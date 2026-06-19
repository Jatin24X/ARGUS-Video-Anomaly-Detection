from __future__ import annotations

import base64
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch

# Ensure project root is on path if needed
PROJECT_ROOT = Path(os.environ.get("ARGUS_STREAM_A_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import gaussian_smooth, minmax_normalize
from src.models.backbones.videomae import (
    CLIP_LENGTH,
    FRAME_SIZE,
    TEMPORAL_STRIDE,
    VideoMAEFeatureExtractor,
)
from src.models.scorers.mulde import MULDEScorer
from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_ANALYSIS_FPS = 12.0
MAX_ANALYSIS_FRAMES = 720
DISPLAY_MAX_EDGE = 720
CACHE_SIZE = 2


@dataclass(frozen=True)
class InferenceProfile:
    key: str
    label: str
    dataset_name: str
    checkpoint_path: Path
    benchmark_report: str
    benchmark_micro: float
    benchmark_macro: float
    benchmark_clip: float
    scoring_mode: str
    signal_kind: str
    smoothing_sigma: float
    gmm_components: int = 0
    single_sigma_index: int = 0
    percentile: float = 85.0
    headline: str = ""
    note: str = ""
    accent: str = "#0f766e"
    accent_soft: str = "#ccfbf1"
    badge: str = ""

    def score_description(self) -> str:
        if self.scoring_mode == "gmm":
            return (
                f"{self.signal_kind} + GMM({self.gmm_components}), "
                f"smoothing={self.smoothing_sigma:g}"
            )
        return (
            f"{self.signal_kind} @ sigma_index={self.single_sigma_index}, "
            f"smoothing={self.smoothing_sigma:g}"
        )


AVENUE_PROFILE = InferenceProfile(
    key="avenue",
    label="Avenue profile",
    dataset_name="Avenue",
    checkpoint_path=PROJECT_ROOT
    / "outputs"
    / "avenue_stream_a_ld_gmm1_beta01_lr4e5_run1"
    / "checkpoints"
    / "stream_a"
    / "best_holdout.pt",
    benchmark_report="outputs/reports/avenue_stream_a_best_test.json",
    benchmark_micro=0.8451,
    benchmark_macro=0.8514,
    benchmark_clip=0.8400,
    scoring_mode="gmm",
    signal_kind="log_density",
    smoothing_sigma=13.0,
    gmm_components=1,
    headline="Avenue analysis profile",
    note="Main saved Avenue profile for the standalone Stream A demo.",
    accent="#0f766e",
    accent_soft="rgba(20, 184, 166, 0.16)",
    badge="Saved profile",
)

UBNORMAL_PROFILE = InferenceProfile(
    key="ubnormal",
    label="UBnormal profile",
    dataset_name="UBnormal",
    checkpoint_path=PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "stream_a_locked_videomae_beta1_score_norm_sigma0.pt",
    benchmark_report="outputs/reports/stream_a_frozen_baseline.json",
    benchmark_micro=0.7394,
    benchmark_macro=0.8410,
    benchmark_clip=0.7309,
    scoring_mode="multiscale",
    signal_kind="score_norm",
    smoothing_sigma=20.0,
    single_sigma_index=0,
    headline="UBnormal analysis profile",
    note="Locked Stream A profile kept in the demo for comparison.",
    accent="#b45309",
    accent_soft="rgba(245, 158, 11, 0.16)",
    badge="Saved profile",
)

PROFILES: Dict[str, InferenceProfile] = {
    AVENUE_PROFILE.label: AVENUE_PROFILE,
    UBNORMAL_PROFILE.label: UBNORMAL_PROFILE,
}


def _recommended_batch_size() -> int:
    if not torch.cuda.is_available():
        return 2
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb >= 20:
        return 12
    if total_gb >= 10:
        return 8
    if total_gb >= 6:
        return 4
    return 2


def _resize_for_display(frame_rgb: np.ndarray) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, DISPLAY_MAX_EDGE / max(h, w))
    if scale >= 1.0:
        return frame_rgb
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _resolve_video_path(video_value: object) -> str | None:
    if video_value is None:
        return None
    if isinstance(video_value, Path):
        return str(video_value)
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        for key in ("path", "name", "video"):
            value = video_value.get(key)
            if isinstance(value, (str, Path)):
                return str(value)
    raise ValueError(f"Unsupported video input type: {type(video_value)!r}")


def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    diff = np.diff(mask.astype(np.int32))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask.size and mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask.size and mask[-1]:
        ends = np.append(ends, mask.size)
    return list(zip(starts.tolist(), ends.tolist()))


def _select_gallery_indices(
    scores: np.ndarray,
    *,
    max_items: int = 4,
    min_gap: int = 8,
) -> List[int]:
    if scores.size == 0:
        return []

    selected: List[int] = []
    for idx in np.argsort(scores)[::-1]:
        idx_i = int(idx)
        if any(abs(idx_i - prev) < min_gap for prev in selected):
            continue
        selected.append(idx_i)
        if len(selected) >= max_items:
            break
    return selected


def _encode_frame_data_uri(frame_rgb: np.ndarray, *, jpeg_quality: int = 80) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError("Failed to encode gallery frame for API response.")
    encoded_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_b64}"


def _gallery_payload(
    frames_rgb: List[np.ndarray],
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_items: int = 4,
    min_gap: int = 8,
) -> List[dict[str, object]]:
    payload: List[dict[str, object]] = []
    for idx in _select_gallery_indices(scores, max_items=max_items, min_gap=min_gap):
        payload.append(
            {
                "index": int(idx),
                "timestamp_sec": float(timestamps[idx]),
                "score": float(scores[idx]),
                "caption": f"{timestamps[idx]:.2f}s  |  score {scores[idx]:.3f}",
                "image_data_url": _encode_frame_data_uri(frames_rgb[idx]),
            }
        )
    return payload


def _anomaly_regions_payload(
    timestamps: np.ndarray,
    anomaly_mask: np.ndarray,
) -> List[dict[str, float]]:
    if timestamps.size == 0:
        return []

    payload: List[dict[str, float]] = []
    for start, end in _contiguous_regions(anomaly_mask):
        end_idx = min(end - 1, len(timestamps) - 1)
        payload.append(
            {
                "start_time_sec": float(timestamps[start]),
                "end_time_sec": float(timestamps[end_idx]),
                "start_index": int(start),
                "end_index": int(end_idx),
            }
        )
    return payload


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _profile_payload(profile: InferenceProfile) -> dict[str, object]:
    return {
        "key": profile.key,
        "label": profile.label,
        "dataset_name": profile.dataset_name,
        "headline": profile.headline,
        "note": profile.note,
        "badge": profile.badge,
        "accent": profile.accent,
        "benchmark_micro_auc_pct": _pct(profile.benchmark_micro),
        "benchmark_macro_auc_pct": _pct(profile.benchmark_macro),
        "benchmark_clip_auc_pct": _pct(profile.benchmark_clip),
        "benchmark_micro_auc": float(profile.benchmark_micro),
        "benchmark_macro_auc": float(profile.benchmark_macro),
        "benchmark_clip_auc": float(profile.benchmark_clip),
        "benchmark_report": profile.benchmark_report,
    }


class InferenceEngine:
    """Core inference engine for model prediction."""

    def __init__(self) -> None:
        self.device = os.environ.get(
            "ARGUS_STREAM_A_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.batch_size = int(os.environ.get("ARGUS_STREAM_A_BATCH_SIZE", "0")) or _recommended_batch_size()
        self.extractor: VideoMAEFeatureExtractor | None = None
        self.scorers: Dict[str, MULDEScorer] = {}
        self.cache: OrderedDict[Tuple[str, int, int], dict] = OrderedDict()

    def preload(
        self,
        *,
        include_extractor: bool = True,
        profile_labels: List[str] | None = None,
    ) -> None:
        logger.info(
            "Preloading ARGUS Stream A assets on %s (extractor=%s)",
            self.device,
            include_extractor,
        )
        if include_extractor:
            self._get_extractor()

        labels = profile_labels or list(PROFILES.keys())
        for label in labels:
            profile = PROFILES[label]
            self._get_scorer(profile)

    def _get_extractor(self) -> VideoMAEFeatureExtractor:
        if self.extractor is None:
            logger.info("Loading VideoMAE extractor on %s", self.device)
            self.extractor = VideoMAEFeatureExtractor(device=self.device)
        return self.extractor

    def _get_scorer(self, profile: InferenceProfile) -> MULDEScorer:
        scorer = self.scorers.get(profile.key)
        if scorer is not None:
            return scorer

        if not profile.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {profile.checkpoint_path}")

        logger.info("Loading scorer for profile %s", profile.key)
        scorer = MULDEScorer.load_checkpoint(profile.checkpoint_path, device=self.device)
        scorer.eval()
        self.scorers[profile.key] = scorer
        return scorer

    @staticmethod
    def _cache_key(video_path: str, roi_sector: str = "full") -> Tuple[str, int, int, str]:
        path = Path(video_path)
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size), roi_sector)

    def _cache_get(self, key: Tuple[str, int, int]) -> dict | None:
        item = self.cache.get(key)
        if item is None:
            return None
        self.cache.move_to_end(key)
        return item

    def _cache_put(self, key: Tuple[str, int, int], value: dict) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def _decode_video(self, video_path: str, roi_sector: str = "full") -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        raw_frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        sample_step = max(1, int(round(source_fps / TARGET_ANALYSIS_FPS)))
        if raw_frame_total > 0:
            sample_step = max(sample_step, int(np.ceil(raw_frame_total / MAX_ANALYSIS_FRAMES)))

        display_frames: List[np.ndarray] = []
        model_frames: List[np.ndarray] = []
        sampled_indices: List[int] = []

        frame_idx = 0
        while len(model_frames) < MAX_ANALYSIS_FRAMES:
            ok = cap.grab()
            if not ok:
                break

            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue

            ok, frame_bgr = cap.retrieve()
            if not ok:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            if roi_sector == "center":
                frame_rgb = frame_rgb[:, int(w * 0.2):int(w * 0.8)]
            elif roi_sector == "left":
                frame_rgb = frame_rgb[:, :int(w * 0.5)]
            elif roi_sector == "right":
                frame_rgb = frame_rgb[:, int(w * 0.5):]

            display_frames.append(_resize_for_display(frame_rgb))
            model_frames.append(
                cv2.resize(
                    frame_rgb,
                    (FRAME_SIZE, FRAME_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            sampled_indices.append(frame_idx)
            frame_idx += 1

        cap.release()

        if raw_frame_total <= 0:
            raw_frame_total = frame_idx

        timestamps = (
            np.asarray(sampled_indices, dtype=np.float64) / source_fps
            if sampled_indices
            else np.empty((0,), dtype=np.float64)
        )

        return {
            "raw_frame_total": raw_frame_total,
            "source_fps": source_fps,
            "sample_step": sample_step,
            "display_frames": display_frames,
            "model_frames": model_frames,
            "timestamps": timestamps,
        }

    def _score_clips(self, cached_video: dict, profile: InferenceProfile) -> np.ndarray:
        score_cache: Dict[str, np.ndarray] = cached_video["score_cache"]
        cached_scores = score_cache.get(profile.key)
        if cached_scores is not None:
            return cached_scores

        features = cached_video["features"]
        if features.size == 0:
            return np.empty((0,), dtype=np.float64)

        scorer = self._get_scorer(profile)
        feat_t = torch.from_numpy(features.astype(np.float32)).to(self.device)

        if profile.scoring_mode == "gmm":
            with torch.inference_mode():
                clip_scores = scorer.score_anomaly(feat_t)
        else:
            signal = scorer.compute_multiscale_signal(feat_t, signal_kind=profile.signal_kind)
            clip_scores = signal[:, profile.single_sigma_index]

        clip_scores = np.asarray(clip_scores, dtype=np.float64)
        score_cache[profile.key] = clip_scores
        return clip_scores

    def _run_analysis(
        self,
        video_path: object,
        profile_label: str,
        *,
        roi_sector: str = "full",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        resolved_video_path = _resolve_video_path(video_path)
        if resolved_video_path is None:
            raise ValueError("Upload a video to start the analysis.")

        try:
            profile = PROFILES[profile_label]
        except KeyError as exc:
            raise ValueError(f"Unknown analysis profile: {profile_label}") from exc

        progress_callback = progress_callback or (lambda _fraction, _desc: None)
        cache_hit = False
        started = time.time()

        progress_callback(0.05, "Opening video")
        key = self._cache_key(resolved_video_path, roi_sector)
        cached_video = self._cache_get(key)

        if cached_video is None:
            decoded = self._decode_video(resolved_video_path, roi_sector)
            sampled_frame_count = len(decoded["model_frames"])
            if sampled_frame_count < CLIP_LENGTH:
                raise ValueError(
                    "Video too short for Stream A analysis. "
                    f"Need at least {CLIP_LENGTH} sampled frames and only found "
                    f"{sampled_frame_count}."
                )

            progress_callback(0.28, "Loading VideoMAE")
            extractor = self._get_extractor()

            progress_callback(0.42, "Extracting clip embeddings")
            features = extractor.extract_from_frames(
                decoded["model_frames"],
                batch_size=self.batch_size,
            )

            cached_video = {
                **decoded,
                "features": features,
                "score_cache": {},
            }
            self._cache_put(key, cached_video)
        else:
            cache_hit = True

        progress_callback(0.72, f"Scoring with {profile.dataset_name}")
        clip_scores = self._score_clips(cached_video, profile)

        timestamps = cached_video["timestamps"]
        frame_count = len(cached_video["display_frames"])
        clip_count = int(len(clip_scores))
        if clip_count == 0 or frame_count == 0:
            raise ValueError("No valid clips were extracted from this video.")

        progress_callback(0.84, "Reconstructing frame-level scores")
        clip_starts = np.arange(
            0,
            frame_count - CLIP_LENGTH + 1,
            TEMPORAL_STRIDE,
            dtype=np.int32,
        )[:clip_count]
        center_offset = (CLIP_LENGTH // 2) * TEMPORAL_STRIDE
        centers = np.minimum(clip_starts + center_offset, frame_count - 1)

        if clip_count == 1:
            frame_scores = np.full((frame_count,), float(clip_scores[0]), dtype=np.float64)
        else:
            frame_scores = np.interp(
                np.arange(frame_count, dtype=np.float64),
                centers.astype(np.float64),
                clip_scores.astype(np.float64),
            )

        smoothed = gaussian_smooth(frame_scores, sigma=profile.smoothing_sigma)
        normalized = minmax_normalize(smoothed)
        threshold = float(np.percentile(normalized, profile.percentile))
        anomaly_mask = normalized >= threshold
        elapsed = time.time() - started

        return {
            "profile": profile,
            "resolved_video_path": resolved_video_path,
            "cache_hit": cache_hit,
            "cached_video": cached_video,
            "timestamps": timestamps,
            "frame_count": frame_count,
            "clip_count": clip_count,
            "scores": normalized,
            "threshold": threshold,
            "anomaly_mask": anomaly_mask,
            "elapsed": elapsed,
        }

    def analyze_payload(self, video_path: object, profile_label: str, roi_sector: str = "full") -> dict[str, Any]:
        analysis = self._run_analysis(video_path, profile_label, roi_sector=roi_sector)
        profile: InferenceProfile = analysis["profile"]
        cached_video = analysis["cached_video"]
        timestamps: np.ndarray = analysis["timestamps"]
        scores: np.ndarray = analysis["scores"]
        anomaly_mask: np.ndarray = analysis["anomaly_mask"]

        peak_idx = int(np.argmax(scores)) if scores.size else 0
        analyzed_duration = float(timestamps[-1]) if timestamps.size else 0.0

        return {
            "profile": _profile_payload(profile),
            "roi_sector": roi_sector,
            "analysis": {
                "video_name": Path(str(analysis["resolved_video_path"])).name,
                "cache_hit": bool(analysis["cache_hit"]),
                "runtime_sec": float(analysis["elapsed"]),
                "timeline": {
                    "timestamps_sec": [float(value) for value in timestamps.tolist()],
                    "scores": [float(value) for value in scores.tolist()],
                    "threshold": float(analysis["threshold"]),
                    "threshold_label": "highlight cutoff",
                    "anomaly_regions": _anomaly_regions_payload(timestamps, anomaly_mask),
                },
                "summary": {
                    "duration_sec": analyzed_duration,
                    "peak_time_sec": float(timestamps[peak_idx]) if timestamps.size else 0.0,
                    "peak_score": float(scores[peak_idx]) if scores.size else 0.0,
                    "raw_frame_count": int(cached_video["raw_frame_total"]),
                    "sampled_frame_count": int(analysis["frame_count"]),
                    "sample_step": int(cached_video["sample_step"]),
                    "source_fps": float(cached_video["source_fps"]),
                    "clip_count": int(analysis["clip_count"]),
                    "profile_label": profile.label,
                    "profile_dataset": profile.dataset_name,
                },
                "frames": _gallery_payload(
                    cached_video["display_frames"],
                    scores,
                    timestamps,
                ),
            },
        }


ENGINE = InferenceEngine()
