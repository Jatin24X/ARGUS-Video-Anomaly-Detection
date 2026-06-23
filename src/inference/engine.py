from __future__ import annotations

import base64
import os
import sys
import time
import hashlib
import queue
import threading
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
    extractor: VideoMAEFeatureExtractor | None,
    scorer: Any,
    device: str,
    vlm_caption: str = "",
    peak_idx: int = -1,
    *,
    max_items: int = 4,
    min_gap: int = 8,
) -> List[dict[str, object]]:
    payload: List[dict[str, object]] = []
    selected_indices = _select_gallery_indices(scores, max_items=max_items, min_gap=min_gap)
    for idx in selected_indices:
        frame_rgb = frames_rgb[idx]
        if extractor is not None and scorer is not None:
            try:
                frame_count = len(frames_rgb)
                clip_starts = np.arange(
                    0,
                    frame_count - CLIP_LENGTH + 1,
                    TEMPORAL_STRIDE,
                    dtype=np.int32,
                )
                center_offset = (CLIP_LENGTH // 2) * TEMPORAL_STRIDE
                centers = np.minimum(clip_starts + center_offset, frame_count - 1)
                
                closest_clip_idx = int(np.argmin(np.abs(centers - idx)))
                clip_start = clip_starts[closest_clip_idx]
                clip_frames = []
                for i in range(CLIP_LENGTH):
                    raw_idx = min(clip_start + i * TEMPORAL_STRIDE, frame_count - 1)
                    frame_model = cv2.resize(
                        frames_rgb[raw_idx],
                        (FRAME_SIZE, FRAME_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    clip_frames.append(frame_model)
                
                heatmap_14x14 = extractor.generate_explainability_heatmap(
                    clip_frames,
                    scorer,
                    target_device=device,
                )
                
                h, w = frame_rgb.shape[:2]
                heatmap_resized = cv2.resize(heatmap_14x14, (w, h), interpolation=cv2.INTER_CUBIC)
                heatmap_resized = np.clip(heatmap_resized, 0, 1)
                heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
                
                color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
                
                blended = cv2.addWeighted(frame_rgb, 0.6, color_heatmap, 0.4, 0)
                
                _, thresh = cv2.threshold(heatmap_uint8, int(0.5 * 255), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                    if w_c * h_c > 100:
                        cv2.rectangle(blended, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
                        
                frame_to_encode = blended
            except Exception as e:
                logger.warning("Failed to generate heatmap for gallery frame %d: %s", idx, e)
                frame_to_encode = frame_rgb
        else:
            frame_to_encode = frame_rgb
            
        description_val = vlm_caption if idx == peak_idx else ""
        
        payload.append(
            {
                "index": int(idx),
                "timestamp_sec": float(timestamps[idx]),
                "score": float(scores[idx]),
                "caption": f"{timestamps[idx]:.2f}s  |  score {scores[idx]:.3f}",
                "image_data_url": _encode_frame_data_uri(frame_to_encode),
                "description": description_val,
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


def generate_video_hash(video_path: Path | str) -> str:
    path = Path(video_path)
    file_size = path.stat().st_size
    sha256 = hashlib.sha256()
    sha256.update(str(file_size).encode("utf-8"))
    
    with open(path, "rb") as f:
        first_1mb = f.read(1024 * 1024)
        sha256.update(first_1mb)
        
        if file_size > 2 * 1024 * 1024:
            f.seek(file_size // 2 - 512 * 1024)
            middle_1mb = f.read(1024 * 1024)
            sha256.update(middle_1mb)
        
        if file_size > 1024 * 1024:
            f.seek(max(0, file_size - 1024 * 1024))
            last_1mb = f.read(1024 * 1024)
            sha256.update(last_1mb)
            
    return sha256.hexdigest()


class PipelinedVideoDecoder:
    def __init__(self, video_path: str, roi_sector: str = "full", max_size: int = 128):
        self.video_path = video_path
        self.roi_sector = roi_sector
        self.queue = queue.Queue(maxsize=max_size)
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.raw_frame_total = 0
        self.source_fps = 30.0
        self.sample_step = 1
        self.error = None
        self.started = False

    def start(self):
        self.started = True
        self.thread.start()

    def _produce(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")

            self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if self.source_fps <= 0:
                self.source_fps = 30.0
            self.raw_frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            self.sample_step = max(1, int(round(self.source_fps / TARGET_ANALYSIS_FPS)))
            if self.raw_frame_total > 0:
                self.sample_step = max(self.sample_step, int(np.ceil(self.raw_frame_total / MAX_ANALYSIS_FRAMES)))

            frame_idx = 0
            sampled_count = 0
            while sampled_count < MAX_ANALYSIS_FRAMES:
                ok = cap.grab()
                if not ok:
                    break

                if frame_idx % self.sample_step != 0:
                    frame_idx += 1
                    continue

                ok, frame_bgr = cap.retrieve()
                if not ok:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                if self.roi_sector == "center":
                    frame_rgb = frame_rgb[:, int(w * 0.2):int(w * 0.8)]
                elif self.roi_sector == "left":
                    frame_rgb = frame_rgb[:, :int(w * 0.5)]
                elif self.roi_sector == "right":
                    frame_rgb = frame_rgb[:, int(w * 0.5):]

                display_f = _resize_for_display(frame_rgb)
                model_f = cv2.resize(
                    frame_rgb,
                    (FRAME_SIZE, FRAME_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                self.queue.put((display_f, model_f, frame_idx))
                sampled_count += 1
                frame_idx += 1

            cap.release()
            if self.raw_frame_total <= 0:
                self.raw_frame_total = frame_idx
        except Exception as e:
            self.error = e
        finally:
            self.queue.put(None)


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
        self.cache: OrderedDict[Tuple[str, str], dict] = OrderedDict()

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
            if self.device == "cuda":
                self._get_vlm()
            else:
                if not os.environ.get("MODAL_PROJECT_NAME") and os.environ.get("ARGUS_SKIP_VLM_PREDOWNLOAD", "1") == "1":
                    logger.info("Skipping Moondream2 pre-download on CPU (local environment).")
                else:
                    try:
                        from huggingface_hub import snapshot_download
                        logger.info("Pre-downloading Moondream2 weights for cache warming...")
                        snapshot_download(repo_id="vikhyatk/moondream2", revision="2024-08-26")
                    except Exception as e:
                        logger.warning(f"Could not pre-download Moondream2 snapshot: {e}")

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

    def _get_vlm(self) -> Tuple[Any, Any] | Tuple[None, None]:
        if not hasattr(self, "_vlm_model") or self._vlm_model is None:
            if self.device != "cuda":
                logger.info("VLM (Moondream2) requires CUDA device. Skipping loading on CPU fallback.")
                self._vlm_model = None
                self._vlm_tokenizer = None
                return None, None
            
            logger.info("Loading Moondream2 VLM model on CUDA...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_id = "vikhyatk/moondream2"
                revision = "2024-08-26"
                
                tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    revision=revision,
                    torch_dtype=torch.float16,
                ).to(self.device)
                model.eval()
                self._vlm_model = model
                self._vlm_tokenizer = tokenizer
                logger.info("Moondream2 VLM loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load Moondream2 VLM: {e}")
                self._vlm_model = None
                self._vlm_tokenizer = None
        return self._vlm_model, self._vlm_tokenizer

    def _run_vlm_caption(self, normal_frame: np.ndarray, anomaly_frame: np.ndarray) -> str:
        model, tokenizer = self._get_vlm()
        if model is None or tokenizer is None:
            return "Anomalous event detected in the video feed."
            
        try:
            from PIL import Image
            pil_normal = Image.fromarray(normal_frame)
            pil_anomaly = Image.fromarray(anomaly_frame)
            
            w1, h1 = pil_normal.size
            w2, h2 = pil_anomaly.size
            new_h = max(h1, h2)
            new_w = w1 + w2
            
            combined = Image.new("RGB", (new_w, new_h))
            combined.paste(pil_normal, (0, (new_h - h1) // 2))
            combined.paste(pil_anomaly, (w1, (new_h - h2) // 2))
            
            prompt = (
                "Compare these two surveillance camera views side-by-side. The left image is normal. "
                "The right image contains an anomalous event. Describe the anomalous activity, safety hazard, "
                "or unusual event in the right image that is not present in the left image. Keep the description "
                "to one concise sentence."
            )
            
            with torch.no_grad():
                enc_image = model.encode_image(combined)
                answer = model.answer_question(enc_image, prompt, tokenizer)
            return answer.strip()
        except Exception as e:
            logger.warning("Error running Moondream2 VLM captioning: %s", e)
            return "Anomalous event detected in the video feed."

    def _cache_key(self, video_path: str, roi_sector: str = "full") -> Tuple[str, str]:
        video_hash = generate_video_hash(video_path)
        return (video_hash, roi_sector)

    def _cache_get(self, key: Tuple[str, str]) -> dict | None:
        item = self.cache.get(key)
        if item is None:
            return None
        self.cache.move_to_end(key)
        return item

    def _cache_put(self, key: Tuple[str, str], value: dict) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

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
        bypass_cache: bool = False,
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

        progress_callback(0.05, "Generating video content hash")
        key = self._cache_key(resolved_video_path, roi_sector)
        cached_video = None if bypass_cache else self._cache_get(key)

        if cached_video is None:
            video_hash, _ = key
            cache_dir = Path("/cache/features")
            cache_path = cache_dir / f"{video_hash}_{roi_sector}.safetensors"
            
            if cache_path.exists() and not bypass_cache:
                logger.info("L2 Volume cache hit for %s", cache_path)
                progress_callback(0.10, "Loading from Volume Cache")
                try:
                    from safetensors.numpy import load_file as load_safetensors
                    data = load_safetensors(str(cache_path))
                    features = data["features"]
                    timestamps = data["timestamps"]
                    display_frames_flat = data["display_frames_flat"]
                    display_frames_sizes = data["display_frames_sizes"]
                    raw_frame_total = int(data["raw_frame_total"][0])
                    source_fps = float(data["source_fps"][0])
                    sample_step = int(data["sample_step"][0])
                        
                    offsets = np.cumsum(np.insert(display_frames_sizes, 0, 0))
                    display_frames = []
                    for i in range(len(display_frames_sizes)):
                        start = offsets[i]
                        end = offsets[i+1]
                        encoded = display_frames_flat[start:end]
                        if len(encoded) > 0:
                            frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            display_frames.append(frame_rgb)
                        else:
                            display_frames.append(np.zeros((720, 720, 3), dtype=np.uint8))
                            
                    cached_video = {
                        "raw_frame_total": raw_frame_total,
                        "source_fps": source_fps,
                        "sample_step": sample_step,
                        "display_frames": display_frames,
                        "model_frames": [],
                        "timestamps": timestamps,
                        "features": features,
                        "score_cache": {},
                    }
                    cache_hit = True
                    self._cache_put(key, cached_video)
                except Exception as e:
                    logger.warning("Failed to load L2 Volume cache at %s: %s. Falling back to fresh run.", cache_path, e)

            if cached_video is None:
                progress_callback(0.15, "Initializing Pipelined Video Decoder")
                decoder = PipelinedVideoDecoder(resolved_video_path, roi_sector, max_size=128)
                decoder.start()

                display_frames = []
                model_frames = []
                sampled_indices = []
                all_features = []
                
                extractor = self._get_extractor()
                required_len = (self.batch_size - 1) * TEMPORAL_STRIDE + CLIP_LENGTH
                next_clip_idx = 0

                progress_callback(0.25, "Decoding & extracting features via ONNX GPU backbone")
                
                while True:
                    item = decoder.queue.get()
                    if item is None:
                        if decoder.error:
                            raise decoder.error
                        break
                    display_f, model_f, frame_idx = item
                    display_frames.append(display_f)
                    model_frames.append(model_f)
                    sampled_indices.append(frame_idx)
                    
                    batch_end_frame_idx = next_clip_idx * TEMPORAL_STRIDE + required_len
                    if len(model_frames) >= batch_end_frame_idx:
                        slice_start = next_clip_idx * TEMPORAL_STRIDE
                        slice_end = batch_end_frame_idx
                        sub_frames = model_frames[slice_start:slice_end]
                        
                        batch_feats = extractor.extract_from_frames(sub_frames, batch_size=self.batch_size)
                        all_features.append(batch_feats)
                        next_clip_idx += self.batch_size
                        
                        pct = min(0.70, 0.25 + 0.45 * (len(model_frames) / max(1, decoder.raw_frame_total)))
                        progress_callback(pct, f"Extracting clip embeddings ({len(model_frames)} frames)")

                num_frames = len(model_frames)
                if num_frames < CLIP_LENGTH:
                    raise ValueError(
                        "Video too short for Stream A analysis. "
                        f"Need at least {CLIP_LENGTH} sampled frames and only found "
                        f"{num_frames}."
                    )
                
                total_clips = (num_frames - CLIP_LENGTH) // TEMPORAL_STRIDE + 1
                if next_clip_idx < total_clips:
                    slice_start = next_clip_idx * TEMPORAL_STRIDE
                    sub_frames = model_frames[slice_start:]
                    batch_feats = extractor.extract_from_frames(sub_frames, batch_size=self.batch_size)
                    all_features.append(batch_feats)
                    
                features = np.concatenate(all_features, axis=0) if all_features else np.empty((0, extractor.hidden_size), dtype=np.float16)
                timestamps = (
                    np.asarray(sampled_indices, dtype=np.float64) / decoder.source_fps
                    if sampled_indices
                    else np.empty((0,), dtype=np.float64)
                )

                cached_video = {
                    "raw_frame_total": decoder.raw_frame_total,
                    "source_fps": decoder.source_fps,
                    "sample_step": decoder.sample_step,
                    "display_frames": display_frames,
                    "model_frames": [],
                    "timestamps": timestamps,
                    "features": features,
                    "score_cache": {},
                }
                self._cache_put(key, cached_video)

                progress_callback(0.70, "Saving to Volume Cache")
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    display_frames_compressed = []
                    for frame in display_frames:
                        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if ok:
                            display_frames_compressed.append(encoded)
                        else:
                            display_frames_compressed.append(np.array([], dtype=np.uint8))
                    
                    display_frames_flat = np.concatenate(display_frames_compressed) if display_frames_compressed else np.empty((0,), dtype=np.uint8)
                    display_frames_sizes = np.array([len(f) for f in display_frames_compressed], dtype=np.int32)
                    
                    from safetensors.numpy import save_file as save_safetensors
                    save_safetensors(
                        {
                            "features": features,
                            "timestamps": timestamps,
                            "display_frames_flat": display_frames_flat,
                            "display_frames_sizes": display_frames_sizes,
                            "raw_frame_total": np.array([decoder.raw_frame_total], dtype=np.int32),
                            "source_fps": np.array([decoder.source_fps], dtype=np.float32),
                            "sample_step": np.array([decoder.sample_step], dtype=np.int32),
                        },
                        str(cache_path)
                    )
                    
                    import modal
                    vol = modal.Volume.from_name("argus-feature-cache")
                    vol.commit()
                    logger.info("Committed feature cache to Modal Volume.")
                except Exception as e:
                    logger.warning("Could not cache features to Modal Volume: %s", e)
        else:
            cache_hit = True

        progress_callback(0.75, f"Scoring behavior via {profile.dataset_name}")
        clip_scores = self._score_clips(cached_video, profile)

        timestamps = cached_video["timestamps"]
        frame_count = len(cached_video["display_frames"])
        clip_count = int(len(clip_scores))
        if clip_count == 0 or frame_count == 0:
            raise ValueError("No valid clips were extracted from this video.")

        progress_callback(0.85, "Reconstructing frame-level scores")
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

    def analyze_payload(
        self,
        video_path: object,
        profile_label: str,
        roi_sector: str = "full",
        bypass_cache: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        progress_callback = progress_callback or (lambda _fraction, _desc: None)
        analysis = self._run_analysis(
            video_path,
            profile_label,
            roi_sector=roi_sector,
            bypass_cache=bypass_cache,
            progress_callback=progress_callback,
        )
        profile: InferenceProfile = analysis["profile"]
        cached_video = analysis["cached_video"]
        timestamps: np.ndarray = analysis["timestamps"]
        scores: np.ndarray = analysis["scores"]
        anomaly_mask: np.ndarray = analysis["anomaly_mask"]

        peak_idx = int(np.argmax(scores)) if scores.size else 0
        lowest_idx = int(np.argmin(scores)) if scores.size else 0
        analyzed_duration = float(timestamps[-1]) if timestamps.size else 0.0

        # Query Moondream2 VLM
        normal_frame = cached_video["display_frames"][lowest_idx]
        anomaly_frame = cached_video["display_frames"][peak_idx]
        
        progress_callback(0.92, "Querying Moondream2 VLM description")
        vlm_caption = self._run_vlm_caption(normal_frame, anomaly_frame)
        
        progress_callback(0.96, "Generating Grad-CAM explainability heatmaps")
        scorer = self._get_scorer(profile)
        extractor = self._get_extractor() if self.extractor is not None else None
        
        frames_payload = _gallery_payload(
            cached_video["display_frames"],
            scores,
            timestamps,
            extractor=extractor,
            scorer=scorer,
            device=self.device,
            vlm_caption=vlm_caption,
            peak_idx=peak_idx,
        )

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
                    "vlm_caption": vlm_caption,
                },
                "frames": frames_payload,
            },
        }


ENGINE = InferenceEngine()
