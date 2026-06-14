"""Shared Stream A evaluation helpers for frame-level benchmarking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from src.data.datasets import (
    VIDEOMAE_CLIP_LENGTH,
    VIDEOMAE_TEMPORAL_STRIDE,
    VideoMAEClipDataset,
    _compute_clip_starts,
    load_metadata,
    resolve_video_feature_path,
)
from src.evaluation.metrics import compute_frame_auc
from src.models.scorers.mulde import MULDEScorer
from src.utils.logging import get_logger

logger = get_logger(__name__)

SIGNAL_KINDS = ("log_density", "score_norm")
SIGMA_STRATEGIES = ("gmm", "max", "mean", "median", "single_sigma")
LOG_DENSITY_LIKE_SIGNALS = {"log_density"}


@dataclass
class VideoSignalRecord:
    """All per-video data needed to evaluate one signal configuration."""

    clip_starts: np.ndarray
    signal_matrix: np.ndarray
    clip_labels: np.ndarray
    frame_labels: np.ndarray
    num_frames: int


def get_stream_a_paths(config) -> tuple[Path, Path]:
    """Resolve the data directories used by Stream A."""
    data_dir = Path(config.data.data_dir)
    dataset_name = getattr(config.data, "dataset", "ubnormal")
    return data_dir / "features" / dataset_name / "videomae", data_dir / "metadata"


def get_stream_a_batch_size(config) -> int:
    """Resolve the effective Stream A batch size."""
    return getattr(config.stream_a, "batch_size", config.training.batch_size)


def get_stream_a_split_names(config) -> tuple[str, str, str]:
    """Resolve train/val/test split names from config."""
    data_cfg = config.data
    return (
        getattr(data_cfg, "train_split", "train"),
        getattr(data_cfg, "val_split", "val"),
        getattr(data_cfg, "test_split", "test"),
    )


def default_eval_params(config) -> dict:
    """Return the current benchmark config from YAML."""
    evaluation_cfg = config.evaluation
    return {
        "signal_kind": getattr(evaluation_cfg, "signal_kind", "log_density"),
        "sigma_strategy": getattr(evaluation_cfg, "sigma_strategy", "gmm"),
        "gmm_components": getattr(
            evaluation_cfg,
            "gmm_components",
            getattr(config.stream_a, "gmm_components", 5),
        ),
        "single_sigma_index": getattr(evaluation_cfg, "single_sigma_index", None),
        "smoothing_sigma": getattr(evaluation_cfg, "smoothing_sigma", 5),
    }


def _reduce_signal_matrix(
    signal_matrix: np.ndarray,
    signal_kind: str,
    sigma_strategy: str,
    single_sigma_index: Optional[int] = None,
) -> np.ndarray:
    """Reduce an NxL signal matrix to one anomaly score per clip."""
    if sigma_strategy == "max":
        reduced = np.max(signal_matrix, axis=1)
    elif sigma_strategy == "mean":
        reduced = np.mean(signal_matrix, axis=1)
    elif sigma_strategy == "median":
        reduced = np.median(signal_matrix, axis=1)
    elif sigma_strategy == "single_sigma":
        if single_sigma_index is None:
            raise ValueError("single_sigma_index is required when sigma_strategy='single_sigma'")
        if single_sigma_index < 0 or single_sigma_index >= signal_matrix.shape[1]:
            raise ValueError(
                f"single_sigma_index={single_sigma_index} out of range for "
                f"{signal_matrix.shape[1]} eval scales"
            )
        reduced = signal_matrix[:, single_sigma_index]
    else:
        raise ValueError(f"Unsupported sigma_strategy={sigma_strategy!r}")

    if signal_kind in LOG_DENSITY_LIKE_SIGNALS:
        return -reduced
    return reduced


def aggregate_signal_scores(
    train_signal_matrix: np.ndarray,
    eval_signal_matrix: np.ndarray,
    signal_kind: str,
    sigma_strategy: str,
    gmm_components: Optional[int] = None,
    single_sigma_index: Optional[int] = None,
) -> tuple[np.ndarray, Optional[GaussianMixture]]:
    """Aggregate multi-sigma signals into one anomaly score per clip."""
    if signal_kind not in SIGNAL_KINDS:
        raise ValueError(f"Unsupported signal_kind={signal_kind!r}")

    if sigma_strategy == "gmm":
        if gmm_components is None:
            raise ValueError("gmm_components is required when sigma_strategy='gmm'")
        if train_signal_matrix.shape[0] < gmm_components:
            raise ValueError(
                f"Need at least {gmm_components} training samples for GMM, "
                f"got {train_signal_matrix.shape[0]}"
            )

        train_signal_matrix = np.asarray(train_signal_matrix, dtype=np.float64)
        eval_signal_matrix = np.asarray(eval_signal_matrix, dtype=np.float64)

        last_error = None
        gmm = None
        for reg_covar in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                candidate = GaussianMixture(
                    n_components=gmm_components,
                    covariance_type="full",
                    random_state=42,
                    max_iter=200,
                    reg_covar=reg_covar,
                )
                candidate.fit(train_signal_matrix)
                gmm = candidate
                break
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "GMM fit failed for signal aggregation (components=%s, reg_covar=%s): %s",
                    gmm_components,
                    reg_covar,
                    exc,
                )
        if gmm is None:
            raise last_error
        return -gmm.score_samples(eval_signal_matrix), gmm

    return (
        _reduce_signal_matrix(
            eval_signal_matrix,
            signal_kind=signal_kind,
            sigma_strategy=sigma_strategy,
            single_sigma_index=single_sigma_index,
        ),
        None,
    )


def collect_train_signal_matrix(
    scorer: MULDEScorer,
    config,
    device: str,
    signal_kind: str,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Compute an NxL multi-sigma signal matrix on normal training clips."""
    features_dir, metadata_dir = get_stream_a_paths(config)
    train_split, _, _ = get_stream_a_split_names(config)
    train_ds = VideoMAEClipDataset(
        features_dir,
        metadata_dir,
        train_split,
        "train",
        dataset_name=getattr(config.data, "dataset", "ubnormal"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size or get_stream_a_batch_size(config),
        shuffle=False,
        num_workers=0,
    )

    all_signal_matrices = []
    for batch in train_loader:
        features = batch[0].to(device)
        with torch.enable_grad():
            all_signal_matrices.append(scorer.compute_multiscale_signal(features, signal_kind))

    return np.concatenate(all_signal_matrices, axis=0)


def collect_split_signal_matrix(
    scorer: MULDEScorer,
    config,
    split: str,
    device: str,
    signal_kind: str,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Compute an NxL multi-sigma signal matrix for one split."""
    features_dir, metadata_dir = get_stream_a_paths(config)
    split_ds = VideoMAEClipDataset(
        features_dir,
        metadata_dir,
        split,
        "eval",
        dataset_name=getattr(config.data, "dataset", "ubnormal"),
    )
    split_loader = DataLoader(
        split_ds,
        batch_size=batch_size or get_stream_a_batch_size(config),
        shuffle=False,
        num_workers=0,
    )

    all_signal_matrices = []
    for batch in split_loader:
        features = batch[0].to(device)
        with torch.enable_grad():
            all_signal_matrices.append(scorer.compute_multiscale_signal(features, signal_kind))

    if not all_signal_matrices:
        return np.zeros((0, int(getattr(config.stream_a, "eval_L", 16))), dtype=np.float32)
    return np.concatenate(all_signal_matrices, axis=0)


def collect_split_signal_records(
    scorer: MULDEScorer,
    config,
    split: str,
    device: str,
    signal_kind: str,
) -> Dict[str, VideoSignalRecord]:
    """Compute per-video multi-sigma signals for one evaluation split."""
    features_dir, metadata_dir = get_stream_a_paths(config)
    dataset_name = getattr(config.data, "dataset", "ubnormal")
    splits, frame_labels, scenes = load_metadata(metadata_dir, dataset_name=dataset_name)

    records: Dict[str, VideoSignalRecord] = {}
    center_offset = (VIDEOMAE_CLIP_LENGTH // 2) * VIDEOMAE_TEMPORAL_STRIDE

    for category in ("normal", "abnormal"):
        is_abnormal = category == "abnormal"
        for video_name in splits.get(split, {}).get(category, []):
            scene_num = int(scenes[video_name])
            try:
                feat_path = resolve_video_feature_path(features_dir, video_name, scene_num)
            except FileNotFoundError:
                continue

            features = np.load(str(feat_path)).astype(np.float32)
            num_clips = features.shape[0]

            if is_abnormal:
                if video_name not in frame_labels:
                    raise ValueError(
                        f"Missing frame labels for abnormal video {video_name!r} in split "
                        f"{split!r}. Add {dataset_name}_frame_labels.json before running "
                        "abnormal evaluation."
                    )
                gt_labels = np.array(frame_labels[video_name], dtype=np.int8)
                if gt_labels.size == 0:
                    raise ValueError(
                        f"Empty frame-label array for abnormal video {video_name!r} in split "
                        f"{split!r}. Populate {dataset_name}_frame_labels.json with real "
                        "frame-level labels before evaluation."
                    )
                num_frames = len(gt_labels)
            else:
                if num_clips <= 1:
                    num_frames = VIDEOMAE_CLIP_LENGTH
                else:
                    num_frames = (
                        (num_clips - 1) * VIDEOMAE_TEMPORAL_STRIDE + VIDEOMAE_CLIP_LENGTH
                    )
                gt_labels = np.zeros(num_frames, dtype=np.int8)

            clip_starts = np.asarray(_compute_clip_starts(num_frames), dtype=np.int32)
            effective_clips = min(len(clip_starts), num_clips)
            if effective_clips == 0:
                continue

            with torch.enable_grad():
                signal_matrix = scorer.compute_multiscale_signal(
                    torch.from_numpy(features[:effective_clips]).to(device),
                    signal_kind=signal_kind,
                )
            clip_starts = clip_starts[:effective_clips]
            centers = np.minimum(clip_starts + center_offset, num_frames - 1)
            clip_labels = gt_labels[centers]

            records[video_name] = VideoSignalRecord(
                clip_starts=clip_starts,
                signal_matrix=signal_matrix,
                clip_labels=clip_labels.astype(np.int8),
                frame_labels=gt_labels.astype(np.int8),
                num_frames=num_frames,
            )

    return records


def reconstruct_frame_scores(
    video_clip_scores: Dict[str, np.ndarray],
    video_clip_starts: Dict[str, np.ndarray],
    video_num_frames: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Project clip-level scores back to per-frame scores using clip centers."""
    frame_scores = {}
    center_offset = (VIDEOMAE_CLIP_LENGTH // 2) * VIDEOMAE_TEMPORAL_STRIDE

    for video_name, clip_scores in video_clip_scores.items():
        num_frames = video_num_frames[video_name]
        if len(clip_scores) == 0:
            frame_scores[video_name] = np.zeros(num_frames, dtype=np.float64)
            continue

        centers = np.minimum(video_clip_starts[video_name] + center_offset, num_frames - 1)
        per_frame_scores = np.interp(
            np.arange(num_frames, dtype=np.float64),
            centers.astype(np.float64),
            clip_scores.astype(np.float64),
        )
        frame_scores[video_name] = per_frame_scores

    return frame_scores


def evaluate_stream_a_from_caches(
    train_signal_matrix: np.ndarray,
    split_records: Dict[str, VideoSignalRecord],
    signal_kind: str,
    sigma_strategy: str,
    gmm_components: Optional[int],
    single_sigma_index: Optional[int],
    smoothing_sigma: float,
    normalize: bool = True,
) -> dict:
    """Evaluate one fixed Stream A signal configuration from cached matrices."""
    video_clip_scores: Dict[str, np.ndarray] = {}
    video_clip_starts: Dict[str, np.ndarray] = {}
    video_num_frames: Dict[str, int] = {}
    video_frame_labels: Dict[str, np.ndarray] = {}
    all_clip_scores = []
    all_clip_labels = []

    fitted_gmm = None
    if sigma_strategy == "gmm":
        _, fitted_gmm = aggregate_signal_scores(
            train_signal_matrix=train_signal_matrix,
            eval_signal_matrix=train_signal_matrix,
            signal_kind=signal_kind,
            sigma_strategy=sigma_strategy,
            gmm_components=gmm_components,
            single_sigma_index=single_sigma_index,
        )

    for video_name, record in split_records.items():
        if sigma_strategy == "gmm":
            clip_scores = -fitted_gmm.score_samples(record.signal_matrix)
        else:
            clip_scores, _ = aggregate_signal_scores(
                train_signal_matrix=train_signal_matrix,
                eval_signal_matrix=record.signal_matrix,
                signal_kind=signal_kind,
                sigma_strategy=sigma_strategy,
                gmm_components=gmm_components,
                single_sigma_index=single_sigma_index,
            )

        video_clip_scores[video_name] = clip_scores
        video_clip_starts[video_name] = record.clip_starts
        video_num_frames[video_name] = record.num_frames
        video_frame_labels[video_name] = record.frame_labels
        all_clip_scores.append(clip_scores)
        all_clip_labels.append(record.clip_labels)

    frame_scores = reconstruct_frame_scores(
        video_clip_scores=video_clip_scores,
        video_clip_starts=video_clip_starts,
        video_num_frames=video_num_frames,
    )
    frame_results = compute_frame_auc(
        video_scores=frame_scores,
        video_labels={k: v.astype(np.float64) for k, v in video_frame_labels.items()},
        smoothing_sigma=smoothing_sigma,
        normalize=normalize,
    )

    clip_scores = np.concatenate(all_clip_scores) if all_clip_scores else np.array([])
    clip_labels = np.concatenate(all_clip_labels) if all_clip_labels else np.array([])
    if clip_scores.size and clip_labels.sum() > 0 and clip_labels.sum() < len(clip_labels):
        clip_auc = float(roc_auc_score(clip_labels, clip_scores))
    else:
        clip_auc = 0.5

    frame_results.update(
        {
            "clip_auc": clip_auc,
            "signal_kind": signal_kind,
            "sigma_strategy": sigma_strategy,
            "gmm_components": gmm_components,
            "single_sigma_index": single_sigma_index,
            "smoothing_sigma": smoothing_sigma,
            "clip_scores_by_video": video_clip_scores,
            "video_num_frames": video_num_frames,
            "video_frame_labels": video_frame_labels,
            "fitted_gmm": fitted_gmm,
        }
    )
    return frame_results


def evaluate_checkpoint(
    checkpoint_path: Path,
    config,
    device: str,
    split: str,
    signal_kind: str,
    sigma_strategy: str,
    gmm_components: Optional[int],
    single_sigma_index: Optional[int],
    smoothing_sigma: float,
) -> dict:
    """Run a full frame-level evaluation for one checkpoint/config combo."""
    scorer = MULDEScorer.load_checkpoint(checkpoint_path, device=device)
    scorer.eval()

    train_signal_matrix = collect_train_signal_matrix(
        scorer=scorer,
        config=config,
        device=device,
        signal_kind=signal_kind,
    )
    split_records = collect_split_signal_records(
        scorer=scorer,
        config=config,
        split=split,
        device=device,
        signal_kind=signal_kind,
    )

    return evaluate_stream_a_from_caches(
        train_signal_matrix=train_signal_matrix,
        split_records=split_records,
        signal_kind=signal_kind,
        sigma_strategy=sigma_strategy,
        gmm_components=gmm_components,
        single_sigma_index=single_sigma_index,
        smoothing_sigma=smoothing_sigma,
    )


def evaluate_normal_holdout_scorer(
    scorer: MULDEScorer,
    config,
    device: str,
    split: str,
    signal_kind: str,
    sigma_strategy: str,
    gmm_components: Optional[int],
    single_sigma_index: Optional[int],
) -> dict:
    """Evaluate one loaded scorer on a normal-only holdout split."""
    train_signal_matrix = collect_train_signal_matrix(
        scorer=scorer,
        config=config,
        device=device,
        signal_kind=signal_kind,
    )
    holdout_signal_matrix = collect_split_signal_matrix(
        scorer=scorer,
        config=config,
        split=split,
        device=device,
        signal_kind=signal_kind,
    )

    if holdout_signal_matrix.shape[0] == 0:
        return {
            "normal_holdout_score": float("inf"),
            "normal_holdout_p95": float("inf"),
            "normal_holdout_std": float("inf"),
            "normal_num_clips": 0,
            "signal_kind": signal_kind,
            "sigma_strategy": sigma_strategy,
            "gmm_components": gmm_components,
            "single_sigma_index": single_sigma_index,
        }

    holdout_scores, _ = aggregate_signal_scores(
        train_signal_matrix=train_signal_matrix,
        eval_signal_matrix=holdout_signal_matrix,
        signal_kind=signal_kind,
        sigma_strategy=sigma_strategy,
        gmm_components=gmm_components,
        single_sigma_index=single_sigma_index,
    )
    return {
        "normal_holdout_score": float(np.mean(holdout_scores)),
        "normal_holdout_p95": float(np.percentile(holdout_scores, 95)),
        "normal_holdout_std": float(np.std(holdout_scores)),
        "normal_num_clips": int(len(holdout_scores)),
        "signal_kind": signal_kind,
        "sigma_strategy": sigma_strategy,
        "gmm_components": gmm_components,
        "single_sigma_index": single_sigma_index,
    }


def evaluate_normal_holdout_checkpoint(
    checkpoint_path: Path,
    config,
    device: str,
    split: str,
    signal_kind: str,
    sigma_strategy: str,
    gmm_components: Optional[int],
    single_sigma_index: Optional[int],
) -> dict:
    """Evaluate one checkpoint on a normal-only holdout split.

    This is intended for datasets such as Avenue that do not have anomalous
    validation videos for checkpoint selection.
    """
    scorer = MULDEScorer.load_checkpoint(checkpoint_path, device=device)
    scorer.eval()
    return evaluate_normal_holdout_scorer(
        scorer=scorer,
        config=config,
        device=device,
        split=split,
        signal_kind=signal_kind,
        sigma_strategy=sigma_strategy,
        gmm_components=gmm_components,
        single_sigma_index=single_sigma_index,
    )
