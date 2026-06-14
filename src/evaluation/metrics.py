"""ARGUS - Evaluation metrics."""

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.logging import get_logger

logger = get_logger(__name__)


def gaussian_smooth(scores: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1D Gaussian smoothing to per-frame scores."""
    if sigma <= 0 or len(scores) < 3:
        return scores
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(scores, sigma=sigma)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def compute_frame_auc(
    video_scores: Dict[str, np.ndarray],
    video_labels: Dict[str, np.ndarray],
    smoothing_sigma: float = 5.0,
    normalize: bool = True,
) -> dict:
    """Compute frame-level micro/macro AUC following UBnormal protocol."""
    micro_scores = []
    micro_labels = []
    per_video_auc = {}
    videos_evaluated = 0

    for video_name in sorted(video_scores.keys()):
        if video_name not in video_labels:
            continue

        scores = video_scores[video_name].copy()
        labels = video_labels[video_name].copy()

        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]

        if len(scores) == 0:
            continue

        smoothed = gaussian_smooth(scores, smoothing_sigma)
        micro_scores.append(smoothed)
        micro_labels.append(labels)

        norm_scores = minmax_normalize(smoothed) if normalize else smoothed
        try:
            vid_auc = roc_auc_score(
                np.concatenate(([0], labels, [1])),
                np.concatenate(([0], norm_scores, [1])),
            )
            per_video_auc[video_name] = vid_auc
        except ValueError:
            logger.warning("Skipping per-video AUC for %s due to invalid labels", video_name)

        videos_evaluated += 1

    if not micro_scores:
        return {
            "micro_auc": 0.5,
            "macro_auc": 0.5,
            "per_video_auc": {},
            "num_videos": 0,
            "num_frames": 0,
        }

    pooled_scores = np.concatenate(micro_scores)
    pooled_labels = np.concatenate(micro_labels)

    if pooled_labels.sum() == 0 or pooled_labels.sum() == len(pooled_labels):
        micro_auc = 0.5
    else:
        try:
            micro_auc = roc_auc_score(pooled_labels, pooled_scores)
        except ValueError:
            micro_auc = 0.5

    macro_auc = float(np.mean(list(per_video_auc.values()))) if per_video_auc else 0.5

    return {
        "micro_auc": micro_auc,
        "macro_auc": macro_auc,
        "per_video_auc": per_video_auc,
        "num_videos": videos_evaluated,
        "num_frames": len(pooled_scores),
    }
