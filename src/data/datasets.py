"""Standalone Stream A datasets.

Only the VideoMAE clip dataset is retained here so the package stays
focused on Stream A and can be adapted to other datasets such as Avenue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)
_MISSING_FRAME_LABEL_WARNED: set[str] = set()

VIDEOMAE_CLIP_LENGTH = 16
VIDEOMAE_TEMPORAL_STRIDE = 4


def _metadata_file(metadata_dir: Path, dataset_name: str, suffix: str) -> Path:
    return metadata_dir / f"{dataset_name}_{suffix}.json"


def load_metadata(metadata_dir: Path, dataset_name: str = "ubnormal") -> Tuple[dict, dict, dict]:
    """Load dataset metadata using a simple portable naming convention."""
    splits_path = _metadata_file(metadata_dir, dataset_name, "splits")
    frame_labels_path = _metadata_file(metadata_dir, dataset_name, "frame_labels")
    scenes_path = _metadata_file(metadata_dir, dataset_name, "scenes")

    if not splits_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {splits_path}")

    with splits_path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    if frame_labels_path.exists():
        with frame_labels_path.open("r", encoding="utf-8") as handle:
            frame_labels = json.load(handle)
    else:
        warning_key = str(frame_labels_path.resolve())
        if warning_key not in _MISSING_FRAME_LABEL_WARNED:
            logger.warning(
                "Frame-label metadata is missing: %s. Normal-only training/holdout can still run, "
                "but abnormal evaluation will fail until labels are added.",
                frame_labels_path,
            )
            _MISSING_FRAME_LABEL_WARNED.add(warning_key)
        frame_labels = {}

    if scenes_path.exists():
        with scenes_path.open("r", encoding="utf-8") as handle:
            scenes = json.load(handle)
    else:
        scenes = {}
        split_payload = splits.values() if isinstance(splits, dict) else []
        for split_entry in split_payload:
            if isinstance(split_entry, dict):
                for videos in split_entry.values():
                    for video_name in videos:
                        scenes.setdefault(video_name, 1)
            elif isinstance(split_entry, list):
                for video_name in split_entry:
                    scenes.setdefault(video_name, 1)

    return splits, frame_labels, scenes


def resolve_video_feature_path(
    features_dir: Path,
    video_name: str,
    scene_num: int | None = None,
) -> Path:
    """Resolve one feature file.

    UBnormal uses Scene{n}/{video}.npy. For portability we also support a flat
    features folder and a final recursive fallback.
    """
    candidates = []
    if scene_num is not None:
        candidates.append(features_dir / f"Scene{int(scene_num)}" / f"{video_name}.npy")
    candidates.append(features_dir / f"{video_name}.npy")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(features_dir.rglob(f"{video_name}.npy"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Feature file not found for {video_name} under {features_dir}")


def get_frame_labels_for_video(
    video_name: str,
    num_frames: int,
    frame_labels: dict,
    is_abnormal: bool,
) -> np.ndarray:
    if not is_abnormal:
        return np.zeros(num_frames, dtype=np.int8)

    raw = frame_labels.get(video_name, [])
    labels = np.array(raw, dtype=np.int8) if raw else np.zeros(0, dtype=np.int8)
    if len(labels) < num_frames:
        labels = np.pad(labels, (0, num_frames - len(labels)))
    elif len(labels) > num_frames:
        labels = labels[:num_frames]
    return labels


def _compute_clip_starts(num_frames: int) -> List[int]:
    if num_frames >= VIDEOMAE_CLIP_LENGTH:
        return list(range(0, num_frames - VIDEOMAE_CLIP_LENGTH + 1, VIDEOMAE_TEMPORAL_STRIDE))
    return [0]


def _compute_clip_label_for_training(
    clip_start: int,
    num_frames: int,
    frame_labels: np.ndarray,
) -> bool:
    span_end = clip_start + (VIDEOMAE_CLIP_LENGTH - 1) * VIDEOMAE_TEMPORAL_STRIDE
    span_end = min(span_end, num_frames - 1)
    return frame_labels[clip_start : span_end + 1].sum() == 0


def _compute_clip_label_for_eval(
    clip_start: int,
    num_frames: int,
    frame_labels: np.ndarray,
) -> int:
    center_offset = (VIDEOMAE_CLIP_LENGTH // 2) * VIDEOMAE_TEMPORAL_STRIDE
    center_frame = min(clip_start + center_offset, num_frames - 1)
    return int(frame_labels[center_frame])


def _iter_split_videos(split_payload) -> List[Tuple[str, bool]]:
    """Return (video_name, is_abnormal) pairs from flexible split metadata."""
    items: List[Tuple[str, bool]] = []
    if isinstance(split_payload, dict):
        for category, videos in split_payload.items():
            is_abnormal = str(category).lower() == "abnormal"
            for video_name in videos:
                items.append((video_name, is_abnormal))
        return items

    if isinstance(split_payload, list):
        for video_name in split_payload:
            items.append((video_name, False))
        return items

    raise ValueError(f"Unsupported split payload type: {type(split_payload)!r}")


class VideoMAEClipDataset(Dataset):
    """Clip-level dataset over pre-extracted VideoMAE features."""

    def __init__(
        self,
        features_dir: Path,
        metadata_dir: Path,
        split: str = "train",
        mode: str = "train",
        dataset_name: str = "ubnormal",
        clip_subsample: int = 1,
    ):
        self.features_dir = Path(features_dir)
        self.metadata_dir = Path(metadata_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.mode = mode
        self.clip_subsample = max(1, int(clip_subsample))
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.sample_metadata: List[Dict[str, object]] = []

        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")

        splits, frame_labels, scenes = load_metadata(self.metadata_dir, dataset_name=dataset_name)
        split_payload = splits.get(split)
        if split_payload is None:
            raise KeyError(f"Split {split!r} not found in metadata for {dataset_name}")

        for video_name, is_abnormal in _iter_split_videos(split_payload):
            if mode == "train" and is_abnormal:
                continue

            scene_num = scenes.get(video_name)
            feature_path = resolve_video_feature_path(self.features_dir, video_name, scene_num)
            features = np.load(str(feature_path)).astype(np.float32)
            num_clips = int(features.shape[0])
            if num_clips == 0:
                continue

            if is_abnormal:
                if video_name not in frame_labels:
                    raise ValueError(
                        f"Missing frame labels for abnormal video {video_name!r} in split "
                        f"{split!r}. Add {dataset_name}_frame_labels.json before running "
                        f"{mode} mode on abnormal clips."
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
                num_frames = (
                    VIDEOMAE_CLIP_LENGTH
                    if num_clips <= 1
                    else ((num_clips - 1) * VIDEOMAE_TEMPORAL_STRIDE + VIDEOMAE_CLIP_LENGTH)
                )
                gt_labels = np.zeros(num_frames, dtype=np.int8)

            clip_starts = _compute_clip_starts(num_frames)
            effective_clips = min(len(clip_starts), num_clips)
            if effective_clips == 0:
                continue

            iterator = range(0, effective_clips, self.clip_subsample if mode == "train" else 1)
            for clip_idx in iterator:
                clip_start = clip_starts[clip_idx]
                keep = True
                if mode == "train":
                    keep = _compute_clip_label_for_training(
                        clip_start=clip_start,
                        num_frames=num_frames,
                        frame_labels=gt_labels,
                    )
                    label = 0
                else:
                    label = _compute_clip_label_for_eval(
                        clip_start=clip_start,
                        num_frames=num_frames,
                        frame_labels=gt_labels,
                    )
                if not keep:
                    continue

                self.samples.append((features[clip_idx], label))
                self.sample_metadata.append(
                    {
                        "video_name": video_name,
                        "scene_id": int(scene_num) if scene_num is not None else 1,
                        "clip_index": int(clip_idx),
                        "clip_start": int(clip_start),
                    }
                )

        logger.info(
            "VideoMAEClipDataset(%s/%s): %s clips from %s videos",
            split,
            mode,
            len(self.samples),
            len({item['video_name'] for item in self.sample_metadata}),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        feature, label = self.samples[index]
        return torch.from_numpy(feature), int(label)
