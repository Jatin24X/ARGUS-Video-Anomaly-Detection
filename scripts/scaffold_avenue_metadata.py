"""Scaffold Avenue metadata from an extracted dataset folder.

This script is intentionally conservative:

- it inventories the frame folders already present on disk
- it writes `avenue_inventory.json`
- it writes `avenue_scenes.json`
- it writes a split template matching the current Stream A loader expectations

It does NOT fabricate frame labels. Real benchmark evaluation still requires
the proper Avenue anomaly annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sorted_video_dirs(root: Path) -> list[Path]:
    dirs = [path for path in root.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda path: path.name)


def _frame_count(video_dir: Path) -> int:
    return sum(1 for path in video_dir.iterdir() if path.is_file())


def _inventory_payload(train_dirs: list[Path], test_dirs: list[Path], dataset_root: Path) -> dict:
    def _rows(paths: list[Path], split_name: str) -> list[dict]:
        return [
            {
                "video_name": f"{split_name}_{path.name}",
                "source_video_name": path.name,
                "split": split_name,
                "num_frames": _frame_count(path),
                "relative_path": str(path.relative_to(dataset_root)).replace("\\", "/"),
            }
            for path in paths
        ]

    return {
        "dataset": "avenue",
        "dataset_root": str(dataset_root),
        "train_videos": _rows(train_dirs, "train"),
        "test_videos": _rows(test_dirs, "test"),
        "train_video_count": len(train_dirs),
        "test_video_count": len(test_dirs),
    }


def _scene_payload(train_dirs: list[Path], test_dirs: list[Path]) -> dict[str, int]:
    payload = {}
    for split_name, paths in (("train", train_dirs), ("test", test_dirs)):
        for path in paths:
            payload[f"{split_name}_{path.name}"] = 1
    return payload


def _split_payload(train_dirs: list[Path], test_dirs: list[Path], val_videos: set[str]) -> dict:
    train_names = [path.name for path in train_dirs]
    test_names = [path.name for path in test_dirs]

    unknown = sorted(val_videos - set(train_names))
    if unknown:
        raise ValueError(
            "Validation videos must come from training frame folders. "
            f"Unknown ids: {', '.join(unknown)}"
        )

    train_normal = [f"train_{name}" for name in train_names if name not in val_videos]
    val_normal = [f"train_{name}" for name in train_names if name in val_videos]

    return {
        "train": {
            "normal": train_normal,
            "abnormal": [],
        },
        "val": {
            "normal": val_normal,
            "abnormal": [],
        },
        "test": {
            "normal": [],
            "abnormal": [f"test_{name}" for name in test_names],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold Avenue metadata templates")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the extracted Avenue Dataset root",
    )
    parser.add_argument(
        "--output-metadata-dir",
        default="data/metadata",
        help="Where metadata JSON files should be written",
    )
    parser.add_argument(
        "--train-frames-subdir",
        default=r"train\frames",
        help="Relative path to the canonical training frame folders",
    )
    parser.add_argument(
        "--test-frames-subdir",
        default=r"test\frames",
        help="Relative path to the canonical test frame folders",
    )
    parser.add_argument(
        "--val-train-videos",
        nargs="*",
        default=[],
        help="Optional held-out training video ids for a val template, e.g. 14 15 16",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_metadata_dir = Path(args.output_metadata_dir)
    train_frames_root = dataset_root / args.train_frames_subdir
    test_frames_root = dataset_root / args.test_frames_subdir

    if not train_frames_root.exists():
        raise FileNotFoundError(f"Training frames folder not found: {train_frames_root}")
    if not test_frames_root.exists():
        raise FileNotFoundError(f"Test frames folder not found: {test_frames_root}")

    train_dirs = _sorted_video_dirs(train_frames_root)
    test_dirs = _sorted_video_dirs(test_frames_root)

    inventory = _inventory_payload(train_dirs, test_dirs, dataset_root)
    scenes = _scene_payload(train_dirs, test_dirs)
    splits = _split_payload(train_dirs, test_dirs, set(args.val_train_videos))

    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    (output_metadata_dir / "avenue_inventory.json").write_text(
        json.dumps(inventory, indent=2),
        encoding="utf-8",
    )
    (output_metadata_dir / "avenue_scenes.json").write_text(
        json.dumps(scenes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_metadata_dir / "avenue_splits_template.json").write_text(
        json.dumps(splits, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote inventory: {output_metadata_dir / 'avenue_inventory.json'}")
    print(f"Wrote scenes:    {output_metadata_dir / 'avenue_scenes.json'}")
    print(f"Wrote splits:    {output_metadata_dir / 'avenue_splits_template.json'}")
    print()
    print("Labels are still required separately.")
    print(
        "Next missing file for real evaluation: "
        f"{output_metadata_dir / 'avenue_frame_labels.json'}"
    )


if __name__ == "__main__":
    main()
