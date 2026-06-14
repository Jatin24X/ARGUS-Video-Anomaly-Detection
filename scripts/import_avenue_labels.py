"""Import Avenue frame-level labels into the standalone Stream A metadata format.

Supports:
- per-frame text labels exported by existing repos (`gt_txt_labels/*.txt`)
- official MATLAB mask files (`testing_label_mask/*_label.mat`)

The output format matches `data/metadata/avenue_frame_labels.json` with keys
like `test_01`, `test_02`, ..., `test_21`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_inventory_lengths(metadata_dir: Path) -> dict[str, int]:
    inventory_path = metadata_dir / "avenue_inventory.json"
    if not inventory_path.exists():
        return {}
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    return {
        row["video_name"]: int(row["num_frames"])
        for row in payload.get("test_videos", [])
    }


def _load_txt_labels(txt_dir: Path) -> dict[str, list[int]]:
    labels: dict[str, list[int]] = {}
    for path in sorted(txt_dir.glob("*.txt")):
        raw_name = path.stem
        video_name = f"test_{raw_name}"
        values = [
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        labels[video_name] = values
    return labels


def _load_mat_labels(mask_dir: Path) -> dict[str, list[int]]:
    labels: dict[str, list[int]] = {}
    for path in sorted(mask_dir.glob("*_label.mat")):
        raw_name = path.stem.replace("_label", "")
        video_name = f"test_{int(raw_name):02d}"
        payload = loadmat(path)
        if "volLabel" not in payload:
            raise ValueError(f"Missing 'volLabel' in {path}")
        vol = payload["volLabel"]
        if vol.ndim != 2 or vol.shape[0] != 1:
            raise ValueError(f"Unexpected volLabel shape in {path}: {vol.shape}")
        frame_values = []
        for index in range(vol.shape[1]):
            mask = np.asarray(vol[0, index])
            frame_values.append(int(np.any(mask)) if mask.size else 0)
        labels[video_name] = frame_values
    return labels


def _validate_lengths(
    labels: dict[str, list[int]],
    *,
    expected_lengths: dict[str, int],
) -> None:
    for video_name, expected in expected_lengths.items():
        if video_name not in labels:
            raise ValueError(f"Missing labels for {video_name}")
        actual = len(labels[video_name])
        if actual != expected:
            raise ValueError(
                f"Length mismatch for {video_name}: expected {expected} frames, got {actual}"
            )


def _validate_binary(labels: dict[str, list[int]]) -> None:
    for video_name, values in labels.items():
        unique = set(values)
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Non-binary labels found for {video_name}: {sorted(unique)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Avenue frame labels")
    parser.add_argument(
        "--txt-label-dir",
        default=None,
        help="Directory containing per-frame 0/1 text labels (preferred when available)",
    )
    parser.add_argument(
        "--mat-mask-dir",
        default=None,
        help="Directory containing official *_label.mat mask files",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/metadata",
        help="Metadata directory containing avenue_inventory.json",
    )
    parser.add_argument(
        "--output-json",
        default="data/metadata/avenue_frame_labels.json",
        help="Where to write the imported frame labels JSON",
    )
    args = parser.parse_args()

    txt_dir = Path(args.txt_label_dir) if args.txt_label_dir else None
    mat_dir = Path(args.mat_mask_dir) if args.mat_mask_dir else None
    metadata_dir = Path(args.metadata_dir)
    output_json = Path(args.output_json)

    if txt_dir is None and mat_dir is None:
        raise SystemExit("Provide at least one of --txt-label-dir or --mat-mask-dir")

    labels = None
    if txt_dir is not None:
        if not txt_dir.exists():
            raise FileNotFoundError(f"Text label directory not found: {txt_dir}")
        labels = _load_txt_labels(txt_dir)

    if mat_dir is not None:
        if not mat_dir.exists():
            raise FileNotFoundError(f"MAT mask directory not found: {mat_dir}")
        mat_labels = _load_mat_labels(mat_dir)
        if labels is None:
            labels = mat_labels
        else:
            if labels != mat_labels:
                raise ValueError(
                    "Text labels and MAT-derived labels do not match. "
                    "Refusing to write inconsistent Avenue labels."
                )

    assert labels is not None
    _validate_binary(labels)

    expected_lengths = _load_inventory_lengths(metadata_dir)
    if expected_lengths:
        _validate_lengths(labels, expected_lengths=expected_lengths)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(labels, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote Avenue frame labels: {output_json}")
    print(f"Videos: {len(labels)}")
    if expected_lengths:
        print("Validated against avenue_inventory.json")


if __name__ == "__main__":
    main()
