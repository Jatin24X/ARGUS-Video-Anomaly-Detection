"""Freeze the official Stream A baseline checkpoint and manifest.

This script copies the locked Stream A checkpoint to a stable immutable path and
writes a machine-readable manifest beside it. It should be run locally after the
winning checkpoint and report artifacts have been synced down from Lightning AI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _git_commit_sha() -> str:
    """Return the current git commit SHA or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the official Stream A baseline")
    parser.add_argument(
        "--source-checkpoint",
        default="outputs/stream_a_beta_1p0/checkpoints/stream_a/best_frame.pt",
        help="Best promoted checkpoint to freeze",
    )
    parser.add_argument(
        "--winning-val-checkpoint",
        default="outputs/stream_a_beta_1p0/checkpoints/stream_a/epochs/epoch_0400.pt",
        help="Epoch checkpoint chosen on frame-level val",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="outputs/checkpoints/stream_a_locked_videomae_beta1_score_norm_sigma0.pt",
        help="Immutable frozen baseline checkpoint path",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest path. Defaults to <output-checkpoint>.manifest.json",
    )
    parser.add_argument(
        "--pointer-json",
        default="outputs/reports/stream_a_frozen_baseline.json",
        help="Optional lightweight pointer JSON for future sessions",
    )
    parser.add_argument(
        "--ranking-json",
        default="outputs/reports/stream_a_beta_1p0_score_norm_sigma0_rank_val.json",
    )
    parser.add_argument(
        "--ranking-csv",
        default="outputs/reports/stream_a_beta_1p0_score_norm_sigma0_rank_val.csv",
    )
    parser.add_argument(
        "--test-json",
        default="outputs/reports/stream_a_beta_1p0_score_norm_sigma0_test.json",
    )
    parser.add_argument("--test-micro-auc", type=float, default=0.7394)
    parser.add_argument("--test-macro-auc", type=float, default=0.8410)
    parser.add_argument("--clip-auc", type=float, default=0.7309)
    args = parser.parse_args()

    source_checkpoint = Path(args.source_checkpoint)
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {source_checkpoint}. "
            "Sync the winning Stream A outputs locally before freezing the baseline."
        )

    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, output_checkpoint)
    logger.info("Copied frozen Stream A checkpoint to %s", output_checkpoint)

    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else output_checkpoint.with_suffix(output_checkpoint.suffix + ".manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_sha(),
        "backbone": "videomae_v2_base",
        "dataset": "ubnormal",
        "training": {
            "beta": 1.0,
            "ema_enabled": False,
            "scheduler": "none",
            "model_selection_metric": "frame_micro_auc",
        },
        "evaluation": {
            "signal_kind": "score_norm",
            "sigma_strategy": "single_sigma",
            "single_sigma_index": 0,
            "smoothing_sigma": 20,
        },
        "winning_val_checkpoint": str(Path(args.winning_val_checkpoint)),
        "frozen_checkpoint": str(output_checkpoint),
        "metrics": {
            "test_micro_auc": args.test_micro_auc,
            "test_macro_auc": args.test_macro_auc,
            "clip_auc": args.clip_auc,
        },
        "reports": {
            "val_ranking_json": args.ranking_json,
            "val_ranking_csv": args.ranking_csv,
            "test_json": args.test_json,
        },
        "notes": [
            "Frozen Stream A baseline after VideoMAE beta=1.0, EMA disabled.",
            "The test split was used during Stream A development and is now frozen.",
            "Use val for all future selection work before re-opening test reporting.",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info("Wrote frozen baseline manifest to %s", manifest_path)

    pointer_path = Path(args.pointer_json)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer = {
        "official_stream_a_baseline": {
            "frozen_checkpoint": str(output_checkpoint),
            "manifest": str(manifest_path),
            "metrics": manifest["metrics"],
        }
    }
    with pointer_path.open("w", encoding="utf-8") as handle:
        json.dump(pointer, handle, indent=2, sort_keys=True)
    logger.info("Wrote frozen baseline pointer JSON to %s", pointer_path)


if __name__ == "__main__":
    main()
