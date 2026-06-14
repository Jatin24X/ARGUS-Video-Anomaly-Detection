"""Select the best Stream A checkpoint using frame-level validation metrics."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stream_a import (
    SIGNAL_KINDS,
    SIGMA_STRATEGIES,
    default_eval_params,
    evaluate_checkpoint,
    evaluate_normal_holdout_checkpoint,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _discover_checkpoints(
    checkpoint_dir: Path,
    include_legacy_best: bool = False,
) -> list[Path]:
    """Collect raw training checkpoints to rank.

    By default this excludes the old monolithic `best.pt` because it may come
    from a different experiment and pollute ranking of the current periodic
    checkpoints. Use `include_legacy_best=True` only when intentionally
    comparing against that legacy artifact.
    """
    checkpoints = []
    epoch_dir = checkpoint_dir / "epochs"
    if epoch_dir.exists():
        checkpoints.extend(sorted(epoch_dir.glob("epoch_*.pt")))

    for name in ("best_clip.pt", "best_holdout.pt", "last.pt"):
        path = checkpoint_dir / name
        if path.exists():
            checkpoints.append(path)

    legacy_best = checkpoint_dir / "best.pt"
    if legacy_best.exists():
        if include_legacy_best:
            checkpoints.append(legacy_best)
        else:
            logger.info(
                "Ignoring legacy checkpoint %s during ranking; pass "
                "--include-legacy-best to include it explicitly.",
                legacy_best,
            )

    unique = []
    seen = set()
    for path in checkpoints:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write ranked checkpoint rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()} | {"rank", "checkpoint"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Select the best Stream A checkpoint")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/stream_a")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--dataset", default="stream_a_locked")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument(
        "--ranking-mode",
        default="auto",
        choices=["auto", "frame_auc", "normal_holdout"],
        help="How to rank checkpoints; auto uses config.training.model_selection_metric",
    )
    parser.add_argument("--signal-kind", choices=SIGNAL_KINDS, default=None)
    parser.add_argument("--sigma-strategy", choices=SIGMA_STRATEGIES, default=None)
    parser.add_argument("--gmm-components", type=int, default=None)
    parser.add_argument("--single-sigma-index", type=int, default=None)
    parser.add_argument("--smoothing-sigma", type=float, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--include-legacy-best",
        action="store_true",
        help="Also rank the legacy best.pt checkpoint from older workflows",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="Copy the best checkpoint to best_frame.pt",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = _discover_checkpoints(
        checkpoint_dir,
        include_legacy_best=args.include_legacy_best,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    config = load_config(config_dir=args.config_dir, dataset=args.dataset)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_defaults = default_eval_params(config)

    signal_kind = args.signal_kind or eval_defaults["signal_kind"]
    sigma_strategy = args.sigma_strategy or eval_defaults["sigma_strategy"]
    gmm_components = (
        args.gmm_components
        if args.gmm_components is not None
        else eval_defaults["gmm_components"]
    )
    single_sigma_index = (
        args.single_sigma_index
        if args.single_sigma_index is not None
        else eval_defaults["single_sigma_index"]
    )
    smoothing_sigma = (
        args.smoothing_sigma
        if args.smoothing_sigma is not None
        else eval_defaults["smoothing_sigma"]
    )

    logger.info("Ranking %s checkpoints on %s", len(checkpoints), args.split)
    logger.info(
        "Config: signal=%s strategy=%s gmm=%s sigma_idx=%s smoothing=%s",
        signal_kind,
        sigma_strategy,
        gmm_components,
        single_sigma_index,
        smoothing_sigma,
    )

    ranking_mode = args.ranking_mode
    if ranking_mode == "auto":
        model_selection_metric = getattr(config.training, "model_selection_metric", "clip_auc")
        ranking_mode = "normal_holdout" if model_selection_metric == "normal_holdout_score" else "frame_auc"
    logger.info("Ranking mode: %s", ranking_mode)

    rows = []
    for index, checkpoint_path in enumerate(checkpoints, start=1):
        logger.info("[%s/%s] Evaluating %s", index, len(checkpoints), checkpoint_path.name)
        if ranking_mode == "normal_holdout":
            results = evaluate_normal_holdout_checkpoint(
                checkpoint_path=checkpoint_path,
                config=config,
                device=device,
                split=args.split,
                signal_kind=signal_kind,
                sigma_strategy=sigma_strategy,
                gmm_components=gmm_components,
                single_sigma_index=single_sigma_index,
            )
            rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "normal_holdout_score": float(results["normal_holdout_score"]),
                    "normal_holdout_p95": float(results["normal_holdout_p95"]),
                    "normal_holdout_std": float(results["normal_holdout_std"]),
                    "normal_num_clips": int(results["normal_num_clips"]),
                    "signal_kind": signal_kind,
                    "sigma_strategy": sigma_strategy,
                    "gmm_components": gmm_components,
                    "single_sigma_index": single_sigma_index,
                }
            )
        else:
            results = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                config=config,
                device=device,
                split=args.split,
                signal_kind=signal_kind,
                sigma_strategy=sigma_strategy,
                gmm_components=gmm_components,
                single_sigma_index=single_sigma_index,
                smoothing_sigma=smoothing_sigma,
            )
            rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "micro_auc": float(results["micro_auc"]),
                    "macro_auc": float(results["macro_auc"]),
                    "clip_auc": float(results["clip_auc"]),
                    "signal_kind": signal_kind,
                    "sigma_strategy": sigma_strategy,
                    "gmm_components": gmm_components,
                    "single_sigma_index": single_sigma_index,
                    "smoothing_sigma": smoothing_sigma,
                }
            )

    if ranking_mode == "normal_holdout":
        rows.sort(
            key=lambda row: (
                row["normal_holdout_score"],
                row["normal_holdout_p95"],
                row["normal_holdout_std"],
                row["checkpoint"],
            ),
        )
    else:
        rows.sort(
            key=lambda row: (
                row["micro_auc"],
                row["macro_auc"],
                row["clip_auc"],
                row["checkpoint"],
            ),
            reverse=True,
        )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    best_row = rows[0]
    if ranking_mode == "normal_holdout":
        logger.info(
            "Best checkpoint: %s | holdout_mean=%.4f p95=%.4f std=%.4f",
            best_row["checkpoint"],
            best_row["normal_holdout_score"],
            best_row["normal_holdout_p95"],
            best_row["normal_holdout_std"],
        )
    else:
        logger.info(
            "Best checkpoint: %s | micro=%.4f macro=%.4f clip=%.4f",
            best_row["checkpoint"],
            best_row["micro_auc"],
            best_row["macro_auc"],
            best_row["clip_auc"],
        )

    if args.promote_best:
        promote_target = checkpoint_dir / "best_frame.pt"
        shutil.copy2(best_row["checkpoint"], promote_target)
        logger.info("Promoted best checkpoint to %s", promote_target)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        logger.info("Wrote ranking JSON to %s", output_json)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        _write_csv(output_csv, rows)
        logger.info("Wrote ranking CSV to %s", output_csv)


if __name__ == "__main__":
    main()
