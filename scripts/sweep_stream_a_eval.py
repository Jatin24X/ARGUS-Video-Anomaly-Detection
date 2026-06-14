"""Run a no-retrain Stream A evaluation sweep on one checkpoint."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stream_a import (
    SIGNAL_KINDS,
    SIGMA_STRATEGIES,
    collect_split_signal_records,
    collect_train_signal_matrix,
    collect_split_signal_matrix,
    evaluate_stream_a_from_caches,
)
from src.models.scorers.mulde import MULDEScorer
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write ranked sweep rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_sweep_rows(
    train_signal_matrix: dict[str, object],
    split_records: dict[str, object],
    split_name: str,
    signal_kind: str,
    smoothing_sigmas: list[float],
    gmm_components_list: list[int],
    eval_L: int,
) -> list[dict]:
    """Evaluate every configured reduction mode for one signal kind."""
    rows = []
    for sigma_strategy in SIGMA_STRATEGIES:
        if sigma_strategy == "gmm":
            strategy_rows = [
                {
                    "gmm_components": gmm_components,
                    "single_sigma_index": None,
                }
                for gmm_components in gmm_components_list
            ]
        elif sigma_strategy == "single_sigma":
            strategy_rows = [
                {
                    "gmm_components": None,
                    "single_sigma_index": sigma_index,
                }
                for sigma_index in range(eval_L)
            ]
        else:
            strategy_rows = [{"gmm_components": None, "single_sigma_index": None}]

        for strategy_row in strategy_rows:
            for smoothing_sigma in smoothing_sigmas:
                results = evaluate_stream_a_from_caches(
                    train_signal_matrix=train_signal_matrix[signal_kind],
                    split_records=split_records[signal_kind],
                    signal_kind=signal_kind,
                    sigma_strategy=sigma_strategy,
                    gmm_components=strategy_row["gmm_components"],
                    single_sigma_index=strategy_row["single_sigma_index"],
                    smoothing_sigma=smoothing_sigma,
                )
                rows.append(
                    {
                        "phase": f"{split_name}_sweep",
                        "split": split_name,
                        "signal_kind": signal_kind,
                        "sigma_strategy": sigma_strategy,
                        "gmm_components": strategy_row["gmm_components"],
                        "single_sigma_index": strategy_row["single_sigma_index"],
                        "smoothing_sigma": smoothing_sigma,
                        "micro_auc": float(results["micro_auc"]),
                        "macro_auc": float(results["macro_auc"]),
                        "clip_auc": float(results["clip_auc"]),
                    }
                )
    return rows


def _build_holdout_sweep_rows(
    train_signal_matrix: dict[str, object],
    split_signal_matrix: dict[str, object],
    split_name: str,
    signal_kind: str,
    gmm_components_list: list[int],
    eval_L: int,
) -> list[dict]:
    """Evaluate signal reduction configs on a normal-only holdout split."""
    from src.evaluation.stream_a import aggregate_signal_scores

    rows = []
    for sigma_strategy in SIGMA_STRATEGIES:
        if sigma_strategy == "gmm":
            strategy_rows = [
                {
                    "gmm_components": gmm_components,
                    "single_sigma_index": None,
                }
                for gmm_components in gmm_components_list
            ]
        elif sigma_strategy == "single_sigma":
            strategy_rows = [
                {
                    "gmm_components": None,
                    "single_sigma_index": sigma_index,
                }
                for sigma_index in range(eval_L)
            ]
        else:
            strategy_rows = [{"gmm_components": None, "single_sigma_index": None}]

        for strategy_row in strategy_rows:
            holdout_scores, _ = aggregate_signal_scores(
                train_signal_matrix=train_signal_matrix[signal_kind],
                eval_signal_matrix=split_signal_matrix[signal_kind],
                signal_kind=signal_kind,
                sigma_strategy=sigma_strategy,
                gmm_components=strategy_row["gmm_components"],
                single_sigma_index=strategy_row["single_sigma_index"],
            )
            rows.append(
                {
                    "phase": f"{split_name}_holdout_sweep",
                    "split": split_name,
                    "signal_kind": signal_kind,
                    "sigma_strategy": sigma_strategy,
                    "gmm_components": strategy_row["gmm_components"],
                    "single_sigma_index": strategy_row["single_sigma_index"],
                    "normal_holdout_score": float(holdout_scores.mean()),
                    "normal_holdout_p95": float(np.percentile(holdout_scores, 95)),
                    "normal_holdout_std": float(np.std(holdout_scores)),
                    "normal_num_clips": int(len(holdout_scores)),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sweep Stream A evaluation configs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--dataset", default="stream_a_locked")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--test-split", default="test", choices=["test"])
    parser.add_argument(
        "--ranking-mode",
        default="auto",
        choices=["auto", "frame_auc", "normal_holdout"],
        help="How to rank eval configs; auto follows config.training.model_selection_metric",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/reports/stream_a_eval_sweep_val.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/reports/stream_a_eval_sweep_val.csv",
    )
    parser.add_argument(
        "--smoothing-sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Optional override for smoothing sigma sweep values",
    )
    parser.add_argument(
        "--gmm-components-list",
        type=int,
        nargs="+",
        default=None,
        help="Optional override for the GMM component counts to test",
    )
    parser.add_argument(
        "--run-test-on-best",
        action="store_true",
        help="Evaluate the single best val configuration on the test split",
    )
    args = parser.parse_args()

    config = load_config(config_dir=args.config_dir, dataset=args.dataset)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    smoothing_sigmas = (
        list(args.smoothing_sigmas)
        if args.smoothing_sigmas is not None
        else list(getattr(config.evaluation, "sweep_smoothing_sigmas", [0, 1, 3, 5, 7, 10, 15, 20]))
    )
    gmm_components_list = (
        list(args.gmm_components_list)
        if args.gmm_components_list is not None
        else list(getattr(config.evaluation, "sweep_gmm_components", [1, 3, 5]))
    )
    ranking_mode = args.ranking_mode
    if ranking_mode == "auto":
        model_selection_metric = getattr(config.training, "model_selection_metric", "clip_auc")
        ranking_mode = "normal_holdout" if model_selection_metric == "normal_holdout_score" else "frame_auc"
    logger.info("Ranking mode: %s", ranking_mode)

    if ranking_mode == "normal_holdout" and args.split != "val":
        raise ValueError("normal_holdout ranking requires --split val")
    if args.run_test_on_best and args.split != "val":
        raise ValueError("--run-test-on-best is only supported when sweeping --split val")

    scorer = MULDEScorer.load_checkpoint(Path(args.checkpoint), device=device)
    scorer.eval()

    train_signal_matrix = {}
    for signal_kind in SIGNAL_KINDS:
        logger.info("Caching train/%s signals for %s", args.split, signal_kind)
        train_signal_matrix[signal_kind] = collect_train_signal_matrix(
            scorer=scorer,
            config=config,
            device=device,
            signal_kind=signal_kind,
        )

    rows = []
    if ranking_mode == "normal_holdout":
        val_split_signal_matrix = {}
        for signal_kind in SIGNAL_KINDS:
            val_split_signal_matrix[signal_kind] = collect_split_signal_matrix(
                scorer=scorer,
                config=config,
                split=args.split,
                device=device,
                signal_kind=signal_kind,
            )
            rows.extend(
                _build_holdout_sweep_rows(
                    train_signal_matrix=train_signal_matrix,
                    split_signal_matrix=val_split_signal_matrix,
                    split_name=args.split,
                    signal_kind=signal_kind,
                    gmm_components_list=gmm_components_list,
                    eval_L=scorer.eval_L,
                )
            )
        rows.sort(
            key=lambda row: (
                row["normal_holdout_score"],
                row["normal_holdout_p95"],
                row["normal_holdout_std"],
                row["signal_kind"],
                row["sigma_strategy"],
            ),
        )
    else:
        val_split_records = {}
        for signal_kind in SIGNAL_KINDS:
            val_split_records[signal_kind] = collect_split_signal_records(
                scorer=scorer,
                config=config,
                split=args.split,
                device=device,
                signal_kind=signal_kind,
            )
            rows.extend(
                _build_sweep_rows(
                    train_signal_matrix=train_signal_matrix,
                    split_records=val_split_records,
                    split_name=args.split,
                    signal_kind=signal_kind,
                    smoothing_sigmas=smoothing_sigmas,
                    gmm_components_list=gmm_components_list,
                    eval_L=scorer.eval_L,
                )
            )
        rows.sort(
            key=lambda row: (
                row["micro_auc"],
                row["macro_auc"],
                row["clip_auc"],
                row["signal_kind"],
                row["sigma_strategy"],
            ),
            reverse=True,
        )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    best_row = rows[0]
    if ranking_mode == "normal_holdout":
        logger.info(
            "Best val holdout config: signal=%s strategy=%s gmm=%s sigma_idx=%s | "
            "mean=%.6f p95=%.6f std=%.6f clips=%s",
            best_row["signal_kind"],
            best_row["sigma_strategy"],
            best_row["gmm_components"],
            best_row["single_sigma_index"],
            best_row["normal_holdout_score"],
            best_row["normal_holdout_p95"],
            best_row["normal_holdout_std"],
            best_row["normal_num_clips"],
        )
    else:
        logger.info(
            "Best %s config: signal=%s strategy=%s gmm=%s sigma_idx=%s smoothing=%s | "
            "micro=%.4f macro=%.4f clip=%.4f",
            args.split,
            best_row["signal_kind"],
            best_row["sigma_strategy"],
            best_row["gmm_components"],
            best_row["single_sigma_index"],
            best_row["smoothing_sigma"],
            best_row["micro_auc"],
            best_row["macro_auc"],
            best_row["clip_auc"],
        )
    if ranking_mode == "frame_auc":
        logger.info(
            "Best %s config metrics: micro=%.4f macro=%.4f clip=%.4f",
            args.split,
            best_row["micro_auc"],
            best_row["macro_auc"],
            best_row["clip_auc"],
        )
    elif ranking_mode == "normal_holdout":
        logger.info(
            "Best val holdout config metrics: mean=%.6f p95=%.6f std=%.6f clips=%s",
            best_row["normal_holdout_score"],
            best_row["normal_holdout_p95"],
            best_row["normal_holdout_std"],
            best_row["normal_num_clips"],
        )

    if args.run_test_on_best:
        logger.info("Evaluating best val config on %s", args.test_split)
        test_split_records = collect_split_signal_records(
            scorer=scorer,
            config=config,
            split=args.test_split,
            device=device,
            signal_kind=best_row["signal_kind"],
        )
        test_results = evaluate_stream_a_from_caches(
            train_signal_matrix=train_signal_matrix[best_row["signal_kind"]],
            split_records=test_split_records,
            signal_kind=best_row["signal_kind"],
            sigma_strategy=best_row["sigma_strategy"],
            gmm_components=best_row["gmm_components"],
            single_sigma_index=best_row["single_sigma_index"],
            smoothing_sigma=best_row.get(
                "smoothing_sigma",
                float(getattr(config.evaluation, "smoothing_sigma", 20)),
            ),
        )
        rows.append(
            {
                "rank": None,
                "phase": "locked_test",
                "split": args.test_split,
                "signal_kind": best_row["signal_kind"],
                "sigma_strategy": best_row["sigma_strategy"],
                "gmm_components": best_row["gmm_components"],
                "single_sigma_index": best_row["single_sigma_index"],
                "smoothing_sigma": best_row.get(
                    "smoothing_sigma",
                    float(getattr(config.evaluation, "smoothing_sigma", 20)),
                ),
                "micro_auc": float(test_results["micro_auc"]),
                "macro_auc": float(test_results["macro_auc"]),
                "clip_auc": float(test_results["clip_auc"]),
            }
        )
        logger.info(
            "Locked test result: micro=%.4f macro=%.4f clip=%.4f",
            test_results["micro_auc"],
            test_results["macro_auc"],
            test_results["clip_auc"],
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    logger.info("Wrote sweep JSON to %s", output_json)

    output_csv = Path(args.output_csv)
    _write_csv(output_csv, rows)
    logger.info("Wrote sweep CSV to %s", output_csv)


if __name__ == "__main__":
    main()
