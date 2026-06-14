"""ARGUS - Frame-level evaluation for Stream A."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stream_a import (
    SIGNAL_KINDS,
    SIGMA_STRATEGIES,
    default_eval_params,
    evaluate_checkpoint,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _serialize_results(results: dict) -> dict:
    """Drop non-serializable heavy fields before writing JSON."""
    serialized = {}
    for key, value in results.items():
        if key in {"clip_scores_by_video", "video_num_frames", "video_frame_labels", "fitted_gmm"}:
            continue
        if key == "per_video_auc":
            serialized[key] = {name: float(auc) for name, auc in value.items()}
        elif isinstance(value, float):
            serialized[key] = float(value)
        elif isinstance(value, int):
            serialized[key] = int(value)
        else:
            serialized[key] = value
    return serialized


def main():
    parser = argparse.ArgumentParser(description="Frame-level evaluation for Stream A")
    parser.add_argument("--checkpoint", required=True, help="Path to MULDE checkpoint")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--dataset", default="stream_a_locked")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--signal-kind",
        choices=SIGNAL_KINDS,
        default=None,
        help="Multi-sigma signal to evaluate",
    )
    parser.add_argument(
        "--sigma-strategy",
        choices=SIGMA_STRATEGIES,
        default=None,
        help="How to reduce the L-scale signal into one clip anomaly score",
    )
    parser.add_argument(
        "--aggregation",
        choices=["gmm", "max", "mean", "median", "single_sigma"],
        default=None,
        help="Deprecated alias for --sigma-strategy",
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        default=None,
        help="Number of GMM components when sigma-strategy=gmm",
    )
    parser.add_argument(
        "--single-sigma-index",
        type=int,
        default=None,
        help="Sigma index to use when sigma-strategy=single_sigma",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=None,
        help="Gaussian smoothing sigma for the UBnormal protocol",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write machine-readable evaluation results",
    )
    args = parser.parse_args()

    config = load_config(config_dir=args.config_dir, dataset=args.dataset)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_defaults = default_eval_params(config)

    signal_kind = args.signal_kind or eval_defaults["signal_kind"]
    sigma_strategy = args.sigma_strategy or args.aggregation or eval_defaults["sigma_strategy"]
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

    logger.info("Frame-level evaluation: %s", args.checkpoint)
    logger.info("Split: %s", args.split)
    logger.info("Signal: %s", signal_kind)
    logger.info("Sigma strategy: %s", sigma_strategy)
    if sigma_strategy == "gmm":
        logger.info("GMM components: %s", gmm_components)
    if sigma_strategy == "single_sigma":
        logger.info("Single sigma index: %s", single_sigma_index)
    logger.info("Smoothing: sigma=%s", smoothing_sigma)

    results = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        config=config,
        device=device,
        split=args.split,
        signal_kind=signal_kind,
        sigma_strategy=sigma_strategy,
        gmm_components=gmm_components,
        single_sigma_index=single_sigma_index,
        smoothing_sigma=smoothing_sigma,
    )

    logger.info("=" * 60)
    logger.info("FRAME-LEVEL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Micro AUC:  %.4f", results["micro_auc"])
    logger.info("  Macro AUC:  %.4f", results["macro_auc"])
    logger.info("  Clip AUC:   %.4f", results["clip_auc"])
    logger.info("  Videos:     %s", results["num_videos"])
    logger.info("  Frames:     %s", results["num_frames"])
    logger.info("=" * 60)

    dataset_name = getattr(config.data, "dataset", "unknown")
    logger.info("")
    if dataset_name == "ubnormal":
        logger.info("COMPARISON:")
        logger.info("  Frame-level AUC:   %.4f   (UBnormal benchmark metric)", results["micro_auc"])
        logger.info("  MULDE paper:       0.7280   (frame-level micro, UBnormal)")
        logger.info("  Delta (ours - paper): %+0.4f", results["micro_auc"] - 0.7280)
    else:
        logger.info("DATASET: %s", dataset_name)
        logger.info("  Frame-level AUC:   %.4f", results["micro_auc"])
        logger.info("  Note: dataset-specific paper comparison is not hardcoded here.")
    logger.info("")

    if results.get("per_video_auc"):
        logger.info("Per-video AUCs (top/bottom 5):")
        sorted_vids = sorted(results["per_video_auc"].items(), key=lambda item: item[1], reverse=True)
        for name, auc in sorted_vids[:5]:
            logger.info("  ^ %s: %.4f", name, auc)
        if len(sorted_vids) > 10:
            logger.info("  ...")
        for name, auc in sorted_vids[-5:]:
            logger.info("  v %s: %.4f", name, auc)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(_serialize_results(results), handle, indent=2, sort_keys=True)
        logger.info("Wrote evaluation JSON to %s", output_path)


if __name__ == "__main__":
    main()
