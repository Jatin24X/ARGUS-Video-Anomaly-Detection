"""Standalone Stream A training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import VideoMAEClipDataset
from src.evaluation.stream_a import default_eval_params, evaluate_normal_holdout_scorer
from src.models.scorers.mulde import MULDEScorer
from src.training.losses import mulde_loss
from src.training.train_stream import _evaluate_stream_a, train_stream
from src.utils.config import load_config
from src.utils.io import set_seed
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _apply_overrides(config, args) -> None:
    if args.stream_a_beta is not None:
        config.stream_a.beta = args.stream_a_beta
    if args.ema_enabled and args.ema_disabled:
        raise SystemExit("Cannot use --ema-enabled and --ema-disabled together.")
    if args.ema_enabled:
        config.training.ema.enabled = True
    if args.ema_disabled:
        config.training.ema.enabled = False
    if args.ema_decay is not None:
        config.training.ema.decay = float(args.ema_decay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone Stream A (VideoMAE + MULDE)")
    parser.add_argument("--dataset", default="stream_a_locked")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stream-a-beta", type=float, default=None)
    parser.add_argument("--ema-enabled", action="store_true")
    parser.add_argument("--ema-disabled", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=None)
    args = parser.parse_args()

    config = load_config(config_dir=args.config_dir, dataset=args.dataset)
    _apply_overrides(config, args)
    set_seed(int(config.project.seed))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config.data.data_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = data_dir / "metadata"
    features_dir = data_dir / "features" / config.data.dataset / "videomae"
    batch_size = int(getattr(config.stream_a, "batch_size", config.training.batch_size))
    train_split = getattr(config.data, "train_split", "train")
    val_split = getattr(config.data, "val_split", "val")
    selection_metric = getattr(config.training, "model_selection_metric", "clip_auc")
    eval_defaults = default_eval_params(config)

    logger.info("Loading Stream A training dataset...")
    train_dataset = VideoMAEClipDataset(
        features_dir=features_dir,
        metadata_dir=metadata_dir,
        split=train_split,
        mode="train",
        dataset_name=config.data.dataset,
    )
    logger.info("Loading Stream A validation dataset...")
    val_dataset = VideoMAEClipDataset(
        features_dir=features_dir,
        metadata_dir=metadata_dir,
        split=val_split,
        mode="eval",
        dataset_name=config.data.dataset,
    )

    logger.info("Computing training feature statistics...")
    stats_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_train_features = []
    for batch in stats_loader:
        all_train_features.append(batch[0].float())
    all_train_features = torch.cat(all_train_features, dim=0)
    feat_mean = all_train_features.mean(dim=0)
    feat_std = all_train_features.std(dim=0)
    logger.info(
        "Feature stats: mean range [%.4f, %.4f], std range [%.4f, %.4f]",
        feat_mean.min().item(),
        feat_mean.max().item(),
        feat_std.min().item(),
        feat_std.max().item(),
    )

    scorer = MULDEScorer(
        feature_dim=768,
        hidden_dim=int(config.stream_a.hidden_dim),
        sigma_low=float(config.stream_a.sigma_low),
        sigma_high=float(config.stream_a.sigma_high),
        eval_L=int(config.stream_a.eval_L),
        beta=float(config.stream_a.beta),
        gmm_components=int(config.stream_a.gmm_components),
        use_layernorm=bool(getattr(config.stream_a, "layernorm", False)),
    )
    scorer.set_feature_stats(feat_mean, feat_std)

    if selection_metric == "normal_holdout_score":
        def _evaluate_stream_a_normal_holdout(model, val_dataset, device, batch_size):
            del val_dataset, batch_size
            results = evaluate_normal_holdout_scorer(
                scorer=model,
                config=config,
                device=device,
                split=val_split,
                signal_kind=eval_defaults["signal_kind"],
                sigma_strategy=eval_defaults["sigma_strategy"],
                gmm_components=eval_defaults["gmm_components"],
                single_sigma_index=eval_defaults["single_sigma_index"],
            )
            return results

        val_evaluator = _evaluate_stream_a_normal_holdout
        val_metric_name = "normal_holdout_score"
        maximize_val_metric = False
        best_checkpoint_name = "best_holdout.pt"
        validation_interval_epochs = int(
            getattr(config.training, "validation_interval_epochs", 10)
        )
    else:
        val_evaluator = _evaluate_stream_a
        val_metric_name = "clip_val_AUC"
        maximize_val_metric = True
        best_checkpoint_name = "best_clip.pt"
        validation_interval_epochs = int(
            getattr(config.training, "validation_interval_epochs", 10)
        )

    results = train_stream(
        model=scorer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=mulde_loss,
        config=config,
        stream_name="a",
        output_dir=output_dir,
        device=device,
        val_evaluator=val_evaluator,
        val_metric_name=val_metric_name,
        maximize_val_metric=maximize_val_metric,
        best_checkpoint_name=best_checkpoint_name,
        validation_interval_epochs=validation_interval_epochs,
    )

    checkpoint_path = Path(results["checkpoint_path"])
    scorer = MULDEScorer.load_checkpoint(checkpoint_path, device=device)
    scorer.eval()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_log_densities = []
    for batch in train_loader:
        all_log_densities.append(scorer.compute_log_densities(batch[0].to(device)))
    scorer.fit_gmm(np.concatenate(all_log_densities, axis=0))
    scorer.save_checkpoint(checkpoint_path)

    if selection_metric == "normal_holdout_score":
        final_metrics = evaluate_normal_holdout_scorer(
            scorer=scorer,
            config=config,
            device=device,
            split=val_split,
            signal_kind=eval_defaults["signal_kind"],
            sigma_strategy=eval_defaults["sigma_strategy"],
            gmm_components=eval_defaults["gmm_components"],
            single_sigma_index=eval_defaults["single_sigma_index"],
        )
        logger.info(
            "Stream A final normal holdout: mean=%.4f p95=%.4f std=%.4f clips=%s",
            final_metrics["normal_holdout_score"],
            final_metrics["normal_holdout_p95"],
            final_metrics["normal_holdout_std"],
            final_metrics["normal_num_clips"],
        )
    else:
        final_auc = _evaluate_stream_a(scorer, val_dataset, device, batch_size)
        logger.info("Stream A final val clip AUC: %.4f", final_auc)


if __name__ == "__main__":
    main()
