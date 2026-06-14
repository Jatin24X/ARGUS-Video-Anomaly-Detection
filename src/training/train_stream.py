"""ARGUS - Generic stream training loop."""

from __future__ import annotations

import os
import csv
from contextlib import nullcontext
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _namespace_to_dict(value) -> dict:
    """Convert a config namespace/dict into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return dict(vars(value))


def _coerce_scalar_metric(value):
    """Convert a metric value into a JSON/CSV-safe Python scalar."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value.item())
    if isinstance(value, (float, int, bool, str)):
        return value
    return None


def _coerce_validation_output(
    value,
    *,
    val_metric_name: str,
) -> tuple[float, dict]:
    """Normalize evaluator output into a primary metric plus flat metric record."""
    if isinstance(value, dict):
        if "primary_metric" in value:
            primary_metric = float(value["primary_metric"])
        elif val_metric_name in value:
            primary_metric = float(value[val_metric_name])
        else:
            raise ValueError(
                f"Validation output is missing primary metric '{val_metric_name}'."
            )

        metric_record = {}
        for key, raw_value in value.items():
            scalar = _coerce_scalar_metric(raw_value)
            if scalar is not None:
                metric_record[key] = scalar
        metric_record.setdefault("primary_metric", primary_metric)
        metric_record.setdefault(val_metric_name, primary_metric)
        return primary_metric, metric_record

    primary_metric = float(value)
    return primary_metric, {
        "primary_metric": primary_metric,
        val_metric_name: primary_metric,
    }


def _validation_sort_key(
    metric_record: dict,
    *,
    val_metric_name: str,
    maximize_val_metric: bool,
) -> tuple:
    """Build a stable comparison key for checkpoint promotion."""
    primary_value = float(metric_record.get(val_metric_name, metric_record["primary_metric"]))
    if not maximize_val_metric:
        if val_metric_name == "normal_holdout_score":
            p95_value = float(metric_record.get("normal_holdout_p95", float("inf")))
            std_value = float(metric_record.get("normal_holdout_std", float("inf")))
            clip_count = int(metric_record.get("normal_num_clips", 0))
            return (-primary_value, -p95_value, -std_value, clip_count)
        return (-primary_value,)

    if val_metric_name != "frame_micro_auc":
        return (primary_value,)

    macro_value = float(
        metric_record.get(
            "frame_macro_auc_benchmark",
            metric_record.get("frame_macro_auc", float("-inf")),
        )
    )
    window_value = float(metric_record.get("window_auc", float("-inf")))
    coverage_value = float(metric_record.get("coverage_ratio", float("-inf")))
    unscored_value = -float(
        metric_record.get("unscored_abnormal_fraction", float("inf"))
    )
    return (
        primary_value,
        macro_value,
        window_value,
        coverage_value,
        unscored_value,
    )


def _preferred_history_fieldnames(records: list[dict]) -> list[str]:
    """Return a stable CSV column order for validation history."""
    preferred = [
        "epoch",
        "train_loss",
        "learning_rate",
        "elapsed_sec",
        "primary_metric",
        "clip_val_AUC",
        "normal_holdout_score",
        "normal_holdout_p95",
        "normal_holdout_std",
        "normal_num_clips",
        "frame_micro_auc",
        "frame_macro_auc",
        "frame_macro_auc_raw_official",
        "frame_macro_auc_benchmark",
        "window_auc",
        "raw_window_auc",
        "coverage_ratio",
        "unscored_abnormal_fraction",
        "num_unscored_frames",
        "num_unscored_abnormal_frames",
    ]
    observed = {key for record in records for key in record.keys()}
    ordered = [key for key in preferred if key in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def _write_validation_history(
    records: list[dict],
    *,
    csv_path: Path,
    jsonl_path: Path,
) -> None:
    """Persist validation history in CSV + JSONL form for run-to-run comparisons."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    fieldnames = _preferred_history_fieldnames(records)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _sample_group_key(metadata, sampling_mode: str):
    """Map one dataset sample metadata entry to a balancing bucket."""
    if isinstance(metadata, dict):
        if sampling_mode in {
            "scene_balanced",
            "scene_weighted",
            "scene_weighted_adaptive_capped",
        }:
            if "scene_id" not in metadata:
                raise ValueError(
                    f"{sampling_mode} sampling requires scene_id metadata"
                )
            return ("scene", int(metadata["scene_id"]))
        if "video_name" not in metadata:
            raise ValueError("video_balanced sampling requires video_name metadata")
        return ("video", str(metadata["video_name"]))

    if isinstance(metadata, tuple):
        if sampling_mode in {
            "scene_balanced",
            "scene_weighted",
            "scene_weighted_adaptive_capped",
        }:
            raise ValueError(
                f"{sampling_mode} sampling is not supported for tuple metadata datasets"
            )
        if not metadata:
            raise ValueError("Empty tuple metadata cannot be balanced")
        return ("video", str(metadata[0]))

    raise ValueError(f"Unsupported sample_metadata entry type: {type(metadata)!r}")


def _load_group_weight_overrides(weight_path: str, *, sampling_mode: str) -> dict[str, float]:
    """Load optional per-group weighting overrides from JSON."""
    payload = json.loads(Path(weight_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sampling weight file must contain a JSON object: {weight_path}")

    preferred_keys = {
        "scene_weighted": ("scene_weights", "scene_weight_overrides"),
        "scene_weighted_adaptive_capped": ("scene_weights", "scene_weight_overrides"),
        "video_weighted": ("video_weights", "video_weight_overrides"),
    }
    raw_weights = None
    for key in preferred_keys.get(sampling_mode, ()):
        if key in payload:
            raw_weights = payload[key]
            break
    if raw_weights is None:
        raw_weights = payload
    if not isinstance(raw_weights, dict):
        raise ValueError(
            f"Sampling weights in {weight_path} must be a JSON object of group->weight."
        )

    return {str(key): float(value) for key, value in raw_weights.items()}


def _build_sample_weights(
    dataset: Dataset,
    *,
    sampling_mode: str,
    group_weight_overrides: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Build per-sample weights for balanced training sampling."""
    sample_metadata = getattr(dataset, "sample_metadata", None)
    if not sample_metadata:
        raise ValueError(
            f"sampling_mode={sampling_mode!r} requires dataset.sample_metadata"
        )

    base_keys = [_sample_group_key(metadata, sampling_mode) for metadata in sample_metadata]
    group_counts = {}
    for key in base_keys:
        group_counts[key] = group_counts.get(key, 0) + 1

    weights = np.empty(len(dataset), dtype=np.float64)
    base_count = len(base_keys)
    for dataset_index in range(len(dataset)):
        base_index = dataset_index % base_count
        group_key = base_keys[base_index]
        group_multiplier = 1.0
        if group_weight_overrides is not None:
            group_multiplier = float(group_weight_overrides.get(str(group_key[1]), 1.0))
        weights[dataset_index] = group_multiplier / float(group_counts[group_key])
    return weights


def _build_train_loader(
    train_dataset: Dataset,
    *,
    batch_size: int,
    device: str,
    sampling_mode: str,
    loader_config: dict,
    sampling_weights_json: Optional[str] = None,
) -> DataLoader:
    """Construct the train loader with optional balanced sampling."""
    if sampling_mode not in {
        "uniform",
        "video_balanced",
        "scene_balanced",
        "scene_weighted",
        "scene_weighted_adaptive_capped",
    }:
        raise ValueError(
            f"Unsupported train_sampling_mode={sampling_mode!r}. "
            "Expected one of {'uniform', 'video_balanced', 'scene_balanced', "
            "'scene_weighted', 'scene_weighted_adaptive_capped'}."
        )

    loader_kwargs = _dataloader_kwargs(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=False,
        **loader_config,
    )
    if sampling_mode == "uniform":
        return DataLoader(
            shuffle=True,
            **loader_kwargs,
        )

    group_weight_overrides = None
    if sampling_mode in {"scene_weighted", "scene_weighted_adaptive_capped"}:
        if not sampling_weights_json:
            raise ValueError(
                f"{sampling_mode} sampling requires stream_b.train_sampling_weights_json"
            )
        group_weight_overrides = _load_group_weight_overrides(
            sampling_weights_json,
            sampling_mode=sampling_mode,
        )

    sample_weights = _build_sample_weights(
        train_dataset,
        sampling_mode=sampling_mode,
        group_weight_overrides=group_weight_overrides,
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        shuffle=False,
        sampler=sampler,
        **loader_kwargs,
    )


def _dataloader_kwargs(**kwargs) -> dict:
    """Normalize optional DataLoader kwargs for worker/no-worker cases."""
    num_workers = int(kwargs.pop("num_workers", 0))
    pin_memory = bool(kwargs.pop("pin_memory", False))
    persistent_workers = bool(kwargs.pop("persistent_workers", False))
    prefetch_factor = kwargs.pop("prefetch_factor", None)
    kwargs.pop("non_blocking", False)
    loader_kwargs = {
        **kwargs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return loader_kwargs


def _resolve_execution_profile(
    stream_name: str,
    stream_config,
    training_config,
) -> str:
    """Resolve one execution profile with a Stream B-focused default."""
    profile = getattr(
        stream_config,
        "execution_profile",
        getattr(training_config, "execution_profile", "confirm"),
    )
    if stream_name != "b":
        return "default"
    if profile not in {"default", "search", "confirm"}:
        raise ValueError(
            f"Unsupported execution_profile={profile!r}. "
            "Expected one of {'default', 'search', 'confirm'}."
        )
    return profile


def _resolve_loader_config(
    *,
    device: str,
    execution_profile: str,
    stream_config,
    training_config,
) -> dict:
    """Resolve loader worker / pin-memory settings for one run."""
    configured_workers = getattr(
        stream_config,
        "loader_num_workers",
        getattr(training_config, "loader_num_workers", None),
    )
    if configured_workers is None:
        if execution_profile in {"search", "confirm"} and device.startswith("cuda"):
            cpu_count = os.cpu_count() or 8
            configured_workers = max(4, min(8, cpu_count // 2))
        else:
            configured_workers = 0
    configured_workers = int(configured_workers)

    configured_prefetch = getattr(
        stream_config,
        "loader_prefetch_factor",
        getattr(training_config, "loader_prefetch_factor", 4),
    )
    persistent_workers = bool(
        getattr(
            stream_config,
            "loader_persistent_workers",
            getattr(training_config, "loader_persistent_workers", configured_workers > 0),
        )
    )
    pin_memory = bool(
        getattr(
            stream_config,
            "loader_pin_memory",
            getattr(training_config, "loader_pin_memory", device.startswith("cuda")),
        )
    )
    non_blocking = bool(
        getattr(
            stream_config,
            "loader_non_blocking",
            getattr(training_config, "loader_non_blocking", device.startswith("cuda")),
        )
    )
    return {
        "num_workers": configured_workers,
        "prefetch_factor": configured_prefetch,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "non_blocking": non_blocking,
    }


def _resolve_amp_dtype(
    *,
    device: str,
    stream_config,
    training_config,
) -> str:
    """Resolve mixed precision mode for one run."""
    amp_dtype = getattr(
        stream_config,
        "amp_dtype",
        getattr(training_config, "amp_dtype", "off"),
    )
    if not device.startswith("cuda"):
        return "off"
    if amp_dtype not in {"off", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported amp_dtype={amp_dtype!r}. Expected one of {'off', 'fp16', 'bf16'}."
        )
    return amp_dtype


class _EMAHelper:
    """Minimal EMA tracker for evaluation-time weights."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self._backup_state = None

    def update(self, model: nn.Module) -> None:
        current_state = model.state_dict()
        for key, value in current_state.items():
            shadow_value = self.shadow_state[key]
            if torch.is_floating_point(value):
                shadow_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value.detach())

    def store(self, model: nn.Module) -> None:
        self._backup_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def copy_to_model(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow_state, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self._backup_state is None:
            return
        model.load_state_dict(self._backup_state, strict=True)
        self._backup_state = None


def _with_eval_weights(model: nn.Module, ema_helper: Optional[_EMAHelper], fn: Callable):
    """Run a callable under EMA weights when enabled."""
    if ema_helper is None:
        return fn()

    ema_helper.store(model)
    ema_helper.copy_to_model(model)
    try:
        return fn()
    finally:
        ema_helper.restore(model)


def train_stream(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    loss_fn: Callable,
    config: object,
    stream_name: str = "a",
    output_dir: Path = Path("outputs"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_evaluator: Optional[Callable[[nn.Module, Dataset, str, int], float]] = None,
    val_metric_name: str = "val_metric",
    maximize_val_metric: bool = True,
    best_checkpoint_name: str = "best_val.pt",
    validation_interval_epochs: int = 10,
) -> dict:
    """Train one stream scorer with optional diagnostic validation."""
    stream_config = getattr(config, f"stream_{stream_name}", config.training)
    training_config = config.training

    epochs = getattr(stream_config, "epochs", training_config.epochs)
    lr = getattr(stream_config, "learning_rate", training_config.learning_rate)
    batch_size = getattr(stream_config, "batch_size", training_config.batch_size)
    optimizer_type = getattr(stream_config, "optimizer", getattr(training_config, "optimizer", "adam"))
    adam_betas = tuple(getattr(training_config, "adam_betas", [0.9, 0.999]))
    weight_decay = float(
        getattr(stream_config, "weight_decay", getattr(training_config, "weight_decay", 0.0))
    )
    scheduler_type = getattr(stream_config, "scheduler", getattr(training_config, "scheduler", "none"))
    scheduler_kwargs = _namespace_to_dict(
        getattr(stream_config, "scheduler_kwargs", getattr(training_config, "scheduler_kwargs", {}))
    )
    grad_clip = getattr(
        stream_config,
        "grad_clip",
        getattr(training_config, "grad_clip", None),
    )
    checkpoint_interval_epochs = getattr(
        stream_config,
        "checkpoint_interval_epochs",
        getattr(training_config, "checkpoint_interval_epochs", 0),
    )
    train_sampling_mode = getattr(stream_config, "train_sampling_mode", "uniform")
    model_selection_metric = getattr(
        stream_config,
        "model_selection_metric",
        getattr(training_config, "model_selection_metric", "clip_auc"),
    )
    ema_config = getattr(training_config, "ema", None)
    ema_enabled = bool(getattr(ema_config, "enabled", False)) if ema_config else False
    ema_decay = float(getattr(ema_config, "decay", 0.999)) if ema_config else 0.999
    execution_profile = _resolve_execution_profile(stream_name, stream_config, training_config)
    loader_config = _resolve_loader_config(
        device=device,
        execution_profile=execution_profile,
        stream_config=stream_config,
        training_config=training_config,
    )
    amp_dtype = _resolve_amp_dtype(
        device=device,
        stream_config=stream_config,
        training_config=training_config,
    )
    if stream_name == "b":
        if execution_profile == "search":
            epochs = min(int(epochs), 6)
            validation_interval_epochs = 1
            checkpoint_interval_epochs = 1
        elif execution_profile == "confirm":
            epochs = max(int(epochs), 8)
            validation_interval_epochs = 1
            checkpoint_interval_epochs = max(int(checkpoint_interval_epochs), 1)

    logger.info("\n%s", "=" * 70)
    logger.info("Training Stream %s", stream_name.upper())
    logger.info("%s", "=" * 70)
    logger.info("  Device:     %s", device)
    logger.info("  Epochs:     %s", epochs)
    logger.info("  LR:         %s", lr)
    logger.info("  Batch size: %s", batch_size)
    logger.info("  Optimizer:  %s", optimizer_type)
    logger.info("  Adam betas: %s", adam_betas)
    logger.info("  Weight decay: %s", weight_decay)
    logger.info("  Scheduler:  %s", scheduler_type)
    logger.info("  Grad clip:  %s", grad_clip)
    logger.info("  EMA:        %s", f"enabled (decay={ema_decay})" if ema_enabled else "disabled")
    logger.info("  Ckpt every: %s", checkpoint_interval_epochs or "off")
    logger.info("  Select by:  %s", model_selection_metric)
    logger.info("  Sampling:   %s", train_sampling_mode)
    logger.info("  Profile:    %s", execution_profile)
    logger.info("  AMP:        %s", amp_dtype)
    logger.info(
        "  Loader:     workers=%s persistent=%s prefetch=%s pin_memory=%s non_blocking=%s",
        loader_config["num_workers"],
        loader_config["persistent_workers"],
        loader_config["prefetch_factor"],
        loader_config["pin_memory"],
        loader_config["non_blocking"],
    )
    logger.info("  Train samples: %s", len(train_dataset))
    if val_dataset:
        logger.info("  Val samples:   %s", len(val_dataset))

    model = model.to(device)
    ema_helper = _EMAHelper(model, ema_decay) if ema_enabled else None

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=adam_betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamax":
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer={optimizer_type!r}. Expected adam, adamax, or sgd."
        )

    scheduler = None
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_kwargs.get("step_size", 50)),
            gamma=float(scheduler_kwargs.get("gamma", 0.9)),
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_kwargs.get("T_max", epochs)),
            eta_min=float(scheduler_kwargs.get("eta_min", 0.0)),
        )
    elif scheduler_type == "exp_decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(scheduler_kwargs.get("gamma", 0.99)),
        )
    elif scheduler_type != "none":
        raise ValueError(
            f"Unsupported scheduler={scheduler_type!r}. Expected none, step, cosine, or exp_decay."
        )

    train_loader = _build_train_loader(
        train_dataset,
        batch_size=batch_size,
        device=device,
        sampling_mode=train_sampling_mode,
        loader_config=loader_config,
        sampling_weights_json=getattr(stream_config, "train_sampling_weights_json", None),
    )
    amp_enabled = device.startswith("cuda") and amp_dtype in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    try:
        grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(amp_enabled and amp_dtype == "fp16"),
        )
    except AttributeError:
        grad_scaler = torch.cuda.amp.GradScaler(
            enabled=bool(amp_enabled and amp_dtype == "fp16")
        )

    checkpoint_dir = output_dir / "checkpoints" / f"stream_{stream_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_checkpoint_dir = checkpoint_dir / "epochs"
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    validation_history_csv_path = reports_dir / f"stream_{stream_name}_validation_history.csv"
    validation_history_jsonl_path = reports_dir / f"stream_{stream_name}_validation_history.jsonl"

    best_val_metric = -float("inf") if maximize_val_metric else float("inf")
    best_val_epoch = 0
    best_metric_record = None
    train_losses = []
    val_metrics = []
    validation_history = []
    periodic_checkpoint_paths = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_losses = []

        for batch in train_loader:
            features = batch[0].to(device, non_blocking=loader_config["non_blocking"])
            segment_scores = (
                batch[2].to(device, non_blocking=loader_config["non_blocking"])
                if len(batch) > 2
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                if segment_scores is None:
                    loss = loss_fn(
                        model,
                        features,
                        loss_context={"stream_config": stream_config},
                    )
                else:
                    loss = loss_fn(
                        model,
                        features,
                        segment_scores,
                        loss_context={"stream_config": stream_config},
                    )
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if grad_clip is not None:
                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            if ema_helper is not None:
                ema_helper.update(model)
            epoch_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        avg_loss = float(np.mean(epoch_losses))
        train_losses.append(avg_loss)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        if checkpoint_interval_epochs and epoch % checkpoint_interval_epochs == 0:
            epoch_path = epoch_checkpoint_dir / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(model, epoch_path, ema_helper=ema_helper)
            periodic_checkpoint_paths.append(str(epoch_path))

        should_validate = (
            val_dataset is not None
            and val_evaluator is not None
            and (epoch % validation_interval_epochs == 0 or epoch == epochs)
        )

        if should_validate:
            val_output = _with_eval_weights(
                model,
                ema_helper,
                lambda: val_evaluator(model, val_dataset, device, batch_size),
            )
            val_metric, metric_record = _coerce_validation_output(
                val_output,
                val_metric_name=val_metric_name,
            )
            val_metrics.append(float(val_metric))
            validation_row = {
                "epoch": int(epoch),
                "train_loss": float(avg_loss),
                "learning_rate": float(current_lr),
                "elapsed_sec": float(elapsed),
                **metric_record,
            }
            validation_history.append(validation_row)
            _write_validation_history(
                validation_history,
                csv_path=validation_history_csv_path,
                jsonl_path=validation_history_jsonl_path,
            )
            improved = (
                _validation_sort_key(
                    metric_record,
                    val_metric_name=val_metric_name,
                    maximize_val_metric=maximize_val_metric,
                )
                > _validation_sort_key(
                    best_metric_record,
                    val_metric_name=val_metric_name,
                    maximize_val_metric=maximize_val_metric,
                )
                if best_metric_record is not None
                else True
            )

            if improved:
                best_val_metric = float(val_metric)
                best_val_epoch = epoch
                best_metric_record = dict(metric_record)
                _save_checkpoint(model, checkpoint_dir / best_checkpoint_name, ema_helper=ema_helper)
                logger.info(
                    "  Epoch %3s/%s | loss=%.4f | %s=%.4f * BEST | lr=%.6f | %.1fs",
                    epoch,
                    epochs,
                    avg_loss,
                    val_metric_name,
                    val_metric,
                    current_lr,
                    elapsed,
                )
            else:
                logger.info(
                    "  Epoch %3s/%s | loss=%.4f | %s=%.4f | lr=%.6f | %.1fs",
                    epoch,
                    epochs,
                    avg_loss,
                    val_metric_name,
                    val_metric,
                    current_lr,
                    elapsed,
                )
        elif epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3s/%s | loss=%.4f | lr=%.6f | %.1fs",
                epoch,
                epochs,
                avg_loss,
                current_lr,
                elapsed,
            )

    last_checkpoint_path = checkpoint_dir / "last.pt"
    _save_checkpoint(model, last_checkpoint_path, ema_helper=ema_helper)

    logger.info("\n%s", "-" * 70)
    logger.info("Training complete - Stream %s", stream_name.upper())
    if best_val_epoch > 0:
        logger.info("  Best %s: %.4f (epoch %s)", val_metric_name, best_val_metric, best_val_epoch)
    logger.info("  Final loss: %.4f", train_losses[-1])
    logger.info("  Checkpoints: %s", checkpoint_dir)
    if validation_history:
        logger.info("  Val history: %s", validation_history_csv_path)
    if stream_name == "b" and val_metric_name != "val_mean_nll":
        logger.info("  Benchmark selection: online via %s", val_metric_name)
    elif model_selection_metric != "clip_auc":
        logger.info("  Benchmark selection: use frame-level selector on %s", epoch_checkpoint_dir)
    logger.info("%s", "-" * 70)

    best_checkpoint_path = checkpoint_dir / best_checkpoint_name
    preferred_checkpoint = best_checkpoint_path if best_val_epoch > 0 else last_checkpoint_path

    results = {
        "best_val_metric": best_val_metric if best_val_epoch > 0 else None,
        "best_epoch": best_val_epoch,
        "best_val_metric_name": val_metric_name,
        "final_train_loss": train_losses[-1],
        "epochs_trained": epochs,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "validation_history": validation_history,
        "checkpoint_path": str(preferred_checkpoint),
        "best_validation_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "periodic_checkpoint_paths": periodic_checkpoint_paths,
        "validation_history_csv_path": str(validation_history_csv_path),
        "validation_history_jsonl_path": str(validation_history_jsonl_path),
        "execution_profile": execution_profile,
        "amp_dtype": amp_dtype,
        "loader_config": dict(loader_config),
    }

    if stream_name == "a" and val_metric_name == "clip_val_AUC":
        results["best_clip_val_auc"] = best_val_metric if best_val_epoch > 0 else None
        results["best_clip_epoch"] = best_val_epoch
        results["best_clip_checkpoint_path"] = str(best_checkpoint_path)
    elif stream_name == "a":
        results["best_stream_a_val_metric"] = best_val_metric if best_val_epoch > 0 else None
        results["best_stream_a_val_metric_name"] = val_metric_name

    return results


def _evaluate_stream_a(
    model: nn.Module,
    val_dataset: Dataset,
    device: str,
    batch_size: int,
) -> float:
    """Evaluate a MULDE scorer on clip-level validation AUC."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    all_log_densities = []
    all_labels = []
    for batch in val_loader:
        features = batch[0].to(device)
        labels = batch[1]
        log_densities = model.compute_log_densities(features)
        all_log_densities.append(log_densities)
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    all_log_densities = np.concatenate(all_log_densities, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        logger.warning("Validation set has only one class - AUC undefined")
        return 0.5

    if model._gmm is not None:
        anomaly_scores = -model._gmm.score_samples(all_log_densities)
    else:
        anomaly_scores = -all_log_densities.mean(axis=1)

    try:
        return float(roc_auc_score(all_labels, anomaly_scores))
    except ValueError:
        return 0.5


def _evaluate_stream_b_nll(
    model: nn.Module,
    val_dataset: Dataset,
    device: str,
    batch_size: int,
) -> float:
    """Evaluate Stream B by mean validation NLL (lower is better)."""
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    all_nll = []
    with torch.no_grad():
        for batch in val_loader:
            segments = batch[0].to(device)
            segment_scores = batch[2].to(device) if len(batch) > 2 else None
            nll = model.negative_log_likelihood(
                segments,
                segment_scores=segment_scores,
                reduction="none",
            )
            all_nll.append(nll.cpu().numpy())

    if not all_nll:
        return float("inf")
    return float(np.concatenate(all_nll, axis=0).mean())


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    ema_helper: Optional[_EMAHelper] = None,
) -> None:
    """Save a checkpoint, delegating to the model when supported."""

    def _save() -> None:
        if hasattr(model, "save_checkpoint"):
            model.save_checkpoint(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)

    _with_eval_weights(model, ema_helper, _save)
