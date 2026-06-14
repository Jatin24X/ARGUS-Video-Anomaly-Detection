"""YAML config loader for ARGUS.

Loads default.yaml and merges with dataset-specific overrides.
All hyperparameters must come from config, not hardcoded (Rule R1).

Source: architecture_detail.md Gap 4.2
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _dict_to_namespace(value)
    return SimpleNamespace(**d)


def load_config(
    config_dir: str = "configs",
    dataset: str = "ubnormal",
) -> SimpleNamespace:
    """Load default config and merge with dataset-specific overrides.

    Args:
        config_dir: Path to configs directory.
        dataset: Dataset name (loads {dataset}.yaml as override).

    Returns:
        SimpleNamespace with dot-access to all config values.
    """
    config_path = Path(config_dir)

    # Load default config
    default_path = config_path / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    overrides = _load_dataset_overrides(config_path, dataset, seen={dataset})
    config = _deep_merge(config, overrides)
    config = _resolve_relative_paths(config, project_root=config_path.parent)
    return _dict_to_namespace(config)


def _load_dataset_overrides(config_path: Path, dataset: str, seen: set[str]) -> dict:
    """Load one dataset override, resolving optional inheritance."""
    override_path = config_path / f"{dataset}.yaml"
    if not override_path.exists():
        return {}

    with open(override_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    inherits = overrides.pop("inherits", None)
    if inherits is None:
        return overrides

    parent_names = [inherits] if isinstance(inherits, str) else list(inherits)
    merged: dict = {}
    for parent_name in parent_names:
        if parent_name in seen:
            chain = " -> ".join([*seen, parent_name])
            raise ValueError(f"Config inheritance cycle detected: {chain}")
        parent_overrides = _load_dataset_overrides(
            config_path,
            parent_name,
            seen={*seen, parent_name},
        )
        merged = _deep_merge(merged, parent_overrides)
    return _deep_merge(merged, overrides)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_relative_paths(config: dict, *, project_root: Path) -> dict:
    """Resolve path-like config values relative to the standalone project root."""
    resolved = config.copy()
    data_cfg = resolved.get("data")
    if isinstance(data_cfg, dict) and "data_dir" in data_cfg:
        data_dir = Path(str(data_cfg["data_dir"]))
        if not data_dir.is_absolute():
            data_cfg = data_cfg.copy()
            data_cfg["data_dir"] = str((project_root / data_dir).resolve())
            resolved["data"] = data_cfg
    return resolved
