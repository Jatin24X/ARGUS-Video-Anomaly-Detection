"""File I/O helpers for ARGUS.

Handles saving/loading features (.npy), checkpoints (.pt), and general I/O.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_features(
    features: np.ndarray,
    path: str | Path,
    dtype: np.dtype = np.float16,
) -> None:
    """Save feature array as .npy file.

    Args:
        features: Feature array to save.
        path: Output path (should end in .npy).
        dtype: Data type for storage (float16 to save space).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), features.astype(dtype))
    logger.info(f"Saved features: {path} | shape={features.shape} | dtype={dtype}")


def load_features(path: str | Path) -> np.ndarray:
    """Load feature array from .npy file.

    Args:
        path: Path to .npy file.

    Returns:
        Loaded numpy array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    features = np.load(str(path))
    logger.debug(f"Loaded features: {path} | shape={features.shape}")
    return features


def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
) -> None:
    """Save model checkpoint.

    Args:
        state: State dict (model, optimizer, epoch, etc.).
        path: Output path (should end in .pt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load model checkpoint.

    Args:
        path: Path to .pt file.

    Returns:
        State dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    logger.info(f"Loaded checkpoint: {path}")
    return state


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Source: implementation_rules.md lines 250-253.

    Args:
        seed: Random seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")
