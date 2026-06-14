"""Standalone Stream A loss wrappers."""

import torch
import torch.nn as nn


def mulde_loss(
    scorer: nn.Module,
    features: torch.Tensor,
    **_: dict,
) -> torch.Tensor:
    return scorer.compute_score_and_loss(features)
