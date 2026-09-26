"""Loss functions for Module 00."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy wrapper used by the training scaffold.

    Args:
        logits: Raw scores, shape (B, K).
        targets: Class indices, shape (B,).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (or per-sample if ``reduction="none"``).
    """
    return F.cross_entropy(logits, targets, reduction=reduction)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-class focal loss — student fill.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw scores, shape (B, K).
        targets: Class indices, shape (B,).
        gamma: Focusing parameter; gamma=0 with alpha=None should match CE.
        alpha: Optional per-class weights, shape (K,), applied as alpha[targets].
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar focal loss (or per-sample vector if ``reduction="none"``).
    """
    raise NotImplementedError(
        "Implement focal_loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)"
    )
