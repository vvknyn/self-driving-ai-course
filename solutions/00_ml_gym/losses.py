"""Reference focal loss for Module 00 assignment tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Stable focal loss via log_softmax."""
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1.0 - pt) ** gamma) * log_pt
    if alpha is not None:
        alpha_t = alpha.to(logits.device)[targets]
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
