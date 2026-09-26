"""Reference metrics for Module 00 assignment tests."""

from __future__ import annotations

import torch


def minority_recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    minority_class: int,
) -> float:
    support = (targets == minority_class).sum().item()
    if support == 0:
        return 0.0
    tp = ((preds == minority_class) & (targets == minority_class)).sum().item()
    return tp / support
