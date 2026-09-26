"""Classification metrics for Module 00."""

from __future__ import annotations

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correct predictions.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground-truth indices, shape (N,).

    Returns:
        Accuracy in [0, 1].
    """
    if targets.numel() == 0:
        return 0.0
    return (preds == targets).float().mean().item()


def per_class_recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> list[float]:
    """Per-class recall = TP / support; 0.0 when support is 0.

    Args:
        preds: Predicted indices, shape (N,).
        targets: Ground-truth indices, shape (N,).
        num_classes: Number of classes K.

    Returns:
        List of length K with recall per class.
    """
    recalls: list[float] = []
    for c in range(num_classes):
        support = (targets == c).sum().item()
        if support == 0:
            recalls.append(0.0)
        else:
            tp = ((preds == c) & (targets == c)).sum().item()
            recalls.append(tp / support)
    return recalls


def minority_recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    minority_class: int,
) -> float:
    """Recall for one minority class — student fill.

    recall = TP / support for ``minority_class``; return 0.0 if support is 0.

    Args:
        preds: Predicted indices, shape (N,).
        targets: Ground-truth indices, shape (N,).
        minority_class: Class id to measure.

    Returns:
        Recall for that class.
    """
    raise NotImplementedError("Implement minority_recall = TP / support for minority_class")
