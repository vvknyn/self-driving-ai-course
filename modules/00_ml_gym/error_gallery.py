"""Misclassification gallery — from-scratch student assignment."""

from __future__ import annotations

from pathlib import Path

import torch


def build_error_gallery(
    images: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    class_names: list[str],
    out_path: str | Path,
    top_k: int = 16,
) -> dict:
    """Build a grid of highest-confidence mistakes — student from-scratch.

    Args:
        images: Batch in [0, 1], shape (N, C, H, W).
        targets: Ground-truth labels, shape (N,).
        logits: Raw scores, shape (N, K).
        class_names: Human-readable names length K.
        out_path: Where to save the figure PNG.
        top_k: Maximum mistakes to show.

    Returns:
        Dict with keys ``path``, ``n_errors``, ``order`` (indices into N batch,
        highest-confidence wrong predictions first). If zero errors, still write
        a figure and return ``n_errors=0``, ``order=[]``.
    """
    raise NotImplementedError(
        "From scratch: save misclassifications sorted by confidence of wrong class"
    )
