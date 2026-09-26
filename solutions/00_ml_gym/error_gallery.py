"""Reference error gallery for Module 00 assignment tests."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def build_error_gallery(
    images: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    class_names: list[str],
    out_path: str | Path,
    top_k: int = 16,
) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    wrong = preds != targets
    n_errors = int(wrong.sum().item())

    if n_errors == 0:
        order: list[int] = []
    else:
        conf_wrong = probs[wrong].max(dim=-1).values
        wrong_idx = wrong.nonzero(as_tuple=False).squeeze(1)
        sorted_local = torch.argsort(conf_wrong, descending=True)
        order = wrong_idx[sorted_local].tolist()[:top_k]

    n_show = max(1, min(len(order), top_k)) if order else 1
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes_flat = [axes] if rows == 1 and cols == 1 else list(axes.flat)

    if not order:
        axes_flat[0].text(0.5, 0.5, "No errors", ha="center", va="center")
        axes_flat[0].axis("off")
        for ax in axes_flat[1:]:
            ax.axis("off")
    else:
        for ax, idx in zip(axes_flat, order):
            img = images[idx].permute(1, 2, 0).numpy()
            pred_name = class_names[preds[idx].item()]
            true_name = class_names[targets[idx].item()]
            conf = probs[idx, preds[idx]].item()
            ax.imshow(img.clip(0, 1))
            ax.set_title(f"T:{true_name}\nP:{pred_name} ({conf:.2f})", fontsize=8)
            ax.axis("off")
        for ax in axes_flat[len(order) :]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {"path": str(out_path), "n_errors": n_errors, "order": order}
