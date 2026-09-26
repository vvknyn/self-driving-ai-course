"""Demo: high accuracy can hide zero minority recall."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or module directory.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import torch

from dataset import CLASS_NAMES
from metrics import accuracy, per_class_recall


def _majority_dominates_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-built batch: 90% clear_road, all pedestrians misclassified."""
    n_road, n_vehicle, n_ped, n_lane = 90, 5, 5, 0
    targets = torch.tensor(
        [0] * n_road + [1] * n_vehicle + [2] * n_ped + [3] * n_lane,
        dtype=torch.long,
    )
    preds = targets.clone()
    preds[targets == 2] = 0  # every pedestrian called clear_road
    return preds, targets


def main() -> None:
    preds, targets = _majority_dominates_batch()
    acc = accuracy(preds, targets)
    recalls = per_class_recall(preds, targets, num_classes=len(CLASS_NAMES))
    ped_idx = CLASS_NAMES.index("pedestrian")

    print("=" * 65)
    print("Break-it: accuracy hides minority failure")
    print("=" * 65)
    print(f"Overall accuracy: {acc * 100:.1f}%  (looks fine)")
    print(f"Pedestrian recall: {recalls[ped_idx] * 100:.1f}%  (safety-critical class ignored)")
    print()
    print("Fix mindset:")
    print("  Principle 2 — The loss/metric defines what 'good' means.")
    print("            Accuracy rewards the majority; track minority recall or use focal loss.")
    print("  Principle 3 — Inspect errors (gallery), not one headline number.")


if __name__ == "__main__":
    main()
