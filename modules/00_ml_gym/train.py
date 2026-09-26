"""Training loop for Module 00 — Driving ML Gym."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from config import TrainConfig
from dataset import CLASS_NAMES, get_dataloaders
from losses import cross_entropy_loss, focal_loss
from metrics import accuracy, minority_recall, per_class_recall
from model import DrivingClassifier


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _select_loss(cfg: TrainConfig):
    if cfg.loss_name == "focal":
        return lambda logits, targets: focal_loss(logits, targets, gamma=cfg.gamma)
    if cfg.loss_name == "cross_entropy":
        return cross_entropy_loss
    raise ValueError(f"Unknown loss_name: {cfg.loss_name!r}")


def train_epoch(
    model: DrivingClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss ({loss.item()}). Check lr, labels, and loss implementation."
            )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
    return total_loss / max(total, 1)


def evaluate(
    model: DrivingClassifier,
    loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[float]]:
    """Evaluate on a split.

    Returns:
        ``(mean_loss, accuracy, per_class_recall_list)``
    """
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite validation loss ({loss.item()}). Inspect model outputs."
                )
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(targets.cpu())
    preds = torch.cat(all_preds)
    tgts = torch.cat(all_targets)
    return (
        total_loss / max(total, 1),
        accuracy(preds, tgts),
        per_class_recall(preds, tgts, num_classes),
    )


def _save_confusion_matrix(
    model: DrivingClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    out_path: Path,
) -> None:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=-1).cpu().numpy()
            for p, t in zip(preds, targets.numpy()):
                cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(cfg: TrainConfig | None = None) -> dict:
    """Train classifier and write artifacts."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.seed)
    model = DrivingClassifier(num_classes=cfg.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = _select_loss(cfg)

    train_losses: list[float] = []
    val_losses: list[float] = []
    for _ in range(cfg.epochs):
        train_losses.append(train_epoch(model, train_loader, optimizer, criterion, device))
        val_loss, _, _ = evaluate(model, val_loader, criterion, device, cfg.num_classes)
        val_losses.append(val_loss)

    val_loss, val_acc, class_recalls = evaluate(
        model, val_loader, criterion, device, cfg.num_classes
    )

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            all_preds.append(model(images.to(device)).argmax(dim=-1).cpu())
            all_targets.append(targets)
    preds_cat = torch.cat(all_preds)
    tgts_cat = torch.cat(all_targets)

    min_rec: float | None
    try:
        min_rec = minority_recall(preds_cat, tgts_cat, cfg.minority_class)
    except NotImplementedError:
        min_rec = None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_root = _repo_root() / cfg.artifacts_root / run_id
    artifacts_root.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_id": run_id,
        "seed": cfg.seed,
        "loss_name": cfg.loss_name,
        "epochs": cfg.epochs,
        "train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "per_class_recall": {CLASS_NAMES[i]: class_recalls[i] for i in range(cfg.num_classes)},
        "minority_class": cfg.minority_class,
        "minority_recall": min_rec,
        "num_train": len(train_loader.dataset),
        "num_val": len(val_loader.dataset),
        "image_shape": [3, cfg.image_size, cfg.image_size],
    }
    metrics_path = artifacts_root / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _modules_root = Path(__file__).resolve().parents[1]
    if str(_modules_root) not in sys.path:
        sys.path.insert(0, str(_modules_root))
    from common.progress import record_event

    record_event(
        "m00",
        "artifact_exported",
        artifacts=[str(metrics_path)],
        next_session_minutes=25,
    )

    _save_confusion_matrix(
        model, val_loader, device, cfg.num_classes, artifacts_root / "confusion_matrix.png"
    )

    print(f"val_accuracy={val_acc:.3f}  metrics -> {metrics_path}")
    return metrics


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--focal":
        main(TrainConfig(loss_name="focal"))
    else:
        main()
