"""Scaffold tests for Module 00 — always run on student stubs."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_M00 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_M00.parents[0]))
from _import_isolation import prepare_module_imports

prepare_module_imports(_M00)

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import TrainConfig
from dataset import CLASS_NAMES, DrivingPatchDataset, get_dataloaders
from error_gallery import build_error_gallery
from losses import cross_entropy_loss, focal_loss
from metrics import accuracy, per_class_recall, minority_recall
from model import DrivingClassifier
from train import evaluate, train_epoch


class TestScaffold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = TrainConfig()
        cls.data_dir = cls.cfg.data_dir

    def test_dataset_real_pngs(self) -> None:
        ds = DrivingPatchDataset(self.data_dir, split="train")
        self.assertEqual(len(ds), len(ds.rows))
        img, label = ds[0]
        self.assertEqual(tuple(img.shape), (3, 64, 64))
        self.assertTrue(0.0 <= img.min().item() <= img.max().item() <= 1.0)
        self.assertTrue(0 <= label.item() < 4)

    def test_pedestrian_minority_in_train(self) -> None:
        ds = DrivingPatchDataset(self.data_dir, split="train")
        counts = ds.class_counts()
        self.assertLess(counts[2], counts[0], "pedestrian must be minority vs clear_road")

    def test_images_not_constant_across_classes(self) -> None:
        ds = DrivingPatchDataset(self.data_dir, split="train")
        by_class: dict[int, list[float]] = {i: [] for i in range(4)}
        for idx in range(len(ds)):
            img, label = ds[idx]
            by_class[int(label.item())].append(img.mean().item())
        road_mean = sum(by_class[0]) / len(by_class[0])
        veh_mean = sum(by_class[1]) / len(by_class[1])
        self.assertGreater(abs(veh_mean - road_mean), 0.02)

    def test_model_forward(self) -> None:
        model = DrivingClassifier(num_classes=4)
        x = torch.randn(2, 3, 64, 64)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 4))
        feats = model.extract_features(x)
        self.assertEqual(feats.ndim, 2)

    def test_optimizer_step_finite_grads(self) -> None:
        train_loader, _ = get_dataloaders(self.data_dir, batch_size=8, seed=0)
        model = DrivingClassifier()
        opt = optim.SGD(model.parameters(), lr=1e-2)
        images, targets = next(iter(train_loader))
        opt.zero_grad()
        loss = cross_entropy_loss(model(images), targets)
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_train_eval_no_nan(self) -> None:
        train_loader, val_loader = get_dataloaders(self.data_dir, batch_size=8, seed=0)
        model = DrivingClassifier()
        opt = optim.SGD(model.parameters(), lr=1e-3)
        train_epoch(model, train_loader, opt, cross_entropy_loss, torch.device("cpu"))
        loss, acc, recalls = evaluate(model, val_loader, cross_entropy_loss, torch.device("cpu"), 4)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(acc, 0.0)

    def test_accuracy_and_per_class_recall(self) -> None:
        preds = torch.tensor([0, 1, 2, 2])
        targets = torch.tensor([0, 1, 2, 0])
        self.assertAlmostEqual(accuracy(preds, targets), 0.75)
        recalls = per_class_recall(preds, targets, 3)
        self.assertAlmostEqual(recalls[2], 1.0)

    def test_student_focal_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            focal_loss(torch.randn(4, 4), torch.tensor([0, 1, 2, 3]))

    def test_student_minority_recall_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            minority_recall(torch.tensor([0]), torch.tensor([2]), 2)

    def test_student_error_gallery_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            build_error_gallery(
                torch.rand(4, 3, 64, 64),
                torch.tensor([0, 1, 2, 3]),
                torch.randn(4, 4),
                CLASS_NAMES,
                "/tmp/x.png",
            )

    def test_example_metrics_json(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        path = repo / "artifacts" / "m00" / "example" / "metrics.json"
        self.assertTrue(path.is_file())
        data = json.loads(path.read_text())
        for key in (
            "run_id",
            "seed",
            "loss_name",
            "epochs",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "per_class_recall",
            "minority_class",
            "minority_recall",
            "num_train",
            "num_val",
            "image_shape",
        ):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
