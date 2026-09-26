"""Tests that reference solutions satisfy the assignment contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_M00 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_M00.parents[0]))
from _import_isolation import prepare_module_imports

prepare_module_imports(_M00)

import torch
import torch.nn.functional as F

import importlib.util

from dataset import CLASS_NAMES

_loader_path = Path(__file__).resolve().parent / "_solution_loader.py"
_spec = importlib.util.spec_from_file_location("m00_solution_loader", _loader_path)
assert _spec and _spec.loader
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)
load_solution_module = _loader.load_solution_module


class TestAssignmentSolutions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sol_losses = load_solution_module("sol_m00_losses", "losses.py")
        cls.sol_metrics = load_solution_module("sol_m00_metrics", "metrics.py")
        cls.sol_gallery = load_solution_module("sol_m00_error_gallery", "error_gallery.py")

    def test_focal_gamma_zero_matches_ce(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(8, 4)
        targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        fl = self.sol_losses.focal_loss(logits, targets, gamma=0.0)
        ce = F.cross_entropy(logits, targets)
        self.assertTrue(torch.allclose(fl, ce, atol=1e-5))

    def test_focal_downweights_confident_correct(self) -> None:
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([0])
        fl = self.sol_losses.focal_loss(logits, target, gamma=2.0).item()
        ce = F.cross_entropy(logits, target).item()
        self.assertLess(fl, ce)

    def test_focal_less_downweight_on_uncertain(self) -> None:
        confident = torch.tensor([[8.0, 0.0, 0.0, 0.0]])
        uncertain = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        target = torch.tensor([0])
        ce_c = F.cross_entropy(confident, target)
        fl_c = self.sol_losses.focal_loss(confident, target, gamma=2.0)
        ce_u = F.cross_entropy(uncertain, target)
        fl_u = self.sol_losses.focal_loss(uncertain, target, gamma=2.0)
        ratio_conf = (fl_c / ce_c).item()
        ratio_unc = (fl_u / ce_u).item()
        self.assertGreater(ratio_unc, ratio_conf)

    def test_focal_alpha_direction(self) -> None:
        logits = torch.tensor([[2.0, -1.0, -1.0, -1.0], [-1.0, 2.0, -1.0, -1.0]])
        targets = torch.tensor([0, 1])
        alpha = torch.tensor([2.0, 0.5, 1.0, 1.0])
        base = self.sol_losses.focal_loss(logits, targets, gamma=1.0)
        weighted = self.sol_losses.focal_loss(logits, targets, gamma=1.0, alpha=alpha)
        self.assertNotAlmostEqual(base.item(), weighted.item(), places=5)

    def test_minority_recall_known(self) -> None:
        preds = torch.tensor([2, 0, 2, 1])
        targets = torch.tensor([2, 2, 0, 2])
        rec = self.sol_metrics.minority_recall(preds, targets, minority_class=2)
        self.assertAlmostEqual(rec, 1 / 3, places=5)

    def test_error_gallery_order_and_file(self) -> None:
        torch.manual_seed(1)
        images = torch.rand(6, 3, 64, 64)
        targets = torch.tensor([0, 1, 2, 3, 0, 1])
        logits = torch.randn(6, 4)
        logits[2, 0] = 10.0  # wrong but confident for idx 2
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gallery.png"
            result = self.sol_gallery.build_error_gallery(
                images, targets, logits, CLASS_NAMES, out, top_k=4
            )
            self.assertTrue(out.is_file())
            self.assertGreater(result["n_errors"], 0)
            preds = logits.argmax(dim=-1)
            for idx in result["order"]:
                self.assertNotEqual(preds[idx].item(), targets[idx].item())


if __name__ == "__main__":
    unittest.main()
