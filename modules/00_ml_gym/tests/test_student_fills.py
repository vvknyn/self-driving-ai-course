"""Optional checks when student fills are implemented."""

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

from dataset import CLASS_NAMES
from error_gallery import build_error_gallery
from losses import focal_loss
from metrics import minority_recall


class TestStudentFills(unittest.TestCase):
    def test_focal_loss_if_implemented(self) -> None:
        try:
            logits = torch.randn(4, 4)
            targets = torch.tensor([0, 1, 2, 3])
            fl = focal_loss(logits, targets, gamma=0.0)
        except NotImplementedError:
            self.skipTest("focal_loss not implemented yet")
        ce = F.cross_entropy(logits, targets)
        self.assertTrue(torch.allclose(fl, ce, atol=1e-5))

    def test_minority_recall_if_implemented(self) -> None:
        try:
            rec = minority_recall(
                torch.tensor([2, 0, 2, 1]),
                torch.tensor([2, 2, 0, 2]),
                minority_class=2,
            )
        except NotImplementedError:
            self.skipTest("minority_recall not implemented yet")
        self.assertAlmostEqual(rec, 1 / 3, places=5)

    def test_error_gallery_if_implemented(self) -> None:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "g.png"
                result = build_error_gallery(
                    torch.rand(4, 3, 64, 64),
                    torch.tensor([0, 1, 2, 3]),
                    torch.randn(4, 4),
                    CLASS_NAMES,
                    out,
                )
        except NotImplementedError:
            self.skipTest("build_error_gallery not implemented yet")
        self.assertTrue(Path(result["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
