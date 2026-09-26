"""Tests for learner progress tracking."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULES_ROOT = _REPO_ROOT / "modules"
if str(_MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULES_ROOT))

from common.progress import (
    come_back_cue,
    format_stack,
    principle_is_written,
    record_event,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class TestPrincipleIsWritten(unittest.TestCase):
    def test_placeholder_rejected(self) -> None:
        self.assertFalse(principle_is_written("_Write 3–6 sentences here."))
        self.assertFalse(principle_is_written("   "))
        self.assertFalse(principle_is_written(""))

    def test_real_text_accepted(self) -> None:
        self.assertTrue(principle_is_written("Focal loss down-weights easy road crops."))


class TestRecordEvent(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "progress.json"
        self.t0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_notebook_opened_no_xp_no_streak(self) -> None:
        doc = record_event("m00", "notebook_opened", now=self.t0, path=self.path)
        self.assertEqual(doc["xp"], 0)
        self.assertEqual(doc["streak_days"], 0)

    def test_artifact_exported_xp_streak_next_module(self) -> None:
        doc = record_event(
            "m00",
            "artifact_exported",
            artifacts=["artifacts/m00/example/metrics.json"],
            now=self.t0,
            path=self.path,
        )
        self.assertEqual(doc["xp"], 10)
        self.assertEqual(doc["streak_days"], 1)
        session = doc["sessions"][-1]
        self.assertEqual(session["next_module_id"], "m01")
        for key in (
            "module_id",
            "completed_at",
            "streak_days",
            "principles_applied",
            "artifacts_shipped",
            "next_session_minutes",
            "next_module_id",
        ):
            self.assertIn(key, session)

    def test_duplicate_artifact_same_day_no_double_xp(self) -> None:
        record_event(
            "m00",
            "artifact_exported",
            artifacts=["a.json"],
            now=self.t0,
            path=self.path,
        )
        doc = record_event(
            "m00",
            "artifact_exported",
            artifacts=["b.json"],
            now=self.t0 + timedelta(hours=2),
            path=self.path,
        )
        self.assertEqual(doc["xp"], 10)

    def test_tests_green_same_day_adds_xp_not_streak(self) -> None:
        record_event(
            "m00",
            "artifact_exported",
            artifacts=["a.json"],
            now=self.t0,
            path=self.path,
        )
        doc = record_event(
            "m00",
            "tests_green",
            now=self.t0 + timedelta(hours=1),
            path=self.path,
        )
        self.assertEqual(doc["xp"], 20)
        self.assertEqual(doc["streak_days"], 1)

    def test_principle_placeholder_no_xp(self) -> None:
        doc = record_event(
            "m00",
            "principle_written",
            principles=["loss_defines_good"],
            principle_texts=["_Write 3–6 sentences here."],
            now=self.t0,
            path=self.path,
        )
        self.assertEqual(doc["xp"], 0)

    def test_principle_written_real_text_awards_xp(self) -> None:
        doc = record_event(
            "m00",
            "principle_written",
            principles=["loss_defines_good"],
            principle_texts=["The loss defines good; focal loss reweights hard examples."],
            now=self.t0,
            path=self.path,
        )
        self.assertEqual(doc["xp"], 10)
        self.assertEqual(doc["streak_days"], 0)
        self.assertIn("loss_defines_good", doc["sessions"][-1]["principles_applied"])

    def test_imbalance_slayer_badge(self) -> None:
        record_event("m00", "artifact_exported", artifacts=["a.json"], now=self.t0, path=self.path)
        self.assertNotIn("Imbalance Slayer", record_event("m00", "tests_green", now=self.t0, path=self.path)["badges"])
        doc = record_event(
            "m00",
            "principle_written",
            principles=["loss_defines_good"],
            principle_texts=["Loss defines what gradient descent optimizes."],
            now=self.t0,
            path=self.path,
        )
        self.assertIn("Imbalance Slayer", doc["badges"])

    def test_pitch_detective_badge(self) -> None:
        t = self.t0
        record_event("m01", "artifact_exported", artifacts=["m.json"], now=t, path=self.path)
        record_event("m01", "tests_green", now=t, path=self.path)
        doc = record_event(
            "m01",
            "principle_written",
            principles=["pitch_error_grows_with_range"],
            principle_texts=["Pitch error grows with range because rays meet Z=0 farther out."],
            now=t,
            path=self.path,
        )
        self.assertIn("Pitch Detective", doc["badges"])

    def test_pause_week_freezes_streak(self) -> None:
        t = self.t0
        record_event("m00", "tests_green", now=t, path=self.path)
        record_event(
            "m00",
            "tests_green",
            now=t + timedelta(days=1),
            path=self.path,
        )
        self.assertEqual(json.loads(self.path.read_text())["streak_days"], 2)
        later = t + timedelta(days=5)
        doc = record_event(
            "m00",
            "tests_green",
            pause_week=True,
            now=later,
            path=self.path,
        )
        self.assertEqual(doc["streak_days"], 2)
        self.assertTrue(doc["pause_week"])

    def test_gap_resets_streak_to_one(self) -> None:
        t = self.t0
        record_event("m00", "tests_green", now=t, path=self.path)
        record_event("m00", "tests_green", now=t + timedelta(days=1), path=self.path)
        doc = record_event(
            "m00",
            "tests_green",
            now=t + timedelta(days=4),
            path=self.path,
        )
        self.assertEqual(doc["streak_days"], 1)


class TestFormatStack(unittest.TestCase):
    def test_m00_checked_m01_unchecked(self) -> None:
        text = format_stack({"modules_touched": ["m00"]})
        self.assertIn("[x] m00", text)
        self.assertIn("[ ] m01", text)


class TestComeBackCue(unittest.TestCase):
    def test_m00_cue(self) -> None:
        cue = come_back_cue("m00")
        self.assertIn("25-min", cue)
        self.assertIn("error-gallery", cue)

    def test_m01_cue(self) -> None:
        cue = come_back_cue("m01")
        self.assertIn("pitch", cue)


class TestLearnerExampleSchema(unittest.TestCase):
    def test_example_matches_schema_keys(self) -> None:
        root = _repo_root()
        example = json.loads((root / "progress" / "learner.example.json").read_text())
        schema = json.loads((root / "progress" / "schema.json").read_text())

        required_top = schema["required"]
        for key in required_top:
            self.assertIn(key, example)

        session_schema = schema["$defs"]["session"]
        session = example["sessions"][0]
        for key in session_schema["required"]:
            self.assertIn(key, session)

        self.assertIn(example["sessions"][0]["next_session_minutes"], [25, 55, 90])
        self.assertRegex(example["sessions"][0]["module_id"], r"^m0[0-9]$")


if __name__ == "__main__":
    unittest.main()
