"""Learner progress tracking — XP, streaks, badges, session cards."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODULE_ORDER = ["m00", "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09"]
MODULE_TITLES = {
    "m00": "Driving ML Gym",
    "m01": "Cameras & IPM",
    "m02": "HydraNet",
    "m03": "BEV transform",
    "m04": "Occupancy",
    "m05": "Vector tracking",
    "m06": "Planning",
    "m07": "Control",
    "m08": "Capstone",
    "m09": "System architecture",
}
NEXT_MODULE = {
    "m00": "m01",
    "m01": "m02",
    "m02": "m03",
    "m03": "m04",
    "m04": "m05",
    "m05": "m06",
    "m06": "m07",
    "m07": "m08",
    "m08": "m09",
    "m09": None,
}

XP_BY_EVENT = {
    "tests_green": 10,
    "artifact_exported": 10,
    "principle_written": 10,
}

STREAK_SENTENCE = (
    "A day counts when you export an artifact or pass the tests for a fill you wrote. "
    "Opening the notebook does not. Set pause_week to true if you need a week off; "
    "the count stays where it is."
)

TODAYS_WIN = {
    "m00": "See pedestrian recall, not just accuracy, on the checked-in crops.",
    "m01": "Warp one camera to the ground plane, then measure how pitch error grows with range.",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_progress_path() -> Path:
    return _repo_root() / "artifacts" / "progress.json"


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _utc_date(dt: datetime) -> str:
    return _utc_now(dt).date().isoformat()


def principle_is_written(text: str) -> bool:
    """Return True when principle text is non-placeholder."""
    if not text or not text.strip():
        return False
    return not text.strip().startswith("_Write")


def _fresh_document() -> dict:
    return {
        "learner_id": "local",
        "pause_week": False,
        "streak_days": 0,
        "xp": 0,
        "badges": [],
        "modules_touched": [],
        "updated_at": "",
        "sessions": [],
    }


def _load_document(path: Path) -> dict:
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return _fresh_document()


def _event_already_awarded_today(doc: dict, module_id: str, event: str, today: str) -> bool:
    for session in doc.get("sessions", []):
        if session.get("module_id") != module_id:
            continue
        if _utc_date(datetime.fromisoformat(session["completed_at"])) != today:
            continue
        events = session.get("events") or []
        if event in events:
            return True
    return False


def _xp_for_event(
    doc: dict,
    module_id: str,
    event: str,
    *,
    principle_texts: list[str] | None,
    today: str,
) -> int:
    if event not in XP_BY_EVENT:
        return 0
    if event == "principle_written":
        texts = principle_texts or []
        if not any(principle_is_written(t) for t in texts):
            return 0
    if _event_already_awarded_today(doc, module_id, event, today):
        return 0
    return XP_BY_EVENT[event]


def _event_qualifies_for_streak(
    event: str,
    *,
    artifacts: list[str] | None,
    principle_texts: list[str] | None,
) -> bool:
    if event == "notebook_opened":
        return False
    if event == "tests_green":
        return True
    if event == "artifact_exported":
        return bool(artifacts)
    # principle_written awards XP only. A streak day is an artifact, or
    # tests_green recorded after the student fill for that island passes.
    return False


def _last_qualifying_session_date(doc: dict) -> str | None:
    for session in reversed(doc.get("sessions", [])):
        events = session.get("events") or []
        artifacts = session.get("artifacts_shipped") or []
        for event in events:
            if _event_qualifies_for_streak(event, artifacts=artifacts, principle_texts=None):
                return _utc_date(datetime.fromisoformat(session["completed_at"]))
        if _event_qualifies_for_streak("artifact_exported", artifacts=artifacts, principle_texts=None):
            return _utc_date(datetime.fromisoformat(session["completed_at"]))
        if _event_qualifies_for_streak("tests_green", artifacts=artifacts, principle_texts=None):
            return _utc_date(datetime.fromisoformat(session["completed_at"]))
    return None


def _compute_streak(
    doc: dict,
    *,
    qualifies: bool,
    paused: bool,
    today: str,
    now_dt: datetime,
) -> int:
    current = int(doc.get("streak_days", 0))
    if not qualifies or paused:
        return current

    last_date = _last_qualifying_session_date(doc)
    if last_date is None:
        return 1
    if last_date == today:
        return current

    yesterday = (_utc_now(now_dt).date() - timedelta(days=1)).isoformat()
    if last_date == yesterday:
        return current + 1
    return 1


def _collect_principles_applied(
    event: str,
    *,
    principles: list[str] | None,
    principle_texts: list[str] | None,
) -> list[str]:
    applied: list[str] = []
    seen: set[str] = set()
    for pid in principles or []:
        if pid not in seen:
            applied.append(pid)
            seen.add(pid)
    if event == "principle_written":
        for text in principle_texts or []:
            if principle_is_written(text):
                for pid in principles or []:
                    if pid not in seen:
                        applied.append(pid)
                        seen.add(pid)
                break
    return applied


def _module_has_event(doc: dict, module_id: str, event: str) -> bool:
    for session in doc.get("sessions", []):
        if session.get("module_id") != module_id:
            continue
        events = session.get("events") or []
        if event in events:
            if event == "principle_written":
                if session.get("principles_applied"):
                    return True
                continue
            return True
    return False


def _all_principles_for_module(doc: dict, module_id: str) -> set[str]:
    found: set[str] = set()
    for session in doc.get("sessions", []):
        if session.get("module_id") != module_id:
            continue
        for pid in session.get("principles_applied") or []:
            found.add(pid)
    return found


def _update_badges(doc: dict) -> None:
    badges = list(doc.get("badges") or [])
    badge_set = set(badges)

    m00_principles = _all_principles_for_module(doc, "m00")
    if (
        _module_has_event(doc, "m00", "tests_green")
        and _module_has_event(doc, "m00", "principle_written")
        and "loss_defines_good" in m00_principles
    ):
        badge_set.add("Imbalance Slayer")

    m01_principles = _all_principles_for_module(doc, "m01")
    if (
        _module_has_event(doc, "m01", "tests_green")
        and _module_has_event(doc, "m01", "principle_written")
        and "pitch_error_grows_with_range" in m01_principles
    ):
        badge_set.add("Pitch Detective")

    doc["badges"] = sorted(badge_set)


def _update_modules_touched(doc: dict) -> None:
    touched: set[str] = set(doc.get("modules_touched") or [])
    for session in doc.get("sessions", []):
        module_id = session.get("module_id")
        if not module_id:
            continue
        xp_awarded = session.get("xp_awarded") or 0
        if xp_awarded > 0:
            touched.add(module_id)
    doc["modules_touched"] = [m for m in MODULE_ORDER if m in touched]


def record_event(
    module_id: str,
    event: str,
    *,
    principles: list[str] | None = None,
    artifacts: list[str] | None = None,
    principle_texts: list[str] | None = None,
    next_session_minutes: int = 25,
    pause_week: bool | None = None,
    now: datetime | None = None,
    path: Path | None = None,
) -> dict:
    """Record a learner event and persist progress JSON.

    Args:
        module_id: Module identifier (``m00`` … ``m09``).
        event: Event name (``tests_green``, ``artifact_exported``, etc.).
        principles: Short principle ids to attach to the session.
        artifacts: Artifact paths shipped this session.
        principle_texts: Free-text principle responses (placeholder text ignored).
        next_session_minutes: Suggested next session length (25, 55, or 90).
        pause_week: When not ``None``, update the pause flag.
        now: Injectable UTC timestamp for tests.
        path: Progress file path (default ``artifacts/progress.json``).

    Returns:
        The full progress document after this event.
    """
    if next_session_minutes not in (25, 55, 90):
        raise ValueError("next_session_minutes must be 25, 55, or 90")

    out_path = path or _default_progress_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = _load_document(out_path)
    now_dt = _utc_now(now)
    today = _utc_date(now_dt)
    completed_at = now_dt.isoformat()

    if pause_week is not None:
        doc["pause_week"] = pause_week
    paused = bool(doc.get("pause_week", False))

    xp_awarded = _xp_for_event(
        doc, module_id, event, principle_texts=principle_texts, today=today
    )

    qualifies = _event_qualifies_for_streak(
        event, artifacts=artifacts, principle_texts=principle_texts
    )
    streak_days = _compute_streak(
        doc, qualifies=qualifies, paused=paused, today=today, now_dt=now_dt
    )

    principles_applied = _collect_principles_applied(
        event, principles=principles, principle_texts=principle_texts
    )

    existing_artifacts: list[str] = []
    for session in doc.get("sessions", []):
        existing_artifacts.extend(session.get("artifacts_shipped") or [])
    artifact_set = set(existing_artifacts)
    artifacts_shipped: list[str] = []
    for art in artifacts or []:
        if art not in artifact_set:
            artifacts_shipped.append(art)
            artifact_set.add(art)

    session: dict = {
        "module_id": module_id,
        "completed_at": completed_at,
        "streak_days": streak_days,
        "principles_applied": principles_applied,
        "artifacts_shipped": artifacts_shipped,
        "next_session_minutes": next_session_minutes,
        "next_module_id": NEXT_MODULE.get(module_id),
    }
    if xp_awarded > 0:
        session["xp_awarded"] = xp_awarded
    session["events"] = [event]

    doc["sessions"].append(session)
    doc["xp"] = int(doc.get("xp", 0)) + xp_awarded
    doc["streak_days"] = streak_days
    doc["updated_at"] = completed_at

    _update_modules_touched(doc)
    _update_badges(doc)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    return doc


def format_stack(progress: dict | None) -> str:
    """Checklist m00–m09. A module is marked done only if it appears in modules_touched."""
    touched = set((progress or {}).get("modules_touched") or [])
    lines = ["**Your stack so far**"]
    for mid in MODULE_ORDER:
        title = MODULE_TITLES[mid]
        mark = "x" if mid in touched else " "
        lines.append(f"- [{mark}] {mid} — {title}")
    return "\n".join(lines)


def come_back_cue(module_id: str) -> str:
    """One-line next action for end-of-notebook return habit."""
    cues = {
        "m00": (
            "Tomorrow: 25-min error-gallery review — implement build_error_gallery "
            "and re-run that cell."
        ),
        "m01": (
            "Tomorrow: 25-min pitch check — implement pitch_shift_meters and plot "
            "10 m, 20 m, and 40 m."
        ),
    }
    return cues.get(
        module_id,
        f"Tomorrow: pick up {MODULE_TITLES.get(module_id, module_id)} where you left off.",
    )


def session_card_text(module_id: str) -> str:
    """Markdown session card for module READMEs and notebooks."""
    win = TODAYS_WIN.get(module_id, "Apply today's principle in code, then inspect the artifact.")
    stack_lines = []
    for mid in MODULE_ORDER:
        title = MODULE_TITLES[mid]
        stack_lines.append(f"- [ ] {mid} — {title}")
    stack = "\n".join(stack_lines)

    return (
        f"## Session card\n\n"
        f"**Today's win:** {win}\n\n"
        f"**Time:** 25 / 55 / 90 minutes\n\n"
        f"{STREAK_SENTENCE}\n\n"
        f"**Your stack so far**\n\n"
        f"{stack}\n"
    )
