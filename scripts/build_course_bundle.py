"""Build course/dist/course.json (+ figures): the ONE interface the site reads.  Schema version 1.

    python scripts/build_course_bundle.py [--root DIR]

Fails loudly (BundleError, exit 1) when a ready unit is missing a file, a field, a figure the lecture
references, or a quiz `section` that is not a heading of the lecture.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from numbers import Real
from pathlib import Path

from _course import REPO, heading_slug, load_yaml, syllabus_units, unit_slug

VERSION, REPO_SLUG = 1, "vvknyn/self-driving-ai-course"
REQUIRED_FILES = ("unit.yaml", "lecture.md", "lab.ipynb", "quiz.yaml", "cards.yaml")
REQUIRED_UNIT_KEYS = ("id", "title", "minutes", "outcomes", "prerequisites", "box", "go_deeper", "tutor_prompt")
QUESTION_KEYS = ("id", "type", "prompt", "answer", "explain", "section")
FIGURE_URL = "/learnfsd-course/figures"
IMAGE = re.compile(r"(!\[[^\]]*\]\()(?P<path>[^)\s]+)(?P<tail>(?:\s+\"[^\"]*\")?\))")
LOCAL_FIGURE = re.compile(r"(?:\./)?figures/(?P<file>[^/]+)")
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)


class BundleError(Exception):
    pass


def _need(condition, message):
    if not condition:
        raise BundleError(message)


def _text(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def rewrite_figures(unit_id: str, slug: str, folder: Path, lecture: str) -> str:
    """Point `figures/<file>` image refs at the site's copy; refuse refs the site could not serve."""

    def rewrite(match):
        path = match["path"]
        if re.match(r"(https?:)?//|/", path):
            return match[0]
        local = LOCAL_FIGURE.fullmatch(path)
        _need(local, f"{unit_id}: lecture image {path!r} must be a URL or live under figures/")
        _need((folder / "figures" / local["file"]).is_file(), f"{unit_id}: lecture references figures/{local['file']}, which does not exist")
        return f"{match[1]}{FIGURE_URL}/{slug}/{local['file']}{match['tail']}"

    return IMAGE.sub(rewrite, lecture)


def build_quiz(unit_id: str, quiz, sections: set[str]) -> list[dict]:
    _need(isinstance(quiz, list) and quiz, f"{unit_id}: quiz.yaml must be a non-empty list of questions")
    out, seen = [], set()
    for q in quiz:
        where = f"{unit_id} quiz question {q.get('id', '?') if isinstance(q, dict) else '?'}"
        _need(isinstance(q, dict) and all(k in q for k in QUESTION_KEYS), f"{where}: needs {', '.join(QUESTION_KEYS)}")
        _need(q["id"] not in seen, f"{where}: duplicate id")
        seen.add(q["id"])
        _need(all(_text(q[k]) for k in ("prompt", "explain", "section")), f"{where}: prompt, explain and section must be text")
        _need(q["section"] in sections, f"{where}: section {q['section']!r} is not a heading of the lecture (headings: {sorted(sections)})")
        if q["type"] == "mcq":
            choices = q.get("choices")
            _need(isinstance(choices, list) and len(choices) >= 2 and all(_text(c) for c in choices), f"{where}: an mcq needs 2+ text choices")
            _need(isinstance(q["answer"], int) and not isinstance(q["answer"], bool) and 0 <= q["answer"] < len(choices),
                  f"{where}: answer must be the index of a choice")
            out.append({"id": q["id"], "type": "mcq", "prompt": q["prompt"], "choices": choices, "answer": q["answer"],
                        "explain": q["explain"], "section": q["section"]})
        elif q["type"] == "numeric":
            _need(isinstance(q["answer"], Real) and not isinstance(q["answer"], bool), f"{where}: a numeric answer must be a number")
            _need(isinstance(q.get("tolerance"), Real) and not isinstance(q["tolerance"], bool) and q["tolerance"] >= 0,
                  f"{where}: a numeric question needs a tolerance >= 0")
            out.append({"id": q["id"], "type": "numeric", "prompt": q["prompt"], "answer": q["answer"], "tolerance": q["tolerance"],
                        "explain": q["explain"], "section": q["section"]})
        else:
            raise BundleError(f"{where}: type must be mcq or numeric, got {q['type']!r}")
    return out


def build_cards(unit_id: str, cards) -> list[dict]:
    _need(isinstance(cards, list), f"{unit_id}: cards.yaml must be a list")
    ids = [c.get("id") if isinstance(c, dict) else None for c in cards]
    _need(all(isinstance(c, dict) and _text(c.get("id")) and _text(c.get("front")) and _text(c.get("back")) for c in cards),
          f"{unit_id}: every card needs id, front and back")
    _need(len(set(ids)) == len(ids), f"{unit_id}: duplicate card ids")
    return [{"id": c["id"], "front": c["front"], "back": c["back"]} for c in cards]


def build_unit(root: Path, entry: dict, known_ids: set[str]) -> tuple[dict, list[str]]:
    """The bundle record of one ready unit plus the file names in its figures/ folder."""
    unit_id, folder = entry["id"], root / "course" / entry["folder"]
    for name in REQUIRED_FILES:
        _need((folder / name).is_file(), f"{unit_id}: ready unit is missing {entry['folder']}/{name}")
    meta = load_yaml(folder / "unit.yaml")
    missing = [k for k in REQUIRED_UNIT_KEYS if k not in meta]
    _need(not missing, f"{unit_id}: unit.yaml is missing {', '.join(missing)}")
    _need(meta["id"] == unit_id, f"{unit_id}: unit.yaml says id {meta['id']!r}")
    _need(all(p in known_ids for p in meta["prerequisites"]), f"{unit_id}: a prerequisite is not a unit in the syllabus: {meta['prerequisites']}")
    _need(all(isinstance(o, dict) and _text(o.get("can")) and _text(o.get("proof")) for o in meta["outcomes"]) and meta["outcomes"],
          f"{unit_id}: outcomes need a can and a proof each")
    paper = meta.get("paper")
    _need(paper is None or all(k in paper for k in ("title", "url", "questions")), f"{unit_id}: paper needs title, url and questions")

    slug = unit_slug(unit_id)
    lecture = (folder / "lecture.md").read_text(encoding="utf-8")
    sections = {heading_slug(h) for h in HEADING.findall(FENCE.sub("", lecture))}
    figures = sorted(p.name for p in (folder / "figures").iterdir() if p.is_file()) if (folder / "figures").is_dir() else []
    record = {
        "id": unit_id, "slug": slug, "title": meta["title"], "minutes": meta["minutes"], "box": meta["box"],
        "outcomes": meta["outcomes"], "prerequisites": [unit_slug(p) for p in meta["prerequisites"]],
        "go_deeper": meta["go_deeper"], "paper": paper, "tutor_prompt": meta["tutor_prompt"],
        "lecture_md": rewrite_figures(unit_id, slug, folder, lecture),
        "colab_url": f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/course/{entry['folder']}/lab.ipynb",
        "quiz": build_quiz(unit_id, load_yaml(folder / "quiz.yaml"), sections),
        "cards": build_cards(unit_id, load_yaml(folder / "cards.yaml")),
    }
    return record, figures


def build_bundle(root: Path) -> tuple[dict, list[tuple[Path, str]]]:
    """(course.json content, [(source figure file, path under dist/figures)])."""
    loops = load_yaml(root / "course" / "syllabus.yaml")["loops"]
    entries = syllabus_units(root)
    known_ids = {e["id"] for e in entries}
    _need(len(known_ids) == len(entries), "syllabus: duplicate unit ids")
    _need(all(e["status"] in ("ready", "planned") for e in entries), "syllabus: unit status must be ready or planned")
    units, minutes, copies = {}, {}, []
    for entry in (e for e in entries if e["status"] == "ready"):
        record, figures = build_unit(root, entry, known_ids)
        units[record["slug"]], minutes[entry["id"]] = record, record["minutes"]
        copies += [(root / "course" / entry["folder"] / "figures" / f, f"{record['slug']}/{f}") for f in figures]

    def unit_row(e):
        return {"id": e["id"], "slug": unit_slug(e["id"]), "title": e["title"], "status": e["status"], "minutes": minutes.get(e["id"])}

    bundle = {
        "version": VERSION, "repo": REPO_SLUG,
        "loops": [
            {"id": loop["id"], "title": loop["title"], "summary": loop["summary"], "modules": [
                {"id": m["id"], "title": m["title"], "summary": m["summary"], "planned_units": m["planned_units"],
                 "paper": m.get("paper"), "units": [unit_row(e) for e in m.get("units", [])]}
                for m in loop["modules"]]}
            for loop in loops],
        "units": units,
        "figures": [dest for _, dest in copies],
    }
    return bundle, copies


def write_bundle(root: Path) -> Path:
    bundle, copies = build_bundle(root)
    dist = root / "course" / "dist"
    shutil.rmtree(dist / "figures", ignore_errors=True)  # never leave a figure no unit uses
    for source, dest in copies:
        (dist / "figures" / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dist / "figures" / dest)
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "course.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dist / "course.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    root = parser.parse_args(argv).root
    try:
        out = write_bundle(root)
    except BundleError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    bundle = json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out} ({len(bundle['units'])} ready units, {len(bundle['figures'])} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
