"""Shared helpers for the course scripts: repo layout, syllabus, unit files, lab.py cells."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # `python scripts/<name>.py` puts scripts/ first, not the repo root: scripts that import zero2fsd get it from here
    sys.path.insert(0, str(REPO))
REPO_SLUG = "vvknyn/self-driving-ai-course"
CELL_HEADER = re.compile(r"^# %%(?P<rest>.*)$")
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
DOTTED_EXERCISE_ID = re.compile(r"\b\d+\.\d+\.\d+\.[a-z]\b")  # 1.1.2.a: learners read "Exercise A", never this
LEARNER_FIELDS = {  # the fields the bundle builder copies into course/dist/ for the learner, per source file
    "unit.yaml": ("title", "box", "outcomes", "go_deeper", "paper", "tutor_prompt"),
    "quiz.yaml": ("prompt", "choices", "explain"),
    "cards.yaml": ("front", "back"),
}


class BuildError(ValueError):
    """What the build scripts raise for a problem in the course sources; their `main`s report it as one `error:` line, and only it."""


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def unit_slug(unit_id: str) -> str:
    """URL slug of a unit id: dots become hyphens."""
    return unit_id.replace(".", "-")


def colab_url(folder: str) -> str:
    """The Colab link of a unit's lab notebook; `folder` is its directory under course/."""
    return f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/course/{folder}/lab.ipynb"


def heading_slug(text: str) -> str:
    """GitHub-style heading anchor: lowercase, spaces to hyphens, everything but [a-z0-9-] dropped."""
    return re.sub(r"[^a-z0-9-]", "", text.strip().lower().replace(" ", "-"))


def syllabus_units(root: Path, status: str | None = None) -> list[dict]:
    """Every unit entry in `root/course/syllabus.yaml` (only those with `status`, if given), in order."""
    loops = load_yaml(root / "course" / "syllabus.yaml")["loops"]
    units = [u for loop in loops for module in loop["modules"] for u in module.get("units", [])]
    return [u for u in units if status is None or u["status"] == status]


def declared_exercises(root: Path, status: str | None = None) -> list[tuple[str, str, str]]:
    """(exercise id, unit id, the name the lab asks the learner to define) from the unit.yaml of every listed unit that has one
    (only those with `status`, if given), in syllabus order."""
    declared = []
    for unit in syllabus_units(root, status):
        meta = root / "course" / unit["folder"] / "unit.yaml"
        if meta.is_file():
            declared += [(e["id"], unit["id"], e["name"]) for e in load_yaml(meta)["exercises"]]
    return declared


def load_solution(root: Path, unit_id: str):
    """The reference solution module of a unit, loaded from its file (labs never import it)."""
    path = root / "solutions" / "course" / unit_id / "solution.py"
    spec = importlib.util.spec_from_file_location(f"solution_{unit_slug(unit_id)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Cell(NamedTuple):
    kind: str  # "code" or "markdown"
    tags: list[str]
    source: str


def parse_lab(text: str) -> list[Cell]:
    """Split a percent-format lab.py into cells: `# %%` code, `# %% [markdown]` text, `tags=[...]` in the header."""
    cells, header, body = [], None, []

    def close():
        if header is not None:
            rest = header.group("rest")
            markdown = "[markdown]" in rest
            tags = json.loads(rest.split("tags=", 1)[1]) if "tags=" in rest else []
            lines = [re.sub(r"^# ?", "", line) for line in body] if markdown else body
            cells.append(Cell("markdown" if markdown else "code", tags, "\n".join(lines).strip("\n")))

    for line in text.splitlines():
        match = CELL_HEADER.match(line)
        if match:
            close()
            header, body = match, []
        elif header is None:
            if line.strip():
                raise BuildError(f"text before the first '# %%' cell header: {line!r}")
        else:
            body.append(line)
    close()
    return cells


def _strings(value, path):
    """(field path, text) for every string under `value`, a url excepted."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != "url":
                yield from _strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def dotted_ids_in_prose(folder: Path) -> list[str]:
    """One `file:line` (or `file: field path`) message per dotted exercise id in the text a learner reads from this unit folder:
    the learner fields of unit.yaml, quiz.yaml and cards.yaml, and lecture.md outside its code fences."""
    found = []
    for name, fields in LEARNER_FIELDS.items():
        path = folder / name
        if not path.is_file():
            continue
        data = load_yaml(path)
        for item in data if isinstance(data, list) else [data]:
            for field in fields:
                for where, text in _strings(item.get(field), field):
                    found += [f"{path}: {where}: {m.group()}" for m in DOTTED_EXERCISE_ID.finditer(text)]
    lecture = folder / "lecture.md"
    if lecture.is_file():
        prose = FENCE.sub(lambda m: "\n" * m.group().count("\n"), lecture.read_text(encoding="utf-8"))  # keep the line numbers
        found += [f"{lecture}:{line}: {m.group()}" for line, text in enumerate(prose.splitlines(), start=1) for m in DOTTED_EXERCISE_ID.finditer(text)]
    return found
