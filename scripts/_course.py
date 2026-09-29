"""Shared helpers for the course scripts: repo layout, syllabus, unit files, lab.py cells."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import NamedTuple

import yaml

REPO = Path(__file__).resolve().parents[1]
CELL_HEADER = re.compile(r"^# %%(?P<rest>.*)$")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def unit_slug(unit_id: str) -> str:
    """URL slug of a unit id: dots become hyphens."""
    return unit_id.replace(".", "-")


def heading_slug(text: str) -> str:
    """GitHub-style heading anchor: lowercase, spaces to hyphens, everything but [a-z0-9-] dropped."""
    return re.sub(r"[^a-z0-9-]", "", text.strip().lower().replace(" ", "-"))


def syllabus_units(root: Path, status: str | None = None) -> list[dict]:
    """Every unit entry in `root/course/syllabus.yaml` (only those with `status`, if given), in order."""
    loops = load_yaml(root / "course" / "syllabus.yaml")["loops"]
    units = [u for loop in loops for module in loop["modules"] for u in module.get("units", [])]
    return [u for u in units if status is None or u["status"] == status]


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
                raise ValueError(f"text before the first '# %%' cell header: {line!r}")
        else:
            body.append(line)
    close()
    return cells
