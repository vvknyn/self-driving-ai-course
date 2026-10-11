"""Compile each unit's percent-format lab.py into the committed lab.ipynb (deterministic).

    python scripts/build_labs.py [--root DIR] [--check] [UNIT_DIR ...]

`--check` writes nothing and exits 1 if any lab.ipynb is missing or stale.

A lab shows a lecture figure with a markdown line `![](figures/x.png)` (empty alt) on its own. The build swaps it for
raw HTML: an <img> with the absolute raw.githubusercontent.com src (Colab cannot resolve repo-relative paths) and the
lecture's own alt text as a numbered caption. Raw HTML, because the website wraps markdown images in its own captions.
The build fails if x.png is not a standalone image of the unit's lecture.md or is missing from its figures/.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import nbformat
from nbformat import v4

from _course import FENCE, REPO, REPO_SLUG, parse_lab

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}
FIGURE_REF = "main"  # the git ref the figure URLs point at
LAB_FIGURE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>figures/[^)\s/]+)\)[ \t]*$", re.M)
LECTURE_FIGURE = re.compile(r"^!\[(?P<alt>.*)\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)[ \t]*$", re.M)


def lecture_figures(lecture: str) -> dict[str, tuple[int, str]]:
    """{image path: (figure number, alt text)} for the lecture's standalone images (one per line), numbered in page order."""
    return {m["path"]: (n, m["alt"]) for n, m in enumerate(LECTURE_FIGURE.finditer(FENCE.sub("", lecture)), 1)}


def figure_html(unit: Path, ref: re.Match) -> str:
    """The raw-HTML figure for a lab's `![](figures/x.png)` line; ValueError unless x.png is a lecture figure that exists."""
    path = ref["path"]
    if ref["alt"]:
        raise ValueError(f"{unit.name}: {ref[0]!r} must have empty alt text; the caption is the lecture's alt")
    figures = lecture_figures((unit / "lecture.md").read_text(encoding="utf-8"))
    if path not in figures:
        raise ValueError(f"{unit.name}: {path} is not a standalone image of lecture.md (its figures: {sorted(figures)})")
    if not (unit / path).is_file():
        raise ValueError(f"{unit.name}: {path} does not exist")
    number, alt = figures[path]
    alt = html.escape(alt)
    src = f"https://raw.githubusercontent.com/{REPO_SLUG}/{FIGURE_REF}/course/{unit.name}/{path}"
    return f'<figure>\n<img src="{src}" alt="{alt}">\n<figcaption><em>Lecture figure {number}. {alt}</em></figcaption>\n</figure>'


def compile_lab(lab: Path) -> str:
    """The notebook JSON for a lab.py; cell ids are c000, c001, ... so rebuilding never churns them."""
    make = {"code": v4.new_code_cell, "markdown": v4.new_markdown_cell}
    cells = []
    for index, cell in enumerate(parse_lab(lab.read_text(encoding="utf-8"))):
        metadata = {"tags": cell.tags} if cell.tags else {}
        source = LAB_FIGURE.sub(lambda ref: figure_html(lab.parent, ref), cell.source) if cell.kind == "markdown" else cell.source
        cells.append(make[cell.kind](source, id=f"c{index:03d}", metadata=metadata))
    notebook = v4.new_notebook(cells=cells, metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}})
    nbformat.validate(notebook)
    return nbformat.writes(notebook) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("units", nargs="*", type=Path)
    args = parser.parse_args(argv)
    labs = [d / "lab.py" for d in args.units] or sorted((args.root / "course").glob("*/lab.py"))
    stale = []
    for lab in labs:
        target, built = lab.with_suffix(".ipynb"), compile_lab(lab)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != built:
                stale.append(target)
        else:
            target.write_text(built, encoding="utf-8")
            print(f"built {target.relative_to(args.root) if target.is_relative_to(args.root) else target}")
    for target in stale:
        print(f"stale: {target} (run scripts/build_labs.py)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
