"""Build course/dist/playground/: the page sources, the Python bundle and the data the playground reads.

    python scripts/build_course_bundle.py     # writes course.json, then calls write_playground

Fails loudly (BuildError, exit 1 through build_course_bundle) when a ready unit lacks a field the page needs, an
exercise has no lab.py cell, or a lecture links to a playground exercise or preset that does not exist.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from _course import BuildError, colab_url, load_yaml, parse_lab, syllabus_units, unit_slug
from zero2fsd.export import roads_dict
from zero2fsd.palette import palette_dict
from zero2fsd.preview import get

_VALUE = r"""([^&"'<>#\s)]*[^&"'<>#\s).,;:!?])"""  # a link's id or preset: ends at a delimiter, never on the punctuation of the sentence around it
PLAYGROUND_LINK = re.compile(rf"playground/index\.html\?exercise={_VALUE}(?:&preset={_VALUE})?")  # the form the site serves
WRONG_FORM_LINK = re.compile(r'playground/(?!index\.html\?exercise=)[^\s)"]*exercise=[^\s)"]*')  # e.g. playground/?exercise=: a 404
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # the earliest date a zip can hold: the same bytes whenever it is built


def _text(where: str, key: str, value) -> str:
    """`value` when it is a string; a BuildError naming `where` and the unit.yaml key otherwise (a bare `task:` loads as None)."""
    if not isinstance(value, str):
        raise BuildError(f"{where}: unit.yaml `{key}` must be text, got {value!r}")
    return value


def exercise_records(repo_root: Path) -> list[dict]:
    """One record per exercise of every ready unit, in syllabus order: everything the page needs to open it."""
    records = []
    for week, unit in enumerate(syllabus_units(repo_root), start=1):  # the week counts every listed unit, ready or not
        if unit["status"] != "ready":
            continue
        folder = repo_root / "course" / unit["folder"]
        meta = load_yaml(folder / "unit.yaml")
        if not isinstance(meta.get("exercises"), list):
            raise BuildError(f"{unit['id']}: unit.yaml has no `exercises` list")
        if meta["exercises"] and not meta.get("car_box"):
            raise BuildError(f"{unit['id']}: unit.yaml has exercises but no `car_box` (the part of the car they build)")
        lab = folder / "lab.py"
        if not lab.is_file():
            raise BuildError(f"{unit['id']}: course/{unit['folder']}/lab.py is missing")
        cells = parse_lab(lab.read_text(encoding="utf-8"))
        for exercise in meta["exercises"]:
            tag = f"exercise:{exercise['id']}"
            starter = "\n\n".join(c.source for c in cells if c.kind == "code" and tag in c.tags)
            if not starter:
                raise BuildError(f"{exercise['id']}: no code cell of {unit['folder']}/lab.py is tagged {tag!r}")
            try:
                preview = get(exercise["id"])
            except ValueError as err:  # zero2fsd cannot raise the scripts' BuildError
                raise BuildError(str(err)) from err
            records.append({
                "id": exercise["id"], "unit_id": unit["id"], "unit_key": unit_slug(unit["id"]), "week": week,
                "unit_title": meta["title"], "label": f"Week {week} · {meta['title']} · Exercise {exercise['id'].rsplit('.', 1)[-1].upper()}",
                "name": _text(exercise["id"], "name", exercise.get("name")), "task": _text(exercise["id"], "task", exercise.get("task", "")),
                "car_box": _text(unit["id"], "car_box", meta["car_box"]),
                "kind": preview.kind, "road": preview.road, "presets": {"starter": starter}, "colab_url": colab_url(unit["folder"]),
            })
    return records


def validate_lecture_links(repo_root: Path, records: list[dict]) -> None:
    """Raise BuildError naming every lecture link to a playground exercise or preset that does not exist."""
    presets = {r["id"]: r["presets"] for r in records}
    bad = []
    for lecture in sorted((repo_root / "course").glob("*/lecture.md")):
        text, where = lecture.read_text(encoding="utf-8"), lecture.relative_to(repo_root).as_posix()
        bad += [f"{where}: {m[0]} (use playground/index.html?exercise=<id>)" for m in WRONG_FORM_LINK.finditer(text)]
        for link in PLAYGROUND_LINK.finditer(text):
            exercise, preset = link.groups()
            if exercise not in presets:
                bad.append(f"{where}: {link[0]} (no such exercise)")
            elif preset and preset not in presets[exercise]:
                bad.append(f"{where}: {link[0]} (exercise {exercise} has presets {sorted(presets[exercise])})")
    if bad:
        raise BuildError("lecture links to the playground that do not resolve:\n  " + "\n  ".join(bad))


def bundle_zip(package: Path) -> bytes:
    """Every .py file of `package`, sorted, under its own name; stored with a fixed date so the bytes depend on content only."""
    sources = sorted(package.rglob("*.py"), key=lambda path: path.as_posix())
    if not sources:
        raise BuildError(f"{package} has no Python files to pack")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for path in sources:
            info = zipfile.ZipInfo(f"{package.name}/{path.relative_to(package).as_posix()}", ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            info.create_system = 3  # unix, whatever machine builds it
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def _json(data) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def page_files(folder: Path):
    """The files of `folder` to publish: none named `.…`, none below a folder named `_…` (`__pycache__`)."""
    for path in folder.iterdir():
        if path.name.startswith("."):
            continue
        if not path.is_dir():
            yield path
        elif not path.name.startswith("_"):
            yield from page_files(path)


def write_playground(repo_root: Path, out_dir: Path) -> None:
    """Replace `out_dir` with the build: the playground/ files, bundle.zip, three data files and manifest.json.
    The build lands in a sibling first and replaces `out_dir` only once it is complete, so a failed build leaves the previous one."""
    records = exercise_records(repo_root)
    validate_lecture_links(repo_root, records)
    building = out_dir.with_name(out_dir.name + ".building")
    shutil.rmtree(building, ignore_errors=True)  # only present after an interrupted build
    building.mkdir(parents=True)
    try:
        page = repo_root / "playground"
        for source in page_files(page):
            target = building / source.relative_to(page)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (building / "bundle.zip").write_bytes(bundle_zip(repo_root / "zero2fsd"))
        for name, data in (("exercises.json", records), ("roads.json", roads_dict()), ("palette.json", palette_dict())):
            (building / name).write_text(_json(data), encoding="utf-8")
        built = sorted((p for p in building.rglob("*") if p.is_file()), key=lambda p: p.as_posix())  # manifest.json does not exist yet, so it is not listed in itself
        files = [{"path": p.relative_to(building).as_posix(), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in built]
        (building / "manifest.json").write_text(_json({"files": files}), encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)  # a source renamed since the last build must not leave its old copy
        os.replace(building, out_dir)
    finally:
        shutil.rmtree(building, ignore_errors=True)  # still there only when the build above failed
