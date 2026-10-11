"""Build course/dist/playground/: the page sources, the Python bundle and the data the playground reads.

    python scripts/build_course_bundle.py     # writes course.json, then calls write_playground

Fails loudly (ValueError, exit 1 through build_course_bundle) when a unit with exercises has no `car_box`, an
exercise has no lab.py cell, or a lecture links to a playground exercise or preset that does not exist.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import string
import zipfile
from pathlib import Path

from _course import colab_url, load_yaml, parse_lab, syllabus_units, unit_slug
from zero2fsd.export import roads_dict
from zero2fsd.palette import palette_dict
from zero2fsd.preview import get

PLAYGROUND_LINK = re.compile(r'playground/index\.html\?exercise=([^&"\s)]+)(?:&preset=([^&"\s)]+))?')  # the form the site serves
WRONG_FORM_LINK = re.compile(r'playground/(?!index\.html\?exercise=)[^\s)"]*exercise=[^\s)"]*')  # e.g. playground/?exercise=: a 404
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # the earliest date a zip can hold: the same bytes whenever it is built


def exercise_records(repo_root: Path) -> list[dict]:
    """One record per exercise of every ready unit, in syllabus order: everything the page needs to open it."""
    records = []
    for week, unit in enumerate(syllabus_units(repo_root), start=1):  # the week counts every listed unit, ready or not
        if unit["status"] != "ready":
            continue
        folder = repo_root / "course" / unit["folder"]
        meta = load_yaml(folder / "unit.yaml")
        if meta["exercises"] and not meta.get("car_box"):
            raise ValueError(f"{unit['id']}: unit.yaml has exercises but no `car_box` (the part of the car they build)")
        cells = parse_lab((folder / "lab.py").read_text(encoding="utf-8"))
        for index, exercise in enumerate(meta["exercises"]):
            tag = f"exercise:{exercise['id']}"
            starter = "\n\n".join(c.source for c in cells if c.kind == "code" and tag in c.tags)
            if not starter:
                raise ValueError(f"{exercise['id']}: no code cell of {unit['folder']}/lab.py is tagged {tag!r}")
            preview = get(exercise["id"])
            records.append({
                "id": exercise["id"], "unit_id": unit["id"], "unit_key": unit_slug(unit["id"]), "week": week,
                "unit_title": meta["title"], "label": f"Week {week} · {meta['title']} · Exercise {string.ascii_uppercase[index]}",
                "name": exercise["name"], "task": exercise.get("task", ""), "car_box": meta["car_box"],
                "kind": preview.kind, "road": preview.road, "presets": {"starter": starter}, "colab_url": colab_url(unit["folder"]),
            })
    return records


def validate_lecture_links(repo_root: Path, records: list[dict]) -> None:
    """Raise ValueError naming every lecture link to a playground exercise or preset that does not exist."""
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
        raise ValueError("lecture links to the playground that do not resolve:\n  " + "\n  ".join(bad))


def bundle_zip(package: Path) -> bytes:
    """Every .py file of `package`, sorted, under its own name; stored with a fixed date so the bytes depend on content only."""
    sources = sorted(package.rglob("*.py"))
    if not sources:
        raise ValueError(f"{package} has no Python files to pack")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for path in sources:
            info = zipfile.ZipInfo(f"{package.name}/{path.relative_to(package).as_posix()}", ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def _json(data) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n"


def write_playground(repo_root: Path, out_dir: Path) -> None:
    """Replace `out_dir` with the build: the playground/ files, bundle.zip, three data files and manifest.json.
    The build lands in a sibling first and replaces `out_dir` only once it is complete, so a failed build leaves the previous one."""
    records = exercise_records(repo_root)
    validate_lecture_links(repo_root, records)
    building = out_dir.with_name(out_dir.name + ".building")
    shutil.rmtree(building, ignore_errors=True)  # only present after an interrupted build
    building.mkdir(parents=True)
    try:
        for source in sorted((repo_root / "playground").iterdir()):
            if source.is_file() and not source.name.startswith("."):
                shutil.copyfile(source, building / source.name)
        (building / "bundle.zip").write_bytes(bundle_zip(repo_root / "zero2fsd"))
        for name, data in (("exercises.json", records), ("roads.json", roads_dict()), ("palette.json", palette_dict())):
            (building / name).write_text(_json(data), encoding="utf-8")
        files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(building.iterdir())]  # manifest.json does not exist yet, so it is not listed in itself
        (building / "manifest.json").write_text(_json({"files": files}), encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)  # a source renamed since the last build must not leave its old copy
        os.replace(building, out_dir)
    finally:
        shutil.rmtree(building, ignore_errors=True)  # still there only when the build above failed
