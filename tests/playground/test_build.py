import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

import build_playground as bp
from _course import REPO, syllabus_units
from build_course_bundle import build_bundle
from zero2fsd.export import roads_dict
from zero2fsd.grade._report import EXERCISES
from zero2fsd.palette import palette_dict
from zero2fsd.preview import get

MINI = REPO / "tests" / "zero2fsd" / "fixtures" / "mini_repo"
DOTTED_ID = re.compile(r"\d+\.\d+\.\d+")
GENERATED = {"bundle.zip", "exercises.json", "roads.json", "palette.json", "manifest.json"}
RECORD_KEYS = {"id", "unit_id", "unit_key", "week", "unit_title", "label", "name", "task", "car_box", "kind", "road", "presets", "colab_url"}
UNIT_YAML = "course/0.1.1-mini/unit.yaml"


@pytest.fixture(scope="module")
def records():
    return bp.exercise_records(REPO)


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    out = tmp_path_factory.mktemp("playground") / "dist"
    bp.write_playground(REPO, out)
    return out


@pytest.fixture
def mini_repo(tmp_path):
    """A writable copy of the one-unit fixture repo the course-script tests use."""
    root = tmp_path / "repo"
    shutil.copytree(MINI, root)
    return root


def rewrite(root, rel, old, new):
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {rel}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def link_in_lecture(root, folder, link):
    path = root / "course" / folder / "lecture.md"
    path.write_text(path.read_text(encoding="utf-8") + f"\n\n{link}\n", encoding="utf-8")


def starter_of(records, exercise_id):
    return next(r for r in records if r["id"] == exercise_id)["presets"]["starter"]


def tracked_files(root):
    """The paths git tracks under `root`, relative to it. An untracked file (a .DS_Store) is not among them."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [name for name in listed.split("\0") if name]


def fingerprints(root, names):
    """{name: bytes} for each of `names`, paths under root. A zip gives {"x.zip::member": CRC32} instead of its bytes, which carry timestamps that differ by machine."""
    found = {}
    for name in names:
        path = root / name
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                found.update({f"{name}::{member.filename}": member.CRC for member in archive.infolist()})
        else:
            found[name] = path.read_bytes()
    return found


# --- exercise_records ---------------------------------------------------------------------------------------


def test_one_record_per_graded_exercise_in_syllabus_order(records):
    ids = [r["id"] for r in records]
    assert sorted(ids) == sorted(EXERCISES) and len(set(ids)) == len(ids)
    assert ids[0] == "0.1.1.a"  # the first exercise of the first unit: what the page opens for an id it does not know
    units = list(dict.fromkeys(r["unit_id"] for r in records))
    assert units == [u["id"] for u in syllabus_units(REPO, "ready")]


def test_week_and_label_follow_the_unit_position(records):
    labels = {r["id"]: r["label"] for r in records}
    assert labels["0.1.4.b"] == "Week 4 · Closing the loop · Exercise B"
    assert labels["0.1.1.a"] == "Week 1 · Meet your car · Exercise A"
    assert [r["week"] for r in records] == [1, 1, 2, 3, 3, 4, 4]


def test_labels_never_show_a_dotted_id(records):
    assert not [r["id"] for r in records if DOTTED_ID.search(r["label"]) or DOTTED_ID.search(r["unit_title"])]


def test_unit_key_and_colab_url_are_the_ones_the_course_bundle_publishes(records):
    units = build_bundle(REPO)[0]["units"]
    for r in records:
        unit = units[r["unit_key"]]  # the site keys progress by this slug, so it must be a unit of course.json
        assert (unit["id"], unit["title"], unit["colab_url"]) == (r["unit_id"], r["unit_title"], r["colab_url"])
    assert {r["unit_id"]: r["unit_key"] for r in records} == {"0.1.1": "0-1-1", "0.1.2": "0-1-2", "0.1.3": "0-1-3", "0.1.4": "0-1-4"}


def test_every_record_has_exactly_the_keys_the_page_reads(records):
    assert all(set(r) == RECORD_KEYS for r in records)


def test_the_starter_is_every_cell_tagged_for_the_exercise_joined_by_a_blank_line(records):
    two_cells = starter_of(records, "0.1.4.b")
    assert "raise NotImplementedError\n\nK_OFF, K_HEAD = 0.35, 2.7" in two_cells
    assert "def my_controller(est, obs):" in two_cells and "my_car = None" in two_cells
    assert "check(" not in two_cells and "inside_the_bend" not in two_cells  # untagged cells stay out
    assert "my_car" not in starter_of(records, "0.1.4.a")  # another exercise's cells stay out


def test_one_cell_tagged_for_two_exercises_is_the_starter_of_both(records):
    assert starter_of(records, "0.1.3.a") == starter_of(records, "0.1.3.b")
    assert "def estimate_lane(obs):" in starter_of(records, "0.1.3.a")


def test_every_starter_runs_after_the_prelude_and_defines_the_exercise_name(records, shim):
    for r in records:
        shim.learner_object(r["presets"]["starter"], r["name"])  # raises NameError when the name is not the learner's own


def test_kind_and_road_come_from_the_preview_registry_and_every_named_road_exists(records):
    assert {(r["id"], r["kind"], r["road"]) for r in records} == {(r["id"], get(r["id"]).kind, get(r["id"]).road) for r in records}
    assert {"drive", "stats", "image"} <= {r["kind"] for r in records}
    assert all(r["road"] in roads_dict() for r in records if r["road"])
    assert {r["kind"] for r in records if not r["road"]} == {"image"}  # a picture exercise names no road: the shim picks its frame


def test_car_box_is_the_units_own_key_in_unit_yaml(records):
    assert {r["unit_id"]: r["car_box"] for r in records} == {
        "0.1.1": "instruments", "0.1.2": "perception", "0.1.3": "perception", "0.1.4": "steer"}


def test_a_unit_with_exercises_and_no_car_box_fails_naming_it_and_one_with_a_car_box_emits_it(mini_repo):
    assert {r["car_box"] for r in bp.exercise_records(mini_repo)} == {"instruments"}
    rewrite(mini_repo, UNIT_YAML, "car_box: instruments", "car_box: steer")
    assert {r["car_box"] for r in bp.exercise_records(mini_repo)} == {"steer"}
    rewrite(mini_repo, UNIT_YAML, "car_box: steer\n", "")
    with pytest.raises(ValueError, match=r"0\.1\.1: unit\.yaml has exercises but no `car_box`"):
        bp.exercise_records(mini_repo)


def test_a_unit_without_exercises_needs_no_car_box(mini_repo):
    rewrite(mini_repo, UNIT_YAML, "car_box: instruments\n", "")
    rewrite(mini_repo, UNIT_YAML, 'exercises:\n  - {id: "0.1.1.a", name: lane_error_stats}\n  - {id: "0.1.1.b", name: steps_off_lane}', "exercises: []")
    assert bp.exercise_records(mini_repo) == []


def test_task_is_empty_until_unit_yaml_gives_one(mini_repo):
    assert [r["task"] for r in bp.exercise_records(mini_repo)] == ["", ""]
    rewrite(mini_repo, "course/0.1.1-mini/unit.yaml", "{id: \"0.1.1.b\", name: steps_off_lane}",
            "{id: \"0.1.1.b\", name: steps_off_lane, task: Count the steps off the lane}")
    assert [r["task"] for r in bp.exercise_records(mini_repo)] == ["", "Count the steps off the lane"]


def test_the_week_counts_every_listed_unit_not_only_the_ready_ones(mini_repo):
    assert {r["week"] for r in bp.exercise_records(mini_repo)} == {1}
    rewrite(mini_repo, "course/syllabus.yaml", "          - {id: \"0.1.1\"",
            "          - {id: \"0.0.9\", folder: 0.0.9-earlier, title: Earlier, status: planned}\n          - {id: \"0.1.1\"")
    records = bp.exercise_records(mini_repo)
    assert {r["week"] for r in records} == {2} and [r["unit_id"] for r in records] == ["0.1.1", "0.1.1"]
    assert records[0]["label"] == "Week 2 · Mini unit · Exercise A" and records[1]["label"].endswith("Exercise B")


def test_an_exercise_without_a_tagged_lab_cell_fails_naming_it(mini_repo):
    rewrite(mini_repo, "course/0.1.1-mini/lab.py", 'tags=["exercise:0.1.1.b"]', 'tags=["exercise:0.1.1.x"]')
    with pytest.raises(ValueError, match="0.1.1.b"):
        bp.exercise_records(mini_repo)


# --- validate_lecture_links ----------------------------------------------------------------------------------


@pytest.mark.parametrize("link", [
    "[Open in the playground](playground/index.html?exercise=0.1.1.a)",
    "[Try it](https://intervuhq.com/learnfsd/playground/index.html?exercise=0.1.1.b&preset=starter)",
    '<a href="playground/index.html?exercise=0.1.1.a&preset=starter">try it</a>',
    "The playground (playground/index.html) opens on its own; the folder `playground/` holds its files.",
])
def test_lecture_links_that_resolve_pass(mini_repo, link):
    link_in_lecture(mini_repo, "0.1.1-mini", link)
    bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))


def test_every_lecture_link_in_the_repo_resolves(records):
    bp.validate_lecture_links(REPO, records)


@pytest.mark.parametrize(("link", "bad_link", "why"), [
    ("[Try it](playground/index.html?exercise=9.9.9.z)", "playground/index.html?exercise=9.9.9.z", "no such exercise"),
    ("[Try it](playground/index.html?exercise=9.9.9.z&preset=starter)", "playground/index.html?exercise=9.9.9.z&preset=starter", "no such exercise"),
    ("[Try it](playground/index.html?exercise=0.1.1.a&preset=nope)", "playground/index.html?exercise=0.1.1.a&preset=nope", "exercise 0.1.1.a has presets ['starter']"),
])
def test_a_lecture_link_to_a_missing_exercise_or_preset_names_the_lecture_the_link_and_why(mini_repo, link, bad_link, why):
    link_in_lecture(mini_repo, "0.1.1-mini", link)
    with pytest.raises(ValueError) as caught:
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))
    assert f"course/0.1.1-mini/lecture.md: {bad_link} ({why}" in str(caught.value)


@pytest.mark.parametrize("link", [
    "[Try it](/learnfsd/playground/?exercise=0.1.1.a)",
    "[Try it](playground/?exercise=0.1.1.a&preset=starter)",
    "[Try it](playground/index.html?preset=starter&exercise=0.1.1.a)",
])
def test_a_link_in_a_form_the_site_does_not_serve_fails_even_when_the_exercise_exists(mini_repo, link):
    link_in_lecture(mini_repo, "0.1.1-mini", link)
    with pytest.raises(ValueError, match=re.escape("use playground/index.html?exercise=<id>")):
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))


def test_every_bad_link_of_every_lecture_is_listed_at_once(mini_repo):
    (mini_repo / "course" / "0.9.9-other").mkdir()
    (mini_repo / "course" / "0.9.9-other" / "lecture.md").write_text("[x](playground/index.html?exercise=8.8.8.y)\n")
    link_in_lecture(mini_repo, "0.1.1-mini", "[x](playground/index.html?exercise=9.9.9.z)")
    with pytest.raises(ValueError) as caught:
        bp.validate_lecture_links(mini_repo, bp.exercise_records(mini_repo))
    message = str(caught.value)
    assert "course/0.9.9-other/lecture.md" in message and "8.8.8.y" in message
    assert "course/0.1.1-mini/lecture.md" in message and "9.9.9.z" in message


def test_a_bad_link_stops_the_build_before_the_previous_one_is_touched(mini_repo, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("previous build")
    link_in_lecture(mini_repo, "0.1.1-mini", "[x](playground/index.html?exercise=9.9.9.z)")
    with pytest.raises(ValueError, match="9.9.9.z"):
        bp.write_playground(mini_repo, out)
    assert (out / "keep.txt").read_text() == "previous build"


def test_a_failure_while_building_leaves_the_previous_build_and_no_half_built_sibling(mini_repo, tmp_path):
    out = tmp_path / "playground"
    out.mkdir()
    (out / "keep.txt").write_text("previous build")
    assert not (mini_repo / "playground").exists()  # the fixture repo has no page files: the build fails after validation, while copying
    with pytest.raises(FileNotFoundError):
        bp.write_playground(mini_repo, out)
    assert (out / "keep.txt").read_text() == "previous build"
    assert not (tmp_path / "playground.building").exists()


# --- write_playground ----------------------------------------------------------------------------------------


def test_dist_holds_the_page_files_and_the_generated_ones_and_nothing_else(dist):
    page = {p.name for p in (REPO / "playground").iterdir() if p.is_file() and not p.name.startswith(".")}
    assert "shim.py" in page
    assert {p.name for p in dist.iterdir()} == page | GENERATED
    assert all(p.is_file() for p in dist.iterdir())


def test_hidden_files_and_folders_of_playground_are_not_published(mini_repo, tmp_path):
    page = mini_repo / "playground"
    (page / "__pycache__").mkdir(parents=True)
    (page / "__pycache__" / "shim.cpython-312.pyc").write_bytes(b"x")
    (page / ".DS_Store").write_bytes(b"x")
    (page / "app.js").write_text("export {};\n")
    (mini_repo / "zero2fsd").mkdir()
    (mini_repo / "zero2fsd" / "__init__.py").write_text('"""stub"""\n')
    bp.write_playground(mini_repo, tmp_path / "out")
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"app.js"} | GENERATED


def test_page_files_are_copied_byte_for_byte(dist):
    for source in (REPO / "playground").iterdir():
        if source.is_file() and not source.name.startswith("."):
            assert (dist / source.name).read_bytes() == source.read_bytes()


def test_the_data_files_read_back_as_what_they_were_built_from(dist, records):
    assert json.loads((dist / "exercises.json").read_text(encoding="utf-8")) == records
    assert json.loads((dist / "roads.json").read_text()) == roads_dict()
    assert json.loads((dist / "palette.json").read_text()) == palette_dict()
    assert (dist / "roads.json").read_text() == json.dumps(roads_dict(), sort_keys=True) + "\n"


def test_the_zip_holds_the_package_sources_and_no_solutions_tests_or_caches(dist):
    with zipfile.ZipFile(dist / "bundle.zip") as archive:
        names = archive.namelist()
        assert "zero2fsd/__init__.py" in names and names == sorted(names)
        assert all(n.startswith("zero2fsd/") and n.endswith(".py") for n in names)
        assert not [n for n in names if any(word in n for word in ("solutions", "tests", "__pycache__"))]
        assert archive.read("zero2fsd/car/car.py") == (REPO / "zero2fsd" / "car" / "car.py").read_bytes()


def test_the_zip_can_be_imported_without_the_repo(dist, tmp_path):
    code = "import sys; sys.path.insert(0, sys.argv[1]); import zero2fsd.car, zero2fsd.grade; print(zero2fsd.car.__file__)"
    done = subprocess.run([sys.executable, "-c", code, str(dist / "bundle.zip")], capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(str(dist / "bundle.zip"))


def test_the_zip_bytes_depend_on_content_only(tmp_path):
    copy = tmp_path / "zero2fsd"
    shutil.copytree(REPO / "zero2fsd", copy)
    (copy / "__pycache__").mkdir(exist_ok=True)
    (copy / "__pycache__" / "stray.cpython-312.pyc").write_bytes(b"x")
    for path in copy.rglob("*"):
        os.utime(path, (0, 0))
    assert bp.bundle_zip(copy) == bp.bundle_zip(REPO / "zero2fsd")


@pytest.mark.parametrize("contents", [{}, {"README.md": "# not code\n", "data.json": "{}\n"}], ids=["empty", "no-py-files"])
def test_a_package_with_no_python_files_does_not_pack(tmp_path, contents):
    package = tmp_path / "zero2fsd"
    package.mkdir()
    for name, text in contents.items():
        (package / name).write_text(text)
    with pytest.raises(ValueError, match="has no Python files to pack"):
        bp.bundle_zip(package)


def test_manifest_lists_every_other_file_with_its_size_and_hash(dist):
    manifest = json.loads((dist / "manifest.json").read_text())
    paths = [f["path"] for f in manifest["files"]]
    assert paths == sorted(p.name for p in dist.iterdir() if p.name != "manifest.json")
    for entry in manifest["files"]:
        data = (dist / entry["path"]).read_bytes()
        assert (entry["bytes"], entry["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())
    assert list(manifest) == ["files"]


def test_write_replaces_the_directory_and_creates_missing_parents(tmp_path):
    out = tmp_path / "deep" / "er" / "dist"
    bp.write_playground(REPO, out)
    (out / "stale.js").write_text("old")
    bp.write_playground(REPO, out)
    assert not (out / "stale.js").exists() and (out / "manifest.json").is_file()


def test_the_committed_playground_is_what_the_source_builds(dist):
    """What the site copies is the committed course/dist/playground/. A tracked file the build no longer makes, a fresh output nobody
    committed or an out-of-date one fails here, so run `python scripts/build_course_bundle.py` after editing playground/ or zero2fsd/.
    Untracked files in the directory are not part of the commit and are ignored."""
    committed_dir = REPO / "course" / "dist" / "playground"
    committed = fingerprints(committed_dir, tracked_files(committed_dir))
    built = fingerprints(dist, [path.relative_to(dist).as_posix() for path in dist.rglob("*") if path.is_file()])
    stale = sorted(name for name in committed.keys() | built.keys() if committed.get(name) != built.get(name))
    assert not stale, "course/dist/playground/ differs from a fresh build in: " + ", ".join(stale)
