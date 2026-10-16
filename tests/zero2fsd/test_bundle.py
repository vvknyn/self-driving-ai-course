import json
import shutil

import pytest
import yaml

import build_course_bundle as bundler
from _course import REPO, BuildError, heading_slug, syllabus_units, unit_slug
from zero2fsd.preview import PREVIEWS

UNIT = "course/0.1.1-mini"


def edit(root, rel, old, new):
    path = root / rel
    text = path.read_text()
    assert old in text, f"{old!r} not in {rel}"
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize(
    ("heading", "slug"),
    [("Measuring lane error", "measuring-lane-error"), ("  Why P, not PID?  ", "why-p-not-pid"), ("Step 2: IoU (≥ 0.85)", "step-2-iou--085"), ("A-B c", "a-b-c")],
)
def test_heading_slug_is_the_github_anchor(heading, slug):
    assert heading_slug(heading) == slug


def test_unit_slug():
    assert (unit_slug("0.1.1"), unit_slug("2.10.3")) == ("0-1-1", "2-10-3")


def test_bundle_shape_for_the_fixture(built_repo):
    bundle, copies = bundler.build_bundle(built_repo)
    assert (bundle["version"], bundle["repo"]) == (1, "vvknyn/self-driving-ai-course")
    module, later = bundle["loops"][0]["modules"]
    assert module["units"] == [
        {"id": "0.1.1", "slug": "0-1-1", "title": "Mini unit", "status": "ready", "minutes": 60},
        {"id": "0.1.2", "slug": "0-1-2", "title": "Not written yet", "status": "planned", "minutes": None},
    ]
    assert module["planned_units"] == 2 and module["paper"] is None
    assert later["units"] == [] and later["paper"] == {"title": "A paper", "url": "https://example.org/paper"}
    assert list(bundle["units"]) == ["0-1-1"]  # planned units carry no record
    unit = bundle["units"]["0-1-1"]
    assert unit["prerequisites"] == [] and unit["paper"] is None and unit["box"] == "none"
    assert unit["colab_url"] == "https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/course/0.1.1-mini/lab.ipynb"
    assert [q["type"] for q in unit["quiz"]] == ["mcq", "numeric"] and unit["cards"] == [
        {"id": "c1", "front": "What is lateral error?", "back": "The sideways distance from lane centre."}]
    assert bundle["figures"] == ["0-1-1/curve.png"] and [dest for _, dest in copies] == bundle["figures"]


def test_lecture_figure_refs_are_rewritten_and_other_urls_left_alone(built_repo):
    edit(built_repo, f"{UNIT}/lecture.md", "```python", "![remote](https://example.org/x.png)\n![abs](/already/served.png)\n\n```python")
    lecture = bundler.build_bundle(built_repo)[0]["units"]["0-1-1"]["lecture_md"]
    assert '![Lateral error over time](/learnfsd-course/figures/0-1-1/curve.png "the curve")' in lecture
    assert "![remote](https://example.org/x.png)" in lecture and "![abs](/already/served.png)" in lecture
    assert "](figures/" not in lecture


def test_a_prerequisite_becomes_a_slug(built_repo):
    edit(built_repo, f"{UNIT}/unit.yaml", "prerequisites: []", 'prerequisites: ["0.1.2"]')
    assert bundler.build_bundle(built_repo)[0]["units"]["0-1-1"]["prerequisites"] == ["0-1-2"]


def test_a_unit_paper_is_carried_through(built_repo):
    paper = {"title": "P", "url": "https://example.org/p", "questions": ["Why?"]}
    edit(built_repo, f"{UNIT}/unit.yaml", "tutor_prompt:", f"paper: {json.dumps(paper)}\ntutor_prompt:")
    assert bundler.build_bundle(built_repo)[0]["units"]["0-1-1"]["paper"] == paper


def test_a_go_deeper_note_survives_bundling_and_its_absence_is_unchanged(built_repo):
    assert bundler.build_bundle(built_repo)[0]["units"]["0-1-1"]["go_deeper"] == [{"title": "An example source", "url": "https://example.org/source"}]
    edit(built_repo, f"{UNIT}/unit.yaml", 'url: "https://example.org/source"}', 'url: "https://example.org/source", note: "Read the intro."}')
    assert bundler.build_bundle(built_repo)[0]["units"]["0-1-1"]["go_deeper"][0]["note"] == "Read the intro."


def test_write_bundle_is_deterministic_and_replaces_stale_figures(built_repo):
    out = bundler.write_bundle(built_repo)
    first = out.read_bytes()
    stale = built_repo / "course" / "dist" / "figures" / "0-1-1" / "gone.png"
    stale.write_bytes(b"x")
    assert bundler.write_bundle(built_repo).read_bytes() == first and not stale.exists()
    assert (built_repo / "course" / "dist" / "figures" / "0-1-1" / "curve.png").is_file()
    assert json.loads(first)["figures"] == ["0-1-1/curve.png"] and first.endswith(b"}\n")


def test_main_exit_codes(built_repo, capsys):
    assert bundler.main(["--root", str(built_repo)]) == 0 and "1 ready units" in capsys.readouterr().out
    assert (built_repo / "course" / "dist" / "playground" / "manifest.json").is_file()
    (built_repo / UNIT / "quiz.yaml").unlink()
    assert bundler.main(["--root", str(built_repo)]) == 1
    assert "quiz.yaml" in capsys.readouterr().err


def test_main_fails_with_one_line_when_an_exercise_has_no_preview(built_repo, capsys):
    edit(built_repo, f"{UNIT}/unit.yaml", '{id: "0.1.1.b", name: steps_off_lane}', '{id: "9.9.9.z", name: steps_off_lane}')
    edit(built_repo, f"{UNIT}/lab.py", "exercise:0.1.1.b", "exercise:9.9.9.z")
    assert bundler.main(["--root", str(built_repo)]) == 1
    err = capsys.readouterr().err
    assert err.splitlines() == [f"error: unknown exercise '9.9.9.z'; known: {', '.join(sorted(PREVIEWS))}"]
    assert not (built_repo / "course" / "dist" / "playground").exists()


def test_main_reports_a_lab_py_with_stray_text_and_a_missing_lab_py_as_one_error_line(built_repo, capsys):
    edit(built_repo, f"{UNIT}/lab.py", "# %%", "stray text\n# %%")
    assert bundler.main(["--root", str(built_repo)]) == 1
    assert capsys.readouterr().err.splitlines() == ["error: text before the first '# %%' cell header: 'stray text'"]
    edit(built_repo, f"{UNIT}/lab.py", "stray text\n", "")
    (built_repo / UNIT / "lab.py").unlink()
    assert bundler.main(["--root", str(built_repo)]) == 1
    assert capsys.readouterr().err.splitlines() == ["error: 0.1.1: course/0.1.1-mini/lab.py is missing"]


def test_main_does_not_swallow_a_value_error_that_is_not_the_builds_own(built_repo):
    edit(built_repo, f"{UNIT}/lab.py", 'tags=["exercise:0.1.1.b"]', 'tags=["exercise:0.1.1.b"')  # corrupt JSON in a cell header
    with pytest.raises(json.JSONDecodeError):
        bundler.main(["--root", str(built_repo)])


def test_main_fails_on_a_lecture_link_to_an_exercise_that_does_not_exist(built_repo, capsys):
    edit(built_repo, f"{UNIT}/lecture.md", "# Mini unit\n", "# Mini unit\n\n[Try it](playground/index.html?exercise=9.9.9.z)\n")
    assert bundler.main(["--root", str(built_repo)]) == 1
    assert "9.9.9.z" in capsys.readouterr().err


def test_a_planned_unit_with_a_folder_is_checked_like_a_ready_one_but_ships_nothing(built_repo):
    folder = built_repo / "course" / "0.1.2-not-written"
    folder.mkdir()
    with pytest.raises(BuildError, match=r"0\.1\.2: planned unit is missing 0\.1\.2-not-written/unit\.yaml"):
        bundler.build_bundle(built_repo)
    shutil.rmtree(folder)
    shutil.copytree(built_repo / UNIT, folder)
    edit(built_repo, "course/0.1.2-not-written/unit.yaml", 'id: "0.1.1"', 'id: "0.1.2"')
    bundle, copies = bundler.build_bundle(built_repo)
    assert list(bundle["units"]) == ["0-1-1"] and not [dest for _, dest in copies if dest.startswith("0-1-2/")]
    (folder / "quiz.yaml").unlink()
    with pytest.raises(BuildError, match=r"0\.1\.2: planned unit is missing 0\.1\.2-not-written/quiz\.yaml"):
        bundler.build_bundle(built_repo)


@pytest.mark.parametrize("name", bundler.REQUIRED_FILES)
def test_a_ready_unit_missing_any_required_file_fails(built_repo, name):
    (built_repo / UNIT / name).unlink()
    with pytest.raises(BuildError, match=name):
        bundler.build_bundle(built_repo)


@pytest.mark.parametrize("key", bundler.REQUIRED_UNIT_KEYS[1:])
def test_a_unit_yaml_missing_any_required_key_fails(built_repo, key):
    path = built_repo / UNIT / "unit.yaml"
    meta = yaml.safe_load(path.read_text())
    del meta[key]
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(BuildError, match=key):
        bundler.build_bundle(built_repo)


@pytest.mark.parametrize(
    ("rel", "old", "new", "message"),
    [
        (f"{UNIT}/unit.yaml", 'id: "0.1.1"', 'id: "0.1.7"', "unit.yaml says id"),
        (f"{UNIT}/unit.yaml", "prerequisites: []", 'prerequisites: ["9.9.9"]', "prerequisite"),
        (f"{UNIT}/unit.yaml", "outcomes:\n  - {can: Summarise a lateral-error trace, proof: lane_error_stats passes its tests}", "outcomes: []", "outcomes"),
        (f"{UNIT}/lecture.md", "figures/curve.png", "figures/missing.png", "figures/missing.png, which does not exist"),
        (f"{UNIT}/lecture.md", "figures/curve.png", "elsewhere/curve.png", "must be a URL or live under figures/"),
        (f"{UNIT}/quiz.yaml", "section: measuring-lane-error\n- id: q2", "section: not-a-heading\n- id: q2", "not a heading of the lecture"),
        (f"{UNIT}/quiz.yaml", "answer: 1", "answer: 3", "index of a choice"),
        (f"{UNIT}/quiz.yaml", "tolerance: 0.01", "tolerance: -1", "tolerance"),
        (f"{UNIT}/quiz.yaml", "type: numeric", "type: essay", "mcq or numeric"),
        (f"{UNIT}/quiz.yaml", "  explain: \"The absolute values are both 0.5.\"\n", "", "needs id, type"),
        (f"{UNIT}/quiz.yaml", "id: q2", "id: q1", "duplicate id"),
        (f"{UNIT}/cards.yaml", 'back: "The sideways distance from lane centre."', 'back: ""', "id, front and back"),
        (f"{UNIT}/unit.yaml", 'url: "https://example.org/source"}', 'url: "https://example.org/source", note: ""}', "go_deeper item"),
        ("course/syllabus.yaml", "status: planned", "status: draft", "ready or planned"),
        ("course/syllabus.yaml", "id: \"0.1.2\"", "id: \"0.1.1\"", "duplicate unit ids"),
    ],
)
def test_bad_content_fails_loudly(built_repo, rel, old, new, message):
    edit(built_repo, rel, old, new)
    with pytest.raises(BuildError, match=message):
        bundler.build_bundle(built_repo)


def test_a_quiz_section_may_not_be_a_heading_that_only_appears_in_a_code_fence(built_repo):
    edit(built_repo, f"{UNIT}/quiz.yaml", "section: measuring-lane-error\n- id: q2", "section: this-comment-is-not-a-heading\n- id: q2")
    with pytest.raises(BuildError, match="not a heading"):
        bundler.build_bundle(built_repo)


def test_the_real_syllabus_builds():
    """Part C: whatever units are `ready` must satisfy the bundle contract (pure build, nothing written)."""
    bundle, _ = bundler.build_bundle(REPO)
    assert bundle["version"] == 1 and [loop["id"] for loop in bundle["loops"]] == ["0", "1", "2", "3"]
    assert [sum(m["planned_units"] for m in loop["modules"]) for loop in bundle["loops"]] == [4, 48, 52, 40]
    assert set(bundle["units"]) == {unit_slug(u["id"]) for loop in bundle["loops"] for m in loop["modules"] for u in m.get("units", []) if u["status"] == "ready"}


READY_WITH_FIGURES = [u for u in syllabus_units(REPO, "ready") if (REPO / "course" / u["folder"] / "figures").is_dir()]


@pytest.mark.parametrize("unit", READY_WITH_FIGURES, ids=[u["folder"] for u in READY_WITH_FIGURES])
def test_the_committed_dist_figures_are_byte_copies_of_the_units_figures(unit):
    """Part C: after scripts/make_figures.py, run scripts/build_course_bundle.py and commit course/dist."""
    pngs = sorted((REPO / "course" / unit["folder"] / "figures").glob("*.png"))
    assert pngs
    for png in pngs:
        copy = REPO / "course" / "dist" / "figures" / unit_slug(unit["id"]) / png.name
        assert copy.is_file() and copy.read_bytes() == png.read_bytes(), copy


@pytest.mark.parametrize("alt", ["`frame[0, 0]`", "a [b] c [d]", "plain"])
def test_figure_refs_are_rewritten_whatever_brackets_the_alt_holds(tmp_path, alt):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "x.png").write_bytes(b"png")
    out = bundler.rewrite_figures("0.1.1", "0-1-1", tmp_path, f"![{alt}](figures/x.png)\n![{alt}](./figures/x.png \"t\")")
    assert out == f"![{alt}](/learnfsd-course/figures/0-1-1/x.png)\n![{alt}](/learnfsd-course/figures/0-1-1/x.png \"t\")"


def test_a_leftover_local_figure_path_fails_the_build(tmp_path):  # alt nested two deep is beyond the matcher
    with pytest.raises(BuildError, match="left unrewritten"):
        bundler.rewrite_figures("0.1.1", "0-1-1", tmp_path, "![deep [[x]]](figures/x.png)")
