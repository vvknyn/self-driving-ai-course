import pytest

import build_labs
import verify_units
from _course import REPO

UNIT = "course/0.1.1-mini"
SOLUTION = "solutions/course/0.1.1/solution.py"


def edit(root, rel, old, new):
    path = root / rel
    text = path.read_text()
    assert old in text, f"{old!r} not in {rel}"
    path.write_text(text.replace(old, new))


def rebuild(root):
    assert build_labs.main(["--root", str(root)]) == 0


def test_a_correct_unit_has_no_violations_and_never_runs_untagged_cells(built_repo, capsys):
    # the fixture lab has an untagged `%pip` cell (not valid Python): running it would be a "cannot build the skeleton"
    assert verify_units.verify(built_repo) == []
    assert capsys.readouterr().out == ""  # the grader's tick/cross lines stay out of the report


def test_planned_units_are_not_verified(built_repo):
    (built_repo / "course" / "0.1.2-not-written").mkdir()  # no files at all: fine, it is only planned
    assert verify_units.verify(built_repo) == []


@pytest.mark.parametrize(
    ("exercise", "old", "new"),
    [
        # exercise a spans two cells: only the LAST one defines the graded name
        ("0.1.1.a", "    return _mean_abs(lat_err)", "    a = np.abs(np.asarray(lat_err, float))\n    return {'mean_abs': float(a.mean()), 'max_abs': float(a.max()), 'rms': float(np.sqrt((a**2).mean()))}"),
        ("0.1.1.b", "def steps_off_lane(lat_err, threshold=0.9):\n    raise NotImplementedError", "def steps_off_lane(lat_err, threshold=0.9):\n    return int(sum(abs(v) > threshold for v in lat_err))"),
    ],
)
def test_a_skeleton_that_already_passes_is_a_violation(built_repo, exercise, old, new):
    edit(built_repo, f"{UNIT}/lab.py", old, new)
    rebuild(built_repo)
    (violation,) = verify_units.verify(built_repo)
    assert exercise in violation and "untouched skeleton PASSES" in violation


@pytest.mark.parametrize(("name", "old", "new"), [
    ("lane_error_stats", "float(np.mean(np.abs(err)))", "0.0"),
    ("steps_off_lane", "> threshold", ">= 5"),
])
def test_a_wrong_reference_solution_is_a_violation(built_repo, name, old, new):
    edit(built_repo, SOLUTION, old, new)
    (violation,) = verify_units.verify(built_repo)
    assert name in violation and "reference solution FAILS" in violation


def test_a_solution_missing_the_declared_name_is_a_violation(built_repo):
    edit(built_repo, SOLUTION, "def steps_off_lane", "def steps_off_the_lane")
    (violation,) = verify_units.verify(built_repo)
    assert "does not define 'steps_off_lane'" in violation


def test_an_exercise_id_the_grader_does_not_know_is_a_violation(built_repo):
    edit(built_repo, f"{UNIT}/unit.yaml", '"0.1.1.b"', '"0.1.1.zz"')
    (violation,) = verify_units.verify(built_repo)
    assert "0.1.1.zz" in violation and "no exercise with this id" in violation


def test_a_cell_tag_missing_from_the_notebook_is_a_violation(built_repo):
    edit(built_repo, f"{UNIT}/lab.py", '# %% tags=["exercise:0.1.1.b"]', "# %%")
    rebuild(built_repo)
    (violation,) = verify_units.verify(built_repo)
    assert "0.1.1.b" in violation and "cannot build the skeleton" in violation and "exercise:0.1.1.b" in violation


def test_a_skeleton_cell_that_does_not_define_the_name_is_a_violation(built_repo):
    edit(built_repo, f"{UNIT}/lab.py", "def steps_off_lane(", "def steps_off_lane_typo(")
    rebuild(built_repo)
    (violation,) = verify_units.verify(built_repo)
    assert "cannot build the skeleton" in violation and "steps_off_lane" in violation


def test_a_ready_unit_that_cannot_be_loaded_is_a_violation_not_a_crash(mini_repo):
    (violation,) = verify_units.verify(mini_repo)  # lab.ipynb was never built
    assert "cannot load the unit" in violation and "lab.ipynb" in violation


def test_a_ready_unit_without_a_solution_file_is_a_violation(built_repo):
    (built_repo / SOLUTION).unlink()
    (violation,) = verify_units.verify(built_repo)
    assert "cannot load the unit" in violation


def test_cli_exit_codes_and_report(built_repo, capsys):
    assert verify_units.main(["--root", str(built_repo)]) == 0
    assert "skeleton fails, solution passes" in capsys.readouterr().out
    edit(built_repo, SOLUTION, "> threshold", ">= 5")
    assert verify_units.main(["--root", str(built_repo)]) == 1
    out = capsys.readouterr().out
    assert "VIOLATION 0.1.1 0.1.1.b" in out and "1 violation(s)" in out


@pytest.mark.slow
def test_every_ready_unit_in_the_repo_verifies():
    """Part C: a unit marked `ready` must ship a lab whose skeleton fails and a solution that passes."""
    assert verify_units.verify(REPO) == []
