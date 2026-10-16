import pytest

from _course import REPO, declared_exercises, load_solution
from zero2fsd.car import run
from zero2fsd.grade import check
from zero2fsd.grade._drive import GATE_MEAN_LAT
from zero2fsd.grade._report import EXERCISES
from zero2fsd.score import driving_score, metrics
from zero2fsd.sim.scenarios import make_scenario

DECLARED = declared_exercises(REPO)  # (exercise id, unit id, the name the lab asks the learner to define), from every unit.yaml


def test_every_registered_exercise_is_declared_by_a_unit_yaml_and_the_other_way_round():
    assert sorted(EXERCISES) == sorted(exercise_id for exercise_id, _, _ in DECLARED)


@pytest.mark.parametrize(("exercise_id", "unit", "name"), DECLARED)
def test_the_reference_solution_exports_the_exercise_name_and_passes_its_exercise(exercise_id, unit, name):
    path = REPO / "solutions" / "course" / unit / "solution.py"
    assert path.is_file(), f"{unit} declares {exercise_id} but has no reference solution at {path.relative_to(REPO)}"
    module = load_solution(REPO, unit)
    assert hasattr(module, name), f"{path.relative_to(REPO)} does not define {name}, the name {exercise_id} asks for"
    result = check(exercise_id, getattr(module, name), show=False)
    assert result.passed, "\n".join(result.details)


@pytest.mark.parametrize("unit", sorted({unit for _, unit, _ in DECLARED}))
def test_a_solution_exports_exactly_what_its_lab_asks_for_and_owns_it(unit):
    module = load_solution(REPO, unit)
    assert sorted(module.__all__) == sorted({name for _, u, name in DECLARED if u == unit})
    source = (REPO / "solutions" / "course" / unit / "solution.py").read_text()
    assert "provided" not in source, "a reference solution must not lean on the provided car"


def test_the_reference_car_completes_gentle_and_leaves_the_lane_on_curvy():
    car = load_solution(REPO, "0.1.4").my_car
    gentle, curvy = (run(car, make_scenario(name)) for name in ("gentle", "curvy"))
    assert gentle.completed and metrics(gentle)["mean_abs_lat"] < GATE_MEAN_LAT and driving_score(gentle) > 0
    assert not curvy.completed and curvy.off_road[-1], "the cliffhanger: a plain P controller cannot hold the curvy road"
