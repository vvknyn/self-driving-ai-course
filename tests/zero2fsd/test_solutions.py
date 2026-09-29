import pytest

from _course import REPO, load_solution
from zero2fsd.car import run
from zero2fsd.grade import check
from zero2fsd.grade._drive import GATE_MEAN_LAT
from zero2fsd.grade._report import EXERCISES
from zero2fsd.score import driving_score, metrics
from zero2fsd.sim.scenarios import make_scenario

# exercise id -> (unit, the name the lab asks the learner to define)
SOLVED = {
    "0.1.1.a": ("0.1.1", "lane_error_stats"),
    "0.1.1.b": ("0.1.1", "steps_off_lane"),
    "0.1.2.a": ("0.1.2", "paint_masks"),
    "0.1.3.a": ("0.1.3", "estimate_lane"),
    "0.1.3.b": ("0.1.3", "estimate_lane"),
    "0.1.4.a": ("0.1.4", "p_steer"),
    "0.1.4.b": ("0.1.4", "my_car"),
}


def test_every_registered_exercise_has_a_reference_solution():
    assert sorted(SOLVED) == sorted(EXERCISES)


@pytest.mark.parametrize("exercise_id", sorted(SOLVED))
def test_the_reference_solution_passes_its_exercise(exercise_id):
    unit, name = SOLVED[exercise_id]
    result = check(exercise_id, getattr(load_solution(REPO, unit), name), show=False)
    assert result.passed, "\n".join(result.details)


@pytest.mark.parametrize("unit", sorted({u for u, _ in SOLVED.values()}))
def test_a_solution_exports_exactly_what_its_lab_asks_for_and_owns_it(unit):
    module = load_solution(REPO, unit)
    assert sorted(module.__all__) == sorted({name for u, name in SOLVED.values() if u == unit})
    source = (REPO / "solutions" / "course" / unit / "solution.py").read_text()
    assert "provided" not in source, "a reference solution must not lean on the provided car"


def test_the_reference_car_completes_gentle_and_leaves_the_lane_on_curvy():
    car = load_solution(REPO, "0.1.4").my_car
    gentle, curvy = (run(car, make_scenario(name)) for name in ("gentle", "curvy"))
    assert gentle.completed and metrics(gentle)["mean_abs_lat"] < GATE_MEAN_LAT and driving_score(gentle) > 0
    assert not curvy.completed and curvy.off_road[-1], "the cliffhanger: a plain P controller cannot hold the curvy road"
