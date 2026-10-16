"""The preview registry mirrors the grade registry: every graded exercise has a preview of the right kind.

The playground runs these previews on the learner's own code (no verdict): drive previews build the Car it
drives, stats previews turn a canned drive into labelled numbers, image previews draw the learner's masks.
"""
import re
import typing

import numpy as np
import pytest

import zero2fsd.preview as preview
from _course import REPO, declared_exercises, load_solution
from zero2fsd.car import Car, run
from zero2fsd.car.provided import K_HEAD, K_OFF, estimate_lane, paint_masks, steer_p
from zero2fsd.grade._report import EXERCISES, Failure
from zero2fsd.preview import PREVIEWS, Kind, get, register
from zero2fsd.preview.units.u0_1 import OVERLAY, STATS_ROAD
from zero2fsd.score import metrics
from zero2fsd.sim import BicycleState, Camera, make_scenario
from zero2fsd.sim.scenarios import SCENARIOS

KINDS = {
    "0.1.1.a": "stats", "0.1.1.b": "stats", "0.1.2.a": "image",
    "0.1.3.a": "drive", "0.1.3.b": "drive", "0.1.4.a": "drive", "0.1.4.b": "drive",
    "1.1.3.a": "stats", "1.1.3.b": "stats",
}
STEPS = 100  # enough steps to tell one car from another without driving a whole lap
STILL_PROVIDED = {"0.1.3.a": ["controller"], "0.1.3.b": ["controller"], "0.1.4.a": ["perception"], "0.1.4.b": ["perception", "controller"]}

# Each answer behaves as the provided box but is a different object, so the Car counts the slot as the learner's.
own_perception = lambda obs: estimate_lane(obs)  # noqa: E731
own_steer = lambda est, k_off, k_head: steer_p(est, k_off, k_head)  # noqa: E731
PROVIDED_ANSWER = {"0.1.3.a": own_perception, "0.1.3.b": own_perception, "0.1.4.a": own_steer, "0.1.4.b": Car()}


def ids_of(kind):
    return [i for i, p in sorted(PREVIEWS.items()) if p.kind == kind]


def reference_answer(exercise_id):
    """What the unit's reference solution defines for this exercise: the learner's right answer."""
    _, unit, name = next(d for d in declared_exercises(REPO) if d[0] == exercise_id)
    return getattr(load_solution(REPO, unit), name)


# --- the registry -------------------------------------------------------------------------------------------------

def test_every_graded_exercise_has_exactly_one_preview():
    assert set(PREVIEWS) == set(EXERCISES)


def test_each_exercise_has_the_kind_its_lab_needs():
    assert {i: p.kind for i, p in PREVIEWS.items()} == KINDS
    assert {p.kind for p in PREVIEWS.values()} <= set(typing.get_args(Kind))


def test_drive_and_stats_previews_name_a_real_scenario_and_image_previews_none():
    for exercise_id, p in PREVIEWS.items():
        assert (p.road == "") if p.kind == "image" else (p.road in SCENARIOS), exercise_id


def test_an_unknown_id_is_a_value_error_that_lists_the_known_ids():
    with pytest.raises(ValueError) as err:
        get("9.9.9.z")
    assert all(i in str(err.value) for i in KINDS)


def test_register_stores_the_function_untouched_and_refuses_a_second_preview(monkeypatch):
    monkeypatch.setattr(preview, "PREVIEWS", {})
    build = lambda fn: Car()  # noqa: E731
    assert register("x.1", "drive", road="gentle")(build) is build
    assert preview.get("x.1") == preview.Preview("drive", build, "gentle")
    with pytest.raises(ValueError, match="x.1"):
        register("x.1", "drive")(build)


# --- drive previews: a Car whose learner-owned slots are the learner's code --------------------------------------

def same_drive(a: Car, b: Car, road: str):
    ta, tb = run(a, road, max_steps=STEPS), run(b, road, max_steps=STEPS)
    return np.array_equal(ta.lat, tb.lat) and np.array_equal(ta.steer, tb.steer)


@pytest.mark.parametrize("exercise_id", ids_of("drive"))
def test_a_drive_preview_builds_a_car_that_drives_its_road_and_leaves_the_other_slot_provided(exercise_id):
    p = get(exercise_id)
    car = p.build(PROVIDED_ANSWER[exercise_id])
    assert isinstance(car, Car)
    assert car.uses_provided() == STILL_PROVIDED[exercise_id]
    assert same_drive(car, Car(), p.road)  # each answer is the provided box in disguise, so the car drives as the provided one


def test_the_steering_preview_uses_the_provided_gains_and_the_provided_perception():
    seen = []

    def my_steer(est, k_off, k_head):
        seen.append((k_off, k_head))
        return 0.0

    run(get("0.1.4.a").build(my_steer), "gentle", max_steps=STEPS)
    assert seen == [(K_OFF, K_HEAD)] * STEPS


def test_the_all_yours_preview_returns_the_learners_own_car_and_refuses_anything_else():
    mine = Car(perception=estimate_lane)
    assert get("0.1.4.b").build(mine) is mine  # a preview may run on the provided boxes too: only the check demands own ones
    with pytest.raises(Failure, match="my_car must be a Car.*got NoneType"):
        get("0.1.4.b").build(None)


# --- stats previews: the learner's function applied to the canned drive ------------------------------------------

@pytest.fixture(scope="module")
def canned():
    return run(Car(), STATS_ROAD)


@pytest.mark.parametrize("road", ["straight", STATS_ROAD])
@pytest.mark.parametrize("exercise_id", ids_of("stats"))
def test_a_stats_preview_returns_labelled_floats(exercise_id, road):
    values = get(exercise_id).build(reference_answer(exercise_id), run(Car(), road, max_steps=STEPS))
    assert values and all(isinstance(k, str) and type(v) is float for k, v in values.items())


def test_the_stats_previews_agree_with_the_score_module_on_the_canned_drive(canned):
    ref = metrics(canned)
    stats = get("0.1.1.a").build(reference_answer("0.1.1.a"), canned)
    assert stats["mean_abs"] == pytest.approx(ref["mean_abs_lat"]) and stats["max_abs"] == pytest.approx(ref["max_abs_lat"])
    assert get("0.1.1.b").build(reference_answer("0.1.1.b"), canned) == {"steps_off_lane": float(ref["steps_off_lane"])}
    assert ref["steps_off_lane"] > 0  # the canned drive has something to count: on `gentle` this lab line would read 0


def test_a_stats_preview_labels_a_bare_number_and_keeps_a_dicts_own_labels(canned):
    assert get("0.1.1.b").build(lambda lat: 7, canned) == {"steps_off_lane": 7.0}
    assert get("0.1.1.a").build(lambda lat: {"mean_abs": 1, "bias": -0.5}, canned) == {"mean_abs": 1.0, "bias": -0.5}


def test_a_stats_preview_fails_loudly_on_a_non_number_and_never_changes_the_telemetry(canned):
    def scribble(lat):
        lat.fill(0.0)
        return 0

    before = canned.lat.copy()
    with pytest.raises(TypeError):
        get("0.1.1.a").build(lambda lat: None, canned)
    get("0.1.1.b").build(scribble, canned)
    assert np.array_equal(canned.lat, before)


@pytest.mark.parametrize("bad, expected_text", [
    ("3", "expected a number or a dict of numbers, got str"),
    (True, "expected a number or a dict of numbers, got bool"),
    ({"mean": "3"}, "expected a number for 'mean', got str"),
    ({"mean": None}, "expected a number for 'mean', got NoneType"),
    ({None: "3"}, "expected a number for None, got str"),  # a None key is a label like any other, not the bare-number marker
])
def test_a_stats_answer_that_is_not_a_number_is_a_plain_language_type_error(canned, bad, expected_text):
    with pytest.raises(TypeError) as err:
        get("0.1.1.a").build(lambda lat: bad, canned)
    assert expected_text in str(err.value)


def test_a_numpy_float_answer_is_accepted_as_a_plain_float(canned):
    got = get("0.1.1.a").build(lambda lat: np.float64(0.25), canned)
    assert got == {"lane_error_stats": 0.25} and type(got["lane_error_stats"]) is float


@pytest.mark.parametrize("exercise_id", ["0.1.1.a", "0.1.1.b"])
def test_a_dict_with_a_none_key_keeps_that_key_as_its_label(canned, exercise_id):
    assert get(exercise_id).build(lambda lat: {None: 3, "mean": 1}, canned) == {"None": 3.0, "mean": 1.0}


# --- image previews: the learner's masks drawn on the frame ------------------------------------------------------

@pytest.fixture(scope="module")
def noisy():
    road = make_scenario("gentle").road
    return Camera().render(road, BicycleState(*road.ego_pose(5.0, 0.3, 0.02), 8.0, 0.0), noise_sigma=6.0, seed=3)


def test_the_image_preview_paints_each_mask_in_its_own_colour_and_leaves_the_rest(noisy):
    before = noisy.copy()
    shown = get("0.1.2.a").build(paint_masks, noisy)
    yellow, white = paint_masks(noisy)
    assert shown.dtype == np.uint8 and shown.shape == noisy.shape
    assert yellow.any() and white.any() and not (yellow & white).any()
    assert (shown[yellow] == OVERLAY["yellow"]).all() and (shown[white] == OVERLAY["white"]).all()
    rest = ~(yellow | white)
    assert np.array_equal(shown[rest], noisy[rest]) and np.array_equal(noisy, before)


@pytest.mark.parametrize("answer, said", [
    (lambda frame: None, "returned NoneType"),
    (lambda frame: (np.zeros((2, 2), bool), np.zeros((2, 2), bool)), "the yellow mask must be a bool array of shape (180, 320)"),
])
def test_the_image_preview_reports_a_wrong_answer_as_a_failure(noisy, answer, said):
    with pytest.raises(Failure, match=re.escape(said)):
        get("0.1.2.a").build(answer, noisy)
