"""1.1.3's graders: the reference passes, the lab's skeletons say "not implemented", and each preset fails the check it is named for."""
import numpy as np
import pytest

from _course import REPO, load_solution, parse_lab
from zero2fsd import geom
from zero2fsd.grade import check
from zero2fsd.grade._report import NOT_IMPLEMENTED
from zero2fsd.grade.units.u1_1_3 import L_ROAD, TIE_POSE, centerline_of, psi_misfit, tie_headings
from zero2fsd.sim import Road, make_scenario

UNIT = next((REPO / "course").glob("1.1.3-*"))
SOLUTION = load_solution(REPO, "1.1.3")
EXERCISES = {"1.1.3.a": "to_road_frame", "1.1.3.b": "lane_errors"}


def define(source: str, name: str):
    """What the playground does with learner code: run it after the geom names are bound, return `name`."""
    ns = {"np": np, **{n: getattr(geom, n) for n in geom.__all__}}
    exec(source, ns)
    return ns[name]


def starter(exercise_id: str) -> str:
    return "\n\n".join(c.source for c in parse_lab((UNIT / "lab.py").read_text()) if f"exercise:{exercise_id}" in c.tags)


def verdict(exercise_id: str, fn) -> str:
    result = check(exercise_id, fn, report=False)
    return "\n".join(result.details)


@pytest.mark.parametrize("exercise_id, name", EXERCISES.items())
def test_the_reference_solution_passes(exercise_id, name):
    assert check(exercise_id, getattr(SOLUTION, name), report=False).passed


@pytest.mark.parametrize("exercise_id, name", EXERCISES.items())
def test_the_labs_skeleton_is_not_implemented_yet(exercise_id, name):
    result = check(exercise_id, define(starter(exercise_id), name), report=False)
    assert not result.passed and NOT_IMPLEMENTED in "\n".join(result.details)


@pytest.mark.parametrize("exercise_id, preset", [("1.1.3.a", "y_minus_lane"), ("1.1.3.a", "nearest_vertex"), ("1.1.3.b", "no_wrap")])
def test_each_preset_fails_the_check_it_is_named_for(exercise_id, preset):
    fn = define((UNIT / "presets" / exercise_id / f"{preset}.py").read_text(), EXERCISES[exercise_id])
    result = check(exercise_id, fn, report=False)
    assert not result.passed
    assert {"y_minus_lane": "does not match", "nearest_vertex": "does not match", "no_wrap": "psi does not match"}[preset] in verdict(exercise_id, fn)


def test_the_nearest_vertex_preset_is_right_on_a_dense_road_and_wrong_at_the_corner():
    fn = define((UNIT / "presets/1.1.3.a/nearest_vertex.py").read_text(), "to_road_frame")
    road = make_scenario("gentle").road
    pts = [road.offset(s, -1.8) for s in np.linspace(0.0, road.length, 50, endpoint=False)]
    assert max(abs(abs(fn(centerline_of(road), True, p)[1]) - abs(road.project(*p)[1])) for p in pts) < 2e-3
    assert fn(L_ROAD, False, (8.0, 3.0))[1] == pytest.approx(3.6056, abs=1e-4)  # the truth is +2.000


def test_a_vertex_tie_accepts_either_segments_heading_and_nothing_else():
    road = Road(L_ROAD, False)
    x, y, yaw = TIE_POSE
    first, second = (float(h) for h in tie_headings(road, x, y))
    assert (first, second) == (0.0, pytest.approx(np.pi / 2))
    for h in (first, second):
        assert psi_misfit(road, TIE_POSE, float(geom.wrap_angle(yaw - h))) < 1e-9
    assert psi_misfit(road, TIE_POSE, float(geom.wrap_angle(yaw - 1.0))) > 0.1


def test_a_heading_a_whole_turn_out_is_rejected_on_any_road():
    right = SOLUTION.lane_errors
    off_by_a_turn = lambda line, closed, pose: (right(line, closed, pose)[0], right(line, closed, pose)[1] + 2 * np.pi)  # noqa: E731
    assert "psi does not match" in verdict("1.1.3.b", off_by_a_turn)


def test_an_open_road_is_given_its_last_vertex_and_a_closed_one_is_not():
    assert np.array_equal(centerline_of(Road(L_ROAD, False)), L_ROAD)
    gentle = make_scenario("gentle").road
    assert len(centerline_of(gentle)) == 7028  # one start per segment: the closing segment returns to the first


def test_a_function_that_returns_one_number_fails_with_a_plain_message():
    assert "two numbers" in verdict("1.1.3.a", lambda line, closed, p: 3.0)


def _wrong(solution_fn, change):
    """A learner function that is the reference answer with `change` applied to its pair."""
    return lambda *args: change(*solution_fn(*args))


@pytest.mark.parametrize("fn, said", [
    (_wrong(SOLUTION.to_road_frame, lambda s, d: (s, -d)), "sign of d the wrong way round"),
    (define((UNIT / "presets/1.1.3.a/nearest_vertex.py").read_text(), "to_road_frame"), "nearest vertex"),
    (define((UNIT / "presets/1.1.3.a/y_minus_lane.py").read_text(), "to_road_frame"), "y minus the y of the nearest vertex"),
])
def test_a_wrong_s_and_d_names_the_mistake_it_matches(fn, said):
    assert said in verdict("1.1.3.a", fn)


def test_a_foot_that_is_not_clamped_is_named():
    def unclamped(line, closed, p):
        a, b = (line, np.roll(line, -1, axis=0)) if closed else (line[:-1], line[1:])
        seg = np.hypot(*(b - a).T)
        u, rel = (b - a) / seg[:, None], np.array(p) - a
        along, cross = np.einsum("ij,ij->i", rel, u), u[:, 0] * rel[:, 1] - u[:, 1] * rel[:, 0]
        i = int(np.argmin(np.abs(cross)))
        return float(np.concatenate([[0.0], np.cumsum(seg)])[i] + along[i]), float(cross[i])
    assert "forgetting to clamp" in verdict("1.1.3.a", unclamped)


@pytest.mark.parametrize("change, said", [
    (lambda e, psi: (e - 1.8, psi), "without adding the 1.8"),
    (lambda e, psi: (-e, psi), "sign of e the wrong way round"),
    (lambda e, psi: (e, -psi), "psi the wrong way round"),
])
def test_a_wrong_e_or_psi_names_the_mistake_it_matches(change, said):
    assert said in verdict("1.1.3.b", _wrong(SOLUTION.lane_errors, change))


def test_the_unwrapped_psi_is_named():
    fn = define((UNIT / "presets/1.1.3.b/no_wrap.py").read_text(), "lane_errors")
    assert "no wrap" in verdict("1.1.3.b", fn)


def test_an_answer_that_matches_no_named_mistake_gets_the_general_hint():
    assert "clamped to the segment" in verdict("1.1.3.a", lambda line, closed, p: (0.0, 0.0))
    assert "wrapped" in verdict("1.1.3.b", _wrong(SOLUTION.lane_errors, lambda e, psi: (e, psi + 0.01)))
