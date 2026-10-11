# tests/zero2fsd/test_export.py
"""Road sampling (`Road.sample_points`) and the JSON-ready road export the browser view draws."""
import json
import math

import numpy as np
import pytest

from zero2fsd.car.runner import OFF_ROAD_OFFSET
from zero2fsd.export import COORD_DECIMALS, LINES, ROAD_SPACING_M, road_dict, roads_dict
from zero2fsd.sim.scenarios import SCENARIOS, make_scenario
from zero2fsd.sim.world import (
    DASH_FRACTION, EGO_LANE_CENTER, LANE_WIDTH, ROAD_HALF_WIDTH, WHITE_LINE_LATERAL, Road,
)

NAMES = sorted(SCENARIOS)


def segment_lengths(points, closed):
    pts = np.asarray(points, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])  # the segment back to the start line
    return np.hypot(*np.diff(pts, axis=0).T)


# --- Road.sample_points -------------------------------------------------------------------------------------------

def straight_10m():
    return Road([[0, 0], [5, 0], [10, 0]], closed=False)


def square_40m():  # counter-clockwise, so "left" is the inside
    return Road([[0, 0], [10, 0], [10, 10], [0, 10]], closed=True)


def test_an_open_road_ends_exactly_at_its_end():
    pts = straight_10m().sample_points(2.5)
    assert pts.shape == (5, 2)
    np.testing.assert_allclose(pts, [[0, 0], [2.5, 0], [5, 0], [7.5, 0], [10, 0]], atol=1e-12)


def test_an_open_road_whose_length_is_not_a_multiple_of_the_spacing_still_ends_at_its_end():
    pts = straight_10m().sample_points(4.0)
    np.testing.assert_allclose(pts, [[0, 0], [4, 0], [8, 0], [10, 0]], atol=1e-12)


def test_a_closed_road_stops_before_its_length_so_the_first_point_closes_the_loop():
    pts = square_40m().sample_points(15.0)
    np.testing.assert_allclose(pts, [[0, 0], [10, 5], [0, 10]], atol=1e-12)  # s = 0, 15, 30; not s = 40
    assert len(square_40m().sample_points(10.0)) == 4  # a multiple of the spacing: s = 40 is the first point again


def test_lateral_is_positive_to_the_left_of_travel_on_an_open_and_a_closed_road():
    np.testing.assert_allclose(straight_10m().sample_points(5.0, lateral=1.0), [[0, 1], [5, 1], [10, 1]], atol=1e-12)
    np.testing.assert_allclose(square_40m().sample_points(10.0, lateral=1.0), [[0, 1], [9, 0], [10, 9], [1, 10]], atol=1e-12)


# --- road_dict / roads_dict ---------------------------------------------------------------------------------------

def test_every_scenario_is_exported_and_the_result_is_plain_json():
    exported = roads_dict()
    assert sorted(exported) == NAMES
    assert json.loads(json.dumps(exported)) == exported  # floats, lists, bools only: nothing numpy


def test_an_unknown_road_is_a_value_error_naming_the_valid_ones():
    with pytest.raises(ValueError, match="unknown scenario 'nope'.*gentle"):
        road_dict("nope")


@pytest.mark.parametrize("name", NAMES)
def test_the_scalars_come_from_the_simulator_not_from_retyped_numbers(name):
    sc, d = make_scenario(name), road_dict(name)
    assert (d["name"], d["closed"], d["length"]) == (name, sc.road.closed, sc.road.length)
    assert (d["route_length"], d["start_s"], d["dash_period"]) == (sc.route_length, sc.start_s, sc.road.dash_period)
    assert (d["dash_fraction"], d["lane_width"], d["ego_lane_center"]) == (DASH_FRACTION, LANE_WIDTH, EGO_LANE_CENTER)
    assert (d["road_half_width"], d["off_road_offset"]) == (ROAD_HALF_WIDTH, OFF_ROAD_OFFSET)
    assert tuple(d["lines"]) == tuple(LINES)


def test_the_closed_and_the_open_scenarios_differ_in_the_fields_the_view_needs():
    assert road_dict("gentle")["closed"] is True and road_dict("gentle")["route_length"] == road_dict("gentle")["length"]
    assert road_dict("straight")["closed"] is False and road_dict("straight")["route_length"] < road_dict("straight")["length"]


def test_lines_sit_at_their_lateral_offsets_left_positive():
    d = road_dict("straight")  # a road along +x, so a line at lateral L has y == L
    for key, lateral in LINES.items():
        np.testing.assert_allclose(np.asarray(d["lines"][key])[:, 1], lateral, atol=10 ** -COORD_DECIMALS)
    assert LINES["tarmac_left"] > 0 > LINES["tarmac_right"] and LINES["ego_center"] == EGO_LANE_CENTER
    assert LINES["white"] == WHITE_LINE_LATERAL and LINES["yellow"] == 0.0


@pytest.mark.parametrize("name", ["gentle", "curvy"])  # on curves a line's lateral is what Road.project reports
def test_curved_lines_keep_their_lateral_offset(name):
    road, d = make_scenario(name).road, road_dict(name)
    for key, lateral in LINES.items():
        for x, y in d["lines"][key][::37]:
            assert road.project(x, y)[1] == pytest.approx(lateral, abs=5e-3), (key, x, y)


@pytest.mark.parametrize("name", NAMES)
def test_points_are_spaced_by_the_export_spacing(name):
    gaps = segment_lengths(road_dict(name)["lines"]["yellow"], closed=False)
    np.testing.assert_allclose(gaps[:-1], ROAD_SPACING_M, atol=2e-3)  # chords of the arc, to the rounding
    assert 0 < gaps[-1] <= ROAD_SPACING_M + 2e-3  # open roads: the remainder before the end; closed: one more full step


@pytest.mark.parametrize("name", NAMES)
def test_the_end_point_rule(name):
    road, d = make_scenario(name).road, road_dict(name)
    yellow = np.asarray(d["lines"]["yellow"])
    if d["closed"]:
        gap = float(np.hypot(*(yellow[-1] - yellow[0])))
        assert 0 < gap <= ROAD_SPACING_M + 2e-3  # the last point is short of the start line, never on it
    else:
        np.testing.assert_allclose(yellow[-1], road.offset(road.length, 0.0), atol=10 ** -COORD_DECIMALS)


@pytest.mark.parametrize("name", NAMES)
def test_the_exported_centre_line_is_as_long_as_the_road(name):
    d = road_dict(name)
    total = segment_lengths(d["lines"]["yellow"], d["closed"]).sum()
    assert math.isclose(total, d["length"], rel_tol=1e-3), (total, d["length"])


def test_coordinates_are_rounded_to_the_export_precision():
    for line in road_dict("gentle")["lines"].values():
        arr = np.asarray(line)
        assert np.array_equal(arr, np.round(arr, COORD_DECIMALS))
