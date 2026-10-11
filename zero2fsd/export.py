"""The road as plain data for the browser view: the polylines the top-down plot draws, ready for `json.dumps`."""
from __future__ import annotations

from .car.runner import OFF_ROAD_OFFSET
from .sim.scenarios import SCENARIOS, make_scenario
from .sim.world import DASH_FRACTION, EGO_LANE_CENTER, LANE_WIDTH, ROAD_HALF_WIDTH, WHITE_LINE_LATERAL

ROAD_SPACING_M = 0.5  # arc length between exported points
COORD_DECIMALS = 3  # millimetres: finer than a canvas shows, and the committed roads.json does not change with the libm
LINES = {  # lateral offsets (left positive) of the polylines the top-down view draws
    "tarmac_left": ROAD_HALF_WIDTH, "tarmac_right": -ROAD_HALF_WIDTH,
    "yellow": 0.0, "white": WHITE_LINE_LATERAL, "ego_center": EGO_LANE_CENTER,
}


def road_dict(name: str) -> dict:
    """One scenario's road as JSON-ready data; an unknown `name` is the ValueError `make_scenario` raises."""
    sc = make_scenario(name)
    road = sc.road
    return {
        "name": sc.name, "closed": road.closed, "length": float(road.length),
        "route_length": float(sc.route_length), "start_s": float(sc.start_s),
        "dash_period": float(road.dash_period), "dash_fraction": DASH_FRACTION,
        "lane_width": LANE_WIDTH, "ego_lane_center": EGO_LANE_CENTER,
        "road_half_width": ROAD_HALF_WIDTH, "off_road_offset": OFF_ROAD_OFFSET,
        "lines": {key: road.sample_points(ROAD_SPACING_M, lateral).round(COORD_DECIMALS).tolist()
                  for key, lateral in LINES.items()},
    }


def roads_dict() -> dict[str, dict]:
    """Every scenario in `SCENARIOS`, keyed by name."""
    return {name: road_dict(name) for name in SCENARIOS}
