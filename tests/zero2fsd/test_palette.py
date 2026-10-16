# tests/zero2fsd/test_palette.py
"""The palette module: every colour is `#RRGGBB`, the paint colours are the camera's, and it is JSON-ready."""
import json
import re

import pytest

from zero2fsd.palette import CHROME, QUANTITY, SCENE, palette_dict
from zero2fsd.sim.camera import GRASS, PALETTE, ROAD, SKY, WHITE, YELLOW

HEX = re.compile(r"#[0-9A-Fa-f]{6}")
SCENE_INDEX = {"sky": SKY, "grass": GRASS, "road": ROAD, "yellow": YELLOW, "white": WHITE}


def rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


@pytest.mark.parametrize("group", [QUANTITY, SCENE, CHROME], ids=["quantity", "scene", "chrome"])
def test_every_colour_is_a_six_digit_hex_string(group):
    assert group and all(HEX.fullmatch(value) for value in group.values())


def test_scene_holds_one_colour_per_camera_palette_entry_and_equals_it():
    assert set(SCENE) == set(SCENE_INDEX)
    for name, index in SCENE_INDEX.items():
        assert rgb(SCENE[name]) == tuple(int(c) for c in PALETTE[index]), name


def test_the_four_quantities_are_told_apart_by_colour():
    assert set(QUANTITY) == {"e_off", "psi", "delta", "perception"}
    assert len(set(QUANTITY.values())) == len(QUANTITY)


def test_the_page_background_is_darker_than_its_panels():
    assert sum(rgb(CHROME["background"])) < sum(rgb(CHROME["panel"]))


def test_palette_dict_names_its_three_groups_and_survives_json():
    palette = palette_dict()
    assert list(palette) == ["quantity", "scene", "chrome"]
    assert json.loads(json.dumps(palette)) == palette
