"""drawing.py obeys Ruling F: no dpi, no colour literal, no font size of its own."""
import ast
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import FancyArrowPatch

from zero2fsd import drawing
from zero2fsd.car import Car, run
from zero2fsd.palette import QUANTITY


@pytest.fixture
def ax():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    yield ax
    plt.close(fig)


def test_draw_frame_on_a_5_5_inch_figure_adds_exactly_2_arrows_and_3_labels(ax):
    drawing.draw_frame(ax, (3.0, 1.0, math.pi / 2), QUANTITY["ego"], "ego")
    assert [type(p) for p in ax.patches] == [FancyArrowPatch] * 2
    assert sorted(t.get_text() for t in ax.texts) == ["ego", "x", "y"]


def test_the_arrows_follow_the_frames_pose(ax):
    drawing.draw_frame(ax, (3.0, 1.0, math.pi / 2), QUANTITY["camera"], "cam", length_m=2.0)
    x_arrow, y_arrow = ax.patches
    assert x_arrow._posA_posB[1] == pytest.approx((3.0, 3.0)) and y_arrow._posA_posB[1] == pytest.approx((1.0, 1.0))


def test_text_inherits_the_figures_font_size_unless_the_caller_passes_one(ax):
    drawing.draw_frame(ax, (0.0, 0.0, 0.0), QUANTITY["ego"], "ego")
    assert {t.get_fontsize() for t in ax.texts} == {matplotlib.rcParams["font.size"]}
    drawing.draw_frame(ax, (0.0, 0.0, 0.0), QUANTITY["ego"], "ego", fontsize=15)
    assert {t.get_fontsize() for t in ax.texts[3:]} == {15}


def test_every_colour_comes_from_the_caller(ax):
    drawing.draw_frame(ax, (0.0, 0.0, 0.0), QUANTITY["theta"], "w")
    assert {matplotlib.colors.to_hex(t.get_color()).upper() for t in ax.texts} == {QUANTITY["theta"]}


def test_the_module_sets_no_dpi_no_hex_colour_and_no_font_size_literal():
    tree = ast.parse(Path(drawing.__file__).read_text())
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    keywords = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    assert not [s for s in strings if s.startswith("#")] and "dpi" not in keywords
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and any(k.arg == "fontsize" and isinstance(k.value, ast.Constant) for k in n.keywords)]


def test_draw_track_plots_the_drive_to_scale(ax):
    tel = run(Car(), "straight", max_steps=5)
    drawing.draw_track(ax, tel, QUANTITY["psi"])
    assert len(ax.lines) == 1 and ax.get_aspect() == 1.0
