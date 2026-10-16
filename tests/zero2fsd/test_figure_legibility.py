# tests/zero2fsd/test_figure_legibility.py
"""Gate G8: every text label of every course figure is legible on a phone.

The rule from the build briefs: any digit is at least 9 px tall when the saved PNG is shown 330 px wide.  A text's digit
height comes from its own font properties (the glyph "0" of its FontProperties), in points, converted to pixels at the
save dpi, then scaled by 330 / (the saved PNG's real width: bbox_inches can change it).  Same mechanism as gate G7: render
each unit's figures through make_unit_figures with Figure.savefig observed.
"""
import struct
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.textpath import TextPath

from make_figures import make_unit_figures

REPO = Path(__file__).resolve().parents[2]
UNITS = sorted(p.parent for p in REPO.glob("course/*/figures.py"))
PHONE_WIDTH_PX = 330
MIN_DIGIT_PX = 9.0
V1_REASON = "v1 figure; rebuilt by the v2 builder"
KNOWN_ILLEGIBLE = {  # unit -> why its figures are expected to fail until redesigned; removed as each unit is rebuilt
    "0.1.1-meet-your-car": "figures below 9 px at phone width (gate G8 first report)",
    "0.1.2-images-are-arrays": "figures below 9 px at phone width (gate G8 first report)",
    "0.1.3-your-first-perception-box": V1_REASON,
    "0.1.4-closing-the-loop": V1_REASON,
}


def png_width(path) -> int:
    """Pixel width from the PNG header (the IHDR width sits at bytes 16..20)."""
    with open(path, "rb") as f:
        return struct.unpack(">I", f.read(24)[16:20])[0]


def digit_px(text: Text, dpi: float) -> float:
    """Height in PNG pixels of the glyph "0" in this text's own font."""
    return TextPath((0, 0), "0", prop=text.get_fontproperties()).get_extents().height * dpi / 72


def _drawn_texts(fig):
    """Visible, non-empty Text artists as drawn: tick labels the axis would draw (not the spare ones), titles, annotations."""
    fig.canvas.draw()
    spare = {t for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for tick in axis.get_major_ticks() + axis.get_minor_ticks()
             for t in (tick.label1, tick.label2)}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            spare -= {t for tick in axis._update_ticks() for t in (tick.label1, tick.label2)}
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in spare]


def smallest_text(fig, path, dpi) -> tuple[str, float]:
    """(string, effective px at phone width) of the smallest text in the figure that was saved to `path`."""
    scale = PHONE_WIDTH_PX / png_width(path)
    return min(((t.get_text().strip(), digit_px(t, dpi) * scale) for t in _drawn_texts(fig)), key=lambda pair: pair[1], default=("", float("inf")))


def render_unit(unit_dir, out_dir, monkeypatch) -> dict[str, tuple[str, float, int]]:
    """{figure file: (smallest text, effective px, PNG width)} for every figure the unit's figures.py saves."""
    found = {}
    save = Figure.savefig

    def observe(self, path, *args, **kwargs):
        save(self, path, *args, **kwargs)
        dpi = kwargs.get("dpi", self.dpi)
        dpi = self.dpi if dpi == "figure" else dpi
        text, px = smallest_text(self, path, dpi)
        found[Path(path).name] = (text, px, png_width(path))

    monkeypatch.setattr(Figure, "savefig", observe)
    make_unit_figures(unit_dir, out_dir)
    return found


def too_small(found) -> list[str]:
    return [f"{name}: '{text}' is {px:.1f} px at {PHONE_WIDTH_PX} px wide (PNG {width} px)"
            for name, (text, px, width) in sorted(found.items(), key=lambda item: item[1][1]) if px < MIN_DIGIT_PX]


@pytest.mark.parametrize("unit", [
    pytest.param(p, marks=pytest.mark.xfail(strict=True, reason=KNOWN_ILLEGIBLE[p.name]) if p.name in KNOWN_ILLEGIBLE else ())
    for p in UNITS], ids=[p.name for p in UNITS])
def test_every_figure_label_is_at_least_9_px_at_phone_width(unit, tmp_path, monkeypatch):
    found = render_unit(unit, tmp_path, monkeypatch)
    assert found, "no figure was drawn"
    assert too_small(found) == []


# --- the measurement itself ---

def _figure_with_text(width_px, size_pt=13, dpi=100):
    fig = Figure(figsize=(width_px / dpi, 1), dpi=dpi)
    fig.text(0.5, 0.5, "0.25", fontsize=size_pt)
    return fig


@pytest.mark.parametrize("width_px, passes", [(330, True), (1300, False)])
def test_13pt_text_at_dpi_100_passes_on_a_330_px_figure_and_fails_on_a_1300_px_one(width_px, passes, tmp_path):
    path = tmp_path / "f.png"
    fig = _figure_with_text(width_px)
    fig.savefig(path, dpi=100)
    assert png_width(path) == width_px
    text, px = smallest_text(fig, path, 100)
    expected = 13 * 100 / 72 * 0.729 * 330 / width_px  # DejaVu Sans digits are 0.729 em tall
    assert text == "0.25" and px == pytest.approx(expected, abs=0.5)
    assert bool(px >= MIN_DIGIT_PX) is passes


def test_spare_tick_labels_and_empty_strings_are_not_measured(tmp_path):
    fig = Figure(figsize=(3.3, 2), dpi=100)
    ax = fig.subplots()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["", ""])
    ax.set_yticks([])
    path = tmp_path / "f.png"
    fig.savefig(path, dpi=100)
    assert smallest_text(fig, path, 100) == ("", float("inf"))
