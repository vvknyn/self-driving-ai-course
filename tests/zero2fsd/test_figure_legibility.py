# tests/zero2fsd/test_figure_legibility.py
"""Gate G8: every text label of every course figure is legible on a phone.

The rule (phone legibility): any digit is at least 9 px tall when the saved PNG is shown on a phone.  The site shows a PNG at
min(330, its pixel width / FIG_DENSITY) CSS px.  A text's digit height comes from its own font properties (the glyph "0"
of its FontProperties), in points, converted to pixels at the dpi the PNG was saved at (savefig.dpi under figstyle.RC), then
scaled by (displayed width / the saved PNG's real width: bbox_inches can change it).  Same mechanism as gate G7: render
each unit's figures through make_unit_figures with Figure.savefig observed.

Gate G9 (no two labels overlap) and the PNG dpi gate (every committed PNG is saved at FIG_DPI) live here too.
"""
import functools
import struct
import tempfile
import zlib
from pathlib import Path

import pytest
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.textpath import TextPath

from figstyle import FIG_DENSITY, FIG_DPI
from make_figures import make_unit_figures

REPO = Path(__file__).resolve().parents[2]
UNITS = sorted(p.parent for p in REPO.glob("course/*/figures.py"))
PHONE_WIDTH_PX = 330
MIN_DIGIT_PX = 9.0
MAX_OVERLAP_PX = 1  # two labels may touch by this much in width or height


def png_width(path) -> int:
    """Pixel width from the PNG header (the IHDR width sits at bytes 16..20)."""
    with open(path, "rb") as f:
        return struct.unpack(">I", f.read(24)[16:20])[0]


def png_dpi(path) -> int | None:
    """Dots per inch from the PNG's pHYs chunk (pixels per metre), or None when it has none."""
    data = Path(path).read_bytes()
    pos = 8
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind == b"pHYs":
            return round(struct.unpack(">I", data[pos + 8:pos + 12])[0] * 0.0254)
        pos += 12 + length
    return None


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
    width = png_width(path)
    scale = min(PHONE_WIDTH_PX, width / FIG_DENSITY) / width
    return min(((t.get_text().strip(), digit_px(t, dpi) * scale) for t in _drawn_texts(fig)), key=lambda pair: pair[1], default=("", float("inf")))


def _label(t: Text) -> str:
    return t.get_text().strip()


def _extent(t: Text):
    return Text.get_window_extent(t)  # the text alone: an Annotation's own extent also covers its arrow


def overlapping_pairs(fig, dpi) -> list[tuple[str, str, float, float]]:
    """(text A, text B, overlap width px, overlap height px) for each pair of drawn texts, at `dpi`, that intersect by
    more than MAX_OVERLAP_PX in both directions.  A pair is skipped when one text sits inside the other's bbox patch."""
    fig.set_dpi(dpi)
    texts = _drawn_texts(fig)
    boxes = [(t, _extent(t)) for t in texts]
    renderer = fig._get_renderer()
    for t in texts:
        if t.get_bbox_patch() is not None:
            t.update_bbox_position_size(renderer)  # a bbox patch holds its size only while the text draws
    patches = {t: t.get_bbox_patch().get_window_extent(renderer) for t in texts if t.get_bbox_patch() is not None}

    def on_badge(inner, outer):
        p, b = patches.get(outer), dict(boxes)[inner]
        return p is not None and p.x0 - 1 <= b.x0 and b.x1 <= p.x1 + 1 and p.y0 - 1 <= b.y0 and b.y1 <= p.y1 + 1

    found = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            w, h = min(ba.x1, bb.x1) - max(ba.x0, bb.x0), min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > MAX_OVERLAP_PX and h > MAX_OVERLAP_PX and not on_badge(a, b) and not on_badge(b, a):
                found.append((_label(a), _label(b), w, h))
    return found


def render_unit(unit_dir, out_dir) -> dict[str, tuple[str, float, int, list]]:
    """{figure file: (smallest text, effective px, PNG width, overlapping pairs)} for every figure the unit's figures.py saves.
    The text is measured at the dpi the PNG was saved at: the savefig call's own dpi, else savefig.dpi as figstyle.RC set it."""
    found = {}
    save = Figure.savefig

    def observe(self, path, *args, **kwargs):
        dpi = kwargs.get("dpi", rcParams["savefig.dpi"])  # read before the save: the rc in force when the save ran
        dpi = self.dpi if dpi == "figure" else dpi
        save(self, path, *args, **kwargs)
        text, px = smallest_text(self, path, dpi)
        found[Path(path).name] = (text, px, png_width(path), overlapping_pairs(self, dpi))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Figure, "savefig", observe)
        make_unit_figures(unit_dir, out_dir)
    return found


@functools.cache
def rendered(unit_dir) -> dict[str, tuple[str, float, int, list]]:
    """render_unit once per unit for the whole test session: G8 and G9 read the same figures."""
    with tempfile.TemporaryDirectory() as out:
        return render_unit(unit_dir, Path(out))


def too_small(found) -> list[str]:
    return [f"{name}: '{text}' is {px:.1f} px at {PHONE_WIDTH_PX} px wide (PNG {width} px)"
            for name, (text, px, width, _) in sorted(found.items(), key=lambda item: item[1][1]) if px < MIN_DIGIT_PX]


def overlaps(found_by_unit) -> list[str]:
    return [f'{unit} / {name}: "{a}" × "{b}" (overlap {w:.0f}×{h:.0f} px)'
            for unit, found in found_by_unit.items() for name, entry in sorted(found.items()) for a, b, w, h in entry[3]]


def wrong_dpi(pngs) -> list[str]:
    return [f"{p}: {png_dpi(p)} dpi" for p in pngs if png_dpi(p) != FIG_DPI]


@pytest.mark.parametrize("unit", UNITS, ids=[p.name for p in UNITS])
def test_every_figure_label_is_at_least_9_px_at_phone_width(unit):
    found = rendered(unit)
    assert found, "no figure was drawn"
    assert too_small(found) == []


def test_no_two_labels_of_any_figure_overlap():
    found = overlaps({unit.name: rendered(unit) for unit in UNITS})
    assert not found, "overlapping labels:\n" + "\n".join(found)


def test_every_committed_figure_png_is_saved_at_the_figstyle_dpi():
    pngs = sorted(REPO.glob("course/*/figures/*.png"))
    assert pngs, "no committed figure PNG"
    assert wrong_dpi(pngs) == [], f"each figure PNG must be saved at {FIG_DPI} dpi (a figures.py must not set its own)"


# --- the measurement itself ---

def _figure_with_text(width_px, size_pt=13, dpi=100):
    fig = Figure(figsize=(width_px / dpi, 1), dpi=dpi)
    fig.text(0.5, 0.5, "0.25", fontsize=size_pt)
    return fig


@pytest.mark.parametrize("width_px, passes", [(660, True), (1300, False)])
def test_13pt_text_at_dpi_200_passes_on_a_660_px_figure_and_fails_on_a_1300_px_one(width_px, passes, tmp_path):
    """Both PNGs show at 330 px (660 / FIG_DENSITY, and the cap), so the scale is 330 / width at dpi 200."""
    path = tmp_path / "f.png"
    fig = _figure_with_text(width_px, dpi=200)
    fig.savefig(path, dpi=200)
    assert png_width(path) == width_px
    text, px = smallest_text(fig, path, 200)
    expected = 13 * 200 / 72 * 0.729 * 330 / width_px  # DejaVu Sans digits are 0.729 em tall
    assert text == "0.25" and px == pytest.approx(expected, abs=0.5)
    assert bool(px >= MIN_DIGIT_PX) is passes


def test_a_png_under_660_px_wide_shows_at_1_over_FIG_DENSITY_of_its_width_not_at_330(tmp_path):
    path = tmp_path / "f.png"
    fig = _figure_with_text(400, dpi=200)
    fig.savefig(path, dpi=200)
    _, px = smallest_text(fig, path, 200)
    assert px == pytest.approx(13 * 200 / 72 * 0.729 / FIG_DENSITY, abs=0.5)  # not x 330 / 400


@pytest.mark.parametrize("dpi, passes", [(100, False), (FIG_DPI, True)])
def test_png_dpi_gate_fails_a_100_dpi_png_and_passes_a_figure_dpi_one(dpi, passes, tmp_path):
    path = tmp_path / "f.png"
    _figure_with_text(300, dpi=dpi).savefig(path, dpi=dpi)
    assert (wrong_dpi([path]) == []) is passes


def test_png_dpi_gate_fails_a_png_with_no_phys_chunk_and_names_every_bad_file(tmp_path):
    good, bad, bare = tmp_path / "g.png", tmp_path / "b.png", tmp_path / "n.png"
    _figure_with_text(300).savefig(good, dpi=FIG_DPI)
    _figure_with_text(300).savefig(bad, dpi=100)
    data = bad.read_bytes()
    pos = data.index(b"pHYs") - 4
    bare.write_bytes(data[:pos] + data[pos + 21:])  # the pHYs chunk is 4 + 4 + 9 + 4 bytes
    assert png_dpi(bare) is None
    assert wrong_dpi([good, bad, bare]) == [f"{bad}: 100 dpi", f"{bare}: None dpi"]


def _figure_with_two_texts(gap_px):
    fig = Figure(figsize=(4, 2), dpi=100)
    fig.text(0.2, 0.5, "alpha", fontsize=14)
    fig.text(0.2, 0.5 - gap_px / 200, "beta", fontsize=14)
    return fig


def test_g9_fails_two_texts_at_the_same_spot_and_passes_them_50_px_apart():
    assert overlapping_pairs(_figure_with_two_texts(0), 100)[0][:2] == ("alpha", "beta")
    assert overlapping_pairs(_figure_with_two_texts(50), 100) == []


def test_g9_skips_a_label_drawn_on_its_own_badge():
    fig = Figure(figsize=(4, 2), dpi=100)
    fig.text(0.2, 0.5, "badge text", fontsize=14, bbox={"boxstyle": "round", "fc": "w"})
    fig.text(0.2, 0.5, "x", fontsize=8)
    assert overlapping_pairs(fig, 100) == []


def test_spare_tick_labels_and_empty_strings_are_not_measured(tmp_path):
    fig = Figure(figsize=(3.3, 2), dpi=100)
    ax = fig.subplots()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["", ""])
    ax.set_yticks([])
    path = tmp_path / "f.png"
    fig.savefig(path, dpi=100)
    assert smallest_text(fig, path, 100) == ("", float("inf"))
