# tests/zero2fsd/test_figure_colours.py
"""Gate G2: no figures.py (nor scripts/figstyle.py) holds a string colour literal; colours come from zero2fsd.palette.
Gate G3: every quantity hue, and every shade of it, reads on the dark figure background (WCAG contrast >= 3:1).

What counts as a colour literal is derived from matplotlib itself, never from a list of names: the kwarg aliases of
every Artist property whose name contains "color", any kwarg or dict key containing "color"/"colour", any "#hex"
string anywhere, and any positional string that matplotlib's own format parser reads as naming a colour ("ro", "C3").
"""
import ast
import re
from pathlib import Path

import pytest
from matplotlib.artist import Artist, ArtistInspector
from matplotlib.axes._base import _process_plot_format

import figstyle
from zero2fsd.palette import CHROME, QUANTITY, shades

REPO = Path(__file__).resolve().parents[2]
SCANNED = sorted(REPO.glob("course/*/figures.py")) + [REPO / "scripts" / "figstyle.py"]
HEX = re.compile(r"#[0-9a-fA-F]{3,8}")


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def colour_aliases() -> set[str]:
    """Every kwarg alias matplotlib accepts for a property whose name contains "color" (c, ec, fc, mec, mfc, ...)."""
    aliases = set()
    for cls in [Artist, *_subclasses(Artist)]:
        for prop, names in ArtistInspector(cls).aliasd.items():
            if "color" in prop:
                aliases |= set(names)
    return aliases


ALIASES = colour_aliases()


def is_colour_name(name: str) -> bool:
    lowered = name.lower()
    return name in ALIASES or "color" in lowered or "colour" in lowered


def _str_constants(node):
    """Every string constant under `node`, not descending into subscript keys (QUANTITY["e_off"] names a palette entry)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            yield node
        return
    for name, child in ast.iter_fields(node):
        if isinstance(node, ast.Subscript) and name == "slice":
            continue
        for item in child if isinstance(child, list) else [child]:
            if isinstance(item, ast.AST):
                yield from _str_constants(item)


def _colour_values(value):
    """(line, text) for the literals a colour-valued expression holds: its strings, or an all-number tuple/list."""
    if isinstance(value, (ast.Tuple, ast.List)) and value.elts and all(
        isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in value.elts
    ):
        return [(value.lineno, ast.unparse(value))]
    return [(c.lineno, c.value) for c in _str_constants(value)]


def _fmt_colour(text: str):
    try:
        return _process_plot_format(text)[2]
    except ValueError:
        return None


def colour_literals(source: str) -> list[tuple[int, str]]:
    """Every (line, literal) in `source` that hands matplotlib a colour as a literal rather than a palette value."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and HEX.search(node.value):
            found.append((node.lineno, node.value))
        elif isinstance(node, ast.keyword) and node.arg and is_colour_name(node.arg):
            found += _colour_values(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and is_colour_name(key.value):
                    found += _colour_values(value)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _fmt_colour(arg.value):
                    found.append((arg.lineno, arg.value))
    return sorted(set(found))


VIOLATIONS = {
    "hex anywhere": 'EDGE = "#666666"\n',
    "colour keyword": 'ax.plot(x, y, color="tab:red")\n',
    "alias keyword": 'ax.plot(x, y, "o", mec="black")\n',
    "keyword containing colour": 'ax.hist(d, edgecolor="0.2")\n',
    "plural keyword": 'ax.eventplot(d, colors=["red"])\n',
    "conditional under a colour keyword": 'ax.text(0, 0, s, color="white" if dark else "black")\n',
    "f-string under a colour keyword": 'ax.plot(t, y, color=f"C{i}")\n',
    "dict key": 'ax.annotate("", xy=p, arrowprops={"arrowstyle": "->", "color": "C3"})\n',
    "format string with a colour": 'ax.plot(x, y, "ro")\n',
    "bare colour string": 'ax.plot(x, y, "C3")\n',
    "rgb tuple": 'ax.plot(x, y, color=(1.0, 0.0, 0.0))\n',
}
ALLOWED = {
    "palette subscript": 'ax.plot(x, y, color=QUANTITY["e_off"])\n',
    "palette shades": 'ax.plot(x, y, color=shades(QUANTITY["e_off"], 3)[i])\n',
    "marker and linestyle formats": 'ax.plot(x, y, "o")\nax.plot(x, y, "--", lw=2)\n',
    "prose": 'NAMES = ("sky", "grass", "road", "yellow", "white")\nax.set_title("the yellow line")\n',
    "label text": 'ax.hist(d, label="true white pixels")\n',
}


def test_the_colour_aliases_come_from_matplotlib_and_include_the_short_ones():
    assert {"c", "ec", "fc", "mec", "mfc"} <= ALIASES


@pytest.mark.parametrize("source", VIOLATIONS.values(), ids=VIOLATIONS)
def test_a_seeded_colour_literal_is_flagged_with_its_line(source):
    hits = colour_literals(source)
    assert hits and all(line == 1 for line, _ in hits), hits


@pytest.mark.parametrize("source", ALLOWED.values(), ids=ALLOWED)
def test_palette_lookups_formats_and_prose_are_not_flagged(source):
    assert colour_literals(source) == []


def test_figstyle_holds_only_palette_colours_and_sets_the_page_colours():
    assert figstyle.RC["figure.facecolor"] == CHROME["background"] and figstyle.RC["axes.facecolor"] == CHROME["panel"]
    assert [c["color"] for c in figstyle.RC["axes.prop_cycle"]] == list(QUANTITY.values())


# --- G3: contrast, computed from the palette values ---


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def luminance(hex_colour: str) -> float:
    """WCAG 2 relative luminance."""
    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in _rgb(hex_colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio, 1.0 (same colour) to 21.0 (black on white)."""
    lighter, darker = sorted((luminance(a), luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def unreadable(hues: dict[str, str], background: str, ratio: float = 3.0) -> dict[str, float]:
    """{name: contrast} for every hue that falls below `ratio` on `background` (empty when all read)."""
    return {name: round(contrast(hue, background), 2) for name, hue in hues.items() if contrast(hue, background) < ratio}


def test_contrast_is_the_wcag_ratio():
    assert round(contrast("#000000", "#FFFFFF"), 1) == 21.0 and contrast("#58C4DD", "#58C4DD") == 1.0
    assert round(contrast("#777777", "#FFFFFF"), 2) == 4.48


def test_a_seeded_dark_hue_is_reported_with_its_ratio():
    assert unreadable({"psi": "#1a1f2b", "e_off": QUANTITY["e_off"]}, CHROME["background"]) == {"psi": 1.18}


@pytest.mark.parametrize("surface", ["background", "panel"])
def test_every_quantity_hue_and_every_shade_reads_on_the_figure_surfaces(surface):
    families = {f"{name}[{i}]": shade for name, hue in QUANTITY.items() for i, shade in enumerate(shades(hue, 4))}
    assert unreadable({**QUANTITY, **families}, CHROME[surface]) == {}


def test_text_reads_on_the_panel_at_the_body_text_ratio():
    assert unreadable({"text": CHROME["text"]}, CHROME["panel"], ratio=4.5) == {}
