# tests/zero2fsd/test_figure_colours.py
"""Gate G2: no figures.py (nor scripts/figstyle.py) holds a string colour literal; colours come from zero2fsd.palette.
Gate G3: every quantity hue, and every shade of it, reads on the dark figure background (WCAG contrast >= 3:1).
Gate G7: no figure plots data outside its axes (an axes that is a deliberate zoom carries the gid "zoom-inset").
Gate G9 (ruling 10b): a symbol has one colour across the text and the figures; a figure label naming a symbol never wears another symbol's colour.

What counts as a colour literal is derived from matplotlib itself, never from a list of names: the kwarg aliases of
every Artist property whose name contains "color", any kwarg or dict key containing "color"/"colour", any "#hex"
string anywhere, and any positional string that matplotlib's own format parser reads as naming a colour ("ro", "C3").
"""
import ast
import os
import re
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgba
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.figure import Figure
from matplotlib.artist import Artist, ArtistInspector
from matplotlib.axes._base import _process_plot_format
from matplotlib.patches import Polygon
from matplotlib.path import Path as Path_

import figstyle
from make_figures import make_unit_figures
from zero2fsd.drawing import draw_frame
from zero2fsd.palette import CHROME, MIX, QUANTITY, shades

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


# --- G2 on the real tree ---


@pytest.mark.parametrize("source", SCANNED, ids=[f"{p.parent.name}/{p.name}" for p in SCANNED])
def test_no_figure_module_holds_a_string_colour_literal(source):
    """Colours come from zero2fsd.palette (QUANTITY, SCENE, CHROME, shades); the style comes from scripts/figstyle.py."""
    assert colour_literals(source.read_text(encoding="utf-8")) == [], f"{source.relative_to(REPO)} holds colour literals (line, text)"


# --- G7: no data outside its axes ---

ZOOM_INSET = "zoom-inset"  # the gid a figures.py gives an axes that deliberately shows only part of its data
EPS = 1e-9


def _data_points(ax):
    """(x, y) arrays of everything drawn in this axes' own data coordinates: lines and patches (not axhline/axvspan, which span the axes)."""
    for line in ax.lines:
        if line.get_transform() is ax.transData:
            yield line.get_xdata(), line.get_ydata()
    for patch in ax.patches:
        if patch.get_transform() is ax.transData:
            path = patch.get_path().transformed(patch.get_patch_transform())
            vertices = path.vertices if path.codes is None else path.vertices[~np.isin(path.codes, (Path_.CLOSEPOLY, Path_.STOP))]  # those two codes are placeholders matplotlib never draws
            yield vertices[:, 0], vertices[:, 1]


def clipped_data(fig) -> list[str]:
    """One message per axes whose plotted data runs past its limits; an axes with gid ZOOM_INSET is exempt."""
    problems = []
    for index, ax in enumerate(ax for top in fig.axes for ax in (top, *top.child_axes)):
        if ax.get_gid() == ZOOM_INSET:
            continue
        for axis, (lo, hi) in (("x", sorted(ax.get_xlim())), ("y", sorted(ax.get_ylim()))):
            for xs, ys in _data_points(ax):
                values = np.asarray(xs if axis == "x" else ys, dtype=float)
                values = values[np.isfinite(values)]
                if values.size and (values.min() < lo - EPS or values.max() > hi + EPS):
                    problems.append(f"axes {index} ('{ax.get_title()}'): {axis} data {values.min():.3g}..{values.max():.3g} outside {lo:.3g}..{hi:.3g}")
                    break
    return problems


def _seeded_figure(inset_gid=None):
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1, 2], [0, 1, 5])
    ax.axhline(9.0)
    ax.set(xlim=(0, 2), ylim=(0, 2))
    inset = ax.inset_axes([0.5, 0.5, 0.4, 0.4], gid=inset_gid)
    inset.plot([0, 1], [0, 3])
    inset.set(xlim=(0, 1), ylim=(0, 1))
    return fig


def test_a_seeded_clipped_trace_is_flagged_and_a_zoom_inset_or_an_axhline_is_not():
    problems = clipped_data(_seeded_figure())
    assert len(problems) == 2 and "y data 0..5 outside 0..2" in problems[0] and "axes 1" in problems[1]
    assert clipped_data(_seeded_figure(inset_gid=ZOOM_INSET)) == problems[:1]
    assert clipped_data(Figure()) == []


def _polygon_axes(points, close_at=None):
    """An axes over 0..2 x 0..2 holding a closed polygon; `close_at` moves its CLOSEPOLY placeholder vertex there."""
    ax = Figure().subplots()
    ax.set(xlim=(0, 2), ylim=(0, 2))
    polygon = Polygon(points, closed=True)
    if close_at is not None:
        polygon.get_path().vertices[-1] = close_at
    ax.add_patch(polygon)
    return ax.figure


def test_a_closepoly_placeholder_outside_the_axes_is_not_data_but_a_real_vertex_outside_is():
    inside = [(0.5, 0.5), (1.5, 0.5), (1.0, 1.5)]
    assert clipped_data(_polygon_axes(inside, close_at=(-50.0, -50.0))) == []
    assert len(clipped_data(_polygon_axes([*inside[:2], (1.0, 5.0)], close_at=(-50.0, -50.0)))) == 1


def test_a_frame_drawn_inside_its_axes_is_not_clipped():
    ax = Figure().subplots()
    ax.set(xlim=(-1, 4), ylim=(-1, 4))
    draw_frame(ax, (1.0, 1.0, 0.3), "k", "ego", length_m=1.0)
    assert clipped_data(ax.figure) == []


KNOWN_CLIPPING = {  # unit -> what still clips: other units' figures, listed in the 0.1.1 fix report rather than reshaped here
}


@pytest.mark.parametrize("source", [
    pytest.param(p, marks=pytest.mark.xfail(strict=True, reason=KNOWN_CLIPPING[p.parent.name]) if p.parent.name in KNOWN_CLIPPING else ())
    for p in SCANNED[:-1]], ids=[p.parent.name for p in SCANNED[:-1]])
def test_no_figure_plots_data_outside_its_axes(source, tmp_path, monkeypatch):
    """Render every figure of the unit and look at the axes it saved: a trace cut off by set_ylim would otherwise ship."""
    drawn = {}
    save = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, path, *args, **kwargs: (drawn.update({Path(path).name: clipped_data(self)}), save(self, path, *args, **kwargs))[1])
    make_unit_figures(source.parent, tmp_path)
    assert drawn, "no figure was drawn"
    assert {name: found for name, found in drawn.items() if found} == {}


def test_render_workers_defaults_to_the_cpu_count_and_reads_the_env_var(monkeypatch):
    monkeypatch.delenv("ZERO2FSD_FIG_WORKERS", raising=False)
    assert figstyle.render_workers() == os.cpu_count()
    monkeypatch.setenv("ZERO2FSD_FIG_WORKERS", "2")
    assert figstyle.render_workers() == 2


@pytest.mark.parametrize("bad", ["0", "-1", "two", "", "2.5"])
def test_render_workers_rejects_a_bad_value_naming_the_variable_and_the_value(monkeypatch, bad):
    monkeypatch.setenv("ZERO2FSD_FIG_WORKERS", bad)
    with pytest.raises(ValueError, match=f"ZERO2FSD_FIG_WORKERS.*{bad!r}"):
        figstyle.render_workers()


ORPHAN_PARENT = """
import os, sys, time
sys.path.insert(0, {scripts!r})
import figstyle
with figstyle.render_pool() as pool:
    pids = {{pool.submit(os.getpid).result() for _ in range(8)}}
    print(*sorted(pids), flush=True)
    time.sleep(600)
"""


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_render_pool_workers_exit_after_their_parent_is_killed(monkeypatch):
    """SIGKILL (a timeout, a killed pytest) cannot run the parent's cleanup, so each worker must notice by itself."""
    import signal
    import subprocess
    import sys
    import time

    monkeypatch.setenv("ZERO2FSD_FIG_WORKERS", "2")
    parent = subprocess.Popen([sys.executable, "-c", ORPHAN_PARENT.format(scripts=str(REPO / "scripts"))], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONPATH": str(REPO)})
    workers = [int(pid) for pid in parent.stdout.readline().split()]
    try:
        assert workers and all(alive(pid) for pid in workers)
        parent.send_signal(signal.SIGKILL)
        parent.wait()
        time.sleep(10)
        assert [pid for pid in workers if alive(pid)] == []
    finally:
        for pid in workers:
            if alive(pid):
                os.kill(pid, signal.SIGKILL)


# --- G9: a symbol has one colour across the text and the figures ---

SYMBOL_TINT = {"e": "e_off", "ψ": "psi", "δ": "delta", "θ": "theta", "d": None}  # None: the lectures leave the symbol untinted (neutral ink)
SYMBOL = re.compile(r"(?<![\w.])(" + "|".join(SYMBOL_TINT) + r")(?![\w.])")  # a lone symbol; d₁ and "deg" are other words
SPELLED = {"psi": "ψ", "theta": "θ", "delta": "δ"}
EXPRESSION_SYMBOL = {re.compile(r"wrap\(\s*θ\s*[-−]\s*θ_road\s*\)"): "ψ"}  # an expression is owned by the quantity it computes: this one is ψ, though it names θ
OWNER_TOLERANCE = 2.5  # per channel, 0-255: `shades` rounds each channel to an integer


def _rgb255(colour) -> np.ndarray:
    return np.array(to_rgba(colour)[:3]) * 255


def tint_owner(colour) -> str | None:
    """The quantity whose family `colour` belongs to, or None: any point on a hue's `shades` line (hue to the lightest shade), whatever n was."""
    c, best = _rgb255(colour), (OWNER_TOLERANCE, None)
    for name, hue in QUANTITY.items():
        h = _rgb255(hue)
        d = _rgb255(CHROME["text"]) - h
        along = float(np.clip((c - h) @ d / (d @ d), 0, MIX))
        miss = float(np.abs(c - (h + along * d)).max())
        if miss <= best[0]:
            best = (miss, name)
    return best[1]


def named_symbols(label: str) -> set[str]:
    """The symbols a label names, whether typeset ($\\psi$), written out (psi) or as the expression that computes one."""
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", label).replace("$", "").replace("{", "").replace("}", "")
    text = re.sub(r"(?<![A-Za-z])(" + "|".join(SPELLED) + r")(?![A-Za-z])", lambda m: SPELLED[m.group(1).lower()], text.replace("\\", " "), flags=re.I)
    for pattern, symbol in EXPRESSION_SYMBOL.items():
        text = pattern.sub(symbol, text)
    return set(SYMBOL.findall(text))


def _patch_colours(patch) -> list:
    return [] if patch is None else [patch.get_facecolor(), patch.get_edgecolor()]


def _handle_colour(handle):
    return handle.get_color() if hasattr(handle, "get_color") else handle.get_facecolor()


def shown_colours(fig) -> list[tuple[str, list]]:
    """(label, every colour it is shown in): its own ink, its bbox, its annotate arrow, and for a legend entry its swatch."""
    shown = []
    for text in fig.findobj(Text):
        arrow = getattr(text, "arrow_patch", None)
        shown.append((text.get_text(), [text.get_color(), *_patch_colours(text.get_bbox_patch()), *_patch_colours(arrow)]))
    for legend in fig.findobj(Legend):
        shown += [(t.get_text(), [_handle_colour(h)]) for t, h in zip(legend.get_texts(), legend.legend_handles)]
    return shown


def symbol_tint_violations(fig) -> list[str]:
    """One message per label that names a symbol while shown in another symbol's quantity colour, or in any tint for an untinted symbol."""
    problems = []
    for label, colours in shown_colours(fig):
        for owner in {tint_owner(c) for c in colours if to_rgba(c)[3] > 0} - {None}:
            wrong = sorted(s for s in named_symbols(label) if SYMBOL_TINT[s] != owner)
            if wrong:
                problems.append(f"{label!r} names {wrong} in the colour of {owner}")
    return problems


def _label_figure(text, colour, **text_kwargs):
    fig = Figure()
    fig.subplots().text(0.5, 0.5, text, color=colour, **text_kwargs)
    return fig


def _legend_figure(label, line_colour):
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], color=line_colour, label=label)
    ax.legend()
    return fig


def _arrow_figure(label, arrow_colour):
    fig = Figure()
    fig.subplots().annotate(label, xy=(0.2, 0.2), xytext=(0.6, 0.6), color=CHROME["text"], arrowprops={"color": arrow_colour})
    return fig


E_BLUE, PSI_PINK = QUANTITY["e_off"], QUANTITY["psi"]
WRONG_COLOUR = {
    "motivating case: d in e's blue": lambda: _label_figure("d = −1.8 m", E_BLUE),
    "another symbol, another unit's colour": lambda: _label_figure("ψ = 0.2", E_BLUE),
    "a mixed label may not be tinted at all": lambda: _label_figure("d = −1.27, e = +0.53", E_BLUE),
    "theta in psi's colour": lambda: _label_figure("θ", PSI_PINK),
    "shade of any n, end": lambda: _label_figure("d", shades(E_BLUE, 3)[2]),
    "shade of n=3, interior": lambda: _label_figure("d", shades(E_BLUE, 3)[1]),
    "shade of n=7, interior": lambda: _label_figure("d", shades(E_BLUE, 7)[3]),
    "mathtext psi in e's blue": lambda: _label_figure(r"$\psi$", E_BLUE),
    "mathtext theta in psi's colour": lambda: _label_figure(r"$\theta = 0.3$", PSI_PINK),
    "mathtext delta in e's blue": lambda: _label_figure(r"$\mathrm{\delta}$", E_BLUE),
    "legend entry d next to an e-blue line": lambda: _legend_figure("d", E_BLUE),
    "neutral d on an e-blue bbox": lambda: _label_figure("d", CHROME["text"], bbox={"facecolor": E_BLUE}),
    "neutral d with an e-blue outline": lambda: _label_figure("d", CHROME["text"], bbox={"facecolor": "none", "edgecolor": E_BLUE}),
    "neutral d with an e-blue arrow": lambda: _arrow_figure("d", E_BLUE),
    "spelled-out psi in e's blue": lambda: _label_figure("psi = 0.3", E_BLUE),
    "spelled-out theta in psi's colour": lambda: _label_figure("Theta", PSI_PINK),
}
RIGHT_COLOUR = {
    "e in its blue": lambda: _label_figure("e = +0.5", E_BLUE),
    "e in a shade of its blue": lambda: _label_figure("e", shades(E_BLUE, 5)[2]),
    "psi in its pink": lambda: _label_figure("ψ", PSI_PINK),
    "mathtext theta in theta's colour": lambda: _label_figure(r"$\theta = 30°$", QUANTITY["theta"]),
    "spelled-out delta in delta's colour": lambda: _label_figure("delta", QUANTITY["delta"]),
    "the wrap expression is psi": lambda: _label_figure(r"$\mathrm{wrap}(\theta - \theta_{\mathrm{road}})$", PSI_PINK),
    "d in neutral ink": lambda: _label_figure("d = −1.8 m", CHROME["text"]),
    "d1 is another quantity": lambda: _label_figure("d₁ = 2.64 m", QUANTITY["perception"]),
    "no symbol": lambda: _label_figure("distance in m", E_BLUE),
    "neutral d on a panel bbox": lambda: _label_figure("d", CHROME["text"], bbox={"facecolor": CHROME["panel"], "edgecolor": CHROME["border"]}),
    "e legend entry next to its line": lambda: _legend_figure("e", E_BLUE),
    "e with its own arrow": lambda: _arrow_figure("e", E_BLUE),
}


@pytest.mark.parametrize("make", WRONG_COLOUR.values(), ids=WRONG_COLOUR)
def test_a_label_naming_a_symbol_in_another_symbols_colour_is_flagged(make):
    assert len(symbol_tint_violations(make())) == 1


@pytest.mark.parametrize("make", RIGHT_COLOUR.values(), ids=RIGHT_COLOUR)
def test_a_label_in_its_own_symbols_colour_or_neutral_ink_or_without_a_symbol_is_not_flagged(make):
    assert symbol_tint_violations(make()) == []


def test_every_shade_of_every_length_belongs_to_its_quantity_and_the_chrome_text_to_none():
    assert {tint_owner(s) for n in range(2, 9) for s in shades(E_BLUE, n)} == {"e_off"}
    assert {tint_owner(c) for c in CHROME.values()} == {None}


@pytest.mark.parametrize("source", SCANNED[:-1], ids=[p.parent.name for p in SCANNED[:-1]])
def test_no_figure_label_names_a_symbol_in_another_symbols_colour(source, tmp_path, monkeypatch):
    """Render every figure of the unit and read the colour of every label: d drawn in e's blue would ship otherwise."""
    drawn = {}
    save = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, path, *args, **kwargs: (drawn.update({Path(path).name: symbol_tint_violations(self)}), save(self, path, *args, **kwargs))[1])
    make_unit_figures(source.parent, tmp_path)
    assert drawn, "no figure was drawn"
    assert {name: found for name, found in drawn.items() if found} == {}


# --- G9 (ruling 10c): a segment between two points is never drawn raw in e's colour; `drawing.draw_offset` is the one place that does ---


def _is_segment_call(node: ast.Call) -> bool:
    """`ax.plot(*zip(p, q), ...)`, `_brace(ax, p, q, ...)` and `ax.annotate("", ...)`: the calls that draw a line between two chosen points."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name == "plot":
        return bool(node.args) and isinstance(node.args[0], ast.Starred) and isinstance(node.args[0].value, ast.Call) and getattr(node.args[0].value.func, "id", "") == "zip"
    if name == "annotate":
        return bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == ""
    return name == "_brace"


def e_coloured_segments(source: str, e_names: set[str]) -> list[int]:
    """Lines of the raw segment calls in `source` that mention e's colour: a name bound to it (`e_names`) or QUANTITY["e_off"]."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _is_segment_call(node):
            mentions = [n for n in ast.walk(node) if (isinstance(n, ast.Name) and n.id in e_names)
                        or (isinstance(n, ast.Subscript) and ast.unparse(n) == "QUANTITY['e_off']")]
            if mentions:
                lines.append(node.lineno)
    return lines


SEGMENT_VIOLATIONS = {
    "plot between two points": 'ax.plot(*zip(foot, car), color=BLUE, lw=3)\n',
    "brace": '_brace(ax, a, b, BLUE)\n',
    "annotate dimension line": 'ax.annotate("", xy=q, xytext=p, arrowprops={"color": QUANTITY["e_off"]})\n',
    "plot in a keyword": 'bot.plot(*zip(B, Q), lw=3, color=BLUE)\n',
}
SEGMENT_ALLOWED = {
    "the helper": 'draw_offset(ax, foot, car, "d")\n',
    "neutral ink": 'ax.plot(*zip(foot, car), color=TEXT, lw=3)\n_brace(ax, a, b, TEXT)\n',
    "another quantity": 'ax.plot(*zip(car, tip), color=PINK, lw=2)\n',
    "a series in e's colour": 'ax.plot(lap.t, lap.e, color=BLUE, lw=2)\n',
}


@pytest.mark.parametrize("source", SEGMENT_VIOLATIONS.values(), ids=SEGMENT_VIOLATIONS)
def test_a_raw_segment_in_e_blue_is_flagged(source):
    assert e_coloured_segments(source, {"BLUE"}) == [1]


@pytest.mark.parametrize("source", SEGMENT_ALLOWED.values(), ids=SEGMENT_ALLOWED)
def test_the_helper_neutral_segments_other_quantities_and_series_are_not_flagged(source):
    assert e_coloured_segments(source, {"BLUE"}) == []


def test_draw_offset_draws_d_in_neutral_ink_and_e_in_blue_and_nothing_else():
    from zero2fsd.drawing import draw_offset
    colours = {}
    for symbol in ("d", "e"):
        ax = Figure().subplots()
        draw_offset(ax, (0, 0), (1, 1), symbol)
        colours[symbol] = to_hex(ax.texts[0].arrow_patch.get_edgecolor())
    assert colours == {"d": CHROME["text"].lower(), "e": QUANTITY["e_off"].lower()}
    with pytest.raises(KeyError):
        draw_offset(Figure().subplots(), (0, 0), (1, 1), "psi")


@pytest.mark.parametrize("source", SCANNED[:-1], ids=[p.parent.name for p in SCANNED[:-1]])
def test_no_figure_module_draws_a_raw_segment_in_e_blue(source):
    """Distances to the road (d) are drawn by draw_offset, in neutral ink; e's blue on a hand-drawn segment is how d went blue."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"colours_{source.parent.name.replace('.', '_').replace('-', '_')}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    e_hues = {shade.upper() for shade in shades(QUANTITY["e_off"], 4)}
    e_names = {name for name, value in vars(module).items() if isinstance(value, str) and value.upper() in e_hues}
    assert e_coloured_segments(source.read_text(encoding="utf-8"), e_names) == []


# --- alt text in the 1.x lessons names the text ink "light grey": "white" is the painted lane line only (0.1 names paint, so it is not scanned) ---

ALT = re.compile(r"!\[(.*?)\]\(figures/(\w+)\.png\)", re.S)
BARE_WHITE = re.compile(r"\bwhite\b(?! (edge |lane )?line)", re.I)


def bare_white(alt: str) -> list[str]:
    return [m.group(0) for m in BARE_WHITE.finditer(alt)]


def test_white_in_alt_text_is_flagged_unless_it_is_the_painted_line():
    assert bare_white("the world axes in white, a white dot") == ["white", "white"]
    assert bare_white("the yellow line and the white edge line, strips of white line, the white line") == []


@pytest.mark.parametrize("lecture", sorted(REPO.glob("course/1.*/lecture.md")), ids=lambda p: p.parent.name)
def test_no_alt_text_calls_the_text_ink_white(lecture):
    found = {name: bare_white(alt) for alt, name in ALT.findall(lecture.read_text())}
    assert {name: words for name, words in found.items() if words} == {}
