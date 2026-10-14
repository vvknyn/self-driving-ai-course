"""app.css's first-paint colours are copies of the palette: no hex is typed that the palette does not own."""
import re
from pathlib import Path

import pytest

from zero2fsd import palette

PLAYGROUND = Path(__file__).resolve().parents[2] / "playground"
APP_CSS = PLAYGROUND / "app.css"
ROOT_BLOCK = re.compile(r":root\s*\{(.*?)\}", re.S)
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
ANY_COLOUR = re.compile(r"#[0-9a-f]{3,8}\b|(?<!\w)(?:rgb|hsl)a?\(", re.I)  # every spelling a stylesheet or script can type a colour in; not `paintRgb(`


def _root() -> str:
    found = ROOT_BLOCK.search(APP_CSS.read_text())
    assert found, "app.css declares a :root block"
    return found.group(1)


def test_every_first_paint_colour_in_root_is_a_palette_value():
    owned = {value.lower() for value in (*palette.CHROME.values(), *palette.QUANTITY.values())}
    literals = HEX.findall(_root())
    assert literals, ":root declares colours"
    strays = [hex_ for hex_ in literals if hex_.lower() not in owned]
    assert strays == [], f"colours in app.css :root that the palette does not own: {strays}"


def test_the_chrome_fallbacks_match_chrome_key_for_key():
    """app.js sets --<key> from palette.json's chrome; the first-paint value under the same name must agree."""
    declared = {name: value.lower() for name, value in re.findall(r"--([a-z]+):\s*(#[0-9a-fA-F]{6})", _root())}
    for key, value in palette.CHROME.items():
        assert declared.get(key) == value.lower(), f"--{key} first-paint fallback differs from CHROME[{key!r}]"


@pytest.mark.parametrize("text", ["color: #fff", "fill: #FF00aa11", "background: rgb(1, 2, 3)", "x = 'rgba(0,0,0,.5)'", "hsl(10 20% 30%)"])
def test_the_colour_pattern_catches_every_spelling(text):
    assert ANY_COLOUR.search(text)


def test_the_colour_pattern_ignores_names_that_merely_contain_one():
    assert not ANY_COLOUR.search("paintRgb(canvas); #editor-note { margin: 0 }")


@pytest.mark.parametrize("name", ["app.css", "app.js", "index.html"])
def test_no_colour_literal_outside_the_root_block(name):
    """Past the first paint every colour is a `var(--...)` or comes from palette.json at runtime."""
    text = (PLAYGROUND / name).read_text()
    outside = ROOT_BLOCK.sub("", text)
    strays = ANY_COLOUR.findall(outside)
    assert strays == [], f"colour literals in {name} outside :root: {strays}"
