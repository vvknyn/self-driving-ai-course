"""app.css's first-paint colours are copies of the palette: no hex is typed that the palette does not own."""
import re
from pathlib import Path

from zero2fsd import palette

APP_CSS = Path(__file__).resolve().parents[2] / "playground" / "app.css"
ROOT_BLOCK = re.compile(r":root\s*\{(.*?)\}", re.S)
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


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
