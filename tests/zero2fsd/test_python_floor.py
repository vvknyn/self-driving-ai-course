"""Every tracked .py (library, tests, lab code that runs in Colab) must compile on the oldest Python
`requires-python` allows. The suite itself runs on a newer Python, so a 3.12-only construct
(same-quote nested f-string, `type X = int`) would pass here and break `import zero2fsd.grade` for a
learner on 3.10 or 3.11. `ast.parse(feature_version=...)` accepts those f-strings, so the real
interpreter does the compiling."""
import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAGIC = re.compile(r"^(\s*)[%!].*$")
DRIVER = (
    "import json,sys\n"
    "for name, src in json.load(sys.stdin):\n"
    "    try: compile(src, name, 'exec')\n"
    "    except SyntaxError as e: print(f'{name}:{e.lineno}: {e.msg}')\n"
)


def oldest_python(pyproject_text: str) -> str:
    return re.search(r'requires-python\s*=\s*"[>=~ ]*(\d+\.\d+)', pyproject_text).group(1)


def blank_magics(source: str) -> str:
    """IPython magic and shell lines are not Python: neutralise each to `pass` (a magic can be an `if` body), keep line numbers."""
    return "\n".join(MAGIC.sub(r"\1pass", line, count=1) if MAGIC.match(line) else line for line in source.splitlines())


def find_interpreter(version: str) -> str | None:
    found = shutil.which(f"python{version}")
    if found:
        return found
    uv_dirs = glob.glob(str(Path.home() / f".local/share/uv/python/cpython-{version}*/bin/python{version}"))
    return uv_dirs[0] if uv_dirs else None


def syntax_errors(interpreter: str, sources: dict[str, str]) -> list[str]:
    payload = json.dumps([[name, blank_magics(src)] for name, src in sources.items()])
    run = subprocess.run([interpreter, "-c", DRIVER], input=payload, capture_output=True, text=True, check=True)
    return run.stdout.splitlines()


@pytest.fixture(scope="module")
def floor_python():
    version = oldest_python((ROOT / "pyproject.toml").read_text())
    interpreter = find_interpreter(version)
    if interpreter is None:
        pytest.fail(f"Python {version} (the oldest pyproject.toml allows) is not installed; run: uv python install {version}")
    return interpreter


BAD_FSTRING = 'x = f"{fmt(v, ".2g")}"\n'
BAD_EXCEPT_STAR = "try:\n    pass\nexcept* ValueError:\n    pass\n"
BAD_TYPE_STMT = "type X = int\n"


@pytest.mark.parametrize("source", [BAD_FSTRING, BAD_EXCEPT_STAR, BAD_TYPE_STMT], ids=["nested-fstring", "except-star", "type-stmt"])
def test_a_construct_newer_than_the_floor_is_flagged(floor_python, source):
    assert syntax_errors(floor_python, {"bad.py": source})


def test_a_lab_with_a_magic_line_and_a_new_fstring_fails_and_with_only_the_magic_line_passes(floor_python):
    magic = "%pip install -q zero2fsd\n!ls\n"
    assert syntax_errors(floor_python, {"lab.py": magic + "x = 1\n"}) == []
    assert syntax_errors(floor_python, {"lab.py": magic + BAD_FSTRING})


def test_the_floor_is_read_from_pyproject():
    assert oldest_python('requires-python = ">=3.10"') == "3.10"


def test_every_tracked_python_file_compiles_on_the_oldest_supported_python(floor_python):
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"], capture_output=True, text=True, check=True).stdout.split()
    sources = {name: (ROOT / name).read_text() for name in tracked if (ROOT / name).is_file()}
    assert syntax_errors(floor_python, sources) == []
