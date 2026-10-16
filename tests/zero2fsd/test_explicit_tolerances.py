"""Every numeric comparison in a lab, a grader or a preview names its tolerances: numpy's and math's defaults (rtol=1e-5, rel_tol=1e-9) are hidden scales."""
import ast

import pytest

from _course import REPO, lab_code, syllabus_units

REQUIRED = {  # call name -> the keywords it must pass
    "allclose": {"rtol", "atol"},
    "isclose": {"rtol", "atol"},
    "assert_allclose": {"rtol", "atol"},
}
MATH_ISCLOSE = {"rel_tol", "abs_tol"}  # math.isclose spells them differently
READY = syllabus_units(REPO, "ready")
FILES = (
    [(f"course/{u['folder']}/lab.py", lab_code) for u in READY]
    + [(str(p.relative_to(REPO)), str) for pattern in ("zero2fsd/grade/units/*.py", "zero2fsd/preview/units/*.py") for p in sorted(REPO.glob(pattern))]
)


def implicit_tolerances(source):
    """['name at line N: missing x, y'] for each tolerance comparison that leaves a tolerance to the library default."""
    out = []
    for call in (n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)):
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name not in REQUIRED:
            continue
        math_call = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math"
        needed = MATH_ISCLOSE if math_call else REQUIRED[name]
        missing = needed - {k.arg for k in call.keywords}
        if missing:
            out.append(f"{name} at line {call.lineno}: missing {', '.join(sorted(missing))}")
    return out


@pytest.mark.parametrize("source, count", [
    ("np.allclose(a, b)", 1),
    ("np.isclose(a, b, atol=1e-9)", 1),
    ("np.testing.assert_allclose(a, b, rtol=0)", 1),
    ("math.isclose(a, b, rel_tol=1e-9)", 1),
    ("np.allclose(a, b, rtol=0, atol=1e-9)", 0),
    ("math.isclose(a, b, rel_tol=0, abs_tol=1e-9)", 0),
    ("x.isclose_to(a)", 0),
])
def test_the_check_flags_a_default_tolerance_and_passes_an_explicit_one(source, count):
    assert len(implicit_tolerances(source)) == count


def test_the_files_scanned_include_a_lab_a_grader_and_a_preview():
    names = [name for name, _ in FILES]
    assert any(n.endswith("lab.py") for n in names) and any("grade/units" in n for n in names) and any("preview/units" in n for n in names)


@pytest.mark.parametrize("name, read", FILES, ids=[n for n, _ in FILES])
def test_every_tolerance_comparison_names_its_tolerances(name, read):
    source = read((REPO / name).read_text(encoding="utf8"))
    assert implicit_tolerances(source) == []
