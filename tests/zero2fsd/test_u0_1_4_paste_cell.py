"""0.1.4's Exercise B says its paste cell "sees every name" that 0.1.3's lab gives a box. Derive both name sets from the labs."""
import ast
import re
from pathlib import Path

COURSE = Path(__file__).resolve().parents[2] / "course"


def cells(lab: Path) -> list[tuple[str, str]]:
    """The `# %%` cells of a lab.py as (header line, source) pairs."""
    parts = re.split(r"^(# %%.*)$", lab.read_text(), flags=re.M)
    return list(zip(parts[1::2], parts[2::2]))


def bound(source: str, *, assignments: bool) -> set[str]:
    """Top-level names the cell's code binds by import, def or class (and by assignment when `assignments`)."""
    code = re.sub(r"^(\s*)%.*$", r"\1pass", source, flags=re.M)  # a Jupyter magic line becomes a no-op
    names: set[str] = set()
    for node in ast.parse(code).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif assignments and isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def names_before_exercise(lab: Path, exercise: str, *, setup_assignments: bool = True) -> set[str]:
    """Names bound by the setup cell, plus the imports/defs/classes of every other cell before the first `exercise` cell."""
    names: set[str] = set()
    for header, source in cells(lab):
        if f'"exercise:{exercise}' in header:
            return names
        if header.startswith("# %%") and "[markdown]" not in header:
            names |= bound(source, assignments=setup_assignments and 'tags=["setup"]' in header)
    raise AssertionError(f"no cell tagged exercise:{exercise} in {lab}")


def test_paste_cell_sees_every_name_a_box_from_the_0_1_3_lab_can_use():
    lab_013 = next(COURSE.glob("0.1.3-*/lab.py"))
    lab_014 = next(COURSE.glob("0.1.4-*/lab.py"))
    given = names_before_exercise(lab_013, "0.1.3.")
    assert given, "found no names in 0.1.3's lab"
    missing = given - names_before_exercise(lab_014, "0.1.4.b")
    assert not missing, f"0.1.4's paste cell does not see {sorted(missing)}, which 0.1.3's lab gives a box"
