import json

import nbformat
import pytest

import build_labs
import make_figures
from _course import REPO, parse_lab

LAB = """\
# %% [markdown]
# # Title
#
# body with `code`

# %%
%pip install -q something

# %% tags=["setup", "exercise:1.1.a"]
x = 1
"""


def test_parse_lab_kinds_tags_and_markdown_stripping():
    md, bootstrap, tagged = parse_lab(LAB)
    assert (md.kind, md.tags, md.source) == ("markdown", [], "# Title\n\nbody with `code`")
    assert (bootstrap.kind, bootstrap.tags, bootstrap.source) == ("code", [], '%pip install -q something')
    assert (tagged.kind, tagged.tags, tagged.source) == ("code", ["setup", "exercise:1.1.a"], "x = 1")


@pytest.mark.parametrize("bad", ["stray line\n# %%\nx = 1\n", "# %% tags=['single-quoted']\nx = 1\n", '# %% tags=["unclosed"\nx = 1\n'])
def test_parse_lab_refuses_malformed_input_loudly(bad):
    with pytest.raises(ValueError):
        parse_lab(bad)


def test_compile_lab_is_a_valid_notebook_with_stable_ids_and_tags(mini_repo):
    text = build_labs.compile_lab(mini_repo / "course" / "0.1.1-mini" / "lab.py")
    nb = nbformat.reads(text, as_version=4)
    nbformat.validate(nb)
    ids = [c.id for c in nb.cells]
    assert ids == [f"c{i:03d}" for i in range(len(nb.cells))]
    assert [c.cell_type for c in nb.cells[:3]] == ["markdown", "code", "code"]
    assert nb.cells[0].source.startswith("# Mini unit") and "`# `" in nb.cells[0].source
    assert nb.cells[1].metadata == {} and nb.cells[2].metadata["tags"] == ["setup"]
    assert sum("exercise:0.1.1.a" in c.metadata.get("tags", []) for c in nb.cells) == 2  # one exercise may span two cells
    assert nb.metadata["kernelspec"]["name"] == "python3"
    assert text == build_labs.compile_lab(mini_repo / "course" / "0.1.1-mini" / "lab.py")  # deterministic
    assert json.loads(text)["cells"][0]["id"] == "c000" and text.endswith("}\n")


def test_main_builds_every_lab_and_check_tracks_freshness(mini_repo, capsys):
    lab = mini_repo / "course" / "0.1.1-mini" / "lab.py"
    notebook = lab.with_suffix(".ipynb")
    assert build_labs.main(["--root", str(mini_repo), "--check"]) == 1  # missing
    assert not notebook.exists()  # --check never writes
    assert build_labs.main(["--root", str(mini_repo)]) == 0 and notebook.exists()
    first = notebook.read_bytes()
    assert build_labs.main(["--root", str(mini_repo), "--check"]) == 0
    assert build_labs.main(["--root", str(mini_repo)]) == 0 and notebook.read_bytes() == first  # byte-identical rebuild
    lab.write_text(lab.read_text() + "\n# %%\nprint('edited')\n")
    assert build_labs.main(["--root", str(mini_repo), "--check"]) == 1  # stale
    assert "stale" in capsys.readouterr().out


def test_main_builds_only_the_named_unit_dirs(mini_repo):
    other = mini_repo / "course" / "0.1.9-other"
    other.mkdir()
    (other / "lab.py").write_text("# %%\nx = 1\n")
    assert build_labs.main(["--root", str(mini_repo), str(other)]) == 0
    assert (other / "lab.ipynb").exists() and not (mini_repo / "course" / "0.1.1-mini" / "lab.ipynb").exists()


def test_figures_are_byte_identical_across_runs(mini_repo):
    png = mini_repo / "course" / "0.1.1-mini" / "figures" / "curve.png"
    assert make_figures.main(["--root", str(mini_repo)]) == 0
    first = png.read_bytes()
    assert first.startswith(b"\x89PNG")
    assert make_figures.main(["--root", str(mini_repo)]) == 0 and png.read_bytes() == first


def test_a_figures_module_without_make_fails_loudly(mini_repo):
    (mini_repo / "course" / "0.1.1-mini" / "figures.py").write_text("x = 1\n")
    with pytest.raises(AttributeError):
        make_figures.main(["--root", str(mini_repo)])


def test_committed_notebooks_are_in_sync_with_their_lab_py():
    """Part C: after editing a lab.py, run scripts/build_labs.py and commit the notebook."""
    assert build_labs.main(["--check"]) == 0, f"stale notebook under {REPO / 'course'}"
