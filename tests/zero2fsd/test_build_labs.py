import json
import re
from html.parser import HTMLParser

import nbformat
from matplotlib import image
import pytest

import build_labs
import make_figures
from _course import REPO, REPO_SLUG, parse_lab
from zero2fsd.palette import CHROME

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


def test_figures_are_drawn_in_the_dark_figure_style(mini_repo):
    assert make_figures.main(["--root", str(mini_repo)]) == 0
    corner = image.imread(mini_repo / "course" / "0.1.1-mini" / "figures" / "curve.png")[0, 0, :3]
    assert tuple(round(float(v) * 255) for v in corner) == tuple(int(CHROME["background"][i:i + 2], 16) for i in (1, 3, 5))


def test_a_figures_module_without_make_fails_loudly(mini_repo):
    (mini_repo / "course" / "0.1.1-mini" / "figures.py").write_text("x = 1\n")
    with pytest.raises(AttributeError):
        make_figures.main(["--root", str(mini_repo)])


def test_committed_notebooks_are_in_sync_with_their_lab_py():
    """Part C: after editing a lab.py, run scripts/build_labs.py and commit the notebook."""
    assert build_labs.main(["--check"]) == 0, f"stale notebook under {REPO / 'course'}"


class _Images(HTMLParser):
    """The (src, alt, caption) of every <img> in a markdown source; the caption is the text of the <em> that follows it."""

    def __init__(self):
        super().__init__()
        self.images, self.in_em = [], False

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append([dict(attrs).get("src"), dict(attrs).get("alt"), ""])
        self.in_em |= tag == "em"

    def handle_endtag(self, tag):
        self.in_em &= tag != "em"

    def handle_data(self, data):
        if self.in_em:
            self.images[-1][2] += data


def shown_images(markdown):
    parser = _Images()
    parser.feed(markdown)
    return [tuple(image) for image in parser.images]


def lecture_figures_of(unit):
    """{file: (number, alt)} read straight from lecture.md: each line that is only an image is the next figure."""
    lines = (unit / "lecture.md").read_text(encoding="utf-8").splitlines()
    found = filter(None, (re.fullmatch(r'!\[(.*)\]\(figures/(\S+?)(?: "[^"]*")?\)', line.strip()) for line in lines))
    return {m[2]: (n, m[1]) for n, m in enumerate(found, 1)}


@pytest.mark.parametrize("notebook", sorted((REPO / "course").glob("*/lab.ipynb")), ids=lambda p: p.parent.name)
def test_every_lecture_figure_is_in_the_lab_with_its_caption_and_a_url_colab_can_load(notebook):
    unit, lecture = notebook.parent, lecture_figures_of(notebook.parent)
    markdown = "\n".join("".join(c["source"]) for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "markdown")
    assert not re.search(r"""(src=["']|\]\()(\./)?figures/""", markdown), "a repo-relative figure path is left: Colab cannot load it"
    prefix = f"https://raw.githubusercontent.com/{REPO_SLUG}/{build_labs.FIGURE_REF}/course/{unit.name}/figures/"
    shown = shown_images(markdown)
    for src, alt, caption in shown:
        assert src.startswith(prefix), src
        file = src[len(prefix):]
        assert (unit / "figures" / file).is_file(), f"{src} is not a file of this repo"
        assert file in lecture, f"{file} is not a figure of lecture.md"
        number, plain = lecture[file][0], lecture[file][1].replace("`", "")  # a `code` span renders as <code>: no backticks reach the learner
        assert (alt, caption) == (plain, f"Lecture figure {number}. {plain}")
    assert {src[len(prefix):] for src, _, _ in shown} == set(lecture)  # none of the lecture's figures is missing from the lab


MINI_LECTURE = """\
# Mini

![Pic one & "two"](figures/one.png)

```
![inside a fence](figures/fenced.png)
```

A sentence with ![inline](figures/inline.png) in it.

![Second](figures/two.png "a title")
"""


def mini_lab_showing(mini_repo, ref):
    """The fixture unit with a lecture of two figures (one, two) and a lab whose one markdown cell holds `ref` on its own line."""
    unit = mini_repo / "course" / "0.1.1-mini"
    (unit / "lecture.md").write_text(MINI_LECTURE, encoding="utf-8")
    (unit / "figures").mkdir(exist_ok=True)
    for name in ("one", "two", "fenced", "inline", "other"):
        (unit / "figures" / f"{name}.png").write_bytes(b"\x89PNG")
    (unit / "lab.py").write_text(f"# %% [markdown]\n# Before\n#\n# {ref}\n#\n# After\n", encoding="utf-8")
    return unit / "lab.py"


def test_a_figure_ref_becomes_the_lectures_numbered_caption_with_prose_untouched(mini_repo):
    for file, number, alt in (("one", 1, 'Pic one & "two"'), ("two", 2, "Second")):  # the fenced and inline images are not figures
        cell = nbformat.reads(build_labs.compile_lab(mini_lab_showing(mini_repo, f"![](figures/{file}.png)")), as_version=4).cells[0]
        url = f"https://raw.githubusercontent.com/{REPO_SLUG}/{build_labs.FIGURE_REF}/course/0.1.1-mini/figures/{file}.png"
        assert shown_images(cell.source) == [(url, alt, f"Lecture figure {number}. {alt}")]
        assert cell.source.startswith("Before\n\n<") and cell.source.endswith(">\n\nAfter") and "![" not in cell.source


@pytest.mark.parametrize("ref", ["![](figures/other.png)", "![](figures/fenced.png)", "![](figures/inline.png)"])
def test_a_ref_to_an_image_that_is_not_a_lecture_figure_fails_the_build(mini_repo, ref):
    with pytest.raises(ValueError, match="not a standalone image of lecture.md"):
        build_labs.compile_lab(mini_lab_showing(mini_repo, ref))


def test_a_ref_to_a_lecture_figure_missing_from_figures_fails_the_build(mini_repo):
    lab = mini_lab_showing(mini_repo, "![](figures/two.png)")
    (lab.parent / "figures" / "two.png").unlink()
    with pytest.raises(ValueError, match="does not exist"):
        build_labs.compile_lab(lab)


def test_a_lab_figure_with_its_own_alt_text_fails_the_build(mini_repo):
    with pytest.raises(ValueError, match="empty alt"):
        build_labs.compile_lab(mini_lab_showing(mini_repo, "![my own caption](figures/two.png)"))


def lecture_with_alt(mini_repo, alt):
    """The fixture lab showing two.png, whose lecture alt text for two.png is `alt`."""
    lab = mini_lab_showing(mini_repo, "![](figures/two.png)")
    lecture = lab.parent / "lecture.md"
    lecture.write_text(lecture.read_text(encoding="utf-8").replace("![Second]", f"![{alt}]"), encoding="utf-8")
    return lab


def test_a_code_span_in_the_alt_is_code_in_the_caption_and_plain_text_in_the_img_alt(mini_repo):
    cell = nbformat.reads(build_labs.compile_lab(lecture_with_alt(mini_repo, "Drive `a<b` on `gentle` & stop")), as_version=4).cells[0]
    url = f"https://raw.githubusercontent.com/{REPO_SLUG}/{build_labs.FIGURE_REF}/course/0.1.1-mini/figures/two.png"
    assert f'<img src="{url}" alt="Drive a&lt;b on gentle &amp; stop">' in cell.source
    assert "<figcaption><em>Lecture figure 2. Drive <code>a&lt;b</code> on <code>gentle</code> &amp; stop</em></figcaption>" in cell.source
    assert "`" not in cell.source
    assert shown_images(cell.source) == [(url, "Drive a<b on gentle & stop", "Lecture figure 2. Drive a<b on gentle & stop")]


def test_markdown_characters_inside_a_code_span_are_allowed(mini_repo):
    cell = nbformat.reads(build_labs.compile_lab(lecture_with_alt(mini_repo, "mask `a[0] * 2` costs `$x`")), as_version=4).cells[0]
    assert "Lecture figure 2. mask <code>a[0] * 2</code> costs <code>$x</code></em>" in cell.source


@pytest.mark.parametrize("alt", ["*bold* roads", "a [link](x) here", "cost $5 each", "`open span", "two `a` and `b` and `c"],
                         ids=["emphasis", "link", "maths", "unbalanced", "unbalanced-after-spans"])
def test_markdown_outside_a_code_span_or_an_unbalanced_backtick_fails_the_build(mini_repo, alt):
    with pytest.raises(ValueError, match="alt text"):
        build_labs.compile_lab(lecture_with_alt(mini_repo, alt))


@pytest.mark.parametrize(("first", "second"), [("two", "one"), ("one", "one")], ids=["reversed", "repeated"])
def test_lab_figures_out_of_lecture_order_fail_the_build(mini_repo, first, second):
    lab = mini_lab_showing(mini_repo, f"![](figures/{first}.png)")
    lab.write_text(lab.read_text(encoding="utf-8") + f"\n# %% [markdown]\n# ![](figures/{second}.png)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of lecture order"):
        build_labs.compile_lab(lab)


def test_lab_figures_in_lecture_order_across_cells_build(mini_repo):
    lab = mini_lab_showing(mini_repo, "![](figures/one.png)")
    lab.write_text(lab.read_text(encoding="utf-8") + "\n# %%\nx = 1\n\n# %% [markdown]\n# ![](figures/two.png)\n", encoding="utf-8")
    cells = nbformat.reads(build_labs.compile_lab(lab), as_version=4).cells
    shown = shown_images(cells[0].source) + shown_images(cells[2].source)
    assert [url.rsplit("/", 1)[1] for url, _, _ in shown] == ["one.png", "two.png"]
