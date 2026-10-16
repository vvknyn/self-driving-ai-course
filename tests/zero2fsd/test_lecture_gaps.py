"""Gate G1: no run of lecture prose over 300 words without a figure.

The count drops what a learner does not read as prose (<details> blocks, code fences, $$ displays), splits the rest on
image links, and counts words in each run between two figures (or an end of the lecture).
"""
import re

import pytest

from _course import REPO, syllabus_units

MAX_RUN_WORDS = 300
READY = syllabus_units(REPO, "ready")


def prose_runs(text):
    """The text of each run of prose between two figures, with details, code fences and $$ displays dropped."""
    for pattern in (r"<details>.*?</details>", r"```.*?```", r"\$\$.*?\$\$"):
        text = re.sub(pattern, " ", text, flags=re.S)
    return re.split(r"!\[.*?\]\(.*?\)", text, flags=re.S)


def word_count(run):
    return len(re.findall(r"[A-Za-z0-9][\w'’.−-]*", run))


def over_the_limit(text):
    """[(words, first non-blank line)] for every run longer than MAX_RUN_WORDS."""
    return [(word_count(run), next((line for line in run.splitlines() if line.strip()), "").strip()[:80])
            for run in prose_runs(text) if word_count(run) > MAX_RUN_WORDS]


@pytest.mark.parametrize("words, flagged", [(300, False), (301, True)])
def test_a_301_word_run_is_flagged_and_a_300_word_one_is_not(words, flagged):
    text = "Intro line.\n" + " ".join(["word"] * (words - 2)) + "\n\n![](figures/a.png)\n\nafter the figure"
    assert bool(over_the_limit(text)) is flagged


def test_words_behind_a_figure_a_details_block_a_fence_or_a_display_do_not_count():
    filler = " ".join(["word"] * 200)
    text = f"{filler}\n![](a.png)\n{filler}\n<details>{filler}</details>\n```\n{filler}\n```\n$$ {filler} $$\n![](b.png)\n{filler}"
    assert over_the_limit(text) == []


@pytest.mark.parametrize("unit", READY, ids=[u["id"] for u in READY])
def test_no_lecture_has_a_run_of_prose_over_300_words_between_figures(unit):
    runs = over_the_limit((REPO / "course" / unit["folder"] / "lecture.md").read_text(encoding="utf8"))
    assert not runs, f"{unit['id']}: runs over {MAX_RUN_WORDS} words: " + "; ".join(f"{n} words, starting {line!r}" for n, line in runs)


def h2_titles(text):
    """The `## ` headings of a lecture, outside code fences and details blocks."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"<details>.*?</details>", " ", text, flags=re.S)
    return re.findall(r"(?m)^## (.+?)\s*$", text)


def recap_after_challenges(text):
    titles = h2_titles(text)
    return "Recap" not in titles or "Challenge problems" not in titles or titles.index("Recap") < titles.index("Challenge problems")


@pytest.mark.parametrize("titles, ok", [(["A", "Recap", "Challenge problems"], True), (["A", "Challenge problems", "Recap"], False), (["A", "Recap"], True), (["A", "Challenge problems"], True)])
def test_recap_must_come_before_challenge_problems(titles, ok):
    assert recap_after_challenges("\n\n".join(f"## {t}" for t in titles)) is ok


LECTURES = sorted((REPO / "course").glob("*/lecture.md"))


@pytest.mark.parametrize("path", LECTURES, ids=[p.parent.name for p in LECTURES])
def test_every_lecture_puts_recap_before_challenge_problems(path):
    assert recap_after_challenges(path.read_text(encoding="utf8")), f"{path.parent.name}: `## Challenge problems` comes before `## Recap`"
