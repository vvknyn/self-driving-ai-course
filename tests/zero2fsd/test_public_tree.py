"""The public tree stays clean: no personal paths, no dangling internal-doc references, no private identifiers.

Rules are structural; the one named list is PRIVATE, because privacy terms are names. The legacy v1 directories
(named in the README) are not reviewed, so the docs/ reference rule skips them; the other two rules scan every tracked file.
"""
import re
import subprocess

import pytest

from _course import REPO

LEGACY = ("modules/", "notebooks/", "docs/", "visual_explainers/")
HOME_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
DOCS_REF = re.compile(r"(?<![\w/.])((?:\.{1,2}/)*)(docs/[\w./-]*[\w/])")  # docs/, ./docs/, ../docs/; not a URL's /docs/
# The named lists: privacy and positioning terms are inherently names. "ruling" is case-insensitive, its id token is not,
# so "ruling a mistake" passes while "Ruling F", "ruling U3a", "Ruling SL12" and "ruling 10b" fail.
PRIVATE = re.compile(r"intervuhq|(?i:ruling)\s+(?:[A-Z]{1,3}\d+[a-z]?|\d+[a-z]?|[A-Z]\b)")
OPEN_SOURCE = re.compile(r"open[- ]?source", re.I)  # the course is "free for non-commercial use", never "open source"
AGENT_WORKFLOW = re.compile(r"For agentic workers|superpowers:|SDD ledger|REQUIRED SUB-SKILL")
INTERNAL_DIRS = ("docs/superpowers/", "docs/writing/", "docs/redesign/", "docs/COURSE_GENERATION_PROMPT_TEMPLATE.md")  # removed from the public tree
THIS_FILE = "tests/zero2fsd/test_public_tree.py"


def home_paths(text):
    return HOME_PATH.findall(text)


def dangling_docs_refs(text, exists):
    return [ref for _, ref in DOCS_REF.findall(text) if not exists(ref.rstrip("/"))]


def private_identifiers(text):
    return PRIVATE.findall(text)


def open_source_claims(text):
    return OPEN_SOURCE.findall(text)


def is_internal_document(name, text):
    return name.startswith(INTERNAL_DIRS) or bool(AGENT_WORKFLOW.search(text))


def tracked_texts(skip_legacy=False):
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    for name in files:
        if name == THIS_FILE or (skip_legacy and name.startswith(LEGACY)):
            continue
        try:
            yield name, (REPO / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):  # binary file, or tracked but deleted on disk
            continue


def exists_in_tree(ref):
    return (REPO / ref).exists()


@pytest.mark.parametrize("text", ["open /Users/jane/Downloads/x", "cd /home/bob/src", "a /Users/vivek/b"])
def test_home_paths_are_found(text):
    assert home_paths(text)


@pytest.mark.parametrize("text", ["src/Users.py", "/usr/local/bin", "the /home page", "~/Downloads"])
def test_home_paths_ignore_other_text(text):
    assert not home_paths(text)


def test_a_docs_reference_to_a_missing_path_is_found():
    assert dangling_docs_refs("see docs/superpowers/plans/x.md.", lambda ref: False) == ["docs/superpowers/plans/x.md"]


def test_a_docs_reference_to_an_existing_path_or_a_url_is_not_flagged():
    assert not dangling_docs_refs("see docs/a.md and https://docs.opencv.org/4.x and x.com/docs/y", lambda ref: ref == "docs/a.md")


@pytest.mark.parametrize("text", ["https://intervuhq.com/x", "per Ruling F", "Ruling 7 says", "ruling U3a", "Ruling SL12", "ruling 10b", "RULING F"])
def test_private_identifiers_are_found(text):
    assert private_identifiers(text)


@pytest.mark.parametrize("text", ["the ruling stands", "Rulings are final", "Ruling out noise", "a ruling a judge made", "the ruling is final"])
def test_private_identifiers_ignore_other_text(text):
    assert not private_identifiers(text)


@pytest.mark.parametrize("text", ["see ./docs/a.md", "see ../docs/a.md", "see docs/a.md"])
def test_docs_reference_prefixes_are_normalised(text):
    assert dangling_docs_refs(text, lambda ref: False) == ["docs/a.md"]


@pytest.mark.parametrize("text", ["an Open-Source project", "open source", "OpenSource"])
def test_open_source_claims_are_found(text):
    assert open_source_claims(text)


def test_open_source_claims_ignore_other_text():
    assert not open_source_claims("free for non-commercial use; the source is available")


@pytest.mark.parametrize(("name", "text"), [
    ("notes/x.md", "> **For agentic workers:** do this"), ("a.md", "REQUIRED SUB-SKILL: foo"), ("a.md", "superpowers:writing"),
    ("a.md", "the SDD ledger"), ("docs/superpowers/plans/p.md", "plain"), ("docs/writing/v2.md", "plain"), ("docs/redesign/00.md", "plain"), ("docs/COURSE_GENERATION_PROMPT_TEMPLATE.md", "plain"),
])
def test_internal_agent_documents_are_found(name, text):
    assert is_internal_document(name, text)


def test_ordinary_documents_are_not_internal():
    assert not is_internal_document("course/0.1.1/lecture.md", "A lecture about cars.")


def test_tracked_public_files_hold_no_home_paths():
    assert [name for name, text in tracked_texts() if home_paths(text)] == []


def test_tracked_public_files_reference_no_missing_docs_path():
    assert [(name, ref) for name, text in tracked_texts(skip_legacy=True) for ref in dangling_docs_refs(text, exists_in_tree)] == []


def test_tracked_public_files_hold_no_private_identifiers():
    assert [name for name, text in tracked_texts() if private_identifiers(text)] == []


def test_tracked_public_files_make_no_open_source_claim():
    assert [name for name, text in tracked_texts() if open_source_claims(text)] == []


def test_tracked_public_files_hold_no_internal_agent_documents():
    assert [name for name, text in tracked_texts() if is_internal_document(name, text)] == []
