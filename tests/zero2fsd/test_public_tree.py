"""The public tree stays clean: no personal paths, no dangling internal-doc references, no private identifiers.

Rules are structural; the one named list is PRIVATE, because privacy terms are names. The legacy v1 directories
(named in the README) are not reviewed, so the docs/ reference rule skips them; the other two rules scan every tracked file.
"""
import functools
import re
import subprocess

import pytest

from _course import REPO

LEGACY = ("modules/", "notebooks/", "docs/", "visual_explainers/")
HOME_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
# docs/, ./docs/, ../docs/ and root-relative /docs/ (GitHub resolves "](/docs/x)" from the repo root); never a URL's or a subfolder's /docs/.
DOCS_REF = re.compile(r"(?:(?<![\w.])(?:\.{1,2}/|/)+|(?<![\w/.]))(docs/[\w./-]*[\w/])")
# The named lists: privacy and positioning terms are inherently names. The word "ruling(s)" is case-insensitive, its id token
# is not, so "ruling a mistake" passes while "Ruling F", "ruling: U3a", "Rulings SL-3" and "ruling #10b" fail.
PRIVATE = re.compile(r"(?i:intervu\s*hq)|(?i:rulings?)[\s:#\u2010-\u2015-]{1,3}(?:[A-Z]{1,3}-?\d+[a-z]?|\d+[a-z]?|[A-Z]\b)")
OPEN_SOURCE = re.compile(r"open[\W_]{0,2}source", re.I)  # the course is "free for non-commercial use", never "open source"
UNICODE_HYPHENS = {c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}
# Internal agent-workflow documents, by what they say wherever they sit: agent tooling, approval notes, prompt templates
# and plan/spec/review headers.
AGENT_WORKFLOW = re.compile(
    r"For agentic workers|superpowers:|SDD ledger|REQUIRED SUB-SKILL"
    r"|approved[- ]by[- ](?:Vivek|delegation)|(?:needs|delegated by) Vivek|Vivek can reverse|Vivek's (?:decisions|review|OK)"
    r"|\bscratchpad/"
    r"|into an? (?:agent|LLM)\b"
    r"|(?m:^\**(?:Status|Authority|Researched|Target course|Sources consulted)\**:)"
)
INTERNAL_DIRS = ("docs/superpowers/", "docs/writing/", "docs/redesign/", "docs/COURSE_GENERATION_PROMPT_TEMPLATE.md")  # removed from the public tree
THIS_FILE = "tests/zero2fsd/test_public_tree.py"


def home_paths(text):
    return HOME_PATH.findall(text)


def dangling_docs_refs(text, exists):
    return [ref for ref in DOCS_REF.findall(text) if not exists(ref.rstrip("/"))]


def private_identifiers(text):
    return PRIVATE.findall(text)


def open_source_claims(text):
    return OPEN_SOURCE.findall(text.translate(UNICODE_HYPHENS))


def is_internal_document(name, text):
    return name.startswith(INTERNAL_DIRS) or bool(AGENT_WORKFLOW.search(text))


@functools.cache
def index_blobs(repo=None):
    """(path, text) for every file in the git index, i.e. what would be published. Undecodable bytes become U+FFFD, never a skip."""
    repo = repo or REPO
    entries = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=repo, capture_output=True, text=True, check=True).stdout.split("\0")
    blobs = [(path, sha) for mode, sha, _, path in (e.replace("\t", " ", 1).split(" ", 3) for e in entries if e) if mode != "160000"]
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, input="".join(f"{sha}\n" for _, sha in blobs).encode(), capture_output=True, check=True).stdout
    pos, texts = 0, []
    for path, _ in blobs:
        header, _, _ = out[pos:].partition(b"\n")
        size = int(header.split()[2])
        start = pos + len(header) + 1
        texts.append((path, out[start:start + size].decode("utf-8", errors="replace")))
        pos = start + size + 1
    return texts


def tracked_texts(skip_legacy=False):
    for name, text in index_blobs(REPO):
        if name != THIS_FILE and not (skip_legacy and name.startswith(LEGACY)):
            yield name, text


def exists_in(paths):
    """A `ref` exists when the index holds that file or anything under that folder (case-sensitive, never the disk)."""
    return lambda ref: ref in paths or any(path.startswith(ref + "/") for path in paths)


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


@pytest.mark.parametrize("text", ["https://intervuhq.com/x", "per Ruling F", "Ruling 7 says", "ruling U3a", "Ruling SL12", "ruling 10b", "RULING F", "ruling: 1.2.6.a", "Rulings F and G", "ruling #12", "ruling-F9", "Ruling SL-3", "IntervuHQ", "INTERVU HQ"])
def test_private_identifiers_are_found(text):
    assert private_identifiers(text)


@pytest.mark.parametrize("text", ["the ruling stands", "Rulings are final", "Ruling out noise", "a ruling a judge made", "the ruling is final"])
def test_private_identifiers_ignore_other_text(text):
    assert not private_identifiers(text)


def test_a_root_relative_or_parent_docs_link_is_found_and_a_subfolder_docs_path_is_not():
    assert dangling_docs_refs("[x](/docs/a.md) and course/../docs/b.md", lambda ref: False) == ["docs/a.md", "docs/b.md"]
    assert not dangling_docs_refs("see src/docs/a.md", lambda ref: False)


def test_a_docs_ref_exists_only_when_the_index_holds_it_exactly():
    exists = exists_in({"docs/a.md", "docs/sub/b.md"})
    assert exists("docs/a.md") and exists("docs/sub") and not exists("docs/A.md") and not exists("docs/c.md")


@pytest.mark.parametrize("text", ["see ./docs/a.md", "see ../docs/a.md", "see docs/a.md"])
def test_docs_reference_prefixes_are_normalised(text):
    assert dangling_docs_refs(text, lambda ref: False) == ["docs/a.md"]


@pytest.mark.parametrize("text", ["an Open-Source project", "open source", "OpenSource", "open\u2011source", "open_source", "open  source", "open\u2013source"])
def test_open_source_claims_are_found(text):
    assert open_source_claims(text)


def test_open_source_claims_ignore_other_text():
    assert not open_source_claims("free for non-commercial use; the source is available")


@pytest.mark.parametrize(("name", "text"), [
    ("notes/x.md", "> **For agentic workers:** do this"), ("a.md", "REQUIRED SUB-SKILL: foo"), ("a.md", "superpowers:writing"),
    ("a.md", "the SDD ledger"), ("n/x.md", "Plan approved by Vivek"), ("n/x.md", "Status: **build from this doc**"),
    ("n/x.md", "**Target course:** x"), ("n/x.md", "Copy this into an LLM session"), ("n/x.md", "needs Vivek's OK"), ("n/x.md", "draft for Vivek's review"), ("n/x.md", "read from scratchpad/notes.md"), ("docs/superpowers/plans/p.md", "plain"), ("docs/writing/v2.md", "plain"), ("docs/redesign/00.md", "plain"), ("docs/COURSE_GENERATION_PROMPT_TEMPLATE.md", "plain"),
])
def test_internal_agent_documents_are_found(name, text):
    assert is_internal_document(name, text)


@pytest.mark.parametrize("text", ["A lecture about cars.", "The car's status: moving", "a prompt about agents and LLMs", "the team approved it"])
def test_ordinary_documents_are_not_internal(text):
    assert not is_internal_document("course/0.1.1/lecture.md", text)


def test_tracked_public_files_hold_no_home_paths():
    assert [name for name, text in tracked_texts() if home_paths(text)] == []


def test_tracked_public_files_reference_no_missing_docs_path():
    exists = exists_in({name for name, _ in index_blobs(REPO)})
    assert [(name, ref) for name, text in tracked_texts(skip_legacy=True) for ref in dangling_docs_refs(text, exists)] == []


def test_tracked_public_files_hold_no_private_identifiers():
    assert [name for name, text in tracked_texts() if private_identifiers(text)] == []


def test_tracked_public_files_make_no_open_source_claim():
    assert [name for name, text in tracked_texts() if open_source_claims(text)] == []


def test_tracked_public_files_hold_no_internal_agent_documents():
    assert [name for name, text in tracked_texts() if is_internal_document(name, text)] == []
