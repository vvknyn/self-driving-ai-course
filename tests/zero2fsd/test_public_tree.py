"""The public tree stays clean: no personal paths, no dangling internal-doc references, no private identifiers.

Rules are structural; the one named list is PRIVATE, because privacy terms are names. The legacy v1 directories
(named in the README) are not reviewed and are skipped.
"""
import re
import subprocess

import pytest

from _course import REPO

LEGACY = ("modules/", "notebooks/", "docs/", "visual_explainers/")
HOME_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
DOCS_REF = re.compile(r"(?<![\w/.])docs/[\w./-]*[\w/]")
PRIVATE = re.compile(r"intervuhq|\bRuling [A-Z0-9]\b")  # the only named list: private identifiers and internal ruling jargon
THIS_FILE = "tests/zero2fsd/test_public_tree.py"


def home_paths(text):
    return HOME_PATH.findall(text)


def dangling_docs_refs(text, exists):
    return [ref for ref in DOCS_REF.findall(text) if not exists(ref.rstrip("/"))]


def private_identifiers(text):
    return PRIVATE.findall(text)


def tracked_texts():
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    for name in files:
        if name == THIS_FILE or name.startswith(LEGACY):
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


@pytest.mark.parametrize("text", ["https://intervuhq.com/x", "per Ruling F", "Ruling 7 says"])
def test_private_identifiers_are_found(text):
    assert private_identifiers(text)


@pytest.mark.parametrize("text", ["the ruling stands", "Rulings are final", "Ruling out noise"])
def test_private_identifiers_ignore_other_text(text):
    assert not private_identifiers(text)


def test_tracked_public_files_hold_no_home_paths():
    assert [name for name, text in tracked_texts() if home_paths(text)] == []


def test_tracked_public_files_reference_no_missing_docs_path():
    assert [(name, ref) for name, text in tracked_texts() for ref in dangling_docs_refs(text, exists_in_tree)] == []


def test_tracked_public_files_hold_no_private_identifiers():
    assert [name for name, text in tracked_texts() if private_identifiers(text)] == []
