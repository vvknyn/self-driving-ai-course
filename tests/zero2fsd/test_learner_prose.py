"""The gates for what a learner reads: a function that returns None is named, and no dotted exercise id shows (WRITING-v2 §F)."""
import numpy as np
import pytest
import yaml

from _course import REPO, dotted_ids_in_prose
from zero2fsd.grade import check
from zero2fsd.grade._report import Failure, call, is_real


# --- call() ---

def forgot_return(x):
    x * 2


def test_a_function_with_no_return_fails_naming_it_and_saying_none():
    with pytest.raises(Failure, match=r"forgot_return returned None.*`return`") as caught:
        call(forgot_return, 1, given="x=1")
    assert caught.value.shown == {"given": "x=1"}


@pytest.mark.parametrize("answer", [0, 0.0, False, np.array([]), "", []])
def test_a_falsy_or_empty_answer_is_not_a_none_return(answer):
    assert call(lambda: answer, given="g") is answer


def test_a_real_exercise_whose_solution_lost_its_return_fails_on_the_none_message():
    namespace: dict = {}
    exec(compile(_drop_returns((REPO / "solutions" / "course" / "0.1.1" / "solution.py").read_text(), "steps_off_lane"), "<solution>", "exec"), namespace)
    result = check("0.1.1.b", namespace["steps_off_lane"], show=False)
    assert not result.passed and "returned None: it needs a `return` line" in "\n".join(result.details)


def _drop_returns(source, function):
    """`source` with every `return` line of `function` (a top-level def) deleted, its body kept parseable by a `pass`."""
    out, inside = [], False
    for line in source.splitlines():
        if line.startswith("def "):
            inside = line.startswith(f"def {function}(")
        out.append(line.replace("return ", "_ = ", 1) if inside and line.strip().startswith("return ") else line)
    return "\n".join(out)


# --- dotted exercise ids ---

def unit_folder(tmp_path, proof="Your steps_off_lane passes the checker for Exercise A in the lab"):
    (tmp_path / "unit.yaml").write_text(yaml.safe_dump({"title": "T", "outcomes": [{"can": "c", "proof": proof}], "exercises": [{"id": "1.1.2.a", "name": "f"}]}))
    (tmp_path / "lecture.md").write_text('# T\n\nSection 1.1.2 is fine.\n\n```python\ncheck("1.1.2.a")\n```\n')
    return tmp_path


def test_exercise_a_passes_and_an_id_in_a_code_fence_or_a_section_number_is_not_prose(tmp_path):
    assert dotted_ids_in_prose(unit_folder(tmp_path)) == []


def test_a_dotted_id_in_proof_text_is_named(tmp_path):
    hits = dotted_ids_in_prose(unit_folder(tmp_path, proof="passes the checker for 1.1.2.a"))
    assert len(hits) == 1 and "outcomes[0].proof" in hits[0] and "1.1.2.a" in hits[0]


def test_a_dotted_id_in_lecture_prose_is_named_with_its_line(tmp_path):
    folder = unit_folder(tmp_path)
    (folder / "lecture.md").write_text("# T\n\n```\n1.1.2.a\n```\nSee 1.1.2.b now.\n")
    hits = dotted_ids_in_prose(folder)
    assert len(hits) == 1 and hits[0].endswith("lecture.md:6: 1.1.2.b")


KNOWN_DOTTED = {  # unit folder -> why it still shows a dotted id: a pending-ship unit, fixed by its own branch rather than edited here
    "0.1.2-images-are-arrays": "unit.yaml outcomes[2].proof shows 0.1.2.a; 0.1.x is in a pending ship",
}


@pytest.mark.parametrize("folder", [
    pytest.param(p, marks=pytest.mark.xfail(strict=True, reason=KNOWN_DOTTED[p.name]) if p.name in KNOWN_DOTTED else ())
    for p in sorted(p for p in (REPO / "course").iterdir() if p.is_dir() and p.name != "dist")], ids=lambda p: p.name)
def test_no_unit_shows_a_dotted_exercise_id_to_the_learner(folder):
    assert dotted_ids_in_prose(folder) == []


# --- a number returned as a 0-d array ---

@pytest.mark.parametrize("answer", [np.asarray(0.0), np.asarray(3), np.float64(0.5), np.float32(0.5), 0.5], ids=["0d-float", "0d-int", "float64", "float32", "float"])
def test_a_scalar_answer_in_any_numpy_or_python_form_is_a_real_number(answer):
    assert is_real(call(lambda: answer, given="g"))


@pytest.mark.parametrize("answer", [np.array([0.0]), np.asarray(True)], ids=["1-element-1d", "0d-bool"])
def test_a_one_element_vector_or_a_bool_is_not_a_number(answer):
    assert not is_real(call(lambda: answer, given="g"))


def test_a_real_scalar_exercise_passes_with_its_solution_wrapped_in_asarray():
    namespace: dict = {}
    exec(compile((REPO / "solutions" / "course" / "0.1.1" / "solution.py").read_text(), "<solution>", "exec"), namespace)
    assert check("0.1.1.b", lambda *a, **k: np.asarray(namespace["steps_off_lane"](*a, **k)), show=False).passed
