"""A lab that calls or redefines a function an earlier lab asked the learner to write must use the same interface.

The learner carries `world_to_ego`, `estimate_lane`, ... from lesson to lesson; a renamed argument in one lab silently breaks the next.
Checked by `ast` over every ready lab, in syllabus order, against the first lab that defines the name.
"""
import ast

import pytest

from _course import REPO, lab_code, syllabus_units

READY = syllabus_units(REPO, "ready")


def params(function):
    """(names in order, number of required positionals, accepts *args, accepts **kwargs)."""
    a = function.args
    positional = [p.arg for p in a.posonlyargs + a.args]
    return positional + [p.arg for p in a.kwonlyargs], len(positional) - len(a.defaults), a.vararg is not None, a.kwarg is not None


def bad_calls(tree, earlier):
    """Messages for each call in `tree` that does not fit the interface of its name in `earlier` {name: FunctionDef}."""
    own = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    problems = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in earlier and n.func.id not in own):
        names, required, _, _ = params(earlier[call.func.id])
        given = {k.arg for k in call.keywords if k.arg}
        positional = len(call.args)
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            continue
        missing = names[positional:required] if positional < required else []
        if given - set(names) or positional > len(names) or [m for m in missing if m not in given]:
            problems.append(f"{call.func.id} called at line {call.lineno} as {positional} positional + {sorted(given)}, defined as {names}")
    return problems


def redefinitions(tree, earlier):
    """Messages for each top-level def in `tree` whose parameter names differ from the earlier def of the same name."""
    return [f"{n.name} redefined with {params(n)[0]}, earlier {params(earlier[n.name])[0]}"
            for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in earlier and params(n)[0] != params(earlier[n.name])[0]]


def problems_in(labs):
    """labs: [(name, source)] in order. Messages for every interface mismatch with an earlier lab."""
    earlier, out = {}, []
    for name, source in labs:
        tree = ast.parse(lab_code(source))
        out += [f"{name}: {m}" for m in bad_calls(tree, earlier) + redefinitions(tree, earlier)]
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                earlier.setdefault(n.name, n)
    return out


def real_labs():
    return [(u["id"], (REPO / "course" / u["folder"] / "lab.py").read_text(encoding="utf8")) for u in READY]


def test_every_ready_lab_uses_the_interfaces_earlier_labs_defined():
    assert problems_in(real_labs()) == []


def test_the_check_has_something_to_check():
    """Real cross-lab uses exist: 0.1.3 uses 0.1.2's paint_masks, 1.1.4 uses 1.1.2's world_to_ego."""
    labs = dict(real_labs())
    assert "paint_masks(" in labs["0.1.3"] and "def paint_masks" in labs["0.1.2"]
    assert "world_to_ego(" in labs["1.1.4"] and "def world_to_ego" in labs["1.1.2"]


@pytest.mark.parametrize("old, new, expected", [
    ("def world_to_ego(pose, p_world):", "def world_to_ego(car_pose, p_world):", False),  # a rename does not break the positional calls the labs make
    ("def world_to_ego(pose, p_world):", "def world_to_ego(pose, p_world, extra):", True),  # a new required argument
    ("def world_to_ego(pose, p_world):", "def world_to_ego(pose, p_world, extra=0):", False),  # a new optional one is compatible
])
def test_changing_an_earlier_labs_signature_is_caught(old, new, expected):
    labs = real_labs()
    index = next(i for i, (name, _) in enumerate(labs) if name == "1.1.2")
    source = labs[index][1]
    assert old in source, "the fixture signature moved: update this test"
    labs[index] = ("1.1.2", source.replace(old, new))
    keyword_uses = "world_to_ego(" in dict(labs)["1.1.4"]
    assert keyword_uses and bool(problems_in(labs)) is expected


def test_a_keyword_call_breaks_when_the_earlier_lab_renames_that_argument():
    earlier = "# %%\ndef lane_point(masks, rows):\n    return masks\n"
    later = "# %%\nlane_point(masks=1, rows=2)\n"
    assert not problems_in([("a", earlier), ("b", later)])
    assert problems_in([("a", earlier.replace("rows", "band")), ("b", later)])


def test_a_later_labs_redefinition_with_other_argument_names_is_caught():
    earlier = "# %%\ndef estimate_lane(obs):\n    return obs\n"
    later = "# %%\ndef estimate_lane(observation):\n    return observation\n"
    assert problems_in([("a", earlier), ("b", later)]) and not problems_in([("a", earlier), ("b", earlier)])
