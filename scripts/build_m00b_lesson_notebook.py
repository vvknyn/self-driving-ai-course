#!/usr/bin/env python3
"""Regenerate the appendix 00b lesson notebook.

Writes ``notebooks/00_neural_networks_and_autograd.ipynb`` next to this course staging tree.
The notebook is the lesson: run it top to bottom. This script does not execute it.

    python staging/self-driving-ai-course/scripts/build_m00b_lesson_notebook.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(source: str):
    return new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return new_code_cell(textwrap.dedent(source).strip() + "\n")


def build() -> nbformat.NotebookNode:
    cells = []

    cells.append(md("""
    # Appendix 00b — Neural networks from scratch

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vvknyn/self-driving-ai-course/blob/main/notebooks/00_neural_networks_and_autograd.ipynb)

    Optional appendix after Module 00. Module 00 already trains with PyTorch. PyTorch hides the graph: you call `backward`, and the partial derivatives appear, but you never see the parents of each number. This notebook builds that graph in ordinary Python with the course engine in `modules/00_nn_scratch/`.

    A **Value** is one number plus a memory of how it was made. A **gradient** is how much a final number would change if you nudged this one. An **operation** (add, multiply, square, ReLU, tanh) is the step that produced a new Value from older ones.

    Each section explains one idea, then runs code. **Predict first**, then execute the cell.
    """))

    cells.append(md("""
    **Purpose:** locate the course repo (clone it on Colab), import the scratch engine, and keep plots in the notebook.

    **Predict:** the cell prints `repo has engine: True`. The engine file name ends with `00_nn_scratch/engine.py`.
    """))

    cells.append(code("""
    import importlib.util
    import io
    import os
    import random
    import subprocess
    import sys
    from contextlib import redirect_stdout
    from pathlib import Path

    import matplotlib
    import matplotlib.pyplot as plt

    def keep_inline():
        # A course import can select a file-only backend.
        # Put figures back on the notebook backend so plt.show() renders here.
        matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
        ip = None
        try:
            from IPython import get_ipython
            ip = get_ipython()
        except Exception:
            ip = None
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")

    %matplotlib inline

    def find_repo(start: Path) -> Path:
        for p in [start, start.parent]:
            if (p / "modules" / "00_nn_scratch" / "engine.py").is_file():
                return p.resolve()
        return start.resolve()

    REPO = find_repo(Path.cwd())
    if not (REPO / "modules" / "00_nn_scratch" / "engine.py").is_file():
        dest = Path.cwd() / "self-driving-ai-course"
        if not (dest / "modules" / "00_nn_scratch" / "engine.py").is_file():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/vvknyn/self-driving-ai-course.git", str(dest)],
                check=True,
            )
        REPO = dest.resolve()
        os.chdir(REPO)
    else:
        os.chdir(REPO)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "matplotlib", "numpy"],
        check=True,
    )

    sys.path.insert(0, str(REPO / "modules" / "00_nn_scratch"))
    import engine as engine_mod
    from engine import Value
    from nn import MLP, Neuron
    from train_toy_driving import generate_driving_dataset, train

    engine_path = Path(engine_mod.__file__).resolve()
    print("repo has engine:", (REPO / "modules" / "00_nn_scratch" / "engine.py").is_file())
    print("engine file ends with:", "modules/" + engine_path.as_posix().split("modules/")[-1])
    keep_inline()
    print("backend:", matplotlib.get_backend())
    """))

    cells.append(md("""
    `repo has engine: True`. The file line ends with `modules/00_nn_scratch/engine.py`. That file is the `Value` class this notebook calls. `backend: inline`, so the later loss plot renders in the cell.
    """))

    cells.append(md("## 1. Why this appendix"))
    cells.append(md("""
    PyTorch's `backward` is the same idea as the method below, written in C++ for big arrays. Here the method is a Python function you can read: walk the graph from the final number back to the inputs, and multiply local slopes along the way.

    **Predict:** a fresh `Value(1.5)` has gradient `0.0` until something calls `backward`. The class really does have a `backward` method.
    """))
    cells.append(code("""
    fresh = Value(1.5, label="fresh")
    print("data:", fresh.data)
    print("grad:", fresh.grad)
    print("has backward:", hasattr(fresh, "backward"))
    print("op so far:", repr(fresh._op))
    print("parents:", len(fresh._prev))
    """))
    cells.append(md("""
    `data: 1.5` and `grad: 0.0`. `has backward: True`. `op so far` is an empty string, and `parents: 0`. A number you typed in has no history yet. The next section builds a number that remembers its parents.
    """))

    cells.append(md("## 2. A Value that remembers"))
    cells.append(md("""
    A **parent** is a Value that was used to make this one. The **op** is the name of that step (`*` for multiply). The engine stores both on the result, so a later backward pass can walk home.

    **Predict:** `2 * -3` is `-6`, the op is `*`, and there are 2 parents.
    """))
    cells.append(code("""
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    product = a * b
    print("a:", a.data)
    print("b:", b.data)
    print("product data:", product.data)
    print("product op:", product._op)
    print("product parents:", len(product._prev))
    print("parent values:", sorted(p.data for p in product._prev))
    """))
    cells.append(md("""
    `product data: -6.0`, `product op: *`, `product parents: 2`. The parent values are `-3.0` and `2.0`, which are `b` and `a`. The result stores the answer and the recipe. It does not yet store a gradient: nothing has asked how the answer depends on `a` or `b`.
    """))

    cells.append(md("## 3. A derivative by hand"))
    cells.append(md("""
    Take three numbers, `a = 2`, `b = -3`, `c = 0.5`. Let `u = a * b + c` and `L = u ** 2`. `L` is the number we care about. The **partial derivative** `dL/da` is the slope of `L` when only `a` moves.

    **First try.** The slope of "something squared" is twice that something, so write `dL/da = 2 * u` and stop. That forgets that `u` itself changes when `a` changes.

    A **nudge check** (central difference) moves `a` by a tiny `eps` in both directions and divides the change in `L` by `2 * eps`. If the hand slope is right, it agrees with the nudge.

    **Predict:** the forgotten-chain slope will not agree with the nudge. The cell prints `match: False`.
    """))
    cells.append(code("""
    a_data, b_data, c_data = 2.0, -3.0, 0.5
    u = a_data * b_data + c_data
    L = u ** 2
    wrong_da = 2 * u
    eps = 1e-5

    def loss_at(a_value):
        return (a_value * b_data + c_data) ** 2

    numeric_da = (loss_at(a_data + eps) - loss_at(a_data - eps)) / (2 * eps)
    print("a, b, c:", a_data, b_data, c_data)
    print("u:", u)
    print("L:", L)
    print("wrong dL/da:", wrong_da)
    print("numeric dL/da:", round(numeric_da, 4))
    print("match:", abs(wrong_da - numeric_da) < 1e-3)
    """))
    cells.append(md("""
    `u: -5.5` and `L: 30.25`. The first try prints `wrong dL/da: -11.0`. The nudge prints `numeric dL/da: 33.0`. `match: False`. Twice `u` is the slope with respect to `u`, not with respect to `a`. The attempt fails the nudge.
    """))
    cells.append(md("""
    **Fix.** The chain rule says multiply the slopes along the path. `dL/du = 2 * u`, and `du/da = b`, so `dL/da = (2 * u) * b`. The same idea gives `dL/db = (2 * u) * a` and `dL/dc = 2 * u`.

    **Predict:** the corrected `dL/da` matches the nudge (`match numeric: True`). You should see `33.0`, `-22.0`, and `-11.0`.
    """))
    cells.append(code("""
    hand_da = (2 * u) * b_data
    hand_db = (2 * u) * a_data
    hand_dc = 2 * u
    print("hand dL/da:", hand_da)
    print("hand dL/db:", hand_db)
    print("hand dL/dc:", hand_dc)
    print("match numeric:", abs(hand_da - numeric_da) < 1e-3)
    """))
    cells.append(md("""
    `hand dL/da: 33.0`, `hand dL/db: -22.0`, `hand dL/dc: -11.0`. `match numeric: True`. The missing factor was `b`, which is `-3`, and `-11 * -3 = 33`. This expression was short enough to fix by hand. A network is thousands of these steps. That is why each Value remembers its parents: the program can apply the chain rule by walking the graph, instead of you rewriting the algebra for every new model.
    """))
    cells.append(md("""
    **Predict:** building `L = (a * b + c) ** 2` out of Values records op `**2` on `L`, data `30.25`, and a parent whose op is `+` and whose data is `-5.5`.
    """))
    cells.append(code("""
    a = Value(a_data, label="a")
    b = Value(b_data, label="b")
    c = Value(c_data, label="c")
    u_node = a * b + c
    loss = u_node ** 2
    parent = next(iter(loss._prev))
    print("L op:", loss._op)
    print("L data:", loss.data)
    print("parent op:", parent._op)
    print("parent data:", parent.data)
    print("parent count:", len(loss._prev))
    """))
    cells.append(md("""
    `L op: **2`, `L data: 30.25`, `parent op: +`, `parent data: -5.5`, `parent count: 1`. The square remembers the sum that produced it. The sum, in turn, remembers the product and `c`. The graph is the hand calculation, stored as objects.
    """))

    cells.append(md("## 4. Backward on the graph"))
    cells.append(md("""
    `backward` sets the final Value's gradient to `1` (the slope of a number with respect to itself) and visits every node from the end back to the inputs. At a multiply, the local rule is "the other parent's data." At an add, the local rule is `1`. At a power, the local rule is the power rule. Each step multiplies by the gradient flowing in from downstream, and **adds** into `.grad` so two uses of the same number both count.

    **Predict:** after `loss.backward()`, the three gradients match the hand numbers `33.0`, `-22.0`, and `-11.0`.
    """))
    cells.append(code("""
    loss.backward()
    print("loss grad:", loss.grad)
    print("engine dL/da:", a.grad)
    print("engine dL/db:", b.grad)
    print("engine dL/dc:", c.grad)
    print("match hand:", a.grad == hand_da and b.grad == hand_db and c.grad == hand_dc)
    """))
    cells.append(md("""
    `loss grad: 1.0`. That is the seed: the slope of the final number with respect to itself. `engine dL/da: 33.0`, `engine dL/db: -22.0`, `engine dL/dc: -11.0`. `match hand: True`. The walk reproduced the corrected hand calculation. You did not type the chain rule again; the ops stored it when the graph was built.
    """))
    cells.append(md("""
    **One number, two uses.** `y = x * x` with `x = 3` is `9`. The parent set stores `x` once, because a set cannot hold the same object twice. The multiply rule still runs both sides, so the gradient should be `3 + 3 = 6`, which is also `2 * x`.

    **Predict:** `unique parents` is `1` and the gradient is `6.0`.
    """))
    cells.append(code("""
    x = Value(3.0, label="x")
    y = x * x
    print("y data:", y.data)
    print("unique parents:", len(y._prev))
    y.backward()
    print("x grad:", x.grad)
    """))
    cells.append(md("""
    `y data: 9.0`, `unique parents: 1`, `x grad: 6.0`. Both uses were added into that one parent. The engine uses `+=` so the second use does not erase the first.
    """))
    cells.append(md("""
    **Another try that fails.** Call `backward` a second time on the same graph without clearing `.grad`. The adds run again on top of the old gradient.

    **Predict:** the second call prints `12.0`. Setting the gradient back to `0` and calling `backward` once more returns `6.0`.
    """))
    cells.append(code("""
    y.backward()
    print("second backward:", x.grad)
    x.grad = 0.0
    y.grad = 0.0
    y.backward()
    print("after clear:", x.grad)
    """))
    cells.append(md("""
    `second backward: 12.0`. The first result `6.0` was still stored, and the second call added the same slope on top of it. `after clear: 6.0`. Training code calls `zero_grad` before every `backward` for this reason: gradients add, so a leftover gradient is a second, unwanted update.
    """))

    cells.append(md("## 5. ReLU and tanh"))
    cells.append(md("""
    **ReLU** keeps positive numbers and turns every other number into `0`. Its slope is `1` on a positive input and `0` on a negative or zero input. A **dead ReLU** is a negative input: the output is `0` and the gradient is `0`, so earlier numbers get no signal through this node.

    **tanh** squashes any real number into the open range from `-1` to `1`. Its slope is `1 - tanh(x) ** 2`, which is near `1` at `0` and near `0` when the output is already close to `1` or `-1`.

    **Predict:** ReLU on `-1.5` prints data `0.0` and grad `0.0`. ReLU on `1.5` prints data `1.5` and grad `1.0`. tanh on `2` is close to `1`, with a small slope. tanh on `8` saturates. The engine has `relu` and `tanh`. It does not have a leaky ReLU method.
    """))
    cells.append(code("""
    print("has relu:", hasattr(Value, "relu"))
    print("has tanh:", hasattr(Value, "tanh"))
    print("has leaky:", hasattr(Value, "leaky_relu") or hasattr(Value, "leaky"))

    for raw in (-1.5, 0.0, 1.5):
        node = Value(raw)
        out = node.relu()
        out.backward()
        print(f"relu input {raw}: data {out.data} grad {node.grad}")

    steep = Value(2.0)
    bent = steep.tanh()
    bent.backward()
    print("tanh input 2 data:", f"{bent.data:.4f}")
    print("tanh input 2 grad:", f"{steep.grad:.4f}")
    flat = Value(8.0)
    flat_out = flat.tanh()
    flat_out.backward()
    print("tanh input 8 data:", f"{flat_out.data:.4f}")
    print("tanh input 8 grad:", f"{flat.grad:.4f}")
    """))
    cells.append(md("""
    `has relu: True`, `has tanh: True`, `has leaky: False`. There is no leaky method to compare against, so a negative input does not pass a small slope. It passes none.

    `relu input -1.5: data 0.0 grad 0.0` is the dead unit. `relu input 0.0` is also data `0.0` and grad `0.0` (the engine treats the kink as slope 0). `relu input 1.5: data 1.5 grad 1.0` is the live unit: the number is unchanged and the slope is 1.

    `tanh input 2 data: 0.9640` and `tanh input 2 grad: 0.0707`. The output is already near the ceiling, so the slope is small. `tanh input 8 data: 1.0000` and `tanh input 8 grad: 0.0000`. Farther out, tanh has saturated and the slope is gone.
    """))
    cells.append(md("""
    Put ReLU on the earlier expression. With `a = 2`, `b = -1`, `c = 1`, the sum is `-1`, and ReLU turns it into `0`. With `b = 1` instead, the sum is `3`, and ReLU leaves it alone.

    **Predict:** the dead graph prints output `0.0` and all three gradients `0.0`. The live graph prints output `3.0` and gradients `1.0`, `2.0`, `1.0`.
    """))
    cells.append(code("""
    dead_a, dead_b, dead_c = Value(2.0), Value(-1.0), Value(1.0)
    dead = (dead_a * dead_b + dead_c).relu()
    dead.backward()
    print("dead pre-activation:", 2.0 * -1.0 + 1.0)
    print("dead output:", dead.data)
    print("dead grads:", dead_a.grad, dead_b.grad, dead_c.grad)

    live_a, live_b, live_c = Value(2.0), Value(1.0), Value(1.0)
    live_pre = live_a * live_b + live_c
    live = live_pre.relu()
    live.backward()
    print("live pre-activation:", live_pre.data)
    print("live output:", live.data)
    print("live grads:", live_a.grad, live_b.grad, live_c.grad)
    """))
    cells.append(md("""
    `dead pre-activation: -1.0`, `dead output: 0.0`, `dead grads: 0.0 0.0 0.0`. The gate is shut, so `a`, `b`, and `c` learn nothing from this output.

    `live pre-activation: 3.0`, `live output: 3.0`, `live grads: 1.0 2.0 1.0`. The slope through ReLU is 1, the slope of the sum with respect to `a` is `b` which is `1`, and the slope with respect to `b` is `a` which is `2`. Same chain rule as section 4, with an extra gate.
    """))

    cells.append(md("## 6. A neuron and an MLP"))
    cells.append(md("""
    A **neuron** is a weighted sum plus a **bias** (one extra number added at the end), then an optional bend such as tanh or ReLU. A **layer** is several neurons reading the same inputs. An **MLP** (multi-layer perceptron) stacks layers. `parameters()` is the flat list of every weight and bias. Those are the numbers training will nudge.

    The toy driving net in the repo is `MLP(nin=2, nouts=[8, 8, 1], activations=["tanh", "tanh", "linear"])`. Count by hand: first layer `2 * 8` weights plus `8` biases, second layer `8 * 8` weights plus `8` biases, last layer `8 * 1` weights plus `1` bias.

    **Predict:** the three layers print `24`, `72`, and `9` parameters. The total prints `105`. One neuron with 2 inputs prints `3` parameters (two weights and a bias).
    """))
    cells.append(code("""
    random.seed(0)
    neuron = Neuron(2, nonlin="tanh")
    print("neuron parameters:", len(neuron.parameters()))
    print("neuron weight and bias:", [round(p.data, 4) for p in neuron.parameters()])
    print("neuron inputs:", 0.5, -0.2)
    neuron_out = neuron([0.5, -0.2])
    print("neuron output:", round(neuron_out.data, 4))

    random.seed(0)
    net = MLP(nin=2, nouts=[8, 8, 1], activations=["tanh", "tanh", "linear"])
    layer_counts = [len(layer.parameters()) for layer in net.layers]
    print("layer parameters:", layer_counts)
    print("total parameters:", len(net.parameters()))
    print("layer1:", 2 * 8, "+", 8, "=", 2 * 8 + 8)
    print("layer2:", 8 * 8, "+", 8, "=", 8 * 8 + 8)
    print("layer3:", 8 * 1, "+", 1, "=", 8 * 1 + 1)
    """))
    cells.append(md("""
    `neuron parameters: 3`. The printed weight and bias are `0.6659`, `-0.9875`, and `0.0` (bias starts at 0). On inputs `0.5` and `-0.2`, `neuron output: 0.4857`. That single number is the whole forward pass of one neuron: weighted sum, then tanh.

    `layer parameters: [24, 72, 9]` and `total parameters: 105`. The breakdown lines are `16 + 8 = 24`, `64 + 8 = 72`, and `8 + 1 = 9`. Add those three layer totals and you get the same `105`.
    """))

    cells.append(md("## 7. Train on a tiny driving toy"))
    cells.append(md("""
    `generate_driving_dataset` draws a speed and a distance, each between 0 and 1. The label is `+1` (safe) when distance is above `0.75 * speed ** 2 + 0.1`, and `-1` (brake) otherwise. The boundary bends with speed squared, so a straight cut in the speed-distance plane cannot follow it exactly. The trainer still uses a small MLP and a hinge: `relu(1 - y * prediction)`, averaged over the rows. A hinge of `0` means the prediction is on the correct side of the margin. The **learning rate** `0.08` is the step size. An **epoch** is one pass over the rows.

    **Predict:** 40 rows, seed 42. More rows are safe than brake. The first row is a brake (`-1`) because its distance sits below the boundary at that speed.
    """))
    cells.append(code("""
    data = generate_driving_dataset(num_samples=40, seed=42)
    safe = sum(1 for _, label in data if label > 0)
    brake = sum(1 for _, label in data if label < 0)
    speed0, dist0 = data[0][0]
    label0 = data[0][1]
    boundary0 = 0.75 * (speed0 ** 2) + 0.1
    print("samples:", len(data))
    print("safe +1:", safe)
    print("brake -1:", brake)
    print("always-safe accuracy %:", round(100.0 * safe / len(data), 1))
    print("first speed:", round(speed0, 4))
    print("first distance:", round(dist0, 4))
    print("first label:", label0)
    print("boundary at that speed:", round(boundary0, 4))
    print("distance below boundary:", dist0 < boundary0)
    """))
    cells.append(md("""
    `samples: 40`, `safe +1: 25`, `brake -1: 15`. Always saying safe scores `62.5` percent, because 25 of 40 labels are `+1`.

    The first row is speed `0.6755`, distance `0.0738`, label `-1.0`. The boundary at that speed is `0.4422`, and `distance below boundary: True`. The label is brake because the lead vehicle is closer than the curved rule allows. Hold onto `62.5`: the training print has to be read against that constant rule.
    """))
    cells.append(md("""
    `train` in `train_toy_driving.py` builds the 105-parameter tanh MLP and steps for 35 epochs at learning rate 0.08, seed 42. The cell records every epoch so it can plot the loss, then runs `train` itself and checks that the milestone losses agree.

    **Predict:** the loss on epoch 1 is about `0.9980` and the loss on epoch 35 is lower. Accuracy will be printed next to those losses. Compare it with the always-safe number `62.5` from the previous cell.
    """))
    cells.append(code("""
    def recorded_run(epochs=35, learning_rate=0.08, seed=42):
        random.seed(seed)
        dataset = generate_driving_dataset(num_samples=40, seed=seed)
        model = MLP(nin=2, nouts=[8, 8, 1], activations=["tanh", "tanh", "linear"])
        losses = []
        for epoch in range(1, epochs + 1):
            total_loss = Value(0.0)
            correct = 0
            for features, label in dataset:
                pred = model(features)
                margin = Value(1.0) - (Value(label) * pred)
                total_loss = total_loss + margin.relu()
                if (pred.data > 0 and label > 0) or (pred.data < 0 and label < 0):
                    correct += 1
            loss = total_loss / len(dataset)
            acc = (correct / len(dataset)) * 100.0
            model.zero_grad()
            loss.backward()
            grad_norm_sq = 0.0
            for p in model.parameters():
                grad_norm_sq += p.grad ** 2
                p.data -= learning_rate * p.grad
            grad_norm = grad_norm_sq ** 0.5
            losses.append(loss.data)
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"record {epoch:02d}/{epochs:02d} loss {loss.data:.4f} "
                    f"acc {acc:5.1f} grad {grad_norm:.4f}"
                )
        return losses

    losses = recorded_run()
    down = sum(losses[i + 1] < losses[i] for i in range(len(losses) - 1))
    print("epochs:", len(losses))
    print("steps down:", down, "of", len(losses) - 1)
    print("loss start:", f"{losses[0]:.4f}")
    print("loss end:", f"{losses[-1]:.4f}")

    buf = io.StringIO()
    with redirect_stdout(buf):
        train(epochs=35, learning_rate=0.08, seed=42)
    train_log = buf.getvalue()
    print(train_log)
    milestones = ["0.9980", "0.7451", "0.7045", "0.6704", "0.6516", "105", "0.08"]
    print("milestones in train log:", all(piece in train_log for piece in milestones))

    keep_inline()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean hinge loss")
    ax.set_title("Toy driving MLP, seed 42")
    plt.tight_layout()
    plt.show()
    print("plotted points:", len(losses))
    """))
    cells.append(md("""
    The record lines and `train` agree. Epoch 1 is loss `0.9980`, accuracy `60.0`, gradient norm `0.5870`. Epoch 10 is loss `0.7451`, accuracy `62.5`, gradient norm `0.3785`. Epoch 20 is loss `0.7045`, accuracy `62.5`, gradient norm `0.2068`. Epoch 30 is loss `0.6704`, accuracy `62.5`, gradient norm `0.2133`. Epoch 35 is loss `0.6516`, accuracy `62.5`, gradient norm `0.2252`. `train` also prints `Total Parameters: 105` and learning rate `0.08`. `milestones in train log: True`.

    `epochs: 35`. `steps down: 34 of 34`. `loss start: 0.9980`. `loss end: 0.6516`. The plot has `35` points, one per epoch, and it slopes down the whole way.

    Accuracy moves from `60.0` to `62.5` and stays there. `62.5` is the always-safe rate from the previous cell (25 of 40). The hinge got smaller on every step, so the margin improved on average. The sign of the prediction ends at the same rate as guessing the common label. The trainer's closing line says the boundary was learned. The accuracy column is the number to trust when you decide whether the signs are right: it is `62.5`.
    """))

    cells.append(md("## 8. Exercises"))
    cells.append(md("""
    **Exercise — power rule.** `Value.__pow__` in `engine.py` uses the local factor `power * x ** (power - 1)`, then multiplies by the upstream gradient. Leave the `TODO` as it is to use the reference. If `solutions/00_nn_scratch/exercises.py` exists, the cell loads `power_rule_grad` from there instead.

    **Predict:** for `x = 3`, power `2`, upstream `1`, both the reference and a fresh `Value` square print `6.0`, and the check prints ✅.
    """))
    cells.append(code("""
    def power_rule_grad_student(x, power, upstream):
        # TODO: power * x ** (power - 1) * upstream
        raise NotImplementedError

    def reference_power_rule_grad(x, power, upstream):
        return (power * (x ** (power - 1))) * upstream

    def load_exercises():
        path = REPO / "solutions" / "00_nn_scratch" / "exercises.py"
        print("solutions file exists:", path.is_file())
        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location("nn_exercises", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    exercise_mod = load_exercises()

    def get_power_rule():
        try:
            power_rule_grad_student(3.0, 2, 1.0)
        except NotImplementedError:
            if exercise_mod is not None and hasattr(exercise_mod, "power_rule_grad"):
                print("Using solutions/00_nn_scratch/exercises.py power_rule_grad")
                return exercise_mod.power_rule_grad
            print("Using reference power_rule_grad (TODO not implemented)")
            return reference_power_rule_grad
        print("Using your power_rule_grad")
        return power_rule_grad_student

    power_fn = get_power_rule()
    print("x, power, upstream:", 3.0, 2, 1.0)
    got = power_fn(3.0, 2, 1.0)
    check_x = Value(3.0)
    (check_x ** 2).backward()
    print("power_rule_grad:", got)
    print("engine grad:", check_x.grad)
    assert abs(got - check_x.grad) < 1e-9
    print("✅ correct: power_rule_grad =", got)
    """))
    cells.append(md("""
    `solutions file exists: False`. The `TODO` still raises, so the cell prints `Using reference power_rule_grad (TODO not implemented)`. `power_rule_grad: 6.0` and `engine grad: 6.0`. The check prints ✅. `2 * 3 ** 1 * 1` is `6`, the same slope as `x * x` in section 4. Replace the `TODO` and run the cell again if you want the check to call your function.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    return (power * (x ** (power - 1))) * upstream
    ```

    </details>
    """))
    cells.append(md("""
    **Exercise — parameter count.** Add weights and biases for an MLP shaped like the one in section 6. Leave the `TODO` as it is to use the reference. A solutions file, if present, should define `count_parameters`.

    **Predict:** the count for inputs `2` and widths `[8, 8, 1]` is `105`, matching `len(MLP.parameters())`. The check prints ✅.
    """))
    cells.append(code("""
    def count_parameters_student(nin, nouts):
        # TODO: for each layer, nin * nout weights plus nout biases
        raise NotImplementedError

    def reference_count_parameters(nin, nouts):
        total = 0
        prev = nin
        for nout in nouts:
            total += prev * nout + nout
            prev = nout
        return total

    def get_count_parameters():
        try:
            count_parameters_student(2, [8, 8, 1])
        except NotImplementedError:
            if exercise_mod is not None and hasattr(exercise_mod, "count_parameters"):
                print("Using solutions/00_nn_scratch/exercises.py count_parameters")
                return exercise_mod.count_parameters
            print("Using reference count_parameters (TODO not implemented)")
            return reference_count_parameters
        print("Using your count_parameters")
        return count_parameters_student

    count_fn = get_count_parameters()
    counted = count_fn(2, [8, 8, 1])
    random.seed(0)
    counted_net = MLP(nin=2, nouts=[8, 8, 1], activations=["tanh", "tanh", "linear"])
    print("count_parameters:", counted)
    print("MLP.parameters():", len(counted_net.parameters()))
    assert counted == len(counted_net.parameters())
    print("✅ correct: count_parameters =", counted)
    """))
    cells.append(md("""
    The `TODO` still raises and the solutions file is still missing, so the cell prints `Using reference count_parameters (TODO not implemented)`. `count_parameters: 105` and `MLP.parameters(): 105`. The check prints ✅. That is the same total as section 6: `24 + 72 + 9`.
    """))
    cells.append(md("""
    <details><summary>Solution</summary>

    ```python
    total = 0
    prev = nin
    for nout in nouts:
        total += prev * nout + nout
        prev = nout
    return total
    ```

    </details>
    """))

    cells.append(md("## 9. Recap"))
    cells.append(md("""
    - A `Value` stores `data`, a gradient that starts at `0.0`, an op, and parents. `2 * -3` stored op `*` and data `-6.0`.
    - The slope of a square is not enough. For `L = (2 * -3 + 0.5) ** 2`, stopping at `2 * u` gave `-11.0`. The nudge and the chain rule both gave `33.0` for `a`, `-22.0` for `b`, and `-11.0` for `c`. `backward` matched those three numbers.
    - Gradients add. `3 * 3` has one unique parent and gradient `6.0`. A second `backward` without a clear moved it to `12.0`. Clearing it brought back `6.0`.
    - ReLU on `-1.5` is data `0.0` and grad `0.0`. ReLU on `1.5` is data `1.5` and grad `1.0`. The engine has no leaky method (`has leaky: False`). tanh(`2`) printed data `0.9640` and grad `0.0707`.
    - The toy MLP has `105` parameters: `24`, `72`, and `9` per layer.
    - On 40 synthetic rows (25 safe, 15 brake), 35 epochs took the mean hinge from `0.9980` to `0.6516`, down on `34` of `34` steps. Accuracy ended at `62.5`, the always-safe rate.

    ### Go deeper
    - [Karpathy — micrograd](https://github.com/karpathy/micrograd) is the scalar engine this `Value` class follows.
    - [Neural Networks: Zero to Hero, lecture 1](https://www.youtube.com/watch?v=VMj-3S1tku0) builds that engine on video. The series index is [karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html).
    """))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata["colab"] = {"provenance": []}
    return nb


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks" / "00_neural_networks_and_autograd.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, out)
    print(f"Wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
