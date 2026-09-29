import numpy as np
from matplotlib.figure import Figure


def make(out_dir):
    fig = Figure(figsize=(3, 2))
    fig.subplots().plot(np.random.default_rng(0).normal(size=20).cumsum())
    fig.savefig(out_dir / "curve.png", dpi=60)
