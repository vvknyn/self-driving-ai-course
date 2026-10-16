import numpy as np


def direction_angle(v):
    """The lecture's first try: the angle from rise over run. It loses the signs of both."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctan(v[..., 1] / v[..., 0])
