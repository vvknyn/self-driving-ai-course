import numpy as np


def to_road_frame(centerline, closed, p):
    """The cleverer guess: d is y minus the y of the nearest centreline point. Wrong wherever the road is not flat."""
    pts = np.asarray(centerline, dtype=float)
    seg = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T) if closed else np.hypot(*np.diff(pts, axis=0).T)
    j = int(np.argmin(np.hypot(*(pts - p).T)))
    return float(np.concatenate([[0.0], np.cumsum(seg)])[j]), float(p[1] - pts[j][1])
