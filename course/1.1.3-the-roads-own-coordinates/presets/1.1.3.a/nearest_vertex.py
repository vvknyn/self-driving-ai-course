import numpy as np


def to_road_frame(centerline, closed, p):
    """The teammate's shortcut: the nearest point of the road is the nearest vertex. Fine at 0.1 m spacing, wrong at a corner."""
    pts = np.asarray(centerline, dtype=float)
    a, b = (pts, np.roll(pts, -1, axis=0)) if closed else (pts[:-1], pts[1:])
    seg = np.hypot(*(b - a).T)
    j = int(np.argmin(np.hypot(*(pts - p).T)))  # the nearest VERTEX
    k = min(j, len(a) - 1)  # the segment that starts there (the last vertex of an open road ends one)
    u = (b[k] - a[k]) / seg[k]
    left = u[0] * (p[1] - pts[j][1]) - u[1] * (p[0] - pts[j][0])
    return float(np.concatenate([[0.0], np.cumsum(seg)])[j]), float(np.copysign(np.hypot(*(pts[j] - p)), left))
