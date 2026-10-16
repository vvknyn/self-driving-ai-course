import numpy as np


def lane_errors(centerline, closed, pose):
    """The teammate's version: psi = theta - theta_road with no wrap, so a whole turn of yaw shows up as 2 pi of error."""
    pts = np.asarray(centerline, dtype=float)
    a, b = (pts, np.roll(pts, -1, axis=0)) if closed else (pts[:-1], pts[1:])
    t, d = project_to_segment(a, b, pose[:2])
    i = int(np.argmin(np.abs(d)))
    return d[i] + 1.8, pose[2] - np.arctan2(b[i, 1] - a[i, 1], b[i, 0] - a[i, 0])
