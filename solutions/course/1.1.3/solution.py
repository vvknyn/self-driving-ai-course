"""Reference answer for 1.1.3 The road's own coordinates.  Labs never import this; the grader never reads it."""
import numpy as np

from zero2fsd.geom import wrap_angle

__all__ = ["to_road_frame", "lane_errors"]

LANE_OFFSET = 1.8  # m: the lane centre is this far right of the yellow line, so e = d + 1.8


def project_to_segment(a, b, p):
    """(t, d): forward part clamped to the segment, and the distance to the foot with the sign of the left part."""
    a, b, p = (np.asarray(q, dtype=float) for q in (a, b, p))
    vec = b - a
    length = np.hypot(vec[..., 0], vec[..., 1])
    u = vec / length[..., None]
    rel = p - a
    t = np.clip(rel[..., 0] * u[..., 0] + rel[..., 1] * u[..., 1], 0.0, length)
    foot = rel - t[..., None] * u
    left = u[..., 0] * rel[..., 1] - u[..., 1] * rel[..., 0]
    return t, np.copysign(np.hypot(foot[..., 0], foot[..., 1]), left)


def _nearest(centerline, closed, p):
    """(s, d, heading) of the segment whose foot is nearest p: arc length there, signed distance, segment direction."""
    pts = np.asarray(centerline, dtype=float)
    a, b = (pts, np.roll(pts, -1, axis=0)) if closed else (pts[:-1], pts[1:])
    vec = b - a
    length = np.hypot(vec[:, 0], vec[:, 1])
    t, d = project_to_segment(a, b, p)
    i = int(np.argmin(np.abs(d)))
    s = float(np.concatenate([[0.0], np.cumsum(length)])[i] + t[i])
    return (s % float(length.sum()) if closed else s), float(d[i]), float(np.arctan2(vec[i, 1], vec[i, 0]))


def to_road_frame(centerline, closed, p):
    s, d, _ = _nearest(centerline, closed, p)
    return s, d


def lane_errors(centerline, closed, pose):
    _, d, heading = _nearest(centerline, closed, pose[:2])
    return d + LANE_OFFSET, float(wrap_angle(pose[2] - heading))
