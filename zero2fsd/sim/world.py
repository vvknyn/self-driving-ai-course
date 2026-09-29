"""Road geometry: a polyline centreline with an arc-length parameterisation.

Sign convention (never flip): lateral distance and heading error are positive to the LEFT
of the direction of travel.  Two lanes of 3.6 m; the ego car drives the right lane, so the
ego-lane centre sits at lateral -1.8 m from the centreline.
"""
from __future__ import annotations

import math

import numpy as np

LANE_WIDTH = 3.6
EGO_LANE_CENTER = -LANE_WIDTH / 2
WHITE_LINE_LATERAL = -LANE_WIDTH  # solid white edge line, right side only
LINE_HALF_WIDTH = 0.1
ROAD_HALF_WIDTH = 3.8  # tarmac extends a little past the lines
DASH_PERIOD = 5.0  # the yellow centreline: 3.5 m of paint, 1.5 m of gap
DASH_FRACTION = 0.7


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Road:
    """A drivable road: `centerline` is an (N, 2) polyline, closed roads wrap last -> first."""

    def __init__(self, centerline, closed: bool):
        pts = np.asarray(centerline, dtype=float)
        self.closed = bool(closed)
        self._a = pts if closed else pts[:-1]
        self._b = np.roll(pts, -1, axis=0) if closed else pts[1:]
        vec = self._b - self._a
        self._seg_len = np.hypot(vec[:, 0], vec[:, 1])
        self._u = vec / self._seg_len[:, None]
        self._heading = np.arctan2(vec[:, 1], vec[:, 0])
        self._s0 = np.concatenate([[0.0], np.cumsum(self._seg_len)[:-1]])
        self.length = float(self._seg_len.sum())
        # A closed road holds a whole number of dashes so the pattern is seamless at the start line.
        self.dash_period = self.length / max(1, round(self.length / DASH_PERIOD)) if closed else DASH_PERIOD

    def _segment(self, s):
        """(segment index, distance along that segment) for arc length(s) `s`."""
        s = np.asarray(s, dtype=float)
        s = s % self.length if self.closed else np.clip(s, 0.0, self.length)
        i = np.clip(np.searchsorted(self._s0, s, side="right") - 1, 0, len(self._s0) - 1)
        return i, s - self._s0[i]

    def pose_at(self, s):
        """(x, y, heading) of the centreline at arc length `s`; vectorised, wraps if closed."""
        i, t = self._segment(s)
        pos = self._a[i] + t[..., None] * self._u[i]
        return pos[..., 0][()], pos[..., 1][()], self._heading[i][()]

    def offset(self, s, d):
        """World point(s) at arc length `s` and lateral `d` (left positive); shape (..., 2)."""
        x, y, h = self.pose_at(s)
        return np.stack([x - np.sin(h) * d, y + np.cos(h) * d], axis=-1)

    def _locate(self, x, y):
        """(s, lateral, heading of the nearest segment) for a world point."""
        rel = np.array([x, y]) - self._a
        t = np.clip(np.einsum("ij,ij->i", rel, self._u), 0.0, self._seg_len)
        foot = rel - t[:, None] * self._u
        i = int(np.argmin(np.einsum("ij,ij->i", foot, foot)))
        cross = self._u[i, 0] * rel[i, 1] - self._u[i, 1] * rel[i, 0]
        lateral = math.copysign(float(np.hypot(*foot[i])), cross)
        return float((self._s0[i] + t[i]) % self.length if self.closed else self._s0[i] + t[i]), lateral, float(self._heading[i])

    def project(self, x, y):
        """(s, lateral): arc length of the nearest centreline point and signed distance from it."""
        s, lateral, _ = self._locate(x, y)
        return s, lateral

    def ego_pose(self, s, offset=0.0, yaw_err=0.0):
        """Pose (x, y, yaw) of a car at arc length `s`, `offset` left of ego-lane centre, yawed `yaw_err` left."""
        x, y = self.offset(s, EGO_LANE_CENTER + offset)
        return float(x), float(y), float(self.pose_at(s)[2] + yaw_err)

    def ego_truth(self, x, y, yaw):
        """(s, offset from ego-lane centre, heading error) of a car pose; the inverse of `ego_pose`."""
        s, lateral, heading = self._locate(x, y)
        return s, lateral - EGO_LANE_CENTER, float(wrap_angle(yaw - heading))

    def dashes(self, s0, s1):
        """Yellow-dash spans (a, b) overlapping [s0, s1], in unwrapped arc length."""
        p = self.dash_period
        k = np.arange(math.floor(s0 / p), math.floor(s1 / p) + 1)
        a, b = np.maximum(k * p, s0), np.minimum(k * p + p * DASH_FRACTION, s1)
        return [(float(lo), float(hi)) for lo, hi in zip(a, b) if hi > lo]

    def span(self, s, behind, ahead):
        """Arc-length window around `s`; open roads clamp to their ends, closed roads run past `length`."""
        lo, hi = s - behind, s + ahead
        return (lo, hi) if self.closed else (max(lo, 0.0), min(hi, self.length))
