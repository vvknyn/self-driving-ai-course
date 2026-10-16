"""The course's frames-and-transforms box: the provided answers to the module 1.1 labs.

Later labs import these, as "Your first perception box" imports `provided.paint_masks`. Pure numpy; leading
dimensions broadcast, so points have shape (..., 2) or (..., 3). A pose is (x, y, theta): a frame's origin and
x-axis direction in its parent frame. The simulator's own code is the grader of every one of these (the tests
compare them), so a bug here cannot pass both.
"""
from __future__ import annotations

import math

import numpy as np

from .sim import dynamics
from .sim.world import EGO_LANE_CENTER, wrap_angle  # wrap_angle is re-exported: one definition of the wrap

Pose = tuple[float, float, float]  # (x, y, theta)


# --- positions and headings ---------------------------------------------------------------------------------------

def heading_vector(theta) -> np.ndarray:
    """(..., 2) unit vector (cos theta, sin theta) pointing along heading `theta`."""
    theta = np.asarray(theta, dtype=float)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def direction_angle(v) -> np.ndarray | float:
    """atan2(v_y, v_x) in [-pi, pi]; nan for a zero vector, which has no direction (never 0)."""
    v = np.asarray(v, dtype=float)
    angle = np.arctan2(v[..., 1], v[..., 0])
    return np.where((v[..., 0] == 0) & (v[..., 1] == 0), np.nan, angle)[()]


def forward_left(theta, v) -> np.ndarray:
    """(..., 2): the forward and left parts (v . h, h x v) of world vector `v` for a car heading `theta`."""
    h, v = heading_vector(theta), np.asarray(v, dtype=float)
    return np.stack([v[..., 0] * h[..., 0] + v[..., 1] * h[..., 1], h[..., 0] * v[..., 1] - h[..., 1] * v[..., 0]], axis=-1)


# --- the view from the driver's seat ------------------------------------------------------------------------------

def rotation_matrix(theta: float) -> np.ndarray:
    """(2, 2) counter-clockwise rotation by `theta`."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def world_to_ego(pose: Pose, p_world) -> np.ndarray:
    """(..., 2) R(theta)^T (p - t): world points in the car's own frame, written with the camera's arithmetic."""
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    p = np.asarray(p_world, dtype=float)
    dx, dy = p[..., 0] - x, p[..., 1] - y
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=-1)


def ego_to_world(pose: Pose, p_ego) -> np.ndarray:
    """(..., 2) R(theta) p + t: points in the car's own frame back in the world."""
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    p = np.asarray(p_ego, dtype=float)
    return np.stack([c * p[..., 0] - s * p[..., 1] + x, s * p[..., 0] + c * p[..., 1] + y], axis=-1)


# --- the road's own coordinates: `centerline` is an (N, 2) polyline, closed wraps last -> first -------------------

def project_to_segment(a, b, p) -> tuple[float, float]:
    """(t, d) of point `p` against segment a -> b: t = clamp((p - a) . u, 0, |b - a|), d = +-|p - foot|, left positive."""
    a, b, p = (np.asarray(q, dtype=float) for q in (a, b, p))
    vec = b - a
    length = np.hypot(vec[..., 0], vec[..., 1])
    u = vec / length[..., None]
    rel = p - a
    t = np.clip(rel[..., 0] * u[..., 0] + rel[..., 1] * u[..., 1], 0.0, length)
    foot = rel - t[..., None] * u
    cross = u[..., 0] * rel[..., 1] - u[..., 1] * rel[..., 0]
    return t[()], np.copysign(np.hypot(foot[..., 0], foot[..., 1]), cross)[()]


def _nearest(centerline, closed: bool, p) -> tuple[float, float, float]:
    """(s, d, heading) of the centreline segment nearest `p`: arc length there, signed distance, segment heading."""
    pts = np.asarray(centerline, dtype=float)
    a, b = (pts, np.roll(pts, -1, axis=0)) if closed else (pts[:-1], pts[1:])
    vec = b - a
    seg_len = np.hypot(vec[:, 0], vec[:, 1])
    t, d = project_to_segment(a, b, p)
    i = int(np.argmin(np.abs(d)))
    s = float(np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])[i] + t[i])
    return (s % float(seg_len.sum()) if closed else s), float(d[i]), float(np.arctan2(vec[i, 1], vec[i, 0]))


def to_road_frame(centerline, closed: bool, p) -> tuple[float, float]:
    """(s, d): arc length of the nearest centreline point, and the SIDEWAYS signed distance from it (left positive)."""
    s, d, _ = _nearest(centerline, closed, p)
    return s, d


def lane_errors(centerline, closed: bool, pose: Pose) -> tuple[float, float]:
    """(e, psi): sideways distance from the ego-lane centre, e = d - EGO_LANE_CENTER = d + 1.8, and heading error."""
    x, y, theta = pose
    _, d, heading = _nearest(centerline, closed, (x, y))
    return d - EGO_LANE_CENTER, float(wrap_angle(theta - heading))


# --- poses you can chain ------------------------------------------------------------------------------------------

def pose_to_matrix(pose: Pose) -> np.ndarray:
    """(3, 3) T_parent_child of a pose."""
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def matrix_to_pose(T) -> Pose:
    """(x, y, theta) of a (3, 3) transform, theta = atan2(T[1, 0], T[0, 0]) in [-pi, pi]."""
    return float(T[0, 2]), float(T[1, 2]), float(np.arctan2(T[1, 0], T[0, 0]))


def invert_pose(T) -> np.ndarray:
    """Closed-form inverse [[R^T, -R^T t], [0, 1]] of any (k+1, k+1) rigid transform."""
    T = np.asarray(T, dtype=float)
    k = T.shape[0] - 1
    inverse = np.eye(k + 1)
    inverse[:k, :k] = T[:k, :k].T
    inverse[:k, k] = -T[:k, :k].T @ T[:k, k]
    return inverse


def relative_pose(T_world_a, T_world_b) -> np.ndarray:
    """T_a_b: frame b seen from frame a."""
    return invert_pose(T_world_a) @ T_world_b


def dead_reckon(T_world_start, steps) -> np.ndarray:
    """(len(steps) + 1, 3, 3): T_world_start, then each running product T @ S_k."""
    out = [np.asarray(T_world_start, dtype=float)]
    for step in steps:
        out.append(out[-1] @ step)
    return np.stack(out)


T_EGO_REAR = pose_to_matrix((-dynamics.WHEELBASE, 0.0, 0.0))  # the rear axle in the front-axle ego frame; 1.2 and 1.3 build on it


# --- the camera's point of view -----------------------------------------------------------------------------------

def rot_x(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


CAM_FROM_EGO_AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # B: ego (forward, left, up) -> camera (right, down, forward)


def T_ego_cam(height_m: float, pitch_rad: float) -> np.ndarray:
    """(4, 4) camera mount: [[rot_y(pitch) @ B^T, (0, 0, height)], [0, 1]]; positive pitch looks down."""
    T = np.eye(4)
    T[:3, :3] = rot_y(pitch_rad) @ CAM_FROM_EGO_AXES.T
    T[2, 3] = height_m
    return T


def transform_points(T, p) -> np.ndarray:
    """(..., k): `T` applied to points (w = 1, so the translation counts)."""
    T, p = np.asarray(T, dtype=float), np.asarray(p, dtype=float)
    k = T.shape[0] - 1
    return p @ T[:k, :k].T + T[:k, k]


def transform_directions(T, v) -> np.ndarray:
    """(..., k): `T` applied to directions (w = 0, so only the rotation counts)."""
    T, v = np.asarray(T, dtype=float), np.asarray(v, dtype=float)
    k = T.shape[0] - 1
    return v @ T[:k, :k].T


def ray_to_ground(T_ego_cam, ray_cam) -> np.ndarray:
    """(..., 3) ego point where the ray from the lens meets z = 0; nan where it never does (d_z >= 0)."""
    origin = np.asarray(T_ego_cam, dtype=float)[:3, 3]
    d = transform_directions(T_ego_cam, ray_cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d[..., 2] < 0, -origin[2] / d[..., 2], np.nan)
    return origin + t[..., None] * d


def pixel_ray(camera, u, v) -> np.ndarray:
    """(..., 3) ((u - cx) / f, (v - cy) / f, 1) in the camera's frame. Provided, not a lab problem (module 1.4 opens it)."""
    x, y = np.broadcast_arrays((np.asarray(u, dtype=float) - camera.cx) / camera.f, (np.asarray(v, dtype=float) - camera.cy) / camera.f)
    return np.stack([x, y, np.ones_like(x)], axis=-1)


__all__ = [
    "CAM_FROM_EGO_AXES", "EGO_LANE_CENTER", "Pose", "T_EGO_REAR", "T_ego_cam", "dead_reckon", "direction_angle",
    "ego_to_world", "forward_left", "heading_vector", "invert_pose", "lane_errors", "matrix_to_pose", "pixel_ray",
    "pose_to_matrix", "project_to_segment", "ray_to_ground", "relative_pose", "rot_x", "rot_y", "rot_z",
    "rotation_matrix", "to_road_frame", "transform_directions", "transform_points", "world_to_ego", "wrap_angle",
]
