"""Pinhole ground-plane camera and a polygon renderer for the course roads.

Vehicle frame: x forward, y LEFT, origin at the front axle (the camera point).  The camera sits
`height_m` above the ground and is pitched down `pitch_deg`.  Pixel coordinates put pixel centres
on integer values, so the optical centre of a 320x180 image is (159.5, 89.5).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .world import LINE_HALF_WIDTH, ROAD_HALF_WIDTH, WHITE_LINE_LATERAL

SKY, GRASS, ROAD, YELLOW, WHITE = range(5)
PALETTE = np.array(
    [(135, 190, 235), (70, 130, 60), (90, 90, 95), (230, 200, 40), (240, 240, 240)], dtype=np.uint8
)
NEAR_PLANE = 0.3  # metres of forward distance; nearer ground is clipped away
GRID_STEP = 0.5  # metres of arc length per rendered quad
BEHIND, AHEAD = 3.0, 70.0  # arc-length window around the car that is drawn


def _clip_near(F, L):
    """Sutherland-Hodgman clip of one polygon (vehicle-frame arrays) to forward >= NEAR_PLANE."""
    out = []
    for i in range(len(F)):
        j = (i + 1) % len(F)
        a_in, b_in = F[i] >= NEAR_PLANE, F[j] >= NEAR_PLANE
        if a_in:
            out.append((F[i], L[i]))
        if a_in != b_in:
            t = (NEAR_PLANE - F[i]) / (F[j] - F[i])
            out.append((NEAR_PLANE, L[i] + t * (L[j] - L[i])))
    return np.array(out)


def _strip(road, s0, s1, d_lo, d_hi):
    """World quads (K, 4, 2) tiling the band between laterals d_lo and d_hi over arc length [s0, s1]."""
    s = np.linspace(s0, s1, max(1, math.ceil((s1 - s0) / GRID_STEP)) + 1)
    lo, hi = road.offset(s, d_lo), road.offset(s, d_hi)
    return np.stack([lo[:-1], lo[1:], hi[1:], hi[:-1]], axis=1)


class Camera:
    def __init__(self, width=320, height=180, hfov_deg=90.0, height_m=1.4, pitch_deg=5.0):
        self.width, self.height, self.mount_height = width, height, height_m
        self.f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2
        pitch = math.radians(pitch_deg)
        self._sin, self._cos = math.sin(pitch), math.cos(pitch)

    def ground_to_pixel(self, forward_m, left_m):
        """Pixel (u, v) of a ground point in the vehicle frame; NaN for points behind the camera."""
        F, L = np.asarray(forward_m, dtype=float), np.asarray(left_m, dtype=float)
        depth = F * self._cos + self.mount_height * self._sin
        depth = np.where(depth > 0, depth, np.nan)
        u = self.cx - self.f * L / depth
        v = self.cy + self.f * (self.mount_height * self._cos - F * self._sin) / depth
        return u[()], v[()]

    def pixel_to_ground(self, u, v):
        """Ground point (forward_m, left_m) seen at pixel (u, v); NaN at or above the horizon."""
        xc = (np.asarray(u, dtype=float) - self.cx) / self.f
        yc = (np.asarray(v, dtype=float) - self.cy) / self.f
        down = yc * self._cos + self._sin  # how steeply the ray points at the ground; <= 0 never lands
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(down > 1e-6, self.mount_height / down, np.nan)
        return (t * (self._cos - yc * self._sin))[()], (-t * xc)[()]

    def render(self, road, state, noise_sigma=0.0, seed=0, return_labels=False):
        """RGB frame uint8[H, W, 3] (plus label image uint8[H, W] if `return_labels`)."""
        labels = self._labels(road, state)
        frame = PALETTE[labels]
        if noise_sigma > 0:
            noise = np.random.default_rng(seed).normal(0.0, noise_sigma, frame.shape)
            frame = np.clip(np.rint(frame + noise), 0, 255).astype(np.uint8)
        return (frame, labels) if return_labels else frame

    def _labels(self, road, state):
        _, horizon = self.ground_to_pixel(1e9, 0.0)
        rows = np.where(np.arange(self.height) < horizon, SKY, GRASS).astype(np.uint8)
        img = Image.fromarray(np.tile(rows[:, None], (1, self.width)))
        draw = ImageDraw.Draw(img)
        s0, s1 = road.span(road.project(state.x, state.y)[0], BEHIND, AHEAD)
        dashes = [_strip(road, a, b, -LINE_HALF_WIDTH, LINE_HALF_WIDTH) for a, b in road.dashes(s0, s1)]
        layers = [
            (ROAD, _strip(road, s0, s1, -ROAD_HALF_WIDTH, ROAD_HALF_WIDTH)),
            (WHITE, _strip(road, s0, s1, WHITE_LINE_LATERAL - LINE_HALF_WIDTH, WHITE_LINE_LATERAL + LINE_HALF_WIDTH)),
            (YELLOW, np.concatenate(dashes) if dashes else np.empty((0, 4, 2))),
        ]
        for label, quads in layers:
            for poly in self._polygons(quads, state):
                draw.polygon(poly, fill=label)
        return np.asarray(img)

    def _polygons(self, quads, state):
        """Near-plane clip world quads in the vehicle frame and project them to flat pixel lists."""
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        dx, dy = quads[..., 0] - state.x, quads[..., 1] - state.y
        F, L = dx * c + dy * s, -dx * s + dy * c
        inside = F >= NEAR_PLANE
        whole = inside.all(axis=1)
        polys = self._flatten(F[whole], L[whole])
        for k in np.nonzero(inside.any(axis=1) & ~whole)[0]:
            cf, cl = _clip_near(F[k], L[k]).T
            polys += self._flatten(cf[None], cl[None])
        return polys

    def _flatten(self, F, L):
        u, v = self.ground_to_pixel(F, L)
        return np.rint(np.stack([u, v], axis=-1)).astype(np.int64).reshape(len(F), 2 * F.shape[1]).tolist()
