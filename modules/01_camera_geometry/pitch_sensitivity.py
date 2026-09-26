"""Pitch miscalibration sensitivity — from-scratch student assignment."""

from __future__ import annotations


def pitch_shift_meters(calib: dict, delta_deg: float, range_m: float) -> float:
    """Estimate forward-range error from pitch bias — from scratch.

    Args:
        calib: Front-camera dict with keys ``fx``, ``fy``, ``cx``, ``cy``,
            ``width``, ``height``, ``cam_position_ego`` (length-3, meters),
            ``pitch_deg``, ``yaw_deg``, ``roll_deg``.
        delta_deg: Extra pitch added to ``pitch_deg`` (same convention as scaffold).
        range_m: Ground-truth forward distance of point (range_m, 0, 0) in ego frame.

    Returns:
        Estimated forward range using biased pitch minus ``range_m``.
        ``delta_deg=0`` should be ~0; absolute error should grow with ``range_m``.
    """
    raise NotImplementedError(
        "Implement pitch_shift_meters from the docstring contract."
    )
