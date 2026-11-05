"""Gripper aperture normalization (was preprocessing/gripper.py).  Contract (user D2): g = aperture_mm / w_open_mm, 0 = closed,
1 = open.

Calibration happens BEFORE clipping: raw ticks -> mm through the monotone caliper knots, then mm / w_open, then the
[0, 1] clip.  ``validateGripper`` is a hard check on the stored values.

Pipeline stage: preprocessing of the dataset stage (convert.py checks every raw gripper stream with validateGripper).
"""
from __future__ import annotations

import numpy as np


class GripperRangeError(ValueError):
    """A stored gripper value lies outside [0, 1]."""


def ticksToMm(ticks, sideCal: dict) -> np.ndarray:
    """servo ticks -> aperture mm through the caliper knots of one side (keys direction, touch_tick, knots_delta_ticks,
    knots_mm, w_open_mm)"""
    tDelta = sideCal.get("direction", 1) * (np.asarray(ticks, np.float64) - sideCal["touch_tick"])
    return np.interp(tDelta, sideCal["knots_delta_ticks"], sideCal["knots_mm"], left=0.0, right=sideCal["w_open_mm"])


def mmToG(mm, wOpenMm: float) -> np.ndarray:
    return np.clip(np.asarray(mm, np.float64) / wOpenMm, 0.0, 1.0)


def validateGripper(g, name: str = "gripper") -> dict:
    """-> dict(min, max) of the finite values; GripperRangeError if any is outside [0, 1]"""
    tG = np.asarray(g)
    tFinite = tG[np.isfinite(tG)]
    if tFinite.size and (tFinite.min() < 0.0 or tFinite.max() > 1.0):
        raise GripperRangeError(f"{name} outside [0, 1]: min {tFinite.min()} max {tFinite.max()}")
    return dict(min=float(tFinite.min()) if tFinite.size else None, max=float(tFinite.max()) if tFinite.size else None)
