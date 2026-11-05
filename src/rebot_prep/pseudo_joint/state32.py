"""[2026-09-23] Track C C6: the ONE function that lays out observation.state s32 (Gen-1 retarget stage).

Used by both the C0 compatibility test (fed R150's own poses/joints, must reproduce r150_umi_v4_3cam_s32 exactly) and the
HandUMI exporter; the Phase-3 census calls it per segment.

Per arm, left (robot0) then right:  [prev_rel pos3 + rot6d (first two ROWS of R), width_m, q6 rad]  -> 16, x2 = 32
    prev_rel[t] = inv(T_t) @ T_{t-1}, T in the DATASET TCP frame; t = 0 uses itself (identity), prev_valid = 0
Rows are float32. The caller decides which rows are stored (Track B drops the last frame, which has no t+1).

The public names (build_state32, prev_rel, FIELDS, PER_ARM) mirror the ~/umi_bridge/track_c/state32.py twin that the census
loads first, so they stay as they are.
"""
from __future__ import annotations

import functools

import numpy as np

from ego_paths import HOLOBRAIN_MAC, prepend_sys_path

PER_ARM = 16
FIELDS = ([f"L_prev_rel_pos{tI}" for tI in range(3)] + [f"L_prev_rel_rot6d{tI}" for tI in range(6)] + ["L_width_m"]
          + [f"L_q{tI}_rad" for tI in range(6)])
FIELDS = FIELDS + [tF.replace("L_", "R_", 1) for tF in FIELDS]
POSE10D = slice(0, 9)
WIDTH = 9
Q6 = slice(10, 16)


@functools.cache
def _matToPose10d():
    """umi.common.pose_util.mat_to_pose10d, the same function Track B's converter uses (the upstream umi_pkg first)"""
    prepend_sys_path(str(HOLOBRAIN_MAC / "umi_pkg"))
    from umi.common.pose_util import mat_to_pose10d
    return mat_to_pose10d


def prev_rel(T) -> np.ndarray:
    """inv(T_t) @ T_{t-1} per row, the first row against itself"""
    tPrev = np.concatenate([T[:1], T[:-1]], axis=0)
    return np.linalg.inv(T) @ tPrev


def build_state32(T_ds, width_m, q_rad) -> tuple[np.ndarray, np.ndarray]:
    """T_ds: [T_left (n,4,4), T_right (n,4,4)] dataset-frame TCP; width_m: [(n,), (n,)]; q_rad: [(n,6), (n,6)]
    -> (state32 (n, 32) float32, prev_valid (n, 1))."""
    tN = len(T_ds[0])
    tOut = np.zeros((tN, 2 * PER_ARM), np.float32)
    for tArm in range(2):
        tO = tArm * PER_ARM
        tOut[:, tO + POSE10D.start:tO + POSE10D.stop] = _matToPose10d()(prev_rel(T_ds[tArm]))
        tOut[:, tO + WIDTH] = width_m[tArm]
        tOut[:, tO + Q6.start:tO + Q6.stop] = q_rad[tArm]
    tPrevValid = np.ones((tN, 1), np.float32)
    tPrevValid[0] = 0.0
    return tOut, tPrevValid
