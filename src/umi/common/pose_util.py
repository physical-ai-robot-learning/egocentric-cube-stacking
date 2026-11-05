"""Minimal stand-in for UMI's umi/common/pose_util.py (upstream: real-stanford/universal_manipulation_interface, MIT).

[2026-10-08] The upstream copy in ~/holobrain-mac-model/umi_pkg was removed; this file keeps the same API and conventions:
pose = xyz + rotvec; rot6d = the first two ROWS of R; rot6d -> R by Gram-Schmidt on those rows. The function and parameter
names are the upstream API and stay snake_case. Used by every stage that builds or reads 10-D poses (dataset, training,
deployment). Usage: `from umi.common.pose_util import mat_to_pose10d`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_mat(pose) -> np.ndarray:
    """(..., 6) xyz + rotvec -> (..., 4, 4)"""
    tPose = np.asarray(pose, np.float64)
    tBatch = tPose.shape[:-1]
    tOut = np.zeros(tBatch + (4, 4))
    tOut[..., :3, :3] = Rotation.from_rotvec(tPose[..., 3:].reshape(-1, 3)).as_matrix().reshape(tBatch + (3, 3))
    tOut[..., :3, 3] = tPose[..., :3]
    tOut[..., 3, 3] = 1
    return tOut


def mat_to_rot6d(mat) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two rows of R (a copy)"""
    tMat = np.asarray(mat)
    return tMat[..., :2, :].copy().reshape(tMat.shape[:-2] + (6,))


def rot6d_to_mat(d6) -> np.ndarray:
    """(..., 6) -> (..., 3, 3): Gram-Schmidt on the two rows, third row = their cross product"""
    tD6 = np.asarray(d6, np.float64)
    tA1, tA2 = tD6[..., :3], tD6[..., 3:]
    tB1 = tA1 / np.linalg.norm(tA1, axis=-1, keepdims=True)
    tB2 = tA2 - np.sum(tB1 * tA2, axis=-1, keepdims=True) * tB1
    tB2 = tB2 / np.linalg.norm(tB2, axis=-1, keepdims=True)
    return np.stack((tB1, tB2, np.cross(tB1, tB2)), axis=-2)


def mat_to_pose10d(mat) -> np.ndarray:
    """(..., 4, 4) -> (..., 9): xyz + rot6d"""
    tMat = np.asarray(mat)
    return np.concatenate([tMat[..., :3, 3], mat_to_rot6d(tMat[..., :3, :3])], axis=-1)


def pose10d_to_mat(d10) -> np.ndarray:
    """(..., 9) xyz + rot6d -> (..., 4, 4)"""
    tD10 = np.asarray(d10, np.float64)
    tOut = np.zeros(tD10.shape[:-1] + (4, 4))
    tOut[..., :3, :3] = rot6d_to_mat(tD10[..., 3:9])
    tOut[..., :3, 3] = tD10[..., :3]
    tOut[..., 3, 3] = 1
    return tOut
