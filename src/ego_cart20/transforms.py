"""SE(3) helpers: 4x4 homogeneous transforms, batched over leading axes (was geometry/transforms.py).

pose9 = [xyz 3 | rot6d 6] of a (relative) pose, the per-arm pose layout of CART20 and of both state20 variants.

Pipeline stage: geometry helpers of the dataset stage.
"""
from __future__ import annotations

import numpy as np

from .rotation6d import matrixToRotation6d, quaternionToMatrix, rotation6dToMatrix


def makeTransform(rotations, positions) -> np.ndarray:
    """R (..., 3, 3), p (..., 3) -> T (..., 4, 4) (was make_T)"""
    tR = np.asarray(rotations, np.float64)
    tP = np.asarray(positions, np.float64)
    tT = np.zeros(tR.shape[:-2] + (4, 4))
    tT[..., :3, :3] = tR
    tT[..., :3, 3] = tP
    tT[..., 3, 3] = 1.0
    return tT


def poseToTransform(position, quaternionWxyz) -> np.ndarray:
    """position (..., 3), wxyz quaternion (..., 4) -> T (was pose_to_T)"""
    return makeTransform(quaternionToMatrix(quaternionWxyz), position)


def inverse(transforms) -> np.ndarray:
    """closed-form SE(3) inverse"""
    tT = np.asarray(transforms, np.float64)
    tRt = np.swapaxes(tT[..., :3, :3], -1, -2)
    return makeTransform(tRt, -np.einsum("...ij,...j->...i", tRt, tT[..., :3, 3]))


def relative(fromTransforms, toTransforms) -> np.ndarray:
    """inv(T_from) @ T_to"""
    return inverse(fromTransforms) @ np.asarray(toTransforms, np.float64)


def translation(transforms) -> np.ndarray:
    return np.asarray(transforms)[..., :3, 3]


def pose9(transforms) -> np.ndarray:
    """[xyz 3 | rot6d 6] of a relative pose"""
    tT = np.asarray(transforms)
    return np.concatenate([tT[..., :3, 3], matrixToRotation6d(tT[..., :3, :3])], -1)


def pose9ToTransform(pose) -> np.ndarray:
    """inverse of pose9 (rot6d through Gram-Schmidt) (was pose9_to_T)"""
    tV = np.asarray(pose, np.float64)
    return makeTransform(rotation6dToMatrix(tV[..., 3:9]), tV[..., :3])
