"""Quaternion / SO(3) / rot6d utilities -- the ONE implementation used everywhere in ego_cart20 (was geometry/rotation6d.py).

rot6d = the first two ROWS of R, flattened [R00 R01 R02 R10 R11 R12] (umi.common.pose_util.mat_to_rot6d = mat[..., :2, :]),
the convention of the v4 RELCART20 state and of every REL16 action.  Identity -> [1, 0, 0, 0, 1, 0].
Quaternions are scalar-first [qw, qx, qy, qz].  All functions are batched over leading axes.

Pipeline stage: geometry helpers of the dataset stage.
"""
from __future__ import annotations

import numpy as np

SLERP_SMALL_SIN = 1e-8      # below this sin(angle) SLERP falls back to linear weights (q0 ~ q1)


def quaternionToMatrix(q) -> np.ndarray:
    """wxyz quaternions (normalised here) -> rotation matrices (..., 3, 3)"""
    tQ = np.asarray(q, np.float64)
    tQ = tQ / np.linalg.norm(tQ, axis=-1, keepdims=True)
    tW, tX, tY, tZ = np.moveaxis(tQ, -1, 0)
    tR = np.stack([
        1 - 2 * (tY * tY + tZ * tZ), 2 * (tX * tY - tW * tZ), 2 * (tX * tZ + tW * tY),
        2 * (tX * tY + tW * tZ), 1 - 2 * (tX * tX + tZ * tZ), 2 * (tY * tZ - tW * tX),
        2 * (tX * tZ - tW * tY), 2 * (tY * tZ + tW * tX), 1 - 2 * (tX * tX + tY * tY)], -1)
    return tR.reshape(tQ.shape[:-1] + (3, 3))


def matrixToQuaternion(rotations) -> np.ndarray:
    """robust (Shepperd) conversion; returns wxyz with qw >= 0"""
    tR = np.asarray(rotations, np.float64)
    tShape = tR.shape[:-2]
    tFlat = tR.reshape(-1, 3, 3)
    tOut = np.empty((len(tFlat), 4))
    for tI, tM in enumerate(tFlat):
        tTrace = np.trace(tM)
        if tTrace > 0:
            tS = 2 * np.sqrt(tTrace + 1)
            tOut[tI] = [0.25 * tS, (tM[2, 1] - tM[1, 2]) / tS, (tM[0, 2] - tM[2, 0]) / tS, (tM[1, 0] - tM[0, 1]) / tS]
        elif tM[0, 0] > tM[1, 1] and tM[0, 0] > tM[2, 2]:
            tS = 2 * np.sqrt(1 + tM[0, 0] - tM[1, 1] - tM[2, 2])
            tOut[tI] = [(tM[2, 1] - tM[1, 2]) / tS, 0.25 * tS, (tM[0, 1] + tM[1, 0]) / tS, (tM[0, 2] + tM[2, 0]) / tS]
        elif tM[1, 1] > tM[2, 2]:
            tS = 2 * np.sqrt(1 + tM[1, 1] - tM[0, 0] - tM[2, 2])
            tOut[tI] = [(tM[0, 2] - tM[2, 0]) / tS, (tM[0, 1] + tM[1, 0]) / tS, 0.25 * tS, (tM[1, 2] + tM[2, 1]) / tS]
        else:
            tS = 2 * np.sqrt(1 + tM[2, 2] - tM[0, 0] - tM[1, 1])
            tOut[tI] = [(tM[1, 0] - tM[0, 1]) / tS, (tM[0, 2] + tM[2, 0]) / tS, (tM[1, 2] + tM[2, 1]) / tS, 0.25 * tS]
    tOut /= np.linalg.norm(tOut, axis=1, keepdims=True)
    tOut[tOut[:, 0] < 0] *= -1
    return tOut.reshape(tShape + (4,))


def matrixToRotation6d(rotations) -> np.ndarray:
    """(..., 3, 3) -> (..., 6), the first two rows"""
    tR = np.asarray(rotations)
    return tR[..., :2, :].reshape(tR.shape[:-2] + (6,))


def rotation6dToMatrix(d6) -> np.ndarray:
    """Gram-Schmidt on the two stored rows (umi pose_util.rot6d_to_mat): rows b1, b2, b3 = b1 x b2"""
    tD6 = np.asarray(d6, np.float64)
    tA1, tA2 = tD6[..., :3], tD6[..., 3:6]
    tB1 = tA1 / np.linalg.norm(tA1, axis=-1, keepdims=True)
    tB2 = tA2 - np.sum(tB1 * tA2, -1, keepdims=True) * tB1
    tB2 = tB2 / np.linalg.norm(tB2, axis=-1, keepdims=True)
    return np.stack([tB1, tB2, np.cross(tB1, tB2)], -2)


def rotationAngle(rotations) -> np.ndarray:
    """geodesic angle (rad) of rotation matrices"""
    tCos = (np.trace(np.asarray(rotations, np.float64), axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(tCos, -1.0, 1.0))


def slerp(q0, q1, w) -> np.ndarray:
    """batched quaternion SLERP (shortest arc), q0/q1 (..., 4) wxyz, w (...)"""
    tQ0 = np.asarray(q0, np.float64)
    tQ1 = np.asarray(q1, np.float64)
    tW = np.asarray(w, np.float64)[..., None]
    tDot = np.sum(tQ0 * tQ1, -1, keepdims=True)
    tQ1 = np.where(tDot < 0, -tQ1, tQ1)
    tDot = np.abs(tDot)
    tTheta = np.arccos(np.clip(tDot, -1, 1))
    tSin = np.sin(tTheta)
    tSmall = tSin < SLERP_SMALL_SIN
    tA = np.where(tSmall, 1 - tW, np.sin((1 - tW) * tTheta) / np.where(tSmall, 1, tSin))
    tB = np.where(tSmall, tW, np.sin(tW * tTheta) / np.where(tSmall, 1, tSin))
    tQ = tA * tQ0 + tB * tQ1
    return tQ / np.linalg.norm(tQ, axis=-1, keepdims=True)
