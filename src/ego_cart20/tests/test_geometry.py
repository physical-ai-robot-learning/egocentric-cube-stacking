"""Characterization of the rotation / SE(3) / canonical-frame helpers on random batched inputs."""
from __future__ import annotations

import numpy as np

from ego_cart20.geometry import canonical_frame as CF
from ego_cart20.geometry import rotation6d as R6
from ego_cart20.geometry import transforms as TF


def _randQuat(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 4))


def test_quaternion_matrix_and_rot6d_round_trip(golden):
    tQ = _randQuat(64)
    tR = R6.quaternion_to_matrix(tQ)
    golden.check("r6.q2m", tR)
    golden.check("r6.q2m_batched", R6.quaternion_to_matrix(tQ.reshape(8, 8, 4)))
    tQBack = R6.matrix_to_quaternion(tR)
    golden.check("r6.m2q", tQBack)
    assert (tQBack[:, 0] >= 0).all()
    golden.check("r6.m2q_batched", R6.matrix_to_quaternion(tR.reshape(4, 16, 3, 3)))
    # every Shepperd branch: trace > 0, and the three diagonal-dominant cases
    tBranches = np.array([np.eye(3), np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])])
    golden.check("r6.m2q_branches", R6.matrix_to_quaternion(tBranches))
    tD6 = R6.matrix_to_rotation_6d(tR)
    golden.check("r6.m2d6", tD6)
    golden.check("r6.d62m_noisy", R6.rotation_6d_to_matrix(tD6 + np.random.default_rng(1).normal(0, 0.05, tD6.shape)))
    golden.check("r6.angle", R6.rotation_angle(tR))


def test_slerp(golden):
    tQ0 = _randQuat(10, 2)
    tQ0 /= np.linalg.norm(tQ0, axis=1, keepdims=True)
    tQ1 = _randQuat(10, 3)
    tQ1 /= np.linalg.norm(tQ1, axis=1, keepdims=True)
    tW = np.linspace(0, 1, 10)
    golden.check("r6.slerp", R6.slerp(tQ0, tQ1, tW))
    golden.check("r6.slerp_identical", R6.slerp(tQ0, tQ0, tW))       # small-angle branch
    golden.check("r6.slerp_scalar", R6.slerp(tQ0[0], tQ1[0], 0.3))


def test_se3_helpers(golden):
    tRng = np.random.default_rng(4)
    tR = R6.quaternion_to_matrix(_randQuat(12, 5))
    tP = tRng.normal(size=(12, 3))
    tT = TF.make_T(tR, tP)
    golden.check("tf.make_T", tT)
    golden.check("tf.pose_to_T", TF.pose_to_T(tP, _randQuat(12, 5)))
    golden.check("tf.inverse", TF.inverse(tT))
    golden.check("tf.relative", TF.relative(tT[:6], tT[6:]))
    golden.check("tf.relative_broadcast", TF.relative(tT[0], tT))
    golden.check("tf.translation", TF.translation(tT))
    tPose9 = TF.pose9(tT)
    golden.check("tf.pose9", tPose9)
    golden.check("tf.pose9_to_T", TF.pose9_to_T(tPose9))


def test_canonical_frame(golden):
    tT = TF.make_T(R6.quaternion_to_matrix(_randQuat(5, 6)), np.random.default_rng(7).normal(size=(5, 3)))
    assert CF.NAME == "per_arm_task_start_v1"
    golden.check("cf.task_start_transform", CF.task_start_transform(tT[0]))
    tCanon = CF.canonicalize(tT, tT[0])
    golden.check("cf.canonicalize", tCanon)
    assert np.allclose(tCanon[0], np.eye(4))
