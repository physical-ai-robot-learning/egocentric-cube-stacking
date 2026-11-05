"""Characterization of the umi.common.pose_util stand-in (upstream UMI API: pose = xyz + rotvec, rot6d = first two ROWS)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from umi.common import pose_util
from umi.tests.golden import assertGolden

GOLDEN = Path(__file__).with_name("golden")
SEED = 7


def _poses() -> np.ndarray:
    tRng = np.random.default_rng(SEED)
    return np.concatenate([tRng.normal(0, 0.3, (2, 3, 3)), tRng.normal(0, 1.0, (2, 3, 3))], -1)


def test_pose_round_trips_through_matrix_and_pose10d():
    tMat = pose_util.pose_to_mat(_poses())
    tD10 = pose_util.mat_to_pose10d(tMat)
    assert tMat.shape == (2, 3, 4, 4) and tD10.shape == (2, 3, 9)
    assert np.allclose(pose_util.pose10d_to_mat(tD10), tMat, atol=1e-12)
    assertGolden(GOLDEN, "pose_util", dict(mat=tMat.round(12).tolist(), d10=tD10.round(12).tolist()))


def test_rot6d_is_the_first_two_rows_and_gram_schmidt_recovers_r():
    tR = pose_util.pose_to_mat(np.array([0.1, 0.2, 0.3, 0.4, -0.5, 0.6]))[:3, :3]
    tD6 = pose_util.mat_to_rot6d(tR)
    assert np.array_equal(tD6, tR[:2, :].reshape(6))
    tNoisy = np.r_[2.0 * tD6[:3], tD6[3:] + 0.3 * tD6[:3]]          # scaled first row, second row not orthogonal
    tBack = pose_util.rot6d_to_mat(tNoisy)
    assert np.allclose(tBack @ tBack.T, np.eye(3), atol=1e-12)
    assertGolden(GOLDEN, "rot6d", dict(d6=tD6.tolist(), back=tBack.round(12).tolist()))


def test_mat_to_rot6d_keeps_input_unmodified_and_returns_a_copy():
    tM = np.eye(4)
    tOut = pose_util.mat_to_rot6d(tM[:3, :3])
    tOut[0] = 5.0
    assert tM[0, 0] == 1.0
