"""Characterization of eef_kin.Kin (FK / continuity IK wrapper) on a fake HandUMI kinematics registry: index mapping between
the 12-joint arm vector and the solver vector, warm start, failure rows and the FK error."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rebot_deploy import eef_kin
from rebot_deploy.tests.fake_kinematics import installFakeRegistry


def test_kin_config_and_index_mapping(monkeypatch):
    tLoaded = installFakeRegistry(monkeypatch)
    tKin = eef_kin.Kin()
    assert tLoaded == ["rebot_b601"]
    assert tKin.mSolver.config.max_joint_delta == 0.35 and tKin.mSolver.config.ori_weight == 0.6
    assert tKin.mSolver.config.pos_weight == 3.0
    assert tKin.mArmJointIdx == {"left": [0, 1, 2, 3, 4, 5], "right": [7, 8, 9, 10, 11, 12]}
    assert tKin.mHomeQ.dtype == np.float32
    tNone = eef_kin.Kin({"max_joint_delta": None})
    assert tNone.mSolver.config.max_joint_delta is None and tNone.mSolver.config.ori_weight == 0.6
    tOther = eef_kin.Kin({"max_joint_delta": 1, "ori_weight": 20})
    assert tOther.mSolver.config.max_joint_delta == 1.0 and tOther.mSolver.config.ori_weight == 20.0


def test_fk_pose7(monkeypatch, golden):
    installFakeRegistry(monkeypatch)
    tKin = eef_kin.Kin()
    tQ = np.linspace(-0.6, 0.5, 12)
    tLeft, tRight = tKin.fkPose7(tQ)
    assert tLeft.dtype == np.float64 and tRight.dtype == np.float64
    golden.check("eef_kin.fk_pose7", [tLeft, tRight])


def test_solve_chunk(monkeypatch, golden):
    installFakeRegistry(monkeypatch)
    tKin = eef_kin.Kin()
    tRng = np.random.default_rng(5)
    tPos = tRng.normal(size=(5, 2, 3)) * 0.1 + np.array([0.3, 0.0, 0.1])
    tPos[2, 0, 0] = 9.0                       # the fake solver raises for this row
    tQuat = np.stack([Rotation.from_rotvec(tRng.normal(size=(5, 3)) * 0.2).as_quat() for _ in range(2)], 1)
    tQ12, tErr, tOk = tKin.solveChunk(np.linspace(-0.3, 0.3, 12), tPos, tQuat)
    golden.check("eef_kin.solve_chunk", [tQ12, tErr, tOk])
    tPos0 = tPos.copy()
    tPos0[0, 0, 0] = 9.0                      # failure on the first row falls back to the start pose
    golden.check("eef_kin.solve_chunk_first_fails", list(tKin.solveChunk(np.linspace(-0.3, 0.3, 12), tPos0[:2], tQuat[:2])))
