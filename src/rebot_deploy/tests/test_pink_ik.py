"""Characterization of pink_ik.PinkIK on the in-repo reBot B601 URDF (upstream/handumi-sw-rebot-ego), with its own *_tcp links
removed so PinkIK adds its TCP frames exactly as on the deploy URDF. FK, the orientation projections and the one-step
differential IK are checked; `solve` needs a QP solver (quadprog), which the test environment may lack."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rebot_deploy import pink_ik

REPO = Path(__file__).resolve().parents[3]
URDF_SOURCE = REPO / "upstream/handumi-sw-rebot-ego/new_files/assets/rebot_b601/rebot_b601_bimanual.urdf"
ARM_JOINTS = [f"{tSide}_joint{tJ}" for tSide in ("left", "right") for tJ in range(1, 7)]


@pytest.fixture()
def ik(tmp_path, monkeypatch):
    if not URDF_SOURCE.exists():
        pytest.skip(f"{URDF_SOURCE} not present")
    tText = URDF_SOURCE.read_text()
    tText = re.sub(r'\s*<link name="(left|right)_tcp" />', "", tText)
    tText = re.sub(r'\s*<joint name="(left|right)_gripper_to_tcp" type="fixed">.*?</joint>', "", tText, flags=re.S)
    tUrdf = tmp_path / "rebot.urdf"
    tUrdf.write_text(tText)
    monkeypatch.setattr(pink_ik, "URDF", str(tUrdf))
    return pink_ik.PinkIK(ARM_JOINTS)


def _q12(rng):
    tQ = rng.uniform(-0.6, 0.6, 12)
    tQ[[1, 2, 7, 8]] = -np.abs(tQ[[1, 2, 7, 8]])
    return tQ


def _rot(rng, scale):
    return Rotation.from_rotvec(rng.normal(size=3) * scale).as_matrix()


def test_model_layout(ik):
    assert ik.model.nq == 12 and sorted(ik.idx) == list(range(12)) and ik.names == ARM_JOINTS


def test_fk(ik, golden):
    tRng = np.random.default_rng(6)
    golden.check("pink.fk", [ik.fk(_q12(tRng)) for _ in range(3)] + [ik.fk(np.zeros(12))])


def test_orientation_projections(ik, golden):
    tRng = np.random.default_rng(7)
    tOut = []
    for _ in range(4):
        tCur, tTgt = _rot(tRng, 1.0), _rot(tRng, 1.0)
        tOut.append([ik._yaw_only_target(tCur, tTgt), ik._pitch_only_target(tCur, tTgt), pink_ik.PinkIK._yaw_hold_target(tCur, tTgt),
                     pink_ik.PinkIK._elev(tTgt)])
    tVertical = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])     # approach axis straight up: heading from target
    tOut.append(ik._pitch_only_target(tVertical, _rot(tRng, 0.5)))
    tOut.append(ik._pitch_only_target(tVertical, tVertical))                         # nothing to rotate: R_cur back
    tFlip = np.diag([1.0, -1.0, -1.0])                                                # 180 deg about x: no twist -> target kept
    tOut.append(pink_ik.PinkIK._yaw_hold_target(np.eye(3), tFlip))
    golden.check("pink.projections", tOut)


def test_diff_step(ik, golden):
    tRng = np.random.default_rng(8)
    tOut = []
    for tLock in ((), ("joint5",), ("joint1", "joint4")):
        tQ = _q12(tRng)
        tTargets = [tT.copy() for tT in ik.fk(tQ)]
        for tSide in (0, 1):
            tTargets[tSide][:3, 3] += tRng.normal(size=3) * 0.03
            tTargets[tSide][:3, :3] = _rot(tRng, 0.1) @ tTargets[tSide][:3, :3]
        tOut.append(list(ik.diff_step(tQ, tTargets, lock=tLock)))
        tOut.append(list(ik.diff_step(tQ, tTargets, ori_weight=1.0, damping=0.1, lock=tLock)))
    tAtLimit = np.zeros(12)                                       # rest pose, J2/J3 at their 0 limit: limit-aware re-solve path
    tTargets = [tT.copy() for tT in ik.fk(tAtLimit)]
    tTargets[0][:3, 3] += [0.0, 0.0, 0.05]
    tTargets[1][:3, 3] += [0.02, 0.0, -0.05]
    tOut.append(list(ik.diff_step(tAtLimit, tTargets)))
    golden.check("pink.diff_step", tOut)


def test_solve_rejects_unknown_orientation_mode(ik):
    with pytest.raises(AssertionError):
        ik.solve(np.zeros(12), ik.fk(np.zeros(12)), ori="roll")
