"""Characterization of robot/: numpy FK (rebot_fk) and batched torch FK (rebot_fk_torch) on a synthetic dual-arm URDF with the
reBot joint names (the real URDF is not in the repository)."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

import numpy as np
import torch

from .conftest import REPO_SRC

Q14 = np.array([-0.398, -0.010, -0.260, 0.583, -0.243, -0.601, 0.4, -0.265, -0.103, -0.149, 0.534, -0.007, -0.220, 0.2])
ARM12 = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def _fresh(name):
    for tName in [tN for tN in sys.modules if tN.startswith("training.robot") or tN in ("rebot_fk", "rebot_fk_torch")]:
        del sys.modules[tName]
    return importlib.import_module(name)


def test_numpy_fk_poses_and_eef(synthUrdf, golden):
    tFk = _fresh("training.robot.rebot_fk")
    tModel = tFk.ReBotFK(str(synthUrdf))
    assert tModel.chains["left"] == ["left_mount"] + [f"left_joint{tI}" for tI in range(1, 7)] + ["left_gripper_joint"]
    golden.check("fk/side_pose_left", tModel.side_pose(Q14[:6], "left"))
    golden.check("fk/side_pose_right", tModel.side_pose(Q14[7:13], "right"))
    golden.check("fk/eef_tip", tModel.eef(Q14))
    golden.check("fk/eef_no_tip", tModel.eef(Q14, tip=False))
    golden.check("fk/rot_axis", tFk._rot_axis([1.0, 2.0, 2.0], 0.7))
    golden.check("fk/T", tFk._T([0.1, 0.2, 0.3], [0.3, -0.2, 0.1]))


def test_torch_fk_matches_numpy_and_is_batched(synthUrdf, golden):
    tFkt = _fresh("training.robot.rebot_fk_torch")
    tModel = tFkt.ReBotFKTorch(dtype=torch.float64)
    tQ = torch.tensor(np.stack([Q14[ARM12], Q14[ARM12] * 0.5, -Q14[ARM12]]), dtype=torch.float64)
    tT = tModel.tcp(tQ.reshape(3, 1, 12))
    assert tuple(tT.shape) == (3, 1, 2, 4, 4)
    golden.check("fkt/tcp", tT)
    tNp = tFkt.rebot_fk.ReBotFK(str(synthUrdf)).eef(Q14, tip=True)
    assert np.allclose(tT[0, 0, 0, :3, 3].numpy(), tNp["left"], atol=1e-12)
    assert np.allclose(tT[0, 0, 1, :3, 3].numpy(), tNp["right"], atol=1e-12)
    tF32 = tModel.to("cpu", torch.float32)
    assert tF32 is not tModel and tF32.dt == torch.float32
    assert tModel.to("cpu") is tModel
    golden.check("fkt/tcp_f32", tF32.tcp(tQ.float()))
    golden.check("fkt/rot", tFkt.ReBotFKTorch._rot(torch.tensor([0.0, 0.6, 0.8]), torch.tensor([0.1, -1.2])))


def _runModule(name, synthUrdf):
    tEnv = dict(os.environ, PYTHONPATH=str(REPO_SRC), REBOT_URDF=str(synthUrdf))
    return subprocess.run([sys.executable, "-m", name], env=tEnv, capture_output=True, text=True, check=True).stdout


def test_fk_self_tests_print_the_same_report(synthUrdf, golden):
    golden.check("fk/selftest_stdout", _runModule("training.robot.rebot_fk", synthUrdf))
    golden.check("fkt/selftest_stdout", _runModule("training.robot.rebot_fk_torch", synthUrdf))
