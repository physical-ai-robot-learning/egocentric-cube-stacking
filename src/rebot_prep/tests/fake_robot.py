"""A deterministic stand-in for the external `handumi.robots.registry` (reBot B601 embodiment + IK solver), so the
kinematics code of rebot_prep can be characterised without the real package.

16 solver joints: left j1..j6, two left gripper joints, right j1..j6, two right gripper joints (arm joints at 0..5, 8..13
like the real solver). FK is a smooth 6-joint chain per arm; IK is one damped least-squares step on a numeric Jacobian,
clamped to max_joint_delta, and raises for targets more than IK_REACH_M from the arm base (so the IK_FAIL path is reachable)."""
from __future__ import annotations

import dataclasses
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation

ARM_BASE = {"left": np.array([0.0, 0.25, 0.0]), "right": np.array([0.0, -0.25, 0.0])}
UPPER_M, FORE_M, HAND_M = 0.30, 0.25, 0.10
IK_REACH_M = 2.0
JACOBIAN_STEP = 1e-6
DAMPING = 1e-3
LIMIT_RAD = 2.6
TIGHT_UPPER_RAD = {3: 1.2}          # left j4 has a tight upper limit (LIMIT status reachable)
HOME_Q = np.array([0.1, -0.5, 0.6, 0.2, 0.1, -0.1, 0.0, 0.0, -0.1, -0.5, 0.6, -0.2, 0.1, 0.1, 0.0, 0.0])
LINK_NAMES = [f"{s}_{n}" for s in ("left", "right") for n in
              ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_link", "tcp", "gripper_left",
               "gripper_right")] + ["base"]


def armPose(q6: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (TCP position, TCP rotation matrix) of one arm"""
    tRot = (Rotation.from_euler("zyx", [q6[0], q6[1] + q6[2], q6[3]]) * Rotation.from_euler("yz", [q6[4], q6[5]])).as_matrix()
    tElbow = np.array([UPPER_M * np.cos(q6[1]), 0.0, UPPER_M * np.sin(q6[1])])
    tWrist = tElbow + np.array([FORE_M * np.cos(q6[1] + q6[2]), 0.0, FORE_M * np.sin(q6[1] + q6[2])])
    tYaw = Rotation.from_euler("z", q6[0]).as_matrix()
    return ARM_BASE[side] + tYaw @ tWrist + tRot @ np.array([HAND_M, 0.0, 0.0]), tRot


@dataclasses.dataclass
class IkWeights:
    max_joint_delta: float | None = 0.35
    ori_weight: float = 0.6


class FakeSolver:
    joint_names = [f"left_joint{i}" for i in range(1, 7)] + ["left_gripper_left", "left_gripper_right"] + \
                  [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper_left", "right_gripper_right"]
    ARM = {"left": list(range(0, 6)), "right": list(range(8, 14))}

    def __init__(self, config: IkWeights) -> None:
        self.config = config
        tLower = np.full(16, -LIMIT_RAD)
        tUpper = np.full(16, LIMIT_RAD)
        for tJ, tHi in TIGHT_UPPER_RAD.items():
            tUpper[tJ] = tHi
        self.robot = types.SimpleNamespace(joints=types.SimpleNamespace(lower_limits=tLower, upper_limits=tUpper),
                                           links=types.SimpleNamespace(names=LINK_NAMES))

    def fk_pose7(self, q):
        tQ = np.asarray(q, np.float64)
        tOut = []
        for tSide in ("left", "right"):
            tP, tR = armPose(tQ[self.ARM[tSide]], tSide)
            tOut.append(np.r_[tP, Rotation.from_matrix(tR).as_quat()].astype(np.float32))
        return tuple(tOut)

    def _residual(self, q6, side, pos, quat):
        tP, tR = armPose(q6, side)
        tRotErr = Rotation.from_matrix(Rotation.from_quat(quat).as_matrix() @ tR.T).as_rotvec()
        return np.r_[np.asarray(pos, np.float64) - tP, self.config.ori_weight * tRotErr]

    def ik(self, q, left_pose, right_pose):
        tQ = np.asarray(q, np.float64).copy()
        for tSide, (tPos, tQuat) in (("left", left_pose), ("right", right_pose)):
            if np.linalg.norm(np.asarray(tPos, np.float64) - ARM_BASE[tSide]) > IK_REACH_M:
                raise RuntimeError(f"{tSide} target out of reach")
            tIdx = self.ARM[tSide]
            tQ6 = tQ[tIdx]
            tE = self._residual(tQ6, tSide, tPos, tQuat)
            tJ = np.zeros((6, 6))
            for tK in range(6):
                tD = np.zeros(6)
                tD[tK] = JACOBIAN_STEP
                tJ[:, tK] = (tE - self._residual(tQ6 + tD, tSide, tPos, tQuat)) / JACOBIAN_STEP
            tStep = np.linalg.solve(tJ.T @ tJ + DAMPING * np.eye(6), tJ.T @ tE)
            if self.config.max_joint_delta is not None:
                tStep = np.clip(tStep, -self.config.max_joint_delta, self.config.max_joint_delta)
            tQ[tIdx] = tQ6 + tStep
        return tQ.astype(np.float32)

    def link_positions(self, q, indices):
        tQ = np.asarray(q, np.float64)
        tOut = []
        for tI in indices:
            tName = LINK_NAMES[tI]
            if tName == "base":
                tOut.append(np.zeros(3))
                continue
            tSide, tLink = tName.split("_", 1)
            tFrac = (LINK_NAMES.index(tName) % 11) / 10.0
            tP, tR = armPose(tQ[self.ARM[tSide]], tSide)
            tOut.append(ARM_BASE[tSide] + tFrac * (tP - ARM_BASE[tSide]) + 0.01 * tR[:, 1] * (tLink == "gripper_left")
                        - 0.01 * tR[:, 1] * (tLink == "gripper_right"))
        return np.array(tOut, np.float32)


def _embodiment(name: str):
    assert name == "rebot_b601"
    tArms = {s: types.SimpleNamespace(joint_names=FakeSolver.joint_names[tLo:tLo + 8],
                                      gripper_joints=[types.SimpleNamespace(name=n) for n in FakeSolver.joint_names[tLo + 6:tLo + 8]])
             for s, tLo in (("left", 0), ("right", 8))}
    return types.SimpleNamespace(config=types.SimpleNamespace(ik_weights=IkWeights(), arms=tArms), solver_cls=FakeSolver,
                                 home_q=lambda: HOME_Q.copy())


def install(monkeypatch) -> None:
    """Make `from handumi.robots.registry import load_embodiment` resolve to the fake for one test."""
    tRegistry = types.ModuleType("handumi.robots.registry")
    tRegistry.load_embodiment = _embodiment
    for tName, tMod in (("handumi", types.ModuleType("handumi")), ("handumi.robots", types.ModuleType("handumi.robots")),
                        ("handumi.robots.registry", tRegistry)):
        monkeypatch.setitem(sys.modules, tName, tMod)
