"""Stand-ins for the HandUMI kinematics registry (`handumi.robots.registry`, not installed in the test environment) and a fake
`Kin` for the inference core: deterministic, smooth functions of the joint vector, so the wiring around them is checkable."""
from __future__ import annotations

import dataclasses
import sys
import types
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

# solver joint order of the fake runtime: a gripper joint between the arms and one at the end, so index mapping matters
SOLVER_JOINTS = (["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6", "left_gripper"]
                 + ["right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6", "right_gripper"])


def pose7Of(q6, side):
    """fake FK of one arm: position and xyzw quaternion, smooth in q6"""
    tQ = np.asarray(q6, np.float64)
    tPos = np.array([0.3 + 0.1 * np.sin(tQ[0]), (0.2 if side == "left" else -0.2) + 0.05 * tQ[1], 0.1 + 0.07 * tQ[2] * tQ[3]])
    tQuat = Rotation.from_rotvec(0.5 * tQ[3:6] + 0.1 * tQ[0:3]).as_quat()
    return np.concatenate([tPos, tQuat])


@dataclasses.dataclass(frozen=True)
class FakeIkWeights:
    max_joint_delta: float | None = 0.5
    ori_weight: float = 1.0
    pos_weight: float = 3.0


class FakeSolver:
    joint_names = list(SOLVER_JOINTS)

    def __init__(self, config):
        self.config = config
        self.mCalls = 0

    def fk_pose7(self, q):
        q = np.asarray(q)
        return pose7Of(q[0:6], "left").astype(np.float32), pose7Of(q[7:13], "right").astype(np.float32)

    def ik(self, q, left_pose, right_pose):
        self.mCalls += 1
        if float(left_pose[0][0]) > 5.0:
            raise RuntimeError("unreachable")
        tOut = np.asarray(q, np.float64).copy()
        tStep = self.config.max_joint_delta or 1.0
        tOut[0:6] += np.clip(0.3 * (left_pose[0][0] - 0.3) + 0.05 * np.arange(6), -tStep, tStep) * self.config.ori_weight
        tOut[7:13] += np.clip(0.2 * (right_pose[0][1] + 0.2) - 0.03 * np.arange(6), -tStep, tStep)
        return tOut.astype(np.float32)


def installFakeRegistry(monkeypatch):
    """put a fake `handumi.robots.registry` into sys.modules for one test"""
    tArms = {tSide: SimpleNamespace(joint_names=[tN for tN in SOLVER_JOINTS if tN.startswith(tSide)],
                                    gripper_joints=[SimpleNamespace(name=f"{tSide}_gripper")]) for tSide in ("left", "right")}
    tRuntime = SimpleNamespace(config=SimpleNamespace(ik_weights=FakeIkWeights(), arms=tArms), solver_cls=FakeSolver,
                               home_q=lambda: np.linspace(-0.2, 0.2, len(SOLVER_JOINTS)))
    tLoaded = []

    def load_embodiment(name):
        tLoaded.append(name)
        return tRuntime

    tRegistry = types.ModuleType("handumi.robots.registry")
    tRegistry.load_embodiment = load_embodiment
    tRobots = types.ModuleType("handumi.robots")
    tRobots.registry = tRegistry
    tRoot = types.ModuleType("handumi")
    tRoot.robots = tRobots
    for tName, tMod in (("handumi", tRoot), ("handumi.robots", tRobots), ("handumi.robots.registry", tRegistry)):
        monkeypatch.setitem(sys.modules, tName, tMod)
    return tLoaded


class FakeKin:
    """the subset of eef_kin.Kin the inference core uses"""

    def __init__(self):
        self.solver = SimpleNamespace(config=SimpleNamespace(max_joint_delta=0.35))
        self._names = list(SOLVER_JOINTS)
        self._arm = {"left": [0, 1, 2, 3, 4, 5], "right": [7, 8, 9, 10, 11, 12]}

    def fk_pose7(self, q12):
        q12 = np.asarray(q12, np.float64)
        return pose7Of(q12[:6], "left"), pose7Of(q12[6:], "right")

    def solve_chunk(self, q12Now, pos, quat):
        tSteps = len(pos)
        tQ = np.zeros((tSteps, 12))
        tErr = np.zeros((tSteps, 2))
        tOk = np.ones((tSteps, 2), bool)
        for tK in range(tSteps):
            tQ[tK] = np.asarray(q12Now, np.float64) + 0.01 * (tK + 1) * np.concatenate([pos[tK, 0] * 3, pos[tK, 1] * 2]).repeat(2)[:12]
            tErr[tK] = [1.0 + tK, 2.0 + 0.5 * tK + float(quat[tK, 0, 0])]
        if tSteps > 3:
            tErr[3, 1] = 100.0
        return tQ, tErr, tOk
