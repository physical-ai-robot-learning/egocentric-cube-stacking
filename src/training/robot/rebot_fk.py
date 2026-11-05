#!/usr/bin/env python3
"""reBot B601-DM dual-arm forward kinematics (numpy, URDF-parsed, no deps) (training / evaluation support).

Input convention = LeRobot dataset `observation.state` / follower-convention action:
  14-D [L: pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll, grip | R: same]
  arm joints in RADIANS here (call with deg->rad if needed). Grippers ignored.
Joint i of each arm maps to URDF {side}_joint{i+1} directly (sign/offset assumed identity; validated empirically by the
set_cubes.py consistency check).
Returns EEF = {side}_gripper_link origin (+ optional tip offset) in `base` frame.
The class / method / attribute names (ReBotFK, side_pose, eef, joints, chains) are shared with the node and Mac copies that
rebot_fk_torch.py may import instead of this file, so they stay as they are.
Env: REBOT_URDF (URDF path), REBOT_TIP_X (tip offset along gripper +x, m).
Usage: python -m training.robot.rebot_fk      (prints the zero pose and one recorded frame)
"""
from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.environ.get("REBOT_URDF", os.path.expanduser("~/robot-cockpit/urdf/reBot_B601_DM_dualarm.urdf"))
ARM_IDX = {"left": [0, 1, 2, 3, 4, 5], "right": [7, 8, 9, 10, 11, 12]}
GRIP_IDX = {"left": 6, "right": 13}
# gripper_link origin = palm. Empirically (80 teleop grasps) the gripper +x axis points DOWN at grasp and TIP_X=+0.07 puts the
# implied cube centre at z~+0.04 m (table-top), while -0.07 would float it at +0.17 m -> fingers/tip extend along +x.
# (verified 2026-08-25)
TIP_OFFSET = np.array([float(os.environ.get("REBOT_TIP_X", "0.07")), 0.0, 0.0])
SIDES = ("left", "right")
N_ARM_JOINTS = 6
# self-test pose: episode 028, frame 0 (arm joints, rad)
EP028_LEFT = [-0.398, -0.010, -0.260, 0.583, -0.243, -0.601]
EP028_RIGHT = [-0.265, -0.103, -0.149, 0.534, -0.007, -0.220]
PRINT_DECIMALS = 3


def _rpy(r, p, y):
    tCr, tSr, tCp, tSp, tCy, tSy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[tCy * tCp, tCy * tSp * tSr - tSy * tCr, tCy * tSp * tCr + tSy * tSr],
                     [tSy * tCp, tSy * tSp * tSr + tCy * tCr, tSy * tSp * tCr - tCy * tSr],
                     [-tSp, tCp * tSr, tCp * tCr]])


def _T(xyz, rpy):
    tT = np.eye(4)
    tT[:3, :3] = _rpy(*rpy)
    tT[:3, 3] = xyz
    return tT


def _rot_axis(axis, th):
    tA = np.asarray(axis, float)
    tA /= np.linalg.norm(tA)
    tK = np.array([[0, -tA[2], tA[1]], [tA[2], 0, -tA[0]], [-tA[1], tA[0], 0]])
    tR = np.eye(3) + math.sin(th) * tK + (1 - math.cos(th)) * tK @ tK
    tT = np.eye(4)
    tT[:3, :3] = tR
    return tT


def _floats(text):
    return [float(tV) for tV in text.split()]


class ReBotFK:
    def __init__(self, urdf=URDF):
        tRoot = ET.parse(urdf).getroot()
        self.joints = {}
        for tJoint in tRoot.findall("joint"):
            tOrigin = tJoint.find("origin")
            tAxis = tJoint.find("axis")
            tXyz = _floats(tOrigin.get("xyz", "0 0 0") if tOrigin is not None else "0 0 0")
            tRpy = _floats(tOrigin.get("rpy", "0 0 0") if tOrigin is not None else "0 0 0")
            tAxisXyz = _floats(tAxis.get("xyz")) if tAxis is not None else [0, 0, 1]
            self.joints[tJoint.get("name")] = dict(type=tJoint.get("type"), parent=tJoint.find("parent").get("link"),
                                                   child=tJoint.find("child").get("link"), T=_T(tXyz, tRpy), axis=tAxisXyz)
        self.chains = {}
        for tSide in SIDES:
            self.chains[tSide] = [f"{tSide}_mount"] + [f"{tSide}_joint{tI}" for tI in range(1, N_ARM_JOINTS + 1)] \
                + [f"{tSide}_gripper_joint"]
            for tName in self.chains[tSide]:
                assert tName in self.joints, tName

    def side_pose(self, q6, side):
        """q6: 6 arm joint angles (rad). returns 4x4 pose of {side}_gripper_link in base frame."""
        tT = np.eye(4)
        tK = 0
        for tName in self.chains[side]:
            tJoint = self.joints[tName]
            tT = tT @ tJoint["T"]
            if tJoint["type"] == "revolute":
                tT = tT @ _rot_axis(tJoint["axis"], q6[tK])
                tK += 1
        return tT

    def eef(self, q14_rad, tip=True):
        """q14 in dataset order (arms rad). returns {'left': xyz, 'right': xyz} (m, base frame)."""
        tOut = {}
        for tSide, tIdx in ARM_IDX.items():
            tT = self.side_pose(np.asarray(q14_rad, float)[tIdx], tSide)
            tP = tT @ np.append(TIP_OFFSET if tip else np.zeros(3), 1.0)
            tOut[tSide] = tP[:3]
        return tOut


def main(argv: list[str] | None = None) -> int:
    tFk = ReBotFK()
    tQ = np.zeros(14)
    print("zero pose:", {tK: np.round(tV, PRINT_DECIMALS) for tK, tV in tFk.eef(tQ).items()})
    tQ[ARM_IDX["left"]] = EP028_LEFT
    tQ[ARM_IDX["right"]] = EP028_RIGHT
    print("ep028 frame0:", {tK: np.round(tV, PRINT_DECIMALS) for tK, tV in tFk.eef(tQ).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
