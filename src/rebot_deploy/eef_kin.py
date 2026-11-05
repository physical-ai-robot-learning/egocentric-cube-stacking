"""[2026-09-18] FK and continuity IK for the EEF-delta deployment path, on the same reBot URDF/TCP model used to build the labels
(deploy stage). Thin wrapper over handumi.robots.registry so the eval UI needs no other project on the path.

In:  12 arm joints [L j1..j6, R j1..j6] (rad, follower frame); desired TCP poses per step and arm.
Out: fkPose7(q12)                   -> (L pose7, R pose7) in the robot base frame
     solveChunk(q12Now, pos, quat)  -> (q12 per step, FK error mm per step/arm, reachable flag), warm-started step by step so the
                                       solution stays on one IK branch (the same continuity rule the offline retarget uses).
usage: from rebot_deploy.eef_kin import Kin; tKin = Kin({"max_joint_delta": None})
"""
from __future__ import annotations

import dataclasses

import numpy as np

IK = dict(max_joint_delta=0.35, ori_weight=0.6)      # same clamp/weight as the frozen offline retarget
EMBODIMENT = "rebot_b601"
SERVICE_DIM = 14                                     # robot_service vector: [L j1..j6, L jaw, R j1..j6, R jaw]
RIGHT_OFFSET = 7                                     # first right-arm index in the service vector
SERVICE_ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
FAILED_FK_ERR_MM = 1e6                               # FK error reported for a step the solver could not solve
MM_PER_M = 1e3


class Kin:
    """HandUMI reBot kinematics: FK and warm-started IK of both arms."""

    def __init__(self, ik_cfg: dict | None = None):
        from handumi.robots.registry import load_embodiment
        tCfg = dict(IK)
        tCfg.update(ik_cfg or {})
        self.mRuntime = load_embodiment(EMBODIMENT)
        tMaxJointDelta = None if tCfg["max_joint_delta"] is None else float(tCfg["max_joint_delta"])
        tWeights = dataclasses.replace(self.mRuntime.config.ik_weights, max_joint_delta=tMaxJointDelta,
                                       ori_weight=float(tCfg["ori_weight"]))
        self.mSolver = self.mRuntime.solver_cls(config=tWeights)
        self.mHomeQ = self.mRuntime.home_q().astype(np.float32)
        self.mJointNames = list(self.mSolver.joint_names)
        self.mArmJointIdx = {}                       # side -> solver indices of its 6 arm joints (gripper joints excluded)
        for tSide in ("left", "right"):
            tArm = self.mRuntime.config.arms[tSide]
            tGripper = {tJoint.name for tJoint in tArm.gripper_joints}
            self.mArmJointIdx[tSide] = [self.mJointNames.index(tName) for tName in tArm.joint_names if tName not in tGripper]

    def _service(self, q12):
        """12 arm joints -> the 14-long robot_service vector (jaws 0)"""
        tOut = [0.0] * SERVICE_DIM
        tOut[0:6] = [float(tX) for tX in q12[:6]]
        tOut[7:13] = [float(tX) for tX in q12[6:]]
        return tOut

    def _solverQ(self, service):
        """robot_service vector -> the solver's joint vector (home pose for every joint that is not an arm joint)"""
        tQ = np.array(self.mHomeQ, np.float64)
        for tK, tJ in enumerate(self.mArmJointIdx["left"]):
            tQ[tJ] = service[tK]
        for tK, tJ in enumerate(self.mArmJointIdx["right"]):
            tQ[tJ] = service[RIGHT_OFFSET + tK]
        return tQ

    def fkPose7(self, q12):
        """(was fk_pose7)"""
        tLeft7, tRight7 = self.mSolver.fk_pose7(self._solverQ(self._service(q12)).astype(np.float32))
        return np.asarray(tLeft7, np.float64), np.asarray(tRight7, np.float64)

    def solveChunk(self, q12Now, pos, quat):
        """pos (T,2,3), quat (T,2,4 xyzw) desired TCP poses -> q12 (T,12), fkerr mm (T,2), ok (T,2). (was solve_chunk)"""
        tSteps = len(pos)
        tQ12 = np.zeros((tSteps, 12))
        tFkErr = np.zeros((tSteps, 2))
        tOk = np.ones((tSteps, 2), bool)
        tQ = self._solverQ(self._service(q12Now))
        for tT in range(tSteps):
            tLeftPos = pos[tT, 0].astype(np.float32)
            tLeftQuat = quat[tT, 0].astype(np.float32)
            tRightPos = pos[tT, 1].astype(np.float32)
            tRightQuat = quat[tT, 1].astype(np.float32)
            try:
                tQ = np.asarray(self.mSolver.ik(tQ.astype(np.float32), left_pose=(tLeftPos, tLeftQuat),
                                                right_pose=(tRightPos, tRightQuat)), np.float64)
            except Exception:
                tOk[tT] = False
                tQ12[tT] = tQ12[tT - 1] if tT else np.asarray(self._service(q12Now))[SERVICE_ARM_IDX]
                tFkErr[tT] = FAILED_FK_ERR_MM
                continue
            tService = [0.0] * SERVICE_DIM
            for tK, tJ in enumerate(self.mArmJointIdx["left"]):
                tService[tK] = tQ[tJ]
            for tK, tJ in enumerate(self.mArmJointIdx["right"]):
                tService[RIGHT_OFFSET + tK] = tQ[tJ]
            tQ12[tT] = np.array(tService)[SERVICE_ARM_IDX]
            tLeft7, tRight7 = self.mSolver.fk_pose7(tQ.astype(np.float32))
            tFkErr[tT, 0] = np.linalg.norm(np.asarray(tLeft7)[:3] - pos[tT, 0]) * MM_PER_M
            tFkErr[tT, 1] = np.linalg.norm(np.asarray(tRight7)[:3] - pos[tT, 1]) * MM_PER_M
        return tQ12, tFkErr, tOk
