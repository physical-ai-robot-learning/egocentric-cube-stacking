"""[2026-09-18] FK and continuity IK for the EEF-delta deployment path, on the same reBot URDF/TCP model used to build the
labels (Gen-1 retarget stage; the deployment copy is src/rebot_deploy/eef_kin.py).

Thin wrapper over the external handumi.robots.registry (the rebot_b601 embodiment), so the eval UI needs no other project
on the path:
  fk_pose7(q12)                       -> (L pose7, R pose7) in the robot base frame
  solve_chunk(q12_now, pos, quat)     -> (q12 per step, FK error mm per step/arm, reachable flag), warm-started step by
                                         step so the solution stays on one IK branch (the same continuity rule the offline
                                         retarget uses).
The class and member names (Kin, fk_pose7, solve_chunk, solver, home_q, rt, _names, _arm, _svc, _q) are shared with the deployment
copy and the ~/holobrain-mac-model twin that pseudo_joint_pipeline loads first; they stay as they are.
"""
from __future__ import annotations

import dataclasses

import numpy as np

IK = dict(max_joint_delta=0.35, ori_weight=0.6)      # same clamp/weight as the frozen offline retarget
EMBODIMENT = "rebot_b601"
SVC_LEN = 14                                         # service vector: left j1..j6, left gripper, right j1..j6, right gripper
RIGHT_OFFSET = 7
Q12_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]   # service vector -> the 12 arm joints
FAIL_ERR_MM = 1e6
M_TO_MM = 1e3


class Kin:
    def __init__(self, ik_cfg: dict | None = None):
        from handumi.robots.registry import load_embodiment
        tCfg = dict(IK)
        tCfg.update(ik_cfg or {})
        self.rt = load_embodiment(EMBODIMENT)
        tMaxDelta = None if tCfg["max_joint_delta"] is None else float(tCfg["max_joint_delta"])
        tWeights = dataclasses.replace(self.rt.config.ik_weights, max_joint_delta=tMaxDelta, ori_weight=float(tCfg["ori_weight"]))
        self.solver = self.rt.solver_cls(config=tWeights)
        self.home_q = self.rt.home_q().astype(np.float32)
        self._names = list(self.solver.joint_names)
        self._arm = {}
        for tSide in ("left", "right"):
            tArm = self.rt.config.arms[tSide]
            tGripper = {tJ.name for tJ in tArm.gripper_joints}
            self._arm[tSide] = [self._names.index(tN) for tN in tArm.joint_names if tN not in tGripper]

    def _svc(self, q12):
        tOut = [0.0] * SVC_LEN
        tOut[0:6] = [float(tX) for tX in q12[:6]]
        tOut[RIGHT_OFFSET:RIGHT_OFFSET + 6] = [float(tX) for tX in q12[6:]]
        return tOut

    def _q(self, svc):
        tQ = np.array(self.home_q, np.float64)
        for tK, tJ in enumerate(self._arm["left"]):
            tQ[tJ] = svc[tK]
        for tK, tJ in enumerate(self._arm["right"]):
            tQ[tJ] = svc[RIGHT_OFFSET + tK]
        return tQ

    def fk_pose7(self, q12):
        tL7, tR7 = self.solver.fk_pose7(self._q(self._svc(q12)).astype(np.float32))
        return np.asarray(tL7, np.float64), np.asarray(tR7, np.float64)

    def solve_chunk(self, q12_now, pos, quat):
        """pos (T,2,3), quat (T,2,4 xyzw) desired TCP poses -> q12 (T,12), fkerr mm (T,2), ok (T,2)."""
        tSteps = len(pos)
        tQ12 = np.zeros((tSteps, 12))
        tFkErr = np.zeros((tSteps, 2))
        tOk = np.ones((tSteps, 2), bool)
        tQ = self._q(self._svc(q12_now))
        for tT in range(tSteps):
            tLp = pos[tT, 0].astype(np.float32)
            tLq = quat[tT, 0].astype(np.float32)
            tRp = pos[tT, 1].astype(np.float32)
            tRq = quat[tT, 1].astype(np.float32)
            try:
                tQ = np.asarray(self.solver.ik(tQ.astype(np.float32), left_pose=(tLp, tLq), right_pose=(tRp, tRq)), np.float64)
            except Exception:
                tOk[tT] = False
                tQ12[tT] = tQ12[tT - 1] if tT else np.asarray(self._svc(q12_now))[Q12_IDX]
                tFkErr[tT] = FAIL_ERR_MM
                continue
            tSvc = [0.0] * SVC_LEN
            for tK, tJ in enumerate(self._arm["left"]):
                tSvc[tK] = tQ[tJ]
            for tK, tJ in enumerate(self._arm["right"]):
                tSvc[RIGHT_OFFSET + tK] = tQ[tJ]
            tQ12[tT] = np.array(tSvc)[Q12_IDX]
            tL7, tR7 = self.solver.fk_pose7(tQ.astype(np.float32))
            tFkErr[tT, 0] = np.linalg.norm(np.asarray(tL7)[:3] - pos[tT, 0]) * M_TO_MM
            tFkErr[tT, 1] = np.linalg.norm(np.asarray(tR7)[:3] - pos[tT, 1]) * M_TO_MM
        return tQ12, tFkErr, tOk
