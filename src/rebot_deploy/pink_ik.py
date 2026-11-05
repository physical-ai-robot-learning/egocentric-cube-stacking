"""[2026-09-29] IK_BACKEND=pink: Pinocchio/Pink QP differential IK, iterated to convergence (vs the PyRoki single LM solve)
(deploy stage).

In:  12 arm joints (rad, q12 = [L j1..j6, R j1..j6], eef_kin arm joint order, follower frame) and one 4x4 TCP target per arm
     in the DATASET / robot base frame.
Out: solve -> q12, per-arm position error (mm), iterations; diffStep -> one damped-least-squares step; fk -> TCP mats.

Model: the same URDF as rebot_fk_torch (reBot_B601_DM_dualarm.urdf, env REBOT_URDF), gripper joints locked at 0.
TCP frames `left_tcp` / `right_tcp` = {side}_gripper_link @ T(+0.07 m along x), exactly rebot_fk_torch's TCP (== dataset TCP),
so targets are given in the DATASET frame directly (no R_DS_TO_FK). Checked against rebot_fk_torch by selftest().
Tasks (fixed before any rollout): FrameTask per arm, position_cost 1.0 (per m), orientation_cost 0.5 (per rad), lm_damping 1e-3;
PostureTask toward the SEED (measured q), cost 1e-3 (weak tie-break only); ConfigurationLimit (URDF limits).
Solver: quadprog, dt 0.05, up to 200 iterations, stop when both arms < 0.5 mm and < 0.5 deg.
usage: python -m rebot_deploy.pink_ik      (selftest against rebot_fk_torch; needs the node's c8old checkout)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pink
import pinocchio as pin
from pink.limits import ConfigurationLimit
from pink.tasks import FrameTask, PostureTask

from ego_paths import DATA_ROOT, HOLOBRAIN_MAC, prepend_sys_path  # machine-specific roots (env-overridable)

URDF = os.path.expanduser(os.environ.get("REBOT_URDF", str(HOLOBRAIN_MAC / "reBot_B601_DM_dualarm.urdf")))
TIP_X = 0.07                          # TCP = gripper link + this along its x axis (m)
SIDES = ("left", "right")
N_ARM_JOINTS = 12
POSITION_COST = 1.0                   # per m
ORIENTATION_COST = 0.5                # per rad
LM_DAMPING = 1e-3
POSTURE_COST = 1e-3                   # weak tie-break toward the seed
SOLVE_ITERS = 200
SOLVE_DT_S = 0.05
QP_SOLVER = "quadprog"
CONVERGED_POS_M = 5e-4                # both arms closer than this ...
CONVERGED_ROT_DEG = 0.5               # ... and this: converged
LOCK_HALF_WIDTH_RAD = 1e-6            # a locked joint's limits are narrowed to its hold value +- this
HEADING_MIN_XY = 1e-6                 # approach axis this close to vertical: heading taken from the target
PARALLEL_AXIS_EPS = 1e-9              # current and desired approach axes parallel: nothing to rotate
TWIST_SINGULAR_EPS = 1e-9             # |(w, z)| of the change below this: 180 deg horizontal flip, no defined twist
DIFF_ORI_WEIGHT = 0.5                 # diffStep: orientation rows scaled by this (pink's ori/pos cost ratio)
DIFF_DAMPING = 1e-2                   # diffStep: damped least squares lambda
MM_PER_M = 1000
ORI_MODES = ("full", "yaw", "pitch", "pitch_hold", "yaw_hold")


class PinkIK:
    """Pink QP IK on the reduced (arm-joints-only) reBot model with the two dataset TCP frames added."""

    def __init__(self, arm_joint_names):
        tFull = pin.buildModelFromUrdf(URDF)
        tKeep = set(arm_joint_names)
        tLock = [tFull.getJointId(tName) for tName in tFull.names[1:] if tName not in tKeep]
        self.mModel = pin.buildReducedModel(tFull, tLock, pin.neutral(tFull)) if tLock else tFull
        for tSide in SIDES:
            tFrameId = self.mModel.getFrameId(f"{tSide}_gripper_link")
            tFrame = self.mModel.frames[tFrameId]
            tTcpPlacement = tFrame.placement * pin.SE3(np.eye(3), np.array([TIP_X, 0, 0]))
            self.mModel.addFrame(pin.Frame(f"{tSide}_tcp", tFrame.parentJoint, tFrameId, tTcpPlacement, pin.FrameType.OP_FRAME))
        self.mData = self.mModel.createData()
        # q index of each of our 12 arm joints in the reduced model (all revolute, nq == nv == 12)
        self.mIdx = [self.mModel.joints[self.mModel.getJointId(tName)].idx_q for tName in arm_joint_names]
        self.mNames = list(arm_joint_names)
        assert self.mModel.nq == N_ARM_JOINTS and sorted(self.mIdx) == list(range(N_ARM_JOINTS)), (self.mModel.nq, self.mIdx)
        self.mTasks = {tSide: FrameTask(f"{tSide}_tcp", position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST,
                                        lm_damping=LM_DAMPING) for tSide in SIDES}
        self.mPosture = PostureTask(cost=POSTURE_COST)
        self.mLimits = [ConfigurationLimit(self.mModel)]

    def _toPin(self, q12):
        tQ = np.zeros(N_ARM_JOINTS)
        tQ[self.mIdx] = q12
        return tQ

    def _fromPin(self, q):
        return np.asarray(q)[self.mIdx].copy()

    def fk(self, q12):
        """q12 -> [T_left_tcp, T_right_tcp] (4x4, base frame)"""
        tQ = self._toPin(q12)
        pin.framesForwardKinematics(self.mModel, self.mData, tQ)
        return [self.mData.oMf[self.mModel.getFrameId(f"{tSide}_tcp")].homogeneous.copy() for tSide in SIDES]

    def _yawOnlyTarget(self, rCur, rTgt):
        """[2026-09-30] ori='yaw': keep only the BASE-z part of the orientation error. World-frame error w = log3(R_tgt R_cur^T);
        the task target becomes exp(w_z e_z) R_cur, so rotation about base x/y (roll/pitch) is free (only the posture tie-break
        acts on it). Well defined for a vertical approach axis too (unlike an Euler yaw of the tool heading)."""
        tErr = pin.log3(rTgt @ rCur.T)
        return pin.exp3(np.array([0.0, 0.0, tErr[2]])) @ rCur

    @staticmethod
    def _elev(rot):
        return np.arcsin(np.clip(rot[2, 0], -1.0, 1.0))     # elevation of the tool x (approach) axis above the base xy plane

    def _pitchOnlyTarget(self, rCur, rTgt):
        """[2026-09-30 user: roll + yaw excluded] ori='pitch': match only the approach-axis elevation (tool x vs base xy plane).
        Target = the minimal rotation that tilts the CURRENT approach axis to the target elevation, keeping its heading, applied
        to R_cur -> heading (yaw) and rotation about the approach axis (roll) stay free (posture tie-break only)."""
        tXCur = rCur[:, 0]
        tElevTgt = self._elev(rTgt)
        tHeading = (np.arctan2(tXCur[1], tXCur[0]) if np.hypot(tXCur[0], tXCur[1]) > HEADING_MIN_XY
                    else np.arctan2(rTgt[1, 0], rTgt[0, 0]))
        tXDes = np.array([np.cos(tElevTgt) * np.cos(tHeading), np.cos(tElevTgt) * np.sin(tHeading), np.sin(tElevTgt)])
        tAxis = np.cross(tXCur, tXDes)
        tSin = np.linalg.norm(tAxis)
        tCos = float(np.clip(tXCur @ tXDes, -1.0, 1.0))
        if tSin < PARALLEL_AXIS_EPS:
            return rCur
        return pin.exp3(tAxis / tSin * np.arctan2(tSin, tCos)) @ rCur

    @staticmethod
    def _yawHoldTarget(rCur, rTgt):
        """[2026-09-30 user: yaw hard-hold] roll/pitch from the policy, yaw held at the CURRENT pose. Swing-twist of the
        world-frame CHANGE D = R_tgt R_cur^T about base z: D = S * T_z; drop T_z -> R' = S R_cur. D is a small waypoint delta,
        so it stays far from the swing-twist singularity (twist taken about base z relative to the BASE frame is near-singular
        at the usual TCP orientation: sqrt(w^2+z^2) median 0.21, min 0.0 on 649 logged poses)."""
        tChange = pin.Quaternion(rTgt @ rCur.T)
        tChange.normalize()
        tNorm = np.hypot(tChange.w, tChange.z)
        if tNorm < TWIST_SINGULAR_EPS:                          # 180 deg horizontal flip: no defined twist, keep the target
            return rTgt
        tTwist = pin.Quaternion(tChange.w / tNorm, 0.0, 0.0, tChange.z / tNorm)
        # [2026-09-30 review] swing = q * t^-1 (D = S T). Both orders leave z-twist 0.12 deg p95 on 654 poses, but t^-1 * q kept
        # the policy's tilt worse (swing err p95 2.18 vs 0.70 deg).
        return (tChange * tTwist.inverse()).matrix() @ rCur

    def solve(self, q12Seed, targets, iters=SOLVE_ITERS, dt=SOLVE_DT_S, postureQ12=None, postureCost=POSTURE_COST, ori="full",
              holdQ12=None, lock=()):
        """targets: [T_left, T_right] 4x4 in the dataset/base frame. Returns q12, per-arm position error (mm), iterations.
        [2026-09-29] postureQ12 / postureCost: tie-break toward a given joint pose (default: the seed, cost 1e-3).
        [2026-09-30] ori: 'full' (default, unchanged) | 'yaw' = position + rotation about base z only (roll/pitch excluded)
                     | 'pitch' = position + approach-axis elevation only (roll/yaw excluded)
                     | 'pitch_hold' = [user: "yaw still moves"] roll/yaw HELD at the seed (measured) pose, pitch from the target:
                       the target rotation becomes the seed TCP rotation tilted to the target elevation, then solved as 'full'
                     | 'yaw_hold' = roll/pitch from the target, yaw (base-z twist) held at the current pose, solved as 'full'.
        holdQ12: the pose whose orientation is HELD (default: the seed). pinkdq seeds at q_ref, so it must pass the measured q.
        [2026-09-30 user: "yaw still moves"] lock: joint-name substrings (e.g. ("wrist_yaw",)) HARD-locked at holdQ12 (default the
        seed): their URDF limits are narrowed to that value for this solve, so the other joints do the IK."""
        assert ori in ORI_MODES, ori
        q12Seed = np.array(q12Seed, dtype=float, copy=True)
        tLimits = self.mLimits
        if lock:
            tHoldQ = q12Seed if holdQ12 is None else np.asarray(holdQ12, float)
            tLocked = [tI for tI, tName in enumerate(self.mNames) if any(tKey in tName for tKey in lock)]
            q12Seed[tLocked] = tHoldQ[tLocked]
            tModel = pin.Model(self.mModel)
            for tI in tLocked:
                tModel.lowerPositionLimit[self.mIdx[tI]] = tHoldQ[tI] - LOCK_HALF_WIDTH_RAD
                tModel.upperPositionLimit[self.mIdx[tI]] = tHoldQ[tI] + LOCK_HALF_WIDTH_RAD
            tLimits = [ConfigurationLimit(tModel)]
        tQ = self._toPin(q12Seed)
        if ori in ("pitch_hold", "yaw_hold"):
            tProject = self._pitchOnlyTarget if ori == "pitch_hold" else self._yawHoldTarget
            tHeld = []
            for tR0, tT in zip(self.fk(q12Seed if holdQ12 is None else holdQ12), targets):
                tTh = np.array(tT, dtype=float, copy=True)
                tTh[:3, :3] = tProject(tR0[:3, :3], tT[:3, :3])
                tHeld.append(tTh)
            targets, ori = tHeld, "full"
        tProj = {"yaw": self._yawOnlyTarget, "pitch": self._pitchOnlyTarget}.get(ori)
        tConf = pink.Configuration(self.mModel, self.mData, tQ)
        for tSide, tT in zip(SIDES, targets):
            self.mTasks[tSide].set_target(pin.SE3(tT[:3, :3], tT[:3, 3]))
        self.mPosture.cost = postureCost
        self.mPosture.set_target(tQ.copy() if postureQ12 is None else self._toPin(postureQ12))
        tTasks = [self.mTasks["left"], self.mTasks["right"], self.mPosture]
        for tIt in range(iters):
            if tProj is not None:   # re-project every iteration: the free rotation components drift and change R_cur
                for tSide, tT in zip(SIDES, targets):
                    tRCur = tConf.get_transform_frame_to_world(f"{tSide}_tcp").rotation
                    self.mTasks[tSide].set_target(pin.SE3(tProj(tRCur, tT[:3, :3]), tT[:3, 3]))
            tVel = pink.solve_ik(tConf, tTasks, dt, solver=QP_SOLVER, limits=tLimits, safety_break=False)
            tConf.integrate_inplace(tVel, dt)
            tErr = [tT[:3, 3] - tConf.get_transform_frame_to_world(f"{tSide}_tcp").translation
                    for tSide, tT in zip(SIDES, targets)]
            if ori == "pitch":
                tRotErr = [abs(self._elev(tT[:3, :3]) - self._elev(tConf.get_transform_frame_to_world(f"{tSide}_tcp").rotation))
                           for tSide, tT in zip(SIDES, targets)]
            elif ori == "yaw":
                tRotErr = [abs(pin.log3(tT[:3, :3] @ tConf.get_transform_frame_to_world(f"{tSide}_tcp").rotation.T)[2])
                           for tSide, tT in zip(SIDES, targets)]
            else:
                tRotErr = [np.linalg.norm(pin.log3(tConf.get_transform_frame_to_world(f"{tSide}_tcp").rotation.T @ tT[:3, :3]))
                           for tSide, tT in zip(SIDES, targets)]
            if max(np.linalg.norm(tE) for tE in tErr) < CONVERGED_POS_M and max(tRotErr) < np.radians(CONVERGED_ROT_DEG):
                break
        tQ12 = self._fromPin(tConf.q)
        return tQ12, np.array([np.linalg.norm(tE) * MM_PER_M for tE in tErr]), tIt + 1

    def diffStep(self, q12, targets, oriWeight=DIFF_ORI_WEIGHT, damping=DIFF_DAMPING, lock=()):
        """[2026-10-01 user: IK from the CURRENT pose, not solved to the target] ONE damped-least-squares Jacobian step at the
        measured q: dq = J^T (J J^T + lambda^2 I)^-1 e, e = [p_tgt - p_cur ; oriWeight * log3(R_tgt R_cur^T)] per arm (base frame,
        LOCAL_WORLD_ALIGNED Jacobian of {side}_tcp). No iteration to convergence: the arm moves along the local linearisation of
        the requested displacement. lock: joint-name substrings whose columns are zeroed (held at q). Joint limits clip the
        result.
        Returns q12, per-arm position error of the RESULT vs the target (mm) by FK, and the requested displacement (mm).
        (was diff_step)"""
        tQ = self._toPin(q12)
        tModel, tData = self.mModel, self.mData
        pin.computeJointJacobians(tModel, tData, tQ)
        pin.framesForwardKinematics(tModel, tData, tQ)
        tJacs, tErrs, tReq = [], [], []
        for tSide, tT in zip(SIDES, targets):
            tFrameId = tModel.getFrameId(f"{tSide}_tcp")
            tPose = tData.oMf[tFrameId]
            # (6, nq): [v; w] in base axes
            tJac = pin.getFrameJacobian(tModel, tData, tFrameId, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            tDp = tT[:3, 3] - tPose.translation
            tDw = pin.log3(tT[:3, :3] @ tPose.rotation.T)
            tJacs.append(np.vstack([tJac[:3], oriWeight * tJac[3:]]))
            tErrs.append(np.concatenate([tDp, oriWeight * tDw]))
            tReq.append(np.linalg.norm(tDp) * MM_PER_M)
        tJac = np.vstack(tJacs)
        tErr = np.concatenate(tErrs)
        if lock:
            for tI, tName in enumerate(self.mNames):
                if any(tKey in tName for tKey in lock):
                    tJac[:, self.mIdx[tI]] = 0.0
        # limit-aware: a joint the step would push past its URDF limit is set AT the limit, its column removed and its
        # contribution taken out of e, and the rest re-solved -- all on the SAME linearisation (still one Jacobian, no iteration
        # toward the target). Without this, clipping after the fact broke the solution near J2/J3 = 0 (rest pose).
        tLo, tHi = tModel.lowerPositionLimit, tModel.upperPositionLimit
        tFixed = np.zeros(tModel.nq, bool)
        tDqFixed = np.zeros(tModel.nq)
        for _ in range(tModel.nq):
            tJacFree = tJac.copy()
            tJacFree[:, tFixed] = 0.0
            tGram = tJacFree @ tJacFree.T + damping ** 2 * np.eye(tJac.shape[0])
            tDq = tJacFree.T @ np.linalg.solve(tGram, tErr - tJac @ tDqFixed) + tDqFixed
            tOver = ~tFixed & ((tQ + tDq > tHi) | (tQ + tDq < tLo))
            if not tOver.any():
                break
            tFixed |= tOver
            tDqFixed[tOver] = np.clip(tQ + tDq, tLo, tHi)[tOver] - tQ[tOver]
        tQNew = np.clip(tQ + tDq, tLo, tHi)
        tQ12New = self._fromPin(tQNew)
        tFk = self.fk(tQ12New)
        tPosErr = np.array([np.linalg.norm(tFk[tI][:3, 3] - targets[tI][:3, 3]) * MM_PER_M for tI in range(2)])
        return tQ12New, tPosErr, np.array(tReq)


SELFTEST_POSES = 50
SELFTEST_SEED = 0
SELFTEST_RANGE_RAD = 0.6


def selftest():
    """Pink TCP vs rebot_fk_torch TCP on random joint vectors (needs the node's c8old checkout and eef_kin's HandUMI registry)"""
    import torch
    prepend_sys_path(str(DATA_ROOT / "c8old"))
    import rebot_fk_torch

    from . import eef_kin
    tKin = eef_kin.Kin()
    tNames = ([tKin.mJointNames[tJ] for tJ in tKin.mArmJointIdx["left"]]
              + [tKin.mJointNames[tJ] for tJ in tKin.mArmJointIdx["right"]])
    tIk = PinkIK(tNames)
    tFkTorch = rebot_fk_torch.ReBotFKTorch(dtype=torch.float64)
    tRng = np.random.default_rng(SELFTEST_SEED)
    tPosErr, tRotErr = [], []
    for _ in range(SELFTEST_POSES):
        tQ = tRng.uniform(-SELFTEST_RANGE_RAD, SELFTEST_RANGE_RAD, N_ARM_JOINTS)
        tQ[[1, 2, 7, 8]] = -np.abs(tQ[[1, 2, 7, 8]])
        tPink = tIk.fk(tQ)
        tTorch = tFkTorch.tcp(torch.tensor(tQ)[None]).numpy()[0]
        for tRow in (0, 1):
            tPosErr.append(np.linalg.norm(tPink[tRow][:3, 3] - tTorch[tRow][:3, 3]) * MM_PER_M)
            tRotErr.append(np.degrees(np.linalg.norm(pin.log3(tPink[tRow][:3, :3].T @ tTorch[tRow][:3, :3]))))
    print(f"Pink TCP vs rebot_fk_torch TCP: pos max {max(tPosErr):.4f} mm, rot max {max(tRotErr):.4f} deg")
    return tIk, tNames


def main(argv: list[str] | None = None) -> int:
    selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
