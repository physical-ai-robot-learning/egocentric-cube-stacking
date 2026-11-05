"""[2026-09-23] Track C pipeline core (Gen-1 retarget stage): legacy HandUMI camera pose -> v2 TCP -> robot-start anchored
retarget -> calibrated continuity IK -> per-row status. Contract: TRACK_C_PSEUDO_JOINT_CONTRACT.md.

In:  camera poses (npz cam_pos / cam_quat or a T_hu trajectory), the camera->TCP v2 calibration
     (<EGO_COLLECTOR_HOME>/configs/calibration/camera_tcp/handumi_camera_tcp_v2.yaml), the reBot kinematics (eef_kin.Kin)
Out: per-row targets, q, status, reasons (ContinuityIK.solve), provenance()
Usage: library; `P = pseudo_joint_pipeline; ik = P.ContinuityIK(P.IKCfg()); rt = P.Retarget(ik.kin)`

Frames (C1 gate + OPEN-1, never re-derived here):
    T_hu_k     = T_cam_k @ X                          X = handumi_camera_tcp_v2      (npz tcp_pose is NOT read)
    rel_ds_k   = F^-1 @ inv(T_hu_0) @ T_hu_k @ F      F = Rx(180 deg)  (OPEN-1 evidence)
    T_tgt_k    = T_fk_R0 @ C^-1 @ rel_ds_k @ C        C = Ry(+90 deg)  (C1 T1: R_fk = R_ds @ C)

IK never holds silently: a failed row keeps its target, gets a status and reason, and the last good q is kept
in a separate field. Nothing is interpolated.

The public names of this module (load_hu, segments, Retarget, IKCfg and its fields, ContinuityIK and its kin / cfg / lo / hi /
q_anchor / _q12 / solve, provenance, X, C, F, ANCHORS, SUCCESS, ...) mirror the ~/umi_bridge/track_c twin that the census
and retarget modules load first and use interchangeably; they stay as they are. X and eef_kin are resolved on first use
(module __getattr__), so importing this module reads no file.
"""
from __future__ import annotations

import dataclasses
import functools

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from ego_paths import EGO_COLLECTOR_HOME, HOLOBRAIN_MAC, prepend_sys_path

CAM_TCP_V2 = EGO_COLLECTOR_HOME / "configs/calibration/camera_tcp/handumi_camera_tcp_v2.yaml"
TRAIN_START_DEG14 = [-4.5, -2.0, -8.3, 30.4, -6.9, -5.2, 0.0, 4.2, -14.8, -1.1, 6.0, 3.2, -3.2, 0.0]
GRIPPER_SVC = (6, 13)                                     # gripper slots of the 14-wide service vector
Q_R0 = np.radians([TRAIN_START_DEG14[tI] for tI in range(14) if tI not in GRIPPER_SVC])
ARM_J = list(range(0, 6)) + list(range(8, 14))           # solver full-q indices of the 12 arm joints
# In-distribution anchor: per-arm median of R150 MEASURED joints over moving frames (|dq| > 0.3 deg/frame at
# 30 fps), 2026-09-23. TRAIN_START is the episode-start home pose; joint2/3 sit 2-8 deg from their 0 upper
# limit there, so human segments (which start mid-task) anchored at it get pinned on the limit.
Q_ACTIVE_DEG12 = [-35.5, -82.7, -55.0, 38.8, -6.9, -12.3, 37.1, -86.6, -55.3, 44.1, 1.9, -11.9]
ANCHORS = {"train_start": Q_R0, "r150_active_median": np.radians(Q_ACTIVE_DEG12)}
LIMIT_TOL_DEG = 0.1                                      # a solver sitting ON its bound (float32) is not a violation
LOST_QUAT_NORM = 0.5                                     # lost frames carry a zero quaternion
SVC_LEN = 14
RIGHT_OFFSET = 7
Q12_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ARM_SLICES = ((0, slice(0, 6)), (1, slice(6, 12)))
REASON_MAX_CHARS = 200
M_TO_MM = 1e3


def _h(R=None, t=None):
    """4x4 homogeneous transform from an optional rotation and translation"""
    tT = np.eye(4)
    if R is not None:
        tT[:3, :3] = R
    if t is not None:
        tT[:3, 3] = t
    return tT


C = _h(Rot.from_rotvec([0, np.pi / 2, 0]).as_matrix())
F = _h(Rot.from_rotvec([np.pi, 0, 0]).as_matrix())


@functools.cache
def cameraTcp() -> np.ndarray:
    """X = handumi_camera_tcp_v2 (camera -> TCP), read once"""
    tV2 = yaml.safe_load(CAM_TCP_V2.read_text())
    return _h(np.asarray(tV2["rotation"]["R_camera_tcp"], float), tV2["translation"]["t_camera_tcp_m"])


@functools.cache
def _eefKin():
    """the ~/holobrain-mac-model eef_kin first (as the original working directory), else the repository copy"""
    prepend_sys_path(str(HOLOBRAIN_MAC), str(HOLOBRAIN_MAC / "umi_pkg"))
    try:
        import eef_kin
    except ImportError:  # clean checkout: the repository copy
        from . import eef_kin
    return eef_kin


def __getattr__(name: str):
    """module attributes resolved on first use: X (camera -> TCP) and eef_kin (the kinematics module)"""
    if name == "X":
        return cameraTcp()
    if name == "eef_kin":
        return _eefKin()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def provenance() -> dict:
    tX = cameraTcp()
    return dict(pseudo_q_scope="local_robot_start_anchored", absolute_workspace_pose="synthetic",
                cross_arm_absolute_geometry="unavailable",
                role="observation-side pseudo proprioception, NOT robot action ground truth",
                C="Ry(+90deg), R_fk = R_ds @ C (c1_frame_regression T1)",
                F="Rx(180deg): R150 FK + wrist-video evidence (OPEN1_EVIDENCE.md); physical inspection PENDING; "
                  "do not flip without it",
                anchor_meaning="r150_active_median is a representative reBot manipulation configuration for generating pseudo q, "
                               "NOT a reconstruction of the human workspace",
                pseudo_q_meaning="plausible continuous joint trajectory if this local human TCP motion started from a typical "
                                 "R150 manipulation configuration",
                X_file=str(CAM_TCP_V2), X=tX.tolist(),
                anchors_deg={tK: np.degrees(tV).round(2).tolist() for tK, tV in ANCHORS.items()},
                limit_tol_deg=LIMIT_TOL_DEG, tcp_pose_used=False)


# ------------------------------------------------------------------ C2 legacy camera pose -> v2 TCP
def load_hu(npz_path) -> dict:
    """-> dict(frame_idx, T_hu (n,4,4), seg_id, lost) ; T_hu is v2 TCP in that side's SLAM frame."""
    tZ = np.load(npz_path)
    tValid = np.linalg.norm(tZ["cam_quat"], axis=1) > LOST_QUAT_NORM
    tTc = np.tile(np.eye(4), (len(tZ["cam_pos"]), 1, 1))
    tTc[tValid, :3, :3] = Rot.from_quat(tZ["cam_quat"][tValid]).as_matrix()
    tTc[tValid, :3, 3] = tZ["cam_pos"][tValid]
    tSeg = np.where(tValid, tZ["seg_id"], -1)
    return dict(frame_idx=tZ["frame_idx"], T_hu=tTc @ cameraTcp(), seg_id=tSeg, lost=tZ["lost"], pose_valid=tValid,
                n_seg_but_no_pose=int(((tZ["seg_id"] >= 0) & ~tValid).sum()))


def segments(seg_id, min_len=2) -> list[tuple[int, int]]:
    """contiguous runs of one seg_id >= 0 -> list of (start, end_exclusive)."""
    tOut, tStart = [], None
    for tI in range(len(seg_id) + 1):
        tCur = seg_id[tI] if tI < len(seg_id) else -2
        if tStart is not None and (tCur != seg_id[tStart]):
            if tI - tStart >= min_len:
                tOut.append((tStart, tI))
            tStart = None
        if tStart is None and tCur >= 0:
            tStart = tI
    return tOut


# ------------------------------------------------------------------ C3 robot-start anchored retarget
class Retarget:
    def __init__(self, kin, q_anchor=None):
        tAnchor = ANCHORS["r150_active_median"] if q_anchor is None else q_anchor
        self.q_anchor = np.asarray(tAnchor, float)
        self.T_R0 = [_h(Rot.from_quat(tP[3:]).as_matrix(), tP[:3]) for tP in kin.fk_pose7(self.q_anchor)]

    def targets(self, T_hu_seg, arm):
        tRel = np.linalg.inv(T_hu_seg[0])[None] @ T_hu_seg
        tRelDs = np.linalg.inv(F)[None] @ tRel @ F[None]
        return self.T_R0[arm][None] @ np.linalg.inv(C)[None] @ tRelDs @ C[None]

    def hold(self, n, arm):
        return np.repeat(self.T_R0[arm][None], n, axis=0)


# ------------------------------------------------------------------ C4 calibrated continuity IK (FROZEN 2026-09-23)
# Decisions (user, 2026-09-23): anchor = r150_active_median, ori_weight = 20 (OPEN-4 sweep, reports/
# open4_ori_weight_r150_active_median.json). dq <= 0.6 is NOT a selection rule for observation-side pseudo q:
# 0.6 < |dq| <= 1.0 is DQ_WARN (soft, row still counts as success), > 1.0 is DQ_FAIL (hard).
HARD = ("IK_FAIL", "LIMIT", "DQ_FAIL", "BRANCH_SWITCH", "NOT_CONVERGED")
SOFT = ("DQ_WARN",)
SUCCESS = ("OK",) + SOFT


@dataclasses.dataclass
class IKCfg:
    """Frozen C4 IK configuration (field names are the serialised contract and the twin's; they stay snake_case)."""
    ori_weight: float = 20.0
    anchor: str = "r150_active_median"
    max_joint_delta: float = 0.35      # per solver call (solver's own clamp)
    iters: int = 10
    conv_dq: float = 1e-4              # stop iterating when max|dq| of a call falls below this
    dq_warn: float = 0.6               # rad per stored step (15 Hz): soft
    dq_fail: float = 1.0               # rad per stored step: hard
    branch_dq: float = 0.3             # BRANCH_SWITCH: max|dq| above this while the target barely moved
    branch_dp_mm: float = 5.0
    branch_dth_deg: float = 5.0
    pos_ok_mm: float = 15.0            # NOT_CONVERGED per row (matches the episode p95 gate)
    rot_ok_deg: float = 5.0


def rowStatus(cfg: IKCfg, q12: np.ndarray, good: np.ndarray, lo: np.ndarray, hi: np.ndarray, sl: slice, arm: int,
              posErrMm: float, rotErrDeg: float, dpMm: float, dthDeg: float) -> tuple[str, str | None, float]:
    """C4 status of one arm on one row (first matching rule wins) -> (status, reason or None, |dq|)"""
    tTol = np.radians(LIMIT_TOL_DEG)
    tD = float(np.abs(q12[sl] - good[sl]).max())
    if (q12[sl] < lo[sl] - tTol).any() or (q12[sl] > hi[sl] + tTol).any():
        return "LIMIT", f"arm{arm} joint limit", tD
    if tD > cfg.dq_fail:
        return "DQ_FAIL", f"arm{arm} |dq|={tD:.3f}>{cfg.dq_fail}", tD
    if tD > cfg.branch_dq and dpMm < cfg.branch_dp_mm and dthDeg < cfg.branch_dth_deg:
        return "BRANCH_SWITCH", f"arm{arm} |dq|={tD:.3f} for target dp {dpMm:.1f}mm dth {dthDeg:.1f}deg", tD
    if posErrMm > cfg.pos_ok_mm or rotErrDeg > cfg.rot_ok_deg:
        return "NOT_CONVERGED", f"arm{arm} {posErrMm:.1f}mm/{rotErrDeg:.1f}deg", tD
    if tD > cfg.dq_warn:
        return "DQ_WARN", f"arm{arm} |dq|={tD:.3f} (warn)", tD
    return "OK", None, tD


def targetStep(targets: np.ndarray, t: int) -> tuple[float, float]:
    """target motion from row t-1 to t (row 0: none) -> (mm, deg)"""
    tPrev = targets[t - 1] if t else targets[t]
    tDp = np.linalg.norm(targets[t, :3, 3] - tPrev[:3, 3]) * M_TO_MM
    tDth = np.degrees(np.linalg.norm(Rot.from_matrix(tPrev[:3, :3].T @ targets[t, :3, :3]).as_rotvec()))
    return tDp, tDth


def fkErrors(kin, qFull: np.ndarray, TL: np.ndarray, TR: np.ndarray) -> list[tuple[float, float]]:
    """per arm (position mm, rotation deg) between the FK of qFull and the row's targets"""
    tL7, tR7 = kin.solver.fk_pose7(qFull.astype(np.float32))
    tOut = []
    for tP7, tTarget in ((tL7, TL), (tR7, TR)):
        tP7 = np.asarray(tP7, float)
        tRf = Rot.from_quat(tP7[3:]).as_matrix()
        tOut.append((np.linalg.norm(tP7[:3] - tTarget[:3, 3]) * M_TO_MM,
                     np.degrees(np.linalg.norm(Rot.from_matrix(tRf.T @ tTarget[:3, :3]).as_rotvec()))))
    return tOut


class ContinuityIK:
    def __init__(self, cfg: IKCfg):
        self.cfg = cfg
        self.kin = _eefKin().Kin(dict(ori_weight=cfg.ori_weight, max_joint_delta=cfg.max_joint_delta))
        tLimits = self.kin.solver.robot.joints
        self.lo = np.asarray(tLimits.lower_limits, float)[ARM_J]
        self.hi = np.asarray(tLimits.upper_limits, float)[ARM_J]
        self.q_anchor = ANCHORS[cfg.anchor]

    def _q12(self, qfull):
        tSvc = [0.0] * SVC_LEN
        for tK, tJ in enumerate(self.kin._arm["left"]):
            tSvc[tK] = qfull[tJ]
        for tK, tJ in enumerate(self.kin._arm["right"]):
            tSvc[RIGHT_OFFSET + tK] = qfull[tJ]
        return np.array(tSvc)[Q12_IDX]

    def solve(self, TL, TR, q12_start=None):
        """TL, TR (n,4,4) targets in the FK frame -> per-row arrays. Each arm seeds from ITS last successful q;
        a failed row keeps its target, gets a status + reason, and never becomes a seed."""
        tStart = self.q_anchor if q12_start is None else q12_start
        tN = len(TL)
        tQ = np.zeros((tN, 12))
        tQLastGood = np.zeros((tN, 12))
        tIters = np.zeros(tN, int)
        tPosErr = np.full((tN, 2), np.nan)
        tRotErr = np.full((tN, 2), np.nan)
        tDq = np.full((tN, 2), np.nan)
        tStatus = np.zeros((tN, 2), "<U14")
        tStatus[:] = "OK"
        tReason = [""] * tN
        tGood = np.array(tStart, float)
        tSeed = self.kin._q(self.kin._svc(tGood))
        tTargets = (TL, TR)
        for tT in range(tN):
            tLp, tLq = TL[tT, :3, 3], Rot.from_matrix(TL[tT, :3, :3]).as_quat()
            tRp, tRq = TR[tT, :3, 3], Rot.from_matrix(TR[tT, :3, :3]).as_quat()
            tQf = tSeed.copy()
            tQLastGood[tT] = tGood
            try:                                                # repeated solver calls until max|dq| < conv_dq
                for tIt in range(self.cfg.iters):
                    tQn = np.asarray(self.kin.solver.ik(tQf.astype(np.float32),
                                                        left_pose=(tLp.astype(np.float32), tLq.astype(np.float32)),
                                                        right_pose=(tRp.astype(np.float32), tRq.astype(np.float32))), np.float64)
                    tStep = np.abs(tQn - tQf).max()
                    tQf = tQn
                    tIters[tT] = tIt + 1
                    if tStep < self.cfg.conv_dq:
                        break
            except Exception as tErr:                           # no hold: the row is a failure with a reason
                tStatus[tT] = "IK_FAIL"
                tReason[tT] = f"solver exception: {type(tErr).__name__}: {tErr}"[:REASON_MAX_CHARS]
                tQ[tT] = np.nan
                continue
            tQ12 = self._q12(tQf)
            tQ[tT] = tQ12
            for tArm, (tPos, tRot) in enumerate(fkErrors(self.kin, tQf, TL[tT], TR[tT])):
                tPosErr[tT, tArm], tRotErr[tT, tArm] = tPos, tRot
            tWhy = []
            for tArm, tSl in ARM_SLICES:
                tDp, tDth = targetStep(tTargets[tArm], tT)
                tStatus[tT, tArm], tNote, tDq[tT, tArm] = rowStatus(self.cfg, tQ12, tGood, self.lo, self.hi, tSl, tArm,
                                                                     tPosErr[tT, tArm], tRotErr[tT, tArm], tDp, tDth)
                if tNote:
                    tWhy.append(tNote)
                if tStatus[tT, tArm] in SUCCESS:
                    tGood[tSl] = tQ12[tSl]
            tReason[tT] = "; ".join(tWhy)
            tSeed = self.kin._q(self.kin._svc(tGood))          # next row seeds from each arm's last SUCCESSFUL q
        tTol = np.radians(LIMIT_TOL_DEG)
        return dict(q=tQ, q_last_good=tQLastGood, pos_err=tPosErr, rot_err=tRotErr, dq=tDq, status=tStatus,
                    reason=np.array(tReason, dtype=object), iters=tIters,
                    at_limit=np.stack([((np.abs(tQ[:, tS] - self.lo[tS]) < tTol) | (np.abs(tQ[:, tS] - self.hi[tS]) < tTol))
                                       .any(1) for _, tS in ARM_SLICES], 1))
