"""[2026-09-28] C-old v1 vs v2-K retarget on the SAME human segments (only the retargeting differs) (Gen-1 retarget stage; was
v2k_retarget.py).

In:  --episodes-json (C8 ids), --raw-root (Hpilot raw episodes), --runs (MASt3R runs), --export (exported sides); the frozen
     contracts/ (seedbank_q12.npy, seedbank_R30_q12.npy + seedbank_R30.json, r30_episodes.json, v2kr_frozen.json);
     ~/umi_bridge/track_c/data/r150_q_tcp.npz; ~/ego_collector/scripts/robotlike_offline_check.py (RLC: its Phase-3 module P3,
     segment builder and Clearance are used as they are)
Out: --out <dir>/<episode>.json (per segment v1 / v2 [/ v2r] records) + <episode>.npz (<start>_v1, <start>_v2_<j>,
     <start>_v2r_<j>)
Usage: python -m rebot_prep.retarget.multi_seed_retarget --episodes-json <json "episodes": [C8 ids]> --raw-root <dataset dir with
       <prefix>_<sess>/episode_N> --runs <dir> --export <dir> --out <dir> [--prefix Hpilot] [--part k/n] [--mode v2k|v2kr|r30|tr]

Segments = exactly those of ~/ego_collector/scripts/robotlike_offline_check.py (== frozen c8_phase3 chain: imu_vi scale +
guard, camera_tcp_v2, |dt| < 20 ms, 65-frame hard-break segments). For each segment and each arm the human motion is
reduced to its RELATIVE trajectory  rel(t) = C^-1 F^-1 inv(T_hu(0)) T_hu(t) F C  (Retarget.targets), then attached to a
robot anchor: T_tgt(t) = FK(q0) rel(t)  -- "robot posture + human relative task motion".

  v1   q0 = r150_active_median (the single anchor C-old v1 used)
  v2-K q0 in the frozen kinematic seed bank (contracts/seedbank_q12.npy, sha in seedbank.json):
       1. prescreen every seed: both arms' targets >= 95 % inside the R150 workspace (c8_phase3.ws_in definition)
       2. rank the survivors by mean workspace NN position distance (margin), roll out continuity IK from the top K
       3. hard gates on each rollout: every row IK status in SUCCESS (residual 15 mm / 5 deg, limits, dq), clearance
          d_min >= 0.075 m on every 5th row
       4. score = mean pos_err/15 + mean rot_err/5 + mean|dq|/0.6 + 0.5 mean|ddq|/0.6 (lower is better); best = top-1,
          variants = further survivors >= VARIANT_RAD (mean 12-D joint trajectory distance) from every kept one, max 3
       One anchor per segment; the seed never changes inside a segment.
  v2-Kr (--mode v2kr, v2kr_frozen.json): score + lambda * d_R150 (lambda 0.5), + K_MANI manifold-nearest seeds
  r30   (--mode r30): v2-K-R30 and v2-Kr-R30 on the R30 bank / workspace / reference (2026-09-28 R30 contract)
  tr    (--mode tr, v2tr_frozen.json): v2-TR-R30, anchors = 65-frame R30 reference windows with the nearest relative motion
Nothing here was tuned on v2 results (weights, K, radii fixed before the first run).

The Phase-3 module (P3), the pseudo_joint_pipeline (P) and their objects come from RLC, i.e. the ~/umi_bridge / ~/c8 copies;
their attribute names (P3.TREE / WR / ik / RUNS / export_dir / init / frame_meta / nearest / DT_MS / SEG, ik.kin / cfg / lo / hi /
_q12 / solve, P.C / F / _h / ANCHORS / SUCCESS / LIMIT_TOL_DEG) are that interface.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from ego_paths import EGO_COLLECTOR_HOME, UMI_BRIDGE

CONTRACTS = pathlib.Path(__file__).resolve().parent / "contracts"
RLC_PATH = EGO_COLLECTOR_HOME / "scripts/robotlike_offline_check.py"
R150_NPZ = UMI_BRIDGE / "track_c/data/r150_q_tcp.npz"
K_ROLL, VARIANT_RAD, MAX_VARIANTS, WS_MIN, D_HARD = 8, 0.5, 3, 0.95, 0.075
K_MANI = 8                       # --mode v2kr / r30: + 8 manifold-nearest seeds
MODES = ["v2k", "v2kr", "r30", "tr"]
WS_POS_M, WS_ORI_DEG, WS_K = 0.05, 20, 50
CLEARANCE_STRIDE = 5             # clearance checked on every 5th row
TREE_STRIDE = 3
POS_NORM_MM, ROT_NORM_DEG, DQ_NORM_RAD, DDQ_WEIGHT = 15, 5, 0.6, 0.5   # rollout score normalisers
SEG = 65
TR_T = np.arange(8, 65, 8)       # feature sample times inside the 65-frame window
TR_ROT_M = 0.1                   # m per rad
TR_STRIDE = 5                    # reference windows every 5 frames
TR_NEIGHBOURS = 64
REF_SAMPLES, REF_SEED = 6000, 0  # workspace reference, exactly like c8_phase3.init()


class ContractError(RuntimeError):
    """A frozen contract file (seed bank hash, reference) does not match what the run would use."""


@dataclass(frozen=True)
class Windows:
    """TR reference windows: q_ref per window, relative-motion features and their tree."""
    mQ: np.ndarray | None = None
    mFeat: np.ndarray | None = None
    mTree: cKDTree | None = None


@dataclass(frozen=True)
class RetargetContext:
    """Everything a segment retarget needs: the external Phase-3 module and pipeline, the mode, the seed bank and the
    reference trees (was the module globals P, MODE, LAMBDA, Q_TREE, BANK, BANK_D, BANK_T0, WIN)."""
    mP3: object
    mP: object
    mMode: str = "v2k"
    mLambda: float = 0.0
    mQTree: cKDTree | None = None
    mBank: np.ndarray | None = None
    mBankD: np.ndarray | None = None
    mBankT0: np.ndarray | None = None
    mWindows: Windows = Windows()
    mKRoll: int = K_ROLL
    mKMani: int = K_MANI


def wsBatch(ctx: RetargetContext, T: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray]:
    """T (..., 4, 4) -> inside mask (...,) and NN position distance (...,), == c8_phase3.ws_in, vectorised (was ws_batch)."""
    tShape = T.shape[:-2]
    tTf = T.reshape(-1, 4, 4)
    tDp, tIi = ctx.mP3.TREE[a].query(tTf[:, :3, 3], k=WS_K)
    tTr = np.einsum("nkij,nij->nk", ctx.mP3.WR[a][tIi], tTf[:, :3, :3])
    tAng = np.degrees(np.arccos(np.clip((tTr - 1) / 2, -1, 1))).min(1)
    return ((tDp[:, 0] < WS_POS_M) & (tAng < WS_ORI_DEG)).reshape(tShape), tDp[:, 0].reshape(tShape)


def relTraj(ctx: RetargetContext, T_hu_seg: np.ndarray) -> np.ndarray:
    """C^-1 F^-1 inv(T_hu(0)) T_hu(t) F C (was rel_traj)"""
    P = ctx.mP
    tRel = np.linalg.inv(T_hu_seg[0])[None] @ T_hu_seg
    return np.linalg.inv(P.C)[None] @ (np.linalg.inv(P.F)[None] @ tRel @ P.F[None]) @ P.C[None]


def fkT(ctx: RetargetContext, q12) -> list[np.ndarray]:
    """[T_left, T_right] FK TCP of q12 (was fk_T)"""
    return [ctx.mP._h(Rot.from_quat(tP[3:]).as_matrix(), tP[:3]) for tP in ctx.mP3.ik.kin.fk_pose7(q12)]


def _score(posErr, rotErr, q: np.ndarray) -> float:
    tDq = np.abs(np.diff(q, axis=0))
    tDdq = np.abs(np.diff(q, 2, axis=0))
    return float(np.nanmean(posErr) / POS_NORM_MM + np.nanmean(rotErr) / ROT_NORM_DEG + tDq.mean() / DQ_NORM_RAD
                 + DDQ_WEIGHT * tDdq.mean() / DQ_NORM_RAD)


def _rolloutRecord(ctx: RetargetContext, res: dict, q: np.ndarray, score: float, dRef, tTgL, tTgR, clr, extra: dict) -> dict:
    tIkOk = bool(np.isin(res["status"], ctx.mP.SUCCESS).all())
    tDmin = float(min(clr.dmin(q[tI])[0] for tI in range(0, len(q), CLEARANCE_STRIDE)))
    tWsL, _ = wsBatch(ctx, tTgL, 0)
    tWsR, _ = wsBatch(ctx, tTgR, 1)
    tPe, tRe = res["pos_err"], res["rot_err"]
    return dict(q=q.astype(np.float32), ik_ok=tIkOk, dmin=tDmin, collision_ok=tDmin >= D_HARD, score=score, d_r150=dRef, **extra,
                ws=[float(tWsL.mean()), float(tWsR.mean())], ws_ok=bool(tWsL.mean() >= WS_MIN and tWsR.mean() >= WS_MIN),
                pos_err_p50=float(np.nanpercentile(tPe, 50)), pos_err_p90=float(np.nanpercentile(tPe, 90)),
                rot_err_p50=float(np.nanpercentile(tRe, 50)), rot_err_p90=float(np.nanpercentile(tRe, 90)),
                status={tK: int(tV) for tK, tV in zip(*np.unique(res["status"], return_counts=True))})


def rollout(ctx: RetargetContext, q0, relL, relR, clr) -> dict:
    """continuity IK from anchor q0 with the frozen C4 solver, gates and score"""
    tT0 = fkT(ctx, q0)
    tTgL, tTgR = tT0[0][None] @ relL, tT0[1][None] @ relR
    tRes = ctx.mP3.ik.solve(tTgL, tTgR, q12_start=q0)
    tQ = np.where(np.isfinite(tRes["q"]), tRes["q"], tRes["q_last_good"])
    tScore = _score(tRes["pos_err"], tRes["rot_err"], tQ)
    tDR150 = float(np.median(ctx.mQTree.query(tQ[::TREE_STRIDE])[0])) if ctx.mQTree is not None else None
    if ctx.mMode == "v2kr":
        tScore += ctx.mLambda * tDR150                                  # (r30 mode keeps the raw score; see doSegment)
    return _rolloutRecord(ctx, tRes, tQ, tScore, tDR150, tTgL, tTgR, clr, {})


# ---------------------------------------------------------------- v2-TR-R30 (v2tr_frozen.json)
def trFeature(relL: np.ndarray, relR: np.ndarray) -> np.ndarray:
    """relative TCP motion of both arms at TR_T: positions (m) and rotation vectors * TR_ROT_M (was _feat)"""
    tF = []
    for tRel in (relL, relR):
        tF.append(tRel[TR_T, :3, 3].ravel())
        tF.append(Rot.from_matrix(tRel[TR_T, :3, :3]).as_rotvec().ravel() * TR_ROT_M)
    return np.concatenate(tF)


def buildWindows(ctx: RetargetContext, d150, m) -> Windows:
    """every 65-frame window (stride 5) inside an R30 episode: q_ref and the FK-frame relative TCP motion feature (was
    build_windows)"""
    tQs, tF = [], []
    tEp = d150["ep"][m]
    tJ = np.radians(np.c_[d150["J"][m, :6], d150["J"][m, 7:13]])
    for tE in np.unique(tEp):
        tJe = tJ[tEp == tE]
        tT = np.array([np.stack(fkT(ctx, tQ)) for tQ in tJe])                                  # (n, 2, 4, 4)
        for tS0 in range(0, len(tJe) - SEG + 1, TR_STRIDE):
            tW = tT[tS0:tS0 + SEG]
            tRels = [np.linalg.inv(tW[0, tA])[None] @ tW[:, tA] for tA in (0, 1)]
            tQs.append(tJe[tS0:tS0 + SEG])
            tF.append(trFeature(*tRels))
    print(f"TR window library: {len(tQs)} R30 windows", flush=True)
    return Windows(np.array(tQs, np.float64), np.array(tF), cKDTree(np.array(tF)))


def solveRef(ctx: RetargetContext, TL, TR, q_ref) -> dict:
    """ContinuityIK.solve with ONE change: the next-row seed = last good q + reference joint step q_ref(t+1) - q_ref(t)
    (was solve_ref)."""
    tIk = ctx.mP3.ik
    P = ctx.mP
    tC = tIk.cfg
    tN = len(TL)
    tQ = np.zeros((tN, 12))
    tQLastGood = np.zeros((tN, 12))
    tPosErr = np.full((tN, 2), np.nan)
    tRotErr = np.full((tN, 2), np.nan)
    tStatus = np.zeros((tN, 2), "<U14")
    tStatus[:] = "OK"
    tGood = np.array(q_ref[0], float)
    tSeed = tIk.kin._q(tIk.kin._svc(tGood))
    tTargets = (TL, TR)
    for tT in range(tN):
        tLp, tLq = TL[tT, :3, 3], Rot.from_matrix(TL[tT, :3, :3]).as_quat()
        tRp, tRq = TR[tT, :3, 3], Rot.from_matrix(TR[tT, :3, :3]).as_quat()
        tQf = tSeed.copy()
        tQLastGood[tT] = tGood
        try:
            for _ in range(tC.iters):
                tQn = np.asarray(tIk.kin.solver.ik(tQf.astype(np.float32),
                                                   left_pose=(tLp.astype(np.float32), tLq.astype(np.float32)),
                                                   right_pose=(tRp.astype(np.float32), tRq.astype(np.float32))), np.float64)
                tStep = np.abs(tQn - tQf).max()
                tQf = tQn
                if tStep < tC.conv_dq:
                    break
        except Exception:
            tStatus[tT] = "IK_FAIL"
            tQ[tT] = np.nan
            continue
        tQ12 = tIk._q12(tQf)
        tQ[tT] = tQ12
        tL7, tR7 = tIk.kin.solver.fk_pose7(tQf.astype(np.float32))
        for tArm, (tP7, tTarget) in enumerate(((tL7, TL[tT]), (tR7, TR[tT]))):
            tP7 = np.asarray(tP7, float)
            tPosErr[tT, tArm] = np.linalg.norm(tP7[:3] - tTarget[:3, 3]) * 1e3
            tRf = Rot.from_quat(tP7[3:]).as_matrix()
            tRotErr[tT, tArm] = np.degrees(np.linalg.norm(Rot.from_matrix(tRf.T @ tTarget[:3, :3]).as_rotvec()))
        tTol = np.radians(P.LIMIT_TOL_DEG)
        for tArm, tSl in ((0, slice(0, 6)), (1, slice(6, 12))):
            tD = float(np.abs(tQ12[tSl] - tGood[tSl]).max())                                  # frozen C4 rule, unchanged
            tPrev = tTargets[tArm][tT - 1] if tT else tTargets[tArm][tT]
            tDp = np.linalg.norm(tTargets[tArm][tT, :3, 3] - tPrev[:3, 3]) * 1e3
            tDth = np.degrees(np.linalg.norm(Rot.from_matrix(tPrev[:3, :3].T @ tTargets[tArm][tT, :3, :3]).as_rotvec()))
            if (tQ12[tSl] < tIk.lo[tSl] - tTol).any() or (tQ12[tSl] > tIk.hi[tSl] + tTol).any():
                tStatus[tT, tArm] = "LIMIT"
            elif tD > tC.dq_fail:
                tStatus[tT, tArm] = "DQ_FAIL"
            elif tD > tC.branch_dq and tDp < tC.branch_dp_mm and tDth < tC.branch_dth_deg:
                tStatus[tT, tArm] = "BRANCH_SWITCH"
            elif tPosErr[tT, tArm] > tC.pos_ok_mm or tRotErr[tT, tArm] > tC.rot_ok_deg:
                tStatus[tT, tArm] = "NOT_CONVERGED"
            elif tD > tC.dq_warn:
                tStatus[tT, tArm] = "DQ_WARN"
            if tStatus[tT, tArm] in P.SUCCESS:
                tGood[tSl] = tQ12[tSl]
        tNext = tGood + (q_ref[min(tT + 1, tN - 1)] - q_ref[tT])
        tSeed = tIk.kin._q(tIk.kin._svc(tNext))
    return dict(q=tQ, q_last_good=tQLastGood, pos_err=tPosErr, rot_err=tRotErr, status=tStatus)


def rolloutTr(ctx: RetargetContext, q_ref, relL, relR, clr) -> dict:
    """rollout from a reference window, seeded along its joint steps (was rollout_tr)"""
    tT0 = fkT(ctx, q_ref[0])
    tTgL, tTgR = tT0[0][None] @ relL, tT0[1][None] @ relR
    tRes = solveRef(ctx, tTgL, tTgR, q_ref)
    tQ = np.where(np.isfinite(tRes["q"]), tRes["q"], tRes["q_last_good"])
    tDRef = float(np.median(ctx.mQTree.query(tQ[::TREE_STRIDE])[0]))
    tScore = _score(tRes["pos_err"], tRes["rot_err"], tQ) + ctx.mLambda * tDRef
    tAdherence = float(np.linalg.norm(tQ - q_ref, axis=1).mean())
    return _rolloutRecord(ctx, tRes, tQ, tScore, tDRef, tTgL, tTgR, clr, dict(adherence_rad=tAdherence))


def _keepVariants(survivors: list) -> list:
    """greedy: up to MAX_VARIANTS survivors (best first), each >= VARIANT_RAD from every kept one"""
    tKept = []
    for tX in survivors:
        if len(tKept) >= MAX_VARIANTS:
            break
        if not tKept or all(np.linalg.norm(tX[2]["q"] - tY[2]["q"], axis=1).mean() >= VARIANT_RAD for tY in tKept):
            tKept.append(tX)
    return tKept


def _ok(r: dict) -> bool:
    return r["ik_ok"] and r["collision_ok"] and r["ws_ok"]


def doSegmentTr(ctx: RetargetContext, TL, TR, clr) -> tuple[dict, dict]:
    """v1 and v2-TR of one segment (was do_segment_tr)"""
    tRelL, tRelR = relTraj(ctx, TL), relTraj(ctx, TR)
    tV1 = rollout(ctx, ctx.mP.ANCHORS["r150_active_median"], tRelL, tRelR, clr)
    _, tNn = ctx.mWindows.mTree.query(trFeature(tRelL, tRelR), k=TR_NEIGHBOURS)
    tCand = []
    for tW in np.atleast_1d(tNn):
        tT0 = fkT(ctx, ctx.mWindows.mQ[tW][0])
        tInL, _ = wsBatch(ctx, tT0[0][None] @ tRelL, 0)
        tInR, _ = wsBatch(ctx, tT0[1][None] @ tRelR, 1)
        if tInL.mean() >= WS_MIN and tInR.mean() >= WS_MIN:
            tCand.append(int(tW))
        if len(tCand) >= ctx.mKRoll:
            break
    tRolls = [(tW, rolloutTr(ctx, ctx.mWindows.mQ[tW], tRelL, tRelR, clr)) for tW in tCand]
    tSurv = sorted([(tR["score"], tW, tR) for tW, tR in tRolls if _ok(tR)], key=lambda x: x[0])
    return tV1, dict(n_ws_candidates=len(tCand), n_rolled=len(tRolls), n_feasible=len(tSurv), kept=_keepVariants(tSurv))


def segments(ctx: RetargetContext, rlc, ep: pathlib.Path):
    """(TL_hu, TR_hu) per segment, exactly as robotlike_offline_check.check_episode builds them"""
    P3 = ctx.mP3
    tTags = {tS: rlc.tag_of(ep, tS) for tS in ("left", "right")}
    if any(not (P3.RUNS / tT / "ss1.csv").is_file() for tT in tTags.values()):
        return None, "mast3r_missing"
    tSides = {tS: rlc.side_nogrip(tTags[tS], ep, tS) for tS in ("left", "right")}
    tL, tR = tSides["left"], tSides["right"]
    tFm = P3.frame_meta(ep)
    tCapL = np.array([tC for _, tC in tFm["left_wrist"][tL["off"]:tL["off"] + tL["n"]]], float)
    tCapR = np.array([tC for _, tC in tFm["right_wrist"][tR["off"]:tR["off"] + tR["n"]]], float)
    tCapH = np.array([tC for _, tC in tFm["head"]], float)
    tJR, tJH = P3.nearest(tCapR, tCapL), P3.nearest(tCapH, tCapL)
    tRowOk = (tL["ok"] & tR["ok"][tJR] & (np.abs(tCapR[tJR] - tCapL) / 1e6 < P3.DT_MS)
              & (np.abs(tCapH[tJH] - tCapL) / 1e6 < P3.DT_MS))
    tIdx = np.flatnonzero(tRowOk)
    tRuns = np.split(tIdx, np.flatnonzero(np.diff(tIdx) != 1) + 1) if len(tIdx) else []
    tSegs = [tR[tK:tK + P3.SEG] for tR in tRuns for tK in range(0, len(tR) - P3.SEG + 1, P3.SEG)]
    if not (tL["scale_valid"] and tR["scale_valid"]):
        return [(int(tS[0]), None, None) for tS in tSegs], "B_metric"
    return [(int(tS[0]), tL["T_tcp"][tS], tR["T_tcp"][tJR[tS]]) for tS in tSegs], None


def doSegment(ctx: RetargetContext, TL, TR, clr):
    """v1 and v2-K / v2-Kr (and, in r30 mode, both K-R30 and Kr-R30) of one segment (was do_segment)"""
    tRelL, tRelR = relTraj(ctx, TL), relTraj(ctx, TR)
    tV1 = rollout(ctx, ctx.mP.ANCHORS["r150_active_median"], tRelL, tRelR, clr)
    # v2-K prescreen: all seeds at once
    tT0 = ctx.mBankT0                                                               # (N, 2, 4, 4), computed once
    tL = tT0[:, 0][:, None] @ tRelL[None]
    tR = tT0[:, 1][:, None] @ tRelR[None]                                          # (N, 65, 4, 4)
    tInL, tDL = wsBatch(ctx, tL, 0)
    tInR, tDR = wsBatch(ctx, tR, 1)
    tCand = np.flatnonzero((tInL.mean(1) >= WS_MIN) & (tInR.mean(1) >= WS_MIN))
    tMarginOrder = tCand[np.argsort((tDL[tCand].mean(1) + tDR[tCand].mean(1)))][:ctx.mKRoll]
    tOrder = tMarginOrder
    if ctx.mMode in ("v2kr", "r30"):                      # + seeds whose own posture is nearest the reference
        tExtra = tCand[np.argsort(ctx.mBankD[tCand])][:ctx.mKMani]
        tOrder = np.array(list(dict.fromkeys(tOrder.tolist() + tExtra.tolist())))
    tRolls = [(int(tI), rollout(ctx, ctx.mBank[tI], tRelL, tRelR, clr)) for tI in tOrder]
    if ctx.mMode == "r30":                                # v2-K-R30 (margin top-8, raw S), v2-Kr-R30 (union, S + lambda d)
        tKSet = set(tMarginOrder.tolist())                                          # exactly v2-K's rollout set
        tSK = sorted([(tR["score"], tI, tR) for tI, tR in tRolls if _ok(tR) and tI in tKSet], key=lambda x: x[0])
        tSR = sorted([(tR["score"] + ctx.mLambda * tR["d_r150"], tI, tR) for tI, tR in tRolls if _ok(tR)], key=lambda x: x[0])
        tBase = dict(n_ws_candidates=int(len(tCand)), n_rolled=len(tRolls))
        return (tV1, dict(tBase, n_feasible=len(tSK), kept=_keepVariants(tSK)),
                dict(tBase, n_feasible=len(tSR), kept=_keepVariants(tSR)))
    tSurv = sorted([(tR["score"], tI, tR) for tI, tR in tRolls if _ok(tR)], key=lambda x: x[0])
    return tV1, dict(n_ws_candidates=int(len(tCand)), n_rolled=len(tRolls), n_feasible=len(tSurv), kept=_keepVariants(tSurv),
                     best_any=min(tRolls, key=lambda x: x[1]["score"])[1] if tRolls else None)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def buildContext(P3, mode: str, initS: float, initEnd: float) -> RetargetContext:
    """seed bank, reference trees and (r30 / tr) the R30 workspace gate on the already initialised P3 (whose TREE / WR are
    replaced in r30 / tr mode, as the original did); initS / initEnd = P3.init() duration and end time, for the log line"""
    P = P3.P
    tBank = np.load(CONTRACTS / "seedbank_q12.npy")
    tBankT0 = np.array([np.stack(fkT(RetargetContext(P3, P), tQ)) for tQ in tBank])
    tCtx = RetargetContext(P3, P, mode, 0.0, None, tBank, None, tBankT0)
    print(f"init {initS:.1f}s, bank FK {time.time() - initEnd:.1f}s", flush=True)
    tD150 = np.load(R150_NPZ)
    tQTree = cKDTree(np.radians(np.c_[tD150["J"][:, :6], tD150["J"][:, 7:13]])[::TREE_STRIDE])
    tCtx = dataclasses.replace(tCtx, mQTree=tQTree, mBankD=tQTree.query(tBank)[0])
    if mode in ("r30", "tr"):
        # 2026-09-28 R30 contract: every real-derived quantity from the frozen R30 calibration episodes ONLY
        tR30 = json.load(open(CONTRACTS / "r30_episodes.json"))["episodes"]
        tMask = np.isin(tD150["ep"], tR30)
        tJ30 = np.radians(np.c_[tD150["J"][tMask, :6], tD150["J"][tMask, 7:13]])
        tBank = np.load(CONTRACTS / "seedbank_R30_q12.npy")
        tSb = json.load(open(CONTRACTS / "seedbank_R30.json"))
        if not (tSb["sha256_q12"] == _sha(tBank) and tSb["reference"] == "r30"):
            raise ContractError("seedbank_R30.json does not match seedbank_R30_q12.npy / reference r30")
        # workspace gate rebuilt exactly like c8_phase3.init() but on R30 frames (6000 random, rng 0)
        tIdx = np.random.default_rng(REF_SEED).choice(len(tJ30), REF_SAMPLES, replace=False)
        tRef = {0: [], 1: []}
        for tI in tIdx:
            for tArm, tP7 in enumerate(P3.ik.kin.fk_pose7(tJ30[tI])):
                tRef[tArm].append(tP7)
        tRef = {tA: np.array(tV) for tA, tV in tRef.items()}
        P3.TREE = {tA: cKDTree(tRef[tA][:, :3]) for tA in (0, 1)}
        P3.WR = {tA: Rot.from_quat(tRef[tA][:, 3:]).as_matrix() for tA in (0, 1)}
        tQTree = cKDTree(tJ30)
        tCtx = dataclasses.replace(tCtx, mQTree=tQTree, mBank=tBank, mBankD=tQTree.query(tBank)[0],
                                   mBankT0=np.array([np.stack(fkT(tCtx, tQ)) for tQ in tBank]))
        tLambda = float(json.load(open(CONTRACTS / "v2kr_frozen.json"))["lambda"])
        tCtx = dataclasses.replace(tCtx, mLambda=tLambda)
        print(f"R30 contract: {len(tR30)} episodes / {len(tJ30)} frames, bank {len(tBank)} (sha {tSb['sha256_q12'][:12]}), "
              f"lambda {tLambda}", flush=True)
        if mode == "tr":
            tCtx = dataclasses.replace(tCtx, mWindows=buildWindows(tCtx, tD150, tMask))
    if mode == "v2kr":
        tFrozen = json.load(open(CONTRACTS / "v2kr_frozen.json"))
        tCtx = dataclasses.replace(tCtx, mLambda=float(tFrozen["lambda"]))
        if tFrozen["seed_bank"]["sha256_q12"] != _sha(tCtx.mBank):
            raise ContractError("seed bank changed")
        print(f"v2kr frozen: lambda {tCtx.mLambda}, K {K_ROLL}+{K_MANI}", flush=True)
    return tCtx


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--episodes-json", required=True)
    tAp.add_argument("--raw-root", required=True)
    tAp.add_argument("--prefix", default="Hpilot")
    tAp.add_argument("--runs", required=True)
    tAp.add_argument("--export", required=True, action="append", help="repeatable; first root holding <tag>/imu_data.json wins")
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--part", default="0/1")
    tAp.add_argument("--mode", choices=MODES, default="v2k")
    return tAp


def loadRlc():
    """~/ego_collector/scripts/robotlike_offline_check.py as a module (its P3 is the Phase-3 module used here)"""
    tSpec = importlib.util.spec_from_file_location("rlc", RLC_PATH)
    tRlc = importlib.util.module_from_spec(tSpec)
    tSpec.loader.exec_module(tRlc)
    return tRlc


def episodeRecord(ctx: RetargetContext, rlc, clr, episode: str, rawRoot: str, prefix: str, out: pathlib.Path) -> dict:
    """retarget every segment of one episode; writes <episode>.npz -> the episode record"""
    tSess, tNum = episode.rsplit("_", 1)
    tEp = pathlib.Path(rawRoot) / f"{prefix}_{tSess}" / f"episode_{tNum}"
    tT0 = time.time()
    try:
        tSegs, tWhy = segments(ctx, rlc, tEp)
        tRec = dict(episode=episode, first_fail=tWhy if tSegs is None else None, segments=[])
        tQs = {}
        for tStart, tTL, tTR in (tSegs or []):
            if tTL is None:
                tRec["segments"].append(dict(start=tStart, first_fail="B_metric"))
                continue
            tRes = doSegmentTr(ctx, tTL, tTR, clr) if ctx.mMode == "tr" else doSegment(ctx, tTL, tTR, clr)
            tV1, tV2 = tRes[0], tRes[1]
            tV2r = tRes[2] if len(tRes) > 2 else None
            tSegRec = dict(start=tStart, v1=_strip(tV1), v1_pass=bool(tV1["ik_ok"] and tV1["collision_ok"] and tV1["ws_ok"]),
                           v2=_pack(tV2))
            if tV2r is not None:
                tSegRec["v2r"] = _pack(tV2r)
            tRec["segments"].append(tSegRec)
            tQs[f"{tStart}_v1"] = tV1["q"]
            for tJ, (_, _, tR) in enumerate(tV2["kept"]):
                tQs[f"{tStart}_v2_{tJ}"] = tR["q"]
            if tV2r is not None:
                for tJ, (_, _, tR) in enumerate(tV2r["kept"]):
                    tQs[f"{tStart}_v2r_{tJ}"] = tR["q"]
        if tQs:
            np.savez_compressed(out / f"{episode}.npz", **tQs)
        tRec["seconds"] = round(time.time() - tT0, 1)
    except Exception as tExc:
        tRec = dict(episode=episode, error=f"{type(tExc).__name__}: {tExc}")
    return tRec


def _strip(r: dict) -> dict:
    return {tK: tV for tK, tV in r.items() if tK != "q"}


def _pack(v: dict) -> dict:
    return dict(n_ws_candidates=v["n_ws_candidates"], n_rolled=v["n_rolled"], n_feasible=v["n_feasible"], pass_=bool(v["kept"]),
                kept=[dict(seed=tI, sel_score=tSc, **_strip(tR)) for tSc, tI, tR in v["kept"]],
                variant_dist_rad=[float(np.linalg.norm(v["kept"][tJ][2]["q"] - v["kept"][0][2]["q"], axis=1).mean())
                                  for tJ in range(1, len(v["kept"]))])


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tRlc = loadRlc()
    P3 = tRlc.P3
    P3.RUNS = pathlib.Path(tArgs.runs)
    tRoots = [pathlib.Path(tX) for tX in tArgs.export]
    P3.export_dir = lambda tag: next((tR / tag for tR in tRoots if (tR / tag / "imu_data.json").exists()), tRoots[0] / tag)
    tT0 = time.time()
    P3.init()
    tT1 = time.time()
    tClr = tRlc.Clearance(P3.ik.kin)
    tCtx = buildContext(P3, tArgs.mode, tT1 - tT0, tT1)
    tOut = pathlib.Path(tArgs.out)
    tOut.mkdir(parents=True, exist_ok=True)
    tK, tN = map(int, tArgs.part.split("/"))
    for tE in json.load(open(tArgs.episodes_json))["episodes"][tK::tN]:
        tDst = tOut / f"{tE}.json"
        if tDst.exists():
            continue
        tRec = episodeRecord(tCtx, tRlc, tClr, tE, tArgs.raw_root, tArgs.prefix, tOut)
        tDst.write_text(json.dumps(tRec, indent=1, default=float))
        tSs = [tS for tS in tRec.get("segments", []) if "v1" in tS]
        print(f"{tE}: segs {len(tSs)}  v1 pass {sum(tS['v1_pass'] for tS in tSs)}  "
              f"v2 pass {sum(tS['v2']['pass_'] for tS in tSs)}  "
              + (f"v2r pass {sum(tS['v2r']['pass_'] for tS in tSs)}  " if tSs and "v2r" in tSs[0] else "") + "  "
              f"feasible seeds/seg {np.mean([tS['v2']['n_feasible'] for tS in tSs]) if tSs else 0:.1f}  "
              f"{tRec.get('seconds', tRec.get('error'))}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
