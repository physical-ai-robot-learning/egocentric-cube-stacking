"""Dataset integrity report (spec 31/32) and the rotation validity check (was validation/integrity.py + geometry_checks.py).

Hard checks: shapes (state 20, CART20 [16,20], action32 [16,32]), action[:, :20] == CART20, AUX12 exactly zero (NaN counts
as non-zero), no NaN / Inf in training rows, gripper in [0, 1], rotations orthonormal (det +1), task-start anchor = identity,
split disjoint, every training row's 16 targets inside the episode (no tail padding).
validateDataset returns the report with `fails` and `pass`; scripts/validate_dataset.py writes it and sets the exit status.
rotationReport: reconstruct R from stored rot6d by Gram-Schmidt and also check the RAW stored rows are already orthonormal
(a stored rot6d that only becomes valid after Gram-Schmidt is a bug upstream).

Pipeline stage: dataset-stage validation of a processed tree.
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

from .config import HORIZON, UMI_DT
from .processed_episode import loadEpisode
from .rotation6d import rotation6dToMatrix, rotationAngle

ROTATION_TOL = 1e-4             # orthonormality / determinant / raw-row tolerance of a stored rot6d
ANCHOR_TOL = 1e-5               # task-start state must be the identity pose within this
ROW_S = 1 / 15                  # one row: a training row's last target may end up to one row after the last valid row
END_EPS_S = 1e-6
IDENTITY_POSE9_X2 = [0, 0, 0, 1, 0, 0, 0, 1, 0] * 2
REPORT_HORIZONS = (1, 4, 8, 16)
ARM_OFFSETS = (0, 10)           # CART20 channel offset of the left / right arm


def pct(x, qs=(50, 95, 99)):
    """{"p50": ..., ...} of the finite values, None if there are none"""
    tX = np.asarray(x, np.float64)
    tX = tX[np.isfinite(tX)]
    return {f"p{tQ}": float(np.percentile(tX, tQ)) for tQ in qs} if len(tX) else None


def rotationReport(d6, tol=ROTATION_TOL) -> dict:
    """validity counts / max errors of the finite rows of a (n, 6) rot6d array"""
    tD6 = np.asarray(d6, np.float64)
    tD6 = tD6[np.isfinite(tD6).all(1)]
    tR = rotation6dToMatrix(tD6)
    tOrth = np.abs(np.swapaxes(tR, -1, -2) @ tR - np.eye(3)).max(axis=(1, 2))
    tDet = np.abs(np.linalg.det(tR) - 1)
    tA, tB = tD6[:, :3], tD6[:, 3:]
    tRaw = np.maximum.reduce([np.abs(np.linalg.norm(tA, axis=1) - 1), np.abs(np.linalg.norm(tB, axis=1) - 1),
                              np.abs(np.sum(tA * tB, 1))])
    tBad = (tOrth > tol) | (tDet > tol) | (tRaw > tol)
    return dict(n=int(len(tD6)), invalid_rotation_count=int(tBad.sum()), max_orthogonality_error=float(tOrth.max()),
                max_determinant_error=float(tDet.max()), max_raw_rot6d_error=float(tRaw.max()))


def validateDataset(root) -> collections.OrderedDict:
    """processed root (with {train,val,test}_manifest.jsonl) -> integrity report"""
    tRoot = pathlib.Path(root)
    tFails = []
    tRep = collections.OrderedDict()
    tManifest = {tS: [json.loads(tLine) for tLine in open(tRoot / f"{tS}_manifest.jsonl")] for tS in ("train", "val", "test")}
    tIds = [tRec["episode_id"] for tS in tManifest for tRec in tManifest[tS]]
    if len(tIds) != len(set(tIds)):
        tFails.append("episode in more than one split")
    tTot = collections.Counter()
    tByOrder = collections.defaultdict(collections.Counter)
    tByStage = collections.Counter()
    tBySplit = collections.Counter()
    tCams = collections.Counter()
    tGaps = []
    tGMin, tGMax = np.inf, -np.inf
    tTransMag = {tK: [] for tK in REPORT_HORIZONS}
    tRotMag = {tK: [] for tK in REPORT_HORIZONS}
    tRots = []
    tAnchorErr = 0.0
    tNan = tInf = tAuxNz = 0
    for tS in tManifest:
        for tRec in tManifest[tS]:
            tEp = loadEpisode(tRoot / tRec["path"])
            tMeta = tEp["metadata"]
            tTrain = tEp["train_rows"]
            tCart = tEp["cart20"]
            tAction = tEp["action"]
            tTot["episodes"] += 1
            tTot["raw_frames"] += sum(tMeta["frames_raw"].values()) // 2
            tTot["rows_15hz"] += tMeta["frames_rows"]
            tTot["rows_valid"] += tMeta["rows_valid"]
            tTot["train_rows"] += len(tTrain)
            tByOrder[tMeta["stack_order"]]["episodes"] += 1
            tByOrder[tMeta["stack_order"]]["frames"] += tMeta["frames_rows"]
            tByOrder[tMeta["stack_order"]]["samples"] += len(tTrain)
            tBySplit[tS] += len(tTrain)
            tByStage.update(tEp["stage_id"][tTrain].tolist())
            for tCam, tOk in tMeta["camera_available"].items():
                tCams[tCam] += int(tOk)
            if tCart.shape[1:] != (HORIZON, 20) or tAction.shape[1:] != (HORIZON, 32) or tEp["state"].shape[1:] != (20,):
                tFails.append(f"{tRec['episode_id']}: shape")
            if not np.array_equal(tAction[..., :20], tCart):
                tFails.append(f"{tRec['episode_id']}: action[:, :20] != CART20")
            tAuxNz += int(np.count_nonzero(tAction[..., 20:]))
            tState = tEp["state"][tTrain]
            tNan += int(np.isnan(tState).sum() + np.isnan(tAction).sum())
            tInf += int(np.isinf(tState).sum() + np.isinf(tAction).sum())
            tG = np.concatenate([tState[:, 18:20].ravel(), tCart[..., [9, 19]].ravel()])
            tGMin, tGMax = min(tGMin, tG.min()), max(tGMax, tG.max())
            tAnchorErr = max(tAnchorErr, float(np.abs(tEp["state"][0, :18] - np.array(IDENTITY_POSE9_X2)).max()))
            tT = tEp["timestamps"]
            tGaps += np.diff(tT).tolist()
            tLastValid = tT[np.flatnonzero(tEp["row_valid"])[-1]]
            if len(tTrain) and tT[tTrain].max() + HORIZON * UMI_DT > tLastValid + ROW_S + END_EPS_S:
                tFails.append(f"{tRec['episode_id']}: target beyond episode end")
            for tK in tTransMag:
                for tO in ARM_OFFSETS:
                    tTransMag[tK] += np.linalg.norm(tCart[:, tK - 1, tO:tO + 3], axis=1).tolist()
                    tRotMag[tK] += np.degrees(rotationAngle(rotation6dToMatrix(tCart[:, tK - 1, tO + 3:tO + 9]))).tolist()
            tRots.append(tCart[..., [3, 4, 5, 6, 7, 8]].reshape(-1, 6))
            tRots.append(tCart[..., 13:19].reshape(-1, 6))
            tRots.append(tState[:, 3:9])
            tRots.append(tState[:, 12:18])
    tRotRep = rotationReport(np.concatenate(tRots))
    tRep["counts"] = dict(tTot)
    tRep["shapes"] = dict(state=[20], cart20=[HORIZON, 20], action32=[HORIZON, 32])
    tRep["nan_count"], tRep["inf_count"], tRep["nonzero_aux12_count"] = tNan, tInf, tAuxNz
    tRep["gripper_min_max"] = [float(tGMin), float(tGMax)]
    tRep["samples_by_order"] = {tK: dict(tV) for tK, tV in sorted(tByOrder.items())}
    tRep["samples_by_stage"] = {str(tK): tV for tK, tV in sorted(tByStage.items())}
    tRep["samples_by_split"] = dict(tBySplit)
    tRep["camera_available_episodes"] = dict(tCams)
    tRep["row_dt_s"] = dict(min=float(np.min(tGaps)), max=float(np.max(tGaps)), mean=float(np.mean(tGaps)))
    tRep["cart_translation_m"] = {f"k{tK}": pct(tV) for tK, tV in tTransMag.items()}
    tRep["cart_rotation_deg"] = {f"k{tK}": pct(tV) for tK, tV in tRotMag.items()}
    tRep["rotation_validity"] = tRotRep
    tRep["task_start_anchor_max_err"] = tAnchorErr
    tMedians = [tRep["cart_translation_m"][f"k{tK}"]["p50"] for tK in REPORT_HORIZONS]
    tRep["horizon_monotone_translation_p50"] = bool(np.all(np.diff(tMedians) > 0))
    if tAuxNz:
        tFails.append(f"AUX12 non-zero entries: {tAuxNz}")
    if tNan or tInf:
        tFails.append(f"NaN {tNan} / Inf {tInf} in training rows")
    if tGMin < 0 or tGMax > 1:
        tFails.append(f"gripper outside [0,1]: {tGMin}..{tGMax}")
    if tRotRep["invalid_rotation_count"]:
        tFails.append(f"invalid rotations {tRotRep['invalid_rotation_count']}")
    if tAnchorErr > ANCHOR_TOL:
        tFails.append(f"task-start anchor not identity ({tAnchorErr})")
    if not tRep["horizon_monotone_translation_p50"]:
        tFails.append(f"translation p50 not increasing with k: {tMedians}")
    tRep["fails"] = tFails
    tRep["pass"] = not tFails
    return tRep
