"""[2026-09-23] C-P6: metric scale source for the (frozen) MASt3R-SLAM RGB-only trajectories (pose stage; was cp6_scale.py).

In:  <M3_BASE>/qa/qa_all.json (per tag: n_src_frames, M3_<variant>.marker.lb_scale), <M3_BASE>/handumi_camera_tcp_v2.yaml,
     <M3_BASE>/val/<split>/<demo>_<side>/{imu_data.json, orbslam_setting.yaml}, <M3_BASE>/runs/<tag>/<variant>.csv
Out: <M3_BASE>/qa/cp6_scale.json (cp6_scale_<variant>.json for another variant) and a printed table.
Usage: M3_BASE=<dir> M3_VARIANT=ss1 python -m mast3r_pose.imu_metric_scale

GT = ArUco long-baseline scale (qa_all.json lb_scale, marker_flip_reject_v1) -- VALIDATION ONLY, never a production input.
Split: FIT = valB (9 demos, 18 sides) chooses every constant / hyperparameter; EVAL = valC (5 demos, 10 sides).

Methods
  global      one constant = median GT scale over FIT sides
  lr_const    per camera (left / right) constant = median GT scale over FIT sides of that camera
  imu_vi      per EPISODE from the wrist IMU, no double integration:
                R_wb f_b - a_lever = s * a_cam - g + R_wb b_a            (per camera frame, 3 rows)
              a_cam  = 2nd derivative of the MASt3R camera position (world units), Savitzky-Golay window W
              a_lever= 2nd derivative of R_wc t_cb (metric lever arm camera->IMU, from IMU.T_b_c1)
              f_b    = accelerometer specific force, box-averaged to camera times, rotated by R_wb = R_wc R_cb,
                       then the SAME SG filter (deriv 0) so both sides carry the same band limit
              unknowns s, g (3), b_a (3, optional); robust LS (Huber IRLS); |g| reported as a sanity check
              hyperparameters (W, bias on/off) chosen on FIT by median |log(s/s_gt)|
Downstream (per side; TCP = camera_tcp_v2 on the metric-scaled trajectory; rotation is the same in every method, so the
difference is scale-induced only):
  scale error        |s_est / s_gt - 1|
  UMI displacement   d_k = translation of inv(T_tcp(t)) @ T_tcp(t + k) on the 15 Hz grid, k = 1, 4, 8, 16, 32;
                     error |d_k(s_est) - d_k(s_gt)| in mm and relative to |d_k(s_gt)|, p50 / p90

The census stage (rebot_prep.census) does NOT import this module: it execs the frozen external copy of the original
cp6_scale.py text, so changes here never reach it.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

from ego_paths import SHARED_ROOT

ENV_BASE = "M3_BASE"
ENV_VARIANT = "M3_VARIANT"
DEFAULT_VARIANT = "ss1"          # [2026-09-23] C8: ss1_5080 = the same benchmark re-run on the RTX 5080
DEFAULT_BASE_NAME = "trackA_mast3r_pose_v1"
KS = (1, 4, 8, 16, 32)           # UMI displacement horizons, in 15 Hz steps
CAMERA_HZ = 30.0
BOX_HALF_S = 0.5 / CAMERA_HZ     # accelerometer box average: +- half a camera frame period
SG_ORDER = 3                     # Savitzky-Golay polynomial order
MIN_WINDOWS = 3                  # a side needs >= 3 SG windows of valid frames
IRLS_ITERS = 10
MAD_TO_SIGMA = 1.4826            # robust sigma from the median absolute residual
HUBER_K = 1.345                  # Huber threshold in robust sigmas
EPS = 1e-9
GRID_WINDOWS = (9, 15, 21, 31, 45)
GRID_BIAS = (False, True)
GRID_STEP = 2                    # 30 fps camera frames -> the 15 Hz UMI grid
REL_MIN_M = 0.005                # relative error only where the true displacement exceeds 5 mm
M_TO_MM = 1e3
METHODS = ("global", "lr_const", "imu_vi")
QUAT_KEYS = ("q_x", "q_y", "q_z", "q_w")
T_B_C_PATTERN = re.compile(r"IMU\.T_b_c1:.*?data:\s*\[([^\]]+)\]", re.S)


@dataclass(frozen=True)
class Benchmark:
    """The benchmark tree: base folder, MASt3R variant, qa_all.json and the camera -> TCP transform X."""
    mBase: Path
    mVariant: str
    mQa: dict
    mCameraTcp: np.ndarray

    @classmethod
    def load(cls, base: Path, variant: str) -> "Benchmark":
        tQa = json.load(open(base / "qa/qa_all.json"))
        tCt = yaml.safe_load(open(base / "handumi_camera_tcp_v2.yaml"))
        tX = np.eye(4)
        tX[:3, :3] = np.array(tCt["rotation"]["R_camera_tcp"], float)
        tX[:3, 3] = tCt["translation"]["t_camera_tcp_m"]
        return cls(base, variant, tQa, tX)

    def runCsv(self, tag: str) -> Path:
        return self.mBase / f"runs/{tag}/{self.mVariant}.csv"

    def sideDir(self, tag: str) -> Path:
        """<base>/val/<split>/<rest of the tag>"""
        tSplit, tRest = tag.split("_", 1)
        return self.mBase / "val" / tSplit / tRest


def loadTrajectory(path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """MASt3R csv -> T_WC (n, 4, 4) indexed by frame_idx, valid mask (is_lost == false); frames >= n are ignored"""
    tT = np.tile(np.eye(4), (n, 1, 1))
    tOk = np.zeros(n, bool)
    for tRow in csv.DictReader(open(path)):
        tI = int(tRow["frame_idx"])
        if tI < n and tRow["is_lost"] == "false":
            tT[tI, :3, :3] = Rot.from_quat([float(tRow[tK]) for tK in QUAT_KEYS]).as_matrix()
            tT[tI, :3, 3] = [float(tRow["x"]), float(tRow["y"]), float(tRow["z"])]
            tOk[tI] = True
    return tT, tOk


def bodyFromCamera(settingPath: Path) -> np.ndarray:
    """IMU.T_b_c1 (camera -> IMU body, 4x4) from an ORB-SLAM3 settings file"""
    tMatch = T_B_C_PATTERN.search(settingPath.read_text())
    return np.array([float(tV) for tV in tMatch.group(1).split(",")]).reshape(4, 4)


def imuViScale(bench: Benchmark, tag: str, window: int, useBias: bool) -> dict | None:
    """IMU visual-inertial scale of one side -> dict(s, g_norm, n, resid_rms), None when too few valid frames"""
    tRoot = bench.sideDir(tag)
    tStreams = json.load(open(tRoot / "imu_data.json"))["1"]["streams"]
    tCamS = np.array([tS["cts"] for tS in tStreams["CORI"]["samples"]]) / 1e3
    tImuS = np.array([tS["cts"] for tS in tStreams["ACCL"]["samples"]]) / 1e3
    tForce = np.array([tS["value"] for tS in tStreams["ACCL"]["samples"]], float)
    tN = len(tCamS)
    tT, tOk = loadTrajectory(bench.runCsv(tag), tN)
    tCb = np.linalg.inv(bodyFromCamera(tRoot / "orbslam_setting.yaml"))
    tRcb, tTcb = tCb[:3, :3], tCb[:3, 3]
    # accelerometer box-averaged over each camera frame period
    tFb = np.full((tN, 3), np.nan)
    for tI, tS in enumerate(tCamS):
        tSel = np.abs(tImuS - tS) <= BOX_HALF_S
        if tSel.any():
            tFb[tI] = tForce[tSel].mean(0)
    tIdx = np.flatnonzero(tOk & np.isfinite(tFb).all(1))
    if len(tIdx) < MIN_WINDOWS * window:
        return None
    # use the longest contiguous valid run (MASt3R coverage is 1.0 on almost every side)
    tRuns = np.split(tIdx, np.flatnonzero(np.diff(tIdx) != 1) + 1)
    tRun = max(tRuns, key=len)
    tDt = float(np.median(np.diff(tCamS[tRun])))
    tRwc = tT[tRun, :3, :3]
    tPc = tT[tRun, :3, 3]
    tRwb = tRwc @ tRcb
    tAcam = savgol_filter(tPc, window, SG_ORDER, deriv=2, delta=tDt, axis=0)
    tAlever = savgol_filter(np.einsum("nij,j->ni", tRwc, tTcb), window, SG_ORDER, deriv=2, delta=tDt, axis=0)
    tFw = savgol_filter(np.einsum("nij,nj->ni", tRwb, tFb[tRun]), window, SG_ORDER, deriv=0, axis=0)
    tHalf = window // 2
    tInner = slice(tHalf, len(tRun) - tHalf)                                  # drop SG edges
    tY = (tFw - tAlever)[tInner].reshape(-1)
    tCols = [tAcam[tInner].reshape(-1)]
    for tK in range(3):
        tE = np.zeros((len(tRun), 3))
        tE[:, tK] = -1.0
        tCols.append(tE[tInner].reshape(-1))                                   # -g
    if useBias:
        for tK in range(3):
            tCols.append(tRwb[tInner][:, :, tK].reshape(-1))                    # R_wb b_a
    tA = np.stack(tCols, 1)
    tW = np.ones(len(tY))
    for _ in range(IRLS_ITERS):
        tXs = np.linalg.lstsq(tA * tW[:, None], tY * tW, rcond=None)[0]
        tRes = tA @ tXs - tY
        tSigma = MAD_TO_SIGMA * np.median(np.abs(tRes)) + EPS
        tW = np.clip(HUBER_K * tSigma / np.maximum(np.abs(tRes), EPS), 0, 1)
    return dict(s=float(tXs[0]), g_norm=float(np.linalg.norm(tXs[1:4])), n=int(len(tY) // 3),
                resid_rms=float(np.sqrt(np.mean(tRes ** 2))))


def _displacement(tA: np.ndarray, tB: np.ndarray, scale: float, cameraTcp: np.ndarray) -> np.ndarray:
    """translation of inv(T_tcp(a)) @ T_tcp(b) with both camera translations scaled"""
    tA = tA.copy()
    tB = tB.copy()
    tA[:3, 3] *= scale
    tB[:3, 3] *= scale
    return (np.linalg.inv(tA @ cameraTcp) @ (tB @ cameraTcp))[:3, 3]


def displacementErrors(bench: Benchmark, tag: str, sEst: float, sGt: float) -> dict:
    """per horizon k: p50 / p90 of |d_k(s_est) - d_k(s_gt)| in mm and relative to |d_k(s_gt)|"""
    tN = bench.mQa[tag]["n_src_frames"]
    tT, tOk = loadTrajectory(bench.runCsv(tag), tN)
    tOut = {}
    for tK in KS:
        tErr, tRel = [], []
        for tI in range(0, tN - GRID_STEP * tK, GRID_STEP):
            if not (tOk[tI] and tOk[tI + GRID_STEP * tK]):
                continue
            tDg = _displacement(tT[tI], tT[tI + GRID_STEP * tK], sGt, bench.mCameraTcp)
            tDe = _displacement(tT[tI], tT[tI + GRID_STEP * tK], sEst, bench.mCameraTcp)
            tErr.append(np.linalg.norm(tDe - tDg) * M_TO_MM)
            if np.linalg.norm(tDg) > REL_MIN_M:
                tRel.append(np.linalg.norm(tDe - tDg) / np.linalg.norm(tDg))
        tOut[tK] = dict(mm_p50=float(np.median(tErr)), mm_p90=float(np.percentile(tErr, 90)),
                        rel_p50=float(np.median(tRel)) if tRel else np.nan,
                        rel_p90=float(np.percentile(tRel, 90)) if tRel else np.nan)
    return tOut


def sideOf(tag: str) -> str:
    return "left" if tag.endswith("_left") else "right"


def _printSplit(rows: list[dict], split: str) -> None:
    print(f"\n=== {split.upper()} ({'valC, held out' if split == 'eval' else 'valB, used for fitting'}) ===")
    print("  method    side   n  scale_err p50/p90 | " + " | ".join(f"k{tK} mm p50/p90 (rel p90)" for tK in KS))
    for tMethod in METHODS:
        for tSide in ("left", "right", "all"):
            tSel = [tR for tR in rows if tR["split"] == split and tR["method"] == tMethod
                    and (tSide == "all" or tR["side"] == tSide) and tR.get("s") == tR.get("s")]
            if not tSel:
                continue
            tScaleErr = np.array([tR["scale_err"] for tR in tSel])
            tCells = []
            for tK in KS:
                tP50 = np.array([tR["umi"][tK]["mm_p50"] for tR in tSel])
                tP90 = np.array([tR["umi"][tK]["mm_p90"] for tR in tSel])
                tRel90 = np.array([tR["umi"][tK]["rel_p90"] for tR in tSel])
                tCells.append(f"{np.median(tP50):5.1f}/{np.median(tP90):5.1f} ({np.nanmedian(tRel90)*100:4.1f}%)")
            print(f"  {tMethod:<9} {tSide:<5} {len(tSel):>2}  "
                  f"{np.median(tScaleErr)*100:5.1f}%/{np.percentile(tScaleErr,90)*100:5.1f}%  | " + " | ".join(tCells))


def runBenchmark(bench: Benchmark) -> dict:
    """fit constants and IMU-VI hyperparameters on FIT, evaluate all methods, print the tables -> the cp6_scale.json content"""
    tMarker = f"M3_{bench.mVariant}"
    tTags = [tT for tT in bench.mQa if "lb_scale" in bench.mQa[tT].get(tMarker, {}).get("marker", {})]
    tGt = {tT: bench.mQa[tT][tMarker]["marker"]["lb_scale"] for tT in tTags}
    tFit = [tT for tT in tTags if tT.startswith("valB")]
    # ---- constants fitted on FIT only
    tGlobal = float(np.median([tGt[tT] for tT in tFit]))
    tLr = {tS: float(np.median([tGt[tT] for tT in tFit if sideOf(tT) == tS])) for tS in ("left", "right")}
    # ---- IMU-VI hyperparameters chosen on FIT only
    tGrid = [(tW, tB) for tW in GRID_WINDOWS for tB in GRID_BIAS]
    tCache = {}
    for tWindow, tBias in tGrid:
        for tTag in tTags:
            tCache[(tTag, tWindow, tBias)] = imuViScale(bench, tTag, tWindow, tBias)

    def tFitScore(tWindow, tBias):
        tV = [abs(np.log(tCache[(tT, tWindow, tBias)]["s"] / tGt[tT])) for tT in tFit
              if tCache[(tT, tWindow, tBias)] and tCache[(tT, tWindow, tBias)]["s"] > 0]
        return float(np.median(tV)) if len(tV) >= len(tFit) // 2 else np.inf

    tScores = {(tW, tB): tFitScore(tW, tB) for (tW, tB) in tGrid}
    tBest = min(tScores, key=tScores.get)
    print("IMU-VI hyperparameter search on FIT (median |log s/s_gt|):")
    for (tWindow, tBias), tV in tScores.items():
        tMark = "   <- chosen" if (tWindow, tBias) == tBest else ""
        print(f"   W {tWindow:>2} bias {'on ' if tBias else 'off'}  {tV:.3f}{tMark}")

    def tEstimate(tMethod, tTag):
        if tMethod == "global":
            return tGlobal
        if tMethod == "lr_const":
            return tLr[sideOf(tTag)]
        tR = tCache[(tTag, *tBest)]
        return tR["s"] if tR and tR["s"] > 0 else np.nan

    tRows = []
    for tTag in tTags:
        for tMethod in METHODS:
            tS = tEstimate(tMethod, tTag)
            tSplit = "fit" if tTag in tFit else "eval"
            if not tS == tS:
                tRows.append(dict(tag=tTag, split=tSplit, side=sideOf(tTag), method=tMethod, s=np.nan))
                continue
            tVi = tCache[(tTag, *tBest)]
            tRows.append(dict(tag=tTag, split=tSplit, side=sideOf(tTag), method=tMethod, s=tS, s_gt=tGt[tTag],
                              scale_err=abs(tS / tGt[tTag] - 1), umi=displacementErrors(bench, tTag, tS, tGt[tTag]),
                              g_norm=tVi["g_norm"] if tMethod == "imu_vi" and tVi else None))

    print(f"\nconstants (FIT): global {tGlobal:.3f}  left {tLr['left']:.3f}  right {tLr['right']:.3f}   "
          f"IMU-VI W {tBest[0]} bias {tBest[1]}")
    tGn = [tR["g_norm"] for tR in tRows if tR["method"] == "imu_vi" and tR.get("g_norm")]
    print(f"IMU-VI |g| estimate: median {np.median(tGn):.2f} (p10 {np.percentile(tGn,10):.2f}, "
          f"p90 {np.percentile(tGn,90):.2f}) m/s2  -- 9.81 expected")
    for tSplit in ("eval", "fit"):
        _printSplit(tRows, tSplit)
    return dict(constants=dict(global_=tGlobal, **tLr), imu_vi_choice=dict(W=tBest[0], bias=tBest[1]),
                fit_scores={f"{tK[0]}_{tK[1]}": tV for tK, tV in tScores.items()}, rows=tRows)


def main(argv: list[str] | None = None) -> int:
    tBase = Path(os.environ.get(ENV_BASE, str(SHARED_ROOT / DEFAULT_BASE_NAME)))
    tVariant = os.environ.get(ENV_VARIANT, DEFAULT_VARIANT)
    tBench = Benchmark.load(tBase, tVariant)
    tOut = runBenchmark(tBench)
    tName = "qa/cp6_scale.json" if tVariant == DEFAULT_VARIANT else f"qa/cp6_scale_{tVariant}.json"
    json.dump(tOut, open(tBase / tName, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
