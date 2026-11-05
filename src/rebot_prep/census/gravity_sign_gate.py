"""[2026-09-25] §23 G1 gravity-sign / frame-convention gate, on REAL data (independent of any selector result) (Gen-1 census
stage; was c8_g1_gravity_sign.py).

In:  ~/c8/phase3_census.json + ~/c8/phase3/<episode>.npz (phase3_segments), ~/c8/c8_quarantine.json, ~/c8/c8_domain_manifest.json,
     ~/c8/export[_early]/<tag>/imu_data.json, ~/c8/stage_p3/runs/<tag>/ss1.csv, ~/umi_bridge/track_c/data/r150_q_tcp.npz
Out: ~/c8/g1_gravity_sign.json and the printed angles + "G1 PASS|FAIL"
Usage: python -m rebot_prep.census.gravity_sign_gate

Tilt vector = gravity expressed in the robot FK-TCP frame convention, unit length (2 DOF, no yaw):
  human  : g_w from imu_vi_g (that side's MASt3R world, pointing down) -> HU TCP frame (R_wtcp^T g_w, TCP = T_wc @ X)
           -> FK-TCP convention via the frozen chain: R_fk = R_hu F C  =>  v_fk = C^T F^T v_hu
  robot  : base -z (assumed z-up URDF base) in the FK TCP frame: v_fk = R_fk(q)^T [0, 0, -1]
Both populations are table-top 3-stack manipulation, so their tilt distributions should overlap. The gate compares the
angle between the mean tilt directions (and the median per-sample NN angle) under the chain as frozen vs with each
single sign / frame flip (g sign, F omitted, C omitted, C^-1). PASS = the frozen chain is the clear best and close.
Human samples: every 15th valid row of every usable (bimanual E) segment of the integrity-filtered census.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from ego_paths import DATA_ROOT, UMI_BRIDGE, prepend_sys_path

C8 = DATA_ROOT
SEG = 65                         # Phase-3 segment rows
HUMAN_STRIDE = 15                # every 15th row of a segment
ROBOT_SAMPLES = 4000
ROBOT_SEED = 0
VI_WINDOW, VI_BIAS = 15, False   # the frozen C-P6 choice
DOWN = np.array([0, 0, -1.0])
FROZEN = "frozen chain (g, F, C)"


def frozenModules():
    """(loadFrozenFit, pseudo_joint_pipeline): the ~/umi_bridge/track_c + ~/c8 copies first, else the repository's"""
    prepend_sys_path(str(UMI_BRIDGE / "track_c"))
    prepend_sys_path(str(C8))
    try:
        import c8_imu_vi_g
        import pseudo_joint_pipeline as P
        return c8_imu_vi_g.load, P
    except ImportError:  # clean checkout: the repository copy
        from rebot_prep.pseudo_joint import pseudo_joint_pipeline as P

        from .imu_vi_gravity import loadFrozenFit
        return loadFrozenFit, P


def humanTilts(ns: dict) -> dict:
    """per side: gravity direction in the HU TCP frame, every HUMAN_STRIDE-th valid row of the usable segments"""
    tX = ns["X"]
    tEpisodes = json.load(open(C8 / "phase3_census.json"))["episodes"]
    tQuarantine = set(json.load(open(C8 / "c8_quarantine.json"))["sessions"])
    tManifest = json.load(open(C8 / "c8_domain_manifest.json"))
    tHum = {"left": [], "right": []}
    for tE, tRec in tEpisodes.items():
        if tE.rsplit("_", 1)[0] in tQuarantine or tRec.get("first_fail") is not None:
            continue
        tStarts = [tS["start"] for tS in tRec.get("segments", []) if tS["first_fail"] is None]
        if not tStarts:
            continue
        tJR = np.load(C8 / "phase3" / f"{tE}.npz")["jR"]
        for tSide in ("left", "right"):
            tTag = f"{tE}_{tSide}"
            tVi = ns["imu_vi"](tTag, VI_WINDOW, VI_BIAS)
            if tVi is None:
                continue
            tExport = C8 / ("export" if tManifest[tTag]["domain"] == "main_calibrated" else "export_early") / tTag
            tN = len(json.load(open(tExport / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
            tT, tOk = ns["load_traj"](C8 / "stage_p3/runs" / tTag / "ss1.csv", tN)
            tG = np.array(tVi["g_world"])
            tG /= np.linalg.norm(tG)
            for tS0 in tStarts:
                tRows = np.arange(tS0, tS0 + SEG, HUMAN_STRIDE)
                tRows = tJR[tRows] if tSide == "right" else tRows
                for tI in tRows:
                    if tOk[tI]:
                        tHum[tSide].append((tT[tI, :3, :3] @ tX[:3, :3]).T @ tG)
    return {tK: np.array(tV) for tK, tV in tHum.items()}


def robotTilts(P) -> dict:
    """per arm: base -z in the FK TCP frame on ROBOT_SAMPLES random R150 frames"""
    tKin = P.ContinuityIK(P.IKCfg()).kin
    tJ = np.load(UMI_BRIDGE / "track_c/data/r150_q_tcp.npz")["J"]
    tIdx = np.random.default_rng(ROBOT_SEED).choice(len(tJ), ROBOT_SAMPLES, replace=False)
    tRob = {"left": [], "right": []}
    for tI in tIdx:
        for tArm, tP7 in enumerate(tKin.fk_pose7(np.radians(np.r_[tJ[tI, :6], tJ[tI, 7:13]]))):
            tRob["left" if tArm == 0 else "right"].append(Rot.from_quat(tP7[3:]).as_matrix().T @ DOWN)
    return {tK: np.array(tV) for tK, tV in tRob.items()}


def angleDeg(a: np.ndarray, b: np.ndarray) -> float:
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def main(argv: list[str] | None = None) -> int:
    os.environ["M3_BASE"] = str(C8 / "stage_p3")
    os.environ["M3_VARIANT"] = "ss1"
    tLoadFrozenFit, P = frozenModules()
    tNs = tLoadFrozenFit()
    tF3, tC3 = P.F[:3, :3], P.C[:3, :3]
    tHum = humanTilts(tNs)
    print("human tilt samples", {tK: len(tV) for tK, tV in tHum.items()})
    tRob = robotTilts(P)
    tVariants = {FROZEN: lambda v: v @ tF3 @ tC3, "g sign flipped": lambda v: -v @ tF3 @ tC3, "F omitted": lambda v: v @ tC3,
                 "C omitted": lambda v: v @ tF3, "C inverted": lambda v: v @ tF3 @ tC3.T}
    tOut = {}
    for tSide in ("left", "right"):
        tRm = tRob[tSide].mean(0)
        tTree = cKDTree(tRob[tSide])
        tOut[tSide] = {}
        print(f"\n== {tSide}: robot mean tilt resultant length {np.linalg.norm(tRm):.2f}")
        for tName, tMap in tVariants.items():
            tH = tMap(tHum[tSide])
            tH /= np.linalg.norm(tH, axis=1, keepdims=True)
            tHm = tH.mean(0)
            tNn = np.degrees(2 * np.arcsin(np.clip(tTree.query(tH)[0] / 2, 0, 1)))
            tOut[tSide][tName] = dict(mean_dir_angle_deg=float(angleDeg(tHm, tRm)), nn_angle_p50_deg=float(np.median(tNn)))
            print(f"   {tName:<24} mean-direction angle {angleDeg(tHm, tRm):6.1f} deg   "
                  f"per-sample NN angle p50 {np.median(tNn):5.1f} deg   (human resultant {np.linalg.norm(tHm):.2f})")
    tBest = {tS: min(tOut[tS], key=lambda k: tOut[tS][k]["mean_dir_angle_deg"]) for tS in tOut}
    tOk = all(tB == FROZEN for tB in tBest.values())
    print("\nG1", "PASS" if tOk else "FAIL", {tS: (tB, round(tOut[tS][tB]["mean_dir_angle_deg"], 1)) for tS, tB in tBest.items()})
    json.dump(tOut, open(C8 / "g1_gravity_sign.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
