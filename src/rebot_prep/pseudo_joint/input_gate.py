"""[2026-09-23] Track C C2 input-quality gate + census (OPEN-5). RAW data only: no smoothing, no IK, no pseudo q.
(Gen-1 retarget stage, input check; was c2_input_gate.py.)

In:  --produce <dir> of <episode>_<side>.npz (frame_idx, cam_pos, cam_quat xyzw, seg_id, lost), the R150 reference
     ~/umi_bridge/track_c/data/r150_q_tcp.npz, the corrected gripper folder (--grip, width availability only)
Out: ~/umi_bridge/track_c/reports/c2/<tag>/{census_segments.csv, census.json, hist_data.npz} and a printed summary
Usage: python -m rebot_prep.pseudo_joint.input_gate [--produce <dir>] [--tag raw] [--grip <dir>]

Per HandUMI segment (v2 TCP from cam_pos/cam_quat, 15 Hz) and per R150 reference window (zarr TCP, 15 Hz, 22-row
non-overlapping windows = the HandUMI median segment length):

    step_p50/p95/max_mm      per-step translation
    rot_step_p95/max_deg     per-step rotation (body frame)
    rev_rate                 among consecutive step pairs with both > 5 mm: fraction turning > 120 deg (cos < -0.5)
    rot_rev_rate             same on body-frame rotation increments with both > 2 deg
    zigzag_bursts            runs of >= 2 consecutive translation reversals (A-B-A-B with steps > 5 mm)
    lost_adjacent_frac       fraction of rows within +-3 rows of a lost / pose-less frame
    boundary_spike           any step within +-3 rows of a lost frame larger than the R150 p99 step_max

Classification, fixed BEFORE the census ran (R150 ref = both arms, all 150 episodes):
    FAIL_POSE    hard gate (step > 70 mm or rot step > 40 deg), net drift from start > 1.0 m, NaN, or boundary_spike
    FAIL_JITTER  rev_rate, rot_rev_rate or zigzag_bursts above the R150 p99 of that metric
    FAIL_OTHER   fewer than 16 rows (none expected: the segmenter already enforces it)
    PASS_RAW     everything else
Step size alone is NOT a fail (a human may move faster than R150 teleop); it is kept as an outlier score.
Outlier score per metric = R150 empirical CDF of the value (>= 0.99 means beyond the R150 p99).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ego_paths import HOME, UMI_BRIDGE
from umi.common.pose_util import pose_to_mat

from . import pseudo_joint_pipeline as P

TRACK_C = UMI_BRIDGE / "track_c"          # the working folder this gate ran in (data/, reports/)
WIN, MIN_ROWS, NEAR = 22, 16, 3           # R150 window rows (HandUMI median segment), shortest segment, lost-frame vicinity
POS_PAIR_MM, ROT_PAIR_DEG, REV_COS = 5.0, 2.0, -0.5
HARD_DP, HARD_DTH, MAX_DRIFT_M = 70.0, 40.0, 1.0
JITTER = ("rev_rate", "rot_rev_rate", "zigzag_bursts")
KEYS = ["step_p95_mm", "step_max_mm", "rot_step_p95_deg", "rot_step_max_deg", "rev_rate", "rot_rev_rate", "zigzag_bursts"]
MOVING_MIN_PAIRS = 3                      # "moving segment" = >= 3 evaluable >5 mm step pairs
FRAME_STEP = 2                            # 30 fps frames -> the 15 Hz grid
P95, P99 = 95, 99
M_TO_MM = 1e3
EPS = 1e-12
SENSITIVITY_REV = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
CLASSES = ("PASS_RAW", "FAIL_JITTER", "FAIL_POSE", "FAIL_OTHER")


def metrics(T, near=None) -> dict:
    """raw quality metrics of one (n, 4, 4) TCP trajectory; with `near` (rows close to a lost frame) also lost_adjacent_frac
    and the steps next to lost frames (_near_step)"""
    tD = np.linalg.inv(T[:-1]) @ T[1:]
    tDpWorld = np.diff(T[:, :3, 3], axis=0) * M_TO_MM                 # reversal is judged on the path itself
    tN = np.linalg.norm(tDpWorld, axis=1)
    tRv = Rot.from_matrix(tD[:, :3, :3]).as_rotvec()
    tTh = np.degrees(np.linalg.norm(tRv, axis=1))
    tOk = (tN[:-1] > POS_PAIR_MM) & (tN[1:] > POS_PAIR_MM)
    tCos = np.einsum("ij,ij->i", tDpWorld[:-1], tDpWorld[1:]) / (tN[:-1] * tN[1:] + EPS)
    tRev = tOk & (tCos < REV_COS)
    tOkRot = (tTh[:-1] > ROT_PAIR_DEG) & (tTh[1:] > ROT_PAIR_DEG)
    tRvNorm = np.linalg.norm(tRv, axis=1)
    tCosRot = np.einsum("ij,ij->i", tRv[:-1], tRv[1:]) / (tRvNorm[:-1] * tRvNorm[1:] + EPS)
    tRevRot = tOkRot & (tCosRot < REV_COS)
    tBursts, tRun = 0, 0
    for tR in tRev:
        tRun = tRun + 1 if tR else 0
        if tRun == 2:
            tBursts += 1
    tM = dict(rows=len(T), step_p50_mm=float(np.median(tN)), step_p95_mm=float(np.percentile(tN, P95)),
              step_max_mm=float(tN.max()),
              rot_step_p95_deg=float(np.percentile(tTh, P95)), rot_step_max_deg=float(tTh.max()),
              rev_rate=float(tRev.sum() / tOk.sum()) if tOk.any() else 0.0, rev_pairs=int(tOk.sum()),
              rot_rev_rate=float(tRevRot.sum() / tOkRot.sum()) if tOkRot.any() else 0.0, rot_rev_pairs=int(tOkRot.sum()),
              zigzag_bursts=int(tBursts), path_mm=float(tN.sum()),
              drift_m=float(np.linalg.norm(T[:, :3, 3] - T[0, :3, 3], axis=1).max()), finite=bool(np.isfinite(T).all()))
    if near is not None:
        tM["lost_adjacent_frac"] = float(near.mean())
        tM["_near_step"] = tN[near[:-1] | near[1:]]
    return tM


def referenceMetrics(path: pathlib.Path) -> list[dict]:
    """metrics of every non-overlapping WIN-row window of both arms of every R150 episode (15 Hz)"""
    tD = np.load(path)
    tEnds = tD["ends"]
    tStarts = np.r_[0, tEnds[:-1]]
    tRef = []
    for tE in range(len(tEnds)):
        tIdx = np.arange(tStarts[tE], tEnds[tE], FRAME_STEP)
        for tArm in range(2):
            tT = pose_to_mat(np.concatenate([tD[f"pos{tArm}"][tIdx], tD[f"rv{tArm}"][tIdx]], -1)).astype(np.float64)
            for tS in range(0, len(tT) - WIN + 1, WIN):
                tRef.append(metrics(tT[tS:tS + WIN]))
    return tRef


def classify(m: dict, spike: bool, thr: dict) -> tuple[str, str]:
    """-> (class, ';'-joined reasons) by the frozen rule"""
    if m["rows"] < MIN_ROWS:
        return "FAIL_OTHER", "too short"
    if (not m["finite"]) or m["step_max_mm"] > HARD_DP or m["rot_step_max_deg"] > HARD_DTH or m["drift_m"] > MAX_DRIFT_M or spike:
        return "FAIL_POSE", ";".join(tX for tX, tC in (("nan", not m["finite"]), ("hard_dp", m["step_max_mm"] > HARD_DP),
                                                    ("hard_dth", m["rot_step_max_deg"] > HARD_DTH),
                                                    ("drift", m["drift_m"] > MAX_DRIFT_M), ("boundary_spike", spike)) if tC)
    if any(m[tK] > thr[tK] for tK in JITTER):
        return "FAIL_JITTER", ";".join(tK for tK in JITTER if m[tK] > thr[tK])
    return "PASS_RAW", ""


def censusRows(produce: str, grip: pathlib.Path, refv: dict, thr: dict, boundaryStep: float) -> tuple[list[dict], list]:
    """one row per HandUMI segment of every <episode>_<side>.npz, and the steps next to lost frames"""
    tRows, tNearSteps = [], []
    for tFile in sorted(glob.glob(str(pathlib.Path(produce) / "*.npz"))):
        tEp, tSide = pathlib.Path(tFile).stem.rsplit("_", 1)
        tHu = P.load_hu(tFile)
        tBad = (tHu["lost"] > 0) | ~tHu["pose_valid"]
        tNearAll = np.zeros(len(tBad), bool)
        for tK in np.flatnonzero(tBad):
            tNearAll[max(0, tK - NEAR):tK + NEAR + 1] = True
        tZ = np.load(tFile)
        for tSi, (tS, tE) in enumerate(P.segments(tHu["seg_id"], 2)):
            tM = metrics(tHu["T_hu"][tS:tE], tNearAll[tS:tE])
            tNs = tM.pop("_near_step")
            tNearSteps.append(tNs)
            tSpike = bool((tNs > boundaryStep).any())
            tTc = np.tile(np.eye(4), (tE - tS, 1, 1))
            tTc[:, :3, 3] = tZ["cam_pos"][tS:tE]
            tTc[:, :3, :3] = Rot.from_quat(tZ["cam_quat"][tS:tE]).as_matrix()
            tMc = metrics(tTc)
            tMc.pop("_near_step", None)
            tCls, tWhy = classify(tM, tSpike, thr)
            tScore = {tK: float((refv[tK] <= tM[tK]).mean()) for tK in KEYS}         # R150 empirical CDF
            tRows.append(dict(episode=tEp, session="_".join(tEp.split("_")[:2]), side=tSide, seg=tSi, start=int(tS), end=int(tE),
                              frame_start=int(tHu["frame_idx"][tS]), frame_end=int(tHu["frame_idx"][tE - 1]),
                              cls=tCls, why=tWhy, width_available=(grip / f"{tEp}.npz").is_file(),
                              outlier_score_max_jitter=max(tScore[tK] for tK in JITTER), outlier_score_step=tScore["step_max_mm"],
                              cam_rev_rate=tMc["rev_rate"], cam_step_p95_mm=tMc["step_p95_mm"],
                              **{tK: tV for tK, tV in tM.items() if tK != "finite"}))
    return tRows, tNearSteps


def bimanualOverlap(byEpisode: dict, grip: pathlib.Path) -> tuple[int, int, int]:
    """episodes / rows where both arms have PASS_RAW frames overlapping for >= MIN_ROWS rows, and episodes with a width file"""
    tEps, tRows, tWithWidth = 0, 0, 0
    for tEp, tV in byEpisode.items():
        tL = [np.arange(tX["frame_start"], tX["frame_end"] + 1, FRAME_STEP) for tX in tV["left"] if tX["cls"] == "PASS_RAW"]
        tR = [np.arange(tX["frame_start"], tX["frame_end"] + 1, FRAME_STEP) for tX in tV["right"] if tX["cls"] == "PASS_RAW"]
        if not (tL and tR):
            continue
        tBoth = np.intersect1d(np.concatenate(tL), np.concatenate(tR))
        tRuns = np.split(tBoth, np.flatnonzero(np.diff(tBoth) != FRAME_STEP) + 1) if len(tBoth) else []
        tGood = sum(len(tX) for tX in tRuns if len(tX) >= MIN_ROWS)
        if tGood:
            tEps += 1
            tRows += tGood
            tWithWidth += int((grip / f"{tEp}.npz").is_file())
    return tEps, tRows, tWithWidth


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description="C2 input gate / OPEN-5 acceptance benchmark (FROZEN 2026-09-23)")
    tAp.add_argument("--produce", default=str(TRACK_C / "data/produce"),
                     help="dir of <episode>_<side>.npz with frame_idx, cam_pos, cam_quat (xyzw), seg_id, lost")
    tAp.add_argument("--tag", default="raw", help="output subdir under reports/c2 (raw | repaired_<name>)")
    tAp.add_argument("--grip", default=str(HOME / ".claude/jobs/e649aef3/tmp/handumi_grip_corrected"))
    return tAp


def printSummary(rows: list[dict], byEpisode: dict, bimanual: tuple[int, int, int], count, rowsByClass, why) -> None:
    tN = len(rows)
    print(f"\nHandUMI segments {tN}  rows {sum(tR['rows'] for tR in rows)}")
    for tCls in CLASSES:
        print(f"   {tCls:<12} {count[tCls]:>5} ({count[tCls]/tN*100:5.1f} % of segments, "
              f"{rowsByClass[tCls]/sum(rowsByClass.values())*100:5.1f} % of rows)")
    print("   reasons (a segment can have several):", dict(why))
    tEpAny = sum(1 for tV in byEpisode.values() if any(tX["cls"] == "PASS_RAW" for tS in tV.values() for tX in tS))
    tEpL = sum(1 for tV in byEpisode.values() if any(tX["cls"] == "PASS_RAW" for tX in tV["left"]))
    tEpR = sum(1 for tV in byEpisode.values() if any(tX["cls"] == "PASS_RAW" for tX in tV["right"]))
    print(f"\nepisodes {len(byEpisode)}: >=1 PASS_RAW segment {tEpAny} | left {tEpL} | right {tEpR} | "
          f"bimanual (both arms PASS_RAW overlapping >= {MIN_ROWS} rows) {bimanual[0]} episodes / {bimanual[1]} rows, "
          f"of which with width file {bimanual[2]}")
    for tSide in ("left", "right"):
        tSel = [tR for tR in rows if tR["side"] == tSide]
        tC = collections.Counter(tR["cls"] for tR in tSel)
        print(f"   {tSide:<5} segments {len(tSel):>5}  PASS_RAW {tC['PASS_RAW']/len(tSel)*100:5.1f} %  "
              f"JITTER {tC['FAIL_JITTER']/len(tSel)*100:5.1f} %  POSE {tC['FAIL_POSE']/len(tSel)*100:5.1f} %")


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = TRACK_C / "reports" / "c2" / tArgs.tag
    tOut.mkdir(parents=True, exist_ok=True)
    tGrip = pathlib.Path(tArgs.grip)

    # ------------------------------------------------------------------ R150 reference
    tRef = referenceMetrics(TRACK_C / "data/r150_q_tcp.npz")
    tRefv = {tK: np.array([tM[tK] for tM in tRef]) for tK in KEYS}
    tRefStat = {tK: dict(median=float(np.median(tV)), p95=float(np.percentile(tV, P95)), p99=float(np.percentile(tV, P99)))
                for tK, tV in tRefv.items()}
    tThr = {tK: tRefStat[tK]["p99"] for tK in JITTER}
    tBoundaryStep = tRefStat["step_max_mm"]["p99"]
    print(f"R150 reference: {len(tRef)} windows of {WIN} rows")
    for tKey, tV in tRefStat.items():
        print(f"   {tKey:<18} median {tV['median']:8.3f}  p95 {tV['p95']:8.3f}  p99 {tV['p99']:8.3f}")

    # ------------------------------------------------------------------ HandUMI census and aggregates
    tRows, tNearSteps = censusRows(tArgs.produce, tGrip, tRefv, tThr, tBoundaryStep)
    tCount = collections.Counter(tR["cls"] for tR in tRows)
    tRowsByClass = collections.Counter()
    for tRow in tRows:
        tRowsByClass.update({tRow["cls"]: tRow["rows"]})
    tWhy = collections.Counter(tW for tR in tRows if tR["why"] for tW in tR["why"].split(";"))
    tByEp = collections.defaultdict(lambda: {"left": [], "right": []})
    for tRow in tRows:
        tByEp[tRow["episode"]][tRow["side"]].append(tRow)
    tBimanual = bimanualOverlap(tByEp, tGrip)
    printSummary(tRows, tByEp, tBimanual, tCount, tRowsByClass, tWhy)
    tSess = collections.defaultdict(list)
    for tRow in tRows:
        tSess[tRow["session"]].append(tRow["cls"])
    print("\nsession pass rate (segments):")
    for tS in sorted(tSess):
        print(f"   {tS}  {len(tSess[tS]):>4}  PASS_RAW {tSess[tS].count('PASS_RAW')/len(tSess[tS])*100:5.1f} %")

    # diagnostics: is jitter a camera-rotation lever-arm effect? is it concentrated near lost frames?
    tRvT = np.array([tR["rev_rate"] for tR in tRows])
    tRvC = np.array([tR["cam_rev_rate"] for tR in tRows])
    tNs = np.concatenate(tNearSteps)
    print(f"\ndiag: rev_rate median TCP {np.median(tRvT):.3f} vs CAMERA {np.median(tRvC):.3f} "
          f"(corr {np.corrcoef(tRvT, tRvC)[0,1]:.2f}); steps near lost frames: {len(tNs)}, "
          f"p95 {np.percentile(tNs,95) if len(tNs) else 0:.1f} mm")
    tSens = {str(tT): float(np.mean([tR["rev_rate"] <= tT for tR in tRows])) for tT in SENSITIVITY_REV}
    print("sensitivity (NOT the rule): share of segments with rev_rate <= t:", {tK: round(tV, 3) for tK, tV in tSens.items()})

    with open(tOut / "census_segments.csv", "w", newline="") as tFh:
        tWriter = csv.DictWriter(tFh, fieldnames=list(tRows[0].keys()))
        tWriter.writeheader()
        tWriter.writerows(tRows)
    tEpAny = sum(1 for tV in tByEp.values() if any(tX["cls"] == "PASS_RAW" for tS in tV.values() for tX in tS))
    tEpL = sum(1 for tV in tByEp.values() if any(tX["cls"] == "PASS_RAW" for tX in tV["left"]))
    tEpR = sum(1 for tV in tByEp.values() if any(tX["cls"] == "PASS_RAW" for tX in tV["right"]))
    (tOut / "census.json").write_text(json.dumps(dict(
        schema="track_c_c2_input_gate/v1", date="2026-09-23", tag=tArgs.tag, produce=tArgs.produce,
        rule=__doc__.split("Classification")[1].strip(),
        r150_reference=dict(windows=len(tRef), window_rows=WIN, stats=tRefStat),
        thresholds=dict(tThr, boundary_step_mm=tBoundaryStep, hard_dp_mm=HARD_DP, hard_dth_deg=HARD_DTH, max_drift_m=MAX_DRIFT_M),
        counts=dict(segments=len(tRows), by_class=dict(tCount), rows_by_class=dict(tRowsByClass), reasons=dict(tWhy),
                    moving_segments=sum(tR["rev_pairs"] >= MOVING_MIN_PAIRS for tR in tRows),
                    moving_pass_raw=sum(tR["rev_pairs"] >= MOVING_MIN_PAIRS and tR["cls"] == "PASS_RAW" for tR in tRows),
                    static_pass_raw=sum(tR["rev_pairs"] == 0 and tR["cls"] == "PASS_RAW" for tR in tRows)),
        episodes=dict(total=len(tByEp), any_pass=tEpAny, left_pass=tEpL, right_pass=tEpR, bimanual_pass=tBimanual[0],
                      bimanual_rows=tBimanual[1], bimanual_pass_with_width=tBimanual[2]),
        session_pass={tS: dict(segments=len(tV), pass_raw=tV.count("PASS_RAW")) for tS, tV in tSess.items()},
        diag=dict(rev_rate_median_tcp=float(np.median(tRvT)), rev_rate_median_camera=float(np.median(tRvC)),
                  corr=float(np.corrcoef(tRvT, tRvC)[0, 1])), sensitivity_rev_rate=tSens), indent=1))
    np.savez(tOut / "hist_data.npz", hu_rev=tRvT, hu_step_p95=np.array([tR["step_p95_mm"] for tR in tRows]),
             hu_rot_rev=np.array([tR["rot_rev_rate"] for tR in tRows]),
             ref_rev=tRefv["rev_rate"], ref_step_p95=tRefv["step_p95_mm"], ref_rot_rev=tRefv["rot_rev_rate"])
    print(f"-> {tOut}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
