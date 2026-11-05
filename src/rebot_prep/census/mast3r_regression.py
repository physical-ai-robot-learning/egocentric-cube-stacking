"""[2026-09-23] C8 Phase-1 regression: reference = 104 episodes / 208 sides, MASt3R-SLAM ss1 on the RTX 5080 (Gen-1 census
stage; was c8_regress_ref208.py).

In:  ~/c8/runs_m3/<tag>/ss1.{csv,json}, <tag>.log, <tag>.vram (env C8_RUNS; synced from the NFS), ~/c8/grip/ (grip_census),
     ~/c8/c8_domain_manifest.json, ~/c8/export[_early]/<tag>/, ~/c8/ref104ep_208sides.txt (default tag list),
     ~/umi_bridge/trackA_mast3r_pose_v1/{cp6_scale.py, qa_compare.py, data/} (frozen texts, exec'd),
     ~/umi_bridge/track_c/data/produce/<tag>.npz (legacy ORB poses)
Out: the staging tree ~/c8/stage (env C8_STAGE), ~/c8/regress_ref208.json and a printed per-side line + summary
Usage: python -m rebot_prep.census.mast3r_regression [tag ...]          (env: ~/handumi-sw/.venv/bin/python)

Not numeric equality with the legacy ORB poses -- trajectory-quality criteria, fixed BEFORE looking at the results:
  R1 completeness   208/208 ss1.csv, rc 0, no MEM_STOP, csv rows == raw_video frames, frame_idx 0..n-1, and
                    n_video == n_CORI + offset (offset = grip_census frame mapping: CORI starts `offset` stream frames late)
  index mapping     video_to_cori_index_v1: cori_idx = video_frame - offset. MASt3R csv frame_idx is the VIDEO frame; the
                    IMU (CORI/GYRO/ACCL) and the gripper are CORI-indexed. Benchmark valB/valC had offset 0, C8 has 1..4.
                    cp6_scale.imu_vi reads frame_idx as the CORI index, so it gets a staged re-indexed copy
                    (stage/runs/<tag>/ss1.csv, frames < offset dropped, nothing else changed). The gyro check is also
                    run WITHOUT the mapping (diagnostic): its best lag must then equal the offset.
  validity rule     nonfinite_pose_invalid_v1: a row with any non-finite x/y/z/q (MASt3R-SLAM can diverge to NaN while
                    still reporting TRACKING / is_lost=false; qa_compare.load lets NaN through) counts as invalid. Applied
                    to a staged copy (stage/runs_video, then re-indexed to stage/runs); counted per side (nonfinite_pose_rows,
                    nonfinite_pose_side).
  R2 width          208/208 grip width length == frames, and identical to legacy handumi_grip_corrected (grip_census)
  R3 coverage       MASt3R valid-frame fraction median >= 0.95, and >= the legacy ORB coverage on >= 90 % of sides
  R4 continuity     reversal rate (qa_compare.tcp_metrics, 15 Hz, > 5 mm steps) median <= ORB median; zigzag bursts
                    and step p99 reported (IMU-VI metric scale)
  R5 gyro           pose-derived body angular velocity vs the wrist gyro (independent sensor, rotation is scale-free):
                    vector cosine p50 >= 0.9 on >= 95 % of sides, best lag within +-1 frame on >= 95 %, no axis with
                    negative correlation (catches frame / sign convention flips)
  R6 IMU-VI scale   per side, cp6_scale.imu_vi EXACTLY (exec of the frozen source, W 15, no bias); newly registered,
                    no GT here; |g| in [8.81, 10.81] on >= 90 % of sides. Reported (not a PASS criterion, added
                    2026-09-24): sides failing imu_vi_degenerate_guard_v1 (s outside [0.0305, 9.18], benchmark GT x0.1 / x10)
Functions are exec'd from the frozen sources (cp6_scale.py up to the FIT/EVAL section, qa_compare.py up to the run loop)
on a staging dir of symlinks (~/c8/stage), so nothing is re-implemented.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ego_paths import DATA_ROOT, UMI_BRIDGE

C8 = DATA_ROOT
TA = UMI_BRIDGE / "trackA_mast3r_pose_v1"
RUNS = pathlib.Path(os.environ.get("C8_RUNS", C8 / "runs_m3"))
PRODUCE = UMI_BRIDGE / "track_c/data/produce"
STAGE = pathlib.Path(os.environ.get("C8_STAGE", C8 / "stage"))
W_VI, BIAS_VI = 15, False                 # frozen C-P6 choice (valB FIT)
CP6_END, QA_END = "tags = [", "res = {}"  # the frozen texts are exec'd up to these lines
POSE_COLS = ("x", "y", "z", "q_x", "q_y", "q_z", "q_w")
LOST_QUAT_NORM = 0.5                      # legacy ORB lost frames carry a zero quaternion
GYRO_LAGS = range(-5, 6)                  # frames
GYRO_MIN_SAMPLES = 30
MOVING_RAD_S = 0.3                        # gyro norm above which a frame counts as moving
EPS = 1e-12
G_MS2, G_TOL = 9.81, 1.0
DEGENERATE_S = (0.0305, 9.18)             # imu_vi_degenerate_guard_v1 (reported only)
VRAM_14GB_MIB = 14336
PASS_COV_P50, PASS_GE_ORB, PASS_FRAC = 0.95, 0.90, 0.95
PASS_COS, PASS_G_FRAC = 0.9, 0.90
MS_PER_S = 1e3
NS_PER_MS = 1e6


def exportDir(manifest: dict, tag: str) -> pathlib.Path:
    """rest490 includes early sides"""
    return C8 / ("export" if manifest[tag]["domain"] == "main_calibrated" else "export_early") / tag


def stageRuns(tags: list[str], manifest: dict, grip: dict) -> dict:
    """the layout cp6_scale / qa_compare expect, as symlinks (no copies, no edits) + the NaN-invalidated, re-indexed csv
    copies -> {tag: nonfinite rows}"""
    (STAGE / "qa").mkdir(parents=True, exist_ok=True)
    for tSrc, tDst in ((TA / "data/qa/qa_all.json", STAGE / "qa/qa_all.json"),
                       (TA / "data/handumi_camera_tcp_v2.yaml", STAGE / "handumi_camera_tcp_v2.yaml")):
        if not tDst.exists():
            tDst.symlink_to(tSrc)
    tNonFinite = {}
    for tTag in tags:
        tA, tB = tTag.split("_", 1)
        tDst = STAGE / "val" / tA / tB
        tDst.parent.mkdir(parents=True, exist_ok=True)
        if not tDst.exists():
            tDst.symlink_to(exportDir(manifest, tTag))
        tSrc = RUNS / tTag / "ss1.csv"
        tDst = STAGE / "runs" / tTag / "ss1.csv"
        tDst.parent.mkdir(parents=True, exist_ok=True)
        tDstVideo = STAGE / "runs_video" / tTag / "ss1.csv"
        tDstVideo.parent.mkdir(parents=True, exist_ok=True)
        if tSrc.is_file():
            tOff = grip[tTag]["offset"]
            tLines = open(tSrc).read().splitlines()
            tHdr = tLines[0].split(",")
            tFi, tIl = tHdr.index("frame_idx"), tHdr.index("is_lost")
            tPoseCols = [tHdr.index(tK) for tK in POSE_COLS]
            tOutVideo, tOut = [tLines[0]], [tLines[0]]
            tNonFinite[tTag] = 0
            for tLine in tLines[1:]:
                tCells = tLine.split(",")
                if not all(np.isfinite(float(tCells[tK])) for tK in tPoseCols) and tCells[tIl] == "false":
                    tCells[tIl] = "true"                                                      # nonfinite_pose_invalid_v1
                    tNonFinite[tTag] += 1
                tOutVideo.append(",".join(tCells))
                tJ = int(tCells[tFi]) - tOff                                                  # video_to_cori_index_v1
                if tJ >= 0:
                    tCells[tFi] = str(tJ)
                    tOut.append(",".join(tCells))
            tDstVideo.write_text("\n".join(tOutVideo) + "\n")
            tDst.write_text("\n".join(tOut) + "\n")
    return tNonFinite


def frozenNamespaces() -> tuple[dict, dict]:
    """(cp6, qa): the frozen cp6_scale.py and qa_compare.py texts exec'd on the staging dir"""
    os.environ["M3_BASE"] = str(STAGE)
    os.environ["M3_VARIANT"] = "ss1"
    tCp6, tQa = {}, {}
    tText = open(TA / "cp6_scale.py").read()
    exec(tText[:tText.index(CP6_END)], tCp6)
    tText = open(TA / "qa_compare.py").read()
    exec(tText[:tText.index(QA_END)], tQa)
    return tCp6, tQa


def orbPoses(tag: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """legacy ORB camera poses of a side on the CORI index (was orb)"""
    tZ = np.load(PRODUCE / f"{tag}.npz")
    tT = np.tile(np.eye(4), (n, 1, 1))
    tOk = np.zeros(n, bool)
    tValid = np.linalg.norm(tZ["cam_quat"], axis=1) > LOST_QUAT_NORM
    for tI, tF in enumerate(tZ["frame_idx"]):
        if tValid[tI] and 0 <= tF < n:
            tT[tF, :3, :3] = Rot.from_quat(tZ["cam_quat"][tI]).as_matrix()
            tT[tF, :3, 3] = tZ["cam_pos"][tI]
            tOk[tF] = True
    return tT, tOk


def gyroCheck(cp6: dict, T: np.ndarray, ok: np.ndarray, sideDir: pathlib.Path) -> dict | None:
    """pose-derived body angular velocity vs the wrist gyro (was gyro_check). T, ok indexed by CORI frame (pass the unmapped
    video-indexed arrays only for the diagnostic)."""
    tStreams = json.load(open(sideDir / "imu_data.json"))["1"]["streams"]
    tTc = np.array([tS["cts"] for tS in tStreams["CORI"]["samples"]]) / MS_PER_S
    tTi = np.array([tS["cts"] for tS in tStreams["GYRO"]["samples"]]) / MS_PER_S
    tW = np.array([tS["value"] for tS in tStreams["GYRO"]["samples"]], float)
    tRbc = cp6["T_b_c"](sideDir / "orbslam_setting.yaml")[:3, :3]
    tN = min(len(tTc), len(T))
    tWp = np.full((tN - 1, 3), np.nan)
    tWg = np.full((tN - 1, 3), np.nan)
    for tI in range(tN - 1):
        tSel = (tTi > tTc[tI]) & (tTi <= tTc[tI + 1])
        if tSel.any():
            tWg[tI] = tW[tSel].mean(0)
        if ok[tI] and ok[tI + 1]:
            tWp[tI] = tRbc @ (Rot.from_matrix(T[tI, :3, :3].T @ T[tI + 1, :3, :3]).as_rotvec() / (tTc[tI + 1] - tTc[tI]))

    def tCorrAt(tLag):
        tA, tB = (tWp[tLag:], tWg[:len(tWg) - tLag]) if tLag >= 0 else (tWp[:tLag], tWg[-tLag:])
        tM = np.isfinite(tA).all(1) & np.isfinite(tB).all(1)
        if tM.sum() < GYRO_MIN_SAMPLES:
            return None
        return np.array([np.corrcoef(tA[tM, tK], tB[tM, tK])[0, 1] for tK in range(3)]), tA[tM], tB[tM]

    tLags = {tLag: tCorrAt(tLag) for tLag in GYRO_LAGS}
    tLags = {tLag: tV for tLag, tV in tLags.items() if tV is not None}
    if 0 not in tLags:
        return None
    tBest = max(tLags, key=lambda tLag: np.nanmean(tLags[tLag][0]))
    tC0, tA, tB = tLags[0]
    tMoving = np.linalg.norm(tB, axis=1) > MOVING_RAD_S
    tNa, tNb = np.linalg.norm(tA[tMoving], axis=1), np.linalg.norm(tB[tMoving], axis=1)
    tCos = np.einsum("ij,ij->i", tA[tMoving], tB[tMoving]) / (tNa * tNb + EPS)
    return dict(axis_corr=tC0.round(4).tolist(), best_lag=int(tBest),
                vec_cos_p50=float(np.median(tCos)) if tMoving.any() else float("nan"),
                gain_p50=float(np.median(tNa / tNb)) if tMoving.any() else float("nan"), n_moving=int(tMoving.sum()))


def sideRecord(tag: str, manifest: dict, grip: dict, nonFinite: dict, cp6: dict, qa: dict) -> dict:
    """the regression record of one side; prints its line"""
    tExport = exportDir(manifest, tag)
    tRec = dict()
    tN = len(json.load(open(tExport / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
    tRec["n_cori"] = tN
    tOff = grip[tag]["offset"]
    tRec["offset"] = tOff
    tNv = int(cv2.VideoCapture(str(tExport / "raw_video.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    tRec["n_video"] = tNv
    tCsv = STAGE / "runs_video" / tag / "ss1.csv"
    tVramFile, tLogFile = RUNS / f"{tag}.vram", RUNS / f"{tag}.log"
    tTok = tVramFile.read_text().split() if tVramFile.is_file() else []
    tRec["vram"] = dict(zip(tTok[::2], map(int, tTok[1::2]))) if tTok else None
    tRec["present"] = (RUNS / tag / "ss1.csv").is_file()
    tRec["nonfinite_pose_rows"] = nonFinite.get(tag, 0)
    tRec["nonfinite_pose_side"] = tRec["nonfinite_pose_rows"] > 0
    if not tRec["present"]:
        return tRec
    tLog = tLogFile.read_text() if tLogFile.is_file() else ""
    tRec["reloc_attempts"] = tLog.count("RELOCALIZING")
    tRec["reloc_failed"] = tLog.count("Failed to relocalize")
    tT, tOk, tStep = qa["load"](tCsv)
    tRows = sum(1 for _ in open(tCsv)) - 1
    tRec["rows_match"] = tRows == tNv == len(tOk) and tStep == 1 and tNv == tN + tOff
    tTv, tOkv = tT, tOk
    tT, tOk = tT[tOff:tOff + tN], tOk[tOff:tOff + tN]                       # video_to_cori_index_v1
    tGrip = np.load(C8 / "grip" / f"{tag}.npz")
    tRec["width_len_match"] = len(tGrip["width_m"]) == tN and np.array_equal(tGrip["video_frame"], np.arange(tN) + tOff)
    tRec["width_legacy_identical"] = bool(grip[tag].get("legacy_width_identical"))
    tRec["coverage"] = float(tOk.mean())
    tRunJson = RUNS / tag / "ss1.json"
    tRec["kf"] = int(json.load(open(tRunJson)).get("n_keyframes", -1)) if tRunJson.is_file() else -1
    tVi = cp6["imu_vi"](tag, W_VI, BIAS_VI)
    tRec["imu_vi"] = tVi
    tS = tVi["s"] if tVi else float("nan")
    tRec["tcp"] = qa["tcp_metrics"](tT, tOk, tS, 1)
    tRec["gyro"] = gyroCheck(cp6, tT, tOk, tExport)
    tRec["gyro_unmapped"] = gyroCheck(cp6, tTv[:tN], tOkv[:tN], tExport)
    if (PRODUCE / f"{tag}.npz").is_file():
        tTo, tOko = orbPoses(tag, tN)
        tRec["orb"] = dict(coverage=float(tOko.mean()), tcp=qa["tcp_metrics"](tTo, tOko, 1.0, 1),
                           gyro=gyroCheck(cp6, tTo, tOko, tExport))
    tGyro = tRec["gyro"]
    print(tag, f"cov {tRec['coverage']:.2f}", f"s {tS:.4f} |g| {tVi['g_norm']:.2f}" if tVi else "no-VI",
          f"rev {tRec['tcp'].get('reversal_rate', float('nan')):.3f}",
          f"gyro cos {tGyro['vec_cos_p50']:.3f} lag {tGyro['best_lag']} "
          f"(unmapped lag {tRec['gyro_unmapped']['best_lag']}, off {tOff})" if tGyro else "no-gyro",
          f"| ORB cov {tRec['orb']['coverage']:.2f} rev {tRec['orb']['tcp'].get('reversal_rate', float('nan')):.3f}"
          if "orb" in tRec else "| no ORB", flush=True)
    return tRec


def summary(rec: dict, tags: list[str]) -> dict:
    """the R1..R6 criteria and the diagnostics over all present sides"""
    tP = [tR for tR in rec.values() if tR["present"]]
    tHasOrb = [tR for tR in tP if "orb" in tR]
    tRes = collections.OrderedDict()
    tRes["R1"] = dict(present=len(tP), rows_match=sum(tR["rows_match"] for tR in tP),
                      rc0=sum(1 for tR in tP if tR["vram"] and tR["vram"]["rc"] == 0))
    tRes["R1"]["PASS"] = tRes["R1"]["present"] == tRes["R1"]["rows_match"] == tRes["R1"]["rc0"] == len(tags)
    tRes["R2"] = dict(len_match=sum(tR["width_len_match"] for tR in tP),
                      legacy_identical=sum(tR["width_legacy_identical"] for tR in tP))
    tRes["R2"]["PASS"] = tRes["R2"]["len_match"] == tRes["R2"]["legacy_identical"] == len(tags)
    tCov = np.array([tR["coverage"] for tR in tP])
    tGe = [tR["coverage"] >= tR["orb"]["coverage"] for tR in tHasOrb]
    tRes["R3"] = dict(cov_p10=float(np.percentile(tCov, 10)), cov_p50=float(np.median(tCov)), n_below_0_9=int((tCov < 0.9).sum()),
                      orb_cov_p50=float(np.median([tR["orb"]["coverage"] for tR in tHasOrb])) if tHasOrb else None,
                      ge_orb_frac=float(np.mean(tGe)) if tGe else None, n_with_orb=len(tHasOrb))
    tRes["R3"]["PASS"] = tRes["R3"]["cov_p50"] >= PASS_COV_P50 and (tRes["R3"]["ge_orb_frac"] or 0) >= PASS_GE_ORB
    tRev = [tR["tcp"]["reversal_rate"] for tR in tP if "reversal_rate" in tR["tcp"]]
    tRevOrb = [tR["orb"]["tcp"]["reversal_rate"] for tR in tHasOrb if "reversal_rate" in tR["orb"]["tcp"]]
    tRes["R4"] = dict(rev_p50=float(np.median(tRev)), orb_rev_p50=float(np.median(tRevOrb)) if tRevOrb else None,
                      zigzag_total=int(sum(tR["tcp"].get("zigzag_bursts", 0) for tR in tP)),
                      orb_zigzag_total=int(sum(tR["orb"]["tcp"].get("zigzag_bursts", 0) for tR in tHasOrb)),
                      step_p99_mm_p50=float(np.median([tR["tcp"]["step_p99_mm"] for tR in tP if "step_p99_mm" in tR["tcp"]])))
    tRes["R4"]["PASS"] = tRevOrb != [] and tRes["R4"]["rev_p50"] <= tRes["R4"]["orb_rev_p50"]
    tGyros = [tR["gyro"] for tR in tP if tR["gyro"]]
    tRes["R5"] = dict(n=len(tGyros), cos_ge_0_9_frac=float(np.mean([tG["vec_cos_p50"] >= PASS_COS for tG in tGyros])),
                      lag_within1_frac=float(np.mean([abs(tG["best_lag"]) <= 1 for tG in tGyros])),
                      n_neg_axis=int(sum(min(tG["axis_corr"]) < 0 for tG in tGyros)),
                      cos_p50=float(np.nanmedian([tG["vec_cos_p50"] for tG in tGyros])),
                      gain_p50=float(np.nanmedian([tG["gain_p50"] for tG in tGyros])),
                      orb_cos_p50=float(np.nanmedian([tR["orb"]["gyro"]["vec_cos_p50"] for tR in tHasOrb if tR["orb"]["gyro"]]))
                      if tHasOrb else None)
    tRes["R5"]["PASS"] = (tRes["R5"]["cos_ge_0_9_frac"] >= PASS_FRAC and tRes["R5"]["lag_within1_frac"] >= PASS_FRAC
                          and tRes["R5"]["n_neg_axis"] == 0)
    tVis = [tR["imu_vi"] for tR in tP if tR["imu_vi"]]
    tGn = np.array([tV["g_norm"] for tV in tVis])
    tRes["R6"] = dict(n=len(tVis), g_in_band_frac=float(np.mean(np.abs(tGn - G_MS2) <= G_TOL)), g_p50=float(np.median(tGn)),
                      s_p10_p50_p90=np.percentile([tV["s"] for tV in tVis], [10, 50, 90]).round(4).tolist(),
                      degenerate_scale_sides={tT: tR["imu_vi"]["s"] for tT, tR in rec.items() if tR.get("imu_vi")
                                              and not (DEGENERATE_S[0] <= tR["imu_vi"]["s"] <= DEGENERATE_S[1])})
    tRes["R6"]["PASS"] = tRes["R6"]["g_in_band_frac"] >= PASS_G_FRAC and len(tVis) == len(tags)
    tVr = [tR["vram"].get("tree_peak_MiB", tR["vram"].get("proc_peak_MiB")) for tR in tP if tR["vram"]]
    tRes["vram"] = dict(tree_peak_max_MiB=max(tVr), tree_peak_p50=float(np.median(tVr)),
                        n_ge_14GB=int(sum(tV >= VRAM_14GB_MIB for tV in tVr)),
                        gpu_peak_max_MiB=max(tR["vram"].get("gpu_peak_MiB", 0) for tR in tP if tR["vram"]),
                        sec_p50=float(np.median([tR["vram"]["sec"] for tR in tP if tR["vram"]])))
    tRes["index_mapping"] = dict(name="video_to_cori_index_v1", offsets=dict(collections.Counter(tR["offset"] for tR in tP)),
                                 unmapped_lag_equals_offset=sum(1 for tR in tP if tR.get("gyro_unmapped")
                                                                and tR["gyro_unmapped"]["best_lag"] == tR["offset"]))
    tRes["nonfinite"] = dict(sides=sum(tR.get("nonfinite_pose_side", False) for tR in tP),
                             rows=sum(tR.get("nonfinite_pose_rows", 0) for tR in tP),
                             which={tT: tR["nonfinite_pose_rows"] for tT, tR in rec.items() if tR.get("nonfinite_pose_rows", 0)})
    tRes["reloc"] = dict(sides_with_reloc=sum(tR["reloc_attempts"] > 0 for tR in tP),
                         sides_reloc_failed=sum(tR["reloc_failed"] > 0 for tR in tP))
    tRes["ALL_PASS"] = all(tRes[tK]["PASS"] for tK in ("R1", "R2", "R3", "R4", "R5", "R6"))
    return tRes


def main(argv: list[str] | None = None) -> int:
    tArgs = sys.argv[1:] if argv is None else argv
    tTags = tArgs or open(C8 / "ref104ep_208sides.txt").read().split()
    tGrip = json.load(open(C8 / "grip/grip_census.json"))["sides"]
    tManifest = json.load(open(C8 / "c8_domain_manifest.json"))
    tNonFinite = stageRuns(tTags, tManifest, tGrip)
    tCp6, tQa = frozenNamespaces()
    tRec = {}
    for tTag in tTags:
        tRec[tTag] = sideRecord(tTag, tManifest, tGrip, tNonFinite, tCp6, tQa)
    tRes = summary(tRec, tTags)
    json.dump(dict(summary=tRes, sides=tRec), open(C8 / "regress_ref208.json", "w"), indent=1, default=float)
    print("\n" + "\n".join(f"{tK}: {tV}" for tK, tV in tRes.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
