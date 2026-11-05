"""[2026-09-23] C8 gripper width for all 698 sides (349 episodes), Track A contract §16 (Gen-1 dataset stage, gripper step 3;
was c8_grip.py).

In:  ~/c8/c8_domain_manifest.json, ~/c8/export[_early]/<tag>/imu_data.json (CORI), the raw Hpilot sensors.mcap, the legacy
     handumi_grip_corrected folder (gripper_contract.json + per-episode npz) from correct_grip
Out: ~/c8/grip/<tag>.npz (val_frame, width_m, open_fraction, raw, dt_ms, video_frame), ~/c8/grip/grip_census.json, a summary
Usage: python -m rebot_prep.grip.grip_census           (env: ~/xvla-mac)

main_calibrated sessions: the STORED anchors of gripper_contract.json (never recomputed).
early_precalibration sessions: the same rule applied fresh (closed = p10 / p90 of per-episode closed extremes, open = closed
+ direction * device span 1136.6 L / 1248.2 R), flagged anchor_source = derived_early.
Frame mapping: val frame i = stream frame (N_stream - n_val + i), verified by CORI intervals == capture_ns intervals
(max |diff| < 1 ms), otherwise the side is REJECTED (census reason grip_frame_mapping).
Regression: for sides whose episode is in handumi_grip_corrected (the 104 reference), the new width must equal the legacy
width_m at the same video_frame.
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

from ego_paths import DATA_ROOT, EGO_COLLECTOR_HOME, HOME, UMI_BRIDGE, prepend_sys_path

C8 = DATA_ROOT
OUT = C8 / "grip"
RAW = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/Hpilot"
LEG = HOME / ".claude/jobs/e649aef3/tmp/handumi_grip_corrected"
SPAN = {"left": 1136.6, "right": 1248.2}           # device span in ticks (correct_grip, sessions reaching the stop)
MAPPING_TOL_MS = 1.0                               # CORI vs capture_ns interval agreement
EARLY_CLOSED_PCT = {1: 10, -1: 90}                 # per direction: percentile of the per-episode closed extremes
NS_PER_MS = 1e6
M_TO_MM = 1e3
P95 = 95


def gripReader():
    """(readEpisode, align, SIDE_STREAM): the ~/umi_bridge copy of extract_grip first (as the original working directory),
    else the repository's"""
    prepend_sys_path(str(UMI_BRIDGE))
    try:
        from extract_grip import SIDE_STREAM, align
        from extract_grip import read_episode as readEpisode
    except ImportError:  # clean checkout: the repository copy
        from .extract_grip import SIDE_STREAM, align, readEpisode
    return readEpisode, align, SIDE_STREAM


def exportDir(manifest: dict, tag: str):
    return C8 / ("export" if manifest[tag]["domain"] == "main_calibrated" else "export_early") / tag


def sideRecords(manifest: dict, reader) -> tuple[dict, dict]:
    """frame-mapped raw ticks per accepted side and the census record of every side"""
    tReadEpisode, tAlign, tSideStream = reader
    tEpisodes = sorted({tV["episode"] for tV in manifest.values()})
    tRaw, tCensus = {}, {}
    for tEpisode in tEpisodes:
        tSess, tEp = tEpisode.rsplit("_", 1)
        tRead = tReadEpisode(tSess, tEp)
        for tSide, tStream in tSideStream.items():
            tTag = f"{tEpisode}_{tSide}"
            tCori = json.load(open(exportDir(manifest, tTag) / "imu_data.json"))["1"]["streams"]["CORI"]["samples"]
            tCts = np.array([tC["cts"] for tC in tCori])
            if tRead is None:
                tCensus[tTag] = dict(ok=False, reason="no_mcap")
                continue
            tFrames = sorted(tRead[0][tStream])
            tNStream, tN = len(tFrames), len(tCts)
            tOff = tNStream - tN
            tCap = np.array([tX[1] for tX in tFrames], float)
            tErr = (float(np.abs(np.diff(tCts) - np.diff(tCap[tOff:]) / NS_PER_MS).max()) if tOff >= 0 and len(tCap[tOff:]) == tN
                    else np.inf)
            if not tErr < MAPPING_TOL_MS:
                tCensus[tTag] = dict(ok=False, reason="grip_frame_mapping", err_ms=tErr)
                continue
            tAl = tAlign(tFrames[tOff:], tRead[1][tSide])
            if tAl is None:
                tCensus[tTag] = dict(ok=False, reason="no_gripper_stream")
                continue
            tRaw[tTag] = dict(sess=tSess, side=tSide, raw=tAl["raw"].astype(float), dt=tAl["dt_ms"],
                              vf=np.array([tX[0] for tX in tFrames[tOff:]]), n=tN)
            tCensus[tTag] = dict(ok=True, offset=tOff, interval_match_ms=tErr,
                                 align_dt_ms_p95=float(np.percentile(tAl["dt_ms"], P95)))
    return tRaw, tCensus


def sessionAnchors(raw: dict, stored: dict) -> dict:
    """(session, side) -> (closed, open, direction, source): stored contract anchors, else derived from the early session"""
    tAnchors = {}
    for tSess in sorted({tV["sess"] for tV in raw.values()}):
        for tSide in ("left", "right"):
            if (tSess, tSide) in stored:
                tCl, tOp, tSign = stored[(tSess, tSide)]
                tAnchors[(tSess, tSide)] = (tCl, tOp, tSign, "stored_contract")
                continue
            tMeta = json.load(open(RAW / f"Hpilot_{tSess}" / "session_meta.json"))["devices"][f"gripper_{tSide}"]
            tSign = 1 if tMeta["ticks_open"] > tMeta["ticks_closed"] else -1
            tExtremes = [(tV["raw"].min() if tSign > 0 else tV["raw"].max()) for tV in raw.values()
                         if tV["sess"] == tSess and tV["side"] == tSide]
            tCl = float(np.percentile(tExtremes, EARLY_CLOSED_PCT[tSign]))
            tAnchors[(tSess, tSide)] = (tCl, tCl + tSign * SPAN[tSide], tSign, "derived_early")
    return tAnchors


def writeWidths(raw: dict, anchors: dict, census: dict, aperture: float) -> collections.Counter:
    """~/c8/grip/<tag>.npz per side; census gains anchor_source / width_p95_mm / legacy_width_identical -> regression counts"""
    tReg = collections.Counter()
    for tTag, tV in raw.items():
        tCl, tOp, _, tSrc = anchors[(tV["sess"], tV["side"])]
        tFrac = np.clip((tV["raw"] - tCl) / (tOp - tCl), 0, 1)
        tWidth = (tFrac * aperture).astype(np.float32)
        np.savez(OUT / f"{tTag}.npz", val_frame=np.arange(tV["n"]), width_m=tWidth, open_fraction=tFrac.astype(np.float32),
                 raw=tV["raw"], dt_ms=tV["dt"], video_frame=tV["vf"])
        census[tTag].update(anchor_source=tSrc, width_p95_mm=float(np.percentile(tWidth, P95) * M_TO_MM))
        tLegacyFile = LEG / f"{tTag.rsplit('_', 1)[0]}.npz"
        if tLegacyFile.is_file():
            tLegacy = np.load(tLegacyFile)
            tLvf, tLw = tLegacy[f"{tV['side']}_video_frame"], tLegacy[f"{tV['side']}_width_m"]
            tPos = np.searchsorted(tLvf, tV["vf"])
            tMatched = (tPos < len(tLvf)) & (tLvf[np.clip(tPos, 0, len(tLvf) - 1)] == tV["vf"])
            tSame = tMatched.all() and np.array_equal(tLw[tPos[tMatched]], tWidth)
            tReg["identical" if tSame else "different"] += 1
            census[tTag]["legacy_width_identical"] = bool(tSame)
    return tReg


def main(argv: list[str] | None = None) -> int:
    OUT.mkdir(exist_ok=True)
    tReader = gripReader()
    tContract = json.load(open(LEG / "gripper_contract.json"))
    tStored = {(tA["session"], tA["side"]): (tA["closed_raw"], tA["open_raw"], tA["direction"]) for tA in tContract["anchors"]}
    tManifest = json.load(open(C8 / "c8_domain_manifest.json"))
    tRaw, tCensus = sideRecords(tManifest, tReader)
    tAnchors = sessionAnchors(tRaw, tStored)
    tReg = writeWidths(tRaw, tAnchors, tCensus, tContract["aperture_m"])
    tAnchorJson = {f"{tK[0]}_{tK[1]}": dict(closed=tA[0], open=tA[1], direction=tA[2], source=tA[3])
                   for tK, tA in tAnchors.items()}
    json.dump(dict(anchors=tAnchorJson, sides=tCensus, regression=dict(tReg)), open(OUT / "grip_census.json", "w"), indent=1,
              default=float)
    tBad = collections.Counter(tC["reason"] for tC in tCensus.values() if not tC["ok"])
    tByDomain = collections.Counter((tManifest[tT]["domain"], tC["ok"]) for tT, tC in tCensus.items())
    print(f"sides {len(tCensus)}  ok {sum(tC['ok'] for tC in tCensus.values())}  rejects {dict(tBad)}  "
          f"by domain {dict(tByDomain)}")
    print(f"regression vs legacy handumi_grip_corrected: {dict(tReg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
