"""[2026-09-23] C8 census, GPU-free stages, per EPISODE (349), main294 vs early55 (Gen-1 census stage; was c8_census_pre.py).

In:  ~/c8/c8_domain_manifest.json, ~/c8/grip/grip_census.json (grip_census), ~/c8/export[_early]/<tag>/imu_data.json,
     the raw Hpilot episode folders (sensors.mcap, the three videos, episode_meta.json)
Out: ~/c8/census_pre.json (per-episode record) and the printed funnel
Usage: python -m rebot_prep.census.sensor_sync_census

Mutually exclusive first-fail reasons:
   1 sensor_complete        sensors.mcap + both wrist videos + IMU both sides + head video
   2 calibration_applicable main = session after the 2026-09-16 11:51 calibration files; early = flagged, NOT rejected here
   3 gripper_valid          grip_census produced width for both sides (frame mapping verified)
   4 sync_pass              left-master pairing with right wrist + head, |dt| < 20 ms, AND at least one 65-frame run
                            (the C7/V3 segment length) exists
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np
from mcap.reader import make_reader

from ego_paths import DATA_ROOT, EGO_COLLECTOR_HOME

C8 = DATA_ROOT
RAW = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/Hpilot"
SEG = 65                       # rows per segment (C7/V3)
SYNC_MS = 20.0                 # left-master pairing tolerance
NS_PER_MS = 1e6
P95 = 95
STREAMS = ("left_wrist", "right_wrist", "head")
REQUIRED_FILES = ("sensors.mcap", "left_wrist.mp4", "right_wrist.mp4", "head.mp4")
STAGES = ("sensor_complete", "gripper_valid", "sync_pass")


def frameMeta(episodeDir) -> dict:
    """{stream: sorted [(video_frame, capture_ns)]} of the three cameras (was frame_meta)"""
    tFrames = {tS: [] for tS in STREAMS}
    with open(episodeDir / "sensors.mcap", "rb") as tFh:
        for _, tChannel, tMsg in make_reader(tFh).iter_messages():
            if tChannel.topic.endswith("/frame_meta"):
                tStream = tChannel.topic.split("/")[1]
                if tStream in tFrames:
                    tData = json.loads(tMsg.data)
                    tFrames[tStream].append((int(tData["video_frame"]), int(tData["capture_ns"])))
    return {tK: sorted(tV) for tK, tV in tFrames.items()}


def nearestTo(cap: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """index into cap of the nearest capture time for every ref time (ties -> the later one)"""
    tJ = np.clip(np.searchsorted(cap, ref), 1, len(cap) - 1)
    return np.where(np.abs(cap[tJ - 1] - ref) < np.abs(cap[tJ] - ref), tJ - 1, tJ)


def coriCount(exportDir) -> int:
    return len(json.load(open(exportDir / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])


def episodeRecord(episode: str, domain: str, grip: dict) -> dict:
    tSess, tEp = episode.rsplit("_", 1)
    tDir = RAW / f"Hpilot_{tSess}" / f"episode_{tEp}"
    tRec = dict(domain=domain)
    tMeta = json.load(open(tDir / "episode_meta.json")) if (tDir / "episode_meta.json").is_file() else {}
    tRec["sensor_complete"] = all((tDir / tF).is_file() for tF in REQUIRED_FILES) and bool(tMeta.get("imu_present"))
    tRec["calibration_applicable"] = domain == "main_calibrated"
    tRec["early_calibration_domain"] = domain != "main_calibrated"
    tRec["gripper_valid"] = all(grip.get(f"{episode}_{tS}", {}).get("ok") for tS in ("left", "right"))
    if tRec["sensor_complete"]:
        tFm = frameMeta(tDir)
        tOffL, tOffR = grip[f"{episode}_left"].get("offset", 0), grip[f"{episode}_right"].get("offset", 0)
        tExport = C8 / ("export" if domain == "main_calibrated" else "export_early")
        tNL, tNR = coriCount(tExport / f"{episode}_left"), coriCount(tExport / f"{episode}_right")
        tCapL = np.array([tC for _, tC in tFm["left_wrist"][tOffL:tOffL + tNL]], float)
        tCapR = np.array([tC for _, tC in tFm["right_wrist"][tOffR:tOffR + tNR]], float)
        tCapH = np.array([tC for _, tC in tFm["head"]], float)
        tDtR = np.abs(tCapR[nearestTo(tCapR, tCapL)] - tCapL) / NS_PER_MS
        tDtH = np.abs(tCapH[nearestTo(tCapH, tCapL)] - tCapL) / NS_PER_MS
        tGood = (tDtR < SYNC_MS) & (tDtH < SYNC_MS)
        tIdx = np.flatnonzero(tGood)
        tRuns = [len(tX) for tX in np.split(tIdx, np.flatnonzero(np.diff(tIdx) != 1) + 1)] if len(tIdx) else []
        tRec.update(sync_good_frac=float(tGood.mean()), sync_longest_run=int(max(tRuns) if tRuns else 0),
                    sync_segments_possible=int(sum(tN // SEG for tN in tRuns)), dtR_p95=float(np.percentile(tDtR, P95)),
                    dtH_p95=float(np.percentile(tDtH, P95)))
        tRec["sync_pass"] = tRec["sync_segments_possible"] > 0
    else:
        tRec["sync_pass"] = False
    tRec["first_fail"] = next((tS for tS in STAGES if not tRec[tS]), None)
    return tRec


def printFunnel(records: dict) -> None:
    for tDomain in ("main_calibrated", "early_precalibration", "all"):
        tSel = [tR for tR in records.values() if tDomain == "all" or tR["domain"] == tDomain]
        tFirst = collections.Counter(tR["first_fail"] for tR in tSel)
        tAlive = len(tSel)
        tLine = f"{tDomain:<22} raw {len(tSel)}"
        for tStage in STAGES:
            tAlive -= tFirst.get(tStage, 0)
            tLine += f" -> {tStage} {tAlive}"
        tSegs = sum(tR.get("sync_segments_possible", 0) for tR in tSel if tR["first_fail"] is None)
        tReasons = dict((tK, tC) for tK, tC in tFirst.items() if tK)
        print(tLine + f"   | 65-frame segments available {tSegs}   | first-fail reasons {tReasons}")


def main(argv: list[str] | None = None) -> int:
    tManifest = json.load(open(C8 / "c8_domain_manifest.json"))
    tGrip = json.load(open(C8 / "grip/grip_census.json"))["sides"]
    tEpisodes = sorted({tV["episode"]: tV["domain"] for tV in tManifest.values()}.items())
    tRecords = {tEpisode: episodeRecord(tEpisode, tDomain, tGrip) for tEpisode, tDomain in tEpisodes}
    json.dump(tRecords, open(C8 / "census_pre.json", "w"), indent=1, default=float)
    printFunnel(tRecords)
    return 0


if __name__ == "__main__":
    sys.exit(main())
