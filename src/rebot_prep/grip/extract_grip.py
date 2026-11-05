"""Per-frame grip01 for the A_bimanual episodes, aligned from sensors.mcap (Gen-1 dataset stage, gripper step 1).

In:  <EGO_COLLECTOR_HOME>/datasets/human_handumi_raw/Hpilot/Hpilot_<session>/episode_<n>/sensors.mcap, a manifest of items
Out: <out>/<episode_id>.npz with <side>_{video_frame, grip01, raw, load, dt_ms} per side, a printed QA summary
Usage: python -m rebot_prep.grip.extract_grip [--manifest <json>] [--out <dir>] [--limit N]

The gripper stream already carries `normalized` = (raw - ticks_closed) / (ticks_open - ticks_closed),
verified against session_meta on both sides, so 0 = closed and 1 = open -- the SAME direction as the
reBot stored width (video-confirmed 2026-09-22: raw 0 = jaws touching, -118 = holding a 50 mm cube,
-270 = wide open). No inversion is applied anywhere.

Gripper runs at ~95 Hz and video at 30, so each frame takes the nearest sample; the alignment error is
reported rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from mcap.reader import make_reader

from ego_paths import EGO_COLLECTOR_HOME, HOME

RAW = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/Hpilot"
SIDE_STREAM = {"left": "left_wrist", "right": "right_wrist"}
JOB_TMP = HOME / ".claude/jobs/e649aef3/tmp"          # the original Track A job folder (default inputs / outputs)
NS_PER_MS = 1e6
FRAME_MS = 1000 / 30                                   # one video frame: the QA 1 pass bound
GRIP_RANGE_TOL = 1e-6


def readEpisode(session: str, episode: str) -> tuple[dict, dict] | None:
    """sensors.mcap of one episode -> ({stream: [(video_frame, capture_ns)]}, {side: [(sample_ns, normalized, raw, load%)]}),
    None when the episode has no sensors.mcap (was read_episode)"""
    tPath = RAW / f"Hpilot_{session}" / f"episode_{episode}" / "sensors.mcap"
    if not tPath.is_file():
        return None
    tFrames = {tS: [] for tS in SIDE_STREAM.values()}
    tGrip = {tS: [] for tS in SIDE_STREAM}
    with open(tPath, "rb") as tFh:
        for _, tChannel, tMsg in make_reader(tFh).iter_messages():
            tTopic = tChannel.topic
            if tTopic.endswith("/frame_meta"):
                tStream = tTopic.split("/")[1]
                if tStream in tFrames:
                    tData = json.loads(tMsg.data)
                    tFrames[tStream].append((int(tData["video_frame"]), int(tData["capture_ns"])))
            elif tTopic.endswith("/gripper"):
                tSide = tTopic.split("/")[1]
                if tSide in tGrip:
                    tData = json.loads(tMsg.data)
                    tTelemetry = tData.get("telemetry") or {}
                    tGrip[tSide].append((int(tData["sample_ns"]), float(tData["normalized"]), int(tData["raw_position"]),
                                         float(tTelemetry.get("load_percent") or 0.0)))
    return tFrames, tGrip


def align(frames, grip) -> dict | None:
    """nearest-sample lookup; returns grip01 per video_frame plus the time error in ms"""
    tFrames = sorted(frames)
    if not tFrames or not grip:
        return None
    tG = np.array(sorted(grip), dtype=np.float64)
    tGt, tGv, tGraw, tGload = tG[:, 0], tG[:, 1], tG[:, 2], tG[:, 3]
    tVf = np.array([tX[0] for tX in tFrames], int)
    tFt = np.array([tX[1] for tX in tFrames], np.float64)
    tJ = np.clip(np.searchsorted(tGt, tFt), 1, len(tGt) - 1)
    tPick = np.where(np.abs(tGt[tJ] - tFt) < np.abs(tGt[tJ - 1] - tFt), tJ, tJ - 1)
    return dict(video_frame=tVf, grip01=tGv[tPick].astype(np.float32),
                raw=tGraw[tPick].astype(np.int32), load=tGload[tPick].astype(np.float32),
                dt_ms=np.abs(tGt[tPick] - tFt) / NS_PER_MS)


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--manifest", default=str(JOB_TMP / "handumi_vla_manifest.json"))
    tAp.add_argument("--out", default=str(JOB_TMP / "handumi_grip"))
    tAp.add_argument("--limit", type=int, default=0)
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = pathlib.Path(tArgs.out)
    tOut.mkdir(parents=True, exist_ok=True)
    tItems = json.loads(pathlib.Path(tArgs.manifest).read_text())["items"]
    if tArgs.limit:
        tItems = tItems[: tArgs.limit]

    tDts, tViol, tJumps, tDone, tFail = [], 0, [], 0, []
    for tItem in tItems:
        tRead = readEpisode(tItem["session_id"], tItem["episode"])
        if tRead is None:
            tFail.append((tItem["episode_id"], "no mcap"))
            continue
        tFrames, tGrip = tRead
        tRec = {}
        for tSide, tStream in SIDE_STREAM.items():
            tAl = align(tFrames[tStream], tGrip[tSide])
            if tAl is None:
                tFail.append((tItem["episode_id"], f"{tSide}: empty"))
                tRec = None
                break
            tRec[f"{tSide}_video_frame"] = tAl["video_frame"]
            tRec[f"{tSide}_grip01"] = tAl["grip01"]
            tRec[f"{tSide}_raw"] = tAl["raw"]
            tRec[f"{tSide}_load"] = tAl["load"]
            tRec[f"{tSide}_dt_ms"] = tAl["dt_ms"].astype(np.float32)
            tDts.append(tAl["dt_ms"])
            tViol += int(((tAl["grip01"] < -GRIP_RANGE_TOL) | (tAl["grip01"] > 1 + GRIP_RANGE_TOL)).sum())
            tDiff = np.abs(np.diff(tAl["grip01"]))
            if len(tDiff):
                tJumps.append(tDiff.max())
        if tRec:
            np.savez_compressed(tOut / f"{tItem['episode_id']}.npz", **tRec)
            tDone += 1

    tD = np.concatenate(tDts) if tDts else np.array([0.0])
    tJ = np.array(tJumps) if tJumps else np.array([0.0])
    print(f"=== gripper extraction — {tDone}/{len(tItems)} episodes ===")
    if tFail:
        print(f"  failed {len(tFail)}: {tFail[:5]}")
    print(f"\n[QA 1] pose-gripper time offset (ms)   p50 {np.percentile(tD,50):.2f}  "
          f"p95 {np.percentile(tD,95):.2f}  max {tD.max():.2f}   "
          f"{'PASS' if tD.max() < FRAME_MS else 'WARN >1 frame'}")
    print(f"[QA 2] grip01 outside [0,1]       {tViol}   {'PASS' if tViol == 0 else 'FAIL'}")
    print(f"[QA 3] max frame-to-frame change  p50 {np.percentile(tJ,50):.4f}  "
          f"p95 {np.percentile(tJ,95):.4f}  max {tJ.max():.4f}   (1.0 = full open/close in one frame)")
    print(f"\n-> {tOut}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
