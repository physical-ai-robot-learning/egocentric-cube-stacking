#!/usr/bin/env python3
"""Corrected HandUMI gripper channel: raw ticks -> physical jaw width in metres (Gen-1 dataset stage, gripper step 2).

In:  <grip>/<session>_<n>.npz from extract_grip (<side>_raw), session_meta.json of every Hpilot session
Out: <out>/<same name>.npz (+ <side>_open_fraction, <side>_width_m), <out>/gripper_contract.json, a printed anchor table
Usage: python -m rebot_prep.grip.correct_grip [--grip <dir>] [--out <dir>]

The stored `normalized` cannot be used directly. Its `ticks_open` sits beyond the mechanical stop in
almost every session (right reaches only ~70% of the calibrated span; left's direction even flips between
sessions and 140419-left's metadata is simply wrong), so the same physical opening maps to different
normalized values across sessions. Everything is therefore re-derived from raw:

    raw tick
      -> direction from session_meta's sign(ticks_open - ticks_closed)   (140419-left: regression-recovered)
      -> endpoints re-anchored to what the hardware actually travelled
      -> physical open fraction in [0,1]
      -> width_m = fraction * APERTURE_M

Anchors are the 90th percentile of the per-episode extremes rather than the absolute extreme: the jaws
hit a hard stop, so per-episode extremes cluster there, and one odd episode should not define the scale.
That single odd episode is exactly what broke 135136-left (its absolute 'closed' frame shows a visible
gap on video).

APERTURE_M is a locked physical constant, video-validated: closed / 50 mm cube grasp / fully-open frames
were read off the wrist camera for all 5 sessions x 2 arms, and the median over the 8 combos with valid
anchors is 67.5 mm. Uncertainty ~±5 mm; it is a single global scale on this one channel, so a later
caliper measurement can be applied by changing this number alone.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

import numpy as np

from ego_paths import EGO_COLLECTOR_HOME, HOME

APERTURE_M = 0.0675
APERTURE_UNCERTAINTY_M = 0.005
RAW = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/Hpilot"
JOB_TMP = HOME / ".claude/jobs/e649aef3/tmp"          # the original Track A job folder (default inputs / outputs)
# 140419-left: session_meta says (6943, 5389) but regressing stored normalized on raw returns this pair
# with R^2 = 1.0, so the metadata is wrong and the recovered pair is what the recording actually used.
RECOVERED = {("20260916_140419", "left"): (2847, 1293)}
EXCLUDED_FROM_APERTURE = {
    ("20260917_135136", "left"): "absolute closed anchor was an outlier episode; video shows a gap",
    ("20260916_140419", "left"): "no closing plateau detected (7 episodes only)",
}
LEFT_SPAN_SESSION = "20260917_150545"    # the one left session with a video-validated closed anchor AND full travel
CLOSED_PCT, OPEN_PCT = 10, 90            # anchors = percentiles of the per-episode extremes (low / high side)
NEVER_OPENED_TRAVEL = 0.85               # travel fraction below which the table flags "never fully opened"
SIDES = ("left", "right")


def sessionMetaTicks() -> dict:
    """{session: {side: (ticks_closed, ticks_open)}} from every Hpilot_<session>/session_meta.json (was session_meta_ticks)"""
    tOut = {}
    for tDir in sorted(RAW.glob("Hpilot_*")):
        tPath = tDir / "session_meta.json"
        if not tPath.is_file():
            continue
        tDevices = json.loads(tPath.read_text()).get("devices", {})
        tOut[tDir.name.replace("Hpilot_", "")] = {
            tS: (tDevices.get(f"gripper_{tS}", {}).get("ticks_closed"), tDevices.get(f"gripper_{tS}", {}).get("ticks_open"))
            for tS in SIDES}
    return tOut


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--grip", default=str(JOB_TMP / "handumi_grip"))
    tAp.add_argument("--out", default=str(JOB_TMP / "handumi_grip_corrected"))
    return tAp


def observedAnchors(bySession: dict, meta: dict) -> tuple[dict, list[dict]]:
    """per (session, side): (closed, observed open, direction) from the per-episode extremes, and the anchor table rows"""
    tAnchors, tTable = {}, []
    for tSess in sorted(bySession):
        for tSide in SIDES:
            tC, tO = RECOVERED.get((tSess, tSide), meta[tSess][tSide])
            tSign = 1 if tO > tC else -1
            tExClosed, tExOpen = [], []
            for tFile in bySession[tSess]:
                tRaw = np.load(tFile)[f"{tSide}_raw"].astype(float)
                tExClosed.append(tRaw.min() if tSign > 0 else tRaw.max())
                tExOpen.append(tRaw.max() if tSign > 0 else tRaw.min())
            # The CLOSED anchor is trustworthy -- the jaws touching is easy to reach and was confirmed
            # on video for every session. The OPEN anchor is not: in 140419/170333/100511 the left jaw
            # travels only ~530 ticks against ~1100 in the sessions that do reach the stop, i.e. the
            # operator never opened it fully there. The mechanical span is a property of the DEVICE, so
            # it is taken from the sessions that reach the stop and applied from each session's own
            # closed anchor, instead of trusting a short session's widest moment as "fully open".
            tClosed = np.percentile(tExClosed, CLOSED_PCT if tSign > 0 else OPEN_PCT)
            tOpened = np.percentile(tExOpen, OPEN_PCT if tSign > 0 else CLOSED_PCT)
            tAnchors[(tSess, tSide)] = (float(tClosed), float(tOpened), tSign)
            tTable.append(dict(session=tSess, side=tSide, direction=int(tSign),
                               closed_raw=round(float(tClosed), 1), open_raw=round(float(tOpened), 1),
                               span=round(abs(float(tOpened) - float(tClosed)), 1),
                               ticks_source="regression-recovered" if (tSess, tSide) in RECOVERED else "session_meta",
                               meta_ticks=[tC, tO],
                               excluded_from_aperture=EXCLUDED_FROM_APERTURE.get((tSess, tSide))))
    return tAnchors, tTable


def applyDeviceSpan(anchors: dict, table: list[dict], sessions) -> dict:
    """open anchor := closed + direction * device span (in place); -> span per side.
    left: the span from LEFT_SPAN_SESSION, the one left session whose closed anchor is video-validated AND whose travel
    reaches the stop. 135136-left also looks full but its closed anchor is the flagged outlier, so its span would drag the
    constant low. right: every session reaches the stop, so the median across them is the device span."""
    tSpan = {}
    tSpanRight = [abs(anchors[(tK, "right")][1] - anchors[(tK, "right")][0]) for tK in sessions]
    tSpan["right"] = float(np.median(tSpanRight))
    tSpan["left"] = float(abs(anchors[(LEFT_SPAN_SESSION, "left")][1] - anchors[(LEFT_SPAN_SESSION, "left")][0]))
    for (tSess, tSide), (tCl, tOp, tSign) in list(anchors.items()):
        tFull = tCl + tSign * tSpan[tSide]
        anchors[(tSess, tSide)] = (tCl, tFull, tSign)
        for tRow in table:
            if tRow["session"] == tSess and tRow["side"] == tSide:
                tRow["open_raw_observed"] = tRow["open_raw"]
                tRow["open_raw"] = round(tFull, 1)
                tRow["span"] = round(tSpan[tSide], 1)
                tRow["span_source"] = "device span (sessions reaching the stop)"
                tRow["travel_fraction"] = round(abs(tOp - tCl) / tSpan[tSide], 3)
    return tSpan


def contract(table: list[dict]) -> dict:
    """the gripper_contract.json content"""
    return dict(schema="handumi_gripper_contract/v1", created="2026-09-22",
                semantic="0 m = fully closed, larger = more open. NO inversion; same direction as reBot.",
                aperture_m=APERTURE_M, aperture_uncertainty_m=APERTURE_UNCERTAINTY_M,
                aperture_method="video-validated closed / 50 mm cube grasp / fully-open frames, "
                                "5 sessions x 2 arms; median over the 8 combos with valid anchors",
                aperture_todo="measure the assembled device's full aperture with a caliper; only this "
                              "scalar changes, no SLAM or video reprocessing",
                anchor_rule="closed = 90th-percentile of per-episode closed extremes (reliable, video-confirmed); "
                            "open = closed + device span. Device span: left from 20260917_150545 (only left session with a "
                            "video-validated closed anchor AND full travel), right from the median across sessions (all reach "
                            "the stop). Three left sessions travel only ~50% of the span -- the operator never opened the left "
                            "jaw fully there, and their width_m correctly saturates below the full aperture.",
                conversion="width_m = clip((raw - closed_raw)/(open_raw - closed_raw), 0, 1) * aperture_m",
                state="width_m[t]", action="width_m[t+1]",
                anchors=table)


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tSrc, tOut = pathlib.Path(tArgs.grip), pathlib.Path(tArgs.out)
    tOut.mkdir(parents=True, exist_ok=True)
    tMeta = sessionMetaTicks()

    tBy = collections.defaultdict(list)
    for tFile in sorted(glob.glob(str(tSrc / "*.npz"))):
        tBy["_".join(pathlib.Path(tFile).stem.split("_")[:2])].append(tFile)

    tAnchors, tTable = observedAnchors(tBy, tMeta)
    tSpan = applyDeviceSpan(tAnchors, tTable, tBy)
    print(f"   device span: left {tSpan['left']:.0f}  right {tSpan['right']:.0f} ticks")

    tN, tClipped = 0, 0
    for tSess in sorted(tBy):
        for tFile in tBy[tSess]:
            tZ = dict(np.load(tFile))
            for tSide in SIDES:
                tCl, tOp, _ = tAnchors[(tSess, tSide)]
                tFrac = (tZ[f"{tSide}_raw"].astype(np.float64) - tCl) / (tOp - tCl)
                tClipped += int(((tFrac < 0) | (tFrac > 1)).sum())
                tFrac = np.clip(tFrac, 0.0, 1.0)
                tZ[f"{tSide}_open_fraction"] = tFrac.astype(np.float32)
                tZ[f"{tSide}_width_m"] = (tFrac * APERTURE_M).astype(np.float32)
            np.savez_compressed(tOut / pathlib.Path(tFile).name, **tZ)
            tN += 1

    (tOut / "gripper_contract.json").write_text(json.dumps(contract(tTable), indent=1))

    print(f"=== corrected gripper — {tN} episodes ===")
    print(f"   aperture_m = {APERTURE_M} (±{APERTURE_UNCERTAINTY_M})   clipped samples {tClipped}")
    print(f"\n   {'session':<18} {'side':<6} {'dir':>4} {'closed':>8} {'open':>8} {'span':>6} {'travel':>6}  source")
    for tRow in tTable:
        tNote = f"   [{tRow['excluded_from_aperture']}]" if tRow["excluded_from_aperture"] else ""
        tTravel = tRow.get("travel_fraction", 1.0)
        tWarn = "  <-- never fully opened" if tTravel < NEVER_OPENED_TRAVEL else ""
        print(f"   {tRow['session']:<18} {tRow['side']:<6} {tRow['direction']:>4} {tRow['closed_raw']:>8.0f} "
              f"{tRow['open_raw']:>8.0f} {tRow['span']:>6.0f} {tTravel:>6.2f}  {tRow['ticks_source']}{tNote}{tWarn}")
    tW = np.concatenate([np.load(tP)[f"{tS}_width_m"] for tP in tOut.glob("2026*.npz") for tS in SIDES])
    print(f"\n   width_m: min {tW.min():.4f}  p50 {np.percentile(tW,50):.4f}  p95 {np.percentile(tW,95):.4f}  max {tW.max():.4f}")
    print(f"-> {tOut}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
