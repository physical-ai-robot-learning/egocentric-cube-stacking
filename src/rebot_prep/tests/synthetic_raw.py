"""Synthetic HandUMI raw sessions (Hpilot layout) and the Gen-1 working tree (~/c8 layout) for the grip and census tests.

    <raw>/Hpilot_<session>/session_meta.json                  gripper ticks_closed / ticks_open per side
    <raw>/Hpilot_<session>/episode_<n>/sensors.mcap            /<stream>/frame_meta (left_wrist, right_wrist, head), /<side>/gripper
    <raw>/Hpilot_<session>/episode_<n>/{episode_meta.json, left_wrist.mp4, right_wrist.mp4, head.mp4}
    <c8>/c8_domain_manifest.json                               tag -> episode, domain
    <c8>/export[_early]/<tag>/imu_data.json                    CORI = the exported (offset) camera frames
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mcap.writer import Writer

FRAME_NS = 33_333_333
GRIP_PERIOD_NS = 10_526_315          # ~95 Hz
T0_NS = 2_000_000_000
STREAMS = ("left_wrist", "right_wrist", "head")
SIDES = ("left", "right")
SESSION_MAIN = "20260917_150545"
SESSION_EARLY = "20260915_090000"
TICKS = {SESSION_MAIN: {"left": (100, 1236), "right": (2000, 752)}, SESSION_EARLY: {"left": (3000, 1863), "right": (500, 1748)}}


def writeMcap(path: Path, nFrames: int, ticks: dict, seed: int, headHalfRate: bool = False, dropGripper: str | None = None) -> dict:
    """-> {stream: [(video_frame, capture_ns)]} and the raw gripper ticks per side"""
    tRng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tFrames = {}
    with open(path, "wb") as tFh:
        tW = Writer(tFh)
        tW.start(profile="handumi", library="test")
        tSchema = tW.register_schema("x", "jsonschema", b"{}")
        tChan = {}
        for tTopic in [f"/{s}/frame_meta" for s in STREAMS] + [f"/{s}/gripper" for s in SIDES] + ["/left/imu"]:
            tChan[tTopic] = tW.register_channel(tTopic, "json", tSchema)
        for tStream in STREAMS:
            tPeriod = 2 * FRAME_NS if (tStream == "head" and headHalfRate) else FRAME_NS
            tCap = T0_NS + np.arange(nFrames) * tPeriod + tRng.integers(-200_000, 200_000, nFrames)
            tFrames[tStream] = [(tI, int(c)) for tI, c in enumerate(tCap)]
            for tI, tC in tFrames[tStream]:
                tW.add_message(tChan[f"/{tStream}/frame_meta"], log_time=tC, publish_time=tC, sequence=tI,
                               data=json.dumps(dict(stream=tStream, frame_index=tI, video_frame=tI, capture_ns=tC)).encode())
        tRaw = {}
        for tSide in SIDES:
            if tSide == dropGripper:
                continue
            tClosed, tOpen = ticks[tSide]
            tN = int(nFrames * FRAME_NS / GRIP_PERIOD_NS) + 4
            tT = T0_NS - 20_000_000 + np.arange(tN) * GRIP_PERIOD_NS
            tFrac = np.clip(0.5 + 0.6 * np.sin(np.arange(tN) / 15.0 + seed), 0, 1)
            tRaw[tSide] = np.round(tClosed + tFrac * (tOpen - tClosed) + tRng.normal(0, 3, tN)).astype(int)
            for tI in range(tN):
                tMsg = dict(sample_ns=int(tT[tI]), normalized=float(tFrac[tI]), raw_position=int(tRaw[tSide][tI]),
                            telemetry=({"load_percent": float(tI % 7)} if tI % 5 else None))
                tW.add_message(tChan[f"/{tSide}/gripper"], log_time=int(tT[tI]), publish_time=int(tT[tI]), sequence=tI,
                               data=json.dumps(tMsg).encode())
        tW.add_message(tChan["/left/imu"], log_time=T0_NS, publish_time=T0_NS, sequence=0, data=b"{}")
        tW.finish()
    return dict(frames=tFrames, raw=tRaw)


def writeCori(exportDir: Path, captureNs: list[int], skew: float = 0.0) -> None:
    exportDir.mkdir(parents=True, exist_ok=True)
    tCts = [(c - captureNs[0]) / 1e6 + 12.5 + skew * k for k, c in enumerate(captureNs)]
    (exportDir / "imu_data.json").write_text(json.dumps({"1": {"streams": {"CORI": {"samples": [{"cts": c} for c in tCts]}}}}))


def buildTree(raw: Path, c8: Path) -> dict:
    """Two sessions (main_calibrated / early_precalibration); per episode nFrames, export offset, special cases."""
    tSpecs = [  # session, episode, n frames, offsets (left, right), case
        (SESSION_MAIN, "0", 90, (2, 1), None),
        (SESSION_MAIN, "1", 80, (0, 3), None),
        (SESSION_MAIN, "2", 70, (1, 1), "bad_mapping"),
        (SESSION_EARLY, "0", 85, (2, 2), None),
        (SESSION_EARLY, "1", 75, (1, 0), "head_half_rate"),
        (SESSION_EARLY, "2", 60, (0, 0), "no_mcap"),
        (SESSION_EARLY, "3", 60, (0, 0), "no_gripper_right"),
    ]
    tManifest, tOut = {}, {}
    for tSess in (SESSION_MAIN, SESSION_EARLY):
        tDir = raw / f"Hpilot_{tSess}"
        tDir.mkdir(parents=True, exist_ok=True)
        (tDir / "session_meta.json").write_text(json.dumps({"devices": {
            f"gripper_{s}": dict(ticks_closed=TICKS[tSess][s][0], ticks_open=TICKS[tSess][s][1]) for s in SIDES}}))
    for tK, (tSess, tEp, tN, tOffsets, tCase) in enumerate(tSpecs):
        tEpisode = f"{tSess}_{tEp}"
        tDomain = "main_calibrated" if tSess == SESSION_MAIN else "early_precalibration"
        tEpDir = raw / f"Hpilot_{tSess}" / f"episode_{tEp}"
        tEpDir.mkdir(parents=True, exist_ok=True)
        (tEpDir / "episode_meta.json").write_text(json.dumps({"imu_present": True}))
        for tVideo in ("left_wrist.mp4", "right_wrist.mp4", "head.mp4"):
            (tEpDir / tVideo).write_bytes(b"")
        tRec = None
        if tCase != "no_mcap":
            tRec = writeMcap(tEpDir / "sensors.mcap", tN, TICKS[tSess], seed=tK, headHalfRate=tCase == "head_half_rate",
                             dropGripper="right" if tCase == "no_gripper_right" else None)
        for tSide, tOff in zip(SIDES, tOffsets):
            tTag = f"{tEpisode}_{tSide}"
            tManifest[tTag] = dict(episode=tEpisode, domain=tDomain)
            tCap = [c for _, c in tRec["frames"][f"{tSide}_wrist"]][tOff:] if tRec else list(T0_NS + np.arange(tN) * FRAME_NS)
            writeCori(c8 / ("export" if tDomain == "main_calibrated" else "export_early") / tTag, tCap,
                      skew=1.5 if (tCase == "bad_mapping" and tSide == "left") else 0.0)
        tOut[tEpisode] = tRec
    (c8 / "c8_domain_manifest.json").write_text(json.dumps(tManifest))
    return tOut
