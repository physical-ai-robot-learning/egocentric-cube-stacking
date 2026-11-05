"""Synthetic inputs of the ego_cart20 characterization tests (numpy / scipy / cv2 only, independent of the package code).

rawArrays()       raw_episode.npz arrays of one two-arm episode: own wrist clocks, smooth motion with rotation, a tracking
                  hole on the left arm, a 200 mm spike and a one-way jump on the right arm, head + left-wrist cameras
rawMeta()         the matching raw_episode.json
writeRawRoot()    <root>/<episode_id>/raw_episode.{json,npz} for several episodes (different stack orders)
writeVideo()      a small mp4 / avi whose frame i is filled with grey level FRAME_GREY_STEP * i
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

ARM_HZ = 30.0
DURATION_S = 6.0
T0_NS = 1_000_000_000_000
FRAME_GREY_STEP = 7
VIDEO_HW = (48, 64)


def _quatWxyz(rotvec: np.ndarray) -> np.ndarray:
    tXyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.concatenate([tXyzw[:, 3:], tXyzw[:, :3]], 1)


def rawArrays(seed: int = 0, spike: bool = True, oneWay: bool = False) -> dict:
    tRng = np.random.default_rng(seed)
    tOut = {}
    for tSide, tOffsetS, tSign in (("left", 0.004, 1.0), ("right", 0.017, -1.0)):
        tT = np.arange(0.0, DURATION_S, 1.0 / ARM_HZ) + tOffsetS
        tN = len(tT)
        tPos = np.stack([0.2 * np.sin(0.7 * tT) * tSign, 0.1 * np.cos(0.5 * tT), 0.05 * tT], 1)
        tPos = tPos + tRng.normal(0, 1e-4, tPos.shape)
        tRotvec = np.stack([0.3 * np.sin(0.9 * tT), 0.2 * tT * tSign, 0.4 * np.cos(0.3 * tT)], 1)
        tValid = np.ones(tN, bool)
        if tSide == "left":
            tValid[40:46] = False                      # 0.2 s tracking hole
            tPos[42] = np.nan                          # NaN pose inside the hole
        else:
            if spike:
                tPos[100:103, 0] += 0.2                # 200 mm spike, then back
            if oneWay:
                tPos[150:, 1] += 0.25                  # 250 mm one-way jump
        tGrip = np.clip(0.5 + 0.4 * np.sin(1.3 * tT + (0.0 if tSide == "left" else 1.0)), 0, 1)
        tOut[f"{tSide}_t_ns"] = (T0_NS + np.round(tT * 1e9)).astype(np.int64)
        tOut[f"{tSide}_position"] = tPos
        tOut[f"{tSide}_quaternion"] = _quatWxyz(tRotvec)
        tOut[f"{tSide}_valid"] = tValid
        tOut[f"{tSide}_gripper"] = tGrip
    for tCam, tOffsetS in (("head", 0.009), ("left_wrist", 0.002)):
        tT = np.arange(0.0, DURATION_S, 1.0 / ARM_HZ) + tOffsetS
        tKeep = np.ones(len(tT), bool)
        if tCam == "head":
            tKeep[60:64] = False                       # dropped head frames
        tOut[f"cam_{tCam}_t_ns"] = (T0_NS + np.round(tT[tKeep] * 1e9)).astype(np.int64)
        tOut[f"cam_{tCam}_frame"] = np.flatnonzero(tKeep).astype(np.int64)
    return tOut


def rawMeta(episodeId: str, stackOrder: str = "RBP", videos: dict | None = None) -> dict:
    return dict(schema="ego_cart20_raw_episode/v1", episode_id=episodeId, source="human_egocentric_handumi", era="synthetic",
                task="cube_stacking", instruction=f"stack {stackOrder}", stack_order=stackOrder, clock="capture_monotonic_ns",
                videos=videos or {}, tool_frame=dict(name="identity"), gripper_calibration=dict(version="synthetic"),
                provenance=dict(note="synthetic"))


def writeRawEpisode(epDir: pathlib.Path, episodeId: str, stackOrder: str, seed: int, videos: dict | None = None,
                    oneWay: bool = False) -> None:
    epDir.mkdir(parents=True)
    np.savez_compressed(epDir / "raw_episode.npz", **rawArrays(seed, oneWay=oneWay))
    (epDir / "raw_episode.json").write_text(json.dumps(rawMeta(episodeId, stackOrder, videos), indent=1))


EPISODES = (("20260101_000000_0001", "RBP", 0), ("20260101_000000_0002", "BRP", 1), ("20260101_000000_0003", "RBP", 2),
            ("20260101_000000_0004", "PRB", 3))


def writeRawRoot(root: pathlib.Path, videos: dict | None = None) -> list[str]:
    for tId, tOrder, tSeed in EPISODES:
        writeRawEpisode(root / tId, tId, tOrder, tSeed, videos, oneWay=tSeed == 3)
    return [tId for tId, _, _ in EPISODES]


def writeVideo(path: pathlib.Path, nFrames: int, codec: str = "mp4v") -> pathlib.Path:
    import cv2
    tH, tW = VIDEO_HW
    tWriter = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 30.0, (tW, tH))
    for tI in range(nFrames):
        tImg = np.full((tH, tW, 3), (FRAME_GREY_STEP * tI) % 256, np.uint8)
        tImg[: tH // 2, : tW // 2, 2] = 255           # a red block (BGR) so colour order is visible
        tWriter.write(tImg)
    tWriter.release()
    return path
