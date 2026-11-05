"""Raw ego episode format: reader and writer (was io/raw_episode_loader.py).

Source-independent; written by a source exporter such as handumi_export.py.
<raw_root>/<episode_id>/raw_episode.json
    episode_id, source, task, instruction, stack_order, clock ("capture_monotonic_ns"),
    videos {camera: absolute mp4 path}, tool_frame (how the hand TCP axes were fixed), gripper_calibration, provenance
<raw_root>/<episode_id>/raw_episode.npz      (per arm, own clock; LEFT and RIGHT are separate streams)
    {arm}_t_ns int64 (n)      {arm}_position (n,3) m      {arm}_quaternion (n,4) wxyz
    {arm}_valid bool (n)      {arm}_gripper (n)  normalized aperture, 0 = closed, 1 = open (calibrated, NOT clipped yet ok)
    cam_{camera}_t_ns int64   cam_{camera}_frame int64  (video frame index in the camera's mp4)
Raw data is preserved untouched; processing never writes into a raw episode directory.

Pipeline stage: input of the dataset stage (convert.py reads it, handumi_export.py writes it).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

from .config import ARMS
from .transforms import poseToTransform

CAMERAS = ("head", "left_wrist", "right_wrist")     # view 0 / 1 / 2
REQUIRED_META = ("episode_id", "instruction", "stack_order", "task")
NS_PER_S = 1e9


class RawEpisodeError(ValueError):
    """raw_episode.json lacks a required metadata field."""


@dataclass(frozen=True)
class RawArm:
    """One arm's raw stream on the episode clock: times (s), poses (n, 4, 4), validity, gripper aperture."""
    mTimeS: np.ndarray
    mPose: np.ndarray
    mValid: np.ndarray
    mGripper: np.ndarray


@dataclass(frozen=True)
class RawEpisode:
    """One raw episode: metadata (raw_episode.json), both arms, camera streams camera -> (t_s, frame_index)."""
    mEpisodeId: str
    mMeta: dict
    mArms: dict
    mCams: dict = field(default_factory=dict)
    mT0Ns: int = 0                                  # clock offset subtracted from every stream


def loadRawEpisode(epDir) -> RawEpisode:
    """<raw_root>/<episode_id> -> RawEpisode; every stream on one clock that starts at the earlier arm's first sample"""
    tDir = pathlib.Path(epDir)
    tMeta = json.load(open(tDir / "raw_episode.json"))
    tZ = np.load(tDir / "raw_episode.npz")
    tMissing = [tKey for tKey in REQUIRED_META if not tMeta.get(tKey)]
    if tMissing:
        raise RawEpisodeError(f"{tDir}: raw_episode.json missing {tMissing}")
    tT0 = int(min(int(tZ[f"{tArm}_t_ns"][0]) for tArm in ARMS))
    tArms = {}
    for tArm in ARMS:
        tTime = (tZ[f"{tArm}_t_ns"].astype(np.int64) - tT0) / NS_PER_S
        tArms[tArm] = RawArm(tTime, poseToTransform(tZ[f"{tArm}_position"], tZ[f"{tArm}_quaternion"]),
                             tZ[f"{tArm}_valid"].astype(bool), tZ[f"{tArm}_gripper"].astype(np.float64))
    tCams = {tCam: ((tZ[f"cam_{tCam}_t_ns"].astype(np.int64) - tT0) / NS_PER_S, tZ[f"cam_{tCam}_frame"].astype(np.int64))
             for tCam in CAMERAS if f"cam_{tCam}_t_ns" in tZ.files}
    return RawEpisode(tMeta["episode_id"], tMeta, tArms, tCams, tT0)


def saveRawEpisode(epDir, meta: dict, arrays: dict) -> None:
    """used by source exporters; refuses to overwrite"""
    tDir = pathlib.Path(epDir)
    tDir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(tDir / "raw_episode.npz", **arrays)
    json.dump(meta, open(tDir / "raw_episode.json", "w"), indent=1)
