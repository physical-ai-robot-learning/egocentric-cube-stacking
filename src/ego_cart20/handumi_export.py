"""HandUMI (old259 Hpilot + HRL80) -> raw ego episodes in the ego_cart20 raw format (was sources/handumi_export.py).
Source adapter ONLY: it reads the frozen C8 pose stack and writes poses/gripper/timestamps; no labels are made here.

Per arm (its own wrist clock -- NOT paired to the other arm's frames):
  pose      c8_phase3.side (old259) / robotlike_offline_check.side_nogrip (HRL80): MASt3R-SLAM + IMU-VI metric scale +
            handumi_camera_tcp_v2 -> human TCP; then right-multiplied by the fixed tool frame X = F @ C @ CT4 so the hand TCP
            axes follow the reBot dataset-TCP convention (the same conjugation as ego_relcart20task:
            CT4^-1 C^-1 F^-1 rel F C CT4).
  valid     the stack's ok mask (lost / NaN pose invalid); the whole arm is refused if its IMU-VI scale is not valid
  t_ns      wrist frame capture_ns (sensors.mcap frame_meta) of each CORI pose sample
  gripper   gcal1 aperture (umi_aperture_cal_v1, caliper): g = G_era,side(open_fraction), 0 = closed, 1 = open; the SAME
            function object as rel16ego_derive_gcal1.G (imported, not re-implemented)
Cameras: head / left_wrist / right_wrist (capture_ns, video_frame) from frame_meta; mp4 paths recorded, never copied.

The C8 pose stack (v2k_retarget, c8_phase3, robotlike_offline_check, rel16ego_derive_gcal1, extract_grip) lives outside this
repository under ego_paths.DATA_ROOT / UMI_BRIDGE; _initStack() loads it once per process.

Pipeline stage: source adapter in front of the dataset stage (writes <raw_root>, read by scripts/convert_dataset.py).
usage (env ~/xvla-mac/bin/python):
  python -m ego_cart20.handumi_export --out ~/c8/ego_cart20_v2_raw [--episodes ID ...] [--workers 4]
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

from ego_paths import DATA_ROOT, EGO_COLLECTOR_HOME, UMI_BRIDGE, prepend_sys_path  # machine-specific roots (env-overridable)

from .raw_episode import saveRawEpisode
from .rotation6d import matrixToQuaternion

C8 = DATA_ROOT
V2K = UMI_BRIDGE / "track_c/v2k"
RL = C8 / "robotlike"
RAW_OLD = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/Hpilot"
RAW_HRL = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/HRL80"
HRL_SESSION = "20260928_101010"
SIDES = ("left", "right")
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class _PoseStack:
    """The external C8 pose-stack modules and the fixed tool frame (was the module globals V, P3, RLC, P, X, GCAL, OLD_RUNS,
    OLD_EXPORT)."""
    mV2k: object          # v2k_retarget module
    mPhase3: object       # c8_phase3 (as imported by v2k_retarget), its RUNS / export_dir are switched per era
    mRobotlike: object    # robotlike_offline_check
    mToolFrame: np.ndarray  # X = F @ C @ CT4
    mGcal: object         # rel16ego_derive_gcal1 (G = gcal1 aperture)
    mOldRuns: object
    mOldExport: object


_STACK: _PoseStack | None = None


def _load(name, path):
    tSpec = importlib.util.spec_from_file_location(name, path)
    tModule = importlib.util.module_from_spec(tSpec)
    tSpec.loader.exec_module(tModule)
    return tModule


def _initStack() -> None:
    """load the C8 pose stack into _STACK (also the ProcessPoolExecutor initializer) (was _init)"""
    global _STACK
    prepend_sys_path(str(C8), str(C8 / "c8old"), str(C8 / "rel16ego/rebot_ee"), str(UMI_BRIDGE / "umi76"), str(UMI_BRIDGE))
    os.environ.setdefault("REBOT_URDF", str(C8 / "rel16ego/rebot_ee/reBot_B601_DM_dualarm.urdf"))
    tV2k = _load("v2k", V2K / "v2k_retarget.py")
    tRobotlike, tPhase3 = tV2k.RLC, tV2k.P3
    tPhase3.init()
    tPipeline = tPhase3.P
    tV2k.P = tPipeline
    tCt4 = np.eye(4)
    tCt4[:3, :3] = tPipeline.C[:3, :3].T
    tToolFrame = np.asarray(tPipeline.F, np.float64) @ np.asarray(tPipeline.C, np.float64) @ tCt4
    tGcal = _load("gcal1", C8 / "rel16ego/rel16ego_derive_gcal1.py")
    _STACK = _PoseStack(tV2k, tPhase3, tRobotlike, tToolFrame, tGcal, tPhase3.RUNS, tPhase3.export_dir)


def sourceList() -> list[str]:
    """inclusion rule (fixed before any CART20 result): old = phase3 census first_fail in {None, no_segment} with BOTH
    IMU-VI scales valid; HRL80 = every episode of the frozen HRL80 split.  No workspace / IK / pseudo-q selection."""
    tCensus = json.load(open(C8 / "phase3_census.json"))["episodes"]
    tOld = sorted(tE for tE, tV in tCensus.items() if tV.get("first_fail") in (None, "no_segment")
                  and tV.get("left", {}).get("scale_valid") and tV.get("right", {}).get("scale_valid"))
    tSplit = json.load(open(V2K / "hrl80_split_frozen.json"))
    return tOld + sorted(tSplit["train_source_episodes"] + tSplit["val_source_episodes"])


def _episodeDir(episodeId):
    tSession, tNum = episodeId.rsplit("_", 1)
    return (RAW_HRL / f"HRL80_{tSession}" if tSession == HRL_SESSION else RAW_OLD / f"Hpilot_{tSession}") / f"episode_{tNum}"


def exportOne(episodeId, outRoot) -> dict:
    """one episode -> <outRoot>/<episodeId>/raw_episode.{json,npz}; status ok / exists / refused"""
    tStack = _STACK
    tOut = pathlib.Path(outRoot) / episodeId
    if (tOut / "raw_episode.json").exists():
        return dict(episode=episodeId, status="exists")
    tEpDir = _episodeDir(episodeId)
    tHrl = episodeId.startswith(HRL_SESSION)
    tPhase3 = tStack.mPhase3
    if tHrl:
        tPhase3.RUNS = RL / "runs"
        tPhase3.export_dir = lambda tag: RL / "export" / tag
        tSides = {tS: tStack.mRobotlike.side_nogrip(tStack.mRobotlike.tag_of(tEpDir, tS), tEpDir, tS) for tS in SIDES}
        tEra = "HRL80_v014"
    else:
        tPhase3.RUNS, tPhase3.export_dir = tStack.mOldRuns, tStack.mOldExport
        tSides = {tS: tPhase3.side(f"{episodeId}_{tS}") for tS in SIDES}
        tEra = "old259_contract"
    tBad = [tS for tS in SIDES if not tSides[tS]["scale_valid"]]
    if tBad:
        return dict(episode=episodeId, status="refused", reason=f"IMU-VI scale invalid: {tBad}")
    tFrameMeta = tPhase3.frame_meta(tEpDir)
    tArrays = {}
    tGripSrc = {}
    for tS in SIDES:
        tSide = tSides[tS]
        tOff, tN = tSide["off"], tSide["n"]
        tFr = tFrameMeta[f"{tS}_wrist"][tOff:tOff + tN]
        assert len(tFr) == tN, (episodeId, tS, len(tFr), tN)
        tVideoFrames = np.array([tV for tV, _ in tFr], np.int64)
        tTNs = np.array([tC for _, tC in tFr], np.int64)
        tOpen = (_hrlGrip(tEpDir, tS, tVideoFrames) if tHrl
                 else np.load(C8 / "grip" / f"{episodeId}_{tS}.npz")["open_fraction"].astype(np.float64))
        assert len(tOpen) == tN
        tG = tStack.mGcal.G(np.clip(tOpen, 0, 1), tS[0].upper(), tEra)
        tT = np.asarray(tSide["T_tcp"], np.float64) @ tStack.mToolFrame[None]
        tOk = np.asarray(tSide["ok"], bool) & np.isfinite(tG)
        tArrays.update({f"{tS}_t_ns": tTNs, f"{tS}_position": tT[:, :3, 3], f"{tS}_quaternion": matrixToQuaternion(tT[:, :3, :3]),
                        f"{tS}_valid": tOk, f"{tS}_gripper": tG, f"{tS}_open_fraction_raw": tOpen,
                        f"{tS}_video_frame": tVideoFrames})
        tGripSrc[tS] = dict(scale=float(tSide["s"]), coverage=float(tOk.mean()))
    for tCam in CAMERA_NAMES:
        tArrays[f"cam_{tCam}_t_ns"] = np.array([tT for _, tT in tFrameMeta[tCam]], np.int64)
        tArrays[f"cam_{tCam}_frame"] = np.array([tV for tV, _ in tFrameMeta[tCam]], np.int64)
    tEpMeta = json.load(open(tEpDir / "episode_meta.json"))
    tMeta = dict(schema="ego_cart20_raw_episode/v1", episode_id=episodeId, source="human_egocentric_handumi", era=tEra,
                 task="cube_stacking",
                 instruction=tEpMeta["instruction"], stack_order=tEpMeta["order"], episode_status=tEpMeta.get("status"),
                 clock="capture_monotonic_ns", videos={tCam: str(tEpDir / f"{tCam}.mp4") for tCam in CAMERA_NAMES},
                 tool_frame=dict(name="handumi_tcp_v2 @ F @ C @ CT4 (reBot dataset-TCP axes)", X=tStack.mToolFrame.tolist()),
                 gripper_calibration=dict(version="umi_aperture_cal_v1", era_map=tEra,
                                          convention="0 = closed, 1 = open, mm / 80 mm",
                                          source="old: ~/c8/grip/<tag>.npz open_fraction" if not tHrl
                                          else "HRL80: sensors.mcap /<side>/gripper normalized via extract_grip.align"),
                 provenance=dict(raw_dir=str(tEpDir),
                                 pose_stack="c8_phase3.side" if not tHrl else "robotlike_offline_check.side_nogrip",
                                 arms=tGripSrc, exporter=str(pathlib.Path(__file__).resolve()),
                                 exported=time.strftime("%Y-%m-%d %H:%M:%S")))
    saveRawEpisode(tOut, tMeta, tArrays)
    return dict(episode=episodeId, status="ok", era=tEra, n={tS: int(tSides[tS]["n"]) for tS in SIDES})


def _hrlGrip(epDir, side, videoFrames):
    """c8oldv2_build_hrl80.grip_open, verbatim logic (that module runs its build at import time, so it cannot be imported)"""
    from extract_grip import align
    from mcap.reader import make_reader
    tFrames, tGrip = [], []
    with open(epDir / "sensors.mcap", "rb") as tF:
        for _, tChannel, tMsg in make_reader(tF).iter_messages(topics=[f"/{side}_wrist/frame_meta", f"/{side}/gripper"]):
            tD = json.loads(tMsg.data)
            if tChannel.topic.endswith("frame_meta"):
                tFrames.append((int(tD["video_frame"]), int(tD["capture_ns"])))
            elif tD.get("normalized") is not None:
                tGrip.append((int(tD["sample_ns"]), float(tD["normalized"]), int(tD.get("raw_position", 0)), 0.0))
    tAligned = align(tFrames, tGrip)
    tLut = dict(zip(tAligned["video_frame"].tolist(), range(len(tAligned["video_frame"]))))
    tIdx = np.array([tLut[int(tV)] for tV in videoFrames])
    return np.clip(tAligned["grip01"][tIdx], 0, 1).astype(np.float64)


def _worker(args):
    tEpisode, tOut = args
    try:
        return exportOne(tEpisode, tOut)
    except Exception as tEx:
        return dict(episode=tEpisode, status="error", reason=f"{type(tEx).__name__}: {tEx}")


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--out", required=True)
    tParser.add_argument("--episodes", nargs="*")
    tParser.add_argument("--workers", type=int, default=1)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tEpisodes = tArgs.episodes or sourceList()
    pathlib.Path(tArgs.out).mkdir(parents=True, exist_ok=True)
    print(f"export {len(tEpisodes)} episodes -> {tArgs.out}", flush=True)
    if tArgs.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(tArgs.workers, initializer=_initStack) as tEx:
            tResults = list(tEx.map(_worker, [(tE, tArgs.out) for tE in tEpisodes]))
    else:
        _initStack()
        tResults = []
        for tE in tEpisodes:
            tResults.append(_worker((tE, tArgs.out)))
            print(json.dumps(tResults[-1]), flush=True)
    tLog = pathlib.Path(tArgs.out) / f"export_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(tResults, open(tLog, "w"), indent=1)
    print("DONE", dict(collections.Counter(tR["status"] for tR in tResults)), "->", tLog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
