"""Synthetic inputs of the Phase-3 segment census (and the other readers of the same ~/c8 / ~/umi_bridge trees):

    <umi_bridge>/trackA_mast3r_pose_v1/cp6_scale.py               the frozen C-P6 text (exec'd, up to "tags = [")
    <umi_bridge>/trackA_mast3r_pose_v1/data/qa/qa_all.json, data/handumi_camera_tcp_v2.yaml
    <umi_bridge>/track_c/{pseudo_joint_pipeline,state32}.py       shims -> the repository modules
    <umi_bridge>/track_c/data/r150_q_tcp.npz                       robot reference: J (deg, 14 columns), ep, tcp
    <c8>/c8_domain_manifest.json, census_pre.json, grip/grip_census.json, grip/<tag>.npz
    <c8>/runs_m3/<tag>/ss1.csv (video-indexed), export/<tag>/{imu_data.json, orbslam_setting.yaml}
    <raw>/Hpilot_<session>/episode_<n>/sensors.mcap                frame_meta of left_wrist / right_wrist / head
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from mast3r_pose.tests import synthetic as M3
from rebot_prep.tests import synthetic_raw

DATA = Path(__file__).with_name("data")
N_CORI = 220
OFFSETS = {"left": 2, "right": 1}
SCALE = {"left": 0.8, "right": 1.25}
ANCHOR_DEG12 = [-35.5, -82.7, -55.0, 38.8, -6.9, -12.3, 37.1, -86.6, -55.3, 44.1, 1.9, -11.9]
N_REF = 6400
REF_NOISE_DEG = 30.0
EPISODE_OK = "20260917_150545_0"
EPISODE_PRE_FAIL = "20260917_150545_1"
EPISODE_NO_RUN = "20260917_150545_2"
NAN_ROW = 12                         # video frame of the left side whose pose is NaN while is_lost == false
SHIM = "import sys\nimport {mod} as _m\nsys.modules[__name__] = _m\n"


def writeReference(path: Path, seed: int = 11) -> None:
    """32 smooth robot episodes of 200 frames around the R150 active anchor (sinusoids per joint, up to REF_NOISE_DEG)"""
    tRng = np.random.default_rng(seed)
    tS = np.arange(N_REF) % 200 / 30.0
    tEp = np.arange(N_REF) // 200
    tAmp = tRng.uniform(0, REF_NOISE_DEG, (N_REF // 200, 12))[tEp]
    tHz = tRng.uniform(0.1, 0.5, (N_REF // 200, 12))[tEp]
    tPhase = tRng.uniform(0, 2 * np.pi, (N_REF // 200, 12))[tEp]
    tQ12 = np.array(ANCHOR_DEG12) + tAmp * np.sin(2 * np.pi * tHz * tS[:, None] + tPhase)
    tJ = np.zeros((N_REF, 14))
    tJ[:, :6], tJ[:, 7:13] = tQ12[:, :6], tQ12[:, 6:]
    tTcp = np.zeros((N_REF, 14))
    tTcp[:, 2] = -0.4 + tRng.normal(0, 0.05, N_REF)
    tTcp[:, 10] = -0.38 + tRng.normal(0, 0.05, N_REF)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, J=tJ, ep=tEp, tcp=tTcp)


def buildTrackDirs(umiBridge: Path) -> dict:
    tTa = umiBridge / "trackA_mast3r_pose_v1"
    (tTa / "data/qa").mkdir(parents=True)
    shutil.copy(DATA / "cp6_scale_frozen.py.txt", tTa / "cp6_scale.py")
    (tTa / "data/qa/qa_all.json").write_text("{}")
    M3.writeCameraTcp(tTa / "data/handumi_camera_tcp_v2.yaml")
    tTc = umiBridge / "track_c"
    (tTc / "data").mkdir(parents=True)
    (tTc / "pseudo_joint_pipeline.py").write_text(SHIM.format(mod="rebot_prep.pseudo_joint.pseudo_joint_pipeline"))
    (tTc / "state32.py").write_text(SHIM.format(mod="rebot_prep.pseudo_joint.state32"))
    writeReference(tTc / "data/r150_q_tcp.npz")
    return dict(ta=tTa, tc=tTc)


def buildC8(c8: Path, raw: Path, seed: int = 5) -> None:
    tRng = np.random.default_rng(seed)
    tSess, tEp = EPISODE_OK.rsplit("_", 1)
    tN = N_CORI + max(OFFSETS.values())
    tRec = synthetic_raw.writeMcap(raw / f"Hpilot_{tSess}" / f"episode_{tEp}" / "sensors.mcap", tN, synthetic_raw.TICKS[tSess], seed=3)
    tManifest, tGrip = {}, {}
    for tEpisode in (EPISODE_OK, EPISODE_PRE_FAIL, EPISODE_NO_RUN):
        for tSide in ("left", "right"):
            tManifest[f"{tEpisode}_{tSide}"] = dict(episode=tEpisode, domain="main_calibrated")
    for tK, tSide in enumerate(("left", "right")):
        tTag = f"{EPISODE_OK}_{tSide}"
        tOff = OFFSETS[tSide]
        tRun = c8 / "runs_m3" / tTag / "ss1.csv"
        M3.writeSide(c8 / "export" / tTag, tRun, SCALE[tSide], 0.3 + 0.5 * tK, tRng, nFrames=N_CORI, frameOffset=tOff,
                     lostFrames=(180 + 3 * tK,))
        if tSide == "left":
            tLines = tRun.read_text().splitlines()
            tCells = tLines[1 + NAN_ROW].split(",")
            tCells[5] = "nan"
            tLines[1 + NAN_ROW] = ",".join(tCells)
            tRun.write_text("\n".join(tLines) + "\n")
        (c8 / "grip").mkdir(parents=True, exist_ok=True)
        np.savez(c8 / "grip" / f"{tTag}.npz", width_m=(0.03 + 0.02 * np.sin(np.arange(N_CORI) / 9.0 + tK)).astype(np.float32),
                 video_frame=np.arange(N_CORI) + tOff)
        tGrip[tTag] = dict(ok=True, offset=tOff)
    (c8 / "c8_domain_manifest.json").write_text(json.dumps(tManifest))
    (c8 / "grip" / "grip_census.json").write_text(json.dumps({"sides": tGrip}))
    (c8 / "census_pre.json").write_text(json.dumps({EPISODE_OK: {"first_fail": None}, EPISODE_PRE_FAIL: {"first_fail": "gripper_valid"},
                                                    EPISODE_NO_RUN: {"first_fail": None}}))
    assert tRec["frames"]["left_wrist"][0][0] == 0
