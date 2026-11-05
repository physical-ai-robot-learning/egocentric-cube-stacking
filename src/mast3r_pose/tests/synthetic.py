"""Synthetic MASt3R + wrist-IMU benchmark tree in the layout the metric-scale fit reads (M3_BASE):

    <base>/qa/qa_all.json                         per tag: n_src_frames, M3_<variant>.marker.lb_scale (the ArUco GT)
    <base>/handumi_camera_tcp_v2.yaml             camera -> TCP
    <base>/val/<split>/<demo>_<side>/imu_data.json    CORI (camera frame times) + ACCL (200 Hz) in ms
    <base>/val/<split>/<demo>_<side>/orbslam_setting.yaml   IMU.T_b_c1
    <base>/runs/<tag>/<variant>.csv               per-frame camera pose in world units (= metres / true scale)

The accelerometer is generated from the metric trajectory of the IMU (camera + lever arm) and gravity, so the IMU-VI fit
has a known answer (TRUE_SCALE per tag)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CAMERA_HZ = 30.0
IMU_HZ = 200.0
N_FRAMES = 240
GRAVITY = np.array([0.0, 0.0, -9.81])
FINITE_DIFF_S = 1e-4
TAGS = {"valB_d1_left": 0.8, "valB_d1_right": 1.1, "valB_d2_left": 0.5, "valB_d2_right": 2.0,
        "valC_d3_left": 0.9, "valC_d3_right": 1.4}
GT_ERROR = {"valB_d1_left": 1.02, "valB_d1_right": 0.97, "valB_d2_left": 1.0, "valB_d2_right": 1.05,
            "valC_d3_left": 0.99, "valC_d3_right": 1.01}
LOST_FRAMES = (50, 51, 52, 180)
R_CAMERA_TCP = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
T_CAMERA_TCP_M = [0.01, -0.03, 0.12]


def _camMetric(tS: np.ndarray, phase: float) -> np.ndarray:
    return np.stack([0.15 * np.sin(1.3 * tS + phase), 0.1 * np.sin(0.9 * tS), 0.08 * np.cos(1.7 * tS + phase)], -1)


def _camRot(tS: np.ndarray, phase: float) -> Rotation:
    return Rotation.from_rotvec(np.stack([0.3 * np.sin(0.7 * tS + phase), 0.2 * np.cos(0.5 * tS), 0.1 * tS], -1))


def tBodyCamera(phase: float) -> np.ndarray:
    """IMU.T_b_c1 (camera -> body) of one side"""
    tT = np.eye(4)
    tT[:3, :3] = Rotation.from_rotvec([0.15, 0.05 + phase / 10, -0.1]).as_matrix()
    tT[:3, 3] = [0.012, -0.02, 0.005]
    return tT


def settingYaml(tBc: np.ndarray) -> str:
    return ("%YAML:1.0\nCamera1.fx: 300.0\nIMU.T_b_c1: !!opencv-matrix\n    rows: 4\n    cols: 4\n    dt: f\n"
            "    data: [" + ", ".join(f"{v:.10f}" for v in tBc.reshape(-1)) + "]\n")


def writeSide(sideDir: Path, runCsv: Path, scale: float, phase: float, rng: np.random.Generator, nFrames: int = N_FRAMES,
              frameOffset: int = 0, lostFrames=LOST_FRAMES) -> None:
    """imu_data.json + orbslam_setting.yaml of one side into sideDir, its MASt3R csv (frame_idx = CORI index + frameOffset)"""
    sideDir.mkdir(parents=True, exist_ok=True)
    tBc = tBodyCamera(phase)
    (sideDir / "orbslam_setting.yaml").write_text(settingYaml(tBc))
    tCb = np.linalg.inv(tBc)
    tCamS = np.arange(nFrames) / CAMERA_HZ + 0.004
    tImuS = np.arange(int(nFrames / CAMERA_HZ * IMU_HZ) + 10) / IMU_HZ - 0.02

    def tImuPos(tS):
        return _camMetric(tS, phase) + _camRot(tS, phase).apply(tCb[:3, 3])

    tAcc = (tImuPos(tImuS + FINITE_DIFF_S) - 2 * tImuPos(tImuS) + tImuPos(tImuS - FINITE_DIFF_S)) / FINITE_DIFF_S ** 2
    tRwb = _camRot(tImuS, phase).as_matrix() @ tCb[:3, :3]
    tF = np.einsum("nji,nj->ni", tRwb, tAcc - GRAVITY) + rng.normal(0, 0.05, (len(tImuS), 3)) + [0.02, -0.01, 0.03]
    tTelemetry = {"1": {"streams": {
        "ACCL": {"samples": [{"cts": float(c * 1e3), "value": v.tolist()} for c, v in zip(tImuS, tF)]},
        "CORI": {"samples": [{"cts": float(c * 1e3), "value": [1.0, 0.0, 0.0, 0.0]} for c in tCamS]}}}}
    (sideDir / "imu_data.json").write_text(json.dumps(tTelemetry))
    tPos = _camMetric(tCamS, phase) / scale + rng.normal(0, 2e-4, (nFrames, 3))
    tQuat = _camRot(tCamS, phase).as_quat()
    runCsv.parent.mkdir(parents=True, exist_ok=True)
    tLines = ["frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w"]
    for tI in range(-frameOffset, nFrames):
        tJ = max(tI, 0)
        tLost = "true" if tI in lostFrames else "false"
        tLines.append(f"{tI + frameOffset},{(tI + frameOffset) / CAMERA_HZ:.6f},TRACKING,{tLost},false,"
                      + ",".join(f"{v:.9f}" for v in (*tPos[tJ], *tQuat[tJ])))
    runCsv.write_text("\n".join(tLines) + "\n")


def writeCameraTcp(path: Path) -> None:
    path.write_text(yaml.safe_dump({"rotation": {"R_camera_tcp": R_CAMERA_TCP.tolist()}, "translation": {"t_camera_tcp_m": T_CAMERA_TCP_M}}))


def writeBenchmark(base: Path, variant: str = "ss1", seed: int = 3) -> Path:
    tRng = np.random.default_rng(seed)
    (base / "qa").mkdir(parents=True, exist_ok=True)
    writeCameraTcp(base / "handumi_camera_tcp_v2.yaml")
    tQa = {}
    for tIdx, (tTag, tScale) in enumerate(TAGS.items()):
        tSplit, tRest = tTag.split("_", 1)
        writeSide(base / "val" / tSplit / tRest, base / "runs" / tTag / f"{variant}.csv", tScale, 0.4 * tIdx, tRng)
        tQa[tTag] = {"n_src_frames": N_FRAMES, f"M3_{variant}": {"marker": {"lb_scale": tScale * GT_ERROR[tTag]}}}
    tQa["valB_nomarker_left"] = {"n_src_frames": N_FRAMES, f"M3_{variant}": {"marker": {}}}
    (base / "qa" / "qa_all.json").write_text(json.dumps(tQa))
    return base
