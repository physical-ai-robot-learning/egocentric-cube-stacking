"""[2026-09-21] Phase U0-B: put a HandUMI episode into the layout UMI's ORB-SLAM3 stage expects (pose stage; was
scripts/umi_export_orbslam.py).

In:  a raw HandUMI episode directory (handumi_collector RawEpisode), one side, the frozen Kalibr calibration in
     ego_paths.CALIBRATION_DIR (default <repo>/configs/calibration, env EGO_CALIBRATION_DIR).
Out: <out>/raw_video.mp4 + <out>/imu_data.json, plus <out>/orbslam_setting.yaml built from that calibration. The same
     layout is the input of mast3r_pose.run_perframe(_resident) and of the IMU-VI metric scale.
Usage: python -m mast3r_pose.export_episode <episode dir> {left,right} <out dir> [--downscale 2] [--start-at-go]

Nothing here is invented: the JSON schema was read out of gopro_slam.cc's LoadTelemetry, the extrinsic convention from the
exporter that MASt3R already ran on, and the clock from the same fit_device_to_host / imu_host_times_ns the production
pipeline uses -- so the camera-IMU time offset sign is never re-decided here.

Two details that are easy to get wrong and are handled explicitly:
  * gopro_slam takes the VIDEO FRAME times from the CORI stream, not from the container. CORI therefore gets
    one sample per frame at that frame's capture time.
  * it shifts the IMU series so the first ACCL sample is t=0 but leaves CORI as it is, so both streams are
    written on one clock whose origin IS the first IMU sample.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import yaml

from ego_paths import CALIBRATION_DIR
from handumi_collector.pose.episode_io import RawEpisode
from handumi_collector.pose.timing import fit_device_to_host, imu_host_times_ns

CAL = CALIBRATION_DIR
FISHEYE = {"left": "fisheye_left_v002.yaml", "right": "fisheye_right_v003.yaml"}
CAMIMU = {"left": "camera_imu_left_v001.yaml", "right": "camera_imu_right_v002.yaml"}
STREAM = {"left": "left_wrist", "right": "right_wrist"}
VIDEO_FPS = 30.0
SPLICE_DT_MS = 1000.0 / 59.97          # what the UMI fork's INIT_RELOCALIZE splice assumes (one GoPro frame)
NS_PER_MS = 1e6
NS_PER_S = 1e9
CORI_IDENTITY = [1.0, 0.0, 0.0, 0.0]
ORB_SETTINGS = [
    "ORBextractor.nFeatures: 1500", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
    "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7", "",
    "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
    "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
    "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -3.5",
    "Viewer.ViewpointF: 500.0", "",
]


class GoCueError(RuntimeError):
    """--start-at-go was asked for an episode whose events.json has no go_cue."""


def settingsYaml(side: str, width: int, height: int, intrinsics: np.ndarray, distortion, cameraImu, noise: dict,
                 fps: float) -> str:
    """ORB-SLAM3 settings (was settings_yaml). IMU.T_b_c1 is the camera->body(IMU) transform, i.e. inv(T_camera_imu):
    our bundle stores the IMU expressed in the camera frame, which is what the validated EuRoC exporter
    inverted to get Tic. Do not re-derive this direction."""
    tTbc = np.linalg.inv(np.asarray(cameraImu, float))
    tD = np.asarray(distortion, float).ravel()
    tK = intrinsics
    tBody = [
        "%YAML:1.0", "",
        f"# HandUMI {side} wrist, generated {pathlib.Path(__file__).name}",
        f"# intrinsics {FISHEYE[side]}  extrinsic/noise {CAMIMU[side]}  (frozen Kalibr, do not edit by hand)",
        'File.version: "1.0"', 'Camera.type: "KannalaBrandt8"', "",
        f"Camera1.fx: {tK[0,0]:.6f}", f"Camera1.fy: {tK[1,1]:.6f}",
        f"Camera1.cx: {tK[0,2]:.6f}", f"Camera1.cy: {tK[1,2]:.6f}", "",
        f"Camera1.k1: {tD[0]:.9f}", f"Camera1.k2: {tD[1]:.9f}",
        f"Camera1.k3: {tD[2]:.9f}", f"Camera1.k4: {tD[3]:.9f}", "",
        # ORB-SLAM3 aborts on a non-integer Camera.fps
        f"Camera.width: {width}", f"Camera.height: {height}", f"Camera.fps: {int(round(fps))}", "Camera.RGB: 1", "",
        "# camera -> body(IMU) = inv(T_camera_imu)",
        "IMU.T_b_c1: !!opencv-matrix", "    rows: 4", "    cols: 4", "    dt: f",
        "    data: [" + ", ".join(f"{tV:.10f}" for tV in tTbc.reshape(-1)) + "]", "",
        f"IMU.NoiseGyro: {noise['gyroscope_noise_density']:.8f}",
        f"IMU.NoiseAcc: {noise['accelerometer_noise_density']:.8f}",
        f"IMU.GyroWalk: {noise['gyroscope_random_walk']:.10f}",
        f"IMU.AccWalk: {noise['accelerometer_random_walk']:.10f}",
        f"IMU.Frequency: {noise['update_rate']:.1f}", "",
    ] + ORB_SETTINGS
    return "\n".join(tBody)


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("episode")
    tAp.add_argument("side", choices=["left", "right"])
    tAp.add_argument("out")
    tAp.add_argument("--downscale", type=int, default=2)
    # [2026-09-21] Every episode opens with a 3 s HOLD STILL. That window exists so OpenVINS could do a
    # static init, but ORB-SLAM3 monocular-inertial needs MOTION to initialise the IMU, and on a loaded map
    # it reacts to an uninitialised IMU by resetting the active map -- throwing away a relocalisation it had
    # already achieved ("Relocalized!!" then "IMU is not or recently initialized. Reseting active map").
    # --start-at-go drops everything before the go_cue so the sequence begins moving.
    tAp.add_argument("--start-at-go", action="store_true")
    # [2026-09-21] IMU PREROLL. UMI's ORB-SLAM3 fork splices a new sequence onto a loaded atlas by rewriting
    # the loaded keyframes' timestamps so the last one sits at -dt, then preintegrating IMU across that gap
    # (Tracking.cc, INIT_RELOCALIZE). dt there is hardcoded to 1/59.97 = 16.68 ms -- a GoPro frame -- and a
    # shadowed local `double dt` means the real mpImuPreintegrated->dT is never read. So the splice always
    # asks for ~17 ms of IMU BEFORE the first camera frame. Our exports had the opposite: the first camera
    # frame preceded the first IMU sample by 7-35 ms, leaving that window empty. Keep real IMU samples from
    # preroll ms before the first kept frame; never fabricate samples, never shift the IMU clock (that would
    # break the frozen camera-IMU calibration).
    tAp.add_argument("--imu-preroll-ms", type=float, default=50.0)
    return tAp


def goCueNs(episodeDir: pathlib.Path) -> int:
    """capture time of the first go_cue event; GoCueError when there is none"""
    tEvents = json.loads((episodeDir / "events.json").read_text())
    tGo = [tE for tE in tEvents if tE.get("kind") == "go_cue"]
    if not tGo:
        raise GoCueError("no go_cue in events.json; cannot --start-at-go")
    print(f"  trimming everything before go_cue at t_rel {tGo[0].get('t_rel_s'):.2f} s")
    return int(tGo[0]["t_ns"])


def writeVideo(ep, side: str, out: pathlib.Path, width: int, height: int, downscale: int, tGo: int | None) -> np.ndarray:
    """raw_video.mp4 and its frame capture times in ONE pass: RawEpisode.iter_frames yields (video_frame, index, capture_ns,
    image), so the timestamps written into CORI are the same frames written into the mp4 (reading them separately invites
    an off-by-one between the two)."""
    tWriter, tCam = None, []
    for _, _, tCapNs, tImg in ep.iter_frames(STREAM[side], downscale=downscale):
        if tGo is not None and int(tCapNs) < tGo:
            continue
        if tWriter is None:
            tH, tW = tImg.shape[:2]
            assert (tW, tH) == (width, height), f"frame {tW}x{tH} != settings {width}x{height}"
            tWriter = cv2.VideoWriter(str(out / "raw_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height))
        tWriter.write(tImg if tImg.ndim == 3 else cv2.cvtColor(tImg, cv2.COLOR_GRAY2BGR))
        tCam.append(float(tCapNs))
    if tWriter is not None:
        tWriter.release()
    return np.asarray(tCam, float)


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tEpDir = pathlib.Path(tArgs.episode).expanduser()
    tOut = pathlib.Path(tArgs.out).expanduser()
    tOut.mkdir(parents=True, exist_ok=True)
    tEp = RawEpisode.load(tEpDir)
    tFisheye = yaml.safe_load(open(CAL / FISHEYE[tArgs.side]))
    tCamImu = yaml.safe_load(open(CAL / CAMIMU[tArgs.side]))
    tK = np.array(tFisheye["K"], float) / tArgs.downscale
    tK[2, 2] = 1.0
    tW, tH = tFisheye["image_size"][0] // tArgs.downscale, tFisheye["image_size"][1] // tArgs.downscale

    try:
        tGo = goCueNs(tEpDir) if tArgs.start_at_go else None
    except GoCueError as tErr:
        print(tErr, file=sys.stderr)
        return 1
    tCam = writeVideo(tEp, tArgs.side, tOut, tW, tH, tArgs.downscale, tGo)
    tNFrames = len(tCam)

    # --- IMU and frame times on ONE clock, via the production clock model
    tImu = tEp.imu[tArgs.side]
    tClock = fit_device_to_host(tImu.device_us, tImu.host_ns)
    tOffNs = int(float(tCamImu["time_offset_ms"]) * NS_PER_MS)
    tImuNs = imu_host_times_ns(tImu, tClock, tOffNs)
    tOrder = np.argsort(tImuNs, kind="stable")
    tImuNs = tImuNs[tOrder]
    tAcc = np.asarray(tImu.accel, float)[tOrder]
    tGyr = np.asarray(tImu.gyro, float)[tOrder]
    tKeep = np.r_[True, np.diff(tImuNs) > 0]
    tImuNs, tAcc, tGyr = tImuNs[tKeep], tAcc[tKeep], tGyr[tKeep]
    tPreNs = int(tArgs.imu_preroll_ms * NS_PER_MS)
    tLo = float(tCam[0]) - tPreNs
    if float(tImuNs[0]) > tLo:                      # IMU cannot reach back far enough: drop leading frames
        tLo = float(tImuNs[0])
        tKeepCam = tCam >= tLo + tPreNs
        tDropped = int((~tKeepCam).sum())
        if tDropped:
            print(f"  dropping {tDropped} leading frames so the IMU leads the camera by {tArgs.imu_preroll_ms:.0f} ms")
        tCam = tCam[tKeepCam]
        tNFrames = len(tCam)
    tSel = tImuNs >= tLo
    tImuNs, tAcc, tGyr = tImuNs[tSel], tAcc[tSel], tGyr[tSel]
    tT0 = float(tImuNs[0])                       # gopro_slam re-bases the IMU to its first sample; match CORI to it

    def tMs(tNs):
        return (np.asarray(tNs, float) - tT0) / NS_PER_MS

    tTelemetry = {"1": {"streams": {
        "ACCL": {"samples": [{"cts": float(tC), "value": [float(tV) for tV in tA3]} for tC, tA3 in zip(tMs(tImuNs), tAcc)]},
        "GYRO": {"samples": [{"cts": float(tC), "value": [float(tV) for tV in tG3]} for tC, tG3 in zip(tMs(tImuNs), tGyr)]},
        "CORI": {"samples": [{"cts": float(tC), "value": list(CORI_IDENTITY)} for tC in tMs(tCam)[:tNFrames]]},
    }}}
    (tOut / "imu_data.json").write_text(json.dumps(tTelemetry))
    tLead = tMs(tCam)[0] - tMs(tImuNs)[0]
    print(f"  first_imu 0.00 ms | first_camera {tMs(tCam)[0]:.2f} ms | camera-imu lead {tLead:.2f} ms "
          f"| splice needs >= {SPLICE_DT_MS:.2f} ms -> {'OK' if tLead >= SPLICE_DT_MS else 'TOO SHORT'}")
    (tOut / "orbslam_setting.yaml").write_text(
        settingsYaml(tArgs.side, tW, tH, tK, tFisheye["D"], tCamImu["T_camera_imu"], tCamImu["imu_noise"], VIDEO_FPS))
    print(f"{tEpDir.name} {tArgs.side}: {tNFrames} frames {tW}x{tH} @ {VIDEO_FPS:.1f} | imu {len(tImuNs)} "
          f"@ {NS_PER_S/np.median(np.diff(tImuNs)):.1f} Hz (+{tCamImu['time_offset_ms']} ms) | cori {min(len(tCam), tNFrames)}")
    print(f"  span imu {tMs(tImuNs)[-1]/1000:.2f} s   cam {tMs(tCam)[:tNFrames][-1]/1000:.2f} s  -> {tOut}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
