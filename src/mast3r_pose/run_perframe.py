"""[2026-09-23] trackA_mast3r_pose_v1, step 2 (GPU, m3slam env): official MASt3R-SLAM with PER-FRAME poses (pose stage).

In:  --video raw_video.mp4 + --setting orbslam_setting.yaml (fisheye, undistorted in memory), or --frames + --calib
Out: --out <name>.csv (ORB-SLAM3 export columns + kf_ref, sim3_scale, match_frac, mode), <name>.json (provenance),
     <name>.calib.yaml (pinhole intrinsics of the undistorted video)
Usage: python -m mast3r_pose.run_perframe --video <dir>/raw_video.mp4 --setting <dir>/orbslam_setting.yaml --out <dir>/ss1.csv

The official main.py saves only KEYFRAME poses (evaluate.save_traj). This wrapper runs the same loop (imports the
official modules; the repo is not modified) and records, per processed frame, the reference keyframe and the
tracked pose, then re-expresses every tracked frame on the FINAL optimised keyframe pose:

    T_WC_final(f) = T_WK_final(k) * T_WK_at_track(k)^-1 * T_WC_at_track(f)        (Sim3, lietorch)

Validity: INIT frame and successfully tracked frames are valid; a frame whose tracking failed (low match fraction /
Cholesky -> RELOC) or that was processed in RELOC mode is INVALID (its pose is only the propagated guess), unless
it later became a keyframe through relocalisation, in which case it takes the final keyframe pose.
single_thread = True (deterministic ordering with the backend). Pose = T_WC (camera -> world), t in world units,
quaternion xyzw -- verified by test_convention.py, not assumed.

Work at import time is deliberate here: the backend runs in a SPAWNED process that re-imports this module, and it needs the
MASt3R-SLAM checkout on sys.path (M3_REPO), the torch.load default and the viewer stub before it unpickles main.run_backend.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import types

from ego_paths import HOME, prepend_sys_path

ROOT = pathlib.Path(os.environ.get("M3_REPO", HOME / "mast3r_slam_official"))
prepend_sys_path(str(ROOT))

import lietorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

# torch >= 2.6 defaults torch.load(weights_only=True); the official naver checkpoints pickle an argparse.Namespace.
# They are trusted (md5 in checkpoints/MD5SUMS), so restore the pre-2.6 default here -- module top level, so the
# spawned backend process gets it too. Recorded in the run provenance.
torch.load = functools.partial(torch.load, weights_only=False)
import torch.multiprocessing as mp  # noqa: E402
from mast3r_slam.config import config, load_config  # noqa: E402
from mast3r_slam.dataloader import Intrinsics, load_dataset  # noqa: E402
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame  # noqa: E402
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono  # noqa: E402
from mast3r_slam.tracker import FrameTracker  # noqa: E402

# main.py imports the in3d/imgui viewer at module level; we never open it (--no-viz equivalent), so a stub
# module stands in for mast3r_slam.visualization instead of patching the official repo.
_viz = types.ModuleType("mast3r_slam.visualization")


class _WindowMsg:
    is_terminated = False
    is_paused = False
    next = False
    C_conf_threshold = 1.5


_viz.WindowMsg = _WindowMsg
_viz.run_visualization = lambda *a, **k: None
sys.modules["mast3r_slam.visualization"] = _viz
import main as M  # noqa: E402  official run_backend / relocalization

SOURCE_FPS = 30.0
POLL_S = 0.01
DEVICE = "cuda:0"
CSV_HEADER = "frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w,map_updated,kf_ref,sim3_scale,match_frac\n"


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--frames", default=None)
    tAp.add_argument("--calib", default=None)
    tAp.add_argument("--video", default=None,
                     help="C8: fisheye raw_video.mp4, undistorted in memory with the SAME fisheye_to_pinhole_v1 "
                          "as prep_undistort.py (no PNG on disk)")
    tAp.add_argument("--setting", default=None, help="C8: orbslam_setting.yaml next to --video (KannalaBrandt8 intrinsics)")
    tAp.add_argument("--subsample", type=int, default=1)
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--config", default=str(ROOT / "config/base.yaml"))
    tAp.add_argument("--seed", type=int, default=0)
    return tAp


def videoDataset(video: str, setting: str, out: str, subsample: int):
    """fisheye video -> in-memory undistorted MonocularDataset (identical to the PNGs prep_undistort.py writes) and the
    pinhole calib yaml written next to `out`; -> (dataset, calib path)"""
    import cv2
    from mast3r_slam.dataloader import MonocularDataset
    tText = pathlib.Path(setting).read_text()

    def tGet(tKey):
        return float(re.search(rf"^{tKey}:\s*([-\d.eE]+)", tText, re.M).group(1))

    tK = np.array([[tGet("Camera1.fx"), 0, tGet("Camera1.cx")], [0, tGet("Camera1.fy"), tGet("Camera1.cy")], [0, 0, 1.0]])
    tD = np.array([tGet(f"Camera1.k{tI}") for tI in (1, 2, 3, 4)]).reshape(4, 1)
    tW, tH = int(tGet("Camera.width")), int(tGet("Camera.height"))
    tP = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(tK, tD, (tW, tH), np.eye(3), balance=0.0)
    tMx, tMy = cv2.fisheye.initUndistortRectifyMap(tK, tD, np.eye(3), tP, (tW, tH), cv2.CV_32FC1)
    tCap = cv2.VideoCapture(video)
    tFrames = []
    while True:
        tOk, tFrame = tCap.read()
        if not tOk:
            break
        tFrames.append(cv2.remap(tFrame, tMx, tMy, cv2.INTER_LINEAR))        # identical to the PNG written by prep_undistort.py
    tCap.release()

    class _VideoUndistorted(MonocularDataset):
        def __init__(self):
            super().__init__()
            self.rgb_files = list(range(len(tFrames)))
            self.timestamps = list(np.arange(len(tFrames), dtype=np.float32) / SOURCE_FPS)

        def read_img(self, idx):
            return cv2.cvtColor(tFrames[self.rgb_files[idx]], cv2.COLOR_BGR2RGB)

    tDataset = _VideoUndistorted()
    tDataset.subsample(subsample)
    tCalib = str(pathlib.Path(out).with_suffix(".calib.yaml"))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(tCalib).write_text("width: %d\nheight: %d\ncalibration: [%.6f, %.6f, %.6f, %.6f]\n"
                                    % (tW, tH, tP[0, 0], tP[1, 1], tP[0, 2], tP[1, 2]))
    return tDataset, tCalib


def _waitUntil(states, done) -> None:
    while True:
        with states.lock:
            if done():
                break
        time.sleep(POLL_S)


def trackAll(dataset, model, keyframes, states, tracker, subsample: int) -> list[dict]:
    """the official tracking loop (single thread), one record per processed frame"""
    tLog = []
    for tI in range(len(dataset)):
        tMode = states.get_mode()
        _, tImg = dataset[tI]
        tTwc = lietorch.Sim3.Identity(1, device=DEVICE) if tI == 0 else states.get_frame().T_WC
        tFrame = create_frame(tI, tImg, tTwc, img_size=dataset.img_size, device=DEVICE)
        tRec = dict(i=tI, src=tI * subsample, mode=tMode.name, valid=False, kf_ref=-1, is_kf=False, match_frac=float("nan"))
        tAddKf = False
        if tMode == Mode.INIT:
            tX, tC = mast3r_inference_mono(model, tFrame)
            tFrame.update_pointmap(tX, tC)
            keyframes.append(tFrame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(tFrame)
            tRec.update(valid=True, is_kf=True, kf_ref=0, T=tFrame.T_WC.data.cpu().numpy().ravel().tolist(),
                        T_kf=keyframes.T_WC[0].cpu().numpy().ravel().tolist())
            tLog.append(tRec)
            continue
        if tMode == Mode.TRACKING:
            tKref = len(keyframes) - 1
            tTkfNow = keyframes.T_WC[tKref].detach().cpu().numpy().ravel().tolist()
            tAddKf, tMatchInfo, tTryReloc = tracker.track(tFrame)
            if tTryReloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(tFrame)
            tRec.update(valid=not tTryReloc, kf_ref=tKref, T=tFrame.T_WC.data.detach().cpu().numpy().ravel().tolist(),
                        T_kf=tTkfNow)
            if isinstance(tMatchInfo, (list, tuple)) and len(tMatchInfo):
                try:
                    tRec["match_frac"] = float(tMatchInfo[0])
                except Exception:
                    pass
        elif tMode == Mode.RELOC:
            tX, tC = mast3r_inference_mono(model, tFrame)
            tFrame.update_pointmap(tX, tC)
            states.set_frame(tFrame)
            states.queue_reloc()
            _waitUntil(states, lambda: states.reloc_sem.value == 0)
        if tAddKf:
            keyframes.append(tFrame)
            states.queue_global_optimization(len(keyframes) - 1)
            tRec["is_kf"] = True
            _waitUntil(states, lambda: len(states.global_optimizer_tasks) == 0)
        tLog.append(tRec)
    return tLog


def finalRows(log: list[dict], keyframes) -> tuple[list[tuple], int]:
    """re-express every valid frame on its final optimised keyframe -> (csv rows, number of keyframes)"""
    tNkf = len(keyframes)
    tKfFinal = {tK: keyframes.T_WC[tK].detach().cpu() for tK in range(tNkf)}
    tFrameKf = {int(keyframes[tK].frame_id): tK for tK in range(tNkf)}
    tRows = []
    for tR in log:
        tT = None
        if tR["i"] in tFrameKf:                                        # became a keyframe (incl. via relocalisation)
            tT = lietorch.Sim3(tKfFinal[tFrameKf[tR["i"]]])
            tR["valid"] = True
            tR["is_kf"] = True
        elif tR["valid"]:
            tT = (lietorch.Sim3(tKfFinal[tR["kf_ref"]]) * lietorch.Sim3(torch.tensor(tR["T_kf"]).view(1, 8)).inv()
                  * lietorch.Sim3(torch.tensor(tR["T"]).view(1, 8)))
        tD = tT.data.view(-1).numpy() if tT is not None else np.zeros(8)
        tRows.append((tR["src"], tR["src"] / SOURCE_FPS, tR["mode"], tR["valid"], tR["is_kf"], tD, tR["kf_ref"],
                      tR["match_frac"]))
    return tRows, tNkf


def writeOutputs(out: str, rows: list[tuple], nKeyframes: int, intr: dict, subsample: int, seed: int, configPath: str,
                 elapsedS: float, runner: str | None = None) -> pathlib.Path:
    """<out>.csv (ORB-SLAM3 export columns + extras) and <out>.json provenance; runner is added only when given"""
    tOut = pathlib.Path(out)
    tOut.parent.mkdir(parents=True, exist_ok=True)
    with open(tOut, "w") as tFh:
        tFh.write(CSV_HEADER)
        for tSrc, tTs, tMode, tValid, tIsKf, tD, tKref, tMf in rows:
            tFh.write(f"{tSrc},{tTs:.6f},{tMode},{'false' if tValid else 'true'},{'true' if tIsKf else 'false'},"
                      f"{tD[0]:.7f},{tD[1]:.7f},{tD[2]:.7f},{tD[3]:.7f},{tD[4]:.7f},{tD[5]:.7f},{tD[6]:.7f},"
                      f"0,{tKref},{tD[7]:.6f},{tMf:.4f}\n")
    tCk = sorted((ROOT / "checkpoints").glob("*"))
    tRunner = {} if runner is None else dict(runner=runner)
    json.dump(dict(pose_source="mast3r_slam", variant="rgb_only_calibrated", repo=str(ROOT),
                   build_patch="build_compat_5090.patch (compile-only)", torch_load_weights_only=False, **tRunner,
                   git_commit=(ROOT / "INSTALLED_COMMIT").read_text().strip(),
                   checkpoints={tC.name: tC.stat().st_size for tC in tCk},
                   calibrated=True, calib=intr, intrinsic_hash=hashlib.md5(json.dumps(intr["calibration"]).encode()).hexdigest(),
                   subsample=subsample, source_fps=30, seed=seed, single_thread=True, config=configPath,
                   pose_convention="T_WC camera->world, Sim3 re-expressed on final keyframes, t xyz + q xyzw "
                                   "(test_convention.py)",
                   scale="monocular Sim3: world units are MASt3R's metric-prior scale, NOT guaranteed metres (checked vs ArUco)",
                   n_frames=len(rows), n_valid=int(sum(tR[3] for tR in rows)), n_keyframes=nKeyframes, seconds=elapsedS),
              open(tOut.with_suffix(".json"), "w"), indent=1)
    return tOut


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    mp.set_start_method("spawn")
    torch.manual_seed(tArgs.seed)
    np.random.seed(tArgs.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_config(tArgs.config)
    config["single_thread"] = True
    config["dataset"]["subsample"] = tArgs.subsample
    tManager = mp.Manager()
    if tArgs.video:
        tDataset, tArgs.calib = videoDataset(tArgs.video, tArgs.setting, tArgs.out, tArgs.subsample)
    else:
        tDataset = load_dataset(tArgs.frames)
        tDataset.subsample(tArgs.subsample)
    tH, tW = tDataset.get_img_shape()[0]
    tIntr = yaml.safe_load(open(tArgs.calib))
    config["use_calib"] = True
    tDataset.use_calibration = True
    tDataset.camera_intrinsics = Intrinsics.from_calib(tDataset.img_size, tIntr["width"], tIntr["height"], tIntr["calibration"])
    tKeyframes = SharedKeyframes(tManager, tH, tW)
    tStates = SharedStates(tManager, tH, tW)
    tModel = load_mast3r(device=DEVICE)
    tModel.share_memory()
    tK = torch.from_numpy(tDataset.camera_intrinsics.K_frame).to(DEVICE, dtype=torch.float32)
    tKeyframes.set_intrinsics(tK)
    tTracker = FrameTracker(tModel, tKeyframes, DEVICE)
    tBackend = mp.Process(target=M.run_backend, args=(config, tModel, tStates, tKeyframes, tK))
    tBackend.start()

    tT0 = time.time()
    tLog = trackAll(tDataset, tModel, tKeyframes, tStates, tTracker, tArgs.subsample)
    tStates.set_mode(Mode.TERMINATED)
    tBackend.join()
    tElapsed = time.time() - tT0

    tRows, tNkf = finalRows(tLog, tKeyframes)
    tOut = writeOutputs(tArgs.out, tRows, tNkf, tIntr, tArgs.subsample, tArgs.seed, tArgs.config, tElapsed)
    print(f"done {tOut}  frames {len(tRows)} valid {sum(tR[3] for tR in tRows)} kf {tNkf}  {tElapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
