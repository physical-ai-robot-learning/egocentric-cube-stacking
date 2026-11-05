"""[2026-09-24] Resident variant of run_perframe (C8 execution infrastructure only; algorithm and config unchanged; pose stage).

In:  --list <file of side tags>, --in-root <dir>: per tag <in-root>/<tag> (or <in-root>/<tag with first '_' -> '/'>) holding
     raw_video.mp4 + orbslam_setting.yaml (video-direct mode only)
Out: <out-dir>/<tag>/<name>.{csv,json,calib.yaml} as run_perframe, plus <out-dir>/<tag>[.<name>].log and .vram per side
Usage: python -m mast3r_pose.run_perframe_resident --list <tags> --in-root <dir> --out-dir <dir> [--name ss1]
       [--mem-stop-mib 15872]

run_perframe pays ~108 s of fixed cost per side (python/torch/CUDA start + 2.75 GB checkpoint load + ...). This runner
loads the MASt3R model ONCE (read-only weights) and processes many sides in one process. Everything episode-specific is
rebuilt from scratch for every side, exactly as a fresh run_perframe process would build it:
    config (load_config again), RNG state (restored to the post-model-load state of a fresh process), mp.Manager,
    SharedKeyframes, SharedStates, FrameTracker, and the BACKEND PROCESS (spawned per side: its retrieval database / global
    optimiser live and die with it).
After each side the per-side objects are released and torch.cuda.empty_cache() is called; VRAM is logged (process-tree
peak during the side, and allocated / reserved / GPU-used after cleanup) so a leak across sides is visible.
The per-side body is run_perframe's (the same functions: videoDataset, trackAll, finalRows, writeOutputs), except that the
model is passed in and `runner="resident_v1"` is added to the provenance json. Validation vs run_perframe outputs: c8
contract §21. Importing run_perframe performs its import-time setup (MASt3R-SLAM path, torch.load default, viewer stub),
which the spawned backend process needs as well.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from .run_perframe import (DEVICE, ROOT, M, FrameTracker, Intrinsics, Mode, SharedKeyframes, SharedStates, config, finalRows,
                           load_config, load_mast3r, trackAll, videoDataset, writeOutputs)

RUNNER = "resident_v1"
DEFAULT_NAME = "ss1"
SAMPLE_PERIOD_S = 0.5
MIB = 2 ** 20
OOM_EXIT_CODE = 4
OOM_TEXT = "out of memory"


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--list", required=True)
    tAp.add_argument("--in-root", required=True)
    tAp.add_argument("--out-dir", required=True)
    tAp.add_argument("--name", default="ss1")
    tAp.add_argument("--subsample", type=int, default=1)
    tAp.add_argument("--seed", type=int, default=0)
    tAp.add_argument("--config", default=str(ROOT / "config/base.yaml"))
    tAp.add_argument("--mem-stop-mib", type=int, default=15872)
    return tAp


def treeMib() -> int:
    """sum of nvidia-smi used_memory over this process and all its descendants (was tree_mib)"""
    tPids, tTodo = {os.getpid()}, [os.getpid()]
    while tTodo:
        tPid = tTodo.pop()
        tChildren = subprocess.run(["pgrep", "-P", str(tPid)], capture_output=True, text=True)
        for tChild in tChildren.stdout.split():
            tPids.add(int(tChild))
            tTodo.append(int(tChild))
    tQuery = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True).stdout
    return sum(int(tU) for tP, tU in (tL.split(", ") for tL in tQuery.strip().splitlines() if tL) if int(tP) in tPids)


def gpuUsedMib() -> int:
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])


def runSide(model, video: str, setting: str, out: str, subsample: int, seed: int, configPath: str, rng0: tuple) -> str:
    """one side, exactly as a fresh run_perframe process (was run_side)"""
    # RNG = the state a fresh run_perframe process has when its loop starts: seeded BEFORE load_mast3r (model
    # instantiation consumes RNG), so the post-load state captured once in main is restored here, not a re-seed.
    torch.set_rng_state(rng0[0])
    torch.cuda.set_rng_state_all(rng0[1])
    np.random.set_state(rng0[2])
    load_config(configPath)
    config["single_thread"] = True
    config["dataset"]["subsample"] = subsample
    tManager = mp.Manager()
    tDataset, tCalib = videoDataset(video, setting, out, subsample)
    tH, tW = tDataset.get_img_shape()[0]
    tIntr = yaml.safe_load(open(tCalib))
    config["use_calib"] = True
    tDataset.use_calibration = True
    tDataset.camera_intrinsics = Intrinsics.from_calib(tDataset.img_size, tIntr["width"], tIntr["height"], tIntr["calibration"])
    tKeyframes = SharedKeyframes(tManager, tH, tW)
    tStates = SharedStates(tManager, tH, tW)
    tK = torch.from_numpy(tDataset.camera_intrinsics.K_frame).to(DEVICE, dtype=torch.float32)
    tKeyframes.set_intrinsics(tK)
    tTracker = FrameTracker(model, tKeyframes, DEVICE)
    tBackend = mp.Process(target=M.run_backend, args=(config, model, tStates, tKeyframes, tK))
    tBackend.start()

    tT0 = time.time()
    tLog = trackAll(tDataset, model, tKeyframes, tStates, tTracker, subsample)
    tStates.set_mode(Mode.TERMINATED)
    tBackend.join()
    tElapsed = time.time() - tT0

    tRows, tNkf = finalRows(tLog, tKeyframes)
    tOut = writeOutputs(out, tRows, tNkf, tIntr, subsample, seed, configPath, tElapsed, runner=RUNNER)
    tMsg = f"done {tOut}  frames {len(tRows)} valid {sum(tR[3] for tR in tRows)} kf {tNkf}  {tElapsed:.0f}s"
    del tTracker, tKeyframes, tStates, tBackend, tDataset, tK, tManager, tLog
    return tMsg


def _isComplete(outDir: pathlib.Path, tag: str, name: str) -> bool:
    """SKIP only a COMPLETE resident output, and trust a fresh READDIR of the out dir, not a path stat: on 2026-09-25 the
    5080 NFS client kept stale dentries for 56 files that had been moved to an archive on the server, so a path stat saw
    them and 56 sides were skipped (contract §21). listdir forces a server round trip."""
    tSuffix = "" if name == DEFAULT_NAME else f".{name}"
    tListed = set(os.listdir(outDir))
    tJson = outDir / tag / f"{name}.json"
    return (tag in tListed and f"{tag}{tSuffix}.vram" in tListed and name + ".csv" in os.listdir(outDir / tag) and tJson.is_file()
            and json.load(open(tJson)).get("runner") == RUNNER)


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(tArgs.seed)                                          # as run_perframe, before the model load
    np.random.seed(tArgs.seed)
    tLoadS = time.time()
    tModel = load_mast3r(device=DEVICE)
    tModel.share_memory()
    tLoadS = time.time() - tLoadS
    tRng0 = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state())
    print(f"RESIDENT model loaded once in {tLoadS:.1f}s  gpu used {gpuUsedMib()} MiB", flush=True)
    tIn, tOutDir = pathlib.Path(tArgs.in_root), pathlib.Path(tArgs.out_dir)
    tDone = 0
    tOutDir.mkdir(parents=True, exist_ok=True)
    for tTag in open(tArgs.list).read().split():
        tOut = tOutDir / tTag / f"{tArgs.name}.csv"
        if _isComplete(tOutDir, tTag, tArgs.name):
            print(f"SKIP {tTag}", flush=True)
            continue
        tDir = tIn / tTag if (tIn / tTag).is_dir() else tIn / tTag.replace("_", "/", 1)
        tPeak = [0]
        tStop = threading.Event()

        def tSampler():
            while not tStop.is_set():
                try:
                    tPeak[0] = max(tPeak[0], treeMib())
                except Exception:
                    pass
                time.sleep(SAMPLE_PERIOD_S)

        tThread = threading.Thread(target=tSampler, daemon=True)
        tThread.start()
        tT0 = time.time()
        # per-side log file <out-dir>/<tag>[.<name>].log, like the per-process runner: fds 1/2 point at it while the side
        # runs, so the spawned backend (inherits them at spawn) logs there too. Logging only; no effect on computation.
        tSuffix = "" if tArgs.name == DEFAULT_NAME else f".{tArgs.name}"
        sys.stdout.flush()
        sys.stderr.flush()
        tLogFile = open(tOutDir / f"{tTag}{tSuffix}.log", "w")
        tSavedOut, tSavedErr = os.dup(1), os.dup(2)
        os.dup2(tLogFile.fileno(), 1)
        os.dup2(tLogFile.fileno(), 2)
        try:
            tMsg = runSide(tModel, str(tDir / "raw_video.mp4"), str(tDir / "orbslam_setting.yaml"), str(tOut), tArgs.subsample,
                           tArgs.seed, tArgs.config, tRng0)
            tRc = 0
        except Exception as tEx:
            traceback.print_exc()
            tMsg = f"ERROR {type(tEx).__name__}: {tEx}"
            tRc = 1
        print(tMsg)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(tSavedOut, 1)
        os.dup2(tSavedErr, 2)
        os.close(tSavedOut)
        os.close(tSavedErr)
        tLogFile.close()
        tStop.set()
        tThread.join()
        gc.collect()
        torch.cuda.empty_cache()
        tWall = time.time() - tT0
        tPost = dict(alloc_MiB=torch.cuda.memory_allocated() // MIB, reserved_MiB=torch.cuda.memory_reserved() // MIB,
                     gpu_used_MiB=gpuUsedMib(), tree_MiB=treeMib())
        (tOutDir / f"{tTag}{tSuffix}.vram").write_text(
            f"tree_peak_MiB {tPeak[0]} post_tree_MiB {tPost['tree_MiB']} post_alloc_MiB {tPost['alloc_MiB']} "
            f"post_reserved_MiB {tPost['reserved_MiB']} post_gpu_MiB {tPost['gpu_used_MiB']} rc {tRc} sec {int(tWall)}\n")
        print(f"{'OK' if tRc == 0 else 'FAIL'} {tTag} {tMsg}  vram tree peak {tPeak[0]} | after cleanup tree {tPost['tree_MiB']} "
              f"alloc {tPost['alloc_MiB']} reserved {tPost['reserved_MiB']} gpu {tPost['gpu_used_MiB']} MiB  {tWall:.0f}s",
              flush=True)
        if tRc or tPeak[0] > tArgs.mem_stop_mib or OOM_TEXT in (tOutDir / f"{tTag}{tSuffix}.log").read_text().lower():
            print(f"STOP {tTag} rc {tRc} peak {tPeak[0]}", flush=True)
            return OOM_EXIT_CODE
        tDone += 1
    print(f"RESIDENT_DONE {tDone}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
