"""processed ego_cart20_v2 -> LeRobot v3 datasets <out>/<name>_{train,val} for the X-VLA REL-only D6 recipe
(umi_bridge/RELCART20_RELONLY_D6_RECIPE.md section 9).

Per LeRobot episode = one processed ego episode; frames = its TRAINING rows only (rows with all 16 real targets and a head
frame), in row order.  Columns:
  observation.images.{global,left_wrist,right_wrist}   head / left wrist / right wrist, full frame INTER_AREA 224x224
                                                        (the existing ego REL16 exporter policy); a missing wrist frame is black
  observation.state (20)       state.npy = RELCART20 task anchor [L9 | R9 | gL gR]       (default; NEVER state_prevrel)
  action (16, 32)              action.npy = CART20 | AUX12 = 0
  state_prev_valid (1)
  aux.q_t (12)                 NaN: required by preflight_rel16v2 for 32-dim actions, unused by rel16_aux_relonly; NaN (not a
                               pseudo q) so any dq / FK plugin pointed at this data fails loudly instead of training on it
  aux.has_dq_supervision (1)   0
  aux.state_prevrel (20)       diagnostic copy (aux.* keys are not policy features)
  aux.stage_id (1), aux.row (1), aux.time_s (1), aux.camera_valid (3)
meta/stats.json: observation.state / action recomputed over all exported rows (action over N*16 rows per channel); AUX12
channels mean 0 / std 1 (the gcal1 placeholder convention, never std 0); aux.q_t stats zeroed.
Resumable: one shard per episode (+ .complete marker), then lerobot aggregate_datasets with the pyarrow agg_patch
(agg_patch.py from ego_paths.DATA_ROOT / rel16ego).

Pipeline stage: dataset stage, last step (after validate_dataset; its output is the X-VLA training input).
usage (env ~/xvla-mac/bin/python, needs lerobot):
  python -m ego_cart20.scripts.export_lerobot <processed_root> <out_root> [--name ego_cart20_v2] [--workers 6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time

import numpy as np

from ego_paths import DATA_ROOT, prepend_sys_path  # machine-specific roots (env-overridable)

from ..processed_episode import loadEpisode

HERE = pathlib.Path(__file__).resolve()
AGG = DATA_ROOT / "rel16ego"
CAMS = (("head", "observation.images.global"), ("left_wrist", "observation.images.left_wrist"),
        ("right_wrist", "observation.images.right_wrist"))
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
IMAGE_PX = 224
FPS = 15
CART_DIM = 20                  # action channels 0:20 = CART20, 20:32 = AUX12
ACTION_SHAPE = (16, 32)
STAT_PERCENTILES = (1, 10, 50, 90, 99)
AUX_Q_DIM = 12
IMAGE_WRITER_THREADS = 4
RECIPE = "umi_bridge/RELCART20_RELONLY_D6_RECIPE.md"


def _videoFeature() -> dict:
    return {"dtype": "video", "shape": (IMAGE_PX, IMAGE_PX, 3), "names": ["height", "width", "channels"]}


def _floatFeature(*shape) -> dict:
    return {"dtype": "float32", "shape": shape, "names": None}


def features() -> dict:
    """the LeRobot feature spec of the export (column -> dtype / shape)"""
    return {"observation.images.global": _videoFeature(), "observation.images.left_wrist": _videoFeature(),
            "observation.images.right_wrist": _videoFeature(), "observation.state": _floatFeature(20),
            "action": _floatFeature(16, 32),
            "state_prev_valid": _floatFeature(1), "aux.q_t": _floatFeature(12), "aux.has_dq_supervision": _floatFeature(1),
            "aux.state_prevrel": _floatFeature(20), "aux.stage_id": _floatFeature(1), "aux.row": _floatFeature(1),
            "aux.time_s": _floatFeature(1), "aux.camera_valid": _floatFeature(3)}


def decode224(path, want) -> dict:
    """{frame index: RGB 224x224 (INTER_AREA)} for the wanted frame indices >= 0 (sequential decode, stops at the last)"""
    import av
    import cv2
    tOut = {}
    tWant = set(int(tX) for tX in want if tX >= 0)
    if not tWant:
        return tOut
    with av.open(str(path)) as tContainer:
        for tI, tFrame in enumerate(tContainer.decode(tContainer.streams.video[0])):
            if tI in tWant:
                tOut[tI] = cv2.resize(tFrame.to_ndarray(format="rgb24"), (IMAGE_PX, IMAGE_PX), interpolation=cv2.INTER_AREA)
            if tI >= max(tWant):
                break
    return tOut


def writeShard(args) -> dict:
    """worker: (processed episode dir, shard root) -> one-episode LeRobot dataset; skipped if its .complete marker exists
    (was write_shard)"""
    tEpDir, tShardRoot = map(pathlib.Path, args)
    tMarker = tShardRoot.with_suffix(".complete")
    if tMarker.exists():
        return json.load(open(tMarker))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    shutil.rmtree(tShardRoot, ignore_errors=True)
    tShardRoot.parent.mkdir(parents=True, exist_ok=True)
    tT0 = time.time()
    tEp = loadEpisode(tEpDir, mmap=False)
    tMeta = tEp["metadata"]
    tRows = tEp["train_rows"]
    assert (np.count_nonzero(tEp["action"][..., CART_DIM:]) == 0 and np.isfinite(tEp["state"][tRows]).all()
            and np.isfinite(tEp["action"]).all())
    tImages, tFrameIdx = {}, {}
    for tCam, _ in CAMS:
        if tCam in tEp["cameras"]:
            tFrameIdx[tCam] = tEp["cameras"][tCam]["frame_index"][tRows]
            tImages[tCam] = decode224(tEp["cameras"][tCam]["video"], tFrameIdx[tCam])
    tBlank = np.zeros((IMAGE_PX, IMAGE_PX, 3), np.uint8)
    tDs = LeRobotDataset.create(repo_id=f"local/{tShardRoot.name}", fps=FPS, features=features(), root=str(tShardRoot),
                                robot_type="rebot_b601", use_videos=True, image_writer_threads=IMAGE_WRITER_THREADS)
    tNan12 = np.full(AUX_Q_DIM, np.nan, np.float32)
    for tJ, tRow in enumerate(tRows):
        tFrame = {}
        for tCam, tKey in CAMS:
            tF = int(tFrameIdx[tCam][tJ]) if tCam in tFrameIdx else -1
            tFrame[tKey] = tImages[tCam][tF] if tF >= 0 else tBlank
        assert tFrame["observation.images.global"] is not tBlank, "training row without head frame"
        tFrame.update({"observation.state": tEp["state"][tRow].astype(np.float32), "action": tEp["action"][tJ].astype(np.float32),
                       "state_prev_valid": np.array([tEp["state_prev_valid"][tRow]], np.float32), "aux.q_t": tNan12,
                       "aux.has_dq_supervision": np.zeros(1, np.float32),
                       "aux.state_prevrel": tEp["state_prevrel"][tRow].astype(np.float32),
                       "aux.stage_id": np.array([tEp["stage_id"][tRow]], np.float32), "aux.row": np.array([tRow], np.float32),
                       "aux.time_s": np.array([tEp["timestamps"][tRow] - tEp["timestamps"][0]], np.float32),
                       "aux.camera_valid": tEp["camera_valid"][tRow].astype(np.float32), "task": tMeta["instruction"]})
        tDs.add_frame(tFrame)
    tDs.save_episode(parallel_encoding=False)
    tDs.finalize()
    tRec = dict(episode_id=tMeta["episode_id"], frames=int(len(tRows)), seconds=round(time.time() - tT0, 1),
                written=time.strftime("%F %T"))
    json.dump(tRec, open(tMarker, "w"))
    return tRec


def stats(x) -> dict:
    """LeRobot stats (min / max / mean / std / count / q01..q99) over axis 0"""
    tX = np.asarray(x, np.float64)
    tStats = dict(min=tX.min(0), max=tX.max(0), mean=tX.mean(0), std=tX.std(0), count=np.array([len(tX)]))
    for tQ in STAT_PERCENTILES:
        tStats[f"q{tQ:02d}"] = np.percentile(tX, tQ, axis=0)
    return tStats


def _column(table, name):
    """a parquet column as a plain (non-extension) arrow array"""
    import pyarrow as pa
    tCol = table.column(name).combine_chunks()
    return tCol.storage if isinstance(tCol, pa.ExtensionArray) else tCol


def fixStats(root) -> dict:
    """recompute state / action stats from the written parquet; AUX12 mean 0 std 1; aux.q_t zeros (was fix_stats)"""
    import glob

    import pyarrow.parquet as pq
    tStates, tActions = [], []
    for tFile in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        tTable = pq.read_table(tFile, columns=["observation.state", "action"])
        tStates.append(np.asarray(_column(tTable, "observation.state").to_pylist(), np.float32))
        tActions.append(np.asarray(_column(tTable, "action").to_pylist(), np.float32))
    tS, tA = np.concatenate(tStates), np.concatenate(tActions)
    assert (tA.shape[1:] == ACTION_SHAPE and np.count_nonzero(tA[..., CART_DIM:]) == 0 and np.isfinite(tS).all()
            and np.isfinite(tA).all())
    tStatsJson = json.load(open(root / "meta/stats.json"))
    tStateStats = stats(tS)
    tActionStats = stats(tA.reshape(-1, ACTION_SHAPE[1]))
    tActionStats["mean"][CART_DIM:] = 0.0
    tActionStats["std"][CART_DIM:] = 1.0
    tStatsJson["observation.state"] = {tK: tV.tolist() for tK, tV in tStateStats.items()}
    tStatsJson["action"] = {tK: tV.tolist() for tK, tV in tActionStats.items()}
    if "aux.q_t" in tStatsJson:
        tStatsJson["aux.q_t"] = {tK: ([0.0] * AUX_Q_DIM if tK != "count" else tStatsJson["aux.q_t"]["count"])
                                 for tK in tStatsJson["aux.q_t"]}
    json.dump(tStatsJson, open(root / "meta/stats.json", "w"), indent=4)
    return dict(rows=int(len(tS)), action_std_0_20=[round(float(tX), 5) for tX in tActionStats["std"][:CART_DIM]])


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("processed")
    tParser.add_argument("out")
    tParser.add_argument("--name", default="ego_cart20_v2")
    tParser.add_argument("--workers", type=int, default=6)
    tParser.add_argument("--splits", nargs="*", default=["train", "val"])
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tProcessed, tOutRoot = pathlib.Path(tArgs.processed), pathlib.Path(tArgs.out)
    tShards = tOutRoot / f"{tArgs.name}_shards"
    tReport = {}
    prepend_sys_path(str(AGG))
    import agg_patch
    agg_patch.install()
    from concurrent.futures import ProcessPoolExecutor

    from lerobot.datasets.aggregate import aggregate_datasets
    for tS in tArgs.splits:
        tManifest = [json.loads(tLine) for tLine in open(tProcessed / f"{tS}_manifest.jsonl")]
        if not tManifest:
            print(f"{tS}: empty, skipped")
            continue
        tJobs = [(str(tProcessed / tR["path"]), str(tShards / tS / tR["episode_id"])) for tR in tManifest]
        with ProcessPoolExecutor(tArgs.workers) as tEx:
            for tRec in tEx.map(writeShard, tJobs):
                print(tS, json.dumps(tRec), flush=True)
        tDst = tOutRoot / f"{tArgs.name}_{tS}"
        tTmp = tOutRoot / f"{tArgs.name}_{tS}.tmp"
        shutil.rmtree(tTmp, ignore_errors=True)
        aggregate_datasets(repo_ids=[f"local/{pathlib.Path(tSr).name}" for _, tSr in tJobs],
                           aggr_repo_id=f"local/{tArgs.name}_{tS}",
                           roots=[pathlib.Path(tSr) for _, tSr in tJobs], aggr_root=tTmp)
        tReport[tS] = dict(episodes=len(tJobs), **fixStats(tTmp))
        assert not tDst.exists(), f"{tDst} exists"
        tTmp.rename(tDst)
        print(tS, tReport[tS], flush=True)
        json.dump(dict(schema="ego_cart20_v2_lerobot/v1", processed=str(tProcessed.resolve()), split=tS, report=tReport[tS],
                       processed_metadata_sha256=hashlib.sha256((tProcessed / "metadata.json").read_bytes()).hexdigest(),
                       exporter_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
                       agg_patch_sha256=hashlib.sha256((AGG / "agg_patch.py").read_bytes()).hexdigest(),
                       recipe=RECIPE, state="RELCART20 task anchor", aux12="zero", aux_q_t="NaN placeholder"),
                  open(tDst / "EXPORT.json", "w"), indent=1)
    print("EXPORT DONE", json.dumps(tReport))
    return 0


if __name__ == "__main__":
    sys.exit(main())
