"""Tiny synthetic LeRobot v3 trees for the dataset-derivation tests: a UMI76 parent (REL16 action, state76, optional aux.q_t),
its per-episode stats parquet, provenance, a hard-linkable video, and the robot source dataset with the 14-D leader command."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
SOURCE_FRAMES = 40
# parquet file -> (episode, first frame, frame count): episode 1 spans both files
LAYOUT = {"file-000.parquet": [(0, 0, 6), (1, 0, 3)], "file-001.parquet": [(1, 3, 3)]}


def _stats(x):
    tQ = np.quantile(x, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=x.std(0), count=np.array([len(x)]),
                q01=tQ[0], q10=tQ[1], q50=tQ[2], q90=tQ[3], q99=tQ[4])


def _listStats(x):
    return {tK: [float(tV) for tV in np.atleast_1d(tArr)] if tK != "count" else [int(tArr[0])]
            for tK, tArr in _stats(x).items()}


def writeSource(root: Path, seed: int = 0) -> Path:
    """robot LeRobot source: action (14,) = leader command, grip dims 6 / 13 in 0..80"""
    tRng = np.random.default_rng(seed)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    tRows = []
    for tEp in (0, 1):
        tT = np.arange(SOURCE_FRAMES)
        tA = tRng.normal(size=(SOURCE_FRAMES, 14))
        tA[:, 6] = 40 + 40 * np.sin(tT / 5.0 + tEp)
        tA[:, 13] = 40 + 40 * np.cos(tT / 7.0 + tEp)
        for tF in range(SOURCE_FRAMES):
            tRows.append((tEp, tF, tA[tF].tolist()))
    tRows = tRows[::-1]                                     # stored out of order: the loader sorts by frame_index
    tTable = pa.table({"action": pa.array([tR[2] for tR in tRows], pa.list_(pa.float32(), 14)),
                       "episode_index": pa.array([tR[0] for tR in tRows], pa.int64()),
                       "frame_index": pa.array([tR[1] for tR in tRows], pa.int64())})
    pq.write_table(tTable, root / "data/chunk-000/file-000.parquet")
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 2}))
    return root


def writeParent(root: Path, sourceRoot: str, actionDim: int = 20, withQt: bool = False, seed: int = 1) -> Path:
    tRng = np.random.default_rng(seed)
    for tSub in ("data/chunk-000", "meta/episodes/chunk-000", "videos/observation.images.global/chunk-000"):
        (root / tSub).mkdir(parents=True)
    (root / "videos/observation.images.global/chunk-000/file-000.mp4").write_bytes(b"not-a-real-video" * 4)
    tHf = {"info": {"features": {"observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 76,
                                                        "_type": "List"}}}}
    tAllA, tAllS, tPerEp = [], [], {}
    tIndex = 0
    for tFile, tParts in LAYOUT.items():
        tCols = {"action": [], "observation.state": [], "episode_index": [], "frame_index": [], "index": [], "task_index": []}
        if withQt:
            tCols["aux.q_t"] = []
        for tEp, tF0, tN in tParts:
            for tF in range(tF0, tF0 + tN):
                tA = tRng.normal(size=(16, actionDim)).astype(np.float32)
                tS = tRng.normal(size=76).astype(np.float32)
                tS[[37, 75]] = tRng.uniform(0.0, 0.13, size=2)
                tCols["action"].append(tA.tolist())
                tCols["observation.state"].append(tS.tolist())
                tCols["episode_index"].append(tEp)
                tCols["frame_index"].append(tF)
                tCols["index"].append(tIndex)
                tCols["task_index"].append(tEp)
                if withQt:
                    tCols["aux.q_t"].append((tRng.normal(size=12) * 0.4).astype(np.float32).tolist())
                tIndex += 1
                tAllA.append(tA)
                tAllS.append(tS)
                tPerEp.setdefault(tEp, ([], []))
                tPerEp[tEp][0].append(tA)
                tPerEp[tEp][1].append(tS)
        tTypes = {"action": pa.list_(pa.list_(pa.float32(), actionDim), 16), "observation.state": pa.list_(pa.float32(), 76),
                  "aux.q_t": pa.list_(pa.float32(), 12)}
        tArrays = [pa.array(tV, tTypes.get(tK, pa.int64())) for tK, tV in tCols.items()]
        tSchema = pa.schema([pa.field(tK, tA.type) for tK, tA in zip(tCols, tArrays)],
                            metadata={b"huggingface": json.dumps(tHf).encode()})
        pq.write_table(pa.Table.from_arrays(tArrays, schema=tSchema), root / "data/chunk-000" / tFile)
    tA = np.stack(tAllA).astype(np.float64)
    tS = np.stack(tAllS).astype(np.float64)
    tStats = {"action": _listStats(tA.reshape(-1, actionDim)), "observation.state": _listStats(tS)}
    (root / "meta/stats.json").write_text(json.dumps(tStats, indent=4))
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 2, "features": {
        "observation.state": {"dtype": "float32", "shape": [76], "names": None},
        "action": {"dtype": "float32", "shape": [16, actionDim], "names": None}}}, indent=4))
    tEpCols = {"episode_index": pa.array([0, 1], pa.int64())}
    for tKey, tIdx in (("action", 0), ("observation.state", 1)):
        tPer = [_listStats(np.stack(tPerEp[tE][tIdx]).astype(np.float64).reshape(-1, actionDim if tIdx == 0 else 76))
                for tE in (0, 1)]
        for tK in STAT_KEYS:
            tEpCols[f"stats/{tKey}/{tK}"] = pa.array([tP[tK] for tP in tPer],
                                                     pa.list_(pa.int64() if tK == "count" else pa.float64()))
    pq.write_table(pa.table(tEpCols), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "v4_provenance.json").write_text(json.dumps({"sources": [{"lerobot": sourceRoot, "zarr": "/node/z.zarr"}],
                                                         "converter": "umi_to_lerobot_v4.py"}, indent=2))
    return root


def readTree(root: Path, tmpRoot: Path) -> dict:
    """every parquet column (as nested lists) and every json text of a dataset tree, tmp paths replaced"""
    tOut = {}
    for tPath in sorted(root.rglob("*")):
        tRel = str(tPath.relative_to(root))
        if tPath.suffix == ".parquet":
            tTable = pq.read_table(tPath)
            for tName in tTable.column_names:
                tVals = tTable.column(tName).to_pylist()
                tOut[f"{tRel}:{tName}"] = np.asarray(tVals, dtype=np.float64)
            tOut[f"{tRel}:schema"] = str(tTable.schema).replace(str(tmpRoot), "<tmp>")
        elif tPath.suffix == ".json":
            tOut[tRel] = tPath.read_text().replace(str(tmpRoot), "<tmp>")
        elif tPath.is_file():
            tOut[tRel] = tPath.read_bytes().decode("latin-1")
    return tOut
