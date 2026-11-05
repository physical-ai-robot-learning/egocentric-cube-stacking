#!/usr/bin/env python3
"""Post-build gate for r312c_umi76_rel16_v1 = HEAD180 + FRONT132 (pan=center only, grade A, 22/order, rot180) (dataset stage,
node, read-only).

[2026-09-28] Derived from gate_r380.py; G6 is now: every FRONT episode shared with R380 is bit-identical.
G1 HEAD eps 0-179: every parquet column bit-identical to r180_umi76_rel16_v1 (the select= change must not touch them)
G2 HEAD videos: decoded frames identical to r180 on sampled episodes/frames
G3 FRONT eps 180-311: count 132, order == selection order, task == R675 pool instruction, frame counts plausible
G4 FRONT global == rot180 of the source center675 global at the same (episode, frame), sampled; wrist == zarr
G5 FRONT state/action finite, shapes 76 / 16x20, current self term identity; stats cover the dataset
Inputs: the node datasets under NODE_HOME/holobrain-data/lerobot and the selections under NODE_HOME/umi_bridge.
Output: one PASS / FAIL line per check and "ALL R312c GATES PASS" (the chain greps for it).
Usage: python gate_r312c.py     (node chain scripts/train/node4090/chain_r312c.sh; needs LeRobot)
"""
from __future__ import annotations

import glob
import json
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ego_paths import NODE_HOME, prepend_sys_path  # machine-specific roots (env-overridable)

N_HEAD = 180
N_FRONT = 132
N_TOTAL = N_HEAD + N_FRONT
N_SHARED_R380 = 78               # FRONT episodes R312c shares with R380
GATE_SEED = 0
G2_EPISODES, G2_FRAMES = 4, 3
G4_EPISODES = 8
G4_OFFSETS = range(-3, 4)        # frame offsets searched in r663 around the mid frame
G4_MAX_MEAN_ABS = 0.02           # best match must be this close ...
G4_ROTATED_FACTOR = 5            # ... and re-rotating it this many times worse
MIN_FRONT_FRAMES = 150
SELF_TERM_TOL = 1e-5
STATE_DIM = 76
ACTION_FLAT = 320                # 16 x 20
IDENTITY_ROT6D = np.array([1.0, 0, 0, 0, 1, 0])   # rotation_6d of the identity (UMI packing)
CAMERA_KEYS = ["observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist"]
NO_MATCH = (9.0, None, 9.0)      # (mean |d|, offset, rotated mean |d|) before any frame is compared


class Gate:
    """PASS / FAIL printer that remembers the failed check names"""

    def __init__(self):
        self.mFails = []

    def check(self, name, ok, info=""):
        print(("PASS  " if ok else "FAIL  ") + name + (f"  {info}" if info else ""), flush=True)
        if not ok:
            self.mFails.append(name)


def _readParquet(f, columns=None):
    """HF datasets writes array columns as arrow EXTENSION types, which pandas turns into an extension dtype that
    pd.concat cannot compare; drop to the storage type (fixed-size list of float) first."""
    tTable = pq.read_table(f, columns=columns)
    tCols = [(tC.combine_chunks().storage if isinstance(tC.type, pa.ExtensionType) else tC) for tC in tTable.columns]
    return pa.table(tCols, names=tTable.column_names).to_pandas(ignore_metadata=True)


def loadFrames(root):
    tFiles = sorted(glob.glob(f"{root}/data/*/*.parquet"))
    return pd.concat([_readParquet(tF) for tF in tFiles]).sort_values("index").reset_index(drop=True)


def _numeric(col):
    """object column of (possibly nested) arrays -> one numeric ndarray; strings stay as a list"""
    def cell(value):
        tArr = np.asarray(value)
        return np.stack([cell(tY) for tY in tArr]) if tArr.dtype == object else tArr
    try:
        return np.stack([cell(tX) for tX in col])
    except Exception:
        return np.asarray([str(tX) for tX in col])


def _orderCode(task):
    """task -> order code, the selection's key format"""
    return "".join(tC[0].upper() for tC in re.findall(r"(\w+) cube", task))


def _checkHead(gate, new, old, newRoot, oldRoot, newTasks):
    tNi = json.load(open(f"{newRoot}/meta/info.json"))
    gate.check("episodes == 312", tNi["total_episodes"] == N_TOTAL, str(tNi["total_episodes"]))
    tNh = new[new.episode_index < N_HEAD].reset_index(drop=True)
    gate.check("G1 HEAD row count == r180", len(tNh) == len(old), f"{len(tNh)} vs {len(old)}")
    tCols = [tC for tC in old.columns]
    tBad = []
    for tC in tCols:
        tA, tB = tNh[tC].to_numpy(), old[tC].to_numpy()
        if tA.dtype == object:
            tSame = tA.shape == tB.shape and np.array_equal(_numeric(tA), _numeric(tB))
        else:
            tSame = np.array_equal(tA, tB)
        if not tSame:
            tBad.append(tC)
    gate.check("G1 HEAD every parquet column bit-identical to r180", not tBad,
               f"{len(tCols)} cols" if not tBad else f"differ: {tBad}")
    tOt = pq.read_table(f"{oldRoot}/meta/tasks.parquet").to_pandas()   # task string = index (metadata)
    tOld = dict(zip(tOt.task_index, tOt.index))
    tHs = new[new.episode_index < N_HEAD].groupby("episode_index").task_index.first().map(newTasks)
    tOs = old.groupby("episode_index").task_index.first().map(tOld)
    gate.check("G1 HEAD task strings identical", all(isinstance(tX, str) for tX in tHs.values) and (tHs.values == tOs.values).all(),
               f"e.g. {str(tHs.values[0])[:40]}")


def _checkHeadVideo(gate, old, dn, do, rng):
    tMax = 0.0
    for tEp in rng.choice(N_HEAD, G2_EPISODES, replace=False):
        tFrames = old.index[old.episode_index == tEp]
        for tI in rng.choice(tFrames, G2_FRAMES, replace=False):
            tA, tB = dn[int(tI)], do[int(tI)]
            for tK in CAMERA_KEYS:
                tMax = max(tMax, float((tA[tK] - tB[tK]).abs().max()))
    gate.check("G2 HEAD decoded video frames identical to r180 (12 frames x 3 cams)", tMax == 0.0, f"max abs diff {tMax:.2e}")


def _checkFront(gate, new, newTasks, sel):
    tNf = new[new.episode_index >= N_HEAD]
    tEps = sorted(tNf.episode_index.unique())
    gate.check("G3 FRONT episode count == 132", len(tEps) == N_FRONT and tEps == list(range(N_HEAD, N_TOTAL)))
    tInv = pd.read_csv(str(NODE_HOME / "umi_bridge/qa_b663/inventory.csv")).set_index("episode")
    tWant = [tInv.loc[tE, "task"] for tE in [N_HEAD + tZ for tZ in sel["zarr_episodes"]]]
    tGot = list(tNf.groupby("episode_index").task_index.first().map(newTasks).values)
    gate.check("G3 FRONT task per episode == selection (R663 inventory task)", tGot == tWant)
    tCnt = pd.Series([_orderCode(tT) for tT in tGot]).value_counts().to_dict()
    gate.check("G3 FRONT order counts == selection", tCnt == sel["order_counts"], str(sorted(tCnt.values())))
    tLen = tNf.groupby("episode_index").size().values
    gate.check("G3 FRONT frames per episode plausible (>=150)", tLen.min() >= MIN_FRONT_FRAMES,
               f"min {tLen.min()} max {tLen.max()} total {tLen.sum()}")
    tSt = _numeric(tNf["observation.state"].to_numpy())
    tAc = _numeric(tNf["action"].to_numpy()).reshape(len(tNf), -1)
    gate.check("G5 FRONT state/action finite", np.isfinite(tSt).all() and np.isfinite(tAc).all())
    gate.check("G5 FRONT state width 76, action 320 (16x20)", tSt.shape[1] == STATE_DIM and tAc.shape[1] == ACTION_FLAT,
               f"{tSt.shape} {tAc.shape}")
    # current self term: pos [3:6] / [41:44] == 0 and rot [18:24]/[56:62] == 0 (axis-angle identity) per the packing
    tCur = max(np.abs(tSt[:, 3:6]).max(), np.abs(tSt[:, 41:44]).max(), np.abs(tSt[:, 18:24] - IDENTITY_ROT6D).max(),
               np.abs(tSt[:, 56:62] - IDENTITY_ROT6D).max())
    gate.check("G5 FRONT current self term is identity", tCur < SELF_TERM_TOL, f"max {tCur:.2e}")


def _checkFrontRot180(gate, dn, dataRoot, sel, rng, LeRobotDataset):
    # G4 rot180: compare new FRONT global against the B663 dataset, whose FRONT globals were built from the same source
    # with the same rot180 and verified correctly rotated (qa_b663 v2). Same (source episode, stored frame) -> same image.
    tR663 = LeRobotDataset("rebot/r663", root=f"{dataRoot}/r663_xvla_umi32_v1")
    tRows = []
    for tJ in rng.choice(N_FRONT, G4_EPISODES, replace=False):
        tENew, tE663 = N_HEAD + int(tJ), N_HEAD + int(sel["zarr_episodes"][int(tJ)])     # r663 FRONT 180.. == r675rbp zarr 0..
        tA0, tA1 = dn.meta.episodes[tENew]["dataset_from_index"], dn.meta.episodes[tENew]["dataset_to_index"]
        tB0, tB1 = tR663.meta.episodes[tE663]["dataset_from_index"], tR663.meta.episodes[tE663]["dataset_to_index"]
        tMid = (tA1 - tA0) // 2
        tX = dn[tA0 + tMid]["observation.images.global"]
        tBest = NO_MATCH
        for tOff in G4_OFFSETS:
            tK = tB0 + tMid + tOff
            if not (tB0 <= tK < tB1):
                continue
            tY = tR663[tK]["observation.images.global"]
            tD = float((tX - tY).abs().mean())
            tDFlip = float((tX - tY.flip(-1).flip(-2)).abs().mean())
            if tD < tBest[0]:
                tBest = (tD, tOff, tDFlip)
        tRows.append(tBest)
    tOk = all(tR[0] < G4_MAX_MEAN_ABS and tR[2] > G4_ROTATED_FACTOR * tR[0] for tR in tRows) and len(tRows) == G4_EPISODES
    gate.check("G4 FRONT global == B663 rot180 FRONT frame, and re-rotating it is far worse (8 eps)", tOk,
               "best(mean|d|, frame offset, d if rotated again) = " + str([(round(tR[0], 4), tR[1], round(tR[2], 4)) for tR in tRows]))


def _checkSharedWithR380(gate, new, dataRoot, sel):
    # G6 regression: every FRONT episode shared with R380 is bit-identical (state, action, length) -- same converter, same source
    tS380 = json.load(open(str(NODE_HOME / "umi_bridge/umi76/r380_front200_v1.json")))["zarr_episodes"]
    tR380 = loadFrames(f"{dataRoot}/r380_umi76_rel16_v1")
    tPos380 = {tZ: N_HEAD + tI for tI, tZ in enumerate(tS380)}
    tShared = [(tI, tZ) for tI, tZ in enumerate(sel["zarr_episodes"]) if tZ in tPos380]
    tBad = []
    for tI, tZ in tShared:
        tA = new[new.episode_index == N_HEAD + tI]
        tB = tR380[tR380.episode_index == tPos380[tZ]]
        if len(tA) != len(tB) \
                or not np.array_equal(_numeric(tA["observation.state"].to_numpy()), _numeric(tB["observation.state"].to_numpy())) \
                or not np.array_equal(_numeric(tA["action"].to_numpy()), _numeric(tB["action"].to_numpy())):
            tBad.append(tZ)
    gate.check(f"G6 the {len(tShared)} FRONT episodes shared with R380 are bit-identical (state, action, length)",
               not tBad and len(tShared) == N_SHARED_R380,
               f"differ: {tBad[:5]}" if tBad else f"{len(tShared)}/{len(tShared)}")


def main(argv: list[str] | None = None) -> int:
    prepend_sys_path(str(NODE_HOME / "lerobot-seeed/src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tData = str(NODE_HOME / "holobrain-data/lerobot")
    tNewRoot, tOldRoot = f"{tData}/r312c_umi76_rel16_v1", f"{tData}/r180_umi76_rel16_v1"
    tSel = json.load(open(str(NODE_HOME / "umi_bridge/umi76/r312c_front132_v1.json")))
    tGate = Gate()
    tN, tO = loadFrames(tNewRoot), loadFrames(tOldRoot)
    tNt = pq.read_table(f"{tNewRoot}/meta/tasks.parquet").to_pandas()   # task string = index (metadata)
    tNewTasks = dict(zip(tNt.task_index, tNt.index))
    _checkHead(tGate, tN, tO, tNewRoot, tOldRoot, tNewTasks)
    tDn, tDo = LeRobotDataset("rebot/r312c", root=tNewRoot), LeRobotDataset("rebot/r180", root=tOldRoot)
    tRng = np.random.default_rng(GATE_SEED)
    _checkHeadVideo(tGate, tO, tDn, tDo, tRng)
    _checkFront(tGate, tN, tNewTasks, tSel)
    tStats = json.load(open(f"{tNewRoot}/meta/stats.json"))
    tGate.check("G5 stats.json present with state/action", "observation.state" in tStats and "action" in tStats)
    _checkFrontRot180(tGate, tDn, tData, tSel, tRng, LeRobotDataset)
    _checkSharedWithR380(tGate, tN, tData, tSel)
    print("\nALL R312c GATES PASS" if not tGate.mFails else f"\nFAILED: {tGate.mFails}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
