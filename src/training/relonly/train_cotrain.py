"""Ego + robot CO-TRAINING on top of the B8 REL-only recipe: patches lerobot_train's dataset and DataLoader, then runs the 8bit
wrapper (code_8bit/train_rel16_relonly.py) through runpy (training stage).

[2026-10-03 user] --dataset.repo_id/--dataset.root = the ROBOT dataset (R312c): its meta/stats/features drive the policy +
normalizer (deploy target). COTRAIN_EGO_ROOT = the ego dataset (ego_cart20_v2b_robot100_train), same CART20 contract, same
domain 20, same loss (REL 0:20).
Sampling: every batch is exactly COTRAIN_EGO_PER_BATCH ego + rest robot rows (default 4 + 4 at batch 8), each source drawn
uniformly without replacement (reshuffled when exhausted) -> dataset-level 50/50, NOT frame-count proportional.
Items are cut to the keys both datasets share (+ task); ego aux.q_t is NaN but unused by the REL-only loss.
Usage: COTRAIN_EGO_ROOT=<ego dataset> python train_cotrain.py <lerobot_train flags>   (node: code_cotrain/train_cotrain.py)
"""
from __future__ import annotations

import copy
import os
import random
import runpy
import sys

import torch

from ego_paths import SHARED_ROOT  # machine-specific roots (env-overridable)

EGO_N = int(os.environ.get("COTRAIN_EGO_PER_BATCH", "4"))       # ego rows in every batch
SAMPLER_SEED = 1000
WRAPPER_8BIT = SHARED_ROOT / "relonly/code_8bit/train_rel16_relonly.py"
_DL = torch.utils.data.DataLoader                                  # the original DataLoader (before main() patches it)


class CoTrainDataset(torch.utils.data.Dataset):
    """robot rows at indices >= 0, ego row j at index ~j; meta / num_frames / num_episodes / episodes are what lerobot_train
    reads (LeRobot dataset interface)"""

    def __init__(self, robot, ego):
        self.mRobot, self.mEgo, self.meta = robot, ego, robot.meta
        self.episodes = None
        self.num_frames = robot.num_frames + ego.num_frames
        self.num_episodes = robot.num_episodes + ego.num_episodes
        self.mKeys = (set(robot[0].keys()) & set(ego[0].keys())) | {"task"}
        print(f"[cotrain] robot {robot.num_frames} fr / {robot.num_episodes} ep + ego {ego.num_frames} fr / {ego.num_episodes} ep; "
              f"shared keys {sorted(self.mKeys)}", flush=True)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):  # i >= 0 robot row, i < 0 ego row ~i
        tItem = self.mRobot[i] if i >= 0 else self.mEgo[~i]
        return {tK: tV for tK, tV in tItem.items() if tK in self.mKeys}


class BalancedBatchSampler(torch.utils.data.Sampler):
    """batches of exactly bs - EGO_N robot + EGO_N ego indices, each pool drawn without replacement, then shuffled"""

    def __init__(self, ds, bs, seed=SAMPLER_SEED):
        self.mNumRobot, self.mNumEgo, self.mBatchSize, self.mRng = len(ds.mRobot), len(ds.mEgo), bs, random.Random(seed)
        assert 0 < EGO_N < bs
        self.mRobotPool, self.mEgoPool = [], []

    def _take(self, pool, n, k):
        tOut = []
        while len(tOut) < k:
            if not pool:
                pool.extend(self.mRng.sample(range(n), n))
            tOut.append(pool.pop())
        return tOut

    def __len__(self):
        return (self.mNumRobot + self.mNumEgo) // self.mBatchSize

    def __iter__(self):
        for _ in range(len(self)):
            tBatch = self._take(self.mRobotPool, self.mNumRobot, self.mBatchSize - EGO_N) \
                + [~tJ for tJ in self._take(self.mEgoPool, self.mNumEgo, EGO_N)]
            self.mRng.shuffle(tBatch)
            yield tBatch


class DataLoader(_DL):
    """lerobot_train's DataLoader: a CoTrainDataset gets the BalancedBatchSampler instead of shuffle / sampler / drop_last"""

    def __init__(self, dataset, *a, **kw):
        if isinstance(dataset, CoTrainDataset):
            tBs = kw.pop("batch_size")
            kw.pop("shuffle", None)
            kw.pop("sampler", None)
            kw.pop("drop_last", None)
            kw["batch_sampler"] = BalancedBatchSampler(dataset, tBs)
            print(f"[cotrain] batch {tBs} = {EGO_N} ego + {tBs - EGO_N} robot (exact, every batch)", flush=True)
        super().__init__(dataset, *a, **kw)


def makeDatasetPatch(origMake, egoRoot):
    """-> lerobot_train.make_dataset replacement: the robot dataset from cfg + the ego dataset at egoRoot"""

    def make_dataset(cfg):
        tRobot = origMake(cfg)
        tEgoCfg = copy.deepcopy(cfg)
        tEgoCfg.dataset.repo_id = "rebot/" + os.path.basename(egoRoot.rstrip("/"))
        tEgoCfg.dataset.root = egoRoot
        return CoTrainDataset(tRobot, origMake(tEgoCfg))
    return make_dataset


def main(argv: list[str] | None = None) -> int:
    """argv is left to lerobot_train (the 8bit wrapper parses sys.argv itself)."""
    import lerobot.scripts.lerobot_train as lerobotTrain
    tEgoRoot = os.environ["COTRAIN_EGO_ROOT"]
    lerobotTrain.make_dataset = makeDatasetPatch(lerobotTrain.make_dataset, tEgoRoot)
    lerobotTrain.torch.utils.data.DataLoader = DataLoader
    sys.argv[0] = str(WRAPPER_8BIT)
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
