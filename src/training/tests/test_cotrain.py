"""Characterization of relonly/train_cotrain.py: the ego + robot co-training dataset, the exact-ratio batch sampler and the
DataLoader / make_dataset patches it puts into lerobot_train (fake lerobot_train; the 8bit wrapper run is intercepted)."""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


class FakeDataset:
    def __init__(self, name, nFrames, nEpisodes, keys):
        self.name, self.num_frames, self.num_episodes, self.mKeys = name, nFrames, nEpisodes, keys
        self.meta = f"meta-{name}"

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):
        return {tK: (self.name, i) for tK in self.mKeys}


def _loadCotrain(fakeLerobot, monkeypatch, egoPerBatch=None):
    tRuns = []
    monkeypatch.setattr(torch.utils.data, "DataLoader", torch.utils.data.DataLoader)   # restored after the test
    monkeypatch.setattr(runpy, "run_path", lambda path, run_name=None: tRuns.append((path, run_name)))
    monkeypatch.setenv("COTRAIN_EGO_ROOT", "/data/ego_cart20_v2b_robot100_train/")
    if egoPerBatch is not None:
        monkeypatch.setenv("COTRAIN_EGO_PER_BATCH", str(egoPerBatch))
    import ego_paths
    monkeypatch.setattr(ego_paths, "SHARED_ROOT", Path("/shared"))
    monkeypatch.setattr(sys, "argv", ["train_cotrain.py", "--dataset.repo_id=x"])
    tLt = fakeLerobot["lerobot.scripts.lerobot_train"]
    tMade = []

    def make_dataset(cfg):
        tMade.append((cfg.dataset.repo_id, cfg.dataset.root))
        if cfg.dataset.root == "/data/ego_cart20_v2b_robot100_train/":
            return FakeDataset("ego", 6, 2, ["action", "observation.state", "aux.q_t", "task"])
        return FakeDataset("robot", 10, 3, ["action", "observation.state", "index", "task"])

    tLt.make_dataset = make_dataset
    tMod = importlib.import_module("training.relonly.train_cotrain")
    return tMod, tLt, tRuns, tMade


def test_cotrain_runs_the_8bit_wrapper_with_patched_lerobot_train(fakeLerobot, monkeypatch, golden, capsys):
    tMod, tLt, tRuns, tMade = _loadCotrain(fakeLerobot, monkeypatch)
    assert tRuns == [("/shared/relonly/code_8bit/train_rel16_relonly.py", "__main__")]
    assert sys.argv[0] == "/shared/relonly/code_8bit/train_rel16_relonly.py"
    tCfg = SimpleNamespace(dataset=SimpleNamespace(repo_id="rebot/r312c_relcart20_rel16_v4", root="/data/r312c"))
    tDs = tLt.make_dataset(tCfg)
    assert tMade == [("rebot/r312c_relcart20_rel16_v4", "/data/r312c"),
                     ("rebot/ego_cart20_v2b_robot100_train", "/data/ego_cart20_v2b_robot100_train/")]
    assert tCfg.dataset.root == "/data/r312c"                      # the robot cfg is not touched (deep copy)
    assert (tDs.num_frames, tDs.num_episodes, tDs.meta, tDs.episodes) == (16, 5, "meta-robot", None)
    assert sorted(tDs.keys) == ["action", "observation.state", "task"]
    assert len(tDs) == 16
    assert tDs[3] == {"action": ("robot", 3), "observation.state": ("robot", 3), "task": ("robot", 3)}
    assert tDs[~2] == {"action": ("ego", 2), "observation.state": ("ego", 2), "task": ("ego", 2)}
    assert tLt.torch.utils.data.DataLoader is tMod.DataLoader
    tLoader = tMod.DataLoader(tDs, batch_size=8, shuffle=True, drop_last=True, num_workers=0, collate_fn=lambda b: b)
    tBatches = [[tItem["action"] for tItem in tBatch] for tBatch in tLoader]
    golden.check("cotrain/batches", [[[tSrc, tI] for tSrc, tI in tB] for tB in tBatches])
    golden.check("cotrain/stdout", capsys.readouterr().out)


def test_balanced_sampler_draws_exact_counts_without_replacement(fakeLerobot, monkeypatch, golden):
    tMod, *_ = _loadCotrain(fakeLerobot, monkeypatch, egoPerBatch=3)
    tDs = SimpleNamespace(robot=list(range(7)), ego=list(range(5)))
    tSampler = tMod.BalancedBatchSampler(tDs, 4, seed=7)
    assert len(tSampler) == 3
    tEpochs = [list(tSampler) for _ in range(3)]
    for tEpoch in tEpochs:
        for tBatch in tEpoch:
            assert sum(1 for tI in tBatch if tI < 0) == 3
    golden.check("cotrain/sampler_epochs", tEpochs)
