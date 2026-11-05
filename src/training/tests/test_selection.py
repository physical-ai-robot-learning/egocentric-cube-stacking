"""Characterization of umi76/select_r312_center.py (deterministic FRONT132 / FRONT66 selection) on a synthetic pool, plus the
integrity of the frozen selection files shipped next to it (their zarr_episodes_sha256 matches their episode list)."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import REPO_SRC

UMI76_DIR = Path(__file__).resolve().parents[1] / "umi76"
COLOURS = ("red", "blue", "pink")


def _writePool(bridge: Path, holobrain: Path):
    tRng = np.random.default_rng(3)
    tAudit = bridge / "rel16_audit/r380"
    tAudit.mkdir(parents=True)
    (bridge / "umi76").mkdir(parents=True)
    tOrders = list(itertools.permutations(COLOURS))
    tTasks = [f"stack the {tA} cube, then the {tB} cube, then the {tC} cube" for tA, tB, tC in tOrders]
    tN = 6 * 40
    tSrc = tRng.permutation(tN) + 1000
    tPan = np.where(tRng.random(tN) < 0.8, "center", "left")
    tGrade = np.where(tRng.random(tN) < 0.85, "A", "B")
    tPool = pd.DataFrame({"zarr_ep": np.arange(tN), "src_ep": tSrc, "pan": tPan, "grade": tGrade})
    tPool.to_csv(tAudit / "front_pool_pan.csv", index=False)
    tSetOf = tRng.integers(0, 6, size=tN + 1000)
    tKeys = {str(tI): f"session{tI // 2}|set{tI}" for tI in range(6)}
    json.dump({"episode_sets": tSetOf.tolist(), "set_keys": tKeys}, open(tAudit / "set_map.json", "w"))
    json.dump({"episodes": (np.arange(tN) * 2 + 5).tolist()}, open(tAudit / "r675_rbp.json", "w"))
    pd.DataFrame({"zarr_ep": np.arange(0, tN, 2), "pan": np.where(np.arange(0, tN, 2) % 3 == 0, "center", "far")}).to_csv(
        tAudit / "front200_pan.csv", index=False)
    tRoot = holobrain / "lerobot/src_rebot_3stack_center675_s96"
    (tRoot / "data/chunk-000").mkdir(parents=True)
    (tRoot / "meta").mkdir(parents=True)
    tEp = np.repeat(np.arange(1000, 1000 + tN), 2)
    pd.DataFrame({"episode_index": tEp, "task_index": (tEp % 6), "x": 0}).to_parquet(tRoot / "data/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": range(6)}, index=pd.Index(tTasks, name="task")).to_parquet(tRoot / "meta/tasks.parquet")


def test_select_r312_center_is_deterministic(tmp_path, golden):
    tBridge, tHolo = tmp_path / "umi_bridge", tmp_path / "holobrain-data"
    _writePool(tBridge, tHolo)
    tEnv = dict(os.environ, PYTHONPATH=str(REPO_SRC), EGO_UMI_BRIDGE=str(tBridge), EGO_HOLOBRAIN_DATA=str(tHolo),
                EGO_NODE_HOME="/home/bh-aiteam")
    tRes = subprocess.run([sys.executable, "-m", "training.umi76.select_r312_center"], env=tEnv, capture_output=True, text=True)
    assert tRes.returncode == 0, tRes.stderr
    golden.check("select/stdout", tRes.stdout)
    for tName in ("r312c_front132_v1", "r246c_front66_v1"):
        golden.check(f"select/{tName}", (tBridge / "umi76" / f"{tName}.json").read_text())


@pytest.mark.parametrize("name", ["r312c_front132_v1.json", "r384_front204_v1.json"])
def test_frozen_selection_files_are_self_consistent(name):
    tDoc = json.loads((UMI76_DIR / name).read_text())
    assert hashlib.sha256(json.dumps(tDoc["zarr_episodes"]).encode()).hexdigest() == tDoc["zarr_episodes_sha256"]
    assert len(set(tDoc["zarr_episodes"])) == len(tDoc["zarr_episodes"]) == sum(tDoc["order_counts"].values())


def test_derive_v2_cli_defaults(parsedArgs, golden):
    import importlib
    tDv = importlib.import_module("training.umi76.derive_v2")
    tNs = parsedArgs(tDv.main, ["--parent", "/p"])
    golden.check("derive_v2/cli_defaults", {tK: str(tV) for tK, tV in vars(tNs).items()})
