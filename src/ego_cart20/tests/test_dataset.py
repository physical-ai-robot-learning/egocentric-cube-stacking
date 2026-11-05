"""Characterization of the dataset side: splits, sample weights, the native dataset reader, the integrity report and the
trajectory plot, on a small processed tree built from synthetic raw episodes."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from ego_cart20.convert import convertEpisode
from ego_cart20 import dataset as SM
from ego_cart20 import splits as SP
from ego_cart20.dataset import XVLAEgoDataset
from ego_cart20.processed_episode import writeEpisode
from ego_cart20.raw_episode import loadRawEpisode
from ego_cart20 import integrity as GC
from ego_cart20 import integrity as IN
from ego_cart20.trajectory_plot import plotSamples

from .synthetic import writeRawRoot, writeVideo


@pytest.fixture(scope="module")
def processedRoot(tmp_path_factory):
    tRoot = tmp_path_factory.mktemp("cart20")
    tVideo = writeVideo(tRoot / "head.mp4", 200)
    tIds = writeRawRoot(tRoot / "raw", videos={"head": str(tVideo), "left_wrist": str(tVideo)})
    tOut = tRoot / "processed"
    tSplit = {tIds[0]: "train", tIds[1]: "train", tIds[2]: "val", tIds[3]: "train"}
    tRows = {"train": [], "val": [], "test": []}
    for tId in tIds:
        tEp = convertEpisode(loadRawEpisode(tRoot / "raw" / tId))
        writeEpisode(tOut / "episodes" / tId, tEp)
        tRows[tSplit[tId]].append(dict(episode_id=tId, path=f"episodes/{tId}", stack_order=tEp["metadata"]["stack_order"]))
    for tS, tR in tRows.items():
        (tOut / f"{tS}_manifest.jsonl").write_text("".join(json.dumps(tX) + "\n" for tX in tR))
    return tOut


def test_hash_split_and_disjoint(golden):
    tIds = [f"ep_{tI:03d}" for tI in range(40)]
    tSplit, tRule = SP.hashSplit(tIds)
    golden.check("sp.hash", tSplit)
    golden.check("sp.hash_rule", tRule)
    golden.check("sp.hash_test", SP.hashSplit(tIds, valFrac=0.2, testFrac=0.1))
    assert SP.checkDisjoint(tSplit) is True


def test_frozen_handumi_split(golden, tmp_path, monkeypatch):
    (tmp_path / "c8" / "c8old_rows").mkdir(parents=True)
    (tmp_path / "c8" / "c8old_rows" / "index.json").write_text(json.dumps({"val_episodes": ["a", "b"]}))
    (tmp_path / "umi" / "track_c" / "v2k").mkdir(parents=True)
    (tmp_path / "umi" / "track_c" / "v2k" / "hrl80_split_frozen.json").write_text(
        json.dumps({"val_source_episodes": ["h1"], "train_source_episodes": ["h2"], "sha256": "abc"}))
    monkeypatch.setattr(SP, "DATA_ROOT", tmp_path / "c8")
    monkeypatch.setattr(SP, "UMI_BRIDGE", tmp_path / "umi")
    golden.check("sp.frozen", SP.frozenHandumiSplit(["a", "c", "h1", "h2"]))


def test_sample_weights_and_sampler(golden):
    tOrders = ["RBP", "RBP", "BRP", "PRB", "RBP", "BRP"]
    tStages = [-1, 0, 1, 3, 2, 7]
    golden.check("sm.weights", SM.sampleWeights(tOrders, tStages))
    golden.check("sm.weights_flat", SM.sampleWeights(tOrders, tStages, orderBalanced=False, stageWeights=(1.0, 2.0)))
    tW = SM.sampleWeights(tOrders, tStages)
    tIt = iter(SM.WeightedSampler(tW, seed=3))
    golden.check("sm.sampler_first", np.array([next(tIt) for _ in range(5000)]))


def test_xvla_ego_dataset(golden, processedRoot):
    tDs = XVLAEgoDataset(processedRoot, "train")
    golden.check("ds.len", len(tDs))
    golden.check("ds.weights", tDs.weights())
    golden.check("ds.weights_flat", tDs.weights(orderBalanced=False, stageWeights=(1.0,)))
    for tI in (0, 7, len(tDs) - 1):
        tItem = tDs[tI]
        golden.check(f"ds.item_{tI}", {tK: tV for tK, tV in tItem.items()})
    tImgDs = XVLAEgoDataset(processedRoot, "val", loadImages=True, imageSize=16)
    tItem = tImgDs[3]
    assert sorted(tItem["images"]) == ["head", "left_wrist"] and tItem["images"]["head"].shape == (16, 16, 3)
    golden.check("ds.image_row", int(tItem["row"]))
    golden.check("ds.image_px", tItem["images"]["head"])


def test_integrity_report(golden, processedRoot):
    golden.check("in.pct", IN.pct([1.0, 2.0, np.nan, 5.0, 9.0]))
    assert IN.pct([np.nan]) is None
    golden.check("in.report", IN.validateDataset(processedRoot))
    golden.check("in.report_str", IN.validateDataset(str(processedRoot)))


def test_rotation_report(golden):
    tRng = np.random.default_rng(0)
    tD6 = np.tile([1.0, 0, 0, 0, 1, 0], (10, 1)) + np.r_[np.zeros((8, 6)), tRng.normal(0, 0.01, (2, 6))]
    tD6[3] = np.nan
    golden.check("gc.rotation_report", GC.rotationReport(tD6))
    golden.check("gc.rotation_report_tol", GC.rotationReport(tD6, tol=0.1))


def test_plot_samples(processedRoot, tmp_path):
    tCart = np.load(processedRoot / "episodes" / "20260101_000000_0001" / "cart20.npy")
    tPng = tmp_path / "s.png"
    plotSamples([tCart[0], tCart[5]], ["a", "b"], tPng)
    assert tPng.stat().st_size > 10_000 and tPng.read_bytes()[:4] == b"\x89PNG"
