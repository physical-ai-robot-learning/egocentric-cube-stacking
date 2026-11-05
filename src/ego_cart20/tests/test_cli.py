"""Characterization of the command-line entry points (run as `python -m <module>` subprocesses on synthetic data) and of the
LeRobot exporter / HandUMI source adapter parts that run without LeRobot or the HandUMI pose stack."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from ego_cart20.scripts import export_lerobot as EL
from ego_cart20 import handumi_export as HE

from .synthetic import writeRawRoot, writeVideo

SRC = pathlib.Path(__file__).resolve().parents[2]
MOD_CONVERT_DATASET = "ego_cart20.scripts.convert_dataset"
MOD_CONVERT_EPISODE = "ego_cart20.scripts.convert_episode"
MOD_VALIDATE = "ego_cart20.scripts.validate_dataset"
MOD_REPORT = "ego_cart20.scripts.report_dataset"
MOD_EXPORT_LEROBOT = "ego_cart20.scripts.export_lerobot"
MOD_HANDUMI_EXPORT = "ego_cart20.handumi_export"
VOLATILE_DATASET_META = ("created", "raw_root", "code_sha256")


def _run(module, *args, env=None, check=True):
    tEnv = dict(os.environ, PYTHONPATH=str(SRC), MPLBACKEND="Agg", **(env or {}))
    tRes = subprocess.run([sys.executable, "-m", module, *map(str, args)], capture_output=True, text=True, env=tEnv, timeout=600)
    if check:
        assert tRes.returncode == 0, tRes.stderr
    return tRes


@pytest.mark.parametrize("module", [MOD_CONVERT_DATASET, MOD_EXPORT_LEROBOT, MOD_HANDUMI_EXPORT])
def test_argparse_help(golden, module):
    tRes = _run(module, "--help")
    golden.check(f"cli.help_{module.rsplit('.', 1)[1]}", tRes.stdout)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tRoot = tmp_path_factory.mktemp("cli")
    tVideo = writeVideo(tRoot / "head.mp4", 200)
    writeRawRoot(tRoot / "raw", videos={"head": str(tVideo), "left_wrist": str(tVideo)})
    (tRoot / "raw" / "not_an_episode").mkdir()
    tRes = _run(MOD_CONVERT_DATASET, tRoot / "raw", tRoot / "hash", "--split", "hash", "--workers", "2")
    tResB = _run(MOD_CONVERT_DATASET, tRoot / "raw", tRoot / "v2b", "--split", "hash", "--workers", "1", "--discontinuity-filter",
                 "--jump-window", "row", "--jump-isolated-step-m", "0.05")
    return tRoot, tRes, tResB


def _datasetMeta(path):
    tMeta = json.loads((path / "metadata.json").read_text())
    assert set(VOLATILE_DATASET_META) <= set(tMeta)
    assert all(tKey.endswith(".py") for tKey in tMeta["code_sha256"])
    return {tK: tV for tK, tV in tMeta.items() if tK not in VOLATILE_DATASET_META}


def test_convert_dataset_cli(golden, converted):
    tRoot, tRes, tResB = converted
    golden.check("cli.convert_stdout", tRes.stdout)
    golden.check("cli.convert_v2b_stdout", tResB.stdout)
    for tName in ("hash", "v2b"):
        tOut = tRoot / tName
        golden.check(f"cli.convert_{tName}_meta", _datasetMeta(tOut))
        golden.check(f"cli.convert_{tName}_meta_keys", list(json.loads((tOut / "metadata.json").read_text())))
        for tS in ("train", "val", "test"):
            golden.check(f"cli.convert_{tName}_{tS}_manifest", (tOut / f"{tS}_manifest.jsonl").read_text())
        tFiles = sorted(str(tP.relative_to(tOut)) for tP in tOut.rglob("*") if tP.name != "metadata.json")
        golden.check(f"cli.convert_{tName}_files", tFiles)
    with pytest.raises(AssertionError, match="exists"):
        _run(MOD_CONVERT_DATASET, tRoot / "raw", tRoot / "hash")


def test_convert_dataset_frozen_split(golden, converted, tmp_path):
    tRoot = converted[0]
    (tmp_path / "c8" / "c8old_rows").mkdir(parents=True)
    (tmp_path / "c8" / "c8old_rows" / "index.json").write_text(json.dumps({"val_episodes": ["20260101_000000_0002"]}))
    (tmp_path / "umi" / "track_c" / "v2k").mkdir(parents=True)
    (tmp_path / "umi" / "track_c" / "v2k" / "hrl80_split_frozen.json").write_text(
        json.dumps({"val_source_episodes": ["20260101_000000_0004"], "sha256": "f00"}))
    tRes = _run(MOD_CONVERT_DATASET, tRoot / "raw", tmp_path / "frozen", "--workers", "1",
                env=dict(EGO_DATA_ROOT=str(tmp_path / "c8"), EGO_UMI_BRIDGE=str(tmp_path / "umi")))
    golden.check("cli.convert_frozen_stdout", tRes.stdout)
    golden.check("cli.convert_frozen_meta", _datasetMeta(tmp_path / "frozen"))


def test_convert_episode_cli(golden, converted, tmp_path):
    tRoot = converted[0]
    tRes = _run(MOD_CONVERT_EPISODE, tRoot / "raw" / "20260101_000000_0003", tmp_path / "one")
    golden.check("cli.convert_episode_stdout", tRes.stdout)
    golden.check("cli.convert_episode_meta", json.loads((tmp_path / "one" / "metadata.json").read_text()))
    tFail = _run(MOD_CONVERT_EPISODE, check=False)
    assert tFail.returncode != 0


def test_validate_dataset_cli(golden, converted, tmp_path):
    tRoot = converted[0]
    tRes = _run(MOD_VALIDATE, tRoot / "hash", check=False)
    golden.check("cli.validate_rc", tRes.returncode)
    golden.check("cli.validate_stdout", tRes.stdout)
    golden.check("cli.validate_report", json.loads((tRoot / "hash" / "integrity_report.json").read_text()))
    tRes2 = _run(MOD_VALIDATE, tRoot / "v2b", tmp_path / "r.json", check=False)
    golden.check("cli.validate_v2b_rc", tRes2.returncode)
    golden.check("cli.validate_v2b_report", json.loads((tmp_path / "r.json").read_text()))


def test_report_dataset_cli(golden, converted, tmp_path):
    tRoot = converted[0]
    tRes = _run(MOD_REPORT, tRoot / "hash", tmp_path / "rep", 2)
    golden.check("cli.report_stdout", tRes.stdout.replace(str(tmp_path), "<TMP>"))
    assert (tmp_path / "rep" / "trajectory_samples.png").stat().st_size > 10_000
    tRes4 = _run(MOD_REPORT, tRoot / "hash", tmp_path / "rep4")
    golden.check("cli.report4_stdout", tRes4.stdout.replace(str(tmp_path), "<TMP>"))


# ---------------------------------------------------------------------------------------------------------------- export_lerobot
def test_export_lerobot_constants(golden):
    golden.check("el.cams", EL.CAMS)
    golden.check("el.stat_keys", EL.STAT_KEYS)
    golden.check("el.features", EL.features())
    assert EL.AGG.name == "rel16ego"


def test_export_lerobot_stats(golden):
    tX = np.random.default_rng(0).normal(size=(50, 4))
    golden.check("el.stats", EL.stats(tX))


def test_export_lerobot_decode224(golden, tmp_path):
    tVideo = writeVideo(tmp_path / "v.mp4", 20)
    tOut = EL.decode224(tVideo, np.array([3, -1, 7, 3]))
    assert sorted(tOut) == [3, 7] and tOut[3].shape == (224, 224, 3)
    golden.check("el.decode_px", np.stack([tOut[3][::32, ::32], tOut[7][::32, ::32]]).astype(np.int64))
    assert EL.decode224(tVideo, np.array([-1])) == {}


def test_export_lerobot_fix_stats(golden, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tRng = np.random.default_rng(1)
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "meta").mkdir()
    for tI in range(2):
        tS = tRng.normal(size=(7, 20)).astype(np.float32)
        tA = np.zeros((7, 16, 32), np.float32)
        tA[..., :20] = tRng.normal(size=(7, 16, 20))
        tTable = pa.table({"observation.state": pa.array(tS.tolist(), pa.list_(pa.float32())),
                           "action": pa.array(tA.tolist(), pa.list_(pa.list_(pa.float32())))})
        pq.write_table(tTable, tmp_path / "data" / "chunk-000" / f"file-{tI:03d}.parquet")
    (tmp_path / "meta" / "stats.json").write_text(json.dumps({"aux.q_t": {"mean": [1.0] * 12, "std": [2.0] * 12, "count": [14]},
                                                              "other": {"mean": [3.0]}}))
    golden.check("el.fix_stats", EL.fixStats(tmp_path))
    golden.check("el.fix_stats_json", json.loads((tmp_path / "meta" / "stats.json").read_text()))


def test_export_lerobot_write_shard_resumes_from_marker(tmp_path):
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "ep1.complete").write_text(json.dumps({"episode_id": "ep1", "frames": 3}))
    assert EL.writeShard((str(tmp_path / "ep1"), str(tmp_path / "shards" / "ep1"))) == {"episode_id": "ep1", "frames": 3}


# ---------------------------------------------------------------------------------------------------------------- handumi_export
def test_handumi_export_layout(golden, tmp_path, monkeypatch):
    golden.check("he.consts", [HE.HRL_SESSION, list(HE.SIDES), str(HE.RAW_OLD.relative_to(HE.EGO_COLLECTOR_HOME)),
                               str(HE.RAW_HRL.relative_to(HE.EGO_COLLECTOR_HOME)), str(HE.V2K.relative_to(HE.UMI_BRIDGE))])
    golden.check("he.episode_dir_old", str(HE._episodeDir("20260901_120000_0007").relative_to(HE.EGO_COLLECTOR_HOME)))
    golden.check("he.episode_dir_hrl", str(HE._episodeDir("20260928_101010_0012").relative_to(HE.EGO_COLLECTOR_HOME)))
    (tmp_path / "c8").mkdir()
    (tmp_path / "c8" / "phase3_census.json").write_text(json.dumps({"episodes": {
        "e3": {"first_fail": None, "left": {"scale_valid": True}, "right": {"scale_valid": True}},
        "e1": {"first_fail": "no_segment", "left": {"scale_valid": True}, "right": {"scale_valid": True}},
        "e2": {"first_fail": "ik", "left": {"scale_valid": True}, "right": {"scale_valid": True}},
        "e4": {"first_fail": None, "left": {"scale_valid": True}, "right": {"scale_valid": False}},
        "e5": {"first_fail": None, "right": {"scale_valid": True}}}}))
    (tmp_path / "v2k").mkdir()
    (tmp_path / "v2k" / "hrl80_split_frozen.json").write_text(
        json.dumps({"train_source_episodes": ["h3", "h1"], "val_source_episodes": ["h2"]}))
    monkeypatch.setattr(HE, "C8", tmp_path / "c8")
    monkeypatch.setattr(HE, "V2K", tmp_path / "v2k")
    golden.check("he.source_list", HE.sourceList())
