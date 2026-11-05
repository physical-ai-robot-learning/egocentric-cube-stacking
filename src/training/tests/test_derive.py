"""Characterization of the UMI76 dataset derivations on synthetic LeRobot trees: derive_v2 (variants A / C, --selftest,
--remap-root) and derive_relcart20 (RELCART20 state from aux.q_t through the torch FK on a synthetic URDF). Both run as CLIs in a
subprocess, exactly like the node chains run them."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

import numpy as np

from .conftest import REPO_SRC
from .synth_lerobot import readTree, writeParent, writeSource


def _run(module, args, tmpPath, extraEnv=None):
    tEnv = dict(os.environ, PYTHONPATH=str(REPO_SRC), EGO_NODE_HOME=str(tmpPath / "no_node_home"), **(extraEnv or {}))
    tRes = subprocess.run([sys.executable, "-m", module, *args], env=tEnv, capture_output=True, text=True)
    assert tRes.returncode == 0, tRes.stderr
    return tRes.stdout.replace(str(tmpPath), "<tmp>")


def _parentV1(tmpPath, sourceRoot=None):
    tSource = writeSource(tmpPath / "src_lerobot")
    return writeParent(tmpPath / "parent", sourceRoot or str(tSource))


def test_derive_v2_variant_a(tmp_path, golden):
    tParent = _parentV1(tmp_path)
    tOut = tmp_path / "out_A"
    tStdout = _run("training.umi76.derive_v2", ["--parent", str(tParent), "--out", str(tOut), "--variant", "A"], tmp_path)
    golden.check("derive_v2/A/stdout", tStdout)
    golden.check("derive_v2/A/tree", readTree(tOut, tmp_path))
    tVideo = "videos/observation.images.global/chunk-000/file-000.mp4"
    assert os.stat(tOut / tVideo).st_ino == os.stat(tParent / tVideo).st_ino          # hard link, never a copy


def test_derive_v2_variant_c_with_remapped_source(tmp_path, golden):
    tParent = _parentV1(tmp_path, sourceRoot="/node/holobrain-data/lerobot/src")
    tOut = tmp_path / "out_C"
    tStdout = _run("training.umi76.derive_v2", ["--parent", str(tParent), "--out", str(tOut), "--variant", "C",
                                                "--remap-root", f"/node/holobrain-data/lerobot/src={tmp_path / 'src_lerobot'}"],
                   tmp_path)
    golden.check("derive_v2/C/stdout", tStdout)
    golden.check("derive_v2/C/tree", readTree(tOut, tmp_path))


def test_derive_v2_selftest_and_refusal(tmp_path, golden):
    tParent = _parentV1(tmp_path)
    golden.check("derive_v2/selftest/stdout", _run("training.umi76.derive_v2", ["--parent", str(tParent), "--selftest"], tmp_path))
    (tmp_path / "exists").mkdir()
    tEnv = dict(os.environ, PYTHONPATH=str(REPO_SRC), EGO_NODE_HOME=str(tmp_path / "no_node_home"))
    tRes = subprocess.run([sys.executable, "-m", "training.umi76.derive_v2", "--parent", str(tParent), "--out",
                           str(tmp_path / "exists"), "--variant", "A"], env=tEnv, capture_output=True, text=True)
    assert tRes.returncode == 1
    assert tRes.stderr.strip().splitlines()[-1] == f"{tmp_path / 'exists'} exists, refusing to overwrite"


def test_derive_v2_helpers(tmp_path, golden):
    tDv = importlib.import_module("training.umi76.derive_v2")
    tX = np.arange(30, dtype=np.float64).reshape(10, 3) ** 1.5
    golden.check("derive_v2/stats", tDv.columnStats(tX))
    tSource = writeSource(tmp_path / "src")
    tBy = tDv.loadSource(str(tSource), ["action"])
    golden.check("derive_v2/load_src", tBy)
    tParent = writeParent(tmp_path / "parent", str(tSource))
    golden.check("derive_v2/row_sources", {str(tK): [str(tV[0]).replace(str(tmp_path), "<tmp>"), *tV[1:]]
                                           for tK, tV in tDv.rowSources(tParent).items()})


def test_derive_relcart20(tmp_path, synthUrdf, golden):
    tSource = writeSource(tmp_path / "src_lerobot")
    tParent = writeParent(tmp_path / "parent_v3d", str(tSource), actionDim=32, withQt=True)
    tOut = tmp_path / "out_relcart20"
    tStdout = _run("training.umi76.derive_relcart20", ["--parent", str(tParent), "--out", str(tOut)], tmp_path,
                   {"REBOT_URDF": str(synthUrdf)})
    golden.check("relcart20/stdout", tStdout)
    tTree = readTree(tOut, tmp_path)
    golden.check("relcart20/tree", tTree)
    tState = tTree["data/chunk-000/file-000.parquet:observation.state"]
    assert np.abs(tState[0, :18] - [0, 0, 0, 1, 0, 0, 0, 1, 0] * 2).max() < 1e-6       # anchor row = identity


def test_derive_v2_variant_b_appends_tcp18(tmp_path, golden, monkeypatch):
    """variant B in-process: the UMI zarr is a stand-in, the pose helpers are the repository copy (src/umi)"""
    import types
    tSource = writeSource(tmp_path / "src_lerobot")
    tParent = writeParent(tmp_path / "parent", str(tSource))
    tRng = np.random.default_rng(5)
    tZarr = {"meta": {"episode_ends": np.array([60, 120])},
             "data": {f"robot{tR}_{tK}": tRng.normal(scale=0.3, size=(120, 3))
                      for tR in (0, 1) for tK in ("eef_pos", "eef_rot_axis_angle")}}
    monkeypatch.setitem(sys.modules, "zarr", types.SimpleNamespace(open=lambda path, mode: tZarr))
    monkeypatch.setenv("UMI_ROOT", str(tmp_path / "no_umi_checkout"))
    tPath = list(sys.path)
    for tName in [tN for tN in sys.modules if tN.startswith("training.umi76")]:
        del sys.modules[tName]
    tDv = importlib.import_module("training.umi76.derive_v2")
    tOut = tmp_path / "out_B"
    monkeypatch.setattr(sys, "argv", ["derive_v2.py", "--parent", str(tParent), "--out", str(tOut), "--variant", "B"])
    try:
        tDv.main()
    finally:
        sys.path[:] = tPath
    golden.check("derive_v2/B/tree", readTree(tOut, tmp_path))
