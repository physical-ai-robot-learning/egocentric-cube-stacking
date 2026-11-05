"""Characterization of ego_paths: every constant, its default and its environment variable; prepend_sys_path order."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import ego_paths

SRC = Path(__file__).resolve().parents[2]
ENV_OF = {"DATA_ROOT": "EGO_DATA_ROOT", "UMI_BRIDGE": "EGO_UMI_BRIDGE", "EGO_COLLECTOR_HOME": "EGO_COLLECTOR_HOME",
          "HOLOBRAIN_MAC": "EGO_HOLOBRAIN_MAC", "HOLOBRAIN_DATA": "EGO_HOLOBRAIN_DATA", "NODE_HOME": "EGO_NODE_HOME",
          "SHARED_ROOT": "EGO_SHARED_ROOT", "CKPT_ROOT": "EGO_CKPT_ROOT", "LAUNDRY_SYSTEM": "EGO_LAUNDRY_SYSTEM",
          "LAUNDRY_DIT_DATASETS": "EGO_LAUNDRY_DIT_DATASETS"}
DEFAULTS = {"DATA_ROOT": "{home}/c8", "UMI_BRIDGE": "{home}/umi_bridge", "EGO_COLLECTOR_HOME": "{home}/ego_collector",
            "HOLOBRAIN_MAC": "{home}/holobrain-mac-model", "HOLOBRAIN_DATA": "{home}/holobrain-data",
            "NODE_HOME": "/home/bh-aiteam", "SHARED_ROOT": "/srv/data/johann",
            "CKPT_ROOT": "/Volumes/PortableSSD/rebot_ckpts_archive", "LAUNDRY_SYSTEM": "{home}/laundry_system",
            "LAUNDRY_DIT_DATASETS": "{home}/laundry_dit_training/datasets"}


def _constants(env: dict) -> dict:
    tCode = "import json, ego_paths; print(json.dumps({k: str(getattr(ego_paths, k)) for k in ego_paths.__all__ if k != 'prepend_sys_path'}))"
    tEnv = {k: v for k, v in os.environ.items() if not k.startswith("EGO_")}
    tEnv.update(env, PYTHONPATH=str(SRC))
    return json.loads(subprocess.run([sys.executable, "-c", tCode], env=tEnv, capture_output=True, text=True, check=True).stdout)


def test_defaults_are_the_original_literals(tmp_path):
    tGot = _constants({"HOME": str(tmp_path)})
    for tName, tDefault in DEFAULTS.items():
        assert tGot[tName] == tDefault.format(home=tmp_path), tName
    assert tGot["HOME"] == str(tmp_path)
    assert tGot["REPO_ROOT"] == str(SRC.parent)


def test_every_root_follows_its_environment_variable(tmp_path):
    tGot = _constants({"HOME": str(tmp_path), **{tEnv: f"/x/{tName}" for tName, tEnv in ENV_OF.items()}})
    for tName in ENV_OF:
        assert tGot[tName] == f"/x/{tName}"


def test_an_empty_environment_variable_means_the_default(tmp_path):
    tGot = _constants({"HOME": str(tmp_path), "EGO_DATA_ROOT": ""})
    assert tGot["DATA_ROOT"] == f"{tmp_path}/c8"


def test_prepend_sys_path_keeps_the_argument_order(monkeypatch):
    monkeypatch.setattr(sys, "path", ["z"])
    ego_paths.prepend_sys_path("a")
    ego_paths.prepend_sys_path(Path("b"), "c")
    assert sys.path == ["b", "c", "a", "z"]
