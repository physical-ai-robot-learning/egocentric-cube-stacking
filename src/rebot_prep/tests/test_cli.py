"""Characterization of the argparse CLIs of rebot_prep: the full --help text (flags, defaults, choices, help strings) with
machine paths normalised, run in a subprocess whose roots point into tmp_path."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from rebot_prep.tests import synthetic_phase3 as P3S
from rebot_prep.tests.golden import assertGolden

GOLDEN = Path(__file__).with_name("golden")
SRC = Path(__file__).resolve().parents[2]
CLIS = {"extract_grip": "rebot_prep.grip.extract_grip", "correct_grip": "rebot_prep.grip.correct_grip",
        "input_gate": "rebot_prep.pseudo_joint.c2_input_gate", "compare": "rebot_prep.retarget.v2k_compare"}


def _help(module: str, tmp_path: Path) -> list[str]:
    tHome = tmp_path / "home"
    tYaml = tHome / "ego_collector/configs/calibration/camera_tcp/handumi_camera_tcp_v2.yaml"
    tYaml.parent.mkdir(parents=True, exist_ok=True)
    tYaml.write_text(yaml.safe_dump({"rotation": {"R_camera_tcp": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                                     "translation": {"t_camera_tcp_m": [0, 0, 0]}}))
    if not (tHome / "umi_bridge/track_c/data/r150_q_tcp.npz").exists():
        P3S.writeReference(tHome / "umi_bridge/track_c/data/r150_q_tcp.npz")
        (tHome / "umi_bridge/track_c/v2k").mkdir(parents=True)
        (tHome / "umi_bridge/track_c/v2k/r30_episodes.json").write_text('{"episodes": [1, 2]}')
    tEnv = {k: v for k, v in os.environ.items() if not k.startswith(("EGO_", "PYTHON"))}
    tEnv.update(HOME=str(tHome), PYTHONPATH=str(SRC), COLUMNS="200")
    tRun = subprocess.run([sys.executable, "-m", module, "--help"], env=tEnv, capture_output=True, text=True, cwd=tmp_path)
    assert tRun.returncode == 0, tRun.stderr[-2000:]
    tText = tRun.stdout.replace(str(tHome), "<home>").replace(str(SRC), "<src>")
    tLines = tText.splitlines()
    tLines[0] = tLines[0].split(":", 1)[1].split(" ", 2)[-1]          # drop "usage: <prog>"
    return tLines


@pytest.mark.parametrize("name", sorted(CLIS))
def test_cli_help_is_unchanged(name, tmp_path):
    assertGolden(GOLDEN, f"help_{name}", _help(CLIS[name], tmp_path))


NAMESPACE_CHILD = """
import argparse, json, runpy, sys
_parse = argparse.ArgumentParser.parse_args
def _capture(self, args=None, namespace=None):
    print("NAMESPACE" + json.dumps(vars(_parse(self, args, namespace)), sort_keys=True))
    sys.exit(0)
argparse.ArgumentParser.parse_args = _capture
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_module(sys.argv[0], run_name="__main__")
"""
ARGV = {"extract_grip": [], "correct_grip": [], "input_gate": [], "compare": ["--group", "a=b", "--out", "o.json"]}


def namespaceOf(module: str, argv: list[str], home: Path, src: Path = SRC) -> dict:
    """the parsed argparse Namespace of `python -m module argv` (defaults), machine paths normalised"""
    tEnv = {k: v for k, v in os.environ.items() if not k.startswith(("EGO_", "PYTHON"))}
    tEnv.update(HOME=str(home), PYTHONPATH=str(src))
    tRun = subprocess.run([sys.executable, "-c", NAMESPACE_CHILD, module, *argv], env=tEnv, capture_output=True, text=True, cwd=home)
    tLine = next((l for l in tRun.stdout.splitlines() if l.startswith("NAMESPACE")), None)
    assert tLine is not None, tRun.stderr[-2000:]
    return json.loads(tLine[len("NAMESPACE"):].replace(str(home), "<home>").replace(str(src), "<src>"))


@pytest.mark.parametrize("name", sorted(CLIS))
def test_cli_defaults_are_unchanged(name, tmp_path):
    _help(CLIS[name], tmp_path)                                         # builds the roots under tmp_path/home
    assertGolden(GOLDEN, f"namespace_{name}", namespaceOf(CLIS[name], ARGV[name], tmp_path / "home"))
