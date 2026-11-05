"""Characterization of the export_episode CLI (--help text with flags, defaults and choices)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mast3r_pose.tests.golden import assertGolden

GOLDEN = Path(__file__).with_name("golden")
SRC = Path(__file__).resolve().parents[2]


def test_export_episode_help_is_unchanged(tmp_path):
    tEnv = {k: v for k, v in os.environ.items() if not k.startswith(("EGO_", "PYTHON"))}
    tEnv.update(HOME=str(tmp_path), PYTHONPATH=str(SRC), COLUMNS="200")
    tRun = subprocess.run([sys.executable, "-m", "mast3r_pose.export_episode", "--help"], env=tEnv, capture_output=True, text=True)
    assert tRun.returncode == 0, tRun.stderr[-2000:]
    tLines = tRun.stdout.splitlines()
    tLines[0] = tLines[0].split(":", 1)[1].split(" ", 2)[-1]
    assertGolden(GOLDEN, "help_export_episode", tLines)


def test_export_episode_defaults_are_unchanged(tmp_path):
    from rebot_prep.tests.test_cli import namespaceOf
    assertGolden(GOLDEN, "namespace_export_episode", namespaceOf("mast3r_pose.export_episode", ["ep", "left", "out"], tmp_path))
