"""Characterization of the C-P6 metric-scale benchmark (IMU-VI Savitzky-Golay + Huber IRLS fit) on a synthetic tree with a
known scale per side: the whole program (hyperparameter search, constants, UMI displacement errors, printed table, JSON)."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from mast3r_pose.tests.golden import assertGolden
from mast3r_pose.tests.synthetic import TAGS, writeBenchmark

GOLDEN = Path(__file__).with_name("golden")
MODULE = "mast3r_pose.imu_metric_scale"


def _run(base: Path, variant: str, monkeypatch, capsys) -> tuple[dict, str]:
    monkeypatch.setenv("M3_BASE", str(base))
    monkeypatch.setenv("M3_VARIANT", variant)
    monkeypatch.setattr("sys.argv", [MODULE])
    try:
        runpy.run_module(MODULE, run_name="__main__")
    except SystemExit as tExit:
        assert tExit.code in (0, None)
    tName = "cp6_scale.json" if variant == "ss1" else f"cp6_scale_{variant}.json"
    return json.loads((base / "qa" / tName).read_text()), capsys.readouterr().out


def test_benchmark_output_and_report_are_unchanged(tmp_path, monkeypatch, capsys):
    tOut, tText = _run(writeBenchmark(tmp_path), "ss1", monkeypatch, capsys)
    assertGolden(GOLDEN, "cp6_scale", dict(json=tOut, stdout=tText.splitlines()))


def test_imu_vi_recovers_the_true_scale_within_a_few_percent(tmp_path, monkeypatch, capsys):
    tOut, _ = _run(writeBenchmark(tmp_path), "ss1", monkeypatch, capsys)
    tVi = {r["tag"]: r["s"] for r in tOut["rows"] if r["method"] == "imu_vi"}
    for tTag, tScale in TAGS.items():
        assert tVi[tTag] == pytest.approx(tScale, rel=0.05)


def test_a_non_default_variant_writes_its_own_file(tmp_path, monkeypatch, capsys):
    tOut, _ = _run(writeBenchmark(tmp_path, variant="ss1_5080"), "ss1_5080", monkeypatch, capsys)
    assert tOut["imu_vi_choice"] == json.loads(json.dumps(tOut["imu_vi_choice"]))
    assert not (tmp_path / "qa" / "cp6_scale.json").exists()
