"""Characterization of the Gen-1 census stage: segment-segment clearance, the imu_vi + gravity variant of the frozen C-P6
fit, and the Phase-3 segment census (staging, metric TCP, 65-frame segments, workspace / IK / state32 funnel) on a
synthetic episode with the fake reBot solver."""
from __future__ import annotations

import importlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from rebot_prep.tests import synthetic_phase3 as P3S
from rebot_prep.tests.golden import assertGolden

GOLDEN = Path(__file__).with_name("golden")
PHASE3 = "rebot_prep.census.phase3_segments"
CLEARANCE = "rebot_prep.census.arm_clearance"
IMU_VI_G = "rebot_prep.census.imu_vi_gravity"


def _round(a, nd=9):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def test_segment_distance_cases():
    tSegseg = importlib.import_module(CLEARANCE).segmentDistance
    tO, tX, tY = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert tSegseg(tO, tO, tX, tX) == 1.0                                         # two points
    assert tSegseg(tO, tO, tY + tX, tY - tX) == 1.0                               # point - segment
    assert tSegseg(tY + tX, tY - tX, tO, tO) == 1.0                               # segment - point
    assert tSegseg(tO, tX, tY, tY + tX) == 1.0                                    # parallel
    assert tSegseg(-tX, tX, -tY + [0, 0, 0.5], tY + [0, 0, 0.5]) == 0.5          # crossing
    assert tSegseg(tO, tX, tX * 3 + tY, tX * 2 + tY) == pytest.approx(np.sqrt(2))  # clamped at both ends
    tRng = np.random.default_rng(4)
    tVals = [tSegseg(*tRng.normal(0, 1, (4, 3))) for _ in range(50)]
    assertGolden(GOLDEN, "segseg", _round(tVals, 12))


def test_clearance_of_the_two_arms(fakeRobot, roots):
    from rebot_prep.pseudo_joint import eef_kin
    importlib.reload(eef_kin)
    tC = importlib.import_module(CLEARANCE).Clearance(eef_kin.Kin())
    assert len(tC.mLinks) == 20 and importlib.import_module(CLEARANCE).CHAIN[-1] == ("gripper_link", "gripper_right")
    tOut = [tC.dmin(q) for q in (np.radians(P3S.ANCHOR_DEG12), np.zeros(12), np.r_[np.full(6, 0.3), np.full(6, -0.3)])]
    tPts = tC.points(np.zeros(12))
    assertGolden(GOLDEN, "clearance", dict(dmin=[[round(d, 12), w] for d, w in tOut], pts={k: _round(v) for k, v in tPts.items()}))


@pytest.fixture
def imuViG(roots, monkeypatch, tmp_path):
    from mast3r_pose.tests import synthetic as M3
    tBase = M3.writeBenchmark(tmp_path / "bench")
    P3S.buildTrackDirs(roots["UMI_BRIDGE"])
    monkeypatch.setenv("M3_BASE", str(tBase))
    monkeypatch.setenv("M3_VARIANT", "ss1")
    monkeypatch.delitem(sys.modules, IMU_VI_G, raising=False)
    return importlib.import_module(IMU_VI_G)


def test_imu_vi_g_equals_the_frozen_fit_plus_gravity(imuViG):
    tNs = imuViG.loadFrozenFit()
    tOut = {}
    for tTag in ("valB_d1_left", "valC_d3_right"):
        tRef, tG = tNs["imu_vi_ref"](tTag, 15, False), tNs["imu_vi"](tTag, 15, False)
        assert {k: tG[k] for k in tRef} == tRef
        assert np.linalg.norm(tG["g_world"]) == pytest.approx(tG["g_norm"])
        tOut[tTag] = tG
    assertGolden(GOLDEN, "imu_vi_g", tOut)


def test_imu_vi_g_script_verifies_the_given_tags(imuViG, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["c8_imu_vi_g", "valB_d1_left", "valB_d2_right"])
    try:
        runpy.run_module(IMU_VI_G, run_name="__main__")
    except SystemExit as tExit:
        assert tExit.code in (0, None)
    tOut = capsys.readouterr().out.splitlines()
    assert tOut[-1] == "IMU_VI_G_VERIFIED"
    assertGolden(GOLDEN, "imu_vi_g_script", tOut)


@pytest.fixture
def phase3(pipeline, roots, monkeypatch):
    P3S.buildTrackDirs(roots["UMI_BRIDGE"])
    tRaw = roots["EGO_COLLECTOR_HOME"] / "datasets/human_handumi_raw/Hpilot"
    P3S.buildC8(roots["DATA_ROOT"], tRaw)
    monkeypatch.delenv("C8_RUNS", raising=False)
    for tVar in ("M3_BASE", "M3_VARIANT"):
        monkeypatch.delenv(tVar, raising=False)
    monkeypatch.delitem(sys.modules, PHASE3, raising=False)
    return importlib.import_module(PHASE3)


def test_phase3_nearest_and_staging(phase3, roots):
    assert phase3.nearest(np.array([0.0, 10.0, 20.0]), np.array([-5.0, 4.0, 6.0, 14.0, 26.0])).tolist() == [0, 0, 1, 1, 2]
    tTag = f"{P3S.EPISODE_OK}_left"
    phase3.loadInputs()
    assert phase3.stageSide(tTag) == (1, 2)
    tStaged = (roots["DATA_ROOT"] / "stage_p3/runs" / tTag / "ss1.csv").read_text().splitlines()
    assert tStaged[1].startswith("0,") and len(tStaged) == 1 + P3S.N_CORI
    assert tStaged[1 + P3S.NAN_ROW - 2].split(",")[3] == "true"
    assert phase3.NF_IDX[tTag] == [P3S.NAN_ROW - 2]
    assert (roots["DATA_ROOT"] / "stage_p3/val/20260917/150545_0_left").resolve() == (roots["DATA_ROOT"] / "export" / tTag).resolve()


def test_phase3_episode_records(phase3, roots, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    phase3.loadInputs()
    phase3.init()
    for tSide in ("left", "right"):
        phase3.stageSide(f"{P3S.EPISODE_OK}_{tSide}")
    tRec = phase3.episodeRecord(P3S.EPISODE_OK)
    assert phase3.episodeRecord(P3S.EPISODE_PRE_FAIL) == dict(episode=P3S.EPISODE_PRE_FAIL, domain="main_calibrated",
                                                             first_fail="gripper_valid")
    assert phase3.episodeRecord(P3S.EPISODE_NO_RUN)["first_fail"] == "mast3r_missing"
    tZ = np.load(roots["DATA_ROOT"] / "phase3" / f"{P3S.EPISODE_OK}.npz")
    tWs = phase3.wsIn(np.stack([np.eye(4)] * 2), 0)
    assertGolden(GOLDEN, "phase3_episode", dict(
        rec=json.loads(json.dumps(tRec, default=float)), npz={k: [list(tZ[k].shape), _round(tZ[k], 5) if tZ[k].dtype.kind == "f" else tZ[k].tolist()]
                                                              for k in sorted(tZ.files)},
        ws_eye=tWs.tolist()))


def test_phase3_funnel_when_the_workspace_gate_passes(phase3, roots, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    phase3.loadInputs()
    phase3.init()
    for tSide in ("left", "right"):
        phase3.stageSide(f"{P3S.EPISODE_OK}_{tSide}")
    monkeypatch.setattr(phase3, "wsIn", lambda T, a: np.ones(len(T), bool))
    tRec = phase3.episodeRecord(P3S.EPISODE_OK)
    assert [s["first_fail"] for s in tRec["segments"]] == [None, None]
    tZ = np.load(roots["DATA_ROOT"] / "phase3" / f"{P3S.EPISODE_OK}.npz")
    assert tZ["usable"].tolist() == [True, True] and tZ["arm_usable"].tolist() == [[True, True], [True, True]]
    assertGolden(GOLDEN, "phase3_episode_ws_pass", json.loads(json.dumps(tRec["segments"], default=float)))


# ---------------------------------------------------------------- Phase-1 regression helpers (ORB poses, gyro check)
REGRESSION = "rebot_prep.census.mast3r_regression"


def test_regression_orb_loader(tmp_path, monkeypatch):
    tQuat = np.array([[0, 0, 0, 1.0], [0, 0, 0, 0], [0, 0, np.sin(0.2), np.cos(0.2)], [0, 0, 0, 1.0]])
    np.savez(tmp_path / "t.npz", frame_idx=np.array([0, 1, 3, 9]), cam_quat=tQuat, cam_pos=np.arange(12.0).reshape(4, 3))
    tMod = importlib.import_module(REGRESSION)
    monkeypatch.setattr(tMod, "PRODUCE", tmp_path)
    tT, tOk = tMod.orbPoses("t", 5)
    assert tOk.tolist() == [True, False, False, True, False]
    assert np.allclose(tT[3, :3, 3], [6, 7, 8]) and np.allclose(tT[1], np.eye(4))


def test_regression_gyro_check_on_the_synthetic_benchmark(tmp_path):
    from mast3r_pose.tests import synthetic as M3
    from rebot_prep.tests.scriptdefs import loadDefs
    tBase = M3.writeBenchmark(tmp_path / "bench")
    tFrozen = Path(__file__).with_name("data") / "cp6_scale_frozen.py.txt"
    tCp6 = loadDefs(tFrozen, {"load_traj", "T_b_c"}, {"__name__": "cp6_defs"})
    tTag = "valB_d1_left"
    tSide = tBase / "val" / "valB" / "d1_left"
    tTelemetry = json.loads((tSide / "imu_data.json").read_text())
    tCori = tTelemetry["1"]["streams"]["CORI"]["samples"]
    tT, tOk = tCp6["load_traj"](tBase / "runs" / tTag / "ss1.csv", M3.N_FRAMES)
    tRbc = tCp6["T_b_c"](tSide / "orbslam_setting.yaml")[:3, :3]
    from scipy.spatial.transform import Rotation
    tRate = [3.0 * tRbc @ Rotation.from_matrix(tT[k, :3, :3].T @ tT[k + 1, :3, :3]).as_rotvec() * 30.0 for k in range(len(tCori) - 1)]
    tTelemetry["1"]["streams"]["GYRO"] = {"samples": [{"cts": s["cts"] + 3.0, "value": tRate[k].tolist()}
                                                      for k, s in enumerate(tCori[:-1])]}
    (tSide / "imu_data.json").write_text(json.dumps(tTelemetry))
    tGyroCheck = importlib.import_module(REGRESSION).gyroCheck
    tRes = tGyroCheck(tCp6, tT, tOk, tSide)
    assert tGyroCheck(tCp6, tT[:20], tOk[:20], tSide) is None
    assertGolden(GOLDEN, "regress_gyro_check", tRes)
