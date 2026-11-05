"""Fixtures of the rebot_prep characterization tests: the fake reBot solver and a fresh pseudo_joint_pipeline import whose
machine roots (camera->TCP yaml, external checkouts) point into tmp_path."""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

import ego_paths
from rebot_prep.tests import fake_robot

R_CAMERA_TCP = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
T_CAMERA_TCP_M = [0.005, -0.04, 0.15]
PIPELINE = "rebot_prep.pseudo_joint.pseudo_joint_pipeline"


@pytest.fixture
def fakeRobot(monkeypatch):
    fake_robot.install(monkeypatch)
    return fake_robot


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """ego_paths roots -> empty folders under tmp_path (nothing from the real home can leak in)."""
    tRoots = {}
    for tName in ("HOME", "DATA_ROOT", "UMI_BRIDGE", "EGO_COLLECTOR_HOME", "HOLOBRAIN_MAC", "SHARED_ROOT"):
        tDir = tmp_path / "roots" / tName.lower()
        tDir.mkdir(parents=True)
        monkeypatch.setattr(ego_paths, tName, tDir)
        tRoots[tName] = tDir
    monkeypatch.setattr(sys, "path", list(sys.path))
    for tMod in ("eef_kin", "pseudo_joint_pipeline", "state32", "c8_imu_vi_g", "c8_collision_v1", "extract_grip", "v2k_compare"):
        monkeypatch.delitem(sys.modules, tMod, raising=False)
    return tRoots


@pytest.fixture
def pipeline(roots, fakeRobot, monkeypatch):
    tYaml = roots["EGO_COLLECTOR_HOME"] / "configs/calibration/camera_tcp/handumi_camera_tcp_v2.yaml"
    tYaml.parent.mkdir(parents=True)
    tYaml.write_text(yaml.safe_dump({"rotation": {"R_camera_tcp": R_CAMERA_TCP.tolist()},
                                     "translation": {"t_camera_tcp_m": T_CAMERA_TCP_M}}))
    for tMod in [m for m in sys.modules if m.startswith("rebot_prep.pseudo_joint.")]:
        monkeypatch.delitem(sys.modules, tMod)
    return importlib.import_module(PIPELINE)


def humanMotion(n: int, phase: float = 0.0) -> np.ndarray:
    """(n, 4, 4) smooth human TCP trajectory (metres, a few cm of motion and some wrist rotation)"""
    tS = np.arange(n) / 30.0
    tT = np.tile(np.eye(4), (n, 1, 1))
    tT[:, :3, 3] = np.stack([0.3 + 0.04 * np.sin(2.0 * tS + phase), 0.1 + 0.03 * np.cos(1.5 * tS), 0.2 + 0.02 * tS], -1)
    tT[:, :3, :3] = Rotation.from_rotvec(np.stack([0.2 * np.sin(tS + phase), 0.1 * tS, 0.05 * np.cos(tS)], -1)).as_matrix()
    return tT
