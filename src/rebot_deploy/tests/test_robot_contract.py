"""Characterization of the robot <-> dataset contract (jaw scales, CART20 / RELCART20 state builders, delta clamp)."""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from rebot_deploy import v4_contract as contract

RAW_COUNTS = (-400.0, -300.0, -270.0, -135.0, -100.0, -2.0, 0.0, 0.01, 50.0, 179.9, 180.0, 181.0, 358.09)
WIDTHS_M = (-0.05, -0.01, 0.0, 0.01, 0.03, 0.0572, 0.1144, 0.2)


def _pose(rng):
    tMat = np.eye(4)
    tMat[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * 0.4).as_matrix()
    tMat[:3, 3] = rng.normal(size=3) * 0.2
    return tMat


def test_constants(golden):
    golden.check("contract.constants", {tName: getattr(contract, tName) for tName in (
        "CART20_W_OPEN", "GRIP_M_PER_RAW", "GRIP_RAW_MIN", "GRIP_RAW_MAX", "GRIP_OBS_PER_CMD", "GRIP_CMD_MAX", "GRIP_BIN_OPEN",
        "GRIP_BIN_CLOSED", "GRIP_CONTRACT_FILE", "GRIP_BIN_THRESH", "GRIP_CMD_FULL", "GRIP_REL_EPS", "GRIP_REL_SCALE",
        "GRIP_REL_MODE", "GRIP_ACC_N", "GRIP_ACC_OPEN", "GRIP_ACC_CLOSE", "FIXED_ANCHOR_R")})


def test_jaw_conversions(golden):
    golden.check("contract.unwrap", [contract.unwrap_grip(tRaw) for tRaw in RAW_COUNTS])
    golden.check("contract.raw_to_width", [contract.raw_to_width(tRaw) for tRaw in RAW_COUNTS])
    golden.check("contract.width_to_raw", [contract.width_to_raw(tW) for tW in WIDTHS_M])
    golden.check("contract.width_to_cmd", [contract.width_to_cmd(tW) for tW in WIDTHS_M])
    golden.check("contract.jaw_hold_cmd", [contract.jaw_hold_cmd(tRaw) for tRaw in RAW_COUNTS])


def test_state_builders(golden):
    tRng = np.random.default_rng(3)
    tAnchor = np.stack([_pose(tRng), _pose(tRng)])
    tCur = np.stack([_pose(tRng), _pose(tRng)])
    golden.check("contract.relcart20", contract.relcart20_state(tAnchor, tCur, [0.01, 0.2]).astype(float))
    golden.check("contract.relcart20_identity", contract.relcart20_state(tCur, tCur, [0.0, 0.05]).astype(float))
    golden.check("contract.cart20", contract.cart20_state(tCur, [-0.01, 0.03]).astype(float))


def test_clamp_delta(golden):
    tRng = np.random.default_rng(4)
    tOut = []
    for tMaxMm, tMaxDeg in ((0, 0), (5, 1), (1000, 90), (3, 0), (0, 2), (50, 10)):
        tDelta = _pose(tRng)
        tClamped, tWasClamped, tNormMm, tAngDeg = contract.clamp_delta(tDelta, tMaxMm, tMaxDeg)
        tOut.append([tClamped, tWasClamped, tNormMm, tAngDeg])
    golden.check("contract.clamp_delta", tOut)


def test_grip_contract_reads_a_passed_file_next_to_the_module(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "__file__", str(tmp_path / "module.py"))
    assert contract._grip_contract() == (42.0, 0.0, None)
    tFile = tmp_path / "gripper_contract_v2.json"
    tFile.write_text(json.dumps({"passed": False, "cmd_open": 30, "cmd_closed": 5}))
    assert contract._grip_contract() == (42.0, 0.0, None)
    tFile.write_text(json.dumps({"passed": True, "cmd_open": 30, "cmd_closed": 5}))
    assert contract._grip_contract() == (30.0, 5.0, str(tFile))
    tFile.write_text("not json")
    assert contract._grip_contract() == (42.0, 0.0, None)
