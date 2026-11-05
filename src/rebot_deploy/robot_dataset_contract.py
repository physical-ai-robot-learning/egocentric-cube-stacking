"""Robot <-> dataset contract of the X-VLA inference core: jaw scales, the CART20 / RELCART20 state builders and the per-step
delta clamp (deploy stage; was v4_contract.py, moved out of infer_core_v4.py on 2026-11-01 unchanged).

In:  follower jaw readings (raw encoder counts), predicted jaw widths / gripper channels, TCP matrices (2, 4, 4).
Out: jaw widths (m), /execute_step jaw commands (leader frame, 0..45), 20-D states, clamped body-frame deltas.

Only constants fixed at import live here: settings the UI switches at run time (GRIPPER_MODE, so gripToCmd; RTC_ON;
REMOTE_INFER; IK_BACKEND; LIMITS) stay in xvla_inference, which re-exports every name below.
usage: from rebot_deploy.robot_dataset_contract import rawToWidth, relCart20State
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from umi.common.pose_util import mat_to_pose10d

# umi_schema.grip_linear_v1_anchor5cm_raw118: a 5 cm jaw opening reads 118 raw counts
GRIP_ANCHOR_WIDTH_M = 0.05
GRIP_ANCHOR_RAW = 118.0
GRIP_FULL_OPEN_RAW = 270.0                                       # |raw| of the fully open jaw
CART20_W_OPEN = GRIP_FULL_OPEN_RAW * GRIP_ANCHOR_WIDTH_M / GRIP_ANCHOR_RAW   # raw -270 = fully open, in m (derive_cart20.py)
STATE_DIM = 20                                                   # CART20 / RELCART20: per arm pos3 + rot6d, then g_L, g_R
POSE9 = 9                                                        # pos3 + rot6d rows
ARM_ROWS = (0, 1)                                                # left, right

# [2026-10-07 user "B를 만들어줄래"] V4_FIXED_ANCHOR_JSON = json with "T_base_F" (4x4, robot base, dataset-TCP convention): the
# RIGHT-arm RELCART20 anchor is this fixed frame instead of the first observation, so the state is the absolute pose in F (the
# HRA_A100 origin-anchored models: F = the HandUMI fingertip-on-X frame). Unset = unchanged behaviour.
FIXED_ANCHOR_R = (np.array(json.load(open(os.path.expanduser(os.environ["V4_FIXED_ANCHOR_JSON"])))["T_base_F"], np.float64)
                  if os.environ.get("V4_FIXED_ANCHOR_JSON") else None)


def relCart20State(anchor, cur, w) -> np.ndarray:
    """[2026-09-29] RELCART20 state (derive_relcart20.py layout): per arm T_rel = inv(T_anchor) @ T_t -> [pos3 | rot6d rows],
    then g_L, g_R = width / W_OPEN clipped to [0, 1]. anchor = TCP mats (2,4,4) at the rollout's first observation (training:
    the episode's frame_index-0 row). anchor == cur gives [0,0,0, 1,0,0,0,1,0] per arm. (was relcart20_state)"""
    tState = np.zeros(STATE_DIM, np.float32)
    for tRow in ARM_ROWS:
        tState[tRow * POSE9:tRow * POSE9 + POSE9] = mat_to_pose10d((np.linalg.inv(anchor[tRow]) @ cur[tRow])[None])[0]
    tState[18:20] = np.clip(np.asarray(w, np.float64) / CART20_W_OPEN, 0.0, 1.0)
    return tState


def cart20State(cur, w) -> np.ndarray:
    """[2026-09-29] CART20 state (derive_cart20.py layout): [L pos3 | L rot6d rows | R pos3 | R rot6d rows | g_L | g_R],
    cur = dataset-frame TCP mats at t (2,4,4), w = follower jaw widths at t (m); g = w / W_OPEN clipped to [0, 1], 1 = OPEN.
    (was cart20_state)"""
    tState = np.zeros(STATE_DIM, np.float32)
    for tRow in ARM_ROWS:
        tState[tRow * POSE9:tRow * POSE9 + POSE9] = mat_to_pose10d(cur[tRow][None])[0]
    tState[18:20] = np.clip(np.asarray(w, np.float64) / CART20_W_OPEN, 0.0, 1.0)
    return tState


GRIP_M_PER_RAW = GRIP_ANCHOR_WIDTH_M / GRIP_ANCHOR_RAW           # umi_schema.grip_linear_v1_anchor5cm_raw118, inverted
GRIP_RAW_MIN, GRIP_RAW_MAX = -270.0, 0.0
GRIP_WRAP_ABOVE_RAW = 180.0                                      # a reading above this is a wrapped closed jaw
GRIP_WRAP_RAW = 360.0                                            # encoder counts per wrap


def unwrapGrip(raw) -> float:
    """The follower's gripper encoder wraps: a jaw that is physically CLOSED can read 358 instead of -2. (was unwrap_grip)

    Measured on this robot 2026-09-24: left closed = 358.09, right closed = 0.01, and the zero re-latches
    on every power cycle (0 / -100 / +90 / 358 have all been seen). The dataset pipeline has always
    unwrapped this -- `grip_norm_dataset.py` does `g - 360 if g > 180` -- and the deployment never did, so
    the state's width channel was nonsense on a wrapped arm and, worse, a command of 0 asked a jaw sitting
    at 358 to travel 358 counts, which is the "TRAIN START throws the gripper wide open" symptom.
    """
    tRaw = float(raw)
    return tRaw - GRIP_WRAP_RAW if tRaw > GRIP_WRAP_ABOVE_RAW else tRaw


def rawToWidth(raw) -> float:
    """follower jaw reading (raw counts) -> jaw width (m) (was raw_to_width)"""
    return float(np.clip(abs(float(np.clip(unwrapGrip(raw), GRIP_RAW_MIN, GRIP_RAW_MAX))), 0, None)
                 * GRIP_M_PER_RAW)


def widthToRaw(widthM) -> float:
    """jaw width (m) -> raw counts, -270 (open) .. 0 (closed) (was width_to_raw)"""
    return float(np.clip(-abs(float(widthM)) / GRIP_M_PER_RAW, GRIP_RAW_MIN, GRIP_RAW_MAX))


# The model predicts a metric jaw WIDTH, /observe reports a raw count, and /execute_step takes a LEADER-frame
# command -- three different scales. Measured on this robot (GRIPPER_COMMAND_SCALE.md, re-checked per session
# because the raw zero re-latches at power-on): cmd * -6 = obs, full travel cmd 0..45, and raw 0 = jaws CLOSED
# with raw -270 = fully OPEN (video-confirmed 2026-09-22). Sending widthToRaw() straight out as the command,
# which the v3 core does, would ask for -270 on a 0..45 scale.
GRIP_OBS_PER_CMD = float(os.environ.get("V4_GRIP_OBS_PER_CMD", "-6.0"))
GRIP_CMD_MAX = float(os.environ.get("V4_GRIP_CMD_MAX", "45"))


def widthToCmd(widthM) -> float:
    """predicted jaw width [m] -> /execute_step leader-frame gripper command. (was width_to_cmd)"""
    return float(np.clip(widthToRaw(widthM) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))


# [2026-09-28] REL16-v2 binary gripper (V4_GRIPPER=binary). The v2 label is the LEADER's intent, 1 = OPEN and
# 0 = CLOSE (derive_v2.py), not a width: the v1 follower width stalls at cube contact, so commanding it gave no grip
# force. The actuator values live in gripper_contract_v2.json, written by `gripper_smoke_test.py --cube` only when
# the squeeze-on-cube gate passes; until then the 2026-09-19 smoke-test endpoints stand (CLOSED 0, OPEN 42).
GRIP_CONTRACT_NAME = "gripper_contract_v2.json"                  # next to this module
SMOKE_TEST_CMD_OPEN = 42.0
SMOKE_TEST_CMD_CLOSED = 0.0


def _gripContract():
    """(cmd_open, cmd_closed, contract path or None) from a passed gripper contract next to this module (was _grip_contract)"""
    tPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), GRIP_CONTRACT_NAME)
    try:
        tContract = json.load(open(tPath))
        if tContract.get("passed"):
            return float(tContract["cmd_open"]), float(tContract["cmd_closed"]), tPath
    except (OSError, ValueError, KeyError):
        pass
    return SMOKE_TEST_CMD_OPEN, SMOKE_TEST_CMD_CLOSED, None


GRIP_BIN_OPEN, GRIP_BIN_CLOSED, GRIP_CONTRACT_FILE = _gripContract()
# [2026-09-29] execution threshold on the model's gripper channel (binary v2 label, or continuous v3 g = clip(leader/45, 0, 1)):
# g >= V4_GRIP_THRESH -> OPEN. Default 0.5 (the v2 behaviour); 0.6 = cmd 27/45, the old binary rule. Frozen per deployment.
GRIP_BIN_THRESH = float(os.environ.get("V4_GRIP_THRESH", "0.5"))
GRIP_CMD_FULL = float(os.environ.get("V4_GRIP_CMD_FULL", "45"))   # cmd 0..45 = full jaw travel (derive_v2 CMD_FULL_OPEN)
GRIP_REL_EPS = float(os.environ.get("V4_GRIP_REL_EPS", "0.05"))   # relative mode deadband on delta-g
# divide g by this first (width-labelled runs: 0.11441 m -> 0..1)
GRIP_REL_SCALE = float(os.environ.get("V4_GRIP_REL_SCALE", "1.0"))
# [2026-09-29] V4_GRIP_REL_MODE=acc: sum of the last N delta-g (per executed waypoint) instead of one delta; OPEN if > ACC_OPEN,
# CLOSE if < ACC_CLOSE, else HOLD; the history is cleared after every OPEN/CLOSE so one burst does not keep re-triggering.
GRIP_REL_MODE = os.environ.get("V4_GRIP_REL_MODE", "delta")
GRIP_ACC_N = int(os.environ.get("V4_GRIP_ACC_N", "4"))
GRIP_ACC_OPEN = float(os.environ.get("V4_GRIP_ACC_OPEN", "0.06"))
GRIP_ACC_CLOSE = float(os.environ.get("V4_GRIP_ACC_CLOSE", "-0.06"))


def jawHoldCmd(raw) -> float:
    """[2026-10-03] the /execute_step jaw COMMAND (0..45) that keeps the jaw where /observe reads it (obs = cmd * -6).
    The generic GRIPPER_MODE=hold path re-sends the RAW reading (0..-270) as a command, which the bridge clips to 0 = CLOSED;
    the one-arm paths use this instead. (was jaw_hold_cmd)"""
    return float(np.clip(unwrapGrip(raw) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))


MM_PER_M = 1000.0
SAFE_DIV_EPS = 1e-9                                              # floor of a norm used as a divisor


def clampDelta(dT, maxMm, maxDeg):
    """Shrink one body-frame delta to the safety envelope, keeping its direction (was clamp_delta).

    Returns (clamped 4x4, whether it was clamped, |translation| mm, rotation angle deg); a limit <= 0 is off."""
    tOut = np.eye(4)
    tTrans = dT[:3, 3].copy()
    tNormMm = float(np.linalg.norm(tTrans)) * MM_PER_M
    tScaleT = 1.0 if (maxMm <= 0 or tNormMm <= maxMm) else maxMm / max(tNormMm, SAFE_DIV_EPS)
    tRotvec = Rot.from_matrix(dT[:3, :3]).as_rotvec()
    tAngDeg = float(np.degrees(np.linalg.norm(tRotvec)))
    tScaleR = 1.0 if (maxDeg <= 0 or tAngDeg <= maxDeg) else maxDeg / max(tAngDeg, SAFE_DIV_EPS)
    tOut[:3, 3] = tTrans * tScaleT
    tOut[:3, :3] = Rot.from_rotvec(tRotvec * tScaleR).as_matrix()
    return tOut, (tScaleT < 1.0 or tScaleR < 1.0), tNormMm, tAngDeg
