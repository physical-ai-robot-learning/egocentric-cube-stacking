"""Robot <-> dataset contract of the v4 inference core: jaw scales, the CART20 / RELCART20 state builders and the per-step
delta clamp. Moved out of infer_core_v4.py (2026-11-01) unchanged; infer_core_v4 re-exports every name, so `IC.raw_to_width`,
`IC.GRIP_BIN_OPEN`, ... still work. Only constants fixed at import live here: settings the UI switches at run time
(GRIPPER_MODE, so grip_to_cmd; RTC_ON; REMOTE_INFER; IK_BACKEND; LIMITS) stay in infer_core_v4.
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from umi.common.pose_util import mat_to_pose10d


CART20_W_OPEN = 270.0 * 0.05 / 118.0                            # raw -270 = fully open, in m (derive_cart20.py)


# [2026-10-07 user "B를 만들어줄래"] V4_FIXED_ANCHOR_JSON = json with "T_base_F" (4x4, robot base, dataset-TCP convention): the RIGHT-arm
# RELCART20 anchor is this fixed frame instead of the first observation, so the state is the absolute pose in F (the HRA_A100
# origin-anchored models: F = the HandUMI fingertip-on-X frame). Unset = unchanged behaviour.
FIXED_ANCHOR_R = (np.array(json.load(open(os.path.expanduser(os.environ["V4_FIXED_ANCHOR_JSON"])))["T_base_F"], np.float64)
                  if os.environ.get("V4_FIXED_ANCHOR_JSON") else None)

def relcart20_state(anchor, cur, w):
    """[2026-09-29] RELCART20 state (derive_relcart20.py layout): per arm T_rel = inv(T_anchor) @ T_t -> [pos3 | rot6d rows], then
    g_L, g_R = width / W_OPEN clipped to [0, 1]. anchor = TCP mats (2,4,4) at the rollout's first observation (training: the
    episode's frame_index-0 row). anchor == cur gives [0,0,0, 1,0,0,0,1,0] per arm."""
    st = np.zeros(20, np.float32)
    for r in (0, 1):
        st[r * 9:r * 9 + 9] = mat_to_pose10d((np.linalg.inv(anchor[r]) @ cur[r])[None])[0]
    st[18:20] = np.clip(np.asarray(w, np.float64) / CART20_W_OPEN, 0.0, 1.0)
    return st


def cart20_state(cur, w):
    """[2026-09-29] CART20 state (derive_cart20.py layout): [L pos3 | L rot6d rows | R pos3 | R rot6d rows | g_L | g_R],
    cur = dataset-frame TCP mats at t (2,4,4), w = follower jaw widths at t (m); g = w / W_OPEN clipped to [0, 1], 1 = OPEN."""
    st = np.zeros(20, np.float32)
    for r in (0, 1):
        st[r * 9:r * 9 + 9] = mat_to_pose10d(cur[r][None])[0]
    st[18:20] = np.clip(np.asarray(w, np.float64) / CART20_W_OPEN, 0.0, 1.0)
    return st


GRIP_M_PER_RAW = 0.05 / 118.0                # umi_schema.grip_linear_v1_anchor5cm_raw118, inverted
GRIP_RAW_MIN, GRIP_RAW_MAX = -270.0, 0.0


def unwrap_grip(raw):
    """The follower's gripper encoder wraps: a jaw that is physically CLOSED can read 358 instead of -2.

    Measured on this robot 2026-09-24: left closed = 358.09, right closed = 0.01, and the zero re-latches
    on every power cycle (0 / -100 / +90 / 358 have all been seen). The dataset pipeline has always
    unwrapped this -- `grip_norm_dataset.py` does `g - 360 if g > 180` -- and the deployment never did, so
    the state's width channel was nonsense on a wrapped arm and, worse, a command of 0 asked a jaw sitting
    at 358 to travel 358 counts, which is the "TRAIN START throws the gripper wide open" symptom.
    """
    r = float(raw)
    return r - 360.0 if r > 180.0 else r


def raw_to_width(raw):
    return float(np.clip(abs(float(np.clip(unwrap_grip(raw), GRIP_RAW_MIN, GRIP_RAW_MAX))), 0, None)
                 * GRIP_M_PER_RAW)


def width_to_raw(width_m):
    return float(np.clip(-abs(float(width_m)) / GRIP_M_PER_RAW, GRIP_RAW_MIN, GRIP_RAW_MAX))


# The model predicts a metric jaw WIDTH, /observe reports a raw count, and /execute_step takes a LEADER-frame
# command -- three different scales. Measured on this robot (GRIPPER_COMMAND_SCALE.md, re-checked per session
# because the raw zero re-latches at power-on): cmd * -6 = obs, full travel cmd 0..45, and raw 0 = jaws CLOSED
# with raw -270 = fully OPEN (video-confirmed 2026-09-22). Sending width_to_raw() straight out as the command,
# which the v3 core does, would ask for -270 on a 0..45 scale.
GRIP_OBS_PER_CMD = float(os.environ.get("V4_GRIP_OBS_PER_CMD", "-6.0"))
GRIP_CMD_MAX = float(os.environ.get("V4_GRIP_CMD_MAX", "45"))


def width_to_cmd(width_m):
    """predicted jaw width [m] -> /execute_step leader-frame gripper command."""
    return float(np.clip(width_to_raw(width_m) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))


# [2026-09-28] REL16-v2 binary gripper (V4_GRIPPER=binary). The v2 label is the LEADER's intent, 1 = OPEN and
# 0 = CLOSE (derive_v2.py), not a width: the v1 follower width stalls at cube contact, so commanding it gave no grip
# force. The actuator values live in gripper_contract_v2.json, written by `gripper_smoke_test.py --cube` only when
# the squeeze-on-cube gate passes; until then the 2026-09-19 smoke-test endpoints stand (CLOSED 0, OPEN 42).
def _grip_contract():
    import json as _json
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_contract_v2.json")
    try:
        c = _json.load(open(p))
        if c.get("passed"):
            return float(c["cmd_open"]), float(c["cmd_closed"]), p
    except (OSError, ValueError, KeyError):
        pass
    return 42.0, 0.0, None


GRIP_BIN_OPEN, GRIP_BIN_CLOSED, GRIP_CONTRACT_FILE = _grip_contract()
# [2026-09-29] execution threshold on the model's gripper channel (binary v2 label, or continuous v3 g = clip(leader/45, 0, 1)):
# g >= V4_GRIP_THRESH -> OPEN. Default 0.5 (the v2 behaviour); 0.6 = cmd 27/45, the old binary rule. Frozen per deployment.
GRIP_BIN_THRESH = float(os.environ.get("V4_GRIP_THRESH", "0.5"))
GRIP_CMD_FULL = float(os.environ.get("V4_GRIP_CMD_FULL", "45"))   # cmd 0..45 = full jaw travel (derive_v2 CMD_FULL_OPEN)
GRIP_REL_EPS = float(os.environ.get("V4_GRIP_REL_EPS", "0.05"))   # relative mode deadband on delta-g
GRIP_REL_SCALE = float(os.environ.get("V4_GRIP_REL_SCALE", "1.0"))  # divide g by this first (width-labelled runs: 0.11441 m -> 0..1)
# [2026-09-29] V4_GRIP_REL_MODE=acc: sum of the last N delta-g (per executed waypoint) instead of one delta; OPEN if > ACC_OPEN,
# CLOSE if < ACC_CLOSE, else HOLD; the history is cleared after every OPEN/CLOSE so one burst does not keep re-triggering.
GRIP_REL_MODE = os.environ.get("V4_GRIP_REL_MODE", "delta")
GRIP_ACC_N = int(os.environ.get("V4_GRIP_ACC_N", "4"))
GRIP_ACC_OPEN = float(os.environ.get("V4_GRIP_ACC_OPEN", "0.06"))
GRIP_ACC_CLOSE = float(os.environ.get("V4_GRIP_ACC_CLOSE", "-0.06"))


def jaw_hold_cmd(raw):
    """[2026-10-03] the /execute_step jaw COMMAND (0..45) that keeps the jaw where /observe reads it (obs = cmd * -6).
    The generic GRIPPER_MODE=hold path re-sends the RAW reading (0..-270) as a command, which the bridge clips to 0 = CLOSED;
    the one-arm paths use this instead."""
    return float(np.clip(unwrap_grip(raw) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))


def clamp_delta(dT, max_mm, max_deg):
    """Shrink one body-frame delta to the safety envelope, keeping its direction."""
    out = np.eye(4); t = dT[:3, 3].copy()
    n = float(np.linalg.norm(t)) * 1000.0
    scale_t = 1.0 if (max_mm <= 0 or n <= max_mm) else max_mm / max(n, 1e-9)
    rv = Rot.from_matrix(dT[:3, :3]).as_rotvec()
    ang = float(np.degrees(np.linalg.norm(rv)))
    scale_r = 1.0 if (max_deg <= 0 or ang <= max_deg) else max_deg / max(ang, 1e-9)
    out[:3, 3] = t * scale_t
    out[:3, :3] = Rot.from_rotvec(rv * scale_r).as_matrix()
    return out, (scale_t < 1.0 or scale_r < 1.0), n, ang
