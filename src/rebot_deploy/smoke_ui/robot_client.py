"""HTTP client side of the robot service (MIT bridge): motor positions, /observe with retries, the leader-frame
joint vector, the fast 14-joint read and jaw torque.
"""
import math
import os
import time

import numpy as np
import requests

from . import settings as cfg
from . import state as rt
from .ic import FLIP_IDX, GRIP_IDX, IC as _IC


def _motor_pos():
    """Light pose read for motion-onset detection: /observe ships three JPEGs, /motor_states does not."""
    try:
        d = rt._SESS.get(cfg.ROBOT + "/motor_states", timeout=5).json()
    except Exception:
        return None
    if not isinstance(d, dict) or "left_arm" in d and not isinstance(d.get("left_arm"), dict):
        return None
    try:
        arm, grip = [], []
        for side in ("left_arm", "right_arm"):
            for name, m in sorted(d[side].items()):
                (grip if name == "gripper" else arm).append(float(m["pos_deg"]))
        return np.asarray(arm), np.asarray(grip)
    except Exception:
        return None


def _wait_motion(base, arm_tol=0.03, grip_tol=0.3, timeout_s=0.6):
    # 0.15 deg was above the first command's realised motion (a 3 deg target moves 0.11 deg on the first
    # send), so every onset timed out. /motor_states polls at 0.4 ms, so the resolution is the threshold.
    """Time from now until the arms (and the jaws) actually start moving. Returns ms, or None on timeout."""
    t0 = time.perf_counter()
    arm_t = grip_t = None
    while time.perf_counter() - t0 < timeout_s:
        cur = _motor_pos()
        if cur is None:
            break
        if arm_t is None and float(np.abs(cur[0] - base[0]).max()) > arm_tol:
            arm_t = (time.perf_counter() - t0) * 1000.0
        if grip_t is None and float(np.abs(cur[1] - base[1]).max()) > grip_tol:
            grip_t = (time.perf_counter() - t0) * 1000.0
        if arm_t is not None and grip_t is not None:
            break
    return arm_t, grip_t


def _robot(path, method="get", **kw):
    try:
        return getattr(requests, method)(cfg.ROBOT + path, timeout=30, **kw).json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"robot_service: {e}"}


def _observe():
    """[2026-09-30 user: "추론 중간에 멈춰"] robot_service /observe returns 500 whenever ONE camera read times out (right wrist,
    ~9 % of reads since the 15:51 USB re-enumeration), and a single failure ended the run loop ("observe failed"). A failed read
    is now retried up to V4_OBS_RETRIES times (50 ms apart) before it counts as a failure; retries are counted in /status."""
    for i in range(cfg.OBS_RETRIES + 1):
        o = _robot("/observe")
        if "error" not in o:
            break
        if i < cfg.OBS_RETRIES:
            rt._OBS_FAILS["retried"] += 1; time.sleep(0.05)
    else:
        rt._OBS_FAILS["gave_up"] += 1
        return None
    # Stamp the observation HERE. umi76 interpolates its history at a physical 50.05 ms, so it needs the
    # time the robot was read, not the time inference happened to start -- the two observations of a cycle
    # are OBS_GAP_S apart when taken and microseconds apart when replayed into the buffer.
    return {"images": o["images"], "joints14": o["joints_rad"], "t": time.time(),
            "cam_age_ms": o.get("cam_age_ms"), "t_wall": o.get("t_wall")}   # [2026-09-30] robot_service frame capture ages


def _leader_now():
    o = _observe()
    if o is None:
        return None
    a = list(o["joints14"])
    for k in FLIP_IDX:
        a[k] = -a[k]
    # the jaw elements are raw encoder counts and the encoder wraps (a closed jaw can read 358, not -2).
    # Anything that ramps from or holds the current pose has to start from the unwrapped value, or it
    # commands a 358-count journey to reach what is already the current position.
    for g in GRIP_IDX:
        a[g] = _IC.unwrapGrip(a[g])
    return a


def _joints14_fast():
    """/motor_states (no images) -> the joints14 of /observe: arm joints in RAD (raw frame), jaws raw degrees."""
    try:
        d = rt._SESS.get(cfg.ROBOT + "/motor_states", timeout=2).json()
        out = []
        for side in ("left_arm", "right_arm"):
            for n in cfg._MOTOR_ORDER:
                v = float(d[side][n]["pos_deg"]); out.append(v if n == "gripper" else math.radians(v))
        return out
    except Exception:
        return None


def _jaw_torque(on):
    """Torque the two gripper motors on or off, without touching the arm motors.

    At rest the jaws are commanded to 0, which presses them into the closed stop, and FORCE_POS with a 10%
    torque limit cannot settle there -- it hunts. Measured: 7.6 deg of oscillation against a stop, 2.5 deg
    off it, and the motor runs a few degrees hotter. That hunt is the buzzing. Cutting torque to the jaw
    alone stops it; `/torque` cannot be used for this because it switches every motor of an arm, and
    dropping the arm motors would let the arm fall. `/gripper_zero` with stage=release disables exactly the
    one motor (it does not re-zero anything until stage=set).
    """
    out = []
    if os.environ.get("V4_JAW_TORQUE", "1") == "0":   # [2026-10-01] MIT bridge: /torque re-ran the arm enable and stalled the bus
        return [{"skipped": "V4_JAW_TORQUE=0"}]
    for arm in ("left", "right"):
        if on:
            out.append(_robot("/torque", "post", json={"arm": arm, "on": True}))
        else:
            out.append(_robot("/gripper_zero", "post", json={"arm": arm, "stage": "release"}))
    return out
