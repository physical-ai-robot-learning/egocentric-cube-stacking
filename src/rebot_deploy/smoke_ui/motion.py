"""Slew-limited joint motion through /execute_step: one sub-step, the dwell-until-reached goto, the arm mask and the
move to the training start pose.
"""
import math
import time

import numpy as np

from . import settings as cfg
from . import state as rt
from .ic import ARM_IDX, FLIP_IDX, GRIP_IDX
from .robot_client import _leader_now, _observe, _robot
from .safety import _hra_gate


def _exec_step(a, substep=1, of=1):
    """Post one /execute_step and remember exactly what went out.

    Every path that moves the robot goes through here -- the rollout loop, the jaw probes, the pose moves --
    so the panel on the page is the actual wire and not a reconstruction of it. Units are execute_step's:
    arm joints in RADIANS in the LEADER frame (sign-flipped on FLIP_IDX against /observe), jaw as 0..45.
    """
    g = _hra_gate(a)
    if g:                                                   # refused: nothing goes on the wire, the rollout stops
        rt.STATE["stop_reason"] = g; rt.STATE["running"] = False
        print(f"[hra] {g}", flush=True)
        return {"error": g}
    r = _robot("/execute_step", "post", json={"action": [float(x) for x in a]})
    rt.STATE["sent"] = {"substep": substep, "of": of, "leader_rad": [round(float(x), 5) for x in a],
                     "arm_deg": [round(math.degrees(a[i]), 2) for i in ARM_IDX],
                     "grip_cmd": [round(float(a[g]), 2) for g in GRIP_IDX],
                     "err": (r or {}).get("error") if isinstance(r, dict) else None,
                     "t": round(time.time() % 10000, 1)}
    return r


def _goto(target14, tcp_target=None, ik_fk_err=0.0, motion_base=None, pass_through=False):
    """Drive to ONE waypoint and hold it until reached or the dwell times out.

    The Damiao follower does not take a setpoint and go: a single send of a +3 deg target realises +0.11 deg,
    and 40 identical sends over ~2 s reach +1.27 deg. So each waypoint is a short closed loop -- keep sending
    the same target at CMD_HZ, watch the error, and stop on reach or timeout. Firing once per waypoint (the
    old behaviour) meant the target moved on before the arm had gone anywhere, which is why a small commanded
    motion looked like no motion at all.
    """
    cur = _leader_now()
    if cur is None:
        return {"error": "observe failed"}
    n = max(1, int(math.ceil(max(abs(target14[i] - cur[i]) for i in ARM_IDX) / cfg.SLEW_MAX_STEP_RAD)))
    arm_lat = grip_lat = None
    for k in range(1, n + 1):
        if rt.STATE["stop_reason"] == "estop":
            return {"stopped": True, "at": k}
        # Ramp the ARM joints only. `cur` comes from /observe, whose jaw element is a RAW count (0..-270),
        # while the target's jaw is a COMMAND (0..45) -- interpolating between the two units sends values
        # like -84.9 on the intermediate sub-steps and only lands on the intended command at the last one.
        # The jaw has its own controller and does not need slewing, so it gets the target directly.
        a = [cur[i] + (target14[i] - cur[i]) * k / n for i in range(14)]
        for _g in GRIP_IDX:
            a[_g] = target14[_g]
        r = _exec_step(a, k, n)
        if isinstance(r, dict) and r.get("error"):
            return r
        time.sleep(1.0 / cfg.CMD_HZ)

    # The joint target came out of IK, so its own FK residual is a floor the robot cannot beat: if
    # ||FK(q_target) - desired_TCP|| is already 7 mm, no amount of dwell brings the measured TCP error under a
    # 6 mm tolerance. Raise the bar by the residual so a timeout means "the follower did not get there",
    # not "IK never offered a pose that close".
    tcp_tol = max(cfg.TCP_TOL_MM, ik_fk_err + 2.0)
    dwell_s = cfg.DWELL_TIMEOUT_S if tcp_target is not None else cfg.GOTO_DWELL_S
    if pass_through and tcp_target is not None:
        dwell_s = cfg.PASS_DWELL_S
    # [2026-09-30] no TCP target (pose goto): measure against FK of the TARGET joint pose instead of logging 0 -- the old
    # tcp_err_mm=0 hid a real 16 mm left-arm gap at TRAIN START.
    if tcp_target is None:
        _tf = list(target14)
        for f in FLIP_IDX:
            _tf[f] = -_tf[f]
        _mt, _ = rt.INF.tcp_now(_tf)
        tgt_pos = [np.asarray(_mt[r_][:3, 3], float) for r_ in (0, 1)]
    else:
        tgt_pos = [np.asarray(tcp_target[0][r_], float) for r_ in (0, 1)]
    t0 = time.time(); reps = 0; reached = False
    joint_err = tcp_err = float("nan"); detail = None; settle = 0; still_mm = None; prev_tcp = None
    while time.time() - t0 < dwell_s:
        if rt.STATE["stop_reason"] == "estop":
            break
        r = _exec_step(target14)        # the dwell repeats the same target; show it too
        if isinstance(r, dict) and r.get("error"):
            return r
        reps += 1
        time.sleep(1.0 / cfg.CMD_HZ)
        o = _observe()
        if o is None:
            break
        now = list(o["joints14"])
        for f in FLIP_IDX:
            now[f] = -now[f]
        joint_err = math.degrees(max(abs(target14[i] - now[i]) for i in ARM_IDX))
        m, _ = rt.INF.tcp_now(o["joints14"])
        exyz = [(np.asarray(m[r_][:3, 3], float) - tgt_pos[r_]) * 1000 for r_ in (0, 1)]
        tcp_err = max(float(np.linalg.norm(e)) for e in exyz)
        jerr = [math.degrees(now[i] - target14[i]) for i in ARM_IDX]
        detail = {arm: {"target_tcp_mm": [round(float(x) * 1000, 1) for x in tgt_pos[r_]],
                        "measured_tcp_mm": [round(float(x) * 1000, 1) for x in m[r_][:3, 3]],
                        "tcp_err_xyz_mm": [round(float(x), 1) for x in exyz[r_]],
                        "tcp_err_norm_mm": round(float(np.linalg.norm(exyz[r_])), 1),
                        "target_q_deg": [round(math.degrees(target14[i]), 2) for i in ARM_IDX[r_ * 6:(r_ + 1) * 6]],
                        "measured_q_deg": [round(math.degrees(now[i]), 2) for i in ARM_IDX[r_ * 6:(r_ + 1) * 6]],
                        "joint_err_deg": dict(zip(cfg.JOINT_NAMES, [round(x, 2) for x in jerr[r_ * 6:(r_ + 1) * 6]])),
                        "joint_err_max_deg": round(max(abs(x) for x in jerr[r_ * 6:(r_ + 1) * 6]), 2),
                        "worst_joint": cfg.JOINT_NAMES[int(np.argmax([abs(x) for x in jerr[r_ * 6:(r_ + 1) * 6]]))]}
                  for arm, r_ in (("L", 0), ("R", 1))}
        if tcp_target is not None:
            if pass_through and tcp_err < max(cfg.PASS_TCP_MM, tcp_tol):
                reached = True
                break
            if joint_err < cfg.JOINT_TOL_DEG and tcp_err < tcp_tol:
                reached = True
                break
        else:                                            # pose goto: 2 consecutive passes (stationary = log/warning only)
            # [2026-09-30 user] the hard stationary gate + reset made false timeouts (final L/R 2.9/3.7 mm, joint 0.24/0.98 deg,
            # reached=false, still_mm=null). Accept on TCP < tol AND max joint < GOTO_JOINT_TOL_DEG for 2 samples in a row.
            ok_now = joint_err < cfg.GOTO_JOINT_TOL_DEG and tcp_err < tcp_tol
            settle = settle + 1 if ok_now else 0
            if prev_tcp is not None:
                still_mm = max(float(np.linalg.norm(np.asarray(m[r_][:3, 3], float) - prev_tcp[r_])) * 1000 for r_ in (0, 1))
            prev_tcp = [np.asarray(m[r_][:3, 3], float).copy() for r_ in (0, 1)]
            if settle >= 2:
                reached = True
                break
    cause = None; still_warn = None
    if not reached:
        # [2026-09-30] cause from the SAME last sample as tcp_err/joint_err/per_arm, with the thresholds this move used
        jt = cfg.JOINT_TOL_DEG if tcp_target is not None else cfg.GOTO_JOINT_TOL_DEG
        if ik_fk_err > cfg.TCP_TOL_MM:
            cause = "IK_RESIDUAL"
        elif joint_err >= jt:
            cause = "FOLLOWER_LAG"
        elif tcp_err >= tcp_tol:
            cause = "CARTESIAN_GAP"
        else:
            cause = "IN_TOL_NOT_CONSECUTIVE"
    if tcp_target is None and still_mm is not None and still_mm >= 1.0:
        still_warn = f"still moving {still_mm:.1f} mm between the last two samples"
    return {"ok": True, "substeps": n, "repeats": reps, "dwell_ms": int((time.time() - t0) * 1000),
            "joint_err_deg": round(joint_err, 2), "tcp_err_mm": round(tcp_err, 2),
            "ik_fk_err_mm": round(float(ik_fk_err), 2), "tcp_tol_mm": round(tcp_tol, 2),
            "reached": reached, "timeout": not reached, "cause": cause, "dwell_limit_s": dwell_s, "per_arm": detail, "still_mm": None if still_mm is None else round(still_mm, 2), "still_warn": still_warn,
            "arm_latency_ms": None if arm_lat is None else round(arm_lat, 1),
            "gripper_latency_ms": None if grip_lat is None else round(grip_lat, 1)}


def _mask_arm(cmd, arm, q_now):
    """Freeze one arm at its current pose so the first tests can move a single arm."""
    if arm == "both":
        return cmd
    hold = list(range(6, 14)) if arm == "left" else list(range(0, 7))     # indices to leave untouched
    lead_now = list(q_now)
    for k in FLIP_IDX:
        lead_now[k] = -lead_now[k]
    out = cmd.copy()
    for i in hold:
        out[:, i] = lead_now[i]
    return out


def _train_start_move():
    tgt = np.asarray(cfg.TRAIN_START_DEG, float).copy()
    for i in ARM_IDX:
        tgt[i] = math.radians(cfg.TRAIN_START_DEG[i])
    tgt[GRIP_IDX[0]] = tgt[GRIP_IDX[1]] = 0.0
    lead = tgt.copy()
    for k in FLIP_IDX:
        lead[k] = -lead[k]
    return _goto([float(x) for x in lead])
