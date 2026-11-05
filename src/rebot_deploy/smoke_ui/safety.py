"""Run gates and stops: arm-only masks, the HRA start latch / gate / stop budgets, the red-cube size stop and the
gripper-command gate.
"""
import time

import numpy as np
import requests

from . import settings as cfg
from . import state as rt
from .ic import GRIP_IDX, IC as _IC
from .robot_client import _leader_now, _observe


def _arm_only():
    return _IC.ARM_ONLY


def _hra_latch():
    """latch the left arm + jaw commands at /run (leader frame, jaws as commands)"""
    a = _leader_now()
    if a is None:
        return "observe failed"
    o = _observe()
    m0, _ = rt.INF.tcp_now(o["joints14"])
    cfg.CUBE_STOP["hits"] = 0
    rt.STATE["hra"] = dict(left=[float(a[i]) for i in range(6)], jaw=[_IC.jawHoldCmd(o["joints14"][g]) for g in GRIP_IDX],
                        t0=time.time(), still=0, last_call=getattr(rt.INF, "calls", 0), gate_refusals=0, max_left_dev=0.0,
                        p0=m0[1][:3, 3].copy(), p_prev=m0[1][:3, 3].copy(), travel_mm=0.0, max_cycle_mm=0.0)
    return None


def _hra_gate(a):
    """None = send; else the refusal text. Only during a rollout of a one-arm model."""
    h = rt.STATE.get("hra")
    if not (cfg.HRA_GATE and _arm_only() and rt.STATE.get("running") and h):
        return None
    dev = max(abs(float(a[i]) - h["left"][i]) for i in range(6)); h["max_left_dev"] = max(h["max_left_dev"], dev)
    jd = max(abs(float(a[g]) - h["jaw"][j]) for j, g in enumerate(GRIP_IDX))
    if dev > cfg.HRA_LEFT_TOL_RAD or jd > cfg.HRA_JAW_TOL:
        h["gate_refusals"] += 1
        return f"HRA gate: left arm command off by {dev:.4f} rad (tol {cfg.HRA_LEFT_TOL_RAD}) / jaw off by {jd:.2f} (tol {cfg.HRA_JAW_TOL})"
    return None


def cube_px(bgr):
    import cv2
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (((h[..., 0] < 10) | (h[..., 0] > 170)) & (h[..., 1] > 110) & (h[..., 2] > 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return 0.0 if n < 2 else float(np.sqrt(st[1:, 4].max()))


def _cube_check():
    c = cfg.CUBE_STOP
    if time.time() - c["last_t"] < c["every_s"]:
        return None
    c["last_t"] = time.time()
    try:
        import cv2
        r = requests.get(cfg.ROBOT + "/frame/right", timeout=1.0)
        px = cube_px(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR))
    except Exception:
        return None
    c["px_now"] = round(px, 1)
    if c["px"] <= 0:
        return None
    c["hits"] = c["hits"] + 1 if px >= c["px"] else 0
    if c["hits"] >= c["n"]:
        c["waiting"] = bool(c["follow"])
        return f"HRA: cube size {px:.0f} px >= stop size {c['px']:.0f} px (right wrist C922)" + (" -- following (resumes when smaller)" if c["follow"] else "")
    return None


def _hra_stop():
    """approach-only stop conditions; None = keep going"""
    h = rt.STATE.get("hra")
    if not (_arm_only() and h):
        return None
    cs = _cube_check()
    if cs:
        return cs
    if cfg.HRA_MAX_RUN_S > 0 and time.time() - h["t0"] > cfg.HRA_MAX_RUN_S:      # 0 = no time limit (user 10-04)
        return f"HRA: max run time {cfg.HRA_MAX_RUN_S:.0f} s"
    c = getattr(rt.INF, "calls", 0)
    if c != h["last_call"] and rt.INF.last.get("tgt_pos") is not None:
        h["last_call"] = c
        p = rt.INF.last["cur_mat"][1][:3, 3]                      # MEASURED right TCP at this inference
        dc = float(np.linalg.norm(p - h["p_prev"])) * 1000.0; h["p_prev"] = p.copy(); h["max_cycle_mm"] = max(h["max_cycle_mm"], dc)
        h["travel_mm"] = float(np.linalg.norm(p - h["p0"])) * 1000.0
        if cfg.HRA_MAX_CYCLE_MM > 0 and dc > cfg.HRA_MAX_CYCLE_MM:
            return f"HRA: right TCP moved {dc:.0f} mm between two inferences (> {cfg.HRA_MAX_CYCLE_MM:.0f} mm)"
        if cfg.HRA_MAX_TRAVEL_MM > 0 and h["travel_mm"] > cfg.HRA_MAX_TRAVEL_MM:
            return f"HRA: right TCP net travel {h['travel_mm']:.0f} mm since /run (> {cfg.HRA_MAX_TRAVEL_MM:.0f} mm)"
        d = float(np.linalg.norm(rt.INF.last["tgt_pos"][-1][1] - rt.INF.last["cur_mat"][1][:3, 3])) * 1000.0
        h["last_pred_mm"] = round(d, 1); h["still"] = h["still"] + 1 if d < cfg.HRA_STILL_MM else 0
        if cfg.HRA_STILL_N > 0 and h["still"] >= cfg.HRA_STILL_N:
            return f"HRA: approach converged (predicted right TCP move < {cfg.HRA_STILL_MM:.0f} mm for {cfg.HRA_STILL_N} chunks)"
    return None


def _grip_gate():
    """None if both jaws read a valid raw, else an error dict (shown on the page)."""
    ob = _observe()
    if not ob or "joints14" not in ob:
        return {"error": "gripper zero gate: observe failed"}
    raw = [float(ob["joints14"][g]) for g in GRIP_IDX]
    bad = [a for a, r in zip(("left", "right"), raw) if not (cfg.GRIP_GATE[0] <= r <= cfg.GRIP_GATE[1])]
    if bad:
        return {"error": f"gripper zero slipped on {'/'.join(bad)}: raw L/R {raw[0]:.1f}/{raw[1]:.1f} outside "
                         f"[{cfg.GRIP_GATE[0]:.0f}, {cfg.GRIP_GATE[1]:.0f}] -- re-zero (1 release -> close by hand -> 2 set zero)",
                "grip_raw_LR": [round(r, 2) for r in raw]}
    return None
