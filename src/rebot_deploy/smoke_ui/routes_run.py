"""Run control routes: /run (STEP, STREAM or PLAN executor), /stop, /step, /observe_only, /profile, and the
cube-follow watcher that re-runs after a cube-size stop.
"""
import threading
import time

from flask import jsonify, request
import numpy as np
import requests

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import IC as _IC
from .plan_exec import _plan_loop
from .robot_client import _observe
from .safety import _arm_only, _grip_gate, _hra_latch, cube_px
from .step_exec import _loop, _one_press
from .stream_exec import _stream_loop


def _cube_follow_watch():
    """[2026-10-07] background: while a cube-size stop is waiting, re-/run when the cube looks smaller again (cube moved away)."""
    import cv2
    ok_n = 0
    while True:
        time.sleep(0.1)
        c = cfg.CUBE_STOP
        if not (c["waiting"] and c["follow"] and c["px"] > 0) or rt.STATE["running"]:
            ok_n = 0; continue
        try:
            r = requests.get(cfg.ROBOT + "/frame/right", timeout=1.0)
            px = cube_px(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR))
        except Exception:
            continue
        c["px_now"] = round(px, 1)
        ok_n = ok_n + 1 if (c["min_px"] <= px < c["resume_frac"] * c["px"]) else 0
        if ok_n >= c["resume_n"]:
            ok_n = 0; c["waiting"] = False; c["hits"] = 0
            with app.test_request_context("/run?reuse_anchor=1", method="POST"):
                rr = run().get_json()
            c["resumes"] += 1
            print(f"[cube-follow] cube {px:.0f} px < {c['resume_frac'] * c['px']:.0f} px -> resumed ({rr})", flush=True)


threading.Thread(target=_cube_follow_watch, daemon=True).start()


@app.route("/run", methods=["POST"])
def run():
    if rt.STATE["running"]:
        return jsonify({"error": "already running"})
    g = _grip_gate()
    if g:
        return jsonify(g)
    # [2026-09-29] RELCART20 anchor guard: a /run never silently re-anchors. First run after start/reset sets it from its first
    # observation; afterwards /run is refused until RESET ANCHOR is pressed (so rollouts from different start poses cannot mix).
    if _IC.STATE_MODE == "relcart20" and getattr(rt.INF, "_anchor", None) is not None and request.args.get("reuse_anchor") != "1":
        a = rt.INF._anchor
        return jsonify({"error": "ANCHOR_ALREADY_SET -- press RESET ANCHOR (after GO TO TRAIN START) for a new episode, or /run?reuse_anchor=1",
                        "anchor_mm": {"L": [round(float(x) * 1000, 1) for x in a[0][:3, 3]], "R": [round(float(x) * 1000, 1) for x in a[1][:3, 3]]}})
    if _arm_only():
        e = _hra_latch()
        if e:
            return jsonify({"error": f"HRA latch failed: {e}"})
    rt.STATE["running"] = True
    rt.INF._g_prev = [None, None]; rt.INF._g_cmd = [None, None]      # [2026-09-29] relative gripper: every /run starts from the measured jaw
    rt.INF._te_hist = None                                         # [2026-10-01] temporal ensemble: every /run starts fresh
    rt.INF._rtc_prev = None                                        # [2026-10-03] RTC: no inpainting prefix across /runs
    if rt._PF["thread"] is not None:                               # [2026-10-02] prefetch from a previous run is never used
        rt._PF["thread"].join(timeout=10.0)
    rt._PF["thread"] = None; rt._PF["res"] = None
    threading.Thread(target=(_plan_loop if cfg.PLAN_MODE else (_stream_loop if cfg.STREAM else _loop)), daemon=True).start()   # [10-06] PLAN mode
    return jsonify({"running": True, "mode": "stream" if cfg.STREAM else "step", "stream_speed": cfg.STREAM_SPEED if cfg.STREAM else None,
                    "max_cycles": cfg.MAX_CYCLES, "max_travel_mm": cfg.MAX_TRAVEL_MM})


@app.route("/stop", methods=["POST"])
def stop():
    rt.STATE["running"] = False; cfg.CUBE_STOP["waiting"] = False      # [2026-10-07] a user STOP also ends cube-follow
    rt.STATE["stop_reason"] = "stopped by user"
    return jsonify({"running": False})


def _profile_loop(n):
    """Profiler only -- it measures, it does not compensate. Latency is reported in action steps so the skip
    decision is made from p50/p95, not from a guess."""
    rt.STATE["profile"] = {"running": True, "cycles": [], "n": n}
    try:
        for _ in range(n):
            if not rt.STATE["profile"]["running"]:
                break
            tel = _one_press()
            if tel.get("error"):
                rt.STATE["profile"]["cycles"].append({"error": tel["error"]})
                continue
            rt.STATE["profile"]["cycles"].append({**tel["latency"], "skip": tel["skip_steps_would_be"]})
        rows = [c for c in rt.STATE["profile"]["cycles"] if "error" not in c]
        if rows:
            def pct(key, q):
                v = sorted(r[key] for r in rows if r.get(key) is not None)
                return round(v[min(int(q * len(v)), len(v) - 1)], 1) if v else None
            rt.STATE["profile"]["summary"] = {
                k: {"p50": pct(k, 0.5), "p95": pct(k, 0.95),
                    "n": sum(1 for r in rows if r.get(k) is not None)}
                for k in ("observe_ms", "infer_ms", "ik_ms", "arm_latency_ms", "gripper_latency_ms",
                          "total_ms", "skip")}
    finally:
        rt.STATE["profile"]["running"] = False
    print(f"[v4-smoke] profile done: {rt.STATE['profile'].get('summary')}", flush=True)


@app.route("/profile", methods=["POST"])
def profile():
    if rt.STATE["running"] or rt.STATE.get("ab", {}).get("running") or rt.STATE.get("profile", {}).get("running"):
        return jsonify({"error": "stop the other run first"})
    n = int(request.args.get("n", "20"))
    threading.Thread(target=_profile_loop, args=(n,), daemon=True).start()
    return jsonify({"started": True, "cycles": n})


@app.route("/observe_only", methods=["POST"])
def observe_only():
    """Dry run: infer and report, send nothing."""
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    try:
        rt.INF.infer([h0, h1])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    L = rt.INF.last
    return jsonify({"dry_run": True, "kept": int(L["kept"]), "valid": int(L["valid_step"]),
                    "pred_mm_step0": [round(float(L["pred_mm"][0, r]), 3) for r in (0, 1)],
                    "pred_deg_step0": [round(float(L["pred_deg"][0, r]), 3) for r in (0, 1)],
                    "cmd_mm_step0": [round(float(L["cmd_mm"][0, r]), 3) for r in (0, 1)],
                    "fkerr_mm": [round(float(L["fkerr_mm"][0, r]), 2) for r in (0, 1)],
                    "state20": [round(float(x), 4) for x in L["state"]]})


@app.route("/step", methods=["POST"])
def step():
    g = _grip_gate()
    if g:
        return jsonify(g)
    if not rt._lock.acquire(blocking=False):
        return jsonify({"error": "busy"})
    try:
        rt.STATE["busy"] = True
        return jsonify(_one_press())
    finally:
        rt.STATE["busy"] = False
        rt._lock.release()
