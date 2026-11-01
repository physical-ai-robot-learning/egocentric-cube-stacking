"""Read-only routes: GET / (the page), /status, /zlog, /frame/<cam>.
"""
import collections
import os

from flask import jsonify, request
import requests

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import IC as _IC
from .page import PAGE
from .robot_client import _robot


@app.route("/status")
def status():
    s = _robot("/status")
    hra = None
    if _IC.ARM_ONLY:
        h = rt.STATE.get("hra") or {}
        hra = dict(gate=cfg.HRA_GATE, mode="HRA RIGHT-ONLY", task=rt.INF.task if rt.INF else None, left="LOCKED", jaws="HOLD" if _IC.ARM_ONLY_GRIP == "hold" else "RIGHT MODEL",
                   max_run_s=cfg.HRA_MAX_RUN_S, still_mm=cfg.HRA_STILL_MM, still_n=cfg.HRA_STILL_N, gate_refusals=h.get("gate_refusals", 0),
                   max_left_dev_rad=round(h.get("max_left_dev", 0.0), 5), last_pred_mm=h.get("last_pred_mm"), still=h.get("still"),
                   travel_mm=round(h.get("travel_mm", 0.0), 1), max_cycle_mm=round(h.get("max_cycle_mm", 0.0), 1),
                   max_travel_cap=cfg.HRA_MAX_TRAVEL_MM, max_cycle_cap=cfg.HRA_MAX_CYCLE_MM,
                   ik=f"{_IC.IK_BACKEND} lock={','.join(_IC.PINK_LOCK) or 'none'} ori={_IC.PINK_ORI} profile={os.environ.get('V4_HRA_IK_PROFILE', '?')}")
    return jsonify({"hra": hra, "robot": s, "ui": {k: rt.STATE[k] for k in ("busy", "presses", "arm", "err", "running", "cycles",
                                                 "travel_mm", "stop_reason", "task")},
                    "last": rt.STATE["last"], "ab": rt.STATE.get("ab"), "observe_retries": dict(rt._OBS_FAILS),
                    "profile": rt.STATE.get("profile"), "denoise": rt.STATE.get("denoise"),
                    "replay": rt.STATE.get("replay"), "sent": rt.STATE.get("sent"),
                    "model_loaded": rt.INF is not None,
                    # which weights and which decode are live. Two action semantics now exist (delta and
                    # UMI current-anchor) and the same chunk commands different targets under each, so this
                    # is not cosmetic -- it is how you check the robot is driven by what you think it is.
                    "model": {"ckpt": os.path.basename(_IC.CKPT.rstrip("/")),
                              "action_mode": _IC.ACTION_MODE, "clamp": dict(_IC.LIMITS),
                              "gripper": _IC.GRIPPER_MODE,
                              "n_action": getattr(rt.INF, "n_action", _IC.N_ACTION),
                              "exec_k": _IC.EXEC_K, "dtype": _IC.DTYPE,
                              "temporal_ensemble": {"w": _IC.TE_W.tolist(), "rot": _IC.TE_ROT, "grip": _IC.TE_GRIP} if _IC.TE_ON else False,
                              "adaptive_k": None if _IC.ADAPTIVE_K is None else {"k": _IC.ADAPTIVE_K, "far_mm": _IC.AK_FAR_MM, "near_mm": _IC.AK_NEAR_MM},
                              "deadband": {"xy_mm": _IC.DB_XY_MM, "z_mm": _IC.DB_Z_MM, "ref_k": _IC.DB_REF_K, "axis_mm": None if _IC.DEADBAND_MM is None else _IC.DEADBAND_MM.tolist()},
                              "global_rot180": _IC.GLOBAL_ROT180,
                              "global_mirror": _IC.GLOBAL_MIRROR,
                              "state_mode": _IC.STATE_MODE,
                              "state_width": {"umi76": 76, "umi94": 94, "cart20": 20, "relcart20": 20}.get(_IC.STATE_MODE, 20),
                              "history_dt_ms": round(_IC.UMI_HISTORY_DT * 1000, 2) if _IC.STATE_MODE in ("umi76", "umi94", "cart20", "relcart20") else None,
                              "chunk": getattr(rt.INF, "chunk", None),
                              "ctl": {"k": getattr(rt.INF, "n_action", None), "stream": cfg.STREAM, "speed_pct": round(cfg.STREAM_SPEED * 100, 1),
                                      "stream_k": cfg.STREAM_K, "prefetch": cfg.PREFETCH, "prefetch_lead": cfg.PREFETCH_LEAD, "plan": cfg.PLAN_MODE, "plan_ema": cfg.PLAN_EMA, "plan_xfade": cfg.PLAN_XFADE, "plan_min_exec": cfg.PLAN_MIN_EXEC, "plan_smooth_deg": cfg.PLAN_SMOOTH_DEG, "plan_stats": {k_: (list(v_) if isinstance(v_, collections.deque) else v_) for k_, v_ in (rt.STATE.get("plan") or {}).items()}, "prefetch_auto": {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in rt._PFT.items()}, "pass_last": cfg.PASS_LAST, "pass": cfg.PASS_THROUGH, "dwell_s": cfg.DWELL_TIMEOUT_S, "rtc": _IC.RTC_ON, "rtc_last": (getattr(rt.INF, "last", {}) or {}).get("rtc"), "ik": _IC.IK_BACKEND, "remote": bool(_IC.REMOTE_INFER), "remote_ckpt": _IC.REMOTE_CKPT,
                                      "remote_last": dict(_IC.REMOTE_LAST), "goal_interp": (s or {}).get("goal_interp") if isinstance(s, dict) else None}}})


@app.route("/zlog")
def zlog():
    """The vertical decomposition, newest last. `n` rows, or `clear=1` to start a fresh window."""
    if request.args.get("clear"):
        rt.STATE["zlog"].clear()
        return jsonify({"cleared": True})
    n = int(request.args.get("n", "40"))
    rows = list(rt.STATE["zlog"])[-n:]
    out = {"rows": rows}
    for arm in ("L", "R"):
        v = [r for r in rows if r["arm"] == arm]
        if v:
            import statistics as st
            a1 = [r["a1_body_dz"] for r in v]
            tg = [r["tgt_dz"] for r in v]
            ac = [r["act_dz"] for r in v if r["act_dz"] is not None]
            out[arm] = {"n": len(v),
                        "a1_body_dz_mean": round(st.fmean(a1), 2),
                        "a1_body_dz_frac_up": round(sum(x > 0 for x in a1) / len(a1), 2),
                        "tgt_dz_mean": round(st.fmean(tg), 2),
                        "tgt_dz_frac_up": round(sum(x > 0 for x in tg) / len(tg), 2),
                        "act_dz_mean": round(st.fmean(ac), 2) if ac else None,
                        "tcp_z_first": v[0]["tcp_z"], "tcp_z_last": v[-1]["tcp_z"],
                        "tcp_z_drift": round(v[-1]["tcp_z"] - v[0]["tcp_z"], 1)}
    return jsonify(out)


@app.route("/frame/<cam>")
def frame(cam):
    """[2026-09-30 user: feeds flicker] robot_service /frame does a full get_observation, so one camera timing out (right wrist,
    ~9 % of reads after the 15:51 USB re-enumeration) makes EVERY feed return 500 and blink. For DISPLAY only, a failed read
    serves this cam's last good jpeg (header X-Frame-Stale: 1). Inference does not use this route."""
    try:
        r = requests.get(f"{cfg.ROBOT}/frame/{cam}", timeout=5)
        if r.status_code == 200 and r.content:
            rt._LAST_FRAME[cam] = r.content
            return (r.content, 200, {"Content-Type": r.headers.get("Content-Type", "image/jpeg")})
    except Exception:  # noqa: BLE001
        pass
    if cam in rt._LAST_FRAME:
        return (rt._LAST_FRAME[cam], 200, {"Content-Type": "image/jpeg", "X-Frame-Stale": "1"})
    return ("frame unavailable", 500)


@app.route("/")
def index():
    return PAGE
