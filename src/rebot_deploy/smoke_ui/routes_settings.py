"""Live settings routes: /mode, /clamp, /ik, /steps, /task, /arm, /gripper_mode, /denoise, /wrist_zoom,
/cube_stop, /reset_anchor.
"""
from ego_paths import UMI_BRIDGE  # machine-specific roots (env-overridable)

import os

from flask import jsonify, request
import numpy as np
import requests

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import IC as _IC, set_limits
from .safety import cube_px


@app.route("/wrist_zoom", methods=["GET", "POST"])
def wrist_zoom_route():
    """[2026-10-07 user "feed 영상을 확대"] right-wrist focal zoom s (f' = s f about the C922 principal point), applied to the model input only."""
    if "s" in request.args:
        v = float(request.args["s"])
        if not (0.7 <= v <= 3.0):
            return jsonify({"error": "s must be 0.7 .. 3.0"})
        _IC.WRIST_ZOOM["s"] = v
        try:
            open(str(UMI_BRIDGE / f".auto_ui_d20_e5/{os.environ.get('V4_UI_PORT', '8056')}.wristzoom"), "w").write(f"{v}\n")
        except Exception:
            pass
    return jsonify({"wrist_zoom": _IC.WRIST_ZOOM["s"]})


@app.route("/cube_stop", methods=["GET", "POST"])
def cube_stop_route():
    """[2026-10-07] cube-size stop: ?px=<sqrt(area) px in the raw 640x480 right-wrist frame> (0 = off); GET returns the live size."""
    if "px" in request.args:
        cfg.CUBE_STOP["px"] = max(0.0, float(request.args["px"])); cfg.CUBE_STOP["hits"] = 0
        try:
            open(str(UMI_BRIDGE / f".auto_ui_d20_e5/{os.environ.get('V4_UI_PORT', '8056')}.cubestop"), "w").write(f"{cfg.CUBE_STOP['px']}\n")
        except Exception:
            pass
    try:
        import cv2
        r = requests.get(cfg.ROBOT + "/frame/right", timeout=2.0)
        now = round(cube_px(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)), 1)
    except Exception as e:
        now = None
    if "follow" in request.args:
        cfg.CUBE_STOP["follow"] = request.args["follow"] == "1"
        if not cfg.CUBE_STOP["follow"]:
            cfg.CUBE_STOP["waiting"] = False
    return jsonify({"stop_px": cfg.CUBE_STOP["px"], "cube_px_now": now, "follow": cfg.CUBE_STOP["follow"], "waiting": cfg.CUBE_STOP["waiting"],
                    "resume_below_px": round(cfg.CUBE_STOP["resume_frac"] * cfg.CUBE_STOP["px"], 1), "resumes": cfg.CUBE_STOP["resumes"]})


@app.route("/reset_anchor", methods=["POST"])
def reset_anchor():
    rt.INF.reset_anchor(); rt.INF._rtc_prev = None
    return jsonify({"anchor": "reset; the next observation becomes the RELCART20 anchor"})


@app.route("/gripper_mode/<mode>", methods=["POST"])
def gripper_mode(mode):
    if mode in ("hold", "predict", "binary"):
        _IC.GRIPPER_MODE = mode
    return jsonify({"gripper": _IC.GRIPPER_MODE})


@app.route("/denoise/<int:n>", methods=["POST"])
def set_denoise(n):
    rt.INF.policy.config.num_denoising_steps = max(1, min(n, 50))
    return jsonify({"num_denoising_steps": rt.INF.policy.config.num_denoising_steps})


@app.route("/clamp/<mm>/<deg>", methods=["POST"])
def clamp(mm, deg):
    return jsonify(set_limits(mm, deg))


# [2026-10-03 user] "k 수도 내가 선택 + streaming / 보간 켜기 + streaming % 직접": live mode control (refused while running).
# k -> step mode n_action AND streaming STREAM_K (rows of each chunk that enter the timeline); stream on/off; speed % -> STREAM_SPEED;
# interp -> MIT bridge goal interpolation (/set_speed goal_interp -- ONE bridge, so this is shared by every UI on :8021).
@app.route("/mode", methods=["POST"])
def mode():
    if rt.STATE["running"] or rt.STATE["busy"]:
        return jsonify({"ok": False, "err": "running -- STOP first"})
    a = request.args; out = {"ok": True}
    if "k" in a:
        k = max(1, min(int(a["k"]), 16)); rt.INF.n_action = k; cfg.STREAM_K = k
    if "stream" in a:
        cfg.STREAM = a["stream"] == "1"
    if "speed" in a:
        cfg.STREAM_SPEED = max(0.05, min(float(a["speed"]) / 100.0, 1.0))
    if "pass" in a:                                     # [2026-10-03 user] step mode: intermediate waypoints pass at V4_PASS_TCP_MM
        cfg.PASS_THROUGH = a["pass"] == "1"
    if "dwell" in a:                                    # reach-test timeout per (last) waypoint, seconds
        cfg.DWELL_TIMEOUT_S = max(0.05, min(float(a["dwell"]), 5.0))
    if "rtc" in a:                                      # [2026-10-03 user] Real-Time Chunking (infer_core_v4.RTC_ON), any mode
        _IC.RTC_ON = a["rtc"] == "1"; rt.INF._rtc_prev = None
    if "prefetch" in a:                                 # [2026-10-03 user] step mode: next observe+infer during the last waypoint
        cfg.PREFETCH = a["prefetch"] == "1"; rt._PF["thread"] = None; rt._PF["res"] = None
    if "plan" in a:                                     # [2026-10-06] PLAN mode (always-one-forward + 100 Hz playback)
        cfg.PLAN_MODE = a["plan"] == "1"
    if "planema" in a:
        cfg.PLAN_EMA = max(0.0, min(float(a["planema"]), 1.0))
    if "planxf" in a:
        cfg.PLAN_XFADE = max(0, min(int(a["planxf"]), 16))
    if "plansm" in a:
        cfg.PLAN_SMOOTH_DEG = max(0, min(int(a["plansm"]), 5))
    if "planmin" in a:
        cfg.PLAN_MIN_EXEC = max(1, min(int(a["planmin"]), 16))
    if "passlast" in a:                                 # [2026-10-04 user] last waypoint passes through while a prefetch runs
        cfg.PASS_LAST = a["passlast"] == "1"
    if "pflead" in a:                                   # [2026-10-04 user] prefetch starts this many waypoints before the last
        cfg.PREFETCH_LEAD = max(-1, min(int(a["pflead"]), 15))
    if "remote" in a:                                   # [2026-10-03] policy forward on the 5090 server (V4_REMOTE_CKPT from the loader)
        if a["remote"] == "1":
            if not _IC.REMOTE_CKPT:
                out.update(ok=False, err="no V4_REMOTE_CKPT (UI not started with V4_REMOTE_INFER)")
            else:
                _IC.REMOTE_INFER = cfg._REMOTE_URL0 or os.environ.get("V4_REMOTE_INFER") or "http://100.64.0.5:8791"   # [10-04] the URL the UI was started with (a remote=0 pops the env)
        else:
            _IC.REMOTE_INFER = ""; os.environ.pop("V4_REMOTE_INFER", None)   # a later checkpoint pick (ckpt_select) stays local too
    if "interp" in a:
        try:
            r = rt._SESS.post(cfg.ROBOT + "/set_speed", json={"goal_interp": a["interp"] == "1"}, timeout=5).json()
            out["bridge_goal_interp"] = r.get("goal_interp")
        except Exception as e:
            out.update(ok=False, err=f"bridge /set_speed failed: {e}")
    out.update(k=rt.INF.n_action, stream=cfg.STREAM, speed_pct=round(cfg.STREAM_SPEED * 100, 1), stream_k=cfg.STREAM_K)
    print(f"[v4-smoke] mode -> {out}", flush=True)
    return jsonify(out)


# [2026-10-03 user] "pink numerical learned curobo 중 하나 선택": live IK backend switch (infer_core_v4.IK_BACKEND, read per solve).
@app.route("/ik/<name>", methods=["POST"])
def ik_backend(name):
    if name not in ("pink", "numerical", "learned", "curobo"):
        return jsonify({"ok": False, "err": f"unknown IK backend {name}"})
    if rt.STATE["running"] or rt.STATE["busy"]:
        return jsonify({"ok": False, "err": "running -- STOP first"})
    if name == "curobo":
        try:
            h = rt._SESS.get(_IC.CUROBO_URL + "/health", timeout=5).json()
        except Exception as e:
            return jsonify({"ok": False, "err": f"cuRobo server {_IC.CUROBO_URL} not reachable: {e}"})
    _IC.IK_BACKEND = name; os.environ["IK_BACKEND"] = name
    print(f"[v4-smoke] IK backend -> {name}", flush=True)
    return jsonify({"ok": True, "ik_backend": name})


@app.route("/steps/<n>", methods=["POST"])
def steps(n):
    rt.INF.n_action = max(1, min(int(n), 16))
    return jsonify({"n_action": rt.INF.n_action})


@app.route("/task/<int:i>", methods=["POST"])
def set_task(i):
    if _IC.ARM_ONLY:                                             # [2026-10-03] one-arm model: one training instruction only
        return jsonify({"task_label": "ARM_ONLY", "task": rt.INF.task, "error": "V4_ARM_ONLY model: task fixed to V4_TASK (no stacking prompts)"})
    if 0 <= i < len(cfg.TASKS):
        rt.INF.task = cfg.TASKS[i][1]
        rt.STATE["task"] = cfg.TASKS[i][0]
    return jsonify({"task_label": rt.STATE.get("task"), "task": rt.INF.task})


@app.route("/arm/<which>", methods=["POST"])
def set_arm(which):
    if which in ("left", "right", "both"):
        rt.STATE["arm"] = which
    return jsonify({"arm": rt.STATE["arm"]})
