"""Closed-loop trial log routes (/trial, /trials): one JSON line per real-robot trial in V4_TRIAL_LOG.
"""
import os
import time

from flask import jsonify, request

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import IC as _IC


def _trial_ckpt_info():
    import re as _re
    name = os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/"))
    m = _re.match(r"ckpt_(?:UI_)?(.+?)_(\d+)k(?:_.*)?$", name)
    run, step = (m.group(1), int(m.group(2)) * 1000) if m else (os.environ.get("RUN", "") or name, None)
    u = run.upper()
    if u.startswith("HRA"): model_type = "hra_rightonly"
    elif u.startswith("COTRAIN"): model_type = "cotrain"
    elif "ROBOT100" in u and u.startswith("FT-"): model_type = "robot100_init"
    elif u.startswith("FT-") and "EGO" in u: model_type = "ego_init"
    elif u.startswith("EGO"): model_type = "ego_only"
    else: model_type = "scratch"
    data = next((d for d in ("R30", "R90", "R384", "R312C", "R150") if _re.search(r"(^|-)" + d + r"(-|$)", u)), None)
    size = {"R30": 30, "R90": 90, "R384": 384, "R312C": 312, "R150": 150}.get(data)
    return dict(ckpt=name, run=run, step=step, model_type=model_type, robot_dataset=data, robot_episodes=size)


@app.route("/trial", methods=["POST"])
def trial():
    import json as _json
    f = request.get_json(silent=True) or {}
    try:
        level = int(f.get("stack_level"))
    except (TypeError, ValueError):
        return jsonify({"error": "stack_level 0-3 is required"})
    if not 0 <= level <= 3:
        return jsonify({"error": "stack_level must be 0, 1, 2 or 3"})
    task_label = rt.STATE.get("task"); instr = getattr(rt.INF, "task", None) if rt.INF else None
    rec = dict(t=round(time.time(), 3), time_local=time.strftime("%Y-%m-%d %H:%M:%S"), ui_port=cfg.UI_PORT, **_trial_ckpt_info(),
               task=task_label, instruction=instr,
               order_executed=(str(f.get("order_executed") or "").strip().upper() or None),
               layout_id=(str(f.get("layout_id") or "").strip() or None),
               stack_level=level, success=(level == 3 and not f.get("wrong_color")),
               wrong_color=bool(f.get("wrong_color")), failure_reason=(f.get("failure_reason") or None) if level < 3 or f.get("wrong_color") else None,
               note=(str(f.get("note") or "").strip() or None), operator=(str(f.get("operator") or "").strip() or None),
               last_run=dict(cycles=rt.STATE.get("cycles"), travel_mm=rt.STATE.get("travel_mm"), stop_reason=rt.STATE.get("stop_reason")))
    try:
        rec["ctl"] = dict(ik=_IC.IK_BACKEND, k=getattr(rt.INF, "n_action", None), stream=cfg.STREAM, rtc=_IC.RTC_ON, dtype=_IC.DTYPE)
    except Exception:  # noqa: BLE001
        pass
    with rt._TRIAL_LOCK:
        n = 0
        if os.path.exists(cfg.TRIAL_LOG):
            with open(cfg.TRIAL_LOG) as fh:
                n = sum(1 for _ in fh)
        rec["trial_id"] = n + 1
        with open(cfg.TRIAL_LOG, "a") as fh:
            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    return jsonify({"saved": True, "trial_id": rec["trial_id"], "log": cfg.TRIAL_LOG, "record": rec})


@app.route("/trials")
def trials():
    import json as _json
    rows = []
    if os.path.exists(cfg.TRIAL_LOG):
        with open(cfg.TRIAL_LOG) as fh:
            for line in fh:
                try: rows.append(_json.loads(line))
                except ValueError: pass
    by = {}
    for r in rows:
        k = r.get("ckpt") or "?"
        b = by.setdefault(k, dict(ckpt=k, model_type=r.get("model_type"), robot_episodes=r.get("robot_episodes"), step=r.get("step"),
                                  n=0, success=0, levels=[0, 0, 0, 0], wrong_color=0))
        b["n"] += 1; b["success"] += bool(r.get("success")); b["wrong_color"] += bool(r.get("wrong_color"))
        lv = r.get("stack_level")
        if isinstance(lv, int) and 0 <= lv <= 3: b["levels"][lv] += 1
    cur = _trial_ckpt_info()["ckpt"]
    return jsonify({"log": cfg.TRIAL_LOG, "n": len(rows), "current_ckpt": cur, "by_ckpt": sorted(by.values(), key=lambda b: (b["model_type"] or "", b["step"] or 0)),
                    "recent": rows[-5:]})
