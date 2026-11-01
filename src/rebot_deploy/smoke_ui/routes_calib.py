"""Gripper and arm calibration routes: /gripper_zero, /gripper_probe, /arm_ident, /grip, /grip_calib.
"""
import time

from flask import jsonify, request
import numpy as np
import requests

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import GRIP_IDX
from .motion import _exec_step
from .robot_client import _leader_now, _observe, _robot


@app.route("/gripper_zero/<arm>/<stage>", methods=["POST"])
def gripper_zero(arm, stage):
    """[2026-09-30 user] re-zero one jaw from the page (the left zero re-latches after every left-arm power drop).
    stage=release drops that jaw's torque -> close the jaw fully BY HAND -> stage=set stores zero there (robot_service
    /gripper_zero). Refused while a loop runs. After set, the page shows both raw jaw readings (closed should be ~0)."""
    if arm not in ("left", "right") or stage not in ("release", "set"):
        return jsonify({"error": f"bad arm/stage {arm}/{stage}"})
    if rt.STATE["running"] or rt.STATE.get("busy"):
        return jsonify({"error": "a loop is running -- STOP first"})
    r = _robot("/gripper_zero", "post", json={"arm": arm, "stage": stage})
    if stage == "set":
        ob = _observe()
        if ob and "joints14" in ob:
            r = {**(r if isinstance(r, dict) else {"result": r}), "raw_now_LR": [round(float(ob["joints14"][g]), 2) for g in GRIP_IDX]}
    return jsonify(r)


@app.route("/gripper_probe/<cmd>", methods=["POST"])
def gripper_probe(cmd):
    """Measure today's gripper command->observation scale. Arms do not move: the whole leader vector is the
    current pose and only the jaw element changes. The raw zero re-latches per power cycle, so this is a
    measurement of THIS session, not a constant."""
    lead = _leader_now()
    if lead is None:
        return jsonify({"error": "observe failed"})
    before = _observe()["joints14"]
    out = list(lead)
    for gi in GRIP_IDX:
        out[gi] = float(cmd)
    r = _exec_step(out)
    if isinstance(r, dict) and r.get("error"):
        return jsonify(r)
    time.sleep(1.5)                                   # hold to a settled readback; mid-travel reads lie
    after = _observe()["joints14"]
    obs_b = [round(float(before[g]), 2) for g in GRIP_IDX]
    obs_a = [round(float(after[g]), 2) for g in GRIP_IDX]
    ratio = [round(obs_a[i] / float(cmd), 3) if float(cmd) else None for i in range(2)]
    return jsonify({"cmd": float(cmd), "obs_before": obs_b, "obs_after": obs_a, "obs_per_cmd": ratio})


@app.route("/arm_ident", methods=["POST"])
def arm_ident():
    """Which wrist feed belongs to which arm, decided by moving one jaw and watching.

    Two different things can look like "left and right are swapped": the camera indices can come up in a
    different order after a service restart, or the arm-to-USB-location mapping can be wrong. Guessing
    between them from the pictures is how the wrong one gets "fixed". So: close ONE gripper, see which feed
    changes and which jaw element of the state vector moves, and put the jaw back.

    No arm joint moves -- the leader vector sent is the current pose with a single jaw element replaced,
    which is the same primitive /gripper_probe uses.
    """
    import cv2 as _cv2
    which = (request.get_json(silent=True) or {}).get("arm", "left")
    gi = GRIP_IDX[0 if which == "left" else 1]
    lead = _leader_now()
    if lead is None:
        return jsonify({"error": "observe failed"})

    def _grab():
        out = {}
        for cam in ("left", "right", "middle"):
            try:
                r = requests.get(f"{cfg.ROBOT}/frame/{cam}", timeout=10)
                a = np.frombuffer(r.content, np.uint8)
                out[cam] = _cv2.imdecode(a, _cv2.IMREAD_GRAYSCALE)
            except Exception:
                out[cam] = None
        return out

    before_img = _grab()
    before_q = _observe()["joints14"]
    # the jaw element of an execute_step vector is a COMMAND in 0..45, not an observation in 0..-270
    # (obs = cmd * -6). Sending -120 here is outside the command range, gets clipped, and the jaw never
    # moves -- which reads as "the probe found nothing" rather than "the probe was wrong".
    target = 20.0 if abs(float(before_q[gi])) < 60 else 0.0
    out = list(lead)
    out[gi] = target
    r = _exec_step(out)
    if isinstance(r, dict) and r.get("error"):
        return jsonify(r)
    time.sleep(2.0)
    after_img = _grab()
    after_q = _observe()["joints14"]

    out[gi] = float(lead[gi])                                      # put the jaw back where it was
    _exec_step(out)

    diff = {}
    for cam in ("left", "right", "middle"):
        a, b = before_img.get(cam), after_img.get(cam)
        diff[cam] = None if a is None or b is None or a.shape != b.shape else \
            round(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()), 3)
    jaw = {"left": round(float(after_q[GRIP_IDX[0]]) - float(before_q[GRIP_IDX[0]]), 2),
           "right": round(float(after_q[GRIP_IDX[1]]) - float(before_q[GRIP_IDX[1]]), 2)}
    moved_cam = max((c for c in ("left", "right") if diff[c] is not None), key=lambda c: diff[c], default=None)
    moved_jaw = max(jaw, key=lambda k: abs(jaw[k]))
    return jsonify({"commanded_arm": which, "jaw_delta_raw": jaw, "frame_mean_abs_diff": diff,
                    "feed_that_moved": moved_cam, "jaw_that_moved": moved_jaw,
                    "cameras_swapped": moved_cam is not None and moved_cam != which,
                    "state_jaw_swapped": moved_jaw != which})


@app.route("/grip/<arm>/<cmd>", methods=["POST"])
def grip_one(arm, cmd):
    """Drive ONE jaw to an explicit command and leave it there.

    Command units are 0..45, not the raw count /observe reports (obs = cmd * -6). 45 is the open end,
    0 the closed one. The arm joints do not move: the vector sent is the current leader pose with a single
    element replaced. /gripper_probe moves both jaws and puts them back, which is a different job.
    """
    gi = GRIP_IDX[0 if arm == "left" else 1]
    lead = _leader_now()
    if lead is None:
        return jsonify({"error": "observe failed"})
    before = _observe()["joints14"]
    out = list(lead)
    out[gi] = float(cmd)
    r = _exec_step(out)
    if isinstance(r, dict) and r.get("error"):
        return jsonify(r)
    time.sleep(1.5)
    after = _observe()["joints14"]
    return jsonify({"arm": arm, "cmd": float(cmd),
                    "raw_before": [round(float(before[g]), 2) for g in GRIP_IDX],
                    "raw_after": [round(float(after[g]), 2) for g in GRIP_IDX]})


@app.route("/grip_calib", methods=["POST"])
def grip_calib():
    """Measure each jaw's command -> raw-count law separately, because they are not the same law.

    The core has been assuming one constant for both arms (obs = cmd * -6, commands 0..45). Measured on this
    robot the left follows roughly that and the right does not: it responds to the opposite sign and a
    different gain, so every gripper command the policy issued for the right arm was wrong. This sweeps a
    command ladder per arm, waits for the jaw to settle at each point, and fits obs = a*cmd + b over the
    points that are not against a stop.

    Arms do not move; only the one jaw element of the leader vector changes.
    """
    import numpy as _np
    body = request.get_json(silent=True) or {}
    cmds = [float(x) for x in body.get("cmds", [0, -45, -30, -20, -10, 10, 20, 30, 45, 0])]
    settle = float(body.get("settle_s", 1.6))
    out = {}
    for arm, gi in (("left", GRIP_IDX[0]), ("right", GRIP_IDX[1])):
        pts = []
        for c in cmds:
            lead = _leader_now()
            if lead is None:
                return jsonify({"error": "observe failed"})
            v = list(lead)
            v[gi] = c
            r = _exec_step(v)
            if isinstance(r, dict) and r.get("error"):
                return jsonify(r)
            time.sleep(settle)
            pts.append((c, float(_observe()["joints14"][gi])))
        # The readback trails the command by exactly ONE command, not by a settling time -- 3.5 s of dwell
        # does not remove it. Pairing obs[i] with cmd[i] therefore measures the PREVIOUS command and makes
        # the gain look arm-dependent and non-linear; pairing obs[i+1] with cmd[i] gives -6.00 on both arms.
        a = _np.array([(pts[i][0], pts[i + 1][1]) for i in range(len(pts) - 1)])
        # drop the ends that sit against a stop: a command that changes nothing carries no information
        lo, hi = a[:, 1].min(), a[:, 1].max()
        span = hi - lo
        keep = a[(a[:, 1] > lo + 0.02 * span) & (a[:, 1] < hi - 0.02 * span)] if span > 1 else a
        fit = None
        if len(keep) >= 2:
            A = _np.vstack([keep[:, 0], _np.ones(len(keep))]).T
            (g, b), res, *_ = _np.linalg.lstsq(A, keep[:, 1], rcond=None)
            pred = A @ _np.array([g, b])
            ss = float(1 - ((keep[:, 1] - pred) ** 2).sum() /
                       max(((keep[:, 1] - keep[:, 1].mean()) ** 2).sum(), 1e-9))
            fit = {"obs_per_cmd": round(float(g), 4), "offset": round(float(b), 3), "r2": round(ss, 5),
                   "n_fit": len(keep)}
        out[arm] = {"points": [[round(c, 1), round(o, 2)] for c, o in pts],
                    "lag_paired": [[round(c, 1), round(o, 2)] for c, o in a], "raw_min": round(float(lo), 2),
                    "raw_max": round(float(hi), 2), "fit": fit}
    return jsonify(out)
