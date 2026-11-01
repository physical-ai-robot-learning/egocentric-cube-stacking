"""Pose routes: training start pose, HRA start pose, the start-pose list and joint jog.
"""
from ego_paths import DATA_ROOT  # machine-specific roots (env-overridable)

import base64
import math
import os
import time

from flask import jsonify, request
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import ARM_IDX, FLIP_IDX, GRIP_IDX, IC as _IC
from .motion import _exec_step, _goto, _mask_arm
from .robot_client import _jaw_torque, _leader_now, _observe, _robot


@app.route("/goto_train_start", methods=["POST"])
def goto_train_start():
    """[2026-09-29] Evaluation initialisation only: drive both arms (slew-limited _goto) to the recorded training start pose
    closest to the median of the 180 episode starts (train_start_pose.json), jaws CLOSED as at those starts. Then /run."""
    if rt.STATE["running"] or not rt._lock.acquire(blocking=False):
        return jsonify({"error": "busy / running"})
    try:
        import json as _j
        P = _j.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_start_pose.json")))
        tgt = np.zeros(14); tgt[ARM_IDX] = P["q12_follower_rad"]; tgt[FLIP_IDX] *= -1.0
        for g_ in GRIP_IDX:
            tgt[g_] = 0.0                                   # CLOSED, as at the training starts
        r = _goto(list(tgt))
        o = _observe(); m, _ = rt.INF.tcp_now(o["joints14"]) if o else (None, None)
        now = None if m is None else {"L": [round(float(x) * 1000, 1) for x in m[0][:3, 3]], "R": [round(float(x) * 1000, 1) for x in m[1][:3, 3]]}
        return jsonify({"goto": r, "target_tcp_mm": P["tcp_mm"], "now_tcp_mm": now, "source": P["source"],
                        "next": "wait until still, then RESET ANCHOR (relcart20) and RUN"})
    finally:
        rt._lock.release()


@app.route("/hra_start_pose", methods=["POST"])
def hra_start_pose():
    """[2026-10-06 user] HRA start pose: tilt the RIGHT gripper so its approach axis points PITCH deg below horizontal (the 166 human
    training starts: p10/p50/p90 = 32/36/39 deg, roll ~ -3 deg), keeping the TCP position (+ optional dz) and the approach heading.
    Closing axis made horizontal (roll 0). Left arm + jaws hold. Walked in STEPS small IK waypoints from the measured pose; stops on an
    IK failure. Then RUN (the next observation becomes the RELCART20 anchor). Query: pitch (deg, default 35), dz_mm (0), steps (7)."""
    if rt.STATE["running"] or not rt._lock.acquire(blocking=False):
        return jsonify({"error": "busy / running"})
    try:
        pitch = float(request.args.get("pitch", 35.0)); dz = float(request.args.get("dz_mm", 0.0)) / 1000.0
        steps = max(1, min(int(request.args.get("steps", 7)), 30))
        if not (0.0 <= pitch <= 60.0) or abs(dz) > 0.15:
            return jsonify({"error": "pitch must be 0..60 deg, |dz_mm| <= 150"})
        o = _observe()
        if o is None:
            return jsonify({"error": "observe failed"})
        m0, _ = rt.INF.tcp_now(o["joints14"]); R0 = m0[1][:3, :3]; p0 = m0[1][:3, 3].copy()
        up = np.array([0.0, 0.0, 1.0])                                   # base z = up (assumes a level mount)
        h = R0[:, 0] - (R0[:, 0] @ up) * up
        if np.linalg.norm(h) < 1e-3:
            return jsonify({"error": "gripper points straight up/down: heading undefined"})
        h /= np.linalg.norm(h); pr = np.radians(pitch)
        a = np.cos(pr) * h - np.sin(pr) * up                              # approach axis, PITCH below horizontal
        y = np.cross(up, a); y /= np.linalg.norm(y)
        if y @ R0[:, 1] < 0: y = -y                                       # keep the jaw-closing side
        rr = np.radians(float(request.args.get("roll", 0.0)))           # roll about the approach axis (human median -3 deg)
        if rr:
            y = np.cos(rr) * y - np.sin(rr) * np.cross(a, y)
        Rt = np.stack([a, y, np.cross(a, y)], 1); pt = p0 + dz * up
        if request.args.get("saved") == "1":          # [10-06] the pose saved as "robot = HandUMI start" (umi_start_pose.json, offset-free frame)
            import json as _j
            _nm = os.path.basename(request.args.get("pose", ""))      # [2026-10-07 user "새 시작점 추가"] choose from ~/c8/hra_red/start_poses/
            _pf = str(DATA_ROOT / f"hra_red/start_poses/{_nm}") if _nm else str(DATA_ROOT / "hra_red/umi_start_pose.json")
            if not os.path.exists(_pf):
                return jsonify({"error": f"no start pose file {_pf}"})
            Ts = np.array(_j.load(open(_pf))["T_base_tcp"])
            # [2026-10-07 user "좀더 오른쪽 그리고 아래"] offsets in the ROBOT WRIST CAMERA view (hand-eye X_tcp_cam): right = camera x,
            # back = -optical axis, both projected on the horizontal plane; down = base -z. Orientation unchanged.
            _r, _b, _d = (float(request.args.get(k_, 0.0)) / 1000.0 for k_ in ("right_mm", "back_mm", "down_mm"))
            if _r or _b or _d:
                _X = np.array(_j.load(open(str(DATA_ROOT / "hra_red/handeye/robot_right_wrist_handeye.json")))["X_tcp_cam"]["park"])
                _C = Ts @ _X; _up = np.array([0.0, 0.0, 1.0])
                _ex = _C[:3, 0] - (_C[:3, 0] @ _up) * _up; _ex /= np.linalg.norm(_ex)
                _ez = _C[:3, 2] - (_C[:3, 2] @ _up) * _up; _ez /= np.linalg.norm(_ez)
                Ts = Ts.copy(); Ts[:3, 3] += _r * _ex - _b * _ez - _d * _up
            Rt = Ts[:3, :3] @ (_IC.R_UMI_OFF if _IC.UMI_CONV else np.eye(3))
            pt = Ts[:3, 3] + (Ts[:3, :3] @ _IC.UMI_TCP_OFF_T if _IC.UMI_CONV else 0.0)   # [10-06] same HandUMI convention as tcp_now
            if np.linalg.norm(pt - p0) > 0.35:     # [2026-10-07] 250 -> 350 mm: the HRA_A100 human-median start is 294 mm from rest
                return jsonify({"error": f"saved start {np.round(pt*1000,1)} mm is > 350 mm from the current TCP -- refused"})
        if all(k_ in request.args for k_ in ("x_mm", "y_mm", "z_mm")):   # [10-06] absolute base target (the recorded run-start position)
            pt = np.array([float(request.args[k_]) for k_ in ("x_mm", "y_mm", "z_mm")]) / 1000.0
            if np.linalg.norm(pt - p0) > 0.25:
                return jsonify({"error": f"target {np.round(pt*1000,1)} mm is > 250 mm from the current TCP -- refused"})
        Rl = m0[0][:3, :3]; pl = m0[0][:3, 3]
        rots = Rot.from_matrix(np.stack([R0, Rt]))
        from scipy.spatial.transform import Slerp
        sl = Slerp([0, 1], rots); log = []
        # [2026-10-07 user "끊겨서 가거든 한번에 가도록"] solve EVERY waypoint first (sequential IK seeded with the previous solution),
        # then stream the whole joint path at CMD_HZ without stopping, and dwell (reach check) only at the final waypoint.
        # The old loop called _goto per waypoint = slew + dwell-until-reached each time -> stop-and-go.
        o0 = _observe()
        if o0 is None:
            return jsonify({"error": "observe failed"})
        q_seed = list(o0["joints14"]); path = []
        for i in range(1, steps + 1):
            f = i / steps; Ri = sl([f]).as_matrix()[0]; pi = p0 + f * (pt - p0)
            P = np.stack([pl, pi]); Q = np.stack([Rot.from_matrix(Rl).as_quat(), Rot.from_matrix(Ri).as_quat()])
            act, ikerr, ok, clipped = rt.INF.solve_waypoint(q_seed, P, Q, None)
            ok_r = bool(np.asarray(ok).reshape(-1, 2)[0][1]) if np.asarray(ok).size >= 2 else bool(np.all(ok))
            if not ok_r:
                return jsonify({"error": f"IK failed at waypoint {i}/{steps} -- robot NOT moved", "steps": log})
            act = list(_mask_arm(np.asarray(act)[None, :], "right", o0["joints14"])[0])
            path.append((act, P, Q, ikerr))
            for j_ in range(7, 13):                                       # next seed = this solution in the /observe frame
                q_seed[j_] = -act[j_] if j_ in FLIP_IDX else act[j_]
        cur = _leader_now()
        if cur is None:
            return jsonify({"error": "observe failed"})
        prev = list(cur); sent = 0
        for wi, (act, P, Q, ikerr) in enumerate(path):
            n = max(1, int(math.ceil(max(abs(act[j_] - prev[j_]) for j_ in ARM_IDX) / cfg.SLEW_MAX_STEP_RAD)))
            for k in range(1, n + 1):
                if rt.STATE["stop_reason"] == "estop":
                    return jsonify({"error": "estop", "steps": log})
                a = [prev[j_] + (act[j_] - prev[j_]) * k / n for j_ in range(14)]
                for _g in GRIP_IDX:
                    a[_g] = act[_g]
                r = _exec_step(a, k, n)
                if isinstance(r, dict) and r.get("error"):
                    return jsonify({"error": r["error"], "steps": log})
                sent += 1; time.sleep(1.0 / cfg.CMD_HZ)
            prev = list(act)
        act, P, Q, ikerr = path[-1]
        r = _goto(list(act), (P, Q), ikerr)
        log.append({"waypoints": steps, "streamed_cmds": sent, "reached": r.get("reached"), "tcp_err_mm": r.get("tcp_err_mm")})
        o = _observe(); m1, _ = rt.INF.tcp_now(o["joints14"]); R1 = m1[1][:3, :3]
        got = float(np.degrees(np.arcsin(np.clip(-R1[:, 0] @ up, -1, 1))))
        roll = float(np.degrees(np.arcsin(np.clip(R1[:, 1] @ up, -1, 1))))
        # [10-06 user] no implicit re-anchor here: the episode starts with RESET ANCHOR -> RUN loop, as before
        return jsonify({"ok": True, "pitch_target_deg": pitch, "pitch_now_deg": round(got, 1), "roll_now_deg": round(roll, 1),
                        "tcp_mm_before": [round(float(x) * 1000, 1) for x in p0], "tcp_mm_now": [round(float(x) * 1000, 1) for x in m1[1][:3, 3]],
                        "steps": log, "next": "place the cube ~5 cm ahead / 15-20 cm inward of the gripper, then RUN"})
    finally:
        rt._lock.release()


@app.route("/start_poses")
def start_poses():
    """[2026-10-07] saved start poses (~/c8/hra_red/start_poses/*.json) for the UMI start button."""
    import json as _j, glob as _g
    out = []
    for f in sorted(_g.glob(str(DATA_ROOT / "hra_red/start_poses/*.json"))):
        try:
            T = np.array(_j.load(open(f))["T_base_tcp"]); out.append({"name": os.path.basename(f), "tcp_mm": [round(float(x) * 1000, 1) for x in T[:3, 3]]})
        except Exception:
            pass
    return jsonify({"poses": out})


@app.route("/jog/<int:idx>/<deg>", methods=["POST"])
def jog(idx, deg):
    """Move ONE arm joint by a few degrees and report which half of the global frame changed.

    This is the only question the pictures could not answer: the live global view shows no arm at rest, so
    "image left = robot left?" is undecidable from a still frame. Moving a known joint decides it, and the
    answer says whether V4_GLOBAL_ROT180 puts the policy's picture in the same physical correspondence the
    training data had, or mirrors it.

    Deliberately small and reversible: arm joints only, |deg| <= 10, through _goto so the slew cap and the
    E-STOP check apply, and the caller is expected to send the negative straight after. Never a jaw index --
    the jaw is a 0..45 command in the same vector and does not belong on this path.
    """
    d = float(deg)
    if idx not in ARM_IDX:
        return jsonify({"error": f"joint {idx} is not an arm joint; ARM_IDX = {ARM_IDX}"})
    # 10 deg is the default cap, but at rest NEITHER arm is inside the global camera's field of view, so a
    # small jog changes nothing there: measured 2026-09-25, an 8 deg shoulder_pan gave a whole-frame diff of
    # max 19.7 / p99 5.0 with ZERO pixels over 30 -- pure sensor noise, and the left/right split of that
    # noise was mistaken for the arm. Bringing an arm into view needs a bigger move, so anything past 10 deg
    # is allowed only with confirm=1.
    if abs(d) > 10.0 and request.args.get("confirm") != "1":
        return jsonify({"error": f"refusing {d} deg without confirm=1; the default cap is 10"})
    if abs(d) > 45.0:
        return jsonify({"error": f"refusing {d} deg; hard cap is 45"})
    if rt.STATE["running"]:
        return jsonify({"error": "stop the running loop first"})

    def half_diff(b64_before, b64_after, rot):
        import cv2
        def dec(b):
            im = cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_COLOR)
            return cv2.rotate(im, cv2.ROTATE_180) if rot else im
        a, b = dec(b64_before), dec(b64_after)
        dif = cv2.absdiff(a, b).mean(axis=2)
        w = dif.shape[1] // 2
        return round(float(dif[:, :w].mean()), 3), round(float(dif[:, w:].mean()), 3)

    o0 = _observe()
    if o0 is None:
        return jsonify({"error": "observe failed"})
    before = o0["images"]["middle"]
    cur = _leader_now()
    if cur is None:
        return jsonify({"error": "observe failed"})
    tgt = list(cur)
    tgt[idx] = cur[idx] + math.radians(d)
    r = _goto(tgt)
    if isinstance(r, dict) and r.get("error"):
        return jsonify({"error": r["error"]})
    time.sleep(0.6)
    o1 = _observe()
    after = o1["images"]["middle"] if o1 else None
    back = _goto(list(cur))                                  # straight back, same ramp
    if after is None:
        return jsonify({"error": "observe failed after the move"})
    rot = bool(_IC.GLOBAL_ROT180)
    import cv2 as _cv
    def _dec(b, rot=False):
        im = _cv.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), _cv.IMREAD_COLOR)
        return _cv.rotate(im, _cv.ROTATE_180) if rot else im
    _d = _cv.absdiff(_dec(before), _dec(after)).mean(axis=2)
    _peak, _npx = _d.max(), (_d > 30).sum()
    # The wrist cameras are bolted to the arms, so they answer the routing question the global view cannot:
    # move 14-vector index 0 and the LEFT wrist feed must be the one that changes. /arm_ident checked this
    # for the jaw channel only; the arm joint block is a separate path and is checked here.
    def feed_diff(cam):
        import cv2 as _c
        def dec(b):
            return _c.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), _c.IMREAD_COLOR)
        return round(float(_c.absdiff(dec(o0["images"][cam]), dec(o1["images"][cam])).mean()), 3)
    wl, wr = feed_diff("left"), feed_diff("right")
    raw_l, raw_r = half_diff(before, after, False)
    mod_l, mod_r = half_diff(before, after, rot)
    return jsonify({
        "joint": idx, "deg": d, "returned": not (isinstance(back, dict) and back.get("error")),
        "global_transform": "rot180" if rot else "none",
        "raw_frame": {"left_half": raw_l, "right_half": raw_r,
                      "changed": "LEFT" if raw_l > raw_r else "RIGHT"},
        "model_frame": {"left_half": mod_l, "right_half": mod_r,
                        "changed": "LEFT" if mod_l > mod_r else "RIGHT"},
        "peak_diff": round(float(_peak), 1), "pixels_over_30": int(_npx),
        "wrist_diff": {"left": wl, "right": wr,
                       "changed": "LEFT" if wl > wr else "RIGHT",
                       "ratio": round(max(wl, wr) / max(1e-6, min(wl, wr)), 2)},
        "note": ("verdict uses model_frame; raw_frame must be its mirror image if rot180 is on. "
                 "pixels_over_30 near zero means the arm never entered the frame and the halves are noise")})


@app.route("/train_start", methods=["POST"])
def train_start():
    """Move to the training data's frame-0 median pose. Rollouts start here so the start pose is in distribution."""
    if rt.STATE["running"]:
        return jsonify({"error": "stop the loop first"})
    st = _robot("/status")
    if not st.get("connected"):
        return jsonify({"error": "robot not connected"})
    if st.get("estop"):
        return jsonify({"error": "E-STOP engaged"})
    tgt = np.asarray(cfg.TRAIN_START_DEG, float).copy()
    for i in ARM_IDX:
        tgt[i] = math.radians(cfg.TRAIN_START_DEG[i])
    tgt[GRIP_IDX[0]] = tgt[GRIP_IDX[1]] = 0.0            # jaws at raw 0
    lead = tgt.copy()
    for k in FLIP_IDX:
        lead[k] = -lead[k]
    rt.STATE["stop_reason"] = None
    _jaw_torque(True)                  # rest leaves them unpowered; a pose move commands them again
    with rt._lock:
        r = _goto([float(x) for x in lead])
    return jsonify(r)
