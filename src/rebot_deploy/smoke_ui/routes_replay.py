"""Recorded-chunk replay (/replay, /replay_stop).
"""
from ego_paths import HOLOBRAIN_MAC  # machine-specific roots (env-overridable)

import math
import threading

from flask import jsonify, request
import numpy as np

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import ARM_IDX, FLIP_IDX, GRIP_IDX
from .motion import _goto
from .robot_client import _observe


def _replay_loop(mode, n_action, stride, max_queries, use_grip, anchor="source", ep=0):
    """Drive the robot through the deployment loop with the POLICY REPLACED BY GROUND TRUTH.

    Same decode, same per-waypoint IK re-solve, same dwell as a real rollout -- only the chunk comes from the
    dataset instead of the model. So if this does not reproduce the demonstration's motion, the fault is in the
    decoder or the execution path, and no checkpoint can fix it.

      delta : T_k = T_{k-1} @ dT_k   anchored on the MEASURED pose  (cumulative, v4)
      umi   : T_k = T_measured @ A_k                                 (current-anchor, upstream UMI)

    The anchor is the measured pose, not the demo's absolute start, so the arm reproduces the demonstration's
    RELATIVE motion from wherever it currently is. That is what deployment does, and it avoids a jump.
    """
    import numpy as _np
    from scipy.spatial.transform import Rotation as _Rot
    path = str(HOLOBRAIN_MAC / f"chunks_ep{ep}.npz")
    d = _np.load(path, allow_pickle=True)
    SRC = d["src"]
    JOINTS = None
    if mode == "joints":
        # the recorded joint trajectory: what the robot physically did during the demo, with no IK, no FK and
        # no frame convention in the way. The purest possible reference for the other three modes.
        jp = str(HOLOBRAIN_MAC / f"joints_ep{ep}.npz")
        JOINTS = _np.load(jp)["joints"]                      # (frames+1, 2, 4, 4) absolute source TCP poses, the teleop itself
    C = d["delta"] if mode == "delta" else (d["umi"] if mode == "umi" else d["delta"])
    if JOINTS is not None:
        C = JOINTS[:max(len(JOINTS) - 1, 1)]
    total = len(C) if max_queries <= 0 else min(len(C), max_queries * stride)
    rt.STATE["replay"] = {"running": True, "mode": mode, "anchor": anchor, "episode": int(ep), "rows": [],
                       "file": path,
                       "done": 0, "total_queries": int(np.ceil(total / stride)), "frames": int(len(C))}
    q = 0
    try:
        # The replay targets the demonstration's absolute poses, so the arm has to be AT its start first --
        # otherwise the very first step is the whole distance to the demo (128.7 mm on the first attempt) and
        # the safety guard stops it, correctly. This approach move is slew-limited like every other motion.
        o0 = _observe()
        if o0 is None:
            rt.STATE["replay"]["rows"].append({"error": "observe failed before approach"})
            raise RuntimeError("no observation")
        if JOINTS is not None:
            tgt0 = list(JOINTS[0])
            for _j in ARM_IDX:
                tgt0[_j] = math.radians(tgt0[_j])
            for _f in FLIP_IDX:
                tgt0[_f] = -tgt0[_f]
            rt.STATE["replay"]["approach"] = {"mode": "joints"}
            print("[v4-smoke] replay approach: recorded joints of frame 0 (no IK)", flush=True)
            r0 = _goto(tgt0)
            if isinstance(r0, dict) and r0.get("error"):
                rt.STATE["replay"]["rows"].append({"error": f"approach failed: {r0['error']}"})
                raise RuntimeError("approach failed")
            approached = True
        else:
            approached = False
        if not approached:
          start_pos = _np.stack([SRC[0, r][:3, 3] for r in (0, 1)])
          start_quat = _np.stack([_Rot.from_matrix(SRC[0, r][:3, :3]).as_quat() for r in (0, 1)])
          m0 = rt.INF.tcp_now(o0["joints14"])[0]
          gap = max(float(_np.linalg.norm(start_pos[r] - m0[r][:3, 3])) * 1000 for r in (0, 1))
          a0, ik0, ok0, _c = rt.INF.solve_waypoint(o0["joints14"], start_pos, start_quat,
                                              C[0, 0, [9, 19]] if use_grip else None)
          rt.STATE["replay"]["approach"] = {"gap_mm": round(gap, 1), "ik_fk_err_mm": round(ik0, 2), "ik_ok": bool(ok0)}
          print(f"[v4-smoke] replay approach: {gap:.1f} mm to episode start, ik residual {ik0:.2f} mm", flush=True)
          r0 = _goto(list(a0), (start_pos, start_quat), ik0)
          if isinstance(r0, dict) and r0.get("error"):
              rt.STATE["replay"]["rows"].append({"error": f"approach failed: {r0['error']}"})
              raise RuntimeError("approach failed")
        while rt.STATE["replay"]["running"] and q < total:
            o = _observe()
            if o is None:
                rt.STATE["replay"]["rows"].append({"error": "observe failed"}); break
            m_now = rt.INF.tcp_now(o["joints14"])[0]
            # anchor=source replays the demonstration itself: every representation decodes against the pose the
            # demo was at, so all three must trace the same path. anchor=measured is what deployment does, and
            # it is where a cumulative representation can drift on execution lag.
            if mode == "joints":
                # send the recorded joints straight through the leader-frame path, slew limited as always
                tgt = list(JOINTS[q])
                for _j in ARM_IDX:
                    tgt[_j] = math.radians(tgt[_j])          # the source dataset stores degrees
                for _f in FLIP_IDX:
                    tgt[_f] = -tgt[_f]
                if not use_grip:
                    lead_now = list(o["joints14"])
                    for _f in FLIP_IDX:
                        lead_now[_f] = -lead_now[_f]
                    for _g in GRIP_IDX:
                        tgt[_g] = lead_now[_g]
                r_ = _goto(tgt)
                o3 = _observe()
                m_after = rt.INF.tcp_now(o3["joints14"])[0] if o3 else None
                row = {"q": int(q), "sent": 1}
                for r, nm in ((0, "L"), (1, "R")):
                    got = (m_after[r][:3, 3] - m_now[r][:3, 3]) if m_after is not None else _np.zeros(3)
                    row[nm] = {"want_mm": None, "got_mm": [round(float(x) * 1000, 2) for x in got], "cos": None}
                rt.STATE["replay"]["rows"].append(row)
                rt.STATE["replay"]["done"] += 1
                q += stride
                continue
            base = m_now if anchor == "measured" else SRC[q]
            ch = C[q]
            pos = _np.zeros((16, 2, 3)); quat = _np.zeros((16, 2, 4)); wid = _np.zeros((16, 2))
            for r in (0, 1):
                cur = base[r].copy()
                for k in range(16):
                    M = _np.eye(4)
                    d6 = ch[k, r * 10 + 3:r * 10 + 9]
                    b1 = d6[:3] / _np.linalg.norm(d6[:3])
                    b2 = d6[3:] - b1 * (b1 @ d6[3:]); b2 /= _np.linalg.norm(b2)
                    M[:3, :3] = _np.stack([b1, b2, _np.cross(b1, b2)])
                    M[:3, 3] = ch[k, r * 10:r * 10 + 3]
                    if mode == "source":
                        Tk = SRC[min(q + k + 1, len(SRC) - 1), r]      # the recorded pose, no representation
                    elif mode == "delta":
                        Tk = cur @ M
                        cur = Tk
                    else:
                        Tk = base[r] @ M
                    pos[k, r] = Tk[:3, 3]
                    quat[k, r] = _Rot.from_matrix(Tk[:3, :3]).as_quat()
                    wid[k, r] = ch[k, r * 10 + 9]
            step_mm = max(float(_np.linalg.norm(pos[0, r] - m_now[r][:3, 3])) * 1000 for r in (0, 1))
            if cfg.REPLAY_MAX_STEP_MM and step_mm > cfg.REPLAY_MAX_STEP_MM:
                rt.STATE["replay"]["rows"].append({"q": int(q), "abort": f"first step {step_mm:.1f} mm > "
                                                f"{cfg.REPLAY_MAX_STEP_MM}"})
                break
            sent = 0
            for k in range(min(n_action, 16)):
                if not rt.STATE["replay"]["running"]:
                    break
                o2 = _observe()
                if o2 is None:
                    break
                a, ikerr, ok, clip = rt.INF.solve_waypoint(o2["joints14"], pos[k], quat[k],
                                                        wid[k] if use_grip else None)
                r_ = _goto(list(a), (pos[k], quat[k]), ikerr)
                if isinstance(r_, dict) and r_.get("error"):
                    rt.STATE["replay"]["rows"].append({"q": int(q), "error": r_["error"]}); rt.STATE["replay"]["running"] = False
                    break
                sent += 1
            o3 = _observe()
            m_after = rt.INF.tcp_now(o3["joints14"])[0] if o3 else None
            row = {"q": int(q), "sent": sent}
            for r, nm in ((0, "L"), (1, "R")):
                want = pos[max(sent - 1, 0), r] - m_now[r][:3, 3]
                got = (m_after[r][:3, 3] - m_now[r][:3, 3]) if m_after is not None else _np.zeros(3)
                row[nm] = {"want_mm": [round(float(x) * 1000, 2) for x in want],
                           "got_mm": [round(float(x) * 1000, 2) for x in got],
                           "cos": round(float(_np.dot(want, got) / max(_np.linalg.norm(want) * _np.linalg.norm(got), 1e-12)), 3)}
            rt.STATE["replay"]["rows"].append(row)
            rt.STATE["replay"]["done"] += 1
            q += stride
    finally:
        rt.STATE["replay"]["running"] = False
    print(f"[v4-smoke] replay {mode} done: {len(rt.STATE['replay']['rows'])} queries", flush=True)


@app.route("/replay", methods=["POST"])
def replay():
    if rt.STATE["running"] or rt.STATE.get("replay", {}).get("running"):
        return jsonify({"error": "stop the other run first"})
    mode = request.args.get("mode", "delta")
    if mode not in ("delta", "umi", "source", "joints"):
        return jsonify({"error": "mode must be joints, source, delta or umi"})
    anchor = request.args.get("anchor", "source")
    ep = int(request.args.get("ep", "0"))
    n_action = int(request.args.get("n_action", "4"))
    stride = int(request.args.get("stride", str(n_action)))
    max_q = int(request.args.get("queries", "0"))          # 0 = the whole episode
    grip = request.args.get("gripper", "1") != "0"
    threading.Thread(target=_replay_loop, args=(mode, n_action, stride, max_q, grip, anchor, ep),
                     daemon=True).start()
    return jsonify({"started": True, "mode": mode, "anchor": anchor, "episode": ep,
                    "n_action": n_action, "stride": stride, "queries": max_q, "gripper": grip})


@app.route("/replay_stop", methods=["POST"])
def replay_stop():
    if rt.STATE.get("replay"):
        rt.STATE["replay"]["running"] = False
    return jsonify({"stopped": True})
