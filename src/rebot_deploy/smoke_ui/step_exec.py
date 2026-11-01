"""STEP executor: one press = observe -> infer -> drive the chunk's waypoints (with optional prefetch of the next
inference while the last waypoint runs), and the continuous loop of presses.
"""
import math
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import settings as cfg
from . import state as rt
from .ic import ARM_IDX, FLIP_IDX, GRIP_IDX, IC_EXEC_K, INF_W_OPEN, LIMITS
from .motion import _goto, _mask_arm
from .robot_client import _jaw_torque, _motor_pos, _observe
from .safety import _hra_stop


def _ema(key, v, a=0.3):
    rt._PFT[key] = v if rt._PFT[key] is None else (1 - a) * rt._PFT[key] + a * v


def _prefetch_worker():
    try:
        t_req = time.perf_counter()
        h0 = _observe()
        time.sleep(cfg.OBS_GAP_S)
        h1 = _observe()
        if h0 is None or h1 is None:
            rt._PF["res"] = {"error": "observe failed"}; return
        obs_ms = round((time.perf_counter() - t_req) * 1000.0, 1)
        t0 = time.time(); t_i = time.perf_counter()
        rt.INF._rtc_delay_hint = rt._PF.get("delay")      # [2026-10-06] RTC delay = waypoints the arm runs while this chunk is inferred
        try:
            cmd = rt.INF.infer([h0, h1])
        finally:
            rt.INF._rtc_delay_hint = None
        rt._PF["res"] = dict(h0=h0, h1=h1, cmd=cmd, L=rt.INF.last, t0=t0, t_obs=time.time(), observe_ms=obs_ms,
                          infer_ms=round((time.perf_counter() - t_i) * 1000.0, 1), exec_k_now=getattr(rt.INF, "exec_k_now", None))
        _ema("inf", rt._PF["res"]["infer_ms"] / 1000.0)
    except Exception as e:  # noqa: BLE001
        rt._PF["res"] = {"error": f"prefetch inference failed: {e}"}


def _prefetch_start():
    rt._PF["res"] = None
    th = threading.Thread(target=_prefetch_worker, daemon=True); rt._PF["thread"] = th; th.start()


def _prefetch_take():
    """the finished prefetch (waits for a running one), or None"""
    th = rt._PF["thread"]
    if th is None:
        return None
    th.join(timeout=10.0); rt._PF["thread"] = None
    r = rt._PF["res"]; rt._PF["res"] = None
    if not r or r.get("error"):
        if r: print(f"[v4-smoke] prefetch dropped: {r.get('error')}", flush=True)
        return None
    if time.time() - r["t_obs"] > cfg.PREFETCH_MAX_AGE_S:
        print(f"[v4-smoke] prefetch dropped: {time.time() - r['t_obs']:.1f}s old", flush=True); return None
    return r


def _one_press():
    T = {}
    pf = _prefetch_take() if cfg.PREFETCH else None
    if pf is not None:
        # [2026-10-02] pipelined cycle: observation + inference already ran during the previous cycle's last waypoint
        h0, h1, cmd, t0 = pf["h0"], pf["h1"], pf["cmd"], pf["t0"]
        rt.INF.last = pf["L"]
        if pf.get("exec_k_now") is not None:
            rt.INF.exec_k_now = pf["exec_k_now"]
        T["observe_ms"] = pf["observe_ms"]; T["infer_ms"] = pf["infer_ms"]; T["prefetched"] = True
        T["prefetch_age_ms"] = round((time.time() - pf["t_obs"]) * 1000.0, 1)
        infer_s = T["infer_ms"] / 1000.0
    else:
        t_req = time.perf_counter()
        h0 = _observe()
        if h0 is None:
            return {"error": "observe failed (robot connected?)"}
        time.sleep(cfg.OBS_GAP_S)
        h1 = _observe()
        if h1 is None:
            return {"error": "observe failed"}

        T["observe_ms"] = round((time.perf_counter() - t_req) * 1000.0, 1)   # upper bound on observation age:
        # /observe returns no capture timestamp, so the true age of the pixels is at most this and cannot be
        # separated from the round trip until robot_service stamps its frames.
        t0 = time.time(); t_i = time.perf_counter()
        try:
            cmd = rt.INF.infer([h0, h1])
        except Exception as e:  # noqa: BLE001
            return {"error": f"inference failed: {e}"}
        infer_s = time.time() - t0
        T["infer_ms"] = round((time.perf_counter() - t_i) * 1000.0, 1)
        T["prefetched"] = False
    # [2026-09-30 user] frame age: capture wall time of each camera frame fed to the model (h1 = the current observation),
    # from robot_service cam_age_ms + t_wall; ages at inference start and at each waypoint command go into the cycle log.
    _cap = {c: h1["t_wall"] - a / 1000.0 for c, a in (h1.get("cam_age_ms") or {}).items() if a is not None and h1.get("t_wall")}
    _age = lambda t_: {c: round((t_ - v) * 1000.0, 1) for c, v in _cap.items()}
    age_infer = _age(t0)
    L = rt.INF.last
    # [2026-09-29] keep the exact inference input of every cycle (both observations: images b64, joints, t) so a real frame
    # can be re-run offline and compared with the training images (grasp-phase OOD check). -> V4_FRAME_DIR/<t>.json
    try:
        _fd = os.path.expanduser(os.environ.get("V4_FRAME_DIR", "~/v4_cycle_frames")); os.makedirs(_fd, exist_ok=True)
        with open(os.path.join(_fd, f"{h1['t']:.3f}.json"), "w") as f_:
            __import__("json").dump(dict(cycle=rt.STATE["cycles"], ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")),
                                         exec_k=int(os.environ.get("V4_EXEC_K", "0") or 0), task=rt.INF.task,
                                         obs=[{"t": h["t"], "joints14": list(map(float, h["joints14"])), "images": h["images"]} for h in (h0, h1)],
                                         state=[float(x) for x in L["state"]]), f_)
    except Exception as e_:  # noqa: BLE001
        print("[frame-log] skipped:", e_, flush=True)
    cmd = _mask_arm(cmd, rt.STATE["arm"], h1["joints14"])

    m_before, _ = rt.INF.tcp_now(h1["joints14"])
    # Axis-wise trace, in the BASE frame, so "it goes up when it should go down" can be attributed instead of
    # argued: the model's own body-frame delta rotated into base (pred), the commanded step after clamping
    # (cmd), and what the TCP actually did (actual). A sign that flips between pred and cmd is a compose bug;
    # between cmd and actual is execution; agreement all the way through is the policy.
    # Three-way vertical decomposition, one row per cycle, kept across cycles. With n_action=1 the loop
    # re-anchors on a fresh T_now every time, so a small per-step bias does not average out -- it is re-applied
    # to the new pose and the arm walks. Separating the three z's says which layer it comes from:
    #   a1_body_dz  the model's own output, before any frame or IK work touches it
    #   tgt_dz      the same step after T_now @ A[1] and the frame adapter, i.e. what was asked of the arm
    #   act_dz      what the TCP did
    # pred up and target up -> the policy. pred flat and target up -> decode/anchor. target flat and actual
    # up -> IK or the follower.
    for r in (0, 1):
        a1 = L["act"][0, r * 10:r * 10 + 3]
        tgt_dz = float(L["tgt_pos"][0][r][2] - m_before[r][2, 3]) * 1000.0
        rt.STATE["zlog"].append({"t": round(time.time() % 10000, 1), "cycle": rt.STATE["cycles"],
                              "arm": "LR"[r], "a1_body_dz": round(float(a1[2]) * 1000, 2),
                              "a1_body_norm": round(float(np.linalg.norm(a1)) * 1000, 2),
                              "tgt_dz": round(tgt_dz, 2), "tcp_z": round(float(m_before[r][2, 3]) * 1000, 1),
                              "act_dz": None})
    axes = {0: [], 1: []}
    prev_tgt = {r: m_before[r][:3, 3].copy() for r in (0, 1)}
    prev_act = {r: m_before[r][:3, 3].copy() for r in (0, 1)}
    sent = 0; track = []
    # execute from EXEC_K, not from the first waypoint; each is anchored on the same measured pose, so this
    # skips the low-signal head of the chunk rather than skipping motion
    _k0 = max(0, int(getattr(rt.INF, "exec_k_now", None) or IC_EXEC_K()) - 1)     # [2026-10-01] adaptive k: what infer() chose
    _shift = 0
    if T.get("prefetched"):
        # the arm kept moving after the prefetch observation: skip the waypoints it has already passed
        _o = _observe()
        if _o is not None:
            _m = rt.INF.tcp_now(_o["joints14"])[0]
            _d = [max(float(np.linalg.norm(np.asarray(L["tgt_pos"][i][r_], float) - _m[r_][:3, 3])) for r_ in (0, 1)) * 1000.0
                  for i in range(min(len(L["tgt_pos"]), _k0 + max(1, len(cmd))))]
            _j = int(np.argmin(_d)); _shift = _j + (1 if _d[_j] < cfg.PASS_TCP_MM else 0) - _k0
            _shift = max(0, min(_shift, max(0, len(cmd) - 1)))
        T["prefetch_shift"] = _shift
    wp_indices = [i for i in range(_k0 + _shift, _k0 + _shift + max(1, len(cmd))) if i < L["valid_step"]]
    _lead = cfg.PREFETCH_LEAD
    if _lead < 0:                                       # [2026-10-04] AUTO prefetch lead (see PREFETCH_LEAD)
        _lead = int(math.ceil(((rt._PFT["inf"] if rt._PFT["inf"] is not None else float(T.get("infer_ms") or 1500) / 1000.0) + 0.2)
                              / max(rt._PFT["wp"] if rt._PFT["wp"] is not None else 0.2, 0.05)))
        rt._PFT["lead"] = _lead
    _t_wp = None
    for k in wp_indices:
        # re-solve this waypoint from where the arm actually is; the Cartesian target itself is untouched
        _now = time.perf_counter()
        if _t_wp is not None: _ema("wp", min(_now - _t_wp, 1.0))
        _t_wp = _now
        o = _observe()
        if o is None:
            return {"error": "observe failed mid-chunk", "sent": sent}
        t_k = time.perf_counter()
        a, ikerr, ok, clipped = rt.INF.solve_waypoint(o["joints14"], L["tgt_pos"][k], L["tgt_quat"][k],
                                                   L["widths"][k] if L["gripper_mode"] in ("predict", "binary", "continuous", "relative") else None)
        ik_ms = (time.perf_counter() - t_k) * 1000.0
        t_cmd_k = time.time()   # [2026-09-30] waypoint command send time (frame age at command)
        if getattr(rt.INF, "_lik_stop", None):        # [2026-09-29] pure learned-IK A/B: stop, do not execute, record why
            print(f"[v4-smoke] LEARNED_IK_STOP: {rt.INF._lik_stop}", flush=True)
            return {"error": f"LEARNED_IK_STOP: {rt.INF._lik_stop}", "sent": sent}
        base = _motor_pos() if k == 0 else None
        a = _mask_arm(np.asarray(a)[None, :], rt.STATE["arm"], o["joints14"])[0]
        _pt = cfg.PASS_THROUGH and (k != wp_indices[-1] or (cfg.PASS_LAST and cfg.PREFETCH and rt.STATE["running"]))       # [2026-10-02] intermediate waypoint: pass through, last one: full reach test
        if cfg.PREFETCH and k == wp_indices[max(0, len(wp_indices) - 1 - _lead)] and rt.STATE["running"] and rt._PF["thread"] is None:
            rt._PF["delay"] = len(wp_indices) - wp_indices.index(k)      # this waypoint + the rest of the cycle run during the inference
            _prefetch_start()
        if base is not None:
            r = _goto(list(a), (L["tgt_pos"][k], L["tgt_quat"][k]), ikerr, motion_base=base, pass_through=_pt)
        else:
            r = _goto(list(a), (L["tgt_pos"][k], L["tgt_quat"][k]), ikerr, pass_through=_pt)
        r["pass_through"] = _pt
        r["ik_ok"] = ok
        r["dq_clipped"] = clipped
        r["ik_ms"] = round(ik_ms, 1)
        if k == 0:
            T["ik_ms"] = round(ik_ms, 1)
            T["arm_latency_ms"] = r.get("arm_latency_ms")
            T["gripper_latency_ms"] = r.get("gripper_latency_ms")
        if isinstance(r, dict) and r.get("error"):
            return {"error": r["error"], "sent": sent}
        if r.get("stopped"):
            break
        sent += 1
        o2 = _observe()
        m_now = rt.INF.tcp_now(o2["joints14"])[0] if o2 else None
        # [2026-09-29] per-cycle grasp diagnostic (arm vs gripper horizon): full predicted g[1:16], the selected g[k], what was
        # sent, TCP now / IK target / after, IK residual, q now / q command (follower frame), limit excess. -> V4_CYCLE_LOG
        try:
            q_cmd_f = np.asarray(a, np.float64).copy(); q_cmd_f[FLIP_IDX] *= -1.0          # leader -> follower frame
            wp = getattr(rt.INF, "last_wp", {}) or {}
            rec_c = dict(t=round(time.time(), 3), cycle=rt.STATE["cycles"], ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")),
                         state_mode=os.environ.get("V4_STATE_MODE"), exec_k=int(k) + 1,
                         frame_age_at_infer_ms=age_infer, frame_age_at_cmd_ms=_age(t_cmd_k), infer_ms=T.get("infer_ms"),
                         prefetched=T.get("prefetched"), prefetch_shift=T.get("prefetch_shift"), prefetch_age_ms=T.get("prefetch_age_ms"), pass_through=r.get("pass_through"),
                         g_pred=[[round(float(x), 3) for x in L["act"][:, r_ * 10 + 9]] for r_ in (0, 1)],
                         g_sel=[round(float(L["act"][k, r_ * 10 + 9]), 3) for r_ in (0, 1)],
                         grip_cmd_sent=[round(float(a[g_]), 1) for g_ in GRIP_IDX],
                         tcp_now_mm=[[round(float(x) * 1000, 1) for x in m_before[r_][:3, 3]] for r_ in (0, 1)],
                         tcp_target_mm=[[round(float(x) * 1000, 1) for x in L["tgt_pos"][k, r_]] for r_ in (0, 1)],
                         tcp_after_mm=None if m_now is None else [[round(float(x) * 1000, 1) for x in m_now[r_][:3, 3]] for r_ in (0, 1)],
                         ik_residual_mm=round(float(ikerr), 2), ik_ok=bool(ok), reached=r.get("reached"), tcp_err_mm=r.get("tcp_err_mm"),
                         q_now=[round(float(x), 4) for x in np.asarray(o["joints14"], np.float64)[ARM_IDX]],
                         q_cmd=[round(float(x), 4) for x in q_cmd_f[ARM_IDX]],
                         ik_backend_executed=getattr(rt.INF, "_ik_used", None), lik_stop_reason=getattr(rt.INF, "_lik_stop", None), shadow_numerical=getattr(rt.INF, "_shadow", None), rel_input_diff_mm=getattr(rt.INF, "_rel_diff_mm", None),
                         grip_relative=getattr(rt.INF, "_grip_rel_log", None),
                         temporal_ensemble=getattr(rt.INF, "te_info", None), deadband=getattr(rt.INF, "db_info", None), adaptive_k=getattr(rt.INF, "ak_info", None),
                         anchor_mm=None if getattr(rt.INF, "_anchor", None) is None else [[round(float(x) * 1000, 1) for x in rt.INF._anchor[r_][:3, 3]] for r_ in (0, 1)],
                         limit_excess=wp.get("limit_excess"), grip_width_now_mm=[round(float(x) * 1000, 1) for x in (L["state"][[36 + 1, 74 + 1]] if len(L["state"]) == 76 else L["state"][18:20] * INF_W_OPEN)])
            with open(cfg.CYCLE_LOG, "a") as f_:
                f_.write(__import__("json").dumps(rec_c) + "\n")
        except Exception as e_:  # noqa: BLE001 -- diagnostics must never break a waypoint
            print("[cycle-log] skipped:", e_, flush=True)
        for r_ in (0, 1):
            raw_body = L["act"][k, r_ * 10:r_ * 10 + 3]                       # model output, body frame, m
            # UMI current-anchor puts EVERY step of the chunk on the same measured T_now, so the body->world
            # rotation is m_before for all k, not only the first. This used to be computed at k == 0 only,
            # which left pred_world null for the whole run whenever exec_k > 1 (execution starts at wp
            # exec_k-1) -- and pred_world is the only field comparable with cmd_world, so the one check the
            # comment above calls for ("a sign that flips between pred and cmd is a compose bug") could not
            # be made. pred_body is in the body frame and must never be compared with cmd_world directly.
            R_cur = m_before[r_][:3, :3]
            pred_world = (R_cur @ raw_body) * 1000.0
            cmd_world = (L["tgt_pos"][k, r_] - prev_tgt[r_]) * 1000.0
            act_world = ((m_now[r_][:3, 3] - prev_act[r_]) * 1000.0) if m_now is not None else None
            axes[r_].append({
                "wp": k,
                "tcp_xyz_mm": [round(float(x) * 1000, 1) for x in prev_act[r_]],
                "pred_body_dxyz_mm": [round(float(x) * 1000, 2) for x in raw_body],
                "pred_world_dxyz_mm": None if pred_world is None else [round(float(x), 2) for x in pred_world],
                "cmd_world_dxyz_mm": [round(float(x), 2) for x in cmd_world],
                "actual_world_dxyz_mm": None if act_world is None else [round(float(x), 2) for x in act_world]})
            prev_tgt[r_] = L["tgt_pos"][k, r_].copy()
            if m_now is not None:
                prev_act[r_] = m_now[r_][:3, 3].copy()
        track.append({k_: r.get(k_) for k_ in ("repeats", "dwell_ms", "joint_err_deg", "tcp_err_mm",
                                               "ik_fk_err_mm", "tcp_tol_mm", "reached", "timeout", "cause",
                                               "ik_ok", "dq_clipped")})
    time.sleep(cfg.CYCLE_SETTLE_S)     # [2026-10-06] was a fixed 0.2 s (only for the m_after travel telemetry); HRA loader sets 0
    h2 = _observe()
    m_after, _ = (rt.INF.tcp_now(h2["joints14"]) if h2 else (None, None))

    rows = []
    for r, nm in ((0, "left"), (1, "right")):
        k = sent - 1 if sent else 0
        want = np.linalg.inv(m_before[r]) @ np.array(
            np.vstack([np.hstack([Rot.from_quat(L["tgt_quat"][k, r]).as_matrix(), L["tgt_pos"][k, r][:, None]]),
                       [0, 0, 0, 1]]))
        got = np.linalg.inv(m_before[r]) @ m_after[r] if m_after is not None else np.eye(4)
        cos = float(np.dot(want[:3, 3], got[:3, 3]) /
                    max(np.linalg.norm(want[:3, 3]) * np.linalg.norm(got[:3, 3]), 1e-12))
        rows.append({
            "arm": nm,
            "pred_mm": round(float(L["pred_mm"][:sent or 1, r].sum()), 3),
            "pred_deg": round(float(L["pred_deg"][:sent or 1, r].sum()), 3),
            "cmd_mm": round(float(np.linalg.norm(want[:3, 3]) * 1000), 3),
            "cmd_deg": round(float(np.degrees(np.linalg.norm(Rot.from_matrix(want[:3, :3]).as_rotvec()))), 3),
            "actual_mm": round(float(np.linalg.norm(got[:3, 3]) * 1000), 3),
            "actual_deg": round(float(np.degrees(np.linalg.norm(Rot.from_matrix(got[:3, :3]).as_rotvec()))), 3),
            "dir_cos": round(cos, 3),
            "fkerr_mm": round(float(L["fkerr_mm"][:sent or 1, r].max()), 2),
        })

    rt.STATE["presses"] += 1
    for i, row in enumerate(rows):
        rt.STATE["travel_mm"][i] += row["actual_mm"]
    tel = {"infer_s": round(infer_s, 2), "sent": sent, "kept": int(L["kept"]), "valid": int(L["valid_step"]),
           "clamped_steps": int(L["clamped_steps"]), "dq_clipped": int(L["clipped"]),
           "arm": rt.STATE["arm"], "presses": rt.STATE["presses"], "rows": rows,
           "state20": [round(float(x), 4) for x in L["state"]],
           "gripper": L["gripper_mode"],
           "latency": {**T,
                       "total_ms": round(sum(v for v in (T.get("observe_ms"), T.get("infer_ms"),
                                                         T.get("ik_ms"), T.get("arm_latency_ms"))
                                             if v is not None), 1),
                       "step_ms": round(cfg.DT_STEP_MS, 1)},
           "skip_steps_would_be": round(sum(v for v in (T.get("observe_ms"), T.get("infer_ms"),
                                                        T.get("ik_ms"), T.get("arm_latency_ms"))
                                            if v is not None) / cfg.DT_STEP_MS, 2),
           "tracking": track,
           "axes": {"left": axes[0], "right": axes[1]},
           "reached": sum(1 for t in track if t.get("reached")),
           "timeouts": sum(1 for t in track if t.get("timeout")),
           "grip_width_mm": [round(float(L["widths"][:sent or 1, r].mean()) * 1000, 1) for r in (0, 1)],
           "grip_cmd": [round(float(L["grip_cmd"][:sent or 1, r].mean()), 1) for r in (0, 1)],
           "clamp": f"{LIMITS['mm']} mm / {LIMITS['deg']} deg per step"}
    rt.STATE["last"] = tel
    return tel


def _loop():
    """Continuous rollout: same one-press cycle, with run-level budgets on top of the per-step clamp."""
    rt.STATE["cycles"] = 0; rt.STATE["travel_mm"] = [0.0, 0.0]; rt.STATE["stop_reason"] = None
    _jaw_torque(True)          # a run can start straight after a rest, which left the jaws unpowered
    while rt.STATE["running"]:
        with rt._lock:
            tel = _one_press()
        rt.STATE["cycles"] += 1
        if isinstance(tel, dict) and tel.get("error"):
            rt.STATE["stop_reason"] = f"error: {tel['error']}"; break
        if cfg.MAX_CYCLES and rt.STATE["cycles"] >= cfg.MAX_CYCLES:
            rt.STATE["stop_reason"] = f"cycle budget {cfg.MAX_CYCLES} reached"; break
        if cfg.MAX_TRAVEL_MM and max(rt.STATE["travel_mm"]) >= cfg.MAX_TRAVEL_MM:
            rt.STATE["stop_reason"] = f"travel budget {cfg.MAX_TRAVEL_MM} mm reached"; break
        if tel.get("valid", 0) < 1:
            rt.STATE["stop_reason"] = "IK could not reach the first predicted pose"; break
        hs = _hra_stop()
        if hs:
            rt.STATE["stop_reason"] = hs; break
    rt.STATE["running"] = False
    if rt.STATE["stop_reason"] is None:
        rt.STATE["stop_reason"] = "stopped by user"
    print(f"[v4-smoke] loop ended: {rt.STATE['stop_reason']} after {rt.STATE['cycles']} cycles, "
          f"travel {[round(x,1) for x in rt.STATE['travel_mm']]} mm", flush=True)
