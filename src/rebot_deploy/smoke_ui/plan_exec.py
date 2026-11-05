"""PLAN executor (2026-10-06): always one forward in flight, chunk smoothing / cross-fade, and fixed-tick playback.
"""
import collections
import json
import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import settings as cfg
from . import state as rt
from .ic import ARM_IDX, FLIP_IDX, GRIP_IDX, IC as _IC, UMI_DT
from .robot_client import _jaw_torque, _joints14_fast, _observe, _robot
from .safety import _hra_stop


def _smooth_chunk(P, Q, cur):
    """P (T,2,3), Q (T,2,4 xyzw), cur (2,4,4) -> smoothed copies (both arms)."""
    T = len(P); deg = min(cfg.PLAN_SMOOTH_DEG, T - 1)
    if deg < 1 or T < 3:
        return P, Q
    P = np.asarray(P, float).copy(); Q = np.asarray(Q, float).copy()
    k = np.arange(0, T + 1, dtype=float); w = np.ones(T + 1); w[0] = 100.0      # k = 0: the current pose, pinned
    for r in range(2):
        Y = np.vstack([cur[r][:3, 3][None], P[:, r]])
        for ax in range(3):
            c = np.polyfit(k, Y[:, ax], deg, w=w); P[:, r, ax] = np.polyval(c, k[1:])
        R0 = cur[r][:3, :3]; rv = np.vstack([np.zeros(3)[None], Rot.from_matrix(np.einsum("ji,tjk->tik", R0, Rot.from_quat(Q[:, r]).as_matrix())).as_rotvec()])
        dr = min(2, deg)
        for ax in range(3):
            c = np.polyfit(k, rv[:, ax], dr, w=w); rv[1:, ax] = np.polyval(c, k[1:])
        Q[:, r] = Rot.from_matrix(np.einsum("ij,tjk->tik", R0, Rot.from_rotvec(rv[1:]).as_matrix())).as_quat()
    return P, Q


def _jag(P):
    v = np.diff(P, axis=0); a = np.diff(v, axis=0)
    return round(float(np.linalg.norm(a, axis=1).mean() / max(np.linalg.norm(v, axis=1).mean(), 1e-9)), 3) if len(P) >= 3 else None      # [10-06] per-tick measured right TCP + per-chunk targets/seams in PLAN_LOG


def _plan_leader(q14):
    a = list(map(float, q14))
    for i in FLIP_IDX:
        a[i] = -a[i]
    for g in GRIP_IDX:
        a[g] = float(_IC.jawHoldCmd(q14[g])) if hasattr(_IC, "jawHoldCmd") else float(np.clip(_IC.unwrapGrip(q14[g]) / -6.0, 0, 45))
    return a


def _plan_forward_job(job, out):
    try:
        t_req = time.time()
        h0 = _observe(); time.sleep(cfg.OBS_GAP_S); h1 = _observe()
        if h0 is None or h1 is None:
            out["res"] = {"error": "observe failed"}; return
        t_obs = time.time()
        rt.INF._rtc_plan_prefix = job["prefix"] if _IC.RTC_ON else None; rt.INF._rtc_delay_hint = job["d"]
        try:
            rt.INF.infer([h0, h1])
        finally:
            rt.INF._rtc_plan_prefix = None; rt.INF._rtc_delay_hint = None
        L_ = rt.INF.last; T = min(int(L_["valid_step"]), len(L_["tgt_pos"]))
        seed = list(map(float, h1["joints14"])); rows = {}; anchor = job["anchor"]; ik_bad = None
        P_all = np.asarray(L_["tgt_pos"][:T], float); Q_all = np.asarray(L_["tgt_quat"][:T], float); jag_raw = _jag(P_all[:, 1]); jag_sm = None
        if cfg.PLAN_SMOOTH_DEG > 0 and L_.get("cur_mat") is not None:
            P_all, Q_all = _smooth_chunk(P_all, Q_all, L_["cur_mat"]); jag_sm = _jag(P_all[:, 1])
        # [2026-10-06 seam fix] compile from the BOUNDARY row (row F, the command the new rows continue from), not from the measured
        # joints of the observation: rows <= F are discarded anyway, and an IK chain seeded at the observation can pick a different
        # wrist posture for the same TCP -> a joint jump at the seam. seam_jump_deg = first new row vs the boundary row.
        if job.get("boundary") is not None:
            seed = list(job["boundary"])
            for i in FLIP_IDX:
                seed[i] = -seed[i]
            seed[GRIP_IDX[0]] = float(h1["joints14"][GRIP_IDX[0]]); seed[GRIP_IDX[1]] = float(h1["joints14"][GRIP_IDX[1]])
        seam_jump = None
        for j in range(T):
            if anchor + j + 1 <= job["F"] and job.get("boundary") is not None:
                continue
            P = P_all[j].copy(); Q = Q_all[j].copy(); W = np.asarray(L_["widths"][j]).copy()
            a, ikerr, ok, clipped = rt.INF.solve_waypoint(seed, P, Q, None)
            if not ok or getattr(rt.INF, "_lik_stop", None):
                ik_bad = f"IK failed at step {j + 1}"; break
            a = list(map(float, a))
            if job.get("left_hold") is not None:
                a[0:7] = job["left_hold"]
            r_no = anchor + j + 1
            if 0.0 < cfg.PLAN_EMA < 1.0 and r_no > job["F"] and job.get("boundary") is not None:
                prev = job["boundary"]
                for i_ in ARM_IDX:
                    a[i_] = prev[i_] + cfg.PLAN_EMA * (a[i_] - prev[i_])
            if r_no > job["F"]:
                if seam_jump is None and job.get("boundary") is not None:
                    seam_jump = round(float(np.degrees(np.abs(np.asarray(a)[ARM_IDX] - np.asarray(job["boundary"])[ARM_IDX]).max())), 2)
                job["boundary"] = list(a)
            rows[r_no] = dict(action=a, P=P, Q=Q, W=W, ikerr=float(ikerr), clipped=int(clipped))
            nxt = list(rows[r_no]["action"]) if False else list(a)
            for i in FLIP_IDX:
                nxt[i] = -nxt[i]
            nxt[GRIP_IDX[0]] = seed[GRIP_IDX[0]]; nxt[GRIP_IDX[1]] = seed[GRIP_IDX[1]]; seed = nxt
        out["res"] = dict(anchor=anchor, rows=rows, t_req=t_req, t_obs=t_obs, t_done=time.time(), ik_bad=ik_bad,
                          rtc=(rt.INF.last or {}).get("rtc"), infer_ms=round((time.time() - t_obs) * 1000, 1), jag_raw=jag_raw, jag_sm=jag_sm,
                          seam_jump_deg=seam_jump)
    except Exception as e:  # noqa: BLE001
        out["res"] = {"error": f"forward failed: {e}"}


def _plan_loop():
    rt.STATE["cycles"] = 0; rt.STATE["travel_mm"] = [0.0, 0.0]; rt.STATE["stop_reason"] = None
    _jaw_torque(True)
    stats = {"chunks": 0, "handed": 0, "dry_ticks": 0, "lat_s": collections.deque(maxlen=50), "pace": None, "last": None}
    rt.STATE["plan"] = stats
    o = _observe()
    if o is None:
        rt.STATE["stop_reason"] = "observe failed"; rt.STATE["running"] = False; return
    left_hold = _plan_leader(o["joints14"])[0:7] if _IC.ARM_ONLY == "right" else None
    sid = int(time.time() * 1000) % 1000000000
    r = _robot("/plan/open", "post", json={"session": sid})
    if r.get("error"):
        rt.STATE["stop_reason"] = f"plan open: {r['error']}"; rt.STATE["running"] = False; return
    plan = {}; plan_end = 0; F = None; fw = {"th": None, "res": None, "job": None, "t0": None}; lat_ema = None
    base = max(0.05, cfg.STREAM_SPEED); pace_sent = 1.0; rows_prev = None; last_cut = 0

    def contiguous_end(c):
        e = c
        while (e + 1) in plan:
            e += 1
        return e

    def log(**kw):
        try:
            with open(cfg.PLAN_LOG, "a") as f_:
                f_.write(json.dumps(dict(t=round(time.time(), 3), session=sid, **kw)) + "\n")
        except Exception:
            pass
    try:
        while rt.STATE["running"]:
            tick = time.time()
            hs = _hra_stop()
            if hs:
                rt.STATE["stop_reason"] = hs; break
            prog = _robot("/plan/progress")
            if prog.get("error") or prog.get("session") != sid:
                rt.STATE["stop_reason"] = f"plan session gone: {prog.get('error') or prog.get('glitch') or 'closed'}"; break
            C = int(prog["last_row"])
            if cfg.PLAN_TRACE:
                q14_ = _joints14_fast()
                if q14_ is not None:
                    try:
                        m_, _ = rt.INF.tcp_now(q14_)
                        log(kind="tick", exec_row=int(prog["exec_row"]), tcp_mm=[round(float(x) * 1000, 1) for x in m_[1][:3, 3]], pace=round(pace_sent, 3))
                    except Exception:
                        pass
            # 1. install a finished forward: replace only the rows after max(C, F)
            if fw["th"] is not None and not fw["th"].is_alive():
                res = fw["res"]; job = fw["job"]; fw["th"] = None
                if not res or res.get("error"):
                    rt.STATE["stop_reason"] = (res or {}).get("error", "forward lost"); break
                cut = max(C, job["F"])
                old = {r_: plan[r_] for r_ in plan if r_ > cut}
                seam_mm = [round(float(np.linalg.norm(np.asarray(res["rows"][r_]["P"])[1] - np.asarray(old[r_]["P"])[1]) * 1000), 1)
                           for r_ in sorted(old)[:6] if r_ in res["rows"]]
                tgt_mm = {int(r_): [round(float(x) * 1000, 1) for x in np.asarray(v["P"])[1]] for r_, v in res["rows"].items()}
                for r_ in old:
                    del plan[r_]
                for r_, v in res["rows"].items():
                    if r_ > cut:
                        i_ = r_ - cut - 1
                        if cfg.PLAN_XFADE > 0 and i_ < cfg.PLAN_XFADE and r_ in old:
                            w_ = (i_ + 1) / (cfg.PLAN_XFADE + 1); a_ = list(v["action"]); o_ = old[r_]["action"]
                            for k_ in ARM_IDX:
                                a_[k_] = (1 - w_) * o_[k_] + w_ * a_[k_]
                            v = dict(v, action=a_)
                        plan[r_] = v
                last_cut = cut
                plan_end = contiguous_end(C)
                lat = time.time() - res["t_req"]; lat_ema = lat if lat_ema is None else 0.7 * lat_ema + 0.3 * lat
                stats["chunks"] += 1; stats["lat_s"].append(round(lat, 3)); rt.STATE["cycles"] = stats["chunks"]
                stats["last"] = dict(anchor=job["anchor"], F=job["F"], C=C, cut=cut, new_rows=sum(1 for r_ in res["rows"] if r_ > cut),
                                     jag_raw=res.get("jag_raw"), jag_sm=res.get("jag_sm"), seam_jump_deg=res.get("seam_jump_deg"),
                                     plan_end=plan_end, lat_s=round(lat, 3), infer_ms=res.get("infer_ms"), ik_bad=res.get("ik_bad"))
                log(kind="chunk", **stats["last"], seam_mm=seam_mm, tgt_mm=tgt_mm, rtc=res.get("rtc") and {k: res["rtc"].get(k) for k in ("delay", "horizon", "leftover", "source")},
                    pace=round(pace_sent, 3), lat_ema=round(lat_ema, 3))
                F = None
                if plan_end <= C and not res["rows"]:
                    rt.STATE["stop_reason"] = res.get("ik_bad") or "empty chunk"; break
            # 2. hand rows: keep ~PLAN_WINDOW_S queued on the bridge; never past F while a forward runs
            limit = F if (fw["th"] is not None and F is not None) else plan_end
            rem = float(prog["remaining_s"]); batch = []; nxt = C + 1
            while rem < cfg.PLAN_WINDOW_S and nxt <= limit and nxt in plan:
                batch.append(dict(row=nxt, action=plan[nxt]["action"], dur_s=UMI_DT)); rem += UMI_DT / max(pace_sent, 1e-3); nxt += 1
            if batch:
                rr = _robot("/plan/rows", "post", json={"session": sid, "rows": batch})
                if rr.get("error"):
                    rt.STATE["stop_reason"] = f"plan rows: {rr['error']}"; break
                stats["handed"] += len(batch); C = nxt - 1
            for r_ in [r_ for r_ in plan if r_ < int(prog["exec_row"]) - 1]:
                del plan[r_]
            # 3. always one forward: anchor = the row the playback heads to now; reserve the rows that run meanwhile
            ex_ = int(prog["exec_row"])
            low = (plan_end - ex_) * UMI_DT / max(pace_sent, 1e-3) < ((lat_ema or 1.0) + cfg.PLAN_BRAKE_LEAD_S) * 1.3
            if fw["th"] is None and (stats["chunks"] == 0 or ex_ - last_cut >= cfg.PLAN_MIN_EXEC or low):
                anchor = int(prog["exec_row"]); L = lat_ema or 1.0
                n_res = int(math.ceil((L + cfg.PLAN_BRAKE_LEAD_S) * max(pace_sent, 0.05) / UMI_DT)) + 1
                F = max(C, min(plan_end, anchor + n_res)) if plan_end > anchor else C
                pre = [(plan[r_]["P"], plan[r_]["Q"], plan[r_]["W"]) for r_ in range(anchor + 1, plan_end + 1) if r_ in plan][:16]
                job = dict(anchor=anchor, F=F, d=max(0, F - anchor), prefix=pre or None, left_hold=left_hold,
                           boundary=list(plan[F]["action"]) if F in plan else None)
                fw.update(res=None, job=job, t0=time.time())
                fw["th"] = threading.Thread(target=_plan_forward_job, args=(job, fw), daemon=True); fw["th"].start()
            # 4. pace: the reserved rows must last until the result arrives (+ brake lead)
            p = base
            if fw["th"] is not None and F is not None and lat_ema is not None:
                avail_1x = float(prog["remaining_1x_s"]) + max(0, F - C) * UMI_DT
                l_rem = max(0.05, lat_ema - (time.time() - fw["t0"]))
                p = float(np.clip(avail_1x / (l_rem + cfg.PLAN_BRAKE_LEAD_S), 0.1, base))
            if abs(p - pace_sent) / max(pace_sent, 1e-3) > 0.05:
                _robot("/plan/pace", "post", json={"session": sid, "pace": p}); pace_sent = p
            stats["pace"] = round(pace_sent, 3)
            if prog.get("dry") and stats["chunks"] > 0:
                stats["dry_ticks"] += 1
            sl = cfg.PLAN_TICK_S - (time.time() - tick)
            if sl > 0:
                time.sleep(sl)
    except Exception as e:  # noqa: BLE001
        rt.STATE["stop_reason"] = f"plan loop: {e}"
    finally:
        _robot("/plan/close", "post", json={"session": sid})
        rt.STATE["running"] = False
        if rt.STATE["stop_reason"] is None:
            rt.STATE["stop_reason"] = "stopped by user"
        log(kind="end", reason=rt.STATE["stop_reason"], chunks=stats["chunks"], handed=stats["handed"], dry_ticks=stats["dry_ticks"],
            lat_p50=float(np.median(stats["lat_s"])) if stats["lat_s"] else None)
        print(f"[v4-smoke] PLAN loop ended: {rt.STATE['stop_reason']} after {stats['chunks']} chunks, {stats['handed']} rows, "
              f"dry ticks {stats['dry_ticks']}", flush=True)
