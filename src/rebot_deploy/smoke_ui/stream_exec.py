"""STREAM executor (2026-10-01): a background inference worker merges every new chunk into a timed joint timeline
that the stream loop sends at a fixed rate.
"""
import collections
import json
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import settings as cfg
from . import state as rt
from .ic import ARM_IDX, GRIP_IDX, UMI_DT
from .motion import _exec_step
from .robot_client import _jaw_torque, _joints14_fast, _observe
from .safety import _hra_stop


def _stream_merge(tl, rows, now):
    """merge_chunk: rows = [(tau, pos(2,3), quat(2,4), width(2), chunk_id, j)] into the pending timeline tl (list, sorted)."""
    half = 0.5 * (UMI_DT / cfg.STREAM_SPEED)
    kept = [r for r in tl if r[0] >= now - half]
    dropped_old = len(tl) - len(kept); dropped_new = 0; blended = 0
    for (tau, P, Q, W, cid, j) in rows:
        if tau < now - half:
            dropped_new += 1; continue
        hit = next((i for i, r in enumerate(kept) if abs(r[0] - tau) <= half), None)
        if hit is None:
            kept.append((tau, P, Q, W, cid, j))
        else:
            o = kept[hit]; w = cfg.STREAM_BLEND
            q = np.stack([Rot.from_quat(np.stack([o[2][r], Q[r]])).mean(weights=[1 - w, w]).as_quat() for r in (0, 1)])
            kept[hit] = (o[0], (1 - w) * o[1] + w * P, q, (1 - w) * o[3] + w * W, cid, j); blended += 1
    kept.sort(key=lambda r: r[0])
    return kept, dict(dropped_old=dropped_old, dropped_new=dropped_new, blended=blended)


def _stream_infer_worker(TL, stats):
    cid = 0; lat_ema = None
    while rt.STATE["running"]:
        if cfg.STREAM_SYNC and cid > 0:
            while rt.STATE["running"]:
                with TL["lock"]:
                    empty = not TL["rows"]
                if empty and time.time() - TL.get("last_sent_t", 0.0) >= cfg.STREAM_SETTLE_S + UMI_DT / cfg.STREAM_SPEED:
                    break
                time.sleep(0.01)
            if not rt.STATE["running"]:
                break
        elif cfg.STREAM_SCHED and cid > 0:
            # wait until the timeline left to play is about one forward long
            while rt.STATE["running"]:
                with TL["lock"]:
                    left = (TL["rows"][-1][0] - time.time()) if TL["rows"] else 0.0
                if left <= (lat_ema or 1.0) + cfg.STREAM_MARGIN_S:
                    break
                time.sleep(0.02)
            if not rt.STATE["running"]:
                break
        t0 = time.time()
        h0 = _observe()
        if h0 is None:
            stats["obs_fail"] += 1; time.sleep(0.05); continue
        time.sleep(cfg.OBS_GAP_S)
        h1 = _observe()
        if h1 is None:
            stats["obs_fail"] += 1; time.sleep(0.05); continue
        t_obs = time.time()
        try:
            rt.INF.infer([h0, h1])
        except Exception as e:  # noqa: BLE001
            rt.STATE["stop_reason"] = f"inference failed: {e}"; rt.STATE["running"] = False; break
        L_ = rt.INF.last; T = min(int(L_["valid_step"]), len(L_["tgt_pos"])); dt = UMI_DT / cfg.STREAM_SPEED
        if cfg.STREAM_SYNC:
            t_start = time.time()
            rows = [(t_start + j * dt, L_["tgt_pos"][j].copy(), L_["tgt_quat"][j].copy(), L_["widths"][j].copy(), cid, j) for j in range(min(T, cfg.STREAM_ROWS))]
        else:
            rows = [(t_obs + (j + 1) * dt, L_["tgt_pos"][j].copy(), L_["tgt_quat"][j].copy(), L_["widths"][j].copy(), cid, j) for j in range(min(T, cfg.STREAM_K) if cfg.STREAM_K > 0 else T)]
        now = time.time()
        with TL["lock"]:
            TL["rows"], m = _stream_merge(TL["rows"], rows, now)
        cid += 1; stats["chunks"] = cid
        lat = now - t0; stats["lat_s"].append(lat); lat_ema = lat if lat_ema is None else 0.7 * lat_ema + 0.3 * lat
        try:
            with open(cfg.STREAM_LOG, "a") as f:
                f.write(json.dumps(dict(t=round(now, 3), kind="chunk", chunk=cid, latency_s=round(lat, 3), infer_ms=round((now - t_obs) * 1000, 1),
                                        speed=cfg.STREAM_SPEED, pending=len(TL["rows"]), lat_ema_s=round(lat_ema, 3), sched=cfg.STREAM_SCHED,
                                        # [2026-10-02 user: gripper never closes] the chunk's full predicted gripper g, k1..k16, per arm
                                        g16=[[round(float(x), 3) for x in np.asarray(L_["widths"])[:T, r_]] for r_ in (0, 1)], stream_k=cfg.STREAM_K, **m)) + "\n")
        except Exception:
            pass


def _stream_loop():
    """Executor: one row per DT_EXEC from the merged timeline, IK from the measured joints, no arrival wait."""
    rt.STATE["cycles"] = 0; rt.STATE["travel_mm"] = [0.0, 0.0]; rt.STATE["stop_reason"] = None
    _jaw_torque(True)
    TL = {"rows": [], "lock": threading.Lock()}
    stats = {"chunks": 0, "sent": 0, "dry": 0, "obs_fail": 0, "lat_s": collections.deque(maxlen=50)}
    rt.STATE["stream"] = stats
    threading.Thread(target=_stream_infer_worker, args=(TL, stats), daemon=True).start()
    dt = UMI_DT / cfg.STREAM_SPEED; nxt = time.time(); last_cid = None
    while rt.STATE["running"]:
        now = time.time()
        hs = _hra_stop()
        if hs:
            rt.STATE["stop_reason"] = hs; break
        with TL["lock"]:
            if not cfg.STREAM_SYNC:          # sync chunks start at the current pose: a late row is still executed, in order
                TL["rows"] = [r for r in TL["rows"] if r[0] >= now - 0.5 * dt]
            row = TL["rows"].pop(0) if TL["rows"] and TL["rows"][0][0] <= now + 0.5 * dt else None
            if row is not None and cfg.STREAM_PAUSE_S > 0 and not cfg.STREAM_SYNC and last_cid is not None and row[4] != last_cid:
                # chunk boundary: put the row back, shift everything pending by the pause, hold
                TL["rows"].insert(0, row)
                TL["rows"] = [(r[0] + cfg.STREAM_PAUSE_S,) + tuple(r[1:]) for r in TL["rows"]]
                last_cid = row[4]; row = "PAUSE"
        if row == "PAUSE":
            stats["pauses"] = stats.get("pauses", 0) + 1
            time.sleep(cfg.STREAM_PAUSE_S); nxt = time.time(); continue
        if row is not None:
            last_cid = row[4]
        if row is None:
            if stats["chunks"] > 0:
                stats["dry"] += 1
        else:
            tau, P, Q, W, cid, j = row
            q14 = _joints14_fast()
            if q14 is None:
                stats["obs_fail"] += 1
            else:
                try:
                    a, ikerr, ok, clipped = rt.INF.solve_waypoint(q14, P, Q, W if rt.INF.last.get("gripper_mode") in ("predict", "binary", "continuous", "relative") else None)
                except Exception as e:  # noqa: BLE001
                    rt.STATE["stop_reason"] = f"IK failed: {e}"; break
                if getattr(rt.INF, "_lik_stop", None):
                    rt.STATE["stop_reason"] = f"IK stop: {rt.INF._lik_stop}"; break
                r = _exec_step(a)
                if isinstance(r, dict) and r.get("error"):
                    rt.STATE["stop_reason"] = f"execute_step: {r['error']}"; break
                stats["sent"] += 1; rt.STATE["cycles"] = stats["sent"]; TL["last_sent_t"] = time.time()
                if cfg.STREAM_WP_PAUSE_S > 0:
                    with TL["lock"]:
                        TL["rows"] = [(r[0] + cfg.STREAM_WP_PAUSE_S,) + tuple(r[1:]) for r in TL["rows"]]
                    nxt += cfg.STREAM_WP_PAUSE_S
                try:
                    m_now, _ = rt.INF.tcp_now(q14)
                    with open(cfg.CYCLE_LOG, "a") as f:
                        f.write(json.dumps(dict(t=round(now, 3), cycle=stats["sent"], stream_chunk=cid, stream_row=j + 1, late_ms=round((now - tau) * 1000, 1),
                                                ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")), exec_k=j + 1, stream=True,
                                                tcp_now_mm=[[round(float(x) * 1000, 1) for x in m_now[r_][:3, 3]] for r_ in (0, 1)],
                                                tcp_target_mm=[[round(float(x) * 1000, 1) for x in P[r_]] for r_ in (0, 1)], tcp_after_mm=None,
                                                ik_residual_mm=round(float(ikerr), 2), ik_ok=bool(ok), reached=None, yaw=getattr(rt.INF, "_yaw_diag", None),
                                                q_now=[round(float(x), 4) for x in np.asarray(q14)[ARM_IDX]],
                                                # [2026-10-02 user: right jaw opens too wide] model g of this row (0 closed..1 open), the jaw command
                                                # sent (UI cmd units, 45 = full open), and the measured jaw raw deg (bridge-clipped to [-270, 0])
                                                g_row=None if W is None else [round(float(x), 3) for x in np.asarray(W).ravel()[:2]],
                                                grip_cmd_sent=[round(float(a[g_]), 1) for g_ in GRIP_IDX],
                                                jaw_raw_now=[round(float(q14[g_]), 1) for g_ in GRIP_IDX])) + "\n")
                except Exception:
                    pass
        nxt += dt; sl = nxt - time.time()
        if sl > 0:
            time.sleep(sl)
        else:
            nxt = time.time()
    rt.STATE["running"] = False
    if rt.STATE["stop_reason"] is None:
        rt.STATE["stop_reason"] = "stopped by user"
    lat = list(stats["lat_s"])
    print(f"[v4-stream] ended: {rt.STATE['stop_reason']} | chunks {stats['chunks']} sent {stats['sent']} dry {stats['dry']} "
          f"obs_fail {stats['obs_fail']} latency med {np.median(lat) if lat else float('nan'):.2f}s", flush=True)
