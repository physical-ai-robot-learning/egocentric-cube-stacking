"""Observation diagnostics: wrist snapshot, camera / frame checks, z probe, 2x2 replay, the observation dump
and the denoising-steps ablation.
"""
from ego_paths import DATA_ROOT, HOLOBRAIN_MAC  # machine-specific roots (env-overridable)

import base64
import os
import threading
import time

from flask import jsonify, request
import numpy as np
import torch

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import IC as _IC
from .robot_client import _observe


@app.route("/wrist_snap", methods=["POST"])
def wrist_snap():
    """[2026-10-07] save the raw right-wrist frame + the zoomed model input + the right TCP (measure cube size: train end vs robot stop)."""
    import json as _j, cv2
    o = _observe()
    if o is None or "right" not in o["images"]:
        return jsonify({"error": "observe failed / no right image"})
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(o["images"]["right"]), np.uint8), cv2.IMREAD_COLOR)
    z = cv2.resize(_IC.wristZoom(bgr, _IC.WRIST_ZOOM["s"]), (224, 224), interpolation=cv2.INTER_AREA)
    d = str(DATA_ROOT / "hra_a100/wrist_snaps"); os.makedirs(d, exist_ok=True); t = time.strftime("%H%M%S")
    cv2.imwrite(f"{d}/{t}_raw.png", bgr); cv2.imwrite(f"{d}/{t}_in224_s{_IC.WRIST_ZOOM['s']:.2f}.png", z)
    m, _ = rt.INF.tcp_now(o["joints14"])
    _j.dump({"t": t, "zoom": _IC.WRIST_ZOOM["s"], "T_base_tcp_right": m[1].tolist()}, open(f"{d}/{t}.json", "w"))
    ok, e = cv2.imencode(".jpg", z)
    return jsonify({"saved": f"{d}/{t}_raw.png", "zoom": _IC.WRIST_ZOOM["s"], "tcp_mm": [round(float(x) * 1000, 1) for x in m[1][:3, 3]],
                    "img": base64.b64encode(e.tobytes()).decode()})


@app.route("/camcheck", methods=["POST"])
def camcheck():
    """No motion. Does rotating the GLOBAL camera 180 deg change what the policy wants?

    B663 mixed two camera domains: the HEAD pool untouched and the FRONT (R675) pool with its global feed
    rotated 180 deg. The live robot has one orientation, and if it is read as the wrong domain the policy
    can be confidently wrong. This is the deterministic test for that, and it moves nothing.

    Both variants run on ONE frozen observation pair, and each draw uses the same seed in both variants --
    flow matching starts from noise, so without that the difference between A and B is mostly the sampler.
    Only the global (middle) image differs; wrists, state and task are byte-identical.

      A  live global as-is
      B  rot180(live global)

    Reported per arm and per k: the world-frame step vector and its magnitude, averaged over K draws, plus
    the spread across draws so a difference can be compared against the sampler's own noise. Magnitude
    alone does not decide anything -- a small output is natural on a still scene -- so the direction is
    reported next to it.
    """
    K = int(request.args.get("k", "4"))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    import cv2
    def rot180_b64(b64):
        a = np.frombuffer(base64.b64decode(b64), np.uint8)
        bgr = cv2.imdecode(a, cv2.IMREAD_COLOR)
        ok, enc = cv2.imencode(".jpg", cv2.rotate(bgr, cv2.ROTATE_180))
        if not ok:
            raise RuntimeError("re-encode failed")
        return base64.b64encode(enc.tobytes()).decode()

    variants = {"A_live": h1["images"],
                "B_rot180": {**h1["images"], "middle": rot180_b64(h1["images"]["middle"])}}
    KS_REQ = [1, 2, 4, 8, 16, 31]
    # valid_step is not constant across draws, so the k list is fixed ONCE from the first draw and reused
    # for every draw and both variants -- otherwise the per-draw arrays have different lengths and, worse,
    # A and B would be compared at different horizons.
    KS = KS_REQ
    out = {"draws": K, "variants": {}, "valid_step": {}}
    try:
        for name, imgs in variants.items():
            hh = [h0, {**h1, "images": imgs}]
            per_draw, vsteps = [], []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)          # same noise sequence in BOTH variants
                rt.INF.infer(hh)
                L = rt.INF.last
                vsteps.append(int(L["valid_step"]))
                m_now = L["cur_mat"]
                # valid_step is the first chunk step IK cannot reach within FK_MAX_MM, and it is NOT constant
                # across draws on the same observation. Steps past it are recorded as NaN rather than dropped,
                # so every draw keeps the same shape and the share of draws that could reach each k becomes a
                # reported number instead of a crash.
                blk = np.full((2, len(KS), 3), np.nan)
                for r in (0, 1):
                    for i, k in enumerate(KS):
                        if k < L["valid_step"]:
                            blk[r, i] = (L["tgt_pos"][k, r] - m_now[r][:3, 3]) * 1000.0
                per_draw.append(blk)
            D = np.stack(per_draw)                         # (K, 2, nk, 3), NaN past valid_step
            out["valid_step"][name] = {"min": min(vsteps), "max": max(vsteps),
                                       "median": int(np.median(vsteps)), "all": vsteps}
            arms = {}
            for r, side in ((0, "left"), (1, "right")):
                rows = []
                for i, k in enumerate(KS):
                    v = D[:, r, i, :]
                    ok = ~np.isnan(v).any(1)
                    if not ok.any():
                        rows.append({"k": k, "reachable_frac": 0.0}); continue
                    vv = v[ok]
                    mean = vv.mean(0)
                    rows.append({"k": k,
                                 "dxyz_mm": [round(float(x), 2) for x in mean],
                                 "norm_mm": round(float(np.linalg.norm(mean)), 2),
                                 "draw_spread_mm": round(float(np.linalg.norm(vv - mean, axis=1).mean()), 2),
                                 "reachable_frac": round(float(ok.mean()), 3)})
                arms[side] = rows
            out["variants"][name] = arms
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})

    # direction agreement between the two variants, per arm and k: cos ~ 1 means rotating the camera did
    # not change WHERE the policy wants to go, only how far.
    cmp = {}
    for side in ("left", "right"):
        rows = []
        for a, b in zip(out["variants"]["A_live"][side], out["variants"]["B_rot180"][side]):
            if "dxyz_mm" not in a or "dxyz_mm" not in b:
                rows.append({"k": a["k"], "unreachable": True}); continue
            va, vb = np.array(a["dxyz_mm"]), np.array(b["dxyz_mm"])
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            rows.append({"k": a["k"], "norm_A": a["norm_mm"], "norm_B": b["norm_mm"],
                         "ratio_B_over_A": round(float(nb / na), 3) if na > 1e-9 else None,
                         "cos_AB": round(float(va @ vb / (na * nb)), 4) if na > 1e-9 and nb > 1e-9 else None})
        cmp[side] = rows
    out["compare"] = cmp
    out["ks"] = KS
    return jsonify(out)


def _chunk_end(L, horizon=None):
    """Both arms' displacement at the LAST VALID chunk step, in mm, as a 6-vector.

    The rule, after 2026-09-25: every trajectory comparison stays inside each draw's `valid_step`, and a
    comparison ACROSS prompts or models uses the minimum valid_step of everything being compared.
    `tgt_pos[-1]` ignores it. DELTA32 290k truncates at 25 of 32 while REL32 runs the full 32, so reading
    index 31 for both fed the unreachable tail into the comparison and produced a "DELTA does not respect
    the prompt's first colour" verdict that a valid-horizon test then contradicted outright.
    Pass `horizon` to pin the common step; leave it None to use this draw's own last valid step.
    """
    k = (L["valid_step"] - 1) if horizon is None else min(horizon, L["valid_step"] - 1)
    k = max(0, int(k))
    return np.concatenate([L["tgt_pos"][k, r] - L["cur_tcp"][r][:3] for r in (0, 1)]) * 1000.0


@app.route("/framecheck", methods=["POST"])
def framecheck():
    """No motion. Is the chunk composed into the base frame with the right sign?

    The running telemetry shows pred_body (model output, BODY frame) next to cmd_world (the commanded step,
    BASE frame). Those are different frames, so a sign difference between them proves nothing -- a TCP that
    points back at the base legitimately flips two axes. The only honest check rotates the model output into
    the base frame first, and that is what this does:

        pred_world[k] = R_now @ act[k][:3]                 model output, rotated, NOT re-composed
        cmd_world[k]  = tgt_pos[k] - tcp_now                what the arm is actually asked to do

    Under UMI current-anchor every step hangs off the same measured T_now, so the two must agree to within
    rotation-representation error at every k. cos < 0 on a step with real magnitude is a compose/sign bug;
    cos ~ +1 with matching magnitude means the decode is right and a wrong direction on the robot is the
    policy or the frame adapter, not the composition.
    """
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    try:
        rt.INF.infer([h0, h1])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    L = rt.INF.last
    m_now = L["cur_mat"]
    out = {"action_mode": L.get("action_mode"), "valid_step": int(L["valid_step"]), "arms": {}}
    for r, side in ((0, "left"), (1, "right")):
        R_now = m_now[r][:3, :3]
        tcp0 = m_now[r][:3, 3]
        rows = []
        for k in (0, 3, 7, 15, 31):
            if k >= L["valid_step"]:
                continue
            pb = np.asarray(L["act"][k, r * 10:r * 10 + 3], dtype=float)
            pw = (R_now @ pb) * 1000.0
            cw = (np.asarray(L["tgt_pos"][k, r], dtype=float) - tcp0) * 1000.0
            npw, ncw = float(np.linalg.norm(pw)), float(np.linalg.norm(cw))
            cos = float(pw @ cw / (npw * ncw)) if npw > 1e-9 and ncw > 1e-9 else None
            rows.append({"k": k,
                         "pred_body_mm": [round(float(x) * 1000, 2) for x in pb],
                         "pred_world_mm": [round(float(x), 2) for x in pw],
                         "cmd_world_mm": [round(float(x), 2) for x in cw],
                         "norm_pred_mm": round(npw, 2), "norm_cmd_mm": round(ncw, 2),
                         "cos": None if cos is None else round(cos, 4)})
        w = np.asarray(L["widths"][:, r], dtype=float) * 1000.0      # mm, whole chunk
        out["arms"][side] = {"tcp_xyz_mm": [round(float(x) * 1000, 1) for x in tcp0], "steps": rows,
                             # the jaw channel over the WHOLE chunk. The dataset's width spans 0..114 mm
                             # (mean 29, std 40), so a chunk that never leaves single digits means the
                             # policy is holding the jaw shut and can never grasp -- a different failure
                             # from the arm going the wrong way, and invisible in the per-step telemetry
                             # which only shows the one executed step.
                             "width_mm": {"min": round(float(w.min()), 2), "max": round(float(w.max()), 2),
                                          "mean": round(float(w.mean()), 2),
                                          "first8": [round(float(x), 2) for x in w[:8]],
                                          "last8": [round(float(x), 2) for x in w[-8:]]}}
    return jsonify(out)


@app.route("/zprobe", methods=["POST"])
def zprobe():
    """Repeat inference on ONE frozen observation, and report the vertical component's mean and spread.

    The robot does not move. This separates a policy that is biased upward from a loop that walks upward:
    if E[dz] is clearly positive here, on a single fixed input, no amount of control tuning will fix it.
    Each repeat draws its own denoising noise, so the spread is the sampler's, not the robot's.
    """
    n = int((request.get_json(silent=True) or {}).get("n", 24))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    import numpy as _np
    KS = [int(x) for x in (request.get_json(silent=True) or {}).get("ks", [1, 2, 4, 8, 16])]
    out = {"n": n, "task": rt.STATE["task"], "ks": KS}
    per = {(r, k): [] for r in (0, 1) for k in KS}
    for _ in range(n):
        try:
            rt.INF.infer([h0, h1])
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"inference failed: {e}"})
        L = rt.INF.last
        for r in (0, 1):
            for k in KS:
                a = L["act"][k - 1, r * 10:r * 10 + 3]
                world = L["cur_mat"][r][:3, :3] @ a
                per[(r, k)].append((float(world[2]) * 1000, float(_np.linalg.norm(a)) * 1000))
    for r, arm in ((0, "L"), (1, "R")):
        out[arm] = {}
        for k in KS:
            v = _np.array(per[(r, k)])
            # the fraction that points the same way as the mean: a direction that is stable across draws is
            # a prediction, one that is half up and half down is the sampler
            agree = float(((v[:, 0] > 0) == (v[:, 0].mean() > 0)).mean())
            out[arm][f"k{k}"] = {"world_dz_mean": round(float(v[:, 0].mean()), 2),
                                 "world_dz_std": round(float(v[:, 0].std()), 2),
                                 "dz_sign_agree": round(agree, 2),
                                 "step_norm_mean": round(float(v[:, 1].mean()), 2),
                                 "dz_frac_of_norm": round(float(abs(v[:, 0].mean()) /
                                                               max(v[:, 1].mean(), 1e-6)), 2)}
    return jsonify(out)


@app.route("/replay2x2", methods=["POST"])
def replay2x2():
    """image x state, dataset x live: which half of the observation costs the amplitude?

    Held-out frames are replayed through the deployment core itself -- same preprocessing, same checkpoint,
    same UMI decode -- and each of the four combinations is scored against the ground-truth chunk stored with
    that frame. The robot does not move.

      dataset / dataset   the core on data the offline evaluator agrees about: the control
      dataset / live      the pictures the policy was trained on, the proprioception of right now
      live    / dataset   the pictures of right now, the proprioception of a demonstration
      live    / live      a rollout step, minus the rollout

    A ratio near the offline one in the control and far from it in live/live puts the fault in the
    observation; a control that is already far puts it in this inference path.
    """
    import numpy as _np
    body = request.get_json(silent=True) or {}
    n_frames = int(body.get("frames", 6))
    KS = [int(x) for x in body.get("ks", [1, 4, 16])]
    path = os.path.expanduser(body.get("pack", str(HOLOBRAIN_MAC / "replay_frames.npy")))
    if not os.path.isfile(path):
        return jsonify({"error": f"no frame pack at {path}"})
    pack = list(_np.load(path, allow_pickle=True))[:n_frames]

    live = _observe()
    if live is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    live2 = _observe()
    if live2 is None:
        return jsonify({"error": "observe failed"})
    live_state, _, _ = rt.INF.build_state([live, live2])
    live_imgs = live2["images"]

    g = torch.Generator().manual_seed(4242)
    noise = torch.randn(1, 16, rt.INF.policy.config.max_action_dim, generator=g)

    cells = {"ds_img/ds_state": (True, True), "ds_img/live_state": (True, False),
             "live_img/ds_state": (False, True), "live_img/live_state": (False, False)}
    out = {"n_frames": len(pack), "ks": KS, "cells": {}}
    for name, (ds_img, ds_state) in cells.items():
        acc = {(arm, k): [] for arm in (0, 1) for k in KS}
        gtacc = {(arm, k): [] for arm in (0, 1) for k in KS}
        for fr in pack:
            imgs = ({"middle": fr["jpg_middle"], "left": fr["jpg_left"], "right": fr["jpg_right"]}
                    if ds_img else live_imgs)
            st = fr["state20"] if ds_state else live_state
            act = rt.INF.infer_raw(imgs, st, task=str(fr["task"]), noise=noise)
            gt = _np.asarray(fr["action"], _np.float64)
            for arm in (0, 1):
                for k in KS:
                    o = arm * 10
                    acc[(arm, k)].append(float(_np.linalg.norm(act[k - 1, o:o + 3])) * 1000)
                    gtacc[(arm, k)].append(float(_np.linalg.norm(gt[k - 1, o:o + 3])) * 1000)
        cell = {}
        for arm, lab in ((0, "L"), (1, "R")):
            for k in KS:
                pn, gn = float(_np.mean(acc[(arm, k)])), float(_np.mean(gtacc[(arm, k)]))
                cell[f"{lab}{k}"] = {"pred_norm_mm": round(pn, 2), "gt_norm_mm": round(gn, 2),
                                     "ratio": round(pn / max(gn, 1e-6), 2)}
        out["cells"][name] = cell
    return jsonify(out)


@app.route("/dump", methods=["POST"])
def dump():
    """Freeze one live observation and this path's raw prediction for it. No motion.

    Everything the policy saw is written out -- the three JPEGs exactly as robot_service sent them, the state
    the core built, the prompt, the pose and joints -- plus the raw (16,20) chunk at a fixed seed. Feeding the
    same file to the offline path answers 'is it the same model seeing the same thing' before anyone argues
    about domain shift.
    """
    import numpy as _np
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    seed = int(request.args.get("seed", "99"))
    st, m_now, poses = rt.INF.build_state([h0, h1])
    # torch.manual_seed does NOT give MPS and CUDA the same numbers, and this policy starts from noise, so a
    # cross-device comparison needs the SAME tensor, generated on the CPU and handed to both paths.
    noise = torch.randn(1, rt.INF.policy.config.chunk_size, 20,
                        generator=torch.Generator().manual_seed(seed))
    try:
        rt.INF.infer([h0, h1], noise=noise)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    L = rt.INF.last
    path = os.path.expanduser(request.args.get("path", "~/live_dump.npz"))
    _np.savez_compressed(
        path,
        jpg_global=_np.frombuffer(base64.b64decode(h1["images"]["middle"]), _np.uint8),
        jpg_left=_np.frombuffer(base64.b64decode(h1["images"]["left"]), _np.uint8),
        jpg_right=_np.frombuffer(base64.b64decode(h1["images"]["right"]), _np.uint8),
        state20=st.astype(_np.float32), prompt=_np.array(rt.INF.task),
        joints_prev=_np.asarray(h0["joints14"], _np.float64),
        joints_now=_np.asarray(h1["joints14"], _np.float64),
        tcp=_np.asarray(m_now, _np.float64), raw_action=L["act"].astype(_np.float64),
        noise=noise.numpy().astype(_np.float32),      # the exact tensor both paths must start from
        seed=_np.array(seed), ckpt=_np.array(os.path.basename(str(rt.INF_CKPT).rstrip("/"))),
        denoise=_np.array(int(rt.INF.policy.config.num_denoising_steps)))
    return jsonify({"wrote": path, "seed": seed,
                    "ckpt": os.path.basename(str(rt.INF_CKPT).rstrip("/")),
                    "denoise": int(rt.INF.policy.config.num_denoising_steps),
                    "state20": [round(float(x), 5) for x in st],
                    "raw_step0_L": [round(float(x), 5) for x in L["act"][0, 0:3]],
                    "raw_step0_R": [round(float(x), 5) for x in L["act"][0, 10:13]],
                    "prompt": rt.INF.task})


def _denoise_ablation(ns, draws):
    """Does a cheaper sampler change the action, or only the clock?

    One observation pair is reused for every setting, and every setting is scored against the 10-step mean
    endpoint, so the comparison is sampler-vs-sampler with the policy, the scene and the state held fixed.
    """
    keep = rt.INF.policy.config.num_denoising_steps
    rt.STATE["denoise"] = {"running": True, "rows": []}
    try:
        h0 = _observe()
        time.sleep(cfg.OBS_GAP_S)
        h1 = _observe()
        if h0 is None or h1 is None:
            rt.STATE["denoise"] = {"running": False, "error": "observe failed"}
            return
        ref = None
        for n in ns:
            rt.INF.policy.config.num_denoising_steps = n
            ends, ms = [], []
            for j in range(draws):
                torch.manual_seed(4321 + 13 * j)
                t = time.perf_counter()
                rt.INF.infer([h0, h1])
                ms.append((time.perf_counter() - t) * 1000.0)
                L = rt.INF.last
                ends.append(_chunk_end(L))
            E = np.stack(ends)
            mean_end = E.mean(axis=0)
            spread = float(np.linalg.norm(E - mean_end, axis=-1).mean())
            if ref is None:
                ref = mean_end
            rt.STATE["denoise"]["rows"].append({
                "steps": n,
                "infer_ms_p50": round(float(np.median(ms)), 1),
                "infer_ms_max": round(float(np.max(ms)), 1),
                "draw_spread_mm": round(spread, 2),
                "delta_vs_10step_mm": round(float(np.linalg.norm(mean_end - ref)), 2),
                "endpoint_mm": [round(float(x), 1) for x in mean_end],
                "latency_steps": round(float(np.median(ms)) / cfg.DT_STEP_MS, 2)})
    finally:
        rt.INF.policy.config.num_denoising_steps = keep
        rt.STATE["denoise"]["running"] = False
    print(f"[v4-smoke] denoise ablation: {rt.STATE['denoise']['rows']}", flush=True)


@app.route("/denoise_ablation", methods=["POST"])
def denoise_ablation():
    if rt.STATE["running"] or rt.STATE.get("profile", {}).get("running"):
        return jsonify({"error": "stop the other run first"})
    ns = [int(x) for x in request.args.get("steps", "10,5,4,2").split(",")]
    draws = int(request.args.get("draws", "4"))
    threading.Thread(target=_denoise_ablation, args=(ns, draws), daemon=True).start()
    return jsonify({"started": True, "steps": ns, "draws": draws})
