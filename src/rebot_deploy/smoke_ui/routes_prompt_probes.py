"""Prompt / language diagnostics: routing line, wrist swap, grounding, swap test, prompt structure, the
prompt A/B loop and the prompt sweep.
"""
import base64
import itertools
import os
import threading
import time

from flask import jsonify, request
import numpy as np
import torch

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import ARM_IDX, IC as _IC
from .motion import _train_start_move
from .robot_client import _observe
from .routes_obs_probes import _chunk_end
from .safety import _grip_gate
from .step_exec import _one_press


@app.route("/routingline", methods=["POST"])
def routingline():
    """No motion. One inference, one line: which prediction slot is large, and which physical arm it drives.

    The chain printed here is the whole L/R question end to end --

        pred[0:10] / pred[10:20]   the model's two arm slots, raw
        -> IK target index 0 / 1
        -> joint block [0..5] / [6..11]
        -> 14-vector indices [0,1,2,3,4,5] / [7,8,9,10,11,12]
        -> physical LEFT / RIGHT arm

    The last hop is the one no code inspection can settle, so it was measured: /jog/0/8 moves the LEFT wrist
    feed 17.7x more than the right, /jog/7/8 moves the RIGHT feed 9.3x more. Those ratios are quoted in the
    response so the line is self-contained.
    """
    K = int(request.args.get("k", "6"))
    prompt = request.args.get("prompt", "RBP")
    text = dict(cfg.TASKS).get(prompt)
    if text is None:
        return jsonify({"error": f"unknown prompt {prompt}"})
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    keep = rt.INF.task
    rt.INF.task = text
    slots, dq_blocks = [], []
    try:
        for j in range(K):
            torch.manual_seed(1234 + 17 * j)
            rt.INF.infer([h0, h1])
            L = rt.INF.last
            k = max(0, int(L["valid_step"]) - 1)
            slots.append([float(np.linalg.norm((L["tgt_pos"][k, r] - L["cur_mat"][r][:3, 3]) * 1000.0))
                          for r in (0, 1)])
            # q_cmd is (T, 12): timesteps x joints. Slice the JOINT axis -- d[:6] would compare the first
            # six timesteps against the rest and read out as a left/right difference that is not there.
            d = np.degrees(np.abs(np.atleast_2d(L["q_cmd"]) - L["q_now"][ARM_IDX]))
            dq_blocks.append([float(d[:, :6].max()), float(d[:, 6:].max())])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        rt.INF.task = keep
    sl = np.asarray(slots).mean(0)
    dq = np.asarray(dq_blocks).mean(0)
    dom = 0 if sl[0] > sl[1] else 1
    return jsonify({
        "prompt": prompt, "draws": K,
        "line": (f"pred slot0(LEFT)={sl[0]:.1f} mm, slot1(RIGHT)={sl[1]:.1f} mm "
                 f"-> dominant slot {dom} -> joint block {'[0..5]' if dom == 0 else '[6..11]'} "
                 f"-> 14-vector {ARM_IDX[:6] if dom == 0 else ARM_IDX[6:]} "
                 f"-> physical {'LEFT' if dom == 0 else 'RIGHT'} arm"),
        "pred_slot_mm": {"slot0_LEFT": round(float(sl[0]), 1), "slot1_RIGHT": round(float(sl[1]), 1)},
        "commanded_joint_delta_deg": {"left_block": round(float(dq[0]), 3), "right_block": round(float(dq[1]), 3)},
        "physical_verification": {"jog_idx0_wrist_ratio": "17.74x LEFT", "jog_idx7_wrist_ratio": "9.26x RIGHT"},
        "global_rot180": bool(_IC.GLOBAL_ROT180)})


@app.route("/wristswap", methods=["POST"])
def wristswap():
    """No motion. Does swapping the two wrist feeds flip which arm the policy drives?

    The global feed has been shown to locate the requested colour correctly, so the open question is the
    next link: target position -> left/right arm. If the live wrist feeds are crossed relative to training,
    the policy can read the right target and still commit to the wrong hand, which is exactly the shape of
    "the opposite arm keeps moving".

    One frozen observation, one seed sequence, global and state untouched; only the two wrist images trade
    places. Arm flips with the swap -> wrist assignment is the bug. Arm stays -> the wrists are not the
    cause and the next suspect is the state/action channel order.
    """
    K = int(request.args.get("k", "10"))
    prompt = request.args.get("prompt", "RBP")
    text = dict(cfg.TASKS).get(prompt)
    if text is None:
        return jsonify({"error": f"unknown prompt {prompt}"})
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    im = h1["images"]
    variants = {"normal": im,
                "wrist_swapped": {**im, "left": im["right"], "right": im["left"]}}
    keep_task = rt.INF.task
    rt.INF.task = text
    out, horizon = {}, None
    try:
        for name, imgs in variants.items():
            hh = [h0, {**h1, "images": imgs}]
            keeps = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)
                rt.INF.infer(hh)
                L = rt.INF.last
                keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_mat": L["cur_mat"].copy()})
                v = int(L["valid_step"]) - 1
                horizon = v if horizon is None else min(horizon, v)
            out[name] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        rt.INF.task = keep_task
    KS = [k for k in (4, 8, 16, 31) if k <= horizon]
    rows = []
    for name, keeps in out.items():
        D = np.stack([np.stack([[np.linalg.norm((L["tgt_pos"][k, r] - L["cur_mat"][r][:3, 3]) * 1000.0)
                                 for r in (0, 1)] for k in KS]) for L in keeps])      # (K, nk, 2)
        m = D.mean(0)
        rows.append({"variant": name,
                     "per_k": [{"k": k, "L_mm": round(float(m[i][0]), 1), "R_mm": round(float(m[i][1]), 1),
                                "acting": "LEFT" if m[i][0] > m[i][1] else "RIGHT"} for i, k in enumerate(KS)]})
    return jsonify({"prompt": prompt, "draws": K, "horizon": horizon, "rows": rows})


@app.route("/grounding", methods=["POST"])
def grounding():
    """No motion. The whole colour-grounding question in ONE frozen observation.

    Every earlier attempt flipped its answer because conditions were collected across separate calls with
    the arms in different poses, or because object coordinates were read in the captured frame while the
    policy consumes the rot180 one. Both are closed here: one observation, one seed sequence, one frame
    convention (model input, i.e. after rot180), and every cell evaluated inside the common valid_step.

    Cells: 4 image conditions (orig, R<->B, R<->P, B<->P) x 3 prompts whose first colour differs.

    Arm choice is NOT used as the target proxy -- it was measured to stay on the right arm in all six
    cells of the previous test, so it carries no target information in this scene. What is reported per
    cell is the requested first colour's image x IN THE MODEL FRAME next to the approach vector, so the
    two can be checked for covariation across cells. That needs no camera calibration: if the policy
    grounds colour, moving the requested colour across the image must move the reach with it, and the
    relationship has to hold with one consistent sign over all twelve cells.
    """
    import cv2
    K = int(request.args.get("k", "8"))
    PROMPTS = [p for p in ("RBP", "BRP", "PRB")]
    TX = dict(cfg.TASKS)
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    rot = bool(_IC.GLOBAL_ROT180)
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(h1["images"]["middle"]), np.uint8), cv2.IMREAD_COLOR)
    if rot:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)          # model frame from here on
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (S > 90) & (V > 50)
    BANDS = {"R": (H <= 10) | (H >= 168), "B": (H >= 95) & (H <= 115), "P": (H >= 118) & (H <= 145)}
    masks, cen = {}, {}
    for c, f in BANDS.items():
        m = cv2.morphologyEx((sat & f).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if int(m.sum()) < 150:
            return jsonify({"error": f"cube {c} not found ({int(m.sum())} px)"})
        masks[c] = m
        ys, xs = np.nonzero(m)
        cen[c] = (float(xs.mean()), float(ys.mean()))

    allm = ((masks["R"] | masks["B"] | masks["P"]) * 255).astype(np.uint8)
    clean = cv2.inpaint(bgr, cv2.dilate(allm, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)

    def build(mapping):
        """mapping: colour -> centroid it should be drawn at"""
        out = clean.copy()
        for c, to in mapping.items():
            m = masks[c].astype(bool)
            dx, dy = int(round(to[0] - cen[c][0])), int(round(to[1] - cen[c][1]))
            ys, xs = np.nonzero(m)
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < out.shape[0]) & (nx >= 0) & (nx < out.shape[1])
            out[ny[ok], nx[ok]] = bgr[ys[ok], xs[ok]]
        return out

    ident = {c: cen[c] for c in "RBP"}
    def sw(a, b):
        m = dict(ident); m[a], m[b] = cen[b], cen[a]; return m
    CONDS = {"orig": ident, "R<->B": sw("R", "B"), "R<->P": sw("R", "P"), "B<->P": sw("B", "P")}

    def enc(img):
        if rot:
            img = cv2.rotate(img, cv2.ROTATE_180)      # back to capture orientation; _img rotates again
        ok, e = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("encode failed")
        return base64.b64encode(e.tobytes()).decode()

    imgs = {name: enc(build(m)) for name, m in CONDS.items()}
    KS = [2, 4, 8, 16]
    keep_task = rt.INF.task
    cells, horizon = {}, None
    try:
        for cond, b64 in imgs.items():
            hh = [h0, {**h1, "images": {**h1["images"], "middle": b64}}]
            for pr in PROMPTS:
                rt.INF.task = TX[pr]
                keeps = []
                for j in range(K):
                    torch.manual_seed(1234 + 17 * j)
                    rt.INF.infer(hh)
                    L = rt.INF.last
                    keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_mat": L["cur_mat"].copy()})
                    v = int(L["valid_step"]) - 1
                    horizon = v if horizon is None else min(horizon, v)
                cells[(cond, pr)] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        rt.INF.task = keep_task

    ks = [k for k in KS if k <= horizon]
    rows = []
    for (cond, pr), keeps in cells.items():
        first = pr[0]
        where = CONDS[cond][first]                     # where the requested colour actually IS in this image
        D = np.stack([np.concatenate([(L["tgt_pos"][ks, r] - L["cur_mat"][r][:3, 3]) * 1000.0
                                      for r in (0, 1)], axis=1) for L in keeps])
        mean = D.mean(0)
        sem = np.linalg.norm(D - mean, axis=2).mean(0) / max(1.0, K ** 0.5)
        rows.append({"cond": cond, "prompt": pr, "first": first,
                     "requested_xy_model_frame": [round(where[0], 1), round(where[1], 1)],
                     "per_k": [{"k": k, "L": [round(float(x), 1) for x in mean[i][:3]],
                                "R": [round(float(x), 1) for x in mean[i][3:]],
                                "sem_mm": round(float(sem[i]), 2)} for i, k in enumerate(ks)]})
    return jsonify({"frame": "model input (rot180 applied)" if rot else "as captured",
                    "global_transform": "rot180" if rot else "none",
                    "reported_object_coords": "model_input_frame",
                    "draws": K, "horizon": horizon, "ks": ks,
                    "camera_raw_xy": {c: [round(bgr.shape[1] - cen[c][0], 1), round(bgr.shape[0] - cen[c][1], 1)]
                                      for c in "RBP"} if rot else {c: [round(v[0], 1), round(v[1], 1)] for c, v in cen.items()},
                    "model_input_xy_after_rot180": {c: [round(v[0], 1), round(v[1], 1)] for c, v in cen.items()},
                    "image_width": bgr.shape[1], "rows": rows})


@app.route("/swaptest", methods=["POST"])
def swaptest():
    """No motion. Does the policy follow the cube's COLOUR or its PLACE?

    The live scene has red and blue on the opposite sides from the training median, so "always goes to
    blue" could be colour grounding or a spatial prior. Editing the picture separates them causally:
    swap the two cubes in the global frame and see whether the reach follows.

    Three variants on ONE frozen observation with one shared seed sequence, global feed only, wrists and
    state untouched:

      orig    the frame as captured
      noop    both cubes cut out, the holes inpainted, then pasted back WHERE THEY WERE
      swap    same cut and inpaint, then pasted at each other's centroid

    `noop` is the control that makes the result readable. Cutting and repasting changes edges, shadows and
    JPEG texture, and a policy can react to that alone, so the comparison that means anything is swap vs
    noop -- not swap vs orig. If noop already moves the output as much as swap does, the test says nothing
    and the masks or the inpainting need work.

    Objects are moved by MASK, not by bounding box, so background does not travel with them.
    """
    import cv2
    K = int(request.args.get("k", "12"))
    prompt = request.args.get("prompt", "RBP")
    text = dict(cfg.TASKS).get(prompt)
    if text is None:
        return jsonify({"error": f"unknown prompt {prompt}"})
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    # Work in the frame the MODEL sees. _img applies GLOBAL_ROT180 downstream, so masks and centroids
    # measured on the captured frame are mirrored with respect to the policy's input, and left/right come
    # out backwards -- that inversion produced a "the policy chases blue, not the prompted colour" reading
    # that was the exact opposite of the truth. Rotate here, and rotate back before handing the edit over.
    raw = np.frombuffer(base64.b64decode(h1["images"]["middle"]), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    _rot = bool(_IC.GLOBAL_ROT180)
    if _rot:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (S > 90) & (V > 50)
    masks = {"R": (sat & ((H <= 10) | (H >= 168))).astype(np.uint8),
             "B": (sat & (H >= 95) & (H <= 115)).astype(np.uint8)}
    for c, m in masks.items():
        m[:] = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if int(m.sum()) < 150:
            return jsonify({"error": f"cube {c} not found in the global frame ({int(m.sum())} px)"})
    cen = {}
    for c, m in masks.items():
        ys, xs = np.nonzero(m)
        cen[c] = (float(xs.mean()), float(ys.mean()))

    both = ((masks["R"] | masks["B"]) * 255).astype(np.uint8)
    clean = cv2.inpaint(bgr, cv2.dilate(both, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)

    def paste(dst, c, to):
        m = masks[c].astype(bool)
        dx, dy = int(round(to[0] - cen[c][0])), int(round(to[1] - cen[c][1]))
        ys, xs = np.nonzero(m)
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < dst.shape[0]) & (nx >= 0) & (nx < dst.shape[1])
        dst[ny[ok], nx[ok]] = bgr[ys[ok], xs[ok]]

    noop = clean.copy(); paste(noop, "R", cen["R"]); paste(noop, "B", cen["B"])
    swap = clean.copy(); paste(swap, "R", cen["B"]); paste(swap, "B", cen["R"])

    def enc(img):
        if _rot:                       # back to capture orientation; _img will rotate it again
            img = cv2.rotate(img, cv2.ROTATE_180)
        ok, e = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("encode failed")
        return base64.b64encode(e.tobytes()).decode()

    variants = {"orig": h1["images"]["middle"], "noop": enc(noop), "swap": enc(swap)}
    KS = [1, 2, 4, 8, 16, 31]
    keep_task = rt.INF.task
    rt.INF.task = text
    per, horizon = {}, None
    try:
        for name, b64 in variants.items():
            hh = [h0, {**h1, "images": {**h1["images"], "middle": b64}}]
            keeps = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)
                rt.INF.infer(hh)
                L = rt.INF.last
                keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_mat": L["cur_mat"].copy(),
                              "valid_step": int(L["valid_step"])})
                v = int(L["valid_step"]) - 1
                horizon = v if horizon is None else min(horizon, v)
            per[name] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        rt.INF.task = keep_task

    ks = [k for k in KS if k <= horizon]
    res = {}
    for name, keeps in per.items():
        D = np.stack([np.concatenate([(L["tgt_pos"][ks, r] - L["cur_mat"][r][:3, 3]) * 1000.0
                                      for r in (0, 1)], axis=1) for L in keeps])   # (K, nk, 6)
        mean = D.mean(0)
        res[name] = {"mean": mean, "sem": np.linalg.norm(D - mean, axis=2).mean(0) / max(1.0, K ** 0.5)}

    rows = []
    for i, k in enumerate(ks):
        o, n, w = res["orig"]["mean"][i], res["noop"]["mean"][i], res["swap"]["mean"][i]
        def cmp(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return (round(float(np.linalg.norm(a - b)), 2),
                    round(float(a @ b / (na * nb)), 4) if na > 1e-9 and nb > 1e-9 else None)
        d_no, c_no = cmp(o, n)
        d_sw, c_sw = cmp(n, w)
        rows.append({"k": k,
                     # left xyz then right xyz, mm, base frame -- the components are what says whether the
                     # reach tracked the cube that moved or merely changed
                     "orig_vec": [round(float(x), 1) for x in o],
                     "noop_vec": [round(float(x), 1) for x in n],
                     "swap_vec": [round(float(x), 1) for x in w],
                     "orig_mm": round(float(np.linalg.norm(o)), 2),
                     "noop_mm": round(float(np.linalg.norm(n)), 2),
                     "swap_mm": round(float(np.linalg.norm(w)), 2),
                     "edit_artifact_mm": d_no, "cos_orig_noop": c_no,
                     "swap_effect_mm": d_sw, "cos_noop_swap": c_sw,
                     "sem_mm": round(float(np.mean([res[v]["sem"][i] for v in res])), 2)})
    return jsonify({"prompt": prompt, "draws": K, "horizon": horizon,
                    "frame": "model view (rot180 applied)" if _rot else "as captured",
                    "cube_centroids_px": {c: [round(x, 1) for x in v] for c, v in cen.items()},
                    "cube_side": {c: ("LEFT" if v[0] < bgr.shape[1] / 2 else "RIGHT") for c, v in cen.items()},
                    "cube_mask_px": {c: int(m.sum()) for c, m in masks.items()},
                    "rows": rows})


@app.route("/promptstruct", methods=["POST"])
def promptstruct():
    """No motion. Does the chunk respect the prompt's FIRST colour, and from which k does it stop?

    Coordinate-free by construction. Cube positions are never used: the six prompts already carry the
    structure, because three pairs share a first colour --

        R: RBP RPB      B: BRP BPR      P: PRB PBR

    A policy that grounds the first colour must keep each pair together while the first cube is being
    approached, and may only separate later, when the second colour starts to matter. So the statistic is,
    at every k, the mean distance between the two members of a pair against the mean distance across pairs.
    Reconstructing cube coordinates from successful-grasp TCPs was considered and rejected: the target would
    then be built in the policy's own frame from the policy's own grasp geometry, and object grounding could
    no longer be told apart from kinematics.

    Every prompt sees ONE frozen observation and the same seed sequence, so the sampler is differenced out.
    The per-k draw spread is reported next to the ratio -- a ratio computed on top of noise means nothing.
    """
    K = int(request.args.get("k", "12"))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})
    hh = [h0, h1]
    PAIRS = {"R": ("RBP", "RPB"), "B": ("BRP", "BPR"), "P": ("PRB", "PBR")}
    TEXT = dict(cfg.TASKS)
    keep_task = rt.INF.task
    per_prompt = {}
    try:
        for label, text in cfg.TASKS:
            rt.INF.task = text
            draws = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)      # identical noise for every prompt
                rt.INF.infer(hh)
                L = rt.INF.last
                m_now = L["cur_mat"]
                T = int(L["valid_step"])
                # both arms' world displacement from the frozen pose, per k -> (T,6)
                d = np.concatenate([(L["tgt_pos"][:T, r] - m_now[r][:3, 3]) * 1000.0 for r in (0, 1)], axis=1)
                draws.append(d)
            n = min(len(d) for d in draws)
            D = np.stack([d[:n] for d in draws])       # (K, n, 6)
            per_prompt[label] = D
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        rt.INF.task = keep_task

    n = min(v.shape[1] for v in per_prompt.values())
    mean = {k: v[:, :n].mean(0) for k, v in per_prompt.items()}
    spread = {k: np.linalg.norm(v[:, :n] - v[:, :n].mean(0), axis=2).mean(0) for k, v in per_prompt.items()}
    rows = []
    for k in range(n):
        within = [np.linalg.norm(mean[a][k] - mean[b][k]) for a, b in PAIRS.values()]
        between = []
        for (c1, (a1, b1)), (c2, (a2, b2)) in itertools.combinations(PAIRS.items(), 2):
            for x in (a1, b1):
                for y in (a2, b2):
                    between.append(np.linalg.norm(mean[x][k] - mean[y][k]))
        w, bt = float(np.mean(within)), float(np.mean(between))
        noise = float(np.mean([spread[p][k] for p in mean])) / max(1.0, K ** 0.5)
        rows.append({"k": k, "within_mm": round(w, 2), "between_mm": round(bt, 2),
                     "ratio": round(bt / w, 2) if w > 1e-6 else None,
                     "sem_mm": round(noise, 2),
                     "per_pair_mm": {c: round(float(np.linalg.norm(mean[a][k] - mean[b][k])), 2)
                                     for c, (a, b) in PAIRS.items()}})
    return jsonify({"draws": K, "steps": n, "rows": rows})


def _ab_loop(chunks, fixed=None):
    """Same scene, same start pose, six prompts: does the instruction change what the robot does?

    Every prompt is run from a freshly re-commanded TRAIN START pose, so the only thing that differs between
    rows is the text. Both the PREDICTED chunk endpoint (what the policy wanted) and the MEASURED one (what
    the arm did) are recorded -- the first is the language test, the second says whether execution preserved
    the difference.
    """
    keep = rt.INF.task
    # `fixed` repeats ONE prompt through the identical procedure: that is the noise floor (sampler draw +
    # start-pose repeatability + scene drift) the six-prompt spread has to beat before it means anything.
    seq = [(f"{fixed}#{i+1}", dict(cfg.TASKS)[fixed]) for i in range(len(cfg.TASKS))] if fixed else list(cfg.TASKS)
    rt.STATE["ab"] = {"running": True, "rows": [], "pairs": {}, "mode": fixed or "six-prompt"}
    ends = {}
    try:
        for label, text in seq:
            if not rt.STATE["ab"]["running"]:
                break
            _train_start_move()
            rt.INF.task = text
            rt.STATE["task"] = label
            agg = None
            for _ in range(chunks):
                tel = _one_press()
                if tel.get("error"):
                    rt.STATE["ab"]["rows"].append({"prompt": label, "error": tel["error"]})
                    agg = None
                    break
                agg = tel
            if agg is None:
                continue
            L = rt.INF.last
            pred_end = _chunk_end(L)
            ends[label] = pred_end
            row = {"prompt": label,
                   "pred_mm": [r_["pred_mm"] for r_ in agg["rows"]],
                   "actual_mm": [r_["actual_mm"] for r_ in agg["rows"]],
                   "dir_cos": [r_["dir_cos"] for r_ in agg["rows"]],
                   "pred_LR_ratio": round(agg["rows"][0]["pred_mm"] / max(agg["rows"][1]["pred_mm"], 1e-6), 2),
                   "grip_width_mm": agg.get("grip_width_mm"),
                   "pred_endpoint_mm": [round(float(x), 1) for x in pred_end]}
            rt.STATE["ab"]["rows"].append(row)
        labels = [k for k in ends]
        rt.STATE["ab"]["pairs"] = {f"{a}-{b}": round(float(np.linalg.norm(ends[a] - ends[b])), 2)
                                for i, a in enumerate(labels) for b in labels[i + 1:]}
        if rt.STATE["ab"]["pairs"]:
            v = list(rt.STATE["ab"]["pairs"].values())
            rt.STATE["ab"]["mean_pair_mm"] = round(sum(v) / len(v), 2)
            rt.STATE["ab"]["max_pair_mm"] = max(v)
    finally:
        rt.INF.task = keep
        rt.STATE["ab"]["running"] = False
    print(f"[v4-smoke] prompt A/B done: {rt.STATE['ab'].get('mean_pair_mm')} mm mean pairwise", flush=True)


@app.route("/prompt_ab", methods=["POST"])
def prompt_ab():
    if rt.STATE["running"] or rt.STATE.get("ab", {}).get("running"):
        return jsonify({"error": "stop the loop first"})
    g = _grip_gate()
    if g:
        return jsonify(g)
    chunks = int(request.args.get("chunks", "1"))
    fixed = request.args.get("fixed")
    if fixed and fixed not in dict(cfg.TASKS):
        return jsonify({"error": f"unknown prompt {fixed}"})
    threading.Thread(target=_ab_loop, args=(chunks, fixed), daemon=True).start()
    return jsonify({"started": True, "mode": fixed or "six-prompt", "chunks": chunks})


@app.route("/prompt_ab_stop", methods=["POST"])
def prompt_ab_stop():
    if rt.STATE.get("ab"):
        rt.STATE["ab"]["running"] = False
    return jsonify({"stopped": True})


@app.route("/prompt_sweep", methods=["POST"])
def prompt_sweep():
    """No motion. One scene, all six prompts, K draws each: does the instruction change what the model wants?

    The sampler starts from noise, so a single draw per prompt cannot answer this -- two runs of the SAME
    prompt already differ. So the comparison is between-prompt distance of the MEAN 16-step endpoint against
    the within-prompt spread of the draws. Anything below the noise floor is not a language effect.
    """
    K = int(request.args.get("k", os.environ.get("V4_SWEEP_K", "4")))
    h0 = _observe()
    if h0 is None:
        return jsonify({"error": "observe failed"})
    time.sleep(cfg.OBS_GAP_S)
    h1 = _observe()
    if h1 is None:
        return jsonify({"error": "observe failed"})

    keep_task, keep_n = rt.INF.task, rt.INF.n_action
    rt.INF.n_action = 1                                   # nothing is sent; this only limits the returned slice
    # Two passes on purpose. The endpoint must be the SAME step for every prompt, and valid_step is not
    # constant across prompts or draws, so the common horizon can only be known after all draws are in.
    # Taking each draw's own last index instead is what produced the retracted "DELTA ignores the prompt's
    # first colour" reading; see _chunk_end.
    raw, ends = {}, {}
    horizon = None
    try:
        for label, text in cfg.TASKS:
            rt.INF.task = text
            keeps = []
            for j in range(K):
                torch.manual_seed(1234 + 17 * j)       # same noise sequence for every prompt: only text varies
                rt.INF.infer([h0, h1])
                L = rt.INF.last
                keeps.append({"tgt_pos": L["tgt_pos"].copy(), "cur_tcp": L["cur_tcp"].copy(),
                              "valid_step": int(L["valid_step"])})
                v = int(L["valid_step"]) - 1
                horizon = v if horizon is None else min(horizon, v)
            raw[label] = keeps
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"inference failed: {e}"})
    finally:
        rt.INF.task, rt.INF.n_action = keep_task, keep_n
    for label, keeps in raw.items():
        ends[label] = np.stack([_chunk_end(L, horizon) for L in keeps])   # (K, 6) mm at the common step

    labels = [t[0] for t in cfg.TASKS]
    means = {k: v.mean(axis=0) for k, v in ends.items()}
    within = float(np.mean([np.linalg.norm(v - v.mean(axis=0), axis=-1).mean() for v in ends.values()]))
    sem = float(np.mean([v.std(axis=0, ddof=1).mean() / np.sqrt(K) for v in ends.values()])) if K > 1 else 0.0
    pairs = {f"{a}-{b}": round(float(np.linalg.norm(means[a] - means[b])), 2)
             for i, a in enumerate(labels) for b in labels[i + 1:]}
    bt = float(np.mean(list(pairs.values())))
    return jsonify({
        "dry_run": True, "draws_per_prompt": K,
        "endpoint_mm_per_prompt": {k: [round(float(x), 1) for x in v] for k, v in means.items()},
        "between_prompt_mean_mm": round(bt, 2),
        "within_prompt_spread_mm": round(within, 2),
        "sem_of_means_mm": round(sem, 2),
        "bt_over_sem": round(bt / max(sem, 1e-9), 2),
        "verdict": ("prompt changes the intended trajectory" if bt > 3 * sem
                    else "NOT distinguishable from sampling noise"),
        "common_horizon_step": horizon,
        "pairs_mm": pairs})
