"""[2026-09-22] Real-robot inference core for the X-VLA checkpoints on the reBot (deploy stage; was infer_core_v4.py, written for
the r150-umi-v4 checkpoint with body-frame SE(3) deltas).

In:  robot observations (14 joint values per /observe, camera JPEGs as base64), a checkpoint and its V4_* contract.
Out: /execute_step joint vectors (leader frame) per waypoint, plus the decoded chunk (`XvlaInferencer.last`) for logging.
usage: from rebot_deploy.xvla_inference import XvlaInferencer   (the smoke UI: python -m rebot_deploy.run_smoke_ui)

Same seat as infer_core_umi.UmiInferencer, and deliberately the same conventions (eef_kin FK/IK, FLIP_IDX,
DQ/FK guards, the umi_schema gripper scale). What differs is the action semantics, and that difference is the
whole reason this file exists:

    v3 (UMI)  action[k] is relative to ONE base, the current pose:  T_k = base @ rel_k        (cumulative)
    v4 (ours) action[k] is relative to the PREVIOUS step:           T_k = T_{k-1} @ dT_k      (incremental)

Deploying an incremental chunk through the cumulative path collapses the trajectory onto its first step -- the
A3 checkpoint already cost a day to that exact confusion -- so the composition here is sequential, and it is a
RIGHT multiplication because the delta lives in the body frame (`inv(T_t) @ T_{t+1}`). A left multiplication
would apply the rotation about the base frame and send the arm somewhere else entirely.

observation.state is rebuilt the way the converter wrote it: per arm [prev_rel pos3+rot6d, gripper width],
prev_rel = inv(T_t) @ T_{t-1}, which needs the two most recent observations.

Safety for the first real-robot runs: every per-step delta is clamped to V4_CLAMP_MM / V4_CLAMP_DEG BEFORE it
is composed (scaling translation and rotation angle, direction preserved), and the gripper is HELD at its
current raw count unless V4_GRIPPER=predict. The point of the first test is direction and magnitude of the
arm motion, not the task.

[2026-09-23] V4_ACTION_MODE=umi switches the decode to the UMI current-anchor chunk the B180/B663 runs are
trained on. It is not a variant of the composition below, it is the absence of one:

    delta   T_k = T_{k-1} @ dT_k        each step on the previous PREDICTED pose
    umi     T_k = T_now   @ A_k         every step on the SAME measured pose

The forbidden form is `T_{k-1} @ A_k`, which would treat a cumulative displacement as an increment and run
the arm k times too far. There is no sequential path in the umi branch at all, so it cannot be written.

The clamp scales with the horizon in umi mode. A_k is the displacement over k steps, not one step, so
clamping every A_k to the per-step limit would flatten the trajectory into a sphere around T_now; the
envelope is k * V4_CLAMP_MM instead.

Env: V4_CKPT, V4_DEVICE (mps), V4_ACTION_MODE (delta|umi), V4_CLAMP_MM (10), V4_CLAMP_DEG (2),
     V4_N_ACTION (4), V4_GRIPPER (hold|predict|binary|continuous|relative), V4_DQ_MAX (0.6), V4_FK_MAX_MM (60), V4_TASK,
     V4_GLOBAL_ROT180 (0) -- rotate the GLOBAL feed 180 deg to match the training convention; the rest is described next to
     each setting below. Settings the UI switches at run time (GRIPPER_MODE, RTC_ON, REMOTE_INFER, IK_BACKEND, LIMITS,
     WRIST_ZOOM, ...) are module attributes read at use time: keep reading them as `xvla_inference.NAME`.
"""
from __future__ import annotations

import base64
import collections
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot, Slerp

from ego_paths import HOLOBRAIN_MAC, UMI_BRIDGE, prepend_sys_path  # machine-specific roots (env-overridable)

# the HoloBrain umi_pkg goes in front of sys.path BEFORE umi.common.pose_util is imported (by the contract and below), as it
# always did, so a machine that has it keeps using that copy
prepend_sys_path(str(HOLOBRAIN_MAC / "umi_pkg"))

from . import eef_kin  # noqa: E402
# The jaw scales, state builders and the delta clamp live in robot_dataset_contract (2026-11-01 split); every name is
# re-exported here, so `xvla_inference.rawToWidth`, `xvla_inference.GRIP_BIN_OPEN`, ... keep working.
from .robot_dataset_contract import (  # noqa: E402,F401
    CART20_W_OPEN, FIXED_ANCHOR_R, GRIP_ACC_CLOSE, GRIP_ACC_N, GRIP_ACC_OPEN, GRIP_BIN_CLOSED, GRIP_BIN_OPEN, GRIP_BIN_THRESH,
    GRIP_CMD_FULL, GRIP_CMD_MAX, GRIP_CONTRACT_FILE, GRIP_M_PER_RAW, GRIP_OBS_PER_CMD, GRIP_RAW_MAX, GRIP_RAW_MIN, GRIP_REL_EPS,
    GRIP_REL_MODE, GRIP_REL_SCALE, _gripContract, cart20State, clampDelta, jawHoldCmd, rawToWidth, relCart20State, unwrapGrip,
    widthToCmd, widthToRaw)
from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat, pose_to_mat  # noqa: E402


class InferenceConfigError(ValueError):
    """A V4_* / IK_BACKEND setting combination the inference core refuses (raised at import)."""


# [2026-09-30 user] REL-only ablation checkpoints (HEAD180-RELCART20-RELONLY-V3 / R312C-RELCART20-RELONLY-V4) were trained with
# action channels 20:32 hard-zeroed at the transformer input AND output. Deployment must apply the SAME contract at every
# denoising step, or the unsupervised 20:32 values re-enter the transformer. V4_RELONLY=1 installs it (the training plugin's own
# function). Off by default: joint/FK checkpoints are unaffected. Such checkpoints must be run with pure Pink (the dq head is
# unsupervised).
RELONLY = os.environ.get("V4_RELONLY", "0") == "1"


def _installRelOnly():
    """V4_RELONLY=1: the training plugin's zero mask on action channels 20:32 (rel16_aux_relonly, an external checkout)"""
    prepend_sys_path(str(UMI_BRIDGE / "rel16_audit/relonly"))
    import rel16_aux_relonly
    rel16_aux_relonly.install_zero_mask()
    if os.environ.get("IK_BACKEND", "numerical") in ("pinkdq", "policydq"):
        raise InferenceConfigError("[infer_core_v4] V4_RELONLY=1 with IK_BACKEND=pinkdq/policydq: "
                                   "the dq head is unsupervised -- use pink")


if RELONLY:
    _installRelOnly()

DEVICE = os.environ.get("V4_DEVICE", "mps")
CKPT = os.environ.get("V4_CKPT", str(HOLOBRAIN_MAC / "ckpt_v4base_of3000"))
# The chunk length is not carried by any weight -- X-VLA uses it only to size the initial flow sample and to
# pad targets -- so a checkpoint will happily emit any number of rows. That is not permission to ask a
# 16-trained checkpoint for 32: rows past its training horizon are extrapolation with nothing to check them
# against. V4_CHUNK is for checkpoints TRAINED at that horizon; left at 0 the checkpoint's own value stands.
CHUNK_OVERRIDE = int(os.environ.get("V4_CHUNK", "0"))
DTYPE = os.environ.get("V4_DTYPE", "fp32")
if DTYPE not in ("fp32", "fp16", "bf16"):   # [2026-10-02 user] bf16 added (weights + X-VLA input cast via config.dtype)
    raise InferenceConfigError(f"V4_DTYPE must be fp32, fp16 or bf16, not {DTYPE!r}")
ACTION_MODE = os.environ.get("V4_ACTION_MODE", "delta")
if ACTION_MODE not in ("delta", "umi"):
    raise InferenceConfigError(f"V4_ACTION_MODE must be delta or umi, not {ACTION_MODE!r}")

# robot_service camera key -> the dataset's camera key -> what the policy was trained to call it
CAM_MAP = {"middle": "global", "left": "left_wrist", "right": "right_wrist"}
RENAME = {"observation.images.global": "observation.images.image",
          "observation.images.left_wrist": "observation.images.image2",
          "observation.images.right_wrist": "observation.images.image3"}

ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP_IDX = [6, 13]
FLIP_IDX = [0, 1, 5, 7, 8, 12]               # follower/observation frame -> leader frame, as /execute_step wants
ARM_ROWS = (0, 1)                            # left, right
SLIM20_ARM_DIMS = 10                         # slim20 / umi actions: per arm pose9 + gripper
POSE9 = 9                                    # pos3 + rot6d rows
DQ_HEAD = slice(20, 32)                      # policydq / pinkdq: the policy's dq12 head (action dims 20:32)
MM_PER_M = 1000.0
DQ_MAX = float(os.environ.get("V4_DQ_MAX", "0.6"))
IK_LOG = os.path.expanduser(os.environ.get("V4_IK_LOG", "~/v4_ik_waypoints.jsonl"))
# [2026-09-29] numerical (PyRoki) | learned (LIK0) | pink | [2026-10-01] diffik (one Jacobian step)
IK_BACKEND = os.environ.get("IK_BACKEND", "numerical")
# [2026-10-03 user] IK_BACKEND=curobo: horizon-aware cuRobo IK on the 5090 (curobo_ik_server.py, CUDA only -> remote). One request
# per chunk = every predicted waypoint from the measured q; the server picks one continuous joint path (GPU multi-seed IK + DP
# over the waypoints). Same TCP/frame as pink (dataset frame, no R_DS_TO_FK). No silent fallback: a failed request stops the
# cycle.
CUROBO_URL = os.environ.get("V4_CUROBO_URL", "http://100.64.0.5:8790")
CUROBO_TIMEOUT_S = 10.0
# [2026-10-03 user "추론 5090 통해서"] V4_REMOTE_INFER=<url of xvla_infer_server.py on the 5090>: the policy forward (pre + flow
# sampling + post) runs there on V4_REMOTE_CKPT (the same checkpoint, a path on the 5090); cameras, state, IK and execution stay
# here. The local model is still loaded (ablation tools, fallback when V4_REMOTE_INFER is cleared at runtime).
# [2026-10-03 user "rtc 구현 + rtc만 켜고 끌 수 있게"] Real-Time Chunking (Black et al. 2025, PI; LeRobot policies/rtc) for X-VLA.
# The new chunk is generated as an inpainting of the previous chunk's not-yet-executed part, so consecutive chunks join without a
# jump. X-VLA's sampler predicts the CLEAN action x^ at every flow step (x_t = noise*t + x^*(1-t), t: 1 -> 0), so LeRobot's
# velocity-form guidance v_g = v - gw * J^T(W*(Y - x^)) becomes, with x^ = x_t - t*v:  x^_g = x^ + t * gw * J^T(W*(Y - x^)),  J =
# d x^ / d x_t (one backward through the action transformer per flow step; the VLM encoding is computed once, no grad).
#   Y   previous chunk targets (absolute base-frame poses + widths, as executed) re-expressed in the NEW observation's
#       current-anchor frame (umi action: A_k = M_now^-1 T_prev), normalised with the checkpoint's own MEAN_STD action stats;
#       channels 20:32 = 0.
#   alignment  prev index j* = the previous target nearest the measured TCP (both arms); new step i <-> prev step j*+1+i
#   W   1 for the first V4_RTC_DELAY steps, linear decay to 0 at V4_RTC_HORIZON (LeRobot LINEAR schedule), 0 after / past the
#       leftover; channels 20:32 never guided.  gw = min(c * inv_r2, V4_RTC_MAX_GW), tau = 1 - t (LeRobot formula).
# No previous chunk (first inference of a /run, RESET, older than V4_RTC_MAX_AGE_S) -> plain sampling. Remote inference: the
# server runs the same sampler (10-06).
RTC_ON = os.environ.get("V4_RTC", "0") == "1"
RTC_DELAY = int(os.environ.get("V4_RTC_DELAY", "2"))
RTC_HORIZON = int(os.environ.get("V4_RTC_HORIZON", "8"))
RTC_MAX_GW = float(os.environ.get("V4_RTC_MAX_GW", "10.0"))
RTC_MAX_AGE_S = float(os.environ.get("V4_RTC_MAX_AGE_S", "10.0"))
RTC_STD_EPS = 1e-8                           # added to the action std when normalising the RTC prefix
RTC_R2_EPS = 1e-12                           # floor of (1 - tau)^2 in the guidance weight


def _rtcWeights(delay, horizon, total) -> torch.Tensor:
    """RTC prefix weights (was _rtc_weights): 1 for the first `delay` steps, linear decay to 0 at `horizon`, 0 after"""
    tWeights = torch.zeros(total)
    tDelay = max(0, min(delay, total))
    tHorizon = max(tDelay, min(horizon, total))
    tWeights[:tDelay] = 1.0
    if tHorizon > tDelay:
        tWeights[tDelay:tHorizon] = torch.linspace(1, 0, tHorizon - tDelay + 2)[1:-1]
    return tWeights


def _installXvlaRtc():
    """Patch X-VLA's generate_actions with the RTC sampler (was _install_xvla_rtc); a no-op without self._rtc or when patched."""
    import lerobot.policies.xvla.modeling_xvla as MX
    M = [c for c in vars(MX).values()
         if isinstance(c, type) and hasattr(c, "generate_actions") and c.__module__ == MX.__name__][0]
    if getattr(M, "_rtc_patched", False):
        return
    orig = M.generate_actions

    def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps, x1=None):
        R = getattr(self, "_rtc", None)
        if R is None:
            return orig(self, input_ids, image_input, image_mask, domain_id, proprio, steps, x1=x1)
        self.eval()
        dt_ = self._get_target_dtype()
        image_input = image_input.to(dtype=dt_)
        proprio = proprio.to(dtype=dt_)
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        assert self.target_align is None and self.xy_source == "none", "RTC path: target_align / xy_source not supported"
        B = input_ids.shape[0]
        A = self.dim_action
        if x1 is None:
            x1 = torch.randn(B, self.chunk_size, A, device=proprio.device, dtype=dt_)
        else:
            x1 = x1.to(device=proprio.device, dtype=dt_)
        Y = R["Y"].to(device=proprio.device, dtype=torch.float32)[None]          # (1, T, A)
        W = R["W"].to(device=proprio.device, dtype=torch.float32)[None]          # (1, T, A)
        action = torch.zeros_like(x1)
        steps = max(1, int(steps))
        gws = []
        errs = []
        for i in range(steps, 0, -1):
            tt = i / steps
            t = torch.full((B,), tt, device=proprio.device, dtype=dt_)
            x_t = (x1 * tt + action * (1 - tt)).detach().requires_grad_(True)
            with torch.enable_grad():
                proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
                pred = self.transformer(target_embed=None, target_token=None, domain_id=domain_id, action_with_noise=x_t_m,
                                        proprio=proprio_m, t=t, **enc)
                err = (Y - pred.float()) * W
                corr = torch.autograd.grad(pred.float(), x_t, err.detach())[0].float()
            tau = 1.0 - tt
            if tau <= 0:
                gw = RTC_MAX_GW
            else:
                inv_r2 = ((1 - tau) ** 2 + tau ** 2) / max((1 - tau) ** 2, RTC_R2_EPS)
                gw = min(((1 - tau) / tau) * inv_r2, RTC_MAX_GW)
            action = (pred.float() + tt * gw * corr).to(dt_).detach()
            gws.append(round(gw, 2))
            errs.append(round(float(err.abs().sum() / W.sum().clamp(min=1)), 4))
        self._rtc_log = dict(gw=gws, err=errs)
        return self.action_space.postprocess(action)
    M.generate_actions = generate_actions
    M._rtc_patched = True
    print("[infer_core_v4] RTC sampler installed on X-VLA (active only when V4_RTC / /mode rtc=1 and a previous chunk exists)",
          flush=True)


REMOTE_INFER = os.environ.get("V4_REMOTE_INFER", "")
REMOTE_CKPT = os.environ.get("V4_REMOTE_CKPT", "")
REMOTE_TIMEOUT_S = 30
_REMOTE_SESS = None
REMOTE_LAST = {}


def remoteForward(imagesB64ByView, state20, task, noise=None, rtc=None) -> np.ndarray:
    """The policy forward on the remote inference server (was remote_forward): (T, A) action chunk in raw units."""
    global _REMOTE_SESS
    if _REMOTE_SESS is None:
        import requests
        _REMOTE_SESS = requests.Session()
    tStart = time.time()
    tPayload = {"ckpt": REMOTE_CKPT, "dtype": DTYPE, "images": imagesB64ByView, "state": np.asarray(state20, float).tolist(),
                "task": task,
                "rot180": ["global"] if GLOBAL_ROT180 else [], "mirror": ["global"] if GLOBAL_MIRROR else [],
                "denoise": int(os.environ["V4_DENOISE_STEPS"]) if os.environ.get("V4_DENOISE_STEPS") else None,
                "noise": None if noise is None else np.asarray(noise.detach().float().cpu().numpy())[0].tolist(),
                # [2026-10-06] RTC on the server (xvla_infer_server_rtc.py): previous-chunk prefix Y / weights W
                "rtc": None if rtc is None else {"Y": rtc["Y"].tolist(), "W": rtc["W"].tolist(), "max_gw": RTC_MAX_GW}}
    tReply = _REMOTE_SESS.post(REMOTE_INFER + "/infer", json=tPayload, timeout=REMOTE_TIMEOUT_S).json()
    if "act" not in tReply:
        raise RuntimeError(f"remote inference failed: {tReply}")
    REMOTE_LAST.update(rt_ms=round((time.time() - tStart) * 1000, 1), server_ms=tReply["ms"], gpu_ms=tReply["gpu_ms"],
                       rtc_log=tReply.get("rtc"))
    return np.asarray(tReply["act"], np.float64)


_CUROBO_SESS = None


def curoboSolve(q12, pos, quat, timeout=CUROBO_TIMEOUT_S):
    """One cuRobo request (was curobo_solve): every waypoint from q12 -> (q (T,12), pos err mm (T,2), success (T,2), info)."""
    global _CUROBO_SESS
    if _CUROBO_SESS is None:
        import requests
        _CUROBO_SESS = requests.Session()          # keep-alive: no TCP/HTTP setup per chunk over the tailnet
    tRequest = {"q0": [float(tX) for tX in q12], "pos": np.asarray(pos, float).tolist(), "quat": np.asarray(quat, float).tolist()}
    tReply = _CUROBO_SESS.post(CUROBO_URL + "/solve", json=tRequest, timeout=timeout).json()
    return (np.asarray(tReply["q"], np.float64), np.asarray(tReply["pos_err_mm"], np.float64),
            np.asarray(tReply["success"], bool), dict(ms=tReply["ms"], rot_err_deg=tReply["rot_err_deg"]))


# [2026-10-01] 0 = off; e.g. 8 with V4_PINK_LOCK= (joint5 free, slew-limited)
YAW_MAX_STEP_DEG = float(os.environ.get("V4_YAW_MAX_STEP_DEG", "0"))
# [2026-09-30] pink|pinkdq: joint-name substrings hard-locked at the measured q (e.g. wrist_yaw)
PINK_LOCK = tuple(x for x in os.environ.get("V4_PINK_LOCK", "").split(",") if x)
# [2026-09-30] IK_BACKEND=pink|pinkdq: full | yaw (position + base-z rotation; roll/pitch excluded) | pitch (position +
# approach-axis elevation; roll/yaw free) | pitch_hold (roll/yaw held at the measured pose, pitch from the model) | yaw_hold
# (roll/pitch from the model, yaw held at the measured pose)
PINK_ORI = os.environ.get("V4_PINK_ORI", "full")
FK_MAX_MM = float(os.environ.get("V4_FK_MAX_MM", "60"))
# mutable so the UI can raise/lower the envelope between presses without a restart: a 3 mm command turned
# out to be smaller than what the servos resolve (2.204 mm commanded -> 0.126 mm measured), so the first
# question "does the arm move the way the model predicts" needs a command big enough to measure.
LIMITS = {"mm": float(os.environ.get("V4_CLAMP_MM", "10")), "deg": float(os.environ.get("V4_CLAMP_DEG", "2"))}
CLAMP_MM = LIMITS["mm"]
CLAMP_DEG = LIMITS["deg"]


def setLimits(mm=None, deg=None) -> dict:
    """change the per-step clamp envelope in place (was set_limits); returns a copy of LIMITS"""
    if mm is not None:
        LIMITS["mm"] = float(mm)
    if deg is not None:
        LIMITS["deg"] = float(deg)
    return dict(LIMITS)


N_ACTION = int(os.environ.get("V4_N_ACTION", "4"))
# Which horizon point to execute, 1-based. A[1] is one 15 Hz step of ground truth -- about 0.4 mm -- which is
# below this policy's own per-step noise, so executing it integrates the noise instead of the motion. A later
# waypoint is a larger, better-conditioned target; it is NOT a sum of the earlier ones, because every A[k]
# hangs off the same T_now. This changes only which point of the predicted trajectory is sent to IK.
EXEC_K = int(os.environ.get("V4_EXEC_K", "1"))
# [2026-10-01 user] inference-side stabilisers, all OFF unless set. Order inside infer(): chunk -> temporal ensemble -> adaptive k
# -> near-zero hold -> (UI) IK on waypoint k-1. Each cycle executes ONE waypoint (chunk index k-1) and re-infers, so the robot
# advances k dataset steps per cycle; a step counter (sum of the k actually chosen) lines older chunks up with the current one.
# V4_TE=1  ACT-style temporal ensemble over the last V4_TE_N (3) chunks, weights V4_TE_W ("1,0.5,0.25", newest first). Waypoint k
#          of the new chunk = weighted mean, in the BASE frame, of the predictions of the same moment by each stored chunk still
#          inside its 16-step horizon (chunk inferred s steps earlier -> its index k + s). Translation only unless V4_TE_ROT=1
#          (weighted quaternion mean) / V4_TE_GRIP=1 (width mean). Raw chunks are the history. Cleared on /run, RESET ANCHOR, gap
#          > V4_TE_MAXGAP_S.
# V4_ADAPTIVE_K="4,2,1"  per-cycle k: far -> 4, medium -> 2, near/align -> 1. Distance = max over arms of the (ensembled)
#          predicted translation at the chunk end (index 15) from the measured TCP: > V4_AK_FAR_MM (40) far, > V4_AK_NEAR_MM (15)
#          medium, else near. Forced to near when either arm's predicted gripper crosses V4_GRIP_THRESH within the chunk
#          (grasp/release = align phase). V4_EXEC_K is ignored while this is set.
# V4_DEADBAND_XY_MM / V4_DEADBAND_Z_MM  near-zero hold: per arm, if the xy norm of (target - measured TCP) < eps_xy both x and y
#          are held at the measured value; z separately if |dz| < eps_z. Rotation untouched. Decided on the (ensembled)
#          displacement at chunk step V4_DEADBAND_REF_K (8) -- "the policy predicts < eps of motion over 8 steps" -- and applied
#          to every waypoint. (V4_DEADBAND_MM="x,y,z" = per-axis variant.)
TE_ON = os.environ.get("V4_TE", "0") == "1"
TE_W = np.array([float(x) for x in os.environ.get("V4_TE_W", "1,0.5,0.25").split(",")],
                np.float64)[:int(os.environ.get("V4_TE_N", "3"))]
TE_ROT = os.environ.get("V4_TE_ROT", "0") == "1"
TE_GRIP = os.environ.get("V4_TE_GRIP", "0") == "1"
TE_MAXGAP_S = float(os.environ.get("V4_TE_MAXGAP_S", "15"))
_ak = os.environ.get("V4_ADAPTIVE_K", "").strip()
ADAPTIVE_K = [int(x) for x in _ak.split(",")] if _ak else None
AK_FAR_MM = float(os.environ.get("V4_AK_FAR_MM", "40"))
AK_NEAR_MM = float(os.environ.get("V4_AK_NEAR_MM", "15"))
_db = os.environ.get("V4_DEADBAND_MM", "").strip()
DEADBAND_MM = np.array([float(x) for x in _db.split(",")], np.float64) if _db else None
if DEADBAND_MM is not None:
    assert DEADBAND_MM.shape == (3,), "V4_DEADBAND_MM needs x,y,z"
DB_XY_MM = float(os.environ["V4_DEADBAND_XY_MM"]) if os.environ.get("V4_DEADBAND_XY_MM") else None
DB_Z_MM = float(os.environ["V4_DEADBAND_Z_MM"]) if os.environ.get("V4_DEADBAND_Z_MM") else None
DB_REF_K = int(os.environ.get("V4_DEADBAND_REF_K", "8"))
GRIPPER_MODE = os.environ.get("V4_GRIPPER", "hold")
JOINT_LIMIT_MARGIN_RAD = math.radians(5)     # commanded joints stay this far inside the URDF limits
LO = np.array([-2.8, -3.14, -3.14, -1.87, -1.57, -3.14] * 2) + JOINT_LIMIT_MARGIN_RAD
HI = np.array([2.8, 0, 0, 1.57, 1.57, 3.14] * 2) - JOINT_LIMIT_MARGIN_RAD
HI[[1, 2, 7, 8]] = 0.0

# The dataset's TCP rotation and this FK's TCP rotation are the same frame up to a FIXED R_y(90 deg):
# measured on 31 frames of recorded joints vs the recorded tcp_state, position identical to 0.00 mm, rotation
# 90.00 deg with standard deviation 0.00, axis [0,1,0]. So  R_fk = R_dataset @ Ry(90).
# Labels are unaffected (they are pose-to-pose relatives, where the fixed rotation cancels), but every robot
# command is not: without this the arm tracks the demonstration's position with its wrist 90 deg out.
R_DS_TO_FK = Rot.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
R_FK_TO_DS = R_DS_TO_FK.T
USE_FRAME_FIX = os.environ.get("V4_FRAME_FIX", "1") != "0"
# [2026-10-06 user] HandUMI TCP rotation = the wrist CAMERA's axes (handumi_camera_tcp_v2: TCP x = camera z), not the fingers: the
# camera is pitched against the jaw, so every human-learned rotation / translation direction is expressed in a frame tilted by a
# fixed angle about the jaw-closing axis (TCP y). V4_UMI_TCP_PITCH_DEG = theta: the robot TCP is used as T @ Ry(-theta) ("HandUMI
# convention") for state, anchors, chunk targets and RTC; IK gets the inverse. Position unchanged. 0 = off.
# 33.6 = robot rest (fingers 2.4 deg below horizontal) placed at the HandUMI start pose vs the human starts' camera-axis tilt
# 36.0.
UMI_TCP_PITCH_DEG = float(os.environ.get("V4_UMI_TCP_PITCH_DEG", "0"))
R_UMI_OFF = Rot.from_rotvec([0.0, -np.radians(UMI_TCP_PITCH_DEG), 0.0]).as_matrix()
# [2026-10-06 user "9.5 기준점 오프셋도"] V4_UMI_TCP_OFFSET_MM = "x,y,z": the HandUMI TCP POINT expressed in the robot dataset TCP frame
# (hand-eye: robot wrist cam X_tcp_cam @ HandUMI X_cam_tcp -> (-94.8, 2.4, 38.4) mm). HandUMI pose = T_robot @ [R_UMI_OFF | t];
# IK targets get the exact inverse. Empty = 0 (rotation-only behaviour unchanged).
UMI_TCP_OFF_T = np.array([float(x) for x in os.environ.get("V4_UMI_TCP_OFFSET_MM", "0,0,0").split(",")]) / 1000.0
UMI_CONV = bool(UMI_TCP_PITCH_DEG) or bool(np.any(UMI_TCP_OFF_T))

# The GLOBAL feed's orientation is part of the deployment contract, not a detail. B663 trained on two
# camera pools -- the HEAD pool as recorded and the FRONT (R675) pool with its global feed rotated 180 deg
# to match it -- so the DATASET has a single global orientation. Verified 2026-09-25 by putting the live
# frame beside frames from both pools: the dark equipment band sits at the BOTTOM in every training frame
# and at the TOP live, i.e. the live global camera is mounted 180 deg from the training convention.
# RETRACTED 2026-09-25, same day: that reasoning was appearance-only (where the dark equipment band sits)
# plus a magnitude change, and it got the answer backwards. The test that decides it compares against the
# relation measured IN the training data -- over 60 HEAD episodes, a first target on the image LEFT is
# grasped at base y +226 mm and one on the image RIGHT at -157 mm, corr(image x, grasp y) = -0.709. Replaying
# the same frozen scene through the policy:
#     rot180 ON   the right arm always acts and runs from -315 out to -418, away from every target,
#                 corr(image x, predicted y) = +0.944 -- the opposite sign
#     rot180 OFF  the left arm acts for left-side targets and moves +315 -> +260, toward the +226 the data
#                 says is there, corr = -0.263 -- the same sign as training
# So the default is 0. Only the global feed would be rotated; the wrist feeds are mounted on the arms.
# Do not turn this back on from appearance alone: re-measure against the training relation.
GLOBAL_ROT180 = os.environ.get("V4_GLOBAL_ROT180", "0") != "0"
# Mirror is a separate axis, applied AFTER the rotation. rot180 flips both image axes and a mirror flips x,
# so rot180 + mirror is a pure vertical flip -- the four settings are the four ways the feed can be oriented.
# A mirror is not a harmless visual change: it reverses handedness, so a left-handed approach in the picture
# becomes right-handed while the action stays as it was. The converter measured this on the FRONT pool and
# rejected it (residual 7.7 px for rot180, 58.7 px for a mirror, 145.6 px for both), but that was FRONT
# against HEAD, not the live camera against training, so the live case is decided by its own measurement.
GLOBAL_MIRROR = os.environ.get("V4_GLOBAL_MIRROR", "0") != "0"
# [2026-10-07 user "feed 영상을 확대"] right-wrist zoom = focal scale s about the robot C922 principal point (318.3, 224.9 @ 640x480):
# f' = s f, c unchanged (not a centre crop). 1.0 = off. Mutable at runtime (UI /wrist_zoom) through WRIST_ZOOM["s"].
WRIST_ZOOM = {"s": float(os.environ.get("V4_WRIST_ZOOM", "1.0")), "c": (318.3126, 224.9012), "wh": (640, 480)}
ZOOM_OFF_TOL = 1e-6                          # |s - 1| below this = no zoom
POLICY_IMAGE_PX = 224                        # the policy's square input image


def wristZoom(bgr, s):
    """focal-scale zoom by s about the C922 principal point (was wrist_zoom); s == 1 returns the image itself"""
    if abs(s - 1.0) < ZOOM_OFF_TOL:
        return bgr
    tH, tW = bgr.shape[:2]
    tCx = WRIST_ZOOM["c"][0] * tW / WRIST_ZOOM["wh"][0]
    tCy = WRIST_ZOOM["c"][1] * tH / WRIST_ZOOM["wh"][1]
    tAffine = np.array([[s, 0.0, tCx * (1 - s)], [0.0, s, tCy * (1 - s)]], np.float32)     # dst = s (src - c) + c
    return cv2.warpAffine(bgr, tAffine, (tW, tH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# UMI76 deployment. The B663 models are slim20 + chunk 32; the HEAD180 UMI76 twins are 76-D state + chunk 16
# with the observation history taken at a PHYSICAL 50.05 ms, not at the previous frame. Serving 15 Hz frames
# straight in would hand the policy a 66.7 ms history it never trained on, so the history pose is
# interpolated out of a timestamped ring buffer. Contract: ~/umi_bridge/umi76/UMI76_CONTRACT.md.
STATE_MODE = os.environ.get("V4_STATE_MODE", "slim20")        # slim20 | umi76 | umi94 (v2 diagnostic twin B)
# [2026-10-03] ONE-ARM models (HRA_red right-only approach prior, ~/ego_cart20/ego_cart20/right_only.py): V4_ARM_ONLY=right serves
# the model exactly as it was trained -- LEFT state dims = dummy (identity pose9, gL = ARM_ONLY_LEFT_G), LEFT wrist image = black
# (the exporter wrote a blank frame), and the outputs that had NO loss (left arm 0:10, gripper 19 when masked) are never executed:
# the left arm and both jaws hold their measured position.  Unset = previous behaviour, byte for byte.  relcart20 state only.
ARM_ONLY = os.environ.get("V4_ARM_ONLY", "").strip().lower()
ARM_ONLY_LEFT_G = float(os.environ.get("V4_ARM_ONLY_LEFT_G", "0.0"))
ARM_ONLY_GRIP = os.environ.get("V4_ARM_ONLY_GRIP", "hold")     # hold | model (only if the right gripper WAS supervised)
# output guard (one-arm models): a chunk whose right-arm target leaves the training range is REFUSED (never clipped into motion).
# HRA training: 0.8 s right translation p95 ~11 cm. A garbage output otherwise drives DQ_MAX (0.6 rad) per waypoint (measured).
ARM_ONLY_MAX_PRED_MM = float(os.environ.get("V4_ARM_ONLY_MAX_PRED_MM", "250"))
ARM_ONLY_MAX_PRED_DEG = float(os.environ.get("V4_ARM_ONLY_MAX_PRED_DEG", "60"))


# jawHoldCmd (the jaw command that holds the measured jaw): robot_dataset_contract.py
_IDENTITY9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float32)
if ARM_ONLY not in ("", "right"):
    raise InferenceConfigError(f"V4_ARM_ONLY={ARM_ONLY!r}: only 'right' is supported")
if ARM_ONLY and STATE_MODE != "relcart20":
    raise InferenceConfigError("V4_ARM_ONLY=right needs V4_STATE_MODE=relcart20 (the right-only dataset's state)")


def _blackB64():
    """the blank 224x224 JPEG the right-only exporter wrote for the left wrist, base64 (was _black_b64)"""
    tOk, tEnc = cv2.imencode(".jpg", np.zeros((POLICY_IMAGE_PX, POLICY_IMAGE_PX, 3), np.uint8))
    return base64.b64encode(tEnc.tobytes()).decode()


def armOnlyImages(imgs):
    """left wrist -> the blank frame the right-only exporter wrote (only with V4_ARM_ONLY=right) (was arm_only_images)"""
    if not ARM_ONLY:
        return imgs
    global _BLACK_B64
    if _BLACK_B64 is None:
        _BLACK_B64 = _blackB64()
    tOut = dict(imgs)
    tOut["left"] = _BLACK_B64
    return tOut


_BLACK_B64 = None
# CART20_W_OPEN, FIXED_ANCHOR_R, relCart20State, cart20State: robot_dataset_contract.py (imported at the top)
UMI_HISTORY_DT = 3.0 / 59.94                                   # 50.05 ms, upstream's own spacing
OBS_BUF_S = float(os.environ.get("V4_OBS_BUF_S", "1.0"))

# jaw scales and conversions (GRIP_*, unwrapGrip, rawToWidth, widthToRaw, widthToCmd), _gripContract: robot_dataset_contract.py


def gripToCmd(v) -> float:
    """model gripper channel -> /execute_step command, for the active GRIPPER_MODE (predict = width, binary = intent).
    (was grip_to_cmd)"""
    if GRIPPER_MODE == "binary":
        return GRIP_BIN_OPEN if float(v) >= GRIP_BIN_THRESH else GRIP_BIN_CLOSED
    # relative: the executed command comes from _gripRelative; this is the log value
    if GRIPPER_MODE in ("continuous", "relative"):
        # [2026-09-29 user] continuous g (label = clip(leader_cmd/45, 0, 1)) -> its inverse, cmd = clip(g, 0, 1) * 45 (full
        # travel)
        return float(np.clip(float(v), 0.0, 1.0) * GRIP_CMD_FULL)
    return widthToCmd(v)


# clampDelta (one body-frame delta shrunk to the safety envelope): robot_dataset_contract.py


DEFAULT_TASK = "Stack the red cube on the bottom, blue cube in the middle, and purple cube on the top, on the plate."
IK_MAX_JOINT_DELTA_OFF = ("none", "0", "")   # V4_IK_MAX_JOINT_DELTA values that switch the solver's per-solve clip off
UMI76_DIM = 76
# UMI76 slice offsets per arm (alphabetical upstream key order, see buildStateUmi76)
UMI76_OFFSETS = {0: dict(pos=0, pos_wrt=6, rot=12, rot_wrt=24, grip=36),
                 1: dict(pos=38, pos_wrt=44, rot=50, rot_wrt=62, grip=74)}
INTERP_TIME_TOL_S = 1e-6                     # a query this far before the buffer start still counts as inside it
ACT_STATS_FILE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
OPEN_JAW_FRACTION = 0.5                      # relative gripper, first cycle: open if the jaw reads > half travel


class XvlaInferencer:
    """The X-VLA policy on the reBot: observation -> state -> chunk -> clamped base-frame targets -> joint commands
    (was V4Inferencer). Method and instance-attribute names (infer, solve_waypoint, tcp_now, reset_anchor, last, task,
    n_action, chunk, policy, _rtc_prev, _rtc_delay_hint, _rtc_plan_prefix, _anchor, _g_prev, ...) are the smoke UI's
    interface to this object and are kept as they were."""
    obs_horizon = 2                      # prev_rel needs t-1 and t

    def __init__(self, ckpt: str = CKPT, device: str = DEVICE):
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        _installXvlaRtc()
        self._ckpt_dir = ckpt
        self._rtc_prev = None
        self.policy = XVLAPolicy.from_pretrained(ckpt)
        self.policy.to(device)
        # fp16 on MPS is 2.50x faster than fp32 (498 -> 199 ms at 5 denoising steps) and, compared at the
        # SAME initial flow sample, moves the predicted chunk by at most 0.365 mm -- against a per-step
        # motion of 4-8 mm and a sampler spread of 18-65 mm. bf16 is no faster and 12x less accurate
        # (4.4 mm), so it is not offered.
        if DTYPE == "fp16":
            self.policy.half()
        elif DTYPE == "bf16":
            self.policy.config.dtype = "bfloat16"
            self.policy.to(torch.bfloat16)
        # [2026-10-02 user] flow-matching denoising steps at inference (checkpoint default 10). Mac MPS bf16, A30k, 80 R312c
        # frames, same noise: 10 steps 382 ms (k4 |err vs GT| med 4.0 / p90 36.4 mm), 5 steps 268 ms (3.8 / 25.9 mm); ~19 ms per
        # step + ~115 ms fixed.
        if os.environ.get("V4_DENOISE_STEPS"):
            self.policy.config.num_denoising_steps = int(os.environ["V4_DENOISE_STEPS"])
            print(f"[infer_core_v4] denoising steps {self.policy.config.num_denoising_steps} (V4_DENOISE_STEPS)", flush=True)
        self.policy.eval()
        self.chunk = int(self.policy.config.chunk_size)
        if CHUNK_OVERRIDE:
            if CHUNK_OVERRIDE != self.chunk:
                print(f"[infer_core_v4] chunk {self.chunk} -> {CHUNK_OVERRIDE} by V4_CHUNK; rows beyond the "
                      f"trained horizon are extrapolation", flush=True)
            # the inner model copies chunk_size at construction, so the config alone is not enough
            self.policy.config.chunk_size = CHUNK_OVERRIDE
            self.policy.config.n_action_steps = CHUNK_OVERRIDE
            self.policy.model.chunk_size = CHUNK_OVERRIDE
            self.chunk = CHUNK_OVERRIDE
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": RENAME}},
            postprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        # [2026-09-29] the HandUMI solver clipped every joint to ±0.35 rad per solve (eef_kin default, inherited from the
        # offline retarget); on HEAD180 GT 33 % of the executed k16 targets need more, so the arm was asked for less.
        # Default stays N0 = 0.35 (official deployed baseline; 13:20 decision: clip-off N1 is offline diagnostic only, never on
        # the real robot). V4_IK_MAX_JOINT_DELTA=none gives N1 for offline use. DQ_MAX/LO/HI apply either way.
        tMaxJointDelta = os.environ.get("V4_IK_MAX_JOINT_DELTA", "0.35").lower()
        self.kin = eef_kin.Kin({"max_joint_delta": None if tMaxJointDelta in IK_MAX_JOINT_DELTA_OFF else float(tMaxJointDelta)})
        self.n_action = N_ACTION
        self.task = os.environ.get("V4_TASK", DEFAULT_TASK)
        self.calls = 0
        self.last = {}
        print(f"[v4] {os.path.basename(str(ckpt).rstrip('/'))} device={device} chunk={self.policy.config.chunk_size} "
              f"state={self.policy.config.max_state_dim} action_mode={self.policy.config.action_mode} "
              f"n_action={self.n_action} clamp={LIMITS['mm']}mm/{LIMITS['deg']}deg gripper={GRIPPER_MODE} "
              f"dq_max={DQ_MAX} ik_max_joint_delta={self.kin.mSolver.config.max_joint_delta} ik_backend={IK_BACKEND} "
              f"fk_guard={FK_MAX_MM}mm"
              + (f" ARM_ONLY={ARM_ONLY} (left state dummy, left image black, left arm + jaws hold, grip {ARM_ONLY_GRIP})"
                 if ARM_ONLY else ""), flush=True)

    # ---------------------------------------------------------------- observation
    @staticmethod
    def _img(b64, rot180=False, mirror=False, zoom=1.0):
        """one camera JPEG (base64) -> the policy's (3, 224, 224) float RGB tensor in [0, 1]"""
        tBytes = np.frombuffer(base64.b64decode(b64), np.uint8)
        tBgr = cv2.imdecode(tBytes, cv2.IMREAD_COLOR)
        tBgr = wristZoom(tBgr, zoom)                          # [2026-10-07] right wrist only (callers pass WRIST_ZOOM["s"])
        if rot180:
            tBgr = cv2.rotate(tBgr, cv2.ROTATE_180)
        if mirror:
            tBgr = cv2.flip(tBgr, 1)
        tRgb = cv2.cvtColor(cv2.resize(tBgr, (POLICY_IMAGE_PX, POLICY_IMAGE_PX), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.moveaxis(tRgb.astype(np.float32) / 255.0, -1, 0)).contiguous()

    def _tcpMat(self, joints14):
        """FK of an observation -> TCP mats (2,4,4) and poses (2,6) [pos, rotvec] in the DATASET frame (was _tcp_mat)"""
        tQ = np.asarray([float(tX) for tX in joints14], np.float64)
        tLeft7, tRight7 = self.kin.fkPose7(tQ[ARM_IDX])
        tPoses = np.stack([np.concatenate([tP7[:3], Rot.from_quat(tP7[3:]).as_rotvec()]) for tP7 in (tLeft7, tRight7)])
        tMats = pose_to_mat(tPoses)
        if USE_FRAME_FIX:                                    # FK frame -> dataset frame
            for tRow in ARM_ROWS:
                tMats[tRow][:3, :3] = tMats[tRow][:3, :3] @ R_FK_TO_DS
        if UMI_CONV:                                         # [10-06] robot TCP -> HandUMI (camera-axis) convention
            for tRow in ARM_ROWS:
                tMats[tRow][:3, 3] = tMats[tRow][:3, 3] + tMats[tRow][:3, :3] @ UMI_TCP_OFF_T
                tMats[tRow][:3, :3] = tMats[tRow][:3, :3] @ R_UMI_OFF
            tPoses = np.stack([np.concatenate([tMats[tRow][:3, 3],
                                               Rot.from_matrix(tMats[tRow][:3, :3]).as_rotvec()]) for tRow in ARM_ROWS])
        return tMats, tPoses                                 # (2,4,4), (2,6) in the DATASET frame

    def push_obs(self, joints14, t=None):
        """Record one observation for the UMI76 history buffer. Cheap: FK only, no model."""
        if not hasattr(self, "_buf"):
            self._buf = collections.deque()
        tTime = time.time() if t is None else float(t)
        tMats, _ = self._tcpMat(joints14)
        tWidths = np.array([rawToWidth(joints14[GRIP_IDX[tRow]]) for tRow in ARM_ROWS], np.float64)
        self._buf.append((tTime, tMats.copy(), tWidths))
        while len(self._buf) > 2 and self._buf[0][0] < tTime - OBS_BUF_S:
            self._buf.popleft()
        return len(self._buf)

    def _interpAt(self, tq):
        """SE(3) + width at an arbitrary time, from the buffer. Linear position/width, slerp rotation. (was _interp_at)"""
        tBuf = getattr(self, "_buf", None)
        if not tBuf or len(tBuf) < 2:
            raise RuntimeError("umi76: need at least two timestamped observations")
        tTimes = np.array([tEntry[0] for tEntry in tBuf])
        if tq < tTimes[0] - INTERP_TIME_TOL_S:
            raise RuntimeError(
                f"umi76: the buffer spans {tTimes[-1]-tTimes[0]:.4f}s but the query needs {UMI_HISTORY_DT:.4f}s of "
                f"history. Observations must be timestamped when the robot is READ (_observe sets 't'); "
                f"pushing them at inference time collapses the gap to microseconds.")
        tJ = int(np.clip(np.searchsorted(tTimes, tq), 1, len(tTimes) - 1))
        tT0, tM0, tW0 = tBuf[tJ - 1]
        tT1, tM1, tW1 = tBuf[tJ]
        tU = 0.0 if tT1 <= tT0 else float(np.clip((tq - tT0) / (tT1 - tT0), 0.0, 1.0))
        tOut = np.zeros((2, 4, 4))
        for tRow in ARM_ROWS:
            tKey = Rot.from_matrix(np.stack([tM0[tRow][:3, :3], tM1[tRow][:3, :3]]))
            tOut[tRow][:3, :3] = Slerp([0.0, 1.0], tKey)([tU]).as_matrix()[0]
            tOut[tRow][:3, 3] = tM0[tRow][:3, 3] * (1 - tU) + tM1[tRow][:3, 3] * tU
            tOut[tRow][3, 3] = 1.0
        return tOut, tW0 * (1 - tU) + tW1 * tU

    def build_state_umi76(self):
        """The 76-D upstream-bimanual packing, from the buffer, at [t-50.05ms, t] (and the cart20 / relcart20 / umi94 states).

        Slice order is alphabetical by upstream key and timestep-major inside each key -- the order
        timm_obs_encoder.py:169 produces after `sorted(low_dim_keys)`. It is NOT the yaml's key order, and
        it must not be re-packed for slim20 compatibility: the widened encoder was trained on this layout.
        """
        tBuf = getattr(self, "_buf", None)
        if not tBuf:
            raise RuntimeError("umi76: no observations buffered")
        tNow = tBuf[-1][0]
        tCur, tWCur = self._interpAt(tNow)
        tHist, tWHist = self._interpAt(tNow - UMI_HISTORY_DT)
        tInvCur = [np.linalg.inv(tCur[tRow]) for tRow in ARM_ROWS]
        tState = np.zeros(UMI76_DIM, np.float32)
        for tRow in ARM_ROWS:
            tOther = 1 - tRow
            tSelfHist = mat_to_pose10d((tInvCur[tRow] @ tHist[tRow])[None])[0]
            tSelfCur = mat_to_pose10d((tInvCur[tRow] @ tCur[tRow])[None])[0]
            tCrossHist = mat_to_pose10d((tInvCur[tOther] @ tHist[tRow])[None])[0]
            tCrossCur = mat_to_pose10d((tInvCur[tOther] @ tCur[tRow])[None])[0]
            tOff = UMI76_OFFSETS[tRow]
            tState[tOff["pos"]:tOff["pos"] + 3] = tSelfHist[:3]
            tState[tOff["pos"] + 3:tOff["pos"] + 6] = tSelfCur[:3]
            tState[tOff["pos_wrt"]:tOff["pos_wrt"] + 3] = tCrossHist[:3]
            tState[tOff["pos_wrt"] + 3:tOff["pos_wrt"] + 6] = tCrossCur[:3]
            tState[tOff["rot"]:tOff["rot"] + 6] = tSelfHist[3:9]
            tState[tOff["rot"] + 6:tOff["rot"] + 12] = tSelfCur[3:9]
            tState[tOff["rot_wrt"]:tOff["rot_wrt"] + 6] = tCrossHist[3:9]
            tState[tOff["rot_wrt"] + 6:tOff["rot_wrt"] + 12] = tCrossCur[3:9]
            tState[tOff["grip"]] = tWHist[tRow]
            tState[tOff["grip"] + 1] = tWCur[tRow]
        tPoses = np.stack([np.concatenate([tCur[tRow][:3, 3], Rot.from_matrix(tCur[tRow][:3, :3]).as_rotvec()])
                           for tRow in ARM_ROWS])
        if STATE_MODE == "cart20":
            return cart20State(tCur, tWCur), tCur, tPoses
        if STATE_MODE == "relcart20":
            # anchor = the first observation after reset_anchor() (the UI calls it on /run): the deploy counterpart of the
            # training episode's frame_index-0 row. Start a rollout from the rest pose, as the recordings did.
            if getattr(self, "_anchor", None) is None:
                self._anchor = [tCur[0].copy(), tCur[1].copy()]
                # [2026-10-07] B / origin-anchored models: RIGHT anchor = the fixed frame F (base)
                if FIXED_ANCHOR_R is not None:
                    self._anchor[1] = FIXED_ANCHOR_R.copy()
                print(f"[relcart20] anchor set: L {np.round(tCur[0][:3, 3] * 1000, 1)} R {np.round(tCur[1][:3, 3] * 1000, 1)} mm",
                      flush=True)
            tState = relCart20State(self._anchor, tCur, tWCur)
            if ARM_ONLY == "right":                                  # trained with a dummy left arm: never show it the real one
                tState[0:9] = _IDENTITY9
                tState[18] = ARM_ONLY_LEFT_G
            return tState, tCur, tPoses
        if STATE_MODE == "umi94":
            # [2026-09-28] twin B: + per-arm base-frame TCP [pos3, rot6d] at t, L then R, in the DATASET frame
            # (cur is already frame-fixed), exactly derive_v2.py's layout. Diagnostic only -- ego has no such pose.
            tState = np.concatenate([tState, mat_to_pose10d(tCur[0][None])[0],
                                     mat_to_pose10d(tCur[1][None])[0]]).astype(np.float32)
        return tState, tCur, tPoses

    def build_state(self, history):
        """per arm [prev_rel pos3 + rot6d, gripper width] -> (20,), exactly the converter's layout."""
        assert len(history) == self.obs_horizon, f"need {self.obs_horizon} observations, got {len(history)}"
        tMPrev, _ = self._tcpMat(history[-2]["joints14"])
        tMNow, tPosesNow = self._tcpMat(history[-1]["joints14"])
        tState = np.zeros(20, np.float32)
        for tRow in ARM_ROWS:
            tPrevRel = np.linalg.inv(tMNow[tRow]) @ tMPrev[tRow]
            tState[tRow * SLIM20_ARM_DIMS:tRow * SLIM20_ARM_DIMS + POSE9] = mat_to_pose10d(tPrevRel)
            tState[tRow * SLIM20_ARM_DIMS + POSE9] = rawToWidth(history[-1]["joints14"][GRIP_IDX[tRow]])
        return tState, tMNow, tPosesNow

    # ---------------------------------------------------------------- one inference
    def infer_raw(self, images_b64: dict, state20, task=None, noise=None) -> np.ndarray:
        """One forward pass on an observation assembled by the caller: (T,20) in raw units.

        `infer` builds the state from two observations and then goes through IK, clamps and the gripper
        scale. This is the same preprocessing and the same policy with none of that, so an image and a state
        from different sources can be combined -- which is the only way to ask whether a live rollout misses
        because of the pictures or because of the proprioception.
        """
        images_b64 = armOnlyImages(images_b64)
        if REMOTE_INFER:
            return remoteForward({tView: images_b64[tKey] for tKey, tView in CAM_MAP.items() if tKey in images_b64}, state20,
                                 task or self.task, noise)
        tObs = {f"observation.images.{tView}": self._img(images_b64[tKey], GLOBAL_ROT180 and tView == "global",
                                                         GLOBAL_MIRROR and tView == "global",
                                                         WRIST_ZOOM["s"] if tView == "right_wrist" else 1.0)
                for tKey, tView in CAM_MAP.items() if tKey in images_b64}
        tObs["observation.state"] = torch.as_tensor(np.asarray(state20, np.float32))
        tObs["task"] = task or self.task
        tBatch = self.pre(tObs)
        with torch.no_grad():
            tChunk = (self.policy.predict_action_chunk(tBatch, noise=noise.to(self.device))
                      if noise is not None else self.policy.predict_action_chunk(tBatch))
        return self.post(tChunk)[0].detach().float().cpu().numpy().astype(np.float64)

    def infer(self, history, noise=None) -> np.ndarray:
        """two (or more) observations -> the joint commands of the first kept waypoints; details in self.last"""
        if STATE_MODE in ("umi76", "umi94", "cart20", "relcart20"):
            for tObsRow in history:
                self.push_obs(tObsRow["joints14"], tObsRow.get("t"))
            tState, tMNow, tPosesNow = self.build_state_umi76()
        else:
            tState, tMNow, tPosesNow = self.build_state(history)
        tImgs = armOnlyImages(history[-1]["images"])
        if REMOTE_INFER:
            tMissing = [tView for tKey, tView in CAM_MAP.items() if tKey not in tImgs]
            if tMissing:
                raise ValueError(f"camera missing: {tMissing} (observation had {list(tImgs)})")
            tRtc = self._rtcBuild(tMNow) if RTC_ON else None          # [2026-10-06] RTC now also with remote inference
            tAct = remoteForward({tView: tImgs[tKey] for tKey, tView in CAM_MAP.items()}, tState, self.task, noise, rtc=tRtc)
            self._rtc_info = None if tRtc is None else dict(tRtc["info"], **(REMOTE_LAST.get("rtc_log") or {}))
        else:
            tRtc = self._rtcBuild(tMNow) if RTC_ON else None
            self.policy.model._rtc = tRtc
            self.policy.model._rtc_log = None
            try:
                tAct = self._localForward(tImgs, tState, noise)
            finally:
                self.policy.model._rtc = None
            self._rtc_info = None if tRtc is None else dict(tRtc["info"], **(self.policy.model._rtc_log or {}))
        if not np.isfinite(tAct).all():
            raise RuntimeError("model output has NaN/Inf -- inference aborted")
        return self._postAct(tAct, tState, tMNow, tPosesNow, history)

    def _actionStats(self):
        """(mean, std) of the checkpoint's action normaliser, loaded once"""
        if not hasattr(self, "_act_ms"):
            from safetensors.torch import load_file
            tStats = load_file(os.path.join(self._ckpt_dir, ACT_STATS_FILE))
            self._act_ms = (tStats["action.mean"].float().numpy(), tStats["action.std"].float().numpy())
        return self._act_ms

    @staticmethod
    def _prefixRows(mNow, pos, quat, widths, rows):
        """absolute base-frame targets of the given rows -> (len(rows), 20) actions in the current-anchor frame of mNow"""
        tRaw = np.zeros((len(rows), 20))
        for tI, tJ in enumerate(rows):
            for tRow in ARM_ROWS:
                tTarget = np.eye(4)
                tTarget[:3, :3] = Rot.from_quat(quat[tJ][tRow]).as_matrix()
                tTarget[:3, 3] = pos[tJ][tRow]
                tRel = np.linalg.inv(mNow[tRow]) @ tTarget
                tRaw[tI, tRow * SLIM20_ARM_DIMS:tRow * SLIM20_ARM_DIMS + POSE9] = mat_to_pose10d(tRel[None])[0]
                tRaw[tI, tRow * SLIM20_ARM_DIMS + POSE9] = widths[tJ][tRow]
        return tRaw

    def _rtcBuild(self, m_now):
        """RTC prefix from the previous chunk (see RTC_ON): Y / W in the model's normalised action space, or None.
        (was _rtc_build)"""
        tPlan = getattr(self, "_rtc_plan_prefix", None)
        if tPlan is not None and ACTION_MODE == "umi":
            # [2026-10-06 PLAN mode] explicit prefix from the UI's plan table: rows anchor+1 .. as (P (2,3), Q (2,4) xyzw, W (2,))
            # absolute base-frame targets; new chunk step i <-> plan row anchor+1+i. Hard (W=1) for the d rows that run during
            # this forward (_rtc_delay_hint), then the LINEAR release to the end of the prefix.
            tMean, tStd = self._actionStats()
            tDimA = int(self.policy.model.dim_action)
            tChunk = int(self.chunk)
            tLen = min(len(tPlan), tChunk)
            if tLen < 1:
                return None
            tRaw = self._prefixRows(m_now, [tP for tP, _, _ in tPlan], [tQ for _, tQ, _ in tPlan], [tW for _, _, tW in tPlan],
                                    range(tLen))
            tY = torch.zeros(tChunk, tDimA)
            tY[:tLen, :20] = torch.as_tensor((tRaw - tMean[:20]) / (tStd[:20] + RTC_STD_EPS), dtype=torch.float32)
            tHint = getattr(self, "_rtc_delay_hint", None)
            tDelay = min(int(tHint) if tHint is not None else 0, tLen)
            tW = torch.zeros(tChunk, tDimA)
            tW[:, :20] = _rtcWeights(tDelay, tLen, tChunk)[:, None]
            return {"Y": tY, "W": tW, "info": dict(prev_idx=-1, nearest_mm=None, leftover=tLen, delay=tDelay, horizon=tLen,
                                                   age_s=0.0, source="plan")}
        tPrev = getattr(self, "_rtc_prev", None)
        if tPrev is None or time.time() - tPrev["t"] > RTC_MAX_AGE_S or ACTION_MODE != "umi":
            return None
        tMean, tStd = self._actionStats()
        tPos, tQuat, tWidths = tPrev["pos"], tPrev["quat"], tPrev["widths"]
        tSteps = len(tPos)
        tDist = sum(np.linalg.norm(tPos[:, tRow] - m_now[tRow][:3, 3], axis=1) for tRow in ARM_ROWS)
        tNearest = int(tDist.argmin())
        tLen = tSteps - 1 - tNearest
        if tLen < 1:
            return None
        tRaw = self._prefixRows(m_now, tPos, tQuat, tWidths, range(tNearest + 1, tNearest + 1 + tLen))
        tDimA = int(self.policy.model.dim_action)
        tChunk = int(self.chunk)
        tY = torch.zeros(tChunk, tDimA)
        tY[:tLen, :20] = torch.as_tensor((tRaw - tMean[:20]) / (tStd[:20] + RTC_STD_EPS), dtype=torch.float32)
        # [2026-10-06] delay = the waypoints the arm executes WHILE this chunk is inferred (the UI's prefetch sets _rtc_delay_hint
        # = lead + 1); those steps are pinned to the previous chunk (W = 1), then a linear release to the horizon. V4_RTC_HORIZON
        # <= 0 -> the whole leftover of the previous chunk.
        tHint = getattr(self, "_rtc_delay_hint", None)
        tDelay = min(int(tHint) if tHint is not None else RTC_DELAY, tLen)
        tHorizon = tLen if RTC_HORIZON <= 0 else min(max(RTC_HORIZON, tDelay), tLen)
        tW = torch.zeros(tChunk, tDimA)
        tW[:, :20] = _rtcWeights(tDelay, tHorizon, tChunk)[:, None]
        return {"Y": tY, "W": tW, "info": dict(prev_idx=tNearest, nearest_mm=round(float(tDist[tNearest]) * 1000 / 2, 1),
                                               leftover=tLen, delay=tDelay, horizon=tHorizon,
                                               age_s=round(time.time() - tPrev["t"], 2))}

    def _localForward(self, imgs, state, noise):
        """the policy forward on this machine (was _local_forward): (T, A) raw units"""
        tObs = {f"observation.images.{tView}": self._img(imgs[tKey], GLOBAL_ROT180 and tView == "global",
                                                         GLOBAL_MIRROR and tView == "global",
                                                         WRIST_ZOOM["s"] if tView == "right_wrist" else 1.0)
                for tKey, tView in CAM_MAP.items() if tKey in imgs}
        tMissing = [f"observation.images.{tView}" for tView in CAM_MAP.values() if f"observation.images.{tView}" not in tObs]
        if tMissing:
            raise ValueError(f"camera missing: {tMissing} (observation had {list(imgs)})")
        tObs["observation.state"] = torch.from_numpy(state)
        tObs["task"] = self.task

        tBatch = self.pre(tObs)
        with torch.no_grad():
            tChunk = (self.policy.predict_action_chunk(tBatch, noise=noise.to(self.device))
                      if noise is not None else self.policy.predict_action_chunk(tBatch))
        return self.post(tChunk)[0].detach().float().cpu().numpy().astype(np.float64)      # (T,20) raw units

    def _postAct(self, act, state, m_now, poses_now, history):
        """Decode one chunk (was _post_act): per-step clamp + composition to base-frame targets, temporal ensemble ->
        adaptive k -> near-zero hold, the chunk IK pre-check, joint / gripper commands (leader frame). Fills self.last;
        returns the kept rows."""
        tSteps = len(act)
        # left outputs had no loss: hold the left TCP (identity delta)
        if ARM_ONLY == "right":
            act = act.copy()
            act[:, 0:9] = _IDENTITY9
            tRightMats = pose10d_to_mat(act[:, 10:19])
            tPredMm = float(np.linalg.norm(tRightMats[:, :3, 3], axis=1).max()) * 1000.0
            tPredDeg = float(np.degrees(np.max([np.linalg.norm(Rot.from_matrix(tM[:3, :3]).as_rotvec()) for tM in tRightMats])))
            if ((ARM_ONLY_MAX_PRED_MM > 0 and tPredMm > ARM_ONLY_MAX_PRED_MM)
                    or (ARM_ONLY_MAX_PRED_DEG > 0 and tPredDeg > ARM_ONLY_MAX_PRED_DEG)):   # 0 = off
                raise RuntimeError(f"ARM_ONLY output guard: right-arm chunk predicts {tPredMm:.0f} mm / {tPredDeg:.0f} deg "
                                   f"(> {ARM_ONLY_MAX_PRED_MM:.0f} mm / {ARM_ONLY_MAX_PRED_DEG:.0f} deg) -- refused, "
                                   f"nothing executed")

        tTgtPos = np.zeros((tSteps, 2, 3))
        tTgtQuat = np.zeros((tSteps, 2, 4))
        tWidths = np.zeros((tSteps, 2))
        tRawMm = np.zeros((tSteps, 2))
        tRawDeg = np.zeros((tSteps, 2))
        tCmdMm = np.zeros((tSteps, 2))
        tNumClamped = 0
        for tRow in ARM_ROWS:
            tOff = tRow * SLIM20_ARM_DIMS
            tMats = pose10d_to_mat(act[:, tOff:tOff + POSE9])
            if ACTION_MODE == "umi":
                # current-anchor: every waypoint hangs off the SAME measured pose, no accumulation
                for tK in range(tSteps):
                    tA, tWasClamped, tMm, tDeg = clampDelta(tMats[tK], LIMITS["mm"] * (tK + 1),
                                                            LIMITS["deg"] * (tK + 1))
                    tRawMm[tK, tRow], tRawDeg[tK, tRow] = tMm, tDeg
                    tCmdMm[tK, tRow] = float(np.linalg.norm(tA[:3, 3])) * 1000.0
                    tNumClamped += int(tWasClamped)
                    tTarget = m_now[tRow] @ tA
                    tTgtPos[tK, tRow] = tTarget[:3, 3]
                    tTgtQuat[tK, tRow] = Rot.from_matrix(tTarget[:3, :3]).as_quat()
            else:
                # sequential body-frame composition, with each step clamped before it is applied
                tCur = m_now[tRow].copy()
                for tK in range(tSteps):
                    tD, tWasClamped, tMm, tDeg = clampDelta(tMats[tK], LIMITS["mm"], LIMITS["deg"])
                    tRawMm[tK, tRow], tRawDeg[tK, tRow] = tMm, tDeg
                    tCmdMm[tK, tRow] = float(np.linalg.norm(tD[:3, 3])) * 1000.0
                    tNumClamped += int(tWasClamped)
                    tCur = tCur @ tD                                 # body frame -> RIGHT multiply
                    tTgtPos[tK, tRow] = tCur[:3, 3]
                    tTgtQuat[tK, tRow] = Rot.from_matrix(tCur[:3, :3]).as_quat()
            tWidths[:, tRow] = act[:, tOff + POSE9]

        # [2026-10-01] temporal ensemble -> adaptive k -> near-zero hold (see TE_ON / ADAPTIVE_K / DB_* above)
        tTeInfo = tDbInfo = tAkInfo = None
        if TE_ON:
            tNowT = time.time()
            if getattr(self, "_te_hist", None) is None or tNowT - getattr(self, "_te_last_t", 0.0) > TE_MAXGAP_S:
                self._te_hist = collections.deque(maxlen=len(TE_W))
                self._te_step = 0
            self._te_last_t = tNowT
            # [0] = current chunk (raw)
            self._te_hist.appendleft((self._te_step, tTgtPos.copy(), tTgtQuat.copy(), tWidths.copy()))
            tNumSrc = np.zeros(tSteps, int)
            tRawPos = tTgtPos.copy()
            for tK in range(tSteps):
                tPs, tQs, tWs, tWt = [], [], [], []
                for tJ, (tStep0, tP, tQ, tW) in enumerate(self._te_hist):
                    tIdx = tK + (self._te_step - tStep0)
                    if tIdx >= tSteps:
                        continue
                    tPs.append(tP[tIdx])
                    tQs.append(tQ[tIdx])
                    tWs.append(tW[tIdx])
                    tWt.append(TE_W[tJ])
                tWt = np.asarray(tWt)
                tNumSrc[tK] = len(tWt)
                if len(tWt) > 1:
                    tWn = tWt / tWt.sum()
                    tTgtPos[tK] = np.einsum("j,jrc->rc", tWn, np.asarray(tPs))
                    if TE_GRIP:
                        tWidths[tK] = tWn @ np.asarray(tWs)
                    if TE_ROT:
                        for tRow in ARM_ROWS:
                            tTgtQuat[tK, tRow] = Rot.from_quat(np.asarray(tQs)[:, tRow]).mean(weights=tWt).as_quat()
            tTeInfo = dict(n_src=tNumSrc.tolist(), w=TE_W.tolist(), rot=TE_ROT, grip=TE_GRIP, _raw=tRawPos)
        tKExec = max(1, EXEC_K)
        if ADAPTIVE_K is not None:
            tDistMm = max(float(np.linalg.norm(tTgtPos[tSteps - 1, tRow] - m_now[tRow][:3, 3])) * 1000.0 for tRow in ARM_ROWS)
            tGripCross = (bool(((tWidths.max(0) > GRIP_BIN_THRESH) & (tWidths.min(0) < GRIP_BIN_THRESH)).any())
                          if GRIPPER_MODE != "hold" else False)
            if tGripCross or tDistMm <= AK_NEAR_MM:
                tKExec, tPhase = ADAPTIVE_K[2], ("align(grip)" if tGripCross else "near")
            elif tDistMm <= AK_FAR_MM:
                tKExec, tPhase = ADAPTIVE_K[1], "medium"
            else:
                tKExec, tPhase = ADAPTIVE_K[0], "far"
            tAkInfo = dict(k=tKExec, phase=tPhase, dist_end_mm=round(tDistMm, 1), grip_cross=tGripCross)
        tKExec = min(tKExec, tSteps)
        tKe = tKExec - 1
        if tTeInfo is not None:
            tRawPos = tTeInfo.pop("_raw")
            tTeInfo["n_src_exec"] = int(tNumSrc[tKe])
            tTeInfo["shift_mm"] = [[round(float(tX) * 1000, 1) for tX in (tTgtPos[tKe, tRow] - tRawPos[tKe, tRow])]
                                   for tRow in ARM_ROWS]
            self._te_step += tKExec                       # the next inference happens tKExec steps later
        if DEADBAND_MM is not None or DB_XY_MM is not None or DB_Z_MM is not None:
            tHeld = {"L": [False] * 3, "R": [False] * 3}
            # the hold is decided ONCE per arm on the displacement at a fixed reference horizon (DB_REF_K), not on the executed
            # waypoint: with adaptive k = 1-2 a real approach moves only a few mm per waypoint and would always be held
            tKRef = min(tSteps, DB_REF_K) - 1
            tDRef = {}
            for tRow in ARM_ROWS:
                tP0 = m_now[tRow][:3, 3]
                tDisp = (tTgtPos[tKRef, tRow] - tP0) * 1000.0
                tHold = np.zeros(3, bool)
                if DEADBAND_MM is not None:
                    tHold |= np.abs(tDisp) < DEADBAND_MM
                if DB_XY_MM is not None and np.linalg.norm(tDisp[:2]) < DB_XY_MM:
                    tHold[:2] = True
                if DB_Z_MM is not None and abs(tDisp[2]) < DB_Z_MM:
                    tHold[2] = True
                tTgtPos[:, tRow, tHold] = tP0[tHold]
                tHeld["LR"[tRow]] = [bool(tH) for tH in tHold]
                tDRef["LR"[tRow]] = [round(float(tX), 1) for tX in tDisp]
            tDbInfo = dict(xy_mm=DB_XY_MM, z_mm=DB_Z_MM, axis_mm=None if DEADBAND_MM is None else DEADBAND_MM.tolist(),
                           ref_k=tKRef + 1, d_ref_mm=tDRef, held_exec=tHeld)
        self.te_info, self.db_info, self.ak_info, self.exec_k_now = tTeInfo, tDbInfo, tAkInfo, tKExec

        tQNow = np.asarray([float(tX) for tX in history[-1]["joints14"]], np.float64)
        tQa = tQNow[ARM_IDX]
        tQCurobo = None
        if IK_BACKEND == "curobo":
            tQTgt, tFkErr, tOk, self._curobo_info = curoboSolve(tQa, tTgtPos, tTgtQuat)
            tQCurobo = tQTgt.copy()
            tBad = np.flatnonzero(tFkErr.max(1) > FK_MAX_MM)
            tValidT = int(tBad[0]) if len(tBad) else tSteps
            print(f"[v4] curobo chunk {tSteps} wp: {self._curobo_info['ms']} ms server, pos err max {tFkErr.max():.1f} mm, "
                  f"valid {tValidT}", flush=True)
            if tValidT < 1:
                raise RuntimeError(f"cuRobo could not reach the first predicted pose (FK error {tFkErr[0].max():.0f} mm)")
        elif IK_BACKEND in ("pink", "learned", "policydq", "pinkdq", "diffik"):
            # [2026-09-29] the validity pre-check used PyRoki WITHOUT the frame fix whatever backend executes; with pink/learned
            # each executed waypoint is guarded by its own backend (FK > FK_MAX_MM / NaN -> stop), so no PyRoki pre-check here.
            tQTgt = np.repeat(tQa[None, :], tSteps, 0)
            tFkErr = np.zeros((tSteps, 2))
            tOk = np.ones((tSteps, 2), bool)
            tValidT = tSteps
        else:
            tQTgt, tFkErr, tOk = self.kin.solveChunk(tQa, tTgtPos, tTgtQuat)
            tBad = np.flatnonzero((tFkErr.max(1) > FK_MAX_MM) | ~tOk.all(1))
            tValidT = int(tBad[0]) if len(tBad) else tSteps
            if tValidT < 1:
                raise RuntimeError(f"IK could not reach the first predicted pose (FK error {tFkErr[0].max():.0f} mm)")
        tDq = tQTgt - tQa[None, :]
        tQCmd = np.clip(tQa[None, :] + np.clip(tDq, -DQ_MAX, DQ_MAX), LO, HI)

        if ARM_ONLY == "right":
            tQCmd[:, 0:6] = tQa[None, 0:6]                            # left arm joints: exactly where they are
        tCmd = np.zeros((tSteps, 14))
        tCmd[:, ARM_IDX] = tQCmd
        for tRow, tGi in enumerate(GRIP_IDX):
            if GRIPPER_MODE in ("predict", "binary", "continuous", "relative"):
                tCmd[:, tGi] = [gripToCmd(tW) for tW in tWidths[:, tRow]]
            else:
                tCmd[:, tGi] = float(tQNow[tGi])                      # HOLD: re-send what the jaw already is
            if ARM_ONLY and (tRow == 0 or ARM_ONLY_GRIP == "hold"):
                tCmd[:, tGi] = jawHoldCmd(tQNow[tGi])                 # one-arm model: unsupervised jaws hold (as a 0..45 COMMAND)
        tCmd[:, FLIP_IDX] *= -1.0

        tKeep = min(tValidT, self.n_action)
        self.last = dict(act=act.copy(), tgt_pos=tTgtPos.copy(), tgt_quat=tTgtQuat.copy(), widths=tWidths.copy(),
                         fkerr_mm=tFkErr.copy(), dq=tDq.copy(), q_cmd=tQCmd.copy(), q_now=tQNow.copy(),
                         valid_step=tValidT, kept=tKeep, clipped=int((np.abs(tDq) > DQ_MAX).sum()),
                         cur_tcp=poses_now.copy(), cur_mat=m_now.copy(), state=state.copy(),
                         pred_mm=tRawMm.copy(), pred_deg=tRawDeg.copy(), cmd_mm=tCmdMm.copy(),
                         clamped_steps=tNumClamped, gripper_mode=GRIPPER_MODE, action_mode=ACTION_MODE,
                         grip_cmd=np.array([[gripToCmd(tWidths[tK, tRow]) for tRow in ARM_ROWS] for tK in range(tSteps)]))
        # [2026-09-29] IK_BACKEND=policydq / pinkdq: the policy's own dq head (action dims 20:32 = q(t+k) - q(t), follower frame,
        # rad)
        self.last["dq_pol"] = act[:, DQ_HEAD].copy() if act.shape[1] >= DQ_HEAD.stop else None
        self.last["q_arm0"] = tQa.copy()
        # [2026-10-03] RTC: this chunk's targets as sent (after TE / deadband) are the next chunk's inpainting prefix
        self._rtc_prev = dict(t=time.time(), pos=tTgtPos.copy(), quat=tTgtQuat.copy(), widths=tWidths.copy())
        self.last["rtc"] = getattr(self, "_rtc_info", None)
        self._rtc_info = None
        self.last["q_curobo"] = tQCurobo
        self.calls += 1
        return tCmd[:tKeep]

    def _learnedWaypoint(self, qa, joints14_now, tgt_pos_k):
        """[2026-09-29] IK_BACKEND=learned: LearnedIK-v0 (umi_bridge/learned_ik, LIK0-P12 best.pt) on the measured q and the REL of the
        whole current chunk re-expressed from the MEASURED TCP (dataset frame); returns the dq of the waypoint being executed.
        Guard: NaN or |FK(q_t + dq) - target| > FK_MAX_MM -> None (the caller falls back to numerical IK and logs it)."""
        try:
            if not hasattr(self, "_lik"):
                prepend_sys_path(str(UMI_BRIDGE / "learned_ik")); import lik0_common as _LC
                ck = torch.load(os.path.expanduser(os.environ.get("V4_LIK_CKPT", str(UMI_BRIDGE / "learned_ik/runs/LIK0-P12/best.pt"))), map_location="cpu", weights_only=False)
                self._lik = _LC.LIK0(ck["stats"]); self._lik.load_state_dict(ck["model"]); self._lik.eval()
                print(f"[v4] learned IK loaded ({os.environ.get('V4_LIK_CKPT', 'LIK0-P12/best.pt')}, step {ck['step']})", flush=True)
            L = self.last; m_meas, _ = self._tcpMat(joints14_now)
            k = int(np.argmin(np.linalg.norm(L["tgt_pos"].reshape(len(L["tgt_pos"]), -1) - np.asarray(tgt_pos_k).reshape(1, -1), axis=1)))
            rel = np.zeros((16, 18))
            for kk in range(16):
                for r in (0, 1):
                    Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(L["tgt_quat"][kk, r]).as_matrix(); Tk[:3, 3] = L["tgt_pos"][kk, r]
                    A = np.linalg.inv(m_meas[r]) @ Tk
                    rel[kk, r * 9:r * 9 + 3] = A[:3, 3]; rel[kk, r * 9 + 3:r * 9 + 9] = A[:2, :3].reshape(6)
            # input check: REL re-expressed from the measured TCP vs the policy's own REL (identical when the arm is still)
            act = L.get("act")
            if act is not None:
                self._rel_diff_mm = round(float(max(np.abs(rel[:, [0, 1, 2, 9, 10, 11]] - act[:, [0, 1, 2, 10, 11, 12]]).max(), 0) * 1000), 2)
            with torch.no_grad():
                dq = self._lik(torch.tensor(qa, dtype=torch.float32)[None], torch.tensor(rel, dtype=torch.float32)[None])[0, k].double().numpy()
            q_t = qa + dq
            if not np.isfinite(q_t).all():
                self._lik_fallback = "nan"; return None, None, None
            # [2026-09-29 user] no extra stop rules: large dq / joint limits are handled by the common DQ_MAX + LO/HI clip like
            # every backend
            q14 = np.zeros(14); q14[ARM_IDX] = q_t; m_new, _ = self._tcpMat(q14)
            fe = np.array([[np.linalg.norm(m_new[r][:3, 3] - np.asarray(tgt_pos_k)[r]) * 1000 for r in (0, 1)]])
            self._lik_fallback = None
            return q_t[None], fe, np.ones((1, 2), bool)
        except Exception as e:  # noqa: BLE001
            self._lik_fallback = f"error {e}"; print("[v4] learned IK failed, numerical fallback:", e, flush=True)
            return None, None, None

    def _gripRelative(self, r, g, raw_now):
        """[2026-09-29] V4_GRIPPER=relative: command from the CHANGE of the predicted g between cycles, with a deadband.
        dg > +eps -> OPEN (GRIP_BIN_OPEN), dg < -eps -> CLOSE (GRIP_BIN_CLOSED), else HOLD the last command. The first cycle of a
        rollout has no previous g: it holds the MEASURED jaw state (open if the jaw reads > half travel). Reset on /run.
        (was _grip_relative)"""
        if not hasattr(self, "_g_prev"):
            self._g_prev = [None, None]
            self._g_cmd = [None, None]
        tG = float(g) / GRIP_REL_SCALE
        tPrev = self._g_prev[r]
        if tPrev is None:
            tAction = "init"
            tCmd = GRIP_BIN_OPEN if rawToWidth(raw_now) > OPEN_JAW_FRACTION * CART20_W_OPEN else GRIP_BIN_CLOSED
        elif GRIP_REL_MODE == "acc":
            tHist = self._g_hist[r] if hasattr(self, "_g_hist") else None
            if tHist is None:
                self._g_hist = getattr(self, "_g_hist", [[], []])
                tHist = self._g_hist[r]
            tHist.append(tG - tPrev)
            del tHist[:-GRIP_ACC_N]
            tSum = sum(tHist)
            if tSum > GRIP_ACC_OPEN:
                tAction = "open"
                tCmd = GRIP_BIN_OPEN
                tHist.clear()
            elif tSum < GRIP_ACC_CLOSE:
                tAction = "close"
                tCmd = GRIP_BIN_CLOSED
                tHist.clear()
            else:
                tAction = "hold"
                tCmd = self._g_cmd[r] if self._g_cmd[r] is not None else GRIP_BIN_CLOSED
            tAction = f"{tAction} (acc {tSum:+.3f})"
        elif tG > tPrev + GRIP_REL_EPS:
            tAction = "open"
            tCmd = GRIP_BIN_OPEN
        elif tG < tPrev - GRIP_REL_EPS:
            tAction = "close"
            tCmd = GRIP_BIN_CLOSED
        else:
            tAction = "hold"
            tCmd = self._g_cmd[r] if self._g_cmd[r] is not None else GRIP_BIN_CLOSED
        self._g_prev[r] = tG
        self._g_cmd[r] = tCmd
        self._grip_rel_log = getattr(self, "_grip_rel_log", [None, None])
        self._grip_rel_log[r] = dict(g_prev=tPrev, g=round(tG, 3), act=tAction, cmd=tCmd)
        return tCmd

    def reset_anchor(self):
        """RELCART20: forget the anchor; the next observation becomes the new one (rollout start)."""
        self._anchor = None
        self._te_hist = None                                           # [2026-10-01] temporal ensemble: new rollout
        self._g_prev = [None, None]                                    # relative gripper: new rollout
        self._g_cmd = [None, None]
        self._g_hist = [[], []]

    def solve_waypoint(self, joints14_now, tgt_pos_k, tgt_quat_k, widths_k=None):
        """Re-solve ONE nominal Cartesian target from the MEASURED joints.

        The chunk's targets T1..T4 come from the model and are never recomputed -- only the IK seed changes.
        Solving the whole chunk once (solve_chunk from the pre-execution pose) stacks each step's solution on
        the previous PREDICTED joints, so the FK residual grew 2.1 -> 4.3 -> 6.7 -> 9.2 mm along one chunk as
        the robot fell behind. Seeding from where the arm actually is cuts that accumulation.
        """
        q_now = np.asarray([float(x) for x in joints14_now], np.float64)
        qa = q_now[ARM_IDX]
        # [10-06] HandUMI convention -> robot TCP (dataset frame), for EVERY IK backend
        if UMI_CONV:
            Rr_ = [Rot.from_quat(q_).as_matrix() @ R_UMI_OFF.T for q_ in np.asarray(tgt_quat_k, np.float64)]
            tgt_quat_k = np.stack([Rot.from_matrix(R_).as_quat() for R_ in Rr_])
            tgt_pos_k = np.stack([np.asarray(p_, np.float64) - R_ @ UMI_TCP_OFF_T for p_, R_ in zip(np.asarray(tgt_pos_k, np.float64), Rr_)])
        tq = np.asarray(tgt_quat_k, np.float64).copy()
        if USE_FRAME_FIX:                                    # dataset frame -> FK frame, for the IK target
            for r in range(tq.shape[0]):
                tq[r] = Rot.from_matrix(Rot.from_quat(tq[r]).as_matrix() @ R_DS_TO_FK).as_quat()
        self._lik_stop = None; self._shadow = None
        if IK_BACKEND == "learned":
            q_tgt, fkerr, ok = self._learnedWaypoint(qa, joints14_now, tgt_pos_k)
            if q_tgt is None:
                # [2026-09-29 advisor] no silent hybrid: the pure-learned A/B stops this cycle and records why (no numerical
                # fallback)
                self._lik_stop = self._lik_fallback or "learned IK failed"; self._ik_used = "learned_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
            else:
                self._ik_used = "learned"
                # shadow numerical: logged, NOT executed
                qs, fs, oks = self.kin.solveChunk(qa, np.asarray(tgt_pos_k)[None], tq[None])
                self._shadow = dict(q=[round(float(x), 4) for x in qs[0]], fk_err_mm=round(float(fs.max()), 2),
                                    dq_diff_vs_learned=[round(float(x), 4) for x in (qs[0] - q_tgt[0])])
        elif IK_BACKEND in ("policydq", "pinkdq"):
            # [2026-09-29] option 5: the policy's own joint head. q_ref = q(inference) + dq_pol[k]  (the same q_t the label used).
            #   policydq: execute q_ref directly (joint-space, no IK).   pinkdq: Pink to the Cartesian target, seeded at q_ref and
            #   pulled toward it (posture cost V4_PINKDQ_POSTURE, default 1e-2) -> demo-like branch + exact FK.
            L = self.last; dqp = L.get("dq_pol")
            if dqp is None:
                raise RuntimeError(f"IK_BACKEND={IK_BACKEND} needs a 32-dim action with the dq12 head (this ckpt has {L['act'].shape[1]})")
            k = int(np.argmin(np.linalg.norm(L["tgt_pos"].reshape(len(L["tgt_pos"]), -1) - np.asarray(tgt_pos_k).reshape(1, -1), axis=1)))
            q_ref = L["q_arm0"] + dqp[k]
            q14 = np.zeros(14); q14[ARM_IDX] = q_ref; m_ref, _ = self._tcpMat(q14)
            fe_ref = [round(float(np.linalg.norm(m_ref[r][:3, 3] - np.asarray(tgt_pos_k)[r]) * 1000), 2) for r in (0, 1)]
            if IK_BACKEND == "policydq":
                self._ik_used = "policydq"; q_tgt, fkerr, ok = q_ref[None], np.array([fe_ref]), np.ones((1, 2), bool)
                self._shadow = dict(k=k, fk_vs_cart_target_mm=fe_ref)
            else:
                if not hasattr(self, "_pink"):
                    from . import pink_ik
                    self._pink = pink_ik.PinkIK([self.kin.mJointNames[j] for j in self.kin.mArmJointIdx["left"]] + [self.kin.mJointNames[j] for j in self.kin.mArmJointIdx["right"]])
                T = []
                for r in range(2):
                    Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(np.asarray(tgt_quat_k, np.float64)[r]).as_matrix(); Tk[:3, 3] = np.asarray(tgt_pos_k)[r]; T.append(Tk)
                qp, perr, it = self._pink.solve(q_ref, T, postureQ12=q_ref, postureCost=float(os.environ.get("V4_PINKDQ_POSTURE", "1e-2")), ori=PINK_ORI, holdQ12=qa, lock=PINK_LOCK)
                if not np.isfinite(qp).all():
                    self._lik_stop = "pinkdq returned NaN"; self._ik_used = "pinkdq_STOP"
                    q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
                else:
                    self._ik_used = "pinkdq"; q_tgt, fkerr, ok = qp[None], perr[None], np.ones((1, 2), bool)
                self._shadow = dict(k=k, policy_fk_vs_cart_target_mm=fe_ref, pink_iters=int(it),
                                    dq_diff_pink_vs_policy=[round(float(x), 4) for x in (q_tgt[0] - q_ref)])
        elif IK_BACKEND == "pink":
            # [2026-09-29] Pinocchio/Pink QP IK to convergence (pink_ik.py); targets in the DATASET frame (its TCP == dataset
            # TCP), so tgt_quat_k is used as given (no R_DS_TO_FK). NaN -> stop this cycle (no fallback). Shadow PyRoki logged.
            if not hasattr(self, "_pink"):
                from . import pink_ik
                self._pink = pink_ik.PinkIK([self.kin.mJointNames[j] for j in self.kin.mArmJointIdx["left"]] + [self.kin.mJointNames[j] for j in self.kin.mArmJointIdx["right"]])
                print(f"[v4] pink IK ready (pos 1.0 / ori 0.5 / posture {os.environ.get('V4_PINK_POSTURE', '1e-3')} to the seed, quadprog, ori={PINK_ORI})", flush=True)
            T = []
            for r in range(2):
                Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(np.asarray(tgt_quat_k, np.float64)[r]).as_matrix(); Tk[:3, 3] = np.asarray(tgt_pos_k)[r]; T.append(Tk)
            qp, perr, it = self._pink.solve(qa, T, ori=PINK_ORI, lock=PINK_LOCK, postureCost=float(os.environ.get("V4_PINK_POSTURE", "1e-3")))   # [2026-09-30] posture prior to the measured q (default unchanged)
            # [2026-10-01 user, option C] joint5 (wrist_yaw) free but slew-limited: if the full-pose solve moves joint5 by more
            # than V4_YAW_MAX_STEP_DEG on an arm, joint5 is pinned at q_now +- cap and the pose is re-solved, so the other joints
            # absorb what they can (instead of clipping joint5 after the fact). Diagnostics per waypoint in self._yaw_diag:
            # target-vs-current TCP rotation (total, base-z part), IK joint5 change, clamped flag.
            J5 = (4, 10)
            import pinocchio as pin
            try:
                Tc = self._pink.fk(qa)
                rot_tot = [round(float(np.degrees(np.linalg.norm(pin.log3(T[r][:3, :3] @ Tc[r][:3, :3].T)))), 2) for r in (0, 1)]
                rot_z = [round(float(np.degrees(pin.log3(T[r][:3, :3] @ Tc[r][:3, :3].T)[2])), 2) for r in (0, 1)]
            except Exception:
                rot_tot = rot_z = None
            dj5 = [float(np.degrees(qp[j] - qa[j])) for j in J5]; clamped = [False, False]
            mode = "none"
            if YAW_MAX_STEP_DEG > 0 and not any("joint5" in x for x in PINK_LOCK) and np.isfinite(qp).all() and max(abs(d) for d in dj5) > YAW_MAX_STEP_DEG:
                # 1) shrink the ORIENTATION step (slerp the target rotation toward the current one by cap/|dj5|) and re-solve with
                #    joint5 free: position stays exact, the wrist turns at most ~cap this waypoint and catches up on the next ones
                T2 = [t_.copy() for t_ in T]
                for r in (0, 1):
                    clamped[r] = abs(dj5[r]) > YAW_MAX_STEP_DEG
                    if clamped[r]:
                        frac = YAW_MAX_STEP_DEG / abs(dj5[r]); Rc = Tc[r][:3, :3]
                        T2[r][:3, :3] = Rc @ pin.exp3(frac * pin.log3(Rc.T @ T[r][:3, :3]))
                q2, p2, it2 = self._pink.solve(qa, T2, ori=PINK_ORI, lock=PINK_LOCK, postureCost=float(os.environ.get("V4_PINK_POSTURE", "1e-3")))
                mode = "ori_slerp"
                # 2) fallback: still over the cap -> pin joint5 at q_now +- cap on the shrunk target
                if np.isfinite(q2).all() and max(abs(float(np.degrees(q2[j] - qa[j]))) for j in J5) > YAW_MAX_STEP_DEG + 0.5:
                    hold = np.asarray(q2, np.float64).copy(); cap = np.radians(YAW_MAX_STEP_DEG)
                    for j in J5: hold[j] = qa[j] + np.clip(q2[j] - qa[j], -cap, cap)
                    q3, p3, it3 = self._pink.solve(qa, T2, ori=PINK_ORI, lock=tuple(PINK_LOCK) + ("joint5",), holdQ12=hold,
                                                   postureCost=float(os.environ.get("V4_PINK_POSTURE", "1e-3")))
                    if np.isfinite(q3).all():
                        q2, p2, it2, mode = q3, p3, it2 + it3, "ori_slerp+pin"
                if np.isfinite(q2).all():
                    qp, perr, it = q2, p2, it + it2
            self._yaw_diag = dict(tgt_rot_deg=rot_tot, tgt_rot_z_deg=rot_z, ik_j5_delta_deg=[round(d, 2) for d in dj5],
                                  j5_clamped=clamped, cap_mode=mode, j5_cmd_delta_deg=[round(float(np.degrees(qp[j] - qa[j])), 2) for j in J5], cap_deg=YAW_MAX_STEP_DEG)
            if not np.isfinite(qp).all():
                self._lik_stop = "pink returned NaN"; self._ik_used = "pink_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
            else:
                self._ik_used = "pink"; q_tgt, fkerr, ok = qp[None], perr[None], np.ones((1, 2), bool)
                qs, fs, _ = self.kin.solveChunk(qa, np.asarray(tgt_pos_k)[None], tq[None])
                self._shadow = dict(q=[round(float(x), 4) for x in qs[0]], fk_err_mm=round(float(fs.max()), 2), pink_iters=int(it),
                                    dq_diff_vs_pink=[round(float(x), 4) for x in (qs[0] - qp)])
        elif IK_BACKEND == "curobo":
            # chunk path from step() (solved once from the measured q at observation time); a target not in the chunk (stream
            # blend) is solved alone from the measured q
            L = getattr(self, "last", {}) or {}; Qc = L.get("q_curobo"); j = None
            if Qc is not None:
                d = np.abs(np.asarray(L["tgt_pos"])[:len(Qc)] - np.asarray(tgt_pos_k)[None]).reshape(len(Qc), -1).max(1)
                j = int(d.argmin()) if d.min() < 1e-9 else None
            try:
                if j is not None:
                    q_tgt, fkerr, ok = Qc[j][None], np.asarray(L["fkerr_mm"])[j][None], np.ones((1, 2), bool); self._ik_used = f"curobo[k{j + 1}]"
                else:
                    qq, fe, okk, _ = curoboSolve(qa, np.asarray(tgt_pos_k)[None], np.asarray(tgt_quat_k)[None])
                    q_tgt, fkerr, ok = qq, fe, np.ones((1, 2), bool); self._ik_used = "curobo[single]"
                if fkerr.max() > FK_MAX_MM or not np.isfinite(q_tgt).all():
                    raise RuntimeError(f"FK error {fkerr.max():.0f} mm")
            except Exception as e:
                self._lik_stop = f"curobo failed: {e}"; self._ik_used = "curobo_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
        elif IK_BACKEND == "diffik":
            # [2026-10-01 user] differential IK: ONE damped-least-squares Jacobian step from the MEASURED q toward the same
            # Cartesian target (pink_ik.diffStep), instead of solving to convergence. Same TCP frame/targets as pink; V4_PINK_LOCK
            # columns zeroed; V4_DIFFIK_ORI_W (0.5 = pink's ori/pos cost ratio), V4_DIFFIK_DAMP (1e-2). Shadow: the converged pink
            # solution.
            if not hasattr(self, "_pink"):
                from . import pink_ik
                self._pink = pink_ik.PinkIK([self.kin.mJointNames[j] for j in self.kin.mArmJointIdx["left"]] + [self.kin.mJointNames[j] for j in self.kin.mArmJointIdx["right"]])
            T = []
            for r in range(2):
                Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(np.asarray(tgt_quat_k, np.float64)[r]).as_matrix(); Tk[:3, 3] = np.asarray(tgt_pos_k)[r]; T.append(Tk)
            qp, perr, req = self._pink.diffStep(qa, T, oriWeight=float(os.environ.get("V4_DIFFIK_ORI_W", "0.5")),
                                                 damping=float(os.environ.get("V4_DIFFIK_DAMP", "1e-2")), lock=PINK_LOCK)
            if not np.isfinite(qp).all():
                self._lik_stop = "diffik returned NaN"; self._ik_used = "diffik_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
            else:
                self._ik_used = "diffik"; q_tgt, fkerr, ok = qp[None], perr[None], np.ones((1, 2), bool)
                qc, pc, it = self._pink.solve(qa, T, ori=PINK_ORI, lock=PINK_LOCK)
                self._shadow = dict(requested_mm=[round(float(x), 1) for x in req], diffik_residual_mm=[round(float(x), 1) for x in perr],
                                    pink_residual_mm=[round(float(x), 1) for x in pc], dq_diff_vs_pink=[round(float(x), 4) for x in (qc - qp)])
        else:
            self._ik_used = "numerical"
            q_tgt, fkerr, ok = self.kin.solveChunk(qa, np.asarray(tgt_pos_k)[None], tq[None])
        dq = q_tgt - qa[None, :]
        q_cmd = np.clip(qa[None, :] + np.clip(dq, -DQ_MAX, DQ_MAX), LO, HI)[0]
        # [2026-09-29] per-waypoint IK log: how often the outer DQ_MAX 0.6 and LO/HI bite once the solver clip is off
        try:
            q_uncl = qa + np.clip(dq[0], -DQ_MAX, DQ_MAX)
            rec = dict(t=round(time.time(), 3), ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")),
                       exec_k=int(os.environ.get("V4_EXEC_K", "0") or 0), state_mode=STATE_MODE, ik_backend_requested=IK_BACKEND, pink_ori=PINK_ORI, pink_lock=list(PINK_LOCK), ik_backend_executed=getattr(self, "_ik_used", IK_BACKEND), lik_stop_reason=getattr(self, "_lik_stop", None), shadow_numerical=getattr(self, "_shadow", None), rel_input_diff_mm=getattr(self, "_rel_diff_mm", None), ik_max_joint_delta=self.kin.mSolver.config.max_joint_delta,
                       max_abs_dq=round(float(np.abs(dq).max()), 4), abs_dq=[round(float(x), 4) for x in np.abs(dq[0])],
                       dqmax_hits=[int(i) for i in np.flatnonzero(np.abs(dq[0]) > DQ_MAX)],
                       limit_hits=[int(i) for i in np.flatnonzero((q_uncl < LO) | (q_uncl > HI))],
                       # rad past LO/HI per joint
                       limit_excess=[round(float(x), 4) for x in np.maximum(LO - q_uncl, q_uncl - HI).clip(0)],
                       q_meas=[round(float(x), 4) for x in qa],
                       fk_err_mm=round(float(fkerr.max()), 2), ok=bool(ok.all()))
            self.last_wp = rec
            with open(IK_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001 -- logging must never break a waypoint
            pass
        if ARM_ONLY == "right":                                       # one-arm model: the left arm never moves (all IK backends)
            q_cmd = np.array(q_cmd, np.float64).reshape(-1, 12) if np.ndim(q_cmd) == 2 else np.array(q_cmd, np.float64)
            if q_cmd.ndim == 2: q_cmd[:, 0:6] = qa[None, 0:6]
            else: q_cmd[0:6] = qa[0:6]
        cmd = np.zeros(14)
        cmd[ARM_IDX] = q_cmd
        for r, gi in enumerate(GRIP_IDX):
            cmd[gi] = (self._gripRelative(r, widths_k[r], q_now[gi]) if (GRIPPER_MODE == "relative" and widths_k is not None) else
                       gripToCmd(widths_k[r]) if (GRIPPER_MODE in ("predict", "binary", "continuous", "relative") and widths_k is not None)
                       else float(q_now[gi]))
            if ARM_ONLY and (r == 0 or ARM_ONLY_GRIP == "hold"):
                cmd[gi] = jawHoldCmd(q_now[gi])                     # one-arm model: unsupervised jaws hold (as a 0..45 COMMAND)
        cmd[FLIP_IDX] *= -1.0
        return cmd, float(fkerr.max()), bool(ok.all()), int((np.abs(dq) > DQ_MAX).sum())

    def tcp_now(self, joints14):
        """FK of an observation, for measuring what the arm actually did."""
        tMats, tPoses = self._tcpMat(joints14)
        return tMats, tPoses
