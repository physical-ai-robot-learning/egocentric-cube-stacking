"""HEAD180 UMI76 converter: UMI zarr (absolute TCP) -> LeRobot dataset, upstream bimanual UMI observation semantics at
upstream's physical timing (dataset stage, node). Contract: UMI76_CONTRACT.md (next to this file).

[2026-09-25] Derived from umi_to_lerobot_v4_3cam.py; the video decoding and dataset writing are unchanged and only the
state/action construction is new. R150 and, later, HandUMI both pass through THIS converter, so "same pipeline" means the
same UMI -> relative -> LeRobot contract, not the same runtime Dataset object. X-VLA's training code is not modified at all.

Per query row t (on the stored 15 Hz grid), with dt = 3/59.94 s (upstream's 50.05 ms):

    action[t]            (16, 20)  per arm [pos3 + rot6d + gripper] of inv(T_t) @ T(t + (k+1) dt), k = 0..15
    observation.state[t] (76,)     upstream low-dim packing: per arm pos / pos_wrt / rot / rot_wrt / grip at [t - dt, t],
                                   self terms inv(T_t) @ T, cross terms inv(T_other_t) @ T
    images                         global (source mp4, optional rot180), left_wrist (arm 0), right_wrist (arm 1), 224x224

So all geometry is baked here and X-VLA sees plain tensors. Rows whose history or 16 targets leave the episode are not
written (nothing is padded); `state_prev_valid` is therefore always 1. No identity action is ever written as if it were a
real label.
Inputs -> outputs: one or more `--source zarr=..,lerobot=..` pairs -> a LeRobot v3 dataset at --out + v4_provenance.json.
Usage: python umi76_to_lerobot.py --source zarr=<z>,lerobot=<root>[,global_transform=rot180,episodes_json=..,select=..]
       --out <dir> --repo-id <id> [--action-mode umi --state-mode slim20 --gripper-target next]   (node launchers:
       scripts/train/node4090/build_r312c_umi76.sh, build_chain_r384.sh)
--state-mode / --gripper-target / --action-mode only change what v4_provenance.json records; the rows are always the UMI76
state and the 16-step current-anchor chunk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp

from ego_paths import NODE_HOME, prepend_sys_path  # machine-specific roots (env-overridable)

SOFT_DP_MM, SOFT_DTH_DEG = 40.0, 12.0
HARD_DP_MM, HARD_DTH_DEG = 70.0, 40.0
N_ROBOT = 2
# state layout recorded in the provenance. full38 keeps the cross-arm term; slim20 drops it so the state is shape-compatible
# with the pretrained xvla-base action encoder (max_state_dim=20 -- setting 38 grows action_encoder.fc from [30, 73728] to
# [30, 92160] and strict load fails). slim20 = per arm [prev_rel pos3+rot6d, gripper].
STATE_MODE = "full38"
# "delta" = the v4 per-frame body increment; "umi" = the shared current-anchor chunk (umi_action.build_chunk)
ACTION_MODE = "delta"
# "current" keeps the original (and frozen) semantics: action gripper = grip[t], identical to the state's
# gripper. "next" = grip[t+1], which puts the gripper on the same instant as the pose target inv(T_t)@T_{t+1}.
GRIPPER_TARGET = "current"

# Upstream timing, in physical seconds. sampler.py:148 walks back `idx * obs_down_sample_steps` on a
# ~59.94 Hz buffer with every latency term zero (dataset_frequeny: 0), so both the observation pair and the
# action targets are spaced 3/59.94 apart. Our zarr is 30 Hz, where that is 1.5 frames, so the offsets are
# interpolated rather than indexed -- 200 ms (3 stored frames) and 66.7 ms (1 stored frame) are both wrong.
UMI_DT = 3.0 / 59.94          # 50.05 ms
RAW_HZ = 30.0
A_HORIZON = 16                # action_horizon
STATE76 = 76
ACTION_PER_ARM = 10           # pos3 + rot6d + gripper
POSE9 = 9                     # pos3 + rot6d
TIME_EPS_S = 1e-9             # window edges are inclusive up to float noise
MIN_ROWS = 3                  # shorter episodes are skipped
GATE_PERCENTILE = 99
M_TO_MM = 1000.0
IMAGE_HW = (224, 224)
GLOBAL_KEY = "observation.images.global"
SEEK_BACK_S = 0.5             # seek this far before from_timestamp, then decode forward
FALLBACK_HALF_FRAME_S = 1 / 60
WRITER_THREADS = "8"          # default of CONV_WRITER_THREADS
PROGRESS_EVERY = 10           # episodes between progress lines
SOURCE_KEYS = {"zarr", "lerobot", "global_transform", "episodes_json", "select"}
# Upstream packing: keys sorted alphabetically (timm_obs_encoder.py:169), each reshaped (T, D) -> flat,
# so timestep-major with the older frame first. Do not re-order for slim20 compatibility.
STATE_OFFSETS = {"r0_pos": 0, "r0_pos_wrt": 6, "r0_rot": 12, "r0_rot_wrt": 24, "r0_grip": 36,
                 "r1_pos": 38, "r1_pos_wrt": 44, "r1_rot": 50, "r1_rot_wrt": 62, "r1_grip": 74}


class SourceSpecError(ValueError):
    """A --source spec, its files or its episode selection is invalid."""


def _loadNodeModules():
    """the node checkouts the converter needs: umi_bridge (umi_action, r150_v4_instructions) and the UMI repository
    (JpegXl codecs for the zarr images, pose_util)"""
    prepend_sys_path(str(NODE_HOME / "umi_bridge"))
    import umi_action
    prepend_sys_path(str(NODE_HOME / "universal_manipulation_interface"))
    prepend_sys_path(str(pathlib.Path(__file__).resolve().parent))
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()
    import r150_v4_instructions
    return umi_action, r150_v4_instructions


def _defaultLerobotRoot():
    from r150_v4_instructions import R150_LEROBOT
    return R150_LEROBOT


def bodyDelta(future_mat, base_mat):
    """consecutive body-frame increments inv(T_{k-1}) @ T_k, T_{-1} = base (was body_delta)"""
    tPrev = np.concatenate([base_mat[None], future_mat[:-1]], axis=0)
    return np.linalg.inv(tPrev) @ future_mat


def deltaMagnitudes(d):
    """-> (translation mm, rotation deg) of relative transforms d (was delta_magnitudes)"""
    tDp = np.linalg.norm(d[..., :3, 3], axis=-1) * M_TO_MM
    tTr = np.clip((np.trace(d[..., :3, :3], axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return tDp, np.degrees(np.arccos(tTr))


def _sourceEpisodeMeta(root=None):
    tRoot = pathlib.Path(root) if root is not None else _defaultLerobotRoot()
    tFiles = sorted(glob.glob(str(tRoot / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    tD = pd.concat([pd.read_parquet(tF) for tF in tFiles], ignore_index=True).sort_values("episode_index")
    return tD.reset_index(drop=True)


def decodeGlobal(row, n, out_res=IMAGE_HW, root=None, transform=None):
    """Decode this episode's global frames straight from the source mp4 (was decode_global).

    Same policy as the UMI exporter used for the wrist views: no crop, INTER_AREA to 224x224, and a
    half-frame tolerance on from_timestamp (a tight comparison drops the intended first frame in 35 of
    150 episodes and the segment then runs one short).

    `transform` applies to the GLOBAL view only. It exists for one pool: R675 was recorded with the third
    camera in front of the table rather than behind the operator's head, and a FRONT->HEAD affine fit came
    out at about +175.6 deg (rot180 residual 7.7 px, against 58.7 px for a mirror and 145.6 px for both).
    Rotating R675's global stream by 180 deg therefore puts both pools in the same global slot. It is never
    applied to a wrist view, and it is never a mirror: a mirror would flip handedness against the state and
    action, which are left untouched.
    """
    tChunk = int(np.asarray(row[f"videos/{GLOBAL_KEY}/chunk_index"]).ravel()[0])
    tFileIdx = int(np.asarray(row[f"videos/{GLOBAL_KEY}/file_index"]).ravel()[0])
    tT0 = float(np.asarray(row[f"videos/{GLOBAL_KEY}/from_timestamp"]).ravel()[0])
    tRoot = pathlib.Path(root) if root is not None else _defaultLerobotRoot()
    tPath = tRoot / "videos" / GLOBAL_KEY / f"chunk-{tChunk:03d}" / f"file-{tFileIdx:03d}.mp4"
    tFrames = []
    with av.open(str(tPath)) as tContainer:
        tStream = tContainer.streams.video[0]
        tStream.thread_type = "AUTO"
        tContainer.seek(int(max(tT0 - SEEK_BACK_S, 0) / float(tStream.time_base)), stream=tStream)
        if tStream.average_rate:
            tTol = 0.5 * float(tStream.average_rate.denominator) / float(tStream.average_rate.numerator)
        else:
            tTol = FALLBACK_HALF_FRAME_S
        for tFrame in tContainer.decode(tStream):
            if float(tFrame.pts * tStream.time_base) < tT0 - tTol:
                continue
            tIm = cv2.resize(tFrame.to_ndarray(format="rgb24"), out_res, interpolation=cv2.INTER_AREA)
            if transform == "rot180":
                tIm = cv2.rotate(tIm, cv2.ROTATE_180)
            elif transform not in (None, "none"):
                raise ValueError(f"unknown global transform {transform!r}")
            tFrames.append(tIm)
            if len(tFrames) >= n:
                break
    if len(tFrames) < n:
        raise RuntimeError(f"global short segment {tPath.name}: {len(tFrames)}/{n}")
    return np.stack(tFrames[:n]).astype(np.uint8)


def features(image_hw=IMAGE_HW):
    tH, tW = image_hw
    return {
        "observation.images.global": {"dtype": "video", "shape": (tH, tW, 3),
                                      "names": ["height", "width", "channels"]},
        "observation.images.left_wrist": {"dtype": "video", "shape": (tH, tW, 3),
                                          "names": ["height", "width", "channels"]},
        "observation.images.right_wrist": {"dtype": "video", "shape": (tH, tW, 3),
                                           "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (STATE76,), "names": None},
        # Always a chunk here. The ancestor branched on ACTION_MODE and would silently emit a flat 20-vector
        # if --action-mode was left at its default, which the trainer accepts and the twins would not share.
        "action": {"dtype": "float32", "shape": (A_HORIZON, ACTION_PER_ARM * N_ROBOT), "names": None},
        "state_prev_valid": {"dtype": "float32", "shape": (1,), "names": None},
    }


def _traj(z, r, s, e):
    """Absolute poses, widths and timestamps for one arm over one episode, at the raw 30 Hz cadence."""
    tIdx = np.arange(s, e)
    tPos = np.asarray(z["data"][f"robot{r}_eef_pos"].oindex[tIdx], np.float64)
    tRv = np.asarray(z["data"][f"robot{r}_eef_rot_axis_angle"].oindex[tIdx], np.float64)
    tW = np.asarray(z["data"][f"robot{r}_gripper_width"].oindex[tIdx], np.float64)[:, 0]
    tTau = np.arange(len(tIdx)) / RAW_HZ
    return tTau, tPos, Rot.from_rotvec(tRv), tW


def _sample(tau, pos, rot, w, t):
    """Pose and width at arbitrary physical times: position and width linear, rotation slerp."""
    t = np.clip(np.asarray(t, np.float64), tau[0], tau[-1])
    tP = np.stack([np.interp(t, tau, pos[:, tK]) for tK in range(3)], axis=-1)
    tQ = Slerp(tau, rot)(t)
    tWw = np.interp(t, tau, w)
    tM = np.zeros((len(t), 4, 4))
    tM[:, :3, :3] = tQ.as_matrix()
    tM[:, :3, 3] = tP
    tM[:, 3, 3] = 1.0
    return tM, tWw


def _packState(n, invCur, hist, cur, gripHist, gripCur, matToPose10d):
    """UMI76 state rows (n, 76) in upstream packing (self / cross terms at [t - dt, t])"""
    tState = np.zeros((n, STATE76), np.float32)
    for tR in range(N_ROBOT):
        tO = 1 - tR
        tSelfH = matToPose10d(invCur[tR] @ hist[tR])          # (n,9)
        tSelfC = matToPose10d(invCur[tR] @ cur[tR])           # identity by construction
        tCrossH = matToPose10d(invCur[tO] @ hist[tR])
        tCrossC = matToPose10d(invCur[tO] @ cur[tR])
        tK = f"r{tR}_"
        tPos, tPosW = STATE_OFFSETS[tK + "pos"], STATE_OFFSETS[tK + "pos_wrt"]
        tRot, tRotW = STATE_OFFSETS[tK + "rot"], STATE_OFFSETS[tK + "rot_wrt"]
        tGrip = STATE_OFFSETS[tK + "grip"]
        tState[:, tPos:tPos + 3] = tSelfH[:, :3]
        tState[:, tPos + 3:tPos + 6] = tSelfC[:, :3]
        tState[:, tPosW:tPosW + 3] = tCrossH[:, :3]
        tState[:, tPosW + 3:tPosW + 6] = tCrossC[:, :3]
        tState[:, tRot:tRot + 6] = tSelfH[:, 3:POSE9]
        tState[:, tRot + 6:tRot + 12] = tSelfC[:, 3:POSE9]
        tState[:, tRotW:tRotW + 6] = tCrossH[:, 3:POSE9]
        tState[:, tRotW + 6:tRotW + 12] = tCrossC[:, 3:POSE9]
        tState[:, tGrip] = gripHist[tR]
        tState[:, tGrip + 1] = gripCur[tR]
    return tState


def episodeFrames(z, s, e, stride):
    """One record per query row: UMI76 state and a 16-step current-anchor action chunk (was episode_frames).

    Query rows sit on the stored 15 Hz grid so each one lines up with a video frame, but every pose the
    state and the action refer to is interpolated at upstream's physical offsets, not at stored indices.
    A row is kept only if its history and all 16 targets fall inside the episode -- nothing is padded,
    because X-VLA does not read action_is_pad and would train on invented targets as if they were real.
    """
    from umi.common.pose_util import mat_to_pose10d
    tTr = {tR: _traj(z, tR, s, e) for tR in range(N_ROBOT)}
    tTau = tTr[0][0]
    tLocal = np.arange(0, len(tTau), stride)
    tTq = tLocal / RAW_HZ
    tOk = (tTq - UMI_DT >= tTau[0] - TIME_EPS_S) & (tTq + A_HORIZON * UMI_DT <= tTau[-1] + TIME_EPS_S)
    tLocal, tTq = tLocal[tOk], tTq[tOk]
    if len(tTq) < MIN_ROWS:
        return None, "too short for the umi76 window"
    tN = len(tTq)

    tCur, tHist, tWc, tWh, tActM, tActW = {}, {}, {}, {}, {}, {}
    for tR in range(N_ROBOT):
        tTa, tPos, tRot, tW = tTr[tR]
        tCur[tR], tWc[tR] = _sample(tTa, tPos, tRot, tW, tTq)
        tHist[tR], tWh[tR] = _sample(tTa, tPos, tRot, tW, tTq - UMI_DT)
        tTk = (tTq[:, None] + (np.arange(A_HORIZON) + 1)[None, :] * UMI_DT).reshape(-1)
        tM, tWw = _sample(tTa, tPos, tRot, tW, tTk)
        tActM[tR] = tM.reshape(tN, A_HORIZON, 4, 4)
        tActW[tR] = tWw.reshape(tN, A_HORIZON)

    tInvCur = {tR: np.linalg.inv(tCur[tR]) for tR in range(N_ROBOT)}
    tState = _packState(tN, tInvCur, tHist, tCur, tWh, tWc, mat_to_pose10d)

    tAction = np.zeros((tN, A_HORIZON, ACTION_PER_ARM * N_ROBOT), np.float32)
    tDpAll, tDthAll = [], []
    for tR in range(N_ROBOT):
        tA = tInvCur[tR][:, None] @ tActM[tR]                      # A[k] = inv(T_t) @ T(t+(k+1)dt)
        tA9 = mat_to_pose10d(tA.reshape(-1, 4, 4)).reshape(tN, A_HORIZON, POSE9)
        tOff = tR * ACTION_PER_ARM
        tAction[:, :, tOff:tOff + POSE9] = tA9
        tAction[:, :, tOff + POSE9] = tActW[tR]
        tStep = np.linalg.inv(tA[:, :-1].reshape(-1, 4, 4)) @ tA[:, 1:].reshape(-1, 4, 4)
        tDp, tDth = deltaMagnitudes(np.concatenate([tA[:, 0], tStep], axis=0))
        tDpAll.append(tDp)
        tDthAll.append(tDth)

    tPrevValid = np.ones((tN, 1), np.float32)                      # history always exists here, by the mask
    tDpAll = np.concatenate(tDpAll)
    tDthAll = np.concatenate(tDthAll)
    tStats = dict(n=tN, dp_p99=float(np.percentile(tDpAll, GATE_PERCENTILE)), dp_max=float(tDpAll.max()),
                  dth_p99=float(np.percentile(tDthAll, GATE_PERCENTILE)), dth_max=float(tDthAll.max()),
                  soft=int(((tDpAll > SOFT_DP_MM) | (tDthAll > SOFT_DTH_DEG)).sum()),
                  hard=int(((tDpAll > HARD_DP_MM) | (tDthAll > HARD_DTH_DEG)).sum()))
    return dict(idx=tLocal + s, state=tState, action=tAction, prev_valid=tPrevValid), tStats


def _parseSource(spec):
    """`zarr=<path>,lerobot=<root>[,global_transform=rot180]` -> dict, with the keys checked (was _parse_source).

    Sources are concatenated in the order given and each keeps its own instruction table and its own global
    video root, so R180 = R150 + R30 is one conversion pass rather than a parquet merge: episode indices,
    statistics and provenance then come out right by construction.
    """
    tD = {}
    for tPart in spec.split(","):
        if not tPart.strip():
            continue
        tK, _, tV = tPart.partition("=")
        tD[tK.strip()] = tV.strip()
    tUnknown = set(tD) - SOURCE_KEYS
    if tUnknown:
        raise SourceSpecError(f"--source: unknown key(s) {sorted(tUnknown)} in {spec!r}")
    for tK in ("zarr", "lerobot"):
        if tK not in tD:
            raise SourceSpecError(f"--source: missing {tK}= in {spec!r}")
        if not pathlib.Path(tD[tK]).exists():
            raise SourceSpecError(f"--source: {tK}={tD[tK]} does not exist")
    tD.setdefault("global_transform", None)
    tD.setdefault("episodes_json", None)
    # [2026-09-28] select=<json>: convert only the zarr episodes listed in its "zarr_episodes" (zarr numbering,
    # i.e. BEFORE episodes_json maps them to the pool). Without it every episode is converted, as before.
    tD.setdefault("select", None)
    if tD["select"] and not pathlib.Path(tD["select"]).exists():
        raise SourceSpecError(f"--source: select={tD['select']} does not exist")
    return tD


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--zarr", default=str(NODE_HOME / "umi_bridge/r150_umi.zarr"),
                     help="single-source shorthand, paired with the default LeRobot root; "
                          "ignored when --source is given")
    tAp.add_argument("--source", action="append", default=[],
                     help="zarr=<path>,lerobot=<root>[,global_transform=rot180]; repeatable")
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--repo-id", default="rebot/r150_umi_v4")
    tAp.add_argument("--episodes", type=int, default=0, help="0 = all")
    tAp.add_argument("--stride", type=int, default=2, help="zarr frames per stored frame (30 fps -> 15 Hz)")
    tAp.add_argument("--fps", type=int, default=15)
    tAp.add_argument("--gripper-target", choices=["current", "next"], default="current",
                     help="which instant the action gripper refers to (see GRIPPER_TARGET)")
    tAp.add_argument("--action-mode", choices=["delta", "umi"], default="delta")
    tAp.add_argument("--state-mode", choices=["full38", "slim20", "s32"], default="full38")
    tAp.add_argument("--drop-hard", action="store_true", help="skip episodes containing a hard-gate delta")
    return tAp


def _episodeOrder(src, ends, episodes):
    """-> (zarr episodes to convert, pool episode map or None, n for the progress line)"""
    tEpMap = None
    if src["episodes_json"]:
        # A zarr built from a SUBSET of a pool is numbered 0..n-1 in the order the subset was added, which is
        # not the pool's episode numbering. Without this map, episode 37 of the zarr would be given episode
        # 37's instruction and episode 37's global video -- both belonging to a different demonstration.
        tEpMap = json.loads(pathlib.Path(src["episodes_json"]).read_text())["episodes"]
        if len(tEpMap) != len(ends):
            raise SourceSpecError(f"episodes_json lists {len(tEpMap)} episodes but the zarr holds {len(ends)}")
        print(f"  subset mapping: zarr 0..{len(ends)-1} -> pool episodes "
              f"{tEpMap[0]}..{tEpMap[-1]}", flush=True)
    tNumEp = len(ends) if episodes == 0 else min(episodes, len(ends))
    tOrder = list(range(tNumEp))
    if src["select"]:
        tOrder = [int(tE) for tE in json.loads(pathlib.Path(src["select"]).read_text())["zarr_episodes"]]
        if len(set(tOrder)) != len(tOrder) or min(tOrder) < 0 or max(tOrder) >= len(ends):
            raise SourceSpecError(f"select={src['select']}: zarr_episodes must be unique and within 0..{len(ends)-1}")
        tNumEp = len(tOrder)
        print(f"  select: {tNumEp} of {len(ends)} zarr episodes from {src['select']}", flush=True)
    return tOrder, tEpMap, tNumEp


def _provenance(args, sources, kept, dropped, agg, umiAction):
    return dict(converter="umi_to_lerobot_v4.py",
                obs_semantics="prev_rel inv(T_t) @ T_{t-1} + gripper + wrt, baked",
                sources=[dict(zarr=tX["zarr"], lerobot=tX["lerobot"], global_transform=tX["global_transform"],
                              **({"select": tX["select"],
                                  "select_zarr_episodes": json.loads(pathlib.Path(tX["select"]).read_text())["zarr_episodes"]}
                                 if tX["select"] else {})) for tX in sources],
                source_zarr=sources[0]["zarr"],
                global_transform=sources[0]["global_transform"],
                camera_domain_merge=" + ".join(
                    f"{pathlib.Path(tX['lerobot']).name}"
                    + (f"({tX['global_transform']} global)" if tX["global_transform"] else "")
                    for tX in sources), repo_id=args.repo_id, fps=args.fps, stride=args.stride,
                episodes_kept=kept, episodes_dropped=dropped,
                frames=int(sum(tS["n"] for tS in agg)),
                gates=dict(soft_dp_mm=SOFT_DP_MM, soft_dth_deg=SOFT_DTH_DEG,
                           hard_dp_mm=HARD_DP_MM, hard_dth_deg=HARD_DTH_DEG,
                           soft_total=int(sum(tS["soft"] for tS in agg)),
                           hard_total=int(sum(tS["hard"] for tS in agg))),
                instruction_source=[str(pathlib.Path(tX["lerobot"]) / "meta") for tX in sources],
                state_mode=STATE_MODE, gripper_target=GRIPPER_TARGET, action_mode=ACTION_MODE,
                action_semantics=("umi current-anchor inv(T_t)@T_(t+k), k=1..16"
                                  if ACTION_MODE == "umi" else "body-frame delta inv(T_t)@T_(t+1)"),
                # hard guard for every consumer: the label is ALREADY relative to the query pose, so no
                # loader, collator or trainer may relativise or horizon-expand it a second time
                pose_repr=("current_anchor_relative" if ACTION_MODE == "umi" else "body_delta"),
                already_relativized=(ACTION_MODE == "umi"),
                horizon=(umiAction.CHUNK if ACTION_MODE == "umi" else 1),
                state_semantics="UMI-style local relative proprioception "
                                "(prev_rel = inv(T_t) @ T_(t-1) + current width), "
                                "not the exact upstream UMI lowdim schema",
                state_dim=STATE76,
                action_dim=ACTION_PER_ARM * N_ROBOT)


def convert(args) -> None:
    global STATE_MODE, GRIPPER_TARGET, ACTION_MODE
    STATE_MODE = args.state_mode
    GRIPPER_TARGET = args.gripper_target
    ACTION_MODE = args.action_mode
    tUmiAction, tInstructions = _loadNodeModules()
    import zarr
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tSources = [_parseSource(tX) for tX in args.source] or [
        dict(zarr=args.zarr, lerobot=str(tInstructions.R150_LEROBOT), global_transform=None)]

    tOut = pathlib.Path(args.out)
    # image_writer_threads is the single biggest lever here: without it add_frame writes each PNG inline and
    # costs 4.3 s per 300 frames, with 8 threads it hands off and costs 0.01 s. Measured on this box.
    tDs = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features(),
                                root=str(tOut), robot_type="rebot_b601", use_videos=True,
                                image_writer_threads=int(os.environ.get("CONV_WRITER_THREADS", WRITER_THREADS)))
    tPool = ThreadPoolExecutor(max_workers=2)

    tT0 = time.time()
    tKept = tDropped = 0
    tAgg = []
    for tSi, tSrc in enumerate(tSources):
        print(f"[source {tSi}] zarr={tSrc['zarr']}  lerobot={tSrc['lerobot']}  "
              f"global_transform={tSrc['global_transform']}", flush=True)
        tZ = zarr.open(tSrc["zarr"], "r")
        tEnds = np.asarray(tZ["meta"]["episode_ends"][:], int)
        tStarts = np.concatenate([[0], tEnds[:-1]])
        tTab = tInstructions.episode_table(pathlib.Path(tSrc["lerobot"]))
        tSrcMeta = _sourceEpisodeMeta(tSrc["lerobot"])
        tLut = dict(zip(tTab["episode_index"].tolist(), tTab["instruction"].tolist()))
        tOrder, tEpMap, tNumEp = _episodeOrder(tSrc, tEnds, args.episodes)
        for tEp in tOrder:
            tRec, tSt = episodeFrames(tZ, tStarts[tEp], tEnds[tEp], args.stride)
            if tRec is None:
                print(f"  ep {tEp:3d} skipped: {tSt}")
                tDropped += 1
                continue
            if args.drop_hard and tSt["hard"] > 0:
                print(f"  ep {tEp:3d} dropped: {tSt['hard']} hard-gate deltas "
                      f"(dp_max {tSt['dp_max']:.1f} mm, dth_max {tSt['dth_max']:.1f} deg)")
                tDropped += 1
                continue
            tSrcEp = tEp if tEpMap is None else int(tEpMap[tEp])
            tTask = tLut[tSrcEp]
            tImgs = {tR: tZ["data"][f"camera{tR}_rgb"] for tR in range(N_ROBOT)}
            # the two wrist reads (JpegXl decode, ~1.1 s) and the global mp4 decode (~2.7 s) are independent;
            # run them together instead of one after the other
            tWristFut = tPool.submit(lambda: {tR: tImgs[tR][tRec["idx"]] for tR in range(N_ROBOT)})
            # global comes from the source mp4 at the SAME (episode, frame) index as the wrist views; the
            # alignment was verified directly (MAE 0.44 / corr 0.9999, and +1 frame raises MAE 17x).
            tRow = tSrcMeta[tSrcMeta["episode_index"] == tSrcEp].iloc[0]
            tGlobal = decodeGlobal(tRow, int(tEnds[tEp] - tStarts[tEp]),
                                   root=tSrc["lerobot"], transform=tSrc["global_transform"])
            tWrist = tWristFut.result()
            for tI, tFi in enumerate(tRec["idx"]):
                tDs.add_frame({
                    "observation.images.global": tGlobal[int(tFi - tStarts[tEp])],
                    "observation.images.left_wrist": tWrist[0][tI],
                    "observation.images.right_wrist": tWrist[1][tI],
                    "observation.state": tRec["state"][tI],
                    "action": tRec["action"][tI],
                    "state_prev_valid": tRec["prev_valid"][tI],
                    "task": tTask,
                })
            # the three camera keys encode in separate processes: 5.17 -> 4.46 s per 300-frame
            # episode, measured. Not the 3x the three streams suggest, because the encode is only
            # part of save_episode and each worker pays process startup.
            tDs.save_episode(parallel_encoding=True)
            tKept += 1
            tAgg.append(tSt)
            if tEp % PROGRESS_EVERY == 0 or tEp == tOrder[-1]:
                print(f"  s{tSi} ep {tEp:3d}/{tNumEp}  frames {tSt['n']:5d}  dp_p99 {tSt['dp_p99']:6.2f} mm  "
                      f"soft {tSt['soft']:3d}  hard {tSt['hard']:2d}  [{time.time()-tT0:6.1f}s]", flush=True)

    tProv = _provenance(args, tSources, tKept, tDropped, tAgg, tUmiAction)
    (tOut / "v4_provenance.json").write_text(json.dumps(tProv, indent=2))
    print(json.dumps(tProv, indent=2))


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        convert(tArgs)
    except SourceSpecError as tErr:
        print(tErr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
