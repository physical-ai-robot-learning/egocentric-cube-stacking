"""Label builders: CART20 horizon chunk, both state20 variants, the action32 packing and stacking-stage ids
(was labels/build_cart20.py + build_state20.py + pack_action32.py + stage.py).

CART20 -- current-anchor relative, never sequential:
    A_k = inv(T(t)) @ T(t + k * UMI_DT),   k = 1..16,   both arms on the SAME physical instants
    CART20_k = [L xyz3 rot6d6 g(t+k dt) | R xyz3 rot6d6 g(t+k dt)]
  T(t + k dt) and g(t + k dt) are interpolated on the RAW tracks (PoseTrack.sample).  A chunk is valid only if every one
  of the 16 targets of both arms is valid -- no tail padding, no terminal-pose repetition.
state20 MAIN (state.npy, user decision D3) -- v4 RELCART20 task-anchor state, identical semantics and layout to the reBot FT data:
    per arm  T_rel = inv(T_task_start) @ T(t);   state20 = [L xyz3 rot6d6 | R xyz3 rot6d6 | gL | gR]   grippers at dims 18, 19
state20 DIAGNOSTIC (state_prevrel.npy) -- spec v2 previous-relative state:
    per arm  T_rel = inv(T(t)) @ T(t - UMI_DT)   (T(t - UMI_DT) interpolated on the raw track)
    state20 = [L xyz3 rot6d6 gL | R xyz3 rot6d6 gR]                                                       grippers at dims 9, 19
action32 -- X-VLA v4 action tensor [16, 32]: channels 0:20 = CART20, 20:32 = AUX12 = 0 (ego-only).
Stages (spec 20) -- stage 0 = approach / before first placement, 1..3 = placement phase 1..3, -1 = unknown.
  No automatic labeler is applied.  On HandUMI the human gripper rests CLOSED (g ~ 0), opens to ~0.7 for the approach and
  closes only to ~0.6 on a cube, so grasp vs. release differ by ~5-8 mm of aperture: a gripper-only stage detector would
  be noise, and a wrong stage label silently biases the stage-weighted sampler.  Stages therefore come only from an explicit
  annotation file (per episode: a sorted list of stage start times in seconds on the episode clock, 4 entries for stages 0..3),
  and rows without one are -1 (weight 1.0 in the sampler).

Pipeline stage: labels of the dataset stage (used by convert.py).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .config import (ACT_GRIP, ACT_POS, ACT_ROT, AUX, CART_DIM, HORIZON, MODEL_ACTION_DIM, PR_GRIP, PR_POSE, ST_GRIP, ST_POSE,
                     STATE_DIM, UMI_DT)
from .transforms import pose9, relative

ANNOTATION_ENV = "EGO_CART20_STAGE_ANNOTATIONS"
N_STAGES = 4
# written into every episode's metadata.json (stage.reason); the path is the module's name when the text was frozen
NO_STAGE_REASON = "no stage annotation; gripper-only detection rejected (see labels/stage.py)"


class AuxNonZeroError(ValueError):
    """An ego-only action32 carries a non-zero (or NaN) AUX12 entry."""


# ---------------------------------------------------------------------------------------------------------------- CART20
def cart20FromPoses(leftNow, leftFuture, gripperLeftFuture, rightNow, rightFuture, gripperRightFuture) -> np.ndarray:
    """pure array form: *Now (4,4), *Future (H,4,4), gripper*Future (H,) -> float32 [H, 20]"""
    tH = len(leftFuture)
    tOut = np.zeros((tH, CART_DIM))
    for tSide, tT0, tFuture, tG in (("left", leftNow, leftFuture, gripperLeftFuture),
                                    ("right", rightNow, rightFuture, gripperRightFuture)):
        tPose9 = pose9(relative(np.asarray(tT0)[None], tFuture))
        tOut[:, ACT_POS[tSide]] = tPose9[:, :3]
        tOut[:, ACT_ROT[tSide]] = tPose9[:, 3:]
        tOut[:, ACT_GRIP[tSide]] = tG
    return tOut.astype(np.float32)


def targetTimes(t, horizon=HORIZON, dt=UMI_DT) -> np.ndarray:
    """t + k * dt, k = 1..horizon"""
    return t + np.arange(1, horizon + 1) * dt


def buildCart20Chunk(leftTrack, rightTrack, t, horizon=HORIZON, dt=UMI_DT, leftNow=None, rightNow=None):
    """-> (cart20 float32 [horizon, 20], ok bool).  *Now: the current pose if already sampled (else sampled at t)."""
    tTargets = targetTimes(t, horizon, dt)
    tFuture = {}
    tOk = True
    for tSide, tTrack, tT0 in (("left", leftTrack, leftNow), ("right", rightTrack, rightNow)):
        if tT0 is None:
            tT0s, _, tOk0 = tTrack.sample([t])
            tT0 = tT0s[0]
            tOk &= bool(tOk0[0])
        tPoses, tGrip, tOkF = tTrack.sample(tTargets)
        tOk &= bool(tOkF.all())
        tFuture[tSide] = (tT0, tPoses, tGrip)
    tChunk = cart20FromPoses(*tFuture["left"], *tFuture["right"])
    return tChunk, tOk


# ---------------------------------------------------------------------------------------------------------------- state20
def buildState20Relcart20(leftAnchor, leftNow, gripperLeftNow, rightAnchor, rightNow, gripperRightNow) -> np.ndarray:
    """batched over leading axes of *Now; returns float32 [..., 20]"""
    tGl = np.asarray(gripperLeftNow, np.float64)
    tOut = np.zeros(tGl.shape + (STATE_DIM,))
    tOut[..., ST_POSE["left"]] = pose9(relative(leftAnchor, leftNow))
    tOut[..., ST_POSE["right"]] = pose9(relative(rightAnchor, rightNow))
    tOut[..., ST_GRIP["left"]] = tGl
    tOut[..., ST_GRIP["right"]] = gripperRightNow
    return tOut.astype(np.float32)


def buildState20Prevrel(leftPrev, leftNow, gripperLeftNow, rightPrev, rightNow, gripperRightNow) -> np.ndarray:
    """spec v2 API (diagnostic only).  Convention: T_rel = inv(T_curr) @ T_prev"""
    tGl = np.asarray(gripperLeftNow, np.float64)
    tOut = np.zeros(tGl.shape + (STATE_DIM,))
    tOut[..., PR_POSE["left"]] = pose9(relative(leftNow, leftPrev))
    tOut[..., PR_POSE["right"]] = pose9(relative(rightNow, rightPrev))
    tOut[..., PR_GRIP["left"]] = tGl
    tOut[..., PR_GRIP["right"]] = gripperRightNow
    return tOut.astype(np.float32)


buildState20 = buildState20Relcart20     # the default state of this pipeline


# ---------------------------------------------------------------------------------------------------------------- action32
def packCart20ToAction32(cart20) -> np.ndarray:
    """[..., 16, 20] -> [..., 16, 32] with AUX12 = 0"""
    tCart = np.asarray(cart20, np.float32)
    assert tCart.shape[-2:] == (HORIZON, CART_DIM), f"CART20 must be [..., {HORIZON}, {CART_DIM}], got {tCart.shape}"
    tAction = np.zeros(tCart.shape[:-1] + (MODEL_ACTION_DIM,), np.float32)
    tAction[..., :CART_DIM] = tCart
    return tAction


def checkAction32(action, cart20) -> None:
    """hard check (spec 27 / 30): never silently fixes a violation"""
    assert action.shape[-2:] == (HORIZON, MODEL_ACTION_DIM), action.shape
    assert np.array_equal(action[..., :CART_DIM], cart20), "action[:, :20] != CART20"
    tNonZero = int(np.count_nonzero(action[..., AUX]))
    if tNonZero:
        raise AuxNonZeroError(f"AUX12 must be exactly zero for ego-only data: {tNonZero} non-zero entries")


# ---------------------------------------------------------------------------------------------------------------- stages
def stageIds(times, gripperLeft, gripperRight, rowValid, starts=None):
    """-> (stage_id int8 [N], info).  starts: list of 4 stage start times (s, same clock as times) or None.
    gripperLeft / gripperRight / rowValid are unused (the rejected gripper-only detector's inputs; kept in the signature)."""
    if starts is None:
        return np.full(len(times), -1, np.int8), dict(source="none", reason=NO_STAGE_REASON)
    tStarts = np.asarray(starts, np.float64)
    assert len(tStarts) == N_STAGES and np.all(np.diff(tStarts) >= 0)
    tStage = (np.searchsorted(tStarts, times, side="right") - 1).astype(np.int8)
    tStage[times < tStarts[0]] = -1
    return tStage, dict(source="annotation", starts_s=tStarts.tolist())


def loadAnnotations(path) -> dict:
    """{episode_id: [4 stage start times]} from the annotation json, {} if the file does not exist"""
    tPath = pathlib.Path(path)
    return json.load(open(tPath)) if tPath.exists() else {}
