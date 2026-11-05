"""Raw ego episode -> processed CART20 episode (the whole label pipeline for one episode).

    1  one PoseTrack per arm on its own raw clock (no pairing to another arm's frames)
    2  task start t_s = first raw instant at which BOTH arms are valid; row grid t_s + n / 15 Hz up to the last instant
       both arms have a valid raw sample
    3  rows: both arms sampled at the row time (position lerp, rotation SLERP, gripper lerp)
    4  canonical / state pose: per arm inv(T(t_s)) @ T(t)                    (canonical_frame, per_arm_task_start_v1)
    5  state.npy (RELCART20 [L9|R9|gL gR]); state_prevrel.npy with T(t - UMI_DT) from the raw track (diagnostic)
    6  CART20 chunk per row: A_k = inv(T(t)) T(t + k UMI_DT), k = 1..16, raw-track interpolation; packed to [16,32]
    7  a TRAINING row needs: both arms valid at t, all 16 targets of both arms valid, every required camera frame within
       image_tol_s.  Nothing is padded.  Invalid rows keep NaN state and are never sampled.
(v2b: the raw discontinuity filter runs before step 1 when Cart20Config.mDiscontinuityFilter is set.)

In: RawEpisode (raw_episode.py).  Out: dict with the ARRAYS of processed_episode.py + cameras + metadata (+ _anchor, _tracks).
Pipeline stage: dataset stage (scripts/convert_dataset.py, scripts/convert_episode.py).
"""
from __future__ import annotations

import numpy as np

from . import canonical_frame
from .config import ARMS, CART_DIM, CONTRACT, HORIZON, Cart20Config
from .gripper import validateGripper
from .labels import buildCart20Chunk, buildState20Prevrel, buildState20Relcart20, checkAction32, packCart20ToAction32, stageIds
from .pose_filter import discontinuityFilter
from .raw_episode import CAMERAS, RawEpisode
from .resample import PoseTrack, canonicalTimes, commonEndTime, nearestIndex, taskStartTime

CAMERA_DT_PERCENTILE = 99
MS_PER_S = 1e3


def convertEpisode(raw: RawEpisode, cfg: Cart20Config = Cart20Config()) -> dict:
    """the whole label pipeline for one episode (steps 1-7 above)"""
    tValid = {tArm: raw.mArms[tArm].mValid for tArm in ARMS}
    tJumpEvents = {}
    if cfg.mDiscontinuityFilter:                   # v2b: raw MASt3R silent-jump filter (before anything else)
        for tArm in ARMS:
            tValid[tArm], tJumpEvents[tArm] = discontinuityFilter(
                raw.mArms[tArm].mTimeS, raw.mArms[tArm].mPose, raw.mArms[tArm].mValid, cfg.mJumpMaxTransPerRowM,
                cfg.mJumpMaxRotPerRowDeg, window=cfg.mJumpWindow, isolatedStepM=cfg.mJumpIsolatedStepM)
    tTracks = {tArm: PoseTrack(raw.mArms[tArm].mTimeS, raw.mArms[tArm].mPose, tValid[tArm], raw.mArms[tArm].mGripper,
                               cfg.mMaxRawGapS) for tArm in ARMS}
    for tArm in ARMS:
        validateGripper(tTracks[tArm].mGripper[tTracks[tArm].mValid], f"{tArm} raw gripper")
    tStartS = taskStartTime([tTracks[tArm] for tArm in ARMS])
    tEndS = commonEndTime([tTracks[tArm] for tArm in ARMS])
    tTimes = canonicalTimes(tStartS, tEndS, cfg.mRowFps)
    tN = len(tTimes)
    tSampled = {tArm: tTracks[tArm].sample(tTimes) for tArm in ARMS}          # (T, g, ok)
    tRowValid = tSampled["left"][2] & tSampled["right"][2]
    assert tRowValid[0], "row 0 must be the task start"
    tAnchor = {tArm: tSampled[tArm][0][0].copy() for tArm in ARMS}

    # canonical poses (NaN on invalid rows)
    tPose = {}
    for tArm in ARMS:
        tCanon = canonical_frame.canonicalize(tSampled[tArm][0], tAnchor[tArm])
        tCanon[~tRowValid] = np.nan
        tPose[tArm] = tCanon
    tGrip = {tArm: np.where(tRowValid, tSampled[tArm][1], np.nan) for tArm in ARMS}

    # state (main) -- RELCART20 task anchor
    tState = buildState20Relcart20(tAnchor["left"], tSampled["left"][0], tSampled["left"][1],
                                   tAnchor["right"], tSampled["right"][0], tSampled["right"][1])
    tState[~tRowValid] = np.nan

    # state_prevrel (diagnostic) -- inv(T(t)) T(t - UMI_DT) on the raw track; identity + mask 0 where history is invalid
    tPrevSampled = {tArm: tTracks[tArm].sample(tTimes - cfg.mTargetDtS) for tArm in ARMS}
    tPrevOk = tPrevSampled["left"][2] & tPrevSampled["right"][2] & tRowValid
    tPrevPose = {tArm: np.where(tPrevOk[:, None, None], tPrevSampled[tArm][0], tSampled[tArm][0]) for tArm in ARMS}
    tStatePrevrel = buildState20Prevrel(tPrevPose["left"], tSampled["left"][0], tSampled["left"][1],
                                        tPrevPose["right"], tSampled["right"][0], tSampled["right"][1])
    tStatePrevrel[~tRowValid] = np.nan
    tStatePrevValid = tPrevOk.astype(np.float32)

    # cameras: nearest frame within tolerance, per row
    tCamValid = np.zeros((tN, len(CAMERAS)), bool)
    tCameras = {}
    for tCi, tCam in enumerate(CAMERAS):
        if tCam not in raw.mCams:
            continue
        tCamT, tCamFrame = raw.mCams[tCam]
        tJ, tDt = nearestIndex(tCamT, tTimes, cfg.mImageTolS)
        tCamValid[:, tCi] = tJ >= 0
        tDtP99 = float(np.percentile(tDt[tJ >= 0], CAMERA_DT_PERCENTILE) * MS_PER_S) if (tJ >= 0).any() else None
        tCameras[tCam] = dict(frame_index=np.where(tJ >= 0, tCamFrame[np.maximum(tJ, 0)], -1).astype(np.int64),
                              video=raw.mMeta.get("videos", {}).get(tCam), tol_s=cfg.mImageTolS, dt_ms_p99=tDtP99)
    tNeed = np.ones(tN, bool)
    for tCam in cfg.mRequiredCameras:
        tNeed &= tCamValid[:, CAMERAS.index(tCam)]

    # CART20 chunks
    tRows, tChunks = [], []
    for tRow in np.flatnonzero(tRowValid & tNeed):
        tChunk, tOk = buildCart20Chunk(tTracks["left"], tTracks["right"], tTimes[tRow], cfg.mHorizon, cfg.mTargetDtS,
                                       leftNow=tSampled["left"][0][tRow], rightNow=tSampled["right"][0][tRow])
        if tOk:
            tRows.append(tRow)
            tChunks.append(tChunk)
    tCart20 = np.stack(tChunks).astype(np.float32) if tChunks else np.zeros((0, HORIZON, CART_DIM), np.float32)
    tAction = packCart20ToAction32(tCart20)
    checkAction32(tAction, tCart20)
    tTrainRows = np.asarray(tRows, np.int64)

    tStage, tStageInfo = stageIds(tTimes, tGrip["left"], tGrip["right"], tRowValid)
    tMeta = dict(
        episode_id=raw.mEpisodeId, source=raw.mMeta.get("source", "human_egocentric"), task=raw.mMeta["task"],
        instruction=raw.mMeta["instruction"], stack_order=raw.mMeta["stack_order"],
        fps=cfg.mRowFps, **CONTRACT, canonical_frame=canonical_frame.NAME,
        task_start_time_s=float(tStartS), clock_offset_ns=int(raw.mT0Ns), duration_s=float(tTimes[-1] - tTimes[0]),
        frames_raw={tArm: int(len(raw.mArms[tArm].mTimeS)) for tArm in ARMS}, frames_rows=int(tN),
        rows_valid=int(tRowValid.sum()),
        train_rows=int(len(tTrainRows)), cameras=list(tCameras), camera_order=list(CAMERAS),
        camera_available={tCam: bool(tCam in tCameras) for tCam in CAMERAS}, stage=tStageInfo, config=cfg.toDict(),
        tool_frame=raw.mMeta.get("tool_frame"), gripper_calibration=raw.mMeta.get("gripper_calibration"),
        raw_provenance=raw.mMeta.get("provenance"), jump_events=tJumpEvents if cfg.mDiscontinuityFilter else None,
    )
    return dict(timestamps=tTimes.astype(np.float64), row_valid=tRowValid, train_rows=tTrainRows, state=tState,
                state_prevrel=tStatePrevrel, state_prev_valid=tStatePrevValid, cart20=tCart20, action=tAction,
                left_pose=tPose["left"], right_pose=tPose["right"], left_gripper=tGrip["left"].astype(np.float32),
                right_gripper=tGrip["right"].astype(np.float32), stage_id=tStage.astype(np.int8), camera_valid=tCamValid,
                cameras=tCameras, metadata=tMeta, _anchor=tAnchor, _tracks=tTracks)
