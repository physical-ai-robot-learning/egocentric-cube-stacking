"""Pose-jump detection on one arm's raw track (was preprocessing/pose_filter.py).

jumpMask             diagnostic only: flags raw steps faster than a speed limit; changes nothing.
discontinuityFilter  the v2b filter (user 2026-10-01): raw MASt3R-SLAM silent jumps (translation jumps with no
                     tracking-lost flag) -> a new validity mask + one event record per jump.  Applied only when
                     Cart20Config.mDiscontinuityFilter is set (the v2 default pipeline does not).

Pipeline stage: preprocessing of the dataset stage (convert.py, before any sampling).
"""
from __future__ import annotations

import numpy as np

from .rotation6d import rotationAngle

ROW_S = 1.0 / 15.0                  # one canonical row; the per-row limits are scaled by dt / ROW_S
RAW_FRAME_FLOOR_S = 1.0 / 30.0      # dt floor = one raw frame, so a short raw step is not judged at an inflated rate
ROW_WINDOW_EPS_S = 1e-3             # "row" window: the sample one row ahead may sit up to 1 ms early
ROW_WINDOW_MAX_STEPS = 3            # "row" window: at most this many raw samples per row (else the window spans a hole)
ROW_STEP_LOCAL_FRAC = 0.5           # "row" window: the raw step itself must carry at least half the row limit
MM_PER_M = 1e3


def jumpMask(timeS, poses, valid, maxLinMps=3.0, maxAngRps=30.0) -> np.ndarray:
    """True at raw sample i when the step (i-1 -> i) between two valid samples exceeds the speed limits"""
    tTime = np.asarray(timeS, np.float64)
    tPoses = np.asarray(poses, np.float64)
    tMask = np.zeros(len(tTime), bool)
    tDt = np.diff(tTime)
    tBoth = valid[1:] & valid[:-1]
    tLin = np.linalg.norm(np.diff(tPoses[:, :3, 3], axis=0), axis=1) / tDt
    tAng = rotationAngle(np.swapaxes(tPoses[:-1, :3, :3], -1, -2) @ tPoses[1:, :3, :3]) / tDt
    tMask[1:] = tBoth & ((tLin > maxLinMps) | (tAng > maxAngRps))
    return tMask


def discontinuityFilter(timeS, poses, valid, maxTransPerRowM=0.100, maxRotPerRowDeg=45.0, groupGap=5, returnFrac=0.3,
                        returnAbsM=0.030, window="frame", isolatedStepM=None, isolatedNeighborMps=0.3):
    """Hard jump detector on ONE arm's raw track.  A raw step i -> i+1 (both valid) is a hard jump if
        |dp| > max_trans_per_row * max(dt, 1/30 s) / (1/15 s)     (100 mm per 66.7 ms row = 1.5 m/s; floor dt at one raw frame)
     or rot > max_rot_per_row * max(dt, 1/30 s) / (1/15 s)       (45 deg per row)
    Consecutive flagged steps (gaps <= group_gap raw samples) form one EVENT spanning samples [a, b+1].
    The event is a SPIKE if the pose after it returns to the pose before it
    (|p(b+1) - p(a)| < max(return_abs, return_frac * largest step) and the rotation also returns within the rotation limit);
    then only samples a+1..b are invalidated, so no interpolation, state, or action window can cross it.  Otherwise it is
    ONE-WAY: every sample from a+1 to the end is invalidated (the task-anchor state would stay offset for the rest of the
    episode).  Returns (new_valid, events)."""
    tTime = np.asarray(timeS, np.float64)
    tPoses = np.asarray(poses, np.float64)
    tValid = np.asarray(valid, bool).copy()
    tN = len(tTime)
    tIdx = np.flatnonzero(tValid)
    tEvents = []
    if len(tIdx) < 2:
        return tValid, tEvents
    tFrom, tTo = tIdx[:-1], tIdx[1:]
    tConsecutive = (tTo - tFrom) == 1
    tDt = np.maximum(tTime[tTo] - tTime[tFrom], RAW_FRAME_FLOOR_S)
    tScale = tDt / ROW_S
    tDp = np.linalg.norm(tPoses[tTo, :3, 3] - tPoses[tFrom, :3, 3], axis=1)
    tDr = np.degrees(rotationAngle(np.swapaxes(tPoses[tFrom, :3, :3], -1, -2) @ tPoses[tTo, :3, :3]))
    if window == "frame":                             # rate limit checked on every raw step (strict)
        tTransJump = tConsecutive & (tDp > maxTransPerRowM * tScale)
        tRotJump = tConsecutive & (tDr > maxRotPerRowDeg * tScale)
        tBad = tTransJump | tRotJump
    else:                                             # "row": displacement over one canonical row (66.7 ms) of raw samples
        tAhead = np.clip(np.searchsorted(tTime, tTime[tFrom] + ROW_S - ROW_WINDOW_EPS_S), 0, tN - 1)
        tWindowOk = tValid[tAhead] & ((tAhead - tFrom) <= ROW_WINDOW_MAX_STEPS)
        tDpRow = np.linalg.norm(tPoses[tAhead, :3, 3] - tPoses[tFrom, :3, 3], axis=1)
        tDrRow = np.degrees(rotationAngle(np.swapaxes(tPoses[tFrom, :3, :3], -1, -2) @ tPoses[tAhead, :3, :3]))
        # the row jump must sit in THIS raw step
        tLocal = (tDp > ROW_STEP_LOCAL_FRAC * maxTransPerRowM) | (tDr > ROW_STEP_LOCAL_FRAC * maxRotPerRowDeg)
        tTransJump = tConsecutive & tWindowOk & (tDpRow > maxTransPerRowM) & tLocal
        tRotJump = tConsecutive & tWindowOk & (tDrRow > maxRotPerRowDeg) & tLocal
        tBad = tTransJump | tRotJump
    if isolatedStepM is not None:                     # glitch signature: a big single step whose neighbouring steps are slow
        tSpeed = tDp / tDt
        tPrev = np.r_[np.inf, tSpeed[:-1]]
        tNext = np.r_[tSpeed[1:], np.inf]
        tPrevFinite = np.where(np.isfinite(tPrev), tPrev, 0)
        tNextFinite = np.where(np.isfinite(tNext), tNext, 0)
        tIsolated = (tConsecutive & (tDp > isolatedStepM) & (np.minimum(tPrev, tNext) < isolatedNeighborMps)
                     & (np.maximum(tPrevFinite, tNextFinite) < isolatedNeighborMps))
        tBad = tBad | tIsolated
    else:
        tIsolated = np.zeros_like(tBad)
    tTags = {"hard_jump_100mm": tTransJump, "hard_rot_45deg": tRotJump, "isolated_glitch_50mm": tIsolated}
    tStepReason = {int(tFrom[tK]): sorted(tName for tName, tMask in tTags.items() if tMask[tK]) for tK in np.flatnonzero(tBad)}
    tSteps = tFrom[tBad]                              # step index = sample before the jump
    tGroups = []
    for tStep in tSteps:
        if tGroups and tStep - tGroups[-1][-1] <= groupGap:
            tGroups[-1].append(tStep)
        else:
            tGroups.append([tStep])
    tCut = None
    for tGroup in tGroups:
        tA, tB = tGroup[0], tGroup[-1] + 1            # pose before the event = a, after = b
        tBig = float(max(tDp[np.searchsorted(tFrom, tX)] for tX in tGroup))
        tReturnP = np.linalg.norm(tPoses[tB, :3, 3] - tPoses[tA, :3, 3])
        tReturnR = float(np.degrees(rotationAngle(tPoses[tA, :3, :3].T @ tPoses[tB, :3, :3])))
        tSpike = tReturnP < max(returnAbsM, returnFrac * tBig) and tReturnR < maxRotPerRowDeg
        tEvents.append(dict(kind="spike" if tSpike else "one_way",
                            reason=sorted({tR for tSt in tGroup for tR in tStepReason[int(tSt)]}),
                            first_sample=int(tA), last_sample=int(tB), t_s=float(tTime[tA + 1]),
                            max_step_mm=round(tBig * MM_PER_M, 1), return_mm=round(float(tReturnP) * MM_PER_M, 1),
                            return_deg=round(tReturnR, 1), n_steps=len(tGroup)))
        if tSpike:
            tValid[tA + 1:tB] = False
        else:
            tCut = tA + 1                             # first one-way jump ends the usable track
            break
    if tCut is not None:
        tValid[tCut:] = False
    return tValid, tEvents
