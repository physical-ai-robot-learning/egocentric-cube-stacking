"""Timestamp-safe pose / gripper sampling on a raw trajectory, and multi-stream synchronisation on capture timestamps
(was preprocessing/resample.py + preprocessing/synchronize.py).

Position: linear.  Rotation: quaternion SLERP.  Gripper: linear.  A query time is VALID only if it falls on a valid
raw sample, or between two consecutive VALID raw samples at most ``maxGapS`` apart.  Nothing is extrapolated,
padded or held: a query outside the valid raw support is invalid (pose = identity placeholder, gripper = NaN).
State history (t - UMI_DT) and action targets (t + k UMI_DT) are evaluated with ``PoseTrack.sample`` on the RAW
track, never on the 15 Hz rows (user decision D1).
All streams share one monotonic capture clock in seconds (raw_episode.py puts them there).

Pipeline stage: preprocessing of the dataset stage (used by convert.py and the label builders).
"""
from __future__ import annotations

import numpy as np

from .rotation6d import matrixToQuaternion, quaternionToMatrix, slerp
from .transforms import makeTransform

TIME_EPS_S = 1e-9             # two times closer than this are the same raw instant
ROW_GRID_EPS = 1e-6           # the last row may sit at t_end within this many rows (float noise of (t_end - t_start) * fps)


class NoCommonInstantError(ValueError):
    """No raw instant at which every arm track is valid."""


class PoseTrack:
    """One arm's raw track: sample pose / gripper at arbitrary times under the valid-support rule above."""

    def __init__(self, timeS, poses, valid, gripper, maxGapS: float = 0.050):
        tTime = np.asarray(timeS, np.float64)
        assert np.all(np.diff(tTime) > 0), "raw timestamps must be strictly increasing"
        self.mTimeS = tTime
        self.mValid = np.asarray(valid, bool)
        self.mMaxGapS = float(maxGapS)
        tPoses = np.asarray(poses, np.float64)
        self.mPos = tPoses[:, :3, 3]
        self.mQuat = matrixToQuaternion(tPoses[:, :3, :3])
        self.mGripper = np.asarray(gripper, np.float64)
        self.mValid &= np.isfinite(self.mPos).all(1) & np.isfinite(self.mGripper)

    def sample(self, times):
        """-> T (m,4,4), g (m,), ok (m,)"""
        tX = np.atleast_1d(np.asarray(times, np.float64))
        tN = len(self.mTimeS)
        tJ = np.clip(np.searchsorted(self.mTimeS, tX, side="right"), 1, tN - 1)
        tA = tJ - 1
        tB = tJ
        tExactA = np.isclose(tX, self.mTimeS[tA], rtol=0, atol=TIME_EPS_S)
        tExactB = np.isclose(tX, self.mTimeS[tB], rtol=0, atol=TIME_EPS_S)
        tInside = (tX >= self.mTimeS[0] - TIME_EPS_S) & (tX <= self.mTimeS[-1] + TIME_EPS_S)
        tOkInterp = self.mValid[tA] & self.mValid[tB] & (self.mTimeS[tB] - self.mTimeS[tA] <= self.mMaxGapS + TIME_EPS_S)
        tOk = tInside & ((tExactA & self.mValid[tA]) | (tExactB & self.mValid[tB]) | tOkInterp)
        tW = np.clip((tX - self.mTimeS[tA]) / (self.mTimeS[tB] - self.mTimeS[tA]), 0.0, 1.0)
        tW = np.where(tExactA, 0.0, np.where(tExactB, 1.0, tW))
        tP = (1 - tW)[:, None] * self.mPos[tA] + tW[:, None] * self.mPos[tB]
        tQ = slerp(self.mQuat[tA], self.mQuat[tB], tW)
        tG = (1 - tW) * self.mGripper[tA] + tW * self.mGripper[tB]
        tT = makeTransform(quaternionToMatrix(tQ), tP)
        tT[~tOk] = np.eye(4)
        tG = np.where(tOk, tG, np.nan)
        return tT, tG, tOk


def canonicalTimes(tStart: float, tEnd: float, fps: float = 15.0) -> np.ndarray:
    """row grid t_start + n / fps, n = 0.. (t_end inclusive within 1 us)"""
    tN = int(np.floor((tEnd - tStart) * fps + ROW_GRID_EPS)) + 1
    return tStart + np.arange(tN) / fps


def resampleEpisode(timestamps, poses, gripper, targetFps=15, valid=None, maxGapS=0.050, tStart=None):
    """spec API: resample one arm's raw track onto the canonical row grid.
    poses: (n,4,4) or dict(position=(n,3), quaternion=(n,4) wxyz).  Returns (times, T, g, ok)."""
    tPoses = makeTransform(quaternionToMatrix(poses["quaternion"]), poses["position"]) if isinstance(poses, dict) else poses
    tValid = np.ones(len(timestamps), bool) if valid is None else valid
    tTrack = PoseTrack(timestamps, tPoses, tValid, gripper, maxGapS)
    tIdx = np.flatnonzero(tTrack.mValid)
    tT0 = tTrack.mTimeS[tIdx[0]] if tStart is None else tStart
    tTimes = canonicalTimes(tT0, tTrack.mTimeS[tIdx[-1]], targetFps)
    tT, tG, tOk = tTrack.sample(tTimes)
    return tTimes, tT, tG, tOk


def nearestIndex(streamT, queryT, tolS):
    """index of the nearest stream sample for every query, -1 if |dt| > tol_s; also returns |dt|"""
    tS = np.asarray(streamT, np.float64)
    tQ = np.atleast_1d(np.asarray(queryT, np.float64))
    if len(tS) == 0:
        return np.full(len(tQ), -1), np.full(len(tQ), np.inf)
    tJ = np.clip(np.searchsorted(tS, tQ), 1, max(len(tS) - 1, 1)) if len(tS) > 1 else np.zeros(len(tQ), int)
    if len(tS) > 1:
        tJ = np.where(np.abs(tS[tJ - 1] - tQ) <= np.abs(tS[tJ] - tQ), tJ - 1, tJ)
    tDt = np.abs(tS[tJ] - tQ)
    return np.where(tDt <= tolS, tJ, -1), tDt


def taskStartTime(tracks) -> float:
    """first raw time at which EVERY arm track has a valid sample there (sampled with its own interpolation rule):
    candidates are each track's valid raw timestamps, checked against all tracks"""
    tCand = np.unique(np.concatenate([tTrack.mTimeS[tTrack.mValid] for tTrack in tracks]))
    tOk = np.ones(len(tCand), bool)
    for tTrack in tracks:
        tOk &= tTrack.sample(tCand)[2]
    if not tOk.any():
        raise NoCommonInstantError("no instant where all arms are valid")
    return float(tCand[np.flatnonzero(tOk)[0]])


def commonEndTime(tracks) -> float:
    """last raw time up to which every track still has a valid sample"""
    return float(min(tTrack.mTimeS[tTrack.mValid][-1] for tTrack in tracks))
