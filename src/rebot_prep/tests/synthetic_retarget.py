"""Synthetic retarget-v2 outputs in the multi-seed retarget layout (<episode>.json + <episode>.npz with <start>_v1,
<start>_v2_<j>, <start>_v2r_<j> q12 trajectories), plus the live / offline-check records the HRL80 report reads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rebot_prep.tests.synthetic_phase3 import ANCHOR_DEG12

SEG = 65
SEG_STARTS = (0, 65, 130, 195)


def _qTraj(rng: np.random.Generator, spread: float) -> np.ndarray:
    tStart = np.radians(ANCHOR_DEG12) + rng.normal(0, spread, 12)
    return (tStart + np.cumsum(rng.normal(0, 0.01, (SEG, 12)), 0)).astype(np.float32)


def _block(rng: np.random.Generator, passed: bool, nKept: int) -> dict:
    tKept = [dict(seed=int(rng.integers(0, 500)), sel_score=float(rng.uniform(0.5, 2)), score=float(rng.uniform(0.5, 2)),
                  pos_err_p50=float(rng.uniform(0, 3)), pos_err_p90=float(rng.uniform(3, 8)), dmin=float(rng.uniform(0.08, 0.3)),
                  d_r150=float(rng.uniform(0, 1))) for _ in range(nKept if passed else 0)]
    return dict(n_ws_candidates=int(rng.integers(0, 3)) if not passed else int(rng.integers(1, 40)), n_rolled=int(rng.integers(0, 16)),
                n_feasible=len(tKept) + int(rng.integers(0, 3)), pass_=passed, kept=tKept,
                variant_dist_rad=[float(rng.uniform(0.5, 1.5)) for _ in range(max(0, len(tKept) - 1))])


def writeRetargetDir(out: Path, episodes: list[str], seed: int, withV2r: bool, extraFail: bool = True) -> None:
    tRng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for tE in episodes:
        tSegs, tQ = [], {}
        for tStart in SEG_STARTS:
            tV1 = dict(ws_ok=bool(tRng.random() < 0.7), ik_ok=bool(tRng.random() < 0.8), collision_ok=bool(tRng.random() < 0.9),
                       pos_err_p50=float(tRng.uniform(0, 3)), pos_err_p90=float(tRng.uniform(3, 9)), score=float(tRng.uniform(1, 3)))
            tV1Pass = tV1["ws_ok"] and tV1["ik_ok"] and tV1["collision_ok"]
            tSeg = dict(start=tStart, v1=tV1, v1_pass=tV1Pass, v2=_block(tRng, bool(tRng.random() < 0.75), int(tRng.integers(1, 4))))
            if withV2r:
                tSeg["v2r"] = _block(tRng, bool(tRng.random() < 0.8), int(tRng.integers(1, 4)))
            tSegs.append(tSeg)
            tQ[f"{tStart}_v1"] = _qTraj(tRng, 0.1)
            for tKey in ("v2", "v2r") if withV2r else ("v2",):
                for tJ in range(len(tSeg[tKey]["kept"])):
                    tQ[f"{tStart}_{tKey}_{tJ}"] = _qTraj(tRng, 0.15)
        if extraFail:
            tSegs.append(dict(start=260, first_fail="B_metric"))
        (out / f"{tE}.json").write_text(json.dumps(dict(episode=tE, first_fail=None, segments=tSegs, seconds=1.0)))
        np.savez_compressed(out / f"{tE}.npz", **tQ)


def writeLive(out: Path, episodes: list[str], seed: int) -> None:
    """robot-like live check records <session dir>__<episode dir>.json"""
    tRng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for tE in episodes:
        tSess, tNum = tE.rsplit("_", 1)
        tSides = {s: dict(over_frac_ang_w=float(tRng.uniform(0, 0.3)), over_frac_ang_a=float(tRng.uniform(0, 0.3))) for s in ("left", "right")}
        (out / f"Hpilot_{tSess}__episode_{tNum}.json").write_text(json.dumps(dict(sides=tSides, grasps_total=int(tRng.integers(0, 6)))))


def writeOfflineCheck(path: Path, seed: int) -> None:
    tRng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tSegs = [dict(start=s, ws_pass=True, first_fail=None if tRng.random() < 0.6 else "C_workspace") for s in SEG_STARTS]
    path.write_text(json.dumps(dict(segments=tSegs + [dict(start=999, first_fail="B_metric")])))
