"""[2026-09-28] Where do v2-K's rescued segments come from, and are they rescued by odd postures? (Gen-1 retarget stage,
diagnostic; was v2k_tail.py)

In:  a multi_seed_retarget output folder (<episode>.json + .npz), ~/umi_bridge/track_c/data/r150_q_tcp.npz
Out: printed tables only
Usage: python -m rebot_prep.retarget.rescued_tail <v2k out dir>

Per segment: v1 first failure (workspace / IK / collision), and the R150-manifold distance of the chosen v2 trajectory
(median over its frames of the 12-D NN distance to R150, rad). Groups: segments usable in both, v2-only (rescued),
v1-only. Also: what share of the v2 tail (frames with NN > v1 p90) comes from rescued vs both segments, and how the
v2-Kr rescoring S + lambda * d_R150 would pick among the kept variants (<= 3 stored per segment) for a few lambdas.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

from ego_paths import UMI_BRIDGE

STRIDE = 3                       # every 3rd frame (reference tree and queried trajectories)
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
P90 = 90


def segmentRows(outDir: pathlib.Path, tree: cKDTree) -> list[dict]:
    """per segment with a v1 rollout: pass flags, v1 failure, v1 NN distance, (score, NN distance) of every kept v2 variant"""
    tRows = []
    for tPath in sorted(glob.glob(str(outDir / "*.json"))):
        tRec = json.load(open(tPath))
        tFile = outDir / f"{tRec['episode']}.npz"
        if not tFile.exists():
            continue
        tZ = np.load(tFile)
        for tSeg in tRec.get("segments", []):
            if "v1" not in tSeg:
                continue
            tV1Fail = ("none" if tSeg["v1_pass"] else
                       "workspace" if not tSeg["v1"]["ws_ok"] else "IK" if not tSeg["v1"]["ik_ok"] else "collision")
            tNnV1 = float(np.median(tree.query(tZ[f"{tSeg['start']}_v1"][::STRIDE])[0]))
            tVariants = []
            for tJ, tKept in enumerate(tSeg["v2"]["kept"]):
                tQ = tZ[f"{tSeg['start']}_v2_{tJ}"]
                tVariants.append((tKept["score"], float(np.median(tree.query(tQ[::STRIDE])[0]))))
            tRows.append(dict(v1=tSeg["v1_pass"], v2=tSeg["v2"]["pass_"], v1_fail=tV1Fail, nn_v1=tNnV1, var=tVariants))
    return tRows


def group(row: dict) -> str:
    return "both" if row["v1"] and row["v2"] else "v2-only" if row["v2"] else "v1-only" if row["v1"] else "neither"


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("out_dir", help="multi_seed_retarget output folder")
    tArgs = tAp.parse_args(argv)
    tD = np.load(UMI_BRIDGE / "track_c/data/r150_q_tcp.npz")
    tR150 = np.radians(np.c_[tD["J"][:, :6], tD["J"][:, 7:13]])
    tRows = segmentRows(pathlib.Path(tArgs.out_dir), cKDTree(tR150[::STRIDE]))
    print("segments", len(tRows), dict(collections.Counter(group(tR) for tR in tRows)))
    print("v2-only (rescued) by v1 failure:", dict(collections.Counter(tR["v1_fail"] for tR in tRows if group(tR) == "v2-only")))
    for tGroup in ("both", "v2-only"):
        tX = [tR["var"][0][1] for tR in tRows if group(tR) == tGroup]
        tY = [tR["nn_v1"] for tR in tRows if group(tR) == tGroup]
        if tX:
            print(f"{tGroup:8s} v2 top-1 NN-to-R150 p50 {np.median(tX):.3f} p90 {np.percentile(tX, P90):.3f} | "
                  f"same segs v1 NN p50 {np.median(tY):.3f} p90 {np.percentile(tY, P90):.3f}")
    tV1P90 = np.percentile([tR["nn_v1"] for tR in tRows if tR["v1"]], P90) if any(tR["v1"] for tR in tRows) else np.inf
    tTail = [group(tR) for tR in tRows if tR["v2"] and tR["var"][0][1] > tV1P90]
    print(f"v2 top-1 segments beyond v1 p90 ({tV1P90:.3f}): {len(tTail)}  from {dict(collections.Counter(tTail))}")
    for tLam in LAMBDAS:
        tPick = [min(tR["var"], key=lambda v: v[0] + tLam * v[1])[1] for tR in tRows if tR["v2"]]
        print(f"v2-Kr lambda {tLam}: chosen NN p50 {np.median(tPick):.3f} p90 {np.percentile(tPick, P90):.3f}  "
              "(among <= 3 stored variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
