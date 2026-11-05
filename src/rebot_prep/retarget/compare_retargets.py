"""[2026-09-28] v1 (single anchor) vs v2-K (kinematic multi-seed) on the same human segments -> one table (Gen-1 retarget
stage; was v2k_compare.py).

In:  one or more multi_seed_retarget output folders (<episode>.json + .npz), ~/umi_bridge/track_c/data/r150_q_tcp.npz,
     ~/umi_bridge/track_c/v2k/r30_episodes.json, optionally the robot-like offline-check folder (v1 parity)
Out: --out <json> (and the same JSON printed)
Usage: python -m rebot_prep.retarget.compare_retargets --group NAME=<dir> [...] [--parity-offline <dir>] --out <json>
       [--eval-ref r150full|heldout120]

Posture-manifold metrics use R150 as the reference (all 173,818 frames, 12 arm joints):
  joint range coverage  per joint, overlap of pseudo-q [p1, p99] with R150 [p1, p99], / R150 width; mean over 12 joints
  cluster coverage      k-means (k=200, seed 0) on R150 q12; share of clusters that receive >= 1 pseudo-q frame
  NN distance           12-D joint distance (rad) from each pseudo-q frame (every 5th) to the nearest R150 frame
  start diversity       mean pairwise 12-D distance between segment-start postures; distinct start clusters
v1 parity: v1 gate flags must equal robotlike_offline_check's for the same segment (old C8 only).
--eval-ref heldout120 = R150 minus the R30 calibration episodes: postures the R30 contract never saw. The reference arrays
are built by loadReference(); the original read --eval-ref from sys.argv at import time.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from ego_paths import UMI_BRIDGE

R150_NPZ = UMI_BRIDGE / "track_c/data/r150_q_tcp.npz"
R30_EPISODES = UMI_BRIDGE / "track_c/v2k/r30_episodes.json"
EVAL_REFS = ["r150full", "heldout120"]
TREE_STRIDE = 3                  # every 3rd reference frame in the NN tree
NN_STRIDE = 5                    # every 5th pseudo-q frame queried
CLUSTER_STRIDE, N_CLUSTERS, CLUSTER_SEED = 10, 200, 0
DYN_STRIDE, DYN_REF_STRIDE = 3, 5
PCT_LO, PCT_HI, P90, P95 = 1, 99, 90, 95


@dataclass(frozen=True)
class Reference:
    """R150 (or held-out) joint reference: q12, [p1, p99] box, NN tree, k-means centres, per-episode |dq| / |ddq| pools."""
    mQ: np.ndarray
    mLo: np.ndarray
    mHi: np.ndarray
    mTree: cKDTree
    mDyn: dict
    mCentres: np.ndarray
    mCentreTree: cKDTree


def _q12(d, mask) -> np.ndarray:
    return np.radians(np.c_[d["J"][mask, :6], d["J"][mask, 7:13]])


def _episodeDiffs(d, mask) -> tuple[np.ndarray, np.ndarray]:
    """|dq|, |ddq| at 30 fps pooled over the episodes selected by mask (never across an episode boundary)"""
    tJ = np.radians(np.c_[d["J"][:, :6], d["J"][:, 7:13]])
    tDv, tDa = [], []
    for tE in np.unique(d["ep"][mask]):
        tQ = tJ[d["ep"] == tE]
        tDv.append(np.abs(np.diff(tQ, axis=0)))
        tDa.append(np.abs(np.diff(tQ, 2, axis=0)))
    return np.concatenate(tDv), np.concatenate(tDa)


def loadReference(evalRef: str = "r150full") -> Reference:
    tD = np.load(R150_NPZ)
    if evalRef == "heldout120":        # R150 minus the R30 calibration episodes: postures the R30 contract never saw
        tMask = ~np.isin(tD["ep"], json.load(open(R30_EPISODES))["episodes"])
    else:
        tMask = np.ones(len(tD["ep"]), bool)
    tQ = _q12(tD, tMask)
    tR30 = np.isin(tD["ep"], json.load(open(R30_EPISODES))["episodes"])
    tDyn = {"R30": _episodeDiffs(tD, tR30), "heldout120": _episodeDiffs(tD, ~tR30)}
    np.random.seed(CLUSTER_SEED)
    tCentres, _ = kmeans2(tQ[::CLUSTER_STRIDE], N_CLUSTERS, seed=CLUSTER_SEED, minit="++")
    return Reference(tQ, np.percentile(tQ, PCT_LO, axis=0), np.percentile(tQ, PCT_HI, axis=0), cKDTree(tQ[::TREE_STRIDE]), tDyn,
                     tCentres, cKDTree(tCentres))


def manifold(ref: Reference, Q: list, starts: list) -> dict:
    """posture / dynamics metrics of a set of pseudo-q trajectories (and their start postures) against the reference"""
    if not len(Q):
        return {}
    tRaw = Q
    tQ = np.concatenate(Q)
    tLo, tHi = np.percentile(tQ, PCT_LO, axis=0), np.percentile(tQ, PCT_HI, axis=0)
    tCov = np.clip(np.minimum(tHi, ref.mHi) - np.maximum(tLo, ref.mLo), 0, None) / (ref.mHi - ref.mLo)
    tNn, _ = ref.mTree.query(tQ[::NN_STRIDE])
    _, tCl = ref.mCentreTree.query(tQ)
    tS = np.array(starts)
    tPair = np.linalg.norm(tS[:, None] - tS[None], axis=-1)[np.triu_indices(len(tS), 1)] if len(tS) > 1 else np.array([0.0])
    _, tStartCl = ref.mCentreTree.query(tS)
    tDv = np.concatenate([np.abs(np.diff(tX, axis=0)) for tX in tRaw])
    tDa = np.concatenate([np.abs(np.diff(tX, 2, axis=0)) for tX in tRaw])
    tDynOut = {}
    for tName, (tRv, tRa) in ref.mDyn.items():
        tDynOut[f"W1_dq_vs_{tName}"] = round(float(np.mean(
            [wasserstein_distance(tDv[::DYN_STRIDE, tJ], tRv[::DYN_REF_STRIDE, tJ]) for tJ in range(12)])), 5)
        tDynOut[f"W1_ddq_vs_{tName}"] = round(float(np.mean(
            [wasserstein_distance(tDa[::DYN_STRIDE, tJ], tRa[::DYN_REF_STRIDE, tJ]) for tJ in range(12)])), 5)
    tDynOut.update(dq_p50=round(float(np.median(tDv)), 5), dq_p95=round(float(np.percentile(tDv, P95)), 5),
                   ddq_p50=round(float(np.median(tDa)), 5), ddq_p95=round(float(np.percentile(tDa, P95)), 5))
    return dict(**tDynOut, joint_range_coverage=round(float(tCov.mean()), 3),
                joint_range_coverage_min=round(float(tCov.min()), 3),
                cluster_coverage=round(len(set(tCl.tolist())) / len(ref.mCentres), 3),
                nn_dist_rad_p50=round(float(np.median(tNn)), 3),
                nn_dist_rad_p90=round(float(np.percentile(tNn, P90)), 3), start_pairwise_rad_mean=round(float(tPair.mean()), 3),
                start_distinct_clusters=int(len(set(tStartCl.tolist()))))


def v2block(ref: Reference, key: str, segs: list, recs: list, dirp: str) -> dict:
    """the v2 (or v2r) part of a summary: usable rate, overlaps with v1, seeds, variants, FK error, clearance, manifold"""
    tOk = [tS for _, tS in segs if tS[key]["pass_"]]
    tN = len(segs)
    tUsable = len(tOk)
    tQ, tS = [], []
    for tRec in recs:
        tFile = pathlib.Path(dirp) / f"{tRec['episode']}.npz"
        if not tFile.exists():
            continue
        tZ = np.load(tFile)
        for tSeg in tRec.get("segments", []):
            if key in tSeg and tSeg[key]["pass_"]:
                tTraj = tZ[f"{tSeg['start']}_{key}_0"]
                tQ.append(tTraj)
                tS.append(tTraj[0])
    tVarD = [tX for _, tS in segs for tX in tS[key]["variant_dist_rad"]]
    return dict(usable=tUsable, rate=round(tUsable / tN, 3) if tN else None,
                both=sum(tS["v1_pass"] and tS[key]["pass_"] for _, tS in segs),
                only_v2=sum((not tS["v1_pass"]) and tS[key]["pass_"] for _, tS in segs),
                only_v1=sum(tS["v1_pass"] and not tS[key]["pass_"] for _, tS in segs),
                ws_feasible_anchor_rate=round(float(np.mean([tS[key]["n_ws_candidates"] > 0 for _, tS in segs])), 3),
                feasible_seeds_per_seg_p50=float(np.median([tS[key]["n_feasible"] for _, tS in segs])),
                segs_with_2plus_variants=int(sum(1 for tS in tOk if len(tS[key]["kept"]) >= 2)),
                variant_dist_rad_p50=round(float(np.median(tVarD)), 3) if tVarD else None,
                fk_pos_mm_p50=round(float(np.median([tS[key]["kept"][0]["pos_err_p50"] for tS in tOk])), 3) if tOk else None,
                fk_pos_mm_p90=round(float(np.median([tS[key]["kept"][0]["pos_err_p90"] for tS in tOk])), 3) if tOk else None,
                dmin_m_p50=round(float(np.median([tS[key]["kept"][0]["dmin"] for tS in tOk])), 3) if tOk else None,
                manifold=manifold(ref, tQ, tS))


def summarize(ref: Reference, dirp: str, parity_dir: str | None = None) -> dict:
    """one group's summary (v1 block, v2 / v2r blocks, optional v1 parity vs the offline check)"""
    tRecs = [json.load(open(tP)) for tP in sorted(glob.glob(f"{dirp}/*.json"))]
    tSegs = [(tR["episode"], tS) for tR in tRecs for tS in tR.get("segments", []) if "v1" in tS]
    tN = len(tSegs)
    tV1Pass = sum(tS["v1_pass"] for _, tS in tSegs)

    def tGate(tKey):
        return round(float(np.mean([bool(tS["v1"][tKey]) for _, tS in tSegs])), 3) if tSegs else None

    tQv1, tS1 = [], []
    for tRec in tRecs:
        tFile = pathlib.Path(dirp) / f"{tRec['episode']}.npz"
        if not tFile.exists():
            continue
        tZ = np.load(tFile)
        for tSeg in tRec.get("segments", []):
            if "v1" not in tSeg:
                continue
            if tSeg["v1_pass"]:
                tQv1.append(tZ[f"{tSeg['start']}_v1"])
                tS1.append(tZ[f"{tSeg['start']}_v1"][0])
    tOut = dict(episodes=len(tRecs), errors=sum(1 for tR in tRecs if "error" in tR), segments=tN,
                v1=dict(usable=tV1Pass, rate=round(tV1Pass / tN, 3) if tN else None, workspace=tGate("ws_ok"), ik=tGate("ik_ok"),
                        collision=tGate("collision_ok"),
                        fk_pos_mm_p50=round(float(np.median([tS["v1"]["pos_err_p50"] for _, tS in tSegs])), 3) if tSegs else None,
                        fk_pos_mm_p90=round(float(np.median([tS["v1"]["pos_err_p90"] for _, tS in tSegs])), 3) if tSegs else None,
                        manifold=manifold(ref, tQv1, tS1)),
                **{tK: v2block(ref, tK, tSegs, tRecs, dirp) for tK in ("v2", "v2r") if tSegs and tK in tSegs[0][1]})
    if parity_dir:
        tMismatch = tTotal = 0
        for tE, tSeg in tSegs:
            tFile = pathlib.Path(parity_dir) / f"{tE}.json"
            if not tFile.exists():
                continue
            tRef = {tX["start"]: tX for tX in json.load(open(tFile)).get("segments", []) if "ws_pass" in tX}
            if tSeg["start"] not in tRef:
                continue
            tX = tRef[tSeg["start"]]
            tTotal += 1
            tMismatch += int((tX["first_fail"] is None) != tSeg["v1_pass"])
        tOut["v1_parity_vs_offline_check"] = dict(segments=tTotal, mismatches=tMismatch)
    return tOut


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--group", action="append", required=True)
    tAp.add_argument("--parity-offline")
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--eval-ref", default="r150full", choices=EVAL_REFS, help="read at import time (reference arrays)")
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tRef = loadReference(tArgs.eval_ref)
    tRes = {}
    for tSpec in tArgs.group:
        tName, tDir = tSpec.split("=", 1)
        tRes[tName] = summarize(tRef, tDir, tArgs.parity_offline if tName.startswith("old") else None)
    pathlib.Path(tArgs.out).write_text(json.dumps(tRes, indent=1))
    print(json.dumps(tRes, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
