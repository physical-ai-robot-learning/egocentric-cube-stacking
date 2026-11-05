"""[2026-09-28] HRL80 vs old259 report exactly as pre-registered (hrl80_interpretation_prereg.json + amendment_1). No new metric
(Gen-1 retarget stage, evaluation).

In:  ~/c8/robotlike/{master_old259_frozen, hybrid_hrl80} (hybrid masters), {baseline_live, hrl80_live} (robot-like live
     checks), baseline_offline/<episode>.json, the HRL80 raw episodes' derived/robot_like/offline_check.json; the
     compare_retargets reference (--eval-ref)
Out: ~/c8/robotlike/hrl80_prereg_report.json (and the same JSON printed)
Usage: python -m rebot_prep.retarget.hrl80_prereg_report --eval-ref heldout120     (env: ~/xvla-mac)

  size-matched: old259 TR segments subsampled to N_HRL80_TR (numpy default_rng(seed) seeds 0..19, without replacement), the SAME
                compare_retargets.manifold() on each subsample -> median / min-max; HRL80 read by its position in that
                distribution
  T0: exact reproduction of the old259 -0.009 (live over_frac_ang_w vs v1 offline-check full-pass fraction, argsort ranks)
  T1: U_e = TR segments / candidate (v1-record) segments per episode in the master; scipy spearmanr (average ties)
The ~/umi_bridge/track_c/v2k copy of v2k_compare is used first (it reads --eval-ref from sys.argv at its import), else the
repository's compare_retargets with the same --eval-ref.
"""
from __future__ import annotations

import functools
import glob
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

from ego_paths import DATA_ROOT, EGO_COLLECTOR_HOME, UMI_BRIDGE, prepend_sys_path

V2 = UMI_BRIDGE / "track_c/v2k"
RL = DATA_ROOT / "robotlike"
RAW = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/HRL80/HRL80_20260928_101010"
N_SUBSAMPLES = 20
KEYS = ["cluster_coverage", "start_distinct_clusters", "start_pairwise_rad_mean", "nn_dist_rad_p50", "nn_dist_rad_p90",
        "W1_dq_vs_heldout120", "W1_ddq_vs_heldout120", "W1_dq_vs_R30", "W1_ddq_vs_R30", "dq_p95"]
LIVE_METRICS = (("w", "live_rot_speed_over_p95 (primary)"), ("a", "live_rot_acc_over_p95"), ("g", "grasps_total"))
ROUND = 3


def manifoldFunction(argv: list[str]):
    """manifold(Q, starts): the ~/umi_bridge v2k_compare's (reference built at its import), else the repository's"""
    prepend_sys_path(str(V2))
    try:
        import v2k_compare as C                     # reads --eval-ref heldout120 at import
        return C.manifold
    except ImportError:  # clean checkout: the repository copy
        from .compare_retargets import loadReference, manifold
        tEvalRef = argv[argv.index("--eval-ref") + 1] if "--eval-ref" in argv else "r150full"
        return functools.partial(manifold, loadReference(tEvalRef))


def trSegments(master: pathlib.Path) -> tuple[list, list]:
    """q12 trajectories (and start postures) of every TR segment of a master (was tr_segments)"""
    tQ, tS = [], []
    for tPath in sorted(master.glob("2*.json")):
        tRec = json.load(open(tPath))
        tFile = master / f"{tRec['episode']}.npz"
        if not tFile.exists():
            continue
        tZ = np.load(tFile)
        for tSeg in tRec["segments"]:
            if isinstance(tSeg.get("v2"), dict) and tSeg["v2"]["retarget_method"] == "TR":
                tTraj = tZ[f"{tSeg['start']}_v2_0"]
                tQ.append(tTraj)
                tS.append(tTraj[0])
    return tQ, tS


def usableFraction(master: pathlib.Path) -> dict:
    """U_e per episode: TR segments / candidate (v1-record) segments (was u_e)"""
    tOut = {}
    for tPath in sorted(master.glob("2*.json")):
        tRec = json.load(open(tPath))
        tCand = [tS for tS in tRec["segments"] if "v1" in tS]
        if tCand:
            tOut[tRec["episode"]] = sum(tS["v2"]["retarget_method"] == "TR" for tS in tCand) / len(tCand)
    return tOut


def liveMetrics(liveDir: pathlib.Path) -> dict:
    """per episode id: max over sides of the live over-threshold fractions and the grasp count (was live)"""
    tOut = {}
    for tPath in glob.glob(f"{liveDir}/*.json"):
        tS = json.load(open(tPath))
        tSess, tEp = pathlib.Path(tPath).stem.split("__")
        tEid = f"{tSess.split('_', 1)[1]}_{tEp.split('_', 1)[1]}"
        tOut[tEid] = dict(w=max(tS["sides"][tX]["over_frac_ang_w"] for tX in ("left", "right")),
                          a=max(tS["sides"][tX]["over_frac_ang_a"] for tX in ("left", "right")), g=tS["grasps_total"])
    return tOut


def argsortRankCorr(x, y) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def sizeMatched(manifold) -> dict:
    """posture / dynamics, size-matched: HRL80 against N_SUBSAMPLES equal-size subsamples of old259"""
    tQo, tSo = trSegments(RL / "master_old259_frozen")
    tQn, tSn = trSegments(RL / "hybrid_hrl80")
    tN = len(tQn)
    tHrl = manifold(tQn, tSn)
    tFullOld = manifold(tQo, tSo)
    tSubs = []
    for tSeed in range(N_SUBSAMPLES):
        tIdx = np.random.default_rng(tSeed).choice(len(tQo), tN, replace=False)
        tSubs.append(manifold([tQo[tI] for tI in tIdx], [tSo[tI] for tI in tIdx]))
    return dict(N_HRL80_TR=tN, N_old259_TR=len(tQo), rows={tK: dict(
        HRL80=tHrl[tK], old259_full=tFullOld[tK], old259_N_matched_median=float(np.median([tS[tK] for tS in tSubs])),
        old259_N_matched_min=float(min(tS[tK] for tS in tSubs)), old259_N_matched_max=float(max(tS[tK] for tS in tSubs)),
        HRL80_position=("above max" if tHrl[tK] > max(tS[tK] for tS in tSubs)
                        else "below min" if tHrl[tK] < min(tS[tK] for tS in tSubs) else "inside range")) for tK in KEYS})


def transferT0(liveOld: dict, liveNew: dict) -> dict:
    """T0: live over_frac_ang_w vs the v1 offline-check full-pass fraction per episode"""
    tT0 = {}
    for tName, tLive, tGet in (("old259", liveOld, lambda e: RL / "baseline_offline" / f"{e}.json"),
                               ("HRL80", liveNew,
                                lambda e: RAW / f"episode_{e.rsplit('_', 1)[1]}" / "derived/robot_like/offline_check.json")):
        tXs, tYs = [], []
        for tE, tV in tLive.items():
            tFile = tGet(tE)
            if not tFile.exists():
                continue
            tSegs = [tG for tG in json.load(open(tFile)).get("segments", []) if "ws_pass" in tG]
            if tSegs:
                tXs.append(tV["w"])
                tYs.append(np.mean([tG["first_fail"] is None for tG in tSegs]))
        tT0[tName] = dict(n=len(tXs), argsort_rank_corr=round(argsortRankCorr(np.array(tXs), np.array(tYs)), ROUND),
                          spearman_avg_ties=round(float(spearmanr(tXs, tYs).statistic), ROUND))
    return tT0


def transferT1(liveOld: dict, liveNew: dict) -> dict:
    """T1: Spearman of the live metrics vs U_e (old259, HRL80, pooled)"""
    tUo, tUn = usableFraction(RL / "master_old259_frozen"), usableFraction(RL / "hybrid_hrl80")
    tT1 = {}
    tGroups = (("old259", liveOld, tUo), ("HRL80", liveNew, tUn), ("pooled", {**liveOld, **liveNew}, {**tUo, **tUn}))
    for tName, tLive, tU in tGroups:
        tEps = [tE for tE in tU if tE in tLive]
        tRow = dict(n=len(tEps), U_mean=round(float(np.mean([tU[tE] for tE in tEps])), ROUND))
        for tKey, tLabel in LIVE_METRICS:
            tR = spearmanr([tLive[tE][tKey] for tE in tEps], [tU[tE] for tE in tEps])
            tRow[tLabel] = dict(rho=round(float(tR.statistic), ROUND), p=round(float(tR.pvalue), ROUND))
        tRow["live_rot_speed_median"] = round(float(np.median([tLive[tE]["w"] for tE in tEps])), ROUND)
        tT1[tName] = tRow
    return tT1


def main(argv: list[str] | None = None) -> int:
    tArgv = sys.argv[1:] if argv is None else argv
    tManifold = manifoldFunction(tArgv)
    tRep = {"size_matched": sizeMatched(tManifold)}
    tLiveOld, tLiveNew = liveMetrics(RL / "baseline_live"), liveMetrics(RL / "hrl80_live")
    tRep["T0_v1_offline_fullpass_fraction"] = transferT0(tLiveOld, tLiveNew)
    tRep["T1_TR_usable_fraction_U_e"] = transferT1(tLiveOld, tLiveNew)
    (RL / "hrl80_prereg_report.json").write_text(json.dumps(tRep, indent=1))
    print(json.dumps(tRep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
