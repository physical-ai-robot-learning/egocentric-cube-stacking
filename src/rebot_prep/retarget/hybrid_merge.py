"""[2026-09-28] Build the hybrid master (hybrid_frozen.json): TR top-1 if TR has a feasible rollout, else Kr top-1 (Gen-1
retarget stage).

In:  ~/c8/robotlike/tr_old (TR, key v2) and ~/c8/robotlike/r30_old (v1 + Kr, key v2r) -- the same 1,229 segments
     (env HM_TR / HM_KR; paths only, the rule is unchanged)
Out: ~/c8/robotlike/hybrid_master/<episode>.json + .npz (env HM_OUT) in the multi_seed_retarget layout (key "v2" = the hybrid
     choice, "v1" kept for comparison), so compare_retargets reads it unchanged. No rollout is recomputed.
Usage: python -m rebot_prep.retarget.hybrid_merge
"""
from __future__ import annotations

import collections
import glob
import json
import os
import pathlib
import sys

import numpy as np

from ego_paths import DATA_ROOT

R = DATA_ROOT / "robotlike"
MASTER = "hybrid_v1"


def _paths() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """HM_TR / HM_KR / HM_OUT, defaults = the old259 inputs / master"""
    tDefaults = (("HM_TR", "tr_old"), ("HM_KR", "r30_old"), ("HM_OUT", "hybrid_master"))
    return tuple(pathlib.Path(os.environ.get(tK, R / tD)) for tK, tD in tDefaults)


def mergeEpisode(episode: str, tr: dict, kr: dict, trNpz, krNpz, counts: collections.Counter,
                 reasons: collections.Counter) -> tuple[dict, dict]:
    """the hybrid record and q arrays of one episode (same segments in both runs, matched by start)"""
    tKrByStart = {tS["start"]: tS for tS in kr.get("segments", [])}
    tRec = dict(episode=episode, first_fail=tr.get("first_fail"), segments=[], master=MASTER)
    tQ = {}
    for tSeg in tr.get("segments", []):
        if "v1" not in tSeg:
            tRec["segments"].append(tSeg)
            continue
        tKrSeg = tKrByStart[tSeg["start"]]                                      # same segment in both runs
        tTrV, tKrV = tSeg["v2"], tKrSeg["v2r"]
        tWhy = None
        if tTrV["pass_"]:
            tMethod, tSrc, tZ, tKey = "TR", tTrV, trNpz, "v2"
        elif tKrV["pass_"]:
            tMethod, tSrc, tZ, tKey = "Kr_fallback", tKrV, krNpz, "v2r"
            tWhy = "no_workspace_window" if tTrV["n_ws_candidates"] == 0 else "no_feasible_rollout"
        else:
            tMethod, tSrc, tZ, tKey = "none", tTrV, None, None
            tWhy = "no_workspace_window" if tTrV["n_ws_candidates"] == 0 else "no_feasible_rollout"
        counts[tMethod] += 1
        if tWhy:
            reasons[(tMethod, tWhy)] += 1
        tTop = tSrc["kept"][:1] if tMethod != "none" else []
        tRec["segments"].append(dict(start=tSeg["start"], v1=tKrSeg["v1"], v1_pass=tKrSeg["v1_pass"],
                                     v2=dict(pass_=tMethod != "none", kept=tTop, n_ws_candidates=tSrc["n_ws_candidates"],
                                             n_rolled=tSrc["n_rolled"], n_feasible=tSrc["n_feasible"],
                                             variant_dist_rad=[], retarget_method=tMethod, fallback_reason=tWhy,
                                             tr_window_id=(tTrV["kept"][0]["seed"] if tTrV["pass_"] else None),
                                             kr_seed_id=(tKrV["kept"][0]["seed"] if tKrV["pass_"] else None),
                                             tr_feasible=tTrV["n_feasible"], kr_feasible=tKrV["n_feasible"])))
        tQ[f"{tSeg['start']}_v1"] = krNpz[f"{tSeg['start']}_v1"]
        if tMethod != "none":
            tQ[f"{tSeg['start']}_v2_0"] = tZ[f"{tSeg['start']}_{tKey}_0"]
    return tRec, tQ


def main(argv: list[str] | None = None) -> int:
    tTr, tKr, tOut = _paths()
    tOut.mkdir(exist_ok=True)
    tCounts, tReasons = collections.Counter(), collections.Counter()
    for tPath in sorted(glob.glob(str(tTr / "*.json"))):
        tE = pathlib.Path(tPath).stem
        tTrRec, tKrRec = json.load(open(tPath)), json.load(open(tKr / f"{tE}.json"))
        tTrNpz = np.load(tTr / f"{tE}.npz") if (tTr / f"{tE}.npz").exists() else None
        tKrNpz = np.load(tKr / f"{tE}.npz") if (tKr / f"{tE}.npz").exists() else None
        tRec, tQ = mergeEpisode(tE, tTrRec, tKrRec, tTrNpz, tKrNpz, tCounts, tReasons)
        (tOut / f"{tE}.json").write_text(json.dumps(tRec, indent=1, default=float))
        if tQ:
            np.savez_compressed(tOut / f"{tE}.npz", **tQ)
    tN = sum(tCounts.values())
    tUsable = tCounts["TR"] + tCounts["Kr_fallback"]
    print("segments", tN, dict(tCounts), "| usable", tUsable, f"({tUsable / tN:.3f})",
          "| TR share of usable", f"{tCounts['TR'] / max(1, tUsable):.3f}")
    print("fallback / none reasons:", {f"{tA}:{tB}": tC for (tA, tB), tC in tReasons.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
