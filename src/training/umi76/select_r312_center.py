"""R312-center FRONT selection (user decision, 2026-09-28): FRONT = explicit pan=center only (phase1 excluded), QA grade A,
exactly 22 episodes per order (PBR, the rarest, has exactly 22) -> FRONT132; nested half FRONT66 = 11 per order (dataset stage).

Deterministic, chosen before any training on it. Within each order, episodes are drawn round-robin over recording
SESSIONS (sorted by name) so no session dominates, and within a session round-robin over its contrastive SET, each
draw by a fixed RNG (seed 20260928). FRONT66 repeats the same procedure on FRONT132 (so R246 is a strict subset).
    R312 = HEAD180 + FRONT132    R246 = HEAD180 + FRONT66    R180 = HEAD180
Inputs: UMI_BRIDGE/rel16_audit/r380/{front_pool_pan.csv, set_map.json, r675_rbp.json, front200_pan.csv} and the center675
source dataset's task table. Outputs: UMI_BRIDGE/umi76/r312c_front132_v1.json and r246c_front66_v1.json in the converter's
select= format (zarr_episodes = r675rbp zarr numbering, i.e. the index into r675_rbp.json "episodes"). The frozen result
r312c_front132_v1.json is shipped next to this module.
Usage: python -m training.umi76.select_r312_center
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

from ego_paths import HOLOBRAIN_DATA, NODE_HOME, UMI_BRIDGE  # machine-specific roots (env-overridable)

SELECTION_SEED = 20260928
FRONT_PER_ORDER = 22             # PBR, the rarest order, has exactly 22 center grade-A episodes
HALF_PER_ORDER = 11
N_ORDERS = 6
CREATED = "2026-09-28"
CENTER675 = "lerobot/src_rebot_3stack_center675_s96"


def orderCode(task: str) -> str:
    """'... red cube ... blue cube ... pink cube' -> 'RBP' (first letters of the cube colours, in order)"""
    return "".join(tC[0].upper() for tC in re.findall(r"(\w+) cube", task))


def loadPool(auditDir, holobrainData):
    """center + grade-A FRONT pool with order, set_key and session columns"""
    tPool = pd.read_csv(f"{auditDir}/front_pool_pan.csv")                  # zarr_ep, src_ep, pan, grade
    tSm = json.load(open(f"{auditDir}/set_map.json"))
    tSet, tKeys = tSm["episode_sets"], tSm["set_keys"]
    tT = pd.concat([pd.read_parquet(tF, columns=["episode_index", "task_index"]) for tF in
                    glob.glob(str(holobrainData / CENTER675 / "data/**/*.parquet"), recursive=True)])
    tTi = tT.groupby("episode_index").task_index.first()
    tTs = {int(tV): tK for tK, tV in pd.read_parquet(str(holobrainData / CENTER675 / "meta/tasks.parquet"))["task_index"].items()}
    tPool["order"] = [orderCode(tTs[int(tTi[tS])]) for tS in tPool.src_ep]
    tPool["set_key"] = [tKeys[str(tSet[tS])] for tS in tPool.src_ep]
    tPool["session"] = tPool.set_key.str.split("|").str[0]
    tCenter = tPool[(tPool.pan == "center") & (tPool.grade == "A")].copy()
    assert (tCenter.set_key != "phase1").all()
    return tCenter


def pick(df, perOrder, rng):
    """perOrder episodes per order: session round-robin, then a random live set of that session, FIFO within the set"""
    tOut = []
    for tOrder in sorted(df.order.unique()):
        tD = df[df.order == tOrder]
        assert len(tD) >= perOrder, (tOrder, len(tD))
        # queue per session -> per set, each shuffled once by the fixed RNG
        tSessions = {}
        for tS in sorted(tD.session.unique()):
            tSessions[tS] = {tK: list(rng.permutation(tG.zarr_ep.values))
                             for tK, tG in sorted(tD[tD.session == tS].groupby("set_key"))}
        tChosen = []
        while len(tChosen) < perOrder:
            tProgressed = False
            for tS in sorted(tSessions):
                if len(tChosen) >= perOrder:
                    break
                tSets = tSessions[tS]
                tLive = [tK for tK in sorted(tSets) if tSets[tK]]
                if not tLive:
                    continue
                tK = tLive[int(rng.integers(len(tLive)))]
                tChosen.append(int(tSets[tK].pop(0)))
                tProgressed = True
            assert tProgressed
        tOut += tChosen
    return sorted(tOut)


def selectionDoc(name, sel, parent, center, rbp):
    tS = center[center.zarr_ep.isin(sel)]
    tDoc = dict(schema=f"{name.split('_')[0]}/v1", created=CREATED,
                rule="pan=center (explicit, phase2_episodes.jsonl) only, phase1 excluded, QA-v2 grade A, exactly "
                     f"{len(sel)//N_ORDERS} per order, session round-robin then set round-robin, rng 20260928"
                     + (f"; nested in {parent}" if parent else ""),
                global_transform="rot180", zarr=str(NODE_HOME / "umi_bridge/r675rbp_umi.zarr"),
                zarr_episodes=[int(tX) for tX in sel], r675_pool_episodes=[int(rbp[tX]) for tX in sel],
                order_counts=tS.order.value_counts().sort_index().to_dict(),
                session_counts=tS.session.value_counts().to_dict(), n_sessions=int(tS.session.nunique()),
                n_sets=int(tS.set_key.nunique()), max_per_set=int(tS.set_key.value_counts().max()), subset_of=parent)
    tDoc["zarr_episodes_sha256"] = hashlib.sha256(json.dumps(tDoc["zarr_episodes"]).encode()).hexdigest()
    return tDoc


def main(argv: list[str] | None = None) -> int:
    tHere = str(UMI_BRIDGE / "rel16_audit/r380")
    tOutDir = str(UMI_BRIDGE / "umi76")
    tCenter = loadPool(tHere, HOLOBRAIN_DATA)
    tRng = np.random.default_rng(SELECTION_SEED)
    tF132 = pick(tCenter, FRONT_PER_ORDER, tRng)
    tF66 = pick(tCenter[tCenter.zarr_ep.isin(tF132)], HALF_PER_ORDER, tRng)
    assert set(tF66) <= set(tF132) and len(tF132) == N_ORDERS * FRONT_PER_ORDER and len(tF66) == N_ORDERS * HALF_PER_ORDER
    tRbp = json.load(open(f"{tHere}/r675_rbp.json"))["episodes"]
    for tName, tSel, tParent in (("r312c_front132_v1", tF132, None), ("r246c_front66_v1", tF66, "r312c_front132_v1.json")):
        tDoc = selectionDoc(tName, tSel, tParent, tCenter, tRbp)
        json.dump(tDoc, open(f"{tOutDir}/{tName}.json", "w"), indent=1)
        print(f"{tName}: {len(tSel)} eps  orders {tDoc['order_counts']}  sessions {tDoc['n_sessions']}  sets {tDoc['n_sets']}  "
              f"max/set {tDoc['max_per_set']}")
    tCur = pd.read_csv(f"{tHere}/front200_pan.csv")
    print(f"overlap with old FRONT200 center subset: {len(set(tF132) & set(tCur[tCur.pan == 'center'].zarr_ep))}/132")
    print("pool per session available (center A):", tCenter.session.value_counts().describe()[["count", "min", "max"]].to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
