"""Episode-level splits (spec 22, was dataset/splits.py).  An episode belongs to exactly one split; frames are never split.

``frozenHandumiSplit``: the split already shared by every ego dataset (C-old v1/v2, REL16 ego, gcal1):
    val = the frozen C-old v1 26 old259 source episodes  +  the HRL80 round-block holdout ep49-54 (hrl80_split_frozen.json)
    train = every other included episode;  test = empty (no third split was ever frozen; adding one would change train)
``hashSplit``: deterministic sha1(episode_id) buckets for other sources.

Pipeline stage: dataset stage (scripts/convert_dataset.py writes the split manifests with it).
"""
from __future__ import annotations

import hashlib
import json

from ego_paths import DATA_ROOT, UMI_BRIDGE  # machine-specific roots (env-overridable)

HASH_HEX_DIGITS = 8                 # sha1 prefix used as the bucket value
HASH_BUCKET_MAX = 0xFFFFFFFF        # = 16 ** HASH_HEX_DIGITS - 1, maps the prefix to [0, 1]


def frozenHandumiSplit(episodeIds) -> tuple[dict, dict]:
    """-> ({episode_id: "train" | "val"}, rule record)"""
    tV1 = set(json.load(open(DATA_ROOT / "c8old_rows/index.json"))["val_episodes"])
    tHrl = json.load(open(UMI_BRIDGE / "track_c/v2k/hrl80_split_frozen.json"))
    tVal = tV1 | set(tHrl["val_source_episodes"])
    tSplit = {tE: ("val" if tE in tVal else "train") for tE in episodeIds}
    return tSplit, dict(rule="frozen: val = C-old v1 26 (c8old_rows/index.json) + HRL80 round-block ep49-54; test empty",
                        hrl80_split_sha256=tHrl.get("sha256"))


def hashSplit(episodeIds, valFrac=0.1, testFrac=0.0) -> tuple[dict, dict]:
    """-> ({episode_id: "train" | "val" | "test"}, rule record)"""
    tSplit = {}
    for tE in episodeIds:
        tU = int(hashlib.sha1(tE.encode()).hexdigest()[:HASH_HEX_DIGITS], 16) / HASH_BUCKET_MAX
        tSplit[tE] = "test" if tU < testFrac else ("val" if tU < testFrac + valFrac else "train")
    return tSplit, dict(rule=f"sha1 bucket val {valFrac} test {testFrac}")


def checkDisjoint(split: dict) -> bool:
    """AssertionError if an episode is assigned two different splits"""
    tSeen = {}
    for tE, tS in split.items():
        assert tSeen.setdefault(tE, tS) == tS
    return True
