"""raw ego episodes -> processed ego_cart20_v2 tree (spec 23): episodes/<id>/..., metadata.json, {train,val,test}_manifest.jsonl.

Every <raw_root>/<id> with a raw_episode.json is converted (convert.convertEpisode, one process per episode); episodes with
at least one training row are split (frozen HandUMI split or sha1 hash split) into the manifests.  The output tree is
write-once.  --discontinuity-filter / --jump-* select the v2b raw jump filter.

Pipeline stage: dataset stage (after handumi_export, before validate_dataset / export_lerobot).
usage: python -m ego_cart20.scripts.convert_dataset <raw_root> <out_root> [--split frozen_handumi|hash] [--workers N]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import time

from ..config import CONTRACT, Cart20Config
from ..convert import convertEpisode
from ..processed_episode import writeEpisode
from ..raw_episode import loadRawEpisode
from ..splits import frozenHandumiSplit, hashSplit

SPLITS = ("train", "val", "test")
CODE_SHA_HEX_DIGITS = 16


def _convertOne(args):
    """worker: (raw_dir, out_root, cfg) -> conversion record (status ok / error) (was one)"""
    tRawDir, tOutRoot, tCfg = args
    try:
        tRaw = loadRawEpisode(tRawDir)
        tEp = convertEpisode(tRaw, tCfg)
        writeEpisode(pathlib.Path(tOutRoot) / "episodes" / tRaw.mEpisodeId, tEp)
        tMeta = tEp["metadata"]
        return dict(episode_id=tRaw.mEpisodeId, status="ok", stack_order=tMeta["stack_order"], instruction=tMeta["instruction"],
                    frames_rows=tMeta["frames_rows"], rows_valid=tMeta["rows_valid"], train_rows=tMeta["train_rows"],
                    era=tRaw.mMeta.get("era"), jump_events=tMeta.get("jump_events"))
    except Exception as tEx:
        return dict(episode_id=pathlib.Path(tRawDir).name, status="error", reason=f"{type(tEx).__name__}: {tEx}")


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("raw_root")
    tParser.add_argument("out_root")
    tParser.add_argument("--split", default="frozen_handumi")
    tParser.add_argument("--workers", type=int, default=6)
    tParser.add_argument("--discontinuity-filter", action="store_true",
                         help="v2b: drop raw MASt3R silent jumps (pose_filter.discontinuity_filter)")
    tParser.add_argument("--jump-window", default="frame", choices=["frame", "row"])
    tParser.add_argument("--jump-isolated-step-m", type=float, default=None)
    return tParser


def _codeSha256() -> dict:
    """{path relative to the package: sha256[:16]} of every .py file of the ego_cart20 package"""
    tPackage = pathlib.Path(__file__).resolve().parents[1]
    return {str(tP.relative_to(tPackage)): hashlib.sha256(tP.read_bytes()).hexdigest()[:CODE_SHA_HEX_DIGITS]
            for tP in sorted(tPackage.rglob("*.py"))}


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tCfg = Cart20Config(mDiscontinuityFilter=tArgs.discontinuity_filter, mJumpWindow=tArgs.jump_window,
                        mJumpIsolatedStepM=tArgs.jump_isolated_step_m)
    tOut = pathlib.Path(tArgs.out_root)
    assert not tOut.exists(), f"{tOut} exists (processed trees are write-once)"
    (tOut / "episodes").mkdir(parents=True)
    tRaws = sorted(tP for tP in pathlib.Path(tArgs.raw_root).iterdir() if (tP / "raw_episode.json").exists())
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(tArgs.workers) as tEx:
        tResults = list(tEx.map(_convertOne, [(str(tR), str(tOut), tCfg) for tR in tRaws]))
    tOk = [tR for tR in tResults if tR["status"] == "ok" and tR["train_rows"] > 0]
    tSplit, tRule = (frozenHandumiSplit if tArgs.split == "frozen_handumi" else hashSplit)([tR["episode_id"] for tR in tOk])
    for tS in SPLITS:
        with open(tOut / f"{tS}_manifest.jsonl", "w") as tF:
            for tR in tOk:
                if tSplit[tR["episode_id"]] == tS:
                    tF.write(json.dumps(dict(tR, split=tS, path=f"episodes/{tR['episode_id']}")) + "\n")
    tCounts = {tS: dict(episodes=sum(tSplit[tR["episode_id"]] == tS for tR in tOk),
                        train_rows=sum(tR["train_rows"] for tR in tOk if tSplit[tR["episode_id"]] == tS),
                        orders=dict(collections.Counter(tR["stack_order"] for tR in tOk if tSplit[tR["episode_id"]] == tS)))
               for tS in SPLITS}
    tMeta = dict(schema="ego_cart20_v2/v1", created=time.strftime("%F %T"), raw_root=str(pathlib.Path(tArgs.raw_root).resolve()),
                 contract=CONTRACT, config=tCfg.toDict(), split=tRule, counts=tCounts, conversion=tResults,
                 code_sha256=_codeSha256())
    json.dump(tMeta, open(tOut / "metadata.json", "w"), indent=1)
    print("DONE", json.dumps(tCounts), collections.Counter(tR["status"] for tR in tResults))
    return 0


if __name__ == "__main__":
    sys.exit(main())
