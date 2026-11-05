"""One raw episode -> one processed episode directory (same code path as convert_dataset, no split / manifest).

Prints a one-line JSON summary (episode_id, stack_order, frames_rows, rows_valid, train_rows, duration_s).
Pipeline stage: dataset stage (single-episode debugging of convert_dataset).
usage: python -m ego_cart20.scripts.convert_episode <raw_episode_dir> <out_episode_dir>
"""
from __future__ import annotations

import argparse
import json
import sys

from ..convert import convertEpisode
from ..processed_episode import writeEpisode
from ..raw_episode import loadRawEpisode

SUMMARY_KEYS = ("episode_id", "stack_order", "frames_rows", "rows_valid", "train_rows", "duration_s")


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description="one raw episode -> one processed episode")
    tParser.add_argument("raw_episode_dir")
    tParser.add_argument("out_episode_dir")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tEp = convertEpisode(loadRawEpisode(tArgs.raw_episode_dir))
    writeEpisode(tArgs.out_episode_dir, tEp)
    print(json.dumps({tK: tEp["metadata"][tK] for tK in SUMMARY_KEYS}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
