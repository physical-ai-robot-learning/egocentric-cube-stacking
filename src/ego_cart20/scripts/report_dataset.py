"""Visual report of a processed ego_cart20 tree: trajectory_samples.png (n random training samples, half of them the most
moving row of their episode) + the per-split order table of metadata.json on stdout.

Pipeline stage: dataset-stage QA (after convert_dataset).
usage: python -m ego_cart20.scripts.report_dataset <processed_root> <out_dir> [n_samples=4]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from ..processed_episode import loadEpisode
from ..trajectory_plot import plotSamples

LAST_TARGET = 15                 # k = 16, the farthest CART20 target
LEFT_XYZ = slice(0, 3)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description="trajectory_samples.png + order / split table of a processed tree")
    tParser.add_argument("processed_root")
    tParser.add_argument("out_dir")
    tParser.add_argument("n_samples", nargs="?", type=int, default=4)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tRoot, tOut = pathlib.Path(tArgs.processed_root), pathlib.Path(tArgs.out_dir)
    tOut.mkdir(parents=True, exist_ok=True)
    tManifest = [json.loads(tLine) for tLine in open(tRoot / "train_manifest.jsonl")]
    tRng = np.random.default_rng(0)
    tCharts, tTitles = [], []
    for tRec in tRng.choice(tManifest, tArgs.n_samples, replace=False):
        tEp = loadEpisode(tRoot / tRec["path"])
        tJ = int(tRng.integers(len(tEp["train_rows"])))
        if tJ % 2:                                       # half random, half the most moving row
            tJ = int(np.argmax(np.linalg.norm(tEp["cart20"][:, LAST_TARGET, LEFT_XYZ], axis=1)))
        tCharts.append(np.asarray(tEp["cart20"][tJ], np.float64))
        tTitles.append(f"{tRec['episode_id']} row {int(tEp['train_rows'][tJ])} ({tRec['stack_order']})")
    plotSamples(tCharts, tTitles, tOut / "trajectory_samples.png")
    print("wrote", tOut / "trajectory_samples.png")
    tMeta = json.load(open(tRoot / "metadata.json"))
    print(json.dumps(tMeta["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
