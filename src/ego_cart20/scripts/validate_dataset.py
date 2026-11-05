"""Integrity report of a processed ego_cart20 tree (integrity.validateDataset); exit 1 on any hard failure.

Writes the report to [report.json] (default <processed_root>/integrity_report.json), prints it, then
"VALIDATION PASS" or "VALIDATION FAIL [...]".
Pipeline stage: dataset-stage validation (after convert_dataset, before export_lerobot).
usage: python -m ego_cart20.scripts.validate_dataset <processed_root> [report.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from ..integrity import validateDataset


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description="integrity report of a processed ego_cart20 tree; exit 1 on any hard failure")
    tParser.add_argument("processed_root")
    tParser.add_argument("report", nargs="?", default=None, help="default <processed_root>/integrity_report.json")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tRep = validateDataset(tArgs.processed_root)
    tDefault = pathlib.Path(tArgs.processed_root) / "integrity_report.json"
    tOut = pathlib.Path(tArgs.report if tArgs.report is not None else tDefault)
    json.dump(tRep, open(tOut, "w"), indent=1)
    print(json.dumps(tRep, indent=1))
    print("VALIDATION", "PASS" if tRep["pass"] else f"FAIL {tRep['fails']}")
    return 0 if tRep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
