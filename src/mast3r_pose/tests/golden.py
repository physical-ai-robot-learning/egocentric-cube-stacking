"""Golden-file helper of the characterization tests: compare a JSON-able result with tests/golden/<name>.json.

REGEN_GOLDEN=1 rewrites the file (only ever against the pre-rewrite code); a missing file is a failure, never a pass."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

REL_TOL = 1e-9
ABS_TOL = 1e-12


def _close(tGot, tWant, tPath: str) -> None:
    if isinstance(tWant, dict):
        assert isinstance(tGot, dict) and sorted(tGot) == sorted(tWant), f"{tPath}: keys {sorted(tGot)} != {sorted(tWant)}"
        for tKey in tWant:
            _close(tGot[tKey], tWant[tKey], f"{tPath}.{tKey}")
    elif isinstance(tWant, list):
        assert isinstance(tGot, list) and len(tGot) == len(tWant), f"{tPath}: length {len(tGot)} != {len(tWant)}"
        for tI, (tA, tB) in enumerate(zip(tGot, tWant)):
            _close(tA, tB, f"{tPath}[{tI}]")
    elif isinstance(tWant, float) or isinstance(tGot, float):
        if tWant is None or tGot is None:
            assert tGot is tWant, f"{tPath}: {tGot} != {tWant}"
        elif math.isnan(tWant):
            assert math.isnan(tGot), f"{tPath}: {tGot} != nan"
        else:
            assert math.isclose(tGot, tWant, rel_tol=REL_TOL, abs_tol=ABS_TOL), f"{tPath}: {tGot!r} != {tWant!r}"
    else:
        assert tGot == tWant, f"{tPath}: {tGot!r} != {tWant!r}"


def assertGolden(goldenDir: Path, name: str, result) -> None:
    """result must survive a JSON round trip (NaN allowed); compared to ~1e-9 relative."""
    tPath = Path(goldenDir) / f"{name}.json"
    tResult = json.loads(json.dumps(result, default=float))
    if os.environ.get("REGEN_GOLDEN") == "1":
        tPath.parent.mkdir(parents=True, exist_ok=True)
        tPath.write_text(json.dumps(tResult, indent=1, sort_keys=True))
    assert tPath.is_file(), f"golden file {tPath} missing (REGEN_GOLDEN=1 against the pre-rewrite code)"
    _close(tResult, json.loads(tPath.read_text()), name)
