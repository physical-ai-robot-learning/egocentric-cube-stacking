"""Test fixtures of rebot_deploy: a golden-value recorder and stand-ins for the hardware / model parts the package touches.

`golden`  compares every value a test hands it against tests/golden_deploy.json (floats to 1e-9 relative, everything else
          exactly). With REBOT_DEPLOY_CHAR_RECORD=1 the values are recorded instead (the characterization run against the
          pre-refactor code).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

GOLDEN_FILE = Path(__file__).with_name("golden_deploy.json")
RECORD = os.environ.get("REBOT_DEPLOY_CHAR_RECORD") == "1"
REL_TOL = 1e-9
ABS_TOL = 1e-12


def _plain(value):
    """numpy / tuple values -> JSON-able python values"""
    if isinstance(value, dict):
        return {str(tKey): _plain(tVal) for tKey, tVal in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(tVal) for tVal in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return _plain(value.detach().cpu().numpy())
    return value


def _same(got, exp, path):
    if isinstance(got, bool) or isinstance(exp, bool) or got is None or exp is None:
        assert got == exp, f"{path}: {got!r} != {exp!r}"
        return
    if isinstance(got, (int, float)) and isinstance(exp, (int, float)):
        assert (math.isnan(got) and math.isnan(exp)) or math.isclose(got, exp, rel_tol=REL_TOL, abs_tol=ABS_TOL), \
            f"{path}: {got} != {exp}"
        return
    if isinstance(got, dict) and isinstance(exp, dict):
        assert sorted(got) == sorted(exp), f"{path}: keys {sorted(set(got) ^ set(exp))}"
        for tKey in exp:
            _same(got[tKey], exp[tKey], f"{path}.{tKey}")
        return
    if isinstance(got, list) and isinstance(exp, list):
        assert len(got) == len(exp), f"{path}: length {len(got)} != {len(exp)}"
        for tIndex, (tGot, tExp) in enumerate(zip(got, exp)):
            _same(tGot, tExp, f"{path}[{tIndex}]")
        return
    assert got == exp, f"{path}: {got!r} != {exp!r}"


class Golden:
    def __init__(self):
        self.mStore = json.loads(GOLDEN_FILE.read_text()) if GOLDEN_FILE.exists() else {}

    def check(self, name, value):
        tPlain = json.loads(json.dumps(_plain(value)))
        if RECORD:
            self.mStore[name] = tPlain
            return
        assert name in self.mStore, f"{name}: no golden value (record with REBOT_DEPLOY_CHAR_RECORD=1)"
        _same(tPlain, self.mStore[name], name)

    def save(self):
        if RECORD:
            GOLDEN_FILE.write_text(json.dumps(self.mStore, indent=1, sort_keys=True) + "\n")


_GOLDEN = Golden()


@pytest.fixture(scope="session")
def golden():
    yield _GOLDEN
    _GOLDEN.save()
