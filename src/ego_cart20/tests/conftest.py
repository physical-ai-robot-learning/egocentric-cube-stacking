"""Shared fixtures of the ego_cart20 characterization tests: the golden-value store.

The expected values were recorded from the package as it was before the 2026-11-05 OA-style rewrite
(`EGO_CART20_RECORD=1 pytest src/ego_cart20/tests` writes golden.npz / golden.json); every later run compares against them.
Arrays: exact for int / bool / str, rtol 1e-9 and atol 1e-12 for floats; JSON-like values are compared after a JSON round
trip with the same float tolerance.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN_NPZ = HERE / "golden.npz"
GOLDEN_JSON = HERE / "golden.json"
RECORD_ENV = "EGO_CART20_RECORD"
FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-12


def _jsonable(value):
    """JSON round trip, numpy scalars / arrays / tuples / NaN made comparable"""
    def _default(tObj):
        if isinstance(tObj, np.ndarray):
            return tObj.tolist()
        if isinstance(tObj, np.generic):
            return tObj.item()
        if isinstance(tObj, (set, frozenset)):
            return sorted(tObj)
        if isinstance(tObj, pathlib.Path):
            return str(tObj)
        raise TypeError(type(tObj))
    return json.loads(json.dumps(value, default=_default))


def _assertClose(got, want, path):
    if isinstance(want, dict):
        assert isinstance(got, dict) and list(got) == list(want), (path, list(got), list(want))
        for tKey in want:
            _assertClose(got[tKey], want[tKey], f"{path}.{tKey}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (path, got, want)
        for tI, (tG, tW) in enumerate(zip(got, want)):
            _assertClose(tG, tW, f"{path}[{tI}]")
    elif isinstance(want, float) and not isinstance(got, bool) and isinstance(got, (int, float)):
        if math.isnan(want):
            assert math.isnan(got), (path, got)
        else:
            assert math.isclose(got, want, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL), (path, got, want)
    else:
        assert got == want and type(got) is type(want), (path, got, want)


class Golden:
    """check(key, value): record (EGO_CART20_RECORD=1) or compare against the recorded value"""

    def __init__(self):
        self.mRecord = os.environ.get(RECORD_ENV) == "1"
        self.mArrays = {}
        self.mJson = {}
        if not self.mRecord:
            with np.load(GOLDEN_NPZ, allow_pickle=False) as tZ:
                self.mArrays = {tK: tZ[tK] for tK in tZ.files}
            self.mJson = json.loads(GOLDEN_JSON.read_text())

    def check(self, key: str, value) -> None:
        if isinstance(value, np.ndarray) and value.dtype != object:
            if self.mRecord:
                assert key not in self.mArrays, f"duplicate golden key {key}"
                self.mArrays[key] = value
                return
            tWant = self.mArrays[key]
            assert value.shape == tWant.shape and value.dtype == tWant.dtype, \
                (key, value.shape, tWant.shape, value.dtype, tWant.dtype)
            if np.issubdtype(tWant.dtype, np.floating):
                np.testing.assert_allclose(value, tWant, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, err_msg=key)
            else:
                np.testing.assert_array_equal(value, tWant, err_msg=key)
            return
        tValue = _jsonable(value)
        if self.mRecord:
            assert key not in self.mJson, f"duplicate golden key {key}"
            self.mJson[key] = tValue
            return
        _assertClose(tValue, self.mJson[key], key)

    def save(self) -> None:
        if self.mRecord:
            np.savez_compressed(GOLDEN_NPZ, **self.mArrays)
            GOLDEN_JSON.write_text(json.dumps(self.mJson, indent=0))


@pytest.fixture(scope="session")
def golden():
    tStore = Golden()
    yield tStore
    tStore.save()
