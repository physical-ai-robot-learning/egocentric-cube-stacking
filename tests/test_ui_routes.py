"""The reBot smoke UI (src/rebot_deploy/mac_v4_smoke_ui.py, split into smoke_ui/ on 2026-11-01) against a golden record.

tests/ui/smoke_ui_probe.py drives every read-only route and a set of POST routes through Flask's test client with the robot and
the network stubbed, and evaluates the pure helpers (chunk smoothing, stream merge, gripper conversions, ...) on synthetic input.
tests/ui/golden/smoke_ui_routes.json is that probe's output on the single-file UI before the split (commit 77a4632); the test
asserts the current code gives the same output: url map, the served page (sha256), JSON of the routes, the live settings after
POST /mode, and the helper values (floats to 1e-9 relative).

The UI needs flask, torch, scipy and cv2. The test runs the probe with UI_TEST_PYTHON (default: this interpreter) and skips
when that interpreter lacks them, e.g.
    UI_TEST_PYTHON=~/xvla-mac/bin/python pytest tests/test_ui_routes.py -q
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests/ui/smoke_ui_probe.py"
GOLDEN = REPO / "tests/ui/golden/smoke_ui_routes.json"
PY = os.path.expanduser(os.environ.get("UI_TEST_PYTHON", sys.executable))


def _usable(py):
    chk = "import flask, torch, scipy, cv2"
    try:
        return subprocess.run([py, "-c", chk], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _same(a, b, path="$"):
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {a} != {b}"
            return
    if isinstance(a, dict) and isinstance(b, dict):
        assert sorted(a) == sorted(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a: _same(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)): _same(x, y, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def probe():
    if not _usable(PY):
        pytest.skip(f"{PY}: needs flask, torch, scipy, cv2 (set UI_TEST_PYTHON)")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "probe.json"
        r = subprocess.run([PY, str(PROBE), str(REPO), str(out)], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        return json.loads(out.read_text())


def test_page_and_url_map(probe):
    g = json.loads(GOLDEN.read_text())
    assert probe["page_sha256"] == g["page_sha256"] and probe["page_len"] == g["page_len"]
    assert probe["url_map"] == g["url_map"]


def test_routes_settings_and_helpers(probe):
    g = json.loads(GOLDEN.read_text())
    assert sorted(probe) == sorted(g)
    for k in g:
        _same(probe[k], g[k], k)
