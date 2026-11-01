"""Import smoke test: every module under src/ must import from a clean checkout.

Each module is imported in its own subprocess with
  - PYTHONPATH = <repo>/src:<repo> (the same as pyproject's pytest `pythonpath`),
  - HOME and the working directory set to an empty temporary folder, so nothing outside the repository
    (data roots, other checkouts under ~) can make an import pass, and nothing is written into the real home,
  - network connections and child processes disabled (a module that talks to the robot at import time fails here
    instead of reaching it).

Result per module:
  ok    imported
  skip  an optional third-party dependency is missing (OPTIONAL below), or the module is an entry script that runs
        its program at import (top-level `main()` call)
  fail  anything else (a module missing from the repository, a data file read at import, a syntax error, a timeout)

The test fails if a module fails that is not listed in tests/import_smoke_known_failures.txt (the list may only shrink).
Report only:  python tests/test_imports.py [--json out.json]
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
KNOWN = Path(__file__).with_name("import_smoke_known_failures.txt")

# Optional third-party packages: a module that needs one of these is skipped, not failed, when it is missing.
OPTIONAL = {
    # heavy / platform-specific (named in the refactor spec)
    "torch", "torchvision", "lerobot", "pinocchio", "pink", "pyorbbecsdk", "cv2", "PySide6", "gtsam",
    # other optional extras and node-side libraries
    "mediapipe", "dex_retargeting", "pupil_apriltags", "rerun", "zarr", "bitsandbytes", "diffusion_policy", "qpsolvers",
    "flask", "mcap", "mcap_protobuf", "serial", "scservo_sdk", "av", "trimesh", "open3d", "sklearn", "numba",
    "pyrealsense2", "requests", "transformers", "safetensors", "huggingface_hub", "accelerate", "datasets", "einops",
    "lietorch", "mast3r_slam", "mast3r", "dust3r", "lerobot_plugin_rebot", "bh_indy7_lerobot", "bh_indy7_lerobot_gui",
    "imagecodecs", "numcodecs", "PIL", "ultralytics",
    "onnxruntime", "coremltools", "ahrs", "filterpy", "pyquaternion", "kornia", "timm", "pytorch3d", "smplx",
    "pyvista", "plotly", "rosbags", "hid", "usb", "uvc", "AVFoundation", "objc", "Foundation", "Quartz",
}

_CHILD = r"""
import importlib, json, socket, subprocess, sys
def _blocked(*a, **k): raise OSError("blocked by the import smoke test")
socket.socket.connect = _blocked; socket.socket.connect_ex = _blocked; socket.create_connection = _blocked
subprocess.Popen.__init__ = _blocked
sys.argv = ["import-smoke"]
name = %r
try:
    importlib.import_module(name)
    print(json.dumps({"status": "ok"}))
except ModuleNotFoundError as e:
    print(json.dumps({"status": "missing", "missing": (e.name or "").split(".")[0], "error": repr(e)[:300]}))
except BaseException as e:
    print(json.dumps({"status": "error", "error": (type(e).__name__ + ": " + str(e))[:300]}))
"""


def discover(src: Path = SRC) -> list[tuple[str, Path]]:
    mods = []
    for p in sorted(src.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(src).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or not all(x.isidentifier() for x in parts):
            continue
        mods.append((".".join(parts), p))
    return mods


def _runs_main_at_import(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if (isinstance(f, ast.Name) and f.id == "main") or (isinstance(f, ast.Attribute) and f.attr == "main"):
                return True
    return False


def check(name: str, path: Path, python: str = sys.executable, timeout: float = 120.0) -> dict:
    if path.name == "__main__.py" or _runs_main_at_import(path):
        return {"module": name, "status": "skip", "reason": "entry script: runs its program at import"}
    with tempfile.TemporaryDirectory(prefix="import_smoke_") as home:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "EGO_", "V4_", "M3_", "UMI_", "REBOT_"))}
        env.update(HOME=home, PYTHONPATH=f"{SRC}{os.pathsep}{REPO}", PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen",
                   MPLBACKEND="Agg")
        try:
            r = subprocess.run([python, "-c", _CHILD % name], cwd=home, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"module": name, "status": "fail", "reason": f"timeout after {timeout:.0f} s"}
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
    if line is None:
        return {"module": name, "status": "fail", "reason": f"no result (exit {r.returncode}): {r.stderr.strip()[-300:]}"}
    res = json.loads(line)
    if res["status"] == "ok":
        return {"module": name, "status": "ok"}
    if res["status"] == "missing" and res["missing"] in OPTIONAL:
        return {"module": name, "status": "skip", "reason": f"optional dependency missing: {res['missing']}"}
    return {"module": name, "status": "fail", "reason": res["error"]}


def run_all(python: str = sys.executable, workers: int = 8) -> list[dict]:
    mods = discover()
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda m: check(m[0], m[1], python), mods))


def known_failures() -> set[str]:
    if not KNOWN.exists():
        return set()
    return {l.split("#")[0].strip() for l in KNOWN.read_text().splitlines() if l.split("#")[0].strip()}


def test_import_smoke():
    res = run_all()
    counts = {s: sum(r["status"] == s for r in res) for s in ("ok", "skip", "fail")}
    print(f"\nimport smoke: {counts}")
    new = sorted(r["module"] + ": " + r["reason"] for r in res if r["status"] == "fail" and r["module"] not in known_failures())
    assert not new, "modules that newly fail to import:\n" + "\n".join(new)


if __name__ == "__main__":
    res = run_all()
    counts = {s: sum(r["status"] == s for r in res) for s in ("ok", "skip", "fail")}
    for r in res:
        if r["status"] != "ok":
            print(f"{r['status']:4s} {r['module']}: {r['reason']}")
    print(json.dumps(counts))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(res, indent=1))
