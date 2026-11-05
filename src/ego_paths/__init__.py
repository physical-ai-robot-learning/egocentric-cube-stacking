"""One place for the machine-specific roots that the code in this repository reads and writes.

Every default is exactly the literal the code used before 2026-11-01, so on the original machines nothing changes.
Override a root with its environment variable (read once, at import time):

| constant | default | environment variable | what lives there |
|---|---|---|---|
| `REPO_ROOT` | this checkout | — | configs/, results/, ... |
| `CALIBRATION_DIR` | `REPO_ROOT/configs/calibration` | `EGO_CALIBRATION_DIR` | frozen HandUMI calibration (fisheye, camera-IMU, gripper) |
| `HOME` | `Path.home()` | (`HOME`) | |
| `DATA_ROOT` | `~/c8` | `EGO_DATA_ROOT` | Mac-side C8 / Gen-1 / Gen-2 working data (census, stage, robotlike, rel16ego, hra_red, ...) |
| `UMI_BRIDGE` | `~/umi_bridge` | `EGO_UMI_BRIDGE` | Track A/B/C working folders, rel16_audit, learned_ik, UI state |
| `EGO_COLLECTOR_HOME` | `~/ego_collector` | `EGO_COLLECTOR_HOME` | the original collector working tree (raw datasets, configs/calibration) |
| `HOLOBRAIN_MAC` | `~/holobrain-mac-model` | `EGO_HOLOBRAIN_MAC` | Mac inference checkout: checkpoints, URDF, rebot_fk |
| `HOLOBRAIN_DATA` | `~/holobrain-data` | `EGO_HOLOBRAIN_DATA` | Mac-side LeRobot datasets |
| `NODE_HOME` | `/home/bh-aiteam` | `EGO_NODE_HOME` | GPU-node home: c8old/xvla, lerobot-seeed, holobrain-data, umi_bridge, bases |
| `SHARED_ROOT` | `/srv/data/johann` | `EGO_SHARED_ROOT` | 5090-node shared storage (MASt3R runs, relonly code bundles) |
| `CKPT_ROOT` | `/Volumes/PortableSSD/rebot_ckpts_archive` | `EGO_CKPT_ROOT` | external SSD checkpoint archive |
| `LAUNDRY_SYSTEM`, `LAUNDRY_DIT_DATASETS` | `~/laundry_system`, `~/laundry_dit_training/datasets` | `EGO_LAUNDRY_SYSTEM`, `EGO_LAUNDRY_DIT_DATASETS` | external checkouts the robotize code reuses |

`prepend_sys_path()` is the one place where code under `src/` puts an external checkout (one that is not part of this
repository, e.g. the GPU node's `c8old/xvla` or LeRobot fork) in front of `sys.path`.
Pipeline stage: none (shared by every stage). Usage: `from ego_paths import DATA_ROOT`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()


def _root(env: str, default: Path | str) -> Path:
    """the environment variable's value when set and non-empty, else the default"""
    tValue = os.environ.get(env)
    return Path(tValue) if tValue else Path(default)


DATA_ROOT = _root("EGO_DATA_ROOT", HOME / "c8")
UMI_BRIDGE = _root("EGO_UMI_BRIDGE", HOME / "umi_bridge")
EGO_COLLECTOR_HOME = _root("EGO_COLLECTOR_HOME", HOME / "ego_collector")
HOLOBRAIN_MAC = _root("EGO_HOLOBRAIN_MAC", HOME / "holobrain-mac-model")
HOLOBRAIN_DATA = _root("EGO_HOLOBRAIN_DATA", HOME / "holobrain-data")
NODE_HOME = _root("EGO_NODE_HOME", "/home/bh-aiteam")
SHARED_ROOT = _root("EGO_SHARED_ROOT", "/srv/data/johann")
CKPT_ROOT = _root("EGO_CKPT_ROOT", "/Volumes/PortableSSD/rebot_ckpts_archive")
LAUNDRY_SYSTEM = _root("EGO_LAUNDRY_SYSTEM", HOME / "laundry_system")
LAUNDRY_DIT_DATASETS = _root("EGO_LAUNDRY_DIT_DATASETS", HOME / "laundry_dit_training/datasets")
CALIBRATION_DIR = _root("EGO_CALIBRATION_DIR", REPO_ROOT / "configs" / "calibration")


def prepend_sys_path(*dirs: Path | str) -> None:
    """Put the directories at the front of sys.path, in the given order: `prepend_sys_path(a)` == `sys.path.insert(0, a)`,
    `prepend_sys_path(a, b)` == `sys.path[:0] = [a, b]` (a is searched first)."""
    sys.path[:0] = [str(tDir) for tDir in dirs]


__all__ = ["REPO_ROOT", "HOME", "DATA_ROOT", "UMI_BRIDGE", "EGO_COLLECTOR_HOME", "HOLOBRAIN_MAC", "HOLOBRAIN_DATA", "NODE_HOME",
           "SHARED_ROOT", "CKPT_ROOT", "LAUNDRY_SYSTEM", "LAUNDRY_DIT_DATASETS", "CALIBRATION_DIR", "prepend_sys_path"]
