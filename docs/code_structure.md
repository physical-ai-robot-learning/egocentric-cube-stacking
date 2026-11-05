# Code structure, interfaces and how the code runs

How the Python code is organised after the 2026-11-05 refactor (OA style; log: [`refactor_2026-11.md`](refactor_2026-11.md)), the
data interfaces between the stages, the machine-specific roots, and where each part ran.

## Packages

One installable project (`pyproject.toml`, src layout). Each package is one pipeline stage and has a README (Status,
Inputs → Outputs, Entry points, Layout, Provenance) and characterization tests in `src/<pkg>/tests/`.

| package | stage | style |
|---|---|---|
| `handumi_collector` (+ `ego_collector.camera`) | 0 capture | shared copy (handumi-common v0.3.0), trimmed to the import closure of the collector, its calibration / QA tools and the exporter; files unchanged, **not restyled** |
| `mast3r_pose` | 1 pose | OA style; `run_perframe*` need MASt3R-SLAM + GPU (not behaviour-checked) |
| `ego_cart20` | 2 Gen-2 conversion | OA style, shallow module tree |
| `rebot_prep` | 2–3 Gen-1 conversion | OA style; public names shared with external twin copies kept |
| `training` | 4 training plugins / dataset tooling | OA style; file names are the node launchers' interface; `relonly/rel16_aux_relonly.py` byte-identical (sha-pinned) |
| `rebot_deploy` | 5 deployment | core OA style (`xvla_inference`, `robot_dataset_contract`, `pink_ik`, `eef_kin`, `usb_arm_ports`); `smoke_ui/` and `robot_service_mit.py` moved, not restyled |
| `ego_paths` | machine-specific roots | OA style |
| `umi` | UMI `pose_util` stand-in | API names = upstream UMI |

Rules: relative imports inside a package, absolute across packages, no `sys.path` edits except `ego_paths.prepend_sys_path(...)`
for external checkouts (LeRobot fork, MASt3R-SLAM, the node's `c8old/xvla`), no work at import time (every runnable module has
`main(argv) -> int` and runs as `python -m <pkg>.<module>`). `tests/test_imports.py` imports every module under `src/` in a clean
subprocess (empty HOME, network and child processes blocked): 2026-11-05, 232 modules, 230 ok, 2 skipped (lietorch), 0 failed.

## Machine-specific roots: `ego_paths`

[`src/ego_paths/__init__.py`](../src/ego_paths/__init__.py) is the one place that names the data roots. Each default is the literal the
code used before, so nothing changes on the original machines; each can be overridden by an environment variable:

| constant | default | override |
|---|---|---|
| `DATA_ROOT` | `~/c8` | `EGO_DATA_ROOT` |
| `UMI_BRIDGE` | `~/umi_bridge` | `EGO_UMI_BRIDGE` |
| `EGO_COLLECTOR_HOME` | `~/ego_collector` | `EGO_COLLECTOR_HOME` |
| `HOLOBRAIN_MAC`, `HOLOBRAIN_DATA` | `~/holobrain-mac-model`, `~/holobrain-data` | `EGO_HOLOBRAIN_MAC`, `EGO_HOLOBRAIN_DATA` |
| `NODE_HOME` | the GPU node's home | `EGO_NODE_HOME` |
| `SHARED_ROOT` | the shared NFS root | `EGO_SHARED_ROOT` |
| `CKPT_ROOT` | `/Volumes/PortableSSD/rebot_ckpts_archive` | `EGO_CKPT_ROOT` |
| `LAUNDRY_SYSTEM`, `LAUNDRY_DIT_DATASETS` | `~/laundry_system`, `~/laundry_dit_training/datasets` | `EGO_LAUNDRY_SYSTEM`, `EGO_LAUNDRY_DIT_DATASETS` |
| `CALIBRATION_DIR` | `<repo>/configs/calibration` | `EGO_CALIBRATION_DIR` |

## Interfaces between the stages

| interface | definition | where it is defined |
|---|---|---|
| raw episode | per-stream mp4 + `sensors.mcap` (IMU, gripper, per-frame camera metadata, sync events), `episode_meta.json`, `events.json` | [`docs/collector/README.md`](collector/README.md), [Sample data](results.md#sample-data) |
| Gen-2 action (CART20) | 16 future poses `inv(T(t)) @ T(t + k·UMI_DT)`, k = 1..16, `UMI_DT` = 3/59.94 s (50.05 ms), so 0.80 s; per arm `[xyz, rot6d, g]`, LEFT first, grippers at dims 9 and 19; model action `[16, 32]` = CART20 + 12 exact zeros (AUX12) | [`src/ego_cart20/config.py`](../src/ego_cart20/config.py), [`docs/cart20/README.md`](cart20/README.md) |
| Gen-2 state (RELCART20) | each arm's pose relative to its task start, `[L pose9, R pose9, gL, gR]` (20-D); rows stored at 15 Hz | same |
| rotation | rot6d = first two **rows** of R (UMI `mat_to_rot6d`); quaternions are wxyz | same |
| gripper (ego labels) | continuous aperture, 0 = closed, 1 = open. Gen-2 (CART20): caliper mm / 80 mm (`src/ego_cart20/config.py`); Gen-1 (`src/rebot_prep/grip/`): fraction × 0.0675 m aperture. Source: the cube data always used the HandUMI **Feetech jaw servo encoder** (the servos were off 10-13 → 10-28 and back from 10-29 only in the later garment project) | same |
| gripper (reBot raw) | raw 0 = CLOSED, −270 = OPEN; commands 0..45 with raw = −6·cmd | [`docs/OPERATIONS_NOTES.md`](OPERATIONS_NOTES.md) |
| Gen-1 action | Δq_pseudo chunk `q[t+5+i] − q[t]`, i = 0..29 (LEAD 5, 30 steps), on 65-frame segments | [`archive/gen1_c8old/training/humanik_delta.py`](../archive/gen1_c8old/training/humanik_delta.py) |
| frames | dataset TCP and the `eef_kin` FK TCP differ by R_y(90°); the IK TCP is `gripper_link` + 0.07 m along x | [`docs/OPERATIONS_NOTES.md`](OPERATIONS_NOTES.md) |
| X-VLA domain id | only base slots 10–17 were pretrained; our embodiment uses slot 20 | [`docs/OPERATIONS_NOTES.md`](OPERATIONS_NOTES.md) |
| data contracts (node side) | UMI76 and RELCART20 contracts | [`src/training/umi76/UMI76_CONTRACT.md`](../src/training/umi76/UMI76_CONTRACT.md), [`RELCART20_CONTRACT.md`](../src/training/umi76/RELCART20_CONTRACT.md), [`docs/training/UMI_XVLA_CONTRACT.md`](training/UMI_XVLA_CONTRACT.md) |

## Where each part ran

This is research code from a single lab environment. Most absolute hostnames, user names and home paths were replaced with
placeholders (`${HOME}`, `${REMOTE_HOME}`, `${DATA_ROOT}`, `${SHARED_ROOT}`, `<gpu-node>`, `<ray-head>`); the launchers in
`scripts/` and a few deploy defaults still name private tailnet addresses (`100.64.0.x`) and the node / shared-storage roots
(`ego_paths` defaults). Many scripts assume the original directory layout (`~/c8`, `~/umi_bridge`, `~/ego_collector`); in
`src/` these roots come from `ego_paths` and can be redirected with environment variables (above).
Where each part ran:

- **Collection:** macOS laptop (PySide6, AVFoundation/UVC, `uvc-util`). Entry point `python -m handumi_collector.collect`, run from the repository root.
- **Export, gripper, census, dataset build, offline eval:** CPU / Apple MPS.
- **MASt3R-SLAM:** one RTX 5080 (16 GB), with the upstream repo patched by `build_compat_5090.patch`.
- **Training:** one RTX 4090, submitted as a Ray job, fp32 with TF32 matmuls, batch 4.

Upstream dependencies (not vendored unless noted):

- LeRobot with X-VLA: <https://github.com/huggingface/lerobot>. The base weights are `lerobot/xvla-base` on the Hugging Face Hub.
- MASt3R-SLAM: <https://github.com/rmurai0610/MASt3R-SLAM>. The pinned commit is in `src/mast3r_pose/INSTALLED_COMMIT` and the checkpoint MD5s are in `MD5SUMS`.
- Universal Manipulation Interface (`umi.common.pose_util`): <https://github.com/real-stanford/universal_manipulation_interface> (MIT). A stand-in with the same API is vendored in [`src/umi/`](../src/umi/) ([`upstream/README.md`](../upstream/README.md)).
- Python packages: `mcap`, `numpy`, `scipy`, `opencv-python`, `av`, `pandas`, `pyarrow`, `torch`, `pyyaml`, `PySide6`, `pyserial`, `scservo_sdk` (Feetech).

Not included (these live in other internal packages):

- `handumi.robots` (the `rebot_b601` embodiment and IK solver): imported first by `src/rebot_prep/pseudo_joint/eef_kin.py` (a fallback is in the repository).
- `umi_bridge/track_c/data/r150_q_tcp.npz`, the robot reference joint/TCP data used by `rebot_prep.census.phase3_segments`, and the R150 robot dataset used by the fine-tune stage of `archive/gen1_c8old/training/c8old_chain.sh`.
- The legacy gripper anchor file `gripper_contract.json` read by `rebot_prep.grip.grip_census`. `src/rebot_prep/grip/correct_grip.py` gives the rule it was derived from (aperture 0.0675 m).
- The episode exporter (`src/mast3r_pose/export_episode.py`, was `scripts/umi_export_orbslam.py`) reads calibrations from `ego_paths.CALIBRATION_DIR` (default `configs/calibration/`); the batch scripts in `scripts/pose/` call the copy in the original `~/ego_collector` working tree.
- `archive/gen1_c8old/training/train_bi.py` imports `eef_delta.py` unconditionally. A copy is included, but it stays inactive unless `EEF_DELTA=1`, which the chain never sets.

## Not included

- `src/training/robot/reBot_B601_DM_dualarm.urdf` (vendor robot model) and `src/mast3r_pose/build_compat_5090.patch`
  (a diff against MASt3R-SLAM) are withheld until their licences are confirmed. The FK code in `src/training/robot/`
  expects the URDF at that path.
- Part II (v4) processing code for HRA_A100 (origin-plane scale, origin-yaw matching, G-frame calibration, the table-aware
  conversion and the robot-camera re-rendering) lives in internal tooling and is not included; its outputs and per-take
  reports are in `results/v4/hra/`.

