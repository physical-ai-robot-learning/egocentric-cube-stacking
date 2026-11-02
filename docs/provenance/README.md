# Provenance and the 2026-11-01 restructure

The MANIFEST files in this folder are the original per-folder provenance records (which working directory, commit and
date each file came from). They are kept verbatim, so the paths **inside** them use the layout before 2026-11-01.
This table maps those paths to the current ones.

| MANIFEST | was |
|---|---|
| [`collection_MANIFEST.md`](collection_MANIFEST.md) | `collection/MANIFEST.md` |
| [`experimental_MANIFEST.md`](experimental_MANIFEST.md) | `experimental/MANIFEST.md` |
| [`pipeline_MANIFEST.md`](pipeline_MANIFEST.md) | `pipeline/MANIFEST.md` (also covers `cart20/` and `calibration/`) |
| `umi_laundry_MANIFEST.md`, `umi_robot_parity_MANIFEST.md` | `pipeline/umi_laundry/MANIFEST.md`, `pipeline/umi_robot_parity/MANIFEST.md` — moved out with the laundry material (see below) |
| [`deploy_MANIFEST.md`](deploy_MANIFEST.md) | `deploy/MANIFEST.md` |
| [`training_MANIFEST.md`](training_MANIFEST.md) | `training/MANIFEST.md` |
| [`evaluation_MANIFEST.md`](evaluation_MANIFEST.md) | `evaluation/MANIFEST.md` |
| [`hra_MANIFEST.md`](hra_MANIFEST.md) | `hra/MANIFEST.md` |

`hardware/MANIFEST.md` and `upstream/MANIFEST.md` did not move.

## Old → new paths

| old | new |
|---|---|
| `collection/handumi_collector/` | `src/handumi_collector/` |
| `collection/configs/{calibration,handumi}/`, `collection/configs/charuco*.yaml` | `configs/{calibration,handumi}/`, `configs/charuco*.yaml` |
| `collection/assets/`, `collection/firmware/` | `assets/`, `firmware/` |
| `collection/scripts/*`, `collection/tools/*` | `scripts/*` (flat) |
| `collection/tests/` | `tests/collector/` |
| `collection/docs/*`, `collection/README.md` | `docs/collector/` |
| `collection/pyproject.toml`, `experimental/pyproject.toml` | merged into the root `pyproject.toml` |
| `experimental/{ego_collector,ego_teleop,visual_teleop}/` | `src/{ego_collector,ego_teleop,visual_teleop}/` |
| `experimental/configs/*` | `configs/*` (the `calibration`/`handumi` symlinks and the duplicate `charuco*.yaml` were dropped: they pointed to / equalled the collection copies) |
| `experimental/assets/aero_hand_open/` | `assets/aero_hand_open/` |
| `experimental/scripts/*` (incl. `umi_teleop_eval/`) | `scripts/*` (`provenance.py` was byte-identical to the collection copy; one kept) |
| `experimental/tests/{teleop,v0,vteleop}` | `tests/experimental/{teleop,v0,vteleop}` |
| `experimental/docs/ego_teleop/`, `experimental/README.md` | `docs/experimental/` |
| `cart20/ego_cart20/` | `src/ego_cart20/` |
| `cart20/tests/`, `cart20/docs/`, `cart20/README.md` | `tests/cart20/`, `docs/cart20/` |
| `cart20/configs/ego_pretrain_relonly_d6.yaml`, `cart20/ego_pretrain_relonly_d6.yaml` | `configs/cart20/ego_pretrain_relonly_d6.yaml`, `configs/cart20/ego_pretrain_relonly_d6.remote_home.yaml` |
| `calibration/` | `configs/dataset_calibration/` |
| `pipeline/mast3r_pose/` (`*.py`, `INSTALLED_COMMIT`, `MD5SUMS`) | `src/mast3r_pose/` |
| `pipeline/mast3r_pose/*.sh` | `scripts/rebot/mast3r/` |
| `pipeline/{census,grip,pseudo_joint}/` | `src/rebot_prep/{census,grip,pseudo_joint}/` |
| `pipeline/retarget_v2/` (incl. `contracts/`, `chains/`) | `src/rebot_prep/retarget/` |
| `pipeline/dataset/`, `pipeline/dataset_v2/` | `legacy/c8old_dataset_v1/` (was `scripts/rebot/dataset/` until the 2026-11-01 second pass), `scripts/rebot/dataset_v2/` |
| `pipeline/export/` | `scripts/rebot/c8/` |
| `pipeline/umi_laundry/` | `scripts/rebot/laundry/` (README → `docs/laundry/`); moved to garment on 2026-11-01 (below) |
| `pipeline/umi_robot_parity/` | `scripts/rebot/parity/` (README → `docs/parity/`); moved to garment on 2026-11-01 (below) |
| `deploy/*.py`, `deploy/learned_ik/`, `deploy/robot_service/` | `src/rebot_deploy/` |
| `deploy/loaders/` (incl. `legacy/`), `deploy/swappers/*` | `scripts/rebot/loaders/` (swappers call `./load_*.sh`, so they share the folder) |
| `deploy/README.md`, `deploy/TODO_gripper_hold.md` | `docs/deploy/` |
| `training/{relonly,umi76,robot}/` | `src/training/{relonly,umi76,robot}/` |
| `training/launchers/<sub>/` | `scripts/training/<sub>/` |
| `training/archivers/` | `scripts/training/archivers/` |
| `training/{train_bi,humanik_delta,eef_delta,lr_groups,aug_defaults,keep_indices,contrastive_sampler}.py`, `training/c8old_{chain,launch}.sh` | `scripts/training/c8old/` |
| `training/docs/*`, `training/README.md` | `docs/training/` |
| `evaluation/rel16_audit/` | `scripts/rebot/eval/rel16_audit/` |
| `evaluation/c8old_*.py` | `scripts/rebot/eval/c8old/` |
| `hra/hra_red/`, `hra/hra_a100/` | `scripts/rebot/hra/hra_red/`, `scripts/rebot/hra/hra_a100/` |
| `hra/README.md` | `docs/hra/README.md` |
| `*/MANIFEST.md` (9 files) | `docs/provenance/` (this folder) |
| `analysis/`, `report/`, `results/`, `sample_data/`, `hardware/`, `upstream/`, `docs/*.md` | unchanged |
| `docs/media/` | `media/` (2026-11-01, second pass: same top-level `media/` as egocentric-garment-manipulation; the v5 report PDFs/LaTeX still print the old path) |

Moves were done with `git mv`, so `git log --follow <new path>` shows each file's history.

## Superseded items (kept, not deleted)

| item | what it was | replaced by |
|---|---|---|
| `ego_collector` (`src/ego_collector/`) | V0 AprilTag-glove egocentric collector | `handumi_collector` (the `ego_collector.camera` layer is still imported) |
| `src/rebot_prep/pseudo_joint/`, `src/rebot_prep/census/c8_phase3.py` | Gen-1 joint-space (pseudo-joint) retargeting | Cartesian CART20 conversion in `src/ego_cart20/` (report v3) |
| `src/rebot_prep/retarget/` v2-K / v2-Kr | multi-seed kinematic retargeting variants | v2-TR (primary) and the frozen hybrid master, same folder |
| `legacy/c8old_dataset_v1/` (was `scripts/rebot/dataset/`) | C-old v1 dataset builder | `scripts/rebot/dataset_v2/` (C-old v2, final, frozen) |
| `scripts/training/c8old/`, `scripts/rebot/eval/c8old/` | Gen-1 trainer (`train_bi.py` + `humanik_delta.py`), C-old chain and offline evaluator | Gen-2 REL-only training in `src/training/relonly/` |
| `legacy/loaders/` (was `scripts/rebot/loaders/legacy/`) | earlier per-run checkpoint loaders | `scripts/rebot/loaders/load_*_ckpt_to_ui.sh` |
| `ego_teleop` (`src/ego_teleop/`) | wearable teleoperation research | not replaced; experimental, not deployed |

## Laundry / garment material moved out (2026-11-01)

Each project now lives in one repository. The laundry (T-shirt flattening) material that had been kept here was removed
and is owned by [egocentric-garment-manipulation](https://github.com/physical-ai-robot-learning/egocentric-garment-manipulation), which holds the same files or later versions of them:

| removed from this repository | what it was |
|---|---|
| `scripts/rebot/laundry/` (+ `docs/laundry/`) | UMIalign pose + visual-gripper pipeline and its run drivers |
| `scripts/rebot/parity/` (+ `docs/parity/`) | UMI TCP tag-rig calibration, Kalibr bundle, UMI → reBot replay parity |
| `src/visual_teleop/`, `tests/experimental/vteleop/` | HandUMI-as-teleop package (older copy) and its tests |
| `scripts/umi_teleop_eval/` | wrist tracking / VIO / keypoint evaluations for `visual_teleop` |
| `docs/UPDATE_2026-10-20.md`, `docs/UPDATE_2026-10-20_ko.md`, README section "Update 2026-10-20" | the post-v5 laundry update note |
| `umi_laundry_MANIFEST.md`, `umi_robot_parity_MANIFEST.md` | provenance records of the two folders above |

The MANIFEST files kept in this folder still mention these paths; they are verbatim records.

## Shared packages (2026-11-01)

`handumi_collector`, `ego_collector`, `ego_teleop` and `umi` are also used by
[egocentric-garment-manipulation](https://github.com/physical-ai-robot-learning/egocentric-garment-manipulation), which until
2026-11-01 held a partial copy (34 files). Both repositories now hold the same full packages under `src/`: identical
copies of every Python file (236 files; md5 list in [`shared_packages_MANIFEST.md`](shared_packages_MANIFEST.md)). Only
the three package `README.md` files differ between the two repositories. Import names are unchanged.

**2026-11-02: copied back from the shared package (v0.3.0) because that repository is private.** On 2026-11-01 the four
packages, their tests and `scripts/mast3r_live_server.py` had been moved out of this repository into a separate shared
package that both repositories installed from git. That repository is private, so a public clone could no longer
install this one. The packages were copied back from its release v0.3.0, byte-identical (the package READMEs restored
to this repository's own text), and the git dependency was removed from `pyproject.toml`:

| copied back | files | notes |
|---|---|---|
| `src/handumi_collector/`, `src/ego_collector/`, `src/ego_teleop/`, `src/umi/` | 239 (236 `.py`, 2,009,728 bytes; 3 `README.md`) | md5 of every `.py` in [`shared_packages_MANIFEST.md`](shared_packages_MANIFEST.md) |
| `tests/collector/`, `tests/experimental/teleop/`, `tests/experimental/v0/` | 75 | identical to the files removed on 2026-11-01 |
| `scripts/mast3r_live_server.py` | 1 | loaded by `tests/experimental/teleop/test_mast3r_live.py`; identical to the file removed on 2026-11-01 |

`configs/`, `assets/`, `firmware/`, `upstream/` and the guides never left. `pyproject.toml` has again the extras
`collector`, `experimental`, `apriltag`, `viz`, `aero`, `dev` with their own dependency lists and the `ego-collector`
console script; `.github/workflows/ci.yml` again runs `tests/experimental/v0` and `tests/experimental/teleop` (without
`test_aero_backends.py`), as before 2026-11-01.

**What differs from the copy removed on 2026-11-01.** The shared package started from this repository's packages and
took the newer revision of some files from the garment copy (taken later from the same `~/ego_collector` working tree):

| file | change against this repository's 2026-11-01 copy | effect |
|---|---|---|
| `ego_teleop/ego_cart20_link.py`, `robot/bimanual.py`, `robot/deadman.py`, `robot/feasibility.py`, `transforms/umi_robot_frame.py`, `recorder/umi_contract_export.py` | new (were only in the garment copy) | new modules; nothing in this repository imports them |
| `ego_collector/camera/orbbec_capture.py` | newer revision | turns the Orbbec colour auto-exposure priority off at start-up (best effort, hardware only) |
| `ego_teleop/robot/rebot_client.py` | newer revision | follower/leader frame conversion, jaw command helpers (`gripper_cmd_from_g`, ...), IK seeded with the previous command. The default joint limits are the MIT bridge's clip range (`BRIDGE_SEEED_LIM_DEG`; the old values stay as `C30_LOWER_RAD` / `C30_UPPER_RAD`), used by the live-robot tools `hud_robot` and `m0_eef_jog` when no limits are given |
| `ego_teleop/robot/safety.py` | newer revision | adds a pure `propose()` and a transactional `commit()`; `step()` gives the same result as before |
| `ego_teleop/transforms/frames.py` | newer revision | docstring only |
| `handumi_collector/devices/uvc_controls.py`, `tools/make_tracking_mesh.py`, `ego_teleop/tracking/backends/openvins.py` | `Path.home()` / `sys.path.insert(0, d)` instead of `ego_paths.HOME` / `ego_paths.prepend_sys_path(d)` | same behaviour; the shared copies do not depend on `ego_paths` |

No reported number depends on these files (`analysis/verify_claims.py`: 736 claims, 0 failed, before and after).

Tests in a fresh Python 3.12 environment (`pip install -e ".[collector,experimental,dev]" rosbags`, the CI install):
`tests/cart20` 40 passed; `tests/experimental/v0` 57 passed, 2 skipped; `tests/experimental/teleop` without
`test_aero_backends.py` 276 passed, 6 skipped; `tests/collector` 224 passed, 12 failed, 1 skipped, 3 errors (the known
mock-gripper / camera-listing / m15 UI failures, as before 2026-11-01; not run in CI); `tests/test_imports.py` passed
(`handumi_collector.tools.metric_qa` is back on the known-failures list).
