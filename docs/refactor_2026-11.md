# Refactor 2026-11: org layout, archive split and OA code style

Branch `refactor/oa-layout` (base `1e16371`). Goal: a reviewer can find the code of every pipeline stage in a few minutes,
and the main-path code reads clean, **without any change of behaviour or of any reported number**. Standard followed: the
org-wide repository standard (layout, OA code style, README skeleton); style reference: `egocentric-laundry-folding`
`src/laundry_*` and its log `docs/laundry/structure_refactor/2026-11-04_oa_layout.md`.

## 1. What changed, in order

1. **Layout** (commit `aadcce9`, `git mv` only): `legacy/` → `archive/`; superseded / exploratory code → `archive/`
   (index: [`archive/README.md`](../archive/README.md)); `scripts/` reduced to stage-organised launchers; `sample_data/` →
   `assets/sample_episodes/`; the episode exporter `scripts/umi_export_orbslam.py` → `src/mast3r_pose/export_episode.py`.
   The shared capture packages were trimmed to the import closure of the main path (static import graph over `src/`,
   absolute + relative + flat-sibling imports, roots = the collector entry points, its calibration / monitor / QA tools and
   the two modules the exporter uses), plus the four modules the kept tests use as fixtures.
2. **pyproject** (`1c443d2`): packages and extras for the new layout; `archive/` is not collected by pytest.
3. **Characterization tests first, then OA rewrite**, per package on its own branch (merged `03656e6`, `21d90e6`, `472f859`,
   `5d9deae`): tests recorded on the old code were committed before each rewrite and pass unchanged after it (only call sites
   renamed).
4. Repository wiring: loaders (`scripts/deploy/`) start the UI as `python -m rebot_deploy.run_smoke_ui`; CI; import smoke
   known-failures list emptied; README restructured (long sections → [`results.md`](results.md), interfaces → [`code_structure.md`](code_structure.md));
   per-folder READMEs; technical report v5.2 (code map only) with the partial v4 documents moved to `report/archive/`.

## 2. Old → new paths

### Top level

| old | new |
|---|---|
| `legacy/c8old_dataset_v1/` | `archive/gen1_c8old/dataset_v1/` |
| `legacy/loaders/` | `archive/deploy_loaders_old/` |
| `legacy/README.md` | content merged into `archive/README.md` |
| `sample_data/` | `assets/sample_episodes/` |
| `assets/aero_hand_open/`, `assets/handumi/` | `archive/ego_teleop/assets/aero_hand_open/`, `archive/shared_unused/assets/handumi/` (only archived code used them) |
| `configs/ego_teleop/` | `archive/ego_teleop/configs/` |
| `configs/{camera/,world_tag_map,wrist_extrinsics,qa,qa_hawor,aperture_calibration,collection_plan}.yaml` | `archive/shared_unused/configs/` |
| `configs/charuco*.yaml` | `archive/collector_tools/configs/` |
| `report/{methods,part1_stacking,part2_approach}_{en,ko}.{tex,pdf}` | `report/archive/` (figure paths `figures/` → `../figures/`) |
| `docs/experimental/` | `archive/ego_teleop/docs/` |
| — | `docs/results.md` (README detail), `docs/refactor_2026-11.md` (this file), `report/README.md`, `src/ego_paths/README.md`, `src/umi/README.md` |

### scripts/

| old | new |
|---|---|
| `scripts/rebot/c8/c8_export*.sh`, `scripts/rebot/mast3r/*.sh` | `scripts/pose/` |
| `src/rebot_prep/retarget/chains/*.sh` | `scripts/dataset/gen1_retarget/` |
| `scripts/training/{node4090,node5090,ego_cart20,mac_chains,archivers}/` | `scripts/train/<same>/` (HRA and soft-fold launchers → archive) |
| `src/training/relonly/train_relonly_5090.sh` | `scripts/train/node5090/train_relonly_5090.sh` |
| `src/training/umi76/build_r312c_umi76.sh` | removed: byte-identical duplicate of `scripts/train/node4090/build_r312c_umi76.sh` |
| `scripts/rebot/loaders/load_relonly{,_5090}_run_ckpt_to_ui.sh`, `auto_ui_d20_e5_v5.sh` | `scripts/deploy/` (default `REPO` = this checkout; UI started with `-m`) |
| `scripts/training/c8old/` | `archive/gen1_c8old/training/` |
| `scripts/rebot/eval/c8old/` | `archive/gen1_c8old/eval/` |
| `scripts/rebot/dataset_v2/` | `archive/gen1_c8old/dataset_v2/` |
| `scripts/rebot/eval/rel16_audit/` | `archive/rel16_audit/` |
| `scripts/rebot/hra/`, `scripts/training/hra_node/`, HRA launchers and loaders | `archive/hra_single_arm/{pipeline,training_wrappers,launchers,loaders}/` |
| `scripts/*.py`, `scripts/*.sh` (flat tools), `scripts/rebot/c8/umi_export_orbslam.py` | `archive/collector_tools/` (the copy → `umi_export_orbslam_c8copy.py`) |
| `scripts/umi_export_orbslam.py` | `src/mast3r_pose/export_episode.py` |

### src/ — modules renamed by the OA rewrite (old names are kept in each module docstring as "was …")

| package | old → new |
|---|---|
| `mast3r_pose` | `cp6_scale.py` → `imu_metric_scale.py` |
| `rebot_prep.grip` | `c8_grip.py` → `grip_census.py` |
| `rebot_prep.census` | `c8_census_pre.py` → `sensor_sync_census.py`; `c8_phase3.py` → `phase3_segments.py`; `c8_regress_ref208.py` → `mast3r_regression.py`; `c8_imu_vi_g.py` → `imu_vi_gravity.py`; `c8_g1_gravity_sign.py` → `gravity_sign_gate.py`; `c8_collision_v1.py` → `arm_clearance.py` |
| `rebot_prep.pseudo_joint` | `c2_input_gate.py` → `input_gate.py` |
| `rebot_prep.retarget` | `v2k_seedbank.py` → `seed_bank.py`; `v2k_retarget.py` → `multi_seed_retarget.py`; `v2k_compare.py` → `compare_retargets.py`; `v2k_tail.py` → `rescued_tail.py` |
| `ego_cart20` | shallow tree: `geometry/{rotation6d,transforms,canonical_frame}.py` → top level; `io/raw_episode_loader.py` → `raw_episode.py`; `io/processed_writer.py` → `processed_episode.py`; `preprocessing/{resample,synchronize}.py` → `resample.py`; `preprocessing/{gripper,pose_filter}.py` → top level; `labels/*.py` → `labels.py`; `dataset/splits.py` → `splits.py`; `dataset/{xvla_ego_dataset,sampler}.py` → `dataset.py`; `validation/{integrity,geometry_checks}.py` → `integrity.py`; `validation/visualize_trajectory.py` → `trajectory_plot.py`; `sources/handumi_export.py` → `handumi_export.py`; `scripts/*` same names |
| `rebot_deploy` | `infer_core_v4.py` → `xvla_inference.py` (`V4Inferencer` → `XvlaInferencer`); `v4_contract.py` → `robot_dataset_contract.py`; `mac_v4_smoke_ui.py` → `run_smoke_ui.py` |
| `training` | no file renamed: the node launchers copy and run these files by name (and check sha256 pins) |

Public functions became camelCase (e.g. `ego_cart20.convert.convert_episode` → `convertEpisode`,
`rebot_deploy` `relcart20_state` → `relCart20State`); the full symbol maps are in the commit messages of the four rewrite commits
(`7a8a69d`, `b526f1d`, `0211576`…`a414f05`, `5135adf`). Names imposed from outside stay: CLI flags and defaults, env switches
(`V4_*`, `MIT_*`, `REL16_*`, `M3_*`, `EGO_*`, …), file formats and keys, LeRobot / torch / Flask hooks, UI routes and JSON fields,
`umi.common.pose_util` API names, and the public names of `pseudo_joint_pipeline` / `state32` / `eef_kin` (`rebot_prep`), which
have twin copies on the original machines that are imported first.

## 3. What was archived

642 files (422 Python) in `archive/`, by folder: `gen1_c8old` 50, `hra_single_arm` 160, `rel16_audit` 57,
`ego_cart20_experiments` 21, `training_variants` 28, `learned_ik` 9, `softfold` 14, `ego_teleop` 124 (+ assets),
`shared_unused` 114 (+ configs, assets), `collector_tools` 41 (+ configs), `deploy_loaders_old` 14. Reasons per folder:
[`archive/README.md`](../archive/README.md). Python files under `src/`: 232 (178 code + 54 test files); before: 404 under `src/` + 146 in `scripts/` (now 2 node-side selection scripts in `scripts/train/node5090/`).

## 4. Restyle status

| package | status |
|---|---|
| `ego_cart20` | OA style, behaviour-checked (57 characterization tests + 15 contract tests); **not behaviour-checked**: the LeRobot parts of `scripts/export_lerobot.py`, `handumi_export.exportOne` (external C8 pose stack + data), H.264 bytes of `encodeRowVideo` |
| `mast3r_pose` | OA style, checked; **not behaviour-checked**: `run_perframe.py`, `run_perframe_resident.py` (MASt3R-SLAM, lietorch, GPU) |
| `rebot_prep` | OA style, checked (51 test files incl. old-vs-new end-to-end runs on synthetic trees); **not behaviour-checked**: `multi_seed_retarget` orchestration (`main` / `segments` / `buildContext`, needs an external module), `seed_bank` at full Sobol size |
| `training` | OA style, checked with stand-ins for LeRobot / zarr / bitsandbytes; `relonly/rel16_aux_relonly.py` **kept byte-identical** (sha256 pinned by 11 node launchers); **restyled mechanically, not behaviour-checked**: `umi76/gate_r312c.py`, `umi76/preflight_cartonly.py` (LeRobot + node data) |
| `rebot_deploy` | `robot_dataset_contract`, `eef_kin`, `pink_ik`, `robot_service/usb_arm_ports` OA style, checked; `xvla_inference` module level checked, but `XvlaInferencer.__init__` / `infer` / `infer_raw` / `_localForward` / RTC patch (need LeRobot) and the non-numerical IK paths of `solve_waypoint` are mechanical only; `PinkIK.solve` not checked (no QP solver in the test env) |
| `rebot_deploy.smoke_ui/` (22 modules + `page.html`), `rebot_deploy/robot_service/robot_service_mit.py` | **moved, not yet restyled** (call sites only); covered by the UI golden test |
| `handumi_collector`, `ego_collector` | **moved, not yet restyled**: shared copies (handumi-common v0.3.0), files unchanged, trimmed |
| `umi`, `ego_paths` | OA style internals; public API unchanged |

## 5. Verification

| check | before (base `1e16371`) | after |
|---|---|---|
| `python analysis/verify_claims.py` | 736 claims, 0 failed | 736 claims, 0 failed (file list now `README.md` + `docs/results.md` + `docs/history.md`; report paths `report/archive/…`; Gen-1 chain read from `archive/gen1_c8old/training/c8old_chain.sh`) |
| `pytest src tests/cart20` | tests/cart20: 40 passed | 207 passed (15 cart20 + 192 package tests; the 25 archived cart20 tests moved with their code) |
| `pytest tests/test_ui_routes.py` (golden UI record) | 2 passed | 2 passed |
| `pytest tests/test_imports.py` | known failures: 48 listed | 232 modules: 230 ok, 2 skip (lietorch), 0 fail; list emptied |
| `pytest tests/collector` | 224 passed, 12 failed, 1 skipped, 3 errors | 157 passed, 12 failed, 3 skipped, 3 errors (same 12 + 3; tests of archived modules moved to `archive/shared_unused/tests/`) |
| `python -m compileall -q .` | clean | clean |
| report | 15 pp | `report_en.pdf` 15 pp, `report_ko.pdf` 15 pp (tectonic); every `\code{}` path exists |
| fresh clone, `pip install -e ".[collector,dev]"` + CI steps | — | see the coordinator report of this branch |

## 6. Not checkable here

Hardware (collector devices, robot service), GPU (MASt3R-SLAM), LeRobot (training entries, export to LeRobot, policy
inference) and the original machines' data roots. These paths were restyled mechanically or left unchanged, as listed in §4.

## 7. Behaviour changes made on purpose (all in `rebot_prep`)

1. `pseudo_joint.input_gate`: paths that were relative to the module folder now use `~/umi_bridge/track_c` (`data/`,
   `reports/c2`, the `--produce` default) — the production folder the module ran from.
2. `retarget.multi_seed_retarget` and `retarget.seed_bank` read the frozen seed banks / episode lists from the package's
   `retarget/contracts/` (the old code read them next to the module, where the repository does not hold them); `seed_bank`
   writes to `~/umi_bridge/track_c/v2k`.
3. `census.phase3_segments.init` and `retarget.seed_bank` fall back to the repository's `pseudo_joint_pipeline` / `state32` /
   `arm_clearance` when the external twin is missing (before: ImportError).
4. Import-time `sys.argv` / data reads moved under `main()` everywhere; missing positional arguments now give an argparse usage
   error (exit 2) instead of an IndexError; `assert`s in library code became `XxxError` exceptions with the same messages.

## 8. Suspected bugs found (recorded, not fixed)

- `ego_cart20.resample.PoseTrack.__init__` modifies a caller's bool `valid` array in place (`&=`); with the jump filter off this mutates the raw episode.
- `ego_cart20` `metadata["code_sha256"]` hashes every `*.py` under the package, now including `tests/`, and its keys follow the new file names (the hash of a new conversion differs from old manifests).
- `ego_cart20.dataset.XVLAEgoDataset(loadImages=True)` crashes for a camera with frame timestamps but no video path.
- `rebot_deploy.pink_ik.PinkIK`: duplicate `*_tcp` frames if the URDF already has them ("Several frames match"); `solve(iters=0)` raises NameError.
- `rebot_deploy.xvla_inference.solve_waypoint` logs `exec_k` from `os.environ["V4_EXEC_K"]`, not the module's `EXEC_K`; `smoke_ui/plan_exec.py` hard-codes `/-6.0` and `0..45` instead of `GRIP_OBS_PER_CMD` / `GRIP_CMD_MAX`.
- `training.umi76.umi76_to_lerobot`: `--action-mode` / `--state-mode` / `--gripper-target` only change what `v4_provenance.json` records (the data is always the 16-step chunk and 76-D state).
- `training`: the repository copies of `train_rel16_relonly.py`, the 8-bit wrapper, `preflight_cartonly.py` and `rel16_aux_masked.py` no longer match the sha256 pins in the node launchers since the 2026-11-01 `ego_paths` edit (`aae3282`); the byte-exact entry is recoverable from `89349a5`. `rel16_aux_relonly.py` still matches.
- `training.umi76.derive_v2` without `--variant` writes labels as variant A with `variant=None` provenance; `gate_r312c` exits 0 even when checks fail (the chain greps the PASS line).
- `rebot_prep`: `correct_grip` docstring says "closed = 90th percentile" but the code uses p10 or p90 by jaw direction; `gravity_sign_gate` crashes with no human tilt samples; `input_gate` IndexError with no segments; `hybrid_merge` KeyError / TypeError on missing Kr data; `seed_bank --ref r30` fails below 6000 frames; `grip_census` hard-codes the device span (1136.6 / 1248.2).
- Docs: the v5.1 report and `docs/provenance/` describe the 2026-11-01 layout; the report is updated (v5.2), provenance records are kept as history.

## 9. Things to know

- `scripts/` launchers are launch records of the original machines: they copy code from working trees under `~/c8`,
  `~/ego_collector`, `~/umi_bridge` and node homes and still name private tailnet addresses (`100.64.0.x`); `ego_paths` defaults
  and two frozen selection JSONs name the node home and the shared NFS root.
- The production robot UI does not run from this repository; no hardware code was run during the refactor.
