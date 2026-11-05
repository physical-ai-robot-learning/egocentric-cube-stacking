# archive

Superseded, exploratory and one-off code, kept **verbatim** (moved with `git mv`, file contents unchanged) so that every
earlier result stays traceable. Nothing in `src/` imports from here, nothing here is packaged or tested, and the code is
not expected to run from this folder: most files are run bundles for particular machines (`~/c8`, `~/umi_bridge`, GPU nodes)
with their own path bootstraps. Git history has the original locations; the full old → new map is in
[`docs/refactor_2026-11.md`](../docs/refactor_2026-11.md).

**Status:** archived (2026-11-05 refactor; `legacy/` of the 2026-11-01 layout was renamed to this folder)

## Inputs → Outputs / Entry points

None from this folder. Each sub-folder documents the commands it was run with in its file headers.

## Index

| folder | what it is | why it is here | was at |
|---|---|---|---|
| [`gen1_c8old/`](gen1_c8old/) | **Gen-1 run bundles behind the Part I joint-space numbers**: X-VLA trainer patches (`training/train_bi.py`, `humanik_delta.py` = the Δq_pseudo action target, `eef_delta.py`, …), the C-old 300k → R150 chain and the R30–R120 ladder (`training/c8old_chain.sh`, read by `analysis/verify_claims.py`), the offline evaluator (`eval/c8old_mac_eval.py`), the C-old v1 and v2 dataset builders / validators (`dataset_v1/`, `dataset_v2/`) | Gen-1 is superseded by the Gen-2 Cartesian path (`src/ego_cart20/`, `src/training/`); its results are frozen in `results/` | `scripts/training/c8old/`, `scripts/rebot/eval/c8old/`, `scripts/rebot/dataset_v2/`, `legacy/c8old_dataset_v1/` |
| [`hra_single_arm/`](hra_single_arm/) | Part II single-arm approach sub-study (HRA_red, HRA_A100): pipeline and calibration runs, node training wrappers and launchers, its loaders, the right-only ego_cart20 variant and its test, the direction probe | separate sub-study, not on the bimanual main path; documented in [`docs/hra/`](../docs/hra/README.md); outputs in `results/v3/hra_red/`, `results/v4/hra/` | `scripts/rebot/hra/`, `scripts/training/hra_node/`, `src/ego_cart20/right_only.py`, … |
| [`rel16_audit/`](rel16_audit/) | Gen-2 checkpoint audits, gate watchers, prompt-swap / drift / latency diagnostics, real-robot log (`REAL_ROBOT_LOG.md`), retention scripts | one-off diagnostics; their conclusions are in `results/v3/` and the report | `scripts/rebot/eval/rel16_audit/` |
| [`ego_cart20_experiments/`](ego_cart20_experiments/) | ego→reBot robotization, UMI tag/IMU/SLAM fusion smoother, ego-vs-R312c comparison and kinematic compatibility diagnostics (produced `results/v3/ego_vs_robot/`), visual jaw / trim sources, gcal1 parity, and their tests | exploratory branches of the Gen-2 conversion | `src/ego_cart20/{robotize,umi_fusion}/`, `src/ego_cart20/scripts/…`, `tests/cart20/…` |
| [`training_variants/`](training_variants/) | superseded UMI76 derivations and gates (v3L / v3d / CART20, R280 / R380 / R246c), REL16 aux loss (replaced by REL-only), base widening, frozen node environment lists | replaced by `src/training/relonly/` and `derive_v2` / `derive_relcart20` | `src/training/umi76/` |
| [`learned_ik/`](learned_ik/) | optional learned IK backend (LearnedIK v0) | never on the deployment path (Pink IK is) | `src/rebot_deploy/learned_ik/` |
| [`softfold/`](softfold/) | soft-fold / h-fold (laundry) source adapters, exporters and launchers left over from the follow-up project | belongs to [egocentric-garment-manipulation](https://github.com/physical-ai-robot-learning/egocentric-garment-manipulation) | `src/ego_cart20/`, `scripts/training/ego_cart20/` |
| [`ego_teleop/`](ego_teleop/) | wearable VIO teleoperation research (fisheye + IMU → relative SE(3), head RGB-D hands → Aero hand), its tests, docs and configs | experimental, never deployed | `src/ego_teleop/`, `tests/experimental/teleop/`, `docs/experimental/`, `configs/ego_teleop/` |
| [`shared_unused/`](shared_unused/) | modules of the shared capture packages (`handumi_collector`, `ego_collector`) that the main path does not import: the V0 AprilTag-glove collector (`ego_collector` except `camera/`), RGB-D / depth / cube-anchor / cross-view pose tooling, V-pilot analysis tools; their tests and the V0 configs | `src/` keeps only what the collector, its calibration / QA tools and the pose export import (import-graph closure) | `src/handumi_collector/`, `src/ego_collector/`, `tests/collector/`, `tests/experimental/v0/`, `configs/` |
| [`collector_tools/`](collector_tools/) | flat command-line tools of the collection side: ChArUco / table-frame calibration helpers, HandUMI RGB-D body-tracking bake-offs and audits, camera / IMU / Orbbec device tests, robot-like protocol checks, a second copy of the episode exporter | one-off tools; the exporter used by the pipeline is now `src/mast3r_pose/export_episode.py` | `scripts/*.py`, `scripts/*.sh`, `scripts/rebot/c8/umi_export_orbslam.py`, `configs/charuco*.yaml` |
| [`deploy_loaders_old/`](deploy_loaders_old/) | earlier per-run checkpoint → UI loaders | replaced by `scripts/deploy/load_*_ckpt_to_ui.sh` | `legacy/loaders/` |

Counts (2026-11-05): 642 files, 422 of them Python.

## Provenance

`legacy/README.md` (2026-11-01) listed `c8old_dataset_v1/` and `loaders/`; both are here (`gen1_c8old/dataset_v1/`,
`deploy_loaders_old/`). Two test files kept in `tests/collector/` skip one check each when the archived module is absent
(`test_pose_core.py::test_se3_matches_legacy_conventions`, `test_fisheye_coverage.py::test_cross_view_…`).
