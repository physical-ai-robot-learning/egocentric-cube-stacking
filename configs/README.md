# configs

All configuration and calibration files, at the repository root so that code which resolves `REPO_ROOT / "configs/..."`
(`handumi_collector.config.REPO_ROOT`) finds them.

**Status:** current

## Inputs → Outputs

Read by the collector, the pose export and the dataset code; written by the calibration tools (new versions only,
latest `vNNN` wins, files are never overwritten).

## Entry points

None (data only). The calibration tools that write here are listed in `calibration/README.md` and `docs/collector/README.md`.

## Layout

- `calibration/` — versioned calibrations (fisheye, camera–IMU, camera–TCP, gripper, head mount, stereo track cams); see `calibration/README.md`
- `envs/requirements-test.txt` — exact CI test environment (Python 3.12, Apple Silicon; the package goldens were recorded there)
- `dataset_calibration/` — the subset of calibrations that the reported dataset references (was top-level `calibration/`; every file is byte-identical to its counterpart in `calibration/`, where the Kalibr files are under `kalibr_raw/` and the gripper files are not in a subfolder). `analysis/verify_claims.py` reads this copy
- `handumi/` — collector hardware profiles, tasks, protocols, pose configs
- `cart20/` — ego REL-only D6 pretrain config (`ego_pretrain_relonly_d6.yaml`; `.remote_home.yaml` keeps the `${REMOTE_HOME}` placeholder version)

## Provenance

2026-11-05: the teleop configs (`ego_teleop/`) moved to `archive/ego_teleop/configs/`, the V0 collector configs
(`camera/`, `world_tag_map.yaml`, `wrist_extrinsics.yaml`, `qa*.yaml`, `aperture_calibration.yaml`, `collection_plan.yaml`) to
`archive/shared_unused/configs/`, the ChArUco board specs to `archive/collector_tools/configs/`.


Merged from `collection/configs`, `experimental/configs` (whose `calibration` and `handumi` were symlinks to the former), `cart20/` and `calibration/`.
See [`docs/provenance/README.md`](../docs/provenance/README.md).
