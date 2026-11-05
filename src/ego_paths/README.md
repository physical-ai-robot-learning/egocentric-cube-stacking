# ego_paths

The one module that names machine-specific data roots (`~/c8`, `~/umi_bridge`, the GPU node home, the shared NFS root, the
checkpoint SSD, the calibration folder), each overridable by an environment variable.

**Status:** current

## Inputs → Outputs

In: environment variables `EGO_DATA_ROOT`, `EGO_UMI_BRIDGE`, `EGO_COLLECTOR_HOME`, `EGO_HOLOBRAIN_MAC`, `EGO_HOLOBRAIN_DATA`,
`EGO_NODE_HOME`, `EGO_SHARED_ROOT`, `EGO_CKPT_ROOT`, `EGO_LAUNDRY_SYSTEM`, `EGO_LAUNDRY_DIT_DATASETS`, `EGO_CALIBRATION_DIR`.
Out: `pathlib.Path` constants; `prepend_sys_path(...)` for external checkouts (the only sanctioned `sys.path` edit in `src/`).

## Entry points

Library only: `from ego_paths import DATA_ROOT, UMI_BRIDGE, CALIBRATION_DIR`.

## Layout

- `__init__.py` — the constants and `prepend_sys_path`; table in [`docs/code_structure.md`](../../docs/code_structure.md#machine-specific-roots-ego_paths)
- `tests/` — defaults and env overrides

## Provenance

Added 2026-11-01 to replace hard-coded home paths; `CALIBRATION_DIR` added 2026-11-05 for `mast3r_pose.export_episode`.
