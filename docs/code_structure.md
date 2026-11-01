# Code structure and import rules

How the Python code in `src/` is organised after the 2026-11-01 code pass, what still depends on the original machines,
and which large files are candidates for a later split. The pass changed imports and path literals only; algorithms,
constants, defaults and outputs are unchanged.

## Packages and how to run them

Install once from the repository root, then use package imports:

```bash
pip install -e ".[collector,experimental,dev]"         # or: export PYTHONPATH=$PWD/src
python -m ego_cart20.scripts.convert_dataset <raw> <out>   # same as: python src/ego_cart20/scripts/convert_dataset.py ...
python -m handumi_collector.collect --hardware handumi_v1 --dataset Hpilot
pytest tests/test_imports.py -q                           # every src/ module imports from a clean checkout (see below)
```

| package | kind | notes |
|---|---|---|
| `ego_cart20`, `mast3r_pose` | regular packages | absolute package imports only |
| `ego_paths` | regular package | the machine-specific roots (next section) |
| `handumi_collector`, `ego_collector`, `ego_teleop`, `umi` | from the shared package [`egocentric`](https://github.com/physical-ai-vla/egocentric) (v0.1.0, since 2026-11-01) | installed by `pyproject.toml`; import names unchanged. `umi` = `umi.common.pose_util` (UMI API stand-in, MIT; source and md5 in [`upstream/README.md`](../upstream/README.md)) |
| `rebot_prep.{census,grip,pseudo_joint,retarget}`, `training.{relonly,umi76,robot}`, `rebot_deploy.{learned_ik,robot_service}` | flat script bundles (no `__init__.py`) | ran as plain scripts from their own folder on the Mac or a GPU node; sibling imports now try the package-relative import first and fall back to the flat one, so both `python -m <pkg>.<mod>` and `python <file>.py` work |
| `rebot_deploy` (`mac_v4_smoke_ui.py` + `smoke_ui/`, `infer_core_v4.py`, `v4_contract.py`, `pink_ik.py`, `eef_kin.py`) | regular package, also runs flat | `python src/rebot_deploy/mac_v4_smoke_ui.py` (its folder is `sys.path[0]`, as the loaders run it) and `python -m rebot_deploy.mac_v4_smoke_ui` both work: siblings are imported package-relative when `__package__` is set and flat otherwise, and the UI imports `infer_core_v4` once (`smoke_ui/ic.py`), so exactly one copy of the core is loaded either way (2026-11-01 UI split) |

## Machine-specific roots: `ego_paths`

[`src/ego_paths/__init__.py`](../src/ego_paths/__init__.py) is the one place that names the data roots. Each default is
exactly the literal the code used before, so nothing changes on the original machines; each can be overridden by an
environment variable:

| constant | default | override |
|---|---|---|
| `DATA_ROOT` | `~/c8` | `EGO_DATA_ROOT` |
| `UMI_BRIDGE` | `~/umi_bridge` | `EGO_UMI_BRIDGE` |
| `EGO_COLLECTOR_HOME` | `~/ego_collector` | `EGO_COLLECTOR_HOME` |
| `HOLOBRAIN_MAC`, `HOLOBRAIN_DATA` | `~/holobrain-mac-model`, `~/holobrain-data` | `EGO_HOLOBRAIN_MAC`, `EGO_HOLOBRAIN_DATA` |
| `NODE_HOME` | `/home/bh-aiteam` | `EGO_NODE_HOME` |
| `SHARED_ROOT` | `/srv/data/johann` | `EGO_SHARED_ROOT` |
| `CKPT_ROOT` | `/Volumes/PortableSSD/rebot_ckpts_archive` | `EGO_CKPT_ROOT` |
| `LAUNDRY_SYSTEM`, `LAUNDRY_DIT_DATASETS` | `~/laundry_system`, `~/laundry_dit_training/datasets` | `EGO_LAUNDRY_SYSTEM`, `EGO_LAUNDRY_DIT_DATASETS` |

`REPO_ROOT` points at the checkout for an editable install or `PYTHONPATH=src` (no code relies on it for a wheel install).
The `egocentric` wheel ships the top-level `umi` package, which would shadow a separately installed UMI. The shared
packages do not use `ego_paths` (it stays here); their three uses of it became `Path.home()` / `sys.path.insert` again.

Code under `src/` no longer edits `sys.path` directly. Where it must load an **external** checkout that is not part of
this repository (the GPU node's `c8old/xvla`, the LeRobot fork `lerobot-seeed`, `~/holobrain-mac-model`, MASt3R-SLAM,
`~/laundry_system`), it calls `ego_paths.prepend_sys_path(...)`, which has exactly the effect of the `sys.path.insert(0, …)` /
`sys.path[:0] = […]` it replaced. Where the repository holds a copy of such a module (`training/robot/rebot_fk_torch.py`,
`rebot_fk.py`, `rebot_prep/pseudo_joint/eef_kin.py`, ...), the external one is still tried first and the repository copy is
the fallback, so a clean clone resolves the import and the original machines behave as before.

Left as they were (listed so nothing is hidden):

- `scripts/` and `analysis/` are run bundles for particular machines and keep their own path bootstraps and absolute
  roots (93 and 8 files with `sys.path` edits; 98 and 10 with absolute roots). Paths inside records (`results/`,
  `docs/provenance/`, `legacy/`) are data and are not edited.
- In `src/`: user-facing log defaults (`~/v4_*.jsonl`, `~/rebot_trials.jsonl`), `~/robot-cockpit/...` in
  `training/robot/rebot_fk.py`, the local `HOME = pathlib.Path.home()` alias in `pseudo_joint_pipeline.py`, and docstrings.
- Node-side files now import `ego_paths`; to run one outside an installed checkout, copy `src/ego_paths/` next to it or set
  `PYTHONPATH=<repo>/src`. (The launchers in `scripts/training/` copy their code from the working trees under `~/c8` and
  `~/ego_cart20`, not from this repository, so the runs they start are unaffected.)

## Import smoke test

[`tests/test_imports.py`](../tests/test_imports.py) imports every module under `src/` in its own subprocess with
`PYTHONPATH=src`, an empty temporary `HOME`, and network and child processes disabled. A module is **skipped** when an
optional heavy dependency (torch, lerobot, pinocchio, pink, pyorbbecsdk, cv2, PySide6, gtsam, ...) is missing or when it is
an entry script that runs its program at import; it **fails** otherwise. Failures that existed on 2026-11-01 are listed in
[`tests/import_smoke_known_failures.txt`](../tests/import_smoke_known_failures.txt); the test fails on any new one. Most
listed modules are scripts that read a data file or `sys.argv` at import time.

Counts (same harness; "main" = `68dfbb5` without the laundry-era `visual_teleop` package, which moved out):

| interpreter | main: ok / skip / fail | after: ok / skip / fail |
|---|---|---|
| `~/ego_collector/.venv` (3.11) | 309 / 15 / 45 | 316 / 16 / 41 |
| `~/xvla-mac` (3.12) | 309 / 10 / 50 | 316 / 10 / 47 |

(After = 373 modules: the 369 of main plus `ego_paths`, `umi`, `umi.common`, `umi.common.pose_util`.)

After the UI split (2026-11-01, 398 modules: + `rebot_deploy.smoke_ui` and its 22 modules, `rebot_deploy.v4_contract`): 3.11 343 / 16 / 39, 3.12 343 / 10 / 45; `rebot_deploy.infer_core_v4` and `rebot_deploy.mac_v4_smoke_ui` left the known-failures list.

After the shared packages moved to `egocentric` (2026-11-01) the test covers the 143 modules left under `src/`; the
230 moved modules (236 with the six that only the garment copy had) are covered by the same test in `egocentric`. Counts with the shared package installed: see
[`docs/provenance/README.md`](provenance/README.md#shared-package-egocentric-2026-11-01).

## Split candidates (not split: behaviour first)

Done on 2026-11-01 (branch `ui-split`; behaviour checked by an AST comparison of every moved statement and by
[`tests/test_ui_routes.py`](../tests/test_ui_routes.py), a golden record of the routes, live settings and helpers taken on the single-file UI):

| file | lines | what was done |
|---|---|---|
| `src/rebot_deploy/mac_v4_smoke_ui.py` | 3815 → 48 | **done.** Split into the package `src/rebot_deploy/smoke_ui/` (22 modules + `page.html`, map in its `__init__.py`): `settings` (the V4_* settings; the 13 that `POST /mode` changes live are read as `cfg.NAME` everywhere), `state` (STATE, INF, locks, caches as `rt.NAME`), `ic` (the one `infer_core_v4` import, replacing 25 inline flat imports), `robot_client`, `motion`, `safety`, the executors `step_exec` / `stream_exec` / `plan_exec`, eleven `routes_*` modules by area, `app` + `server`, and the page as a byte-identical `page.html`. The old file is the entry point and resolves every old top-level name (module `__getattr__`). Function bodies are unchanged apart from the `cfg.` / `rt.` / `_IC` rewrites; route names, JSON fields and the page are identical. |
| `src/rebot_deploy/infer_core_v4.py` | 1203 → 1115 | **partly done.** Jaw scales and conversions, `_grip_contract`, the CART20 / RELCART20 state builders, `jaw_hold_cmd` and `clamp_delta` moved to `v4_contract.py` and are re-exported. `grip_to_cmd`, the RTC helpers, `remote_forward` and `curobo_solve` stay: they read settings the UI switches at run time on this module (`GRIPPER_MODE`, `RTC_ON`, `REMOTE_INFER`, ...), so moving them would change which value they see. Sibling imports (`eef_kin`, `v4_contract`, lazily `pink_ik`) follow `__package__`; the one nested-quote f-string was rewritten with single quotes (same AST), so the module now also parses on Python 3.11. |

Still candidates:

| file | lines | proposed split |
|---|---|---|
| `src/rebot_deploy/robot_service/robot_service_mit.py` | 837 | camera capture vs arm control vs HTTP handlers. |
| scripts that parse `sys.argv` or read data at import (`ego_cart20.scripts.{convert_episode,report_dataset,validate_dataset*}`, `training.umi76.{widen_*,norm_parity,derive_*}`, `rebot_prep.census.*`, `rebot_prep.retarget.v2k_*`) | — | move the module-level work under `main()`; this is what keeps them in the known-failures list. |
