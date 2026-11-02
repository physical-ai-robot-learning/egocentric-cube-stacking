# tests

Tests that lived outside the packages, grouped by the component they test.

**Status:** current

## Inputs → Outputs

Synthetic data and small fixtures only; tests that need hardware, recorded data or optional packages skip or are listed as known failures.

## Entry points

```bash
pip install -e ".[collector,experimental,dev]"     # or rely on pyproject's pytest pythonpath = src
pytest tests/cart20 -q                  # 40 passed
pytest tests/test_imports.py -q         # every src/ module imports from a clean checkout (known failures: import_smoke_known_failures.txt)
pytest tests/test_ui_routes.py -q       # the reBot smoke UI vs its pre-split golden record (needs flask, torch, cv2; UI_TEST_PYTHON=...)
```

The former `tests/experimental/vteleop` (tests of `visual_teleop`, which did not collect: `ImportError: gripper_cmd_from_g`) moved with
`visual_teleop` to [egocentric-garment-manipulation](https://github.com/physical-ai-vla/egocentric-garment-manipulation) on 2026-11-01.
The tests of `handumi_collector`, `ego_teleop` and `ego_collector` (`tests/collector`, `tests/experimental/{teleop,v0}`)
moved with those packages to the shared package [`handumi-common`](https://github.com/physical-ai-vla/handumi-common) on 2026-11-01 (same results there: 224/12/1/3,
297, 59).
`src/training/relonly/test_rel16_relonly_lossmask.py` and `src/training/umi76/test_rel16_aux*.py` stay next to the code they test (node-side).

## Layout

- `cart20/` — `ego_cart20` contract, cube scale, right-only, robotize, UMI fusion (was `cart20/tests`)
- `test_imports.py` + `import_smoke_known_failures.txt` — import smoke test of every module under `src/` (see `docs/code_structure.md`)

## Provenance

Counts measured on 2026-11-01 before and after the restructure with the same interpreter (Python 3.11, `~/ego_collector/.venv`).
