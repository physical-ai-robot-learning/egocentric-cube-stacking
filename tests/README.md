# tests

Tests that lived outside the packages, grouped by the component they test.

**Status:** current

## Inputs → Outputs

Synthetic data and small fixtures only; tests that need hardware, recorded data or optional packages skip or are listed as known failures.

## Entry points

```bash
pip install -e ".[collector,experimental,dev]"     # or rely on pyproject's pytest pythonpath = src
pytest tests/cart20 -q                  # 40 passed
pytest tests/experimental/teleop -q     # 297 passed
pytest tests/experimental/v0 -q         # 59 passed
pytest tests/collector -q               # 224 passed, 12 failed, 1 skipped, 3 errors (known; same on main before the restructure)
```

`tests/experimental/vteleop` does not collect (`ImportError: gripper_cmd_from_g`), on `main` as well.
`src/training/relonly/test_rel16_relonly_lossmask.py` and `src/training/umi76/test_rel16_aux*.py` stay next to the code they test (node-side).

## Layout

- `collector/` — `handumi_collector` (was `collection/tests`)
- `experimental/teleop`, `experimental/v0`, `experimental/vteleop` — `ego_teleop`, `ego_collector` V0, `visual_teleop` (was `experimental/tests`)
- `cart20/` — `ego_cart20` contract, cube scale, right-only, robotize, UMI fusion (was `cart20/tests`)

## Provenance

Counts measured on 2026-11-01 before and after the restructure with the same interpreter (Python 3.11, `~/ego_collector/.venv`).
