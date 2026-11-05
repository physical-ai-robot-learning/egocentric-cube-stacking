# tests

Repository-level tests. Package characterization tests live next to their code in `src/<pkg>/tests/` (OA layout).

**Status:** current

## Inputs → Outputs

Synthetic data and small fixtures only; nothing needs hardware, recorded data, a GPU or LeRobot.

## Entry points

```bash
pip install -e ".[collector,dev]"
pytest -q src tests/cart20                  # 207 passed: package characterization tests (+ the CART20 contract / cube-scale tests)
pip install torch flask requests pin pin-pink
pytest -q tests/test_ui_routes.py           # 2 passed: reBot smoke UI vs its golden record (UI_TEST_PYTHON=<python> to use another env)
pytest -q tests/test_imports.py             # 1 passed: every module under src/ imports in a clean subprocess
pytest -q tests/collector                   # 157 passed, 3 skipped, 12 failed + 3 errors (known since 2026-11-01; not run in CI)
```

Counts measured 2026-11-05 on Python 3.12 (macOS).

## Layout

- `cart20/` — `ego_cart20` contract and cube-scale tests
- `collector/` — `handumi_collector` tests (shared-package tests; two of them skip one check when the archived module is absent)
- `test_ui_routes.py` + `ui/` — UI golden test: `ui/smoke_ui_probe.py` drives every route through Flask's test client with the robot stubbed; `ui/golden/smoke_ui_routes.json` = its output on the pre-split UI
- `test_imports.py` + `import_smoke_known_failures.txt` — import smoke test (the list is empty since 2026-11-05)

## Provenance

`tests/experimental/{teleop,v0}` and the cart20 tests of archived modules (right-only, robotize, UMI fusion) moved to
`archive/` with the code they test (2026-11-05). Before / after counts of the refactor: [`docs/refactor_2026-11.md`](../docs/refactor_2026-11.md).
