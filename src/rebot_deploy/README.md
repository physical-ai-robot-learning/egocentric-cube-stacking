# rebot_deploy

Serves X-VLA checkpoints on the reBot: inference core, robot web UI/controller, Pink IK, FK and the MIT robot service.

**Status:** current

## Inputs → Outputs

- In: a checkpoint + its V4_* contract (set by a loader in `scripts/deploy/`), live robot observations.
- Out: joint commands to the robot service, trial logs.

## Entry points

```bash
PYTHONPATH=src python -m rebot_deploy.run_smoke_ui                    # the smoke UI (usually started by scripts/deploy/load_*_ckpt_to_ui.sh)
python src/rebot_deploy/robot_service/robot_service_mit.py           # in the reBot plugin env (not included); MIT_MOCK=1 without hardware
PYTHONPATH=src python -m rebot_deploy.robot_service.usb_arm_ports     # list the arm controllers' USB locationIDs (LEFT_LOC / RIGHT_LOC)
PYTHONPATH=src python -m rebot_deploy.pink_ik                         # Pink TCP vs rebot_fk_torch self-test (needs the node's c8old)
```

## Layout

- `xvla_inference.py` (was `infer_core_v4.py`) — inference core: obs → state, flow sampling (+ RTC), chunk decode, IK backends,
  gripper mapping. Its V4_* settings are module attributes the UI switches at run time.
- `robot_dataset_contract.py` (was `v4_contract.py`) — jaw scales, CART20 / RELCART20 state builders, per-step delta clamp
- `run_smoke_ui.py` (was `mac_v4_smoke_ui.py`) — entry point of the Flask UI and controller (PLAN / RTC / trial logger); the code is
  in `smoke_ui/` (settings, state, robot client, safety gates, the STEP / STREAM / PLAN executors, `routes_*.py`, `page.html`; map in
  [`smoke_ui/__init__.py`](smoke_ui/__init__.py))
- `pink_ik.py`, `eef_kin.py` — IK and FK
- `robot_service/` — MIT-mode HTTP robot service and USB arm-port mapping
- `tests/` — characterization tests (golden values in `tests/golden_deploy.json`); the UI routes are covered by
  `tests/test_ui_routes.py` at the repository root

## Provenance

Was `deploy/`; full guide [`docs/deploy/README.md`](../../docs/deploy/README.md), sources [`docs/provenance/deploy_MANIFEST.md`](../../docs/provenance/deploy_MANIFEST.md).
