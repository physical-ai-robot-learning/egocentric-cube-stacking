# rebot_deploy

Serves X-VLA checkpoints on the reBot: inference core, robot web UI/controller, Pink IK, FK, the MIT robot service and an optional learned-IK backend.

**Status:** current

## Inputs → Outputs

- In: a checkpoint + its V4_* contract (set by a loader in `scripts/rebot/loaders/`), live robot observations.
- Out: joint commands to the robot service, trial logs.

## Entry points

```bash
python src/rebot_deploy/mac_v4_smoke_ui.py              # usually started by scripts/rebot/loaders/load_*_ckpt_to_ui.sh
python src/rebot_deploy/robot_service/robot_service_mit.py   # in the reBot plugin env (not included); MIT_MOCK=1 without hardware
python src/rebot_deploy/hra_direction_probe.py ...
```

## Layout

- `infer_core_v4.py` — inference core (obs → state, flow sampling, chunk decode, IK backends, gripper mapping)
- `mac_v4_smoke_ui.py` — Flask UI and controller (PLAN / RTC / trial logger)
- `pink_ik.py`, `eef_kin.py` — IK and FK
- `robot_service/` — MIT-mode HTTP robot service and USB arm-port mapping
- `learned_ik/` — optional learned IK (contract docs + scripts)
- `hra_direction_probe.py` — offline HRA direction probe

## Provenance

Was `deploy/`; full guide [`docs/deploy/README.md`](../../docs/deploy/README.md), sources [`docs/provenance/deploy_MANIFEST.md`](../../docs/provenance/deploy_MANIFEST.md).
