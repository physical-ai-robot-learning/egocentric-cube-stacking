# handumi_collector

Raw recorder for the human (HandUMI) side: head C922 + one or two HandUMI wrist units (fisheye camera, wrist IMU, jaw encoder) → per-episode mp4 + `sensors.mcap`, plus calibration and QA tools.

**Status:** current

## Inputs → Outputs

- In: live devices; hardware profiles and task definitions in `configs/handumi/`; calibrations in `configs/calibration/`.
- Out: one episode directory per take (`episode_meta.json`, `events.json`, `sensors.mcap`, one mp4 per camera). Pose estimation is offline (`src/mast3r_pose/`, `src/ego_cart20/`).

## Entry points

```bash
python -m handumi_collector.collect --hardware handumi_v1 --dataset Hpilot
python -m handumi_collector.collect --hardware handumi_v1 --dataset HRL80 --protocol robot_like_v1
python -m handumi_collector.tools.calibrate_fisheye ...      # see docs/collector/README.md for every tool
python -m handumi_collector.tools.inspect_episode ...
```

## Layout

- `collect.py` — collector entry point (UI + session)
- `collector/` — session, episode manager, auto-loop, integrity checks
- `devices/` — camera, IMU and jaw-servo drivers
- `ui/` — PySide6 UI
- `pose/` — offline pose tooling (RGB-D, IMU, sync, QA)
- `robotlike/` — `robot_like_v1` protocol monitor and the C922-view tile
- `tools/` — calibration, monitors and QA command-line tools
- `config.py` — `REPO_ROOT` (repository root) and config loading; `calibration.py` — versioned calibration lookup

## Provenance

Copied from `~/ego_collector` on 2026-10-07 (HEAD `12db478` + working tree). Full guide: [`docs/collector/README.md`](../../docs/collector/README.md); file-level source table: [`docs/provenance/collection_MANIFEST.md`](../../docs/provenance/collection_MANIFEST.md). Tests: `tests/collector/`.

The Python files of this package are shared with the garment project ([egocentric-garment-manipulation](https://github.com/physical-ai-vla/egocentric-garment-manipulation), `src/handumi_collector/`): identical copies, see [`docs/provenance/README.md`](../../docs/provenance/README.md).
