# ego_cart20

Converts HandUMI ego episodes into X-VLA training samples with the robot's own relative-Cartesian contract (CART20 action, RELCART20 state), validates them and exports LeRobot datasets.

**Status:** current (Gen-2 / report v3+; replaces the Gen-1 joint-space path in `src/rebot_prep/`)

## Inputs → Outputs

- In: raw episodes in the `raw_episode.{json,npz}` format (per-arm poses + gripper + camera timestamps), written by a source adapter (`handumi_export.py`).
- Out: processed episodes + `{train,val,test}_manifest.jsonl`, validation reports, LeRobot v3 datasets.

## Entry points

```bash
python -m ego_cart20.handumi_export --out <raw_root>
python -m ego_cart20.scripts.convert_dataset <raw_root> <processed_root> [--split frozen_handumi|hash] [--discontinuity-filter]
python -m ego_cart20.scripts.convert_episode <raw_episode_dir> <out_episode_dir>
python -m ego_cart20.scripts.validate_dataset <processed_root> [report.json]
python -m ego_cart20.scripts.report_dataset <processed_root> <out_dir> [n_samples]
python -m ego_cart20.scripts.export_lerobot <processed_root> <lerobot_root>     # needs lerobot
```

## Layout

- `config.py` — frozen contract constants (UMI_DT, CART20 channel map, gripper polarity) and `Cart20Config`
- `rotation6d.py`, `transforms.py`, `canonical_frame.py` — SO(3) / rot6d, SE(3) / pose9, the per-arm task-start frame
- `raw_episode.py`, `processed_episode.py` — raw episode reader / writer, processed episode writer / reader
- `resample.py`, `gripper.py`, `pose_filter.py` — raw-track sampling and stream sync, gripper normalisation, jump filter
- `labels.py`, `convert.py` — CART20 / state20 / action32 / stage builders, the one-episode conversion
- `splits.py`, `dataset.py` — episode splits, native (LeRobot-free) dataset reader and sample weights
- `integrity.py`, `trajectory_plot.py` — dataset integrity report, CART20 trajectory plot
- `handumi_export.py` — HandUMI source adapter (needs the external C8 pose stack)
- `cube_pnp.py`, `cube_scale.py` — cube-based metric scale of a MASt3R-SLAM trajectory
- `scripts/` — command-line entry points; `tests/` — characterization tests (golden values)

Superseded and exploratory parts are archived: robotize / UMI fusion / extra source adapters in `archive/ego_cart20_experiments/`,
the right-only (HRA) variant in `archive/hra_single_arm/`, soft-fold in `archive/softfold/`.

## Provenance

Full guide: [`docs/cart20/README.md`](../../docs/cart20/README.md). Sources: [`docs/provenance/pipeline_MANIFEST.md`](../../docs/provenance/pipeline_MANIFEST.md). Tests: `tests/cart20/`, `src/ego_cart20/tests/`.
