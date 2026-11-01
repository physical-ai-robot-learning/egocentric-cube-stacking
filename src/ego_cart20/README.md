# ego_cart20

Converts HandUMI ego episodes into X-VLA training samples with the robot's own relative-Cartesian contract (CART20 action, RELCART20 state), validates them and exports LeRobot datasets.

**Status:** current (Gen-2 / report v3+; replaces the Gen-1 joint-space path in `src/rebot_prep/`)

## Inputs → Outputs

- In: raw episodes in the `raw_episode.{json,npz}` format (per-arm poses + gripper + camera timestamps), written by a source adapter in `sources/`.
- Out: processed episodes + `{train,val,test}_manifest.jsonl`, validation reports, LeRobot v3 datasets.

## Entry points

```bash
python src/ego_cart20/sources/handumi_export.py --out <raw_root>
python src/ego_cart20/scripts/convert_dataset.py <raw_root> <processed_root>
python src/ego_cart20/scripts/validate_dataset.py <processed_root>
python src/ego_cart20/scripts/export_lerobot.py <processed_root> <lerobot_root>
python -m ego_cart20.right_only export|convert|sanity ...   # single-arm (HRA) variant
```

## Layout

- `config.py` — frozen contract constants (UMI_DT, CART20 channel map, gripper polarity)
- `sources/` — source adapters (HandUMI, soft-fold, h-fold, visual jaw)
- `io/`, `geometry/`, `preprocessing/`, `labels/` — loader/writer, frames, resampling, jump filter, label builders
- `scripts/` — convert / validate / report / LeRobot exporters
- `validation/`, `dataset/` — gcal1 parity, native dataset reader
- `robotize/`, `umi_fusion/` — UMI → robot robotization; tag/IMU/SLAM smoother (10-20)
- `cube_pnp.py`, `cube_scale.py`, `right_only.py` — cube-PnP metric scale and the right-only (HRA) conversion

## Provenance

Full guide: [`docs/cart20/README.md`](../../docs/cart20/README.md). Sources: [`docs/provenance/pipeline_MANIFEST.md`](../../docs/provenance/pipeline_MANIFEST.md). Tests: `tests/cart20/`.
