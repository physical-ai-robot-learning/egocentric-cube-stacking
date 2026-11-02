# scripts

Command-line entry points. Tools that find the repository root with `Path(__file__).parents[1]` sit flat in this folder;
machine- and stage-specific run bundles sit in subfolders.

**Status:** current, except the items marked superseded below

## Inputs → Outputs

Each script documents its own inputs in its header. Most reBot and node scripts read and write the data roots
(`~/c8`, `~/umi_bridge`, `/home/bh-aiteam`, `/srv/data/johann`) directly, not paths inside this repository.

## Entry points

```bash
python scripts/charuco_calibrate.py ...          # flat tools: calibration, camera QA, robot-like checks, RGB-D body tracking
python scripts/robotlike_offline_check.py ...
python scripts/umi_export_orbslam.py <episode> <side> <out>
scripts/rebot/mast3r/c8_run_5080.sh <side list> <out dir>
python legacy/c8old_dataset_v1/c8old_build_dataset.py gate|rows|write
scripts/training/c8old/c8old_chain.sh            # Gen-1 C-old chain (4090)
```

## Layout

- `*.py`, `*.sh` (flat) — from the former `collection/scripts`, `collection/tools` and `experimental/scripts`: calibration
  (ChArUco, table frame, exposure), robot-like protocol checks, HandUMI RGB-D body tracking (parked), `provenance.py`
- `rebot/c8/` — episode export for the C8 pipeline (`c8_export*.sh`, `umi_export_orbslam.py` copy)
- `rebot/mast3r/` — MASt3R-SLAM run scripts for the GPU nodes
- Gen-1 C-old v1 dataset builder: moved to [`legacy/c8old_dataset_v1/`](../legacy/c8old_dataset_v1/) (superseded by `rebot/dataset_v2/`)
- `rebot/dataset_v2/` — C-old v2 (TR / hybrid / HRL80) builders and validators
- `rebot/eval/rel16_audit/` — Gen-2 checkpoint audits and gate watchers; `rebot/eval/c8old/` — Gen-1 offline evaluator
- `rebot/hra/` — single-arm approach (HRA) pipeline: `hra_red/`, `hra_a100/` (see `docs/hra/README.md`)
- The laundry / garment material that used to sit here (`rebot/laundry/`, `rebot/parity/`, `umi_teleop_eval/`) lives in
  [egocentric-garment-manipulation](https://github.com/physical-ai-robot-learning/egocentric-garment-manipulation) since 2026-11-01.
- `rebot/loaders/` — checkpoint → UI loaders and auto-swappers; earlier per-run loaders are in [`legacy/loaders/`](../legacy/loaders/) (superseded)
- `training/node4090/`, `training/node5090/` — node launchers; `training/hra_node/`, `training/mac_chains/`, `training/ego_cart20/` — wrappers and Mac-side chains;
  `training/archivers/` — checkpoint archivers; `training/c8old/` — Gen-1 trainer, patches and the C-old ladder (superseded by `src/training/relonly`)

## Provenance

Old → new paths: [`docs/provenance/README.md`](../docs/provenance/README.md). Per-file sources: the MANIFEST files in `docs/provenance/`.
