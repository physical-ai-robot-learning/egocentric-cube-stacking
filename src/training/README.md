# training

X-VLA training plugins and dataset tooling that run on the GPU nodes: REL-only loss (Gen-2), UMI76/RELCART20 dataset derivation and gates, reBot FK.

**Status:** current (Gen-2). The Gen-1 trainer (`train_bi.py`, `humanik_delta.py`) is in `scripts/training/c8old/`.

## Inputs → Outputs

- In: LeRobot v3 datasets (robot or ego), the base model `lerobot/xvla-base`.
- Out: checkpoints (on the nodes; archived by `scripts/train/archivers/`).

## Entry points

Every runnable module has `main()`. On the nodes the launchers (`scripts/train/node4090|node5090/*.sh`) copy single files
next to each other and run them as flat scripts (`$PY train_rel16_relonly.py ...`, matched by name with `pgrep`), so the
file names are part of that interface and do not change; the same files also run as modules from a checkout:

```bash
python -m training.relonly.train_rel16_relonly <lerobot_train flags>            # REL-only plugin + lerobot_train
python -m training.relonly.train_rel16_relonly_lossmask <lerobot_train flags>   # + per-dim loss mask (HRA)
python -m training.relonly.train_relonly_adamw8bit <lerobot_train flags>        # AdamW8bit wrapper (node: code_8bit/train_rel16_relonly.py)
COTRAIN_EGO_ROOT=<ego set> python -m training.relonly.train_cotrain <flags>    # ego + robot co-training (node: code_cotrain/)
python -m training.umi76.umi76_to_lerobot --source zarr=..,lerobot=.. --out <dir> ...
python -m training.umi76.derive_v2 --parent <dir> --out <dir> --variant A|B|C   # or --selftest
python -m training.umi76.derive_relcart20 --parent <v3d dir> --out <dir>
python -m training.umi76.select_r312_center                                     # writes the FRONT132 / FRONT66 selections
python -m training.umi76.gate_r312c                                             # post-build gate (node data, LeRobot)
python -m training.umi76.preflight_cartonly <dataset root> <base dir> <state dim>
python -m training.robot.rebot_fk_torch                                         # torch vs numpy FK self-test (REBOT_URDF)
```

`relonly/rel16_aux_relonly.py` is kept **byte-identical**: the node launchers check its sha256 (`AUX_SHA=3511f81a...`)
before a run starts (`tests/test_relonly_plugin.py` asserts it).

## Layout

- `relonly/` — REL-only plugin (pinned), loss-mask variant, the two lerobot_train entries, AdamW8bit wrapper, co-training sampler
- `umi76/` — converter, dataset derivation (v2 / RELCART20), gate, selection, masked rel16_aux + preflight, contracts
  (`UMI76_CONTRACT.md`, `RELCART20_CONTRACT.md`) and the frozen selections (`r312c_front132_v1.json`, `r384_front204_v1.json`)
- `robot/` — numpy/torch FK (`rebot_fk*.py`); the URDF is not included
- `tests/` — characterization tests on synthetic data with stand-ins for LeRobot / zarr / bitsandbytes
  (`PYTHONPATH=src python -m pytest src/training/tests`)

## Provenance

Was `training/{relonly,umi76,robot}`; full guide [`docs/training/README.md`](../../docs/training/README.md), sources [`docs/provenance/training_MANIFEST.md`](../../docs/provenance/training_MANIFEST.md).
