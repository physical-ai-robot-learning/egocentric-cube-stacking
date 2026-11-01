# training

X-VLA training plugins and dataset tooling that run on the GPU nodes: REL-only loss (Gen-2), UMI76/RELCART20 dataset derivation and gates, reBot FK.

**Status:** current (Gen-2). The Gen-1 trainer (`train_bi.py`, `humanik_delta.py`) is in `scripts/training/c8old/`.

## Inputs → Outputs

- In: LeRobot v3 datasets (robot or ego), the base model `lerobot/xvla-base`.
- Out: checkpoints (on the nodes; archived by `scripts/training/archivers/`).

## Entry points

```bash
python src/training/relonly/train_rel16_relonly.py ...            # usually via scripts/training/node4090|node5090 launchers
python src/training/relonly/train_rel16_relonly_lossmask.py ...   # loss-masked (HRA)
python src/training/umi76/umi76_to_lerobot.py ...
```

## Layout

- `relonly/` — REL-only plugin, loss-mask variant (+ test), AdamW8bit wrapper, co-training sampler
- `umi76/` — dataset derivation (v2/v3/CART20/RELCART20), gates, base widening, contracts (`UMI76_CONTRACT.md`, `RELCART20_CONTRACT.md`)
- `robot/` — numpy/torch FK (`rebot_fk*.py`); the URDF is not included

## Provenance

Was `training/{relonly,umi76,robot}`; full guide [`docs/training/README.md`](../../docs/training/README.md), sources [`docs/provenance/training_MANIFEST.md`](../../docs/provenance/training_MANIFEST.md).
