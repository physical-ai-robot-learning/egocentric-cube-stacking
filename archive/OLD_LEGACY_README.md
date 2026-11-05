# legacy

Superseded material, kept so that earlier results stay traceable. Nothing here is needed for the current (Gen-2)
pipeline.

**Status:** superseded

## Moved here

| path | what it was | was at | replaced by |
|---|---|---|---|
| [`c8old_dataset_v1/`](c8old_dataset_v1/) | Gen-1 C-old v1 dataset builder: segment rows → LeRobot v3 (`c8old_build_dataset.py`, `write_stage.py`), post-build and gripper-semantics gates | `scripts/rebot/dataset/` | `scripts/rebot/dataset_v2/` (C-old v2, final, frozen) |
| [`loaders/`](loaders/) | earlier per-run checkpoint → UI loaders | `scripts/rebot/loaders/legacy/` | `scripts/rebot/loaders/load_*_ckpt_to_ui.sh` |

## Superseded but kept in place

These stay at their paths because current code imports them or `analysis/verify_claims.py` reads them as evidence.

| path | what it was | replaced by | why it stays in place |
|---|---|---|---|
| `src/ego_collector/` | V0 AprilTag-glove egocentric collector | `src/handumi_collector/` | `ego_collector.camera` is imported by `handumi_collector` |
| `src/rebot_prep/pseudo_joint/`, `src/rebot_prep/census/c8_phase3.py` | Gen-1 joint-space (pseudo-joint) retargeting | Cartesian CART20 conversion in `src/ego_cart20/` | part of the `rebot_prep` package; produced the first-release data |
| `src/rebot_prep/retarget/` v2-K / v2-Kr | multi-seed kinematic retargeting variants | v2-TR and the frozen hybrid master (same folder) | same package as the current method |
| `scripts/training/c8old/`, `scripts/rebot/eval/c8old/` | Gen-1 trainer, C-old chain and offline evaluator | `src/training/relonly/` | `c8old_chain.sh` is claim evidence |
| `report/part*`, `report/methods_*`, `report/archive/` | split reports (v4) and v3 | `report/report_{en,ko}` (v5) | the archived full-length reports are claim evidence |

Old → new paths for the whole repository: [`docs/provenance/README.md`](../docs/provenance/README.md).
