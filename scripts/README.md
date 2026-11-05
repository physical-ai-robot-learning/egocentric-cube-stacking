# scripts

Thin operational entry points only: shell launchers for the GPU nodes (Ray / ssh), the MASt3R-SLAM GPU and the robot
computer, grouped by pipeline stage. Library code lives in `src/`; one-off tools are in [`archive/`](../archive/README.md).

**Status:** current (records of how the reported runs were launched)

## Inputs → Outputs

Each script documents its inputs in its header. Most read and write the data roots of the machine they ran on
(`~/c8`, `~/umi_bridge`, `/home/<node user>`, the shared NFS root) and copy their code from working trees on those machines,
not from this checkout; they are kept as the exact launch records and are not portable as they are. Hostnames and IPs in
them are the lab's private tailnet addresses.

## Entry points / Layout

| folder | stage | what |
|---|---|---|
| `pose/` | 1 pose | `c8_export.sh` / `c8_export_early.sh`: export every episode side for MASt3R-SLAM (the exporter is `src/mast3r_pose/export_episode.py`; the scripts call the copy in the original `~/ego_collector` working tree); `c8_run_5080.sh <side list> <out dir>`, `c8_rest490_resident.sh`: per-frame MASt3R-SLAM on the RTX 5080 (`python -m mast3r_pose.run_perframe[_resident]`) |
| `dataset/gen1_retarget/` | 2–3 Gen-1 | chains of the multi-seed retarget v2 (TR, v2-Kr, HRL80) around `python -m rebot_prep.retarget.multi_seed_retarget` (they call the original `v2k_retarget.py` working copy) |
| `train/node4090/`, `train/node5090/` | 4 train | node launchers: dataset builds (`build_r312c_umi76.sh`, `build_chain_r384.sh`, R90 selection `select_r90.py` / `build_r90.py`) and training runs of `src/training/relonly/` (ego pretrain, R312c fine-tune, co-training, R30 / R90 ablations) |
| `train/ego_cart20/`, `train/mac_chains/` | 4 train | Mac-side submitters and waiters (`submit_ego_cart20v2.sh <gpu>`, `ft_r312c_when_*_done.sh`, …): wait for a free GPU, then submit |
| `train/archivers/` | 4 train | checkpoint archivers (node → SSD with sha256 checks); see [`docs/training/README.md`](../docs/training/README.md#archiving) |
| `deploy/` | 5 deploy | `load_relonly_run_ckpt_to_ui.sh <step> [port]`, `load_relonly_5090_run_ckpt_to_ui.sh`: fetch a checkpoint, verify sha256 + identity, start the UI (`python -m rebot_deploy.run_smoke_ui`); `auto_ui_d20_e5_v5.sh`: load each new checkpoint on the bimanual ports |

The Gen-1 C-old trainer and its chain (`c8old_chain.sh`) are in [`archive/gen1_c8old/training/`](../archive/gen1_c8old/training/).

## Provenance

`scripts/rebot/{c8,mast3r,loaders}` and `scripts/training/*` of the 2026-11-01 layout; map in
[`docs/refactor_2026-11.md`](../docs/refactor_2026-11.md). `scripts/dataset/gen1_retarget/` was `src/rebot_prep/retarget/chains/`.
