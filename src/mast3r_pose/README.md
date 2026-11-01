# mast3r_pose

Per-frame MASt3R-SLAM wrapper for the wrist fisheye videos (RGB-only camera pose) and the per-episode IMU visual-inertial metric scale.

**Status:** current

## Inputs → Outputs

- In: wrist video + intrinsics (exported episodes), a MASt3R-SLAM checkout at the commit in `INSTALLED_COMMIT` with the checkpoints in `MD5SUMS`.
- Out: per-frame camera poses (`T_WC`) with tracking QA; metric scale per episode (`cp6_scale.py`).

## Entry points

```bash
python src/mast3r_pose/run_perframe.py ...            # GPU, MASt3R-SLAM env
python src/mast3r_pose/run_perframe_resident.py ...   # resident queue variant
scripts/rebot/mast3r/c8_run_5080.sh <side list> <out dir> [<input root> <output name>]
```

## Layout

- `run_perframe.py`, `run_perframe_resident.py` — per-frame pose runners
- `cp6_scale.py` — IMU-VI metric scale
- `INSTALLED_COMMIT`, `MD5SUMS` — pinned upstream commit and checkpoint hashes
- Run scripts: `scripts/rebot/mast3r/`

## Provenance

Was `pipeline/mast3r_pose/`; see [`docs/provenance/pipeline_MANIFEST.md`](../../docs/provenance/pipeline_MANIFEST.md). Upstream: <https://github.com/rmurai0610/MASt3R-SLAM>.
