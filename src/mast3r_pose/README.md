# mast3r_pose

Per-frame MASt3R-SLAM wrapper for the wrist fisheye videos (RGB-only camera pose) and the per-episode IMU visual-inertial metric scale.

**Status:** current

## Inputs → Outputs

- In: a raw HandUMI episode (exported per side to the UMI / ORB-SLAM3 layout by `export_episode`), a MASt3R-SLAM checkout at the commit in `INSTALLED_COMMIT` with the checkpoints in `MD5SUMS` (`M3_REPO`, default `~/mast3r_slam_official`).
- Out: per-frame camera poses (`T_WC`) with tracking QA; metric scale per episode (`imu_metric_scale`).

## Entry points

```bash
python -m mast3r_pose.export_episode <episode dir> {left,right} <out dir> [--downscale 2] [--start-at-go] [--imu-preroll-ms 50]
python -m mast3r_pose.run_perframe --video <dir>/raw_video.mp4 --setting <dir>/orbslam_setting.yaml --out <dir>/ss1.csv   # GPU, m3slam env
python -m mast3r_pose.run_perframe_resident --list <tags> --in-root <dir> --out-dir <dir> [--name ss1]                      # many sides, one process
M3_BASE=<benchmark dir> M3_VARIANT=ss1 python -m mast3r_pose.imu_metric_scale                                              # C-P6 benchmark
```

The calibration folder of `export_episode` is `ego_paths.CALIBRATION_DIR` (default `<repo>/configs/calibration`, env `EGO_CALIBRATION_DIR`).

## Layout

- `export_episode.py` — raw episode side → `raw_video.mp4`, `imu_data.json`, `orbslam_setting.yaml` (was `scripts/umi_export_orbslam.py`)
- `run_perframe.py`, `run_perframe_resident.py` — per-frame pose runners (the resident one reuses run_perframe's functions)
- `imu_metric_scale.py` — IMU-VI metric scale (was `cp6_scale.py`; the census stage execs the frozen external copy of the original text, not this module)
- `INSTALLED_COMMIT`, `MD5SUMS` — pinned upstream commit and checkpoint hashes
- `tests/` — characterization tests (synthetic benchmark tree, fake episode)
- Run scripts: `scripts/pose/`

## Provenance

Was `pipeline/mast3r_pose/`; see [`docs/provenance/pipeline_MANIFEST.md`](../../docs/provenance/pipeline_MANIFEST.md). Upstream: <https://github.com/rmurai0610/MASt3R-SLAM>.
