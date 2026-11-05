"""Pose stage (stage 1): HandUMI episode export, per-frame MASt3R-SLAM wrist camera poses and the IMU visual-inertial metric
scale.

Modules: export_episode (raw HandUMI episode side -> UMI / ORB-SLAM3 layout), run_perframe (per-frame MASt3R-SLAM, GPU),
run_perframe_resident (many sides in one process), imu_metric_scale (C-P6 IMU-VI scale benchmark).
"""
