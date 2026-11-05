"""ego_cart20: HandUMI ego episodes -> X-VLA training data in the robot's relative-Cartesian contract (Gen-2).

CART20 action (16 future poses inv(T(t)) @ T(t + k UMI_DT) per arm + gripper) and RELCART20 task-anchor state, packed to the
X-VLA v4 action32 / state20 layout; validation; LeRobot v3 export; cube-based metric scale.

Pipeline (dataset stage):
  handumi_export          HandUMI pose stack -> raw episodes (raw_episode.py format)
  scripts.convert_dataset raw episodes -> processed tree (convert.py: resample.py, pose_filter.py, labels.py, canonical_frame.py)
  scripts.validate_dataset, scripts.report_dataset   integrity report (integrity.py) / trajectory plot (trajectory_plot.py)
  scripts.export_lerobot  processed tree -> LeRobot v3 datasets (dataset.py is the native, LeRobot-free reader)
  cube_scale, cube_pnp    per-episode metric scale of a MASt3R-SLAM trajectory from a cube of known edge
The frozen contract (UMI_DT, channel map, gripper polarity) is config.py.
"""
