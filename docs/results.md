# Results, dataset and reports

The detailed material behind the README's result box: the dataset and its processing funnel, the sample data, the
headline offline results, the demonstration videos, the reports and the release history. Moved here verbatim from the
README on 2026-11-05 (only relative links changed). `analysis/verify_claims.py` checks the numbers in this file together
with `README.md` and `docs/history.md`.

## Dataset

- **Task:** stacking three cubes (red, blue, purple) on a plate, in 6 stacking orders (RBP, RPB, BRP, BPR, PRB, PBR).
  The instruction text is `Stack the {bottom} cube on the bottom, {middle} cube in the middle, and {top} cube on the top, on the plate.`
- **Rig:** each hand wears a HandUMI unit with a wrist fisheye camera (1080p, 30 fps), a Teensy IMU (200 Hz) and a
  Feetech jaw encoder. A fixed C922 head camera (640x480, 30 fps) looks down at the table.
- **Processing funnel:**
  - 349 recorded episodes
  - 284 pass sync (left wrist is the master clock; the right wrist and head must be within 20 ms)
  - 259 source episodes remain after pose, gripper and gate filtering
  - 561 bimanual segments of 65 frames each
- **Pose:** MASt3R-SLAM gives the RGB-only wrist-camera pose. Metric scale comes from a per-episode IMU visual-inertial
  fit (`mast3r_pose.imu_metric_scale.imuViScale`, was `cp6_scale.imu_vi`). On a separate 14-episode ArUco benchmark (10 tracks held out) the scale error is
  4.6 / 17.0 % (p50 / p90) and the 16-step relative position error p90 is 11.3 mm (`results/pose_benchmark/`). The camera-to-TCP transform is `handumi_camera_tcp_v2`.
- **Retargeting:** trajectories are anchored at the robot start pose and retargeted to reBot B601 pseudo-joints with
  continuity IK. Segments then pass workspace, metric and IK gates (an inter-arm clearance check is used only in
  retargeting v2). Sessions with inconsistent
  IMU and camera rotation are quarantined (`results/c8_quarantine.json`).
- **Split:** by source episode, never by segment. `numpy.random.default_rng(0)` gives 233 train episodes (504 segments)
  and 26 held-out episodes (57 segments). See `results/split_index.json`.
- **Model:** X-VLA (879M parameters), initialised from `lerobot/xvla-base`, pretrained for 300k steps on the ego set. Loss terms:
  - Main target: a Δq_pseudo action chunk, `q[t+5+i] - q[t]` for i = 0..29 (LEAD 5, 30 steps). It is defined in `archive/gen1_c8old/training/humanik_delta.py`.
  - EEF auxiliary loss on relative TCP xyz (`XVLA_EE_AUX`).
  - FK-consistency loss.
- **Evaluation:** offline, on the 26 held-out episodes. Metrics are joint MAE against a zero-action baseline, FK
  position error, direction cosine and a collapse ratio. The checkpoint selection rule was registered before any
  results (`geo_score`); a second, motion-gated selector was added post hoc and is labelled as such
  (`results/selection.json`).

The held-out set is a random 10% of source episodes. It is not balanced by order or by session, and the cube layouts
were not systematically randomised. None of the results here are claims about generalisation.


## Sample data

`assets/sample_episodes/<session>_<episode>/` holds three episodes from session `Hpilot_20260917_150545`:

| id | order | split |
|---|---|---|
| `20260917_150545_000010` | BPR | held-out (val) |
| `20260917_150545_000001` | RBP | train |
| `20260917_150545_000005` | PRB | train |

Each episode directory contains:

- `episode_meta.json`, `events.json`
- `sensors.mcap`: IMU, gripper, per-frame camera metadata and sync events
- `head.mp4`: head camera, re-muxed video-only
- `derived/pose_mast3r_filtered/`: per-side camera/TCP pose parquet files, canonical table and pose QA
- `segments_<id>.npz`: the processed 65-frame segments. Keys are `starts, st, ac, tcp_tgt, vfL, vfR, vfH, epd, instr, val`. `epd` is the episode path relative to the raw dataset root (`Hpilot/<session>/<episode>`).
- `right_wrist_frame.jpg`: one 640 px frame from the right wrist camera. It was checked by eye to show only the tabletop, cubes, plate and HandUMI hardware.

The sample episodes ship the head video only; wrist fisheye views of three HRL80 takes are shown in the
[MASt3R-SLAM trajectory videos](history.md#mast3r-slam-trajectory-videos).


## Key results (offline, 26 held-out episodes, final 300k checkpoint)

| | all arm-samples (798) | moving, TCP ≥ 20 mm by k=30 (187) |
|---|---|---|
| median TCP error, k=1 / k=30 | 1.5 / 5.7 mm | 11.2 / 39.9 mm |
| joint MAE k=30 (zero-motion predictor) | 0.97° (0.51°) | 6.59° (8.68°) |
| TCP direction cosine, k=30 | 0.81 | 0.90 |

- Aggregate errors are dominated by static phases; on static samples the policy predicts spurious motion.
- Prompt swap: the 6 order instructions move the k=30 prediction by 4.8 mm (≈8 % of the 62 mm predicted displacement),
  and the true instruction is no closer to the recorded motion than a wrong one (mean rank 3.55 of 6, chance 3.5).
  The instruction measurably changes predictions but does not make them more accurate.


## Demonstration videos

Head-camera view of three recorded human demonstrations (2x speed). Left to right: RBP (train), PRB (train),
BPR (held-out validation episode). Full-length videos (640x480, 30 fps, 23 s, no audio):

| Episode | Order (bottom → top) | Split | Video |
|---|---|---|---|
| `20260917_150545_000001` | red, blue, purple | train | [head.mp4](../assets/sample_episodes/20260917_150545_000001/head.mp4) |
| `20260917_150545_000005` | purple, red, blue | train | [head.mp4](../assets/sample_episodes/20260917_150545_000005/head.mp4) |
| `20260917_150545_000010` | blue, purple, red | held-out | [head.mp4](../assets/sample_episodes/20260917_150545_000010/head.mp4) |

**MASt3R-SLAM wrist trajectories** (v3): head + both wrist cameras, gravity-aligned 3D wrist trajectory and gripper
trace. Details in [MASt3R-SLAM trajectory videos](history.md#mast3r-slam-trajectory-videos).

| Video | Take | Metric scale |
|---|---|---|
| [hrl80_000035_wrist.mp4](../media/mast3r_trajectories/hrl80_000035_wrist.mp4) | HRL80 bimanual stacking, PRB, 33 s | IMU-VI |
| [hrl80_000055_wrist.mp4](../media/mast3r_trajectories/hrl80_000055_wrist.mp4) | HRL80 bimanual stacking, RBP, 33 s | IMU-VI |
| [hrl80_000060_wrist.mp4](../media/mast3r_trajectories/hrl80_000060_wrist.mp4) | HRL80 bimanual stacking, PBR, 33 s | IMU-VI |

These are human demonstrations recorded with the hand-worn rig, not robot rollouts. Wrist-camera videos are not
published (see [Sample data](#sample-data)).


## Reports

| Report | English | 한국어 | Full-length version |
|---|---|---|---|
| **Technical report (v5.2, one document, 15 pp; v5.2 = v5 with the code map updated to the 11-05 layout)** | [`report/report_en.pdf`](../report/report_en.pdf) | [`report/report_ko.pdf`](../report/report_ko.pdf) | (is the full report) |
| Part I: three-cube stacking (bimanual), 4 pp | [`report/archive/part1_stacking_en.pdf`](../report/archive/part1_stacking_en.pdf) | [`report/archive/part1_stacking_ko.pdf`](../report/archive/part1_stacking_ko.pdf) | `report/archive/part1_stacking_{en,ko}_full.pdf` (11 pp, v4.1) |
| Part II: single-arm approach to a cube, 2 pp | [`report/archive/part2_approach_en.pdf`](../report/archive/part2_approach_en.pdf) | [`report/archive/part2_approach_ko.pdf`](../report/archive/part2_approach_ko.pdf) | `report/archive/part2_approach_{en,ko}_full.pdf` (5 pp) |
| Methods and pipeline (how every stage works), 6 pp | [`report/archive/methods_en.pdf`](../report/archive/methods_en.pdf) | [`report/archive/methods_ko.pdf`](../report/archive/methods_ko.pdf) | — |

The v5.2 technical report is the main document. The earlier split documents remain in `report/archive/` (short and
full-length versions). LaTeX
sources sit next to the PDFs; figures are rebuilt by `report/figures/make_figures*.py`, `analysis/v4_r30_ablation.py` and
`analysis/v4_hra_summary.py`. The v3 single report is also in `report/archive/`. `analysis/verify_claims.py` also checks that every value required in the full-length reports appears in the v5 report.
All reports now cite the upstream HandUMI hardware repository.
The Part I full version 4.1 (2026-10-12) adds Sec. III, *HandUMI hardware*: the upstream HandUMI project (murobotics-ai/handumi-hw by BrikHMP18,
formerly robonet-ai; we built `e58de33` and forked handumi-sw at `33cc437`, the fork not used for the reported data), print and electronics, our deviations (no VR tracking, added Teensy 4.1 + ICM-42688-P wrist IMU, fixed-focus head
C922), our three modified parts (Fig. 2, `report/figures/make_fig10_handumi_mods.py`) and device-identity pitfalls. The hardware
files are in `hardware/` and in our build copy [`physical-ai-robot-learning/umihand`](https://github.com/physical-ai-robot-learning/umihand); the design itself is
upstream [murobotics-ai/handumi-hw](https://github.com/murobotics-ai/handumi-hw).


## Release history

Research code, calibration, processing records and a small data sample from one experiment. We collected bimanual
human demonstrations with a hand-worn UMI-style gripper ([HandUMI](https://github.com/murobotics-ai/handumi-hw), not our
design; see [Hardware source](../README.md#hardware-source-handumi-is-not-our-design)), converted them into pseudo-joint trajectories
for a reBot B601 dual-arm robot, and used them to pretrain X-VLA. Version 3 (October 2026) replaces the joint-space
retargeting with a Cartesian conversion that matches the robot's own action contract, and tests what transfers to the
robot ([Update v3](history.md#update-v3-cartesian-ego-pretraining-and-transfer)). Version 4 (7 October 2026) adds a 30-episode
low-data ablation and splits the write-up into two reports, each in English and Korean
([Update v4](history.md#update-v4-two-reports-r30-ablation-and-single-arm-approach)). Version 5 (12 October 2026) merges everything
into **one technical report**: [`report/report_en.pdf`](../report/report_en.pdf) ([한국어](../report/report_ko.pdf)), 15 pages
covering hardware, every pipeline stage, all results and their limits, with no result changed. Earlier states are tagged
`report-v2-2026-09-30` and `report-v3-2026-10-04`.

The results added by each release (multi-seed retargeting, robot-like collection and fine-tuning in v2; Cartesian ego
pretraining and transfer diagnostics in v3; the R30 ablation and single-arm approach in v4) and the repository maps of
every release are in [`docs/history.md`](history.md):
[Update v2](history.md#update-v2-multi-seed-retargeting-robot-like-collection-combined-dataset) ·
[Update v3](history.md#update-v3-cartesian-ego-pretraining-and-transfer) ·
[Update v4](history.md#update-v4-two-reports-r30-ablation-and-single-arm-approach) ·
[Repository maps](history.md#repository-maps-by-release-historical).

