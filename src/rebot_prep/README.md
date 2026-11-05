# rebot_prep

Gen-1 human → reBot preparation: gripper width from jaw ticks, census and gates, pseudo-joint retargeting with continuity IK, and the multi-seed retarget v2 with its frozen contracts.

**Status:** superseded (by `ego_cart20`, report v3) for new training data; it produced the Part I C-old datasets and is kept as their evidence

## Inputs → Outputs

- In: exported episodes, MASt3R poses + scale, jaw ticks, robot reference data (not included).
- Out: per-episode census/gate records, pseudo-joint segments (65 frames), retarget v2 manifests.

## Entry points

```bash
python -m rebot_prep.grip.extract_grip [--manifest <json>] [--out <dir>]
python -m rebot_prep.grip.correct_grip [--grip <dir>] [--out <dir>]
python -m rebot_prep.grip.grip_census
python -m rebot_prep.census.sensor_sync_census
python -m rebot_prep.census.phase3_segments [episode ...]
python -m rebot_prep.census.mast3r_regression [tag ...]
python -m rebot_prep.census.imu_vi_gravity <tag> [...]
python -m rebot_prep.census.gravity_sign_gate
python -m rebot_prep.pseudo_joint.input_gate [--produce <dir>] [--tag raw] [--grip <dir>]
python -m rebot_prep.retarget.seed_bank [--ref r30]
python -m rebot_prep.retarget.multi_seed_retarget --episodes-json <json> --raw-root <dir> --runs <dir> --export <dir> --out <dir> [--mode v2k|v2kr|r30|tr]
python -m rebot_prep.retarget.compare_retargets --group NAME=<dir> [...] --out <json> [--eval-ref r150full|heldout120]
python -m rebot_prep.retarget.rescued_tail <out dir>
python -m rebot_prep.retarget.hybrid_merge                     # env HM_TR / HM_KR / HM_OUT
python -m rebot_prep.retarget.hrl80_prereg_report --eval-ref heldout120
```

The chain scripts are in `scripts/dataset/gen1_retarget/` (they call the original working copies under `~/umi_bridge/track_c/v2k`).

## Layout

- `grip/` — `extract_grip.py`, `correct_grip.py` (aperture rule, `APERTURE_M`), `grip_census.py` (was `c8_grip.py`)
- `census/` — `sensor_sync_census.py` (was `c8_census_pre.py`), `phase3_segments.py` (`c8_phase3.py`), `mast3r_regression.py` (`c8_regress_ref208.py`), `imu_vi_gravity.py` (`c8_imu_vi_g.py`), `gravity_sign_gate.py` (`c8_g1_gravity_sign.py`), `arm_clearance.py` (`c8_collision_v1.py`)
- `pseudo_joint/` — `pseudo_joint_pipeline.py` (retarget + continuity IK core), `state32.py`, `eef_kin.py`, `input_gate.py` (was `c2_input_gate.py`)
- `retarget/` — `seed_bank.py` (`v2k_seedbank.py`), `multi_seed_retarget.py` (`v2k_retarget.py`), `compare_retargets.py` (`v2k_compare.py`), `rescued_tail.py` (`v2k_tail.py`), `hybrid_merge.py`, `hrl80_prereg_report.py`; `contracts/` = frozen rules, splits and seed banks (data, names unchanged)
- `tests/` — characterization tests on synthetic data with a fake reBot solver (`tests/fake_robot.py`)

Machine roots come from `ego_paths` (`~/c8`, `~/umi_bridge`, `~/ego_collector`). Where the original working directories held a twin of a module (`~/umi_bridge/track_c/pseudo_joint_pipeline.py`, `state32.py`, `extract_grip.py`, `v2k/v2k_compare.py`, `~/c8/c8_collision_v1.py`, `~/holobrain-mac-model/eef_kin.py`), that copy is imported first and the repository's is the fallback; the public names those twins share (e.g. `ContinuityIK.solve`, `Kin.fk_pose7`, `build_state32`) therefore stay as they are.

## Provenance

Was `pipeline/{grip,census,pseudo_joint,retarget_v2}/`; sources in [`docs/provenance/pipeline_MANIFEST.md`](../../docs/provenance/pipeline_MANIFEST.md) and [`docs/SOURCES_v2.md`](../../docs/SOURCES_v2.md).
