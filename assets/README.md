# assets

Small bundled data: three recorded demonstration episodes.

**Status:** current

## Inputs → Outputs

Read-only. `sample_episodes/<session>_<episode>/` holds three episodes of session `Hpilot_20260917_150545` (two train, one
held-out): `episode_meta.json`, `events.json`, `sensors.mcap` (IMU, gripper, per-frame camera metadata, sync events),
`head.mp4`, `derived/pose_mast3r_filtered/` (per-side camera / TCP poses + pose QA), `segments_<id>.npz` (the processed
65-frame Gen-1 segments) and one checked right-wrist frame. Format and keys: [`docs/results.md`](../docs/results.md#sample-data).

## Entry points

None (data only).

## Layout

- `sample_episodes/20260917_150545_000001/` — RBP, train
- `sample_episodes/20260917_150545_000005/` — PRB, train
- `sample_episodes/20260917_150545_000010/` — BPR, held-out

## Provenance

Was `sample_data/` (moved 2026-11-05). The tracking-body meshes and the Aero Hand URDFs that used to sit here are only used by
archived code: `archive/shared_unused/assets/handumi/`, `archive/ego_teleop/assets/aero_hand_open/` (Apache-2.0, NOTICE kept).
