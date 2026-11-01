# rebot_prep

Gen-1 human → reBot preparation: gripper width from jaw ticks, census and gates, pseudo-joint retargeting with continuity IK, and the multi-seed retarget v2 with its frozen contracts.

**Status:** superseded (by `ego_cart20`, report v3) for new training data; it produced the Part I C-old datasets and is kept as their evidence

## Inputs → Outputs

- In: exported episodes, MASt3R poses + scale, jaw ticks, robot reference data (not included).
- Out: per-episode census/gate records, pseudo-joint segments (65 frames), retarget v2 manifests.

## Entry points

```bash
python src/rebot_prep/grip/c8_grip.py ...
python src/rebot_prep/census/c8_census_pre.py ...
python src/rebot_prep/census/c8_phase3.py ...
python src/rebot_prep/retarget/v2k_retarget.py ...      # chains in src/rebot_prep/retarget/chains/
```

## Layout

- `grip/` — `extract_grip.py`, `correct_grip.py` (aperture rule), `c8_grip.py`
- `census/` — sensor/sync census, Phase-3 pose → metric TCP → pseudo-q, collision, gravity-sign and regression checks
- `pseudo_joint/` — retarget + continuity IK core, state layout, input gate, `eef_kin.py`
- `retarget/` — retarget v2 (seed bank, v2-K/Kr/TR, comparison, hybrid merge, HRL80 prereg); `contracts/` = frozen rules and splits; `chains/`

## Provenance

Was `pipeline/{grip,census,pseudo_joint,retarget_v2}/`; sources in [`docs/provenance/pipeline_MANIFEST.md`](../../docs/provenance/pipeline_MANIFEST.md) and [`docs/SOURCES_v2.md`](../../docs/SOURCES_v2.md). Scripts use the data roots `~/c8`, `~/umi_bridge` and import siblings by file location, as in the original working directories.
