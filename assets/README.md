# assets

Runtime assets read by the code.

**Status:** current

## Inputs → Outputs

Read-only inputs: tracking-body meshes for the RGB-D tools and the Aero Hand URDFs for the FK tests and the Aero branch.

## Entry points

None. `handumi_collector.tools.make_tracking_mesh` writes `handumi/*_tracking_body.yaml`.

## Layout

- `handumi/` — HandUMI tracking-body meshes (STL) and yaml/json
- `aero_hand_open/` — Aero Hand Open URDFs (Apache-2.0, TetherIA; see `aero_hand_open/NOTICE`)

## Provenance

Was `collection/assets/` and `experimental/assets/`.
