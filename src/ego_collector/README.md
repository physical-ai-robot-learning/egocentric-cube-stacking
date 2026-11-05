# ego_collector

Camera capture layer (`ego_collector.camera`: UVC / AVFoundation capture with fixed exposure, Orbbec RGB-D capture) used by
`handumi_collector.devices.camera`.

**Status:** current (camera layer only). The rest of the V0 AprilTag-glove collector (`record → process-tags → process-hands →
generate-actions → qa`, superseded by `handumi_collector`) is in [`archive/shared_unused/ego_collector/`](../../archive/shared_unused/ego_collector/).

## Inputs → Outputs

In: a camera index / Orbbec device and a capture config. Out: timestamped frames to the collector's video writer.

## Entry points

Library only (imported by `handumi_collector`).

## Layout

- `camera/capture.py` — UVC / AVFoundation capture (`CaptureConfig`)
- `camera/orbbec_capture.py` — Orbbec RGB-D capture (needs `pyorbbecsdk`, installed by hand)

## Provenance

Shared copy (handumi-common v0.3.0), files unchanged, not restyled; trimmed 2026-11-05. Sources:
[`docs/provenance/experimental_MANIFEST.md`](../../docs/provenance/experimental_MANIFEST.md).
