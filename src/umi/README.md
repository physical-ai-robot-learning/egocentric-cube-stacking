# umi

Stand-in for `umi.common.pose_util` of the Universal Manipulation Interface (UMI, MIT licence): the pose conversions
(`pose_to_mat`, `mat_to_rot6d`, `rot6d_to_mat`, `mat_to_pose10d`, `pose10d_to_mat`) that the CART20 contract and the deploy code use, with the same API names.

**Status:** current

## Inputs → Outputs

Numpy arrays in, numpy arrays out (positions, rotation vectors, 4×4 matrices, rot6d = first two **rows** of R).

## Entry points

Library only: `from umi.common.pose_util import mat_to_rot6d`.

## Layout

- `common/pose_util.py` — the functions (API names = upstream UMI; internals in OA style)
- `tests/` — characterization tests

## Provenance

Upstream <https://github.com/real-stanford/universal_manipulation_interface> (MIT, `upstream/LICENSE-umi-mit.txt`); source and
md5 record in [`upstream/README.md`](../../upstream/README.md). A wheel of this repository ships a top-level `umi` package, which would
shadow a separately installed UMI.
