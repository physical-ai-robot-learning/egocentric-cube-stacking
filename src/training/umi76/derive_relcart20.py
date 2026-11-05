"""RELCART20 state (primary ego->robot state candidate) from REL16-v3d: the ONLY change is observation.state (76 -> 20)
(dataset stage, node).

[2026-09-29]
    state (20) = [ L_rel pos3 (m) | L_rel rot6d | R_rel pos3 (m) | R_rel rot6d | g_L | g_R ]
        T_rel(t) = inv(T_anchor) @ T(t), per arm, T = rebot_fk_torch.tcp(aux.q_t) in the robot base frame (== dataset TCP)
        anchor   = the episode's FIRST ROW (frame_index 0). Segment = one LeRobot episode (one teleop recording of the whole
                   3-stack task); v3d has no finer segmentation and none is added here.
        rot6d    = first two ROWS of R (umi pose_util.mat_to_rot6d, the REL16 action convention); identity = [1,0,0,0,1,0]
        g        = follower jaw openness at t = width / W_OPEN, W_OPEN = 270*0.05/118 m, clip [0,1]; 0 CLOSED, 1 OPEN
                   (the AbsCart20 converter, unchanged)
Action (16, 32) REL16 + continuous grip + dq12, aux.q_t, videos (hard-linked), prompts, episode order: bit-identical to the parent.
Gates: QA1 anchor rows == identity (pos < 1e-6 m, rot < 1e-4 rad); QA2 = relcart20_qa.py (independent recomputation, Mac deploy FK);
QA3 every non-state column byte-identical to the parent.
Inputs -> outputs: v3d LeRobot dir (with aux.q_t) -> new LeRobot dir with state20 (data, stats, info, episode stats, provenance).
usage: derive_relcart20.py --parent <v3d dir> --out <dir>      (node, holobrain env; also python -m training.umi76.derive_relcart20)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# DV's old helper names (_col, read_parent, stats) are used on purpose: on the node this file sits next to the node's own
# derive_v2.py, which has only those names.
try:
    from . import derive_v2 as DV
except ImportError:  # run as a flat script on the node (its folder is sys.path[0])
    import derive_v2 as DV

W_OPEN = 270.0 * 0.05 / 118.0     # m, follower jaw width at full open (0.11441 m)
STATE_DIM = 20
POSE9 = 9                         # pos3 + rot6d per arm
GRIP_STATE76 = [37, 75]           # follower width of L / R in state76
QA1_POS_M = 1e-6                  # anchor rows must be identity to these tolerances
QA1_ROT = 1e-4
DIR_MODE = 0o755
FILE_MODE = 0o644


class DeriveError(RuntimeError):
    """The derivation cannot run (e.g. the output directory already exists)."""


def _loadFkModule():
    """the node / Mac rebot_fk_torch on NODE_HOME/c8old/xvla when present (unchanged behaviour), else the repository copy"""
    from ego_paths import NODE_HOME, prepend_sys_path  # machine-specific roots (env-overridable)
    prepend_sys_path(str(NODE_HOME / "c8old/xvla"))
    try:
        import rebot_fk_torch
    except ImportError:  # clean checkout: the repository copy, src/training/robot/rebot_fk_torch.py
        from training.robot import rebot_fk_torch
    return rebot_fk_torch


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--parent", required=True)
    tAp.add_argument("--out", required=True)
    return tAp


def _absoluteTcp(tables, fk):
    """pass 1: absolute TCP of every row, and the anchor (frame_index 0) of every episode -- episodes may span parquet files"""
    tAbsT, tRows = {}, []
    for tFi, tTable in enumerate(tables):
        tEp = np.asarray(DV._col(tTable, "episode_index").to_numpy())
        tFr = np.asarray(DV._col(tTable, "frame_index").to_numpy())
        with torch.no_grad():
            tQ = torch.tensor(np.asarray(DV._col(tTable, "aux.q_t").to_pylist(), np.float64))
            tAbsT[tFi] = fk.tcp(tQ).numpy()                                                  # (n, 2, 4, 4)
        tRows.append((tEp, tFr))
    tAnchor = {}
    for tFi, (tEp, tFr) in enumerate(tRows):
        for tJ in np.flatnonzero(tFr == 0):
            assert int(tEp[tJ]) not in tAnchor, f"episode {tEp[tJ]} has two frame_index-0 rows"
            tAnchor[int(tEp[tJ])] = tAbsT[tFi][tJ]
    assert len(tAnchor) == len({int(tE) for tEp, _ in tRows for tE in tEp}), "an episode has no frame_index-0 row"
    return tAbsT, tRows, tAnchor


def _linkTree(parent, out):
    """videos hard-linked, meta copied (modes normalised: the parent is frozen read-only), data rewritten later"""
    for tP in parent.rglob("*"):
        tRel = tP.relative_to(parent)
        tQ = out / tRel
        if tP.is_dir():
            tQ.mkdir(parents=True, exist_ok=True)
            os.chmod(tQ, DIR_MODE)
        elif tRel.parts[0] == "videos":
            os.link(tP, tQ)
        elif tRel.parts[0] != "data":
            shutil.copy2(tP, tQ)
            os.chmod(tQ, FILE_MODE)
    (out / "data").mkdir(exist_ok=True)


def _state20(table, rows, absT, anchor):
    """RELCART20 state of one table -> (S (n, 20) float32, anchor-row |pos| max or None, anchor-row |R - I| max or None)"""
    tEp, tFr = rows
    tS76 = np.asarray(DV._col(table, "observation.state").to_pylist(), np.float64)
    tA0 = np.stack([anchor[int(tE)] for tE in tEp])                                          # (n, 2, 4, 4)
    tRel = np.einsum("naij,najk->naik", np.linalg.inv(tA0), absT)
    tS = np.zeros((len(tEp), STATE_DIM), np.float32)
    for tR in (0, 1):
        tS[:, tR * POSE9:tR * POSE9 + 3] = tRel[:, tR, :3, 3]
        tS[:, tR * POSE9 + 3:tR * POSE9 + POSE9] = tRel[:, tR, :2, :3].reshape(-1, 6)
    tS[:, 2 * POSE9:STATE_DIM] = np.clip(tS76[:, GRIP_STATE76] / W_OPEN, 0.0, 1.0)
    tZero = tFr == 0
    if not tZero.any():
        return tS, None, None
    return tS, np.abs(tRel[tZero][..., :3, 3]).max(), np.abs(tRel[tZero][..., :3, :3] - np.eye(3)).max()


def _writeTable(path, table, state, parent, out):
    tCols, tFields = [], []
    for tName in table.column_names:
        if tName == "observation.state":
            tType = pa.list_(pa.float32(), STATE_DIM)
            tCols.append(pa.array(state.tolist(), type=tType))
            tFields.append(pa.field(tName, tType))
        else:
            tCols.append(table.column(tName))
            tFields.append(table.schema.field(tName))
    tMeta = dict(table.schema.metadata or {})
    if b"huggingface" in tMeta:
        tHf = json.loads(tMeta[b"huggingface"])
        tHf["info"]["features"]["observation.state"] = {"feature": {"dtype": "float32", "_type": "Value"}, "length": STATE_DIM,
                                                        "_type": "List"}
        tMeta[b"huggingface"] = json.dumps(tHf).encode()
    pq.write_table(pa.Table.from_arrays(tCols, schema=pa.schema(tFields, metadata=tMeta)), out / pathlib.Path(path).relative_to(parent))


def _writeMeta(parent, out, state, perEp, qa1Pos, qa1Rot):
    tSt = json.load(open(parent / "meta/stats.json"))
    tNs = DV.stats(state.astype(np.float64))
    tSt["observation.state"] = {tK: ([float(tX) for tX in tNs[tK]] if tK != "count" else [int(len(state))]) for tK in DV.STAT_KEYS}
    json.dump(tSt, open(out / "meta/stats.json", "w"), indent=4)
    tInfo = json.load(open(parent / "meta/info.json"))
    tInfo["features"]["observation.state"] = {"dtype": "float32", "shape": [STATE_DIM], "names": None}
    json.dump(tInfo, open(out / "meta/info.json", "w"), indent=4)
    for tPath in sorted(glob.glob(str(out / "meta/episodes/chunk-*/*.parquet"))):
        tTable = pq.read_table(tPath)
        tCols = {tC: tTable.column(tC) for tC in tTable.column_names}
        tEps = np.asarray(DV._col(tTable, "episode_index").to_numpy())
        for tK in DV.STAT_KEYS:
            tVals = [(list(DV.stats(perEp[int(tE)].astype(np.float64))[tK]) if tK != "count" else [len(perEp[int(tE)])])
                     for tE in tEps]
            tCols[f"stats/observation.state/{tK}"] = pa.array(tVals, type=tTable.schema.field(f"stats/observation.state/{tK}").type)
        pq.write_table(pa.table(tCols).replace_schema_metadata(tTable.schema.metadata), tPath)
    tProv = json.load(open(parent / "v4_provenance.json"))
    tProv.update(parent=str(parent), derived_by="derive_relcart20.py", variant="RELCART20",
                 state_layout="[L_rel pos3 rot6d(rows) | R_rel pos3 rot6d(rows) | g_L g_R], T_rel = inv(T_anchor) T_t, "
                              "anchor = episode frame_index 0",
                 anchor="first row (frame_index 0) of each LeRobot episode; segment = episode",
                 gripper="follower width / 0.11441 m, clip [0,1], 1 OPEN",
                 qa1=dict(max_pos_m=float(qa1Pos), max_rot=float(qa1Rot)))
    json.dump(tProv, open(out / "v4_provenance.json", "w"), indent=2)


def derive(parent: pathlib.Path, out: pathlib.Path) -> None:
    if out.exists():
        raise DeriveError(f"{out} exists, refusing to overwrite")
    tFiles, tTables = DV.read_parent(parent)
    tFk = _loadFkModule().ReBotFKTorch(dtype=torch.float64)
    tAbsT, tRows, tAnchor = _absoluteTcp(tTables, tFk)
    out.mkdir(parents=True)
    _linkTree(parent, out)
    tAllS, tPerEp, tQa1Pos, tQa1Rot = [], {}, [], []
    for tFi, (tPath, tTable) in enumerate(zip(tFiles, tTables)):
        tS, tPos, tRot = _state20(tTable, tRows[tFi], tAbsT[tFi], tAnchor)
        if tPos is not None:
            tQa1Pos.append(tPos)
            tQa1Rot.append(tRot)
        tAllS.append(tS)
        tEp = tRows[tFi][0]
        for tE in np.unique(tEp):
            tPerEp.setdefault(int(tE), []).append(tS[tEp == tE])
        _writeTable(tPath, tTable, tS, parent, out)
    tPerEp = {tE: np.concatenate(tV) for tE, tV in tPerEp.items()}
    tS = np.concatenate(tAllS)
    print(f"QA1 anchor identity over {len(tAnchor)} episodes: max |pos| {max(tQa1Pos):.2e} m, max |R - I| {max(tQa1Rot):.2e}")
    assert max(tQa1Pos) < QA1_POS_M and max(tQa1Rot) < QA1_ROT, "QA1 FAILED"
    for tPath in tFiles:                                                                     # QA3 (node side)
        tParentTable = pq.read_table(tPath)
        tOutTable = pq.read_table(out / pathlib.Path(tPath).relative_to(parent))
        for tName in tParentTable.column_names:
            if tName != "observation.state":
                assert tParentTable.column(tName).equals(tOutTable.column(tName)), f"QA3: {tName} changed in {tPath}"
    print("QA3 non-state columns byte-identical to the parent: PASS")
    _writeMeta(parent, out, tS, tPerEp, max(tQa1Pos), max(tQa1Rot))
    print(f"done {out}: rows {len(tS)} state (20,)")


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        derive(pathlib.Path(tArgs.parent), pathlib.Path(tArgs.out))
    except DeriveError as tErr:
        print(tErr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
