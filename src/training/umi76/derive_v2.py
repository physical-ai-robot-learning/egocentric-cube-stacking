"""REL16-v2 datasets derived from an existing umi76 REL16 dataset by rewriting parquet columns only (dataset stage, node).

[2026-09-28] Why not reconvert: the twin requirement is that arm REL16 labels, state76, video, prompts and episode order are
BIT-IDENTICAL to the parent. Rewriting two action channels (and, for B, appending 18 state dims) guarantees that by
construction and skips the multi-hour video encode. Videos are hard-linked, never re-encoded.

    A  --variant A   action[:, :, 9] / [:, :, 19] := binary LEADER gripper intent, state76 unchanged
    C  --variant C   [2026-09-29] action[:, :, 9|19] := CONTINUOUS leader command g = clip(cmd / 45, 0, 1), 0 = fully CLOSE,
                     1 = fully OPEN (45 = the measured full jaw travel, GRIPPER_COMMAND_SCALE.md; above it the follower is
                     saturated at raw -270). state76 unchanged. Inference binarises (g >= threshold -> OPEN) -- the model
                     learns the common ego/robot semantic "how open", the robot interface alone is binary.
    B  --variant B   A + observation.state 76 -> 94: per arm base-frame TCP [pos3, rot6d] at t, L then R
                     (DIAGNOSTIC ONLY: ego has no robot-base absolute pose; never a shared representation)

Gripper label contract (v2): 1 = OPEN, 0 = CLOSE, from the LEADER command at the same physical times as the arm
targets, t + (k+1) * 3/59.94 s (source 30 Hz, linear interp, then threshold). The follower width used in v1 stalls
at cube contact while the leader squeezes (R150 hold: follower raw -79..-118, leader 0-7), so it carried no grasp
intent. The actuator side is separate: the deploy adapter maps 1 -> cmd OPEN, 0 -> cmd CLOSED from
gripper_contract_v2.json (gripper_smoke_test.py --cube). Leader: 0 = closed (rest, frame 0), open ~50-80, same
sign in R150/R30/center675 (checked 2026-09-28: frame-0 leader 0 / follower 0, corr(follower, leader) -0.8).

Row -> source mapping (converter umi76_to_lerobot.py): sources in provenance order, episodes in select order
(or 0..n-1), src_ep via episodes_json; stored row f of an episode = source frame 2f + 2 (first kept t >= 50.05 ms).

Inputs -> outputs: parent LeRobot dir (+ its source robot datasets, + the UMI zarr for B) -> new LeRobot dir (data rewritten,
videos hard-linked, meta / stats / provenance updated).
usage: derive_v2.py --parent <dir> --out <dir> --variant A|B|C [--rbp <json>] [--remap-root old=new ...]
       derive_v2.py --parent <dir> --selftest       recompute the parent's stats and compare (no writes)
The node chains run it as a flat script (`$PY derive_v2.py ...`); the node-side siblings (derive_v3d / derive_v3L /
derive_cart20, now in archive/training_variants/umi76) import its helpers under their old names (aliases at the end).
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

from ego_paths import NODE_HOME, prepend_sys_path  # machine-specific roots (env-overridable)

UMI_DT_FRAMES = 30.0 * 3.0 / 59.94          # 1.5015 source frames
OPEN_THRESH = 27.0                          # leader cmd >= 27 = opening (humanik rule, video-verified on R150)
CMD_FULL_OPEN = 45.0                        # variant C: measured full jaw travel cmd 0..45
GRIP_IDX = {0: 6, 1: 13}                    # arm -> leader gripper dim of the 14-D source action
REMAP = {}                                  # source root remap (old -> new), filled from --remap-root
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
STAT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
HORIZON = 16                                # action chunk length
REL_DIMS = 20                               # REL16 action width per row (2 arms x [pos3 rot6d g])
ACTION_PER_ARM = 10
GRIP_IN_ARM = 9                             # gripper dim inside an arm's 10
GRIP_DIMS = [9, 19]
STATE76 = 76
STATE94 = 94
SOURCE_ROW_OFFSET = 2                       # stored row f = source frame 2f + 2
SOURCE_ROW_STRIDE = 2
FRAME_EPS = 1e-6                            # last target may sit exactly on the last source frame
BAND_LO, BAND_HI = 10, 40                   # "ambiguous" leader command band reported at the end
SELFTEST_KEYS = ("min", "max", "mean", "std", "q50")
DEFAULT_RBP = NODE_HOME / "umi_bridge/r675_rbp.json"


class DeriveError(RuntimeError):
    """The derivation cannot run (e.g. the output directory already exists)."""


def storageColumn(t, c):
    """column c of table t as one array; HF extension arrays are reduced to their storage (was _col)"""
    tArr = t.column(c).combine_chunks()
    return tArr.storage if isinstance(tArr, pa.ExtensionArray) else tArr


def loadSource(root, cols):
    """{episode: {col: (n, D) float64}} of a LeRobot dataset, rows ordered by frame_index (was load_src)"""
    tOut = {tC: [] for tC in cols}
    tEp = []
    tFr = []
    for tFile in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)):
        tTable = pq.read_table(tFile, columns=list(cols) + ["episode_index", "frame_index"])
        for tC in cols:
            tOut[tC].append(np.asarray(storageColumn(tTable, tC).to_pylist(), np.float64))
        tEp.append(np.asarray(storageColumn(tTable, "episode_index").to_numpy()))
        tFr.append(np.asarray(storageColumn(tTable, "frame_index").to_numpy()))
    tEp = np.concatenate(tEp)
    tFr = np.concatenate(tFr)
    tArr = {tC: np.concatenate(tV) for tC, tV in tOut.items()}
    tBy = {}
    for tE in np.unique(tEp):
        tRows = np.flatnonzero(tEp == tE)
        tRows = tRows[np.argsort(tFr[tRows])]
        assert (tFr[tRows] == np.arange(len(tRows))).all(), f"{root} ep {tE}: frame_index not contiguous"
        tBy[int(tE)] = {tC: tArr[tC][tRows] for tC in cols}
    return tBy


def columnStats(x):
    """LeRobot-style stats over axis 0 of a (N, D) array (action: frames*16 rows) (was stats)."""
    tQ = np.quantile(x, STAT_QUANTILES, axis=0)
    return dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=x.std(0), count=np.array([len(x)]),
                q01=tQ[0], q10=tQ[1], q50=tQ[2], q90=tQ[3], q99=tQ[4])


def readParent(parent):
    """-> (sorted data parquet paths, their tables) (was read_parent)"""
    tFiles = sorted(glob.glob(f"{parent}/data/chunk-*/*.parquet"))
    tTables = [pq.read_table(tF) for tF in tFiles]
    return tFiles, tTables


def rowSources(parent, rbpOverride=None):
    """episode_index -> (source lerobot root, source episode, zarr, zarr episode) from the parent's provenance
    (was row_sources)"""
    tProv = json.load(open(f"{parent}/v4_provenance.json"))
    tMapping, tE = {}, 0
    for tSource in tProv["sources"]:
        tRoot = tSource["lerobot"]
        tSel = tSource.get("select_zarr_episodes")
        if tSel is None:
            tN = json.load(open(f"{REMAP.get(tRoot, tRoot)}/meta/info.json"))["total_episodes"]
            tZeps = list(range(tN))
            tEpMap = None
        else:
            tZeps = [int(tX) for tX in tSel]
            tEpMap = json.load(open(rbpOverride or tSource.get("episodes_json") or str(DEFAULT_RBP)))["episodes"]
        for tZ in tZeps:
            tMapping[tE] = (tRoot, tZ if tEpMap is None else int(tEpMap[tZ]), tSource["zarr"], tZ)
            tE += 1
    return tMapping


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--parent", required=True)
    tAp.add_argument("--out")
    tAp.add_argument("--variant", choices=["A", "B", "C"])
    tAp.add_argument("--selftest", action="store_true")
    tAp.add_argument("--remap-root", action="append", default=[], help="old=new, to run off-node with local copies")
    tAp.add_argument("--rbp", default=None)
    return tAp


def _selftest(tables, stored):
    """recompute the parent's action / state stats and print the max difference to the stored ones"""
    tA = np.concatenate([np.asarray(storageColumn(tT, "action").to_pylist(), np.float64) for tT in tables])
    tS = np.concatenate([np.asarray(storageColumn(tT, "observation.state").to_pylist(), np.float64) for tT in tables])
    for tName, tX in (("action", tA.reshape(-1, tA.shape[-1])), ("observation.state", tS)):
        tMine = columnStats(tX)
        for tK in SELFTEST_KEYS:
            tD = np.abs(np.asarray(stored[tName][tK]) - tMine[tK]).max()
            print(f"selftest {tName:18s} {tK:5s} max|stored-recomputed| {tD:.3e}")
        print(f"selftest {tName} count stored {stored[tName]['count']} recomputed {tMine['count']}")


def _loadTcpZarrs(srcOf, remap):
    """variant B: per zarr, episode starts and both arms' eef pos / rotvec (the poses state76 was computed from)"""
    prepend_sys_path(os.environ.get("UMI_ROOT", str(NODE_HOME / "universal_manipulation_interface")))
    from umi.common.pose_util import mat_to_pose10d
    from scipy.spatial.transform import Rotation as Rot
    import zarr
    # TCP18 comes from the SAME zarr eef poses state76 was computed from (converter _traj/_sample; the query
    # rows sit on the stored grid, so the pose at t is the raw zarr sample, no interpolation). center675_s96
    # carries no observation.ee.tcp_state, and one source for every pool keeps B consistent by construction.
    tZp = {}
    for tPath in sorted({tV[2] for tV in srcOf.values()}):
        tZ = zarr.open(remap.get(tPath, tPath), "r")
        tEnds = np.asarray(tZ["meta"]["episode_ends"][:], int)
        tZp[tPath] = dict(starts=np.concatenate([[0], tEnds[:-1]]),
                          pos=[np.asarray(tZ["data"][f"robot{tR}_eef_pos"][:], np.float64) for tR in (0, 1)],
                          rv=[np.asarray(tZ["data"][f"robot{tR}_eef_rot_axis_angle"][:], np.float64) for tR in (0, 1)])
    return tZp, mat_to_pose10d, Rot


def _linkTree(parent, out):
    """videos: hard links (same inode, zero bytes, bit-identical by construction); meta copied then rewritten"""
    for tP in parent.rglob("*"):
        tRel = tP.relative_to(parent)
        tQ = out / tRel
        if tP.is_dir():
            tQ.mkdir(parents=True, exist_ok=True)
        elif tRel.parts[0] == "videos":
            os.link(tP, tQ)
        elif tRel.parts[0] != "data":
            shutil.copy2(tP, tQ)
    (out / "data").mkdir(exist_ok=True)


def _splice(old, x, dims=None, append=None):
    """stats dict: the parent's values kept for every unchanged dim, `dims` recomputed from x, dims from `append` on added"""
    tMine = columnStats(x.astype(np.float64))
    tOut = {}
    for tK in STAT_KEYS:
        if tK == "count":
            tOut[tK] = list(old[tK])
            continue
        tV = np.asarray(old[tK], np.float64).copy()
        if dims is not None:
            tV[dims] = tMine[tK][dims]
        if append is not None:
            tV = np.concatenate([tV, tMine[tK][append:]])
        tOut[tK] = tV.tolist()
    return tOut


def _rewriteTable(path, table, actions, states, variant, parent, out):
    """write back with the parent's schema; only the replaced columns change type/length"""
    tCols, tFields = [], []
    for tName in table.column_names:
        tField = table.schema.field(tName)
        if tName == "action":
            tArr = pa.array(actions.tolist(), type=storageColumn(table, "action").type)
        elif tName == "observation.state" and variant == "B":
            tField = pa.field(tName, pa.list_(pa.float32(), STATE94))
            tArr = pa.array(states.tolist(), type=tField.type)
        else:
            tCols.append(table.column(tName))
            tFields.append(tField)
            continue
        tCols.append(tArr)
        tFields.append(tField)
    tMeta = dict(table.schema.metadata or {})
    if variant == "B" and b"huggingface" in tMeta:
        tHf = json.loads(tMeta[b"huggingface"])
        tHf["info"]["features"]["observation.state"]["length"] = STATE94
        tMeta[b"huggingface"] = json.dumps(tHf).encode()
    pq.write_table(pa.Table.from_arrays(tCols, schema=pa.schema(tFields, metadata=tMeta)),
                   out / pathlib.Path(path).relative_to(parent))


def _rewriteData(args, srcOf, src, tcp, files, tables, parent, out):
    """rewrite every data file; -> (all actions, all states, per-episode (A, S), ambiguous-band fractions)"""
    tAllA, tAllS, tBand, tPerEp = [], [], [], {}
    for tPath, tTable in zip(files, tables):
        tEp = np.asarray(storageColumn(tTable, "episode_index").to_numpy())
        tFr = np.asarray(storageColumn(tTable, "frame_index").to_numpy())
        tA = np.asarray(storageColumn(tTable, "action").to_pylist(), np.float32)          # (n,16,20)
        tS = np.asarray(storageColumn(tTable, "observation.state").to_pylist(), np.float32)
        tNewS = []
        for tI in range(len(tEp)):
            tRoot, tSe, tZpath, tZep = srcOf[int(tEp[tI])]
            tLead = src[tRoot][tSe]["action"]
            tN = len(tLead)
            tI0 = SOURCE_ROW_STRIDE * int(tFr[tI]) + SOURCE_ROW_OFFSET
            tTk = tI0 + UMI_DT_FRAMES * (np.arange(HORIZON) + 1)
            assert tTk[-1] <= tN - 1 + FRAME_EPS, \
                f"ep {tEp[tI]} f {tFr[tI]}: target beyond source episode ({tTk[-1]:.2f} > {tN-1})"
            for tR, tGi in GRIP_IDX.items():
                tV = np.interp(tTk, np.arange(tN), tLead[:, tGi])
                tLabel = np.clip(tV / CMD_FULL_OPEN, 0.0, 1.0) if args.variant == "C" else (tV >= OPEN_THRESH)
                tA[tI, :, tR * ACTION_PER_ARM + GRIP_IN_ARM] = tLabel.astype(np.float32)
                tBand.append(float(((tV > BAND_LO) & (tV < BAND_HI)).mean()))
            if args.variant == "B":
                tZp, tMatToPose10d, tRot = tcp
                tZi = int(tZp[tZpath]["starts"][tZep]) + tI0
                tExt = []
                for tR in (0, 1):
                    tM = np.eye(4)
                    tM[:3, :3] = tRot.from_rotvec(tZp[tZpath]["rv"][tR][tZi]).as_matrix()
                    tM[:3, 3] = tZp[tZpath]["pos"][tR][tZi]
                    tExt.append(tMatToPose10d(tM[None])[0])
                tNewS.append(np.concatenate([tS[tI], *tExt]).astype(np.float32))
        if args.variant == "B":
            tS = np.stack(tNewS)
        tAllA.append(tA)
        tAllS.append(tS)
        for tE in np.unique(tEp):
            tRows = tEp == tE
            tPerEp[int(tE)] = (tA[tRows], tS[tRows])
        _rewriteTable(tPath, tTable, tA, tS, args.variant, parent, out)
    return tAllA, tAllS, tPerEp, tBand


def _writeMeta(args, parent, out, actions, states, perEp):
    # stats: the parent's stored values are kept BIT-FOR-BIT for every unchanged dim (arm action dims, state76);
    # only gripper dims 9/19 (and B's 18 appended state dims) are computed here, and `count` stays the parent's.
    tSt = json.load(open(parent / "meta/stats.json"))
    tSt["action"] = _splice(tSt["action"], actions.reshape(-1, REL_DIMS), dims=GRIP_DIMS)
    if args.variant == "B":
        tSt["observation.state"] = _splice(tSt["observation.state"], states, append=STATE76)
    json.dump(tSt, open(out / "meta/stats.json", "w"), indent=4)
    tInfo = json.load(open(parent / "meta/info.json"))
    if args.variant == "B":
        tInfo["features"]["observation.state"]["shape"] = [STATE94]
    json.dump(tInfo, open(out / "meta/info.json", "w"), indent=4)
    for tPath in sorted(glob.glob(str(out / "meta/episodes/chunk-*/*.parquet"))):
        tTable = pq.read_table(tPath)
        tCols = {tC: tTable.column(tC) for tC in tTable.column_names}
        tEps = np.asarray(storageColumn(tTable, "episode_index").to_numpy())
        for tKey in ("action",) + (("observation.state",) if args.variant == "B" else ()):
            tOlds = [{tK: storageColumn(tTable, f"stats/{tKey}/{tK}")[tJ].as_py() for tK in STAT_KEYS} for tJ in range(len(tEps))]
            tPer = [_splice(tOlds[tJ], perEp[int(tE)][0].reshape(-1, REL_DIMS), dims=GRIP_DIMS) if tKey == "action"
                    else _splice(tOlds[tJ], perEp[int(tE)][1], append=STATE76) for tJ, tE in enumerate(tEps)]
            for tK in STAT_KEYS:
                tCols[f"stats/{tKey}/{tK}"] = pa.array([tP[tK] for tP in tPer], type=tTable.schema.field(f"stats/{tKey}/{tK}").type)
        pq.write_table(pa.table(tCols).replace_schema_metadata(tTable.schema.metadata), tPath)
    tProv = json.load(open(parent / "v4_provenance.json"))
    if args.variant == "C":
        tLabel = "continuous leader command g = clip(cmd/45, 0, 1), 0 = CLOSE, 1 = OPEN, at t+(k+1)*50.05ms"
    else:
        tLabel = "binary leader intent, 1 = OPEN, 0 = CLOSE, leader cmd >= 27 at t+(k+1)*50.05ms"
    tProv.update(parent=str(parent), derived_by="derive_v2.py", variant=args.variant, gripper_label=tLabel,
                 state_dim=STATE94 if args.variant == "B" else STATE76,
                 state_extra=("[L pos3 rot6d, R pos3 rot6d] base-frame TCP at t (DIAGNOSTIC)" if args.variant == "B" else None))
    json.dump(tProv, open(out / "v4_provenance.json", "w"), indent=2)


def derive(args) -> None:
    """the whole derivation for parsed CLI args (selftest or write)"""
    tParent = pathlib.Path(args.parent)
    tFiles, tTables = readParent(tParent)
    tStored = json.load(open(tParent / "meta/stats.json"))
    if args.selftest:
        _selftest(tTables, tStored)
        return
    tRemap = dict(tX.split("=", 1) for tX in args.remap_root)
    REMAP.update(tRemap)
    tSrcOf = rowSources(tParent, args.rbp)
    tRoots = sorted({tV[0] for tV in tSrcOf.values()})
    tSrc = {tR: loadSource(tRemap.get(tR, tR), ["action"]) for tR in tRoots}
    tTcp = _loadTcpZarrs(tSrcOf, tRemap) if args.variant == "B" else None
    tOut = pathlib.Path(args.out)
    if tOut.exists():
        raise DeriveError(f"{tOut} exists, refusing to overwrite")
    tOut.mkdir(parents=True)
    _linkTree(tParent, tOut)
    tAllA, tAllS, tPerEp, tBand = _rewriteData(args, tSrcOf, tSrc, tTcp, tFiles, tTables, tParent, tOut)
    tA = np.concatenate(tAllA)
    tS = np.concatenate(tAllS)
    _writeMeta(args, tParent, tOut, tA, tS, tPerEp)
    tOpen = tA[:, :, GRIP_DIMS]
    print(f"done {tOut}: rows {len(tA)}  open frac L {tOpen[..., 0].mean():.3f} R {tOpen[..., 1].mean():.3f}  "
          f"leader in ambiguous 10-40 band {np.mean(tBand):.4f}")


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        derive(tArgs)
    except DeriveError as tErr:
        print(tErr, file=sys.stderr)
        return 1
    return 0


# old names, imported by the node-side siblings (derive_relcart20 here; derive_v3d / derive_v3L / derive_cart20 in archive/)
_col = storageColumn
load_src = loadSource
stats = columnStats
read_parent = readParent
row_sources = rowSources

if __name__ == "__main__":
    sys.exit(main())
