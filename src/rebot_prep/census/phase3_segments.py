"""[2026-09-24] C8 Phase 3: CPU postprocess of the 5080 MASt3R-SLAM runs -> segment census + pseudo q, per episode (Gen-1
census stage; was c8_phase3.py).

In:  ~/c8/c8_domain_manifest.json, ~/c8/grip/grip_census.json + grip/<tag>.npz, ~/c8/census_pre.json, ~/c8/runs_m3/<tag>/ss1.csv
     (env C8_RUNS), ~/c8/export[_early]/<tag>/, the raw Hpilot sensors.mcap; ~/umi_bridge/trackA_mast3r_pose_v1 (frozen
     cp6_scale.py text + qa / camera_tcp data), ~/umi_bridge/track_c (pseudo_joint_pipeline, state32, data/r150_q_tcp.npz)
Out: ~/c8/stage_p3/ (staged, re-indexed csv + symlinks), ~/c8/phase3/<episode>.npz (per usable-or-not segment: frames, status,
     q, state32, gates) and ~/c8/phase3_census.json (phase3_census_partial.json when episodes are given)
Usage: python -m rebot_prep.census.phase3_segments [episode ...]   (default: every episode of the manifest; env ~/xvla-mac)

Per side (rules frozen in contract §21, applied in this order, all named):
  nonfinite_pose_invalid_v1   valid_pose = (is_lost == false) AND all_finite(x, y, z, q)
  video_to_cori_index_v1      MASt3R frame_idx is the VIDEO frame; cori_idx = video_frame - offset (grip_census frame mapping)
  imu_vi (C-P6 frozen)        cp6_scale.imu_vi EXACTLY (exec of the frozen source), SG window 15, no bias
  metric TCP (C-P5)           T_world_tcp = (T_WC with t * s) @ X, X = handumi_camera_tcp_v2
  gripper                     ~/c8/grip/<tag>.npz width_m (Track A §16), CORI-indexed
Per episode (LEFT wrist = master clock, V3 pairing): right / head frame = nearest capture_ns, |dt| < 20 ms.
  row_ok[i] = valid_L[i] AND valid_R[jR[i]] AND dtR < 20 ms AND dtH < 20 ms
  Segments = non-overlapping 65 consecutive row_ok frames, tail dropped. SEGMENT LENGTH (explicit, never re-derive):
    65 rows at 30 fps INCLUDING the anchor row  ==  33 rows at 15 Hz (even frames 0, 2, ..., 64)  ==  anchor + 32 future steps.
  Every invalid row (lost, NaN, unsynced) is a HARD BREAK -- no segment spans it.
  Each segment re-anchored at r150_active_median, continuity IK (C4 frozen IKCfg) at 30 fps.
Segment funnel (first fail, mutually exclusive; fixed before any Phase-3 result was seen):
  A_candidates     65-frame run as above (sync <= 20 ms, video_to_cori_index_v1, valid_pose, lost/NaN hard break)
  B_metric         both sides: imu_vi returned, s finite > 0, |g| in [8.81, 10.81], AND imu_vi_degenerate_guard_v1:
                   s in [0.1 x min, 10 x max] of the ArUco GT scales of the 28-side benchmark = [0.0305, 9.18] (added
                   2026-09-24 after the dry run found s = 1.4e-4 with |g| 9.71 on a coverage-0.40 side: |g| is solved
                   independently of s, so the |g| band cannot see a degenerate scale; the bound comes from the benchmark
                   GT only, never from C8 values) -> metric TCP
  C_workspace      BOTH arms >= 95 % rows inside the R150 workspace (pos NN < 50 mm AND ori NN-50 < 20 deg)
  D_IK             BOTH arms: every row status in P.SUCCESS (continuity IK)
  E_pseudo_q       state32 (s32) all finite
  Every stage is judged PER ARM (cumulative: an arm credited at a stage passed all earlier ones). PRIMARY (bimanual) =
  both arms pass; SECONDARY = >= 1 arm passes B..E (the V3 per-arm mask rule). A side with a degenerate / invalid scale
  therefore drops only its own arm from the secondary yield. No residual threshold is used (resid_rms is reported only).
  Thresholds FROZEN 2026-09-24 (incl. imu_vi_degenerate_guard_v1 [0.0305, 9.18], a catastrophic-degeneracy guard, NOT a
  scale-quality gate).
Episode first-fail (before segments): sensor/gripper/sync from census_pre.json, then mast3r_missing, then no_segment.

Parallel: the staging is serial; episodes run in a ProcessPoolExecutor whose workers call init() (the model context lives in the
module global CONTEXT, the inputs in INPUTS, the NaN-invalidated rows of the staging in NF_IDX).
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from ego_paths import DATA_ROOT, EGO_COLLECTOR_HOME, UMI_BRIDGE, prepend_sys_path

C8 = DATA_ROOT
TA = UMI_BRIDGE / "trackA_mast3r_pose_v1"
TC = UMI_BRIDGE / "track_c"
RUNS = pathlib.Path(os.environ.get("C8_RUNS", C8 / "runs_m3"))
STAGE = C8 / "stage_p3"
OUT = C8 / "phase3"
RAW = EGO_COLLECTOR_HOME / "datasets/human_handumi_raw/Hpilot"
SEG, DT_MS, G_BAND = 65, 20.0, (8.81, 10.81)
S_GUARD = (0.1 * 0.3049, 10 * 0.9178)                                     # imu_vi_degenerate_guard_v1 (benchmark s_gt min / max)
W_VI, BIAS_VI = 15, False
WS_MIN = 0.95                     # C_workspace: share of rows inside the R150 workspace, both arms
WS_POS_M, WS_ORI_DEG, WS_K = 0.05, 20, 50   # == build_v3_dataset.ws_in: position NN, orientation among the 50 nearest
REF_SAMPLES, REF_SEED = 6000, 0   # R150 frames of the workspace reference
STATE_PER_ARM = 16
NS_PER_MS = 1e6
FROZEN_END = "tags = ["            # the frozen cp6 text is exec'd up to its FIT/EVAL section
STAGES = ("B_metric", "C_workspace", "D_IK", "E_pseudo_q")
FUNNEL = ["A_candidates", "B_metric", "C_workspace", "D_IK", "E_pseudo_q"]
FIRST_FAIL_RANK = {"B_metric": 1, "C_workspace": 2, "D_IK": 3, "E_pseudo_q": 4}
RULES = ["nonfinite_pose_invalid_v1", "video_to_cori_index_v1", "imu_vi W15 no-bias", "camera_tcp_v2", "left-master |dt|<20ms",
         "65-frame hard-break segments", "r150_active_median re-anchor", "C4 IKCfg", "ws>=95% both arms"]

NF_IDX = {}                                                                   # tag -> CORI indices made invalid by the NaN rule


@dataclass(frozen=True)
class Inputs:
    """The census inputs: domain manifest, grip census sides, census_pre records."""
    mManifest: dict
    mGrip: dict
    mPre: dict


@dataclass(frozen=True)
class Context:
    """Frozen models: the exec'd C-P6 namespace, pseudo_joint_pipeline (P), its IK and retarget, the R150 workspace trees."""
    mCp6: dict
    mP: object
    mIk: object
    mRt: object
    mTree: dict
    mWr: dict
    mBuildState32: object


INPUTS: Inputs | None = None
CONTEXT: Context | None = None


def loadInputs() -> Inputs:
    global INPUTS
    INPUTS = Inputs(json.load(open(C8 / "c8_domain_manifest.json")), json.load(open(C8 / "grip/grip_census.json"))["sides"],
                    json.load(open(C8 / "census_pre.json")))
    return INPUTS


def _pipelineModules():
    """(pseudo_joint_pipeline, build_state32): the ~/umi_bridge/track_c copies first, else the repository's"""
    prepend_sys_path(str(TC))
    try:
        import pseudo_joint_pipeline as P
        from state32 import build_state32
    except ImportError:  # clean checkout: the repository copy
        from rebot_prep.pseudo_joint import pseudo_joint_pipeline as P
        from rebot_prep.pseudo_joint.state32 import build_state32
    return P, build_state32


def init() -> Context:
    """stage the C-P6 inputs, exec the frozen cp6 text, build IK / retarget and the R150 workspace reference (CONTEXT)"""
    global CONTEXT
    (STAGE / "qa").mkdir(parents=True, exist_ok=True)
    for tSrc, tDst in ((TA / "data/qa/qa_all.json", STAGE / "qa/qa_all.json"),
                       (TA / "data/handumi_camera_tcp_v2.yaml", STAGE / "handumi_camera_tcp_v2.yaml")):
        if not tDst.exists():
            try:
                tDst.symlink_to(tSrc)
            except FileExistsError:
                pass
    os.environ["M3_BASE"] = str(STAGE)
    os.environ["M3_VARIANT"] = "ss1"
    tCp6 = {}
    tText = open(TA / "cp6_scale.py").read()
    exec(tText[:tText.index(FROZEN_END)], tCp6)
    P, tBuildState32 = _pipelineModules()
    tCfg = P.IKCfg()
    tIk = P.ContinuityIK(tCfg)
    tRt = P.Retarget(tIk.kin, P.ANCHORS[tCfg.anchor])
    tD150 = np.load(TC / "data/r150_q_tcp.npz")
    tIdx = np.random.default_rng(REF_SEED).choice(len(tD150["J"]), REF_SAMPLES, replace=False)
    tRef = {0: [], 1: []}
    for tI in tIdx:
        for tArm, tP7 in enumerate(tIk.kin.fk_pose7(np.radians(np.r_[tD150["J"][tI, :6], tD150["J"][tI, 7:13]]))):
            tRef[tArm].append(tP7)
    tRef = {tA: np.array(tV) for tA, tV in tRef.items()}
    CONTEXT = Context(tCp6, P, tIk, tRt, {tA: cKDTree(tRef[tA][:, :3]) for tA in (0, 1)},
                      {tA: Rot.from_quat(tRef[tA][:, 3:]).as_matrix() for tA in (0, 1)}, tBuildState32)
    return CONTEXT


def exportDir(tag: str) -> pathlib.Path:
    return C8 / ("export" if INPUTS.mManifest[tag]["domain"] == "main_calibrated" else "export_early") / tag


def stageSide(tag: str) -> tuple[int, int]:
    """nonfinite_pose_invalid_v1 + video_to_cori_index_v1 into STAGE/runs/<tag>/ss1.csv; returns (n_nonfinite, offset)
    (was stage_side)."""
    tA, tB = tag.split("_", 1)
    tDst = STAGE / "val" / tA / tB
    tDst.parent.mkdir(parents=True, exist_ok=True)
    if not tDst.exists():
        tDst.symlink_to(exportDir(tag))
    tSrc = RUNS / tag / "ss1.csv"
    tOut = STAGE / "runs" / tag / "ss1.csv"
    tOut.parent.mkdir(parents=True, exist_ok=True)
    tOff = INPUTS.mGrip[tag]["offset"]
    tLines = open(tSrc).read().splitlines()
    tHdr = tLines[0].split(",")
    tFi, tIl = tHdr.index("frame_idx"), tHdr.index("is_lost")
    tPoseCols = [tHdr.index(tK) for tK in ("x", "y", "z", "q_x", "q_y", "q_z", "q_w")]
    tRows = [tLines[0]]
    tNonFinite = 0
    NF_IDX[tag] = []
    for tLine in tLines[1:]:
        tCells = tLine.split(",")
        if tCells[tIl] == "false" and not all(np.isfinite(float(tCells[tK])) for tK in tPoseCols):
            tCells[tIl] = "true"
            tNonFinite += 1
            NF_IDX[tag].append(int(tCells[tFi]) - tOff)
        tJ = int(tCells[tFi]) - tOff
        if tJ >= 0:
            tCells[tFi] = str(tJ)
            tRows.append(",".join(tCells))
    tOut.write_text("\n".join(tRows) + "\n")
    return tNonFinite, tOff


def wsIn(T: np.ndarray, arm: int) -> np.ndarray:
    """inside the R150 workspace per row (== build_v3_dataset.ws_in; was ws_in)"""
    tDp, tIi = CONTEXT.mTree[arm].query(T[:, :3, 3], k=WS_K)
    tWr = CONTEXT.mWr[arm]
    tDo = np.array([np.degrees(np.linalg.norm(Rot.from_matrix(tWr[tIi[tT]].transpose(0, 2, 1) @ T[tT, :3, :3]).as_rotvec(),
                                              axis=1)).min() for tT in range(len(T))])
    return (tDp[:, 0] < WS_POS_M) & (tDo < WS_ORI_DEG)


def frameMeta(episodeDir: pathlib.Path) -> dict:
    """{stream: sorted [(video_frame, capture_ns)]} of the three cameras (was frame_meta)"""
    from mcap.reader import make_reader
    tFrames = {"left_wrist": [], "right_wrist": [], "head": []}
    with open(episodeDir / "sensors.mcap", "rb") as tFh:
        for _, tChannel, tMsg in make_reader(tFh).iter_messages():
            if tChannel.topic.endswith("/frame_meta"):
                tStream = tChannel.topic.split("/")[1]
                if tStream in tFrames:
                    tData = json.loads(tMsg.data)
                    tFrames[tStream].append((int(tData["video_frame"]), int(tData["capture_ns"])))
    return {tK: sorted(tV) for tK, tV in tFrames.items()}


def nearest(cap: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """index into cap of the nearest capture time for every ref time (ties -> the later one)"""
    tJ = np.clip(np.searchsorted(cap, ref), 1, len(cap) - 1)
    return np.where(np.abs(cap[tJ - 1] - ref) < np.abs(cap[tJ] - ref), tJ - 1, tJ)


def sideData(tag: str) -> dict:
    """stage the side again (stageSide fills NF_IDX in this process), then its metric TCP, validity, scale and width (was side)"""
    tCp6 = CONTEXT.mCp6
    tNonFinite, tOff = stageSide(tag)
    tN = len(json.load(open(exportDir(tag) / "imu_data.json"))["1"]["streams"]["CORI"]["samples"])
    tT, tOk = tCp6["load_traj"](STAGE / "runs" / tag / "ss1.csv", tN)             # CORI-indexed, is_lost rows (incl. NaN) invalid
    tVi = tCp6["imu_vi"](tag, W_VI, BIAS_VI)
    tS = tVi["s"] if tVi else float("nan")
    tScaleValid = bool(tVi and np.isfinite(tS) and tS > 0 and G_BAND[0] <= tVi["g_norm"] <= G_BAND[1]
                       and S_GUARD[0] <= tS <= S_GUARD[1])
    tTm = tT.copy()
    tTm[:, :3, 3] *= (tS if np.isfinite(tS) else 1.0)
    tTt = tTm @ tCp6["X"]
    tTt[~tOk] = np.eye(4)
    tGrip = np.load(C8 / "grip" / f"{tag}.npz")
    assert len(tGrip["width_m"]) == tN and tGrip["video_frame"][0] == tOff, tag
    tNfMask = np.zeros(tN, bool)
    tNfMask[[tJ for tJ in NF_IDX[tag] if 0 <= tJ < tN]] = True
    return dict(T_tcp=tTt, ok=tOk, n=tN, off=tOff, s=tS, scale_valid=tScaleValid, imu_vi=tVi,
                nonfinite_pose_rows=tNonFinite, nonfinite_mask=tNfMask, width=tGrip["width_m"])


def segmentRecord(start: int, TL: np.ndarray, TR: np.ndarray, widthL: np.ndarray, widthR: np.ndarray, metricOk: list[bool],
                  store: dict) -> dict:
    """one 65-row segment through the B..E funnel; appends its arrays to store"""
    tCtx = CONTEXT
    P = tCtx.mP
    tRec = dict(start=int(start))
    if not any(metricOk):
        tRec.update(arm_pass={tK: [False, False] for tK in STAGES}, first_fail="B_metric",
                    arm_first_fail=["B_metric", "B_metric"], one_arm_usable=False)
        return tRec
    tTgL, tTgR = tCtx.mRt.targets(TL, 0), tCtx.mRt.targets(TR, 1)
    tWs = [float(wsIn(tTgL, 0).mean()), float(wsIn(tTgR, 1).mean())]
    tRes = tCtx.mIk.solve(tTgL, tTgR)
    tQn = np.where(np.isfinite(tRes["q"]), tRes["q"], tRes["q_last_good"])
    tS32, _ = tCtx.mBuildState32([TL @ P.F[None], TR @ P.F[None]], [widthL.astype(np.float32), widthR.astype(np.float32)],
                                 [tQn[:, :6], tQn[:, 6:]])
    # per-arm stage flags, cumulative (an arm that failed B is not credited later); E = that arm's 16 state32 columns finite
    tC = [tWs[tA] >= WS_MIN for tA in (0, 1)]
    tD = [bool(np.isin(tRes["status"][:, tA], P.SUCCESS).all()) for tA in (0, 1)]
    tE = [bool(np.isfinite(tS32[:, STATE_PER_ARM * tA:STATE_PER_ARM * tA + STATE_PER_ARM]).all()) for tA in (0, 1)]
    tAp = {"B_metric": list(metricOk)}
    tAp["C_workspace"] = [tAp["B_metric"][tA] and tC[tA] for tA in (0, 1)]
    tAp["D_IK"] = [tAp["C_workspace"][tA] and tD[tA] for tA in (0, 1)]
    tAp["E_pseudo_q"] = [tAp["D_IK"][tA] and tE[tA] for tA in (0, 1)]
    tArmFirstFail = [next((tK for tK in STAGES if not tAp[tK][tA]), None) for tA in (0, 1)]
    tRec.update(ws_in=tWs, arm_pass=tAp, arm_first_fail=tArmFirstFail, s32_finite=bool(np.isfinite(tS32).all()),
                ik_success_rows=[int(np.isin(tRes["status"][:, tA], P.SUCCESS).sum()) for tA in (0, 1)],
                status=[dict(collections.Counter(tRes["status"][:, tA].tolist())) for tA in (0, 1)],
                first_fail=next((tK for tK in STAGES if not all(tAp[tK])), None),               # PRIMARY: both arms
                one_arm_usable=any(tAp["E_pseudo_q"]))                                       # SECONDARY: >= 1 arm
    store["start"].append(start)
    store["q"].append(tQn.astype(np.float32))
    store["state32"].append(tS32.astype(np.float32))
    store["status"].append(tRes["status"])
    store["arm_usable"].append(tAp["E_pseudo_q"])
    store["usable"].append(tRec["first_fail"] is None)
    return tRec


def episodeRecord(e: str) -> dict:
    """the census record of one episode; writes OUT/<episode>.npz when it has segments that reached the IK (was episode)"""
    tInputs = INPUTS
    tDomain = tInputs.mManifest[f"{e}_left"]["domain"]
    tPre = tInputs.mPre[e]
    tRec = dict(episode=e, domain=tDomain)
    if tPre["first_fail"] is not None:
        tRec["first_fail"] = tPre["first_fail"]
        return tRec
    if not all((RUNS / f"{e}_{tS}" / "ss1.csv").is_file() for tS in ("left", "right")):
        tRec["first_fail"] = "mast3r_missing"
        return tRec
    tSides = {tS: sideData(f"{e}_{tS}") for tS in ("left", "right")}
    for tSide in ("left", "right"):
        tS = tSides[tSide]
        tRec[f"{tSide}"] = dict(scale=tS["s"], scale_valid=tS["scale_valid"],
                                g_norm=tS["imu_vi"]["g_norm"] if tS["imu_vi"] else None,
                                resid_rms=tS["imu_vi"]["resid_rms"] if tS["imu_vi"] else None, coverage=float(tS["ok"].mean()),
                                nonfinite_pose_rows=tS["nonfinite_pose_rows"], nonfinite_pose_side=tS["nonfinite_pose_rows"] > 0,
                                offset=tS["off"])
    tSess, tEp = e.rsplit("_", 1)
    tFm = frameMeta(RAW / f"Hpilot_{tSess}" / f"episode_{tEp}")
    tL, tR = tSides["left"], tSides["right"]
    tCapL = np.array([tC for _, tC in tFm["left_wrist"][tL["off"]:tL["off"] + tL["n"]]], float)
    tCapR = np.array([tC for _, tC in tFm["right_wrist"][tR["off"]:tR["off"] + tR["n"]]], float)
    tCapH = np.array([tC for _, tC in tFm["head"]], float)
    tJR, tJH = nearest(tCapR, tCapL), nearest(tCapH, tCapL)
    tDtR, tDtH = np.abs(tCapR[tJR] - tCapL) / NS_PER_MS, np.abs(tCapH[tJH] - tCapL) / NS_PER_MS
    tPoseOk = tL["ok"] & tR["ok"][tJR]
    tSynced = (tDtR < DT_MS) & (tDtH < DT_MS)
    tRowOk = tPoseOk & tSynced
    tIdx = np.flatnonzero(tRowOk)
    tRuns = np.split(tIdx, np.flatnonzero(np.diff(tIdx) != 1) + 1) if len(tIdx) else []
    tSegs = [tR[tK:tK + SEG] for tR in tRuns for tK in range(0, len(tR) - SEG + 1, SEG)]
    tNonFinite = tL["nonfinite_mask"] | tR["nonfinite_mask"][tJR]
    tRec.update(rows=int(tL["n"]), row_ok=int(tRowOk.sum()), runs_ge65=int(sum(len(tR) >= SEG for tR in tRuns)),
                rows_lost_to_nonfinite=int(tNonFinite.sum()), rows_lost_to_pose=int((~tPoseOk & ~tNonFinite).sum()),
                rows_lost_to_sync=int((tPoseOk & ~tSynced).sum()))
    tMetricOk = [tL["scale_valid"], tR["scale_valid"]]                                  # B_metric per ARM
    tStore = collections.defaultdict(list)
    tSegRecs = [segmentRecord(tS[0], tL["T_tcp"][tS], tR["T_tcp"][tJR[tS]], tL["width"][tS], tR["width"][tJR[tS]], tMetricOk,
                              tStore) for tS in tSegs]
    if tStore:
        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / f"{e}.npz", **{tK: np.array(tV) for tK, tV in tStore.items()}, seg_len=SEG, jR=tJR, jH=tJH)
    tRec["segments"] = tSegRecs
    tRec["first_fail"] = None if tSegs else "no_segment"
    return tRec


def _initWorker() -> None:
    loadInputs()
    init()


def printFunnel(records: dict) -> None:
    for tDomain in ("main_calibrated", "early_precalibration", "all"):
        tSel = [tR for tR in records.values() if tDomain == "all" or tR["domain"] == tDomain]
        tEf = collections.Counter(tR["first_fail"] for tR in tSel)
        tSegs = [tS for tR in tSel for tS in tR.get("segments", [])]
        tAlive = [len(tSegs)] + [0] * 4
        for tK in range(1, 5):
            tAlive[tK] = sum(1 for tS in tSegs if tS["first_fail"] is None or FIRST_FAIL_RANK[tS["first_fail"]] > tK)
        tSecondary = [len(tSegs)] + [sum(1 for tS in tSegs if any(tS.get("arm_pass", {}).get(tK, [False, False])))
                                     for tK in FUNNEL[1:]]
        tNfSides = sum(1 for tR in tSel for tS in ("left", "right") if tR.get(tS, {}).get("nonfinite_pose_side"))
        print(f"{tDomain:<22} episodes {len(tSel)}  episode first-fail {dict((tK, tC) for tK, tC in tEf.items() if tK)}  "
              f"with segments {tEf.get(None, 0)}  nonfinite sides {tNfSides}")
        print(" " * 23 + "Bimanual production yield (both arms)  " + "  ".join(f"{tN} {tA}" for tN, tA in zip(FUNNEL, tAlive)))
        print(" " * 23 + "Single-arm salvage yield (>=1 arm)     "
              + "  ".join(f"{tN} {tA}" for tN, tA in zip(FUNNEL, tSecondary)))
        tWithRows = [tR for tR in tSel if "rows" in tR]

        def tSum(tKey):
            return sum(tR[tKey] for tR in tWithRows)

        tTotal = tSum("rows") if tWithRows else 0
        tSegFail = collections.Counter(tS["first_fail"] for tS in tSegs if tS["first_fail"])
        print(" " * 23 + f"drop reasons | episodes: sync {tEf.get('sync_pass', 0)}  sensor {tEf.get('sensor_complete', 0)}  "
              f"gripper {tEf.get('gripper_valid', 0)}"
              f"  mast3r_missing {tEf.get('mast3r_missing', 0)}  no_segment {tEf.get('no_segment', 0)}"
              f" | rows (of {tTotal}): lost {tSum('rows_lost_to_pose') if tWithRows else 0}  "
              f"nonfinite {tSum('rows_lost_to_nonfinite') if tWithRows else 0}  "
              f"unsynced {tSum('rows_lost_to_sync') if tWithRows else 0}"
              f" | segments (both-arms first fail): metric/scale {tSegFail.get('B_metric', 0)}  "
              f"workspace {tSegFail.get('C_workspace', 0)}"
              f"  IK {tSegFail.get('D_IK', 0)}  pseudo_q {tSegFail.get('E_pseudo_q', 0)}")
        tWsFails = [tS["arm_first_fail"] for tS in tSegs if tS["first_fail"] == "C_workspace"]
        tBoth = sum(1 for tF in tWsFails if tF == ["C_workspace", "C_workspace"])
        tLeft = sum(1 for tF in tWsFails if tF[0] == "C_workspace" and tF[1] != "C_workspace")
        tRight = sum(1 for tF in tWsFails if tF[1] == "C_workspace" and tF[0] != "C_workspace")
        print(" " * 23 + f"workspace fails {len(tWsFails)}: one arm only {tLeft + tRight} (left {tLeft} / right {tRight}), "
              f"both arms {tBoth}"
              + (f"  -> one-arm share {100 * (tLeft + tRight) / len(tWsFails):.1f} %" if tWsFails else ""))


def main(argv: list[str] | None = None) -> int:
    tArgs = sys.argv[1:] if argv is None else argv
    OUT.mkdir(exist_ok=True)
    tInputs = loadInputs()
    tEpisodes = tArgs or sorted({tV["episode"] for tV in tInputs.mManifest.values()})
    tCtx = init()
    for tE in tEpisodes:                                                       # staging is serial (symlinks, csv rewrite)
        for tSide in ("left", "right"):
            if (RUNS / f"{tE}_{tSide}" / "ss1.csv").is_file() and tInputs.mPre[tE]["first_fail"] is None:
                stageSide(f"{tE}_{tSide}")
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2), initializer=_initWorker) as tEx:
        tRecs = dict(zip(tEpisodes, tEx.map(episodeRecord, tEpisodes)))
    tOut = dict(schema="c8_phase3/v1", date="2026-09-24", rules=RULES, provenance=tCtx.mP.provenance(), episodes=tRecs)
    tName = "phase3_census.json" if not tArgs else "phase3_census_partial.json"
    json.dump(tOut, open(C8 / tName, "w"), indent=1, default=float)
    printFunnel(tRecs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
