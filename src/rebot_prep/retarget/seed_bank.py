"""[2026-09-28] C-old v2-K seed bank: paired bimanual (qL, qR) configurations generated from the reBot's OWN kinematics
(Gen-1 retarget stage; was v2k_seedbank.py).

In:  ~/umi_bridge/track_c/data/r150_q_tcp.npz (envelope only), contracts/r30_episodes.json (with --ref r30), the reBot kinematics
     (pseudo_joint_pipeline.eef_kin) and the arm clearance
Out: ~/umi_bridge/track_c/v2k/seedbank[_R30]_q12.npy + seedbank[_R30].json (the frozen copies used by multi_seed_retarget are in
     contracts/), a printed funnel
Usage: python -m rebot_prep.retarget.seed_bank [--ref r30]

The frozen rule (written verbatim into seedbank.json "rule", and identical to contracts/seedbank.json) is RULE below.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import qmc

from ego_paths import DATA_ROOT, UMI_BRIDGE, prepend_sys_path

OUT = UMI_BRIDGE / "track_c/v2k"                          # the working folder the bank was generated in
CONTRACTS = pathlib.Path(__file__).resolve().parent / "contracts"
SOBOL_M, SEED, DEDUP_RAD, FLOOR_MARGIN_M, WS_POS_M, WS_ORI_DEG, D_HARD = 17, 20260928, 0.25, 0.010, 0.05, 20.0, 0.075
LINKS = ["link3", "link4", "link5", "link6", "gripper_link", "tcp", "gripper_left", "gripper_right"]
IK_CFG = dict(ori_weight=20.0, max_joint_delta=0.35)     # the frozen C4 IKCfg values
REF_SAMPLES, REF_SEED = 6000, 0                           # == phase3_segments.init()
SIGMA_REF_SAMPLES = 1500
JACOBIAN_STEP_RAD = 1e-4
WS_K = 50
FLOOR_PCT, ENVELOPE_LO_PCT, ENVELOPE_HI_PCT, SIGMA_FLOOR_PCT = 0.1, 1, 99, 1
TCP_Z_COLS = (2, 10)                                      # left / right TCP z in r150_q_tcp.npz "tcp"
RULE = """[2026-09-28] C-old v2-K seed bank: paired bimanual (qL, qR) configurations generated from the reBot's OWN kinematics.

No teleop episode, trajectory or episode identity is used as a seed. R150 enters only as an ENVELOPE (calibration of
bounds), never as a sample: per-joint p1-p99 range, the same R150 workspace NN test that Phase 3 gates targets with,
the table floor (TCP z p0.1 minus 10 mm) and a manipulability floor (p1 of R150's position-Jacobian sigma_min).
FROZEN before any v2 retarget result exists (user, 2026-09-28): rule + params + output hash in seedbank.json.

  1. Sobol 2^SOBOL_M points in 12-D (6 joints per arm) inside the R150 p1-p99 joint box
  2. per arm: FK TCP inside the R150 workspace (pos NN < 50 mm AND ori NN-50 < 20 deg == c8_phase3.ws_in)
  3. per arm: every link / finger point above the table floor
  4. per arm: position-Jacobian sigma_min >= R150 p1 (no near-singular pose)
  5. bimanual clearance d_min >= 0.075 m (c8_collision_v1, contract D_HARD)
  6. greedy dedup in 12-D joint space: keep a sample only if >= DEDUP_RAD from every kept one"""


def kinematicModules():
    """(pseudo_joint_pipeline, Clearance): the ~/umi_bridge/track_c and ~/c8 copies first, else the repository's"""
    prepend_sys_path(str(DATA_ROOT))
    prepend_sys_path(str(UMI_BRIDGE / "track_c"))
    try:
        import pseudo_joint_pipeline as P
    except ImportError:  # clean checkout: the repository copy
        from rebot_prep.pseudo_joint import pseudo_joint_pipeline as P
    try:
        from c8_collision_v1 import Clearance
    except ImportError:  # clean checkout: the repository copy
        from rebot_prep.census.arm_clearance import Clearance
    return P, Clearance


def main(argv: list[str] | None = None) -> int:
    tArgv = sys.argv if argv is None else ["seed_bank", *argv]
    # --ref r30 (2026-09-28 contract): EVERY R150-derived quantity below comes from the frozen R30 calibration episodes only
    # (r30_episodes.json = nested set_id < 5); the other 120 R150 episodes are never read. Default = legacy full R150.
    tRefName = tArgv[tArgv.index("--ref") + 1] if "--ref" in tArgv else "r150full"
    P, Clearance = kinematicModules()
    tT0 = time.time()
    tKin = P.eef_kin.Kin(dict(IK_CFG))
    tClr = Clearance(tKin)
    tD = np.load(UMI_BRIDGE / "track_c/data/r150_q_tcp.npz")
    tKeepRows = (np.isin(tD["ep"], json.load(open(CONTRACTS / "r30_episodes.json"))["episodes"]) if tRefName == "r30"
                 else np.ones(len(tD["ep"]), bool))
    tD = {tK: tD[tK][tKeepRows] for tK in ("J", "tcp")}
    tJ = np.radians(np.c_[tD["J"][:, :6], tD["J"][:, 7:13]])
    tLo, tHi = np.percentile(tJ, ENVELOPE_LO_PCT, axis=0), np.percentile(tJ, ENVELOPE_HI_PCT, axis=0)
    # envelope references (R150 frames: never used as samples)
    tIdx = np.random.default_rng(REF_SEED).choice(len(tJ), REF_SAMPLES, replace=False)               # == phase3_segments.init()
    tRef = {0: [], 1: []}
    for tI in tIdx:
        for tArm, tP7 in enumerate(tKin.fk_pose7(tJ[tI])):
            tRef[tArm].append(tP7)
    tRef = {tA: np.array(tV) for tA, tV in tRef.items()}
    tTree = {tA: cKDTree(tRef[tA][:, :3]) for tA in (0, 1)}
    tWr = {tA: Rot.from_quat(tRef[tA][:, 3:]).as_matrix() for tA in (0, 1)}
    tFloor = [float(np.percentile(tD["tcp"][:, tC], FLOOR_PCT)) - FLOOR_MARGIN_M for tC in TCP_Z_COLS]

    def tFkPos(tQ12):
        tL7, tR7 = tKin.fk_pose7(tQ12)
        return np.array([tL7[:3], tR7[:3]])

    def tSigmaMin(tQ12):
        """smallest singular value of each arm's 3x6 position Jacobian (finite differences, 1e-4 rad)"""
        tP0 = tFkPos(tQ12)
        tOut = []
        for tArm in (0, 1):
            tJp = np.zeros((3, 6))
            for tK in range(6):
                tDq = np.zeros(12)
                tDq[6 * tArm + tK] = JACOBIAN_STEP_RAD
                tJp[:, tK] = (tFkPos(tQ12 + tDq)[tArm] - tP0[tArm]) / JACOBIAN_STEP_RAD
            tOut.append(float(np.linalg.svd(tJp, compute_uv=False)[-1]))
        return tOut

    tSmFloor = np.percentile(np.array([tSigmaMin(tJ[tI]) for tI in tIdx[:SIGMA_REF_SAMPLES]]), SIGMA_FLOOR_PCT, axis=0)

    def tWsOk(tP7, tArm):
        tDp, tIi = tTree[tArm].query(tP7[:3], k=WS_K)
        tR = Rot.from_quat(tP7[3:]).as_matrix()
        tTr = np.einsum("nij,ij->n", tWr[tArm][tIi], tR)                      # trace(R_ref^T R)
        tAng = np.degrees(np.arccos(np.clip((tTr - 1) / 2, -1, 1))).min()
        return tDp[0] < WS_POS_M and tAng < WS_ORI_DEG

    tQ = tLo + qmc.Sobol(12, scramble=True, seed=SEED).random_base2(SOBOL_M) * (tHi - tLo)
    tFunnel = dict(sobol=len(tQ))
    tKeep = []
    for tQ12 in tQ:
        tL7, tR7 = tKin.fk_pose7(tQ12)
        if not (tWsOk(tL7, 0) and tWsOk(tR7, 1)):
            continue
        tKeep.append(tQ12)
    tFunnel["workspace"] = len(tKeep)
    tAboveFloor = []
    for tQ12 in tKeep:
        tPts = tClr.points(tQ12)
        if all(tPts[f"{tS}_{tN}"][2] >= tFloor[tA] for tA, tS in enumerate(("left", "right")) for tN in LINKS):
            tAboveFloor.append(tQ12)
    tFunnel["table_floor"] = len(tAboveFloor)
    tManip = [tQ for tQ in tAboveFloor if all(tS >= tF for tS, tF in zip(tSigmaMin(tQ), tSmFloor))]
    tFunnel["manipulability"] = len(tManip)
    tClear = [tQ for tQ in tManip if tClr.dmin(tQ)[0] >= D_HARD]
    tFunnel["collision"] = len(tClear)
    tBank = []
    for tQ12 in tClear:                                                          # greedy dedup, Sobol order
        if not tBank or np.min(np.linalg.norm(np.array(tBank) - tQ12, axis=1)) >= DEDUP_RAD:
            tBank.append(tQ12)
    tBank = np.array(tBank, np.float64)
    tFunnel["dedup"] = len(tBank)
    tTag = "" if tRefName == "r150full" else f"_{tRefName.upper()}"
    np.save(OUT / f"seedbank{tTag}_q12.npy", tBank)
    tSha = hashlib.sha256(tBank.tobytes()).hexdigest()
    tMeta = dict(schema="c_old_v2k_seedbank/v1", frozen="2026-09-28 (before any v2 retarget result)", rule=RULE,
                 reference=tRefName, reference_frames=int(len(tJ)),
                 params=dict(sobol_m=SOBOL_M, sobol_seed=SEED, dedup_rad=DEDUP_RAD, floor_margin_m=FLOOR_MARGIN_M,
                             ws_pos_m=WS_POS_M, ws_ori_deg=WS_ORI_DEG, d_hard=D_HARD, floor_links=LINKS),
                 envelope=dict(joint_lo_deg=np.degrees(tLo).round(2).tolist(), joint_hi_deg=np.degrees(tHi).round(2).tolist(),
                               table_floor_m=[round(tF, 4) for tF in tFloor], sigma_min_floor=tSmFloor.round(5).tolist()),
                 funnel=tFunnel, n=len(tBank), sha256_q12=tSha, seconds=round(time.time() - tT0, 1),
                 columns="q12 rad: left j1..j6, right j1..j6 (URDF / solver frame), paired same configuration")
    (OUT / f"seedbank{tTag}.json").write_text(json.dumps(tMeta, indent=1))
    print(json.dumps({tK: tMeta[tK] for tK in ("funnel", "n", "sha256_q12", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
