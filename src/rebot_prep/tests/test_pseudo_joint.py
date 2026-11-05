"""Characterization of the Gen-1 pseudo-joint core: state32 layout, eef_kin (FK / chunk IK) and pseudo_joint_pipeline
(camera pose -> v2 TCP, segments, robot-anchored retarget, continuity IK status rules), on the fake reBot solver."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from rebot_prep.tests.conftest import humanMotion
from rebot_prep.tests.golden import assertGolden

GOLDEN = Path(__file__).with_name("golden")


def _round(a, nd=9):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def test_state32_layout_and_prev_rel():
    from rebot_prep.pseudo_joint import state32
    tL, tR = humanMotion(5), humanMotion(5, 0.7)
    tOut, tValid = state32.build_state32([tL, tR], [np.linspace(0, 0.05, 5), np.full(5, 0.02)],
                                         [np.tile(np.arange(6) * 0.1, (5, 1)), -np.tile(np.arange(6) * 0.1, (5, 1))])
    assert tOut.dtype == np.float32 and tOut.shape == (5, 32)
    assert np.allclose(tOut[0, :9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-7)
    assert tValid.ravel().tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert len(state32.FIELDS) == 32 and state32.FIELDS[16] == "R_prev_rel_pos0" and state32.PER_ARM == 16
    assertGolden(GOLDEN, "state32", dict(out=_round(tOut, 6), prev_rel=_round(state32.prev_rel(tL))))


def test_eef_kin_fk_and_chunk_ik(fakeRobot, roots):
    from rebot_prep.pseudo_joint import eef_kin
    importlib.reload(eef_kin)
    tKin = eef_kin.Kin()
    tQ = np.array([0.1, -0.4, 0.5, 0.2, 0.1, -0.1, -0.1, -0.4, 0.5, -0.2, 0.1, 0.1])
    tL7, tR7 = tKin.fk_pose7(tQ)
    tPos = np.stack([np.stack([tL7[:3], tR7[:3]])] * 3) + np.array([0.0, 0.01, 0.02])[:, None, None]
    tQuat = np.stack([np.stack([tL7[3:], tR7[3:]])] * 3)
    tPos[2, 0] = [5.0, 0.0, 0.0]                                     # out of reach -> exception row
    tQ12, tErr, tOk = tKin.solve_chunk(tQ, tPos, tQuat)
    assert tOk.tolist() == [[True, True], [True, True], [False, False]]
    assert tErr[2].tolist() == [1e6, 1e6] and np.array_equal(tQ12[2], tQ12[1])
    assert eef_kin.IK == dict(max_joint_delta=0.35, ori_weight=0.6)
    assert tKin._arm == {"left": [0, 1, 2, 3, 4, 5], "right": [8, 9, 10, 11, 12, 13]}
    assertGolden(GOLDEN, "eef_kin", dict(l7=_round(tL7), r7=_round(tR7), q12=_round(tQ12, 6), err=_round(tErr, 4),
                                         svc=tKin._svc(tQ), q=_round(tKin._q(tKin._svc(tQ)))))


def test_segments_split_runs_of_one_id():
    from rebot_prep.pseudo_joint import pseudo_joint_pipeline as P
    tSeg = np.array([-1, 0, 0, 0, -1, 1, 1, 2, -1, 3, 3, 3, 3])
    assert P.segments(tSeg) == [(1, 4), (5, 7), (9, 13)]
    assert P.segments(tSeg, 3) == [(1, 4), (9, 13)]
    assert P.segments(np.array([5, 5])) == [(0, 2)]


def test_load_hu_applies_camera_tcp_v2_and_masks_lost_frames(pipeline, tmp_path):
    tT = humanMotion(8)
    tQuat = np.array([np.r_[0, 0, 0, 1.0]] * 8)
    from scipy.spatial.transform import Rotation
    tQuat = Rotation.from_matrix(tT[:, :3, :3]).as_quat()
    tQuat[3] = 0.0
    tSeg = np.array([0, 0, 0, 0, 0, 1, 1, -1])
    np.savez(tmp_path / "x.npz", frame_idx=np.arange(8) * 2, cam_pos=tT[:, :3, 3], cam_quat=tQuat, seg_id=tSeg,
             lost=np.array([0, 0, 0, 1, 0, 0, 0, 0]))
    tHu = pipeline.load_hu(tmp_path / "x.npz")
    assert tHu["seg_id"].tolist() == [0, 0, 0, -1, 0, 1, 1, -1] and tHu["n_seg_but_no_pose"] == 1
    assert np.allclose(tHu["T_hu"][0], tT[0] @ pipeline.X)
    assertGolden(GOLDEN, "load_hu", dict(T_hu=_round(tHu["T_hu"]), X=_round(pipeline.X)))


def test_retarget_and_continuity_ik_statuses(pipeline):
    tCfg = pipeline.IKCfg()
    tIk = pipeline.ContinuityIK(tCfg)
    tRt = pipeline.Retarget(tIk.kin, pipeline.ANCHORS[tCfg.anchor])
    tTL, tTR = tRt.targets(humanMotion(12), 0), tRt.targets(humanMotion(12, 1.0), 1)
    tTL[5, :3, 3] += [0.0, 0.0, 0.25]                              # jump -> large dq on the left arm
    tTR[8, :3, 3] = [9.0, 0.0, 0.0]                                # unreachable -> IK_FAIL row
    tRes = tIk.solve(tTL, tTR)
    assert tRes["status"].shape == (12, 2) and tRes["status"][8].tolist() == ["IK_FAIL", "IK_FAIL"]
    assert np.isnan(tRes["q"][8]).all()
    assert np.allclose(tRt.hold(3, 1), tRt.T_R0[1][None])
    assertGolden(GOLDEN, "continuity_ik", dict(
        T_R0=[_round(t) for t in tRt.T_R0], TL=_round(tTL), q=_round(tRes["q"], 6), q_last_good=_round(tRes["q_last_good"], 6),
        pos_err=_round(tRes["pos_err"], 4), rot_err=_round(tRes["rot_err"], 4), dq=_round(tRes["dq"], 6),
        status=tRes["status"].tolist(), reason=[r[:60] for r in tRes["reason"].tolist()], iters=tRes["iters"].tolist(),
        at_limit=tRes["at_limit"].tolist()))


def test_continuity_ik_from_a_given_start_and_cfg_defaults(pipeline):
    tCfg = pipeline.IKCfg(ori_weight=5.0, dq_warn=0.05, branch_dq=0.01)
    tIk = pipeline.ContinuityIK(tCfg)
    tRt = pipeline.Retarget(tIk.kin)
    tRes = tIk.solve(tRt.targets(humanMotion(6), 0), tRt.targets(humanMotion(6, 2.0), 1), q12_start=pipeline.Q_R0)
    assert pipeline.IKCfg().__dict__ == dict(ori_weight=20.0, anchor="r150_active_median", max_joint_delta=0.35, iters=10, conv_dq=1e-4,
                                             dq_warn=0.6, dq_fail=1.0, branch_dq=0.3, branch_dp_mm=5.0, branch_dth_deg=5.0,
                                             pos_ok_mm=15.0, rot_ok_deg=5.0)
    assert pipeline.SUCCESS == ("OK", "DQ_WARN") and pipeline.HARD[0] == "IK_FAIL"
    assertGolden(GOLDEN, "continuity_ik_start", dict(q=_round(tRes["q"], 6), status=tRes["status"].tolist(),
                                                     reason=tRes["reason"].tolist(), lo=_round(tIk.lo), hi=_round(tIk.hi)))


def test_provenance_and_frames(pipeline):
    tProv = pipeline.provenance()
    tProv["X_file"] = Path(tProv["X_file"]).name
    assertGolden(GOLDEN, "provenance", dict(prov=tProv, C=_round(pipeline.C), F=_round(pipeline.F), Q_R0=_round(pipeline.Q_R0),
                                            ARM_J=pipeline.ARM_J))


def test_continuity_ik_soft_and_convergence_statuses(pipeline):
    tCfg = pipeline.IKCfg(iters=1, pos_ok_mm=2.0, dq_warn=0.01, branch_dq=0.03, branch_dp_mm=12.0, branch_dth_deg=30.0)
    tIk = pipeline.ContinuityIK(tCfg)
    tRt = pipeline.Retarget(tIk.kin)
    tTL, tTR = tRt.targets(humanMotion(10), 0), tRt.targets(humanMotion(10, 0.5), 1)
    tTR[4:, :3, 3] += [0.0, -0.06, 0.0]
    tRes = tIk.solve(tTL, tTR)
    tSeen = set(tRes["status"].ravel().tolist())
    assert {"NOT_CONVERGED", "DQ_WARN"} <= tSeen or {"NOT_CONVERGED", "BRANCH_SWITCH"} <= tSeen
    assertGolden(GOLDEN, "continuity_ik_soft", dict(q=_round(tRes["q"], 6), status=tRes["status"].tolist(), reason=tRes["reason"].tolist()))


# ---------------------------------------------------------------- C2 input gate (raw TCP quality metrics)
GATE_PATH = Path(__file__).resolve().parents[1] / "pseudo_joint" / "c2_input_gate.py"


def _gateMetrics():
    from rebot_prep.tests.scriptdefs import loadDefs
    return loadDefs(GATE_PATH, {"POS_PAIR_MM", "ROT_PAIR_DEG", "REV_COS", "metrics"}, {"__name__": "gate_defs"})["metrics"]


def test_input_gate_metrics_on_a_zigzag_path():
    tN = 30
    tT = humanMotion(tN)
    tT[10:20, 0, 3] += np.where(np.arange(10) % 2, 0.012, -0.012)          # A-B-A-B reversals of 24 mm
    tNear = np.zeros(tN, bool)
    tNear[12:16] = True
    tM = _gateMetrics()(tT, tNear)
    tNearStep = tM.pop("_near_step")
    assert tM["zigzag_bursts"] >= 1 and tM["rows"] == tN and tM["finite"]
    tPlain = _gateMetrics()(humanMotion(tN))
    assert "_near_step" not in tPlain and "lost_adjacent_frac" not in tPlain
    assertGolden(GOLDEN, "input_gate_metrics", dict(m=tM, near=_round(tNearStep), plain=tPlain))
