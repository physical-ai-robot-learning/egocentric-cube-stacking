"""Characterization of the raw-track sampling, stream synchronisation, gripper normalisation and pose-jump filter."""
from __future__ import annotations

import numpy as np
import pytest

from ego_cart20.geometry.transforms import pose_to_T
from ego_cart20.preprocessing import gripper as GR
from ego_cart20.preprocessing import pose_filter as PF
from ego_cart20.preprocessing import resample as RS
from ego_cart20.preprocessing import synchronize as SY

from .synthetic import rawArrays


def _track(side="left", seed=0, **kw):
    tA = rawArrays(seed, **kw)
    tT = (tA[f"{side}_t_ns"] - tA["left_t_ns"][0]) / 1e9
    return tT, pose_to_T(tA[f"{side}_position"], tA[f"{side}_quaternion"]), tA[f"{side}_valid"], tA[f"{side}_gripper"]


def test_pose_track_sample(golden):
    tT, tPose, tValid, tGrip = _track("left")
    tTrack = RS.PoseTrack(tT, tPose, tValid, tGrip, 0.050)
    golden.check("rs.track_valid", tTrack.valid)
    golden.check("rs.track_q", tTrack.q)
    tQuery = np.concatenate([[-0.1, tT[0], tT[5], tT[-1], tT[-1] + 0.01], np.linspace(0, 6.1, 97)])
    tS, tG, tOk = tTrack.sample(tQuery)
    golden.check("rs.sample_T", tS)
    golden.check("rs.sample_g", tG)
    golden.check("rs.sample_ok", tOk)
    tS1, tG1, tOk1 = tTrack.sample(1.234)
    golden.check("rs.sample_scalar", np.concatenate([tS1.ravel(), tG1, tOk1.astype(float)]))
    tNarrow = RS.PoseTrack(tT, tPose, tValid, tGrip, 0.020)            # gap rule tighter than the raw spacing
    golden.check("rs.sample_ok_narrow", tNarrow.sample(tQuery)[2])
    with pytest.raises(AssertionError):
        RS.PoseTrack(tT[::-1], tPose, tValid, tGrip)


def test_canonical_times_and_resample_episode(golden):
    golden.check("rs.canonical_times", RS.canonical_times(0.013, 2.0, 15.0))
    golden.check("rs.canonical_times_exact_end", RS.canonical_times(0.0, 1.0, 15.0))
    tT, tPose, tValid, tGrip = _track("right")
    tOut = RS.resample_episode(tT, tPose, tGrip, 15, tValid)
    for tName, tArr in zip(("times", "T", "g", "ok"), tOut):
        golden.check(f"rs.resample_{tName}", tArr)
    tA = rawArrays(0)
    tOutDict = RS.resample_episode(tT, dict(position=tA["right_position"], quaternion=tA["right_quaternion"]), tGrip,
                                   target_fps=10, valid=None, max_gap_s=0.1, t_start=0.5)
    for tName, tArr in zip(("times", "T", "g", "ok"), tOutDict):
        golden.check(f"rs.resample_dict_{tName}", tArr)


def test_synchronize(golden):
    tStream = np.array([0.0, 0.1, 0.2, 0.35, 0.4])
    tJ, tDt = SY.nearest_index(tStream, np.array([-0.05, 0.04, 0.05, 0.06, 0.27, 0.3, 0.5]), 0.03)
    golden.check("sy.nearest_j", tJ)
    golden.check("sy.nearest_dt", tDt)
    tJ1, tDt1 = SY.nearest_index(np.array([0.2]), [0.19, 0.5], 0.05)
    golden.check("sy.nearest_single", [tJ1.tolist(), tDt1.tolist()])
    tJ0, tDt0 = SY.nearest_index(np.array([]), [0.1, 0.2], 0.05)
    golden.check("sy.nearest_empty", [tJ0.tolist(), tDt0.tolist()])
    tLeft = RS.PoseTrack(*_track("left"))
    tRight = RS.PoseTrack(*_track("right"))
    golden.check("sy.task_start", SY.task_start_time([tLeft, tRight]))
    golden.check("sy.common_end", SY.common_end_time([tLeft, tRight]))
    tT, tPose, _, tGrip = _track("left")
    tNone = RS.PoseTrack(tT, tPose, np.zeros(len(tT), bool), tGrip)
    tLate = RS.PoseTrack(tT + 100.0, tPose, np.ones(len(tT), bool), tGrip)
    with pytest.raises(ValueError, match="no instant where all arms are valid"):
        SY.task_start_time([tLeft, tLate])
    assert tNone.valid.sum() == 0


def test_gripper(golden):
    tCal = dict(direction=-1, touch_tick=2000, knots_delta_ticks=[0, 100, 400, 900], knots_mm=[0.0, 5.0, 30.0, 80.0], w_open_mm=80.0)
    tTicks = np.array([2100, 2000, 1950, 1700, 1100, 900, 0])
    tMm = GR.ticks_to_mm(tTicks, tCal)
    golden.check("gr.ticks_to_mm", tMm)
    golden.check("gr.ticks_to_mm_default_dir", GR.ticks_to_mm(tTicks, {k: v for k, v in tCal.items() if k != "direction"}))
    golden.check("gr.mm_to_g", GR.mm_to_g(np.array([-3.0, 0.0, 40.0, 80.0, 95.0]), 80.0))
    golden.check("gr.validate_ok", GR.validate_gripper(np.array([0.1, np.nan, 0.9])))
    golden.check("gr.validate_empty", GR.validate_gripper(np.array([np.nan])))
    with pytest.raises(ValueError, match=r"left raw gripper outside \[0, 1\]: min -0.1 max 0.5"):
        GR.validate_gripper(np.array([-0.1, 0.5]), "left raw gripper")


def test_jump_mask(golden):
    tT, tPose, tValid, _ = _track("right")
    golden.check("pf.jump_mask", PF.jump_mask(tT, tPose, tValid))
    golden.check("pf.jump_mask_tight", PF.jump_mask(tT, tPose, tValid, max_lin_mps=0.5, max_ang_rps=0.5))


@pytest.mark.parametrize("window", ["frame", "row"])
@pytest.mark.parametrize("isolated", [None, 0.05])
def test_discontinuity_filter(golden, window, isolated):
    for tName, tKw in (("spike", {}), ("oneway", dict(spike=False, oneWay=True)), ("both", dict(oneWay=True))):
        tT, tPose, tValid, _ = _track("right", **tKw)
        tNew, tEvents = PF.discontinuity_filter(tT, tPose, tValid, 0.100, 45.0, window=window, isolated_step_m=isolated)
        golden.check(f"pf.dfilter_{window}_{isolated}_{tName}_valid", tNew)
        golden.check(f"pf.dfilter_{window}_{isolated}_{tName}_events", tEvents)
    tNew, tEvents = PF.discontinuity_filter(tT[:1], tPose[:1], tValid[:1])
    assert tNew.tolist() == [True] and tEvents == []
    assert PF.ROW_S == 1.0 / 15.0
