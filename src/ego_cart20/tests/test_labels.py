"""Characterization of the label builders (CART20 chunk, both state20 variants, action32 packing, stage ids) and the
frozen contract constants."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ego_cart20 import config as CFG
from ego_cart20.rotation6d import quaternionToMatrix
from ego_cart20.transforms import makeTransform, poseToTransform
from ego_cart20 import labels as BC
from ego_cart20 import labels as BS
from ego_cart20 import labels as PA
from ego_cart20 import labels as ST
from ego_cart20.resample import PoseTrack

from .synthetic import rawArrays


def _poses(n, seed):
    tRng = np.random.default_rng(seed)
    return makeTransform(quaternionToMatrix(tRng.normal(size=(n, 4))), tRng.normal(size=(n, 3)))


def test_contract_constants(golden):
    tConsts = {tK: getattr(CFG, tK) for tK in ("UMI_DT", "ROW_FPS", "HORIZON", "ARM_DIM", "CART_DIM", "AUX_DIM",
                                              "MODEL_ACTION_DIM", "STATE_DIM", "ARMS", "ACT_GRIP", "ST_GRIP", "PR_GRIP")}
    golden.check("cfg.constants", tConsts)
    tSlices = {tK: {tA: [tS.start, tS.stop] for tA, tS in getattr(CFG, tK).items()}
               for tK in ("ACT_POS", "ACT_ROT", "ST_POSE", "PR_POSE")}
    golden.check("cfg.slices", tSlices)
    assert (CFG.AUX.start, CFG.AUX.stop) == (20, 32)
    golden.check("cfg.contract", CFG.CONTRACT)
    assert list(json.loads(json.dumps(CFG.CONTRACT))) == list(CFG.CONTRACT)


def test_cart20_config_serialisation(golden):
    golden.check("cfg.default_dict", CFG.Cart20Config().toDict())
    tCfg = CFG.Cart20Config(mDiscontinuityFilter=True, mJumpWindow="row", mJumpIsolatedStepM=0.05)
    golden.check("cfg.v2b_dict", tCfg.toDict())
    assert tCfg == CFG.Cart20Config(mDiscontinuityFilter=True, mJumpWindow="row", mJumpIsolatedStepM=0.05)
    with pytest.raises(Exception):
        tCfg.horizon = 3


def test_cart20_from_poses_and_chunk(golden):
    tA = _poses(2, 0)
    tF = _poses(32, 1)
    tG = np.linspace(0, 1, 32)
    golden.check("bc.from_poses", BC.cart20FromPoses(tA[0], tF[:16], tG[:16], tA[1], tF[16:], tG[16:]))
    golden.check("bc.target_times", BC.targetTimes(1.25))
    golden.check("bc.target_times_custom", BC.targetTimes(0.5, 4, 0.1))
    tArr = rawArrays(0)
    tTracks = []
    for tSide in ("left", "right"):
        tT = (tArr[f"{tSide}_t_ns"] - tArr["left_t_ns"][0]) / 1e9
        tPose = poseToTransform(tArr[f"{tSide}_position"], tArr[f"{tSide}_quaternion"])
        tTracks.append(PoseTrack(tT, tPose, tArr[f"{tSide}_valid"],
                                 tArr[f"{tSide}_gripper"]))
    for tTime in (0.5, 1.2, 3.0, 5.5):
        tC, tOk = BC.buildCart20Chunk(tTracks[0], tTracks[1], tTime)
        golden.check(f"bc.chunk_{tTime}", tC)
        golden.check(f"bc.chunk_ok_{tTime}", tOk)
    tT0 = tTracks[0].sample([2.0])[0][0]
    tC, tOk = BC.buildCart20Chunk(tTracks[0], tTracks[1], 2.0, 8, 0.03, leftNow=tT0, rightNow=None)
    golden.check("bc.chunk_given_T0", tC)
    assert tOk is True


def test_state20(golden):
    tP = _poses(6, 2)
    tCur = _poses(10, 3)
    tGl = np.linspace(0, 1, 5)
    tGr = 1 - tGl
    golden.check("bs.relcart20", BS.buildState20Relcart20(tP[0], tCur[:5], tGl, tP[1], tCur[5:], tGr))
    golden.check("bs.prevrel", BS.buildState20Prevrel(tP[1:6], tCur[:5], tGl, tP[0], tCur[5:], tGr))
    golden.check("bs.relcart20_single", BS.buildState20Relcart20(tP[0], tCur[0], 0.3, tP[1], tCur[1], 0.6))
    assert BS.buildState20 is BS.buildState20Relcart20


def test_pack_action32(golden):
    tC = np.random.default_rng(5).normal(size=(3, 16, 20))
    tA = PA.packCart20ToAction32(tC)
    golden.check("pa.pack", tA)
    PA.checkAction32(tA, tC.astype(np.float32))
    with pytest.raises(AssertionError, match=r"CART20 must be \[\.\.\., 16, 20\], got \(16, 19\)"):
        PA.packCart20ToAction32(np.zeros((16, 19)))
    tBad = tA.copy()
    tBad[0, 0, 0] += 1
    with pytest.raises(AssertionError, match=r"action\[:, :20\] != CART20"):
        PA.checkAction32(tBad, tC.astype(np.float32))
    tBad = tA.copy()
    tBad[1, 2, 30] = 1
    tBad[0, 2, 21] = 2
    with pytest.raises(ValueError, match="AUX12 must be exactly zero for ego-only data: 2 non-zero entries"):
        PA.checkAction32(tBad, tC.astype(np.float32))
    with pytest.raises(AssertionError):
        PA.checkAction32(np.zeros((16, 31), np.float32), np.zeros((16, 20), np.float32))


def test_stage_ids(golden, tmp_path):
    tTimes = np.linspace(0, 10, 41)
    tIds, tInfo = ST.stageIds(tTimes, None, None, None)
    golden.check("st.none_ids", tIds)
    golden.check("st.none_info", tInfo)
    tIds, tInfo = ST.stageIds(tTimes, None, None, None, starts=[1.0, 3.0, 3.0, 7.5])
    golden.check("st.annot_ids", tIds)
    golden.check("st.annot_info", tInfo)
    with pytest.raises(AssertionError):
        ST.stageIds(tTimes, None, None, None, starts=[1.0, 3.0, 2.0, 7.5])
    assert ST.ANNOTATION_ENV == "EGO_CART20_STAGE_ANNOTATIONS"
    assert ST.loadAnnotations(tmp_path / "missing.json") == {}
    (tmp_path / "a.json").write_text(json.dumps({"ep": [0, 1, 2, 3]}))
    assert ST.loadAnnotations(str(tmp_path / "a.json")) == {"ep": [0, 1, 2, 3]}
