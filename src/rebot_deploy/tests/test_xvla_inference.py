"""Characterization of the inference core's pure parts on synthetic inputs: RTC weights and the RTC prefix, limits, the gripper
mapping per GRIPPER_MODE, image decode / zoom, the one-arm image swap, TCP frames (frame fix, HandUMI convention), the
observation buffer and the slim20 / umi76 / cart20 / relcart20 / umi94 state builders, chunk decode (umi / delta, clamp,
temporal ensemble, adaptive k, deadband) and the per-waypoint IK path with the numerical backend. The policy, the HandUMI
kinematics and the network are replaced by fakes; the module settings each case depends on are set explicitly.

The module reads V4_* at import: run these tests with no V4_* / IK_BACKEND variables set."""
from __future__ import annotations

import base64
import json
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from rebot_deploy import infer_core_v4 as core
from rebot_deploy.tests.fake_kinematics import FakeKin

pytestmark = pytest.mark.skipif(any(tK.startswith("V4_") for tK in os.environ) or "IK_BACKEND" in os.environ,
                                reason="V4_* / IK_BACKEND set in the environment")


def _jpegB64(seed, shape=(480, 640, 3)):
    tRng = np.random.default_rng(seed)
    tImg = (tRng.random(shape) * 255).astype(np.uint8)
    tImg[40:120, 60:200] = (0, 0, 230)
    tOk, tEnc = cv2.imencode(".png", tImg)
    return base64.b64encode(tEnc.tobytes()).decode()


def _joints14(rng):
    tQ = rng.uniform(-0.5, 0.5, 14)
    tQ[6] = -120.0
    tQ[13] = 358.0
    return tQ.tolist()


def _inferencer(monkeypatch, tmp_path, **settings):
    """a V4Inferencer without a policy: the fake kin, n_action 4, chunk 8, and the module settings given"""
    tDefaults = dict(ACTION_MODE="umi", GRIPPER_MODE="hold", IK_BACKEND="numerical", STATE_MODE="slim20", ARM_ONLY="",
                     TE_ON=False, ADAPTIVE_K=None, DEADBAND_MM=None, DB_XY_MM=None, DB_Z_MM=None, EXEC_K=1, USE_FRAME_FIX=True,
                     UMI_CONV=False, LIMITS={"mm": 10.0, "deg": 2.0}, IK_LOG=str(tmp_path / "ik.jsonl"), FIXED_ANCHOR_R=None)
    tDefaults.update(settings)
    for tName, tValue in tDefaults.items():
        monkeypatch.setattr(core, tName, tValue)
    tInf = object.__new__(core.V4Inferencer)
    tInf.kin = FakeKin()
    tInf.n_action = 4
    tInf.chunk = 8
    tInf.calls = 0
    tInf.last = {}
    tInf._rtc_prev = None
    tInf.task = "stack"
    return tInf


def _chunk(rng, steps, dims=20):
    """a (steps, dims) action chunk in raw units: per arm pose9 (small rotation, a few cm) + gripper"""
    tAct = np.zeros((steps, dims))
    for tR in (0, 1):
        for tK in range(steps):
            tRot = Rotation.from_rotvec(rng.normal(size=3) * 0.05 * (tK + 1)).as_matrix()
            tAct[tK, tR * 10:tR * 10 + 3] = rng.normal(size=3) * 0.004 * (tK + 1)
            tAct[tK, tR * 10 + 3:tR * 10 + 9] = tRot[:2, :].reshape(6)
        tAct[:, tR * 10 + 9] = np.linspace(0.2, 0.8, steps)[::(1 if tR else -1)]
    return tAct


def test_constants(golden):
    golden.check("core.constants", {tName: getattr(core, tName) for tName in (
        "DEVICE", "CHUNK_OVERRIDE", "DTYPE", "ACTION_MODE", "CAM_MAP", "RENAME", "ARM_IDX", "GRIP_IDX", "FLIP_IDX", "DQ_MAX",
        "IK_BACKEND", "CUROBO_URL", "RTC_ON", "RTC_DELAY", "RTC_HORIZON", "RTC_MAX_GW", "RTC_MAX_AGE_S", "REMOTE_INFER", "REMOTE_CKPT",
        "YAW_MAX_STEP_DEG", "PINK_LOCK", "PINK_ORI", "FK_MAX_MM", "LIMITS", "CLAMP_MM", "CLAMP_DEG", "N_ACTION", "EXEC_K", "TE_ON",
        "TE_W", "TE_ROT", "TE_GRIP", "TE_MAXGAP_S", "ADAPTIVE_K", "AK_FAR_MM", "AK_NEAR_MM", "DEADBAND_MM", "DB_XY_MM", "DB_Z_MM",
        "DB_REF_K", "GRIPPER_MODE", "LO", "HI", "R_DS_TO_FK", "R_FK_TO_DS", "USE_FRAME_FIX", "UMI_TCP_PITCH_DEG", "R_UMI_OFF",
        "UMI_TCP_OFF_T", "UMI_CONV", "GLOBAL_ROT180", "GLOBAL_MIRROR", "STATE_MODE", "ARM_ONLY", "ARM_ONLY_LEFT_G", "ARM_ONLY_GRIP",
        "ARM_ONLY_MAX_PRED_MM", "ARM_ONLY_MAX_PRED_DEG", "UMI_HISTORY_DT", "OBS_BUF_S", "RELONLY")})
    assert core.WRIST_ZOOM == {"s": 1.0, "c": (318.3126, 224.9012), "wh": (640, 480)}
    assert core.V4Inferencer.obs_horizon == 2


def test_rtc_weights(golden):
    tCases = ((0, 0, 8), (2, 5, 8), (3, 2, 8), (9, 12, 8), (0, 8, 8), (-2, 3, 6), (1, 1, 4))
    golden.check("core.rtc_weights", [core._rtc_weights(tD, tH, tT) for tD, tH, tT in tCases])


def test_set_limits(monkeypatch):
    monkeypatch.setattr(core, "LIMITS", {"mm": 10.0, "deg": 2.0})
    assert core.set_limits() == {"mm": 10.0, "deg": 2.0}
    assert core.set_limits(7, None) == {"mm": 7.0, "deg": 2.0}
    assert core.set_limits(None, "1.5") == {"mm": 7.0, "deg": 1.5}
    assert core.LIMITS == {"mm": 7.0, "deg": 1.5}


def test_grip_to_cmd_per_mode(monkeypatch, golden):
    tValues = (-0.1, 0.0, 0.03, 0.1144, 0.49, 0.5, 0.7, 1.2)
    tOut = {}
    for tMode in ("hold", "predict", "binary", "continuous", "relative"):
        monkeypatch.setattr(core, "GRIPPER_MODE", tMode)
        tOut[tMode] = [core.grip_to_cmd(tV) for tV in tValues]
    golden.check("core.grip_to_cmd", tOut)


def test_images(golden):
    tImg = cv2.imdecode(np.frombuffer(base64.b64decode(_jpegB64(1)), np.uint8), cv2.IMREAD_COLOR)
    assert core.wrist_zoom(tImg, 1.0) is tImg
    tZoom = core.wrist_zoom(tImg, 1.3)
    tSmall = core.wrist_zoom(tImg[:240, :320].copy(), 0.8)
    tOut = {"zoom": [int(tZoom.astype(np.int64).sum()), tZoom[200:203, 300:303]], "small": [list(tSmall.shape), int(tSmall.sum())]}
    for tName, tArgs in (("plain", ()), ("rot", (True,)), ("mirror", (False, True)), ("both", (True, True)), ("zoom", (False, False, 1.4))):
        tTensor = core.V4Inferencer._img(_jpegB64(2), *tArgs)
        assert tTensor.shape == (3, 224, 224) and tTensor.dtype == torch.float32
        tOut[tName] = [float(tTensor.double().sum()), tTensor[:, 100, 50:53]]
    golden.check("core.images", tOut)


def test_arm_only_images(monkeypatch, golden):
    tImgs = {"left": "L", "right": "R", "middle": "M"}
    monkeypatch.setattr(core, "ARM_ONLY", "")
    assert core.arm_only_images(tImgs) is tImgs
    monkeypatch.setattr(core, "ARM_ONLY", "right")
    monkeypatch.setattr(core, "_BLACK_B64", None)
    tOut = core.arm_only_images(tImgs)
    assert tImgs == {"left": "L", "right": "R", "middle": "M"} and tOut["right"] == "R" and tOut["middle"] == "M"
    tBlack = cv2.imdecode(np.frombuffer(base64.b64decode(tOut["left"]), np.uint8), cv2.IMREAD_COLOR)
    assert tBlack.shape == (224, 224, 3) and int(tBlack.max()) == 0
    assert core._BLACK_B64 == tOut["left"] == core._black_b64()


def test_tcp_frames(monkeypatch, tmp_path, golden):
    tRng = np.random.default_rng(10)
    tQ = _joints14(tRng)
    tOut = []
    for tFix, tConv in ((True, False), (False, False), (True, True)):
        tInf = _inferencer(monkeypatch, tmp_path, USE_FRAME_FIX=tFix, UMI_CONV=tConv)
        if tConv:
            monkeypatch.setattr(core, "UMI_TCP_OFF_T", np.array([-0.0948, 0.0024, 0.0384]))
            monkeypatch.setattr(core, "R_UMI_OFF", Rotation.from_rotvec([0.0, -np.radians(33.6), 0.0]).as_matrix())
        tOut.append(list(tInf._tcp_mat(tQ)))
        tOut.append(list(tInf.tcp_now(tQ)))
    golden.check("core.tcp_frames", tOut)


def test_slim20_state(monkeypatch, tmp_path, golden):
    tRng = np.random.default_rng(11)
    tInf = _inferencer(monkeypatch, tmp_path)
    tHistory = [{"joints14": _joints14(tRng)}, {"joints14": _joints14(tRng)}]
    golden.check("core.slim20_state", list(tInf.build_state(tHistory)))
    with pytest.raises(AssertionError):
        tInf.build_state(tHistory[:1])


def test_buffered_states(monkeypatch, tmp_path, golden):
    tRng = np.random.default_rng(12)
    tJoints = [_joints14(tRng) for _ in range(6)]
    tOut = {}
    for tMode in ("umi76", "umi94", "cart20", "relcart20"):
        tInf = _inferencer(monkeypatch, tmp_path, STATE_MODE=tMode)
        with pytest.raises(RuntimeError):
            tInf.build_state_umi76()
        tSizes = [tInf.push_obs(tQ, 100.0 + 0.0667 * tI) for tI, tQ in enumerate(tJoints[:3])]
        tStates = [list(tInf.build_state_umi76())]
        tSizes += [tInf.push_obs(tQ, 100.0 + 0.0667 * tI) for tI, tQ in enumerate(tJoints[3:], start=3)]
        tStates.append(list(tInf.build_state_umi76()))
        tOut[tMode] = [tSizes, tStates, [tInf._interp_at(100.05)[0], tInf._interp_at(100.05)[1]]]
        if tMode == "relcart20":
            tInf.reset_anchor()
            tOut["relcart20_reset"] = list(tInf.build_state_umi76())
            monkeypatch.setattr(core, "ARM_ONLY", "right")
            monkeypatch.setattr(core, "FIXED_ANCHOR_R", np.diag([1.0, -1.0, -1.0, 1.0]))
            tInf.reset_anchor()
            tOut["relcart20_arm_only_fixed"] = list(tInf.build_state_umi76())
    tShort = _inferencer(monkeypatch, tmp_path, STATE_MODE="umi76")
    tShort.push_obs(tJoints[0], 10.0)
    tShort.push_obs(tJoints[1], 10.001)
    with pytest.raises(RuntimeError, match="history"):
        tShort.build_state_umi76()
    tFull = _inferencer(monkeypatch, tmp_path, STATE_MODE="umi76", OBS_BUF_S=0.1)
    tOut["buffer_trim"] = [tFull.push_obs(tQ, 50.0 + 0.04 * tI) for tI, tQ in enumerate(tJoints)]
    golden.check("core.buffered_states", tOut)


def _decode(monkeypatch, tmp_path, rng, steps=8, dims=20, **settings):
    tInf = _inferencer(monkeypatch, tmp_path, **settings)
    tHistory = [{"joints14": _joints14(rng)}, {"joints14": _joints14(rng)}]
    tState, tMNow, tPosesNow = tInf.build_state(tHistory)
    tAct = _chunk(rng, steps, dims)
    tCmd = tInf._post_act(tAct, tState, tMNow, tPosesNow, tHistory)
    tLast = {tK: tV for tK, tV in tInf.last.items()}
    tInfo = [tInf.te_info, tInf.db_info, tInf.ak_info, tInf.exec_k_now]
    return tInf, [tCmd, tLast, tInfo, tInf._rtc_prev["pos"], tInf.calls]


def test_chunk_decode(monkeypatch, tmp_path, golden):
    tCases = {
        "umi_hold": dict(),
        "delta_predict": dict(ACTION_MODE="delta", GRIPPER_MODE="predict"),
        "umi_binary_tight": dict(GRIPPER_MODE="binary", LIMITS={"mm": 2.0, "deg": 0.5}),
        "umi_continuous_noclamp": dict(GRIPPER_MODE="continuous", LIMITS={"mm": 0.0, "deg": 0.0}),
        "pink_backend": dict(IK_BACKEND="pink", GRIPPER_MODE="relative"),
        "adaptive_k": dict(ADAPTIVE_K=[4, 2, 1], GRIPPER_MODE="continuous"),
        "adaptive_k_hold": dict(ADAPTIVE_K=[4, 2, 1], AK_FAR_MM=1e6, AK_NEAR_MM=1e5),
        "deadband": dict(DB_XY_MM=30.0, DB_Z_MM=5.0, DEADBAND_MM=np.array([1.0, 50.0, 2.0])),
        "exec_k": dict(EXEC_K=3),
        "arm_only": dict(ARM_ONLY="right", ARM_ONLY_GRIP="model", GRIPPER_MODE="predict"),
    }
    tOut = {}
    for tName, tSettings in tCases.items():
        _, tOut[tName] = _decode(monkeypatch, tmp_path, np.random.default_rng(20), **tSettings)
    _, tOut["dq_head"] = _decode(monkeypatch, tmp_path, np.random.default_rng(21), dims=32)
    golden.check("core.chunk_decode", tOut)


def test_chunk_decode_guards(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="ARM_ONLY output guard"):
        _decode(monkeypatch, tmp_path, np.random.default_rng(22), ARM_ONLY="right", ARM_ONLY_MAX_PRED_MM=1.0)
    tInf = _inferencer(monkeypatch, tmp_path, FK_MAX_MM=0.5)
    tRng = np.random.default_rng(23)
    tHistory = [{"joints14": _joints14(tRng)}, {"joints14": _joints14(tRng)}]
    tState, tMNow, tPosesNow = tInf.build_state(tHistory)
    with pytest.raises(RuntimeError, match="IK could not reach"):
        tInf._post_act(_chunk(tRng, 8), tState, tMNow, tPosesNow, tHistory)


def test_temporal_ensemble(monkeypatch, tmp_path, golden):
    tInf = _inferencer(monkeypatch, tmp_path, TE_ON=True, TE_W=np.array([1.0, 0.5, 0.25]), TE_ROT=True, TE_GRIP=True,
                       ADAPTIVE_K=[3, 2, 1], GRIPPER_MODE="continuous")
    tRng = np.random.default_rng(24)
    tOut = []
    for _ in range(4):
        tHistory = [{"joints14": _joints14(tRng)}, {"joints14": _joints14(tRng)}]
        tState, tMNow, tPosesNow = tInf.build_state(tHistory)
        tCmd = tInf._post_act(_chunk(tRng, 8), tState, tMNow, tPosesNow, tHistory)
        tOut.append([tCmd, tInf.te_info, tInf.ak_info, tInf.last["tgt_pos"], tInf.last["tgt_quat"], tInf.last["widths"], tInf._te_step])
    golden.check("core.temporal_ensemble", tOut)


def test_rtc_prefix(monkeypatch, tmp_path, golden):
    tInf = _inferencer(monkeypatch, tmp_path, RTC_DELAY=2, RTC_HORIZON=5)
    tRng = np.random.default_rng(25)
    tInf._act_ms = (tRng.normal(size=32) * 0.01, 0.05 + tRng.random(32) * 0.1)
    tInf.policy = SimpleNamespace(model=SimpleNamespace(dim_action=32))
    tMNow = np.stack([np.eye(4), np.eye(4)])
    tMNow[0][:3, 3] = [0.3, 0.2, 0.1]
    tMNow[1][:3, 3] = [0.3, -0.2, 0.1]
    assert tInf._rtc_build(tMNow) is None                                  # no previous chunk
    tPos = tMNow[None, :, :3, 3] + np.linspace(0, 0.05, 8)[:, None, None] * np.array([1.0, 0.5, -0.3])
    tQuat = np.stack([Rotation.from_rotvec(tRng.normal(size=(8, 3)) * 0.1).as_quat() for _ in range(2)], 1)
    tWidths = np.linspace(0.0, 1.0, 16).reshape(8, 2)
    tInf._rtc_prev = dict(t=time.time(), pos=tPos, quat=tQuat, widths=tWidths)
    tOut = {}
    for tName, tHint in (("default", None), ("hint", 4)):
        tInf._rtc_delay_hint = tHint
        tRes = tInf._rtc_build(tMNow + np.array([0, 0, 0, 0.0])[None, None])
        tInfo = dict(tRes["info"])
        assert tInfo.pop("age_s") < 1.0
        tOut[tName] = [tRes["Y"], tRes["W"], tInfo]
    monkeypatch.setattr(core, "RTC_HORIZON", 0)
    tInf._rtc_delay_hint = None
    tOut["whole_leftover"] = tInf._rtc_build(tMNow)["W"][:, 0]
    tInf._rtc_plan_prefix = [(tPos[tI], tQuat[tI], tWidths[tI]) for tI in range(2, 6)]
    tInf._rtc_delay_hint = 1
    tPlan = tInf._rtc_build(tMNow)
    tOut["plan"] = [tPlan["Y"], tPlan["W"], tPlan["info"]]
    tInf._rtc_plan_prefix = None
    tInf._rtc_prev["t"] = time.time() - 100.0
    assert tInf._rtc_build(tMNow) is None                                  # too old
    monkeypatch.setattr(core, "ACTION_MODE", "delta")
    tInf._rtc_prev["t"] = time.time()
    assert tInf._rtc_build(tMNow) is None                                  # delta action mode: no RTC
    golden.check("core.rtc_prefix", tOut)


def test_grip_relative(monkeypatch, tmp_path, golden):
    tSeq = (0.5, 0.52, 0.6, 0.58, 0.4, 0.41, 0.47, 0.49, 0.55, 0.6)
    tOut = {}
    for tMode in ("delta", "acc"):
        tInf = _inferencer(monkeypatch, tmp_path, GRIP_REL_MODE=tMode, GRIP_ACC_N=3)
        tCmds = []
        for tI, tG in enumerate(tSeq):
            tCmds.append([tInf._grip_relative(0, tG, -200.0), tInf._grip_relative(1, 1.0 - tG, -10.0), tInf._grip_rel_log])
            tCmds[-1][2] = json.loads(json.dumps(tCmds[-1][2]))
        tInf.reset_anchor()
        tCmds.append([tInf._grip_relative(0, 0.3, 358.0), tInf._g_prev, tInf._g_cmd, tInf._g_hist])
        tOut[tMode] = tCmds
    golden.check("core.grip_relative", tOut)


def test_solve_waypoint_numerical(monkeypatch, tmp_path, golden):
    tRng = np.random.default_rng(26)
    tOut = {}
    for tName, tSettings in (("hold", dict()), ("predict_nofix", dict(GRIPPER_MODE="predict", USE_FRAME_FIX=False)),
                             ("relative", dict(GRIPPER_MODE="relative")), ("arm_only", dict(ARM_ONLY="right", ARM_ONLY_GRIP="hold")),
                             ("umi_conv", dict(UMI_CONV=True, GRIPPER_MODE="binary"))):
        tInf = _inferencer(monkeypatch, tmp_path, **tSettings)
        if tName == "umi_conv":
            monkeypatch.setattr(core, "UMI_TCP_OFF_T", np.array([-0.0948, 0.0024, 0.0384]))
            monkeypatch.setattr(core, "R_UMI_OFF", Rotation.from_rotvec([0.0, -np.radians(33.6), 0.0]).as_matrix())
        tQ = _joints14(tRng)
        tPos = np.array([[0.32, 0.21, 0.12], [0.28, -0.19, 0.08]]) + tRng.normal(size=(2, 3)) * 0.01
        tQuat = Rotation.from_rotvec(tRng.normal(size=(2, 3)) * 0.2).as_quat()
        tRes = tInf.solve_waypoint(tQ, tPos, tQuat, [0.3, 0.7])
        tRec = dict(tInf.last_wp)
        tRec.pop("t")
        tOut[tName] = [list(tRes), tRec, tInf._ik_used, tInf._lik_stop, tInf._shadow]
    tLines = (tmp_path / "ik.jsonl").read_text().splitlines()
    assert len(tLines) == 5 and all(json.loads(tL)["ik_backend_executed"] == "numerical" for tL in tLines)
    golden.check("core.solve_waypoint", tOut)


class _FakeSession:
    def __init__(self, reply):
        self.mReply = reply
        self.mCalls = []

    def post(self, url, json, timeout):
        self.mCalls.append((url, json, timeout))
        return SimpleNamespace(json=lambda: self.mReply)


def test_remote_forward_payload(monkeypatch, golden):
    tSess = _FakeSession({"act": [[0.1, 0.2], [0.3, 0.4]], "ms": 12.0, "gpu_ms": 8.0, "rtc": {"gw": [1.0]}})
    monkeypatch.setattr(core, "_REMOTE_SESS", tSess)
    monkeypatch.setattr(core, "REMOTE_INFER", "http://node:8790")
    monkeypatch.setattr(core, "REMOTE_CKPT", "/srv/ckpt")
    monkeypatch.setattr(core, "GLOBAL_ROT180", True)
    monkeypatch.setattr(core, "REMOTE_LAST", {})
    tAct = core.remote_forward({"global": "G", "left_wrist": "L"}, np.arange(20) * 0.1, "stack", torch.ones(1, 2, 3),
                               rtc={"Y": np.zeros((2, 3)), "W": np.ones((2, 3))})
    assert tAct.dtype == np.float64
    tUrl, tPayload, tTimeout = tSess.mCalls[0]
    assert set(core.REMOTE_LAST) == {"rt_ms", "server_ms", "gpu_ms", "rtc_log"} and core.REMOTE_LAST["server_ms"] == 12.0
    golden.check("core.remote_forward", [tAct, tUrl, tPayload, tTimeout])
    monkeypatch.setattr(core, "_REMOTE_SESS", _FakeSession({"error": "boom"}))
    with pytest.raises(RuntimeError, match="remote inference failed"):
        core.remote_forward({}, np.zeros(20), "t")


def test_curobo_solve_request(monkeypatch, golden):
    tSess = _FakeSession({"q": [[0.1] * 12], "pos_err_mm": [[1.0, 2.0]], "success": [[True, False]], "ms": 5, "rot_err_deg": [0.3]})
    monkeypatch.setattr(core, "_CUROBO_SESS", tSess)
    tRes = core.curobo_solve(np.arange(12) * 0.1, np.zeros((1, 2, 3)), np.ones((1, 2, 4)))
    assert tRes[0].dtype == np.float64 and tRes[2].dtype == bool
    golden.check("core.curobo_solve", [list(tRes), tSess.mCalls])
