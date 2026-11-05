"""Characterization of one-episode conversion (raw episode -> processed CART20 episode) and the raw / processed episode
file formats."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ego_cart20.config import Cart20Config
from ego_cart20.convert import convertEpisode
from ego_cart20 import processed_episode as PW
from ego_cart20 import raw_episode as RL

from .synthetic import rawArrays, rawMeta, writeRawEpisode, writeVideo

ARRAY_KEYS = ("timestamps", "row_valid", "train_rows", "state", "state_prevrel", "state_prev_valid", "cart20", "action",
              "left_pose", "right_pose", "left_gripper", "right_gripper", "stage_id", "camera_valid")


def _raw(tmp_path, name="ep"):
    writeRawEpisode(tmp_path / name, name, "RBP", 0, videos={"head": "/videos/head.mp4"})
    return RL.loadRawEpisode(tmp_path / name)


def test_raw_episode_round_trip(golden, tmp_path):
    tRaw = _raw(tmp_path)
    assert RL.CAMERAS == ("head", "left_wrist", "right_wrist")
    assert RL.REQUIRED_META == ("episode_id", "instruction", "stack_order", "task")
    golden.check("rl.t0_ns", tRaw.mT0Ns)
    golden.check("rl.meta", tRaw.mMeta)
    assert tRaw.mEpisodeId == "ep" and sorted(tRaw.mCams) == ["head", "left_wrist"]
    for tSide in ("left", "right"):
        tArm = tRaw.mArms[tSide]
        golden.check(f"rl.{tSide}_t_s", tArm.mTimeS)
        golden.check(f"rl.{tSide}_T", tArm.mPose)
        golden.check(f"rl.{tSide}_valid", tArm.mValid)
        golden.check(f"rl.{tSide}_gripper", tArm.mGripper)
    golden.check("rl.cam_head_t", tRaw.mCams["head"][0])
    golden.check("rl.cam_head_frame", tRaw.mCams["head"][1])
    with pytest.raises(FileExistsError):
        RL.saveRawEpisode(tmp_path / "ep", {}, {})
    tMeta = rawMeta("bad")
    tMeta["instruction"] = ""
    RL.saveRawEpisode(tmp_path / "bad", tMeta, rawArrays(0))
    with pytest.raises(ValueError, match=r"raw_episode\.json missing \['instruction'\]"):
        RL.loadRawEpisode(tmp_path / "bad")


def _checkEpisode(golden, prefix, ep):
    for tKey in ARRAY_KEYS:
        golden.check(f"{prefix}.{tKey}", ep[tKey])
    golden.check(f"{prefix}.cameras", ep["cameras"])
    golden.check(f"{prefix}.metadata", ep["metadata"])
    golden.check(f"{prefix}.anchor_left", ep["_anchor"]["left"])
    golden.check(f"{prefix}.anchor_right", ep["_anchor"]["right"])
    assert sorted(ep["_tracks"]) == ["left", "right"]
    golden.check(f"{prefix}.tracks_valid_right", ep["_tracks"]["right"].mValid)
    assert list(ep) == list(ARRAY_KEYS[:13]) + ["camera_valid", "cameras", "metadata", "_anchor", "_tracks"]


def test_convert_episode_default(golden, tmp_path):
    _checkEpisode(golden, "cv.default", convertEpisode(_raw(tmp_path)))


def test_convert_episode_v2b_filter(golden, tmp_path):
    tCfg = Cart20Config(mDiscontinuityFilter=True, mJumpWindow="frame", mJumpIsolatedStepM=0.05)
    _checkEpisode(golden, "cv.v2b", convertEpisode(_raw(tmp_path), tCfg))


def test_convert_episode_custom_config(golden, tmp_path):
    tCfg = Cart20Config(mRequiredCameras=("head", "left_wrist"), mImageTolS=0.005, mHorizon=8, mMaxRawGapS=0.04)
    _checkEpisode(golden, "cv.custom", convertEpisode(_raw(tmp_path), tCfg))


def test_processed_writer_round_trip(golden, tmp_path):
    tEp = convertEpisode(_raw(tmp_path))
    tOut = PW.writeEpisode(tmp_path / "out" / "ep", tEp)
    assert tOut == tmp_path / "out" / "ep"
    assert PW.ARRAYS == ARRAY_KEYS
    golden.check("pw.files", sorted(str(p.relative_to(tOut)) for p in tOut.rglob("*")))
    tMeta = json.loads((tOut / "metadata.json").read_text())
    golden.check("pw.metadata", tMeta)
    golden.check("pw.head_source", json.loads((tOut / "head" / "source.json").read_text()))
    with pytest.raises(FileExistsError):
        PW.writeEpisode(tOut, tEp)
    PW.writeEpisode(tOut, tEp, overwrite=True)
    for tMmap in (True, False):
        tBack = PW.loadEpisode(tOut, mmap=tMmap)
        for tKey in ARRAY_KEYS:
            np.testing.assert_array_equal(np.asarray(tBack[tKey]), tEp[tKey])
        assert tBack["metadata"] == tMeta
        assert sorted(tBack["cameras"]) == ["head", "left_wrist"]
        np.testing.assert_array_equal(tBack["cameras"]["head"]["frame_index"], tEp["cameras"]["head"]["frame_index"])
        assert tBack["cameras"]["head"]["video"] == "/videos/head.mp4"


def test_encode_row_video(tmp_path):
    import av
    tSrc = writeVideo(tmp_path / "src.mp4", 12)
    tIndex = np.array([3, -1, 0, 3, 11, 5])
    PW.encodeRowVideo(tSrc, tIndex, tmp_path / "rows.mp4", size=32, fps=15)
    with av.open(str(tmp_path / "rows.mp4")) as tC:
        tFrames = [tF.to_ndarray(format="rgb24") for tF in tC.decode(video=0)]
    assert len(tFrames) == 6 and tFrames[0].shape == (32, 32, 3)
    assert tFrames[1].max() < 20                       # -1 row = black frame
    assert abs(int(tFrames[0][24, 24, 1]) - 21) < 6 and abs(int(tFrames[4][24, 24, 1]) - 77) < 6   # grey 7 * frame index
