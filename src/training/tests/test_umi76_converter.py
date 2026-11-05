"""Characterization of umi76/umi76_to_lerobot.py (UMI zarr -> LeRobot UMI76 converter) with stand-ins for the node-only modules
(zarr, umi_action, diffusion_policy codecs, r150_v4_instructions, LeRobotDataset.create); the UMI pose helpers are the repository
copy (src/umi). A synthetic two-episode zarr and a small global mp4 drive the whole main() loop."""
from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as Rot

N_RAW = 40                    # raw 30 Hz frames per synthetic episode
VIDEO_HW = (32, 48)


class FakeArray:
    def __init__(self, data):
        self.mData = np.asarray(data)
        self.oindex = self.mData

    def __getitem__(self, key):
        return self.mData[key]


def makeZarr(seed=0, nEpisodes=2):
    tRng = np.random.default_rng(seed)
    tN = N_RAW * nEpisodes
    tT = np.arange(tN) / 30.0
    tData = {}
    for tR in (0, 1):
        tPos = np.stack([0.3 + 0.05 * np.sin(tT * (1 + tR)), 0.1 * np.cos(tT), 0.2 + 0.02 * tT], -1)
        tRv = np.stack([0.2 * np.sin(tT), 0.1 + 0.3 * tT, -0.2 * np.cos(tT * 0.5)], -1) + tR * 0.1
        tData[f"robot{tR}_eef_pos"] = FakeArray(tPos + tRng.normal(scale=1e-3, size=tPos.shape))
        tData[f"robot{tR}_eef_rot_axis_angle"] = FakeArray(tRv)
        tData[f"robot{tR}_gripper_width"] = FakeArray((0.05 + 0.03 * np.sin(tT * 2 + tR))[:, None])
        tData[f"camera{tR}_rgb"] = FakeArray(tRng.integers(0, 255, size=(tN, 4, 4, 3), dtype=np.uint8))
    return {"data": tData, "meta": {"episode_ends": FakeArray(np.arange(1, nEpisodes + 1) * N_RAW)}}


def writeVideo(path: Path, nFrames: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as tC:
        tS = tC.add_stream("mpeg4", rate=30)
        tS.height, tS.width = VIDEO_HW
        tS.pix_fmt = "yuv420p"
        for tI in range(nFrames):
            tIm = np.zeros(VIDEO_HW + (3,), np.uint8)
            tIm[: VIDEO_HW[0] // 2, :, 0] = (tI * 3) % 255
            tIm[:, : VIDEO_HW[1] // 3, 1] = 200 - tI
            tIm[VIDEO_HW[0] // 2:, VIDEO_HW[1] // 2:, 2] = 50 + tI
            for tP in tS.encode(av.VideoFrame.from_ndarray(tIm, format="rgb24")):
                tC.mux(tP)
        for tP in tS.encode():
            tC.mux(tP)


def writeSourceLerobot(root: Path, nEpisodes=2):
    writeVideo(root / "videos/observation.images.global/chunk-000/file-000.mp4", N_RAW * nEpisodes)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    tKey = "videos/observation.images.global"
    pd.DataFrame({"episode_index": list(range(nEpisodes))[::-1],
                  f"{tKey}/chunk_index": [0] * nEpisodes, f"{tKey}/file_index": [0] * nEpisodes,
                  f"{tKey}/from_timestamp": [tE * N_RAW / 30.0 for tE in range(nEpisodes)][::-1]}).to_parquet(
        root / "meta/episodes/chunk-000/file-000.parquet")
    return root


@pytest.fixture
def conv(fakeLerobot, monkeypatch, tmp_path):
    """the converter module imported under stand-ins; .zarrs maps a zarr path to its fake store, .created the fake datasets"""
    tZarrs, tCreated = {}, []
    tSourceRoot = writeSourceLerobot(tmp_path / "r150_lerobot")

    def mod(name, **attrs):
        tM = types.ModuleType(name)
        tM.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, tM)
        return tM

    def episode_table(root):
        return pd.DataFrame({"episode_index": [0, 1, 2, 3], "instruction": [f"task {tI} of {Path(root).name}" for tI in range(4)]})

    class FakeDataset:
        def __init__(self, **kw):
            self.mKw = kw
            self.mFrames, self.mSaves = [], []
            Path(kw["root"]).mkdir(parents=True)

        def add_frame(self, frame):
            self.mFrames.append(frame)

        def save_episode(self, **kw):
            self.mSaves.append(kw)

    class LeRobotDataset:
        @staticmethod
        def create(**kw):
            tCreated.append(FakeDataset(**kw))
            return tCreated[-1]

    mod("zarr", open=lambda path, mode: tZarrs[str(path)])
    mod("umi_action", CHUNK=16)
    mod("diffusion_policy")
    mod("diffusion_policy.codecs")
    mod("diffusion_policy.codecs.imagecodecs_numcodecs", register_codecs=lambda: None)
    mod("r150_v4_instructions", episode_table=episode_table, R150_LEROBOT=tSourceRoot)
    mod("lerobot.datasets")
    mod("lerobot.datasets.lerobot_dataset", LeRobotDataset=LeRobotDataset)
    tPath = list(sys.path)
    tM = importlib.import_module("training.umi76.umi76_to_lerobot")
    sys.path[:] = tPath
    tM.zarrs, tM.created, tM.sourceRoot = tZarrs, tCreated, tSourceRoot
    return tM


def test_body_delta_and_magnitudes(conv, golden):
    tRng = np.random.default_rng(4)
    tMats = np.tile(np.eye(4), (5, 1, 1))
    tMats[:, :3, :3] = Rot.from_rotvec(tRng.normal(scale=0.3, size=(5, 3))).as_matrix()
    tMats[:, :3, 3] = tRng.normal(scale=0.05, size=(5, 3))
    tD = conv.body_delta(tMats[1:], tMats[0])
    golden.check("conv/body_delta", tD)
    golden.check("conv/delta_magnitudes", conv.delta_magnitudes(tD))


def test_episode_frames_state76_and_action_chunk(conv, golden):
    tZ = makeZarr()
    tRec, tSt = conv.episode_frames(tZ, 0, N_RAW, 2)
    golden.check("conv/episode_frames/rec", tRec)
    golden.check("conv/episode_frames/stats", tSt)
    assert conv.episode_frames(tZ, 0, 25, 2) == (None, "too short for the umi76 window")
    tTau, tPos, tRot, tW = conv._traj(tZ, 1, N_RAW, 2 * N_RAW)
    golden.check("conv/sample", conv._sample(tTau, tPos, tRot, tW, [-0.1, 0.0, 0.0334, 0.5, 9.0]))


def test_features_and_source_spec(conv, golden, tmp_path):
    golden.check("conv/features", {tK: {tKk: (list(tVv) if isinstance(tVv, tuple) else tVv) for tKk, tVv in tV.items()}
                                   for tK, tV in conv.features().items()})
    (tmp_path / "z.zarr").mkdir()
    tSpec = conv._parse_source(f"zarr={tmp_path / 'z.zarr'}, lerobot={conv.sourceRoot},global_transform=rot180,")
    assert tSpec == {"zarr": str(tmp_path / "z.zarr"), "lerobot": str(conv.sourceRoot), "global_transform": "rot180",
                     "episodes_json": None, "select": None}
    for tBad, tMsg in ((f"zarr={tmp_path}/z.zarr,lerobot=/x,bogus=1", f"--source: unknown key(s) ['bogus'] in "),
                       (f"lerobot={conv.sourceRoot}", "--source: missing zarr= in "),
                       (f"zarr=/does/not/exist,lerobot={conv.sourceRoot}", "--source: zarr=/does/not/exist does not exist"),
                       (f"zarr={tmp_path}/z.zarr,lerobot={conv.sourceRoot},select=/no.json", "--source: select=/no.json does not exist")):
        with pytest.raises(BaseException) as tErr:
            conv._parse_source(tBad)
        assert str(tErr.value).startswith(tMsg)


def test_decode_global_rot180_and_errors(conv, golden):
    tMeta = conv._source_episode_meta(conv.sourceRoot)
    assert tMeta["episode_index"].tolist() == [0, 1]
    tRow = tMeta.iloc[1]
    tPlain = conv.decode_global(tRow, 12, out_res=(16, 8), root=conv.sourceRoot)
    tRot = conv.decode_global(tRow, 12, out_res=(16, 8), root=conv.sourceRoot, transform="rot180")
    assert tPlain.shape == (12, 8, 16, 3) and np.array_equal(tRot, tPlain[:, ::-1, ::-1])
    golden.check("conv/decode_global", tPlain)
    with pytest.raises(ValueError, match="unknown global transform 'mirror'"):
        conv.decode_global(tRow, 2, root=conv.sourceRoot, transform="mirror")
    with pytest.raises(RuntimeError, match=r"global short segment file-000.mp4: 40/41"):
        conv.decode_global(tRow, 41, root=conv.sourceRoot)


def test_cli_defaults(conv, parsedArgs, golden):
    tNs = parsedArgs(conv.main, ["--out", "/tmp/x"])
    golden.check("conv/cli_defaults", {tK: str(tV) for tK, tV in vars(tNs).items()})


def test_main_converts_two_sources(conv, tmp_path, monkeypatch, golden, capsys):
    tZa, tZb = tmp_path / "a.zarr", tmp_path / "b.zarr"
    tZa.mkdir()
    tZb.mkdir()
    conv.zarrs[str(tZa)] = makeZarr(seed=1)
    conv.zarrs[str(tZb)] = makeZarr(seed=2)
    (tmp_path / "eps.json").write_text(json.dumps({"episodes": [1, 0]}))
    (tmp_path / "sel.json").write_text(json.dumps({"zarr_episodes": [1]}))
    tOut = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["umi76_to_lerobot.py", "--out", str(tOut), "--repo-id", "rebot/t",
                                      "--source", f"zarr={tZa},lerobot={conv.sourceRoot}",
                                      "--source", f"zarr={tZb},lerobot={conv.sourceRoot},global_transform=rot180,"
                                                  f"episodes_json={tmp_path / 'eps.json'},select={tmp_path / 'sel.json'}"])
    conv.main()
    tDs = conv.created[-1]
    assert tDs.mKw["repo_id"] == "rebot/t" and tDs.mKw["fps"] == 15 and tDs.mKw["image_writer_threads"] == 8
    assert tDs.mSaves == [{"parallel_encoding": True}] * 3
    golden.check("conv/main/frames", [{tK: (tV if tK != "task" else str(tV)) for tK, tV in tF.items()} for tF in tDs.mFrames])
    golden.check("conv/main/provenance", (tOut / "v4_provenance.json").read_text().replace(str(tmp_path), "<tmp>"))
    tStdout = re.sub(r"\[\s*[0-9.]+s\]", "[t]", capsys.readouterr().out.replace(str(tmp_path), "<tmp>"))   # wall time
    golden.check("conv/main/stdout", tStdout)
