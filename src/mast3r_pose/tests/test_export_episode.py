"""Characterization of the HandUMI -> UMI/ORB-SLAM3 episode exporter: telemetry JSON, settings YAML, video, printed log,
on a fake RawEpisode (frames + IMU) with the repository's frozen calibration files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mast3r_pose import export_episode
from mast3r_pose.tests.golden import assertGolden

GOLDEN = Path(__file__).with_name("golden")
REPO_CALIBRATION = Path(__file__).resolve().parents[3] / "configs" / "calibration"
N_FRAMES = 20
FRAME_NS = 33_333_333
T0_NS = 5_000_000_000
IMU_PERIOD_US = 5000
DOWNSCALE = 4


class FakeEpisode:
    def __init__(self, imuLeadNs: int) -> None:
        tN = 400
        tDevice = np.arange(tN, dtype=np.int64) * IMU_PERIOD_US + 1_000_000
        tHost = T0_NS - imuLeadNs + (tDevice - tDevice[0]) * 1000 + (np.arange(tN) % 3) * 200_000
        tRng = np.random.default_rng(1)
        self.imu = {s: SimpleNamespace(device_us=tDevice, host_ns=tHost, accel=tRng.normal(0, 1, (tN, 3)) + [0, 0, 9.8],
                                       gyro=tRng.normal(0, 0.1, (tN, 3))) for s in ("left", "right")}

    def iter_frames(self, stream, downscale=1):
        tW, tH = 1920 // downscale, 1080 // downscale
        for tI in range(N_FRAMES):
            tImg = np.zeros((tH, tW, 3), np.uint8)
            tImg[:, : (tI + 1) * 10] = 200
            yield tI, tI, T0_NS + tI * FRAME_NS, tImg


def _run(tmp_path, monkeypatch, capsys, side: str, imuLeadNs: int, extra: list[str]) -> dict:
    tEp = tmp_path / "episode_0"
    tEp.mkdir()
    (tEp / "events.json").write_text(json.dumps([{"kind": "go_cue", "t_ns": T0_NS + 5 * FRAME_NS - 10, "t_rel_s": 3.0}]))
    monkeypatch.setattr(export_episode, "CAL", REPO_CALIBRATION)
    monkeypatch.setattr(export_episode.RawEpisode, "load", classmethod(lambda cls, p: FakeEpisode(imuLeadNs)))
    tOut = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["export_episode", str(tEp), side, str(tOut), "--downscale", str(DOWNSCALE), *extra])
    export_episode.main()
    tCap = cv2.VideoCapture(str(tOut / "raw_video.mp4"))
    tFrames = int(tCap.get(cv2.CAP_PROP_FRAME_COUNT))
    tCap.release()
    tTel = json.loads((tOut / "imu_data.json").read_text())
    return dict(stdout=capsys.readouterr().out.replace(str(tOut), "<out>").splitlines(),
                setting=(tOut / "orbslam_setting.yaml").read_text().splitlines(),
                telemetry_sha256=hashlib.sha256(json.dumps(tTel, sort_keys=True).encode()).hexdigest(),
                cori_first=tTel["1"]["streams"]["CORI"]["samples"][0]["cts"], n_cori=len(tTel["1"]["streams"]["CORI"]["samples"]),
                n_accl=len(tTel["1"]["streams"]["ACCL"]["samples"]), video_frames=tFrames)


@pytest.mark.parametrize("side", ["left", "right"])
def test_export_with_imu_preroll_available(tmp_path, monkeypatch, capsys, side):
    assertGolden(GOLDEN, f"export_{side}_preroll_ok", _run(tmp_path, monkeypatch, capsys, side, 200_000_000, []))


def test_export_drops_leading_frames_when_the_imu_starts_late(tmp_path, monkeypatch, capsys):
    assertGolden(GOLDEN, "export_left_drop_frames", _run(tmp_path, monkeypatch, capsys, "left", 10_000_000, []))


def test_export_start_at_go_trims_before_the_go_cue(tmp_path, monkeypatch, capsys):
    assertGolden(GOLDEN, "export_right_start_at_go",
                 _run(tmp_path, monkeypatch, capsys, "right", 200_000_000, ["--start-at-go", "--imu-preroll-ms", "20"]))


def test_settings_yaml_inverts_the_camera_imu_transform():
    tT = np.eye(4)
    tT[:3, 3] = [0.1, 0.2, 0.3]
    tText = export_episode.settings_yaml("left", 480, 270, np.diag([300.0, 310.0, 1.0]), [0.1, 0.2, 0.3, 0.4], tT,
                                         dict(gyroscope_noise_density=1e-3, accelerometer_noise_density=2e-3,
                                              gyroscope_random_walk=3e-5, accelerometer_random_walk=4e-5, update_rate=200.0),
                                         29.97)
    assert "Camera.fps: 30" in tText
    assert "data: [1.0000000000, 0.0000000000, 0.0000000000, -0.1000000000," in tText
    assertGolden(GOLDEN, "settings_yaml", tText.splitlines())
