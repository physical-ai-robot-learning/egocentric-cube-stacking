"""Characterization of the Gen-1 gripper chain on synthetic raw sessions: extract_grip (mcap -> per-frame ticks),
correct_grip (anchors, device span, aperture -> width_m) and the per-side C8 grip census (frame mapping, stored /
derived anchors, legacy regression), then the GPU-free episode census that reads its output."""
from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

from rebot_prep.grip import correct_grip, extract_grip
from rebot_prep.tests import synthetic_raw as S
from rebot_prep.tests.golden import assertGolden

GOLDEN = Path(__file__).with_name("golden")
GRIP_CENSUS = "rebot_prep.grip.c8_grip"
EPISODE_CENSUS = "rebot_prep.census.c8_census_pre"
LEGACY_JOB = ".claude/jobs/e649aef3/tmp"


def _npzDigest(path: Path) -> dict:
    tZ = np.load(path)
    return {k: [str(tZ[k].dtype), list(tZ[k].shape), np.round(tZ[k].astype(np.float64), 7).tolist()] for k in sorted(tZ.files)}


def _runModule(name: str, argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", [name, *argv])
    try:
        runpy.run_module(name, run_name="__main__")
    except SystemExit as tExit:
        assert tExit.code in (0, None)


@pytest.fixture
def tree(roots, monkeypatch):
    tRaw = roots["EGO_COLLECTOR_HOME"] / "datasets/human_handumi_raw/Hpilot"
    tRec = S.buildTree(tRaw, roots["DATA_ROOT"])
    monkeypatch.setattr(extract_grip, "RAW", tRaw)
    monkeypatch.setattr(correct_grip, "RAW", tRaw)
    return dict(raw=tRaw, c8=roots["DATA_ROOT"], home=roots["HOME"], rec=tRec)


def test_align_picks_the_nearest_sample_and_reports_the_error():
    tFrames = [(2, 300), (0, 100), (1, 210)]
    tGrip = [(95, 0.1, 10, 1.0), (205, 0.2, 20, 2.0), (260, 0.3, 30, 3.0), (330, 0.4, 40, 4.0)]
    tAl = extract_grip.align(tFrames, tGrip)
    assert tAl["video_frame"].tolist() == [0, 1, 2]
    assert tAl["raw"].tolist() == [10, 20, 40] and tAl["raw"].dtype == np.int32
    assert tAl["grip01"].dtype == np.float32 and tAl["load"].tolist() == [1.0, 2.0, 4.0]
    assert np.allclose(tAl["dt_ms"], [5e-6, 5e-6, 30e-6])
    assert extract_grip.align([], tGrip) is None and extract_grip.align(tFrames, []) is None
    assert extract_grip.SIDE_STREAM == {"left": "left_wrist", "right": "right_wrist"}


def test_read_episode_from_sensors_mcap(tree):
    tFrames, tGrip = extract_grip.read_episode(S.SESSION_MAIN, "0")
    assert sorted(tFrames) == ["left_wrist", "right_wrist"] and len(tFrames["left_wrist"]) == 90
    assert tFrames["left_wrist"][:2] == tree["rec"][f"{S.SESSION_MAIN}_0"]["frames"]["left_wrist"][:2]
    assert [g[2] for g in tGrip["left"]] == tree["rec"][f"{S.SESSION_MAIN}_0"]["raw"]["left"].tolist()
    assert tGrip["left"][0][3] == 0.0 and tGrip["left"][1][3] == 1.0
    assert extract_grip.read_episode(S.SESSION_EARLY, "2") is None


def _extractAndCorrect(tree, monkeypatch, capsys) -> dict:
    tItems = [dict(session_id=s, episode=e, episode_id=f"{s}_{e}") for s, e in
              [(S.SESSION_MAIN, "0"), (S.SESSION_MAIN, "1"), (S.SESSION_MAIN, "2"), (S.SESSION_EARLY, "0"), (S.SESSION_EARLY, "1"),
               (S.SESSION_EARLY, "2"), (S.SESSION_EARLY, "3")]]
    tManifest = tree["home"] / "manifest.json"
    tManifest.write_text(json.dumps({"items": tItems}))
    tGrip, tCorrected = tree["home"] / LEGACY_JOB / "handumi_grip", tree["home"] / LEGACY_JOB / "handumi_grip_corrected"
    monkeypatch.setattr("sys.argv", ["extract_grip", "--manifest", str(tManifest), "--out", str(tGrip)])
    extract_grip.main()
    tExtractOut = capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["correct_grip", "--grip", str(tGrip), "--out", str(tCorrected)])
    correct_grip.main()
    tCorrectOut = capsys.readouterr().out
    return dict(grip=tGrip, corrected=tCorrected, extract_out=tExtractOut.replace(str(tree["home"]), "<home>").splitlines(),
                correct_out=tCorrectOut.replace(str(tree["home"]), "<home>").splitlines())


def test_extract_and_correct_grip_outputs(tree, monkeypatch, capsys):
    tRun = _extractAndCorrect(tree, monkeypatch, capsys)
    tContract = json.loads((tRun["corrected"] / "gripper_contract.json").read_text())
    assert tContract["aperture_m"] == correct_grip.APERTURE_M == 0.0675 and correct_grip.APERTURE_UNCERTAINTY_M == 0.005
    assertGolden(GOLDEN, "extract_correct_grip", dict(
        extract_out=tRun["extract_out"], correct_out=tRun["correct_out"], contract=tContract,
        extracted={p.name: _npzDigest(p) for p in sorted(tRun["grip"].glob("*.npz"))},
        corrected={p.name: _npzDigest(p) for p in sorted(tRun["corrected"].glob("*.npz"))}))


def test_session_meta_ticks_reads_every_session(tree):
    assert correct_grip.session_meta_ticks() == {s: {k: tuple(v) for k, v in S.TICKS[s].items()} for s in sorted(S.TICKS)}


def test_grip_census_and_episode_census(tree, monkeypatch, capsys):
    tRun = _extractAndCorrect(tree, monkeypatch, capsys)
    tContractPath = tRun["corrected"] / "gripper_contract.json"
    tContract = json.loads(tContractPath.read_text())
    tContract["anchors"] = [a for a in tContract["anchors"] if a["session"] == S.SESSION_MAIN]   # early sessions: derived
    tContractPath.write_text(json.dumps(tContract))
    _runModule(GRIP_CENSUS, [], monkeypatch)
    tFirst = capsys.readouterr().out
    tEpisode = f"{S.SESSION_MAIN}_1"                   # make the legacy width identical for one episode, then re-run
    tLegacy = dict(np.load(tRun["corrected"] / f"{tEpisode}.npz"))
    for tSide in ("left", "right"):
        tNew = np.load(tree["c8"] / "grip" / f"{tEpisode}_{tSide}.npz")
        tLegacy[f"{tSide}_video_frame"], tLegacy[f"{tSide}_width_m"] = tNew["video_frame"], tNew["width_m"]
    np.savez(tRun["corrected"] / f"{tEpisode}.npz", **tLegacy)
    _runModule(GRIP_CENSUS, [], monkeypatch)
    tSecond = capsys.readouterr().out
    tCensus = json.loads((tree["c8"] / "grip" / "grip_census.json").read_text())
    _runModule(EPISODE_CENSUS, [], monkeypatch)
    tPre = capsys.readouterr().out
    assertGolden(GOLDEN, "grip_census", dict(
        first=tFirst.splitlines(), second=tSecond.splitlines(), census=tCensus,
        npz={p.name: _npzDigest(p) for p in sorted((tree["c8"] / "grip").glob("*.npz"))},
        pre_out=tPre.splitlines(), census_pre=json.loads((tree["c8"] / "census_pre.json").read_text())))
