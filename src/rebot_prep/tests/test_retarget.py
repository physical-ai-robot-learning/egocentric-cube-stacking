"""Characterization of the retarget-v2 stage on synthetic inputs: the multi-seed retarget core (workspace prescreen,
rollouts, TR reference windows), the comparison metrics, the rescued-tail diagnostic, the hybrid master merge, the HRL80
pre-registered report and the kinematic seed bank (fake reBot solver)."""
from __future__ import annotations

import importlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from rebot_prep.tests import synthetic_phase3 as P3S
from rebot_prep.tests import synthetic_retarget as SR
from rebot_prep.tests.golden import assertGolden
from rebot_prep.tests.scriptdefs import loadDefs

GOLDEN = Path(__file__).with_name("golden")
RETARGET_DIR = Path(__file__).resolve().parents[1] / "retarget"
COMPARE = "rebot_prep.retarget.v2k_compare"
TAIL = "rebot_prep.retarget.v2k_tail"
HYBRID = "rebot_prep.retarget.hybrid_merge"
HRL80 = "rebot_prep.retarget.hrl80_prereg_report"
SEEDBANK = "rebot_prep.retarget.v2k_seedbank"
OLD_EPISODES = [f"20260917_15054{k}_{n}" for k in range(3) for n in range(2)]
HRL_EPISODES = ["20260928_101010_0", "20260928_101010_1", "20260928_101010_2"]


def _round(a, nd=9):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def _run(name: str, argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", [name, *argv])
    try:
        runpy.run_module(name, run_name="__main__")
    except SystemExit as tExit:
        assert tExit.code in (0, None)


@pytest.fixture
def refTree(roots):
    P3S.writeReference(roots["UMI_BRIDGE"] / "track_c/data/r150_q_tcp.npz")
    tV2k = roots["UMI_BRIDGE"] / "track_c/v2k"
    tV2k.mkdir(parents=True)
    (tV2k / "r30_episodes.json").write_text(json.dumps({"episodes": [0, 3, 7, 11, 20]}))
    return roots


def _freshCompare(argv: list[str], monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", *argv])
    monkeypatch.delitem(sys.modules, COMPARE, raising=False)
    return importlib.import_module(COMPARE)


@pytest.mark.parametrize("evalRef", ["r150full", "heldout120"])
def test_compare_summary(refTree, tmp_path, monkeypatch, capsys, evalRef):
    tDir = tmp_path / "v2k_old"
    SR.writeRetargetDir(tDir, OLD_EPISODES, seed=1, withV2r=True)
    tParity = tmp_path / "baseline_offline"
    for tK, tE in enumerate(OLD_EPISODES[:4]):
        SR.writeOfflineCheck(tParity / f"{tE}.json", tK)
    tOut = tmp_path / "cmp.json"
    _run(COMPARE, ["--group", f"old259_v2k={tDir}", "--group", f"plain={tDir}", "--parity-offline", str(tParity), "--out", str(tOut),
                   "--eval-ref", evalRef], monkeypatch)
    assert capsys.readouterr().out.strip() == tOut.read_text().strip()
    assertGolden(GOLDEN, f"compare_{evalRef}", json.loads(tOut.read_text()))


def test_compare_manifold_of_nothing_is_empty(refTree, monkeypatch):
    assert _freshCompare([], monkeypatch).manifold([], []) == {}


def test_rescued_tail_report(refTree, tmp_path, monkeypatch, capsys):
    tDir = tmp_path / "v2k_old"
    SR.writeRetargetDir(tDir, OLD_EPISODES, seed=2, withV2r=False)
    _run(TAIL, [str(tDir)], monkeypatch)
    assertGolden(GOLDEN, "rescued_tail", capsys.readouterr().out.splitlines())


def _hybrid(roots, monkeypatch, capsys, episodes, tr, kr, out, seed):
    SR.writeRetargetDir(tr, episodes, seed=seed, withV2r=False)
    SR.writeRetargetDir(kr, episodes, seed=seed + 100, withV2r=True)
    for tK, tE in enumerate(episodes):          # the merge pairs segments by start: give KR the TR starts
        tKr = json.loads((kr / f"{tE}.json").read_text())
        tKr["segments"] = [dict(s, start=t["start"]) for s, t in zip(tKr["segments"], json.loads((tr / f"{tE}.json").read_text())["segments"])]
        (kr / f"{tE}.json").write_text(json.dumps(tKr))
    monkeypatch.setenv("HM_TR", str(tr))
    monkeypatch.setenv("HM_KR", str(kr))
    monkeypatch.setenv("HM_OUT", str(out))
    _run(HYBRID, [], monkeypatch)
    return capsys.readouterr().out


def test_hybrid_master_merge(refTree, tmp_path, monkeypatch, capsys):
    tOut = tmp_path / "hybrid_master"
    tText = _hybrid(refTree, monkeypatch, capsys, OLD_EPISODES, tmp_path / "tr_old", tmp_path / "r30_old", tOut, seed=3)
    tFiles = {p.name: json.loads(p.read_text()) for p in sorted(tOut.glob("*.json"))}
    tNpz = {p.name: {k: _round(v, 6) for k, v in sorted(np.load(p).items())} for p in sorted(tOut.glob("*.npz"))}
    assertGolden(GOLDEN, "hybrid_merge", dict(out=tText.splitlines(), json=tFiles, npz_keys={k: sorted(v) for k, v in tNpz.items()},
                                              npz_sum={k: float(np.sum([np.sum(x) for x in v.values()])) for k, v in tNpz.items()}))


def test_hrl80_prereg_report(refTree, tmp_path, monkeypatch, capsys):
    tRl = refTree["DATA_ROOT"] / "robotlike"
    tRl.mkdir()
    tOld = OLD_EPISODES + [f"20260918_12000{k}_{n}" for k in range(3) for n in range(3)]
    _hybrid(refTree, monkeypatch, capsys, tOld, tmp_path / "tr_old", tmp_path / "r30_old", tRl / "master_old259_frozen", seed=5)
    _hybrid(refTree, monkeypatch, capsys, HRL_EPISODES, tmp_path / "tr_h", tmp_path / "r30_h", tRl / "hybrid_hrl80", seed=6)
    SR.writeLive(tRl / "baseline_live", tOld, seed=7)
    SR.writeLive(tRl / "hrl80_live", HRL_EPISODES, seed=8)
    for tK, tE in enumerate(tOld):
        SR.writeOfflineCheck(tRl / "baseline_offline" / f"{tE}.json", 20 + tK)
    tRaw = refTree["EGO_COLLECTOR_HOME"] / "datasets/human_handumi_raw/HRL80/HRL80_20260928_101010"
    for tK, tE in enumerate(HRL_EPISODES):
        SR.writeOfflineCheck(tRaw / f"episode_{tE.rsplit('_', 1)[1]}" / "derived/robot_like/offline_check.json", 40 + tK)
    _run(HRL80, ["--eval-ref", "heldout120"], monkeypatch)
    tRep = json.loads((tRl / "hrl80_prereg_report.json").read_text())
    assert capsys.readouterr().out.strip() == json.dumps(tRep, indent=1)
    assertGolden(GOLDEN, "hrl80_report", tRep)


# ---------------------------------------------------------------- multi-seed retarget core (phase3 context + fake solver)
RETARGET_NAMES = {"K_ROLL", "VARIANT_RAD", "MAX_VARIANTS", "WS_MIN", "D_HARD", "TR_T", "TR_ROT_M", "ws_batch", "rel_traj", "fk_T",
                  "rollout", "_feat", "build_windows", "solve_ref", "rollout_tr", "do_segment_tr", "do_segment"}


@pytest.fixture
def retargetCore(pipeline, roots, monkeypatch):
    from rebot_prep.tests.test_census import PHASE3
    P3S.buildTrackDirs(roots["UMI_BRIDGE"])
    P3S.buildC8(roots["DATA_ROOT"], roots["EGO_COLLECTOR_HOME"] / "datasets/human_handumi_raw/Hpilot")
    for tVar in ("C8_RUNS", "M3_BASE", "M3_VARIANT"):
        monkeypatch.delenv(tVar, raising=False)
    monkeypatch.delitem(sys.modules, PHASE3, raising=False)
    tP3 = importlib.import_module(PHASE3)
    tP3.init()
    from rebot_prep.census.c8_collision_v1 import Clearance
    tNs = loadDefs(RETARGET_DIR / "v2k_retarget.py", RETARGET_NAMES, {"__name__": "v2k_retarget_defs"})
    tNs.update(P3=tP3, P=tP3.P, MODE="v2k", LAMBDA=0.0, K_MANI=8, Q_TREE=None, BANK_D=None, WIN=dict(q=None, feat=None, tree=None))
    return tNs, Clearance(tP3.ik.kin), tP3


def _segment(p3, start: int = 11):
    from rebot_prep.tests.conftest import humanMotion
    return humanMotion(65, 0.2 + start / 100), humanMotion(65, 1.3 + start / 100)


def _pack(res):
    tV1, tV2 = res[0], res[1]
    tOut = dict(v1={k: (_round(v, 5) if k == "q" else v) for k, v in tV1.items()},
                v2={k: v for k, v in tV2.items() if k not in ("kept", "best_any")},
                kept=[[round(sc, 9), int(i), {k: (_round(v, 5) if k == "q" else v) for k, v in r.items()}] for sc, i, r in tV2["kept"]])
    if len(res) > 2:
        tOut["v2r"] = [[round(sc, 9), int(i)] for sc, i, r in res[2]["kept"]]
    return json.loads(json.dumps(tOut, default=float))


@pytest.mark.parametrize("mode", ["v2k", "v2kr", "r30"])
def test_multi_seed_segment(retargetCore, mode):
    tNs, tClr, tP3 = retargetCore
    from scipy.spatial import cKDTree
    tRng = np.random.default_rng(9)
    tBank = np.radians(P3S.ANCHOR_DEG12) + tRng.normal(0, 0.12, (12, 12))
    tD150 = np.load(Path(tP3.TC) / "data/r150_q_tcp.npz")
    tQTree = cKDTree(np.radians(np.c_[tD150["J"][:, :6], tD150["J"][:, 7:13]])[::3])
    tNs.update(BANK=tBank, BANK_T0=np.array([np.stack(tNs["fk_T"](q)) for q in tBank]), MODE=mode, LAMBDA=0.5 if mode != "v2k" else 0.0,
               Q_TREE=tQTree, BANK_D=tQTree.query(tBank)[0], K_ROLL=3, K_MANI=2)
    tTL, tTR = _segment(tP3)
    tRes = tNs["do_segment"](tTL, tTR, tClr)
    tWs = tNs["ws_batch"](tNs["BANK_T0"][:, 0], 0)
    assertGolden(GOLDEN, f"retarget_segment_{mode}", dict(res=_pack(tRes), ws=[tWs[0].tolist(), _round(tWs[1])],
                                                          rel=_round(tNs["rel_traj"](tTL)[::16])))


def test_tr_reference_window_segment(retargetCore, capsys):
    tNs, tClr, tP3 = retargetCore
    from scipy.spatial import cKDTree
    tD150 = dict(np.load(Path(tP3.TC) / "data/r150_q_tcp.npz"))
    tMask = np.isin(tD150["ep"], [0, 1])
    tNs.update(MODE="tr", LAMBDA=0.5, K_ROLL=3, Q_TREE=cKDTree(np.radians(np.c_[tD150["J"][tMask, :6], tD150["J"][tMask, 7:13]])))
    tNs["build_windows"](tD150, tMask)
    tTL, tTR = _segment(tP3, 40)
    tRes = tNs["do_segment_tr"](tTL, tTR, tClr)
    assertGolden(GOLDEN, "retarget_segment_tr", dict(res=_pack(tRes), out=capsys.readouterr().out.splitlines(),
                                                     feat=_round(tNs["_feat"](tNs["rel_traj"](tTL), tNs["rel_traj"](tTR)))))


# ---------------------------------------------------------------- seed bank
@pytest.fixture
def seedbank(pipeline, refTree, monkeypatch, tmp_path):
    (refTree["DATA_ROOT"] / "c8_collision_v1.py").write_text(P3S.SHIM.format(mod="rebot_prep.census.c8_collision_v1"))
    monkeypatch.delitem(sys.modules, SEEDBANK, raising=False)
    tMod = importlib.import_module(SEEDBANK)
    tOut = tmp_path / "bank"
    tOut.mkdir()
    (tOut / "r30_episodes.json").write_text(json.dumps({"episodes": list(range(1, 32))}))
    monkeypatch.setattr(tMod, "OUT", tOut)
    monkeypatch.setattr(tMod, "SOBOL_M", 7)
    return tMod, tOut


@pytest.mark.parametrize("ref", ["r150full", "r30"])
def test_seed_bank(seedbank, monkeypatch, capsys, ref):
    tMod, tOut = seedbank
    monkeypatch.setattr("sys.argv", ["v2k_seedbank"] + (["--ref", ref] if ref != "r150full" else []))
    tMod.main()
    tTag = "" if ref == "r150full" else "_R30"
    tMeta = json.loads((tOut / f"seedbank{tTag}.json").read_text())
    tMeta.pop("seconds")
    tPrinted = json.loads(capsys.readouterr().out)
    tPrinted.pop("seconds")
    assertGolden(GOLDEN, f"seedbank_{ref}", dict(meta=tMeta, printed=tPrinted, bank=_round(np.load(tOut / f"seedbank{tTag}_q12.npy"), 9)))
