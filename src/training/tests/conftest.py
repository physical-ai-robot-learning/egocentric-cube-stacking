"""Test fixtures of the training package: a golden-value recorder, stand-ins for the LeRobot X-VLA modules the plugins patch
(LeRobot is not installed in the test environment), a synthetic dual-arm URDF for the FK modules, and a CLI argument catcher.

`golden`       compares every value a test hands it against tests/golden_training.npz, exactly (array_equal, NaN == NaN).
               With TRAINING_CHAR_RECORD=1 the values are recorded instead (the characterization run against the pre-refactor
               code).
`fakeLerobot`  installs fake `lerobot.*` modules (X-VLA soft transformer / action hub / policy / factory / lerobot_train) for one
               test, purges every `training.*` module before and after, so each test imports the plugins fresh (they keep
               module-level counters and patch classes on install).
`synthUrdf`    writes a small dual-arm URDF with the joint names rebot_fk expects and points REBOT_URDF at it.
`parsedArgs`   runs a CLI entry and returns the argparse Namespace it parsed, stopping right after parse_args.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

GOLDEN_FILE = Path(__file__).with_name("golden_training.npz")
RECORD = os.environ.get("TRAINING_CHAR_RECORD") == "1"
REPO_SRC = Path(__file__).resolve().parents[2]


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for tKey in sorted(value, key=str):
            _flatten(f"{prefix}/{tKey}", value[tKey], out)
    elif isinstance(value, (list, tuple)) and not (value and all(isinstance(tX, (int, float, bool, np.number)) for tX in value)):
        out[f"{prefix}#len"] = np.asarray(len(value))
        for tIndex, tItem in enumerate(value):
            _flatten(f"{prefix}[{tIndex}]", tItem, out)
    elif torch.is_tensor(value):
        out[prefix] = value.detach().cpu().numpy()
    elif value is None:
        out[prefix] = np.asarray("<None>")
    elif isinstance(value, (str, Path)):
        out[prefix] = np.asarray(str(value))
    else:
        out[prefix] = np.asarray(value)


class Golden:
    def __init__(self):
        self.mStore = dict(np.load(GOLDEN_FILE, allow_pickle=False)) if GOLDEN_FILE.exists() else {}
        self.mNew = {}

    def check(self, name, value):
        tFlat = {}
        _flatten(name, value, tFlat)
        if RECORD:
            self.mNew.update(tFlat)
            return
        tExpectedKeys = sorted(tK for tK in self.mStore if tK == name or tK.startswith(name + "/") or tK.startswith(name + "[")
                               or tK.startswith(name + "#"))
        assert sorted(tFlat) == tExpectedKeys, f"{name}: keys differ: {sorted(set(tFlat) ^ set(tExpectedKeys))[:10]}"
        for tKey, tValue in tFlat.items():
            tExp = self.mStore[tKey]
            assert tValue.shape == tExp.shape, f"{tKey}: shape {tValue.shape} != {tExp.shape}"
            if tExp.dtype.kind in "fc" or tValue.dtype.kind in "fc":
                assert np.array_equal(tValue.astype(np.float64), tExp.astype(np.float64), equal_nan=True), \
                    f"{tKey}: max diff {np.nanmax(np.abs(tValue.astype(np.float64) - tExp.astype(np.float64)))}"
            else:
                assert np.array_equal(tValue, tExp), f"{tKey}: {tValue!r} != {tExp!r}"


_GOLDEN = Golden()


@pytest.fixture(scope="session")
def golden():
    yield _GOLDEN
    if RECORD and _GOLDEN.mNew:
        tAll = dict(_GOLDEN.mStore)
        tAll.update(_GOLDEN.mNew)
        np.savez_compressed(GOLDEN_FILE, **tAll)


# ---------------------------------------------------------------- stand-ins for the LeRobot X-VLA modules
class FakeDecoder(torch.nn.Module):
    """action_decoder: fc.weight (domains, I * O), bias.weight (domains, O) like X-VLA's domain-indexed linear"""

    def __init__(self, inputSize=4, outputSize=32, domains=2):
        super().__init__()
        self.input_size, self.output_size = inputSize, outputSize
        self.fc = torch.nn.Embedding(domains, inputSize * outputSize)
        self.bias = torch.nn.Embedding(domains, outputSize)


def _makeLerobot():
    tMods = {}

    def mod(name, **attrs):
        tM = types.ModuleType(name)
        tM.__dict__.update(attrs)
        tMods[name] = tM
        return tM

    class SoftPromptedTransformer:
        def __init__(self, *a, **kw):
            self.action_decoder = FakeDecoder()

        def forward(self, domain_id, action_with_noise, t=None):
            return action_with_noise * 2.0 + 1.0

    class AutoActionSpace:
        JOINTS_SCALE = 3.0

        def __init__(self, realDim=32):
            self.real_dim = realDim

        def _pad_to_model_dim(self, x):
            return x

        def compute_loss(self, pred, target):
            raise AssertionError("original compute_loss must be replaced")

    class XVLAPolicy:
        def __init__(self):
            self.model = SimpleNamespace(action_space=AutoActionSpace())
            self.seen = []

        def forward(self, batch):
            self.seen.append(batch)
            return "orig-forward"

    def make_pre_post_processors(*a, **kw):
        def pre(batch):
            return {tK: tV for tK, tV in batch.items() if tK in ("action", "observation.state")}
        return pre, "post"

    tCalls = {"main": 0}

    def lt_main():
        tCalls["main"] += 1

    mod("lerobot")
    mod("lerobot.policies")
    mod("lerobot.policies.xvla")
    mod("lerobot.policies.xvla.soft_transformer", SoftPromptedTransformer=SoftPromptedTransformer)
    mod("lerobot.policies.xvla.action_hub", AutoActionSpace=AutoActionSpace)
    mod("lerobot.policies.xvla.modeling_xvla", XVLAPolicy=XVLAPolicy)
    mod("lerobot.policies.factory", make_pre_post_processors=make_pre_post_processors)
    mod("lerobot.scripts")
    mod("lerobot.scripts.lerobot_train", main=lt_main, make_dataset=None, torch=torch, calls=tCalls)
    return tMods


# modules a test may import fresh: the package itself, the flat (node-layout) copies of its plugins, and the stand-ins
_FRESH_PREFIXES = ("training", "lerobot", "rel16_", "rebot_fk", "train_rel16", "zarr", "umi_action", "diffusion_policy",
                   "r150_v4_instructions")


def _isFresh(name):
    return any(name == tP or name.startswith(tP + ".") or (tP.endswith("_") and name.startswith(tP)) for tP in _FRESH_PREFIXES) \
        or name in ("rebot_fk", "rebot_fk_torch", "rel16_aux_relonly", "rel16_relonly_lossmask")


def _purgeTraining():
    for tName in [tN for tN in sys.modules if _isFresh(tN)]:
        del sys.modules[tName]


@pytest.fixture
def fakeLerobot(monkeypatch):
    """-> dict of fake modules by name; the training modules are imported fresh inside the test"""
    tBefore = dict(sys.modules)
    tPath = list(sys.path)
    _purgeTraining()
    tMods = _makeLerobot()
    for tName, tMod in tMods.items():
        monkeypatch.setitem(sys.modules, tName, tMod)
    try:
        yield tMods
    finally:
        _purgeTraining()
        sys.modules.update({tK: tV for tK, tV in tBefore.items() if not _isFresh(tK)})
        sys.path[:] = tPath


# ---------------------------------------------------------------- synthetic dual-arm URDF
_JOINT_AXES = ["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"]
_JOINT_XYZ = ["0 0 0.05", "0.01 0 0.08", "0.2 0.0 0.01", "0.15 0.01 0", "0.05 0 -0.01", "0.04 0.0 0.0"]
_JOINT_RPY = ["0 0 0", "0.1 0 0", "0 0.2 0", "0 0 0.3", "0.05 0.1 0", "0 0 0"]


def urdfText() -> str:
    tLines = ['<robot name="synthetic_dualarm">', '  <link name="base"/>']
    for tSide, tY, tYaw in (("left", 0.3, "0.1"), ("right", -0.3, "-0.2")):
        tLines.append(f'  <joint name="{tSide}_mount" type="fixed"><parent link="base"/><child link="{tSide}_l0"/>'
                      f'<origin xyz="0 {tY} 0.02" rpy="0 0 {tYaw}"/></joint>')
        for tI in range(6):
            tLines.append(f'  <joint name="{tSide}_joint{tI + 1}" type="revolute"><parent link="{tSide}_l{tI}"/>'
                          f'<child link="{tSide}_l{tI + 1}"/><origin xyz="{_JOINT_XYZ[tI]}" rpy="{_JOINT_RPY[tI]}"/>'
                          f'<axis xyz="{_JOINT_AXES[tI]}"/></joint>')
        tLines.append(f'  <joint name="{tSide}_gripper_joint" type="fixed"><parent link="{tSide}_l6"/>'
                      f'<child link="{tSide}_gripper_link"/><origin xyz="0.03 0 0" rpy="0 0.1 0"/></joint>')
    tLines.append("</robot>")
    return "\n".join(tLines) + "\n"


@pytest.fixture
def synthUrdf(tmp_path, monkeypatch):
    tPath = tmp_path / "synthetic_dualarm.urdf"
    tPath.write_text(urdfText())
    monkeypatch.setenv("REBOT_URDF", str(tPath))
    return tPath


# ---------------------------------------------------------------- argparse catcher
class _Parsed(BaseException):
    pass


@pytest.fixture
def parsedArgs(monkeypatch):
    """parsedArgs(fn, argv) -> Namespace: fn() parses argv and is stopped right after parse_args"""

    def run(fn, argv):
        tOrig = argparse.ArgumentParser.parse_args
        tBox = {}

        def parse_args(self, args=None, namespace=None):
            tBox["ns"] = tOrig(self, argv, namespace)
            raise _Parsed()

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
        monkeypatch.setattr(sys, "argv", ["prog"] + list(argv))
        try:
            fn()
        except _Parsed:
            pass
        finally:
            monkeypatch.setattr(argparse.ArgumentParser, "parse_args", tOrig)
        return tBox["ns"]

    return run
