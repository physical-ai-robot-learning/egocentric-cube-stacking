"""Characterization of relonly/train_relonly_adamw8bit.py: the AdamW8bit shim's checkpoint-safe state_dict / load_state_dict and
the hand-over to the REL-only entry, with a stand-in for bitsandbytes (python-int step, one qmap tensor shared by all params)."""
from __future__ import annotations

import importlib
import runpy
import sys
import types
from pathlib import Path

import pytest
import torch


class FakeAdamW8bit(torch.optim.Optimizer):
    QMAP = torch.linspace(-1.0, 1.0, 8)

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self, closure=None):
        for tGroup in self.param_groups:
            for tP in tGroup["params"]:
                tState = self.state[tP]
                if not tState:
                    tState["step"] = 0
                    tState["state1"] = torch.zeros_like(tP)
                    tState["qmap1"] = self.QMAP
                tState["step"] += 1
                tState["state1"].add_(tP.grad)
                tP.data.add_(tP.grad, alpha=-tGroup["lr"])


@pytest.fixture
def shimModule(fakeLerobot, monkeypatch):
    tBnb = types.ModuleType("bitsandbytes")
    tBnb.__version__ = "0.45.5"
    tBnb.optim = types.SimpleNamespace(AdamW8bit=FakeAdamW8bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", tBnb)
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.AdamW)          # restored after the test
    tRuns = []
    monkeypatch.setattr(runpy, "run_path", lambda path, run_name=None: tRuns.append((path, run_name)))
    import ego_paths
    monkeypatch.setattr(ego_paths, "SHARED_ROOT", Path("/shared"))
    monkeypatch.setattr(sys, "argv", ["train_relonly_adamw8bit.py", "--steps=1"])
    tPath = list(sys.path)
    tMod = importlib.import_module("training.relonly.train_relonly_adamw8bit")
    tMod.runs, tMod.addedPath = tRuns, [tP for tP in sys.path if tP not in tPath]
    sys.path[:] = tPath
    return tMod


def test_wrapper_replaces_adamw_and_runs_the_relonly_entry(shimModule):
    assert torch.optim.AdamW is shimModule.AdamW8bitShim
    assert shimModule.runs == [("/shared/relonly/code/train_rel16_relonly.py", "__main__")]
    assert sys.argv == ["/shared/relonly/code/train_rel16_relonly.py", "--steps=1"]
    assert shimModule.addedPath == ["/shared/relonly/pylib_bnb"]


def test_state_dict_tensor_steps_and_unshared_qmaps(shimModule, golden, capsys):
    torch.manual_seed(0)
    tParams = [torch.nn.Parameter(torch.randn(3)), torch.nn.Parameter(torch.randn(2))]
    tOpt = shimModule.AdamW8bitShim(tParams, lr=0.1, weight_decay=0.0)
    with pytest.raises(AssertionError, match="amsgrad not supported"):
        shimModule.AdamW8bitShim(tParams, amsgrad=True)
    for _ in range(2):
        for tP in tParams:
            tP.grad = torch.ones_like(tP)
        tOpt.step()
    tSd = tOpt.state_dict()
    assert all(isinstance(tS["step"], torch.Tensor) and tS["step"].dtype == torch.int64 for tS in tSd["state"].values())
    assert all(isinstance(tS["step"], int) for tS in tOpt.state.values())                 # live state untouched
    tPtrs = [tS["qmap1"].untyped_storage().data_ptr() for tS in tSd["state"].values()]
    assert len(set(tPtrs)) == 2 and tPtrs[0] == FakeAdamW8bit.QMAP.untyped_storage().data_ptr()
    tNew = shimModule.AdamW8bitShim([torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))])
    tNew.load_state_dict(tSd)
    assert sorted(tS["step"] for tS in tNew.state.values()) == [2, 2]
    assert all(isinstance(tS["step"], int) for tS in tNew.state.values())
    golden.check("adamw8bit/state_dict", {str(tK): tV for tK, tV in tSd["state"].items()})
    golden.check("adamw8bit/stdout", capsys.readouterr().out)
