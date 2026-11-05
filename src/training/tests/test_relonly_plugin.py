"""Characterization of relonly/: the pinned REL-only plugin (rel16_aux_relonly), the loss-mask variant (rel16_relonly_lossmask)
and the two lerobot_train entries, on fake LeRobot X-VLA classes and small synthetic tensors."""
from __future__ import annotations

import hashlib
import importlib
import runpy
from pathlib import Path

import pytest
import torch

RELONLY_DIR = Path(__file__).resolve().parents[1] / "relonly"
# the AUX_SHA the node launchers (scripts/train/node4090/*relonly*.sh) verify before a run starts
PINNED_RELONLY_SHA256 = "3511f81aec809103ddc97c7440c3f111c08c90d3d7083b577fc9f5fc004a4bf3"


def test_relonly_plugin_bytes_match_the_launcher_pin():
    tData = (RELONLY_DIR / "rel16_aux_relonly.py").read_bytes()
    assert hashlib.sha256(tData).hexdigest() == PINNED_RELONLY_SHA256


def _actionSpace(mods):
    return mods["lerobot.policies.xvla.action_hub"].AutoActionSpace()


def test_zero_mask_zeroes_channels_20_32_in_and_out(fakeLerobot, golden, capsys):
    tR = importlib.import_module("training.relonly.rel16_aux_relonly")
    tR.install_zero_mask()
    tR.install_zero_mask()                                   # idempotent
    tSt = fakeLerobot["lerobot.policies.xvla.soft_transformer"].SoftPromptedTransformer
    tX = torch.arange(2 * 3 * 34, dtype=torch.float32).reshape(2, 3, 34) / 7.0
    tOut = tSt().forward(0, tX, t=0.5)
    assert (tOut[..., 20:32] == 0).all()
    assert torch.equal(tOut[..., :20], tX[..., :20] * 2.0 + 1.0)
    assert torch.equal(tOut[..., 32:], tX[..., 32:] * 2.0 + 1.0)
    golden.check("relonly/zero_mask_out", tOut)
    golden.check("relonly/zero_mask_counters", dict(tR._S))
    golden.check("relonly/zero_mask_stdout", capsys.readouterr().out)


def test_relonly_compute_loss_and_preflight_lines(fakeLerobot, golden, capsys, monkeypatch):
    monkeypatch.setenv("RELONLY_PREFLIGHT", "2")
    monkeypatch.setenv("REL16_LOG_EVERY", "2")
    tR = importlib.import_module("training.relonly.rel16_aux_relonly")
    tR.install()
    tSpace = _actionSpace(fakeLerobot)
    tGen = torch.Generator().manual_seed(3)
    tLosses = []
    for _ in range(3):
        tPred = torch.randn(2, 16, 32, generator=tGen)
        tPred[..., 20:32] = 0
        tTarget = torch.randn(2, 16, 32, generator=tGen)
        torch.manual_seed(11)
        tLosses.append(tSpace.compute_loss(tPred, tTarget)["rel_loss"])
    golden.check("relonly/rel_losses", torch.stack(tLosses))
    golden.check("relonly/rel_stdout", capsys.readouterr().out)


def test_lossmask_install_keeps_mask_and_masks_the_loss(fakeLerobot, golden, capsys, monkeypatch):
    monkeypatch.setenv("RELONLY_PREFLIGHT", "2")
    monkeypatch.setenv("REL16_LOG_EVERY", "1")
    tL = importlib.import_module("training.relonly.rel16_relonly_lossmask")
    tL.install()
    tFactory = fakeLerobot["lerobot.policies.factory"]
    tPre, tPost = tFactory.make_pre_post_processors()
    assert tPost == "post"
    tMask = torch.ones(2, 20)
    tMask[1, :10] = 0
    tMask[1, 19] = 0
    tBatch = tPre({"action": torch.zeros(2, 16, 32), "observation.state": torch.zeros(2, 20), "aux.loss_mask": tMask,
                   "other": 1})
    assert sorted(tBatch) == ["action", "aux.loss_mask", "observation.state"]
    tPolicy = fakeLerobot["lerobot.policies.xvla.modeling_xvla"].XVLAPolicy()
    assert tPolicy.forward(tBatch) == "orig-forward"
    tSpace = tPolicy.model.action_space
    golden.check("lossmask/stored_mask", tSpace._lossmask)
    golden.check("lossmask/dims_supervised", tL.COUNTERS["dims_supervised"])
    tGen = torch.Generator().manual_seed(5)
    tPred = torch.randn(2, 16, 32, generator=tGen)
    tPred[..., 20:32] = 0
    tTarget = torch.randn(2, 16, 32, generator=tGen)
    torch.manual_seed(12)
    tMasked = tSpace.compute_loss(tPred, tTarget)["rel_loss"]
    tPolicy.forward({"action": torch.zeros(2, 16, 32)})          # no mask -> plain REL-only mean
    assert tSpace._lossmask is None
    tPlain = tSpace.compute_loss(tPred, tTarget)["rel_loss"]
    golden.check("lossmask/losses", torch.stack([tMasked, tPlain]))
    golden.check("lossmask/counters", {tK: tV for tK, tV in tL.COUNTERS.items() if tK != "dims_supervised"})
    golden.check("lossmask/stdout", capsys.readouterr().out)


def test_lossmask_rejects_an_all_masked_batch(fakeLerobot):
    tL = importlib.import_module("training.relonly.rel16_relonly_lossmask")
    with pytest.raises(RuntimeError, match="no supervised action element"):
        tL.maskedRelLoss(torch.zeros(1, 2, 20), torch.zeros(1, 2, 20), torch.zeros(1, 20, dtype=torch.bool))


def _runEntry(name):
    with pytest.raises(SystemExit) as tExit:                         # sys.exit(main()) after lerobot_train returns
        runpy.run_path(str(RELONLY_DIR / name), run_name="__main__")
    assert tExit.value.code == 0


@pytest.mark.parametrize("entry", ["train_rel16_relonly.py", "train_rel16_relonly_lossmask.py"])
def test_train_entry_installs_plugin_and_calls_lerobot_main(fakeLerobot, monkeypatch, entry, capsys, golden):
    monkeypatch.syspath_prepend(str(RELONLY_DIR))                    # node layout: run as a flat script
    _runEntry(entry)
    assert fakeLerobot["lerobot.scripts.lerobot_train"].calls["main"] == 1
    golden.check(f"entry/{entry}/stdout", capsys.readouterr().out)


@pytest.mark.parametrize("entry", ["train_rel16_relonly.py", "train_rel16_relonly_lossmask.py"])
def test_train_entry_preflight_gradient_hooks(fakeLerobot, monkeypatch, entry, capsys, golden):
    monkeypatch.setenv("RELONLY_PREFLIGHT", "1")
    monkeypatch.syspath_prepend(str(RELONLY_DIR))
    _runEntry(entry)
    tSt = fakeLerobot["lerobot.policies.xvla.soft_transformer"].SoftPromptedTransformer
    tModel = tSt()
    tDec = tModel.action_decoder
    if entry.endswith("lossmask.py"):
        import rel16_relonly_lossmask as tPlugin                      # the flat-imported copy the entry installed
        tSup = torch.ones(20, dtype=torch.bool)
        tSup[:3] = False
        tPlugin.COUNTERS["dims_supervised"] = tSup
    tGradW = torch.zeros(2, tDec.input_size * tDec.output_size)
    tGradW.view(2, tDec.input_size, tDec.output_size)[..., 3:20] = 0.5
    (tDec.fc.weight * tGradW).sum().backward()
    tGradB = torch.zeros(2, tDec.output_size)
    tGradB[:, 20:32] = 1.0
    (tDec.bias.weight * tGradB).sum().backward()
    (tDec.bias.weight * tGradB).sum().backward()                     # second batch: no more lines (n reached 1)
    golden.check(f"preflight/{entry}/stdout", capsys.readouterr().out)
