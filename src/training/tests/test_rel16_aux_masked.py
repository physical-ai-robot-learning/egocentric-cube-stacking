"""Characterization of umi76/rel16_aux_masked.py: NaN sanitising of masked dq targets, the aux-key carrier around the LeRobot
preprocessor, and the rel / masked dq / masked FK losses (fake X-VLA classes, synthetic URDF for the FK term)."""
from __future__ import annotations

import importlib
import json

import pytest
import torch


def _stats(tmp_path):
    tGen = torch.Generator().manual_seed(1)
    tMean = (torch.randn(32, generator=tGen) * 0.01).tolist()
    tStd = (torch.rand(32, generator=tGen) * 0.05 + 0.01).tolist()
    tPath = tmp_path / "stats.json"
    tPath.write_text(json.dumps({"action": {"mean": tMean, "std": tStd}}))
    return tPath


def test_sanitize_action_zeroes_masked_dq_and_rejects_other_nan(fakeLerobot, golden):
    tM = importlib.import_module("training.umi76.rel16_aux_masked")
    tA = torch.arange(2 * 3 * 32, dtype=torch.float32).reshape(2, 3, 32)
    assert tM.sanitize_action(tA, None) is tA
    tA[1, :, 20:32] = float("nan")
    tMdq = torch.tensor([True, False])
    tOut = tM.sanitize_action(tA, tMdq)
    golden.check("masked/sanitized", tOut)
    with pytest.raises(RuntimeError, match="non-finite dq target"):
        tM.sanitize_action(tA, torch.tensor([True, True]))
    tBad = tA.clone()
    tBad[0, 0, 3] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite REL20 target"):
        tM.sanitize_action(tBad, tMdq)


def test_masked_losses_with_and_without_masks(fakeLerobot, synthUrdf, tmp_path, monkeypatch, golden, capsys):
    monkeypatch.setenv("REL16_STATS", str(_stats(tmp_path)))
    monkeypatch.setenv("REL16_LOG_EVERY", "2")
    tM = importlib.import_module("training.umi76.rel16_aux_masked")
    tM.install()
    tPre, _ = fakeLerobot["lerobot.policies.factory"].make_pre_post_processors()
    tGen = torch.Generator().manual_seed(9)
    tAction = torch.randn(2, 16, 32, generator=tGen) * 0.1
    tAction[1, :, 20:32] = float("nan")
    tQt = torch.randn(2, 12, generator=tGen) * 0.3
    tBatch = tPre({"action": tAction, "observation.state": torch.zeros(2, 20), "aux.q_t": tQt,
                   "aux.has_dq_supervision": torch.tensor([1.0, 0.0]), "aux.has_fk_supervision": torch.tensor([1.0, 0.0])})
    assert sorted(tBatch) == ["action", "aux.has_dq_supervision", "aux.has_fk_supervision", "aux.q_t", "observation.state"]
    tPolicy = fakeLerobot["lerobot.policies.xvla.modeling_xvla"].XVLAPolicy()
    tPolicy.forward(tBatch)
    golden.check("masked/forwarded_action", tPolicy.seen[-1]["action"])
    tSpace = tPolicy.model.action_space
    tPred = (torch.randn(2, 16, 32, generator=tGen) * 0.1).requires_grad_(True)
    tTarget = tPolicy.seen[-1]["action"]
    tOut = tSpace.compute_loss(tPred, tTarget)
    sum(tOut.values()).backward()
    golden.check("masked/losses_masked", tOut)
    golden.check("masked/grad_masked", tPred.grad)
    assert (tPred.grad[1, :, 20:32] == 0).all()

    tPolicy.forward({"action": tAction.nan_to_num(0.0), "aux.q_t": tQt})        # no mask columns: original rel16_aux
    golden.check("masked/losses_plain", tSpace.compute_loss(tPred.detach(), tAction.nan_to_num(0.0)))
    tPolicy.forward({"action": tAction.nan_to_num(0.0), "aux.has_fk_supervision": torch.zeros(2)})
    golden.check("masked/losses_no_fk", tSpace.compute_loss(tPred.detach(), tAction.nan_to_num(0.0)))
    with pytest.raises(RuntimeError, match="no aux.q_t"):
        tPolicy.forward({"action": tAction})
    golden.check("masked/counters", {tK: tV for tK, tV in tM._S.items() if tK != "fk_mm"})
    golden.check("masked/stdout", capsys.readouterr().out.replace(str(tmp_path), "<tmp>"))
