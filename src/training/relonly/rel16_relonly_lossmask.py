"""REL-only X-VLA loss with a per-sample, per-dim LOSS MASK over action dims 0:20 (training stage, LeRobot plugin).

[2026-10-03 user] For right-arm-only ego tasks. Copy-and-extend of rel16_aux_relonly.py (that file is sha-pinned by running
launchers and is NOT modified). Everything of the REL-only recipe stays: action dims 20:32 are hard-zeroed at the transformer
input and output (install_zero_mask from rel16_aux_relonly), no dq loss, no FK loss. One addition:

    aux.loss_mask  (B, 20) float, 1 = supervise this action dim, 0 = no target exists (e.g. the dummy LEFT arm and a
                   constant right gripper of the HRA_red right-only approach task).  Same mask for all 16 horizon steps.

    rel_loss = sum_{b,k,d<20} m[b,d] * (pred - target)^2 / sum_{b,k,d<20} m[b,d]   * JOINTS_SCALE

  * a batch WITHOUT aux.loss_mask (every existing bimanual dataset) uses m = 1, which is exactly the REL-only loss
    (mean over B*16*20) -- tested against rel16_aux_relonly's expression
  * masked dims contribute exactly 0 loss and 0 gradient, also with a NaN placeholder target (target replaced first)
  * normalising by the number of SUPERVISED elements keeps the per-element weight of a right-only sample equal to a
    bimanual one when the two are mixed in one batch (cotrain)

Inputs -> outputs: LeRobot batches with optional aux.loss_mask -> {"rel_loss"} from AutoActionSpace.compute_loss.
Usage: `install()` before lerobot_train builds the policy (train_rel16_relonly_lossmask.py does this).
Env: REL16_LOG_EVERY (200), RELONLY_PREFLIGHT=N (self-checks on the first N batches), RELONLY_ZERO (20:32).
"""
from __future__ import annotations

import os
import sys

import torch

try:  # the pinned zero contract (install_zero_mask)
    from . import rel16_aux_relonly as relonly
except ImportError:  # run as a flat script on the node (its folder is sys.path[0])
    import rel16_aux_relonly as relonly

MASK_KEY = "aux.loss_mask"
LOG_EVERY = int(os.environ.get("REL16_LOG_EVERY", "200"))
PREFLIGHT = int(os.environ.get("RELONLY_PREFLIGHT", "0"))
REL_DIMS = 20                    # REL20 = per arm pos3 + rot6d + gripper
ACTION_DIMS = 32                 # REL20 + dq12 (the dq part is zeroed, never supervised)
MASK_ON = 0.5                    # aux.loss_mask value above this = supervised
PREFLIGHT_NOISE = 10.0           # scale of the random targets the preflight swaps in (the loss must not move)
# counters / last-batch info; dims_supervised is read by the entry's preflight gradient hook (was _S)
COUNTERS = {"n": 0, "masked_samples": 0, "samples": 0, "supervised_frac": []}


class LossMaskError(RuntimeError):
    """A batch has no supervised action element (the masked mean would divide by zero)."""


def maskedRelLoss(pred20, target20, mask):
    """pred20/target20 (B, H, 20); mask (B, 20) bool or None.  Returns the mean over supervised elements (was masked_rel_loss)."""
    if mask is None:
        return torch.mean((pred20 - target20) ** 2)
    tMask = mask[:, None, :].expand_as(pred20)
    # replace masked targets BEFORE the subtraction: torch.where alone still back-propagates 0 * NaN = NaN through the
    # unselected branch when a masked target is NaN (caught by the unit test)
    tSafe = torch.where(tMask, target20, pred20.detach())
    tSquared = torch.where(tMask, (pred20 - tSafe) ** 2, torch.zeros_like(pred20))
    tCount = tMask.sum()
    if tCount == 0:
        raise LossMaskError("[relonly-lossmask] a batch with no supervised action element")
    return tSquared.sum() / tCount


class _KeepMask:
    """Preprocessor wrapper: the LeRobot preprocessor drops unknown keys, so aux.loss_mask is carried past it."""

    def __init__(self, pre):
        self.mPre = pre

    def __call__(self, batch):
        tMask = batch.get(MASK_KEY) if isinstance(batch, dict) else None
        tOut = self.mPre(batch)
        if tMask is not None and isinstance(tOut, dict):
            tRef = tOut.get("action", tOut.get("observation.state"))
            tOut[MASK_KEY] = tMask.to(tRef.device) if tRef is not None else tMask
        return tOut

    def __getattr__(self, name):
        return getattr(self.mPre, name)


def _randomiseUnusedTargets(target, mask):
    """preflight: a copy of target with dims 20:32 and every masked dim replaced by noise (the loss must not move)"""
    tTarget2 = target.clone()
    tTarget2[..., REL_DIMS:ACTION_DIMS] = torch.randn_like(tTarget2[..., REL_DIMS:ACTION_DIMS]) * PREFLIGHT_NOISE
    if mask is not None:     # masked targets randomised -> loss must not move
        tMask = mask[:, None, :].expand_as(tTarget2[..., :REL_DIMS])
        tTarget2[..., :REL_DIMS] = torch.where(tMask, tTarget2[..., :REL_DIMS],
                                               torch.randn_like(tTarget2[..., :REL_DIMS]) * PREFLIGHT_NOISE)
    return tTarget2


def install() -> None:
    """Patch LeRobot X-VLA (once per process): zero contract, mask carrier, mask hand-over in forward, masked compute_loss."""
    import lerobot.policies.factory as factory
    import lerobot.policies.xvla.action_hub as actionHub
    import lerobot.policies.xvla.modeling_xvla as modeling
    relonly.install_zero_mask()

    tOrigMake = factory.make_pre_post_processors

    def make_pre_post_processors(*a, **kw):
        tPre, tPost = tOrigMake(*a, **kw)
        return _KeepMask(tPre), tPost
    factory.make_pre_post_processors = make_pre_post_processors

    tOrigForward = modeling.XVLAPolicy.forward

    def forward(self, batch):
        tAction = batch["action"]
        tB = tAction.shape[0]
        tMask = batch.get(MASK_KEY)
        if tMask is not None:
            tMask = tMask.reshape(tB, -1).to(tAction.device)
            assert tMask.shape[1] == REL_DIMS, f"[relonly-lossmask] {MASK_KEY} must be (B, 20), got {tuple(tMask.shape)}"
            tMask = tMask > MASK_ON
            COUNTERS["masked_samples"] += int((~tMask).any(1).sum())
            COUNTERS["supervised_frac"].append(float(tMask.float().mean()))
        COUNTERS["dims_supervised"] = None if tMask is None else tMask.any(0).detach().cpu()   # read by the preflight grad hook
        COUNTERS["samples"] += tB
        self.model.action_space._lossmask = tMask
        return tOrigForward(self, batch)
    modeling.XVLAPolicy.forward = forward

    def compute_loss(self, pred, target):
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape[-1] >= ACTION_DIMS and target.shape[-1] >= ACTION_DIMS and self.real_dim == ACTION_DIMS, \
            f"[relonly-lossmask] expected real_dim 32, got {self.real_dim} / {tuple(pred.shape)}"
        tMask = getattr(self, "_lossmask", None)
        tOut = {"rel_loss": maskedRelLoss(pred[..., :REL_DIMS], target[..., :REL_DIMS], tMask) * self.JOINTS_SCALE}
        COUNTERS["n"] += 1
        tN = COUNTERS["n"]
        if PREFLIGHT and tN <= PREFLIGHT:
            tTarget2 = _randomiseUnusedTargets(target, tMask)
            tLoss2 = maskedRelLoss(pred[..., :REL_DIMS], tTarget2[..., :REL_DIMS], tMask) * self.JOINTS_SCALE
            tOk = torch.equal(tLoss2.detach(), tOut["rel_loss"].detach())
            if tMask is not None:
                tPerSample = tMask.sum(1).tolist()
                tH = pred.shape[1]
                print(f"[relonly-lossmask-preflight] batch {tN}: supervised_dims_per_sample {tPerSample} | effective_mask_sum "
                      f"{int(tMask.sum()) * tH} (= sum dims x {tH} horizon; the loss denominator) | full would be "
                      f"{tMask.shape[0] * REL_DIMS * tH}", flush=True)
            tPredZero = pred[..., REL_DIMS:ACTION_DIMS].abs().max().item() == 0
            if tMask is None:
                tMaskText = "absent (all ones)"
            else:
                tMaskText = (f"{tMask.float().mean().item():.3f} supervised, {int((~tMask).any(1).sum())}/{tMask.shape[0]} "
                             "samples masked")
            print(f"[relonly-lossmask-preflight] batch {tN}: {'PASS' if tOk else 'FAIL'} loss invariant to masked + 20:32 "
                  f"targets | {'PASS' if tPredZero else 'FAIL'} pred 20:32 == 0 | mask {tMaskText} | "
                  f"loss {tOut['rel_loss'].item():.4f}", flush=True)
        if tN % LOG_EVERY == 1:
            tFrac = COUNTERS["supervised_frac"][-1] if COUNTERS["supervised_frac"] else 1.0
            print(f"[relonly-lossmask] batch {tN}: rel {tOut['rel_loss'].item():.4f} | supervised dims {tFrac:.3f} | "
                  f"masked samples so far {COUNTERS['masked_samples']}/{COUNTERS['samples']}", flush=True)
        return tOut
    actionHub.AutoActionSpace.compute_loss = compute_loss
    assert "rebot_fk_torch" not in sys.modules, "[relonly-lossmask] FK module must not be imported"
    print(f"[relonly-lossmask] installed: REL(0:20) loss with per-sample {MASK_KEY} (absent = all ones); dims 20:32 zeroed; "
          "no dq / FK", flush=True)
