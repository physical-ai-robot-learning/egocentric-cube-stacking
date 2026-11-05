"""rel16_aux with per-sample supervision masks, for the ego CART-ONLY pretrain (ego_relcart20task_rel16_v4_nopseudoq)
(training stage, LeRobot plugin).

[2026-09-30] rel16_aux.py (sha-pinned by the running launchers) is NOT modified; this is a separate copy. Same architecture /
schema: action (16, 32) = REL20 (REL16 pos3 rot6d g per arm) + dq12, same rel / dq / FK terms and lambdas. Two optional loss-only
columns:
    aux.has_dq_supervision  (B,)  1 = the dq12 target exists, 0 = no joint label (ego)
    aux.has_fk_supervision  (B,)  1 = aux.q_t exists and FK consistency applies, 0 = no q_t (ego)
Contract (generic, never by dataset name):
  * a column that is ABSENT means ones -> the loss is computed with the ORIGINAL rel16_aux expressions, so real R312c v4 FT is
    unchanged (tested bit-exact against rel16_aux on a real v4 batch: test_rel16_aux_masked.py)
  * aux.q_t may be absent only if every sample has has_fk_supervision == 0
  * masked dq target dims may be NaN (ego stores NaN: no value exists). Before the model runs they are set to 0 in the
    normalized space (the flow-matching input x_t = noise*t + action*(1-t) needs a finite value); NaN anywhere else, or in a
    dq dim of a sample with has_dq_supervision == 1, raises
  * masked samples contribute EXACTLY 0 to dq_loss / fk_loss and exactly 0 gradient (torch.where, not multiply). The mean is
    over the FULL batch (B * 16 * 12 for dq, B * 16 * 2 for FK), i.e. per-sample weight identical to the unmasked recipe
  * rel_loss (dims 0:20, gripper dims 9 / 19 included, as in the real v4 recipe) is never masked
Inputs -> outputs: LeRobot batches (+ optional aux columns) -> {"rel_loss", "dq_loss", "fk_loss"} from compute_loss.
Usage: `install()` before lerobot_train builds the policy (the node entry train_rel16aux_masked.py; preflight_cartonly.py).
Env: as rel16_aux (REL16_STATS, REL16_LAMBDA_Q, REL16_LAMBDA_FK, REL16_LOG_EVERY).
"""
from __future__ import annotations

import json
import os

import torch

from ego_paths import NODE_HOME, prepend_sys_path  # machine-specific roots (env-overridable)

LAMBDA_Q = float(os.environ.get("REL16_LAMBDA_Q", "1.0"))
LAMBDA_FK = float(os.environ.get("REL16_LAMBDA_FK", "20"))
LOG_EVERY = int(os.environ.get("REL16_LOG_EVERY", "200"))
MASK_KEYS = ("aux.has_dq_supervision", "aux.has_fk_supervision")
AUX_KEYS = ("aux.q_t",) + MASK_KEYS
REL_DIMS = 20                    # REL20 = per arm pos3 + rot6d + gripper
ACTION_DIMS = 32                 # REL20 + dq12
MASK_ON = 0.5                    # mask value above this = supervised
L_POS = slice(0, 3)              # predicted REL16 positions of the left / right arm inside REL20
R_POS = slice(10, 13)
M_TO_MM = 1000
# counters for the periodic log line (was _S)
COUNTERS = {"n": 0, "fk_mm": [], "masked_dq": 0, "masked_fk": 0, "samples": 0}


class Rel16MaskError(RuntimeError):
    """A batch violates the mask contract (non-finite supervised target, FK on without aux.q_t)."""


def _mask(batch, key, b, dev):
    tM = batch.get(key)
    return None if tM is None else (tM.reshape(b).to(dev) > MASK_ON)


def sanitizeAction(action, mdq):
    """normalized action (B, 16, 32): masked dq dims NaN -> 0; any other non-finite value raises (was sanitize_action)"""
    if mdq is None:
        return action
    tDq = action[..., REL_DIMS:ACTION_DIMS]
    tKeep = mdq[:, None, None]
    if not torch.isfinite(action[..., :REL_DIMS]).all():
        raise Rel16MaskError("[rel16_aux_masked] non-finite REL20 target")
    if not torch.isfinite(tDq[mdq]).all():
        raise Rel16MaskError("[rel16_aux_masked] non-finite dq target on a sample with has_dq_supervision = 1")
    return torch.cat([action[..., :REL_DIMS], torch.where(tKeep, tDq, torch.zeros_like(tDq))], dim=-1)


class _KeepAux:
    """Preprocessor wrapper: the LeRobot preprocessor drops unknown keys; keep every loss-only aux column (rel16_aux kept only
    aux.q_t)."""

    def __init__(self, pre):
        self.mPre = pre

    def __call__(self, batch):
        tAux = {tK: batch[tK] for tK in AUX_KEYS if isinstance(batch, dict) and tK in batch}
        tOut = self.mPre(batch)
        if tAux and isinstance(tOut, dict):
            tRef = tOut.get("action", tOut.get("observation.state"))
            for tK, tV in tAux.items():
                tOut[tK] = tV.to(tRef.device) if tRef is not None else tV
        return tOut

    def __getattr__(self, name):
        return getattr(self.mPre, name)


def _fkLoss(pred, qt, mfk, mean, std, fkFor):
    """FK(q_t + predicted dq) position vs predicted REL16 position, in the frame of the current TCP; -> (loss, d2)"""
    with torch.autocast(device_type="cuda", enabled=False):
        tDev = pred.device
        tMean, tStd = mean.to(tDev), std.to(tDev)
        tX = pred[..., :ACTION_DIMS].float() * tStd + tMean                         # unnormalized prediction (B, 16, 32)
        tQt = qt.to(tDev).float()                                                   # (B, 12)
        tFk = fkFor(tDev)
        with torch.no_grad():
            tTt = tFk.tcp(tQt)                                                      # (B, 2, 4, 4)
        tTk = tFk.tcp(tQt[:, None] + tX[..., REL_DIMS:ACTION_DIMS])                 # (B, 16, 2, 4, 4), grad -> dq_hat
        tRt = tTt[..., :3, :3]
        tPt = tTt[..., :3, 3]
        tPFk = torch.einsum("baji,bkaj->bkai", tRt, tTk[..., :3, 3] - tPt[:, None])  # R_t^T (p_k - p_t), (B, 16, 2, 3)
        tPRel = torch.stack([tX[..., L_POS], tX[..., R_POS]], dim=2)                # predicted REL16 position, (B, 16, 2, 3)
        tD2 = ((tPFk - tPRel) ** 2).sum(-1)                                         # m^2
        if mfk is None:
            tLoss = (LAMBDA_FK * tD2.mean()).to(pred.dtype)
        else:
            tD2m = torch.where(mfk[:, None, None], tD2, torch.zeros_like(tD2))
            tLoss = (LAMBDA_FK * tD2m.sum() / tD2m.numel()).to(pred.dtype)
            COUNTERS["masked_fk"] += int((~mfk).sum())
    return tLoss, tD2


def install() -> None:
    """Patch LeRobot X-VLA (once per process): aux-key carrier, mask hand-over in forward, masked rel / dq / FK compute_loss."""
    prepend_sys_path(str(NODE_HOME / "c8old/xvla"))
    try:
        import rebot_fk_torch
    except ImportError:  # clean checkout: the repository copy (src/training/robot/rebot_fk_torch.py)
        from training.robot import rebot_fk_torch
    import lerobot.policies.xvla.modeling_xvla as modeling
    import lerobot.policies.xvla.action_hub as actionHub
    tStats = json.load(open(os.environ["REL16_STATS"]))["action"]
    tMean = torch.tensor(tStats["mean"], dtype=torch.float32)
    tStd = torch.tensor(tStats["std"], dtype=torch.float32)
    assert tMean.numel() == ACTION_DIMS, f"REL16_STATS action dim {tMean.numel()} != 32 (need a REL20 + dq12 dataset)"
    tFkCache = {}

    def fkFor(dev):
        if dev not in tFkCache:
            tFkCache[dev] = rebot_fk_torch.ReBotFKTorch(device=dev, dtype=torch.float32)
        return tFkCache[dev]

    import lerobot.policies.factory as factory
    tOrigMake = factory.make_pre_post_processors

    def make_pre_post_processors(*args, **kw):
        tPre, tPost = tOrigMake(*args, **kw)
        return _KeepAux(tPre), tPost
    factory.make_pre_post_processors = make_pre_post_processors

    tOrigForward = modeling.XVLAPolicy.forward

    def forward(self, batch):
        tAction = batch["action"]
        tB, tDev = tAction.shape[0], tAction.device
        tMdq, tMfk = _mask(batch, MASK_KEYS[0], tB, tDev), _mask(batch, MASK_KEYS[1], tB, tDev)
        tQ = batch.get("aux.q_t")
        if tQ is None and (tMfk is None or tMfk.any()):
            raise Rel16MaskError("[rel16_aux_masked] batch has no aux.q_t but FK supervision is on for some sample")
        batch = dict(batch)
        batch["action"] = sanitizeAction(tAction, tMdq)
        tSpace = self.model.action_space
        tSpace._rel16_qt, tSpace._rel16_mdq, tSpace._rel16_mfk = tQ, tMdq, tMfk
        return tOrigForward(self, batch)
    modeling.XVLAPolicy.forward = forward

    def compute_loss(self, pred, target):
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape[-1] >= ACTION_DIMS and target.shape[-1] >= ACTION_DIMS and self.real_dim == ACTION_DIMS, \
            f"[rel16_aux_masked] expected real_dim 32, got {self.real_dim} / {tuple(pred.shape)}"
        tMdq, tMfk = getattr(self, "_rel16_mdq", None), getattr(self, "_rel16_mfk", None)
        tB = pred.shape[0]
        COUNTERS["samples"] += tB
        tOut = {"rel_loss": torch.mean((pred[..., :REL_DIMS] - target[..., :REL_DIMS]) ** 2) * self.JOINTS_SCALE}
        tDqPred, tDqTarget = pred[..., REL_DIMS:ACTION_DIMS], target[..., REL_DIMS:ACTION_DIMS]
        if tMdq is None:                                                            # original rel16_aux expression
            tOut["dq_loss"] = LAMBDA_Q * torch.mean((tDqPred - tDqTarget) ** 2) * self.JOINTS_SCALE
        else:
            tSe = torch.where(tMdq[:, None, None], (tDqPred - tDqTarget) ** 2, torch.zeros_like(tDqPred))
            tOut["dq_loss"] = LAMBDA_Q * tSe.sum() / tSe.numel() * self.JOINTS_SCALE
            COUNTERS["masked_dq"] += int((~tMdq).sum())
        tQt = getattr(self, "_rel16_qt", None)
        if LAMBDA_FK > 0 and tMfk is not None and not tMfk.any():                   # no FK label in this batch (ego)
            tOut["fk_loss"] = torch.zeros((), dtype=pred.dtype, device=pred.device)
            COUNTERS["masked_fk"] += tB
        elif LAMBDA_FK > 0 and tQt is not None:
            tOut["fk_loss"], tD2 = _fkLoss(pred, tQt, tMfk, tMean, tStd, fkFor)
            with torch.no_grad():
                tSel = tD2 if tMfk is None else tD2[tMfk]
                if tSel.numel():
                    COUNTERS["fk_mm"].append(tSel.detach().sqrt().reshape(-1, 2).cpu() * M_TO_MM)
        COUNTERS["n"] += 1
        if COUNTERS["n"] % LOG_EVERY == 1:
            tMsg = f"[rel16_aux_masked] batch {COUNTERS['n']}: rel {tOut['rel_loss'].item():.4f} dq {tOut['dq_loss'].item():.4f} " \
                   f"fk {tOut.get('fk_loss', torch.tensor(0.)).item():.5f}" \
                   f" | masked samples so far dq {COUNTERS['masked_dq']} / fk {COUNTERS['masked_fk']} of {COUNTERS['samples']}"
            if COUNTERS["fk_mm"]:
                tMm = torch.cat(COUNTERS["fk_mm"])
                COUNTERS["fk_mm"] = []
                tMsg += f" | FK(dq) vs REL pos: L p50 {tMm[:, 0].median():.1f} R p50 {tMm[:, 1].median():.1f} mm"
            print(tMsg + f"  (lambda_q {LAMBDA_Q}, lambda_fk {LAMBDA_FK})", flush=True)
        return tOut
    actionHub.AutoActionSpace.compute_loss = compute_loss
    print(f"[rel16_aux_masked] installed: REL16 + masked dq12 aux (lambda_q {LAMBDA_Q}) + masked FK (lambda_fk {LAMBDA_FK}), "
          f"stats {os.environ['REL16_STATS']}", flush=True)


# old name, used by the node-side unit test (archive/training_variants/umi76/test_rel16_aux_masked.py)
sanitize_action = sanitizeAction
