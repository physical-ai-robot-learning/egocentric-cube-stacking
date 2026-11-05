"""lerobot_train with the REL-only ablation plugin (rel16_aux_relonly.py); same CLI as lerobot_train.py (training stage).

[2026-09-30] RELONLY_PREFLIGHT=N runs the plugin self-checks on the first N batches and adds a gradient hook on the action
decoder that asserts rows 20:32 get exactly zero task gradient and rows 0:20 a non-zero one.
The node launchers run this file as a flat script ($R/code/train_rel16_relonly.py) and match it by name with pgrep;
the 8bit / co-train wrappers start it through runpy with run_name "__main__".
Usage: python train_rel16_relonly.py <lerobot_train flags>     (or python -m training.relonly.train_rel16_relonly)
"""
from __future__ import annotations

import os
import sys

try:
    from . import rel16_aux_relonly
except ImportError:  # run as a flat script on the node (its folder is sys.path[0])
    import rel16_aux_relonly

ZERO_LO, ZERO_HI = 20, 32        # decoder output rows that must get exactly zero task gradient


def _preflightBatches() -> int:
    """RELONLY_PREFLIGHT = number of batches the gradient hooks check (0 = no hooks)"""
    return int(os.environ.get("RELONLY_PREFLIGHT", "0"))


def installPreflightHooks() -> None:
    """Gradient hooks on every new SoftPromptedTransformer's action_decoder: print PASS / FAIL for the first N batches."""
    import lerobot.policies.xvla.soft_transformer as softTransformer
    tOrigInit = softTransformer.SoftPromptedTransformer.__init__
    tCount = {"n": 0}

    def __init__(self, *a, **kw):
        tOrigInit(self, *a, **kw)
        tDecoder = self.action_decoder
        tIn, tOutDim = tDecoder.input_size, tDecoder.output_size

        def hookWeight(grad):
            if tCount["n"] < int(os.environ["RELONLY_PREFLIGHT"]):
                tView = grad.view(grad.shape[0], tIn, tOutDim)
                tZero = tView[..., ZERO_LO:ZERO_HI].abs().max().item()
                tLive = tView[..., :ZERO_LO].abs().max().item()
                print(f"[relonly-preflight] decoder weight grad: rows 20:32 max {tZero:.3e} -> {'PASS' if tZero == 0 else 'FAIL'} | "
                      f"rows 0:20 max {tLive:.3e} -> {'PASS' if tLive > 0 else 'FAIL'}", flush=True)
            return grad

        def hookBias(grad):
            if tCount["n"] < int(os.environ["RELONLY_PREFLIGHT"]):
                tZero = grad[:, ZERO_LO:ZERO_HI].abs().max().item()
                print(f"[relonly-preflight] decoder bias grad rows 20:32 max {tZero:.3e} -> {'PASS' if tZero == 0 else 'FAIL'}",
                      flush=True)
                tCount["n"] += 1
            return grad
        tDecoder.fc.weight.register_hook(hookWeight)
        tDecoder.bias.weight.register_hook(hookBias)
        print(f"[relonly-preflight] gradient hooks on action_decoder ({tIn} -> {tOutDim})", flush=True)
    softTransformer.SoftPromptedTransformer.__init__ = __init__


def main(argv: list[str] | None = None) -> int:
    """argv is left to lerobot_train (it parses sys.argv itself)."""
    rel16_aux_relonly.install()
    if _preflightBatches():
        installPreflightHooks()
    from lerobot.scripts.lerobot_train import main as lerobotMain
    lerobotMain()
    return 0


if __name__ == "__main__":
    sys.exit(main())
