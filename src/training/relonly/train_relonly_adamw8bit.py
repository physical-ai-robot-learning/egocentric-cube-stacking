"""AdamW8bit variant of train_rel16_relonly.py: torch.optim.AdamW -> bitsandbytes AdamW8bit, then the unchanged REL-only entry
runs through runpy (training stage).

[2026-10-02 user] bitsandbytes 0.45.5 comes from SHARED_ROOT/relonly/pylib_bnb (NOT installed into the holobrain env).
Checkpoint fix: bnb keeps state['step'] as a python int, which lerobot's safetensors optimizer save rejects
("Key state/0/step ... expected torch.Tensor"). state_dict() returns COPIES of the per-param state with step as a 0-d int64
tensor (the live state is not touched); load_state_dict() turns it back into an int before bnb loads it.
On the nodes this file is deployed as code_8bit/train_rel16_relonly.py (WRAP in the launchers).
Usage: python train_relonly_adamw8bit.py <lerobot_train flags>     (or python -m training.relonly.train_relonly_adamw8bit)
"""
from __future__ import annotations

import runpy
import sys

import torch

from ego_paths import SHARED_ROOT, prepend_sys_path  # machine-specific roots (env-overridable)

BNB_DIR = SHARED_ROOT / "relonly/pylib_bnb"
RELONLY_ENTRY = SHARED_ROOT / "relonly/code/train_rel16_relonly.py"
AdamW8bitShim = None             # the shim class, built once bitsandbytes is importable (see makeShim)


def makeShim(bnb):
    """-> AdamW8bitShim(bnb.optim.AdamW8bit) with tensor `step` in state_dict and unshared tensors (safetensors-safe)"""

    class AdamW8bitShim(bnb.optim.AdamW8bit):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, **kw):
            assert not kw.get("amsgrad"), "amsgrad not supported by the 8bit shim"
            super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            print(f"[8bit] bitsandbytes {bnb.__version__} AdamW8bit replaces torch AdamW (groups {len(self.param_groups)})",
                  flush=True)

        def state_dict(self):
            # also: bnb shares ONE qmap tensor across all params; safetensors refuses shared storage -> clone repeats
            tSd = super().state_dict()
            tSeen = set()
            tOut = {}
            for tKey, tState in tSd["state"].items():
                tCopy = {}
                for tName, tValue in tState.items():
                    if tName == "step" and isinstance(tValue, int):
                        tValue = torch.tensor(tValue, dtype=torch.int64)
                    elif torch.is_tensor(tValue):
                        tPtr = tValue.untyped_storage().data_ptr()
                        if tPtr in tSeen:
                            tValue = tValue.clone()
                        else:
                            tSeen.add(tPtr)
                    tCopy[tName] = tValue
                tOut[tKey] = tCopy
            tSd["state"] = tOut
            return tSd

        def load_state_dict(self, state_dict, *a, **kw):
            tState = {tKey: {tName: (int(tValue.item()) if tName == "step" and torch.is_tensor(tValue) else tValue)
                             for tName, tValue in tParamState.items()}
                      for tKey, tParamState in state_dict["state"].items()}
            tResult = super().load_state_dict({**state_dict, "state": tState}, *a, **kw)
            tSteps = {tS.get("step") for tS in self.state.values() if "step" in tS}
            print(f"[8bit] optimizer state loaded: {len(self.state)} params, step values {sorted(tSteps)[:3]}", flush=True)
            return tResult

    return AdamW8bitShim


def main(argv: list[str] | None = None) -> int:
    """argv is left to lerobot_train (the REL-only entry parses sys.argv itself)."""
    global AdamW8bitShim
    prepend_sys_path(str(BNB_DIR))
    import bitsandbytes as bnb
    AdamW8bitShim = makeShim(bnb)
    torch.optim.AdamW = AdamW8bitShim
    sys.argv[0] = str(RELONLY_ENTRY)
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
