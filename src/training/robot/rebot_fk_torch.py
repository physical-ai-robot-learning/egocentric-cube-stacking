"""Batched, differentiable reBot FK in torch (URDF-parsed via rebot_fk.ReBotFK) (training stage).

[2026-09-08] Used by humanik_delta for the EE auxiliary target (C) and the FK-consistency loss (D), by rel16_aux_masked (FK
loss) and derive_relcart20 (RELCART20 state). TCP = {side}_gripper_link @ T(TIP_X along +x) (pinch centre, TIP_X=0.07 m).
The numpy FK comes from the node copy next to the URDF (NODE_HOME/rebot_ee) when present, else from this repository.
Env: REBOT_URDF (defaults to NODE_HOME/rebot_ee/reBot_B601_DM_dualarm.urdf), REBOT_TIP_X.
Usage: python -m training.robot.rebot_fk_torch      (self-test against the numpy FK on 50 random poses)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from ego_paths import NODE_HOME, prepend_sys_path  # machine-specific roots (env-overridable)

prepend_sys_path(str(NODE_HOME / "rebot_ee"))
os.environ.setdefault("REBOT_URDF", str(NODE_HOME / "rebot_ee/reBot_B601_DM_dualarm.urdf"))
try:
    import rebot_fk  # noqa: E402  (the node copy next to the URDF, unchanged behaviour)
except ImportError:  # clean checkout: the repository copy next to this file
    from . import rebot_fk

TIP_X = float(os.environ.get("REBOT_TIP_X", "0.07"))
AXIS_EPS = 1e-12                 # guards the axis normalisation
SIDES = ("left", "right")
ARM12_OF_Q14 = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
SELFTEST_POSES = 50
SELFTEST_RANGE_RAD = 1.5


class ReBotFKTorch:
    def __init__(self, device="cpu", dtype=torch.float32):
        tFk = rebot_fk.ReBotFK(os.environ["REBOT_URDF"])
        self.mDevice, self.mDtype = device, dtype
        self.mChains = {}
        for tSide in SIDES:
            tSteps = []
            for tName in tFk.chains[tSide]:
                tJoint = tFk.joints[tName]
                tAxis = np.asarray(tJoint["axis"], float) / (np.linalg.norm(tJoint["axis"]) + AXIS_EPS)
                tSteps.append((torch.tensor(tJoint["T"], dtype=dtype, device=device), tJoint["type"] == "revolute",
                               torch.tensor(tAxis, dtype=dtype, device=device)))
            self.mChains[tSide] = tSteps
        self.mTipT = torch.eye(4, dtype=dtype, device=device)
        self.mTipT[0, 3] = TIP_X

    def to(self, device, dtype=None):
        dtype = dtype or self.mDtype
        if device == self.mDevice and dtype == self.mDtype:
            return self
        tNew = ReBotFKTorch.__new__(ReBotFKTorch)
        tNew.mDevice, tNew.mDtype = device, dtype
        tNew.mChains = {tSide: [(tT.to(device=device, dtype=dtype), tRev, tA.to(device=device, dtype=dtype))
                                for tT, tRev, tA in tSteps] for tSide, tSteps in self.mChains.items()}
        tNew.mTipT = self.mTipT.to(device=device, dtype=dtype)
        return tNew

    @staticmethod
    def _rot(axis, th):
        """Rodrigues: axis (3,), th (N,) -> (N,4,4)."""
        tN = th.shape[0]
        tC = torch.cos(th)
        tS = torch.sin(th)
        tOneMinusC = 1 - tC
        tX, tY, tZ = axis
        tR = torch.stack([
            torch.stack([tC + tX * tX * tOneMinusC, tX * tY * tOneMinusC - tZ * tS, tX * tZ * tOneMinusC + tY * tS], -1),
            torch.stack([tY * tX * tOneMinusC + tZ * tS, tC + tY * tY * tOneMinusC, tY * tZ * tOneMinusC - tX * tS], -1),
            torch.stack([tZ * tX * tOneMinusC - tY * tS, tZ * tY * tOneMinusC + tX * tS, tC + tZ * tZ * tOneMinusC], -1)], -2)
        tT = torch.eye(4, dtype=th.dtype, device=th.device).expand(tN, 4, 4).clone()
        tT[:, :3, :3] = tR
        return tT

    def side(self, q6, side):
        """q6 (N,6) rad -> (N,4,4) base->TCP."""
        tN = q6.shape[0]
        tT = torch.eye(4, dtype=q6.dtype, device=q6.device).expand(tN, 4, 4).clone()
        tK = 0
        for tFixed, tRevolute, tAxis in self.mChains[side]:
            tT = tT @ tFixed.to(q6.dtype)
            if tRevolute:
                tT = tT @ self._rot(tAxis.to(q6.dtype), q6[:, tK])
                tK += 1
        return tT @ self.mTipT.to(q6.dtype)

    def tcp(self, q12):
        """q12 (..., 12) rad = [L6, R6] -> (..., 2, 4, 4)."""
        tShape = q12.shape[:-1]
        tQ = q12.reshape(-1, 12)
        return torch.stack([self.side(tQ[:, :6], "left"), self.side(tQ[:, 6:], "right")], 1).reshape(*tShape, 2, 4, 4)


def main(argv: list[str] | None = None) -> int:
    """self-test vs numpy FK"""
    tFk = ReBotFKTorch()
    tNumpyFk = rebot_fk.ReBotFK(os.environ["REBOT_URDF"])
    tRng = np.random.default_rng(0)
    tErr = 0
    for _ in range(SELFTEST_POSES):
        tQ14 = np.zeros(14)
        tQ14[ARM12_OF_Q14] = tRng.uniform(-SELFTEST_RANGE_RAD, SELFTEST_RANGE_RAD, 12)
        tRef = tNumpyFk.eef(tQ14, tip=True)
        tT = tFk.tcp(torch.tensor(tQ14[ARM12_OF_Q14], dtype=torch.float32)[None])[0]
        tErr = max(tErr, np.abs(tT[0, :3, 3].numpy() - tRef["left"]).max(), np.abs(tT[1, :3, 3].numpy() - tRef["right"]).max())
    print("torch FK vs numpy FK max |dpos| = %.2e m" % tErr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
