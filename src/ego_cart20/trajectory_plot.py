"""3D view of random CART20 samples (spec 33, was validation/visualize_trajectory.py).

Current pose T_t (origin of the anchor frame) and the k = 1, 4, 8, 16 targets, all drawn as A_k = inv(T_t) T(t + k dt)
from ONE origin -- a sequential-delta bug would show all k collapsed near the origin.

Pipeline stage: dataset-stage QA (scripts/report_dataset.py).
"""
from __future__ import annotations

import numpy as np

from .config import HORIZON
from .transforms import pose9ToTransform

ARM_STYLES = ((0, "tab:blue"), (10, "tab:red"))     # CART20 channel offset and colour of the left / right arm
AXIS_COLOURS = ("r", "g", "b")
PANEL_INCH = 5
DPI = 110


def plotSamples(cart20List, titles, outPng, ks=(1, 4, 8, 16), axisLen=0.01) -> None:
    """one 3D panel per CART20 chunk [16, 20] -> outPng"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tN = len(cart20List)
    tFig = plt.figure(figsize=(PANEL_INCH * tN, PANEL_INCH))
    for tI, (tCart, tTitle) in enumerate(zip(cart20List, titles)):
        tAx = tFig.add_subplot(1, tN, tI + 1, projection="3d")
        for tO, tColour in ARM_STYLES:
            tP = np.array([pose9ToTransform(tCart[tK - 1, tO:tO + 9])[:3, 3] for tK in range(1, HORIZON + 1)])
            tAx.plot([0, *tP[:, 0]], [0, *tP[:, 1]], [0, *tP[:, 2]], color=tColour, alpha=0.4)
            for tK in ks:
                tT = pose9ToTransform(tCart[tK - 1, tO:tO + 9])
                tPos = tT[:3, 3]
                tAx.plot([0, tPos[0]], [0, tPos[1]], [0, tPos[2]], color=tColour, ls=":", lw=0.8)
                for tA, tAxisColour in zip(range(3), AXIS_COLOURS):
                    tEnd = tPos + tT[:3, tA] * axisLen
                    tAx.plot([tPos[0], tEnd[0]], [tPos[1], tEnd[1]], [tPos[2], tEnd[2]], color=tAxisColour, lw=1)
                tAx.text(*tPos, f"k{tK}", fontsize=7, color=tColour)
        tAx.scatter([0], [0], [0], color="k", s=20)
        tAx.set_title(tTitle, fontsize=8)
        tAx.set_xlabel("x")
        tAx.set_ylabel("y")
        tAx.set_zlabel("z")
    tFig.suptitle("CART20 targets from the current pose T_t (blue = left, red = right; RGB = target axes)")
    tFig.tight_layout()
    tFig.savefig(outPng, dpi=DPI)
    plt.close(tFig)
