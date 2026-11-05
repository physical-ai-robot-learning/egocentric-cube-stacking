"""[2026-09-25] E_collision v1 (contract §23): inter-arm clearance from URDF link NAMES (Gen-1 census / retarget stage; was
c8_collision_v1.py).

In:  a kinematics object with the eef_kin.Kin interface (solver.robot.links.names, solver.link_positions, _q, _svc), q12
Out: Clearance.dmin(q12) -> (d_min in m, argmin pair by link names); Clearance.points(q12) -> {link name: xyz}
Usage: `Clearance(kin).dmin(q12)`

v0 used joint-index lists as link indices -> it measured left-arm / fixed-mount distances and is void. Exclusion rule by link
semantics, not by distance: `base` and `*_base_link` are fixed pedestal mounts -> excluded. Per-arm chain of segments:
  link1-link2, link2-link3, link3-link4, link4-link5, link5-link6, link6-gripper_link, gripper_link-tcp,
  gripper_link-gripper_left, gripper_link-gripper_right                         (9 segments per arm, 81 cross pairs)
Exact segment-segment distance (closest points, clamped). The method names points / dmin stay: the retarget code also calls
them on the ~/c8 twin.
"""
from __future__ import annotations

import numpy as np

CHAIN = [("link1", "link2"), ("link2", "link3"), ("link3", "link4"), ("link4", "link5"), ("link5", "link6"),
         ("link6", "gripper_link"), ("gripper_link", "tcp"), ("gripper_link", "gripper_left"), ("gripper_link", "gripper_right")]
DEGENERATE_SQ = 1e-12            # squared length (or determinant) below which a segment is a point (or segments are parallel)


def segmentDistance(p0, p1, q0, q1) -> float:
    """exact distance between segments p0-p1 and q0-q1 (closest points, clamped to the segments; was segseg)"""
    tD1, tD2, tR = p1 - p0, q1 - q0, p0 - q0
    tA, tE, tF = tD1 @ tD1, tD2 @ tD2, tD2 @ tR
    if tA <= DEGENERATE_SQ and tE <= DEGENERATE_SQ:
        return float(np.linalg.norm(tR))
    if tA <= DEGENERATE_SQ:
        tSc, tTc = 0.0, np.clip(tF / tE, 0, 1)
    else:
        tC = tD1 @ tR
        if tE <= DEGENERATE_SQ:
            tSc, tTc = np.clip(-tC / tA, 0, 1), 0.0
        else:
            tB = tD1 @ tD2
            tDen = tA * tE - tB * tB
            tSc = np.clip((tB * tF - tC * tE) / tDen, 0, 1) if tDen > DEGENERATE_SQ else 0.0
            tTc = (tB * tSc + tF) / tE
            if tTc < 0:
                tTc, tSc = 0.0, np.clip(-tC / tA, 0, 1)
            elif tTc > 1:
                tTc, tSc = 1.0, np.clip((tB - tC) / tA, 0, 1)
    return float(np.linalg.norm((p0 + tSc * tD1) - (q0 + tTc * tD2)))


class Clearance:
    """Minimum distance between the left and the right arm's link-chain segments."""

    def __init__(self, kin):
        self.mKin = kin
        self.mSolver = kin.solver
        tIndex = {tN: tI for tI, tN in enumerate(self.mSolver.robot.links.names)}
        self.mLinks = sorted({f"{tSide}_{tN}" for tSide in ("left", "right") for tPair in CHAIN for tN in tPair})
        self.mLinkIdx = [tIndex[tN] for tN in self.mLinks]

    def points(self, q12) -> dict:
        tQ = self.mKin._q(self.mKin._svc(q12)).astype(np.float32)
        tP = np.asarray(self.mSolver.link_positions(tQ, self.mLinkIdx))
        return dict(zip(self.mLinks, tP))

    def dmin(self, q12) -> tuple[float, str | None]:
        tP = self.points(q12)
        tBest = (np.inf, None)
        for tA, tB in CHAIN:
            for tC, tD in CHAIN:
                tV = segmentDistance(tP[f"left_{tA}"], tP[f"left_{tB}"], tP[f"right_{tC}"], tP[f"right_{tD}"])
                if tV < tBest[0]:
                    tBest = (tV, f"L:{tA}-{tB}|R:{tC}-{tD}")
        return tBest
