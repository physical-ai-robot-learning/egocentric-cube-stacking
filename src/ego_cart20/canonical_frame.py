"""Canonical task frame: the ONE place that defines how raw hand poses become canonical poses (was geometry/canonical_frame.py).

Convention ``per_arm_task_start_v1``:
    T_canonical(t) = inv(T_world_hand(t_task_start)) @ T_world_hand(t)        per arm, t_task_start = episode row 0
So every arm's canonical pose at the task start is the identity, and the canonical pose IS the v4 RELCART20 state pose.

Why per arm and not one bimanual frame: each HandUMI wrist runs its own MASt3R-SLAM map, so the left and right raw
poses live in two unrelated world frames; no shared L-R frame exists (memory: handumi-shared-atlas-v1 retracted).
Why this does not touch the labels: a left-multiplied frame change W cancels in every label,
    inv(W T_t) (W T_s) = inv(T_t) T_s,
so CART20 actions and both state variants are invariant to it.  The tool frame (right-multiplied) is NOT invariant and
is fixed upstream by the source exporter (hand TCP expressed in the reBot TCP axis convention, metadata ``tool_frame``).

Pipeline stage: geometry helpers of the dataset stage (used by convert.py).
"""
from __future__ import annotations

import numpy as np

from .transforms import inverse

NAME = "per_arm_task_start_v1"


def taskStartTransform(worldHandStart) -> np.ndarray:
    """T_canonical_world for one arm"""
    return inverse(worldHandStart)


def canonicalize(worldHand, worldHandStart) -> np.ndarray:
    """T_world_hand (n, 4, 4) -> canonical poses inv(T_world_hand(t_task_start)) @ T_world_hand"""
    return taskStartTransform(worldHandStart)[None] @ np.asarray(worldHand, np.float64)
