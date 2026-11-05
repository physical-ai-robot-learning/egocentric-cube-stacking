"""The one import of the inference core (xvla_inference, was infer_core_v4) shared by every smoke-UI module.

`IC` is the module itself: settings the UI switches live (RTC_ON, REMOTE_INFER, IK_BACKEND, GRIPPER_MODE, WRIST_ZOOM, ...)
are read and written as `IC.NAME`. The plain names below are the values the UI took with `from infer_core_v4 import ...`
when it was one file; they are bound once, at import, exactly as before.
"""
from .. import xvla_inference as IC

XvlaInferencer, CAM_MAP, GRIP_IDX, FLIP_IDX, ARM_IDX, LIMITS, set_limits = (
    IC.XvlaInferencer, IC.CAM_MAP, IC.GRIP_IDX, IC.FLIP_IDX, IC.ARM_IDX, IC.LIMITS, IC.setLimits)
INF_W_OPEN = IC.CART20_W_OPEN
UMI_DT = IC.UMI_HISTORY_DT       # 50.05 ms: the chunk's own step


def IC_EXEC_K():
    return IC.EXEC_K
