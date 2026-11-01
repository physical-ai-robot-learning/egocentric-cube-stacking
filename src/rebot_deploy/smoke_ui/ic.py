"""The one import of the inference core (infer_core_v4) shared by every smoke-UI module.

`IC` is the module itself: settings the UI switches live (RTC_ON, REMOTE_INFER, IK_BACKEND, GRIPPER_MODE, WRIST_ZOOM, ...)
are read and written as `IC.NAME`. The plain names below are the values the UI took with `from infer_core_v4 import ...`
when it was one file; they are bound once, at import, exactly as before.

The package or the flat import is chosen from __package__ (not by catching ImportError, which would hide a real
import error), so `python -m rebot_deploy.mac_v4_smoke_ui` and
`python src/rebot_deploy/mac_v4_smoke_ui.py` (folder = sys.path[0]) each load exactly one copy of the core.
"""
if "." in (__package__ or ""):   # rebot_deploy.smoke_ui: package import
    from .. import infer_core_v4 as IC
else:                             # flat launch: smoke_ui is a top-level package next to infer_core_v4.py
    import infer_core_v4 as IC

V4Inferencer, CAM_MAP, GRIP_IDX, FLIP_IDX, ARM_IDX, LIMITS, set_limits = (
    IC.V4Inferencer, IC.CAM_MAP, IC.GRIP_IDX, IC.FLIP_IDX, IC.ARM_IDX, IC.LIMITS, IC.set_limits)
INF_W_OPEN = IC.CART20_W_OPEN
UMI_DT = IC.UMI_HISTORY_DT       # 50.05 ms: the chunk's own step


def IC_EXEC_K():
    return IC.EXEC_K
