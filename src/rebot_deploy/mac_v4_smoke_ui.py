"""[2026-09-22] Real-robot SMOKE panel for the r150-umi-v4 X-VLA checkpoint (body-frame SE(3) deltas).

This is not a rollout UI. It exists to answer one question with the robot powered: does a predicted body-frame
delta arrive at the arm as the same motion, in direction and magnitude?

    predicted dT  ->  clamped dT  ->  commanded joints  ->  measured TCP dT

so all four are logged side by side for every press. There is no continuous loop on purpose: one press moves
the arm by at most V4_CLAMP_MM per step, for V4_N_ACTION steps, and then stops. The gripper is held.

Sending uses the same path the other UIs use -- observe -> leader frame (FLIP_IDX) -> slew-limited
/execute_step sub-steps -- because posting a target straight to /execute_step has already driven this arm into
its limits once.

Env: V4_* (see infer_core_v4), V4_UI_PORT (8016), V4_ROBOT (http://localhost:8020).

[2026-11-01] The code is split by area into the package `smoke_ui/` next to this file (settings, state, robot client,
safety gates, the STEP / STREAM / PLAN executors, route modules, page.html); see the README of this folder for the map.
This file is the unchanged entry point: `python src/rebot_deploy/mac_v4_smoke_ui.py` (what the loaders run) and
`python -m rebot_deploy.mac_v4_smoke_ui` both start the same server, and `mac_v4_smoke_ui.<name>` still resolves every
top-level name of the old single file (read live from the module that now holds it).
"""
if __package__:        # python -m rebot_deploy.mac_v4_smoke_ui / import rebot_deploy.mac_v4_smoke_ui
    from .smoke_ui import server as _server
else:                  # python src/rebot_deploy/mac_v4_smoke_ui.py: this folder is sys.path[0]
    from smoke_ui import server as _server

import sys as _sys

_PKG = _server.__name__.rsplit(".", 1)[0]
_AREAS = ("settings", "state", "ic", "app", "page", "robot_client", "safety", "motion", "step_exec", "stream_exec", "plan_exec",
          "routes_run", "routes_robot", "routes_pose", "routes_replay", "routes_settings", "routes_status", "routes_calib",
          "routes_prompt_probes", "routes_obs_probes", "routes_ckpt", "routes_trial")
main = _server.main
app = _server.app


def __getattr__(name):
    """old single-file names (STATE, INF, STREAM, _goto, status, PAGE, ...) -> the current value in the module that holds them"""
    for area in _AREAS:
        mod = _sys.modules[f"{_PKG}.{area}"]
        if name in vars(mod) and name not in ("cfg", "rt", "_IC"):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
