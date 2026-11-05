"""[2026-09-22] Real-robot SMOKE panel for the X-VLA checkpoints on the reBot (deploy stage; was mac_v4_smoke_ui.py, first written
for the r150-umi-v4 checkpoint with body-frame SE(3) deltas).

This is not a rollout UI. It exists to answer one question with the robot powered: does a predicted body-frame
delta arrive at the arm as the same motion, in direction and magnitude?

    predicted dT  ->  clamped dT  ->  commanded joints  ->  measured TCP dT

so all four are logged side by side for every press. There is no continuous loop on purpose: one press moves
the arm by at most V4_CLAMP_MM per step, for V4_N_ACTION steps, and then stops. The gripper is held.

Sending uses the same path the other UIs use -- observe -> leader frame (FLIP_IDX) -> slew-limited
/execute_step sub-steps -- because posting a target straight to /execute_step has already driven this arm into
its limits once.

In:  the robot service (V4_ROBOT, http://localhost:8020) and a checkpoint (V4_CKPT + the V4_* contract, see xvla_inference).
Out: a Flask UI on V4_UI_PORT (8016) driving the robot; cycle / IK / trial logs under ~.

[2026-11-01] The code is split by area into the package `smoke_ui/` next to this file (settings, state, robot client,
safety gates, the STEP / STREAM / PLAN executors, route modules, page.html); see smoke_ui/__init__.py for the map.
`run_smoke_ui.<name>` still resolves every top-level name of the old single file (read live from the module that now holds it).
usage: PYTHONPATH=src python -m rebot_deploy.run_smoke_ui
"""
from __future__ import annotations

import sys

from .smoke_ui import server as _server

_PKG = _server.__name__.rsplit(".", 1)[0]
_AREAS = ("settings", "state", "ic", "app", "page", "robot_client", "safety", "motion", "step_exec", "stream_exec", "plan_exec",
          "routes_run", "routes_robot", "routes_pose", "routes_replay", "routes_settings", "routes_status", "routes_calib",
          "routes_prompt_probes", "routes_obs_probes", "routes_ckpt", "routes_trial")
_SHARED_ALIASES = ("cfg", "rt", "_IC")          # module aliases every area binds: never resolved through this facade
app = _server.app


def __getattr__(name):
    """old single-file names (STATE, INF, STREAM, _goto, status, PAGE, ...) -> the current value in the module that holds them"""
    for tArea in _AREAS:
        tModule = sys.modules[f"{_PKG}.{tArea}"]
        if name in vars(tModule) and name not in _SHARED_ALIASES:
            return getattr(tModule, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv: list[str] | None = None) -> int:
    """load the inferencer and serve the UI until the process is stopped"""
    _server.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
