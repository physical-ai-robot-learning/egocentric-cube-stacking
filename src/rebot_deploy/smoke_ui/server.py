"""Entry point of the smoke UI: imports every route module (each registers its routes on `app`), loads the inferencer
and serves on V4_UI_PORT. Started through `python -m rebot_deploy.run_smoke_ui` (was mac_v4_smoke_ui.py)."""
import os

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import XvlaInferencer
from . import (routes_calib, routes_ckpt, routes_obs_probes, routes_pose, routes_prompt_probes, routes_replay,  # noqa: F401
               routes_robot, routes_run, routes_settings, routes_status, routes_trial)


def main():
    rt.INF_CKPT = os.environ.get("V4_CKPT", "")
    rt.INF = XvlaInferencer()
    print(f"[v4-smoke] http://localhost:{cfg.UI_PORT}  robot={cfg.ROBOT}", flush=True)
    app.run(host="0.0.0.0", port=cfg.UI_PORT, threaded=True)


if __name__ == "__main__":
    main()
