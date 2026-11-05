"""The reBot smoke UI (Flask), split by area out of the former single file mac_v4_smoke_ui.py (2026-11-01, behaviour unchanged).

Start it with `python -m rebot_deploy.run_smoke_ui` (`src` on PYTHONPATH; the loaders do).

  settings / state / ic     V4_* settings (live ones read as cfg.X), runtime state (rt.X), the one xvla_inference import
  robot_client / motion     robot-service HTTP client; slew-limited /execute_step motion and the dwell-until-reached goto
  safety                    HRA latch / gate / stop budgets, cube-size stop, gripper-zero gate
  step_exec / stream_exec / plan_exec    the three executors behind /run (STEP presses, STREAM timeline, PLAN playback)
  routes_*                  HTTP routes by area; app.py holds the Flask object, server.py imports the routes and runs it
  page.py + page.html       the single-page front end served by GET /
"""
