"""Process-wide runtime state of the smoke UI: the STATE dict shown by /status, the loaded inferencer (INF, set by
server.main), locks, the keep-alive HTTP session and small caches. Modules access it as `rt.NAME`.
"""
import collections as _collections
import os
import threading

import requests


_PFT = {"inf": None, "wp": None, "lead": None}
_PF = {"thread": None, "res": None}
INF_CKPT = os.environ.get("V4_CKPT", "")
STATE = {"busy": False, "presses": 0, "last": {}, "err": None, "arm": ("right" if os.environ.get("V4_ARM_ONLY", "").strip().lower() == "right" else "both"),
         "running": False, "cycles": 0, "travel_mm": [0.0, 0.0], "stop_reason": None, "task": "RBP",
         "zlog": _collections.deque(maxlen=400)}
INF = None
_lock = threading.Lock()
_SESS = requests.Session()          # keep-alive: /motor_states answers in 0.4 ms, /observe in 33 ms
_OBS_FAILS = {"retried": 0, "gave_up": 0}
_LAST_FRAME = {}   # [2026-09-30] last good jpeg per cam, display only
_CKSEL_CACHE = {"t": 0.0, "v": set()}
_TRIAL_LOCK = threading.Lock()
