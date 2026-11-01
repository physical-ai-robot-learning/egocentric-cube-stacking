"""Checkpoint selector routes (/ckpts, /ckpt_select): list archived steps of this UI's run and restart the UI on one
through the loader script.
"""
from ego_paths import HOLOBRAIN_MAC, NODE_HOME, UMI_BRIDGE  # machine-specific roots (env-overridable)

import os
import time

from flask import jsonify
import requests

from . import settings as cfg
from . import state as rt
from .app import app
from .ic import IC as _IC


def _cksel_local_step(name, run):
    pre = f"ckpt_UI_{run}_"
    if name.startswith(pre) and name.endswith("k_pinklockwy") and name[len(pre):-len("k_pinklockwy")].isdigit():
        return f"{int(name[len(pre):-len('k_pinklockwy')]) * 1000:06d}"
    return None


def _cksel_steps():
    run = os.environ.get("RUN", ""); d = os.path.join(cfg.CKSEL_SSD, os.environ.get("CKSEL_SSD_PREFIX", "trackb_") + run); out = set()
    if run and os.path.isdir(d):
        out |= {s for s in os.listdir(d) if s.isdigit() and len(s) == 6 and os.path.isfile(os.path.join(d, s, ".archived"))}
    # [2026-10-03 user "새 체크포인트 나올때마다 드롭다운 업데이트"]: list the training node too (4090 runs before SSD archiving)
    node = cfg.CKSEL_NODE or f"bh-aiteam@100.64.0.2:{NODE_HOME}/holobrain-data/trainB/{run}/checkpoints"
    jump = [] if cfg.CKSEL_NODE else ["-J", "head-lp"]
    if run:
        if time.time() - rt._CKSEL_CACHE["t"] > 25:
            import subprocess
            host, cdir = node.split(":", 1)
            try:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", *jump, host,
                                    f"cd {cdir} && for c in [0-9]*; do [ -f $c/pretrained_model/model.safetensors ] && echo $c; done"],
                                   capture_output=True, text=True, timeout=20)
                rt._CKSEL_CACHE.update(t=time.time(), v={x for x in r.stdout.split() if x.isdigit() and len(x) == 6})
            except Exception:
                pass
        out |= rt._CKSEL_CACHE["v"]
    hm = str(HOLOBRAIN_MAC); pre = f"ckpt_UI_{run}_"
    for n in os.listdir(hm):
        st = _cksel_local_step(n, run) if run else None
        if st and os.path.isfile(os.path.join(hm, n, "model.safetensors")): out.add(st)
    return sorted(out)


@app.route("/ckpts")
def ckpts():
    pin = os.path.join(cfg.CKSEL_ST, str(cfg.UI_PORT) + ".pin")
    # [2026-10-06 user "pinned 말고 실제 load된 ckpt도 표기"] serving step (this UI's model), a load in progress, the remote server's models
    import re as _re
    serving_dir = os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")); m_ = _re.search(r"_(\d+)k_", serving_dir)
    lk = os.path.join(cfg.CKSEL_ST, str(cfg.UI_PORT) + ".loading"); loading = None
    if os.path.exists(lk):
        try:
            pid_, st_ = open(lk).read().split()[:2]; os.kill(int(pid_), 0); loading = st_
        except Exception:
            loading = None
    remote = None
    if _IC.REMOTE_INFER:
        try:
            h = requests.get(_IC.REMOTE_INFER + "/health", timeout=1.0).json()
            remote = [(_re.search(r"_(\d+)k_", c[0]).group(1) + "k") if _re.search(r"_(\d+)k_", c[0]) else c[0] for c in h.get("loaded", [])]
        except Exception:
            remote = "busy (loading a checkpoint)"
    return jsonify({"serving_k": (m_.group(1) + "k") if m_ else serving_dir, "loading": loading, "remote_url": _IC.REMOTE_INFER, "remote_loaded": remote,
                    "run": os.environ.get("RUN", ""), "steps": _cksel_steps(),
                    "pinned": open(pin).read().strip() if os.path.exists(pin) else None,
                    "serving": os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/"))})


@app.route("/ckpt_select/<sel>", methods=["POST"])
def ckpt_select(sel):
    import subprocess
    run = os.environ.get("RUN", "")
    if not run or not os.environ.get("DS_EXPECT"):
        return jsonify({"ok": False, "err": "this UI was not started by load_relonly_run_ckpt_to_ui.sh (no RUN/DS_EXPECT)"})
    if rt.STATE.get("running") or rt.STATE.get("busy"):
        return jsonify({"ok": False, "err": "episode running -- STOP first"})
    os.makedirs(cfg.CKSEL_ST, exist_ok=True); pin = os.path.join(cfg.CKSEL_ST, str(cfg.UI_PORT) + ".pin")
    lk = os.path.join(cfg.CKSEL_ST, str(cfg.UI_PORT) + ".loading")    # one load at a time (a double click raced two rsyncs into one .part)
    if os.path.exists(lk):
        try:
            os.kill(int(open(lk).read().split()[0]), 0)
            return jsonify({"ok": False, "err": "a checkpoint load is already running for this port -- wait"})
        except (OSError, ValueError, IndexError):
            pass
    if sel == "auto":
        if os.path.exists(pin):
            old = open(pin).read().strip()
            with open(os.path.join(cfg.CKSEL_ST, "pending_delete"), "a") as f: f.write(f"{run} {old}\n")   # the pinned copy goes too
            os.remove(pin)
        st = os.path.join(cfg.CKSEL_ST, str(cfg.UI_PORT))
        if os.path.exists(st): os.remove(st)          # swapper reloads the newest on its next pass (<= 2 min)
        return jsonify({"ok": True, "msg": "auto: newest checkpoint loads within ~2 min"})
    if sel not in _cksel_steps():
        return jsonify({"ok": False, "err": f"step {sel} not archived for {run}"})
    with open(pin, "w") as f: f.write(sel + "\n")
    pd = os.path.join(cfg.CKSEL_ST, "pending_delete")     # never let the swapper's cleanup delete the copy we are about to serve
    if os.path.exists(pd):
        keep = [l for l in open(pd) if l.split() != [run, sel]]
        open(pd, "w").writelines(keep)
    cs = _cksel_local_step(os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")), run)
    if cs and cs != sel:                              # old copy: swapper deletes it (4090 runs once archived, 5090 runs at once)
        with open(pd, "a") as f: f.write(f"{run} {cs}\n")
    log = os.path.expanduser(f"~/ckpt_select_{cfg.UI_PORT}.log")
    _pl = subprocess.Popen(["/bin/bash", cfg.CKSEL_LOADER, str(int(sel)), str(cfg.UI_PORT)],
                     cwd=str(UMI_BRIDGE), env=dict(os.environ), stdout=open(log, "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    with open(lk, "w") as f: f.write(f"{_pl.pid} {sel}\n")
    return jsonify({"ok": True, "msg": f"loading {run} {sel} (pinned) -- UI restarts in ~30-60 s, log {log}"})
