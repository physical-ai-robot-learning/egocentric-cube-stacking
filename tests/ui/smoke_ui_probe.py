"""Probe of the reBot smoke UI (src/rebot_deploy/run_smoke_ui.py + smoke_ui/): its routes, live settings and pure helpers,
with every robot / network call stubbed (requests raises ConnectionError) and an empty HOME, so the result is the same on any
machine. Used by tests/ui/test_ui_routes.py; it runs in its own interpreter because the UI starts threads and reads env at import.

usage: python tests/ui/smoke_ui_probe.py <repo root> <out.json>      (needs flask, torch, scipy, cv2)"""
import hashlib, json, os, sys, tempfile, types
repo, out = sys.argv[1], sys.argv[2]
tmp = tempfile.mkdtemp()
os.environ["HOME"] = tmp
for k in list(os.environ):
    if k.startswith(("V4_", "CKSEL_")) or k in ("RUN", "DS_EXPECT"): del os.environ[k]
os.environ["V4_TRIAL_LOG"] = os.path.join(tmp, "trials.jsonl")
sys.path[:0] = [os.path.join(repo, "src")]

import requests
def _no_net(self, method, url, *a, **k): raise requests.ConnectionError(f"stubbed: {method} {url}")
requests.sessions.Session.request = _no_net
import numpy as np, torch

import rebot_deploy.run_smoke_ui as M
IC = sys.modules["rebot_deploy.xvla_inference"]
R = {}

def set_rt(name, val):
    st = sys.modules.get("rebot_deploy.smoke_ui.state")
    setattr(st if st is not None else M, name, val)

class FakeINF:
    def __init__(self):
        self.n_action = 4; self.task = "Stack the red cube on the bottom, blue cube in the middle, and purple cube on the top, on the plate."
        self.chunk = 32; self.last = {}; self._rtc_prev = "x"; self.anchor_resets = 0; self.n_denoise = None
        self.policy = types.SimpleNamespace(config=types.SimpleNamespace(num_denoising_steps=10))
    def reset_anchor(self): self.anchor_resets += 1
    def set_task(self, t): self.task = t
    def __getattr__(self, k): raise AttributeError(k)

c = M.app.test_client()
def J(r):
    try: return r.status_code, r.get_json()
    except Exception: return r.status_code, None

def mask(o):
    if isinstance(o, dict): return {k: ("<t>" if k in ("t", "time_local", "log") else mask(v)) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [mask(v) for v in o]
    return o

R["url_map"] = sorted((r.rule, r.endpoint, sorted(r.methods)) for r in M.app.url_map.iter_rules())
page = c.get("/").data
R["page_sha256"] = hashlib.sha256(page).hexdigest(); R["page_len"] = len(page)
R["status_noinf"] = J(c.get("/status"))
fake = FakeINF(); set_rt("INF", fake)
for path in ["/status", "/zlog", "/zlog?n=5", "/start_poses", "/cube_stop", "/wrist_zoom", "/frame/left", "/trials"]:
    R["GET " + path] = mask(J(c.get(path)))
R["GET /ckpts"] = mask(J(c.get("/ckpts")))
mode_q = "/mode?k=3&stream=1&speed=40&plan=1&planema=0.5&planxf=4&plansm=2&planmin=3&passlast=1&pflead=2&pass=1&dwell=0.3&prefetch=1&rtc=1"
for path in [mode_q, "/mode?remote=1", "/mode?remote=0", "/mode?interp=1", "/clamp/12/3", "/steps/5", "/task/2", "/arm/right", "/arm/both",
             "/ik/pink", "/gripper_mode/hold", "/gripper_mode/predict", "/stop", "/reset_anchor", "/denoise/5", "/prompt_ab_stop", "/replay_stop",
             "/ckpt_select/000100", "/estop", "/connect"]:
    R["POST " + path] = mask(J(c.post(path)))
R["status_after_mode"] = mask(J(c.get("/status")))
R["fake_after"] = dict(n_action=fake.n_action, task=fake.task, anchor_resets=fake.anchor_resets, rtc_prev=fake._rtc_prev, denoise=fake.policy.config.num_denoising_steps)
R["ic_after"] = dict(RTC_ON=IC.RTC_ON, REMOTE_INFER=IC.REMOTE_INFER, IK_BACKEND=IC.IK_BACKEND, GRIPPER_MODE=IC.GRIPPER_MODE, LIMITS=dict(IC.LIMITS))
R["trial_bad"] = mask(J(c.post("/trial", json={"stack_level": 7})))
R["trial_ok"] = mask(J(c.post("/trial", json={"stack_level": 3, "order_executed": "rbp", "layout_id": "L1", "note": " hi ", "operator": "jh"})))
R["trials_after"] = mask(J(c.get("/trials")))
for n in ("STREAM", "STREAM_SPEED", "STREAM_K", "PREFETCH", "PASS_THROUGH", "DWELL_TIMEOUT_S", "PREFETCH_LEAD", "PASS_LAST", "PLAN_MODE",
          "PLAN_EMA", "PLAN_XFADE", "PLAN_MIN_EXEC", "PLAN_SMOOTH_DEG", "ROBOT", "UI_PORT", "CMD_HZ", "TASKS", "TRAIN_START_DEG", "CUBE_STOP", "GRIP_GATE"):
    v = getattr(M, n); R["setting " + n] = v if not isinstance(v, dict) else {k: x for k, x in v.items() if k != "last_t"}

# ---- pure helpers on synthetic inputs
rng = np.random.default_rng(0)
def T(r):
    m = np.eye(4); from scipy.spatial.transform import Rotation as Rt
    m[:3, :3] = Rt.from_rotvec(r.normal(size=3) * 0.3).as_matrix(); m[:3, 3] = r.normal(size=3) * 0.2; return m
P = rng.normal(size=(16, 2, 3)) * 0.05; from scipy.spatial.transform import Rotation as Rt
Q = np.stack([Rt.from_rotvec(rng.normal(size=(16, 3)) * 0.2).as_quat(), Rt.from_rotvec(rng.normal(size=(16, 3)) * 0.2).as_quat()], 1)
cur = np.stack([T(rng), T(rng)])
for deg in (0, 1, 3, 5):
    setattr(sys.modules.get("rebot_deploy.smoke_ui.settings") or M, "PLAN_SMOOTH_DEG", deg)
    Ps, Qs = M._smooth_chunk(P, Q, cur); R[f"_smooth_chunk deg{deg}"] = [np.round(Ps, 10).tolist(), np.round(Qs, 10).tolist()]
R["_jag"] = [M._jag(P[:, 0]), M._jag(P[:2, 0])]
q14 = rng.normal(size=14).tolist(); q14[6] = 350.0; q14[13] = -120.0
R["_plan_leader"] = M._plan_leader(q14)
cmd = rng.normal(size=(3, 14))
R["_mask_arm"] = {a: np.round(M._mask_arm(cmd, a, q14), 12).tolist() for a in ("both", "left", "right")}
L = {"valid_step": 20, "tgt_pos": rng.normal(size=(32, 2, 3)), "cur_tcp": [rng.normal(size=4), rng.normal(size=4)]}
R["_chunk_end"] = [np.round(M._chunk_end(L), 10).tolist(), np.round(M._chunk_end(L, 5), 10).tolist(), np.round(M._chunk_end(L, 40), 10).tolist()]
rows = [(0.05 * i, rng.normal(size=(2, 3)), Rt.from_rotvec(rng.normal(size=(2, 3)) * 0.1).as_quat(), rng.random(2), 1, i) for i in range(8)]
tl = [(0.05 * i + 0.01, rng.normal(size=(2, 3)), Rt.from_rotvec(rng.normal(size=(2, 3)) * 0.1).as_quat(), rng.random(2), 0, i) for i in range(6)]
kept, st = M._stream_merge(tl, rows, 0.1)
R["_stream_merge"] = [st, [[round(k[0], 10), np.round(k[1], 10).tolist(), np.round(k[2], 10).tolist(), np.round(k[3], 10).tolist(), k[4], k[5]] for k in kept]]
img = np.zeros((480, 640, 3), np.uint8); img[100:160, 200:270] = (0, 0, 220)
R["cube_px"] = [M.cube_px(img), M.cube_px(np.zeros((48, 64, 3), np.uint8))]
R["_cksel_local_step"] = [M._cksel_local_step("ckpt_UI_RUNX_150k_pinklockwy", "RUNX"), M._cksel_local_step("ckpt_UI_RUNX_15xk_pinklockwy", "RUNX")]
for name in ("ckpt_UI_FT-R312C-FROM-EGOV2B-D20_150k_pinklockwy", "ckpt_HRA_RED-R30_40k", "x"):
    os.environ["V4_CKPT"] = "/a/" + name + "/"; R["_trial_ckpt_info " + name] = M._trial_ckpt_info()
os.environ.pop("V4_CKPT")
R["IC_EXEC_K"] = M.IC_EXEC_K(); R["_arm_only"] = M._arm_only()
# infer core pure functions
R["ic.unwrap_grip"] = [IC.unwrapGrip(x) for x in (-300, -10, 0, 179, 181, 358)]
R["ic.raw_to_width"] = [IC.rawToWidth(x) for x in (-300, -270, -100, 0, 50, 358)]
R["ic.width_to_raw"] = [IC.widthToRaw(x) for x in (-0.01, 0, 0.03, 0.2)]
R["ic.width_to_cmd"] = [IC.widthToCmd(x) for x in (0, 0.03, 0.2)]
R["ic.jaw_hold_cmd"] = [IC.jawHoldCmd(x) for x in (-300, -100, 0, 358)]
for gm in ("predict", "binary", "continuous", "relative"):
    IC.GRIPPER_MODE = gm; R["ic.grip_to_cmd " + gm] = [IC.gripToCmd(x) for x in (-0.1, 0.0, 0.03, 0.5, 0.7, 1.2)]
A = np.stack([T(rng), T(rng)]); C = np.stack([T(rng), T(rng)])
R["ic.relcart20_state"] = np.round(IC.relCart20State(A, C, [0.01, 0.2]).astype(float), 7).tolist()
R["ic.cart20_state"] = np.round(IC.cart20State(C, [0.0, 0.05]).astype(float), 7).tolist()
for mm, dg in ((0, 0), (5, 1), (1000, 90)):
    o, cl, n, ang = IC.clampDelta(T(rng), mm, dg); R[f"ic.clamp_delta {mm} {dg}"] = [np.round(o, 12).tolist(), cl, round(n, 9), round(ang, 9)]
R["ic._rtc_weights"] = [IC._rtcWeights(d, h, t).tolist() for d, h, t in ((0, 0, 8), (2, 5, 8), (3, 2, 8), (9, 12, 8))]
R["ic.set_limits"] = [IC.setLimits(7, None), IC.setLimits(None, 1.5)]
R["ic.consts"] = {k: (getattr(IC, k).tolist() if hasattr(getattr(IC, k), "tolist") else getattr(IC, k)) for k in (
    "CART20_W_OPEN", "GRIP_M_PER_RAW", "GRIP_RAW_MIN", "GRIP_RAW_MAX", "GRIP_OBS_PER_CMD", "GRIP_CMD_MAX", "GRIP_BIN_OPEN",
    "GRIP_BIN_CLOSED", "GRIP_BIN_THRESH", "GRIP_CMD_FULL", "GRIP_REL_EPS", "GRIP_REL_SCALE", "GRIP_REL_MODE", "GRIP_ACC_N",
    "GRIP_ACC_OPEN", "GRIP_ACC_CLOSE", "FIXED_ANCHOR_R", "RTC_MAX_GW", "UMI_HISTORY_DT")}
json.dump(R, open(out, "w"), indent=1, default=str, sort_keys=True)
print("ok", len(R), "entries; page", R["page_len"], R["page_sha256"][:12])
