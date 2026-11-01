"""Environment-derived settings of the smoke UI (V4_* variables), unchanged names and defaults.

Some are changed live by POST /mode (STREAM, STREAM_SPEED, STREAM_K, PREFETCH, PASS_THROUGH, DWELL_TIMEOUT_S,
PREFETCH_LEAD, PASS_LAST, PLAN_*); every module therefore reads them as attributes (`cfg.NAME`) at use time, never
copies them with `from ... import NAME`.
"""
from ego_paths import CKPT_ROOT, UMI_BRIDGE  # machine-specific roots (env-overridable)

import os


CYCLE_LOG = os.path.expanduser(os.environ.get("V4_CYCLE_LOG", "~/v4_cycles.jsonl"))
ROBOT = os.environ.get("V4_ROBOT", "http://localhost:8020")
UI_PORT = int(os.environ.get("V4_UI_PORT", "8016"))
OBS_GAP_S = 1.0 / float(os.environ.get("V4_EXEC_HZ", "15"))
SLEW_MAX_STEP_RAD = float(os.environ.get("V4_SLEW_RAD", "0.02"))     # half the v3 cap: this is a first test
SLEW_HZ = float(os.environ.get("V4_SLEW_HZ", "20"))


# The follower tracks slowly: one send of a +3 deg target realises +0.11 deg, and 40 identical sends over ~2 s
# reach +1.27 deg. A waypoint therefore has to be HELD, not fired once, or the arm executes a fraction of the
# commanded motion (and a small command looks like no motion at all). Hold until the joint error stops
# shrinking or the budget runs out.
# A waypoint is reached when BOTH the joint error and the Cartesian error are small -- joints alone can read
# "close" while the TCP is still millimetres out, and TCP alone hides a wrist that is still turning.
CMD_HZ = float(os.environ.get("V4_CMD_HZ", "20"))
JOINT_TOL_DEG = float(os.environ.get("V4_JOINT_TOL_DEG", "1.2"))
TCP_TOL_MM = float(os.environ.get("V4_TCP_TOL_MM", "6"))
DWELL_TIMEOUT_S = float(os.environ.get("V4_DWELL_TIMEOUT_S", "0.6"))


# [2026-10-02 user] pass-through: with several waypoints per cycle (V4_N_ACTION > 1), the intermediate ones are NOT dwelt on until
# "reached" -- the next target is sent as soon as the TCP is within V4_PASS_TCP_MM (or V4_PASS_DWELL_S runs out). Only the LAST
# waypoint of the cycle keeps the full reach test (JOINT_TOL_DEG + TCP_TOL_MM, V4_DWELL_TIMEOUT_S), so the end target is unchanged.
PASS_THROUGH = os.environ.get("V4_PASS_THROUGH", "0") == "1"
PASS_TCP_MM = float(os.environ.get("V4_PASS_TCP_MM", "12"))
PASS_DWELL_S = float(os.environ.get("V4_PASS_DWELL_S", "0.25"))


# [2026-10-02 user] prefetch (pipelining): while the LAST waypoint of a cycle is being driven, a worker observes and runs the next
# inference, so the next cycle starts without the inference stop. The prefetched chunk is anchored on the pose observed during that
# last motion (absolute base-frame targets, as always); the next cycle skips the head waypoints the arm has already passed
# (nearest target to the measured TCP, then one beyond it if within V4_PASS_TCP_MM) and executes V4_N_ACTION waypoints from there.
# A prefetch older than V4_PREFETCH_MAX_AGE_S, or one that failed, is discarded and the cycle observes + infers as before.
PREFETCH = os.environ.get("V4_PREFETCH", "0") == "1"
CYCLE_SETTLE_S = float(os.environ.get("V4_CYCLE_SETTLE_S", "0.2"))   # [2026-10-06] post-cycle settle before the travel measurement
_REMOTE_URL0 = os.environ.get("V4_REMOTE_INFER", "")      # [2026-10-04] remote server URL at startup (5080b / 5090), survives remote=0 -> 1
PREFETCH_MAX_AGE_S = float(os.environ.get("V4_PREFETCH_MAX_AGE_S", "2.5"))


# [2026-10-04 user] prefetch lead: start the prefetch V4_PREFETCH_LEAD waypoints BEFORE the last one (0 = at the last, the old
# behaviour), so the inference overlaps more of the motion; the next cycle's nearest-target shift skips what was passed meanwhile.
PREFETCH_LEAD = max(-1, int(os.environ.get("V4_PREFETCH_LEAD", "0")))
# -1 = AUTO: lead = ceil((inference EMA + 0.2 s) / waypoint-period EMA), so the prefetch finishes before the cycle ends


# [2026-10-04 user "움직임 연결 부드럽게"] pass-last: with prefetch running, the LAST waypoint of a cycle is passed through too (no
# full reach dwell) -- the next, already-inferred chunk takes over while the arm is still moving. 0 = old behaviour.
PASS_LAST = os.environ.get("V4_PASS_LAST", "0") == "1"


# [2026-09-30 user] pose gotos (TRAIN START, probes: no TCP target) need an accurate start pose, policy steps need a fast loop:
# separate dwell for gotos. Policy steps keep V4_DWELL_TIMEOUT_S.
GOTO_DWELL_S = float(os.environ.get("V4_GOTO_DWELL_S", "2.0"))   # [2026-09-30] 1.5 -> 2.0: left pan follower lag left 9.6 mm at 1.5 s


# [2026-09-30 user] goto settle: TCP err < tol AND max joint err < V4_GOTO_JOINT_TOL_DEG for 2 consecutive samples. Stationary motion
# between the last two samples is logged (still_mm / still_warn), not a gate. V4_GOTO_STILL_S is unused. Policy steps keep the old rule.
GOTO_JOINT_TOL_DEG = float(os.environ.get("V4_GOTO_JOINT_TOL_DEG", "1.0"))
GOTO_STILL_S = float(os.environ.get("V4_GOTO_STILL_S", "0.25"))
JOINT_NAMES = ["pan", "lift", "elbow", "wflex", "wyaw", "wroll"]


# R150 150 episodes' frame-0 median pose (arms deg, gripper raw) -- the same value the v3/EEF UIs use. Start a
# rollout here or the very first observation is a start pose the policy never saw.
# The six order instructions exactly as meta/tasks.parquet stores them -- a paraphrase is a different input to
# the language encoder, so these are copied verbatim rather than retyped.
TASKS = [
    ("RBP", "Stack the red cube on the bottom, blue cube in the middle, and purple cube on the top, on the plate."),
    ("RPB", "Stack the red cube on the bottom, purple cube in the middle, and blue cube on the top, on the plate."),
    ("BRP", "Stack the blue cube on the bottom, red cube in the middle, and purple cube on the top, on the plate."),
    ("BPR", "Stack the blue cube on the bottom, purple cube in the middle, and red cube on the top, on the plate."),
    ("PRB", "Stack the purple cube on the bottom, red cube in the middle, and blue cube on the top, on the plate."),
    ("PBR", "Stack the purple cube on the bottom, blue cube in the middle, and red cube on the top, on the plate."),
]
TRAIN_START_DEG = [float(x) for x in os.environ.get(
    "V4_TRAIN_START_DEG",
    "-4.5,-2.0,-8.3,30.4,-6.9,-5.2,0.0, 4.2,-14.8,-1.1,6.0,3.2,-3.2,0.0").split(",")]
REPLAY_FILE = os.environ.get("V4_REPLAY", "~/holobrain-mac-model/chunks_ep0.npz")


# 0 = no guard, as asked: the replay stops on E-STOP or `replay stop`, nothing else
REPLAY_MAX_STEP_MM = float(os.environ.get("V4_REPLAY_MAX_STEP_MM", "0"))


# A continuous loop needs its own stops, because the per-step clamp only bounds ONE step: 8 steps x 25 mm is
# 200 mm of travel. These are budgets on the whole run, and any of them ends it.
# 0 = no budget. The loop then ends only on E-STOP, STOP, an error, or IK failing to reach the first
# predicted pose -- the per-step clamp is still what bounds any single move.
MAX_CYCLES = int(os.environ.get("V4_MAX_CYCLES", "0"))
MAX_TRAVEL_MM = float(os.environ.get("V4_MAX_TRAVEL_MM", "0"))
DT_STEP_MS = 1000.0 / 15.0                 # the policy's action step; latency is reported in these units
OBS_RETRIES = int(os.environ.get("V4_OBS_RETRIES", "5"))   # [2026-09-30] see _observe


# [2026-10-03 user] HRA right-only safety (only when the core runs with V4_ARM_ONLY=right):
#   * at /run the LEFT arm (leader frame) and both jaws (as 0..45 commands) are latched; during a rollout every /execute_step
#     must carry exactly those values -- a deviation > HRA_LEFT_TOL_RAD / HRA_JAW_TOL means a leak of an unsupervised output
#     and the step is REFUSED and the run stopped (never silently corrected)
#   * approach-only stop: run time > HRA_MAX_RUN_S, or the predicted right TCP displacement at the chunk end < HRA_STILL_MM for
#     HRA_STILL_N inferences in a row (the approach has converged; the task has no grasp, so it must not keep pushing)
HRA_MAX_RUN_S = float(os.environ.get("V4_HRA_MAX_RUN_S", "0"))    # 0 = off (user 10-04: "최대시간같은거 없애줘"; 6 s stopped after 1-4 step-mode cycles)
HRA_STILL_MM = float(os.environ.get("V4_HRA_STILL_MM", "5"))
HRA_STILL_N = int(os.environ.get("V4_HRA_STILL_N", "3"))
HRA_LEFT_TOL_RAD = float(os.environ.get("V4_HRA_LEFT_TOL_RAD", "0.02"))
HRA_JAW_TOL = float(os.environ.get("V4_HRA_JAW_TOL", "2.0"))
HRA_GATE = os.environ.get("V4_HRA_GATE", "1") != "0"        # 0 = no refusal gate (user 10-04 "다 빼줘"); the core still holds left + jaws
HRA_MAX_CYCLE_MM = float(os.environ.get("V4_HRA_MAX_CYCLE_MM", "80"))     # measured right TCP move between two inferences
HRA_MAX_TRAVEL_MM = float(os.environ.get("V4_HRA_MAX_TRAVEL_MM", "450"))  # measured right TCP net move since /run


# [2026-10-07 user "일정크기 되면 그냥 멈추게"] vision stop: the red cube in the RAW right-wrist C922 frame (640x480, before any zoom)
# measured as sqrt(area) of the largest red blob; stop when it is >= CUBE_STOP["px"] on CUBE_STOP["n"] consecutive checks (0 = off).
CUBE_STOP = {"px": float(os.environ.get("V4_CUBE_STOP_PX", "0")), "n": 2, "every_s": 0.1, "last_t": 0.0, "hits": 0, "px_now": None,
             # [2026-10-07 user "큐브가 다시 그 사이즈보다 작으면 추론 계속"] follow: after a cube-size stop, watch the cube and /run again
             # (same fixed anchor) once it is VISIBLE (>= min_px) and smaller than resume_frac * px on resume_n checks; user STOP ends it
             "follow": os.environ.get("V4_CUBE_FOLLOW", "1") != "0", "waiting": False, "resume_frac": 0.85, "resume_n": 3, "min_px": 6.0, "resumes": 0}


# [2026-09-30 user] gripper zero gate: the left zero re-latches after every left-arm power drop (read +18/+22/357/362 today).
# A slipped zero reads outside the valid raw range, the state clips it to "closed" and commands land in the wrong frame,
# so RUN / MOVE 1 CHUNK / PROMPT A/B are refused until both jaws read inside [V4_GRIP_GATE_LO, V4_GRIP_GATE_HI] (-280, +5).
GRIP_GATE = (float(os.environ.get("V4_GRIP_GATE_LO", "-280")), float(os.environ.get("V4_GRIP_GATE_HI", "5")))
STREAM = os.environ.get("V4_STREAM", "0") == "1"
STREAM_SPEED = float(os.environ.get("V4_STREAM_SPEED", "0.4"))
STREAM_BLEND = float(os.environ.get("V4_STREAM_BLEND", "0.7"))
STREAM_LOG = os.path.expanduser(os.environ.get("V4_STREAM_LOG", "~/v4_stream.jsonl"))


# [2026-10-01 user: "맞는때에 추론을 넣게"] ChunkLookahead schedule: the next observe+forward starts only when the pending timeline
# covers no more than the expected latency (EMA of the measured one) + V4_STREAM_MARGIN_S, so the new chunk lands just before
# the current one runs out (fresh observation, no wasted rows). V4_STREAM_SCHED=0 = back-to-back forwards (previous behaviour).
STREAM_SCHED = os.environ.get("V4_STREAM_SCHED", "1") == "1"
STREAM_MARGIN_S = float(os.environ.get("V4_STREAM_MARGIN_S", "0.2"))


# [2026-10-01 user: "사이에 약간의 텀을 줘"] pause at every chunk boundary (the executor switches to rows of a newer chunk): the arm
# holds for V4_STREAM_PAUSE_S and the whole pending timeline is shifted by the same amount (nothing is dropped as late).
STREAM_PAUSE_S = float(os.environ.get("V4_STREAM_PAUSE_S", "0.3"))
STREAM_WP_PAUSE_S = float(os.environ.get("V4_STREAM_WP_PAUSE_S", "0.0"))


# [2026-10-01 user: "사이가 멈춰도 돼 ... 갑자기 확 튀던가 예전 동작으로 돌아가"] SYNC streaming: a pipelined chunk is predicted from
# an observation ~1 s old while the arm kept moving, so at the boundary the new plan jumps or points BACK along the path. With
# V4_STREAM_SYNC=1 the next observe+forward starts only when the timeline is empty and the arm has settled
# (V4_STREAM_SETTLE_S), and the chunk's first V4_STREAM_ROWS rows are timed from NOW (tau_j = now + j*dt, nothing dropped or
# blended): stream within a chunk, stop between chunks for the forward.
STREAM_SYNC = os.environ.get("V4_STREAM_SYNC", "0") == "1"
STREAM_ROWS = int(os.environ.get("V4_STREAM_ROWS", "8"))


# [2026-10-02 user "step k는 8"] non-sync streaming: only the first V4_STREAM_K rows (k1..kK) of every chunk enter the timeline
# (0 = all valid rows, the previous behaviour); the next chunk is scheduled/merged as before, so each plan is executed up to kK.
STREAM_K = int(os.environ.get("V4_STREAM_K", "0"))
STREAM_SETTLE_S = float(os.environ.get("V4_STREAM_SETTLE_S", "0.15"))   # [2026-10-01 user] extra hold after every waypoint (timeline shifted too)
_MOTOR_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]


# ==== [2026-10-06 user: "추론에서_모터까지" port] PLAN mode ==================================================================
# One forward is always in flight. Rows of the plan table carry a global row number; the bridge (/plan/*) plays them at 100 Hz as
# knots (segment position + slope as velocity feed-forward). A forward is requested with anchor = the row the playback heads to
# at the observation; the rows that will run while it computes are RESERVED (anchor .. F) and never replaced: the result replaces
# only the rows after max(C handed, F). Rows are handed in ~V4_PLAN_WINDOW_S portions; while a forward runs nothing beyond F is
# handed, and PACE slows the playback so the reserved rows last until the result (+ a brake lead). If the rows still run out the
# arm settles on the last knot. RTC (if on) uses the plan rows as the prefix: hard for the reserved d = F - anchor rows.
# Model rows = UMI_DT (50.05 ms, the training spacing); playback pace <= V4_STREAM_SPEED (the UI speed %).
PLAN_MODE = os.environ.get("V4_PLAN", "0") == "1"
PLAN_WINDOW_S = float(os.environ.get("V4_PLAN_WINDOW_S", "0.12"))
PLAN_BRAKE_LEAD_S = float(os.environ.get("V4_PLAN_BRAKE_LEAD_S", "0.15"))
PLAN_TICK_S = float(os.environ.get("V4_PLAN_TICK_S", "0.02"))
PLAN_LOG = os.path.expanduser(os.environ.get("V4_PLAN_LOG", "~/v4_plan.jsonl"))


# [2026-10-06 user: "지그재그"] seam smoothing (the doc's §07 EMA + an overlap crossfade) and a minimum execution horizon:
#   PLAN_EMA     q = prev + a (q_raw - prev) on the arm joints, prev = the boundary row (row F, already fixed); 0 or >= 1 = off
#   PLAN_XFADE   the first N new rows are blended with the previous plan's rows at the same row numbers, weight (i+1)/(N+1)
#   PLAN_MIN_EXEC the next forward is requested only after this many rows of the installed chunk have started (or earlier if the
#                 plan would run dry first): fewer, longer-lived chunks = fewer direction changes
PLAN_EMA = float(os.environ.get("V4_PLAN_EMA", "0.35"))
PLAN_XFADE = int(os.environ.get("V4_PLAN_XFADE", "4"))
PLAN_MIN_EXEC = int(os.environ.get("V4_PLAN_MIN_EXEC", "6"))
PLAN_TRACE = os.environ.get("V4_PLAN_TRACE", "1") == "1"


# [2026-10-06 user: "평활화"] within-chunk smoothing of the model's 16 targets before IK (the jitter is IN the chunk: 2nd difference
# ~69 % of the step). Positions: least-squares polynomial of degree PLAN_SMOOTH_DEG in the step index, the current TCP (k = 0) pinned
# with a large weight so the chunk still starts where the arm is; rotations: the same fit on the rotation vectors relative to the
# current TCP rotation (degree min(2, deg)). 0 = off. No lag across chunks (one chunk at a time).
PLAN_SMOOTH_DEG = int(os.environ.get("V4_PLAN_SMOOTH_DEG", "3"))


# [2026-10-03 user] "ckpoint 쌓인거 내가 ui에서 선택할수 있게 해줘": pick any archived step of this UI's run. A pick PINS the port
# (auto_ui_d20_e5.sh skips pinned ports) and re-runs load_relonly_run_ckpt_to_ui.sh with this process's env (same deploy contract),
# which restarts this UI on the chosen weights (~30-60 s). "auto" removes the pin and makes the swapper load the newest again.
CKSEL_SSD = str(CKPT_ROOT)
CKSEL_ST = str(UMI_BRIDGE / ".auto_ui_d20_e5")
CKSEL_LOADER = os.environ.get("CKSEL_LOADER", str(UMI_BRIDGE / "load_relonly_run_ckpt_to_ui.sh"))
CKSEL_NODE = os.environ.get("CKSEL_NODE", "")          # 5090 runs: "<ssh host>:<checkpoints dir>" (no SSD archive, list the node)


# [2026-10-04] closed-loop TRIAL LOG: one JSON line per real-robot trial, written by the operator after the trial.
# Auto fields come from the served checkpoint / UI state; outcome fields come from the form. Success rates are only ever
# computed from this file (the per-cycle log has no outcome). File: V4_TRIAL_LOG (default ~/rebot_trials.jsonl).
TRIAL_LOG = os.path.expanduser(os.environ.get("V4_TRIAL_LOG", "~/rebot_trials.jsonl"))
