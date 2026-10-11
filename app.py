# =====================================================================
# SELF-ADAPTIVE CLOUD SECURITY FRAMEWORK
# Render-ready deployment (no ngrok, no cloudflared)
# =====================================================================

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import os, io, csv, time, base64, sqlite3, random, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from flask import Flask, jsonify, request, send_file

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NUM_VMS     = 3
RANDOM_SEED = 42

ALPHA            = 0.1
GAMMA            = 0.9
EPSILON_START    = 1.0
EPSILON_MIN      = 0.05
EPSILON_DECAY    = 0.995
EPISODE_LENGTH   = 20

# On Render, /content doesn't exist — use the current working directory.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "security.db")
CSV_PATH   = os.path.join(BASE_DIR, "security_logs.csv")
EXPORT_DIR = os.path.join(BASE_DIR, "rl_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


# =====================================================================
# PART A — 6-ATTACK / 6-ACTION MODEL
# =====================================================================
ATTACKS = ["Normal", "DDoS", "Brute Force", "Port Scan",
           "Data Exfiltration", "Malware/Lateral Move"]
ACTIONS = ["Allow/Monitor", "Rate-Limit", "Block IP",
           "Enforce MFA/Lock Acct", "Isolate VM", "Throttle Egress"]
FEATS   = ["Req Rate", "Failed Logins", "Port Diversity",
           "Outbound Vol", "Anomaly Score"]
NA, NT, NF = len(ACTIONS), len(ATTACKS), len(FEATS)

BASE = np.array([
    [0.25, 0.05, 0.10, 0.20, 0.10],
    [0.95, 0.10, 0.15, 0.30, 0.40],
    [0.40, 0.90, 0.10, 0.15, 0.30],
    [0.45, 0.10, 0.95, 0.10, 0.35],
    [0.30, 0.08, 0.10, 0.95, 0.45],
    [0.35, 0.15, 0.40, 0.45, 0.95],
])
R = np.array([
    [  2, -3, -5, -4, -9, -4 ],
    [-10, 10,  6, -3, -2, -3 ],
    [-10,  2,  6, 10, -1, -3 ],
    [ -8,  4, 10, -2,  0, -3 ],
    [-12, -2,  3, -2,  6, 10 ],
    [-12, -2,  3,  0, 10,  4 ],
], dtype=float)
OPT = R.argmax(axis=1)
POW = 3 ** np.arange(NF)
LEVEL_VALUE = [0.15, 0.50, 0.90]

STATE_NAMES  = {i: ATTACKS[i] for i in range(NT)}
ACTION_NAMES = {i: ACTIONS[i] for i in range(NA)}

def encode_all(F):
    F = np.asarray(F, dtype=float)
    B = (F > 0.65).astype(int) + (F > 0.35).astype(int)
    return int(B @ POW)

def make_profiles(strength):
    P = BASE.copy()
    if strength > 0:
        for i in range(1, NT):
            P[i] = P[i] + strength * (P[0] - P[i])
    return P


# =====================================================================
# PART B — CLOUD ENVIRONMENT
# =====================================================================
class VirtualMachine:
    def __init__(self, vm_id):
        self.vm_id = vm_id
        self.cpu = random.uniform(15, 45)
        self.memory = random.uniform(20, 55)
        self.isolated = False
    def status(self):
        return {"vm_id": self.vm_id, "cpu": round(self.cpu, 2),
                "memory": round(self.memory, 2), "isolated": self.isolated}

class CloudEnvironment:
    def __init__(self, num_vms=NUM_VMS):
        self.vms = [VirtualMachine(f"VM-{i+1}") for i in range(num_vms)]
    def snapshot(self):
        return [vm.status() for vm in self.vms]
    def get_vm(self, vm_id):
        for vm in self.vms:
            if vm.vm_id == vm_id: return vm
        return None
    def reset(self):
        for vm in self.vms:
            vm.cpu = random.uniform(15, 45)
            vm.memory = random.uniform(20, 55)
            vm.isolated = False


# =====================================================================
# PART C — DATA GENERATION AND DETECTION
# =====================================================================
def generate_event(event_type="normal"):
    rng = np.random.default_rng()
    P = make_profiles(0.0)
    table = {"normal": (0, 0.0), "ddos": (1, 0.9), "brute": (2, 0.9),
             "portscan": (3, 0.8), "exfil": (4, 0.9), "malware": (5, 0.95)}
    if event_type in table:
        k, sev = table[event_type]
    else:
        k = int(rng.integers(0, NT)); sev = float(rng.uniform(0.3, 1.0))

    mean = P[0] + sev * (P[k] - P[0])
    feats = np.clip(mean + rng.normal(0, 0.05, NF), 0, 1)
    cpu = float(20 + 75 * feats[0])
    mem = float(25 + 70 * feats[4])
    net = int(10 + 900 * feats[0])
    flg = int(30 * feats[1])

    return {"timestamp": datetime.now().strftime("%H:%M:%S"),
            "vm_id": f"VM-{int(rng.integers(1, NUM_VMS+1))}",
            "features": feats, "cpu": round(cpu, 2), "memory": round(mem, 2),
            "network_requests": net, "failed_logins": flg,
            "traffic_level": ["Low","Medium","High","Very High"][min(3, int(feats[0]*4))],
            "actual_threat": ATTACKS[k], "true_class": k}

def detect_state(event):
    f = event["features"]
    s = encode_all(f)
    score = float(np.max(f))
    if score >= 0.85: sev = "Critical"
    elif score >= 0.65: sev = "High"
    elif score >= 0.45: sev = "Medium"
    elif score >= 0.25: sev = "Low"
    else: sev = "None"
    return s, sev


# =====================================================================
# PART D — Q-LEARNING AGENT
# =====================================================================
class QLearningAgent:
    def __init__(self, n_states=3**NF, n_actions=NA):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        return int(np.argmax(self.q_table[state]))
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

def rule_action(f):
    if f[4] > 0.8: return 4
    if f[3] > 0.7: return 5
    if f[1] > 0.7: return 3
    if f[2] > 0.7: return 2
    if f[0] > 0.8: return 1
    return 0

def train_full_agent(agent, episodes=1000, steps=20, drift=True, adaptive=True,
                     drift_strength=0.4, p_attack=0.35, noise=0.08,
                     seed=RANDOM_SEED):
    rng_env = np.random.default_rng(seed)
    rng_ag  = np.random.default_rng(seed + 1)
    P0 = make_profiles(0.0)
    P1 = make_profiles(drift_strength if drift else 0.0)
    history = {"episodes": [], "rewards": [], "threats": [], "successes": [],
               "failures": [], "epsilon": [], "adapt": []}
    best_roll, last_adapt = -1.0, -999
    w = 10
    for ep in range(episodes):
        P = P1 if (drift and ep >= episodes // 2) else P0
        types = np.zeros(steps, dtype=int)
        t = 0
        while t < steps:
            if rng_env.random() < p_attack:
                a_ = int(rng_env.integers(1, NT))
                d  = int(rng_env.integers(3, 9))
                types[t:t+d] = a_
                t += d
            else:
                t += 1
        feats = np.clip(P[types] + rng_env.normal(0, noise, (steps, NF)), 0, 1)
        S = np.array([encode_all(feats[i]) for i in range(steps)])
        total_reward = 0
        threats = successes = failures = 0
        correct_so_far = 0
        for t in range(steps):
            s = int(S[t]); ty = int(types[t])
            a = (int(rng_ag.integers(NA)) if rng_ag.random() < agent.epsilon
                 else int(np.argmax(agent.q_table[s])))
            r = R[ty, a] + rng_ag.normal(0, 0.5)
            nxt = (agent.gamma * agent.q_table[int(S[t+1])].max()
                   if t + 1 < steps else 0.0)
            agent.q_table[s, a] += agent.alpha * (r + nxt - agent.q_table[s, a])
            total_reward += R[ty, a]
            if ty > 0: threats += 1
            if R[ty, a] >= 6 or (ty == 0 and a == 0):
                successes += 1; correct_so_far += 1
            else:
                failures += 1
        acc_now = correct_so_far / steps
        if adaptive and ep >= 2 * w:
            recent = float(np.mean(history.get("_acc_recent", [acc_now])))
            if recent < best_roll - 0.12 and (ep - last_adapt) > 3 * w:
                agent.epsilon = max(agent.epsilon, 0.4)
                last_adapt = ep
                history["adapt"].append(ep + 1)
            else:
                best_roll = max(best_roll, recent)
        history.setdefault("_acc_recent", []).append(acc_now)
        if len(history["_acc_recent"]) > w:
            history["_acc_recent"].pop(0)
        agent.decay_epsilon()
        history["episodes"].append(ep + 1)
        history["rewards"].append(total_reward)
        history["threats"].append(threats)
        history["successes"].append(successes)
        history["failures"].append(failures)
        history["epsilon"].append(round(agent.epsilon, 4))
    history.pop("_acc_recent", None)
    return history


# =====================================================================
# PART E — DATABASE
# =====================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS security_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, vm_id TEXT, event_type TEXT, threat_type TEXT,
        severity TEXT, state INTEGER, rl_action INTEGER, action_name TEXT,
        mitigation TEXT, reward INTEGER, success INTEGER)""")
    conn.commit(); conn.close()

def log_event(e):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO security_events
        (timestamp, vm_id, event_type, threat_type, severity, state,
         rl_action, action_name, mitigation, reward, success)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (e.get("timestamp"), e.get("vm_id"), e.get("event_type"),
         e.get("threat_type"), e.get("severity"), e.get("state"),
         e.get("rl_action"), e.get("action_name"), e.get("mitigation"),
         e.get("reward"), 1 if e.get("success") else 0))
    conn.commit(); conn.close()
    try:
        fe = os.path.isfile(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            if not fe:
                wr.writerow(["timestamp","vm_id","event_type","threat_type",
                             "severity","state","rl_action","action_name",
                             "mitigation","reward","success"])
            wr.writerow([e.get("timestamp"), e.get("vm_id"), e.get("event_type"),
                         e.get("threat_type"), e.get("severity"), e.get("state"),
                         e.get("rl_action"), e.get("action_name"),
                         e.get("mitigation"), e.get("reward"), e.get("success")])
    except Exception as ex:
        print("[CSV WARN]", ex)

def get_recent_events(limit=20):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM security_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
    conn.close(); return rows

def get_all_events():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM security_events ORDER BY id ASC").fetchall()]
    conn.close(); return rows

def clear_events():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM security_events"); conn.commit(); conn.close()
    if os.path.exists(CSV_PATH): os.remove(CSV_PATH)

def summarize(events):
    total = len(events)
    threats = sum(1 for e in events if e["threat_type"] != "Normal")
    successes = sum(1 for e in events if e["success"] == 1)
    failures = total - successes
    total_reward = sum(e["reward"] for e in events)
    detection_rate = (threats / total * 100) if total else 0.0
    tc = {a: 0 for a in ATTACKS}
    for e in events:
        if e["threat_type"] in tc: tc[e["threat_type"]] += 1
    if threats == 0: status, emoji = "Secure", "🟢"
    elif successes >= threats * 0.7: status, emoji = "Threat Detected (Mitigated)", "🟡"
    else: status, emoji = "Active Threat", "🔴"
    return {"total_events": total, "threats_detected": threats,
            "successful_mitigations": successes, "failed_mitigations": failures,
            "total_reward": total_reward,
            "detection_rate": round(detection_rate, 2),
            "threat_counts": tc, "status": status, "status_emoji": emoji}


# =====================================================================
# PART F — GLOBAL STATE
# =====================================================================
cloud_env = CloudEnvironment(num_vms=NUM_VMS)
agent     = QLearningAgent()
last_result = {}
training_history = {"episodes": [], "rewards": [], "threats": [],
                    "successes": [], "failures": [], "epsilon": [], "adapt": []}

init_db()

print("Pre-training Q-Learning agent (brief)...")
training_history = train_full_agent(
    agent, episodes=200, steps=20, drift=True, adaptive=True,
    drift_strength=0.3, p_attack=0.35, noise=0.08, seed=RANDOM_SEED)
print(f"Pre-train done. epsilon = {agent.epsilon:.3f} | last reward = {training_history['rewards'][-1]}")


def process_event(event):
    global last_result
    try:
        state, severity = detect_state(event)
        true_class = int(event.get("true_class", 0))
        if true_class < 0 or true_class >= NT:
            true_class = 0
        action = agent.choose_action(state)
        reward = float(R[true_class, action])
        success = (action == 0) if true_class == 0 else (reward >= 6)
        agent.update(state, action, reward, state)

        vm = cloud_env.get_vm(event["vm_id"])
        if vm:
            if action == 4: vm.isolated = True
            elif action == 5:
                vm.cpu = min(vm.cpu, 40); vm.memory = min(vm.memory, 50)

        mit_desc = {0:"Normal operation continues.",
                    1:"Traffic rate limited on affected service.",
                    2:"Source IP / user account blocked (simulated).",
                    3:"MFA enforced / account locked (simulated).",
                    4:"Virtual Machine isolated from the network.",
                    5:"Outbound egress throttled."}[action]

        log_event({"timestamp": event["timestamp"], "vm_id": event["vm_id"],
                   "event_type": event["traffic_level"],
                   "threat_type": event["actual_threat"], "severity": severity,
                   "state": state, "rl_action": action,
                   "action_name": ACTIONS[action], "mitigation": mit_desc,
                   "reward": int(reward), "success": bool(success)})

        last_result = {"timestamp": event["timestamp"], "vm_id": event["vm_id"],
                       "actual_threat": event["actual_threat"], "severity": severity,
                       "state": state, "state_name": ATTACKS[true_class],
                       "action": action, "action_name": ACTIONS[action],
                       "reward": int(reward), "mitigation": mit_desc,
                       "success": bool(success), "cpu": event["cpu"],
                       "memory": event["memory"],
                       "network_requests": event["network_requests"],
                       "failed_logins": event["failed_logins"],
                       "features": [round(float(x), 2) for x in event["features"]]}
        return last_result
    except Exception as e:
        print(f"[process_event ERROR] {e!r}")
        return {"timestamp": event.get("timestamp", "-"),
                "vm_id": event.get("vm_id", "VM-?"),
                "actual_threat": event.get("actual_threat", "Unknown"),
                "severity": "Unknown", "state": 0, "state_name": "Normal",
                "action": 0, "action_name": ACTIONS[0], "reward": 0,
                "mitigation": "no-op (internal error)",
                "success": False, "cpu": 0.0, "memory": 0.0,
                "network_requests": 0, "failed_logins": 0,
                "features": [0.0]*NF}


# =====================================================================
# PART G — FIGURES
# =====================================================================
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()

def _smooth(v, w):
    return pd.Series(v).rolling(w, min_periods=1).mean().values


# =====================================================================
# PART H — FLASK APP + UI
# =====================================================================
app = Flask(__name__)

STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#0b1220; --panel:#131f38; --panel-2:#1a2748;
  --border:#24345c; --text:#e6edf7; --muted:#8ba0c8;
  --accent:#3b82f6; --accent-2:#22d3ee; --green:#22c55e;
  --yellow:#eab308; --red:#ef4444; --radius:12px;
}
body { font-family:"Segoe UI",Roboto,Arial,sans-serif;
  background:radial-gradient(circle at top left,#0f1a30 0%,#0b1220 60%);
  color:var(--text); min-height:100vh; line-height:1.5; }
.layout { display:flex; min-height:100vh; }
.sidebar { width:230px; background:#0a1424; border-right:1px solid var(--border);
  padding:24px 16px; display:flex; flex-direction:column; }
.logo { font-size:20px; margin-bottom:28px; }
.logo span { color:var(--accent-2); }
.sidebar nav { display:flex; flex-direction:column; gap:6px; flex:1; }
.sidebar nav a { color:var(--muted); text-decoration:none; padding:10px 14px;
  border-radius:8px; font-size:15px; cursor:pointer; }
.sidebar nav a:hover { background:var(--panel); color:var(--text); }
.sidebar nav a.active { background:var(--accent); color:white; }
.sidebar-footer { font-size:12px; color:var(--muted); margin-top:20px; }
.sidebar-footer .epsilon { color:var(--accent-2); font-weight:600; }
.main { flex:1; padding:28px 32px; overflow-x:auto; }
header { display:flex; justify-content:space-between; align-items:center;
  margin-bottom:24px; flex-wrap:wrap; gap:12px; }
header h1 { font-size:22px; margin-bottom:4px; }
.subtitle { color:var(--muted); font-size:14px; }
.status-pill { padding:8px 16px; border-radius:999px; font-size:14px;
  font-weight:600; background:#0f2b1a; color:var(--green);
  border:1px solid #1f4a2c; }
.status-pill.warn { background:#2b250f; color:var(--yellow); border-color:#4a3f1f; }
.status-pill.danger { background:#2b0f0f; color:var(--red); border-color:#4a1f1f; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:16px; margin-bottom:24px; }
.card { background:linear-gradient(160deg,var(--panel) 0%,var(--panel-2) 100%);
  border:1px solid var(--border); border-radius:var(--radius);
  padding:18px; display:flex; flex-direction:column; gap:8px; }
.card-label { font-size:13px; color:var(--muted); }
.card-value { font-size:24px; font-weight:700; }
.controls { background:var(--panel); border:1px solid var(--border);
  border-radius:var(--radius); padding:20px; margin-bottom:24px; }
.controls h2 { font-size:16px; margin-bottom:14px; color:var(--muted); }
.btn-row { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:10px; }
.btn { background:#1c2b4d; color:var(--text); border:1px solid var(--border);
  padding:9px 16px; border-radius:8px; font-size:14px; cursor:pointer;
  transition:transform .08s, background .15s, box-shadow .2s; position:relative; }
.btn:hover { background:#263b66; }
.btn:active { transform:scale(0.96); }
.btn.clicked { background:var(--accent-2) !important; color:#0b1220 !important;
  box-shadow:0 0 12px rgba(34,211,238,0.7); }
.btn.primary { background:var(--accent); border-color:var(--accent); color:white; }
.btn.green { background:#14532d; border-color:#1f7a44; }
.btn.blue  { background:#1e3a8a; border-color:#2b4fb0; }
.btn.red   { background:#7f1d1d; border-color:#b91c1c; }
.btn:disabled { opacity:.5; cursor:not-allowed; }
.panel { background:var(--panel); border:1px solid var(--border);
  border-radius:var(--radius); padding:20px; margin-bottom:24px; }
.panel h3 { font-size:15px; margin-bottom:14px; color:var(--muted); }
.panel h4 { font-size:14px; margin:14px 0 6px 0; color:var(--accent-2); }
.panel p, .panel li { font-size:13.5px; color:var(--text); line-height:1.65; }
.panel ul, .panel ol { margin-left:20px; margin-bottom:8px; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px; }
@media (max-width:900px){ .two-col{grid-template-columns:1fr;} }
.rl-result { font-size:14px; line-height:1.7; }
.rl-result.empty { color:var(--muted); font-style:italic; }
.rl-result .row { display:flex; gap:8px; }
.rl-result .label { color:var(--muted); min-width:140px; }
.rl-result .good { color:var(--green); font-weight:600; }
.rl-result .bad  { color:var(--red); font-weight:600; }
.rl-result .action { color:var(--accent-2); font-weight:700; }
.cloud-grid { display:grid; grid-template-columns:1fr; gap:10px; }
.vm { background:var(--panel-2); border:1px solid var(--border);
  border-radius:10px; padding:10px 14px; font-size:13px; }
.vm .vm-title { font-weight:600; margin-bottom:4px; }
.vm .bar { background:#0a1424; border-radius:6px; overflow:hidden;
  height:8px; margin:4px 0 6px; }
.vm .bar > span { display:block; height:100%; background:var(--accent-2); }
.vm.iso { border-color:var(--red); }
.vm.iso .vm-title::after { content:" - ISOLATED"; color:var(--red); font-size:11px; }
.charts { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }
@media (max-width:900px){ .charts{grid-template-columns:1fr;} }
.chart-box { background:var(--panel); border:1px solid var(--border);
  border-radius:var(--radius); padding:16px; }
.chart-box h3 { font-size:14px; color:var(--muted); margin-bottom:10px; }
.chart-box canvas { max-height:200px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { padding:9px 10px; text-align:left; border-bottom:1px solid var(--border); }
th { color:var(--muted); background:#0e1830; font-weight:600; }
tr:hover td { background:#16223f; }
td .success { color:var(--green); } td .fail { color:var(--red); }
.empty { color:var(--muted); text-align:center; font-style:italic; padding:20px; }
.tab-pane { display:none; }
.tab-pane.active { display:block; }
.feature-bar { display:inline-block; width:220px; height:10px; background:#334155;
  border-radius:5px; vertical-align:middle; margin:0 8px; overflow:hidden; }
.feature-bar > span { display:block; height:100%; background:#3b82f6; }
pre { background:#0e1830; padding:12px; border-radius:8px; color:#c7d2fe;
  overflow-x:auto; font-size:12px; }
.info-banner { background:linear-gradient(90deg,#0f2744 0%,#132f52 100%);
  border-left:4px solid var(--accent-2); padding:14px 18px; border-radius:8px;
  margin-bottom:20px; font-size:13px; color:#c7d2fe; }
.info-banner b { color:var(--accent-2); }
.info-banner ul { margin-left:18px; margin-top:6px; }
.info-banner li { margin:3px 0; }
.error-box { background:#7f1d1d; color:#fee2e2; padding:12px; border-radius:8px;
  margin-bottom:12px; font-size:13px; }
.about-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
@media (max-width:900px){ .about-grid{grid-template-columns:1fr;} }
.about-card { background:var(--panel-2); border:1px solid var(--border);
  border-radius:10px; padding:16px; }
.about-card h4 { color:var(--accent-2); margin-bottom:8px; font-size:14px; }
.about-card p, .about-card li { font-size:13px; color:var(--text); }
.about-card ul, .about-card ol { margin-left:18px; margin-top:6px; }
.term { color:var(--accent-2); font-weight:600; }
.kbd { background:#0e1830; padding:2px 6px; border-radius:4px;
  font-family:monospace; color:var(--accent-2); font-size:12px; }
"""

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Self-Adaptive Cloud Security Framework</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>__STYLE__</style></head><body>
<div class="layout">
 <aside class="sidebar">
  <h2 class="logo">CloudSec<span>RL</span></h2>
  <nav>
   <a class="active" data-tab="dash">Dashboard</a>
   <a data-tab="live">Live Simulation</a>
   <a data-tab="inject">Threat Injection</a>
   <a data-tab="policy">Policy Explorer</a>
   <a data-tab="ablation">Ablation Study</a>
   <a data-tab="export">Export</a>
   <a data-tab="about">About</a>
   <a href="/logs">Full Logs</a>
  </nav>
  <div class="sidebar-footer">
   <p>Q-Learning Agent</p>
   <p class="epsilon">e = <span id="eps">1.00</span></p>
  </div>
 </aside>
 <main class="main">

  <header>
   <div><h1>Self-Adaptive Cloud Security Framework</h1>
     <p class="subtitle">Reinforcement Learning Based Dynamic Threat Detection &amp; Automated Mitigation</p></div>
   <div id="status-pill" class="status-pill">Secure</div>
  </header>

  <div id="global-error" class="error-box" style="display:none"></div>

  <section id="tab-dash" class="tab-pane active">
    <div class="info-banner">
      <b>Dashboard</b> - The main control room. Click any colored button to simulate an event and
      watch the RL agent respond in real time.
    </div>

    <section class="cards">
      <div class="card"><span class="card-label">Total Events</span><span class="card-value" id="c-total">0</span></div>
      <div class="card"><span class="card-label">Threats Detected</span><span class="card-value" id="c-threats">0</span></div>
      <div class="card"><span class="card-label">Successful Mitigations</span><span class="card-value" id="c-mit">0</span></div>
      <div class="card"><span class="card-label">Total Reward</span><span class="card-value" id="c-reward">0</span></div>
      <div class="card"><span class="card-label">Detection Rate</span><span class="card-value" id="c-rate">0%</span></div>
    </section>

    <section class="controls">
      <h2>Simulation Controls</h2>
      <div class="btn-row">
        <button class="btn green" onclick="simulate('normal',event)">Normal</button>
        <button class="btn" onclick="simulate('ddos',event)">DDoS</button>
        <button class="btn" onclick="simulate('brute',event)">Brute Force</button>
        <button class="btn" onclick="simulate('portscan',event)">Port Scan</button>
        <button class="btn" onclick="simulate('exfil',event)">Data Exfiltration</button>
        <button class="btn" onclick="simulate('malware',event)">Malware</button>
        <button class="btn primary" onclick="simulate('random',event)">Random Event</button>
      </div>
      <div class="btn-row">
        <button class="btn blue" onclick="trainAgent(event)">Train RL Agent</button>
        <button class="btn red" onclick="resetAll()">Reset</button>
      </div>
    </section>

    <section class="two-col">
      <div class="panel"><h3>RL Decision (Latest Event)</h3>
        <div id="rl-result" class="rl-result empty">No event yet. Click a control above.</div></div>
      <div class="panel"><h3>Cloud Environment</h3>
        <div id="cloud" class="cloud-grid"></div></div>
    </section>

    <section class="charts">
      <div class="chart-box"><h3>Rewards Over Training</h3><canvas id="chart-rewards"></canvas></div>
      <div class="chart-box"><h3>Threats Detected Per Episode</h3><canvas id="chart-threats"></canvas></div>
      <div class="chart-box"><h3>Successful vs Failed Mitigations</h3><canvas id="chart-mit"></canvas></div>
      <div class="chart-box"><h3>Attack vs Normal Traffic</h3><canvas id="chart-traffic"></canvas></div>
    </section>

    <section class="panel"><h3>Q-Table Sample (top-visited states)</h3>
      <table class="q-table" id="qtable"><thead></thead><tbody></tbody></table></section>

    <section class="panel"><h3>Recent Security Events</h3>
      <table class="events-table" id="events-table"><thead>
        <tr><th>Time</th><th>VM</th><th>Threat</th><th>Severity</th><th>State</th>
        <th>RL Action</th><th>Mitigation</th><th>Result</th><th>Reward</th></tr>
      </thead><tbody></tbody></table></section>
  </section>

  <section id="tab-live" class="tab-pane">
    <div class="info-banner">
      <b>Live Simulation</b> - Streams simulated events one after another. Click Start and watch
      the agent respond in real time.
    </div>
    <div class="panel">
      <div class="btn-row">
        <button class="btn primary" onclick="startLive()">Start Live Simulation</button>
        <button class="btn red" onclick="stopLive()">Stop</button>
      </div>
      <div id="live-out" style="margin-top:16px">
        <p class="empty">Press Start Live Simulation to begin.</p>
      </div>
    </div>
  </section>

  <section id="tab-inject" class="tab-pane">
    <div class="info-banner">
      <b>Threat Injection</b> - Pick a specific attack and severity, then see the agent's response.
    </div>
    <div class="panel">
      <div class="btn-row">
        <select id="inj-attack" class="btn">
          <option value="Normal">Normal</option>
          <option value="DDoS">DDoS</option>
          <option value="Brute Force">Brute Force</option>
          <option value="Port Scan">Port Scan</option>
          <option value="Data Exfiltration">Data Exfiltration</option>
          <option value="Malware/Lateral Move">Malware</option>
        </select>
        <select id="inj-sev" class="btn">
          <option value="0.4">Low (0.4)</option>
          <option value="0.6">Medium (0.6)</option>
          <option value="0.8">High (0.8)</option>
          <option value="1.0" selected>Critical (1.0)</option>
        </select>
        <button class="btn primary" onclick="inject()">Inject &amp; Respond</button>
        <button class="btn" onclick="randomInject()">Random Attack</button>
      </div>
      <div id="inj-out" style="margin-top:16px"><p class="empty">Choose an attack and click Inject.</p></div>
    </div>
  </section>

  <section id="tab-policy" class="tab-pane">
    <div class="info-banner">
      <b>Policy Explorer</b> - Set the 5 features to Low / Medium / High and see what the agent
      would do.
    </div>
    <div class="panel">
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px">
        <label>Req Rate<br><select class="btn" id="p0"><option value="0">Low</option><option value="1">Med</option><option value="2">High</option></select></label>
        <label>Failed Logins<br><select class="btn" id="p1"><option value="0">Low</option><option value="1">Med</option><option value="2">High</option></select></label>
        <label>Port Diversity<br><select class="btn" id="p2"><option value="0">Low</option><option value="1">Med</option><option value="2">High</option></select></label>
        <label>Outbound Vol<br><select class="btn" id="p3"><option value="0">Low</option><option value="1">Med</option><option value="2">High</option></select></label>
        <label>Anomaly Score<br><select class="btn" id="p4"><option value="0">Low</option><option value="1">Med</option><option value="2">High</option></select></label>
      </div>
      <div class="btn-row" style="margin-top:16px">
        <button class="btn primary" onclick="policyLookup()">Get Agent Decision</button>
      </div>
      <div id="pol-out" style="margin-top:16px"><p class="empty">Pick feature levels and click the button.</p></div>
    </div>
  </section>

  <section id="tab-ablation" class="tab-pane">
    <div class="info-banner">
      <b>Ablation Study</b> - Trains two agents (with and without self-adaptation) and compares them.
    </div>
    <div class="panel">
      <div class="btn-row">
        <button class="btn primary" onclick="runAblation(event)">Run Ablation Study</button>
      </div>
      <div id="abl-out" style="margin-top:16px"><p class="empty">Click the button to run the study.</p></div>
    </div>
  </section>

  <section id="tab-export" class="tab-pane">
    <div class="info-banner">
      <b>Export</b> - Download the event log CSV or the current Q-table CSV.
    </div>
    <div class="panel">
      <div class="btn-row">
        <button class="btn blue" onclick="exportCSV()">Download event log CSV</button>
        <button class="btn blue" onclick="exportQ()">Download Q-table CSV</button>
      </div>
    </div>
  </section>

  <section id="tab-about" class="tab-pane">
    <div class="info-banner">
      <b>About</b> - A plain-English reference for every term used in this project.
    </div>

    <div class="panel">
      <h3>Project in One Paragraph</h3>
      <p>
        This project is a <span class="term">simulated cloud security system</span>. It generates fake
        cloud traffic (some normal, some malicious), uses simple rules to detect what kind of traffic
        it is, and then asks a <span class="term">reinforcement learning agent</span> to choose the best
        defence action. The agent learns by trial and error - good choices earn reward, bad choices
        earn penalties.
      </p>
    </div>

    <div class="panel">
      <h3>Cloud Security Terms</h3>
      <div class="about-grid">
        <div class="about-card"><h4>Cloud</h4><p>A set of computers you rent from a provider (AWS, Azure, GCP).</p></div>
        <div class="about-card"><h4>Virtual Machine (VM)</h4><p>A software-defined computer running inside a real computer.</p></div>
        <div class="about-card"><h4>Cloud Security</h4><p>Protecting cloud data, apps and infrastructure.</p></div>
        <div class="about-card"><h4>Threat</h4><p>Anything that tries to harm the system.</p></div>
        <div class="about-card"><h4>Detection</h4><p>Recognising that a threat is happening.</p></div>
        <div class="about-card"><h4>Mitigation</h4><p>The defensive action taken in response.</p></div>
        <div class="about-card"><h4>False Positive</h4><p>Normal traffic wrongly flagged as an attack.</p></div>
        <div class="about-card"><h4>False Negative</h4><p>A real attack that was not detected.</p></div>
      </div>
    </div>

    <div class="panel">
      <h3>The 6 Traffic Types</h3>
      <div class="about-grid">
        <div class="about-card"><h4>Normal</h4><p>Regular traffic - low request rate, no failed logins. Action: Allow.</p></div>
        <div class="about-card"><h4>DDoS</h4><p>Massive request rate. Action: Rate-Limit.</p></div>
        <div class="about-card"><h4>Brute Force</h4><p>Repeated login attempts. Action: Enforce MFA / Lock.</p></div>
        <div class="about-card"><h4>Port Scan</h4><p>Probing many ports. Action: Block IP.</p></div>
        <div class="about-card"><h4>Data Exfiltration</h4><p>Large outbound data. Action: Throttle Egress.</p></div>
        <div class="about-card"><h4>Malware / Lateral Move</h4><p>Infection spreading. Action: Isolate VM.</p></div>
      </div>
    </div>

    <div class="panel">
      <h3>The 6 Defence Actions</h3>
      <ul>
        <li><b>Allow/Monitor</b> - do nothing except watch.</li>
        <li><b>Rate-Limit</b> - slow down traffic from a source.</li>
        <li><b>Block IP</b> - refuse all traffic from an IP.</li>
        <li><b>Enforce MFA / Lock Account</b> - require a second factor.</li>
        <li><b>Isolate VM</b> - disconnect a VM from the network.</li>
        <li><b>Throttle Egress</b> - limit outgoing data volume.</li>
      </ul>
    </div>

    <div class="panel">
      <h3>The 5 Features</h3>
      <ul>
        <li><b>Req Rate</b> - requests per second.</li>
        <li><b>Failed Logins</b> - recent failed login attempts.</li>
        <li><b>Port Diversity</b> - distinct ports touched.</li>
        <li><b>Outbound Vol</b> - data leaving the network.</li>
        <li><b>Anomaly Score</b> - general "how weird" score.</li>
      </ul>
    </div>

    <div class="panel">
      <h3>Reinforcement Learning Terms</h3>
      <div class="about-grid">
        <div class="about-card"><h4>Reinforcement Learning</h4><p>Agent learns by trying actions and getting rewards or penalties.</p></div>
        <div class="about-card"><h4>Agent</h4><p>The decision-maker. Here: the Q-Learning agent.</p></div>
        <div class="about-card"><h4>Environment</h4><p>The world the agent interacts with.</p></div>
        <div class="about-card"><h4>State</h4><p>A compact description of the current situation.</p></div>
        <div class="about-card"><h4>Action</h4><p>What the agent can do.</p></div>
        <div class="about-card"><h4>Reward</h4><p>Number telling how good the action was.</p></div>
        <div class="about-card"><h4>Penalty</h4><p>Negative reward for bad action.</p></div>
        <div class="about-card"><h4>Q-Table</h4><p>Grid: rows = states, columns = actions, cells = Q-values.</p></div>
        <div class="about-card"><h4>Q-Learning</h4><p>Algorithm that updates Q(s,a) with:<br><span class="kbd">Q(s,a) = Q(s,a) + a[r + g*max Q(s',a') - Q(s,a)]</span></p></div>
        <div class="about-card"><h4>Learning Rate (a)</h4><p>How much new experience changes the Q-table. Here: 0.1.</p></div>
        <div class="about-card"><h4>Discount (g)</h4><p>Weight of future rewards. Here: 0.9.</p></div>
        <div class="about-card"><h4>Epsilon</h4><p>Probability of exploring. Starts at 1.0.</p></div>
        <div class="about-card"><h4>Epsilon Decay</h4><p>Slowly reduces exploration. Here: x0.995 per episode.</p></div>
        <div class="about-card"><h4>Self-Adaptation</h4><p>Re-raise epsilon when accuracy drops.</p></div>
        <div class="about-card"><h4>Attack Drift</h4><p>Attacker changes tactics mid-training.</p></div>
        <div class="about-card"><h4>Ablation Study</h4><p>Compare agents with/without a feature.</p></div>
      </div>
    </div>

    <div class="panel">
      <h3>How to Demo This Project</h3>
      <ol>
        <li>Open Dashboard, click Normal, DDoS, Brute Force, etc.</li>
        <li>Open Policy Explorer and try different feature levels.</li>
        <li>Click Train RL Agent to see the reward curve rise.</li>
        <li>Open Ablation Study to show self-adaptation matters.</li>
        <li>Open Export to download the CSV for your report.</li>
      </ol>
    </div>
  </section>

 </main>
</div>

<script>
function showError(msg){
  const box = document.getElementById('global-error');
  box.style.display = 'block';
  box.textContent = 'Warning: ' + msg;
  setTimeout(()=>{ box.style.display='none'; }, 8000);
}

async function safeFetch(url, options){
  const resp = await fetch(url, options);
  const text = await resp.text();
  try { return JSON.parse(text); }
  catch (e) {
    const preview = text.slice(0, 120).replace(/\s+/g, ' ');
    throw new Error(`Server returned non-JSON (HTTP ${resp.status}). Preview: "${preview}..."`);
  }
}
async function apiGet(u){ return await safeFetch(u); }
async function apiPost(u,b){
  return await safeFetch(u, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(b || {})
  });
}

document.querySelectorAll('.sidebar nav a[data-tab]').forEach(a => {
  a.addEventListener('click', () => {
    document.querySelectorAll('.sidebar nav a').forEach(x=>x.classList.remove('active'));
    a.classList.add('active');
    document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
    document.getElementById('tab-' + a.dataset.tab).classList.add('active');
  });
});

let chartRewards, chartThreats, chartMit, chartTraffic;
const commonOpts = {
  responsive:true, maintainAspectRatio:false, animation:{duration:250},
  plugins:{legend:{labels:{color:'#8ba0c8'}}},
  scales:{x:{ticks:{color:'#8ba0c8'},grid:{color:'#1a2748'}},
          y:{ticks:{color:'#8ba0c8'},grid:{color:'#1a2748'}}}
};
function initCharts(){
  chartRewards = new Chart(document.getElementById('chart-rewards'),{
    type:'line', data:{labels:[],datasets:[{label:'Total Reward',data:[],
      borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,0.15)',
      fill:true,tension:.25,pointRadius:0}]}, options:commonOpts});
  chartThreats = new Chart(document.getElementById('chart-threats'),{
    type:'bar', data:{labels:[],datasets:[{label:'Threats',data:[],
      backgroundColor:'#ef4444'}]}, options:commonOpts});
  chartMit = new Chart(document.getElementById('chart-mit'),{
    type:'doughnut', data:{labels:['Successful','Failed'],
      datasets:[{data:[0,0],backgroundColor:['#22c55e','#ef4444']}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:250},
      plugins:{legend:{labels:{color:'#8ba0c8'}}}}});
  chartTraffic = new Chart(document.getElementById('chart-traffic'),{
    type:'pie', data:{labels:['Normal','DDoS','Brute Force','Port Scan',
      'Data Exfiltration','Malware'],
      datasets:[{data:[0,0,0,0,0,0],backgroundColor:['#22c55e','#ef4444',
        '#f59e0b','#a855f7','#3b82f6','#ec4899']}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:250},
      plugins:{legend:{labels:{color:'#8ba0c8'}}}}});
}

function downsample(xs, ys, max=120){
  if (xs.length <= max) return {x: xs, y: ys};
  const step = Math.ceil(xs.length / max);
  const ox=[], oy=[];
  for (let i=0;i<xs.length;i+=step){
    const j = Math.min(i+step, xs.length);
    ox.push(xs[Math.floor((i+j)/2)]);
    let s = 0; for (let k=i;k<j;k++) s += ys[k];
    oy.push(s / (j - i));
  }
  return {x: ox, y: oy};
}

async function refreshStatus(){
  try{
    const data = await apiGet('/api/status');
    const m = data.metrics;
    document.getElementById('c-total').textContent  = m.total_events;
    document.getElementById('c-threats').textContent = m.threats_detected;
    document.getElementById('c-mit').textContent    = m.successful_mitigations;
    document.getElementById('c-reward').textContent = m.total_reward;
    document.getElementById('c-rate').textContent   = m.detection_rate+'%';
    const pill = document.getElementById('status-pill');
    pill.textContent = m.status_emoji+' '+m.status;
    pill.className = 'status-pill';
    if(m.status.includes('Active')) pill.classList.add('danger');
    else if(m.status.includes('Threat')) pill.classList.add('warn');

    const cloudDiv = document.getElementById('cloud'); cloudDiv.innerHTML='';
    data.cloud.forEach(vm=>{
      const d = document.createElement('div');
      d.className='vm'+(vm.isolated?' iso':'');
      d.innerHTML = `<div class="vm-title">${vm.vm_id}</div>
        <div>CPU: ${vm.cpu}%</div><div class="bar"><span style="width:${vm.cpu}%"></span></div>
        <div>Memory: ${vm.memory}%</div><div class="bar"><span style="width:${vm.memory}%"></span></div>`;
      cloudDiv.appendChild(d);
    });
    chartTraffic.data.datasets[0].data = [
      m.threat_counts['Normal'], m.threat_counts['DDoS'],
      m.threat_counts['Brute Force'], m.threat_counts['Port Scan'],
      m.threat_counts['Data Exfiltration'], m.threat_counts['Malware/Lateral Move']];
    chartTraffic.update();
    chartMit.data.datasets[0].data=[m.successful_mitigations,m.failed_mitigations];
    chartMit.update();
  }catch(e){ showError(e.message); }
}
async function refreshEvents(){
  try{
    const data = await apiGet('/api/events?limit=15');
    const tb = document.querySelector('#events-table tbody'); tb.innerHTML='';
    if(!data.events || !data.events.length){
      tb.innerHTML='<tr><td colspan="9" class="empty">No events yet.</td></tr>'; return;
    }
    data.events.forEach(e=>{
      const tr=document.createElement('tr');
      tr.innerHTML = `<td>${e.timestamp}</td><td>${e.vm_id}</td>
        <td>${e.threat_type}</td><td>${e.severity}</td><td>${e.state}</td>
        <td>${e.action_name}</td><td>${e.mitigation}</td>
        <td>${e.success?'<span class="success">Success</span>':'<span class="fail">Failed</span>'}</td>
        <td>${e.reward>0?'+'+e.reward:e.reward}</td>`;
      tb.appendChild(tr);
    });
  }catch(e){ showError(e.message); }
}
async function refreshQTable(){
  try{
    const data = await apiGet('/api/qtable');
    document.getElementById('eps').textContent = data.epsilon.toFixed(2);
    const th = document.querySelector('#qtable thead');
    const tb = document.querySelector('#qtable tbody');
    th.innerHTML = '<tr><th>State</th>'+data.actions.map(a=>`<th>${a}</th>`).join('')+'</tr>';
    tb.innerHTML='';
    data.top_states.forEach(row=>{
      const tr=document.createElement('tr');
      tr.innerHTML = `<td>${row.label}</td>`+row.values.map(v=>`<td>${v.toFixed(2)}</td>`).join('');
      tb.appendChild(tr);
    });
  }catch(e){ showError(e.message); }
}
function showRLResult(r){
  const box = document.getElementById('rl-result'); box.classList.remove('empty');
  const cls = r.success?'good':'bad';
  const txt = r.success?'Successful Mitigation':'Failed Mitigation';
  const featBars = (r.features||[]).map((v,i) =>
    `<div class="row"><span class="label">${['Req Rate','Failed Logins','Port Diversity','Outbound Vol','Anomaly Score'][i]}:</span>
     <span class="feature-bar"><span style="width:${(v*100).toFixed(0)}%"></span></span> ${v}</div>`).join('');
  box.innerHTML = `
    <div class="row"><span class="label">Detected State:</span><span>${r.state_name} (state=${r.state})</span></div>
    <div class="row"><span class="label">Actual Threat:</span><span>${r.actual_threat}</span></div>
    <div class="row"><span class="label">Severity:</span><span>${r.severity}</span></div>
    ${featBars}
    <div class="row"><span class="label">RL Action:</span><span class="action">${r.action_name} (action=${r.action})</span></div>
    <div class="row"><span class="label">Reward:</span><span class="${r.reward>=0?'good':'bad'}">${r.reward>0?'+'+r.reward:r.reward}</span></div>
    <div class="row"><span class="label">Mitigation:</span><span>${r.mitigation}</span></div>
    <div class="row"><span class="label">Result:</span><span class="${cls}">${txt}</span></div>`;
}

function flashButton(el){
  if (!el) return;
  el.classList.add('clicked');
  setTimeout(()=>el.classList.remove('clicked'), 250);
}

async function simulate(type, ev){
  if (ev && ev.target) flashButton(ev.target);
  try {
    const data = await apiPost('/api/simulate',{type});
    if(!data.ok){ showError(data.error || 'simulate failed'); return; }
    showRLResult(data.result);
    await Promise.all([refreshStatus(), refreshEvents(), refreshQTable()]);
  } catch(e){ showError(e.message); }
}

async function trainAgent(ev){
  const btn = ev.target; btn.disabled=true;
  const oldText = btn.textContent; btn.textContent='Training...';
  try{
    const data = await apiPost('/api/train',{episodes:600});
    if(!data.ok){ showError(data.error || 'train failed'); return; }
    const h = data.history;
    const reduced = downsample(h.episodes, h.rewards, 120);
    chartRewards.data.labels = reduced.x;
    chartRewards.data.datasets[0].data = reduced.y;
    chartRewards.update();
    const start = Math.max(0, h.episodes.length - 60);
    chartThreats.data.labels = h.episodes.slice(start);
    chartThreats.data.datasets[0].data = h.threats.slice(start);
    chartThreats.update();
    const totS = h.successes.reduce((a,b)=>a+b,0);
    const totF = h.failures.reduce((a,b)=>a+b,0);
    chartMit.data.datasets[0].data = [totS,totF]; chartMit.update();
    await refreshQTable(); await refreshStatus();
  } catch(e){ showError(e.message); }
  finally { btn.disabled=false; btn.textContent=oldText; }
}
async function resetAll(){
  if(!confirm('Reset all logs, Q-table and state?')) return;
  try{
    await apiPost('/api/reset',{});
    chartRewards.data.labels=[]; chartRewards.data.datasets[0].data=[]; chartRewards.update();
    chartThreats.data.labels=[]; chartThreats.data.datasets[0].data=[]; chartThreats.update();
    chartMit.data.datasets[0].data=[0,0]; chartMit.update();
    chartTraffic.data.datasets[0].data=[0,0,0,0,0,0]; chartTraffic.update();
    document.getElementById('rl-result').classList.add('empty');
    document.getElementById('rl-result').innerHTML='No event yet. Click a control above.';
    await refreshStatus(); await refreshEvents(); await refreshQTable();
  }catch(e){ showError(e.message); }
}

let liveTimer = null;
function stopLive(){ if(liveTimer){ clearInterval(liveTimer); liveTimer=null; } }
function startLive(){
  stopLive();
  const out = document.getElementById('live-out');
  out.innerHTML = '<h3 style="color:var(--muted);font-size:14px;margin-bottom:8px">Streaming events...</h3>'
                + '<table class="events-table"><thead><tr><th>Step</th><th>Threat</th>'
                + '<th>RL Action</th><th>Reward</th><th>Result</th></tr></thead>'
                + '<tbody id="live-tbody"></tbody></table>';
  let step = 0;
  const tbody = document.getElementById('live-tbody');
  const types = ['normal','ddos','brute','portscan','exfil','malware'];
  liveTimer = setInterval(async ()=>{
    step++;
    const t = types[Math.floor(Math.random()*types.length)];
    try{
      const data = await apiPost('/api/simulate',{type:t});
      if(!data.ok) return;
      const r = data.result;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${step}</td><td>${r.actual_threat}</td>
        <td>${r.action_name}</td><td>${r.reward>0?'+'+r.reward:r.reward}</td>
        <td>${r.success?'OK':(r.actual_threat==='Normal'?'False Alarm':'Missed')}</td>`;
      tbody.prepend(tr);
      if(tbody.children.length > 14) tbody.removeChild(tbody.lastChild);
      refreshStatus(); refreshEvents();
    }catch(e){ }
    if(step >= 40) stopLive();
  }, 900);
}

async function inject(){
  const a = document.getElementById('inj-attack').value;
  const s = parseFloat(document.getElementById('inj-sev').value);
  try{
    const data = await apiPost('/api/inject',{attack:a,severity:s});
    if(!data.ok){ document.getElementById('inj-out').innerHTML='<p class="empty">Error: '+(data.error||'unknown')+'</p>'; return; }
    document.getElementById('inj-out').innerHTML = data.html;
    refreshStatus(); refreshEvents();
  }catch(e){ document.getElementById('inj-out').innerHTML='<p class="empty">Error: '+e.message+'</p>'; }
}
async function randomInject(){
  try{
    const data = await apiPost('/api/inject',{attack:'random',severity:0});
    if(!data.ok){ document.getElementById('inj-out').innerHTML='<p class="empty">Error: '+(data.error||'unknown')+'</p>'; return; }
    document.getElementById('inj-out').innerHTML = data.html;
    refreshStatus(); refreshEvents();
  }catch(e){ document.getElementById('inj-out').innerHTML='<p class="empty">Error: '+e.message+'</p>'; }
}

async function policyLookup(){
  const levels = ['p0','p1','p2','p3','p4'].map(id => parseInt(document.getElementById(id).value));
  try{
    const data = await apiPost('/api/policy',{levels});
    if(!data.ok){ document.getElementById('pol-out').innerHTML='<p class="empty">Error: '+(data.error||'unknown')+'</p>'; return; }
    document.getElementById('pol-out').innerHTML = data.html;
  }catch(e){ document.getElementById('pol-out').innerHTML='<p class="empty">Error: '+e.message+'</p>'; }
}

async function runAblation(ev){
  const b = ev.target; b.disabled=true; const old=b.textContent; b.textContent='Running...';
  try{
    const data = await apiPost('/api/ablation',{episodes:400});
    if(!data.ok){
      document.getElementById('abl-out').innerHTML='<p class="empty">Error: '+(data.error||'unknown')+'</p>'; return;
    }
    document.getElementById('abl-out').innerHTML = `<pre>${data.summary}</pre>
      <img src="data:image/png;base64,${data.image}" style="max-width:100%;margin-top:12px;border-radius:8px"/>`;
  }catch(e){ document.getElementById('abl-out').innerHTML='<p class="empty">Error: '+e.message+'</p>'; }
  finally { b.disabled=false; b.textContent=old; }
}

function exportCSV(){ window.location.href = '/api/export_log'; }
function exportQ(){   window.location.href = '/api/export_q'; }

document.addEventListener('DOMContentLoaded', async ()=>{
  initCharts();
  await refreshStatus(); await refreshEvents(); await refreshQTable();
  setInterval(refreshStatus, 5000);
});
</script></body></html>
"""


LOGS_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Logs</title>
<style>__STYLE__</style></head><body>
<div class="layout">
 <aside class="sidebar"><h2 class="logo">CloudSec<span>RL</span></h2>
  <nav><a href="/">Dashboard</a><a class="active" href="/logs">Full Logs</a></nav>
 </aside>
 <main class="main">
  <header><div><h1>Security Event Logs</h1>
   <p class="subtitle">All stored events (SQLite + CSV)</p></div></header>
  <div class="info-banner">
    Every event your agent processed is stored here. The same data is written to
    <code>security_logs.csv</code>.
  </div>
  <section class="panel">
   <table class="events-table" id="logs-table"><thead>
    <tr><th>Time</th><th>VM</th><th>Threat</th><th>Severity</th><th>State</th>
    <th>RL Action</th><th>Mitigation</th><th>Result</th><th>Reward</th></tr>
   </thead><tbody></tbody></table>
  </section>
 </main>
</div>
<script>
async function loadLogs(){
  const r = await fetch('/api/events?limit=200');
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch(e){ data = {events:[]}; }
  const tb = document.querySelector('#logs-table tbody'); tb.innerHTML='';
  if(!data.events.length){ tb.innerHTML='<tr><td colspan="9" class="empty">No events logged yet.</td></tr>'; return; }
  data.events.forEach(e=>{
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>${e.timestamp}</td><td>${e.vm_id}</td>
      <td>${e.threat_type}</td><td>${e.severity}</td><td>${e.state}</td>
      <td>${e.action_name}</td><td>${e.mitigation}</td>
      <td>${e.success?'Success':'Failed'}</td>
      <td>${e.reward>0?'+'+e.reward:e.reward}</td>`;
    tb.appendChild(tr);
  });
}
loadLogs();
</script></body></html>
"""


def render(html):
    return html.replace("__STYLE__", STYLE)


# =====================================================================
# ROUTES
# =====================================================================
def _json_ok(payload=None, **kw):
    d = {"ok": True}
    if payload is not None: d.update(payload)
    d.update(kw)
    return jsonify(d)

def _json_err(msg):
    return jsonify({"ok": False, "error": str(msg)})


@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render(PAGE_HTML)

@app.route("/logs")
def logs_page():
    return render(LOGS_HTML)

@app.route("/api/status")
def api_status():
    try:
        return _json_ok({
            "cloud": cloud_env.snapshot(),
            "metrics": summarize(get_all_events()),
            "last_result": last_result
        })
    except Exception as e:
        return _json_err(e)

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body = request.get_json(silent=True) or {}
        t = (body.get("type") or "random").lower()
        event = generate_event(t)
        return _json_ok({"result": process_event(event)})
    except Exception as e:
        return _json_err(e)

@app.route("/api/train", methods=["POST"])
def api_train():
    global training_history, agent
    try:
        body = request.get_json(silent=True) or {}
        episodes = max(10, min(int(body.get("episodes", 600)), 3000))
        agent = QLearningAgent()
        hist = train_full_agent(agent, episodes=episodes, steps=EPISODE_LENGTH,
                                drift=True, adaptive=True, drift_strength=0.4,
                                p_attack=0.35, noise=0.08, seed=RANDOM_SEED)
        training_history = hist
        return _json_ok({"episodes": episodes, "history": hist})
    except Exception as e:
        return _json_err(e)

@app.route("/api/events")
def api_events():
    try:
        return _json_ok({"events": get_recent_events(int(request.args.get("limit", 20)))})
    except Exception as e:
        return _json_err(e)

@app.route("/api/qtable")
def api_qtable():
    try:
        q = agent.q_table
        scores = np.abs(q).max(axis=1)
        top_idx = np.argsort(-scores)[:15]
        rows = []
        lv = ["L","M","H"]
        for s in top_idx:
            parts = [lv[(int(s) // (3**i)) % 3] for i in range(NF)]
            label = " | ".join(f"{FEATS[i]}={parts[i]}" for i in range(NF))
            rows.append({"label": f"#{int(s)}: {label}",
                         "values": [float(x) for x in q[s]]})
        return _json_ok({"states": ATTACKS, "actions": ACTIONS,
                         "q_values": q.tolist(), "top_states": rows,
                         "epsilon": round(agent.epsilon, 4)})
    except Exception as e:
        return _json_err(e)

@app.route("/api/inject", methods=["POST"])
def api_inject():
    try:
        body = request.get_json(silent=True) or {}
        attack = body.get("attack", "random")
        severity = float(body.get("severity", 0.0))
        if attack == "random" or attack not in ATTACKS:
            event = generate_event("random")
        else:
            mapping = {"Normal":"normal", "DDoS":"ddos", "Brute Force":"brute",
                       "Port Scan":"portscan", "Data Exfiltration":"exfil",
                       "Malware/Lateral Move":"malware"}
            k = ATTACKS.index(attack)
            rng = np.random.default_rng()
            P = make_profiles(0.0)
            mean = P[0] + severity * (P[k] - P[0])
            feats = np.clip(mean + rng.normal(0, 0.05, NF), 0, 1)
            event = generate_event(mapping[attack])
            event["features"] = feats
            event["cpu"] = round(float(20 + 75 * feats[0]), 2)
            event["memory"] = round(float(25 + 70 * feats[4]), 2)
            event["network_requests"] = int(10 + 900 * feats[0])
            event["failed_logins"] = int(30 * feats[1])

        r = process_event(event)
        bars = "".join(
            f"<div style='margin:4px 0'><span style='display:inline-block;width:150px'>{FEATS[i]}</span>"
            f"<span class='feature-bar'><span style='width:{r['features'][i]*100:.0f}%'></span></span>"
            f" {r['features'][i]:.2f}</div>"
            for i in range(NF))
        col = "#15803d" if r["success"] else "#b91c1c"
        verdict = "Effective mitigation" if r["success"] else "Suboptimal / missed"
        try:
            best_action = ACTIONS[int(OPT[ATTACKS.index(r['actual_threat'])])]
        except Exception:
            best_action = ACTIONS[0]
        html = (f"<div class='panel'><h3>Injected threat: {r['actual_threat']}</h3>"
                f"{bars}"
                f"<p style='margin-top:10px'>Agent response: <b>{r['action_name']}</b></p>"
                f"<p>Best-known response: <b>{best_action}</b></p>"
                f"<p>Reward: <b>{r['reward']}</b> - "
                f"<b style='color:{col} !important'>{verdict}</b></p></div>")
        return _json_ok({"html": html})
    except Exception as e:
        return _json_err(e)

@app.route("/api/policy", methods=["POST"])
def api_policy():
    try:
        body = request.get_json(silent=True) or {}
        levels = body.get("levels", [0,0,0,0,0])
        f = np.array([LEVEL_VALUE[int(x)] for x in levels], dtype=float)
        s = int(encode_all(f))
        q = agent.q_table[s]
        a = int(np.argmax(q))
        ra = rule_action(f)
        unseen = bool(np.all(q == 0))
        lo, hi = float(q.min()), float(q.max())
        bars = ""
        for i in range(NA):
            wd = 100 * (q[i] - lo) / (hi - lo + 1e-9)
            c = "#16a34a" if i == a else "#64748b"
            bars += (f"<div style='margin:4px 0'><span style='display:inline-block;width:190px'>{ACTIONS[i]}</span>"
                     f"<span style='display:inline-block;width:240px;height:12px;background:#334155;border-radius:6px;vertical-align:middle'>"
                     f"<span style='display:block;width:{wd:.0f}%;height:12px;background:{c};border-radius:6px'></span></span> Q = {q[i]:.2f}</div>")
        warn = ("<p style='color:#b45309 !important'>This state was never visited in training (defaults to Allow).</p>"
                if unseen else "")
        html = (f"<div class='panel'><h3>Decision for state #{s}</h3>{bars}{warn}"
                f"<p style='margin-top:10px'>RL agent chooses: <b>{ACTIONS[a]}</b></p>"
                f"<p>Static rule engine chooses: <b>{ACTIONS[ra]}</b></p></div>")
        return _json_ok({"html": html})
    except Exception as e:
        return _json_err(e)

@app.route("/api/ablation", methods=["POST"])
def api_ablation():
    try:
        body = request.get_json(silent=True) or {}
        episodes = max(50, min(int(body.get("episodes", 400)), 1500))
        a1 = QLearningAgent()
        h1 = train_full_agent(a1, episodes=episodes, steps=EPISODE_LENGTH,
                              drift=True, adaptive=True, drift_strength=0.4,
                              p_attack=0.35, noise=0.08, seed=RANDOM_SEED)
        a2 = QLearningAgent()
        h2 = train_full_agent(a2, episodes=episodes, steps=EPISODE_LENGTH,
                              drift=True, adaptive=False, drift_strength=0.4,
                              p_attack=0.35, noise=0.08, seed=RANDOM_SEED)
        fig = Figure(figsize=(12, 4.5))
        ax = fig.subplots(1, 2)
        E = len(h1["rewards"]); w = max(3, E // 30)
        x = np.arange(1, E + 1)
        a = ax[0]
        a.plot(x, _smooth(h1["successes"], w), lw=2, label="With adaptation")
        a.plot(x, _smooth(h2["successes"], w), lw=2, label="Without adaptation")
        a.axvline(E // 2 + 0.5, color="red", ls=":", lw=2, label="Attack drift")
        a.set_title("Successful mitigations per episode"); a.grid(alpha=.3); a.legend()
        a = ax[1]
        a.plot(x, h1["epsilon"], lw=2, label="With adaptation")
        a.plot(x, h2["epsilon"], lw=2, label="Without adaptation")
        a.axvline(E // 2 + 0.5, color="red", ls=":", lw=2)
        a.set_title("Exploration rate"); a.grid(alpha=.3); a.legend()
        fig.tight_layout()
        b64 = fig_to_b64(fig)
        k = max(5, E // 5)
        s1 = float(np.mean(h1["successes"][-k:]))
        s2 = float(np.mean(h2["successes"][-k:]))
        summary = (f"Final {k} episodes (post-drift):\n"
                   f"  With self-adaptation    : {s1:.1f} successful mitigations / episode\n"
                   f"  Without self-adaptation : {s2:.1f} successful mitigations / episode\n"
                   f"  Adaptation triggers (with adaptation): "
                   f"{', '.join(str(e) for e in h1['adapt']) or 'none'}")
        return _json_ok({"summary": summary, "image": b64})
    except Exception as e:
        return _json_err(e)

@app.route("/api/export_log")
def api_export_log():
    try:
        if not os.path.exists(CSV_PATH):
            return _json_err("No log file yet."), 404
        return send_file(CSV_PATH, as_attachment=True,
                         download_name="security_logs.csv", mimetype="text/csv")
    except Exception as e:
        return _json_err(e)

@app.route("/api/export_q")
def api_export_q():
    try:
        q = agent.q_table
        path = os.path.join(EXPORT_DIR, "q_table.csv")
        lv = ["Low","Med","High"]
        rows = []
        for s in range(q.shape[0]):
            row = {"state_id": s}
            for i in range(NF):
                row[FEATS[i]] = lv[(s // (3**i)) % 3]
            for a in range(NA):
                row["Q: " + ACTIONS[a]] = round(float(q[s, a]), 3)
            row["Best Action"] = (ACTIONS[int(np.argmax(q[s]))]
                                  if np.any(q[s] != 0) else "Allow/Monitor (unvisited)")
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        return send_file(path, as_attachment=True,
                         download_name="q_table.csv", mimetype="text/csv")
    except Exception as e:
        return _json_err(e)

@app.route("/api/reset", methods=["POST"])
def api_reset():
    global agent, last_result, training_history, cloud_env
    try:
        clear_events()
        agent = QLearningAgent()
        cloud_env = CloudEnvironment(num_vms=NUM_VMS)
        last_result = {}
        training_history = {"episodes": [], "rewards": [], "threats": [],
                            "successes": [], "failures": [], "epsilon": [], "adapt": []}
        return _json_ok()
    except Exception as e:
        return _json_err(e)

@app.errorhandler(Exception)
def _global_err(e):
    return _json_err(e), 200


# =====================================================================
# LAUNCH — Render compatible
# =====================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
