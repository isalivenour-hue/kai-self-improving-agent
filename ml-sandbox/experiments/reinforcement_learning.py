"""Persistent tabular reinforcement-learning demo in a small grid world."""
from pathlib import Path
import numpy as np
import json
STATE = Path(__file__).resolve().parents[1] / "state"
STATE.mkdir(exist_ok=True)
QFILE = STATE / "q_table.npy"
policy_path = STATE / "learning_policy.json"
policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
exploration = float(policy.get('exploration_epsilon', 0.10))
rng = np.random.default_rng(123)
q = np.load(QFILE) if QFILE.exists() else np.zeros((25, 4))
# 5x5 world: start 0, goal 24, a few fixed obstacles.
obstacles = {6, 7, 13, 18}
def step(s, a):
    r,c = divmod(s,5); nr,nc = r,c
    if a==0: nr-=1
    elif a==1: nr+=1
    elif a==2: nc-=1
    else: nc+=1
    ns = s if not (0<=nr<5 and 0<=nc<5) else nr*5+nc
    if ns in obstacles: ns=s
    return ns, (10.0 if ns==24 else -0.15)
for _ in range(500):
    s=0
    for _ in range(80):
        a = rng.integers(4) if rng.random()<exploration else int(np.argmax(q[s]))
        ns, reward = step(s,a)
        q[s,a] += 0.15*(reward + 0.95*np.max(q[ns]) - q[s,a])
        s=ns
        if s==24: break
np.save(QFILE,q)
print(f"reinforcement_learning: saved {QFILE}")
print(f"best_action_from_start={int(np.argmax(q[0]))} q_value={q[0].max():.3f}")
