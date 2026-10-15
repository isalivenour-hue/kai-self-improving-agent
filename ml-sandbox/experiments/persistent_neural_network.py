"""Persistent neural-network experiment for Noor.

The model, checkpoint, training history, and identity context stay inside the
sandbox. Each run performs more training and saves the new weights.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import urllib.request
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
CONTEXT_PATH = ROOT / "data" / "chatgpt_context.json"
STATE.mkdir(exist_ok=True)
IDENTITY_PATH = STATE / "identity.json"
CHECKPOINT = STATE / "model.npz"
HISTORY = STATE / "training_log.jsonl"
POLICY_PATH = STATE / "learning_policy.json"

identity = {
    "name": "KAI",
    "system_type": "persistent artificial cognitive system",
    "architecture": ["memory", "cognition", "agency", "meta-cognition", "learning", "self-model", "world-model", "hardware-model"],
    "owner": "Nour",
    "creator": "Nour",
    "name_spelling": "N as in Nile, O as in ocean, U as in universe, R as in rabbit.",
    "learning_sources": ["public internet", "Wikipedia", "GitHub"],
    "human_identity": "Nour is the human who built and operates this experiment.",
    "purpose": "Develop measurable learning, memory, reasoning, and self-improvement capabilities over time.",
    "runtime": "isolated ML sandbox with internet access",
    "limitations": [
        "This identity is stored context, not consciousness or self-awareness.",
        "The model learns statistical patterns; it does not independently understand a person."
    ]
}
IDENTITY_PATH.write_text(json.dumps(identity, indent=2) + "\n")
context = json.loads(CONTEXT_PATH.read_text()) if CONTEXT_PATH.exists() else {}
if POLICY_PATH.exists():
    policy = json.loads(POLICY_PATH.read_text())
else:
    policy = {"learning_rate": 0.003, "epochs_per_cycle": 5, "revision": 0, "reason": "initial bounded policy"}
learning_rate = float(np.clip(policy.get("learning_rate", 0.003), 0.0005, 0.01))
epochs_per_cycle = int(np.clip(policy.get("epochs_per_cycle", 5), 3, 8))

# Verify the sandbox still has internet access without executing downloaded data.
internet_urls = {"example": "https://example.com", "wikipedia": "https://www.wikipedia.org/", "github": "https://github.com/"}
internet_status = {}
for label, url in internet_urls.items():
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "NoorSandboxML/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            internet_status[label] = response.status
    except Exception as exc:
        internet_status[label] = f"unavailable: {type(exc).__name__}"

# Real dataset: 569 samples and 30 measurements.
data = load_breast_cancer()
X = data.data.astype(np.float64)
y = data.target.astype(np.float64).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

rng = np.random.default_rng(42)
if CHECKPOINT.exists():
    saved = np.load(CHECKPOINT)
    W1, b1, W2, b2, W3, b3 = [saved[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
    step = int(saved["step"])
    print(f"Loaded persistent checkpoint at training_step={step}")
else:
    def init(a, b): return rng.normal(0, np.sqrt(2 / a), (a, b))
    W1, b1 = init(30, 24), np.zeros((1, 24))
    W2, b2 = init(24, 12), np.zeros((1, 12))
    W3, b3 = init(12, 1), np.zeros((1, 1))
    step = 0

params = [W1, b1, W2, b2, W3, b3]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]

def relu(z): return np.maximum(z, 0)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -40, 40)))
def forward(x):
    z1 = x @ W1 + b1; h1 = relu(z1)
    z2 = h1 @ W2 + b2; h2 = relu(z2)
    return z1, h1, z2, h2, sigmoid(h2 @ W3 + b3)
def bce(p, t):
    e = 1e-8
    return float(-np.mean(t*np.log(p+e) + (1-t)*np.log(1-p+e)))
def metrics(x, t):
    *_, p = forward(x); pred = (p >= .5).astype(int); ti = t.astype(int)
    tp = int(((pred == 1) & (ti == 1)).sum()); tn = int(((pred == 0) & (ti == 0)).sum())
    fp = int(((pred == 1) & (ti == 0)).sum()); fn = int(((pred == 0) & (ti == 1)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2*precision*recall / max(precision+recall, 1e-12)
    return bce(p,t), float((pred == ti).mean()), precision, recall, f1

# Five more epochs per invocation: progress survives restarts.
for _ in range(epochs_per_cycle):
    for start in range(0, len(X_train), 32):
        idx = rng.permutation(len(X_train))[start:start+32]
        xb, yb = X_train[idx], y_train[idx]
        z1,h1,z2,h2,out = forward(xb); n = len(xb)
        d3 = (out-yb)/n; grads = [h1.T*0, d3.sum(axis=0,keepdims=True), h1.T @ (d3 @ W3.T * (z2>0)), None, h2.T @ d3, d3.sum(axis=0,keepdims=True)]
        dz2 = (d3 @ W3.T) * (z2 > 0); grads[2] = h1.T @ dz2; grads[3] = dz2.sum(axis=0,keepdims=True)
        dz1 = (dz2 @ W2.T) * (z1 > 0); grads[0] = xb.T @ dz1; grads[1] = dz1.sum(axis=0,keepdims=True)
        step += 1
        for i,(p,g) in enumerate(zip(params,grads)):
            m[i] = .9*m[i] + .1*g; v[i] = .999*v[i] + .001*g*g
            p -= learning_rate*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)

np.savez(CHECKPOINT, W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3,step=step)
loss, acc, precision, recall, f1 = metrics(X_test, y_test)
record = {"time": datetime.now(timezone.utc).isoformat(), "owner": "Nour", "policy_revision": policy.get("revision", 0), "learning_rate": learning_rate, "epochs_per_cycle": epochs_per_cycle, "internet_status": internet_status, "training_step": step, "test_loss": loss, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
with HISTORY.open("a") as f: f.write(json.dumps(record) + "\n")
print(f"identity: {identity['human_identity']}")
print(f"context loaded: {context.get('source', 'none')}")
print(f"internet: {internet_status}")
print(f"saved: {CHECKPOINT}")
print(f"training_step={step} test_accuracy={acc:.1%} precision={precision:.1%} recall={recall:.1%} F1={f1:.1%}")
print("Run this script again to continue learning from the saved checkpoint.")
