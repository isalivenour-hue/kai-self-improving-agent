"""A more realistic neural network built from scratch.

Task: classify real breast-cancer measurements as benign or malignant.
The model, loss, backpropagation, mini-batches, and optimizer are NumPy-only.
Scikit-learn is used only to load the public dataset and make a split.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
data = load_breast_cancer()
X = data.data.astype(np.float64)
y = data.target.astype(np.float64).reshape(-1, 1)

# Keep the test set untouched until the end. Standardize using training data only.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Two hidden layers: 30 -> 24 -> 12 -> 1.
def init(fan_in, fan_out):
    return rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))

W1, b1 = init(30, 24), np.zeros((1, 24))
W2, b2 = init(24, 12), np.zeros((1, 12))
W3, b3 = init(12, 1), np.zeros((1, 1))

def relu(z):
    return np.maximum(z, 0)

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))

def forward(x):
    z1 = x @ W1 + b1; h1 = relu(z1)
    z2 = h1 @ W2 + b2; h2 = relu(z2)
    out = sigmoid(h2 @ W3 + b3)
    return z1, h1, z2, h2, out

def loss(pred, truth):
    eps = 1e-8
    return -np.mean(truth*np.log(pred+eps) + (1-truth)*np.log(1-pred+eps))

def evaluate(x, truth):
    *_, prob = forward(x)
    pred = (prob >= 0.5).astype(int)
    truth_i = truth.astype(int)
    tp = int(((pred == 1) & (truth_i == 1)).sum())
    tn = int(((pred == 0) & (truth_i == 0)).sum())
    fp = int(((pred == 1) & (truth_i == 0)).sum())
    fn = int(((pred == 0) & (truth_i == 1)).sum())
    accuracy = (tp + tn) / len(truth)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return loss(prob, truth), accuracy, precision, recall, f1, (tn, fp, fn, tp)

# Adam optimizer, implemented directly.
params = [W1, b1, W2, b2, W3, b3]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8
learning_rate = 0.003
batch_size = 32

for step in range(1, 1201):
    order = rng.permutation(len(X_train))
    for start in range(0, len(order), batch_size):
        idx = order[start:start+batch_size]
        xb, yb = X_train[idx], y_train[idx]
        z1, h1, z2, h2, out = forward(xb)
        n = len(xb)
        # Backpropagation through sigmoid, ReLU, and both hidden layers.
        d3 = (out - yb) / n
        gW3, gb3 = h2.T @ d3, d3.sum(axis=0, keepdims=True)
        dh2 = d3 @ W3.T
        dz2 = dh2 * (z2 > 0)
        gW2, gb2 = h1.T @ dz2, dz2.sum(axis=0, keepdims=True)
        dh1 = dz2 @ W2.T
        dz1 = dh1 * (z1 > 0)
        gW1, gb1 = xb.T @ dz1, dz1.sum(axis=0, keepdims=True)
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1*m[i] + (1-beta1)*g
            v[i] = beta2*v[i] + (1-beta2)*(g*g)
            mhat = m[i] / (1-beta1**step)
            vhat = v[i] / (1-beta2**step)
            p -= learning_rate*mhat/(np.sqrt(vhat)+eps)
    if step in (1, 100, 300, 600, 900, 1200):
        tr = evaluate(X_train, y_train)
        te = evaluate(X_test, y_test)
        print(f"epoch={step:4d} train_loss={tr[0]:.4f} val_loss={te[0]:.4f} val_acc={te[1]:.1%} val_f1={te[4]:.1%}")

final = evaluate(X_test, y_test)
print("\nFinal test metrics")
print(f"accuracy={final[1]:.1%}  precision={final[2]:.1%}  recall={final[3]:.1%}  F1={final[4]:.1%}")
print(f"confusion_matrix=[[TN={final[5][0]}, FP={final[5][1]}], [FN={final[5][2]}, TP={final[5][3]}]]")
print("\nThis is a demonstration model, not a medical diagnostic system.")
