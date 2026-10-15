"""A tiny neural network built from scratch with NumPy."""
import numpy as np

# XOR is not linearly separable, so it needs a hidden layer.
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([[0.], [1.], [1.], [0.]])

rng = np.random.default_rng(7)
W1 = rng.normal(0, 0.8, (2, 4)); b1 = np.zeros((1, 4))
W2 = rng.normal(0, 0.8, (4, 1)); b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(pred, truth):
    eps = 1e-8
    return -np.mean(truth*np.log(pred+eps) + (1-truth)*np.log(1-pred+eps))

learning_rate = 1.2
for step in range(5000):
    # Forward pass
    hidden = np.tanh(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    loss = binary_cross_entropy(output, y)

    # Backpropagation: derivatives from the output back to the inputs
    d_output = output - y
    dW2 = hidden.T @ d_output / len(X)
    db2 = np.mean(d_output, axis=0, keepdims=True)
    d_hidden = (d_output @ W2.T) * (1 - hidden**2)
    dW1 = X.T @ d_hidden / len(X)
    db1 = np.mean(d_hidden, axis=0, keepdims=True)

    # Gradient descent update
    W2 -= learning_rate*dW2; b2 -= learning_rate*db2
    W1 -= learning_rate*dW1; b1 -= learning_rate*db1

    if step in (0, 100, 1000, 4999):
        print(f"step={step:4d} loss={loss:.5f}")

predictions = (output >= 0.5).astype(int)
print("\nXOR predictions:")
for features, probability, prediction, expected in zip(X, output.ravel(), predictions.ravel(), y.ravel()):
    print(f"{features.astype(int)} -> probability={probability:.3f}, prediction={prediction}, expected={int(expected)}")
print(f"\naccuracy={(predictions == y).mean():.1%}")
