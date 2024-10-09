#random input taken
import numpy as np
import matplotlib.pyplot as plt

# Sample data
np.random.seed(42)
X = 2 * np.random.rand(50, 1)
y = 4 + 3 * X + np.random.randn(50, 1)

# Initial values
W = np.random.randn(1, 1)
b = np.random.randn(1)

# Parameters
alpha = 0.1
tolerance = 1e-6
m = len(X)

# Gradient descent loop
while True:
    pred = W * X + b
    dW = -(2/m) * np.sum(X * (y - pred))
    db = -(2/m) * np.sum(y - pred)
    
    new_W = W - alpha * dW
    new_b = b - alpha * db
    
    if np.abs(new_W - W).sum() < tolerance and np.abs(new_b - b).sum() < tolerance:
        break
    
    W, b = new_W, new_b

# Print results
print(W[0][0], b[0])

# Plot
plt.scatter(X, y)
X_new = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = W * X_new + b
plt.plot(X_new, y_pred, color='red')
plt.show()
