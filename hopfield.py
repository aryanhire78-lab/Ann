import numpy as np

# Sign Function
def sign(x):
    return np.where(x >= 0, 1, -1)

# Stored Patterns (Bipolar: -1, +1)
patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

# Training (Hebbian Learning)
n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    W += np.outer(p, p)

# Remove self-connections
np.fill_diagonal(W, 0)
print("Weight Matrix W:\n", W)

# Recall Function
def recall(input_pattern, W, steps=5):
    x = input_pattern.copy()
    for _ in range(steps):
        x = sign(np.dot(W, x))  # synchronous update
    return x

# Test with Noisy Input
if __name__ == "__main__":
    test_pattern = np.array([1, -1, -1, -1])  # noisy version
    recalled = recall(test_pattern, W)
    
    print("\nNoisy Input:", test_pattern)
    print("Recalled Pattern:", recalled)