import numpy as np

# Input Dataset (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target Output
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize Weights and Biases
np.random.seed(42)
W1 = np.random.rand(2, 2)
b1 = np.random.rand(1, 2)
W2 = np.random.rand(2, 1)
b2 = np.random.rand(1, 1)

# Training Parameters
learning_rate = 0.5
epochs = 5000

# Training Loop
for epoch in range(epochs):
    # Forward Propagation
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Loss (Mean Squared Error)
    error = Y - A2
    loss = np.mean(np.square(error))
    
    # Backpropagation
    dA2 = error * sigmoid_derivative(A2)
    error_hidden = dA2.dot(W2.T)
    dA1 = error_hidden * sigmoid_derivative(A1)
    
    # Update Weights and Biases
    W2 += A1.T.dot(dA2) * learning_rate
    b2 += np.sum(dA2, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(dA1) * learning_rate
    b1 += np.sum(dA1, axis=0, keepdims=True) * learning_rate
    
    # Print loss occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final Output
print("\nFinal Output:")
print(np.round(A2))