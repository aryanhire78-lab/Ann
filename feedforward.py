import numpy as np

# Input Dataset (XOR Example)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

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

# Initialize Network
np.random.seed(42)
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

W1 = np.random.rand(input_neurons, hidden_neurons)
b1 = np.random.rand(1, hidden_neurons)
W2 = np.random.rand(hidden_neurons, output_neurons)
b2 = np.random.rand(1, output_neurons)

# Training Parameters
learning_rate = 0.5
epochs = 10000

# Training (Feedforward + Backpropagation)
for epoch in range(epochs):
    # Feed Forward
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Error
    error = Y - A2
    
    # Backpropagation
    dA2 = error * sigmoid_derivative(A2)
    error_hidden = dA2.dot(W2.T)
    dA1 = error_hidden * sigmoid_derivative(A1)
    
    # Update Weights
    W2 += A1.T.dot(dA2) * learning_rate
    b2 += np.sum(dA2, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(dA1) * learning_rate
    b1 += np.sum(dA1, axis=0, keepdims=True) * learning_rate
    
    # Print Loss Occasionally
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final Output
print("\nFinal Output:")
print(np.round(A2))