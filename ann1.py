import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = activation

    def forward(self, inputs):
        inputs = np.array(inputs)
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Create neurons using the same class
# Same weights and bias for fair comparison
weights = [1.0]
bias = 0.0

sigmoid_neuron = Neuron(weights, bias, sigmoid)
relu_neuron = Neuron(weights, bias, relu)
tanh_neuron = Neuron(weights, bias, tanh)

# Input range
x = np.linspace(-10, 10, 400)

# Forward pass through neurons
y_sigmoid = [sigmoid_neuron.forward([i]) for i in x]
y_relu = [relu_neuron.forward([i]) for i in x]
y_tanh = [tanh_neuron.forward([i]) for i in x]

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(x, y_sigmoid, label="Sigmoid", color="blue")
plt.title("Sigmoid Neuron Output")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, y_relu, label="ReLU", color="green")
plt.title("ReLU Neuron Output")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, y_tanh, label="Tanh", color="red")
plt.title("Tanh Neuron Output")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()