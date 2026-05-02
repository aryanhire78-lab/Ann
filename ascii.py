import numpy as np

# ASCII values for '0' to '9' (48 to 57)
ascii_values = list(range(48, 58))

# Convert ASCII values to 6-bit binary
inputs = [list(map(int, format(a, '06b'))) for a in ascii_values]

# Target: Even = 1, Odd = 0
targets = [1 if (a % 2 == 0) else 0 for a in ascii_values]

# Initialize weights and bias
weights = np.zeros(6)
bias = 0.0
learning_rate = 0.1

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Training
for epoch in range(25):
    for x, target in zip(inputs, targets):
        net_input = np.dot(weights, x) + bias
        output = step(net_input)
        
        # Calculate error
        error = target - output
        
        # Update weights and bias
        weights += learning_rate * error * np.array(x)
        bias += learning_rate * error

print("Training Completed")
print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")

# Testing
print("\nASCII | Digit | Prediction (1=Even, 0=Odd)")
print("-" * 44)

for a in ascii_values:
    x = list(map(int, format(a, '06b')))
    prediction = step(np.dot(weights, x) + bias)
    # Using f-strings for clean table alignment
    print(f"  {a}  |   {chr(a)}   | {prediction}")