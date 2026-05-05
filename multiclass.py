import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
data = load_iris()
X = data.data
y = data.target   # NOTE: no one-hot needed in PyTorch

# 2. Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# 5. Model (inbuilt activations)
model = nn.Sequential(
    nn.Linear(4, 20),
    nn.ReLU(),          # inbuilt activation
    nn.Linear(20, 3)
)

# 6. Loss + optimizer (inbuilt)
criterion = nn.CrossEntropyLoss()   # combines softmax + loss
optimizer = optim.Adam(model.parameters(), lr=0.02)

# 7. Training
epochs = 20

for epoch in range(epochs):
    model.train()

    # Forward
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 8. Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test).float().mean()
    print("\nFinal Accuracy:", accuracy.item())
