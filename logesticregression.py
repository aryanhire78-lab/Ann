# use "pip install tensorflow" if you dont downloaded the tensorflow

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
data = load_iris()
X, y = data.data, data.target

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Neural Network Model
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

nn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train NN
nn_model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate NN
nn_loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)

# 2. Logistic Regression Model
log_model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', input_shape=(4,))
])

log_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Logistic Regression
log_model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate Logistic Regression
log_loss, log_acc = log_model.evaluate(X_test, y_test, verbose=0)

# Results
print(f"Neural Network Accuracy: {nn_acc:.4f}")
print(f"Logistic Regression Accuracy: {log_acc:.4f}")