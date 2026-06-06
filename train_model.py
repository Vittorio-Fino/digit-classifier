"""
Train an MLP digit classifier on MNIST (28x28) and export it with joblib.
Run this once before starting the server.
"""

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

print("Fetching MNIST dataset (downloads ~12 MB on first run)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        max_iter=60,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        learning_rate_init=0.001,
    )),
])

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")
