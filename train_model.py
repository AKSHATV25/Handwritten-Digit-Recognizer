"""
Trains a handwritten digit classifier on the scikit-learn digits dataset
(8x8 grayscale images, 0-9) and saves the trained model + scaler to disk.

Run once locally: python train_model.py
This produces model.pkl, which app.py loads at runtime.
"""

import pickle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    print("Loading digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc * 100:.2f}%")

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler}, f)

    print("Saved model.pkl")


if __name__ == "__main__":
    main()
