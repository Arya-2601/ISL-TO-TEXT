# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading data...", flush=True)

# Load CSV
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\ISL_project\isl_angles.csv")

# Split features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {len(X_train)}", flush=True)
print(f"Testing samples  : {len(X_test)}", flush=True)

# Train Random Forest
print("\nTraining model...", flush=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%", flush=True)
print("\nPer-letter accuracy:", flush=True)
print(classification_report(y_test, y_pred), flush=True)

# Save the model
model_path = r"C:\Users\user\OneDrive\Desktop\projects frontend\ipd\ISL-TO-TEXT\isl_angles_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved at: {model_path}", flush=True)
print("Day 4 Complete!", flush=True)