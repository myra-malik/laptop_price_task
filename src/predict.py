# src/predict.py — ultra simple, uses columns stored inside the model

import os, pickle
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path      = os.path.join(ROOT, "models", "regression_model_final.pkl")
data_path       = os.path.join(ROOT, "data", "train_data.csv")
metrics_path    = os.path.join(ROOT, "results", "test_metrics.txt")
predictions_path= os.path.join(ROOT, "results", "train_predictions.csv")

# load model (weights + feature_columns live together)
with open(model_path, "rb") as f:
    model = pickle.load(f)
w = model["weights"]
train_cols = model["feature_columns"]

# load data
df = pd.read_csv(data_path)
has_y = "Price" in df.columns
y = df["Price"].to_numpy(dtype=float).ravel() if has_y else None
X = df.drop(columns=["Price"]) if has_y else df

# one-hot encode, then align EXACTLY to training columns
X_enc = pd.get_dummies(X, drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)
X_enc = X_enc.reindex(columns=train_cols, fill_value=0.0)

# add bias
X_b = np.c_[np.ones((X_enc.shape[0], 1)), X_enc.to_numpy(dtype=float)]

# predict
yhat = X_b @ w
np.savetxt(predictions_path, yhat, fmt="%.2f")

# metrics (only if Price exists)
if has_y:
    err = y - yhat
    mse  = float((err**2).mean())
    rmse = float(np.sqrt(mse))
    r2   = 1.0 - float(np.sum(err**2)) / float(np.sum((y - y.mean())**2))
    with open(metrics_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")

print("Predictions ->", predictions_path)
if has_y:
    print("Metrics ->", metrics_path)
