from data_preprocessing import load_and_preprocess
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
x_train_encoded, x_test_encoded, y_train, y_test, feature_columns = load_and_preprocess(
    os.path.join(ROOT, "data", "train_data.csv"),
    test_size=0.2,
    seed=36
)

# src/train_model.py  — minimal ridge retrain (no extra files)

import pickle
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path    = os.path.join(ROOT, "data", "train_data.csv")
models_dir   = os.path.join(ROOT, "models")
results_dir  = os.path.join(ROOT, "results")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# load
df = pd.read_csv(data_path)

# split target
y = df["Price"].to_numpy(dtype=float).ravel()
X = df.drop(columns=["Price"])

# one-hot encode (this defines the training columns)
X_enc = pd.get_dummies(X, drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)
feature_columns = list(X_enc.columns)
X_np = X_enc.to_numpy(dtype=float)

# simple split (80/20, fixed)
rng = np.random.default_rng(42)
idx = rng.permutation(len(X_np))
cut = int(0.8 * len(X_np))
tr, te = idx[:cut], idx[cut:]
Xtr, Xte, ytr, yte = X_np[tr], X_np[te], y[tr], y[te]

# add bias
Xtr_b = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
Xte_b = np.c_[np.ones((Xte.shape[0], 1)), Xte]

# ridge fit: w = (X'X + λI)^(-1) X'y  (no penalty on bias)
lam = 0.1
I = np.eye(Xtr_b.shape[1]); I[0,0] = 0
w = np.linalg.pinv(Xtr_b.T @ Xtr_b + lam * I) @ (Xtr_b.T @ ytr)

# predict + metrics
yhat = Xte_b @ w
err = yte - yhat
mse  = float((err**2).mean())
rmse = float(np.sqrt(mse))
r2   = 1.0 - float(np.sum(err**2)) / float(np.sum((yte - yte.mean())**2))
print("Ridge λ=0.1  MSE:", round(mse,2), "RMSE:", round(rmse,2), "R²:", round(r2,2))

# save ONE final model, embedding the training columns inside
with open(os.path.join(models_dir, "regression_model_final.pkl"), "wb") as f:
    pickle.dump({
        "weights": w,
        "has_bias": True,
        "model": "ridge",
        "lambda": lam,
        "feature_columns": feature_columns  # <-- embedded here
    }, f)

# write results (from the test split above)
with open(os.path.join(results_dir, "train_predictions.csv"), "w") as f:
    for v in yhat:
        f.write(f"{float(v):.2f}\n")

with open(os.path.join(results_dir, "train_metrics.txt"), "w") as f:
    f.write("Regression Metrics:\n")
    f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
    f.write(f"R-squared (R²) Score: {r2:.2f}\n")


