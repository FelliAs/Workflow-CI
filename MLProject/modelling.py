"""
modelling.py
============
KNeighborsRegressor — MLflow Tracking (Kriteria 3: Workflow CI)
Tidak ada DagsHub. Tracking URI & Run ID dikontrol sepenuhnya oleh mlflow run CLI.
Jika dijalankan langsung (python modelling.py), tracking ke ./mlruns secara default.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ===========================================================
# 0. KONFIGURASI
# ===========================================================
BASELINE_PARAMS = {
    "n_neighbors": 10,
    "weights"    : "uniform",
    "metric"     : "euclidean",
    "algorithm"  : "ball_tree",
    "leaf_size"  : 30,
    "p"          : 2,
    "n_jobs"     : -1,
}

# ===========================================================
# 1. SETUP MLFLOW
# ===========================================================
# PENTING: Tidak ada mlflow.set_tracking_uri() atau mlflow.set_experiment() di sini.
# Jika dijalankan via "mlflow run .", CLI yang mengontrol MLFLOW_TRACKING_URI & MLFLOW_RUN_ID.
# Jika dijalankan langsung "python modelling.py", MLflow default ke ./mlruns.
print(f"[1/4] MLflow tracking URI : {mlflow.get_tracking_uri()}")

# ===========================================================
# 2. LOAD DATA
# ===========================================================
print("[2/4] Memuat data preprocessing ...")
BASE_DIR = Path(__file__).parent / "memory_crisis_preprocessing"

X_train = pd.read_csv(BASE_DIR / "X_train.csv")
X_test  = pd.read_csv(BASE_DIR / "X_test.csv")
y_train = pd.read_csv(BASE_DIR / "y_train.csv").squeeze()
y_test  = pd.read_csv(BASE_DIR / "y_test.csv").squeeze()

print(f"      X_train : {X_train.shape}")
print(f"      X_test  : {X_test.shape}")

# ===========================================================
# 3. TRAIN MODEL
# ===========================================================
print("[3/4] Training KNeighborsRegressor ...")
model = KNeighborsRegressor(**BASELINE_PARAMS)
model.fit(X_train, y_train)
print("      Training selesai.")

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

metrics = {
    "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    "test_rmse" : np.sqrt(mean_squared_error(y_test,  y_pred_test)),
    "train_mae" : mean_absolute_error(y_train, y_pred_train),
    "test_mae"  : mean_absolute_error(y_test,  y_pred_test),
    "train_r2"  : r2_score(y_train, y_pred_train),
    "test_r2"   : r2_score(y_test,  y_pred_test),
}
for k, v in metrics.items():
    print(f"      {k:12s}: {v:.4f}")

# ===========================================================
# 4. MLFLOW MANUAL LOGGING
# ===========================================================
print("[4/4] Logging ke MLflow ...")

artifact_dir = Path(__file__).parent / "artifacts_temp"
artifact_dir.mkdir(exist_ok=True)

# --- Residual Plot ---
residuals = y_test.values - y_pred_test
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc = axes[0].scatter(y_test, y_pred_test, alpha=0.4, s=15,
                     c=np.abs(residuals), cmap="plasma")
lims = [
    min(float(y_test.min()), y_pred_test.min()),
    max(float(y_test.max()), y_pred_test.max()),
]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Price (USD)")
axes[0].set_ylabel("Predicted Price (USD)")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()
plt.colorbar(sc, ax=axes[0], label="|Residual|")

axes[1].hist(residuals, bins=50, color="#5C6BC0", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].axvline(np.mean(residuals), color="orange", linestyle="-",
                label=f"Mean={np.mean(residuals):.2f}")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")
axes[1].legend()

fig.suptitle("Residual Analysis — KNN | RAM Price", fontsize=13, fontweight="bold")
plt.tight_layout()
residual_path = artifact_dir / "residual_plot.png"
fig.savefig(residual_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# Log langsung — jika ada active run dari CLI, log ke sana.
# Jika tidak ada, buka run baru.
active = mlflow.active_run()
if active:
    run_id = active.info.run_id
    mlflow.log_param("model", "KNeighborsRegressor")
    for pname, pval in BASELINE_PARAMS.items():
        mlflow.log_param(pname, pval)
    for mname, mval in metrics.items():
        mlflow.log_metric(mname, mval)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="knn_model")
    mlflow.log_artifact(str(residual_path), artifact_path="plots")
    print(f"\n      Run ID    : {run_id}")
else:
    with mlflow.start_run(run_name="KNN_CI_Run") as run:
        mlflow.log_param("model", "KNeighborsRegressor")
        for pname, pval in BASELINE_PARAMS.items():
            mlflow.log_param(pname, pval)
        for mname, mval in metrics.items():
            mlflow.log_metric(mname, mval)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="knn_model")
        mlflow.log_artifact(str(residual_path), artifact_path="plots")
        print(f"\n      Run ID    : {run.info.run_id}")

print("\n[SELESAI] Training & logging selesai.")
