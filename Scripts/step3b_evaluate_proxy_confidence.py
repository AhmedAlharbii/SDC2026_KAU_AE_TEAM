"""
STEP 3B: Offline Evaluation (Inference-Style + Truth Diagnostics)

Goal:
- Run the same style of scoring used in Step 4 (threat/confidence from model outputs).
- Keep this step offline for validation with truth-based diagnostics.

Why this exists:
- Engineers need a pre-production gate after training.
- This step verifies that confidence behaves correctly before operational inference.

Operational rule:
- Step 3B may use truth labels for diagnostics only.
- Step 4 must remain production-safe (no truth labels in scoring logic).
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from model_builder import build_model_from_config

print("=" * 80)
print("STEP 3B: OFFLINE EVALUATION GATE")
print("DebriSolver Competition - KAU Team")
print("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

MODEL_DIR = "trained_model"
SEQUENCE_DIR = "processed_sequences"
OUTPUT_DIR = "evaluation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MC_SAMPLES = 50


def require_file(path):
    """Fail fast when required artifacts are missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact not found: {path}")


def load_model_weights(model):
    """Load checkpoints in priority order and stop if none are valid."""
    candidates = [
        os.path.join(MODEL_DIR, "final_model.h5"),
        os.path.join(MODEL_DIR, "model_weights.weights.h5"),
        os.path.join(MODEL_DIR, "best_model.h5"),
    ]

    for ckpt in candidates:
        try:
            model.load_weights(ckpt)
            print(f"      ✓ Loaded weights: {ckpt}")
            return
        except Exception:
            continue

    raise RuntimeError("Could not load model weights from any known checkpoint")


def compute_threat_and_confidence(pred_mean, pred_std, x_input, feature_names):
    """
    Production-style scoring logic (no truth labels).

    Threat:
    - Uses current vs predicted Pc trend + urgency.

    Confidence:
    - Uses prediction uncertainty + data quantity + covariance quality.
    """

    n_samples = pred_mean.shape[0]
    threat_scores = np.zeros(n_samples)
    confidence_levels = np.zeros(n_samples)

    pc_idx = None
    log10_pc_idx = None
    time_to_tca_idx = None
    cr_r_idx = None

    for i, feat in enumerate(feature_names):
        if feat == "COLLISION_PROBABILITY":
            pc_idx = i
        elif feat == "log10_pc":
            log10_pc_idx = i
        elif feat == "time_to_tca_hours":
            time_to_tca_idx = i
        elif feat == "combined_cr_r":
            cr_r_idx = i

    main_pc_idx = log10_pc_idx if log10_pc_idx is not None else pc_idx

    for i in range(n_samples):
        input_seq = x_input[i]
        non_zero_mask = np.any(input_seq != 0, axis=1)

        if np.any(non_zero_mask):
            last_valid_idx = np.where(non_zero_mask)[0][-1]
            current_pc_scaled = input_seq[last_valid_idx, main_pc_idx]
        else:
            current_pc_scaled = 0.0

        predicted_pc_scaled = pred_mean[i, main_pc_idx]
        pc_change = predicted_pc_scaled - current_pc_scaled

        if log10_pc_idx is not None:
            base_threat = np.clip((predicted_pc_scaled + 2) * 17.5, 0, 70)
        else:
            base_threat = predicted_pc_scaled * 70

        trend_modifier = np.clip(pc_change * 30, -15, 15)

        if time_to_tca_idx is not None:
            predicted_time = pred_mean[i, time_to_tca_idx]
            urgency = max(0, (1 - (predicted_time + 1) / 2) * 15)
        else:
            urgency = 0

        threat_scores[i] = np.clip(base_threat + trend_modifier + urgency, 0, 100)

        avg_std = np.mean(pred_std[i])
        uncertainty_confidence = 1 / (1 + avg_std * 3)

        n_valid_timesteps = np.sum(non_zero_mask)
        data_confidence = min(1.0, n_valid_timesteps / 10)

        if cr_r_idx is not None:
            predicted_cov = pred_mean[i, cr_r_idx]
            cov_confidence = 1 / (1 + abs(predicted_cov))
        else:
            cov_confidence = 0.5

        confidence_levels[i] = np.clip(
            0.5 * uncertainty_confidence + 0.3 * data_confidence + 0.2 * cov_confidence,
            0.1,
            1.0,
        )

    return threat_scores, confidence_levels


print("\n[1/6] Validating artifacts and loading data...")
required = [
    os.path.join(MODEL_DIR, "model_config.json"),
    os.path.join(SEQUENCE_DIR, "X_test.npy"),
    os.path.join(SEQUENCE_DIR, "Y_test.npy"),
    os.path.join(SEQUENCE_DIR, "test_metadata.csv"),
    os.path.join(SEQUENCE_DIR, "feature_scaler.pkl"),
]
for path in required:
    require_file(path)
    print(f"      ✓ Found: {path}")

with open(os.path.join(MODEL_DIR, "model_config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

feature_names = config["feature_names"]
n_timesteps = config["n_timesteps"]
n_features = config["n_features"]

x_test = np.load(os.path.join(SEQUENCE_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SEQUENCE_DIR, "Y_test.npy"))
test_meta = pd.read_csv(os.path.join(SEQUENCE_DIR, "test_metadata.csv"))

print(f"      ✓ Test samples: {len(x_test):,}")
print(f"      ✓ Features: {n_features}")

print("\n[2/6] Loading model and running MC dropout predictions...")
model = build_model_from_config(config)
_ = model(np.zeros((1, n_timesteps, n_features)))
load_model_weights(model)

mc_predictions = []
for i in range(MC_SAMPLES):
    if (i + 1) % 10 == 0:
        print(f"      Progress: {i + 1}/{MC_SAMPLES}")
    mc_predictions.append(model(x_test, training=True).numpy())

mc_predictions = np.array(mc_predictions)
pred_mean = np.mean(mc_predictions, axis=0)
pred_std = np.std(mc_predictions, axis=0)

print("\n[3/6] Running production-style scoring (threat/confidence)...")
threat_scores, confidence_levels = compute_threat_and_confidence(
    pred_mean, pred_std, x_test, feature_names
)

sample_df = test_meta.copy()
sample_df["threat_score"] = threat_scores
sample_df["confidence_level"] = confidence_levels

for i, feat in enumerate(feature_names):
    sample_df[f"pred_{feat}"] = pred_mean[:, i]
    sample_df[f"std_{feat}"] = pred_std[:, i]
    sample_df[f"actual_{feat}"] = y_test[:, i]

sample_df["avg_uncertainty"] = np.mean(pred_std, axis=1)

# Event-level offline dashboard (mirrors Step 4 structure, but offline context).
event_dashboard = sample_df.groupby("event_id", as_index=False).agg({
    "threat_score": "last",
    "confidence_level": "last",
    "tca": "first",
    "total_cdms": "max",
    "target_pc": "last",
    "target_time_to_tca": "last",
})

high_threat = event_dashboard["threat_score"] > 50
high_conf = event_dashboard["confidence_level"] > 0.5
event_dashboard["quadrant"] = "NOT PRIORITY"
event_dashboard.loc[high_threat & high_conf, "quadrant"] = "ACT NOW"
event_dashboard.loc[high_threat & ~high_conf, "quadrant"] = "WATCH CLOSELY"
event_dashboard.loc[~high_threat & high_conf, "quadrant"] = "SAFELY IGNORE"

print("\n[4/6] Computing truth-based diagnostics (offline only)...")
actual_cols = [f"actual_{feat}" for feat in feature_names if f"actual_{feat}" in sample_df.columns]
pred_cols = [f"pred_{feat}" for feat in feature_names if f"pred_{feat}" in sample_df.columns]

actual_values = sample_df[actual_cols].values
pred_values = sample_df[pred_cols].values
sample_df["sample_mae"] = np.mean(np.abs(pred_values - actual_values), axis=1)

if "actual_log10_pc" in sample_df.columns and "pred_log10_pc" in sample_df.columns:
    sample_df["pc_abs_error"] = np.abs(sample_df["pred_log10_pc"] - sample_df["actual_log10_pc"])
else:
    sample_df["pc_abs_error"] = np.nan

sample_df["proxy_confidence_from_truth"] = 1.0 / (1.0 + 2.0 * sample_df["sample_mae"])

corr_conf_vs_error = float(sample_df["confidence_level"].corr(sample_df["sample_mae"]))
corr_proxy_vs_error = float(sample_df["proxy_confidence_from_truth"].corr(sample_df["sample_mae"]))

bins = np.linspace(0.0, 1.0, 11).tolist()
sample_df["confidence_bin"] = pd.cut(sample_df["confidence_level"], bins=bins, include_lowest=True)
calib = sample_df.groupby("confidence_bin", observed=False).agg(
    n_samples=("sample_mae", "count"),
    mean_confidence=("confidence_level", "mean"),
    mean_sample_mae=("sample_mae", "mean"),
    mean_pc_abs_error=("pc_abs_error", "mean"),
).reset_index()

print(f"      ✓ Corr(confidence_level, sample_mae): {corr_conf_vs_error:.4f}")
print(f"      ✓ Corr(proxy_confidence_from_truth, sample_mae): {corr_proxy_vs_error:.4f}")

print("\n[5/6] Saving offline outputs...")
sample_df.to_csv(os.path.join(OUTPUT_DIR, "offline_sample_predictions.csv"), index=False)
event_dashboard.to_csv(os.path.join(OUTPUT_DIR, "offline_event_dashboard.csv"), index=False)
calib.to_csv(os.path.join(OUTPUT_DIR, "proxy_confidence_calibration_bins.csv"), index=False)

summary = {
    "n_samples": int(len(sample_df)),
    "n_events": int(len(event_dashboard)),
    "mean_threat_score": float(sample_df["threat_score"].mean()),
    "mean_confidence_level": float(sample_df["confidence_level"].mean()),
    "mean_sample_mae": float(sample_df["sample_mae"].mean()),
    "mean_pc_abs_error": float(sample_df["pc_abs_error"].mean(skipna=True)),
    "corr_confidence_vs_error": corr_conf_vs_error,
    "corr_proxy_vs_error": corr_proxy_vs_error,
    "interpretation": {
        "desired": "corr_confidence_vs_error should be negative",
        "meaning": "higher confidence should correspond to lower observed error"
    },
    "note": "This is offline validation and uses truth labels post-hoc."
}

with open(os.path.join(OUTPUT_DIR, "offline_evaluation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Plot A: confidence vs observed error.
plt.figure(figsize=(10, 6))
plt.scatter(sample_df["confidence_level"], sample_df["sample_mae"], alpha=0.35, s=20, edgecolors="none")
plt.title("Offline Gate: Confidence vs Observed Error")
plt.xlabel("Confidence Level")
plt.ylabel("Observed Sample MAE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confidence_vs_observed_error.png"), dpi=200)
plt.close()

# Plot B: binned calibration.
valid_bins = calib[calib["n_samples"] > 0]
plt.figure(figsize=(10, 6))
plt.plot(valid_bins["mean_confidence"], valid_bins["mean_sample_mae"], marker="o", linewidth=2)
for _, row in valid_bins.iterrows():
    plt.text(row["mean_confidence"], row["mean_sample_mae"], f"n={int(row['n_samples'])}", fontsize=8)
plt.title("Offline Gate: Binned Confidence Calibration")
plt.xlabel("Mean Confidence (bin)")
plt.ylabel("Mean Observed Sample MAE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confidence_calibration_bins.png"), dpi=200)
plt.close()

print("      ✓ Saved: offline_sample_predictions.csv")
print("      ✓ Saved: offline_event_dashboard.csv")
print("      ✓ Saved: proxy_confidence_calibration_bins.csv")
print("      ✓ Saved: offline_evaluation_summary.json")
print("      ✓ Saved: confidence_vs_observed_error.png")
print("      ✓ Saved: confidence_calibration_bins.png")

print("\n[6/6] Engineer interpretation...")
print("  - Negative correlation between confidence and error is desired.")
print("  - If correlation is near zero/positive, confidence needs recalibration.")
print("  - If gate passes, proceed to Step 4 for production-safe inference.")

print("\n" + "=" * 80)
print("✓ STEP 3B COMPLETE")
print("=" * 80)
print("Next: python step4_inference_dashboard.py")
