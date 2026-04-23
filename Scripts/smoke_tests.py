"""Lightweight smoke checks for generated DebriSolver artifacts."""

import argparse
import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def check_parser_artifact():
    parsed_path = "parsed_cdm_data.csv"
    require_file(parsed_path)
    parsed = pd.read_csv(parsed_path)

    required_cols = [
        "event_id",
        "CREATION_DATE",
        "TCA",
        "COLLISION_PROBABILITY",
        "MISS_DISTANCE",
    ]
    missing = [col for col in required_cols if col not in parsed.columns]
    if missing:
        raise ValueError(f"parsed_cdm_data.csv missing required columns: {missing}")

    if parsed.empty:
        raise ValueError("parsed_cdm_data.csv is empty")

    print("[PASS] Parser artifact check")


def check_sequence_artifacts():
    sequence_dir = "processed_sequences"
    required = [
        "X_train.npy",
        "Y_train.npy",
        "X_val.npy",
        "Y_val.npy",
        "X_test.npy",
        "Y_test.npy",
        "test_metadata.csv",
        "feature_scaler.pkl",
        "feature_imputer.pkl",
    ]
    for name in required:
        require_file(os.path.join(sequence_dir, name))

    x_train = np.load(os.path.join(sequence_dir, "X_train.npy"))
    y_train = np.load(os.path.join(sequence_dir, "Y_train.npy"))
    x_test = np.load(os.path.join(sequence_dir, "X_test.npy"))

    if x_train.ndim != 3 or x_test.ndim != 3:
        raise ValueError("Sequence arrays must be 3D")
    if y_train.ndim != 2:
        raise ValueError("Target arrays must be 2D")
    if np.isnan(x_train).any() or np.isinf(x_train).any():
        raise ValueError("X_train contains NaN or Inf")

    print("[PASS] Sequence artifact check")


def check_model_artifacts():
    model_dir = "trained_model"
    required = ["model_config.json", "training_history.csv", "test_predictions.csv"]
    for name in required:
        require_file(os.path.join(model_dir, name))

    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    required_keys = ["n_timesteps", "n_features", "feature_names", "seed"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"model_config.json missing keys: {missing_keys}")

    print("[PASS] Model artifact check")


def check_inference_artifacts():
    dashboard_path = os.path.join("dashboard_output", "event_dashboard.csv")
    require_file(dashboard_path)

    dashboard = pd.read_csv(dashboard_path)
    required_cols = ["event_id", "threat_score", "confidence_level", "quadrant"]
    missing = [col for col in required_cols if col not in dashboard.columns]
    if missing:
        raise ValueError(f"event_dashboard.csv missing columns: {missing}")

    print("[PASS] Inference artifact check")


def main():
    parser = argparse.ArgumentParser(description="Run smoke checks on DebriSolver artifacts")
    parser.add_argument(
        "--stage",
        choices=["data", "model", "inference", "full"],
        default="full",
        help="Scope of smoke checks",
    )
    args = parser.parse_args()

    check_parser_artifact()
    check_sequence_artifacts()

    if args.stage in {"model", "inference", "full"}:
        check_model_artifacts()

    if args.stage in {"inference", "full"}:
        check_inference_artifacts()

    print("[DONE] Smoke checks completed successfully")


if __name__ == "__main__":
    main()
