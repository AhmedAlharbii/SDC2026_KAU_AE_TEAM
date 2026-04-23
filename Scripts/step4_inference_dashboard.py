"""
STEP 4: Inference & Dashboard Generation
Converts self-supervised model predictions into operator-actionable threat assessments

Author: Ahmed Alharbi (Team Leader)  
Date: November 2025
DebriSolver Competition - KAU Team

KEY CONCEPT:
- Model predicts next CDM parameters (self-supervised)
- Threat score: derived from predicted trajectory (is Pc increasing/decreasing?)
- Confidence: derived from model's prediction uncertainty (MC Dropout std)
- TCA: taken directly from latest CDM data (trusted)
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("STEP 4: INFERENCE & DASHBOARD GENERATION")
print("DebriSolver Competition - KAU Team")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = 'trained_model'
SEQUENCE_DIR = 'processed_sequences'
OUTPUT_DIR = 'dashboard_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MC Dropout samples for uncertainty estimation
MC_SAMPLES = 50

# ============================================================================
# LOAD CONFIG AND REBUILD MODEL
# ============================================================================

print(f"\n[1/5] Loading model and data...")

# Load config first
with open(os.path.join(MODEL_DIR, 'model_config.json'), 'r') as f:
    config = json.load(f)

feature_names = config['feature_names']
n_timesteps = config['n_timesteps']
n_features = config['n_features']

print(f"      ✓ Config loaded")
print(f"      ✓ Features: {feature_names}")

# Rebuild the model architecture (must match training)
from tensorflow.keras import layers, Model, regularizers

def build_model(n_timesteps, n_features):
    """Rebuild model architecture matching training."""
    inputs = layers.Input(shape=(n_timesteps, n_features), name='cdm_sequence')
    x = layers.Masking(mask_value=0.0)(inputs)
    
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.001),
                   recurrent_regularizer=regularizers.l2(0.001)),
        name='bigru_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.001),
                   recurrent_regularizer=regularizers.l2(0.001)),
        name='bigru_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     name='dense_1')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     name='dense_2')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(n_features, name='predicted_cdm')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='SelfSupervised_CDM_GRU')

# Build model
model = build_model(n_timesteps, n_features)

# Load weights from H5 file
# We need to call the model once to build it before loading weights
dummy_input = np.zeros((1, n_timesteps, n_features))
_ = model(dummy_input)

# Try loading the full model's weights
weights_loaded = False
try:
    model.load_weights(os.path.join(MODEL_DIR, 'final_model.h5'))
    print(f"      ✓ Model weights loaded from final_model.h5")
    weights_loaded = True
except:
    # If that fails, try the weights file
    try:
        model.load_weights(os.path.join(MODEL_DIR, 'model_weights.weights.h5'))
        print(f"      ✓ Model weights loaded from model_weights.weights.h5")
        weights_loaded = True
    except:
        try:
            model.load_weights(os.path.join(MODEL_DIR, 'best_model.h5'))
            print(f"      ✓ Model weights loaded from best_model.h5")
            weights_loaded = True
        except Exception as e:
            print(f"      ✗ Could not load weights: {e}")
            raise RuntimeError("Model weights could not be loaded from any checkpoint") from e

if not weights_loaded:
    raise RuntimeError("No model checkpoint was loaded; aborting inference")

print(f"      ✓ Model rebuilt and weights loaded")

# Load scaler
scaler = joblib.load(os.path.join(SEQUENCE_DIR, 'feature_scaler.pkl'))
print(f"      ✓ Scaler loaded")

# Load test data
X_test = np.load(os.path.join(SEQUENCE_DIR, 'X_test.npy'))
test_meta = pd.read_csv(os.path.join(SEQUENCE_DIR, 'test_metadata.csv'))
print(f"      ✓ Test data: {X_test.shape[0]} samples")

# Load original parsed data for additional context
parsed_data = pd.read_csv('parsed_cdm_data.csv')
parsed_data['TCA'] = pd.to_datetime(parsed_data['TCA'], errors='coerce')
parsed_data['CREATION_DATE'] = pd.to_datetime(parsed_data['CREATION_DATE'], errors='coerce')
print(f"      ✓ Original CDM data loaded")


def validate_inference_inputs(x_test, test_meta, parsed_data, feature_names, config):
    """Validate inference artifacts before running MC Dropout."""

    required_meta_cols = ['event_id', 'tca', 'total_cdms', 'target_pc', 'target_time_to_tca']
    required_parsed_cols = ['event_id', 'TCA', 'CREATION_DATE', 'COLLISION_PROBABILITY', 'MISS_DISTANCE']

    if x_test.ndim != 3:
        raise ValueError(f"X_test must be 3D (samples, timesteps, features); got shape {x_test.shape}")

    if x_test.shape[1] != config['n_timesteps']:
        raise ValueError(
            f"X_test timestep mismatch: expected {config['n_timesteps']}, got {x_test.shape[1]}"
        )

    if x_test.shape[2] != config['n_features']:
        raise ValueError(
            f"X_test feature mismatch: expected {config['n_features']}, got {x_test.shape[2]}"
        )

    if len(feature_names) != config['n_features']:
        raise ValueError(
            f"feature_names length mismatch: expected {config['n_features']}, got {len(feature_names)}"
        )

    if len(test_meta) != len(x_test):
        raise ValueError(
            f"test_metadata row count mismatch: expected {len(x_test)}, got {len(test_meta)}"
        )

    missing_meta_cols = [col for col in required_meta_cols if col not in test_meta.columns]
    if missing_meta_cols:
        raise ValueError(f"test_metadata missing required columns: {missing_meta_cols}")

    missing_parsed_cols = [col for col in required_parsed_cols if col not in parsed_data.columns]
    if missing_parsed_cols:
        raise ValueError(f"parsed_cdm_data missing required columns: {missing_parsed_cols}")

    if np.isnan(x_test).any() or np.isinf(x_test).any():
        raise ValueError("X_test contains NaN or infinite values")

    if parsed_data['event_id'].isna().any():
        raise ValueError("parsed_cdm_data contains missing event_id values")

    print("      ✓ Inference input validation passed")


validate_inference_inputs(X_test, test_meta, parsed_data, feature_names, config)

# ============================================================================
# MC DROPOUT INFERENCE
# ============================================================================

print(f"\n[2/5] Running MC Dropout inference ({MC_SAMPLES} samples)...")

mc_predictions = []
for i in range(MC_SAMPLES):
    if (i + 1) % 10 == 0:
        print(f"      Progress: {i+1}/{MC_SAMPLES}")
    # Use training=True to keep dropout active for MC sampling
    pred = model(X_test, training=True)
    mc_predictions.append(pred.numpy())

mc_predictions = np.array(mc_predictions)

# Calculate mean and std of predictions
pred_mean = np.mean(mc_predictions, axis=0)
pred_std = np.std(mc_predictions, axis=0)

print(f"      ✓ Predictions shape: {pred_mean.shape}")

# ============================================================================
# COMPUTE THREAT SCORES AND CONFIDENCE
# ============================================================================

print(f"\n[3/5] Computing threat scores and confidence levels...")

def compute_threat_and_confidence(
    pred_mean, 
    pred_std, 
    X_input,
    feature_names,
    scaler
):
    """
    Compute threat score and confidence from model predictions.
    
    Threat Score (0-100):
    - Based on PREDICTED trajectory of Pc
    - If model predicts Pc will increase → higher threat
    - If model predicts Pc will decrease → lower threat
    - Also considers absolute Pc level
    
    Confidence (0-1):
    - Based on model's prediction uncertainty (MC Dropout std)
    - Lower std → higher confidence
    - Also considers how much input data (non-zero timesteps)
    - Does NOT use ground-truth future labels (deployment-safe)
    """
    
    n_samples = pred_mean.shape[0]
    threat_scores = np.zeros(n_samples)
    confidence_levels = np.zeros(n_samples)
    
    # Find feature indices
    pc_idx = None
    log10_pc_idx = None
    miss_dist_idx = None
    time_to_tca_idx = None
    cr_r_idx = None
    
    for i, feat in enumerate(feature_names):
        if feat == 'COLLISION_PROBABILITY':
            pc_idx = i
        elif feat == 'log10_pc':
            log10_pc_idx = i
        elif feat == 'MISS_DISTANCE':
            miss_dist_idx = i
        elif feat == 'time_to_tca_hours':
            time_to_tca_idx = i
        elif feat == 'combined_cr_r':
            cr_r_idx = i
    
    # Use log10_pc if available, otherwise COLLISION_PROBABILITY
    main_pc_idx = log10_pc_idx if log10_pc_idx is not None else pc_idx
    
    for i in range(n_samples):
        # =====================================================================
        # THREAT SCORE
        # =====================================================================
        
        # 1. Get current Pc from input sequence (last non-zero timestep)
        input_seq = X_input[i]
        non_zero_mask = np.any(input_seq != 0, axis=1)
        if np.any(non_zero_mask):
            last_valid_idx = np.where(non_zero_mask)[0][-1]
            current_pc_scaled = input_seq[last_valid_idx, main_pc_idx]
        else:
            current_pc_scaled = 0
        
        # 2. Get predicted Pc
        predicted_pc_scaled = pred_mean[i, main_pc_idx]
        
        # 4. Calculate Pc trend (predicted vs current)
        pc_change = predicted_pc_scaled - current_pc_scaled
        
        # 5. Base threat from absolute predicted Pc level
        if log10_pc_idx is not None:
            # log10_pc after StandardScaler: roughly -2 to 2
            # Map to 0-70 range
            base_threat = (predicted_pc_scaled + 2) * 17.5  # Rough scaling
            base_threat = np.clip(base_threat, 0, 70)
        else:
            base_threat = predicted_pc_scaled * 70
        
        # 6. Trend modifier (-15 to +15)
        # Positive pc_change = Pc increasing = more threat
        trend_modifier = np.clip(pc_change * 30, -15, 15)
        
        # 7. Time urgency (if available)
        if time_to_tca_idx is not None:
            predicted_time = pred_mean[i, time_to_tca_idx]
            # Lower time = more urgent (after scaling, interpret as normalized)
            urgency = max(0, (1 - (predicted_time + 1) / 2) * 15)
        else:
            urgency = 0
        
        # Combine threat components
        threat = base_threat + trend_modifier + urgency
        threat_scores[i] = np.clip(threat, 0, 100)
        
        # =====================================================================
        # CONFIDENCE
        # =====================================================================
        
        # 1. Model uncertainty (from MC Dropout std)
        # Lower std = more confident
        avg_std = np.mean(pred_std[i])
        uncertainty_confidence = 1 / (1 + avg_std * 3)  # Normalize to 0-1
        
        # 2. Data quantity confidence
        # More input timesteps = more confident
        n_valid_timesteps = np.sum(non_zero_mask)
        data_confidence = min(1.0, n_valid_timesteps / 10)  # Max at 10 CDMs
        
        # 3. Covariance-based confidence (if available)
        if cr_r_idx is not None:
            predicted_cov = pred_mean[i, cr_r_idx]
            # Higher covariance = less confident
            cov_confidence = 1 / (1 + abs(predicted_cov))
        else:
            cov_confidence = 0.5
        
        # Combine confidence components
        confidence = (
            uncertainty_confidence * 0.5 +  # Model uncertainty is key
            data_confidence * 0.3 +
            cov_confidence * 0.2
        )
        confidence_levels[i] = np.clip(confidence, 0.1, 1.0)
    
    return threat_scores, confidence_levels


threat_scores, confidence_levels = compute_threat_and_confidence(
    pred_mean, pred_std, X_test, feature_names, scaler
)

print(f"      ✓ Threat scores computed")
print(f"        Range: [{threat_scores.min():.1f}, {threat_scores.max():.1f}]")
print(f"        Mean:  {threat_scores.mean():.1f}")
print(f"        Std:   {threat_scores.std():.1f}")

print(f"      ✓ Confidence levels computed")
print(f"        Range: [{confidence_levels.min():.2f}, {confidence_levels.max():.2f}]")
print(f"        Mean:  {confidence_levels.mean():.2f}")

# ============================================================================
# CREATE DASHBOARD DATA
# ============================================================================

print(f"\n[4/5] Creating dashboard data...")

# Aggregate to event level
dashboard_data = test_meta.copy()
dashboard_data['threat_score'] = threat_scores
dashboard_data['confidence_level'] = confidence_levels

# Add prediction details
for i, feat in enumerate(feature_names):
    dashboard_data[f'pred_{feat}'] = pred_mean[:, i]
    dashboard_data[f'uncertainty_{feat}'] = pred_std[:, i]

# Aggregate by event (take the latest prediction for each event)
event_dashboard = dashboard_data.groupby('event_id').agg({
    'threat_score': 'last',
    'confidence_level': 'last',
    'tca': 'first',
    'total_cdms': 'max',
    'target_pc': 'last',
    'target_time_to_tca': 'last',
}).reset_index()

# Get additional event info from original data
event_info = parsed_data.groupby('event_id').agg({
    'TCA': 'last',
    'COLLISION_PROBABILITY': 'last',
    'MISS_DISTANCE': 'last',
    'object1_OBJECT_NAME': 'first',
    'object2_OBJECT_NAME': 'first',
    'object1_OBJECT_TYPE': 'first',
    'object2_OBJECT_TYPE': 'first',
    'CREATION_DATE': ['min', 'max'],
}).reset_index()

event_info.columns = ['event_id', 'tca_datetime', 'final_pc', 'final_miss_distance',
                      'object1_name', 'object2_name', 'object1_type', 'object2_type',
                      'first_cdm_date', 'last_cdm_date']

# Merge
event_dashboard = event_dashboard.merge(event_info, on='event_id', how='left')

# Calculate observation period
event_dashboard['observation_days'] = (
    pd.to_datetime(event_dashboard['last_cdm_date']) - 
    pd.to_datetime(event_dashboard['first_cdm_date'])
).dt.total_seconds() / 86400

# Classify into quadrants
event_dashboard['quadrant'] = 'NOT PRIORITY'
high_threat = event_dashboard['threat_score'] > 50
high_conf = event_dashboard['confidence_level'] > 0.5

event_dashboard.loc[high_threat & high_conf, 'quadrant'] = 'ACT NOW'
event_dashboard.loc[high_threat & ~high_conf, 'quadrant'] = 'WATCH CLOSELY'
event_dashboard.loc[~high_threat & high_conf, 'quadrant'] = 'SAFELY IGNORE'
event_dashboard.loc[~high_threat & ~high_conf, 'quadrant'] = 'NOT PRIORITY'

print(f"      ✓ Dashboard data created for {len(event_dashboard)} events")

# Quadrant distribution
print(f"\n      Quadrant Distribution:")
quadrant_counts = event_dashboard['quadrant'].value_counts()
for quadrant, count in quadrant_counts.items():
    pct = count / len(event_dashboard) * 100
    print(f"        {quadrant}: {count:,} ({pct:.1f}%)")

# ============================================================================
# SAVE DASHBOARD DATA
# ============================================================================

print(f"\n[5/5] Saving dashboard data...")

# Save full dashboard data
event_dashboard.to_csv(os.path.join(OUTPUT_DIR, 'event_dashboard.csv'), index=False)
print(f"      ✓ Saved event_dashboard.csv")

# Save sample-level data
dashboard_data.to_csv(os.path.join(OUTPUT_DIR, 'sample_predictions.csv'), index=False)
print(f"      ✓ Saved sample_predictions.csv")

# Create summary report
summary = {
    'total_events': len(event_dashboard),
    'quadrant_distribution': quadrant_counts.to_dict(),
    'threat_score_stats': {
        'mean': float(event_dashboard['threat_score'].mean()),
        'std': float(event_dashboard['threat_score'].std()),
        'min': float(event_dashboard['threat_score'].min()),
        'max': float(event_dashboard['threat_score'].max()),
    },
    'confidence_stats': {
        'mean': float(event_dashboard['confidence_level'].mean()),
        'std': float(event_dashboard['confidence_level'].std()),
        'min': float(event_dashboard['confidence_level'].min()),
        'max': float(event_dashboard['confidence_level'].max()),
    },
    'high_priority_events': event_dashboard[
        event_dashboard['quadrant'] == 'ACT NOW'
    ][['event_id', 'threat_score', 'confidence_level', 'final_pc', 'tca_datetime', 
       'object1_name', 'object2_name']].head(10).to_dict('records')
}

with open(os.path.join(OUTPUT_DIR, 'dashboard_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"      ✓ Saved dashboard_summary.json")

# ============================================================================
# PRINT HIGH PRIORITY EVENTS
# ============================================================================

print(f"\n{'=' * 80}")
print("HIGH PRIORITY EVENTS (ACT NOW)")
print(f"{'=' * 80}")

act_now = event_dashboard[event_dashboard['quadrant'] == 'ACT NOW'].sort_values(
    'threat_score', ascending=False
).head(10)

if len(act_now) > 0:
    for _, row in act_now.iterrows():
        print(f"\n  Event: {row['event_id']}")
        print(f"  Objects: {row['object1_name']} vs {row['object2_name']}")
        print(f"  TCA: {row['tca_datetime']}")
        print(f"  Threat Score: {row['threat_score']:.1f}")
        print(f"  Confidence: {row['confidence_level']:.2f}")
        print(f"  Final Pc: {row['final_pc']:.2e}" if pd.notna(row['final_pc']) else "  Final Pc: N/A")
        print(f"  CDMs: {row['total_cdms']} over {row['observation_days']:.1f} days")
        print(f"  " + "-" * 50)
else:
    print("\n  No events in ACT NOW quadrant.")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 80}")
print("DASHBOARD GENERATION COMPLETE")
print(f"{'=' * 80}")

print(f"\nOutput files in {OUTPUT_DIR}/:")
print(f"  - event_dashboard.csv     (event-level threat/confidence)")
print(f"  - sample_predictions.csv  (sample-level predictions)")
print(f"  - dashboard_summary.json  (summary statistics)")

print(f"\nQuadrant Summary:")
for quadrant in ['ACT NOW', 'WATCH CLOSELY', 'SAFELY IGNORE', 'NOT PRIORITY']:
    count = quadrant_counts.get(quadrant, 0)
    pct = count / len(event_dashboard) * 100
    print(f"  {quadrant:15s}: {count:5,} events ({pct:5.1f}%)")

print(f"\n{'=' * 80}")
print("✓ STEP 4 COMPLETE")
print(f"{'=' * 80}")
print("\nNext: python step5_visualize.py")
