"""
STEP 3: Train Self-Supervised GRU Model
Model learns to predict next CDM parameters from historical sequence

Author: Ahmed Alharbi (Team Leader)  
Date: November 2025
DebriSolver Competition - KAU Team

ARCHITECTURE:
- Bidirectional GRU to capture temporal patterns
- MC Dropout for uncertainty quantification
- Multi-output: predicts all CDM parameters simultaneously

IMPROVEMENTS:
- Standard Dropout with training=True workaround for MC inference
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set environment before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("STEP 3: TRAIN SELF-SUPERVISED GRU MODEL")
print("DebriSolver Competition - KAU Team")
print("=" * 80)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ GPU detected: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("\n⚠ No GPU detected, using CPU")

print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = 'processed_sequences'
OUTPUT_DIR = 'trained_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model hyperparameters
GRU_UNITS_1 = 128
GRU_UNITS_2 = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
L2_REG = 0.001

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 150  
LEARNING_RATE = 0.001
PATIENCE = 20 
MC_SAMPLES = 50

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n[1/6] Loading prepared sequences...")

try:
    X_train = np.load(os.path.join(INPUT_DIR, 'X_train.npy'))
    Y_train = np.load(os.path.join(INPUT_DIR, 'Y_train.npy'))
    X_val = np.load(os.path.join(INPUT_DIR, 'X_val.npy'))
    Y_val = np.load(os.path.join(INPUT_DIR, 'Y_val.npy'))
    X_test = np.load(os.path.join(INPUT_DIR, 'X_test.npy'))
    Y_test = np.load(os.path.join(INPUT_DIR, 'Y_test.npy'))
    
    with open(os.path.join(INPUT_DIR, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().strip().split('\n')
    
    print(f"      ✓ X_train: {X_train.shape}")
    print(f"      ✓ Y_train: {Y_train.shape}")
    print(f"      ✓ X_val:   {X_val.shape}")
    print(f"      ✓ Y_val:   {Y_val.shape}")
    print(f"      ✓ X_test:  {X_test.shape}")
    print(f"      ✓ Y_test:  {Y_test.shape}")
    print(f"      ✓ Features: {feature_names}")
    
except FileNotFoundError as e:
    print(f"      ✗ Error loading data: {e}")
    print(f"      Run step2_prepare_sequences.py first!")
    exit(1)

n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]

# ============================================================================
# BUILD MODEL (Using standard Dropout - compatible with all TF versions)
# ============================================================================

print(f"\n[2/6] Building self-supervised GRU model...")

def build_self_supervised_gru(n_timesteps, n_features, gru_units_1, gru_units_2, 
                               dense_units, dropout_rate, l2_reg):
    """
    Build a GRU model for self-supervised CDM prediction.
    
    Input: Sequence of CDM features
    Output: Predicted next CDM feature values
    
    Uses standard Dropout - we'll handle MC inference separately.
    """
    
    inputs = layers.Input(shape=(n_timesteps, n_features), name='cdm_sequence')
    
    # Masking for padded sequences (zeros)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # First Bidirectional GRU layer
    x = layers.Bidirectional(
        layers.GRU(
            gru_units_1, 
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg)
        ),
        name='bigru_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second Bidirectional GRU layer
    x = layers.Bidirectional(
        layers.GRU(
            gru_units_2,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg)
        ),
        name='bigru_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    x = layers.Dense(
        dense_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_1'
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        dense_units // 2,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_2'
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer: predict all features of the next CDM
    outputs = layers.Dense(n_features, name='predicted_cdm')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='SelfSupervised_CDM_GRU')
    
    return model


# Build the model
model = build_self_supervised_gru(
    n_timesteps=n_timesteps,
    n_features=n_features,
    gru_units_1=GRU_UNITS_1,
    gru_units_2=GRU_UNITS_2,
    dense_units=DENSE_UNITS,
    dropout_rate=DROPOUT_RATE,
    l2_reg=L2_REG
)

model.summary()

# ============================================================================
# COMPILE MODEL
# ============================================================================

print(f"\n[3/6] Compiling model...")

# Feature weights - give more importance to key features
feature_weights = np.ones(n_features)

for i, feat in enumerate(feature_names):
    if 'collision_probability' in feat.lower() or 'log10_pc' in feat.lower():
        feature_weights[i] = 2.0  # Pc is most important
    elif 'miss_distance' in feat.lower():
        feature_weights[i] = 1.5
    elif 'time_to_tca' in feat.lower():
        feature_weights[i] = 1.5
    elif 'cr_r' in feat.lower() or 'ct_t' in feat.lower() or 'cn_n' in feat.lower():
        feature_weights[i] = 1.2

print(f"      Feature weights:")
for i, (feat, weight) in enumerate(zip(feature_names, feature_weights)):
    print(f"        {feat}: {weight}")

feature_weights_tensor = tf.constant(feature_weights, dtype=tf.float32)

def weighted_mse(y_true, y_pred):
    """Weighted MSE loss giving more importance to key features."""
    squared_diff = tf.square(y_true - y_pred)
    weighted_squared_diff = squared_diff * feature_weights_tensor
    return tf.reduce_mean(weighted_squared_diff)

def pc_mae(y_true, y_pred):
    """MAE for collision probability."""
    pc_idx = 0
    for i, feat in enumerate(feature_names):
        if 'log10_pc' in feat.lower():
            pc_idx = i
            break
    return tf.reduce_mean(tf.abs(y_true[:, pc_idx] - y_pred[:, pc_idx]))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=weighted_mse,
    metrics=['mae', pc_mae]
)

print(f"      ✓ Model compiled")

# ============================================================================
# CALLBACKS
# ============================================================================

print(f"\n[4/6] Setting up training callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

print(f"      ✓ Early stopping (patience={PATIENCE})")
print(f"      ✓ Model checkpoint")
print(f"      ✓ Learning rate reduction")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print(f"\n[5/6] Training model...")
print(f"      Batch size: {BATCH_SIZE}")
print(f"      Max epochs: {EPOCHS}")
print(f"      Learning rate: {LEARNING_RATE}")

start_time = datetime.now()

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

training_time = (datetime.now() - start_time).total_seconds() / 60

print(f"\n      ✓ Training completed in {training_time:.1f} minutes")
print(f"      ✓ Best val_loss: {min(history.history['val_loss']):.4f}")
print(f"      ✓ Epochs trained: {len(history.history['loss'])}")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print(f"\n[6/6] Evaluating on test set...")

# Standard evaluation
test_loss, test_mae, test_pc_mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"      Test Loss: {test_loss:.4f}")
print(f"      Test MAE:  {test_mae:.4f}")
print(f"      Test Pc MAE: {test_pc_mae:.4f}")

# MC Dropout evaluation for uncertainty
# We'll use training=True to keep dropout active
print(f"\n      Running MC Dropout ({MC_SAMPLES} samples) for uncertainty...")

mc_predictions = []
for i in range(MC_SAMPLES):
    # Call model with training=True to keep dropout active
    pred = model(X_test, training=True)
    mc_predictions.append(pred.numpy())

mc_predictions = np.array(mc_predictions)

# Calculate mean and std of predictions
pred_mean = np.mean(mc_predictions, axis=0)
pred_std = np.std(mc_predictions, axis=0)

# Prediction uncertainty
avg_uncertainty = np.mean(pred_std, axis=1)

print(f"      Average prediction uncertainty: {np.mean(avg_uncertainty):.4f}")
print(f"      Uncertainty range: [{np.min(avg_uncertainty):.4f}, {np.max(avg_uncertainty):.4f}]")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================

print(f"\nSaving model and results...")

# Save model in H5 format (more compatible)
model.save(os.path.join(OUTPUT_DIR, 'final_model.h5'))
print(f"      ✓ Model saved to {OUTPUT_DIR}/final_model.h5")

# Also save weights separately (with proper extension for newer Keras)
try:
    model.save_weights(os.path.join(OUTPUT_DIR, 'model_weights.weights.h5'))
    print(f"      ✓ Weights saved to {OUTPUT_DIR}/model_weights.weights.h5")
except Exception as e:
    print(f"      ⚠ Could not save weights separately: {e}")
    print(f"      (Model is still saved in final_model.h5)")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)
print(f"      ✓ Training history saved")

# Save test predictions with uncertainty
test_meta = pd.read_csv(os.path.join(INPUT_DIR, 'test_metadata.csv'))

# Add predictions to metadata
for i, feat in enumerate(feature_names):
    test_meta[f'pred_{feat}'] = pred_mean[:, i]
    test_meta[f'std_{feat}'] = pred_std[:, i]
    test_meta[f'actual_{feat}'] = Y_test[:, i]

test_meta['avg_uncertainty'] = avg_uncertainty
test_meta.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)
print(f"      ✓ Test predictions saved")

# Save model configuration
config = {
    'n_timesteps': int(n_timesteps),
    'n_features': int(n_features),
    'feature_names': feature_names,
    'gru_units_1': GRU_UNITS_1,
    'gru_units_2': GRU_UNITS_2,
    'dense_units': DENSE_UNITS,
    'dropout_rate': DROPOUT_RATE,
    'l2_reg': L2_REG,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'mc_samples': MC_SAMPLES,
    'test_loss': float(test_loss),
    'test_mae': float(test_mae),
    'test_pc_mae': float(test_pc_mae),
    'training_time_minutes': training_time,
    'best_val_loss': float(min(history.history['val_loss'])),
    'epochs_trained': len(history.history['loss'])
}

with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"      ✓ Model config saved")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 80}")
print("TRAINING COMPLETE")
print(f"{'=' * 80}")

print(f"\nModel Performance:")
print(f"  Training samples:   {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")
print(f"  Test samples:       {len(X_test):,}")
print(f"  Epochs trained:     {len(history.history['loss'])}")
print(f"  Best val loss:      {min(history.history['val_loss']):.4f}")
print(f"  Test MAE:           {test_mae:.4f}")
print(f"  Test Pc MAE:        {test_pc_mae:.4f}")

print(f"\nUncertainty Quantification:")
print(f"  MC Dropout samples: {MC_SAMPLES}")
print(f"  Avg uncertainty:    {np.mean(avg_uncertainty):.4f}")

print(f"\nFiles saved to {OUTPUT_DIR}/:")
print(f"  - final_model.h5")
print(f"  - best_model.h5")
print(f"  - model_weights.h5")
print(f"  - training_history.csv")
print(f"  - test_predictions.csv")
print(f"  - model_config.json")

print(f"\n{'=' * 80}")
print("✓ STEP 3 COMPLETE")
print(f"{'=' * 80}")
print("\nNext: python step3b_evaluate_proxy_confidence.py")
print("Then run production inference: python step4_inference_dashboard.py")
