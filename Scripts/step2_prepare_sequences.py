"""
STEP 2: Prepare Sequences for Self-Supervised Learning
Groups CDMs by event, creates sequences, temporal train/test split

Author: Ahmed Alharbi (Team Leader)  
Date: November 2025
DebriSolver Competition - KAU Team

KEY CONCEPT:
- Training: Events where TCA has passed (we know full history)
- Testing: Events where TCA was in the future (model predicts trajectory)
- Self-supervised: Model predicts next CDM values from previous CDMs
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import joblib
import yaml
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("STEP 2: PREPARE SEQUENCES FOR SELF-SUPERVISED LEARNING")
print("DebriSolver Competition - KAU Team")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = 'parsed_cdm_data.csv'
OUTPUT_DIR = 'processed_sequences'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load central config
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Sequence parameters
MAX_SEQUENCE_LENGTH = cfg['data']['max_sequence_length']
MIN_CDMS_FOR_TRAINING = cfg['data']['min_cdms']

# Features the model will learn to predict (and use as input)
# NOTE: COLLISION_PROBABILITY (raw linear scale) is intentionally EXCLUDED.
# It is redundant with log10_pc and has catastrophic outliers (117+ std devs)
# when Pc is even moderately high (~0.07). log10_pc captures all the same
# information on a well-behaved log scale.
FEATURES = [
    'log10_pc',
    'MISS_DISTANCE',
    'time_to_tca_hours',
    'combined_cr_r',
    'combined_ct_t', 
    'combined_cn_n',
    'RELATIVE_SPEED',
    'RELATIVE_POSITION_R',
    'RELATIVE_POSITION_T',
    'RELATIVE_POSITION_N',
]

# Covariance columns to log-transform before scaling.
# Raw covariance in m² spans 0 to 20 billion — StandardScaler can't handle this.
# log1p compresses to [0, ~22] range, making the scaler effective.
LOG_TRANSFORM_COLS = ['combined_cr_r', 'combined_ct_t', 'combined_cn_n']

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n[1/7] Loading parsed CDM data...")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"      ✓ Loaded {len(df):,} CDMs")
except FileNotFoundError:
    print(f"      ✗ File not found: {INPUT_FILE}")
    print(f"      Run step1_parse_kvn.py first!")
    exit(1)

# Convert timestamps
df['CREATION_DATE'] = pd.to_datetime(df['CREATION_DATE'], errors='coerce')
df['TCA'] = pd.to_datetime(df['TCA'], errors='coerce')

print(f"      ✓ Unique events: {df['event_id'].nunique():,}")

# ============================================================================
# LOG-TRANSFORM COVARIANCE FEATURES
# ============================================================================

print(f"\n[1b/7] Applying log1p transform to covariance features...")
print(f"       Reason: raw covariance in m² spans 0 to ~20 billion.")
print(f"       StandardScaler fails on this range — log1p compresses to [0,~22].")

for col in LOG_TRANSFORM_COLS:
    if col in df.columns:
        before_max = df[col].max()
        df[col] = np.log1p(df[col].clip(0))   # clip(0) enforces physical non-negativity
        after_max = df[col].max()
        print(f"      ✓ {col}: max {before_max:.0f} → {after_max:.3f} (log1p)")
    else:
        print(f"      ⚠ {col} not in dataframe — skipping")

# ============================================================================
# DETERMINE AVAILABLE FEATURES
# ============================================================================

print(f"\n[2/7] Checking available features...")

available_features = []
for feat in FEATURES:
    if feat in df.columns:
        valid_pct = df[feat].notna().sum() / len(df) * 100
        if valid_pct > 50:  # Only use features with >50% valid data
            available_features.append(feat)
            print(f"      ✓ {feat}: {valid_pct:.1f}% valid")
        else:
            print(f"      ⚠ {feat}: {valid_pct:.1f}% valid (skipping)")
    else:
        print(f"      ✗ {feat}: not found")

FEATURES = available_features
print(f"\n      Using {len(FEATURES)} features")

if len(FEATURES) < 3:
    print("      ✗ ERROR: Need at least 3 features")
    exit(1)

# ============================================================================
# RANDOM SPLIT BY EVENT (Standard ML Practice for Alert Ranking)
# ============================================================================

print(f"\n[3/7] Creating random train/validation/test split...")
print(f"      Approach: Random event split (appropriate for alert credibility assessment)")

# Get all unique events
unique_events = df['event_id'].unique()
print(f"\n      Total events: {len(unique_events):,}")

# Check TCA distribution
if 'TCA' in df.columns:
    event_tca = df.groupby('event_id')['TCA'].first()
    event_tca = pd.to_datetime(event_tca, errors='coerce')
    valid_tca = event_tca.dropna()
    
    if len(valid_tca) > 0:
        print(f"      TCA range: {valid_tca.min()} to {valid_tca.max()}")
        print(f"      Events with valid TCA: {len(valid_tca):,} ({len(valid_tca)/len(unique_events)*100:.1f}%)")

# Random split: 80% train, 10% validation, 10% test
# Seed comes from config.yaml so the split is reproducible and tied to the same
# seed used for model training — change it in one place, affects everything.
from sklearn.model_selection import train_test_split

SPLIT_SEED = cfg['training']['seed']

train_events, temp_events = train_test_split(
    unique_events,
    test_size=0.2,
    random_state=SPLIT_SEED
)

val_events, test_events = train_test_split(
    temp_events,
    test_size=0.5,
    random_state=SPLIT_SEED
)

print(f"\n      Random split (80/10/10):")
print(f"        Train:      {len(train_events):,} events ({len(train_events)/len(unique_events)*100:.1f}%)")
print(f"        Validation: {len(val_events):,} events ({len(val_events)/len(unique_events)*100:.1f}%)")
print(f"        Test:       {len(test_events):,} events ({len(test_events)/len(unique_events)*100:.1f}%)")

# Verify no overlap
assert len(set(train_events) & set(val_events)) == 0, "Train/Val overlap detected!"
assert len(set(train_events) & set(test_events)) == 0, "Train/Test overlap detected!"
assert len(set(val_events) & set(test_events)) == 0, "Val/Test overlap detected!"
print(f"      ✓ No overlap between splits verified")

# Show TCA distribution across splits (if available)
if 'TCA' in df.columns and len(valid_tca) > 0:
    print(f"\n      TCA distribution across splits:")
    for split_name, split_events in [('Train', train_events), ('Val', val_events), ('Test', test_events)]:
        split_tcas = event_tca[event_tca.index.isin(split_events)].dropna()
        if len(split_tcas) > 0:
            print(f"        {split_name:10s}: {split_tcas.min()} to {split_tcas.max()}")
    print(f"      (All splits contain events across full time range)")

print(f"\n      Rationale: Random split ensures model generalizes to unseen")
print(f"                 conjunction events, appropriate for alert credibility")
print(f"                 assessment (not temporal forecasting).")

# ============================================================================
# FILTER EVENTS BY CDM COUNT
# ============================================================================

print(f"\n[4/7] Filtering events (minimum {MIN_CDMS_FOR_TRAINING} CDMs)...")

event_cdm_counts = df.groupby('event_id').size()
valid_events = event_cdm_counts[event_cdm_counts >= MIN_CDMS_FOR_TRAINING].index.tolist()

train_events = [e for e in train_events if e in valid_events]
val_events = [e for e in val_events if e in valid_events]
test_events = [e for e in test_events if e in valid_events]

print(f"      After filtering:")
print(f"        Train:      {len(train_events):,} events")
print(f"        Validation: {len(val_events):,} events")  
print(f"        Test:       {len(test_events):,} events")

# ============================================================================
# FEATURE SCALING
# ============================================================================

print(f"\n[5/7] Fitting feature scaler on training data...")

# Get training data
train_df = df[df['event_id'].isin(train_events)]

# Fit imputer and scaler on training features only
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
train_features = train_df[FEATURES]
train_features_imputed = imputer.fit_transform(train_features)
scaler.fit(train_features_imputed)

# Save imputer and scaler for reproducibility
joblib.dump(imputer, os.path.join(OUTPUT_DIR, 'feature_imputer.pkl'))
print(f"      ✓ Imputer fitted and saved")

# Save scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'feature_scaler.pkl'))
print(f"      ✓ Scaler fitted and saved")

# Save feature names
with open(os.path.join(OUTPUT_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(FEATURES))
print(f"      ✓ Feature names saved")

# ============================================================================
# CREATE SEQUENCES
# ============================================================================

print(f"\n[6/7] Creating sequences for self-supervised learning...")

def create_sequences_for_events(event_list, df, scaler, features, max_len):
    """
    Create sequences for self-supervised learning.
    
    For each event with N CDMs, we create training samples:
    - Input: CDMs 1 to N-1 (history)
    - Target: CDM N (what we predict)
    
    We also create intermediate samples:
    - Input: CDMs 1 to k-1 → Target: CDM k (for k = 2, 3, ..., N)
    
    Returns:
    - X: Input sequences (batch, timesteps, features)
    - Y: Target CDM values (batch, features)
    - event_ids: Which event each sample belongs to
    - sequence_positions: Which CDM position is being predicted
    """
    
    X_list = []
    Y_list = []
    event_ids = []
    sequence_positions = []
    metadata = []  # Store additional info for each sample
    
    for event_id in event_list:
        event_data = df[df['event_id'] == event_id].sort_values('CREATION_DATE')
        
        if len(event_data) < 2:
            continue
        
        # Extract, impute, and scale features using training-derived statistics
        event_features = event_data[features]
        event_features_imputed = imputer.transform(event_features)
        event_features_scaled = scaler.transform(event_features_imputed)
        
        # Get metadata
        event_tca = event_data['TCA'].iloc[-1]
        event_pc = event_data['COLLISION_PROBABILITY'].values
        event_times = event_data['time_to_tca_hours'].values
        
        n_cdms = len(event_features_scaled)
        
        # Create samples: predict CDM k from CDMs 1..k-1
        for k in range(2, min(n_cdms + 1, max_len + 1)):
            # Input: all CDMs up to k-1
            input_seq = event_features_scaled[:k-1]
            
            # Target: CDM k (index k-1)
            target = event_features_scaled[k-1]
            
            # Pad input sequence to max_len
            if len(input_seq) < max_len:
                padding = np.full((max_len - len(input_seq), len(features)), -999.0)
                input_seq_padded = np.vstack([padding, input_seq])
            else:
                input_seq_padded = input_seq[-max_len:]
            
            X_list.append(input_seq_padded)
            Y_list.append(target)
            event_ids.append(event_id)
            sequence_positions.append(k)
            
            # Store metadata for this sample
            metadata.append({
                'event_id': event_id,
                'prediction_position': k,
                'total_cdms': n_cdms,
                'input_cdms': k - 1,
                'tca': event_tca,
                'target_pc': event_pc[k-1] if k-1 < len(event_pc) else np.nan,
                'target_time_to_tca': event_times[k-1] if k-1 < len(event_times) else np.nan
            })
    
    X = np.array(X_list) if X_list else np.array([]).reshape(0, max_len, len(features))
    Y = np.array(Y_list) if Y_list else np.array([]).reshape(0, len(features))
    
    return X, Y, event_ids, sequence_positions, metadata

# Create sequences for each split
print("      Creating training sequences...")
X_train, Y_train, train_ids, train_pos, train_meta = create_sequences_for_events(
    train_events, df, scaler, FEATURES, MAX_SEQUENCE_LENGTH
)
print(f"        ✓ Train: {X_train.shape[0]:,} samples")

print("      Creating validation sequences...")
X_val, Y_val, val_ids, val_pos, val_meta = create_sequences_for_events(
    val_events, df, scaler, FEATURES, MAX_SEQUENCE_LENGTH
)
print(f"        ✓ Val: {X_val.shape[0]:,} samples")

print("      Creating test sequences...")
X_test, Y_test, test_ids, test_pos, test_meta = create_sequences_for_events(
    test_events, df, scaler, FEATURES, MAX_SEQUENCE_LENGTH
)
print(f"        ✓ Test: {X_test.shape[0]:,} samples")

# ============================================================================
# SAVE SEQUENCES
# ============================================================================

print(f"\n[7/7] Saving sequences and metadata...")

# Save numpy arrays
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'Y_train.npy'), Y_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(OUTPUT_DIR, 'Y_val.npy'), Y_val)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_DIR, 'Y_test.npy'), Y_test)

print(f"      ✓ Saved sequence arrays")

# Save metadata
train_meta_df = pd.DataFrame(train_meta)
val_meta_df = pd.DataFrame(val_meta)
test_meta_df = pd.DataFrame(test_meta)

train_meta_df.to_csv(os.path.join(OUTPUT_DIR, 'train_metadata.csv'), index=False)
val_meta_df.to_csv(os.path.join(OUTPUT_DIR, 'val_metadata.csv'), index=False)
test_meta_df.to_csv(os.path.join(OUTPUT_DIR, 'test_metadata.csv'), index=False)

print(f"      ✓ Saved metadata")

# Save split information
split_info = {
    'split_method': 'random_by_event',
    'split_ratios': '80% train / 10% validation / 10% test',
    'random_seed': 42,
    'n_train_events': len(train_events),
    'n_val_events': len(val_events),
    'n_test_events': len(test_events),
    'n_train_samples': len(X_train),
    'n_val_samples': len(X_val),
    'n_test_samples': len(X_test),
    'n_features': len(FEATURES),
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'features': FEATURES,
    'rationale': 'Random event split appropriate for alert credibility assessment task'
}

pd.DataFrame([split_info]).to_csv(os.path.join(OUTPUT_DIR, 'split_info.csv'), index=False)
print(f"      ✓ Saved split information")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 80}")
print("SEQUENCE PREPARATION COMPLETE")
print(f"{'=' * 80}")

print(f"\nData Shapes:")
print(f"  X_train: {X_train.shape} (samples, timesteps, features)")
print(f"  Y_train: {Y_train.shape} (samples, features)")
print(f"  X_val:   {X_val.shape}")
print(f"  Y_val:   {Y_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Y_test:  {Y_test.shape}")

print(f"\nFeatures ({len(FEATURES)}):")
for i, feat in enumerate(FEATURES, 1):
    print(f"  {i}. {feat}")

print(f"\nTemporal Split:")
print(f"  Method: Random event split (80/10/10)")
print(f"  Ensures model generalizes to unseen conjunction events")
print(f"  Appropriate for alert credibility assessment task")

print(f"\nSelf-Supervised Task:")
print(f"  Input:  Sequence of CDMs (history)")
print(f"  Output: Predict next CDM's parameter values")

print(f"\nFiles saved to: {OUTPUT_DIR}/")

print(f"\n{'=' * 80}")
print("✓ STEP 2 COMPLETE")
print(f"{'=' * 80}")
print("\nNext: python step3_train_model.py")
