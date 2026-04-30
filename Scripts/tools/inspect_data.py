"""
Section 2 Data Inspection Script
Checks all processed_sequences files for shape, distribution, outliers,
scaler/imputer integrity, and feature sanity.
Run from project root: python Scripts/inspect_data.py
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Write ALL output to file AND terminal
log_path = 'inspect_data_output.txt'
log_file = open(log_path, 'w', encoding='utf-8')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)

SEQ_DIR = 'processed_sequences'
MODEL_DIR = 'trained_model'

RESET  = '\033[0m'
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{msg}{RESET}")
def sep():     print("─" * 70)

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ FILE INVENTORY ]")
sep()
files = os.listdir(SEQ_DIR)
for f in sorted(files):
    path = os.path.join(SEQ_DIR, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {f:<45} {size_kb:>10.1f} KB")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ LOAD ARRAYS ]")
sep()
arrays = {}
for split in ['train', 'val', 'test']:
    for xy in ['X', 'Y']:
        key = f"{xy}_{split}"
        path = os.path.join(SEQ_DIR, f"{key}.npy")
        if os.path.exists(path):
            arrays[key] = np.load(path)
            ok(f"{key}: {arrays[key].shape}  dtype={arrays[key].dtype}")
        else:
            err(f"{key}.npy NOT FOUND")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ FEATURE NAMES ]")
sep()
feat_path = os.path.join(SEQ_DIR, 'feature_names.txt')
if os.path.exists(feat_path):
    with open(feat_path) as f:
        feature_names = f.read().strip().split('\n')
    for i, name in enumerate(feature_names):
        print(f"  [{i:2d}] {name}")
    ok(f"{len(feature_names)} features total")
else:
    err("feature_names.txt NOT FOUND")
    feature_names = [f"feature_{i}" for i in range(11)]

n_features = len(feature_names)

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ NaN / Inf CHECK ]")
sep()
for key, arr in arrays.items():
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count > 0:
        err(f"{key}: {nan_count} NaN values!")
    elif inf_count > 0:
        err(f"{key}: {inf_count} Inf values!")
    else:
        ok(f"{key}: clean (no NaN, no Inf)")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ Y DISTRIBUTION — THE SUSPECTED CULPRIT ]")
sep()
print("  Checking Y arrays (targets the model predicts):\n")

for split in ['train', 'val', 'test']:
    key = f"Y_{split}"
    if key not in arrays:
        continue
    Y = arrays[key]
    print(f"  {BOLD}{key}{RESET}  shape={Y.shape}")
    print(f"  {'Feature':<35} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'|max|>10?':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, name in enumerate(feature_names):
        col = Y[:, i]
        mn, mx, mu, sd = col.min(), col.max(), col.mean(), col.std()
        flag = f"  {RED}← OUTLIER{RESET}" if abs(mx) > 10 or abs(mn) > 10 else ""
        print(f"  {name:<35} {mn:>10.3f} {mx:>10.3f} {mu:>10.3f} {sd:>10.3f}{flag}")
    
    # Extreme value count
    extreme = np.abs(Y) > 10
    extreme_count = extreme.sum()
    extreme_pct = extreme_count / Y.size * 100
    if extreme_count > 0:
        err(f"  {key}: {extreme_count} values with |v| > 10 ({extreme_pct:.2f}% of all values)")
        # Show which feature has the most extremes
        per_feature = extreme.sum(axis=0)
        worst_idx = np.argmax(per_feature)
        err(f"  Worst feature: [{worst_idx}] {feature_names[worst_idx]} — {per_feature[worst_idx]} extreme values")
    else:
        ok(f"  {key}: no extreme values (all within ±10 standard deviations)")
    print()

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ X DISTRIBUTION — PADDING SENTINEL CHECK ]")
sep()
for split in ['train', 'val', 'test']:
    key = f"X_{split}"
    if key not in arrays:
        continue
    X = arrays[key]
    
    # Count padding
    padding_mask = (X == -999.0)
    n_padding = padding_mask.sum()
    n_total = X.size
    pct_padding = n_padding / n_total * 100
    
    # Real values only
    real_vals = X[~padding_mask]
    real_outside_range = np.abs(real_vals) > 10
    outside_count = real_outside_range.sum()
    
    print(f"  {BOLD}{key}{RESET}")
    ok(f"  Padding (-999.0): {n_padding:,} / {n_total:,} values ({pct_padding:.1f}%)")
    if outside_count > 0:
        warn(f"  Real values with |v| > 10: {outside_count} ({outside_count/len(real_vals)*100:.2f}%)")
    else:
        ok(f"  All real values within ±10 standard deviations")
    print(f"  Real values range: [{real_vals.min():.3f}, {real_vals.max():.3f}]  mean={real_vals.mean():.3f}  std={real_vals.std():.3f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ SCALER INSPECTION ]")
sep()
scaler_path = os.path.join(SEQ_DIR, 'feature_scaler.pkl')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    ok(f"Scaler loaded: {type(scaler).__name__}")
    print(f"\n  {'Feature':<35} {'mean':>12} {'std':>12} {'std > 5?':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")
    for i, name in enumerate(feature_names):
        mu  = scaler.mean_[i]
        std = scaler.scale_[i]
        flag = f"  {RED}← LARGE STD{RESET}" if std > 5 else ""
        print(f"  {name:<35} {mu:>12.4f} {std:>12.4f}{flag}")
    
    max_std_idx = np.argmax(scaler.scale_)
    print(f"\n  Largest std: [{max_std_idx}] {feature_names[max_std_idx]} = {scaler.scale_[max_std_idx]:.4f}")
    warn("If std >> 1, it means some features weren't properly normalized before scaling")
    warn("A large std means raw data had extreme range — and Y may have unscaled outliers from those features")
else:
    err("feature_scaler.pkl NOT FOUND")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ IMPUTER INSPECTION ]")
sep()
imp_path = os.path.join(SEQ_DIR, 'feature_imputer.pkl')
if os.path.exists(imp_path):
    imputer = joblib.load(imp_path)
    ok(f"Imputer loaded: {type(imputer).__name__} strategy={imputer.strategy}")
    print(f"\n  {'Feature':<35} {'imputed value':>15}")
    print(f"  {'-'*35} {'-'*15}")
    for i, name in enumerate(feature_names):
        val = imputer.statistics_[i]
        print(f"  {name:<35} {val:>15.6f}")
else:
    warn("feature_imputer.pkl NOT FOUND — check if step2 saves it")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ TRAIN vs VAL DISTRIBUTION COMPARISON ]")
sep()
print("  Checking if val distribution matches train distribution:")
print("  (if val mean/std is very different from train, data split may be non-representative)\n")

for i, name in enumerate(feature_names):
    if 'Y_train' not in arrays or 'Y_val' not in arrays:
        break
    tr = arrays['Y_train'][:, i]
    va = arrays['Y_val'][:, i]
    
    mean_diff = abs(tr.mean() - va.mean())
    std_ratio = va.std() / (tr.std() + 1e-8)
    
    flag = ""
    if mean_diff > 1.0:
        flag = f"  {RED}← mean shift > 1 std{RESET}"
    elif std_ratio > 2.0 or std_ratio < 0.5:
        flag = f"  {YELLOW}← std ratio = {std_ratio:.2f}{RESET}"
    
    print(f"  [{i:2d}] {name:<33} train_mean={tr.mean():>7.3f} val_mean={va.mean():>7.3f} | train_std={tr.std():>6.3f} val_std={va.std():>6.3f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ TRAINING HISTORY ]")
sep()
hist_path = os.path.join(MODEL_DIR, 'training_history.csv')
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    ok(f"training_history.csv loaded — {len(hist)} epochs trained")
    print(f"\n  {hist[['loss', 'val_loss', 'mae', 'val_mae']].to_string(index=True)}")
    print(f"\n  Best val_loss: {hist['val_loss'].min():.4f} at epoch {hist['val_loss'].idxmin()}")
    print(f"  Last val_loss: {hist['val_loss'].iloc[-1]:.4f}")
    print(f"  val_loss trend: {'INCREASING' if hist['val_loss'].iloc[-1] > hist['val_loss'].iloc[0] else 'DECREASING'}")
else:
    warn("training_history.csv not found")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ model_config.json ]")
sep()
cfg_path = os.path.join(MODEL_DIR, 'model_config.json')
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    ok(f"model_config.json loaded")
    print(f"  Features:       {cfg.get('feature_names')}")
    print(f"  n_features:     {cfg.get('n_features')}")
    print(f"  n_timesteps:    {cfg.get('n_timesteps')}")
    print(f"  best_val_loss:  {cfg.get('best_val_loss'):.4f}")
    print(f"  epochs_trained: {cfg.get('epochs_trained')}")
    print(f"  test_loss:      {cfg.get('test_loss'):.4f}")
    print(f"  test_mae:       {cfg.get('test_mae'):.4f}")
else:
    warn("model_config.json not found — run step3 first")

# ─────────────────────────────────────────────────────────────────────────────
hdr("[ SPLIT INFO ]")
sep()
split_path = os.path.join(SEQ_DIR, 'split_info.csv')
if os.path.exists(split_path):
    split_info = pd.read_csv(split_path)
    ok("split_info.csv loaded")
    print(split_info.to_string(index=False))
else:
    warn("split_info.csv not found")

hdr("[ INSPECTION COMPLETE ]")
sep()
print("""
  Summary of what to look for in the output above:
  1. Any Y feature with |max| > 10 → extreme outliers causing val_loss explosion
  2. Scaler std >> 1 for any feature → suggests a normalization bug
  3. Mean difference > 1.0 between train/val for any feature → bad split
  4. val_loss trend: if always INCREASING from epoch 1 → systematic data issue
  5. X real values range: should be roughly [-5, 5] after scaling
""")
