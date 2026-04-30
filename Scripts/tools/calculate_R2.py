import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# Load test predictions (saved from step3)
test_pred = pd.read_csv('trained_model/test_predictions.csv')

# Load feature names
with open('processed_sequences/feature_names.txt', 'r') as f:
    features = f.read().strip().split('\n')

# Calculate overall R² (all features combined)
y_true_all = []
y_pred_all = []

for feat in features:
    y_true_all.extend(test_pred[f'actual_{feat}'].values)
    y_pred_all.extend(test_pred[f'pred_{feat}'].values)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

overall_r2 = r2_score(y_true_all, y_pred_all)
overall_mae = mean_absolute_error(y_true_all, y_pred_all)

print("="*70)
print("BiGRU MODEL PERFORMANCE - R² SCORE")
print("="*70)
print(f"\nOverall R² Score:        {overall_r2:.4f}")
print(f"Overall MAE:             {overall_mae:.4f}")
print(f"Test Samples:            {len(test_pred):,}")

# Per-feature R²
print(f"\n{'Feature':<30} {'R² Score':>12} {'MAE':>12}")
print("-"*70)

for feat in features:
    y_true = test_pred[f'actual_{feat}'].values
    y_pred = test_pred[f'pred_{feat}'].values
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"{feat:<30} {r2:>12.4f} {mae:>12.4f}")

print("="*70)

# Interpretation
print("\nINTERPRETATION:")
if overall_r2 > 0.7:
    print("✓ EXCELLENT: Model explains >70% of variance")
elif overall_r2 > 0.5:
    print("✓ GOOD: Model explains >50% of variance")
elif overall_r2 > 0.3:
    print("✓ MODERATE: Model explains >30% of variance")
elif overall_r2 > 0:
    print("✓ POSITIVE: Model performs better than baseline")
else:
    print("⚠ NEGATIVE: Check if this is due to feature normalization")

print("\nNote: For normalized features, MAE is often more interpretable than R²")