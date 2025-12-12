"""
Figure 8: Training, Validation, and Test Performance
Using actual training history - FIXED MARKER

Author: Ahmad Alharbi
Date: December 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Load training history
history_df = pd.read_csv('trained_model/training_history.csv')

epochs = np.arange(1, len(history_df) + 1)
train_mae = history_df['mae'].values
val_mae = history_df['val_mae'].values
train_pc_mae = history_df['pc_mae'].values
val_pc_mae = history_df['val_pc_mae'].values

best_epoch = 28
test_mae = 0.464
test_pc_mae = 0.403

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ============================================================================
# Subplot 1: Overall MAE
# ============================================================================
ax1.plot(epochs, train_mae, label='Training', color='#3498db', 
         linewidth=2.5, alpha=0.8)
ax1.plot(epochs, val_mae, label='Validation', color='#e74c3c', 
         linewidth=2.5, alpha=0.8)

# FIXED: Changed marker from '★' to '*'
ax1.scatter([best_epoch], [test_mae], color='#2ecc71', s=300, 
            marker='*', edgecolor='black', linewidth=2.5, 
            label=f'Test: {test_mae:.3f}', zorder=5)

ax1.axvline(x=best_epoch, color='gray', linestyle=':', 
            linewidth=2, alpha=0.6, label=f'Best Epoch ({best_epoch})')

ax1.annotate('Test at Best Epoch', 
             xy=(best_epoch, test_mae), 
             xytext=(best_epoch + 9, test_mae + 0.025),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='darkgreen'),
             fontsize=10, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                       alpha=0.7, edgecolor='darkgreen'))

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax1.set_title('Overall MAE Evolution\n(All 11 Features)', 
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, 
           edgecolor='black', fancybox=True)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_xlim(0, len(epochs) + 2)
ax1.set_ylim(0.45, 0.65)

# ============================================================================
# Subplot 2: Collision Probability MAE
# ============================================================================
ax2.plot(epochs, train_pc_mae, label='Training', color='#3498db', 
         linewidth=2.5, alpha=0.8)
ax2.plot(epochs, val_pc_mae, label='Validation', color='#e74c3c', 
         linewidth=2.5, alpha=0.8)

# FIXED: Changed marker from '★' to '*'
ax2.scatter([best_epoch], [test_pc_mae], color='#2ecc71', s=300, 
            marker='*', edgecolor='black', linewidth=2.5, 
            label=f'Test: {test_pc_mae:.3f}', zorder=5)

ax2.axvline(x=best_epoch, color='gray', linestyle=':', 
            linewidth=2, alpha=0.6, label=f'Best Epoch ({best_epoch})')

ax2.annotate('Test at Best Epoch', 
             xy=(best_epoch, test_pc_mae), 
             xytext=(best_epoch + 9, test_pc_mae + 0.02),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='darkgreen'),
             fontsize=10, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                       alpha=0.7, edgecolor='darkgreen'))

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax2.set_title('Collision Probability MAE Evolution\n(Most Critical Feature)', 
              fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95, 
           edgecolor='black', fancybox=True)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_xlim(0, len(epochs) + 2)
ax2.set_ylim(0.38, 0.55)

# ============================================================================
# Title and interpretation
# ============================================================================
fig.suptitle('Model Performance Evolution: Training vs Validation vs Test', 
             fontsize=16, fontweight='bold')

interpretation = (
    f'Model trained for {len(epochs)} epochs with early stopping identifying epoch {best_epoch} as optimal. '
    f'Green stars indicate test set performance at the best model checkpoint. '
    f'Test MAE ({test_mae:.3f} overall, {test_pc_mae:.3f} Pc) aligns closely with validation '
    f'({val_mae[best_epoch-1]:.3f} overall, {val_pc_mae[best_epoch-1]:.3f} Pc at epoch {best_epoch}), '
    f'confirming robust generalization without overfitting.'
)

fig.text(0.5, 0.01, interpretation, ha='center', fontsize=9.5,
         bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', 
                   alpha=0.9, edgecolor='black', linewidth=2))

plt.tight_layout(rect=[0, 0.065, 1, 0.96])

plt.savefig('figures/Figure_8_Train_Val_Test_Curves.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: figures/Figure_8_Train_Val_Test_Curves.png")

plt.show()