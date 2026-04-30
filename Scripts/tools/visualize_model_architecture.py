"""
Visualize BiGRU Model Architecture
Creates publication-ready diagram for methodology section

Author: Ahmed Alharbi (Team Leader)  
Date: December 2025
DebriSolver Competition - KAU Team
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Always resolve paths relative to Scripts/ regardless of which subdirectory this lives in
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

# Create figure with more vertical space
fig, ax = plt.subplots(figsize=(12, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis('off')

# Color scheme
color_input = '#e8f4f8'      # Light blue
color_gru = '#90caf9'        # Blue
color_dense = '#ffb74d'      # Orange
color_output = '#81c784'     # Green
color_dropout = '#ef9a9a'    # Red
color_batch = '#ce93d8'      # Purple

# Helper function for boxes
def draw_box(ax, x, y, width, height, label, color, fontsize=10, fontweight='normal'):
    """Draw a fancy box with label"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", 
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', 
            fontsize=fontsize, fontweight=fontweight, wrap=True)

# Helper function for arrows
def draw_arrow(ax, x1, y1, x2, y2, label='', color='black', linestyle='-'):
    """Draw arrow between layers"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color=color, linestyle=linestyle)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label, fontsize=8, style='italic', color=color)

# ============================================================================
# Title
# ============================================================================
ax.text(5, 17.5, 'Bidirectional GRU Architecture for CDM Trajectory Learning', 
        ha='center', fontsize=16, fontweight='bold')
ax.text(5, 17, 'Self-Supervised Sequence Prediction Model', 
        ha='center', fontsize=12, style='italic', color='gray')

# ============================================================================
# LAYER 1: Input Layer
# ============================================================================
y_pos = 15.5
draw_box(ax, 5, y_pos, 4, 0.8, 
         'Input: CDM Sequence\n(batch, 20 timesteps, 11 features)', 
         color_input, fontsize=10, fontweight='bold')

# Feature list
ax.text(0.5, y_pos, 
        '11 Features:\n• Collision Probability\n• log₁₀(Pc)\n• Miss Distance\n• Time to TCA\n• Covariance (R, T, N)\n• Relative Velocity\n• Relative Position (R, T, N)', 
        fontsize=8, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 1.2)

# ============================================================================
# LAYER 2: Masking Layer
# ============================================================================
y_pos = 14
draw_box(ax, 5, y_pos, 3.5, 0.6, 
         'Masking Layer\n(Ignores padded zeros)', 
         color_batch, fontsize=9)

ax.text(9, y_pos, 
        'Purpose:\nHandle variable-length\nCDM sequences', 
        fontsize=7, va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

draw_arrow(ax, 5, y_pos - 0.4, 5, y_pos - 0.8)

# ============================================================================
# LAYER 3: Bidirectional GRU Layer 1
# ============================================================================
y_pos = 12.5

# Forward GRU
draw_box(ax, 3.5, y_pos, 2.5, 1, 
         'Forward GRU\n128 units\nreturn_sequences=True', 
         color_gru, fontsize=9, fontweight='bold')

# Backward GRU
draw_box(ax, 6.5, y_pos, 2.5, 1, 
         'Backward GRU\n128 units\nreturn_sequences=True', 
         color_gru, fontsize=9, fontweight='bold')

# Bidirectional explanation
ax.text(0.5, y_pos, 
        'Bidirectional:\nReads sequence\nforward AND backward\nto capture full context', 
        fontsize=7, va='center',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

# Concatenate arrows - improved clarity
ax.annotate('', xy=(5, y_pos - 0.65), xytext=(3.5, y_pos - 0.55),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
ax.annotate('', xy=(5, y_pos - 0.65), xytext=(6.5, y_pos - 0.55),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
ax.text(5, y_pos - 0.95, 'Concatenate → 256 features', 
        ha='center', fontsize=8, style='italic', fontweight='bold')

draw_arrow(ax, 5, y_pos - 1.3, 5, y_pos - 1.8)

# ============================================================================
# LAYER 4: Batch Normalization + Dropout
# ============================================================================
y_pos = 10.5
draw_box(ax, 3.5, y_pos, 2, 0.5, 
         'Batch Normalization', 
         color_batch, fontsize=8)
draw_box(ax, 6.5, y_pos, 2, 0.5, 
         'Dropout (0.3)', 
         color_dropout, fontsize=8)

ax.text(9, y_pos, 
        'Regularization:\nPrevent overfitting\n& enable MC Dropout', 
        fontsize=7, va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

draw_arrow(ax, 5, y_pos - 0.4, 5, y_pos - 0.8)

# ============================================================================
# LAYER 5: Bidirectional GRU Layer 2
# ============================================================================
y_pos = 9

# Forward GRU
draw_box(ax, 3.5, y_pos, 2.5, 1, 
         'Forward GRU\n64 units\nreturn_sequences=False', 
         color_gru, fontsize=9, fontweight='bold')

# Backward GRU
draw_box(ax, 6.5, y_pos, 2.5, 1, 
         'Backward GRU\n64 units\nreturn_sequences=False', 
         color_gru, fontsize=9, fontweight='bold')

ax.text(0.5, y_pos, 
        'return_sequences=False:\nOutputs single vector\n(final hidden state)', 
        fontsize=7, va='center',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

# Concatenate arrows - improved clarity
ax.annotate('', xy=(5, y_pos - 0.65), xytext=(3.5, y_pos - 0.55),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
ax.annotate('', xy=(5, y_pos - 0.65), xytext=(6.5, y_pos - 0.55),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
ax.text(5, y_pos - 0.95, 'Concatenate → 128 features', 
        ha='center', fontsize=8, style='italic', fontweight='bold')

draw_arrow(ax, 5, y_pos - 1.3, 5, y_pos - 1.8)

# ============================================================================
# LAYER 6: Batch Normalization + Dropout
# ============================================================================
y_pos = 7
draw_box(ax, 3.5, y_pos, 2, 0.5, 
         'Batch Normalization', 
         color_batch, fontsize=8)
draw_box(ax, 6.5, y_pos, 2, 0.5, 
         'Dropout (0.3)', 
         color_dropout, fontsize=8)

draw_arrow(ax, 5, y_pos - 0.4, 5, y_pos - 0.8)

# ============================================================================
# LAYER 7: Dense Layer 1
# ============================================================================
y_pos = 5.8
draw_box(ax, 5, y_pos, 3, 0.7, 
         'Dense Layer (64 neurons)\nActivation: ReLU', 
         color_dense, fontsize=9, fontweight='bold')

draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 0.8)

# ============================================================================
# LAYER 8: Dropout
# ============================================================================
y_pos = 4.8
draw_box(ax, 5, y_pos, 2, 0.5, 
         'Dropout (0.3)', 
         color_dropout, fontsize=8)

draw_arrow(ax, 5, y_pos - 0.4, 5, y_pos - 0.7)

# ============================================================================
# LAYER 9: Dense Layer 2
# ============================================================================
y_pos = 3.9
draw_box(ax, 5, y_pos, 3, 0.7, 
         'Dense Layer (32 neurons)\nActivation: ReLU', 
         color_dense, fontsize=9, fontweight='bold')

draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 0.8)

# ============================================================================
# LAYER 10: Dropout
# ============================================================================
y_pos = 2.9
draw_box(ax, 5, y_pos, 2, 0.5, 
         'Dropout (0.3)', 
         color_dropout, fontsize=8)

ax.text(9, y_pos, 
        'MC Dropout:\nDuring inference,\nkeep dropout active\nfor uncertainty', 
        fontsize=7, va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

draw_arrow(ax, 5, y_pos - 0.4, 5, y_pos - 0.7, color='red', linestyle='--')
ax.text(5, 2.1, 'training=True during inference', fontsize=8, color='red', 
        style='italic', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

# ============================================================================
# LAYER 11: Output Layer
# ============================================================================
y_pos = 1
draw_box(ax, 5, y_pos, 4, 0.8, 
         'Output: Predicted Next CDM\n(11 features)', 
         color_output, fontsize=10, fontweight='bold')

# ============================================================================
# Model Summary Box
# ============================================================================
summary_text = """Model Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Parameters:      244,171
Trainable Parameters:  243,403
Non-trainable:         768

Architecture Type: Sequence-to-Vector
Task: Self-Supervised Learning
Loss: Weighted MSE (2× for Pc)
Optimizer: Adam (lr=0.001)
"""

ax.text(8.5, 6.5, summary_text, fontsize=8, va='top', ha='left',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9, 
                  edgecolor='black', linewidth=2))

# ============================================================================
# Legend
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input/Output'),
    mpatches.Patch(facecolor=color_gru, edgecolor='black', label='GRU Layers'),
    mpatches.Patch(facecolor=color_dense, edgecolor='black', label='Dense Layers'),
    mpatches.Patch(facecolor=color_dropout, edgecolor='black', label='Dropout'),
    mpatches.Patch(facecolor=color_batch, edgecolor='black', label='Batch Normalization')
]

ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

# ============================================================================
# Self-Supervised Learning Explanation (Back to original top-right position)
# ============================================================================
explanation_text = """Self-Supervised Learning Task:

Given: CDMs 1, 2, ..., N-1 (input sequence)
Predict: CDM N (next in sequence)
Loss: Compare prediction with actual CDM N

The model learns conjunction dynamics by
predicting how CDM parameters evolve over time."""

ax.text(11, 15.5, explanation_text, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, 
                  edgecolor='darkgreen', linewidth=2))

# Save figure
plt.tight_layout()
plt.savefig('figures/Model_Architecture_BiGRU.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Saved: figures/Model_Architecture_BiGRU.png")

plt.show()
