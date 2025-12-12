"""
STEP 5: Visualization Dashboard - REVISED
Creates individual, publication-ready figures for research report

Author: Ahmed Talal Alharbi (Team Leader)  
Date: December 2025
DebriSolver Competition - KAU Team

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("STEP 5: VISUALIZATION DASHBOARD")
print("DebriSolver Competition - KAU Team")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

DASHBOARD_DIR = 'dashboard_output'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n[1/10] Loading dashboard data...")

dashboard = pd.read_csv(os.path.join(DASHBOARD_DIR, 'event_dashboard.csv'))
print(f"      ✓ Loaded {len(dashboard)} events")

# ============================================================================
# FIGURE 1: QUADRANT DASHBOARD
# ============================================================================

print(f"\n[2/10] Creating quadrant dashboard...")

fig, ax = plt.subplots(figsize=(10, 8))

# Color mapping
colors = {
    'ACT NOW': '#d62728',
    'WATCH CLOSELY': '#ff7f0e',
    'SAFELY IGNORE': '#2ca02c',
    'NOT PRIORITY': '#7f7f7f'
}

# Plot each quadrant
for quadrant, color in colors.items():
    mask = dashboard['quadrant'] == quadrant
    subset = dashboard[mask]
    
    # Size based on number of CDMs
    sizes = np.clip(subset['total_cdms'] * 10, 30, 200)
    
    ax.scatter(subset['confidence_level'], subset['threat_score'],
               c=color, s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5,
               label=f'{quadrant}: {len(subset)}')

# Decision boundaries
ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Labels and formatting
ax.set_xlabel('Model Confidence Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Threat Score', fontsize=12, fontweight='bold')
ax.set_title('Risk Assessment Quadrant Classification', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

# Legend moved to upper right
ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=9)

# Add summary box in bottom left
total = len(dashboard)
summary_text = f"Total Events: {total:,}\n"
for quadrant, color in colors.items():
    count = len(dashboard[dashboard['quadrant'] == quadrant])
    pct = count / total * 100
    summary_text += f"{quadrant}: {count} ({pct:.1f}%)\n"

ax.text(0.02, 0.35, summary_text.strip(), transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))

# Add quadrant labels (moved away from legend)
ax.text(0.25, 85, 'WATCH CLOSELY\n(High Threat, Low Confidence)', 
        ha='center', va='center', fontsize=10, color='#ff7f0e', 
        fontweight='bold', alpha=0.6, style='italic')
ax.text(0.75, 75, 'ACT NOW\n(High Threat, High Confidence)',  # Moved down slightly
        ha='center', va='center', fontsize=10, color='#d62728', 
        fontweight='bold', alpha=0.6, style='italic')
ax.text(0.25, 15, 'NOT PRIORITY\n(Low Threat, Low Confidence)', 
        ha='center', va='center', fontsize=10, color='#7f7f7f', 
        fontweight='bold', alpha=0.6, style='italic')
ax.text(0.75, 15, 'SAFELY IGNORE\n(Low Threat, High Confidence)', 
        ha='center', va='center', fontsize=10, color='#2ca02c', 
        fontweight='bold', alpha=0.6, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_1_Quadrant_Dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Saved Figure_1_Quadrant_Dashboard.png")

# ============================================================================
# FIGURE 2: THREAT SCORE DISTRIBUTION
# ============================================================================

print(f"\n[3/10] Creating threat score distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

threat_scores = dashboard['threat_score'].values
mean_threat = threat_scores.mean()

ax.hist(threat_scores, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Threshold (50)', alpha=0.8)
ax.axvline(x=mean_threat, color='green', linestyle='-', linewidth=2, label=f'Mean ({mean_threat:.1f})', alpha=0.8)

ax.set_xlabel('Threat Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Predicted Threat Scores', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.95, edgecolor='black')
ax.grid(alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_2_Threat_Distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Saved Figure_2_Threat_Distribution.png")

# ============================================================================
# FIGURE 3: CONFIDENCE DISTRIBUTION
# ============================================================================

print(f"\n[4/10] Creating confidence distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

confidence_levels = dashboard['confidence_level'].values
mean_conf = confidence_levels.mean()

ax.hist(confidence_levels, bins=50, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)', alpha=0.8)
ax.axvline(x=mean_conf, color='green', linestyle='-', linewidth=2, label=f'Mean ({mean_conf:.2f})', alpha=0.8)

ax.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Model Confidence Levels', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.95, edgecolor='black')
ax.grid(alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_3_Confidence_Distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Saved Figure_3_Confidence_Distribution.png")

# ============================================================================
# FIGURE 4: THREAT vs ACTUAL Pc
# ============================================================================

print(f"\n[5/10] Creating threat vs collision probability correlation...")

fig, ax = plt.subplots(figsize=(10, 7))

# Get log10 Pc values (need to load from parsed data for actual Pc)
try:
    parsed_data = pd.read_csv('parsed_cdm_data.csv')
    
    # Merge with dashboard to get final Pc for each event
    event_pc = parsed_data.groupby('event_id').agg({
        'log10_pc': 'last',
        'COLLISION_PROBABILITY': 'last'
    }).reset_index()
    
    merged = dashboard.merge(event_pc, on='event_id', how='left')
    
    # Filter valid data
    valid = merged[merged['log10_pc'].notna() & merged['COLLISION_PROBABILITY'].notna()]
    
    scatter = ax.scatter(valid['log10_pc'], valid['threat_score'],
                        c=valid['confidence_level'], cmap='RdYlGn',  # Changed from RdYlGn_r
                        s=50, alpha=0.6, edgecolors='white', linewidth=0.5,
                        vmin=0, vmax=1)
    
    # Trend line
    z = np.polyfit(valid['log10_pc'], valid['threat_score'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(valid['log10_pc'].min(), valid['log10_pc'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend (slope={z[0]:.2f})', alpha=0.8)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Confidence Level')
    
    ax.set_xlabel('Log₁₀(Collision Probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Threat Score', fontsize=12, fontweight='bold')
    ax.set_title('Threat Score vs. Actual Collision Probability', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4_Threat_vs_Pc_Correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved Figure_4_Threat_vs_Pc_Correlation.png")
    
except Exception as e:
    print(f"      ⚠ Could not create Threat vs Pc plot: {e}")

# ============================================================================
# FIGURE 5: CONFIDENCE vs DATA QUANTITY
# ============================================================================

print(f"\n[6/10] Creating confidence vs data quantity...")

fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(dashboard['total_cdms'], dashboard['confidence_level'],
                    c=dashboard['threat_score'], cmap='RdYlGn_r',
                    s=50, alpha=0.6, edgecolors='white', linewidth=0.5,
                    vmin=0, vmax=100)

cbar = plt.colorbar(scatter, ax=ax, label='Threat Score')

# Add visual markers for key thresholds
max_cdm = dashboard['total_cdms'].max()

ax.set_xlabel('Number of CDMs (Observation Data)', fontsize=12, fontweight='bold')
ax.set_ylabel('Model Confidence Level', fontsize=12, fontweight='bold')
ax.set_title('Confidence Level vs. Amount of Observation Data', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(-1, min(max_cdm * 1.05, 50))
ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_5_Confidence_vs_Data_Quantity.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Saved Figure_5_Confidence_vs_Data_Quantity.png")

# ============================================================================
# FIGURE 6: TRAINING HISTORY (LOSS)
# ============================================================================

print(f"\n[7/10] Creating training loss curves...")

try:
    history = pd.read_csv(os.path.join('trained_model', 'training_history.csv'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history) + 1)
    
    ax.plot(epochs, history['loss'], label='Training Loss', linewidth=2, color='#1f77b4', marker='o', markersize=3, markevery=5)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e', marker='s', markersize=3, markevery=5)
    
    # Mark best epoch
    best_epoch = history['val_loss'].idxmin() + 1
    best_val_loss = history['val_loss'].min()
    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Best Epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (Weighted MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_6_Training_Loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved Figure_6_Training_Loss.png")
    
except FileNotFoundError:
    print(f"      ⚠ Training history not found - skipping loss curves")

# ============================================================================
# FIGURE 7: TRAINING HISTORY (MAE)
# ============================================================================

print(f"\n[8/10] Creating training MAE curves...")

try:
    history = pd.read_csv(os.path.join('trained_model', 'training_history.csv'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history) + 1)
    
    ax.plot(epochs, history['mae'], label='Training MAE', linewidth=2, color='#2ca02c', marker='o', markersize=3, markevery=5)
    ax.plot(epochs, history['val_mae'], label='Validation MAE', linewidth=2, color='#d62728', marker='s', markersize=3, markevery=5)
    
    # Mark best epoch
    best_epoch = history['val_loss'].idxmin() + 1
    best_val_mae = history['val_mae'].iloc[best_epoch - 1]
    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Best Epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val_mae], color='red', s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation MAE Curves', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_7_Training_MAE.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved Figure_7_Training_MAE.png")
    
except FileNotFoundError:
    print(f"      ⚠ Training history not found - skipping MAE curves")

# ============================================================================
# FIGURE 8: TRAINING HISTORY (Pc MAE)
# ============================================================================

print(f"\n[9/10] Creating Pc MAE curves...")

try:
    history = pd.read_csv(os.path.join('trained_model', 'training_history.csv'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history) + 1)
    
    ax.plot(epochs, history['pc_mae'], label='Training Pc MAE', linewidth=2, color='#9467bd', marker='o', markersize=3, markevery=5)
    ax.plot(epochs, history['val_pc_mae'], label='Validation Pc MAE', linewidth=2, color='#e377c2', marker='s', markersize=3, markevery=5)
    
    # Mark best epoch
    best_epoch = history['val_loss'].idxmin() + 1
    best_val_pc_mae = history['val_pc_mae'].iloc[best_epoch - 1]
    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Best Epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val_pc_mae], color='red', s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Collision Probability MAE', fontsize=12, fontweight='bold')
    ax.set_title('Collision Probability Prediction Error', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_8_Pc_MAE.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved Figure_8_Pc_MAE.png")
    
except FileNotFoundError:
    print(f"      ⚠ Training history not found - skipping Pc MAE curves")

# ============================================================================
# TABLE 1: TOP 20 ACT NOW EVENTS (Excel Export)
# ============================================================================

print(f"\n[10/10] Creating priority events table...")

act_now = dashboard[dashboard['quadrant'] == 'ACT NOW'].sort_values(
    ['threat_score', 'confidence_level'], ascending=[False, False]
).head(20)

if len(act_now) > 0:
    # Prepare table data
    table_data = act_now[[
        'event_id', 'threat_score', 'confidence_level', 'final_pc', 
        'tca_datetime', 'total_cdms', 'object1_name', 'object2_name'
    ]].copy()
    
    # Round numerical values
    table_data['threat_score'] = table_data['threat_score'].round(1)
    table_data['confidence_level'] = table_data['confidence_level'].round(2)
    
    # Rename columns
    table_data.columns = [
        'Event ID', 'Threat', 'Confidence', 'Final Pc', 
        'TCA', 'CDMs', 'Object 1', 'Object 2'
    ]
    
    # Export to Excel (with error handling)
    excel_path = os.path.join(OUTPUT_DIR, 'Table_1_High_Priority_Events.xlsx')
    try:
        table_data.to_excel(excel_path, index=False, sheet_name='ACT NOW Events')
        print(f"      ✓ Saved Table_1_High_Priority_Events.xlsx")
    except ImportError:
        print(f"      ⚠ openpyxl not installed - skipping Excel export")
        print(f"        Install with: pip install openpyxl")
        # Save as CSV instead
        csv_path = os.path.join(OUTPUT_DIR, 'Table_1_High_Priority_Events.csv')
        table_data.to_csv(csv_path, index=False)
        print(f"      ✓ Saved Table_1_High_Priority_Events.csv instead")
    
    # Create PNG visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.12, 0.08, 0.08, 0.10, 0.15, 0.06, 0.20, 0.20])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#d62728')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Table_1_High_Priority_Events.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved Table_1_High_Priority_Events.png")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 80}")
print("VISUALIZATION COMPLETE")
print(f"{'=' * 80}")

print(f"\nGenerated figures in {OUTPUT_DIR}/:")
print(f"  Figure 1: Quadrant Dashboard")
print(f"  Figure 2: Threat Score Distribution")
print(f"  Figure 3: Confidence Distribution")
print(f"  Figure 4: Threat vs Collision Probability")
print(f"  Figure 5: Confidence vs Data Quantity")
print(f"  Figure 6: Training Loss Curves")
print(f"  Figure 7: Training MAE Curves")
print(f"  Figure 8: Collision Probability MAE")
print(f"  Table 1:  High Priority Events (Excel + PNG)")

print(f"\nAll figures are:")
print(f"  - High resolution (300 DPI)")
print(f"  - Individual files (no overlap)")
print(f"  - Publication-ready formatting")
print(f"  - Numbered for easy citation")

print(f"\n{'=' * 80}")
print("✓ STEP 5 COMPLETE")
print(f"{'=' * 80}")
