"""
STEP 5B: Generate Detailed Event Reports
Creates individual detailed reports for high-priority and safe events

Author: Ahmed Alharbi (Team Leader)  
Date: November 2025
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
print("STEP 5B: DETAILED EVENT REPORTS")
print("DebriSolver Competition - KAU Team")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

DASHBOARD_DIR = 'dashboard_output'
OUTPUT_DIR = 'detailed_reports'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ACT_NOW = 20
N_SAFELY_IGNORE = 5

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n[1/3] Loading data...")

dashboard = pd.read_csv(os.path.join(DASHBOARD_DIR, 'event_dashboard.csv'))
parsed_data = pd.read_csv('parsed_cdm_data.csv')

# Convert timestamps
parsed_data['TCA'] = pd.to_datetime(parsed_data['TCA'], errors='coerce')
parsed_data['CREATION_DATE'] = pd.to_datetime(parsed_data['CREATION_DATE'], errors='coerce')

print(f"      ✓ Dashboard: {len(dashboard)} events")
print(f"      ✓ Parsed data: {len(parsed_data):,} CDMs")

# ============================================================================
# SELECT EVENTS
# ============================================================================

print(f"\n[2/3] Selecting events for detailed reports...")

# Top 20 ACT NOW
act_now = dashboard[dashboard['quadrant'] == 'ACT NOW'].sort_values(
    ['threat_score', 'confidence_level'], ascending=[False, False]
).head(N_ACT_NOW)

# Random 5 from SAFELY IGNORE
safely_ignore = dashboard[dashboard['quadrant'] == 'SAFELY IGNORE'].sample(
    n=min(N_SAFELY_IGNORE, len(dashboard[dashboard['quadrant'] == 'SAFELY IGNORE'])),
    random_state=42
)

selected_events = pd.concat([act_now, safely_ignore])
print(f"      ✓ Selected {len(selected_events)} events:")
print(f"        - {len(act_now)} ACT NOW events")
print(f"        - {len(safely_ignore)} SAFELY IGNORE events")

# ============================================================================
# GENERATE REPORTS
# ============================================================================

print(f"\n[3/3] Generating detailed reports...")

def create_event_report(event_id, event_info, event_cdms):
    """Create detailed report for a single event with no overlapping."""
    
    # Sort CDMs by creation date
    event_cdms = event_cdms.sort_values('CREATION_DATE')
    
    fig = plt.figure(figsize=(16, 12))  # Increased height
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)  # Increased hspace
    
    # Title
    quadrant_colors = {
        'ACT NOW': '#d62728',
        'WATCH CLOSELY': '#ff7f0e',
        'SAFELY IGNORE': '#2ca02c',
        'NOT PRIORITY': '#7f7f7f'
    }
    title_color = quadrant_colors.get(event_info['quadrant'], 'black')
    
    fig.suptitle(f"Event: {event_id} | Threat: {event_info['threat_score']:.1f} | Confidence: {event_info['confidence_level']:.2f}",
                 fontsize=16, fontweight='bold', color=title_color, y=0.98)
    
    # ==========================================================================
    # Panel 1: Event Summary (Top Left)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    summary_text = f"""EVENT SUMMARY
{'='*40}

Event ID:      {event_id}
Object 1:      {str(event_info.get('object1_name', 'N/A'))[:25]}
Type:          {event_info.get('object1_type', 'N/A')}
Object 2:      {str(event_info.get('object2_name', 'N/A'))[:25]}
Type:          {event_info.get('object2_type', 'N/A')}

TCA:           {str(event_info.get('tca_datetime', 'N/A'))[:19]}
Final Pc:      {event_info.get('final_pc', 0):.2e}
Miss Dist:     {event_info.get('final_miss_distance', 0):.1f} m

Observation:   {event_info.get('total_cdms', 0)} CDMs
Period:        {event_info.get('observation_days', 0):.1f} days

ASSESSMENT
{'='*40}

Threat:        {event_info['threat_score']:.1f} / 100
Confidence:    {event_info['confidence_level']:.2f}
Quadrant:      {event_info['quadrant']}"""
    
    ax1.text(0.05, 1.5, summary_text, transform=ax1.transAxes, fontsize=8.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ==========================================================================
    # Panel 2: Quadrant Position (Top Middle)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax2.scatter([event_info['confidence_level']], [event_info['threat_score']], 
                c=title_color, s=500, marker='*', edgecolors='black', linewidth=2, zorder=5)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Confidence', fontsize=10)
    ax2.set_ylabel('Threat Score', fontsize=10)
    ax2.set_title('Quadrant Position', fontsize=11, fontweight='bold')
    
    # Add quadrant labels (smaller)
    ax2.text(0.25, 75, 'WATCH\nCLOSELY', ha='center', fontsize=7, color='#ff7f0e', alpha=0.4)
    ax2.text(0.75, 75, 'ACT\nNOW', ha='center', fontsize=7, color='#d62728', alpha=0.4)
    ax2.text(0.25, 25, 'NOT\nPRIORITY', ha='center', fontsize=7, color='#7f7f7f', alpha=0.4)
    ax2.text(0.75, 25, 'SAFELY\nIGNORE', ha='center', fontsize=7, color='#2ca02c', alpha=0.4)
    
    # ==========================================================================
    # Panel 3: Observation Info (Top Right)
    # ==========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    obs_text = f"""OBSERVATION HISTORY
{'='*25}

Total CDMs:    {len(event_cdms)}
Period:        {event_info.get('observation_days', 0):.1f} days

First CDM:     {str(event_cdms['CREATION_DATE'].iloc[0])[:19]}
Last CDM:      {str(event_cdms['CREATION_DATE'].iloc[-1])[:19]}

TCA:           {str(event_cdms['TCA'].iloc[0])[:19]}

DATA QUALITY
{'='*25}

Valid stamps:  {event_cdms['CREATION_DATE'].notna().sum()}/{len(event_cdms)}
Valid Pc:      {event_cdms['COLLISION_PROBABILITY'].notna().sum()}/{len(event_cdms)}
Covariance:    {'Yes' if event_cdms['combined_cr_r'].notna().any() else 'No'}
"""
    
    ax3.text(0.05, 1.25, obs_text, transform=ax3.transAxes, fontsize=8.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # ==========================================================================
    # Panel 4: Collision Probability Trend (Row 2, spans 1.5 width)
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1:3, 0])
    
    if len(event_cdms) >= 2 and event_cdms['COLLISION_PROBABILITY'].notna().sum() >= 2:
        times = event_cdms['CREATION_DATE']
        pc_values = event_cdms['COLLISION_PROBABILITY']
        valid_mask = pc_values.notna()
        
        ax4.semilogy(times[valid_mask], pc_values[valid_mask], marker='o', linestyle='-', linewidth=2, 
                     markersize=7, color='#d62728', markerfacecolor='white', 
                     markeredgewidth=1.5, label='Observed Pc')
        
        ax4.axhline(y=1e-4, color='red', linestyle='--', alpha=0.3, linewidth=1, label='High Risk (10⁻⁴)')
        ax4.axhline(y=1e-6, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='Medium Risk (10⁻⁶)')
        
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_ylabel('Collision Probability (log scale)', fontsize=10)
        ax4.set_title('Collision Probability Evolution', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8, loc='best')
        ax4.grid(alpha=0.3, linestyle=':')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    else:
        ax4.text(0.5, 0.5, f'Insufficient Data\\nLast Pc: {event_cdms["COLLISION_PROBABILITY"].iloc[-1]:.2e}',
                ha='center', va='center', fontsize=11, transform=ax4.transAxes)
        ax4.axis('off')
    
    # ==========================================================================
    # Panel 5: Miss Distance Trend (Row 2-3, middle)
    # ==========================================================================
    ax5 = fig.add_subplot(gs[1:3, 1])
    
    if len(event_cdms) >= 2 and event_cdms['MISS_DISTANCE'].notna().sum() >= 2:
        times = event_cdms['CREATION_DATE']
        miss_dist = event_cdms['MISS_DISTANCE']
        valid_mask = miss_dist.notna()
        
        ax5.plot(times[valid_mask], miss_dist[valid_mask], marker='s', linestyle='-', linewidth=2, 
                 markersize=7, color='#2ca02c', markerfacecolor='white', 
                 markeredgewidth=1.5, label='Miss Distance')
        
        ax5.set_xlabel('Date', fontsize=10)
        ax5.set_ylabel('Miss Distance (meters)', fontsize=10)
        ax5.set_title('Miss Distance Evolution', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8, loc='best')
        ax5.grid(alpha=0.3, linestyle=':')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    else:
        ax5.text(0.5, 0.5, f'Insufficient Data\\nLast Miss Dist: {event_cdms["MISS_DISTANCE"].iloc[-1]:.1f} m',
                ha='center', va='center', fontsize=11, transform=ax5.transAxes)
        ax5.axis('off')
    
    # ==========================================================================
    # Panel 6: Covariance Trend (Row 2-3, right)
    # ==========================================================================
    ax6 = fig.add_subplot(gs[1:3, 2])
    
    if len(event_cdms) >= 2 and event_cdms['combined_cr_r'].notna().sum() >= 2:
        times = event_cdms['CREATION_DATE']
        cr_r = event_cdms['combined_cr_r']
        valid_mask = cr_r.notna()
        
        ax6.semilogy(times[valid_mask], cr_r[valid_mask], marker='^', linestyle='-', linewidth=2, 
                     markersize=7, color='#ff7f0e', markerfacecolor='white', 
                     markeredgewidth=1.5, label='Radial Covariance')
        
        ax6.set_xlabel('Date', fontsize=10)
        ax6.set_ylabel('Combined Cov (m² log scale)', fontsize=10)
        ax6.set_title('Measurement Uncertainty', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8, loc='best')
        ax6.grid(alpha=0.3, linestyle=':')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'Covariance Data\\nNot Available',
                ha='center', va='center', fontsize=11, transform=ax6.transAxes)
        ax6.axis('off')
    
    # Save figure
    safe_filename = event_id.replace('/', '_').replace('\\', '_')
    output_path = os.path.join(OUTPUT_DIR, f'event_{safe_filename}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path
    """Create detailed report for a single event."""
    
    # Sort CDMs by creation date
    event_cdms = event_cdms.sort_values('CREATION_DATE')
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    quadrant_colors = {
        'ACT NOW': '#d62728',
        'WATCH CLOSELY': '#ff7f0e',
        'SAFELY IGNORE': '#2ca02c',
        'NOT PRIORITY': '#7f7f7f'
    }
    title_color = quadrant_colors.get(event_info['quadrant'], 'black')
    
    fig.suptitle(f"Event Detail: {event_id}\n"
                 f"Threat: {event_info['threat_score']:.1f} | Confidence: {event_info['confidence_level']:.2f} | "
                 f"Quadrant: {event_info['quadrant']}",
                 fontsize=16, fontweight='bold', color=title_color)
    
    # ==========================================================================
    # Panel 1: Event Summary (Top Left)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    summary_text = f"""
EVENT SUMMARY
{'='*45}

Event ID:         {event_id}

Object 1:         {event_info.get('object1_name', 'N/A')}
Type:             {event_info.get('object1_type', 'N/A')}

Object 2:         {event_info.get('object2_name', 'N/A')}
Type:             {event_info.get('object2_type', 'N/A')}

TCA:              {event_info.get('tca_datetime', 'N/A')}
Final Pc:         {event_info.get('final_pc', 0):.2e}
Miss Distance:    {event_info.get('final_miss_distance', 0):.1f} m

Observation:      {event_info.get('total_cdms', 0)} CDMs
Period:           {event_info.get('observation_days', 0):.1f} days

MODEL ASSESSMENT
{'='*45}

Threat Score:     {event_info['threat_score']:.1f} / 100
Confidence:       {event_info['confidence_level']:.2f}
Quadrant:         {event_info['quadrant']}
"""
    
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ==========================================================================
    # Panel 2: Quadrant Position (Top Middle)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax2.scatter([event_info['confidence_level']], [event_info['threat_score']], 
                c=title_color, s=500, marker='*', edgecolors='black', linewidth=2, zorder=5)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Confidence Level', fontsize=10)
    ax2.set_ylabel('Threat Score', fontsize=10)
    ax2.set_title('Position in Risk Quadrant', fontsize=11, fontweight='bold')
    
    # Add quadrant labels
    ax2.text(0.25, 75, 'WATCH\nCLOSELY', ha='center', fontsize=8, color='#ff7f0e', alpha=0.5)
    ax2.text(0.75, 75, 'ACT\nNOW', ha='center', fontsize=8, color='#d62728', alpha=0.5)
    ax2.text(0.25, 25, 'NOT\nPRIORITY', ha='center', fontsize=8, color='#7f7f7f', alpha=0.5)
    ax2.text(0.75, 25, 'SAFELY\nIGNORE', ha='center', fontsize=8, color='#2ca02c', alpha=0.5)
    
    # ==========================================================================
    # Panel 3: CDM Count Info (Top Right)
    # ==========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    cdm_info_text = f"""
OBSERVATION HISTORY
{'='*30}

Total CDMs:        {len(event_cdms)}
Observation Days:  {event_info.get('observation_days', 0):.1f}

First CDM:         {event_cdms['CREATION_DATE'].iloc[0]}
Last CDM:          {event_cdms['CREATION_DATE'].iloc[-1]}

TCA:               {event_cdms['TCA'].iloc[0]}
Time to TCA:       {event_cdms['time_to_tca_hours'].iloc[-1]:.1f} hours
                   ({event_cdms['time_to_tca_hours'].iloc[-1]/24:.1f} days)

DATA QUALITY
{'='*30}

Valid timestamps:  {event_cdms['CREATION_DATE'].notna().sum()}/{len(event_cdms)}
Valid Pc values:   {event_cdms['COLLISION_PROBABILITY'].notna().sum()}/{len(event_cdms)}
Covariance data:   {'Yes' if event_cdms['combined_cr_r'].notna().any() else 'No'}
"""
    
    ax3.text(0.05, 0.95, cdm_info_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # ==========================================================================
    # Panel 4: Collision Probability Trend (Bottom Left)
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1:, 0])
    
    if len(event_cdms) >= 2:
        times = event_cdms['CREATION_DATE']
        pc_values = event_cdms['COLLISION_PROBABILITY']
        
        ax4.semilogy(times, pc_values, marker='o', linestyle='-', linewidth=2, 
                     markersize=8, color='#d62728', markerfacecolor='white', 
                     markeredgewidth=2, label='Observed Pc')
        
        # Add threshold lines
        ax4.axhline(y=1e-4, color='red', linestyle='--', alpha=0.3, linewidth=1, label='High Risk (1e-4)')
        ax4.axhline(y=1e-6, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='Medium Risk (1e-6)')
        
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_ylabel('Collision Probability (log scale)', fontsize=10)
        ax4.set_title('Collision Probability Evolution', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, f'Single CDM\nPc: {event_cdms["COLLISION_PROBABILITY"].iloc[0]:.2e}',
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.axis('off')
    
    # ==========================================================================
    # Panel 5: Miss Distance Trend (Bottom Middle)
    # ==========================================================================
    ax5 = fig.add_subplot(gs[1:, 1])
    
    if len(event_cdms) >= 2 and event_cdms['MISS_DISTANCE'].notna().sum() >= 2:
        times = event_cdms['CREATION_DATE']
        miss_dist = event_cdms['MISS_DISTANCE']
        
        ax5.plot(times, miss_dist, marker='s', linestyle='-', linewidth=2, 
                 markersize=8, color='#2ca02c', markerfacecolor='white', 
                 markeredgewidth=2, label='Miss Distance')
        
        ax5.set_xlabel('Date', fontsize=10)
        ax5.set_ylabel('Miss Distance (meters)', fontsize=10)
        ax5.set_title('Miss Distance Evolution', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax5.text(0.5, 0.5, f'Insufficient Data\nLast Miss Distance: {event_cdms["MISS_DISTANCE"].iloc[-1]:.1f} m',
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.axis('off')
    
    # ==========================================================================
    # Panel 6: Covariance/Uncertainty Trend (Bottom Right)
    # ==========================================================================
    ax6 = fig.add_subplot(gs[1:, 2])
    
    if len(event_cdms) >= 2 and event_cdms['combined_cr_r'].notna().sum() >= 2:
        times = event_cdms['CREATION_DATE']
        cr_r = event_cdms['combined_cr_r']
        
        ax6.semilogy(times, cr_r, marker='^', linestyle='-', linewidth=2, 
                     markersize=8, color='#ff7f0e', markerfacecolor='white', 
                     markeredgewidth=2, label='Radial Covariance (CR_R)')
        
        ax6.set_xlabel('Date', fontsize=10)
        ax6.set_ylabel('Combined Covariance (m² log scale)', fontsize=10)
        ax6.set_title('Measurement Uncertainty Evolution', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax6.text(0.5, 0.5, 'Covariance Data\nNot Available',
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.axis('off')
    
    # Save figure
    safe_filename = event_id.replace('/', '_').replace('\\', '_')
    output_path = os.path.join(OUTPUT_DIR, f'event_{safe_filename}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# Generate reports
for idx, (_, event_row) in enumerate(selected_events.iterrows(), 1):
    event_id = event_row['event_id']
    
    # Get all CDMs for this event
    event_cdms = parsed_data[parsed_data['event_id'] == event_id]
    
    if len(event_cdms) == 0:
        print(f"      ⚠ No CDMs found for event {event_id}")
        continue
    
    try:
        output_path = create_event_report(event_id, event_row, event_cdms)
        quadrant_label = event_row['quadrant'].replace(' ', '_')
        print(f"      [{idx:2d}/{len(selected_events)}] ✓ {quadrant_label:15s} | {event_id}")
    except Exception as e:
        print(f"      [{idx:2d}/{len(selected_events)}] ✗ Error: {event_id} - {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 80}")
print("DETAILED REPORTS COMPLETE")
print(f"{'=' * 80}")

print(f"\nGenerated {len(selected_events)} detailed event reports in {OUTPUT_DIR}/")
print(f"  - {len(act_now)} ACT NOW events")
print(f"  - {len(safely_ignore)} SAFELY IGNORE events")

print(f"\nEach report includes:")
print(f"  - Event summary and metadata")
print(f"  - Quadrant position visualization")
print(f"  - Collision probability trend over time")
print(f"  - Miss distance evolution")
print(f"  - Measurement uncertainty trends")

print(f"\n{'=' * 80}")
print("✓ STEP 5B COMPLETE")
print(f"{'=' * 80}")
