"""
STEP 1: KVN Parser for Self-Supervised CDM Learning
Extracts all parameters from KVN files for trajectory learning

Author: Ahmed Talal Alharbi (Team Leader)  
Date: November 2025
DebriSolver Competition - KAU Team
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("STEP 1: KVN PARSER")
print("Self-Supervised CDM Trajectory Learning")
print("DebriSolver Competition - KAU Team")
print("=" * 80)

# ============================================================================
# KVN PARSING FUNCTION
# ============================================================================

def parse_kvn_file(filepath):
    """
    Parse a KVN file and extract all relevant parameters.
    
    Returns a dictionary with:
    - Event metadata (TCA, creation date, message ID)
    - Conjunction parameters (Pc, miss distance, relative velocity)
    - Object 1 data (state vector, covariance, metadata)
    - Object 2 data (state vector, covariance, metadata)
    """
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data = {}
        current_object = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('COMMENT'):
                continue
            
            # Parse key = value pairs
            if '=' in line:
                parts = re.split(r'\s*=\s*', line, maxsplit=1)
                
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Skip empty values
                    if not value:
                        continue
                    
                    # Remove units like [m], [m**2], [km], [km/s], [m/s]
                    value = re.sub(r'\s*\[[^\]]+\]\s*$', '', value).strip()
                    
                    # Track which object section we're in
                    if key == 'OBJECT':
                        if value == 'OBJECT1':
                            current_object = 'object1'
                        elif value == 'OBJECT2':
                            current_object = 'object2'
                        continue
                    
                    # Fields that belong to specific objects
                    object_specific_fields = [
                        # Covariance matrix (diagonal elements we care about most)
                        'CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N',
                        'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT',
                        'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT',
                        'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT',
                        # State vector
                        'X', 'Y', 'Z', 'X_DOT', 'Y_DOT', 'Z_DOT',
                        # Object metadata
                        'OBJECT_DESIGNATOR', 'OBJECT_NAME', 'OBJECT_TYPE',
                        'INTERNATIONAL_DESIGNATOR', 'CATALOG_NAME',
                        'EPHEMERIS_NAME', 'COVARIANCE_METHOD', 'MANEUVERABLE',
                        'REF_FRAME', 'COV_REF_FRAME', 'EPOCH', 'OPERATOR'
                    ]
                    
                    if key in object_specific_fields and current_object:
                        data[f"{current_object}_{key}"] = value
                    else:
                        data[key] = value
        
        # Extract event_id from filename: CDM_<obj1>_<obj2>_<msg_id>.kvn
        filename = Path(filepath).stem
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'CDM':
            data['object1_norad_id'] = parts[1]
            data['object2_norad_id'] = parts[2]
            data['event_id'] = f"{parts[1]}_{parts[2]}"  # Unique pair identifier
            data['cdm_id'] = '_'.join(parts[3:])  # Message identifier
        else:
            data['event_id'] = filename
            data['cdm_id'] = filename
        
        data['source_file'] = filename
        
        return data
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n[1/5] Searching for KVN files...")

kvn_directory = input("\nEnter path to KVN files directory (or press Enter for current): ").strip()
if not kvn_directory:
    kvn_directory = '.'

kvn_files = list(Path(kvn_directory).glob('**/*.kvn'))

if len(kvn_files) == 0:
    print(f"✗ No .kvn files found in '{kvn_directory}'")
    exit(1)

print(f"    ✓ Found {len(kvn_files):,} KVN files")

# ============================================================================
# TEST PARSER
# ============================================================================

print("\n[2/5] Testing parser on sample file...")

test_data = parse_kvn_file(kvn_files[0])
if test_data:
    print(f"    ✓ Parsed {len(test_data)} fields from: {kvn_files[0].name}")
    
    # Check key fields
    key_fields = ['CREATION_DATE', 'TCA', 'COLLISION_PROBABILITY', 'MISS_DISTANCE',
                  'object1_CR_R', 'object2_CR_R', 'event_id']
    print("    Key fields check:")
    for field in key_fields:
        status = "✓" if field in test_data else "✗"
        value = str(test_data.get(field, 'MISSING'))[:40]
        print(f"      {status} {field}: {value}")
else:
    print("    ✗ Failed to parse test file")
    exit(1)

# ============================================================================
# PARSE ALL FILES
# ============================================================================

print(f"\n[3/5] Parsing all {len(kvn_files):,} files...")

all_data = []
errors = 0
last_progress = 0

for i, kvn_file in enumerate(kvn_files, 1):
    # Progress update every 5%
    progress = int(i / len(kvn_files) * 100)
    if progress >= last_progress + 5:
        print(f"    Progress: {i:,}/{len(kvn_files):,} ({progress}%)")
        last_progress = progress
    
    parsed = parse_kvn_file(kvn_file)
    if parsed:
        all_data.append(parsed)
    else:
        errors += 1

print(f"\n    ✓ Successfully parsed: {len(all_data):,}")
if errors > 0:
    print(f"    ✗ Errors: {errors:,}")

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

print("\n[4/5] Creating DataFrame and converting types...")

df = pd.DataFrame(all_data)

# Convert timestamps
for col in ['CREATION_DATE', 'TCA']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Convert numeric columns
numeric_patterns = [
    'COLLISION_PROBABILITY', 'MISS_DISTANCE', 'RELATIVE_SPEED',
    'RELATIVE_POSITION_R', 'RELATIVE_POSITION_T', 'RELATIVE_POSITION_N',
    'RELATIVE_VELOCITY_R', 'RELATIVE_VELOCITY_T', 'RELATIVE_VELOCITY_N',
    'CR_R', 'CT_T', 'CN_N', 'CT_R', 'CN_R', 'CN_T',
    'X', 'Y', 'Z', 'X_DOT', 'Y_DOT', 'Z_DOT',
    'SCREEN_VOLUME_X', 'SCREEN_VOLUME_Y', 'SCREEN_VOLUME_Z'
]

for col in df.columns:
    for pattern in numeric_patterns:
        if pattern in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            break

# Calculate derived features
print("    Calculating derived features...")

# Time to TCA (hours)
if 'CREATION_DATE' in df.columns and 'TCA' in df.columns:
    valid_mask = df['CREATION_DATE'].notna() & df['TCA'].notna()
    df.loc[valid_mask, 'time_to_tca_hours'] = (
        df.loc[valid_mask, 'TCA'] - df.loc[valid_mask, 'CREATION_DATE']
    ).dt.total_seconds() / 3600

# Combined covariance (sum of both objects' position uncertainty)
if 'object1_CR_R' in df.columns and 'object2_CR_R' in df.columns:
    df['combined_cr_r'] = df['object1_CR_R'].fillna(0) + df['object2_CR_R'].fillna(0)
if 'object1_CT_T' in df.columns and 'object2_CT_T' in df.columns:
    df['combined_ct_t'] = df['object1_CT_T'].fillna(0) + df['object2_CT_T'].fillna(0)
if 'object1_CN_N' in df.columns and 'object2_CN_N' in df.columns:
    df['combined_cn_n'] = df['object1_CN_N'].fillna(0) + df['object2_CN_N'].fillna(0)

# Log10 of collision probability
if 'COLLISION_PROBABILITY' in df.columns:
    df['log10_pc'] = np.log10(df['COLLISION_PROBABILITY'].replace(0, 1e-15))

# Total covariance trace (measure of overall uncertainty)
if all(col in df.columns for col in ['combined_cr_r', 'combined_ct_t', 'combined_cn_n']):
    df['covariance_trace'] = df['combined_cr_r'] + df['combined_ct_t'] + df['combined_cn_n']
    df['log10_cov_trace'] = np.log10(df['covariance_trace'].replace(0, 1))

print(f"    ✓ DataFrame: {len(df):,} rows, {len(df.columns)} columns")

# ============================================================================
# SAVE
# ============================================================================

print("\n[5/5] Saving parsed data...")

output_file = 'parsed_cdm_data.csv'
df.to_csv(output_file, index=False)
print(f"    ✓ Saved to: {output_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print(f"\n{'=' * 80}")
print("PARSING COMPLETE - SUMMARY")
print(f"{'=' * 80}")

print(f"\nDataset Overview:")
print(f"  Total CDMs:         {len(df):,}")
print(f"  Unique events:      {df['event_id'].nunique():,}")
print(f"  CDMs per event:     {len(df) / df['event_id'].nunique():.2f} (avg)")

# Date range
if 'CREATION_DATE' in df.columns:
    valid_dates = df['CREATION_DATE'].dropna()
    if len(valid_dates) > 0:
        print(f"\nDate Range:")
        print(f"  First CDM:  {valid_dates.min()}")
        print(f"  Last CDM:   {valid_dates.max()}")

# Collision probability
if 'COLLISION_PROBABILITY' in df.columns:
    print(f"\nCollision Probability:")
    print(f"  Min:    {df['COLLISION_PROBABILITY'].min():.2e}")
    print(f"  Median: {df['COLLISION_PROBABILITY'].median():.2e}")
    print(f"  Max:    {df['COLLISION_PROBABILITY'].max():.2e}")

# Time to TCA
if 'time_to_tca_hours' in df.columns:
    valid = df['time_to_tca_hours'].dropna()
    print(f"\nTime to TCA (hours):")
    print(f"  Valid:  {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    print(f"  Min:    {valid.min():.1f}")
    print(f"  Median: {valid.median():.1f}")
    print(f"  Max:    {valid.max():.1f}")

# Covariance
if 'combined_cr_r' in df.columns:
    print(f"\nCombined Covariance (Object1 + Object2):")
    print(f"  CR_R mean: {df['combined_cr_r'].mean():.2e} m²")
    print(f"  CT_T mean: {df['combined_ct_t'].mean():.2e} m²")
    print(f"  CN_N mean: {df['combined_cn_n'].mean():.2e} m²")

# Event distribution
event_counts = df.groupby('event_id').size()
print(f"\nEvents by CDM count:")
print(f"  1 CDM:      {(event_counts == 1).sum():,}")
print(f"  2-3 CDMs:   {((event_counts >= 2) & (event_counts <= 3)).sum():,}")
print(f"  4-10 CDMs:  {((event_counts >= 4) & (event_counts <= 10)).sum():,}")
print(f"  11-20 CDMs: {((event_counts >= 11) & (event_counts <= 20)).sum():,}")
print(f"  20+ CDMs:   {(event_counts > 20).sum():,}")

# Object types
if 'object1_OBJECT_TYPE' in df.columns:
    print(f"\nObject 1 Types:")
    print(df['object1_OBJECT_TYPE'].value_counts().head())

if 'object2_OBJECT_TYPE' in df.columns:
    print(f"\nObject 2 Types:")
    print(df['object2_OBJECT_TYPE'].value_counts().head())

print(f"\n{'=' * 80}")
print("✓ STEP 1 COMPLETE")
print(f"{'=' * 80}")
print("\nNext: python step2_prepare_sequences.py")
