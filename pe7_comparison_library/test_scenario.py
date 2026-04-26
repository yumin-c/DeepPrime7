#!/usr/bin/env python3
"""
Test scenario comparison with small dataset
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR     = Path(__file__).resolve().parent   # pe7_comparison_library/
REPO_ROOT      = SCRIPT_DIR.parent                 # DeepPrime7 repo root
_dp_default    = REPO_ROOT.parent / 'DeepPrime'
DEEPPRIME_ROOT = Path(os.environ.get('DEEPPRIME_ROOT', _dp_default))

# Load one chunk to verify structure
chunk_file = SCRIPT_DIR / 'deepprime_features' / 'deepprime_pegrna_features_000001-000100.csv'
df = pd.read_csv(chunk_file)

print(f"Sample data shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nPAM distribution:")
print(f"  NGG: {(df['PAM_NGG'] == 1).sum()}")
print(f"  NRCH: {(df['PAM_NRCH'] == 1).sum()}")
print(f"  NRTH: {(df['PAM_NRTH'] == 1).sum()}")

print(f"\nVariant IDs: {df['variant_id'].nunique()}")
print(f"Sample variant: {df['variant_id'].iloc[0]}")

# Test import of rdo
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEEPPRIME_ROOT))

try:
    import run_deepprime_original as rdo
    print("\n✓ Successfully imported run_deepprime_original")
    print(f"  Available functions: {len([x for x in dir(rdo) if not x.startswith('_')])}")
    
    # Check key functions
    funcs = ['_tensorize_inputs', 'ensemble_predict', '_dp_dataset_to_dp7', 'deepprime7_predict']
    for func in funcs:
        if hasattr(rdo, func):
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func}")
except Exception as e:
    print(f"\n✗ Error importing: {e}")

print("\n✓ Test complete!")
