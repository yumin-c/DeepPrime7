import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
train_path = "data/processed_train.csv"
scaler_path = "models/deepprime7_scaler.pth"

FEATURE_GROUPS = {
    'tm_features': ['Tm1_PBS', 'Tm2_RTT_cTarget_sameLength', 'Tm3_RTT_cTarget_replaced',
                    'Tm4_cDNA_PAM-oppositeTarget', 'Tm5_RTT_cDNA'],
    'gc_count_features': ['GC_count_PBS', 'GC_count_RTT'],
    'gc_contents_features': ['GC_contents_PBS', 'GC_contents_RTT'],
    'mfe_features': ['MFE_RT-PBS-polyT', 'MFE_Spacer'],
    'deepspcas9_score': ['DeepSpCas9_score']
}

USE_FEATURES = {
    'tm_features': True,
    'gc_count_features': False,
    'gc_contents_features': True,
    'mfe_features': True,
    'deepspcas9_score': False,
}

# --- Setup ---
# Collect selected features
selected_features = []
for group, include in USE_FEATURES.items():
    if include:
        selected_features.extend(FEATURE_GROUPS[group])

# --- Load data ---
df = pd.read_csv(train_path)

# --- Fit scaler ---
scaler = StandardScaler()
scaler.fit(df[selected_features])

# --- Save ---
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
joblib.dump(scaler, scaler_path)

print(f"✅ Scaler trained and saved to {scaler_path}")