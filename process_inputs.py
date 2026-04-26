import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/250716_24k_final.csv")

# --- Step 1: Calculate binomial-based std and clip mean to >= 0 ---
target_prefixes = ["DN", "DC", "DT", "AN"]
epsilon = 1e-6

for prefix in target_prefixes:
    total_count = df[f"{prefix}R1_Total"] + df[f"{prefix}R2_Total"]

    df[f"{prefix}_M_EF"] = df[f"{prefix}_M_EF"].clip(lower=0, upper=100)
    df[f"{prefix}_M_EF_std"] = np.sqrt(df[f"{prefix}_M_EF"]/100 * (1 - df[f"{prefix}_M_EF"]/100) / (total_count + epsilon) + epsilon) * 100

# --- Step 2: Split assignment ---
df["Split"] = "NA"

# Exclude subgroups and assign to test
excluded_subgroups = ['Nature_Britt_PE7', 'NBT_PRIDICT', 'PAM_64combi', 'Lib6K_PE2vsPE7_NRCH', 'Lib6K_PE2vsPE7_NGG']
df.loc[df['Subgroup'].isin(excluded_subgroups[1:]), 'Split'] = 'Train'
df.loc[df['Subgroup'].isin(excluded_subgroups[:1]), 'Split'] = 'Test'

# --- Group 1,2,3,5 split by Target using representative Subgroup ---
mask_g135 = (df['Group'].isin([1, 2, 3, 5])) & (~df['Subgroup'].isin(excluded_subgroups))
df_g135 = df[mask_g135].copy()

# Determine representative Subgroup for each Target
target_to_rep_subgroup = (
    df_g135.groupby("Target")["Subgroup"]
    .agg(lambda x: x.value_counts().idxmax())
)

targets = target_to_rep_subgroup.index.to_numpy()
subgroup_labels = target_to_rep_subgroup.values

# Stratified split by representative Subgroup
train_t, temp_t = train_test_split(targets, test_size=0.2, random_state=42, stratify=subgroup_labels)
valid_t, test_t = train_test_split(temp_t, test_size=0.5, random_state=42,
                                   stratify=target_to_rep_subgroup.loc[temp_t].values)

# Assign splits
df.loc[mask_g135 & df['Target'].isin(train_t), 'Split'] = 'Train'
df.loc[mask_g135 & df['Target'].isin(valid_t), 'Split'] = 'Valid'
df.loc[mask_g135 & df['Target'].isin(test_t), 'Split'] = 'Test'

# --- Group 4 split by subtarget (Target[:26]) ---
mask_g4 = (df['Group'] == 4) & (~df['Subgroup'].isin(excluded_subgroups))
df['subtarget'] = df['Target'].str[:26]
unique_subtargets = df.loc[mask_g4, 'subtarget'].unique()
train_s, temp_s = train_test_split(unique_subtargets, test_size=0.2, random_state=42)
valid_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=42)

df.loc[mask_g4 & df['subtarget'].isin(train_s), 'Split'] = 'Train'
df.loc[mask_g4 & df['subtarget'].isin(valid_s), 'Split'] = 'Valid'
df.loc[mask_g4 & df['subtarget'].isin(test_s), 'Split'] = 'Test'

# --- Step 3: Keep only required columns ---
keep_cols = [f"{p}_M_EF" for p in target_prefixes] + \
            [f"{p}_M_EF_std" for p in target_prefixes] + [
    'idx', 'Subpool', 'Group', 'Subgroup', 'Target', 'Target_extended', 'Masked_EditSeq', 'edit_type', 'Edit_pos', 'Edit_len',
    'Tm1_PBS', 'Tm2_RTT_cTarget_sameLength', 'Tm3_RTT_cTarget_replaced',
    'Tm4_cDNA_PAM-oppositeTarget', 'Tm5_RTT_cDNA',
    'GC_count_PBS', 'GC_count_RTT', 'GC_contents_PBS', 'GC_contents_RTT',
    'MFE_RT-PBS-polyT', 'MFE_Spacer', 'DeepSpCas9_score', 'DLD1_PE2max_DeepPrime_score', 'Split'
]

df_final = df[keep_cols]

# --- Step 4: Save ---
df_final.to_csv("data/processed_all.csv", index=False)

# Save split-specific files
df_final[df_final['Split'] == 'Train'].to_csv("data/processed_train.csv", index=False)
df_final[df_final['Split'] == 'Valid'].to_csv("data/processed_valid.csv", index=False)
df_final[df_final['Split'] == 'Test'].to_csv("data/processed_test.csv", index=False)
