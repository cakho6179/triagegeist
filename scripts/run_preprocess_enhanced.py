import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.preprocess import (load_raw, clip_outliers, add_missing_indicators, impute_missing,
                             add_engineered_features, extract_text_features, encode_categorical,
                             temporal_split, CATEGORICAL_FEATURES)

from src.features_extra import add_extra_features

PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42

print("=" * 60)
print("ENHANCED PREPROCESSING (with extra clinical features)")
print("=" * 60)

# Step 1: Load
df = load_raw()

# Step 2: Clip outliers
print("\n--- Step 1: Clip outliers ---")
df = clip_outliers(df)

# Step 3: Add missing indicators
print("\n--- Step 2: Add missing indicators ---")
df = add_missing_indicators(df)

# Step 4: Impute
print("\n--- Step 3: Impute missing values ---")
df = impute_missing(df)

# Step 5: Add engineered features
print("\n--- Step 4: Add engineered features ---")
df = add_engineered_features(df)

# Step 6: Add EXTRA features
print("\n--- Step 5: Add EXTRA clinical features ---")
df = add_extra_features(df)

# Step 7: Extract text features
print("\n--- Step 6: Extract text features ---")
df, tfidf_df, _ = extract_text_features(df, max_features=500)
df = pd.concat([df, tfidf_df], axis=1)

# Step 8: Encode categoricals
print("\n--- Step 7: Encode categoricals ---")
df = encode_categorical(df)

# Step 9: Drop raw text
print("\n--- Step 8: Drop raw text columns ---")
for col in ['chief_complaint_text', 'injury_cause_text']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Step 10: Split
print("\n--- Step 9: Split data ---")
train, val = temporal_split(df)

X_train = train.drop(columns=['target_triage_acuity', 'wait_time_minutes'], errors='ignore')
y_train = train['target_triage_acuity'].values

X_val = val.drop(columns=['target_triage_acuity', 'wait_time_minutes'], errors='ignore')
y_val = val['target_triage_acuity'].values

print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,}")
print(f"  Features: {X_train.shape[1]}")
for name, target_vals in [("Train", y_train), ("Val", y_val)]:
    dist = pd.Series(target_vals).value_counts().sort_index()
    dist_str = ", ".join([f"{k}:{v}" for k, v in dist.items()])
    print(f"  {name} target: {dist_str}")

# Save
suffix = "_enhanced"
np.savez_compressed(
    PROCESSED_DIR / f"processed_data{suffix}.npz",
    X_train=X_train.values, y_train=y_train,
    X_val=X_val.values, y_val=y_val,
    feature_names=np.array(X_train.columns.tolist(), dtype=object),
)

pd.DataFrame(X_train, columns=X_train.columns).to_parquet(
    PROCESSED_DIR / f"X_train{suffix}.parquet", index=False
)
pd.DataFrame(X_val, columns=X_val.columns).to_parquet(
    PROCESSED_DIR / f"X_val{suffix}.parquet", index=False
)
pd.Series(y_train).to_csv(PROCESSED_DIR / f"y_train{suffix}.csv", index=False)
pd.Series(y_val).to_csv(PROCESSED_DIR / f"y_val{suffix}.csv", index=False)

print(f"\n  Saved to {PROCESSED_DIR}/ (suffix: {suffix})")
print("=" * 60)
