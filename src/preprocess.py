import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/raw/nhamcs_data_2018_22.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CLIP_RANGES = {
    'temp': (90, 110),
    'heart_rate': (20, 250),
    'resp_rate': (4, 60),
    'sys_bp': (50, 260),
    'dias_bp': (20, 160),
    'spo2': (50, 100),
    'pain_score': (0, 10),
}

HIGH_RISK_KEYWORDS = [
    'chest pain', 'shortness of breath', 'sob', 'syncope', 'seizure',
    'trauma', 'bleeding', 'altered mental', 'unconscious', 'stroke',
    'mi', 'stem', 'cardiac arrest', 'respiratory arrest', 'intubated',
    'overdose', 'anaphylaxis', 'sepsis', 'meningitis',
    'difficulty breathing', 'respiratory distress', 'hypoxia', 'cyanosis',
    'hematemesis', 'hemoptysis', 'melena', 'blood', 'hemorrhage',
    'head injury', 'fall', 'mva', 'motor vehicle', 'burn',
    'suicidal', 'hallucination', 'agitation', 'violent',
    'severe', 'critical', 'unstable', 'deteriorating',
    'amputation', 'open fracture', 'dislocation', 'fracture',
    'ectopic', 'pregnancy complication', 'preterm labor',
    'newborn', 'neonatal', 'pediatric', 'infant',
]

NUMERIC_FEATURES = [
    'temp', 'heart_rate', 'resp_rate', 'sys_bp', 'dias_bp', 'spo2', 'pain_score',
]

ALL_IMPUTE_COLS = [
    'temp', 'heart_rate', 'resp_rate', 'sys_bp', 'dias_bp', 'spo2', 'pain_score',
    'vitals_during_visit', 'seen_last_72h', 'arrival_time',
]

CATEGORICAL_FEATURES = [
    'sex', 'race', 'insurance', 'region', 'residence', 'ems_arrival',
    'episode', 'is_injury_poison', 'no_payment', 'vitals_during_visit',
    'seen_last_72h',
]

COMORBIDITY_COLS = [
    'hist_alzheimers', 'hist_asthma', 'hist_cancer', 'hist_stroke', 'hist_ckd',
    'hist_copd', 'hist_chf', 'hist_cad', 'hist_depression', 'hist_diabetes_t1',
    'hist_diabetes_t2', 'hist_diabetes_unspec', 'hist_esrd', 'hist_pe', 'hist_hiv',
    'hist_high_cholesterol', 'hist_hypertension', 'hist_obesity', 'hist_sleep_apnea',
    'hist_osteoporosis', 'hist_substance_abuse',
]

def load_raw():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def clip_outliers(df, ranges=CLIP_RANGES):
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            before = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            if before > 0:
                print(f"  Clipped {col}: {before} values to [{lo}, {hi}]")
    return df

def add_missing_indicators(df, cols=NUMERIC_FEATURES):
    # Also add for other imputed columns
    extra = [c for c in ALL_IMPUTE_COLS if c not in cols]
    cols = list(cols) + list(extra)
    for col in cols:
        if col in df.columns:
            missing_col = f"{col}_missing"
            df[missing_col] = df[col].isnull().astype(int)
    return df

def impute_missing(df, cols=ALL_IMPUTE_COLS):
    for col in cols:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            n = df[col].isnull().sum()
            if n > 0:
                print(f"  Warning: {col} still has {n} NaN after median imputation")
    cat_with_missing = [c for c in ['is_injury_poison'] if c in df.columns and df[c].isnull().any()]
    for col in cat_with_missing:
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)
        print(f"  Imputed {col}: filled {df[col].isnull().sum()} with mode='{mode_val}'")
    return df

def add_engineered_features(df):
    # ========= VITAL SIGNS DERIVATIVES =========
    df['shock_index'] = df['heart_rate'] / df['sys_bp'].replace(0, np.nan)
    df['shock_index'] = df['shock_index'].clip(0, 5)

    df['pulse_pressure'] = df['sys_bp'] - df['dias_bp']
    df['pulse_pressure'] = df['pulse_pressure'].clip(0, 200)

    df['map'] = df['dias_bp'] + (df['pulse_pressure'] / 3)

    df['spo2_bp_ratio'] = df['spo2'] / df['sys_bp'].replace(0, np.nan)
    df['spo2_bp_ratio'] = df['spo2_bp_ratio'].clip(0, 5)

    df['rr_temp_product'] = df['resp_rate'] * df['temp'] / 100

    df['sys_bp_norm'] = df['sys_bp'] / 100
    df['hr_bp_ratio'] = df['heart_rate'] / df['sys_bp_norm'].replace(0, np.nan)
    df['hr_bp_ratio'] = df['hr_bp_ratio'].clip(0, 5)

    df['age_bp_product'] = df['age'] * df['sys_bp'] / 100

    # ========= VITAL FLAGS (severity thresholds) =========
    df['elderly'] = (df['age'] >= 65).astype(int)
    df['pediatric'] = (df['age'] <= 5).astype(int)
    df['young_adult'] = ((df['age'] >= 18) & (df['age'] < 40)).astype(int)
    df['middle_aged'] = ((df['age'] >= 40) & (df['age'] < 65)).astype(int)

    df['hypoxia'] = (df['spo2'] < 92).astype(int)
    df['severe_hypoxia'] = (df['spo2'] < 85).astype(int)
    df['tachycardia'] = (df['heart_rate'] > 100).astype(int)
    df['severe_tachy'] = (df['heart_rate'] > 130).astype(int)
    df['bradycardia'] = (df['heart_rate'] < 60).astype(int)
    df['severe_brady'] = (df['heart_rate'] < 40).astype(int)
    df['hypotension'] = (df['sys_bp'] < 90).astype(int)
    df['severe_hypotension'] = (df['sys_bp'] < 70).astype(int)
    df['hypertension_flag'] = (df['sys_bp'] > 180).astype(int)
    df['severe_hypertension'] = (df['sys_bp'] > 220).astype(int)
    df['tachypnea'] = (df['resp_rate'] > 24).astype(int)
    df['severe_tachypnea'] = (df['resp_rate'] > 30).astype(int)
    df['bradypnea'] = (df['resp_rate'] < 10).astype(int)
    df['fever'] = (df['temp'] > 100.4).astype(int)
    df['high_fever'] = (df['temp'] > 103).astype(int)
    df['hypothermia'] = (df['temp'] < 95.0).astype(int)
    df['severe_hypothermia'] = (df['temp'] < 93.0).astype(int)
    df['high_pain'] = (df['pain_score'] >= 7).astype(int)
    df['severe_pain'] = (df['pain_score'] == 10).astype(int)
    df['moderate_pain'] = ((df['pain_score'] >= 4) & (df['pain_score'] <= 6)).astype(int)

    # ========= CLINICAL SCORING SYSTEMS =========
    df['qsofa'] = (df['tachypnea'].fillna(0).astype(int) +
                   (df['sys_bp'].fillna(120) <= 100).astype(int))

    df['mews_points'] = (np.where(df['heart_rate'].fillna(80) > 120, 2,
                         np.where(df['heart_rate'].fillna(80) > 100, 1, 0)) +
                         np.where(df['sys_bp'].fillna(120) < 80, 2,
                         np.where(df['sys_bp'].fillna(120) < 100, 1, 0)) +
                         np.where(df['resp_rate'].fillna(18) > 30, 2,
                         np.where(df['resp_rate'].fillna(18) > 20, 1, 0)) +
                         np.where(df['temp'].fillna(98.6) < 95, 2,
                         np.where(df['temp'].fillna(98.6) > 100.4, 1, 0)) +
                         np.where(df['pain_score'].fillna(5) >= 7, 1, 0))

    df['sirs_criteria'] = ((df['temp'].fillna(98.6) > 100.4).astype(int) +
                           (df['temp'].fillna(98.6) < 96.8).astype(int) +
                           (df['heart_rate'].fillna(80) > 90).astype(int) +
                           (df['resp_rate'].fillna(18) > 20).astype(int))

    df['sepsis_risk'] = (df['sirs_criteria'] >= 2).astype(int) * df['hypoxia']

    df['cardiac_risk'] = ((df['elderly'] + df['tachycardia'] +
                           df['hypotension'] + df['shock_index'].fillna(0.5).gt(0.9).astype(int)) >= 2).astype(int)

    df['resp_distress'] = (df['hypoxia'] | df['severe_tachypnea']).astype(int)

    df['critical_vitals'] = (df['severe_hypoxia'] | df['severe_tachy'] |
                             df['severe_hypotension'] | df['severe_tachypnea'] |
                             df['high_fever'] | df['severe_brady']).astype(int)

    # ========= AGE DERIVED FEATURES =========
    df['age_sq'] = (df['age'] / 10) ** 2
    df['age_cube'] = (df['age'] / 30) ** 3
    df['age_decade'] = (df['age'] // 10).astype(int)

    df['age_x_hr'] = df['age'] * df['heart_rate'].fillna(80) / 100
    df['age_x_rr'] = df['age'] * df['resp_rate'].fillna(18) / 100

    # ========= TEMPORAL FEATURES =========
    df['arrival_hour'] = (df['arrival_time'].fillna(1200) / 100).astype(int)
    df['arrival_hour_sin'] = np.sin(2 * np.pi * df['arrival_hour'] / 24)
    df['arrival_hour_cos'] = np.cos(2 * np.pi * df['arrival_hour'] / 24)

    df['visit_month_sin'] = np.sin(2 * np.pi * df['visit_month'] / 12)
    df['visit_month_cos'] = np.cos(2 * np.pi * df['visit_month'] / 12)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['night_shift'] = ((df['arrival_hour'] >= 22) | (df['arrival_hour'] < 6)).astype(int)
    df['evening'] = ((df['arrival_hour'] >= 17) & (df['arrival_hour'] < 22)).astype(int)
    df['morning'] = ((df['arrival_hour'] >= 6) & (df['arrival_hour'] < 12)).astype(int)
    df['afternoon'] = ((df['arrival_hour'] >= 12) & (df['arrival_hour'] < 17)).astype(int)

    df['weekend'] = df['day_of_week'].isin([1, 7]).astype(int)

    df['winter'] = df['visit_month'].isin([12, 1, 2]).astype(int)
    df['spring'] = df['visit_month'].isin([3, 4, 5]).astype(int)
    df['summer'] = df['visit_month'].isin([6, 7, 8]).astype(int)
    df['fall'] = df['visit_month'].isin([9, 10, 11]).astype(int)

    # ========= DEMOGRAPHIC DERIVED =========
    df['ems'] = (df['ems_arrival'].str.lower() == 'yes').astype(int)

    df['no_insurance'] = (df['insurance'].str.lower().str.contains('none|self|uninsured|blank', na=False)).astype(int)

    # ========= COMORBIDITY FEATURES =========
    df['comorbidity_count'] = df[COMORBIDITY_COLS].sum(axis=1)

    cardio_cols = ['hist_cad', 'hist_chf', 'hist_hypertension', 'hist_high_cholesterol']
    df['cardio_burden'] = df[[c for c in cardio_cols if c in df.columns]].sum(axis=1)

    diabetes_cols = ['hist_diabetes_t1', 'hist_diabetes_t2', 'hist_diabetes_unspec']
    df['diabetes_burden'] = df[[c for c in diabetes_cols if c in df.columns]].sum(axis=1)

    resp_cols = ['hist_asthma', 'hist_copd', 'hist_sleep_apnea']
    df['resp_burden'] = df[[c for c in resp_cols if c in df.columns]].sum(axis=1)

    df['ckd_esrd'] = (df.get('hist_ckd', 0) | df.get('hist_esrd', 0)).astype(int)
    df['cardio_resp_combo'] = (df['cardio_burden'] > 0).astype(int) & (df['resp_burden'] > 0).astype(int)

    df['multi_comorbidity'] = (df['comorbidity_count'] >= 3).astype(int)

    # ========= COMORBIDITY x VITAL INTERACTIONS =========
    df['elderly_x_hypoxia'] = df['elderly'] * df['hypoxia']
    df['elderly_x_hypotension'] = df['elderly'] * df['hypotension']
    df['elderly_x_tachy'] = df['elderly'] * df['tachycardia']
    df['cardio_x_hypoxia'] = (df['cardio_burden'] > 0).astype(int) * df['hypoxia']
    df['cardio_x_chest'] = (df['cardio_burden'] > 0).astype(int) * df['high_pain']

    # ========= INTERACTION FEATURES =========
    df['age_x_comorbidity'] = df['age'] * df['comorbidity_count'] / 100
    df['age_x_cardio'] = df['age'] * df['cardio_burden'] / 100

    df['shock_x_elderly'] = df['shock_index'] * df['elderly']
    df['shock_x_cardio'] = df['shock_index'] * (df['cardio_burden'] > 0).astype(int)
    df['shock_x_hypoxia'] = df['shock_index'] * df['hypoxia']

    df['hr_x_shock'] = df['heart_rate'] * df['shock_index']
    df['rr_x_tachy'] = df['resp_rate'] * df['tachycardia']

    df['hypoxia_x_tachy'] = df['hypoxia'] * df['tachycardia']
    df['hypoxia_x_fever'] = df['hypoxia'] * df['fever']
    df['tachy_x_hypotension'] = df['tachycardia'] * df['hypotension']
    df['pain_x_tachy'] = df['high_pain'] * df['tachycardia']
    df['pain_x_hypoxia'] = df['high_pain'] * df['hypoxia']

    df['hypoxia_x_elderly_fever'] = df['hypoxia'] * df['elderly'] * df['fever']

    # ========= COMPOSITE RISK SCORES =========
    agg_vitals = (df['hypoxia'] + df['tachycardia'] + df['hypotension'] +
                  df['tachypnea'] + df['fever'] + df['high_pain'] +
                  df['hypothermia'] + df['bradycardia'])
    df['abnormal_vital_count'] = agg_vitals
    df['severe_vital_count'] = (df['severe_hypoxia'] + df['severe_tachy'] +
                                df['severe_hypotension'] + df['severe_tachypnea'] +
                                df['high_fever'] + df['severe_brady'])

    df['triage_risk_score'] = (df['qsofa'] * 3 + df['mews_points'] +
                               df['abnormal_vital_count'] + df['elderly'] * 2 +
                               df['comorbidity_count'])

    df['age_vital_composite'] = (df['age'] / 10 + df['abnormal_vital_count'] * 2 +
                                 df['shock_index'].fillna(0.5) * 5)

    print(f"  Added {len(df.columns)} engineered features")
    return df

def extract_text_features(df, max_features=200):
    print("\nExtracting TF-IDF from chief complaint text...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
    )
    text_data = df['chief_complaint_text'].fillna('')
    tfidf_matrix = tfidf.fit_transform(text_data)
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'cc_{w}' for w in feature_names],
        index=df.index,
    )

    df['cc_high_risk'] = df['chief_complaint_text'].fillna('').str.lower().apply(
        lambda t: any(kw in t for kw in HIGH_RISK_KEYWORDS)
    ).astype(int)

    df['cc_length'] = df['chief_complaint_text'].fillna('').str.len()

    print(f"  TF-IDF: {len(feature_names)} features from {len(text_data)} documents")
    print(f"  High risk complaints: {df['cc_high_risk'].sum():,} ({df['cc_high_risk'].mean()*100:.1f}%)")
    return df, tfidf_df, tfidf

def encode_categorical(df):
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    encoded_dfs = []
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dummy_na=False)
        encoded_dfs.append(dummies)
        df.drop(columns=[col], inplace=True)
    if encoded_dfs:
        df = pd.concat([df] + encoded_dfs, axis=1)
    return df

def temporal_split(df, train_years=None, val_years=None):
    if train_years is None:
        train_years = [2018, 2019, 2020]
    if val_years is None:
        val_years = [2021, 2022]
    train = df[df['year'].isin(train_years)].copy()
    val = df[df['year'].isin(val_years)].copy()
    print(f"  Temporal split: train={len(train):,} ({train_years}), val={len(val):,} ({val_years})")
    return train, val

def run(include_text=False, max_text_features=500, temporal=True):
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_raw()

    print("\n--- Step 1: Clip outliers ---")
    df = clip_outliers(df)

    print("\n--- Step 2: Add missing indicators ---")
    df = add_missing_indicators(df)

    print("\n--- Step 3: Impute missing values ---")
    df = impute_missing(df)

    print("\n--- Step 4: Add engineered features ---")
    df = add_engineered_features(df)

    if include_text:
        print("\n--- Step 5: Extract text features ---")
        df, tfidf_df, _ = extract_text_features(df, max_features=max_text_features)
        df = pd.concat([df, tfidf_df], axis=1)
        step_label = "Step 6"
    else:
        step_label = "Step 5"

    print(f"\n--- {step_label}: Encode categoricals ---")
    df = encode_categorical(df)

    print(f"\n--- {step_label}: Drop raw text columns ---")
    for col in ['chief_complaint_text', 'injury_cause_text']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"  Dropped {col}")

    print(f"\n--- {step_label.replace('Step', 'Step ')}: Split data ---")
    target = df['target_triage_acuity'].values
    wait_time = df['wait_time_minutes'].values

    if temporal:
        train, val = temporal_split(df)
    else:
        train, val = train_test_split(
            df, test_size=0.2, random_state=RANDOM_STATE,
            stratify=df['target_triage_acuity'],
        )

    X_train = train.drop(columns=['target_triage_acuity', 'wait_time_minutes'], errors='ignore')
    y_train = train['target_triage_acuity'].values
    w_train = train.get('wait_time_minutes', pd.Series(np.nan)).values

    X_val = val.drop(columns=['target_triage_acuity', 'wait_time_minutes'], errors='ignore')
    y_val = val['target_triage_acuity'].values
    w_val = val.get('wait_time_minutes', pd.Series(np.nan)).values

    print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,}")
    print(f"  Features: {X_train.shape[1]}")

    for name, data, target_vals in [
        ("Train", X_train, y_train),
        ("Val", X_val, y_val),
    ]:
        dist = pd.Series(target_vals).value_counts().sort_index()
        dist_str = ", ".join([f"{k}:{v}" for k, v in dist.items()])
        print(f"  {name} target: {dist_str}")

    suffix = "_with_text" if include_text else "_tabular_only"
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

    print(f"\n  Saved to {PROCESSED_DIR}/")
    print("=" * 60)
    return X_train, X_val, y_train, y_val, w_train, w_val

if __name__ == "__main__":
    run(include_text=False)
