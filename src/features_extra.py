import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

HIGH_RISK_KW = [
    'chest pain', 'shortness of breath', 'sob', 'syncope', 'seizure',
    'trauma', 'bleeding', 'altered mental', 'unconscious', 'stroke',
    'mi', 'stemi', 'cardiac arrest', 'respiratory arrest', 'intubated',
    'overdose', 'anaphylaxis', 'sepsis', 'meningitis',
    'difficulty breathing', 'respiratory distress', 'hypoxia', 'cyanosis',
    'hematemesis', 'hemoptysis', 'melena', 'blood', 'hemorrhage',
    'head injury', 'fall', 'mva', 'motor vehicle', 'burn',
    'suicidal', 'hallucination', 'agitation', 'violent',
    'severe pain', 'cannot walk', 'paralysis', 'numbness',
    'ectopic', 'pregnancy complication',
    'newborn', 'neonatal', 'infant',
]

CARDIO_KW = ['chest pain', 'chest tightness', 'palpitations', 'heart racing',
             'mi', 'stemi', 'heart attack', 'angina', 'cardiac', 'coronary']

RESP_KW = ['shortness of breath', 'sob', 'difficulty breathing', 'wheezing',
           'cough', 'hemoptysis', 'respiratory distress', 'hypoxia',
           'copd', 'asthma', 'pneumonia', 'bronchitis', 'breath']

NEURO_KW = ['headache', 'dizziness', 'vertigo', 'seizure', 'convulsion',
            'stroke', 'cva', 'tia', 'altered mental', 'confusion',
            'unconscious', 'syncope', 'faint', 'passed out',
            'paralysis', 'weakness', 'numbness', 'tingling',
            'slurred speech', 'facial droop', 'gait']

GI_KW = ['abdominal pain', 'nausea', 'vomiting', 'diarrhea',
         'hematemesis', 'melena', 'hematochezia', 'gi bleed',
         'constipation', 'belly pain']

TRAUMA_KW = ['fall', 'mva', 'motor vehicle', 'car accident', 'burn',
             'fracture', 'laceration', 'head injury', 'trauma',
             'assault', 'stab', 'gsv', 'gunshot']

INFECT_KW = ['fever', 'chills', 'infection', 'sepsis', 'meningitis',
             'cellulitis', 'uti', 'pneumonia', 'flu', 'covid']

PSYCH_KW = ['suicidal', 'depression', 'anxiety', 'hallucination',
            'agitation', 'violent', 'psychosis', 'panic', 'overdose',
            'intoxication', 'withdrawal']

SYMPTOM_GROUPS = {
    'cardiac': CARDIO_KW,
    'respiratory': RESP_KW,
    'neurologic': NEURO_KW,
    'gi': GI_KW,
    'trauma': TRAUMA_KW,
    'infectious': INFECT_KW,
    'psychiatric': PSYCH_KW,
}

SEVERITY_WEIGHTS = {
    'cardiac': 4, 'respiratory': 4, 'neurologic': 3,
    'gi': 2, 'trauma': 3, 'infectious': 3, 'psychiatric': 2,
}

URGENCY_KW = ['acute', 'sudden', 'severe', 'progressive', 'worsening',
              'constant', 'unrelieved', 'intractable', 'emergency']

NEGATION_KW = ['no', 'denies', 'without', 'negative', 'denied', 'none']

def add_extra_features(df):
    print("\n--- Adding Enhanced Clinical Features ---")
    n_orig = len(df.columns)

    text = df['chief_complaint_text'].fillna('').str.lower()

    # ========= 1. SYMPTOM CATEGORY FLAGS =========
    print("  1. Symptom category flags...")
    for group_name, keywords in SYMPTOM_GROUPS.items():
        df[f'symp_{group_name}'] = text.apply(
            lambda t: any(kw in t for kw in keywords)
        ).astype(int)

    # ========= 2. TEXT-DERIVED FEATURES =========
    print("  2. Text-derived features...")
    df['cc_symptom_count'] = text.apply(lambda t: sum(1 for kw in HIGH_RISK_KW if kw in t))
    df['cc_has_urgency'] = text.apply(lambda t: any(kw in t for kw in URGENCY_KW)).astype(int)
    df['cc_has_negation'] = text.apply(lambda t: any(kw in t for kw in NEGATION_KW)).astype(int)
    df['cc_has_both_urgency_negation'] = (df['cc_has_urgency'] & df['cc_has_negation']).astype(int)
    df['cc_word_count'] = text.str.split().str.len()
    df['cc_symptom_density'] = df['cc_symptom_count'] / df['cc_word_count'].replace(0, 1)

    top_words = ['pain', 'chest', 'shortness', 'fever', 'headache',
                 'abdominal', 'cough', 'nausea', 'weakness', 'dizziness',
                 'fall', 'bleeding', 'breathing', 'back', 'vomiting']
    for w in top_words:
        df[f'cc_has_{w}'] = text.str.contains(r'\b' + w + r'\b', regex=True).astype(int)

    symptom_systems = ['cardiac', 'respiratory', 'neurologic', 'gi', 'trauma', 'infectious', 'psychiatric']
    df['cc_system_count'] = df[[f'symp_{s}' for s in symptom_systems]].sum(axis=1)
    df['cc_text_severity_score'] = sum(
        df[f'symp_{s}'] * w for s, w in SEVERITY_WEIGHTS.items()
    ) + df['cc_has_urgency'] * 2 - df['cc_has_negation']

    # ========= 3. AGE-ADJUSTED VITAL ABNORMALITIES =========
    print("  3. Age-adjusted vital thresholds...")
    age = df['age'].fillna(40)
    is_child = (age < 12).astype(int)
    is_adult = ((age >= 12) & (age < 65)).astype(int)
    is_elderly = (age >= 65).astype(int)

    hr = df['heart_rate'].fillna(80)
    sbp = df['sys_bp'].fillna(120)
    dbp = df['dias_bp'].fillna(70)
    rr = df['resp_rate'].fillna(18)
    spo2 = df['spo2'].fillna(98)
    temp = df['temp'].fillna(98.6)

    df['child_tachy'] = (is_child & (hr > 140)).astype(int)
    df['child_brady'] = (is_child & (hr < 80)).astype(int)
    df['elderly_tachy'] = (is_elderly & (hr > 110)).astype(int)
    df['elderly_hypotension'] = (is_elderly & (sbp < 100)).astype(int)
    df['child_hypoxia'] = (is_child & (spo2 < 94)).astype(int)
    df['elderly_hypoxia'] = (is_elderly & (spo2 < 93)).astype(int)
    df['elderly_fever'] = (is_elderly & (temp > 99.5)).astype(int)
    df['adult_sepsis_vitals'] = (is_adult & (temp > 100.4) & (hr > 90) & (rr > 20)).astype(int)

    # ========= 4. COMPOSITE CLINICAL FLAGS =========
    print("  4. Composite clinical flags...")
    cardio_hx = any(c in df.columns for c in ['hist_cad', 'hist_chf', 'hist_hypertension',
                                                'hist_high_cholesterol', 'hist_heart_disease'])
    cardiac_risk = df.get('cardio_burden', 0) if 'cardio_burden' in df.columns else \
                   (df.get('hist_cad', 0) + df.get('hist_chf', 0) + df.get('hist_hypertension', 0))

    if 'cardio_burden' in df.columns:
        cardiac_hx_flag = (df['cardio_burden'] > 0).astype(int)
    else:
        cardiac_hx_flag = (df.get('hist_cad', 0) | df.get('hist_chf', 0)).astype(int)

    df['flag_mi_suspect'] = (df.get('symp_cardiac', 0) &
                             (age >= 45) &
                             cardiac_hx_flag &
                             (hr > 90)).astype(int)

    df['flag_sepsis_suspect'] = (df.get('symp_infectious', 0) &
                                 (temp > 100.4) &
                                 (hr > 90) &
                                 (rr > 20) &
                                 ((spo2 < 95) | (sbp < 100))).astype(int)

    df['flag_stroke_suspect'] = (df.get('symp_neurologic', 0) &
                                 (sbp > 140) &
                                 (age >= 50)).astype(int)

    df['flag_copd_exacerbation'] = (df.get('symp_respiratory', 0) &
                                    df.get('hist_copd', 0) &
                                    (spo2 < 93)).astype(int)

    df['flag_dehydration'] = ((age >= 65) &
                              (hr > 100) &
                              (sbp < 110) &
                              (dbp < 70)).astype(int)

    df['flag_overdose_suspect'] = (df.get('symp_psychiatric', 0) &
                                   (rr < 12) &
                                   (spo2 < 93)).astype(int)

    df['flag_gi_bleed'] = (df.get('symp_gi', 0) &
                           (hr > 100) &
                           (sbp < 110)).astype(int)

    df['flag_anaphylaxis'] = (df.get('symp_respiratory', 0) &
                              (spo2 < 93) &
                              (sbp < 100)).astype(int)

    df['high_risk_combo_flag'] = (df[[c for c in df.columns if c.startswith('flag_')]]).sum(axis=1)

    # ========= 5. INTERACTION & CROSS FEATURES =========
    print("  5. Cross-domain interaction features...")
    df['cardio_x_pain'] = df.get('symp_cardiac', 0) * df.get('high_pain', 0)
    df['resp_x_fever'] = df.get('symp_respiratory', 0) * df.get('fever', 0)
    df['neuro_x_elderly'] = df.get('symp_neurologic', 0) * df.get('elderly', 0)
    df['trauma_x_child'] = df.get('symp_trauma', 0) * df.get('pediatric', 0)
    df['psych_x_substance'] = df.get('symp_psychiatric', 0) * df.get('hist_substance_abuse', 0)
    df['cardio_x_diabetes'] = df.get('symp_cardiac', 0) * df.get('diabetes_burden', 0)

    pain_score = df['pain_score'].fillna(5)
    df['cardiac_x_pain_level'] = df.get('symp_cardiac', 0) * pain_score
    df['trauma_x_pain_level'] = df.get('symp_trauma', 0) * pain_score
    df['cardio_x_age'] = df.get('cardio_burden', 0) * age / 10
    df['resp_x_age'] = df.get('resp_burden', 0) * age / 10

    df['cc_high_risk_no_cardio'] = (df.get('cc_high_risk', 0) & ~df.get('symp_cardiac', 0)).astype(int)
    df['high_risk_text_x_hypoxia'] = df.get('cc_high_risk', 0) * df.get('hypoxia', 0)
    df['cardio_text_x_tachy'] = df.get('symp_cardiac', 0) * df.get('tachycardia', 0)
    df['neuro_text_x_hypertension'] = df.get('symp_neurologic', 0) * df.get('hypertension_flag', 0)

    # ========= 6. TRANSFORMED VITALS =========
    print("  6. Transformed vital signs...")
    df['hr_squared'] = (hr / 100) ** 2
    df['rr_squared'] = (rr / 20) ** 2
    df['sbp_log'] = np.log(sbp.clip(50, 260))
    df['hr_rr_ratio'] = hr / rr.replace(0, np.nan)
    df['hr_rr_ratio'] = df['hr_rr_ratio'].clip(0, 20)

    df['temp_x_hr'] = temp * hr / 1000
    df['temp_x_rr'] = temp * rr / 100
    df['spo2_x_hr'] = spo2 * hr / 1000

    df['pain_x_age'] = pain_score * age / 100

    df['comorbidity_x_age_product'] = df.get('comorbidity_count', 0) * age / 100
    df['abnormal_vitals_x_age'] = df.get('abnormal_vital_count', 0) * age / 100

    n_new = len(df.columns) - n_orig
    print(f"  Added {n_new} enhanced features (total: {len(df.columns)})")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/nhamcs_data_2018_22.csv", low_memory=False, nrows=1000)
    df = add_extra_features(df)
    print(f"\nNew columns: {[c for c in df.columns if c not in ['chief_complaint_text']][:20]}...")
