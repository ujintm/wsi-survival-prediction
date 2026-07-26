import pandas as pd
import numpy as np
import os

clinical = pd.read_csv('/home/yuz/pathML/clinical.cart.2026-06-08/clinical.tsv', sep='\t')
unique_patients = clinical['cases.case_id'].nunique()
print(clinical['demographic.vital_status'].value_counts(dropna=False))

survival_cols = [
    'cases.submitter_id',
    'demographic.days_to_death',
    'demographic.vital_status', 
    'diagnoses.days_to_last_follow_up',
    'demographic.age_at_index',
    'demographic.race',
    'demographic.gender',
    'diagnoses.ajcc_pathologic_stage',
    'diagnoses.tumor_grade',
    'diagnoses.primary_diagnosis'
]


df_survival = clinical[survival_cols].copy()
df_survival = df_survival.mask(df_survival == "'--")
df_patient = df_survival.groupby('cases.submitter_id').first().reset_index()
df_patient = df_patient.drop(columns=['diagnoses.tumor_grade'])

df_patient['survival_time'] = df_patient['demographic.days_to_death'].fillna(
    df_patient['diagnoses.days_to_last_follow_up'])
df_patient['event'] = (df_patient['demographic.vital_status'] == 'Dead').astype(int)
df_patient['survival_time'] = pd.to_numeric(df_patient['survival_time'])



from sksurv.util import Surv

y = Surv.from_arrays(
    event=df_patient['event'].astype(bool),
    time=df_patient['survival_time']
)

stage_map = {
    'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
    'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
    'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
    'Stage IV': 4
}
df_patient['stage_numeric'] = df_patient['diagnoses.ajcc_pathologic_stage'].map(stage_map)

# Gender
df_patient['gender_numeric'] = (df_patient['demographic.gender'] == 'male').astype(int)
# Race - One-hot encoding
df_patient = pd.get_dummies(df_patient, columns=['demographic.race'], prefix='race', drop_first=True)


# Primary diagnosis를 3개 카테고리로 단순화
def simplify_diagnosis(dx):
    if pd.isna(dx) or dx == 'Not Reported':
        return 'Other'
    elif 'duct' in dx.lower():
        return 'Ductal'
    elif 'Lobular' in dx:
        return 'Lobular'
    else:
        return 'Other'

df_patient['diagnosis_simplified'] = df_patient['diagnoses.primary_diagnosis'].apply(simplify_diagnosis)
df_patient = pd.get_dummies(df_patient, columns=['diagnosis_simplified'], prefix='dx', drop_first=True)

feature_cols = ['demographic.age_at_index', 'stage_numeric', 'gender_numeric']
race_cols = [col for col in df_patient.columns if col.startswith('race_')]
dx_cols = [col for col in df_patient.columns if col.startswith('dx_')]

feature_cols = feature_cols + race_cols + dx_cols
final_cols = ['cases.submitter_id', 'survival_time', 'event'] + feature_cols

# debug
print("df_patient 전체:", df_patient["cases.submitter_id"].nunique())
print("dropna 전:", df_patient[final_cols]["cases.submitter_id"].nunique())


df_final = df_patient[final_cols].dropna()

# debug
print("dropna 후:", df_patient[final_cols].dropna()["cases.submitter_id"].nunique())

print(
    df_final[['cases.submitter_id','event']]
    .drop_duplicates()
    ['event']
    .value_counts()
)


print(
    sorted(df_patient["diagnoses.ajcc_pathologic_stage"]
           .dropna()
           .unique())
)


# WSI feature 매칭
feature_dir = '/home/yuz/wsi-survival/features_uni/pt_files'
feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]

path_data = []
for f in feature_files:
    # 환자 ID 추출 (TCGA-A2-A0T1)
    patient_id = '-'.join(f.split('-')[:3])
    # full_path = os.path.abspath(os.path.join(feature_dir, f))
    # path_data.append({'cases.submitter_id': patient_id, '': full_path})
    path_data.append({
    "cases.submitter_id": patient_id,
    "feature_path": f})

df_path = pd.DataFrame(path_data)
df_final = df_final.merge(df_path, on='cases.submitter_id', how='inner')

df_final.to_csv('/home/yuz/wsi-survival/clinical_survival_processed_final.csv', index=False)
print(f"최종 병합 완료: {len(df_final)}")
print(f"{df_final['cases.submitter_id'].nunique()}명")
print(f"컬럼 구성: {df_final.columns.tolist()}")
print(
    df_final[['cases.submitter_id','event']]
    .drop_duplicates()
    ['event']
    .value_counts()
)