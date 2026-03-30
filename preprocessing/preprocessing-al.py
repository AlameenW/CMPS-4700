"""
MindMetrics — CMPS 4700
Assignment 1: Preprocessing Code
Template Format: SAMPLE_ID | TARGET | A1 | A2 | ... | An
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load Data 
df = pd.read_csv('smmh.csv')
# print(f"Raw dataset shape: {df.shape}")
# print(f"Missing values:\n{df.isnull().sum()}")

# STEP 2: Rename Columns 
col_map = {
    '1. What is your age?':                          'Age',
    '2. Gender':                                     'Gender',
    '3. Relationship Status':                        'Relationship_Status',
    '4. Occupation Status':                          'Occupation',
    '5. What type of organizations are you affiliated with?': 'Organization',
    '8. What is the average time you spend on social media every day?': 'Daily_Usage',
    '9. How often do you find yourself using Social media without a specific purpose?':  'Q9',
    '10. How often do you get distracted by Social media when you are busy doing something?': 'Q10',
    "11. Do you feel restless if you haven't used Social media in a while?": 'Q11',
    '12. On a scale of 1 to 5, how easily distracted are you?':             'Q12',
    '13. On a scale of 1 to 5, how much are you bothered by worries?':      'Q13',
    '14. Do you find it difficult to concentrate on things?':                'Q14',
    '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?': 'Q15',
    '16. Following the previous question, how do you feel about these comparisons, generally speaking?': 'Q16',
    '17. How often do you look to seek validation from features of social media?': 'Q17',
    '18. How often do you feel depressed or down?':                          'Q18',
    '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?': 'Q19',
    '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': 'Q20',
}
print('RUN STARTS HERE')
df = df.rename(columns=col_map)
df = df.drop(columns=['Timestamp', '6. Do you use social media?',
                       '7. What social media platforms do you commonly use?'])

# STEP 3: Handle Missing Values 
# Organization: 30 missing → fill with 'Unknown'
df['Organization'] = df['Organization'].fillna('Unknown')

# STEP 4: Normalize Gender 
def normalize_gender(g):
    g = str(g).strip().lower()
    if g in ['male', 'm']: return 'Male'
    if g in ['female', 'f']: return 'Female'
    return 'Other'
df['Gender'] = df['Gender'].apply(normalize_gender)

# Normalize organization

# STEP 5: Ordinal Encode Daily Usage 
usage_map = {
    'Less than an Hour': 0, 'Between 1 and 2 hours': 1,
    'Between 2 and 3 hours': 2, 'Between 3 and 4 hours': 3,
    'Between 4 and 5 hours': 4, 'More than 5 hours': 5,
}
df['Daily_Usage'] = df['Daily_Usage'].map(usage_map)
# STEP 6: Build Target Variable 
# Composite mental health score from Q9–Q20 (12 Likert items)
score_cols = [f'Q{i}' for i in range(9, 21)]
df['MH_Score'] = df[score_cols].sum(axis=1)

def risk_tier(s):
    if s <= 25:  return 'Low Risk'
    elif s <= 40: return 'Moderate Risk'
    else:         return 'High Risk'

df['TARGET'] = df['MH_Score'].apply(risk_tier)
df = df.drop(columns=['MH_Score'])
print(f"\nClass distribution:\n{df['TARGET'].value_counts()}")
# STEP 7: One-Hot Encode Categoricals 
cat_cols = ['Gender', 'Relationship_Status', 'Occupation', 'Organization']
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# STEP 8: Label Encode Target 
le = LabelEncoder()
df['TARGET'] = le.fit_transform(df['TARGET'])

# Classes: 0=High Risk, 1=Low Risk, 2=Moderate Risk

# STEP 9: Min-Max Normalize Numeric Features 
num_cols = score_cols + ['Age', 'Daily_Usage']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# STEP 10: Reformat to Template (SAMPLE_ID, TARGET, A1, A2...) 
feature_cols = [c for c in df.columns if c != 'TARGET']
df_out = df[['TARGET'] + feature_cols].copy()
df_out.index.name = 'SAMPLE_ID'
df_out = df_out.reset_index()
df_out['SAMPLE_ID'] = df_out['SAMPLE_ID'] + 1
a_col_map = {fc: f'A{i+1}' for i, fc in enumerate(feature_cols)}
df_out = df_out.rename(columns=a_col_map)
print(f"\nPreprocessed template shape: {df_out.shape}")
print(df_out.head())

# STEP 11: SMOTE — Balance Classes 
X = df_out.drop(columns=['SAMPLE_ID', 'TARGET'])
y = df_out['TARGET']
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)
# print(f"\nAfter SMOTE: {pd.Series(y_bal).value_counts().to_dict()}")

# STEP 12: Train/Val/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_bal, y_bal, test_size=0.30, random_state=42, stratify=y_bal)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# STEP 13: Save Outputs
df_out.to_excel('MindMetrics_preprocessed.xlsx', index=False, sheet_name='Preprocessed')
print("Saved preprocessed Excel file.")