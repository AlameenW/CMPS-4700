"""
MindMetrics — Preprocessing Visualizations
Compares raw vs preprocessed dataset to show cleaning, scaling, and class balancing
"""
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data 
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_file = os.path.join(script_dir, 'smmh.csv')
pre_file = os.path.join(script_dir, 'MindMetrics_preprocessed.xlsx')

raw_df = pd.read_csv(raw_file)
pre_df = pd.read_excel(pre_file)

# 2. Rename raw_df columns to match preprocessing 
col_map = { 
    '1. What is your age?': 'Age',
    '2. Gender': 'Gender',
    '3. Relationship Status': 'Relationship_Status',
    '4. Occupation Status': 'Occupation',
    '5. What type of organizations are you affiliated with?': 'Organization',
    '8. What is the average time you spend on social media every day?': 'Daily_Usage',
    '9. How often do you find yourself using Social media without a specific purpose?':  'Q9',
    '10. How often do you get distracted by Social media when you are busy doing something?': 'Q10',
    "11. Do you feel restless if you haven't used Social media in a while?": 'Q11',
    '12. On a scale of 1 to 5, how easily distracted are you?': 'Q12',
    '13. On a scale of 1 to 5, how much are you bothered by worries?': 'Q13',
    '14. Do you find it difficult to concentrate on things?': 'Q14',
    '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?': 'Q15',
    '16. Following the previous question, how do you feel about these comparisons, generally speaking?': 'Q16',
    '17. How often do you look to seek validation from features of social media?': 'Q17',
    '18. How often do you feel depressed or down?': 'Q18',
    '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?': 'Q19',
    '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': 'Q20',
}

raw_df = raw_df.rename(columns=col_map)

# 3. Create target in raw_df
score_cols = [f'Q{i}' for i in range(9,21)]
raw_df['MH_Score'] = raw_df[score_cols].sum(axis=1)

def risk_tier(s):
    if s < -25:   
        return 'Low Risk'
    elif s <= 40: 
        return 'Moderate Risk'
    else:         
        return 'High Risk'

raw_df['TARGET'] = raw_df['MH_Score'].apply(risk_tier)

# 4. Missing values heatmap before preprocessing 
plt.figure(figsize=(12,6))
sns.heatmap(raw_df.isnull(), cbar=False)
plt.title("Missing Values Before Preprocessing")
plt.show()

# 5. Plot class distribution before vs after preprocessing 
plt.figure(figsize=(8,4))
sns.countplot(x='TARGET', data=raw_df)
plt.title("Class Distribution Before Preprocessing")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x='TARGET', data=pre_df)
plt.title("Class Distribution After Preprocessing + SMOTE")
plt.show()

# 6. Numeric feature distributions
numeric_cols = ['Age', 'Daily_Usage'] + score_cols

# Before preprocessing
raw_df[numeric_cols].hist(figsize=(15,10), bins=10)
plt.suptitle("Raw Numeric Feature Distributions")
plt.show()

# After preprocessing (scaled 0–1)
pre_df[numeric_cols].hist(figsize=(15,10), bins=10)
plt.suptitle("Scaled Numeric Feature Distributions")
plt.show()

# 7. Categorical distributions before vs after 

# --- Gender ---
plt.figure(figsize=(8,4))
sns.countplot(x='Gender', data=raw_df)
plt.title("Gender Distribution Before Preprocessing")
plt.show()

gender_cols = [c for c in pre_df.columns if 'Gender_' in c]
pre_df[gender_cols].sum().plot(kind='bar', figsize=(6,4))
plt.title("Gender Distribution After One-Hot Encoding")
plt.ylabel("Count")
plt.show()

# --- Relationship_Status ---
plt.figure(figsize=(8,4))
sns.countplot(x='Relationship_Status', data=raw_df)
plt.title("Relationship Status Before Preprocessing")
plt.show()

rel_cols = [c for c in pre_df.columns if 'Relationship_Status_' in c]
pre_df[rel_cols].sum().plot(kind='bar', figsize=(8,4))
plt.title("Relationship Status Distribution After One-Hot Encoding")
plt.ylabel("Count")
plt.show()

# --- Organization ---
plt.figure(figsize=(12,4))
sns.countplot(x='Organization', data=raw_df)
plt.title("Organization Distribution Before Preprocessing")
plt.xticks(rotation=45, ha='right')
plt.show()

org_cols = [c for c in pre_df.columns if 'Organization_' in c]
pre_df[org_cols].sum().plot(kind='bar', figsize=(12,4))
plt.title("Organization Distribution After One-Hot Encoding")
plt.ylabel("Count")
plt.show()

print("All plots generated successfully!")

'''