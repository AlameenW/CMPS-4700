from imblearn.over_sampling import SMOTE
import pandas as pd


# class branching module (SMOTE)
def balance_data(X_train, y_train):

    """ 
    Applies SMOTE oversampling to balance class distribution in training data.
    """
    
    # Class distribution (before)
    print("Before SMOTE:")
    print(pd.Series(y_train).value_counts())
    # #


    # Safety check (SMOTE requirements)
    unique_class_count = len(set(y_train))
    sample_count = len(y_train)

    if unique_class_count < 2 or sample_count < 10:
        print("SMOTE skipped: Insufficient class diversity or sample size!")
        return X_train, y_train
    # # 
    
    # SMOTE Initialization
    smote_model = SMOTE(random_state=42)
    # #

    # Resampling
    X_resampled, y_resampled = smote_model.fit_resample(X_train, y_train)
    # #

    # Class distribution (After)
    print("After SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    # #


    return X_resampled, y_resampled