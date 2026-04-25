import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


# FEATURE SELECTION MODULE
def select_features(X_train, y_train, X_val, X_test):
    """
    Selects informative features using:
    - Mutual Information
    - Absolute correlation with encoded target
    """

    # DATA QUALITY CHECK (NaNs)
    nan_report = X_train.isna().sum().sort_values(ascending=False).head(10)
    print("NaNs in X_train:\n", nan_report)
    # #

    # CLEAN DATA (avoid mutation inside loops)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    # #

    # SAFETY CHECK (single-class edge case)
    unique_class_count = len(set(y_train))

    if unique_class_count < 2:
        print("⚠️ Feature selection skipped (only one class detected)")
        return X_train, X_val, X_test, X_train.columns
    # #

    # ENCODE TARGET VARIABLE (for numerical scoring)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(y_train)
    # #

    # MUTUAL INFORMATION SCORING
    mutual_info_scores = mutual_info_classif(
        X_train,
        encoded_target,
        random_state=42
    )

    mutual_info_series = pd.Series(mutual_info_scores, index=X_train.columns)
    # #

    # CORRELATION SCORING
    correlation_scores = X_train.corrwith(
        pd.Series(encoded_target, index=X_train.index)
    ).abs()
    # #

    # FEATURE SELECTION MASK
    selection_mask = (
        (mutual_info_series > 0.01) |
        (correlation_scores > 0.05)
    )

    selected_features = X_train.columns[selection_mask]
    # #

    # FALLBACK (prevent empty feature set)
    if len(selected_features) == 0:
        print("⚠️ No features selected; reverting to full feature set")
        selected_features = X_train.columns
    # #


    # RETURN FILTERED DATASETS

    return (
        X_train[selected_features],
        X_val[selected_features],
        X_test[selected_features],
        selected_features
    )
 # #
