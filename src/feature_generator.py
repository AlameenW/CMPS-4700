import pandas as pd
import numpy as np


# Feature generation
def generate_features(df):

    """
    Creates engineered features and target labels from raw dataset.
    """

    # Score column selection (safe filter)
    score_columns = [f"Q{i}" for i in range(9, 21)]
    score_columns = [col for col in score_columns if col in df.columns]

    # Ensure numeric and clean missing values
    for column in score_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[score_columns] = df[score_columns].fillna(df[score_columns].median())

    # Features - Mental Health Score
    df["MH_Score"] = df[score_columns].sum(axis=1)
    # #

    # Threshold computation (quantiles)
    low_threshold = df["MH_Score"].quantile(0.33)
    high_threshold = df["MH_Score"].quantile(0.66)
    # #

    # Vector label assignment
    conditions = [
        df["MH_Score"] <= low_threshold,
        df["MH_Score"] <= high_threshold
    ]

    choices = [
        "Low Risk",
        "Moderate Risk"
    ]

    df["TARGET"] = pd.Series(
        np.select(conditions, choices, default="High Risk")
    )

   # Final data cleanup
    df = df.drop(columns=["MH_Score"])


    # Split features and labels
    feature_matrix = df.drop(columns=["TARGET"])
    label_vector = df["TARGET"]
    

    return feature_matrix, label_vector
    # #
# #
