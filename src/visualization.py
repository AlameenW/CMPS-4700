import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve


# Dataset distribution
def plot_dataset_distribution(y_train, y_val, y_test):

    train_counts = pd.Series(y_train).value_counts().sort_index()
    val_counts = pd.Series(y_val).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()

    plt.figure()

    plt.plot(train_counts, marker="o", label="Train")
    plt.plot(val_counts, marker="o", label="Validation")
    plt.plot(test_counts, marker="o", label="Test")

    plt.title("Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    plt.show()
# #


# General learning curve SVM, DT, KNN
def plot_learning_curve(model, X, y, model_name="Model"):

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()

    plt.plot(train_sizes, train_mean, marker="o", label="Train Accuracy")
    plt.plot(train_sizes, val_mean, marker="o", label="Validation Accuracy")

    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.show()
# #


# ANN Epoch-error curve
def plot_ann_training_curve(history, model_name="ANN"):

    plt.figure()

    plt.plot(history.history["loss"], label="Train Loss")

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title(f"Epoch-Error Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.show()
# #

# Boxplot (5 random trials)
def plot_model_boxplot(scores, model_name="Model"):

    plt.figure()

    plt.boxplot(scores)

    plt.title(f"Performance Stability (5 Trials) - {model_name}")
    plt.ylabel("Accuracy")
    plt.xticks([1], [model_name])

    plt.tight_layout()
    plt.show()
# #

# Feature Distribution Plot
def plot_feature_distribution(df, feature_list):

    valid_features = [col for col in feature_list if col in df.columns]

    if len(valid_features) == 0:
        print("Skipping feature distribution plot: no valid features found")
        return

    df[valid_features].hist(bins=15, figsize=(10, 6))

    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()
# #

# Target class distribution
def plot_target_distribution(y):

    plt.figure()

    pd.Series(y).value_counts().plot(kind="bar")

    plt.title("Target Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
# #

# Mutual Information Importance
def plot_feature_importance(mi_series, top_n=15):

    top_features = mi_series.sort_values(ascending=False).head(top_n)

    plt.figure()

    plt.barh(top_features.index, top_features.values)

    plt.title(f"Top {top_n} Features (Mutual Information)")
    plt.xlabel("Score")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
# #

# MI vs CORRELATION scatterplot
def plot_mi_vs_correlation(mi_series, corr_series):

    plt.figure()

    plt.scatter(mi_series, corr_series)

    plt.title("Mutual Information vs Correlation")
    plt.xlabel("Mutual Information")
    plt.ylabel("Correlation")

    plt.tight_layout()
    plt.show()
# #

# Correlation distribution
def plot_correlation_distribution(corr_matrix):

    plt.figure()

    corr_vals = corr_matrix.values.flatten()
    corr_vals = corr_vals[~np.isnan(corr_vals)]

    plt.hist(corr_vals, bins=30)

    plt.title("Feature Correlation Distribution")
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
# #

# SMOTE class balance
def plot_smote_distribution(y_before, y_after):

    plt.figure()

    plt.subplot(1,2,1)
    pd.Series(y_before).value_counts().plot(kind="bar")
    plt.title("Before SMOTE")

    plt.subplot(1,2,2)
    pd.Series(y_after).value_counts().plot(kind="bar")
    plt.title("After SMOTE")

    plt.tight_layout()
    plt.show()
# #

# feature sampple check
def plot_feature_sample(df, n=5):

    plt.figure()

    sample = df.head(n)

    plt.table(cellText=sample.values,
              colLabels=sample.columns,
              loc="center")

    plt.axis("off")
    plt.title("Feature Sample Preview")

    plt.tight_layout()
    plt.show()
# #

