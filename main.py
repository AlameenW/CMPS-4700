import pandas as pd

from src.data_loader import load_train_data, load_test_data
from src.preprocessing import preprocess
from src.feature_generator import generate_features
from src.split_data import split_data
from src.correlation import select_features
from src.pca import apply_pca
from src.smote_balance import balance_data
from src.export import save_predictions

from src.visualization import (
    plot_dataset_distribution,
    plot_feature_distribution,
    plot_target_distribution,
    plot_smote_distribution,
    plot_model_boxplot,
    plot_learning_curve
)

from classifiers.svm import train as train_svm
from classifiers.svm import predict as predict_svm
from classifiers.svm import save as save_model
from classifiers.svm import load as load_model

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# -------------------------
# CONFIG
# -------------------------
MODE = "train"
MODEL_PATH = "svm_model.pkl"


# -------------------------
# SAFE DATAFRAME CONVERTER
# -------------------------
def ensure_dataframe(X):

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    return X


# -------------------------
# TRAIN PIPELINE
# -------------------------
def train_pipeline():

    print("=== TRAINING MODE ===")

    # -------------------------
    # LOAD + PREPROCESS
    # -------------------------
    df = load_train_data()
    df = preprocess(df)

    # -------------------------
    # FEATURE GENERATION
    # -------------------------
    X, y = generate_features(df)

    # -------------------------
    # TARGET PLOT
    # -------------------------
    plot_target_distribution(y)

    # -------------------------
    # SAFE FEATURE PLOT (FIXED)
    # -------------------------
    X = ensure_dataframe(X)

    safe_features = [c for c in ['Age', 'Daily_Usage'] if c in X.columns]

    if len(safe_features) > 0:
        plot_feature_distribution(X, safe_features)
    else:
        print("Skipping feature distribution plot (no valid columns)")

    print("Target distribution BEFORE split:")
    print(pd.Series(y).value_counts())

    # -------------------------
    # SPLIT
    # -------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # -------------------------
    # DATA DISTRIBUTION PLOT
    # -------------------------
    plot_dataset_distribution(y_train, y_val, y_test)

    # -------------------------
    # FEATURE SELECTION
    # -------------------------
    X_train, X_val, X_test, selected_features = select_features(
        X_train, y_train, X_val, X_test
    )

    print("Selected features:", len(selected_features))

    # -------------------------
    # PCA (SAFE OUTPUT HANDLING)
    # -------------------------
    if X_train.shape[1] > 1:

        X_train, X_val, X_test, _ = apply_pca(X_train, X_val, X_test)

        # FORCE DF AFTER PCA (IMPORTANT FIX)
        X_train = ensure_dataframe(X_train)
        X_val = ensure_dataframe(X_val)
        X_test = ensure_dataframe(X_test)

    else:
        print("PCA skipped")

    # -------------------------
    # SMOTE (TRAIN ONLY)
    # -------------------------
    X_train, y_train = balance_data(X_train, y_train)

    # -------------------------
    # SMOTE PLOT (FIXED INPUT)
    # -------------------------
    plot_smote_distribution(y, y_train)

    print("Final training class count:", len(set(y_train)))

    # -------------------------
    # MODEL TRAINING
    # -------------------------
    model = train_svm(X_train, y_train)

    # -------------------------
    # LEARNING CURVE
    # -------------------------
    plot_learning_curve(model, X_train, y_train, model_name="SVM")

    # -------------------------
    # 5-TRIAL STABILITY TEST
    # -------------------------
    scores = []

    for i in range(5):

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=i,
            stratify=y_train
        )

        m = train_svm(X_tr, y_tr)
        preds = m.predict(X_te)

        scores.append(accuracy_score(y_te, preds))

    # -------------------------
    # BOXPLOT
    # -------------------------
    plot_model_boxplot(scores, model_name="SVM")

    # -------------------------
    # SAVE MODEL
    # -------------------------
    save_model(model, MODEL_PATH)

    print("Model trained and saved successfully.")


# -------------------------
# TEST PIPELINE
# -------------------------
def test_pipeline():

    print("=== TESTING MODE ===")

    df = load_test_data()
    df = preprocess(df)

    X_test, y_test = generate_features(df)

    model = load_model(MODEL_PATH)

    preds = predict_svm(model, X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Test Accuracy: {acc:.4f}")

    save_predictions(y_test, preds)


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":

    if MODE == "train":
        train_pipeline()
    else:
        test_pipeline()