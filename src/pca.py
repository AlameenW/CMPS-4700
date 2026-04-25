import numpy as np
from sklearn.decomposition import PCA



# PCA Dimensionality Reduction

def apply_pca(X_train, X_val, X_test, n_components=0.95):

    """
    Applies PCA transformation fitted on training data
    and transforms validation and test sets accordingly.
    """

   
    # Clean INPUT data ( handle INF and NaN)
    
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)


    # Remove low-variance features

    valid_columns_mask = X_train.nunique() > 1
    X_train = X_train.loc[:, valid_columns_mask]

    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]
    

    ## Insufficient features check
    feature_count = X_train.shape[1]

    if feature_count < 2:
        print("⚠️ PCA skipped (insufficient feature dimensions)")
        return X_train, X_val, X_test, None
    # #

    # Train-only PCA fitting
    pca_model = PCA(n_components=n_components)

    X_train_transformed = pca_model.fit_transform(X_train)
    X_val_transformed = pca_model.transform(X_val)
    X_test_transformed = pca_model.transform(X_test)
    
    # #

    # Return transformed Datasets
    return (
        X_train_transformed,
        X_val_transformed,
        X_test_transformed,
        pca_model
    )
    # #
# #
