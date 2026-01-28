# src/utils/scaling.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def estandarizar_train_val_test(train_df, val_df, test_df):
    """
    Fit a scaler on the training set and apply the same transformation
    to validation and test sets.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.

    Returns:
        train_scaled (pd.DataFrame): Scaled training data.
        val_scaled (pd.DataFrame): Scaled validation data.
        test_scaled (pd.DataFrame): Scaled test data.
        scaler (StandardScaler): Fitted scaler.
    """
    scaler = StandardScaler()

    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index
    )

    val_scaled = pd.DataFrame(
        scaler.transform(val_df),
        columns=val_df.columns,
        index=val_df.index
    )

    test_scaled = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index
    )

    return train_scaled, val_scaled, test_scaled, scaler


def inverse_transform_labels_only(scaler, X_scaled, label_cols, total_features=None):
    """
    Inverse-transform ONLY the target columns using a scaler fitted
    on the full feature space.

    Supports arrays of shape (N, O) or (B, H, O).

    Args:
        scaler: Fitted scaler with inverse_transform method.
        X_scaled (np.ndarray): Scaled target values.
        label_cols (Iterable[int]): Indices of target columns in full feature space.
        total_features (int, optional): Total number of features F used to fit the scaler.

    Returns:
        np.ndarray: Target values in original scale, same shape as X_scaled.
    """
    arr = np.asarray(X_scaled, dtype=np.float32)
    squeeze_back = False

    if arr.ndim == 2:          # (N, O)
        arr = arr[None, ...]  # (1, N, O)
        squeeze_back = True

    if arr.ndim != 3:
        raise ValueError(f"X_scaled must be (N,O) or (B,H,O), got {arr.shape}")

    B, H, O = arr.shape

    # Infer total number of features F
    if total_features is None:
        if hasattr(scaler, "n_features_in_"):
            F = int(scaler.n_features_in_)
        elif hasattr(scaler, "mean_"):
            F = int(len(scaler.mean_))
        else:
            F = O
    else:
        F = int(total_features)

    label_idx = list(label_cols)

    flat = arr.reshape(B * H, O)
    dummy = np.zeros((flat.shape[0], F), dtype=np.float32)
    dummy[:, label_idx] = flat

    inv = scaler.inverse_transform(dummy)[:, label_idx]
    inv = inv.reshape(B, H, O)

    if squeeze_back:
        inv = inv[0]

    return inv
