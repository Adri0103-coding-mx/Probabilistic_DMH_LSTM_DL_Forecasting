import numpy as np
import pandas as pd
import copy


def backtesting_multistep(
    series_std: pd.DataFrame,
    scaler,
    base_model,
    train_fn,
    predict_fn,
    *,
    L: int,
    H: int,
    stride: int,
    label_cols,
    idx_start: int,
    retrain: bool = True,
    train_opts=None
):
    """
    Walk-forward backtesting with direct multi-horizon forecasting.

    At each forecast origin t0:
      1) Uses only past information [t0-L, t0) to avoid look-ahead bias.
      2) Optionally retrains the model on all data available up to t0.
      3) Generates a direct H-step-ahead forecast.
      4) Inverse-transforms predictions for target variables only.

    Parameters
    ----------
    series_std : pd.DataFrame
        Standardized full dataset (T, F) with DatetimeIndex.
    scaler : object
        Scaler fitted on training data (used to inverse-transform labels).
    base_model : nn.Module
        Base model to clone for each retraining window.
    train_fn : callable
        Training wrapper: train_fn(model, X, Y, opts) -> trained model.
    predict_fn : callable
        Inference function: predict_fn(model, X) -> (1, H, O).
    L : int
        Lookback window length.
    H : int
        Forecast horizon.
    stride : int
        Step between forecast origins.
    label_cols : iterable of int
        Column indices corresponding to target variables.
    idx_start : int
        First forecast origin index.
    retrain : bool, optional
        Whether to retrain the model at each window.
    train_opts : dict, optional
        Training options passed to train_fn.

    Returns
    -------
    dict with:
        - Y_pred_win : np.ndarray (Nw, H, O)
        - Y_true_win : np.ndarray (Nw, H, O)
        - dates_win  : np.ndarray (Nw, H)
    """

    if train_opts is None:
        train_opts = {}

    idx = series_std.index
    Z = series_std.values.astype(float)
    T, F = Z.shape
    label_cols = list(label_cols)
    O = len(label_cols)

    Y_pred_win, Y_true_win, dates_win = [], [], []

    for t0 in range(idx_start, T - H + 1, stride):

        # ----- optional retraining -----
        if retrain:
            Xtr, Ytr = [], []
            for i in range(L, t0 - H + 1):
                Xtr.append(Z[i - L:i, :])
                Ytr.append(Z[i:i + H, label_cols])

            if len(Xtr) == 0:
                continue

            Xtr = np.asarray(Xtr, dtype=np.float32)
            Ytr = np.asarray(Ytr, dtype=np.float32)

            model = copy.deepcopy(base_model)
            model = train_fn(model, Xtr, Ytr, train_opts)
        else:
            model = base_model

        # ----- inference -----
        X_in = Z[t0 - L:t0, :].reshape(1, L, F).astype(np.float32)
        y_hat_std = predict_fn(model, X_in)
        y_hat_std = np.asarray(y_hat_std).reshape(1, H, O)

        y_true_std = Z[t0:t0 + H, :][:, label_cols].reshape(1, H, O)

        # ----- inverse scaling (labels only) -----
        y_hat = inverse_transform_labels_only(scaler, y_hat_std, label_cols, F)
        y_true = inverse_transform_labels_only(scaler, y_true_std, label_cols, F)

        Y_pred_win.append(y_hat)
        Y_true_win.append(y_true)
        dates_win.append(idx[t0:t0 + H].to_numpy())

    return {
        "Y_pred_win": np.vstack(Y_pred_win),   # (Nw, H, O)
        "Y_true_win": np.vstack(Y_true_win),   # (Nw, H, O)
        "dates_win":  np.stack(dates_win)      # (Nw, H)
    }
