import numpy as np
def conformal_interval(
    y_true_cal,
    y_pred_cal,
    y_pred_test,
    alpha=0.1
):
    """
    Construct conformal prediction intervals using calibration residuals.

    Parameters
    ----------
    y_true_cal : np.ndarray (N_cal, O)
        True values in calibration set.
    y_pred_cal : np.ndarray (N_cal, O)
        Point predictions in calibration set.
    y_pred_test : np.ndarray (N_test, O)
        Point predictions for test set.
    alpha : float
        Miscoverage level (e.g., 0.1 for 90% intervals).

    Returns
    -------
    (lower, upper) : tuple of np.ndarray
        Lower and upper conformal bounds, shape (N_test, O).
    """

    residuals = np.abs(y_true_cal - y_pred_cal)
    q = np.quantile(residuals, 1 - alpha, axis=0)

    lower = y_pred_test - q
    upper = y_pred_test + q

    return lower, upper
