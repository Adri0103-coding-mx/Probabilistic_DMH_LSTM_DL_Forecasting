def build_xy(arr, lookback, horizon):
    """
    Constructs input–output sequences for direct multi-horizon forecasting
    using a sliding window approach.

    Parameters
    ----------
    arr : array-like, shape (T, ...)
        Preprocessed time series data.
    lookback : int
        Length of the input window.
    horizon : int
        Forecast horizon (number of steps ahead).

    Returns
    -------
    X : list
        List of input sequences of length `lookback`.
    y : list
        List of corresponding output sequences of length `horizon`.
    """
    X, y = [], []
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i:i + lookback])
        y.append(arr[i + lookback:i + lookback + horizon])
    return X, y
