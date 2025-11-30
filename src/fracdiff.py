import numpy as np
import pandas as pd


def fracdiff_weights(d: float, max_lag: int = 128, tol: float = 1e-2):
    """
    Compute fractional differencing weights for parameter d.

    w_0 = 1
    w_k = w_{k-1} * ( (k - 1 - d) / k )

    Stop when either:
      - k reaches max_lag
      - abs(w_k) < tol
    """
    weights = [1.0]
    for k in range(1, max_lag):
        w_prev = weights[-1]
        w_new = w_prev * (k - 1 - d) / k
        if abs(w_new) < tol:
            break
        weights.append(w_new)

    w = np.array(weights, dtype=np.float64)
    return w, len(w)


def fracdiff_series(x: pd.Series, d: float, max_lag: int = 128, tol: float = 1e-2):
    """
    Apply fractional differencing to a 1D pandas Series.

    Returns a numpy array of same length as x and the effective kernel length K.
    First K-1 entries will be NaN.
    """
    x = x.astype(float)
    w, K = fracdiff_weights(d, max_lag=max_lag, tol=tol)

    fx = np.full(len(x), np.nan, dtype=np.float64)

    for t in range(K - 1, len(x)):
        window = x.iloc[t - K + 1 : t + 1].values
        fx[t] = np.dot(w[::-1], window)

    return fx, K