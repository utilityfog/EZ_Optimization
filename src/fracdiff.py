import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


class LearnableFracDiff1D(nn.Module):
    """
    Learnable fractional differencing over a fixed-length window.

    Input:  x  [B, K]   (window of raw returns)
    Output: y  [B, K]   (fractionally differenced series over that window)

    We use the standard recursive weights:

        w_0 = 1
        w_k = w_{k-1} * ((k - 1 - d) / k)

    where d is a learned scalar.
    """

    def __init__(self, max_lag: int = 12, init_d: float = 0.4):
        super().__init__()
        self.max_lag = max_lag
        # d is unconstrained; in practice it will learn something near init_d
        self.d_param = nn.Parameter(torch.tensor(float(init_d), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K] (K should equal self.max_lag in our setup)

        Returns:
          y: [B, K] fractional diff of x
        """
        if x.dim() != 2:
            raise ValueError(f"LearnableFracDiff1D expects [B, K] got {x.shape}")

        B, K = x.shape
        K_use = min(K, self.max_lag)
        # use last K_use entries if K > max_lag
        x_use = x[:, -K_use:]

        d = torch.clamp(self.d_param, -0.9, 1.0)

        # compute weights w_0..w_{K_use-1} on the correct device/dtype
        weights = []
        w_prev = torch.ones(1, device=x.device, dtype=x.dtype)
        for k in range(K_use):
            if k == 0:
                w_k = w_prev
            else:
                w_k = w_prev * ((k - 1 - d) / k)
            weights.append(w_k)
            w_prev = w_k

        w = torch.cat(weights, dim=0)  # [K_use]

        # y_t = sum_{j=0}^t w_j * x_{t-j}
        y = torch.zeros(B, K_use, device=x.device, dtype=x.dtype)

        for t in range(K_use):
            coeffs = w[: t + 1]           # [t+1]
            x_slice = x_use[:, : t + 1]   # [B, t+1], indices 0..t
            # reverse to align t-j
            y[:, t] = (x_slice.flip(dims=[1]) * coeffs.view(1, -1)).sum(dim=1)

        std = y.std(dim=1, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        y = (y - y.mean(dim=1, keepdim=True)) / std

        return y