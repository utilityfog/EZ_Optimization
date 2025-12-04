import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")


def _to_datetime(df: pd.DataFrame):
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError("No date column found")
    c = date_cols[0]
    df[c] = pd.to_datetime(df[c])
    return df, c

def load_raw_data():
    """
    Load:
      sp500_df: OHLCV daily S&P 500 data
      psavert_df: monthly personal savings rate
      unemploy_df: monthly unemployment
    Resample everything to month end and merge.
    """
    sp500_df = pd.read_csv(os.path.join(RAW_DIR, "sp500_df.csv"))
    psavert_df = pd.read_csv(os.path.join(RAW_DIR, "psavert_df.csv"))
    unemploy_df = pd.read_csv(os.path.join(RAW_DIR, "unemploy_df.csv"))

    # Parse dates
    sp500_df, sp_date = _to_datetime(sp500_df)
    psavert_df, ps_date = _to_datetime(psavert_df)
    unemploy_df, un_date = _to_datetime(unemploy_df)

    # Identify close/ohlcv columns
    close_col = None
    for c in ["SP500_Close", "close", "Close"]:
        if c in sp500_df.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError("Could not find close or SP500_Close column in sp500_df")

    open_col = "open" if "open" in sp500_df.columns else "Open"
    high_col = "high" if "high" in sp500_df.columns else "High"
    low_col = "low" if "low" in sp500_df.columns else "Low"
    vol_col = "volume" if "volume" in sp500_df.columns else "Volume"

    for col in [open_col, high_col, low_col, vol_col]:
        if col not in sp500_df.columns:
            raise ValueError(f"Missing OHLCV column: {col}")

    # Convert S&P daily OHLCV to monthly OHLCV via last trading day
    sp500_df["date_som"] = sp500_df[sp_date].values.astype("datetime64[M]")

    monthly_ohlcv = (
        sp500_df.sort_values(sp_date)
        .groupby("date_som", as_index=False)
        .tail(1)[["date_som", open_col, high_col, low_col, close_col, vol_col]]
        .sort_values("date_som")
        .reset_index(drop=True)
    )

    # Compute monthly returns from close prices
    monthly_ohlcv["SP500_Returns"] = monthly_ohlcv[close_col].pct_change()

    # Macro data
    # personal savings
    if "Personal_Savings_Rate" not in psavert_df.columns:
        candidates = [c for c in psavert_df.columns if "save" in c.lower()]
        psavert_df = psavert_df.rename(
            columns={candidates[0]: "Personal_Savings_Rate"}
        )

    psavert_df["date_som"] = psavert_df[ps_date].values.astype("datetime64[M]")
    psavert_df = psavert_df[["date_som", "Personal_Savings_Rate"]]

    # unemployment
    if "Unemployment" not in unemploy_df.columns:
        candidates = [c for c in unemploy_df.columns if "unemploy" in c.lower()]
        unemploy_df = unemploy_df.rename(columns={candidates[0]: "Unemployment"})

    unemploy_df["date_som"] = unemploy_df[un_date].values.astype("datetime64[M]")
    unemploy_df = unemploy_df[["date_som", "Unemployment"]]

    # Final merged monthly dataset
    df = (
        monthly_ohlcv
        .merge(psavert_df, on="date_som", how="left")
        .merge(unemploy_df, on="date_som", how="left")
        .sort_values("date_som")
        .reset_index(drop=True)
    )

    return df, (open_col, high_col, low_col, close_col, vol_col)

def expanding_normalize(feature_matrix):
    """
    Expanding window z score normalization.
    """
    n, d = feature_matrix.shape
    out = np.zeros((n, d), dtype=np.float32)

    csum = np.cumsum(feature_matrix, axis=0)
    csum_sq = np.cumsum(feature_matrix**2, axis=0)

    for t in range(n):
        count = t + 1
        mean = csum[t] / count
        var = csum_sq[t] / count - mean**2
        std = np.sqrt(np.maximum(var, 1e-8))
        out[t] = (feature_matrix[t] - mean) / std

    return out


def build_processed(
    frac_d: float | None = None,
    max_lag: int | None = None,
    tol: float = 1e-6,
    train_frac: float = 0.8,
):
    """
    Build processed feature and return arrays and do a chronological train test split.

    frac_d, max_lag, tol are accepted for compatibility with older configs
    but are not used in this simplified pipeline.
    """
    os.makedirs(PROC_DIR, exist_ok=True)

    df, (open_col, high_col, low_col, close_col, vol_col) = load_raw_data()

    # drop first row (return NaN)
    df = df.dropna(subset=["SP500_Returns"]).reset_index(drop=True)

    # Features
    # OHLCV + macro
    features_raw = df[
        [
            open_col,
            high_col,
            low_col,
            close_col,
            vol_col,
            "Personal_Savings_Rate",
            "Unemployment",
        ]
    ].astype("float32").values

    features_norm = expanding_normalize(features_raw)

    # Targets
    gross_returns = (1.0 + df["SP500_Returns"].astype("float32").values)

    # Chronological train test split
    n = features_norm.shape[0]
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

    split_idx = int(n * train_frac)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(
            f"Bad split index {split_idx} for n={n}. Check train_frac={train_frac}."
        )

    X_train = features_norm[:split_idx]
    X_test = features_norm[split_idx:]
    y_train = gross_returns[:split_idx]
    y_test = gross_returns[split_idx:]

    # Save
    # Training arrays, for compatibility with train.py ensure_processed
    np.save(os.path.join(PROC_DIR, "features.npy"), X_train)
    np.save(os.path.join(PROC_DIR, "returns.npy"), y_train)

    # Test arrays, for evaluation
    np.save(os.path.join(PROC_DIR, "features_test.npy"), X_test)
    np.save(os.path.join(PROC_DIR, "returns_test.npy"), y_test)

    print("Saved processed features and returns")
    print("  train features shape:", X_train.shape)
    print("  train returns shape:", y_train.shape)
    print("  test features shape:", X_test.shape)
    print("  test returns shape:", y_test.shape)


if __name__ == "__main__":
    build_processed()