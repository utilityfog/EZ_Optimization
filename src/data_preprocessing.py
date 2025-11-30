import os
import numpy as np
import pandas as pd

from .fracdiff import fracdiff_series

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")

def _to_datetime(df: pd.DataFrame):
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError("No date-like column found in dataframe")
    c = date_cols[0]
    df[c] = pd.to_datetime(df[c])
    return df, c


def load_raw_data():
    sp500_df = pd.read_csv(os.path.join(RAW_DIR, "sp500_df.csv"))
    psavert_df = pd.read_csv(os.path.join(RAW_DIR, "psavert_df.csv"))
    unemploy_df = pd.read_csv(os.path.join(RAW_DIR, "unemploy_df.csv"))

    sp500_df, sp_date = _to_datetime(sp500_df)
    psavert_df, ps_date = _to_datetime(psavert_df)
    unemploy_df, un_date = _to_datetime(unemploy_df)

    sp500_df["date_som"] = sp500_df[sp_date].values.astype("datetime64[M]")
    psavert_df["date_som"] = psavert_df[ps_date].values.astype("datetime64[M]")
    unemploy_df["date_som"] = unemploy_df[un_date].values.astype("datetime64[M]")

    # handle SP500_simple_returns if present
    ret_path = os.path.join(RAW_DIR, "SP500_simple_returns.csv")
    if os.path.exists(ret_path):
        SP500_simple_returns = pd.read_csv(ret_path)
        SP500_simple_returns, ret_date = _to_datetime(SP500_simple_returns)
        if "SP500_Returns" not in SP500_simple_returns.columns:
            guess = [
                c
                for c in SP500_simple_returns.columns
                if "return" in c.lower() or "ret" in c.lower()
            ]
            if not guess:
                raise ValueError("Could not find return column in SP500_simple_returns")
            SP500_simple_returns = SP500_simple_returns.rename(
                columns={guess[0]: "SP500_Returns"}
            )
        SP500_simple_returns["date_som"] = SP500_simple_returns[
            ret_date
        ].values.astype("datetime64[M]")
        SP500_simple_returns = SP500_simple_returns[["date_som", "SP500_Returns"]]
    else:
        # build monthly returns from daily close
        close_col = (
            "SP500_Close" if "SP500_Close" in sp500_df.columns else "close"
        )
        if close_col not in sp500_df.columns:
            raise ValueError(
                "Could not find a closing-price column (SP500_Close or close) in sp500_df"
            )

        monthly_price = (
            sp500_df.sort_values(sp_date)
            .groupby("date_som", as_index=False)
            .tail(1)[["date_som", close_col]]
            .sort_values("date_som")
            .reset_index(drop=True)
        )
        monthly_price["SP500_Returns"] = monthly_price[close_col].pct_change()
        SP500_simple_returns = monthly_price[["date_som", "SP500_Returns"]]

    # normalize macro column names
    if "Personal_Savings_Rate" not in psavert_df.columns:
        candidates = [
            c
            for c in psavert_df.columns
            if "save" in c.lower() or "psavert" in c.lower()
        ]
        if not candidates:
            raise ValueError("Personal savings rate column not found in psavert_df")
        psavert_df = psavert_df.rename(
            columns={candidates[0]: "Personal_Savings_Rate"}
        )

    if "Unemployment" not in unemploy_df.columns:
        candidates = [c for c in unemploy_df.columns if "unemploy" in c.lower()]
        if not candidates:
            raise ValueError("Unemployment column not found in unemploy_df")
        unemploy_df = unemploy_df.rename(columns={candidates[0]: "Unemployment"})

    psavert_df = psavert_df[["date_som", "Personal_Savings_Rate"]]
    unemploy_df = unemploy_df[["date_som", "Unemployment"]]

    df_wide = (
        SP500_simple_returns.merge(psavert_df, on="date_som", how="left")
        .merge(unemploy_df, on="date_som", how="left")
        .sort_values("date_som")
        .reset_index(drop=True)
    )

    return df_wide


def build_processed(frac_d: float = 0.4, max_lag: int = 128, tol: float = 1e-2):
    os.makedirs(PROC_DIR, exist_ok=True)

    df = load_raw_data()

    # fractional differencing on each series
    r_fd, K_r = fracdiff_series(df["SP500_Returns"], d=frac_d, max_lag=max_lag, tol=tol)
    s_fd, K_s = fracdiff_series(
        df["Personal_Savings_Rate"], d=frac_d, max_lag=max_lag, tol=tol
    )
    u_fd, K_u = fracdiff_series(
        df["Unemployment"], d=frac_d, max_lag=max_lag, tol=tol
    )

    df["R_fd"] = r_fd
    df["Savings_fd"] = s_fd
    df["Unemp_fd"] = u_fd

    # drop initial rows where any FD is NaN
    df = df.dropna(subset=["R_fd", "Savings_fd", "Unemp_fd"]).reset_index(drop=True)

    # feature matrix and returns
    features = df[["R_fd", "Savings_fd", "Unemp_fd"]].values.astype("float32")
    returns = (1.0 + df["SP500_Returns"].values.astype("float32"))  # gross returns

    # simple z-score normalization per feature
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features - mean) / std

    np.save(os.path.join(PROC_DIR, "features.npy"), features_norm)
    np.save(os.path.join(PROC_DIR, "returns.npy"), returns)
    np.save(os.path.join(PROC_DIR, "feature_mean.npy"), mean)
    np.save(os.path.join(PROC_DIR, "feature_std.npy"), std)

    print("Saved processed features and returns")
    print("features shape:", features_norm.shape)
    print("returns shape:", returns.shape)


if __name__ == "__main__":
    build_processed()