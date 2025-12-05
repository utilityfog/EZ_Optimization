import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import load_raw_data


def compute_summary_statistics(df):
    """
    Prints summary stats for all features.
    """
    print("\n===== SUMMARY STATISTICS =====")
    print(df.describe())  # mean, std, min, max, etc.
    
def compute_covariance_matrix(df):
    """
    Prints covariance matrix for numeric features.
    """
    print("\n===== COVARIANCE MATRIX =====")
    print(df.cov())
    
def plot_returns(df):
    """
    Produces a simple line plot of S&P 500 monthly returns.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['SP500_Returns'], label='Monthly Returns')
    plt.title("S&P 500 Monthly Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def run_data_exploration():
    df, (open_col, high_col, low_col, close_col, vol_col) = load_raw_data()
    
    # drop first row (return NaN)
    df = df.dropna(subset=["SP500_Returns"]).reset_index(drop=True)
    
    features_df = df[
        [
            open_col,
            high_col,
            low_col,
            close_col,
            vol_col,
            "Personal_Savings_Rate",
            "Unemployment",
            "SP500_Returns",
        ]
    ].astype("float32")

    # 1. Summary Stats
    compute_summary_statistics(features_df)

    # 2. Covariance Matrix
    compute_covariance_matrix(features_df)

    # 3. Return Plot
    plot_returns(features_df)
    
if __name__ == "__main__":
    run_data_exploration()