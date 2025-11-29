import pandas as pd
import numpy as np

def load_sp500(csv_path="data/raw/sp500.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("date")
    prices = df["close"].values.astype(np.float32)
    return prices

def compute_gross_returns(prices):
    # Simple gross returns: R[t] = price[t] / price[t-1]
    R = prices[1:] / prices[:-1]
    return R.astype(np.float32)

def save_returns(returns, out_path="data/processed/returns.npy"):
    np.save(out_path, returns)

if __name__ == "__main__":
    prices = load_sp500()
    returns = compute_gross_returns(prices)
    save_returns(returns)
    print("Processed returns saved:", returns.shape)