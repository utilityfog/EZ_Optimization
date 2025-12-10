"""
ez_runner.py

Bridge between R (reticulate) and the EZ_Optimization Python package.

Typical R usage:

    library(reticulate)
    use_python("/path/to/EZ_Optimization/venv/bin/python", required = TRUE)
    py_config()
    source_python("ez_runner.py")
    ez_out <- run_ez_pipeline(retrain = TRUE, num_episodes = 1)

This file assumes the project layout:

    EZ_Optimization/
      src/
        config.py
        data_preprocessing.py
        train.py
        ...

You normally run training with:

    (venv) python -m src.train

Here we import the same code as a package, so relative imports inside src/*
keep working.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Locate project root and make sure Python can see `src` as a package
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# Imports from your package
# ---------------------------------------------------------------------

try:
    from src.config import Config
    from src.data_preprocessing import PROC_DIR, expanding_normalize
    from src.train import main as train_main
except ImportError as e:
    raise ImportError(
        f"Failed to import from src. "
        f"ROOT seen by ez_runner.py = {ROOT}. "
        f"sys.path[0] = {sys.path[0]}. "
        f"Original error: {e}"
    )


# def _build_from_r_csv_or_die(
#     train_frac: float = 0.8,
# ) -> Tuple[np.ndarray, np.ndarray, list]:
#     """
#     Build features.npy and returns.npy from the R-created CSV, but
#     do it using the same logic as src.data_preprocessing.build_processed
#     other than the choice of features.

#     R is expected to have written
#         data/processed/Total_data_for_python.csv
#     via build_total_data() in R/data_preprocessing.R.

#     Steps:
#       - load CSV
#       - pick feature columns (everything except date and predicting_return)
#       - apply expanding window z-score normalization
#       - compute gross returns = 1 + monthly_return / 100
#       - chronological train/test split with train_frac
#       - save train and test arrays under PROC_DIR
#     """
#     csv_path = os.path.join(PROC_DIR, "Total_data_for_python.csv")
#     print(f"ez_runner: looking for R CSV at {csv_path}")

#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(
#             "Total_data_for_python.csv not found at:\n"
#             f"  {csv_path}\n\n"
#             "You must run build_total_data() in R (R/data_preprocessing.R) "
#             "BEFORE calling run_ez_pipeline(). In your Rmd, that usually means "
#             "executing the chunk:\n\n"
#             "```{r build-data}\n"
#             "Total_data <- build_total_data()\n"
#             "```\n"
#         )

#     df = pd.read_csv(csv_path, parse_dates=["predicting_date"])

#     # Choose features: everything except date and the future target.
#     # This gives the 10 predictors you built in R:
#     # monthly_return, lag_return, GDP_g, Personal_Savings_Rate, Unemployment,
#     # CPI_inflation, FedFunds_lag1, Term_Spread_lag1, BAA_Yield_lag1, VIX_ret_lag1
#     drop_cols = ["predicting_date", "predicting_return"]
#     feature_cols = [c for c in df.columns if c not in drop_cols]

#     features_raw = df[feature_cols].astype("float32").to_numpy()

#     # Use the same expanding window normalization as src.data_preprocessing
#     features_norm = expanding_normalize(features_raw)

#     # Env returns: realized monthly simple return converted to gross,
#     # consistent with build_processed using SP500_Returns.
#     if "monthly_return" in df.columns:
#         r_simple = df["monthly_return"].astype("float32").to_numpy() / 100.0
#     else:
#         r_simple = df["predicting_return"].astype("float32").to_numpy() / 100.0

#     gross_returns = 1.0 + r_simple

#     # Chronological train/test split, same pattern as build_processed
#     n = features_norm.shape[0]
#     if not (0.0 < train_frac < 1.0):
#         raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

#     split_idx = int(n * train_frac)
#     if split_idx <= 0 or split_idx >= n:
#         raise ValueError(
#             f"Bad split index {split_idx} for n={n}. Check train_frac={train_frac}."
#         )

#     X_train = features_norm[:split_idx]
#     X_test = features_norm[split_idx:]
#     y_train = gross_returns[:split_idx]
#     y_test = gross_returns[split_idx:]

#     # Save under PROC_DIR as in src.data_preprocessing.build_processed
#     os.makedirs(PROC_DIR, exist_ok=True)
#     feat_path = os.path.join(PROC_DIR, "features.npy")
#     ret_path = os.path.join(PROC_DIR, "returns.npy")
#     feat_test_path = os.path.join(PROC_DIR, "features_test.npy")
#     ret_test_path = os.path.join(PROC_DIR, "returns_test.npy")

#     np.save(feat_path, X_train)
#     np.save(ret_path, y_train)
#     np.save(feat_test_path, X_test)
#     np.save(ret_test_path, y_test)

#     print("ez_runner: wrote numpy files from R CSV (with expanding normalization)")
#     print("  train features shape:", X_train.shape)
#     print("  train returns shape :", y_train.shape)
#     print("  test features shape :", X_test.shape)
#     print("  test returns shape  :", y_test.shape)
#     print("  feature columns     :", feature_cols)

#     return X_train, y_train, feature_cols

def _build_from_r_csv_or_die(
    train_frac: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build features.npy and returns.npy from the R-created CSV, but
    now prefer the ECON370 40/20/40 split if the column `sample_set`
    is present:

        sample_set = "train"       -> training
        sample_set = "validation"  -> ALSO treated as training for PPO
        sample_set = "test"        -> held-out test set

    If `sample_set` is missing, fall back to a simple chronological
    split using `train_frac` (default 0.8).

    R is expected to have written:
        data/processed/Total_data_for_python.csv
    via build_total_data() in R/data_preprocessing.R.
    """
    csv_path = os.path.join(PROC_DIR, "Total_data_for_python.csv")
    print(f"ez_runner: looking for R CSV at {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Total_data_for_python.csv not found at:\n"
            f"  {csv_path}\n\n"
            "You must run build_total_data() in R (R/data_preprocessing.R) "
            "BEFORE calling run_ez_pipeline()."
        )

    df = pd.read_csv(csv_path, parse_dates=["predicting_date"])
    df = df.sort_values("predicting_date").reset_index(drop=True)

    # Choose features: everything except date, future target, and sample_set tag
    drop_cols = ["predicting_date", "predicting_return", "sample_set"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    features_raw = df[feature_cols].astype("float32").to_numpy()

    # Use the same expanding window normalization as src.data_preprocessing
    features_norm = expanding_normalize(features_raw)

    # Env returns: realized monthly simple return converted to gross,
    # consistent with build_processed using SP500_Returns.
    if "monthly_return" in df.columns:
        r_simple = df["monthly_return"].astype("float32").to_numpy() / 100.0
    else:
        r_simple = df["predicting_return"].astype("float32").to_numpy() / 100.0

    gross_returns = 1.0 + r_simple
    n_total = features_norm.shape[0]

    # ------------------------------------------------------------------
    # ECON370 project split: use sample_set if present
    # ------------------------------------------------------------------
    if "sample_set" in df.columns:
        mask_train = df["sample_set"].isin(["train", "validation"])
        mask_test = df["sample_set"] == "test"

        n_train_raw = int((df["sample_set"] == "train").sum())
        n_val_raw = int((df["sample_set"] == "validation").sum())
        n_test_raw = int((df["sample_set"] == "test").sum())

        print("ez_runner: using ECON370 40/20/40 split from R:")
        print(f"  train rows      : {n_train_raw}")
        print(f"  validation rows : {n_val_raw}")
        print(f"  test rows       : {n_test_raw}")
        print("  PPO training uses train + validation (first 60%),")
        print("  PPO testing uses the last 40% (test).")

    else:
        # Fallback: simple chronological train_frac split (old behavior)
        if not (0.0 < train_frac < 1.0):
            raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

        split_idx = int(n_total * train_frac)
        if split_idx <= 0 or split_idx >= n_total:
            raise ValueError(
                f"Bad split index {split_idx} for n={n_total}. "
                f"Check train_frac={train_frac}."
            )

        idx = np.arange(n_total)
        mask_train = idx < split_idx
        mask_test = ~mask_train

        print("ez_runner: WARNING: `sample_set` not found in CSV,")
        print(f"  falling back to chronological split with train_frac={train_frac}.")
        n_train_raw = mask_train.sum()
        n_val_raw = 0
        n_test_raw = mask_test.sum()

    # Slice normalized features and returns
    X_train = features_norm[mask_train]
    X_test = features_norm[mask_test]
    y_train = gross_returns[mask_train]
    y_test = gross_returns[mask_test]

    # Save under PROC_DIR as in src.data_preprocessing.build_processed
    os.makedirs(PROC_DIR, exist_ok=True)
    feat_path = os.path.join(PROC_DIR, "features.npy")
    ret_path = os.path.join(PROC_DIR, "returns.npy")
    feat_test_path = os.path.join(PROC_DIR, "features_test.npy")
    ret_test_path = os.path.join(PROC_DIR, "returns_test.npy")

    np.save(feat_path, X_train)
    np.save(ret_path, y_train)
    np.save(feat_test_path, X_test)
    np.save(ret_test_path, y_test)

    print("ez_runner: wrote numpy files from R CSV (with expanding normalization)")
    print("  total rows        :", n_total)
    print("  train+val rows    :", X_train.shape[0])
    print("    (train raw      :", n_train_raw, ")")
    print("    (validation raw :", n_val_raw, ")")
    print("  test rows         :", X_test.shape[0])
    print("  feature columns   :", feature_cols)

    return X_train, y_train, feature_cols

def run_ez_pipeline(
    retrain: bool = False,
    num_episodes: int = None,
) -> Dict[str, Any]:
    """
    Entry point called from R via reticulate.

    Parameters
    ----------
    retrain : bool
        If True, re-run src.train.main() to train a fresh model.
        If False, reuse existing weights in data/processed/actor_critic_ez.pt.
    num_episodes : int or None
        If provided, override Config.num_episodes for a short run from R.

    Returns
    -------
    dict
        A plain Python dict that reticulate converts to an R list.
    """
    cfg = Config()

    if num_episodes is not None:
        cfg.num_episodes = int(num_episodes)

    # Force features.npy / returns.npy to be derived from the R CSV,
    # using the same expanding window normalization and chronological
    # split as src.data_preprocessing.build_processed, but with the
    # 10 feature columns chosen in R.
    X_train, y_train, feature_cols = _build_from_r_csv_or_die(train_frac=0.8)

    model_path = os.path.join(PROC_DIR, "actor_critic_ez.pt")

    if retrain or not os.path.exists(model_path):
        print("ez_runner: training via src.train.main()")
        train_main()
    else:
        print(
            f"ez_runner: reusing existing model at {model_path} "
            "(set retrain=True to retrain)"
        )

    out: Dict[str, Any] = {
        "proc_dir": PROC_DIR,
        "model_path": model_path,
        "retrained": retrain,
        "num_episodes": cfg.num_episodes,
        "used_r_csv": True,
        "feature_dim": int(X_train.shape[1]),
        "n_obs_train": int(X_train.shape[0]),
        "feature_cols": feature_cols,
    }

    return out


if __name__ == "__main__":
    info = run_ez_pipeline(retrain=True, num_episodes=1)
    print("run_ez_pipeline returned:")
    print(info)