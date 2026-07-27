"""Shared feature engineering, IO paths, and dataframe helpers for pipeline notebooks.

Scaling policy
--------------
We deliberately do NOT expose a function that fits a scaler on the whole
dataframe. Doing so before walk-forward validation leaks the mean/std of the
test windows into the training windows. The recommended path is:

    from src.feature_engineering import fit_scaler_on_train
    scaler = fit_scaler_on_train(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

`get_feature_columns` exposes the numeric, non-target, non-binary columns
that should participate in scaling. Binary indicators (e.g. ``is_weekend``)
are left untouched on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ENG_DIR = PROJECT_ROOT / "data" / "engineered"

DEFAULT_ASSETS: tuple[str, ...] = ("btc", "ada")

EXCLUDE_PREFIXES: tuple[str, ...] = ("fwd_return", "signal")
EXCLUDE_COLUMNS: frozenset[str] = frozenset({"PriceUSD", "date", "asset", "weekday_name"})
BINARY_COLUMNS: frozenset[str] = frozenset({"is_weekend"})

# Features derivable from the price series alone (plus rolling stats of it).
# Used by the ablation study (VALIDATION_PLAN.md): Model A = price-only,
# Model B = price-only + on-chain. This set is a strict subset of
# ``get_feature_columns`` output, so A's columns are always a subset of B's and
# the two models can be compared on identical rows/folds.
PRICE_ONLY_COLUMNS: frozenset[str] = frozenset({
    "Log_Return",
    "Price_Dist_lag_30d",
    "Price_Dist_lag_90d",
    "Price_Dist_lag_182d",
    "Price_vs_MA7",
    "Volatility_30d",
})


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lags, calendar flags, volatility, and forward-return targets."""
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)

    df["Tx_Intensity"] = df["TxCnt"] / df["AdrActCnt"].replace(0, np.nan)
    df["Velocity_Momentum"] = df["Activity_Velocity"].pct_change(7)
    df["Log_Return"] = np.log(df["PriceUSD"] / df["PriceUSD"].shift(1))

    lag_targets = ["Tx_Intensity", "AdrActCnt", "NVT_Tx_Basis", "Velocity_Momentum"]
    for lag in (1, 3, 7, 14, 30):
        for col in lag_targets:
            df[f"{col}_lag_{lag}d"] = df[col].shift(lag)
    for lag in (30, 90, 182):
        df[f"Price_Dist_lag_{lag}d"] = df["PriceUSD"] / df["PriceUSD"].shift(lag)

    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["Price_vs_MA7"] = df["PriceUSD"] / df["PriceUSD"].rolling(7).mean()
    df["Volatility_30d"] = df["Log_Return"].rolling(30).std()

    for h in (3, 7, 14, 30):
        df[f"fwd_return_{h}d"] = df["PriceUSD"].shift(-h) / df["PriceUSD"] - 1

    # --- Optional features: only computed when source columns are present ---

    # Exchange net flow and 7-day smoothed version (BTC only)
    if {"FlowInExUSD", "FlowOutExUSD"}.issubset(df.columns):
        df["ExchangeNetFlow"] = df["FlowInExUSD"] - df["FlowOutExUSD"]
        df["ExchangeNetFlow_7d_MA"] = df["ExchangeNetFlow"].rolling(7).mean()

    # Puell Multiple: daily issuance / 365-day MA of issuance (BTC only)
    if "IssTotUSD" in df.columns:
        df["Puell_Multiple"] = df["IssTotUSD"] / df["IssTotUSD"].rolling(365).mean()

    # Hash rate deviation from its 30-day trend (BTC only)
    if "HashRate" in df.columns:
        df["HashRate_30d_MA"] = df["HashRate"].rolling(30).mean()
        df["HashRate_vs_MA30"] = df["HashRate"] / df["HashRate_30d_MA"]

    # Staking ratio: fraction of circulating supply delegated (ADA only)
    if "ActiveStakeADA" in df.columns and "SplyCur" in df.columns:
        df["StakingRatio"] = df["ActiveStakeADA"] / df["SplyCur"]

    # CapMrktCurUSD and TxCnt are kept inside ratios (NVT, Velocity, Tx_Intensity)
    # and dropped here to avoid feeding both the ratio and its raw building blocks.
    return df.drop(columns=["CapMrktCurUSD", "TxCnt", "asset"], errors="ignore")


def _tabular_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index for disk storage; first column becomes ``date``."""
    out = df.reset_index()
    first = out.columns[0]
    if first != "date":
        out = out.rename(columns={first: "date"})
    return out


def save_engineered(df: pd.DataFrame, asset: str, *, eng_dir: Path | None = None) -> None:
    """Write engineered data to CSV and Parquet under ``data/engineered``."""
    target = eng_dir or ENG_DIR
    target.mkdir(parents=True, exist_ok=True)
    tab = _tabular_engineered(df)
    stem = target / f"engineered_{asset}"
    tab.to_csv(f"{stem}.csv", index=False)
    tab.to_parquet(f"{stem}.parquet", index=False)
    print(f"Engineered {asset.upper()} saved to:\n  - {stem}.csv\n  - {stem}.parquet")


def _load_single_engineered(
    asset: str,
    *,
    raw_dir: Path,
    eng_dir: Path,
) -> pd.DataFrame:
    eng_csv = eng_dir / f"engineered_{asset}.csv"
    if eng_csv.exists():
        df = pd.read_csv(eng_csv)
    else:
        raw_csv = raw_dir / f"coinmetrics_{asset}.csv"
        df = pd.read_csv(raw_csv)
        engineered = engineer_features(df)
        save_engineered(engineered, asset, eng_dir=eng_dir)
        df = pd.read_csv(eng_csv)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col])
    return df.set_index("date")


def load_engineered_frames(
    assets: Sequence[str] = DEFAULT_ASSETS,
    *,
    raw_dir: Path | None = None,
    eng_dir: Path | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Load engineered CSVs for each asset, rebuilding from raw CSV if missing."""
    raw = raw_dir or RAW_DIR
    eng = eng_dir or ENG_DIR
    return tuple(_load_single_engineered(a, raw_dir=raw, eng_dir=eng) for a in assets)


def classify_signal(return_val: float, threshold: float) -> float:
    """Map a (possibly NaN) forward return into {-1, 0, 1} (or NaN).

    NaN inputs propagate to NaN outputs so that the caller's `dropna()` removes
    rows where the forward return is unknown (typically the last `h` rows of
    the dataframe). Returning 0 for NaN would silently inject synthetic Hold
    labels into the training set.
    """
    if pd.isna(return_val):
        return np.nan
    if return_val > threshold:
        return 1
    if return_val < -threshold:
        return -1
    return 0


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [
        c
        for c in num_cols
        if not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
        and c not in EXCLUDE_COLUMNS
        and c not in BINARY_COLUMNS
    ]


def get_price_only_feature_columns(df: pd.DataFrame) -> list[str]:
    """Price/rolling-price features only (ablation Model A).

    Returns the intersection of ``get_feature_columns(df)`` with
    ``PRICE_ONLY_COLUMNS``, preserving order. Always a subset of the full
    feature set, so Model A (price-only) and Model B (price + on-chain) can be
    trained and scored on identical rows.
    """
    return [c for c in get_feature_columns(df) if c in PRICE_ONLY_COLUMNS]


def get_onchain_feature_columns(df: pd.DataFrame) -> list[str]:
    """On-chain features only: the full feature set minus the price-only subset."""
    return [c for c in get_feature_columns(df) if c not in PRICE_ONLY_COLUMNS]


def fit_scaler_on_train(
    X_train: pd.DataFrame | np.ndarray,
    feature_cols: Sequence[str] | None = None,
) -> StandardScaler:
    """Fit a ``StandardScaler`` on training data only.

    Use this inside walk-forward validation, never on the full dataset.
    """
    scaler = StandardScaler()
    if isinstance(X_train, pd.DataFrame):
        cols = list(feature_cols) if feature_cols is not None else X_train.columns.tolist()
        scaler.fit(X_train[cols].astype(np.float64).values)
    else:
        scaler.fit(np.asarray(X_train, dtype=np.float64))
    return scaler
