from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import DATA_DIR


def load_hourly_data(path: Path = DATA_DIR / "merged.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def make_daily_features(hourly: pd.DataFrame, horizon_days: int = 6) -> pd.DataFrame:
    df = hourly.copy()
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    precipitation = df.pivot(index="date", columns="hour", values="pr")
    precipitation.columns = [f"pr_{hour:02d}" for hour in precipitation.columns]

    discharge = df.pivot(index="date", columns="hour", values="Q")
    discharge.columns = [f"Q_{hour:02d}" for hour in discharge.columns]

    daily = pd.concat([precipitation, discharge], axis=1).dropna().reset_index()

    pr_cols = [f"pr_{hour:02d}" for hour in range(24)]
    q_cols = [f"Q_{hour:02d}" for hour in range(24)]

    daily["pr_daily_sum"] = daily[pr_cols].sum(axis=1)
    daily["pr_daily_max"] = daily[pr_cols].max(axis=1)
    daily["Q_daily_mean"] = daily[q_cols].mean(axis=1)
    daily["Q_daily_max"] = daily[q_cols].max(axis=1)
    daily["Q_future_6d"] = daily["Q_daily_mean"].shift(-horizon_days)

    for lag in [1, 2, 3, 6, 7]:
        daily[f"Q_daily_mean_lag_{lag}d"] = daily["Q_daily_mean"].shift(lag)
        daily[f"pr_daily_sum_lag_{lag}d"] = daily["pr_daily_sum"].shift(lag)

    daily["pr_3d_sum"] = daily["pr_daily_sum"].rolling(window=3, min_periods=3).sum()
    daily["pr_7d_sum"] = daily["pr_daily_sum"].rolling(window=7, min_periods=7).sum()
    daily["Q_3d_mean"] = daily["Q_daily_mean"].rolling(window=3, min_periods=3).mean()
    daily["Q_7d_mean"] = daily["Q_daily_mean"].rolling(window=7, min_periods=7).mean()

    return daily.dropna().reset_index(drop=True)


def split_features_target(daily: pd.DataFrame, target_col: str = "Q_future_6d") -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [column for column in daily.columns if column not in ["date", target_col]]
    return daily[feature_cols], daily[target_col]
