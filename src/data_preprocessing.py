from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_DIR


def clean_hydro_series(series: pd.Series, variable_name: str, z_thresh: float = 4.0, jump_factor: float = 4.0) -> pd.Series:
    values = series.copy().astype(float)
    values = values.replace([-9999, -8888, -7777, -5555], np.nan)

    if "pr" in variable_name.lower():
        values[values < 0] = np.nan
        values[values > 300] = np.nan

    if "q" in variable_name.lower():
        values[values < 0] = np.nan

    std = values.std()
    if std and not np.isnan(std):
        z_score = (values - values.mean()) / std
        values[np.abs(z_score) > z_thresh] = np.nan

    diff = values.diff().abs()
    median_diff = diff.median()
    if median_diff and not np.isnan(median_diff):
        values[diff > jump_factor * median_diff] = np.nan

    values = values.rolling(window=12, center=True, min_periods=1).median()
    return values.interpolate(method="time")


def load_precipitation(path: Path) -> pd.DataFrame:
    precipitation = pd.read_csv(path, sep=r"\s+", header=None)
    precipitation.columns = ["year", "month", "day", "hour", "pr"]
    return precipitation


def load_discharge(path: Path) -> pd.DataFrame:
    discharge = pd.read_csv(path, sep=",", header=0)
    discharge.columns = ["year", "month", "day", "hour", "Q"]
    return discharge


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["datetime"] = pd.to_datetime(
        dict(year=result["year"], month=result["month"], day=result["day"], hour=result["hour"]),
        errors="coerce",
    )
    return result.dropna(subset=["datetime"])


def preprocess(
    precipitation_path: Path = DATA_DIR / "pr_hourly_DWD_ID1550.dat",
    discharge_path: Path = DATA_DIR / "Q_hourly_ID16425004.dat",
    output_path: Path = DATA_DIR / "merged.csv",
) -> pd.DataFrame:
    precipitation = add_datetime(load_precipitation(precipitation_path))
    discharge = add_datetime(load_discharge(discharge_path))

    precipitation = precipitation.drop_duplicates(subset="datetime", keep="first")
    discharge = discharge.drop_duplicates(subset="datetime", keep="first")

    precipitation = precipitation.sort_values("datetime").set_index("datetime")
    discharge = discharge.sort_values("datetime").set_index("datetime")

    precipitation["pr"] = clean_hydro_series(precipitation["pr"], variable_name="pr")
    discharge["Q"] = clean_hydro_series(discharge["Q"], variable_name="Q")

    merged = pd.merge(
        precipitation.reset_index()[["datetime", "pr"]],
        discharge.reset_index()[["datetime", "Q"]],
        on="datetime",
        how="inner",
    ).sort_values("datetime")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and clean hourly precipitation and discharge files.")
    parser.add_argument("--precipitation", type=Path, default=DATA_DIR / "pr_hourly_DWD_ID1550.dat")
    parser.add_argument("--discharge", type=Path, default=DATA_DIR / "Q_hourly_ID16425004.dat")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "merged.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = preprocess(args.precipitation, args.discharge, args.output)
    print(f"Saved {len(merged):,} hourly rows to {args.output}")


if __name__ == "__main__":
    main()
