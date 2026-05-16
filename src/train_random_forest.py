from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from evaluate import nash_sutcliffe_efficiency, regression_metrics, save_prediction_plot, save_predictions_csv
from feature_engineering import load_hourly_data, make_daily_features, split_features_target
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_project_dirs, set_global_seed, update_metrics


def train_random_forest(test_size: float = 0.2, random_state: int = 42) -> dict[str, float]:
    set_global_seed(random_state)
    ensure_project_dirs()

    hourly = load_hourly_data(DATA_DIR / "merged.csv")
    daily = make_daily_features(hourly, horizon_days=6)
    x, y = split_features_target(daily)

    split_idx = int(len(daily) * (1 - test_size))
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=25,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    persistence_predictions = x_test["Q_daily_mean"].to_numpy()

    rf_metrics = regression_metrics(y_test.to_numpy(), predictions)
    rf_metrics["nse"] = nash_sutcliffe_efficiency(y_test.to_numpy(), predictions)
    rf_metrics["train_rows"] = int(len(x_train))
    rf_metrics["test_rows"] = int(len(x_test))

    baseline_metrics = regression_metrics(y_test.to_numpy(), persistence_predictions)
    baseline_metrics["nse"] = nash_sutcliffe_efficiency(y_test.to_numpy(), persistence_predictions)
    baseline_metrics["train_rows"] = int(len(x_train))
    baseline_metrics["test_rows"] = int(len(x_test))

    with (MODELS_DIR / "random_forest.pkl").open("wb") as file:
        pickle.dump(model, file)

    importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    importances.to_csv(RESULTS_DIR / "random_forest_feature_importance.csv", header=["importance"])

    save_prediction_plot(
        y_test.to_numpy(),
        predictions,
        "Random Forest - 6-Day Ahead Daily Mean Discharge",
        RESULTS_DIR / "rf_actual_vs_predicted.png",
    )
    save_predictions_csv(
        y_test.to_numpy(),
        predictions,
        RESULTS_DIR / "random_forest_predictions.csv",
        index_values=y_test.index.to_numpy(),
    )
    save_predictions_csv(
        y_test.to_numpy(),
        persistence_predictions,
        RESULTS_DIR / "baseline_predictions.csv",
        index_values=y_test.index.to_numpy(),
    )

    update_metrics("persistence_baseline", baseline_metrics, RESULTS_DIR / "metrics.json")
    update_metrics("random_forest", rf_metrics, RESULTS_DIR / "metrics.json")

    return rf_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Random Forest flood forecasting model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_random_forest(test_size=args.test_size, random_state=args.random_state)
    print("Random Forest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
