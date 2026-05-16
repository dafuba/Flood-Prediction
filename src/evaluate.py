from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import RESULTS_DIR, load_metrics, save_json


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - numerator / denominator)


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual", linewidth=1.8)
    plt.plot(y_pred, label="Predicted", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Evaluation sample")
    plt.ylabel("Discharge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_predictions_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    index_values: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "sample": np.arange(len(y_true)) if index_values is None else index_values,
            "actual": y_true,
            "predicted": y_pred,
            "residual": y_true - y_pred,
        }
    )
    predictions.to_csv(output_path, index=False)


def save_model_comparison(metrics_path: Path = RESULTS_DIR / "metrics.json") -> None:
    metrics = load_metrics(metrics_path)
    if not metrics:
        print(f"No metrics found at {metrics_path}")
        return

    rows = []
    for model_name, model_metrics in metrics.items():
        row = {"model": model_name}
        row.update(model_metrics)
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("rmse")
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(comparison["model"], comparison["rmse"])
    plt.title("Model Comparison by RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    save_json(metrics, metrics_path)
    print(comparison.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize saved model metrics.")
    parser.add_argument("--metrics", type=Path, default=RESULTS_DIR / "metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_model_comparison(args.metrics)


if __name__ == "__main__":
    main()
