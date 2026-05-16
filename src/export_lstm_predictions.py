from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from evaluate import save_predictions_csv
from feature_engineering import load_hourly_data
from train_lstm import FEATURE_COLS, make_dataset
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_project_dirs


def export_lstm_predictions(
    model_path: Path = MODELS_DIR / "lstm.keras",
    window_hours: int = 24 * 7,
    horizon_hours: int = 24 * 6,
    sequence_stride: int = 6,
    batch_size: int = 64,
) -> None:
    ensure_project_dirs()
    model = tf.keras.models.load_model(model_path)

    df = load_hourly_data(DATA_DIR / "merged.csv")
    df["Q_future_144"] = df["Q"].shift(-horizon_hours)
    df = df.dropna(subset=["Q_future_144"]).reset_index(drop=True)

    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(df[FEATURE_COLS].iloc[:train_end])
    target_scaler.fit(df[["Q_future_144"]].iloc[:train_end])

    scaled_features = feature_scaler.transform(df[FEATURE_COLS]).astype("float32")
    scaled_targets = target_scaler.transform(df[["Q_future_144"]]).flatten().astype("float32")

    test_dataset = make_dataset(
        scaled_features[val_end:],
        scaled_targets[val_end:],
        window_hours,
        sequence_stride,
        batch_size,
    )

    y_true_scaled = []
    y_pred_scaled = []
    for batch_x, batch_y in test_dataset:
        y_true_scaled.append(batch_y.numpy())
        y_pred_scaled.append(model.predict(batch_x, verbose=0))

    y_true = target_scaler.inverse_transform(np.concatenate(y_true_scaled).reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(np.concatenate(y_pred_scaled).reshape(-1, 1)).flatten()

    save_predictions_csv(y_true, y_pred, RESULTS_DIR / "lstm_predictions.csv")
    print(f"Saved LSTM predictions to {RESULTS_DIR / 'lstm_predictions.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LSTM predictions from a saved Keras model.")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "lstm.keras")
    parser.add_argument("--window-hours", type=int, default=24 * 7)
    parser.add_argument("--horizon-hours", type=int, default=24 * 6)
    parser.add_argument("--sequence-stride", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_lstm_predictions(
        model_path=args.model_path,
        window_hours=args.window_hours,
        horizon_hours=args.horizon_hours,
        sequence_stride=args.sequence_stride,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
