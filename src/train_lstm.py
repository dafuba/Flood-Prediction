from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from evaluate import nash_sutcliffe_efficiency, regression_metrics, save_prediction_plot, save_predictions_csv
from feature_engineering import load_hourly_data
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_project_dirs, set_global_seed, update_metrics


FEATURE_COLS = ["pr", "Q"]


def make_dataset(
    data: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
    sequence_stride: int,
    batch_size: int,
) -> tf.data.Dataset:
    return tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=targets,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        shuffle=False,
        batch_size=batch_size,
    )


def train_lstm(
    epochs: int = 20,
    batch_size: int = 64,
    window_hours: int = 24 * 7,
    horizon_hours: int = 24 * 6,
    sequence_stride: int = 6,
    random_state: int = 42,
) -> dict[str, float]:
    set_global_seed(random_state)
    ensure_project_dirs()

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

    train_dataset = make_dataset(
        scaled_features[:train_end],
        scaled_targets[:train_end],
        window_hours,
        sequence_stride,
        batch_size,
    )
    val_dataset = make_dataset(
        scaled_features[train_end:val_end],
        scaled_targets[train_end:val_end],
        window_hours,
        sequence_stride,
        batch_size,
    )
    test_dataset = make_dataset(
        scaled_features[val_end:],
        scaled_targets[val_end:],
        window_hours,
        sequence_stride,
        batch_size,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_hours, len(FEATURE_COLS))),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_true_scaled = []
    y_pred_scaled = []
    for batch_x, batch_y in test_dataset:
        y_true_scaled.append(batch_y.numpy())
        y_pred_scaled.append(model.predict(batch_x, verbose=0))

    y_true = target_scaler.inverse_transform(np.concatenate(y_true_scaled).reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(np.concatenate(y_pred_scaled).reshape(-1, 1)).flatten()

    metrics = regression_metrics(y_true, y_pred)
    metrics["nse"] = nash_sutcliffe_efficiency(y_true, y_pred)
    metrics["train_rows"] = int(train_end)
    metrics["validation_rows"] = int(val_end - train_end)
    metrics["test_rows"] = int(len(df) - val_end)
    metrics["epochs_trained"] = int(len(history.history["loss"]))
    metrics["window_hours"] = int(window_hours)
    metrics["horizon_hours"] = int(horizon_hours)

    model.save(MODELS_DIR / "lstm.keras")
    save_prediction_plot(
        y_true,
        y_pred,
        "LSTM - 6-Day Ahead Hourly Discharge",
        RESULTS_DIR / "lstm_actual_vs_predicted.png",
    )
    save_predictions_csv(
        y_true,
        y_pred,
        RESULTS_DIR / "lstm_predictions.csv",
    )
    update_metrics("lstm", metrics, RESULTS_DIR / "metrics.json")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM flood forecasting model.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-hours", type=int, default=24 * 7)
    parser.add_argument("--horizon-hours", type=int, default=24 * 6)
    parser.add_argument("--sequence-stride", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_lstm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_hours=args.window_hours,
        horizon_hours=args.horizon_hours,
        sequence_stride=args.sequence_stride,
        random_state=args.random_state,
    )
    print("LSTM metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
