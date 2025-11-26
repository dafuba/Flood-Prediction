📘 README — Flood Level Forecasting (6-Day Ahead)
📌 Overview

This project builds and compares machine-learning models to forecast river discharge 6 days (144 hours) into the future using hourly precipitation and discharge measurements.

Two prediction strategies are implemented:

Random Forest (RF) using daily aggregated features

Long Short-Term Memory (LSTM) using raw hourly sequences

The LSTM significantly outperforms the Random Forest.

📂 Data Pipeline
1. Raw datasets

Two raw .dat files were provided:

Precipitation file → hourly precipitation (pr)

Discharge file → hourly streamflow (Q)

Both include timestamp components:

year, month, day, hour, value

2. Merging

The two datasets were merged on a constructed datetime column:

datetime = pd.to_datetime(year, month, day, hour)


The merged file is saved as:

merged.csv

📊 Preprocessing & Feature Engineering
Daily Aggregation

Each day is converted into a single row with:

24× hourly precipitation values (pr_00 … pr_23)

24× hourly discharge values (Q_00 … Q_23)

Daily totals and statistics:

pr_daily_sum

pr_daily_max

Q_daily_mean

Q_daily_max

6-Day Ahead Target

The goal is to predict daily mean discharge 6 days ahead:

Q_future_6d = Q_daily_mean.shift(-6)


Rows with missing futures are removed.

🌲 Model 1 — Random Forest (Daily Features)

The RF model uses:

daily rainfall totals

daily rainfall max

daily mean discharge

daily max discharge

hourly PR and Q values

engineered lag features

Limitations:
Random Forest cannot learn temporal memory → weaker long-horizon forecasts.

Performance:

Metric	Value
RMSE	1.96
R²	0.647
🧠 Model 2 — LSTM (Hourly Sequences)

The LSTM uses raw hourly data in sequential windows:

Sliding windows of 7 days (168 hours)

Predict the daily mean discharge 6 days later

Features used each hour:

precipitation (pr)

discharge (Q)

Data is scaled properly using train-only fit.

Network architecture:
LSTM(64) → Dropout(0.3)
LSTM(32) → Dropout(0.3)
Dense(16, relu)
Dense(1)


With EarlyStopping and no shuffling.

Performance:
Metric	Value
RMSE	0.734
R²	0.952

Conclusion:
The LSTM captures long-term hydrologic memory and predicts 6-day-ahead discharge with high accuracy.

📈 Model Comparison
Model	RMSE ↓	R² ↑	Notes
Random Forest	1.961	0.647	Overfits, no temporal memory
LSTM	0.734	0.952	Best overall; learns sequence patterns

The LSTM performs much better, especially for medium-range forecasting (144 hours).

📉 Visualization

Both models are plotted against actual discharge:

RF: good seasonal shape but weak peaks

LSTM: smoother but highly accurate trends and timing

LSTM predictions closely follow real hydrological cycles
