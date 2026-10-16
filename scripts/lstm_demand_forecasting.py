
# File: scripts/lstm_demand_forecasting.py

from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "daily_sales_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "lstm_demand_model.keras"
EVALUATION_FILE = OUTPUT_DIR / "lstm_forecast_evaluation.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "lstm_test_predictions.csv"
FUTURE_FILE = OUTPUT_DIR / "lstm_revenue_forecast_90_days.csv"

LOOKBACK = 30
TEST_RATIO = 0.20
FORECAST_DAYS = 90
EPOCHS = 50
BATCH_SIZE = 32

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# --------------------------------------------------
# 2. Load and prepare daily revenue data
# --------------------------------------------------

print("\n[1/7] Loading daily sales data...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run the existing demand forecasting preparation first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {"Date", "DailyRevenue"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["DailyRevenue"] = pd.to_numeric(
    df["DailyRevenue"], errors="coerce"
)

df = df.dropna(subset=["Date", "DailyRevenue"])
df = df.sort_values("Date")

# Aggregate duplicate dates, if any.
daily = (
    df.groupby("Date", as_index=False)["DailyRevenue"]
    .sum()
    .sort_values("Date")
)

# Create a continuous calendar-day series.
# Missing dates are treated as zero-sales days.
daily = daily.set_index("Date").asfreq("D", fill_value=0)
daily["DailyRevenue"] = daily["DailyRevenue"].fillna(0)

values = daily["DailyRevenue"].to_numpy(dtype=np.float32)
dates = daily.index

if len(values) <= LOOKBACK + 20:
    raise ValueError(
        f"Not enough daily records ({len(values)}) for "
        f"a {LOOKBACK}-day LSTM window and evaluation."
    )

if not np.isfinite(values).all():
    raise ValueError("DailyRevenue contains invalid numeric values.")

print(f"Daily records: {len(values):,}")
print(f"Date range: {dates.min().date()} to {dates.max().date()}")
print(f"Lookback window: {LOOKBACK} days")


# --------------------------------------------------
# 3. Chronological train/test split
# --------------------------------------------------

print("\n[2/7] Creating chronological train/test split...")

split_index = int(len(values) * (1 - TEST_RATIO))

if split_index <= LOOKBACK:
    raise ValueError("Training period is too short for the lookback window.")

train_values = values[:split_index]

# Fit the evaluation scaler only on training data.
evaluation_scaler = MinMaxScaler()
evaluation_scaler.fit(train_values.reshape(-1, 1))

scaled_all = evaluation_scaler.transform(
    values.reshape(-1, 1)
).flatten()

X_train, y_train = [], []

# Training sequences target only training-period observations.
for i in range(LOOKBACK, split_index):
    X_train.append(scaled_all[i - LOOKBACK:i])
    y_train.append(scaled_all[i])

X_train = np.asarray(X_train, dtype=np.float32).reshape(
    -1, LOOKBACK, 1
)
y_train = np.asarray(y_train, dtype=np.float32)

# Test sequences use the preceding 30 observed days.
X_test, y_test = [], []

for i in range(split_index, len(values)):
    X_test.append(scaled_all[i - LOOKBACK:i])
    y_test.append(values[i])

X_test = np.asarray(X_test, dtype=np.float32).reshape(
    -1, LOOKBACK, 1
)
y_test = np.asarray(y_test, dtype=np.float32)

print(f"Training sequences: {len(X_train):,}")
print(f"Testing sequences: {len(X_test):,}")


# --------------------------------------------------
# 4. Build and train LSTM model
# --------------------------------------------------

print("\n[3/7] Training LSTM model...")

def build_lstm():
    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )
    return model


evaluation_model = build_lstm()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = evaluation_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 5. Evaluate LSTM against a naive baseline
# --------------------------------------------------

print("\n[4/7] Evaluating model on the held-out test period...")

scaled_predictions = evaluation_model.predict(
    X_test, verbose=0
).reshape(-1, 1)

lstm_predictions = evaluation_scaler.inverse_transform(
    scaled_predictions
).flatten()

# Naive forecast: today's prediction equals yesterday's actual revenue.
naive_predictions = values[split_index - 1:-1]

mae_lstm = mean_absolute_error(y_test, lstm_predictions)
rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_predictions))

mae_naive = mean_absolute_error(y_test, naive_predictions)
rmse_naive = np.sqrt(mean_squared_error(y_test, naive_predictions))

# MAPE is calculated only for nonzero actual revenue.
nonzero = y_test != 0

if nonzero.any():
    mape_lstm = np.mean(
        np.abs((y_test[nonzero] - lstm_predictions[nonzero])
               / y_test[nonzero])
    ) * 100

    mape_naive = np.mean(
        np.abs((y_test[nonzero] - naive_predictions[nonzero])
               / y_test[nonzero])
    ) * 100
else:
    mape_lstm = np.nan
    mape_naive = np.nan

evaluation = pd.DataFrame([
    {
        "Model": "Naive Baseline",
        "MAE": mae_naive,
        "RMSE": rmse_naive,
        "MAPE": mape_naive
    },
    {
        "Model": "LSTM",
        "MAE": mae_lstm,
        "RMSE": rmse_lstm,
        "MAPE": mape_lstm
    }
])

evaluation.to_csv(EVALUATION_FILE, index=False)

predictions_df = pd.DataFrame({
    "Date": dates[split_index:],
    "ActualRevenue": y_test,
    "LSTMForecast": lstm_predictions,
    "NaiveForecast": naive_predictions
})

predictions_df.to_csv(PREDICTIONS_FILE, index=False)

print("\nHeld-out test results:")
print(evaluation.to_string(index=False))

print(f"\nSaved evaluation: {EVALUATION_FILE}")
print(f"Saved test predictions: {PREDICTIONS_FILE}")


# --------------------------------------------------
# 6. Train final model on all available data
# --------------------------------------------------

print("\n[5/7] Training final model on all historical data...")

# Use a new scaler fitted on all available history for final forecasting.
final_scaler = MinMaxScaler()
scaled_full = final_scaler.fit_transform(
    values.reshape(-1, 1)
).flatten()

X_full, y_full = [], []

for i in range(LOOKBACK, len(scaled_full)):
    X_full.append(scaled_full[i - LOOKBACK:i])
    y_full.append(scaled_full[i])

X_full = np.asarray(X_full, dtype=np.float32).reshape(
    -1, LOOKBACK, 1
)
y_full = np.asarray(y_full, dtype=np.float32)

final_model = build_lstm()

final_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

final_model.fit(
    X_full,
    y_full,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    shuffle=False,
    callbacks=[final_early_stopping],
    verbose=1
)

final_model.save(MODEL_FILE)
print(f"Saved final LSTM model: {MODEL_FILE}")


# --------------------------------------------------
# 7. Generate recursive 90-day forecast
# --------------------------------------------------

print("\n[6/7] Forecasting the next 90 days...")

window = list(scaled_full[-LOOKBACK:])
future_scaled = []

for _ in range(FORECAST_DAYS):
    model_input = np.asarray(
        window[-LOOKBACK:], dtype=np.float32
    ).reshape(1, LOOKBACK, 1)

    next_scaled = float(
        final_model.predict(model_input, verbose=0)[0, 0]
    )

    future_scaled.append(next_scaled)
    window.append(next_scaled)

future_revenue = final_scaler.inverse_transform(
    np.asarray(future_scaled).reshape(-1, 1)
).flatten()

# Revenue cannot be negative.
future_revenue = np.maximum(future_revenue, 0)

future_dates = pd.date_range(
    start=dates.max() + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

future_df = pd.DataFrame({
    "Date": future_dates,
    "ForecastRevenue": future_revenue
})

future_df["Year"] = future_df["Date"].dt.year
future_df["Month"] = future_df["Date"].dt.month
future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek
future_df["IsWeekend"] = (
    future_df["DayOfWeek"] >= 5
).astype(int)

future_df.to_csv(FUTURE_FILE, index=False)

print(f"Saved 90-day forecast: {FUTURE_FILE}")


# --------------------------------------------------
# 8. Final summary
# --------------------------------------------------

print("\n[7/7] STEP 11 COMPLETED")
print("-" * 55)
print(f"Model file:       {MODEL_FILE}")
print(f"Evaluation file:  {EVALUATION_FILE}")
print(f"Test predictions: {PREDICTIONS_FILE}")
print(f"90-day forecast:  {FUTURE_FILE}")
print(f"Test MAE (LSTM):  {mae_lstm:,.2f}")
print(f"Test RMSE (LSTM): {rmse_lstm:,.2f}")
print(f"Test MAPE (LSTM): {mape_lstm:.2f}%")
print("-" * 55)

if mae_lstm < mae_naive:
    print("LSTM achieved lower MAE than the naive baseline.")
else:
    print("LSTM did not beat the naive baseline on MAE.")
    print("Do not assume LSTM is better; compare results before choosing.")
