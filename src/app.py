from utils import db_connect
engine = db_connect()

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=2546&path=sales.csv"
CSV_PATH = DATA_DIR / "sales.csv"

MODEL_PATH = MODELS_DIR / "sales_baseline_naive.joblib"
META_PATH = PROCESSED_DIR / "sales_series_meta.joblib"


def detect_date_and_value_columns(df):
    date_candidates = []
    for c in df.columns:
        if df[c].dtype == "object" or "date" in c.lower() or "time" in c.lower() or "month" in c.lower():
            date_candidates.append(c)

    best_date = None
    best_ok = -1

    for c in date_candidates:
        parsed = pd.to_datetime(df[c], errors="coerce")
        ok = int(parsed.notna().sum())
        if ok > best_ok:
            best_ok = ok
            best_date = c

    if best_date is None:
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            ok = int(parsed.notna().sum())
            if ok > best_ok:
                best_ok = ok
                best_date = c

    numeric_cols = [c for c in df.columns if c != best_date]
    best_value = None
    best_numeric_ok = -1

    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        ok = int(s.notna().sum())
        if ok > best_numeric_ok:
            best_numeric_ok = ok
            best_value = c

    return best_date, best_value


def load_series():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.read_csv(CSV_URL)
        df.to_csv(CSV_PATH, index=False)

    date_col, value_col = detect_date_and_value_columns(df)

    ts_df = df[[date_col, value_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df[value_col] = pd.to_numeric(ts_df[value_col], errors="coerce")
    ts_df = ts_df.dropna().sort_values(date_col)

    ts_df = ts_df.set_index(date_col)
    ts_df = ts_df[~ts_df.index.duplicated(keep="last")]
    ts = ts_df[value_col].astype(float)

    return ts, date_col, value_col


def naive_forecast_last_value(train_series, steps):
    last = float(train_series.iloc[-1])
    return np.array([last] * int(steps), dtype=float)


def train_and_evaluate(ts):
    n = len(ts)
    test_size = max(1, int(round(n * 0.2)))

    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]

    forecast = naive_forecast_last_value(train, len(test))
    forecast = pd.Series(forecast, index=test.index)

    mae = float(mean_absolute_error(test.values, forecast.values))
    rmse = float(np.sqrt(mean_squared_error(test.values, forecast.values)))

    eps = 1e-9
    mape = float(np.mean(np.abs((test.values - forecast.values) / (np.abs(test.values) + eps))) * 100.0)

    return train, test, forecast, mae, rmse, mape


def main():
    ts, date_col, value_col = load_series()

    deltas = ts.index.to_series().diff().dropna()
    tensor = deltas.mode().iloc[0] if len(deltas) > 0 else None
    freq_inferred = pd.infer_freq(ts.index)

    train, test, forecast, mae, rmse, mape = train_and_evaluate(ts)

    joblib.dump({"type": "naive_last_value"}, MODEL_PATH)

    meta = {
        "date_col": date_col,
        "value_col": value_col,
        "tensor": str(tensor),
        "freq_inferred": freq_inferred,
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
    joblib.dump(meta, META_PATH)

    print("CSV:", CSV_PATH)
    print("Model:", MODEL_PATH)
    print("Meta:", META_PATH)
    print("Tensor:", tensor)
    print("Freq inferred:", freq_inferred)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE (%):", mape)


if __name__ == "__main__":
    main()

