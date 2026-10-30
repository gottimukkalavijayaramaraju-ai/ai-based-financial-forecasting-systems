"""
AI-Based Financial Forecasting System (base project)
=====================================================
Pipeline:
  1. Load data      -> Yahoo Finance (yfinance), a local CSV, or synthetic data
  2. Features       -> lags, returns, moving averages, volatility, RSI
  3. Models         -> Linear Regression, Random Forest, Gradient Boosting
  4. Evaluation     -> RMSE, MAE, MAPE, directional accuracy (time-based split)
  5. Forecast       -> next-day closing price + plot

Usage:
  python financial_forecasting.py                      # synthetic data (works offline)
  python financial_forecasting.py --ticker AAPL        # real data via yfinance
  python financial_forecasting.py --csv mydata.csv     # CSV with Date, Close columns

Install:
  pip install numpy pandas scikit-learn matplotlib yfinance
"""

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")  # save plots to file; remove this line to show windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def load_from_yfinance(ticker: str, period: str = "5y") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    return df[["Close"]].dropna()


def make_synthetic(n_days: int = 1500) -> pd.DataFrame:
    """Geometric random walk with drift and a seasonal wave (demo data)."""
    rng = np.random.default_rng(RANDOM_STATE)
    t = np.arange(n_days)
    returns = 0.0004 + 0.012 * rng.standard_normal(n_days) + 0.002 * np.sin(t / 20)
    price = 100 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame({"Close": price}, index=idx)


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["Return"] = df["Close"].pct_change()

    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = df["Close"].shift(lag)
        out[f"ret_lag_{lag}"] = out["Return"].shift(lag)

    for w in (5, 10, 20):
        out[f"sma_{w}"] = df["Close"].rolling(w).mean()
        out[f"vol_{w}"] = out["Return"].rolling(w).std()

    out["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["rsi_14"] = rsi(df["Close"])

    # Target: NEXT day's close
    out["target"] = df["Close"].shift(-1)
    return out


def get_xy(features: pd.DataFrame):
    data = features.dropna()
    X = data.drop(columns=["target"])
    y = data["target"]
    return X, y


# ----------------------------------------------------------------------------
# 3 & 4. MODELS AND EVALUATION
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, prev_close) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # Did the model get the direction (up/down vs previous close) right?
    direction = np.mean(np.sign(y_true - prev_close) == np.sign(y_pred - prev_close)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE %": mape, "Direction %": direction}


def main():
    parser = argparse.ArgumentParser(description="AI-Based Financial Forecasting")
    parser.add_argument("--ticker", type=str, help="Ticker symbol, e.g. AAPL, TCS.NS")
    parser.add_argument("--csv", type=str, help="CSV path with Date and Close columns")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    # ---- Load
    if args.csv:
        df, name = load_from_csv(args.csv), args.csv
    elif args.ticker:
        df, name = load_from_yfinance(args.ticker), args.ticker
    else:
        df, name = make_synthetic(), "Synthetic data"
    print(f"Loaded {len(df)} rows for {name}  ({df.index[0].date()} -> {df.index[-1].date()})")

    # ---- Features
    features = build_features(df)
    X, y = get_xy(features)

    # ---- Time-ordered split (NEVER shuffle time series)
    split = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }

    # ---- Train + evaluate
    results, predictions = {}, {}
    prev_close = X_test["Close"].values
    for label, model in models.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        predictions[label] = pred
        results[label] = evaluate(y_test.values, pred, prev_close)

    # Naive baseline: "tomorrow = today". Any useful model should beat this.
    results["Naive baseline"] = evaluate(y_test.values, prev_close, prev_close)

    report = pd.DataFrame(results).T.round(3).sort_values("RMSE")
    print("\nModel comparison (test set):")
    print(report.to_string())

    best = report.drop(index="Naive baseline").index[0]
    print(f"\nBest model: {best}")

    # ---- Next-day forecast using the latest row of features
    latest = features.drop(columns=["target"]).iloc[[-1]]
    if latest.isna().any(axis=1).iloc[0]:
        latest = features.drop(columns=["target"]).dropna().iloc[[-1]]
    next_price = models[best].predict(scaler.transform(latest))[0]
    last_close = df["Close"].iloc[-1]
    change = (next_price / last_close - 1) * 100
    print(f"Last close: {last_close:.2f} | Next-day forecast: {next_price:.2f} ({change:+.2f}%)")

    # ---- Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label="Actual", color="black", linewidth=1.5)
    for label, pred in predictions.items():
        plt.plot(y_test.index, pred, label=label, alpha=0.7)
    plt.title(f"Financial Forecasting - {name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("forecast_plot.png", dpi=150)
    print("Plot saved to forecast_plot.png")

    report.to_csv("model_results.csv")
    print("Metrics saved to model_results.csv")


if __name__ == "__main__":
    main()
