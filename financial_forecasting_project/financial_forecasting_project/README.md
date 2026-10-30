# AI-Based Financial Forecasting System

Base project that forecasts next-day closing prices using machine learning.

## Setup
    pip install -r requirements.txt

## Run
    python financial_forecasting.py                   # synthetic demo data
    python financial_forecasting.py --ticker AAPL     # real data (yfinance)
    python financial_forecasting.py --csv mydata.csv  # CSV with Date, Close columns

## Output
- Model comparison table (RMSE, MAE, MAPE, directional accuracy)
- Next-day price forecast
- forecast_plot.png and model_results.csv
