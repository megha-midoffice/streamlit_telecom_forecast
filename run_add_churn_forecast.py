"""
run_add_churn_forecast.py

Forecast prepaid/postpaid adds and churn.
Automatically bridges gap between last training date and today.
"""

import pickle
import pandas as pd


MODEL_PATH = "add_churn_sarimax.pkl"


def run_add_churn_forecast(horizon_days: int) -> pd.DataFrame:
    """
    Forecast adds & churn for given number of future days.
    Returns DataFrame.
    """
    
    # -----------------------------
    # Load model payload
    # -----------------------------
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    
    prepaid_adds_model = payload["prepaid_adds_model"]
    prepaid_churn_model = payload["prepaid_churn_model"]
    postpaid_adds_model = payload["postpaid_adds_model"]
    postpaid_churn_model = payload["postpaid_churn_model"]
    
    last_train_date = pd.to_datetime(payload["last_train_date"])
    
    today = pd.Timestamp.today().normalize()
    
    # -----------------------------
    # Compute gap
    # -----------------------------
    gap_days = (today - last_train_date).days
    
    if gap_days < 0:
        gap_days = 0
    
    steps_needed = gap_days + horizon_days
    
    # -----------------------------
    # Forecast all 4
    # -----------------------------
    prepaid_adds_forecast = prepaid_adds_model.forecast(steps=steps_needed)
    prepaid_churn_forecast = prepaid_churn_model.forecast(steps=steps_needed)
    postpaid_adds_forecast = postpaid_adds_model.forecast(steps=steps_needed)
    postpaid_churn_forecast = postpaid_churn_model.forecast(steps=steps_needed)
    
    # -----------------------------
    # Build forecast index
    # -----------------------------
    forecast_index = pd.date_range(
        start=last_train_date + pd.Timedelta(days=1),
        periods=steps_needed,
        freq="D"
    )
    
    forecast_df = pd.DataFrame({
        "date": forecast_index,
        "prepaid_adds_forecast": prepaid_adds_forecast.values,
        "prepaid_churn_forecast": prepaid_churn_forecast.values,
        "postpaid_adds_forecast": postpaid_adds_forecast.values,
        "postpaid_churn_forecast": postpaid_churn_forecast.values,
    })
    
    
    # -----------------------------
    # Return only requested horizon
    # -----------------------------
    final_output = forecast_df.tail(horizon_days).reset_index(drop=True)
    
    return final_output
