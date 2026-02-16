"""
run_subscription_forecast.py

Forecast subscription adds using trained SARIMAX models.
Handles gap between last training date and today automatically.
"""

import pickle
import pandas as pd


MODEL_PATH = "subscription_sarimax.pkl"


def run_subscription_forecast(horizon_days: int) -> pd.DataFrame:
    """
    Forecast subscriptions for given number of future days.
    
    Automatically bridges gap between last training date and today.
    Returns DataFrame with prepaid, postpaid, and total forecasts.
    """
    
    # -----------------------------
    # Load model payload
    # -----------------------------
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    
    prepaid_model = payload["prepaid_model"]
    postpaid_model = payload["postpaid_model"]
    last_train_date = pd.to_datetime(payload["last_train_date"])
    
    today = pd.Timestamp.today().normalize()
    
    # -----------------------------
    # Compute gap
    # -----------------------------
    gap_days = (today - last_train_date).days
    
    if gap_days < 0:
        gap_days = 0  # safety
    
    steps_needed = gap_days + horizon_days
    
    # -----------------------------
    # Forecast
    # -----------------------------
    prepaid_forecast_full = prepaid_model.forecast(steps=steps_needed)
    postpaid_forecast_full = postpaid_model.forecast(steps=steps_needed)
    
    # -----------------------------
    # Create forecast index
    # -----------------------------
    forecast_index = pd.date_range(
        start=last_train_date + pd.Timedelta(days=1),
        periods=steps_needed,
        freq="D"
    )
    
    forecast_df = pd.DataFrame({
        "date": forecast_index,
        "prepaid_forecast": prepaid_forecast_full.values,
        "postpaid_forecast": postpaid_forecast_full.values
    })
    
    forecast_df["total_forecast"] = (
        forecast_df["prepaid_forecast"] +
        forecast_df["postpaid_forecast"]
    )
    
    # -----------------------------
    # Return only requested horizon
    # -----------------------------
    final_output = forecast_df.tail(horizon_days).reset_index(drop=True)
    
    return final_output
