# -*- coding: utf-8 -*-

from run_subscription_forecast import run_subscription_forecast
from run_add_churn_forecast import run_add_churn_forecast
from detect_subscription_drivers import detect_subscription_drivers
from detect_add_churn_drivers import detect_add_churn_drivers

from run_full_analysis import (
    load_master_data,
    load_orders_data,
    prepare_add_churn_driver_frames
)

from agent.narrator_subscriptions import generate_subscription_driver_narrative
from agent.narrator_add_churn import generate_add_churn_driver_narrative


class TelecomAgent:
    """
    Clean execution agent.

    Flow:
    Button → Forecast → Driver Detection → Driver Narration

    No routing.
    No mixed contexts.
    No forecast narration.
    """

    def __init__(self, master_path: str, orders_path: str):
        self.master_path = master_path
        self.orders_path = orders_path

        # Load once
        self.master_df = load_master_data(master_path)
        self.orders_df = load_orders_data(orders_path)

        (
            self.postpaid_add_df,
            self.postpaid_churn_df,
            self.prepaid_add_df,
            self.prepaid_churn_df
        ) = prepare_add_churn_driver_frames(self.orders_df)

    # -------------------------------------------------------
    # DEBUG — RAW OUTPUTS
    # -------------------------------------------------------

    def debug_outputs(self, forecast_days: int = 14):

        print("\n=== SUBSCRIPTION FORECAST ===")
        sub_forecast = run_subscription_forecast(forecast_days)
        print(sub_forecast)

        print("\n=== ADD-CHURN FORECAST ===")
        add_churn_forecast = run_add_churn_forecast(forecast_days)
        print(add_churn_forecast)

        print("\n=== SUBSCRIPTION DRIVERS ===")
        sub_drivers = detect_subscription_drivers(self.master_df)
        print(sub_drivers)

        print("\n=== ADD-CHURN DRIVERS ===")
        add_churn_drivers = detect_add_churn_drivers(
            self.postpaid_add_df,
            self.postpaid_churn_df,
            self.prepaid_add_df,
            self.prepaid_churn_df
        )
        print(add_churn_drivers)

        return {
            "subscription_forecast": sub_forecast,
            "add_churn_forecast": add_churn_forecast,
            "subscription_drivers": sub_drivers,
            "add_churn_drivers": add_churn_drivers
        }

    # -------------------------------------------------------
    # SUBSCRIPTION EXECUTION
    # -------------------------------------------------------

    def run_subscription_analysis(self, forecast_days: int = 14):
        """
        Runs:
        1) Subscription forecast
        2) Subscription driver detection
        3) Subscription driver narration

        Returns:
        dict with forecast, drivers, narration
        """

        forecast = run_subscription_forecast(forecast_days)
        drivers = detect_subscription_drivers(self.master_df)

        narration = generate_subscription_driver_narrative(drivers)

        return {
            "forecast": forecast,
            "drivers": drivers,
            "narration": narration
        }

    # -------------------------------------------------------
    # ADD / CHURN EXECUTION
    # -------------------------------------------------------

    def run_add_churn_analysis(self, forecast_days: int = 14):
        """
        Runs:
        1) Add/Churn forecast
        2) Add/Churn driver detection
        3) Add/Churn driver narration

        Returns:
        dict with forecast, drivers, narration
        """

        forecast = run_add_churn_forecast(forecast_days)

        drivers = detect_add_churn_drivers(
            self.postpaid_add_df,
            self.postpaid_churn_df,
            self.prepaid_add_df,
            self.prepaid_churn_df
        )

        narration = generate_add_churn_driver_narrative(drivers)

        return {
            "forecast": forecast,
            "drivers": drivers,
            "narration": narration
        }
