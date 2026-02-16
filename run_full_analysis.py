"""
run_full_analysis.py

End-to-end execution:

- Prepare subscription data
- Prepare add/churn data
- Run forecasts
- Detect structural drivers

Returns final structured result for agent.
"""

import pandas as pd

from run_subscription_forecast import run_subscription_forecast
from run_add_churn_forecast import run_add_churn_forecast

from detect_subscription_drivers import detect_subscription_drivers
from detect_add_churn_drivers import detect_add_churn_drivers


# ============================================================
# DATA LOADERS
# ============================================================

def load_master_data(path):

    df = pd.read_parquet(path)

    # Dates
    df["created_date"] = pd.to_datetime(
        df["created_date"],
        errors="coerce"
    )

    # Segment
    df["segment"] = (
        df["vlocity_account_payment_type"]
        .astype(str)
        .str.lower()
    )

    # Unique subscription ID (CRITICAL)
    id_cols = [
        "asset_msisdn",
        "asset_id",
        "account_id",
        "subscription_id"
    ]

    df["unique_subscription_id"] = (
        df[id_cols]
        .astype(str)
        .agg("|".join, axis=1)
    )

    # Weekly spine
    df["week_start"] = (
        df["created_date"]
        .dt.to_period("W-MON")
        .dt.start_time
    )

    return df




def load_orders_data(path):
    orders = pd.read_parquet(path)
    orders["report_date"] = pd.to_datetime(
        orders["activateddate"],
        errors="coerce"
    )
    orders["week_start"] = orders["report_date"].dt.to_period("W-MON").dt.start_time

    orders["order_key"] = (
        orders["orderid"].astype(str) + "_" +
        orders["ordernumber"].astype(str)
    )

    return orders


# ============================================================
# ADD / CHURN DRIVER PREP
# ============================================================

def prepare_add_churn_driver_frames(orders):

    # POSTPAID ADDS
    adds_postpaid = orders[
        (orders['segment__c'] == 'B2C') &
        (orders['vlocity_cmt__accountpaymenttype__c'] == 'Postpaid') &
        (orders['type'].isin(['Sales', 'Add'])) &
        (orders['account_classification_order'] == 'Prod Accounts')
    ].copy()

    weekly_postpaid_adds = (
        adds_postpaid
        .groupby("week_start")["order_key"]
        .nunique()
        .reset_index(name="adds")
    )

    weekly_postpaid_adds["is_retail"] = (
        adds_postpaid.groupby("week_start")["vlocity_cmt__originatingchannel__c"]
        .apply(lambda x: (x == "Retail").sum())
        .values
    )

    weekly_postpaid_adds["is_portin"] = (
        adds_postpaid.groupby("week_start")["vlocity_cmt__reason__c"]
        .apply(lambda x: (x == "Port In").sum())
        .values
    )

    # POSTPAID CHURN
    churn_postpaid = orders[
        (orders['type'] == 'Disconnect') &
        (orders['vlocity_cmt__accountpaymenttype__c'] == 'Postpaid') &
        (orders['segment__c'] == 'B2C')
    ].copy()

    weekly_postpaid_churn = (
        churn_postpaid
        .groupby("week_start")["order_key"]
        .nunique()
        .reset_index(name="churn")
    )

    weekly_postpaid_churn["is_competitive"] = (
        churn_postpaid.groupby("week_start")["vlocity_cmt__reason__c"]
        .apply(lambda x: x.isin(["Port Out", "Competitive offer"]).sum())
        .values
    )

    weekly_postpaid_churn["is_financial"] = (
        churn_postpaid.groupby("week_start")["vlocity_cmt__reason__c"]
        .apply(lambda x: x.isin(["Non-Pay", "Billing Cancel", "Cutting back"]).sum())
        .values
    )

    # PREPAID ADDS
    adds_prepaid = orders[
        (orders['segment__c'] == 'B2C') &
        (orders['vlocity_cmt__accountpaymenttype__c'] == 'Prepaid') &
        (orders['type'].isin(['Sales', 'Add']))
    ].copy()

    weekly_prepaid_adds = (
        adds_prepaid
        .groupby("week_start")["order_key"]
        .nunique()
        .reset_index(name="adds")
    )

    weekly_prepaid_adds["is_portin"] = (
        adds_prepaid.groupby("week_start")["vlocity_cmt__reason__c"]
        .apply(lambda x: (x == "Port In").sum())
        .values
    )

    # PREPAID CHURN
    churn_prepaid = orders[
        (orders['type'] == 'Disconnect') &
        (orders['vlocity_cmt__accountpaymenttype__c'] == 'Prepaid') &
        (orders['segment__c'] == 'B2C')
    ].copy()

    weekly_prepaid_churn = (
        churn_prepaid
        .groupby("week_start")["order_key"]
        .nunique()
        .reset_index(name="churn")
    )

    weekly_prepaid_churn["is_financial"] = (
        churn_prepaid.groupby("week_start")["vlocity_cmt__reason__c"]
        .apply(lambda x: x.isin(["Non-Pay", "Billing Cancel", "Cutting back"]).sum())
        .values
    )

    return (
        weekly_postpaid_adds,
        weekly_postpaid_churn,
        weekly_prepaid_adds,
        weekly_prepaid_churn
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_analysis(master_path, orders_path, forecast_days=14):

    master_df = load_master_data(master_path)
    orders_df = load_orders_data(orders_path)

    # Forecasts
    subscription_forecast = run_subscription_forecast(forecast_days)
    add_churn_forecast = run_add_churn_forecast(forecast_days)

    # Drivers
    subscription_drivers = detect_subscription_drivers(master_df)

    (
        postpaid_add_df,
        postpaid_churn_df,
        prepaid_add_df,
        prepaid_churn_df
    ) = prepare_add_churn_driver_frames(orders_df)

    add_churn_drivers = detect_add_churn_drivers(
        postpaid_add_df,
        postpaid_churn_df,
        prepaid_add_df,
        prepaid_churn_df
    )


    return {
        "forecast": {
            "subscriptions": subscription_forecast,
            "add_churn": add_churn_forecast
        },
        "drivers": {
            "subscriptions": subscription_drivers,
            "add_churn": add_churn_drivers
        }
    }
