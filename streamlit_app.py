# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 20:35:43 2026

@author: MeghaGhosh_b5xx485
"""

import streamlit as st
import pandas as pd
from io import BytesIO

from agent.telecom_agent import TelecomAgent


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Telecom Forecast Dashboard",
    layout="wide"
)

st.title("Telecom Forecast & Driver Insights")

# ---------------------------------------------------
# FORECAST HORIZON CONTROL
# ---------------------------------------------------

st.markdown("### Forecast Horizon")

forecast_days = st.slider(
    "Select forecast horizon (days)",
    min_value=7,
    max_value=60,
    value=14,
    step=7
)


# ---------------------------------------------------
# LOAD AGENT (CACHED)
# ---------------------------------------------------

@st.cache_resource
def load_agent():
    return TelecomAgent(
        master_path="customer_master_small.parquet",
        orders_path="orders_activity_small.parquet"
    )

agent = load_agent()


# ---------------------------------------------------
# EXCEL EXPORT HELPER
# ---------------------------------------------------

def to_excel(df: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output


# ---------------------------------------------------
# BUTTONS
# ---------------------------------------------------

col1, col2 = st.columns(2)

run_subscription = col1.button("Run Subscription Forecast")
run_add_churn = col2.button("Run Add / Churn Forecast")


# ===================================================
# SUBSCRIPTION FLOW
# ===================================================

if run_subscription:

    with st.spinner("Running subscription analysis..."):
        result = agent.run_subscription_analysis(forecast_days)

    forecast_df = result["forecast"]
    narration = result["narration"]

    st.subheader("Subscription Forecast")

    col_graph, col_ai = st.columns([3, 1])

    with col_graph:
        st.line_chart(
            forecast_df.set_index("date")[
                ["prepaid_forecast", "postpaid_forecast", "total_forecast"]
            ]
        )

    with col_ai:
        st.markdown("### AI Insight")
        st.info(narration)

    st.download_button(
        label="Download Forecast (Excel)",
        data=to_excel(forecast_df),
        file_name="subscription_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ===================================================
# ADD / CHURN FLOW
# ===================================================


if run_add_churn:

    with st.spinner("Running add/churn analysis..."):
        result = agent.run_add_churn_analysis(forecast_days)

    forecast_df = result["forecast"]
    narration = result["narration"]

    st.subheader("Add / Churn Forecast")

    # ---- Adds + Churn split layout
    col_graph, col_ai = st.columns([3, 1])

    with col_graph:

        st.markdown("#### Adds Forecast")
        st.line_chart(
            forecast_df.set_index("date")[
                [
                    "prepaid_adds_forecast",
                    "postpaid_adds_forecast"
                ]
            ]
        )

        st.markdown("#### Churn Forecast")
        st.line_chart(
            forecast_df.set_index("date")[
                [
                    "prepaid_churn_forecast",
                    "postpaid_churn_forecast"
                ]
            ]
        )

    with col_ai:
        st.markdown("### AI Insight")
        st.info(narration)

    st.download_button(
        label="Download Add/Churn Forecast (Excel)",
        data=to_excel(forecast_df),
        file_name="add_churn_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

