# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 17:43:26 2026

@author: MeghaGhosh_b5xx485
"""

from agent.telecom_agent import TelecomAgent


if __name__ == "__main__":

    agent = TelecomAgent(
        master_path="customer_master_19_jan.csv",
        orders_path="orders_activity_30_jan.csv"
    )

    print("\n==================================================")
    print("DEBUG: RAW OUTPUTS")
    print("==================================================")

    debug_data = agent.debug_outputs(14)

    print("\n==================================================")
    print("SUBSCRIPTION ANALYSIS (WITH NARRATION)")
    print("==================================================")

    sub_result = agent.run_subscription_analysis(14)

    print("\n--- Forecast ---")
    print(sub_result["forecast"])

    print("\n--- Drivers ---")
    print(sub_result["drivers"])

    print("\n--- Narration ---")
    print(sub_result["narration"])

    print("\n==================================================")
    print("ADD / CHURN ANALYSIS (WITH NARRATION)")
    print("==================================================")

    add_churn_result = agent.run_add_churn_analysis(14)

    print("\n--- Forecast ---")
    print(add_churn_result["forecast"])

    print("\n--- Drivers ---")
    print(add_churn_result["drivers"])

    print("\n--- Narration ---")
    print(add_churn_result["narration"])
