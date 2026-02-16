"""
detect_subscription_drivers.py

Detects:
- Prepaid premium shift
- Postpaid value mix shift
"""

import pandas as pd
import re


# ============================================================
# CONFIGURATION (Production thresholds)
# ============================================================

PREMIUM_REGIME_THRESHOLD = 0.15

PREPAID_STRONG_SHARE_DELTA = 0.10
PREPAID_STRONG_ADDS_RATIO = 1.2
PREPAID_STRONG_CORR = 0.6

POSTPAID_STRONG_SHARE_DELTA = 0.05
POSTPAID_STRONG_ADDS_RATIO = 1.2
POSTPAID_STRONG_CORR = 0.5


# ============================================================
# HELPER — GB Extraction (Consistent with Notebook)
# ============================================================

def _extract_gb(product_name):
    if pd.isna(product_name):
        return None
    
    name = str(product_name).lower()
    
    if "unlimited" in name:
        return "unlimited"
    
    match = re.search(r'(\d+)gb', name)
    if match:
        return int(match.group(1))
    
    return None


def _is_high_tier(product_name):
    gb = _extract_gb(product_name)
    
    if gb == "unlimited":
        return True
    
    if isinstance(gb, int) and gb > 50:
        return True
    
    return False


# ============================================================
# PREPAID DRIVER
# ============================================================

def _detect_prepaid_driver(master_df: pd.DataFrame) -> dict:
    
    df = master_df[master_df["segment"] == "prepaid"].copy()
    
    if df.empty:
        return {
            "segment": "prepaid",
            "detected": False,
            "reason": "No prepaid data"
        }
    
    df["high_tier_flag"] = df["product_name"].apply(_is_high_tier)
    
    weekly_total = (
        df.groupby("week_start")["unique_subscription_id"]
        .nunique()
        .reset_index(name="total_adds")
    )
    
    weekly_high = (
        df[df["high_tier_flag"]]
        .groupby("week_start")["unique_subscription_id"]
        .nunique()
        .reset_index(name="high_adds")
    )
    
    spine = weekly_total.merge(
        weekly_high,
        on="week_start",
        how="left"
    ).fillna(0)
    
    spine = spine.sort_values("week_start")
    
    spine["adds_4w"] = spine["total_adds"].rolling(4).sum()
    spine["high_share"] = spine["high_adds"] / spine["total_adds"]
    spine["high_4w"] = spine["high_share"].rolling(4).mean()
    
    spine = spine.dropna().reset_index(drop=True)
    
    if len(spine) < 8:
        return {
            "segment": "prepaid",
            "detected": False,
            "reason": "Insufficient data"
        }
    
    # Regime split
    pre = spine[spine["high_4w"] < PREMIUM_REGIME_THRESHOLD]
    post = spine[spine["high_4w"] >= PREMIUM_REGIME_THRESHOLD]
    
    if len(post) < 4:
        return {
            "segment": "prepaid",
            "detected": False,
            "reason": "No premium regime"
        }
    
    avg_adds_pre = pre["adds_4w"].mean()
    avg_adds_post = post["adds_4w"].mean()
    
    avg_high_pre = pre["high_4w"].mean()
    avg_high_post = post["high_4w"].mean()
    
    adds_ratio = avg_adds_post / avg_adds_pre if avg_adds_pre > 0 else 0
    share_delta = avg_high_post - avg_high_pre
    
    # Rolling correlation
    spine["rolling_corr"] = (
        spine["high_4w"]
        .rolling(6)
        .corr(spine["adds_4w"])
    )
    
    recent_corr = spine["rolling_corr"].iloc[-1]
    
    detected = (
        share_delta >= PREPAID_STRONG_SHARE_DELTA and
        adds_ratio >= PREPAID_STRONG_ADDS_RATIO and
        recent_corr >= PREPAID_STRONG_CORR
    )
    
    return {
        "segment": "prepaid",
        "driver": "premium_shift",
        "detected": bool(detected),
        "confidence": "high" if detected else "low",
        "metrics": {
            "adds_ratio": float(adds_ratio),
            "share_delta": float(share_delta),
            "recent_correlation": float(recent_corr),
            "avg_adds_pre": float(avg_adds_pre),
            "avg_adds_post": float(avg_adds_post),
            "high_share_pre": float(avg_high_pre),
            "high_share_post": float(avg_high_post)
        }
    }


# ============================================================
# POSTPAID DRIVER
# ============================================================

def _detect_postpaid_driver(master_df: pd.DataFrame) -> dict:
    
    df = master_df[master_df["segment"] == "postpaid"].copy()
    
    if df.empty:
        return {
            "segment": "postpaid",
            "detected": False,
            "reason": "No postpaid data"
        }
    
    df["value_flag"] = df["product_name"].str.contains(
        r"(basic|save|4all|lifeline)",
        case=False,
        regex=True,
        na=False
    )
    
    weekly_total = (
        df.groupby("week_start")["unique_subscription_id"]
        .nunique()
        .reset_index(name="total_adds")
    )
    
    weekly_value = (
        df[df["value_flag"]]
        .groupby("week_start")["unique_subscription_id"]
        .nunique()
        .reset_index(name="value_adds")
    )
    
    spine = weekly_total.merge(
        weekly_value,
        on="week_start",
        how="left"
    ).fillna(0)
    
    spine = spine.sort_values("week_start")
    
    spine["adds_4w"] = spine["total_adds"].rolling(4).sum()
    spine["value_share"] = spine["value_adds"] / spine["total_adds"]
    spine["value_4w"] = spine["value_share"].rolling(4).mean()
    
    spine = spine.dropna().reset_index(drop=True)
    
    if len(spine) < 16:
        return {
            "segment": "postpaid",
            "detected": False,
            "reason": "Insufficient data"
        }
    
    recent = spine.tail(8)
    previous = spine.iloc[-16:-8]
    
    avg_adds_recent = recent["adds_4w"].mean()
    avg_adds_previous = previous["adds_4w"].mean()
    
    avg_value_recent = recent["value_4w"].mean()
    avg_value_previous = previous["value_4w"].mean()
    
    adds_ratio = avg_adds_recent / avg_adds_previous if avg_adds_previous > 0 else 0
    share_delta = avg_value_recent - avg_value_previous
    
    spine["rolling_corr"] = (
        spine["value_4w"]
        .rolling(6)
        .corr(spine["adds_4w"])
    )
    
    recent_corr = spine["rolling_corr"].iloc[-1]
    
    detected = (
        share_delta >= POSTPAID_STRONG_SHARE_DELTA and
        adds_ratio >= POSTPAID_STRONG_ADDS_RATIO and
        recent_corr >= POSTPAID_STRONG_CORR
    )
    
    return {
        "segment": "postpaid",
        "driver": "value_mix_shift",
        "detected": bool(detected),
        "confidence": "high" if detected else "low",
        "metrics": {
            "adds_ratio": float(adds_ratio),
            "share_delta": float(share_delta),
            "recent_correlation": float(recent_corr),
            "avg_adds_previous": float(avg_adds_previous),
            "avg_adds_recent": float(avg_adds_recent),
            "value_share_previous": float(avg_value_previous),
            "value_share_recent": float(avg_value_recent)
        }
    }



def detect_subscription_drivers(master_df: pd.DataFrame) -> dict:
    """
    Public entrypoint for agent.
    """
    
    return {
        "subscription_driver_analysis": {
            "prepaid": _detect_prepaid_driver(master_df),
            "postpaid": _detect_postpaid_driver(master_df)
        }
    }
