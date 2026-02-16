"""
detect_add_churn_drivers.py

Structural Alignment driver analysis for:
- Postpaid
- Prepaid

Goal:
Explain what components are aligned with adds/churn (not regime shifts).

Approach:
- Smooth target and driver (rolling sums)
- Compute driver share of target
- Compute rolling correlation (driver vs target)
- Compute a simple alignment score
- Return "aligned" True/False based on thresholds

No model training here. Modeling stays offline.
"""

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROLLING_WINDOW = 4      # smoothing window (weeks)
CORR_WINDOW = 8         # correlation window (weeks)

MIN_POINTS = 12         # minimum spine points after rolling for analysis

# Alignment thresholds (practical)
CORR_ALIGNED = 0.60     # "tracks closely"
SHARE_MIN = 0.10        # driver is meaningful fraction of total (10%)
SCORE_ALIGNED = 0.65    # overall alignment score threshold


# ============================================================
# HELPERS
# ============================================================

def _safe_div(n, d):
    return np.where(d == 0, np.nan, n / d)

def _alignment_score(corr, share, corr_weight=0.7, share_weight=0.3):
    """
    Simple bounded score in [0, 1].
    - corr mapped from [-1,1] -> [0,1]
    - share capped at 1.0
    """
    if corr is None or np.isnan(corr):
        corr_component = 0.0
    else:
        corr_component = (corr + 1) / 2  # [-1,1] -> [0,1]

    share_component = 0.0 if share is None or np.isnan(share) else float(min(max(share, 0.0), 1.0))
    return float(corr_weight * corr_component + share_weight * share_component)

def _build_spine(df, date_col, target_col, driver_col):
    """
    Build weekly spine with rolling sums and share.
    Expects df already weekly-grained or daily with a week_start column.
    """
    spine = df.copy().sort_values(date_col)

    # rolling sums
    spine[f"{target_col}_rw"] = spine[target_col].rolling(ROLLING_WINDOW).sum()
    spine[f"{driver_col}_rw"] = spine[driver_col].rolling(ROLLING_WINDOW).sum()

    spine = spine.dropna(subset=[f"{target_col}_rw", f"{driver_col}_rw"]).reset_index(drop=True)

    if len(spine) < MIN_POINTS:
        return None

    # share of driver in target
    spine["driver_share"] = _safe_div(spine[f"{driver_col}_rw"].values, spine[f"{target_col}_rw"].values)

    # rolling correlation (driver vs target)
    spine["rolling_corr"] = (
        spine[f"{driver_col}_rw"]
        .rolling(CORR_WINDOW)
        .corr(spine[f"{target_col}_rw"])
    )

    return spine

def _summarize_alignment(spine, target_rw_col, driver_rw_col):
    """
    Summarize alignment using recent windows.
    """
    recent_corr = float(spine["rolling_corr"].iloc[-1]) if len(spine) else np.nan
    avg_corr_recent = float(spine["rolling_corr"].tail(CORR_WINDOW).mean()) if len(spine) >= CORR_WINDOW else float(spine["rolling_corr"].mean())

    share_recent = float(spine["driver_share"].tail(CORR_WINDOW).mean()) if len(spine) >= CORR_WINDOW else float(spine["driver_share"].mean())
    share_latest = float(spine["driver_share"].iloc[-1])

    target_level_recent = float(spine[target_rw_col].tail(CORR_WINDOW).mean())
    driver_level_recent = float(spine[driver_rw_col].tail(CORR_WINDOW).mean())

    score = _alignment_score(avg_corr_recent, share_recent)

    aligned = (
        (avg_corr_recent >= CORR_ALIGNED) and
        (share_recent >= SHARE_MIN) and
        (score >= SCORE_ALIGNED)
    )

    # confidence buckets (simple + narratable)
    if aligned and score >= 0.80:
        confidence = "high"
    elif aligned:
        confidence = "medium"
    else:
        confidence = "low"

    return aligned, confidence, {
        "corr_latest": float(recent_corr) if not np.isnan(recent_corr) else None,
        "corr_recent_avg": float(avg_corr_recent) if not np.isnan(avg_corr_recent) else None,
        "share_latest": float(share_latest) if not np.isnan(share_latest) else None,
        "share_recent_avg": float(share_recent) if not np.isnan(share_recent) else None,
        "alignment_score": float(score),
        "target_level_recent": float(target_level_recent),
        "driver_level_recent": float(driver_level_recent),
    }


# ============================================================
# POSTPAID ADDS (Retail)
# ============================================================

def _detect_postpaid_add_driver(df):
    # expects: week_start, adds, is_retail
    spine = _build_spine(df, "week_start", "adds", "is_retail")

    if spine is None:
        return {"driver": "retail_add_growth", "aligned": False, "confidence": "low", "reason": "Insufficient data"}

    aligned, confidence, metrics = _summarize_alignment(
        spine,
        target_rw_col="adds_rw",
        driver_rw_col="is_retail_rw"
    )

    return {
        "driver": "retail_add_growth",
        "aligned": bool(aligned),
        "confidence": confidence,
        "metrics": metrics
    }


# ============================================================
# POSTPAID CHURN (Competitive)
# ============================================================

def _detect_postpaid_churn_driver(df):
    # expects: week_start, churn, is_competitive
    spine = _build_spine(df, "week_start", "churn", "is_competitive")

    if spine is None:
        return {"driver": "competitive_churn_spike", "aligned": False, "confidence": "low", "reason": "Insufficient data"}

    aligned, confidence, metrics = _summarize_alignment(
        spine,
        target_rw_col="churn_rw",
        driver_rw_col="is_competitive_rw"
    )

    return {
        "driver": "competitive_churn_spike",
        "aligned": bool(aligned),
        "confidence": confidence,
        "metrics": metrics
    }


# ============================================================
# PREPAID ADDS (Port-in)
# ============================================================

def _detect_prepaid_add_driver(df):
    # expects: week_start, adds, is_portin
    spine = _build_spine(df, "week_start", "adds", "is_portin")

    if spine is None:
        return {"driver": "portin_add_growth", "aligned": False, "confidence": "low", "reason": "Insufficient data"}

    aligned, confidence, metrics = _summarize_alignment(
        spine,
        target_rw_col="adds_rw",
        driver_rw_col="is_portin_rw"
    )

    return {
        "driver": "portin_add_growth",
        "aligned": bool(aligned),
        "confidence": confidence,
        "metrics": metrics
    }


# ============================================================
# PREPAID CHURN (None)
# ============================================================

def _detect_prepaid_churn_driver(df):
    return {
        "driver": "no_structural_driver_detected",
        "aligned": False,
        "confidence": "low",
        "metrics": {}
    }


# ============================================================
# PUBLIC ENTRYPOINT
# ============================================================

def detect_add_churn_drivers(postpaid_add_df,
                             postpaid_churn_df,
                             prepaid_add_df,
                             prepaid_churn_df):

    return {
        "add_churn_driver_analysis": {
            "postpaid": {
                "adds": _detect_postpaid_add_driver(postpaid_add_df),
                "churn": _detect_postpaid_churn_driver(postpaid_churn_df)
            },
            "prepaid": {
                "adds": _detect_prepaid_add_driver(prepaid_add_df),
                "churn": _detect_prepaid_churn_driver(prepaid_churn_df)
            }
        }
    }
