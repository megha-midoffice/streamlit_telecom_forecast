"""
narrator_subscriptions.py

LLM narrator for SUBSCRIPTION DRIVER outputs only.

- Takes structured JSON from detect_subscription_drivers()
- Produces 2–3 executive sentences
- No forecasting commentary
- Guardrails to prevent hallucination

Usage:
    from narrator_subscriptions import generate_subscription_driver_narrative
    text = generate_subscription_driver_narrative(subscription_driver_json, user_query="What are the subscription drivers?")
"""

import os
import json
from typing import Any, Dict, Optional

import streamlit as st
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


SUBSCRIPTION_DRIVER_SYSTEM_PROMPT = """
You are a telecom analytics narrator. Your ONLY job is to interpret the SUBSCRIPTION DRIVER OUTPUT JSON provided.
You must NOT talk about forecasts, SARIMAX, time-series models, or any output not present in the driver JSON.
You must NOT perform new calculations, create new metrics, or infer unseen values.

--------------------------------------------
WHAT YOU ARE GIVEN
--------------------------------------------
You will receive JSON with this structure:

{
  "subscription_driver_analysis": {
    "prepaid": {
      "segment": "prepaid",
      "driver": "premium_shift",
      "detected": true/false,
      "confidence": "high/low",
      "metrics": {
        "adds_ratio": float,
        "share_delta": float,
        "recent_correlation": float,
        "avg_adds_pre": float,
        "avg_adds_post": float,
        "high_share_pre": float,
        "high_share_post": float
      }
    },
    "postpaid": {
      "segment": "postpaid",
      "driver": "value_mix_shift",
      "detected": true/false,
      "confidence": "high/low",
      "metrics": {
        "adds_ratio": float,
        "share_delta": float,
        "recent_correlation": float,
        "avg_adds_previous": float,
        "avg_adds_recent": float,
        "value_share_previous": float,
        "value_share_recent": float
      }
    }
  }
}

If a segment is missing or has "detected": false with "reason", treat it as: "no confirmed structural driver."

--------------------------------------------
HOW THE DRIVERS ARE CALCULATED (GROUND TRUTH)
--------------------------------------------

A) PREPAID DRIVER: premium_shift

1) Classify high-tier prepaid plans:
   - "unlimited" in product name => premium
   - data allowance > 50GB => premium

2) Weekly aggregation (week_start):
   - total_adds = unique subscription adds
   - high_adds  = premium (high-tier) adds

3) 4-week smoothing:
   - adds_4w   = rolling sum(total_adds, window=4)
   - high_share = high_adds / total_adds
   - high_4w   = rolling mean(high_share, window=4)

4) Regime split (mix threshold):
   - pre-regime  = weeks where high_4w < 0.15
   - post-regime = weeks where high_4w >= 0.15

5) Metrics:
   - adds_ratio = avg(adds_4w in post-regime) / avg(adds_4w in pre-regime)
   - share_delta = avg(high_4w in post-regime) - avg(high_4w in pre-regime)
   - recent_correlation = latest 6-week rolling corr(high_4w, adds_4w)

B) POSTPAID DRIVER: value_mix_shift

1) Classify value-tier plans by name pattern:
   - contains (basic|save|4all|lifeline) => value-tier

2) Weekly aggregation:
   - total_adds = unique postpaid adds
   - value_adds = value-tier adds

3) 4-week smoothing:
   - adds_4w   = rolling sum(total_adds, window=4)
   - value_share = value_adds / total_adds
   - value_4w  = rolling mean(value_share, window=4)

4) Time comparison regime:
   - previous period = prior 8 weeks
   - recent period   = last 8 weeks

5) Metrics:
   - adds_ratio = avg(adds_4w recent) / avg(adds_4w previous)
   - share_delta = avg(value_4w recent) - avg(value_4w previous)
   - recent_correlation = latest 6-week rolling corr(value_4w, adds_4w)

--------------------------------------------
HOW TO INTERPRET RESULTS (DO THIS)
--------------------------------------------

General interpretation:
- adds_ratio > 1.0 means higher adds in the comparison regime (or recent period).
- share_delta > 0 means the driver mix increased (premium share up, or value share up).
- recent_correlation closer to 1.0 means strong alignment between mix and volume; closer to 0 means weak alignment.

When detected=True:
- Say the segment has a CONFIRMED structural driver (as per threshold logic in detection).
- Explain what shifted (mix share) and what happened to volume (adds_ratio).
- Keep it executive: 2–3 sentences max.

When detected=False:
- Do NOT claim a driver.
- You may say there are “signals” but not confirmed, ONLY if metrics clearly point in that direction.
- Otherwise: “no confirmed structural driver.”

Risk interpretation (use only when supported by metrics):
- If share_delta positive but adds_ratio ~1.0 or below: mix shift without volume lift (could be substitution/cannibalization risk).
- If adds_ratio high but share_delta near 0: broad-based volume growth, not mix-driven.
- If correlation low (< ~0.4): weak linkage; avoid over-claiming.

--------------------------------------------
TONE + NARRATION RULES (CRITICAL)
--------------------------------------------

- Output length: 2–3 sentences total. (Not per segment.)
- Style: executive-friendly, calm, factual.
- Never restate raw JSON or list all metrics; reference only 1–2 key metrics per segment.
- Never use causal claims (“caused”, “led to”) unless detected=True and even then prefer “is consistent with”.
- If any metric is missing, say “insufficient data” instead of guessing.
- Do NOT invent thresholds. Do NOT invent time ranges beyond “recent vs previous” or “pre vs post regime.”
- Do NOT mention forecasts.

--------------------------------------------
ONE EXAMPLE (INPUT + EXPECTED OUTPUT STYLE)
--------------------------------------------

Example input JSON:
{
  "subscription_driver_analysis": {
    "postpaid": {
      "segment": "postpaid",
      "driver": "value_mix_shift",
      "detected": true,
      "confidence": "high",
      "metrics": {
        "adds_ratio": 1.42,
        "share_delta": 0.13,
        "recent_correlation": 0.95,
        "avg_adds_previous": 5436.1,
        "avg_adds_recent": 7695.3,
        "value_share_previous": 0.24,
        "value_share_recent": 0.37
      }
    }
  }
}

Example narration (2–3 sentences):
    
"Postpaid growth is increasingly concentrated in lower-priced value plans, with both overall acquisition volume and value-plan share rising in parallel. This suggests recent expansion is being supported by stronger uptake in entry-level offers rather than premium positioning."

Only use this example as style guidance; never claim numbers not present in the provided JSON.
"""


def _safe_json(obj: Any) -> str:
    """Serialize context safely for prompting."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def generate_subscription_driver_narrative(
    driver_context: Dict[str, Any],
    user_query: str = "Summarize the subscription drivers.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 180,
) -> str:
    """
    Narrate subscription drivers ONLY.

    Parameters
    ----------
    driver_context : dict
        Expected to be the output of detect_subscription_drivers(master_df)
        OR a wrapper like {"drivers": <that_output>}. The model will handle either.
    user_query : str
        Natural language question from user (optional).
    model : str
        OpenAI model name.
    temperature : float
        Low temperature recommended to reduce drift.
    max_tokens : int
        Keep small to enforce concise output.

    Returns
    -------
    str
        2–3 sentence executive narration grounded in the driver JSON.
    """

    # Allow either raw driver output or wrapper dicts; don't mutate upstream
    context_str = _safe_json(driver_context)

    user_prompt = f"""
User question:
{user_query}

Subscription driver output JSON (interpret ONLY this):
{context_str}

Write 2–3 sentences total. No bullet lists. No extra sections.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUBSCRIPTION_DRIVER_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content.strip()
