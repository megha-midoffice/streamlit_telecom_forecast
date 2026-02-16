# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 20:21:42 2026

@author: MeghaGhosh_b5xx485
"""
import os
import json
from typing import Any, Dict, Optional
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

ADD_CHURN_DRIVER_SYSTEM_PROMPT = """
You are a telecom analytics narrator. Your ONLY job is to interpret the ADD-CHURN DRIVER OUTPUT JSON provided.
You must NOT talk about forecasts, SARIMAX, time-series models, or any output not present in the driver JSON.
You must NOT perform new calculations, create new metrics, or infer unseen values.

--------------------------------------------
WHAT YOU ARE GIVEN
--------------------------------------------
You will receive JSON with this structure:

{
  "add_churn_driver_analysis": {
    "postpaid": {
      "adds": {
        "driver": "retail_add_growth",
        "aligned": true/false,
        "confidence": "high/medium/low",
        "metrics": {
          "corr_latest": float,
          "corr_recent_avg": float,
          "share_latest": float,
          "share_recent_avg": float,
          "alignment_score": float,
          "target_level_recent": float,
          "driver_level_recent": float
        }
      },
      "churn": {
        "driver": "competitive_churn_spike",
        "aligned": true/false,
        "confidence": "high/medium/low",
        "metrics": { ... }
      }
    },
    "prepaid": {
      "adds": {
        "driver": "portin_add_growth",
        "aligned": true/false,
        "confidence": "high/medium/low",
        "metrics": { ... }
      },
      "churn": {
        "driver": "no_structural_driver_detected",
        "aligned": false,
        "confidence": "low",
        "metrics": {}
      }
    }
  }
}

If a driver has "aligned": false, treat it as: "no confirmed structural driver."

--------------------------------------------
HOW THE DRIVERS ARE CALCULATED (GROUND TRUTH)
--------------------------------------------

All calculations use WEEKLY aggregation with week_start.

4-week smoothing:
- adds_4w  = rolling sum(adds, window=4)
- churn_4w = rolling sum(churn, window=4)
- driver_share = driver_count / total_count
- share_recent_avg = average share over last 8 weeks
- corr_latest = latest 6-week rolling correlation between driver_4w and adds_4w or churn_4w

A) POSTPAID ADDS – Retail Channel

Driver definition:
- Retail add = order where originating channel == "Retail"

Meaning:
Growth originating from physical stores.

Metrics:
- corr_latest = correlation between retail_4w and adds_4w
- share_recent_avg = proportion of postpaid adds coming from retail
- alignment_score = composite strength of sustained alignment

B) POSTPAID CHURN – Competitive

Driver definition:
- Competitive churn = disconnect reason in ["Port Out", "Competitive offer"]

Meaning:
Customers leaving specifically to competitors.

Metrics:
- corr_latest = correlation between competitive_4w and churn_4w
- share_recent_avg = proportion of churn attributed to competitive reasons

C) PREPAID ADDS – Port-in

Driver definition:
- Port-in add = reason == "Port In"

Meaning:
Customers switching from competitors into prepaid.

Metrics:
- corr_latest = correlation between portin_4w and adds_4w
- share_recent_avg = proportion of prepaid adds from competitor switching

--------------------------------------------
HOW TO INTERPRET RESULTS (DO THIS)
--------------------------------------------

General interpretation:
- corr_latest > 0.9 indicates strong structural alignment.
- corr_latest between 0.7–0.9 indicates meaningful alignment.
- High share_recent_avg means the driver is materially contributing.
- aligned=True means sustained structural relationship.
- aligned=False means no confirmed structural driver.

When aligned=True:
- State clearly what channel or customer behavior is driving movement.
- Indicate whether it appears structural (based on correlation strength).

When aligned=False:
- Say there is no confirmed structural driver.
- Do NOT speculate.

Risk interpretation:
- If competitive churn aligned=True → indicates market share pressure.
- If retail adds aligned=True → store channel is supporting growth.
- If port-in adds aligned=True → competitive capture is supporting growth.

--------------------------------------------
TONE + NARRATION RULES (CRITICAL)
--------------------------------------------

- Output length: 2–3 sentences TOTAL.
- Style: executive-friendly, calm, factual.
- Never restate raw JSON or list all metrics.
- Reference only 1–2 key signals per segment.
- Never use internal variable names like "retail_add_growth".
- Never claim causation; prefer "is consistent with" or "appears aligned with".
- If data missing, say "insufficient data."
- Do NOT mention forecasts.
- Do NOT invent thresholds or time ranges beyond "recent."

--------------------------------------------
ONE EXAMPLE (INPUT + EXPECTED OUTPUT STYLE)
--------------------------------------------

Example input JSON:

{
  "add_churn_driver_analysis": {
    "postpaid": {
      "adds": {
        "driver": "retail_add_growth",
        "aligned": true,
        "confidence": "high",
        "metrics": {
          "corr_latest": 0.99,
          "share_recent_avg": 0.55
        }
      },
      "churn": {
        "driver": "competitive_churn_spike",
        "aligned": true,
        "confidence": "medium",
        "metrics": {
          "corr_latest": 0.93,
          "share_recent_avg": 0.40
        }
      }
    },
    "prepaid": {
      "adds": {
        "driver": "portin_add_growth",
        "aligned": true,
        "confidence": "high",
        "metrics": {
          "corr_latest": 0.98,
          "share_recent_avg": 0.50
        }
      }
    }
  }
}

Example narration (2–3 sentences total):

“Postpaid acquisition momentum appears closely aligned with retail channel performance, indicating store-based sales are supporting current growth. At the same time, a meaningful share of churn is competitive in nature, suggesting ongoing market share pressure. Prepaid growth is also being supported by customers switching from competitors, pointing to sustained competitive capture.”

Only use this example for style guidance. Never introduce numbers not present in the provided JSON.
"""

import json
from typing import Any, Dict


def _safe_json(obj: Any) -> str:
    """Serialize context safely for prompting."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def generate_add_churn_driver_narrative(
    driver_context: Dict[str, Any],
    user_query: str = "Summarize the add and churn drivers.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 180,
) -> str:
    """
    Narrate add/churn drivers ONLY.

    Parameters
    ----------
    driver_context : dict
        Expected to be the output of detect_add_churn_drivers(...)
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

Add/Churn driver output JSON (interpret ONLY this):
{context_str}

Write 2–3 sentences total. No bullet lists. No extra sections.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ADD_CHURN_DRIVER_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content.strip()
