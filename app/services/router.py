"""
router.py — RouteLLM: Smart model routing
==========================================
Routes each prompt to either the small (cheap) or large (expensive) model
using a two-stage strategy:

    1. Rule-based check  — keyword matching + code detection (instant, high confidence)
    2. ML model fallback — trained classifier (router_model.joblib) for ambiguous cases

The router returns a routing decision dict AND the matching LLMConfig so the
caller doesn't need to do any mapping.

Escalation note (future):
    When the ML model is uncertain (probability 0.45–0.65), it returns
    route="small_then_escalate". For now we route to small. A future version
    could try small first, evaluate quality, and auto-retry with large if
    the response is poor.
"""

import re
from pathlib import Path

import joblib

from app.core.config import LLMConfig, small_model_config, large_model_config


# ── Load the trained routing model (once at import time) ──────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "router_model.joblib"
model = joblib.load(MODEL_PATH)


# ── Keyword lists ─────────────────────────────────────────────────────────────
EASY_KEYWORDS = [
    "rewrite",
    "make this polite",
    "summarize",
    "translate",
    "extract",
    "convert to json",
    "fix grammar",
    "paraphrase",
]

HARD_KEYWORDS = [
    "debug",
    "race condition",
    "distributed",
    "optimize",
    "architecture",
    "legal",
    "medical",
    "tax",
    "financial strategy",
    "prove",
    "derive",
    "multi-step",
    "reason step by step",
]

# ── Code detection patterns ───────────────────────────────────────────────────
_CODE_PATTERNS = [
    r"```",
    r"\bdef\b",
    r"\bclass\b",
    r"\bfunction\b",
    r"\bconst\b",
    r"\blet\b",
    r"\bvar\b",
    r"\bimport\b",
    r"\bSELECT\b",
    r"\basync\b",
    r"\bawait\b",
]


def has_code(prompt: str) -> bool:
    """Check if the prompt contains code-like patterns."""
    return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in _CODE_PATTERNS)


# ── Stage 1: Rule-based routing ───────────────────────────────────────────────

def rule_based_route(prompt: str) -> dict | None:
    """
    Fast keyword/code matching. Returns a routing decision dict or None
    if no rule matched (falls through to ML).
    """
    text = prompt.lower()

    if any(keyword in text for keyword in HARD_KEYWORDS):
        return {
            "route": "large",
            "confidence": 0.95,
            "reason": "Matched hard-task keyword rule",
        }

    if has_code(prompt):
        return {
            "route": "large",
            "confidence": 0.90,
            "reason": "Prompt contains code",
        }

    if any(keyword in text for keyword in EASY_KEYWORDS):
        return {
            "route": "small",
            "confidence": 0.90,
            "reason": "Matched easy-task keyword rule",
        }

    return None


# ── Stage 2: ML model routing ────────────────────────────────────────────────

def ml_route(prompt: str) -> dict:
    """
    Use the trained ML classifier to predict whether the small model
    can handle this prompt successfully.
    """
    probability_small_success = model.predict_proba([prompt])[0][1]

    if probability_small_success >= 0.65:
        return {
            "route": "small",
            "confidence": float(probability_small_success),
            "reason": "ML router: small model likely succeeds",
        }

    elif probability_small_success <= 0.45:
        return {
            "route": "large",
            "confidence": float(1 - probability_small_success),
            "reason": "ML router: small model likely fails",
        }

    else:
        # Uncertain zone → use small for now (escalation deferred)
        return {
            "route": "small_then_escalate",
            "confidence": float(probability_small_success),
            "reason": "ML router: uncertain — using small model (escalation deferred)",
        }


# ── Combined router ──────────────────────────────────────────────────────────

def route_prompt(prompt: str) -> dict:
    """
    Route a prompt: rules first, ML fallback.
    Returns dict with keys: route, confidence, reason.
    """
    rule_result = rule_based_route(prompt)
    if rule_result:
        return rule_result
    return ml_route(prompt)


# ── Convenience function for routes.py ────────────────────────────────────────

def get_model_config(prompt: str) -> tuple[LLMConfig, dict]:
    """
    One-call helper: routes the prompt and returns the right LLMConfig.

    Returns:
        (config, decision) where:
            config   — the LLMConfig to use (small or large)
            decision — routing metadata dict with route, confidence, reason

    Usage in routes.py:
        config, decision = get_model_config(prompt)
        async for token in async_stream_llm(
            base_url=config.base_url,
            api_key=config.api_key,
            model_name=config.model_name,
            prompt=prompt,
        ):
            yield token
    """
    decision = route_prompt(prompt)

    if decision["route"] == "large":
        config = large_model_config
    else:
        # "small" and "small_then_escalate" both go to small for now
        config = small_model_config

    return config, decision
