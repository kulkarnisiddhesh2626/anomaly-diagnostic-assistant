"""Central configuration: dimension vocabularies, file paths, and LLM settings.

Everything that another module might want to tweak lives here so the rest of the
codebase stays declarative and easy to reason about.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
INJECTED_TRUTH_JSON = DATA_DIR / "injected_anomalies.json"

# --------------------------------------------------------------------------- #
# Synthetic-data dimensions
# --------------------------------------------------------------------------- #
# Merchant Category Codes (issuer-relevant subset) with human labels.
MCCS: dict[str, str] = {
    "5411": "Grocery Stores",
    "5812": "Restaurants",
    "5999": "Misc Retail",
    "4829": "Money Transfer",
    "7995": "Gambling",
}

# Issuing/acquiring countries. "domestic" is defined relative to ISSUER_COUNTRY.
COUNTRIES: list[str] = ["US", "GB", "IN", "DE", "BR"]
ISSUER_COUNTRY = "US"  # cards are issued here; everything else is cross-border

# (channel, card_present_flag, allowed auth_types)
CHANNELS: dict[str, tuple[bool, list[str]]] = {
    "ecom": (False, ["3DS", "non-3DS"]),       # card-not-present
    "pos": (True, ["non-3DS"]),                # card-present, chip/swipe
    "contactless": (True, ["non-3DS"]),        # card-present, tap
}

# Decline reason codes (ISO-8583-flavoured) plus the synthetic APPROVED bucket.
APPROVED_CODE = "00_APPROVED"
DECLINE_REASON_CODES: dict[str, str] = {
    "05_DO_NOT_HONOR": "Do Not Honor (issuer risk decline)",
    "51_INSUFFICIENT_FUNDS": "Insufficient Funds",
    "14_INVALID_CARD": "Invalid Card Number",
    "59_SUSPECTED_FRAUD": "Suspected Fraud",
    "91_ISSUER_UNAVAILABLE": "Issuer/Processor Unavailable (technical)",
    "N7_3DS_FAILURE": "3DS Authentication Failure",
}
ALL_OUTCOME_CODES: list[str] = [APPROVED_CODE] + list(DECLINE_REASON_CODES)

# --------------------------------------------------------------------------- #
# LLM settings (read from environment; safe defaults)
# --------------------------------------------------------------------------- #
# Default to a fast, capable model. Override with ANOMALY_LLM_MODEL.
# Cheaper option for high volume: "claude-haiku-4-5".
DEFAULT_MODEL = os.environ.get("ANOMALY_LLM_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
LLM_MAX_TOKENS = int(os.environ.get("ANOMALY_LLM_MAX_TOKENS", "1200"))
# Low temperature: we want grounded, repeatable diagnostics, not creativity.
LLM_TEMPERATURE = float(os.environ.get("ANOMALY_LLM_TEMPERATURE", "0.2"))

# --------------------------------------------------------------------------- #
# Detection defaults (overridable from the UI)
# --------------------------------------------------------------------------- #
ROBUST_Z_THRESHOLD = 5.0     # MAD/proportion-scaled z above which a point is anomalous
MIN_HOURLY_VOLUME = 30       # ignore buckets too small to be statistically meaningful
EVENT_MERGE_GAP_HOURS = 2    # contiguous flagged hours within this gap = one event
