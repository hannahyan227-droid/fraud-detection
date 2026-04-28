from __future__ import annotations

from typing import Any, Dict, Optional

# Merchant categories with disproportionately high fraud rates because payouts
# are hard to reverse after a chargeback (gift card PINs, crypto wallets, wires).
_HIGH_RISK_MERCHANTS = {"gift_cards", "crypto", "wire_transfer"}

# These categories attract fraud but are less irreversible than the top tier.
_MEDIUM_RISK_MERCHANTS = {"electronics", "jewelry", "travel"}


def _num(val: Any, default: Optional[float] = 0) -> Optional[float]:
    """Return a numeric value, mapping None and NaN to *default*.

    Pass default=None when the caller needs to distinguish "genuinely absent /
    unknown" from zero — e.g. account_age_days where 0 is a valid value and
    NaN (pandas join miss) should be treated as unknown rather than brand-new.
    """
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f   # f != f is True only for NaN
    except (TypeError, ValueError):
        return default


def score_transaction(tx: Dict[str, Any]) -> int:
    """Return a fraud risk score from 0 to 100. Higher means riskier.

    Each rule adds points for a signal that empirically correlates with fraud.
    The final score is clamped so callers always receive a value in [0, 100].
    New optional fields (account_age_days, kyc_level, merchant_category) are
    read via .get() so existing call-sites that don't supply them still work.
    """
    score = 0

    # --- Device risk score ---
    # Third-party device fingerprinting assigns scores based on known-bad
    # devices, proxies, and emulators. High scores are a strong fraud signal.
    device_risk = _num(tx.get("device_risk_score"))
    if device_risk >= 70:
        score += 25   # High-risk device: likely a flagged or spoofed device
    elif device_risk >= 40:
        score += 10   # Elevated risk: anonymising software or unusual config

    # --- International transactions ---
    # Cross-border transactions have materially higher fraud rates because
    # they complicate dispute resolution and frequently mismatch cardholder
    # location data.
    if tx.get("is_international") == 1:
        score += 15

    # --- Transaction amount ---
    # Larger amounts drive higher loss exposure and correlate with targeted
    # account-takeover fraud rather than opportunistic small-amount abuse.
    amount = _num(tx.get("amount_usd"))
    if amount >= 1000:
        score += 25
    elif amount >= 500:
        score += 10

    # --- Transaction velocity (card testing / account takeover signal) ---
    # Fraudsters probe stolen credentials with a burst of transactions to
    # validate a card before a large purchase. Six or more transactions in
    # 24 hours is well above normal spending behaviour.
    velocity = _num(tx.get("velocity_24h"))
    if velocity >= 6:
        score += 20   # Burst pattern consistent with card testing
    elif velocity >= 3:
        score += 5    # Mildly elevated, warrants light upweight

    # --- Failed login attempts ---
    # Multiple failed logins immediately before a transaction strongly suggest
    # an account takeover attempt where the attacker is guessing credentials.
    failed_logins = _num(tx.get("failed_logins_24h"))
    if failed_logins >= 5:
        score += 20
    elif failed_logins >= 2:
        score += 10

    # --- Prior chargeback history ---
    # Past chargebacks are the strongest single predictor of future fraud.
    # A repeat offender pattern (2+) warrants a heavier penalty than a single
    # prior incident.
    prior_cbs = _num(tx.get("prior_chargebacks"))
    if prior_cbs >= 2:
        score += 20
    elif prior_cbs == 1:
        score += 5

    # --- Account age ---
    # Newly created accounts are disproportionately used in fraud because
    # attackers open synthetic or stolen-identity accounts to bypass velocity
    # checks on existing accounts.  Missing account_age_days (e.g. guest
    # checkout) or NaN (no matching account row in the join) contributes no
    # points; default=None distinguishes "unknown" from "zero days old".
    account_age = _num(tx.get("account_age_days"), default=None)
    if account_age is not None:
        if account_age < 30:
            score += 20   # Very new — high synthetic/stolen-identity risk
        elif account_age < 90:
            score += 10   # Still within the elevated-risk window

    # --- KYC (Know Your Customer) level ---
    # Basic KYC means the account owner's identity has not been fully verified.
    # Fraudsters deliberately stop short of full verification to avoid leaving
    # a paper trail.  "none" (if ever encountered) is treated most harshly.
    kyc = tx.get("kyc_level", "")
    if kyc == "none":
        score += 25
    elif kyc == "basic":
        score += 15

    # --- Merchant category ---
    # Certain merchant types are disproportionately targeted because proceeds
    # are quickly liquidated and chargebacks rarely recover the loss.
    merchant = tx.get("merchant_category", "")
    if merchant in _HIGH_RISK_MERCHANTS:
        score += 20
    elif merchant in _MEDIUM_RISK_MERCHANTS:
        score += 10

    return max(0, min(score, 100))


def label_risk(score: int) -> str:
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"
