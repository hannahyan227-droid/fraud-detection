from risk_rules import label_risk, score_transaction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A baseline transaction that is low-risk across every signal so individual
# tests can override just the field they are exercising.
_LOW_RISK_BASE = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 50,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
    "account_age_days": 500,
    "kyc_level": "full",
    "merchant_category": "grocery",
}


def _tx(**overrides):
    """Return a copy of the low-risk baseline with the given fields replaced."""
    return {**_LOW_RISK_BASE, **overrides}


# ---------------------------------------------------------------------------
# Original tests (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 1200,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    assert score_transaction(tx) >= 25


# ---------------------------------------------------------------------------
# Regression tests for the four previously inverted rules
# ---------------------------------------------------------------------------

def test_high_device_risk_raises_score():
    """device_risk_score >= 70 must add points, not subtract them."""
    low = score_transaction(_tx(device_risk_score=10))
    high = score_transaction(_tx(device_risk_score=80))
    assert high > low, "high device risk should increase the score"


def test_international_raises_score():
    """is_international == 1 must add points, not subtract them."""
    domestic = score_transaction(_tx(is_international=0))
    intl = score_transaction(_tx(is_international=1))
    assert intl > domestic, "international transaction should increase the score"


def test_high_velocity_raises_score():
    """velocity_24h >= 6 must add points, not subtract them."""
    low_v = score_transaction(_tx(velocity_24h=1))
    high_v = score_transaction(_tx(velocity_24h=8))
    assert high_v > low_v, "high transaction velocity should increase the score"


def test_prior_chargebacks_raise_score():
    """prior_chargebacks must add points, not subtract them."""
    clean = score_transaction(_tx(prior_chargebacks=0))
    one_cb = score_transaction(_tx(prior_chargebacks=1))
    two_cb = score_transaction(_tx(prior_chargebacks=2))
    assert one_cb > clean, "one prior chargeback should increase the score"
    assert two_cb > one_cb, "two prior chargebacks should score higher than one"


# ---------------------------------------------------------------------------
# Tests for the device risk mid-tier
# ---------------------------------------------------------------------------

def test_medium_device_risk_adds_points():
    low = score_transaction(_tx(device_risk_score=10))
    med = score_transaction(_tx(device_risk_score=55))
    assert med > low


# ---------------------------------------------------------------------------
# Tests for new signals: account age
# ---------------------------------------------------------------------------

def test_very_new_account_raises_score():
    old = score_transaction(_tx(account_age_days=500))
    new = score_transaction(_tx(account_age_days=15))
    assert new > old, "accounts < 30 days old should score higher"


def test_recent_account_raises_score_less_than_very_new():
    old = score_transaction(_tx(account_age_days=500))
    recent = score_transaction(_tx(account_age_days=60))
    brand_new = score_transaction(_tx(account_age_days=10))
    assert old < recent < brand_new


def test_missing_account_age_does_not_crash():
    tx = {k: v for k, v in _LOW_RISK_BASE.items() if k != "account_age_days"}
    assert isinstance(score_transaction(tx), int)


# ---------------------------------------------------------------------------
# Tests for new signals: KYC level
# ---------------------------------------------------------------------------

def test_basic_kyc_raises_score():
    full = score_transaction(_tx(kyc_level="full"))
    basic = score_transaction(_tx(kyc_level="basic"))
    assert basic > full


def test_no_kyc_raises_score_more_than_basic():
    basic = score_transaction(_tx(kyc_level="basic"))
    none_ = score_transaction(_tx(kyc_level="none"))
    assert none_ > basic


# ---------------------------------------------------------------------------
# Tests for new signals: merchant category
# ---------------------------------------------------------------------------

def test_high_risk_merchant_raises_score():
    grocery = score_transaction(_tx(merchant_category="grocery"))
    gift = score_transaction(_tx(merchant_category="gift_cards"))
    crypto = score_transaction(_tx(merchant_category="crypto"))
    assert gift > grocery
    assert crypto > grocery


def test_medium_risk_merchant_raises_score_less_than_high():
    low = score_transaction(_tx(merchant_category="grocery"))
    med = score_transaction(_tx(merchant_category="electronics"))
    high = score_transaction(_tx(merchant_category="gift_cards"))
    assert low < med < high


def test_unknown_merchant_category_does_not_raise():
    score = score_transaction(_tx(merchant_category="pet_supplies"))
    assert isinstance(score, int)


# ---------------------------------------------------------------------------
# Missing / NaN value robustness
# ---------------------------------------------------------------------------

def test_completely_empty_transaction_does_not_crash():
    """score_transaction must not raise even with an empty dict."""
    assert isinstance(score_transaction({}), int)


def test_none_values_treated_as_zero():
    """None for numeric fields should not raise and should score as 0 risk."""
    tx = {
        "device_risk_score": None,
        "is_international": None,
        "amount_usd": None,
        "velocity_24h": None,
        "failed_logins_24h": None,
        "prior_chargebacks": None,
        "account_age_days": None,
    }
    assert score_transaction(tx) == 0


def test_nan_numeric_fields_do_not_crash():
    """NaN values (as produced by a pandas left-join miss) must not raise."""
    import math
    nan = float("nan")
    tx = {
        "device_risk_score": nan,
        "is_international": 0,
        "amount_usd": nan,
        "velocity_24h": nan,
        "failed_logins_24h": nan,
        "prior_chargebacks": nan,
        "account_age_days": nan,
    }
    result = score_transaction(tx)
    assert isinstance(result, int)
    assert not math.isnan(result)


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------

def test_score_never_exceeds_100():
    """All signals at maximum should clamp to 100, not overflow."""
    tx = {
        "device_risk_score": 90,
        "is_international": 1,
        "amount_usd": 5000,
        "velocity_24h": 10,
        "failed_logins_24h": 8,
        "prior_chargebacks": 3,
        "account_age_days": 5,
        "kyc_level": "none",
        "merchant_category": "gift_cards",
    }
    assert score_transaction(tx) == 100


def test_score_never_below_zero():
    """A minimum-signal transaction should score 0, never negative."""
    tx = {
        "device_risk_score": 0,
        "is_international": 0,
        "amount_usd": 1,
        "velocity_24h": 0,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
        "account_age_days": 1000,
        "kyc_level": "full",
        "merchant_category": "grocery",
    }
    assert score_transaction(tx) == 0


# ---------------------------------------------------------------------------
# Known confirmed-fraud profile (transaction 50003 from sample data)
# ---------------------------------------------------------------------------

def test_confirmed_fraud_profile_labels_high():
    """Transaction 50003: $1,250 gift-card purchase, device_risk=81,
    international, velocity=6, failed_logins=5, basic KYC.
    This resulted in a confirmed $1,250 chargeback and must score 'high'."""
    tx = {
        "device_risk_score": 81,
        "is_international": 1,
        "amount_usd": 1250,
        "velocity_24h": 6,
        "failed_logins_24h": 5,
        "prior_chargebacks": 0,
        "account_age_days": 85,
        "kyc_level": "basic",
        "merchant_category": "gift_cards",
    }
    assert label_risk(score_transaction(tx)) == "high"
