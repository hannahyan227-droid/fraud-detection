"""
Comprehensive test suite for risk_rules.py.

Organisation
------------
1.  Helpers / shared fixtures
2.  _num() helper — unit tests
3.  label_risk() — unit tests
4.  score_transaction() — per-rule unit tests (one section per signal)
5.  Boundary conditions across every threshold
6.  Regression tests for the four previously inverted rules
7.  Missing-field / None / NaN robustness
8.  Score clamping (floor and ceiling)
9.  Integration tests — fraudulent transaction profiles
10. Integration tests — legitimate transaction profiles
11. Integration test — known confirmed-fraud row from sample data
"""

import math

import pytest

from risk_rules import _num, label_risk, score_transaction

# ---------------------------------------------------------------------------
# 1. Helpers / shared fixtures
# ---------------------------------------------------------------------------

# Every signal at its minimum contribution: score_transaction(_ZERO) == 0.
_ZERO = {
    "device_risk_score": 0,
    "is_international": 0,
    "amount_usd": 0,
    "velocity_24h": 0,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
    "account_age_days": 1000,   # old account → 0 pts
    "kyc_level": "full",        # full KYC   → 0 pts
    "merchant_category": "grocery",  # low risk  → 0 pts
}


def _tx(**overrides):
    """Return a zero-risk baseline with the given fields replaced."""
    return {**_ZERO, **overrides}


# ---------------------------------------------------------------------------
# 2. _num() helper — unit tests
# ---------------------------------------------------------------------------

class TestNum:
    def test_none_returns_zero_by_default(self):
        assert _num(None) == 0

    def test_nan_returns_zero_by_default(self):
        assert _num(float("nan")) == 0

    def test_valid_integer(self):
        assert _num(42) == 42.0

    def test_valid_float(self):
        assert _num(3.14) == pytest.approx(3.14)

    def test_zero_stays_zero(self):
        assert _num(0) == 0

    def test_numeric_string_is_parsed(self):
        assert _num("99") == 99.0

    def test_non_numeric_string_returns_default(self):
        assert _num("abc") == 0

    def test_none_with_none_default_returns_none(self):
        assert _num(None, default=None) is None

    def test_nan_with_none_default_returns_none(self):
        assert _num(float("nan"), default=None) is None

    def test_valid_value_ignores_default(self):
        assert _num(5, default=None) == 5.0

    def test_negative_value_preserved(self):
        # Negative ages/amounts are unusual but _num should not alter the sign.
        assert _num(-1) == -1.0


# ---------------------------------------------------------------------------
# 3. label_risk() — unit tests
# ---------------------------------------------------------------------------

class TestLabelRisk:
    # Original preserved tests
    def test_low_mid_high_examples(self):
        assert label_risk(10) == "low"
        assert label_risk(35) == "medium"
        assert label_risk(75) == "high"

    # Exact boundary: score == 30 → medium (not low)
    def test_boundary_low_to_medium(self):
        assert label_risk(29) == "low"
        assert label_risk(30) == "medium"

    # Exact boundary: score == 60 → high (not medium)
    def test_boundary_medium_to_high(self):
        assert label_risk(59) == "medium"
        assert label_risk(60) == "high"

    def test_score_zero_is_low(self):
        assert label_risk(0) == "low"

    def test_score_100_is_high(self):
        assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# 4a. score_transaction() — device_risk_score
# ---------------------------------------------------------------------------

class TestDeviceRisk:
    """device_risk_score: +25 if >= 70, +10 if >= 40, 0 otherwise."""

    def test_below_medium_threshold_is_zero(self):
        assert score_transaction(_tx(device_risk_score=0)) == 0
        assert score_transaction(_tx(device_risk_score=39)) == 0

    def test_at_medium_threshold_adds_10(self):
        assert score_transaction(_tx(device_risk_score=40)) == 10

    def test_between_thresholds_adds_10(self):
        assert score_transaction(_tx(device_risk_score=55)) == 10
        assert score_transaction(_tx(device_risk_score=69)) == 10

    def test_at_high_threshold_adds_25(self):
        assert score_transaction(_tx(device_risk_score=70)) == 25

    def test_above_high_threshold_still_adds_25(self):
        assert score_transaction(_tx(device_risk_score=100)) == 25

    def test_high_risk_strictly_greater_than_medium(self):
        assert score_transaction(_tx(device_risk_score=70)) > score_transaction(_tx(device_risk_score=40))


# ---------------------------------------------------------------------------
# 4b. score_transaction() — is_international
# ---------------------------------------------------------------------------

class TestInternational:
    """is_international: +15 if 1, 0 otherwise."""

    def test_domestic_adds_zero(self):
        assert score_transaction(_tx(is_international=0)) == 0

    def test_international_adds_15(self):
        assert score_transaction(_tx(is_international=1)) == 15

    def test_exact_point_value(self):
        diff = (
            score_transaction(_tx(is_international=1))
            - score_transaction(_tx(is_international=0))
        )
        assert diff == 15


# ---------------------------------------------------------------------------
# 4c. score_transaction() — amount_usd
# ---------------------------------------------------------------------------

class TestAmountUsd:
    """amount_usd: +25 if >= 1000, +10 if >= 500, 0 otherwise."""

    def test_below_medium_threshold_is_zero(self):
        assert score_transaction(_tx(amount_usd=0)) == 0
        assert score_transaction(_tx(amount_usd=499)) == 0
        assert score_transaction(_tx(amount_usd=499.99)) == 0

    def test_at_medium_threshold_adds_10(self):
        assert score_transaction(_tx(amount_usd=500)) == 10

    def test_between_thresholds_adds_10(self):
        assert score_transaction(_tx(amount_usd=750)) == 10
        assert score_transaction(_tx(amount_usd=999.99)) == 10

    def test_at_large_threshold_adds_25(self):
        assert score_transaction(_tx(amount_usd=1000)) == 25

    def test_above_large_threshold_still_adds_25(self):
        assert score_transaction(_tx(amount_usd=50000)) == 25

    # Original preserved test (standalone dict, no account fields)
    def test_large_amount_adds_risk_original(self):
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
# 4d. score_transaction() — velocity_24h
# ---------------------------------------------------------------------------

class TestVelocity:
    """velocity_24h: +20 if >= 6, +5 if >= 3, 0 otherwise."""

    def test_below_medium_threshold_is_zero(self):
        assert score_transaction(_tx(velocity_24h=0)) == 0
        assert score_transaction(_tx(velocity_24h=2)) == 0

    def test_at_medium_threshold_adds_5(self):
        assert score_transaction(_tx(velocity_24h=3)) == 5

    def test_between_thresholds_adds_5(self):
        assert score_transaction(_tx(velocity_24h=5)) == 5

    def test_at_high_threshold_adds_20(self):
        assert score_transaction(_tx(velocity_24h=6)) == 20

    def test_above_high_threshold_still_adds_20(self):
        assert score_transaction(_tx(velocity_24h=100)) == 20


# ---------------------------------------------------------------------------
# 4e. score_transaction() — failed_logins_24h
# ---------------------------------------------------------------------------

class TestFailedLogins:
    """failed_logins_24h: +20 if >= 5, +10 if >= 2, 0 otherwise."""

    def test_below_medium_threshold_is_zero(self):
        assert score_transaction(_tx(failed_logins_24h=0)) == 0
        assert score_transaction(_tx(failed_logins_24h=1)) == 0

    def test_at_medium_threshold_adds_10(self):
        assert score_transaction(_tx(failed_logins_24h=2)) == 10

    def test_between_thresholds_adds_10(self):
        assert score_transaction(_tx(failed_logins_24h=4)) == 10

    def test_at_high_threshold_adds_20(self):
        assert score_transaction(_tx(failed_logins_24h=5)) == 20

    def test_above_high_threshold_still_adds_20(self):
        assert score_transaction(_tx(failed_logins_24h=20)) == 20


# ---------------------------------------------------------------------------
# 4f. score_transaction() — prior_chargebacks
# ---------------------------------------------------------------------------

class TestPriorChargebacks:
    """prior_chargebacks: +20 if >= 2, +5 if == 1, 0 if 0."""

    def test_zero_chargebacks_adds_nothing(self):
        assert score_transaction(_tx(prior_chargebacks=0)) == 0

    def test_one_chargeback_adds_5(self):
        assert score_transaction(_tx(prior_chargebacks=1)) == 5

    def test_two_chargebacks_adds_20(self):
        assert score_transaction(_tx(prior_chargebacks=2)) == 20

    def test_many_chargebacks_still_adds_20(self):
        assert score_transaction(_tx(prior_chargebacks=10)) == 20

    def test_two_chargebacks_strictly_greater_than_one(self):
        assert (
            score_transaction(_tx(prior_chargebacks=2))
            > score_transaction(_tx(prior_chargebacks=1))
        )


# ---------------------------------------------------------------------------
# 4g. score_transaction() — account_age_days
# ---------------------------------------------------------------------------

class TestAccountAge:
    """account_age_days: +20 if < 30, +10 if < 90, 0 otherwise."""

    def test_very_new_account_adds_20(self):
        assert score_transaction(_tx(account_age_days=1)) == 20
        assert score_transaction(_tx(account_age_days=29)) == 20

    def test_at_young_boundary_adds_10(self):
        # Exactly 30 days: no longer "very new", still in the elevated window.
        assert score_transaction(_tx(account_age_days=30)) == 10

    def test_recent_account_adds_10(self):
        assert score_transaction(_tx(account_age_days=60)) == 10
        assert score_transaction(_tx(account_age_days=89)) == 10

    def test_at_mature_boundary_adds_zero(self):
        # Exactly 90 days: out of the elevated-risk window.
        assert score_transaction(_tx(account_age_days=90)) == 0

    def test_old_account_adds_nothing(self):
        assert score_transaction(_tx(account_age_days=365)) == 0
        assert score_transaction(_tx(account_age_days=1000)) == 0

    def test_missing_field_adds_nothing(self):
        # Key absent entirely — no points, no crash.
        tx = {k: v for k, v in _ZERO.items() if k != "account_age_days"}
        assert score_transaction(tx) == 0

    def test_nan_account_age_adds_nothing(self):
        # NaN comes from a pandas left-join miss; should be treated as unknown,
        # not as age=0 which would incorrectly fire the <30 rule.
        assert score_transaction(_tx(account_age_days=float("nan"))) == 0

    def test_none_account_age_adds_nothing(self):
        assert score_transaction(_tx(account_age_days=None)) == 0


# ---------------------------------------------------------------------------
# 4h. score_transaction() — kyc_level
# ---------------------------------------------------------------------------

class TestKycLevel:
    """kyc_level: +25 if 'none', +15 if 'basic', 0 for 'full' or unknown."""

    def test_full_kyc_adds_nothing(self):
        assert score_transaction(_tx(kyc_level="full")) == 0

    def test_basic_kyc_adds_15(self):
        assert score_transaction(_tx(kyc_level="basic")) == 15

    def test_no_kyc_adds_25(self):
        assert score_transaction(_tx(kyc_level="none")) == 25

    def test_unknown_kyc_value_adds_nothing(self):
        assert score_transaction(_tx(kyc_level="pending")) == 0
        assert score_transaction(_tx(kyc_level="")) == 0

    def test_missing_kyc_field_adds_nothing(self):
        tx = {k: v for k, v in _ZERO.items() if k != "kyc_level"}
        assert score_transaction(tx) == 0

    def test_no_kyc_strictly_greater_than_basic(self):
        assert (
            score_transaction(_tx(kyc_level="none"))
            > score_transaction(_tx(kyc_level="basic"))
        )


# ---------------------------------------------------------------------------
# 4i. score_transaction() — merchant_category
# ---------------------------------------------------------------------------

class TestMerchantCategory:
    """High-risk merchants: +20. Medium-risk: +10. Others: 0."""

    def test_low_risk_merchant_adds_nothing(self):
        assert score_transaction(_tx(merchant_category="grocery")) == 0
        assert score_transaction(_tx(merchant_category="streaming")) == 0
        assert score_transaction(_tx(merchant_category="unknown_category")) == 0

    def test_medium_risk_merchants_add_10(self):
        for cat in ("electronics", "jewelry", "travel"):
            assert score_transaction(_tx(merchant_category=cat)) == 10, cat

    def test_high_risk_merchants_add_20(self):
        for cat in ("gift_cards", "crypto", "wire_transfer"):
            assert score_transaction(_tx(merchant_category=cat)) == 20, cat

    def test_high_risk_strictly_greater_than_medium(self):
        assert (
            score_transaction(_tx(merchant_category="gift_cards"))
            > score_transaction(_tx(merchant_category="electronics"))
        )

    def test_missing_merchant_field_adds_nothing(self):
        tx = {k: v for k, v in _ZERO.items() if k != "merchant_category"}
        assert score_transaction(tx) == 0


# ---------------------------------------------------------------------------
# 5. Boundary conditions across every threshold (additive combination checks)
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_score_increases_strictly_at_each_device_threshold(self):
        s39 = score_transaction(_tx(device_risk_score=39))
        s40 = score_transaction(_tx(device_risk_score=40))
        s69 = score_transaction(_tx(device_risk_score=69))
        s70 = score_transaction(_tx(device_risk_score=70))
        assert s39 < s40 <= s69 < s70

    def test_score_increases_strictly_at_each_amount_threshold(self):
        s499 = score_transaction(_tx(amount_usd=499))
        s500 = score_transaction(_tx(amount_usd=500))
        s999 = score_transaction(_tx(amount_usd=999))
        s1000 = score_transaction(_tx(amount_usd=1000))
        assert s499 < s500 <= s999 < s1000

    def test_score_increases_strictly_at_each_velocity_threshold(self):
        s2 = score_transaction(_tx(velocity_24h=2))
        s3 = score_transaction(_tx(velocity_24h=3))
        s5 = score_transaction(_tx(velocity_24h=5))
        s6 = score_transaction(_tx(velocity_24h=6))
        assert s2 < s3 <= s5 < s6

    def test_score_increases_strictly_at_each_login_threshold(self):
        s1 = score_transaction(_tx(failed_logins_24h=1))
        s2 = score_transaction(_tx(failed_logins_24h=2))
        s4 = score_transaction(_tx(failed_logins_24h=4))
        s5 = score_transaction(_tx(failed_logins_24h=5))
        assert s1 < s2 <= s4 < s5

    def test_score_increases_strictly_at_chargeback_thresholds(self):
        s0 = score_transaction(_tx(prior_chargebacks=0))
        s1 = score_transaction(_tx(prior_chargebacks=1))
        s2 = score_transaction(_tx(prior_chargebacks=2))
        assert s0 < s1 < s2

    def test_score_increases_strictly_at_each_age_threshold(self):
        s90 = score_transaction(_tx(account_age_days=90))
        s89 = score_transaction(_tx(account_age_days=89))
        s30 = score_transaction(_tx(account_age_days=30))
        s29 = score_transaction(_tx(account_age_days=29))
        assert s90 < s89
        assert s89 == s30   # both in the 30–89 band
        assert s30 < s29

    def test_kyc_ordering(self):
        full = score_transaction(_tx(kyc_level="full"))
        basic = score_transaction(_tx(kyc_level="basic"))
        none_ = score_transaction(_tx(kyc_level="none"))
        assert full < basic < none_

    def test_merchant_ordering(self):
        low = score_transaction(_tx(merchant_category="grocery"))
        med = score_transaction(_tx(merchant_category="electronics"))
        high = score_transaction(_tx(merchant_category="gift_cards"))
        assert low < med < high


# ---------------------------------------------------------------------------
# 6. Regression tests — four previously inverted rules
# ---------------------------------------------------------------------------

class TestInvertedRuleRegressions:
    """Each rule previously subtracted points; verify it now adds them."""

    def test_high_device_risk_adds_not_subtracts(self):
        low = score_transaction(_tx(device_risk_score=10))
        high = score_transaction(_tx(device_risk_score=80))
        assert high > low

    def test_international_adds_not_subtracts(self):
        domestic = score_transaction(_tx(is_international=0))
        intl = score_transaction(_tx(is_international=1))
        assert intl > domestic

    def test_high_velocity_adds_not_subtracts(self):
        low_v = score_transaction(_tx(velocity_24h=1))
        high_v = score_transaction(_tx(velocity_24h=8))
        assert high_v > low_v

    def test_prior_chargebacks_adds_not_subtracts(self):
        clean = score_transaction(_tx(prior_chargebacks=0))
        one = score_transaction(_tx(prior_chargebacks=1))
        two = score_transaction(_tx(prior_chargebacks=2))
        assert clean < one < two

    def test_combined_worst_case_scores_high_not_low(self):
        # Before the fixes, stacking all four inverted signals would have
        # produced a score of 0 (maximum discount). After, they sum to 100.
        tx = _tx(
            device_risk_score=80,
            is_international=1,
            velocity_24h=8,
            prior_chargebacks=3,
        )
        assert label_risk(score_transaction(tx)) == "high"


# ---------------------------------------------------------------------------
# 7. Missing-field / None / NaN robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_empty_dict_does_not_crash(self):
        result = score_transaction({})
        assert isinstance(result, int)
        assert result == 0

    def test_all_none_numeric_fields_score_zero(self):
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

    def test_nan_numeric_fields_do_not_crash_or_return_nan(self):
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

    def test_nan_account_age_does_not_trigger_new_account_penalty(self):
        # NaN from a join miss must not be treated as age=0 (very new).
        score_no_age = score_transaction({k: v for k, v in _ZERO.items() if k != "account_age_days"})
        score_nan_age = score_transaction(_tx(account_age_days=float("nan")))
        assert score_nan_age == score_no_age == 0

    def test_missing_new_optional_fields_do_not_affect_original_signals(self):
        # Callers that only pass the original six fields still work correctly.
        original_tx = {
            "device_risk_score": 10,
            "is_international": 0,
            "amount_usd": 1200,
            "velocity_24h": 1,
            "failed_logins_24h": 0,
            "prior_chargebacks": 0,
        }
        assert score_transaction(original_tx) == 25  # only amount fires (+25)

    def test_non_numeric_string_for_numeric_field_treated_as_zero(self):
        assert score_transaction(_tx(device_risk_score="high")) == 0

    def test_float_inputs_for_integer_fields_work_correctly(self):
        # pandas may pass float64 values for integer-typed columns.
        assert score_transaction(_tx(prior_chargebacks=2.0)) == 20
        assert score_transaction(_tx(velocity_24h=6.0)) == 20


# ---------------------------------------------------------------------------
# 8. Score clamping
# ---------------------------------------------------------------------------

class TestScoreClamping:
    def test_maximum_all_signals_clamps_to_100(self):
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

    def test_minimum_all_signals_clamps_to_zero(self):
        assert score_transaction(_ZERO) == 0

    def test_result_always_in_valid_range(self):
        cases = [
            _tx(device_risk_score=50, amount_usd=600),
            _tx(velocity_24h=4, failed_logins_24h=3),
            _tx(kyc_level="basic", merchant_category="electronics"),
        ]
        for tx in cases:
            s = score_transaction(tx)
            assert 0 <= s <= 100, f"score {s} out of range for {tx}"


# ---------------------------------------------------------------------------
# 9. Integration tests — fraudulent transaction profiles
# ---------------------------------------------------------------------------

class TestFraudulentProfiles:
    """Multi-signal transactions that should clearly score 'high'."""

    def test_account_takeover_profile(self):
        # Many failed logins, burst velocity, risky device, international.
        tx = _tx(
            device_risk_score=82,
            is_international=1,
            amount_usd=750,
            velocity_24h=9,
            failed_logins_24h=6,
            prior_chargebacks=1,
            merchant_category="electronics",
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_synthetic_identity_fraud_profile(self):
        # Brand-new account, basic KYC, gift-card purchase, high device risk.
        tx = _tx(
            device_risk_score=75,
            amount_usd=1500,
            account_age_days=10,
            kyc_level="basic",
            merchant_category="gift_cards",
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_repeat_fraudster_profile(self):
        # History of chargebacks, crypto purchase, burst velocity, international.
        tx = _tx(
            device_risk_score=60,
            is_international=1,
            velocity_24h=7,
            prior_chargebacks=3,
            merchant_category="crypto",
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_card_testing_profile(self):
        # Very high velocity + international + no-KYC + risky device.
        # Small amount is typical card-testing behaviour.
        tx = _tx(
            device_risk_score=80,
            is_international=1,
            amount_usd=5,
            velocity_24h=10,
            account_age_days=5,
            kyc_level="none",
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_wire_transfer_high_risk_profile(self):
        # Wire transfer from new account, basic KYC, high device risk.
        tx = _tx(
            device_risk_score=72,
            amount_usd=2000,
            account_age_days=20,
            kyc_level="basic",
            merchant_category="wire_transfer",
        )
        assert label_risk(score_transaction(tx)) == "high"


# ---------------------------------------------------------------------------
# 10. Integration tests — legitimate transaction profiles
# ---------------------------------------------------------------------------

class TestLegitimateProfiles:
    """Multi-signal transactions that should score 'low' or at most 'medium'."""

    def test_routine_low_value_grocery_purchase(self):
        # Everyday low-value domestic grocery purchase.
        tx = _tx(
            device_risk_score=8,
            amount_usd=45,
            account_age_days=600,
        )
        assert label_risk(score_transaction(tx)) == "low"

    def test_established_customer_large_purchase(self):
        # Trusted account with full KYC buying electronics at a high price.
        # Should not exceed "medium" — large amount adds some risk but other
        # signals are clean.
        tx = _tx(
            device_risk_score=5,
            amount_usd=1200,
            account_age_days=800,
            merchant_category="electronics",
        )
        score = score_transaction(tx)
        assert label_risk(score) in ("low", "medium")

    def test_new_account_small_purchase_scores_low(self):
        # New account but zero risk elsewhere — should not hit medium just
        # because the account is recent.
        tx = _tx(
            device_risk_score=15,
            amount_usd=30,
            account_age_days=20,
            merchant_category="streaming",
        )
        assert label_risk(score_transaction(tx)) == "low"

    def test_international_travel_purchase_established_account(self):
        # Customer travelling abroad: international + travel merchant raises
        # score to medium, which is expected and correct.
        tx = _tx(
            device_risk_score=12,
            is_international=1,
            amount_usd=650,
            velocity_24h=2,
            account_age_days=1200,
            merchant_category="travel",
        )
        assert label_risk(score_transaction(tx)) in ("low", "medium")

    def test_clean_profile_scores_zero(self):
        # A transaction that fires no rules at all should score exactly 0.
        assert score_transaction(_ZERO) == 0


# ---------------------------------------------------------------------------
# 11. Integration test — known confirmed-fraud row from sample data
# ---------------------------------------------------------------------------

class TestKnownFraudSampleData:
    def test_transaction_50003_labels_high(self):
        """
        Transaction 50003 from transactions.csv: $1,250 gift-card purchase,
        device_risk=81, international, velocity=6, failed_logins=5, account
        85 days old, basic KYC.  It is confirmed fraud (chargebacks.csv).
        The scorer must label it 'high'.
        """
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

    def test_transaction_50003_exact_score_above_60(self):
        """Score must be >= 60 to qualify as 'high' (not just barely medium)."""
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
        assert score_transaction(tx) >= 60
