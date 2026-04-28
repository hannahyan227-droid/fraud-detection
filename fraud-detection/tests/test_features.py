"""
Tests for features.py — build_model_frame().

Covers:
- Account join: transaction receives correct account-level fields
- is_large_amount: binary flag correctness and boundary
- login_pressure: categorical bucketing and boundary values
- Unmatched account_id: graceful NaN rather than crash
- Output structure: required columns present, row count preserved
"""

import math

import pandas as pd
import pytest

from features import build_model_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACCOUNT_DEFAULTS = {
    "account_id": 1,
    "customer_name": "Test User",
    "country": "US",
    "signup_date": "2020-01-01",
    "kyc_level": "full",
    "account_age_days": 500,
    "prior_chargebacks": 0,
    "is_vip": "N",
}

_TRANSACTION_DEFAULTS = {
    "transaction_id": 1001,
    "account_id": 1,
    "timestamp": "2026-01-01 12:00:00",
    "amount_usd": 100.0,
    "merchant_category": "grocery",
    "channel": "web",
    "device_risk_score": 10,
    "ip_country": "US",
    "is_international": 0,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "chargeback_within_60d": 0,
}


def _accounts(*rows):
    """Build an accounts DataFrame.

    No args → one row with all defaults.
    One or more dicts → each dict is merged with defaults to form a row.
    """
    if not rows:
        return pd.DataFrame([_ACCOUNT_DEFAULTS])
    return pd.DataFrame([{**_ACCOUNT_DEFAULTS, **r} for r in rows])


def _transactions(*rows):
    """Build a transactions DataFrame.

    No args → one row with all defaults.
    One or more dicts → each dict is merged with defaults to form a row.
    """
    if not rows:
        return pd.DataFrame([_TRANSACTION_DEFAULTS])
    return pd.DataFrame([{**_TRANSACTION_DEFAULTS, **r} for r in rows])


# ---------------------------------------------------------------------------
# Join correctness
# ---------------------------------------------------------------------------

class TestMerge:
    def test_account_fields_appear_in_output(self):
        txn = _transactions()
        acc = _accounts({"kyc_level": "basic", "account_age_days": 45})
        result = build_model_frame(txn, acc)
        assert result.loc[0, "kyc_level"] == "basic"
        assert result.loc[0, "account_age_days"] == 45

    def test_transaction_fields_are_preserved(self):
        txn = _transactions({"amount_usd": 999.0, "merchant_category": "electronics"})
        acc = _accounts()
        result = build_model_frame(txn, acc)
        assert result.loc[0, "amount_usd"] == 999.0
        assert result.loc[0, "merchant_category"] == "electronics"

    def test_unmatched_account_id_produces_nan_not_error(self):
        txn = _transactions({"account_id": 9999})
        acc = _accounts({"account_id": 1})
        result = build_model_frame(txn, acc)
        assert len(result) == 1
        assert math.isnan(result.loc[0, "account_age_days"])

    def test_multiple_transactions_all_joined(self):
        txn = _transactions(
            {"transaction_id": 1001, "account_id": 1},
            {"transaction_id": 1002, "account_id": 2},
        )
        acc = _accounts(
            {"account_id": 1, "kyc_level": "full"},
            {"account_id": 2, "kyc_level": "basic"},
        )
        result = build_model_frame(txn, acc)
        assert len(result) == 2
        kyc_map = dict(zip(result["transaction_id"], result["kyc_level"]))
        assert kyc_map[1001] == "full"
        assert kyc_map[1002] == "basic"

    def test_output_row_count_matches_transactions(self):
        txn = _transactions(
            {"transaction_id": 1001},
            {"transaction_id": 1002},
            {"transaction_id": 1003},
        )
        acc = _accounts()  # account_id=1; all three transactions join to it
        result = build_model_frame(txn, acc)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# is_large_amount
# ---------------------------------------------------------------------------

class TestIsLargeAmount:
    def test_below_threshold_is_zero(self):
        result = build_model_frame(_transactions({"amount_usd": 999.99}), _accounts())
        assert result.loc[0, "is_large_amount"] == 0

    def test_at_threshold_is_one(self):
        result = build_model_frame(_transactions({"amount_usd": 1000.0}), _accounts())
        assert result.loc[0, "is_large_amount"] == 1

    def test_above_threshold_is_one(self):
        result = build_model_frame(_transactions({"amount_usd": 5000.0}), _accounts())
        assert result.loc[0, "is_large_amount"] == 1

    def test_zero_amount_is_zero(self):
        result = build_model_frame(_transactions({"amount_usd": 0.0}), _accounts())
        assert result.loc[0, "is_large_amount"] == 0

    def test_flag_is_integer_dtype(self):
        result = build_model_frame(_transactions(), _accounts())
        assert result["is_large_amount"].dtype in (int, "int32", "int64")


# ---------------------------------------------------------------------------
# login_pressure
# ---------------------------------------------------------------------------

class TestLoginPressure:
    @pytest.mark.parametrize("logins,expected", [
        (0,   "none"),
        (1,   "low"),
        (2,   "low"),
        (3,   "high"),
        (10,  "high"),
        (100, "high"),
    ])
    def test_login_pressure_bucketing(self, logins, expected):
        txn = _transactions({"failed_logins_24h": logins})
        result = build_model_frame(txn, _accounts())
        assert str(result.loc[0, "login_pressure"]) == expected

    def test_login_pressure_boundary_none_to_low(self):
        """0 logins → 'none'; 1 login → 'low'."""
        r0 = build_model_frame(_transactions({"failed_logins_24h": 0}), _accounts())
        r1 = build_model_frame(_transactions({"failed_logins_24h": 1}), _accounts())
        assert str(r0.loc[0, "login_pressure"]) == "none"
        assert str(r1.loc[0, "login_pressure"]) == "low"

    def test_login_pressure_boundary_low_to_high(self):
        """2 logins → 'low'; 3 logins → 'high'."""
        r2 = build_model_frame(_transactions({"failed_logins_24h": 2}), _accounts())
        r3 = build_model_frame(_transactions({"failed_logins_24h": 3}), _accounts())
        assert str(r2.loc[0, "login_pressure"]) == "low"
        assert str(r3.loc[0, "login_pressure"]) == "high"


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_required_columns_present(self):
        result = build_model_frame(_transactions(), _accounts())
        for col in (
            "transaction_id", "account_id", "amount_usd",
            "device_risk_score", "is_international", "velocity_24h",
            "failed_logins_24h", "prior_chargebacks",
            "kyc_level", "account_age_days",
            "is_large_amount", "login_pressure",
        ):
            assert col in result.columns, f"missing column: {col}"

    def test_derived_columns_added(self):
        result = build_model_frame(_transactions(), _accounts())
        assert "is_large_amount" in result.columns
        assert "login_pressure" in result.columns
