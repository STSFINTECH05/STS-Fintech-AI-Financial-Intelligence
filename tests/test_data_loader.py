"""
Unit tests for ML Database Loader (Step 7.2A)
Verifies:
- NumPy scalar conversion to native Python types
- pandas missing-value (NaN, NaT, pd.NA) sanitization to None (SQL NULL)
- Date and Timestamp conversion
- Budget column mapping and data integrity
- Database transaction rollback and idempotency
"""

import os
import sys
from datetime import datetime, date
import numpy as np
import pandas as pd
import pytest

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.database_loader import sanitize_value, DATABASE_CONFIG


class TestSanitizeValue:
    """Test suite for sanitize_value scalar conversion."""

    def test_numpy_integer_conversion(self):
        val_int64 = np.int64(12345)
        val_int32 = np.int32(678)

        sanitized_64 = sanitize_value(val_int64)
        sanitized_32 = sanitize_value(val_int32)

        assert isinstance(sanitized_64, int)
        assert sanitized_64 == 12345
        assert not isinstance(sanitized_64, np.generic)

        assert isinstance(sanitized_32, int)
        assert sanitized_32 == 678
        assert not isinstance(sanitized_32, np.generic)

    def test_numpy_float_conversion(self):
        val_float64 = np.float64(785156.03)
        val_float32 = np.float32(107.45)

        sanitized_64 = sanitize_value(val_float64)
        sanitized_32 = sanitize_value(val_float32)

        assert isinstance(sanitized_64, float)
        assert sanitized_64 == pytest.approx(785156.03)
        assert not isinstance(sanitized_64, np.generic)

        assert isinstance(sanitized_32, float)
        assert sanitized_32 == pytest.approx(107.45, rel=1e-3)
        assert not isinstance(sanitized_32, np.generic)

    def test_numpy_boolean_conversion(self):
        val_true = np.bool_(True)
        val_false = np.bool_(False)

        sanitized_true = sanitize_value(val_true)
        sanitized_false = sanitize_value(val_false)

        assert isinstance(sanitized_true, bool)
        assert sanitized_true is True
        assert not isinstance(sanitized_true, np.generic)

        assert isinstance(sanitized_false, bool)
        assert sanitized_false is False
        assert not isinstance(sanitized_false, np.generic)

    def test_missing_values_become_none(self):
        assert sanitize_value(None) is None
        assert sanitize_value(np.nan) is None
        assert sanitize_value(float("nan")) is None
        assert sanitize_value(float("inf")) is None
        assert sanitize_value(float("-inf")) is None
        assert sanitize_value(pd.NA) is None
        assert sanitize_value(pd.NaT) is None
        assert sanitize_value(np.float64(np.nan)) is None

    def test_timestamp_and_date_conversion(self):
        ts = pd.Timestamp("2025-07-01 14:30:00")
        dt = datetime(2025, 7, 1, 14, 30, 0)
        d = date(2025, 7, 1)

        sanitized_ts = sanitize_value(ts)
        sanitized_dt = sanitize_value(dt)
        sanitized_d = sanitize_value(d)

        assert isinstance(sanitized_ts, datetime)
        assert sanitized_ts == dt
        assert isinstance(sanitized_dt, datetime)
        assert isinstance(sanitized_d, date)

    def test_standard_python_types_preserved(self):
        assert sanitize_value("sample text") == "sample text"
        assert sanitize_value(999) == 999
        assert sanitize_value(45.67) == 45.67
        assert sanitize_value(True) is True
        assert sanitize_value(False) is False


class TestBudgetDataPreparation:
    """Test suite for budget dataset structure and conversions."""

    @pytest.fixture
    def budget_df(self):
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "budgets.csv"
        )
        return pd.read_csv(csv_path)

    def test_budget_csv_row_count(self, budget_df):
        assert len(budget_df) == 5000, "budgets.csv must contain exactly 5,000 records"

    def test_budget_columns(self, budget_df):
        expected_columns = [
            "budget_id",
            "department_id",
            "budget_year",
            "budget_month",
            "allocated_amount",
            "actual_amount",
            "remaining_amount",
            "utilization_percentage"
        ]
        assert list(budget_df.columns) == expected_columns

    def test_budget_id_uniqueness(self, budget_df):
        assert budget_df["budget_id"].nunique() == 5000
        assert budget_df["budget_id"].min() == 1
        assert budget_df["budget_id"].max() == 5000

    def test_budget_department_range(self, budget_df):
        assert budget_df["department_id"].min() >= 1
        assert budget_df["department_id"].max() <= 50

    def test_budget_sanitization_produces_native_types(self, budget_df):
        cols = list(budget_df.columns)
        for row in budget_df[cols].head(50).itertuples(index=False):
            record = [sanitize_value(val) for val in row]

            # budget_id, department_id, budget_year, budget_month must be native int
            assert isinstance(record[0], int)
            assert isinstance(record[1], int)
            assert isinstance(record[2], int)
            assert isinstance(record[3], int)

            # allocated_amount, actual_amount, remaining_amount, utilization_percentage must be float
            assert isinstance(record[4], float)
            assert isinstance(record[5], float)
            assert isinstance(record[6], float)
            assert isinstance(record[7], float)

            # None of them should be numpy types
            for val in record:
                assert not isinstance(val, np.generic), f"Value {val} must not be a NumPy generic"


class TestDatabaseLoaderSafety:
    """Test database rollback and transaction safety."""

    def test_database_connection(self):
        import psycopg2
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            database=DATABASE_CONFIG["database"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"]
        )
        assert conn.status == psycopg2.extensions.STATUS_READY
        conn.close()

    def test_rollback_on_intentional_failure(self):
        import psycopg2
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            database=DATABASE_CONFIG["database"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"]
        )
        cur = conn.cursor()

        # Test transaction rollback
        try:
            cur.execute("SELECT count(*) FROM budgets")
            initial_count = cur.fetchone()[0]

            # Attempt an invalid insert that violates foreign key
            cur.execute("""
                INSERT INTO budgets (budget_id, department_id, budget_year)
                VALUES (999999, 999999, 2026)
            """)
            conn.commit()
            pytest.fail("Should have failed foreign key constraint")
        except Exception:
            conn.rollback()

        # Confirm rollback left the database in consistent state
        cur.execute("SELECT count(*) FROM budgets")
        count_after = cur.fetchone()[0]
        assert count_after == initial_count

        cur.close()
        conn.close()
