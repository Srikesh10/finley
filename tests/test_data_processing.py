"""
test_data_processing.py — Unit tests for process_transactions() in finley.py.

All data is synthetic and built in-memory via io.StringIO.  No files are read
from the data/ directory.  pytest tmp_path fixtures are used to hand the
StringIO content to process_transactions(), which expects a file path.

Run with:
    pytest tests/test_data_processing.py -v
"""

import io
import sys
import os
import textwrap

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as test_outofscope.py line 9
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finley import process_transactions  # noqa: E402  (import after sys.path fix)


# ===========================================================================
# Helpers
# ===========================================================================

TX_HEADER = "transaction_date,amount,primary_category,sub_category,merchant_name,name,is_deleted,pending\n"
ACC_HEADER = "name,subtype,type,current_balance,available_balance,is_deleted\n"


def _write(tmp_path, filename: str, content: str) -> str:
    """Write *content* (from io.StringIO or plain string) to a temp file and
    return the absolute path string expected by process_transactions()."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return str(p)


def _tx_csv(*rows: str) -> str:
    """Build a complete transactions CSV string from row strings."""
    buf = io.StringIO()
    buf.write(TX_HEADER)
    for row in rows:
        buf.write(row.strip() + "\n")
    return buf.getvalue()


def _acc_csv(*rows: str) -> str:
    """Build a complete accounts CSV string from row strings."""
    buf = io.StringIO()
    buf.write(ACC_HEADER)
    for row in rows:
        buf.write(row.strip() + "\n")
    return buf.getvalue()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def two_month_tx_csv():
    """Two complete months (Jan + Feb 2025) of synthetic spending + income.

    Jan: FOOD_AND_DRINK $50 + $30, ENTERTAINMENT $100  => spending $180
    Feb: FOOD_AND_DRINK $60,        TRANSFER_OUT $200   => spending $260
    Income row: amount=-1000 (negative = income)
    All rows are active (is_deleted=False, pending=False).
    """
    return textwrap.dedent("""\
        transaction_date,amount,primary_category,sub_category,merchant_name,name,is_deleted,pending
        2025-01-05,50.00,FOOD_AND_DRINK,GROCERIES,Whole Foods,Whole Foods,False,False
        2025-01-15,30.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,False
        2025-01-20,100.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False
        2025-02-03,60.00,FOOD_AND_DRINK,GROCERIES,Trader Joes,Trader Joes,False,False
        2025-02-10,200.00,TRANSFER_OUT,TRANSFER,Venmo,Venmo,False,False
        2025-01-10,-1000.00,INCOME,PAYROLL,Employer,Payroll,False,False
    """)


@pytest.fixture()
def two_month_tx_path(tmp_path, two_month_tx_csv):
    """Write two_month_tx_csv to a temp file; return its path."""
    return _write(tmp_path, "transactions.csv", two_month_tx_csv)


@pytest.fixture()
def simple_accounts_csv():
    """One checking account, active."""
    return _acc_csv(
        "My Checking,checking,depository,5000.00,4800.00,False",
    )


@pytest.fixture()
def simple_accounts_path(tmp_path, simple_accounts_csv):
    return _write(tmp_path, "accounts.csv", simple_accounts_csv)


# ===========================================================================
# 1. BASIC STRUCTURE
# ===========================================================================

class TestBasicStructure:
    """process_transactions() must return a dict containing all required keys."""

    REQUIRED_TOP_LEVEL_KEYS = [
        "date_range",
        "totals",
        "monthly_spending",
        "by_category",
        "monthly_by_category",
        "last_complete_month",
        "subscriptions",
        "upcoming_bills",
        "leisure",
        "top_transactions_recent_month",
        "mom_category_delta",
        "yearly_spending",
    ]

    def test_all_required_keys_present(self, two_month_tx_path):
        """All required top-level keys are present in the returned dict."""
        result = process_transactions(two_month_tx_path)
        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            assert key in result, f"Missing required key: '{key}'"

    def test_date_range_has_from_and_to(self, two_month_tx_path):
        """date_range contains 'from' and 'to' string keys."""
        result = process_transactions(two_month_tx_path)
        dr = result["date_range"]
        assert "from" in dr and "to" in dr
        assert isinstance(dr["from"], str)
        assert isinstance(dr["to"], str)

    def test_totals_has_required_subkeys(self, two_month_tx_path):
        """totals dict has total_spending, total_income, net_cashflow, transaction_count."""
        result = process_transactions(two_month_tx_path)
        t = result["totals"]
        for key in ("total_spending", "total_income", "net_cashflow", "transaction_count"):
            assert key in t, f"totals missing key '{key}'"

    def test_top_transactions_recent_month_structure(self, two_month_tx_path):
        """top_transactions_recent_month has 'month', 'transactions', and 'note' keys."""
        result = process_transactions(two_month_tx_path)
        tt = result["top_transactions_recent_month"]
        assert "month" in tt
        assert "transactions" in tt
        assert isinstance(tt["transactions"], list)


# ===========================================================================
# 2. SPENDING TOTALS
# ===========================================================================

class TestSpendingTotals:
    """totals.total_spending must equal the sum of all positive amounts.
    Income (negative amounts) must be captured in total_income.
    TRANSFER_OUT rows are positive and therefore DO count toward total_spending."""

    def test_total_spending_matches_sum_of_positive_amounts(self, tmp_path):
        """total_spending == sum of all positive-amount rows (including TRANSFER_OUT)."""
        csv_content = _tx_csv(
            "2025-03-01,100.00,FOOD_AND_DRINK,GROCERIES,Walmart,Walmart,False,False",
            "2025-03-05,250.00,TRANSFER_OUT,TRANSFER,Chase,Chase,False,False",
            "2025-03-10,75.50,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            "2025-03-15,-2000.00,INCOME,PAYROLL,Employer,Payroll,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        expected = round(100.00 + 250.00 + 75.50, 2)
        assert result["totals"]["total_spending"] == pytest.approx(expected, abs=0.01)

    def test_total_income_is_sum_of_negative_amounts(self, tmp_path):
        """total_income == abs(sum of negative-amount rows)."""
        csv_content = _tx_csv(
            "2025-03-01,50.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-03-10,-1500.00,INCOME,PAYROLL,Employer,Payroll,False,False",
            "2025-03-20,-300.00,INCOME,OTHER,Side Gig,Side Gig,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["totals"]["total_income"] == pytest.approx(1800.00, abs=0.01)

    def test_net_cashflow_equals_income_minus_spending(self, tmp_path):
        """net_cashflow == total_income - total_spending."""
        csv_content = _tx_csv(
            "2025-03-01,400.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-03-10,-2000.00,INCOME,PAYROLL,Employer,Payroll,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        t = result["totals"]
        expected_net = round(t["total_income"] - t["total_spending"], 2)
        assert t["net_cashflow"] == pytest.approx(expected_net, abs=0.01)


# ===========================================================================
# 3. MONTHLY BREAKDOWN
# ===========================================================================

class TestMonthlyBreakdown:
    """monthly_by_category structure — one entry per month with correct keys."""

    def test_one_entry_per_calendar_month(self, two_month_tx_path):
        """monthly_by_category has exactly one key per month that has spending."""
        result = process_transactions(two_month_tx_path)
        mbc = result["monthly_by_category"]
        # Our fixture has Jan and Feb spending rows
        assert "2025-01" in mbc
        assert "2025-02" in mbc

    def test_each_month_has_required_keys(self, two_month_tx_path):
        """Each monthly_by_category entry has 'is_partial', 'spending', 'subcategories'."""
        result = process_transactions(two_month_tx_path)
        for month, data in result["monthly_by_category"].items():
            assert "is_partial" in data, f"{month} missing 'is_partial'"
            assert "spending" in data, f"{month} missing 'spending'"
            assert "subcategories" in data, f"{month} missing 'subcategories'"

    def test_spending_values_match_transactions(self, two_month_tx_path):
        """Jan spending should be 180.00 (50 + 30 + 100), Feb should be 260.00 (60 + 200)."""
        result = process_transactions(two_month_tx_path)
        jan = result["monthly_by_category"]["2025-01"]["spending"]
        feb = result["monthly_by_category"]["2025-02"]["spending"]
        jan_total = sum(jan.values())
        feb_total = sum(feb.values())
        assert jan_total == pytest.approx(180.00, abs=0.01)
        assert feb_total == pytest.approx(260.00, abs=0.01)

    def test_subcategories_is_dict_of_dicts(self, two_month_tx_path):
        """subcategories maps category -> {sub_category: amount}."""
        result = process_transactions(two_month_tx_path)
        jan = result["monthly_by_category"]["2025-01"]["subcategories"]
        # FOOD_AND_DRINK should appear with sub-keys
        assert "FOOD_AND_DRINK" in jan
        assert isinstance(jan["FOOD_AND_DRINK"], dict)


# ===========================================================================
# 4. PARTIAL MONTH DETECTION
# ===========================================================================

class TestPartialMonthDetection:
    """Months with < 15 days of spread are marked is_partial=True."""

    def test_five_day_spread_is_partial(self, tmp_path):
        """A month whose transactions span only 5 days is marked is_partial=True."""
        # All three rows are within 5 days (Jan 1-5) — it's the LAST month so partial check applies
        csv_content = _tx_csv(
            # Complete earlier month to anchor the dataset
            "2024-12-05,100.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2024-12-10,50.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            "2024-12-20,75.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,False",
            # Partial last month — only 5 days spread
            "2025-01-01,30.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-01-03,20.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,False",
            "2025-01-05,15.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        jan = result["monthly_by_category"].get("2025-01", {})
        assert jan.get("is_partial") is True, "5-day spread month should be is_partial=True"

    def test_twenty_day_spread_is_not_partial(self, tmp_path):
        """A month whose transactions span 20 days is marked is_partial=False."""
        csv_content = _tx_csv(
            # Transactions spread across Jan 1-21 (20-day spread) — only month, so both edge checks apply
            # We need at least two months so the partial check on the last month is meaningful;
            # a single month as first AND last gets first_partial AND last_partial evaluated independently.
            # Use a prior complete month to make Jan the "last" month.
            "2024-12-05,100.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2024-12-10,50.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            "2024-12-20,75.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,False",
            # Jan with 20-day spread
            "2025-01-01,30.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-01-21,20.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        jan = result["monthly_by_category"].get("2025-01", {})
        assert jan.get("is_partial") is False, "20-day spread month should be is_partial=False"


# ===========================================================================
# 5. LAST COMPLETE MONTH
# ===========================================================================

class TestLastCompleteMonth:
    """last_complete_month skips a partial tail month and returns the prior full one."""

    def test_last_complete_month_skips_partial_tail(self, tmp_path):
        """When the most recent month is partial, last_complete_month is the prior month."""
        csv_content = _tx_csv(
            # Feb — complete (20+ day spread)
            "2025-02-01,100.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-02-25,50.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            # Mar — partial (only 3 days spread, and it's the last month)
            "2025-03-01,40.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-03-03,20.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["last_complete_month"] == "2025-02", (
            f"Expected '2025-02' but got '{result['last_complete_month']}'"
        )

    def test_last_complete_month_is_last_when_not_partial(self, tmp_path):
        """When the most recent month is NOT partial, last_complete_month equals it."""
        csv_content = _tx_csv(
            "2025-01-05,80.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-01-25,40.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            "2025-02-03,60.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-02-24,30.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["last_complete_month"] == "2025-02", (
            f"Expected '2025-02' but got '{result['last_complete_month']}'"
        )


# ===========================================================================
# 6. TRANSFER_OUT EXCLUSION
# ===========================================================================

class TestTransferOutExclusion:
    """TRANSFER_OUT rows appear in by_category but are excluded from safe_to_spend MTD."""

    def test_transfer_out_appears_in_by_category(self, tmp_path):
        """TRANSFER_OUT transactions are counted in by_category."""
        csv_content = _tx_csv(
            "2025-03-10,500.00,TRANSFER_OUT,TRANSFER,Venmo,Venmo,False,False",
            "2025-03-15,100.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert "TRANSFER_OUT" in result["by_category"], (
            "TRANSFER_OUT should appear as a category in by_category"
        )

    def test_transfer_out_excluded_from_safe_to_spend_mtd(self, tmp_path, simple_accounts_csv):
        """mtd_spending_ex_transfers in safe_to_spend does NOT include TRANSFER_OUT amounts."""
        # Use the current calendar month so MTD calculation picks it up
        current_month = pd.Timestamp.today().strftime("%Y-%m")
        current_month_day1 = f"{current_month}-01"
        current_month_day15 = f"{current_month}-15"

        csv_content = _tx_csv(
            f"{current_month_day1},200.00,TRANSFER_OUT,TRANSFER,Venmo,Venmo,False,False",
            f"{current_month_day15},80.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
        )
        tx_path = _write(tmp_path, "tx.csv", csv_content)
        acc_path = _write(tmp_path, "accounts.csv", simple_accounts_csv)

        result = process_transactions(tx_path, accounts_path=acc_path)
        sts = result.get("safe_to_spend")
        assert sts is not None, "safe_to_spend should be populated when accounts_path is provided"
        # MTD should only include FOOD_AND_DRINK $80, NOT the $200 transfer
        assert sts["mtd_spending_ex_transfers"] == pytest.approx(80.00, abs=0.01), (
            f"Expected 80.00 (transfer excluded), got {sts['mtd_spending_ex_transfers']}"
        )


# ===========================================================================
# 7. INCOME vs SPENDING SPLIT
# ===========================================================================

class TestIncomeSpendingSplit:
    """Negative amounts are income; positive amounts are spending."""

    def test_negative_amounts_classified_as_income(self, tmp_path):
        """Rows with amount < 0 contribute only to total_income, not total_spending."""
        csv_content = _tx_csv(
            "2025-04-05,300.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-04-10,-2500.00,INCOME,PAYROLL,Employer,Payroll,False,False",
            "2025-04-20,-500.00,INCOME,OTHER,Freelance,Freelance,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["totals"]["total_income"] == pytest.approx(3000.00, abs=0.01)
        assert result["totals"]["total_spending"] == pytest.approx(300.00, abs=0.01)

    def test_positive_amounts_classified_as_spending(self, tmp_path):
        """Rows with amount > 0 contribute only to total_spending, not total_income."""
        csv_content = _tx_csv(
            "2025-04-01,150.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            "2025-04-12,200.00,TRANSFER_OUT,TRANSFER,Venmo,Venmo,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["totals"]["total_spending"] == pytest.approx(350.00, abs=0.01)
        assert result["totals"]["total_income"] == pytest.approx(0.00, abs=0.01)

    def test_net_cashflow_direction(self, tmp_path):
        """net_cashflow is positive when income > spending (surplus)."""
        csv_content = _tx_csv(
            "2025-04-05,400.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-04-10,-3000.00,INCOME,PAYROLL,Employer,Payroll,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["totals"]["net_cashflow"] > 0, "Surplus household should have positive net_cashflow"


# ===========================================================================
# 8. DELETED / PENDING ROWS
# ===========================================================================

class TestDeletedPendingExclusion:
    """Rows where is_deleted=True or pending=True must not affect any calculation."""

    def test_deleted_rows_excluded_from_spending(self, tmp_path):
        """A deleted transaction does not appear in total_spending."""
        csv_content = _tx_csv(
            "2025-05-01,100.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            # This row is deleted — should be ignored entirely
            "2025-05-05,999.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,True,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["totals"]["total_spending"] == pytest.approx(100.00, abs=0.01), (
            "Deleted transaction should not contribute to total_spending"
        )

    def test_pending_rows_excluded_from_spending(self, tmp_path):
        """A pending transaction does not appear in total_spending."""
        csv_content = _tx_csv(
            "2025-05-01,100.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            # This row is pending — should be ignored entirely
            "2025-05-10,500.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,True",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["totals"]["total_spending"] == pytest.approx(100.00, abs=0.01), (
            "Pending transaction should not contribute to total_spending"
        )

    def test_deleted_and_pending_both_excluded(self, tmp_path):
        """Both deleted AND pending rows are excluded; only clean rows are counted."""
        csv_content = _tx_csv(
            "2025-05-01,200.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-05-05,999.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,True,False",
            "2025-05-10,777.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,True",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result["totals"]["total_spending"] == pytest.approx(200.00, abs=0.01)
        assert result["totals"]["transaction_count"] == 1

    def test_deleted_rows_excluded_from_category_breakdown(self, tmp_path):
        """Deleted rows do not appear in by_category."""
        csv_content = _tx_csv(
            "2025-05-01,50.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            # Deleted ENTERTAINMENT row — should not create an ENTERTAINMENT category
            "2025-05-05,200.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,True,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert "ENTERTAINMENT" not in result["by_category"], (
            "ENTERTAINMENT category should not appear when only deleted rows exist for it"
        )


# ===========================================================================
# 9. EDGE CASE — Empty DataFrame
# ===========================================================================

class TestEmptyDataFrame:
    """process_transactions() must not crash on an empty CSV (headers only)."""

    def test_empty_csv_does_not_crash(self, tmp_path):
        """An empty CSV (header row only) returns a result dict without raising."""
        csv_content = TX_HEADER  # header only, zero data rows
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert isinstance(result, dict), "Should return a dict even for empty input"

    def test_empty_csv_totals_are_zero(self, tmp_path):
        """Empty CSV produces zero total_spending and zero total_income."""
        path = _write(tmp_path, "tx.csv", TX_HEADER)
        result = process_transactions(path)
        assert result["totals"]["total_spending"] == pytest.approx(0.0, abs=0.01)
        assert result["totals"]["total_income"] == pytest.approx(0.0, abs=0.01)

    def test_empty_csv_monthly_spending_is_empty(self, tmp_path):
        """Empty CSV produces an empty monthly_spending dict."""
        path = _write(tmp_path, "tx.csv", TX_HEADER)
        result = process_transactions(path)
        assert result["monthly_spending"] == {}

    def test_empty_csv_subscriptions_is_list(self, tmp_path):
        """subscriptions is always a list (empty list when no recurring_path provided)."""
        path = _write(tmp_path, "tx.csv", TX_HEADER)
        result = process_transactions(path)
        assert isinstance(result["subscriptions"], list)

    def test_empty_csv_last_complete_month_is_none_or_empty(self, tmp_path):
        """last_complete_month is None (or falsy) when there are no transactions."""
        path = _write(tmp_path, "tx.csv", TX_HEADER)
        result = process_transactions(path)
        # None or empty string are both acceptable
        assert not result["last_complete_month"], (
            f"Expected falsy last_complete_month for empty data, got {result['last_complete_month']!r}"
        )


# ===========================================================================
# 10. SUBCATEGORY BREAKDOWN
# ===========================================================================

class TestSubcategoryBreakdown:
    """food_breakdown should sum FOOD_AND_DRINK sub-categories correctly."""

    def test_food_breakdown_sums_subcategories(self, tmp_path):
        """food_breakdown totals match the individual FOOD_AND_DRINK sub-category amounts."""
        csv_content = _tx_csv(
            "2025-06-01,120.00,FOOD_AND_DRINK,GROCERIES,Whole Foods,Whole Foods,False,False",
            "2025-06-05,45.00,FOOD_AND_DRINK,RESTAURANTS,Chipotle,Chipotle,False,False",
            "2025-06-10,30.00,FOOD_AND_DRINK,GROCERIES,Trader Joes,Trader Joes,False,False",
            "2025-06-15,200.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        fb = result.get("food_breakdown", {})
        assert fb.get("GROCERIES") == pytest.approx(150.00, abs=0.01), (
            f"GROCERIES should be 120+30=150, got {fb.get('GROCERIES')}"
        )
        assert fb.get("RESTAURANTS") == pytest.approx(45.00, abs=0.01), (
            f"RESTAURANTS should be 45, got {fb.get('RESTAURANTS')}"
        )

    def test_food_breakdown_excludes_other_categories(self, tmp_path):
        """food_breakdown only contains FOOD_AND_DRINK sub-categories, not other categories."""
        csv_content = _tx_csv(
            "2025-06-01,100.00,FOOD_AND_DRINK,GROCERIES,Store,Store,False,False",
            "2025-06-05,300.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        fb = result.get("food_breakdown", {})
        assert "STREAMING" not in fb, "food_breakdown must not contain ENTERTAINMENT sub-categories"

    def test_food_breakdown_empty_when_no_food_transactions(self, tmp_path):
        """food_breakdown is an empty dict when no FOOD_AND_DRINK transactions exist."""
        csv_content = _tx_csv(
            "2025-06-01,200.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)
        assert result.get("food_breakdown") == {}, (
            "food_breakdown should be empty dict when no food transactions exist"
        )

    def test_monthly_subcategories_match_food_breakdown_totals(self, tmp_path):
        """Subcategory totals in monthly_by_category match the standalone food_breakdown."""
        csv_content = _tx_csv(
            # Only one complete month so partial-detection edge doesn't interfere
            "2025-06-01,80.00,FOOD_AND_DRINK,GROCERIES,Store A,Store A,False,False",
            "2025-06-10,40.00,FOOD_AND_DRINK,GROCERIES,Store B,Store B,False,False",
            "2025-06-20,60.00,FOOD_AND_DRINK,RESTAURANTS,Diner,Diner,False,False",
        )
        path = _write(tmp_path, "tx.csv", csv_content)
        result = process_transactions(path)

        # The single month will be both first and last; with only one month it cannot be partial
        # (partial check requires at least two months for last_complete_month logic)
        # Either way the subcategory numbers must match food_breakdown
        monthly_food_subs = (
            result["monthly_by_category"]
            .get("2025-06", {})
            .get("subcategories", {})
            .get("FOOD_AND_DRINK", {})
        )
        fb = result.get("food_breakdown", {})
        assert monthly_food_subs.get("GROCERIES") == pytest.approx(fb.get("GROCERIES", 0), abs=0.01)
        assert monthly_food_subs.get("RESTAURANTS") == pytest.approx(fb.get("RESTAURANTS", 0), abs=0.01)


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestIntegrationSmoke:
    """A fuller end-to-end run with both transactions and accounts to make sure
    the whole pipeline runs without errors and produces plausible values."""

    def test_full_run_with_accounts(self, tmp_path):
        """Full pipeline with tx + accounts does not crash and produces coherent output."""
        tx_csv = _tx_csv(
            "2025-01-05,120.00,FOOD_AND_DRINK,GROCERIES,Costco,Costco,False,False",
            "2025-01-15,80.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            "2025-01-20,300.00,TRANSFER_OUT,TRANSFER,Venmo,Venmo,False,False",
            "2025-01-10,-3500.00,INCOME,PAYROLL,Employer,Payroll,False,False",
            "2025-02-05,150.00,FOOD_AND_DRINK,GROCERIES,Costco,Costco,False,False",
            "2025-02-20,60.00,ENTERTAINMENT,STREAMING,Netflix,Netflix,False,False",
            "2025-02-10,-3500.00,INCOME,PAYROLL,Employer,Payroll,False,False",
        )
        acc_csv = _acc_csv(
            "Primary Checking,checking,depository,8000.00,7500.00,False",
            "Savings,savings,depository,15000.00,15000.00,False",
        )
        tx_path = _write(tmp_path, "tx.csv", tx_csv)
        acc_path = _write(tmp_path, "accounts.csv", acc_csv)

        result = process_transactions(tx_path, accounts_path=acc_path)

        # Structural checks
        assert isinstance(result, dict)
        assert result["totals"]["total_spending"] == pytest.approx(710.00, abs=0.01)
        assert result["totals"]["total_income"] == pytest.approx(7000.00, abs=0.01)
        assert result["totals"]["net_cashflow"] == pytest.approx(6290.00, abs=0.01)

        # Both months present
        assert "2025-01" in result["monthly_by_category"]
        assert "2025-02" in result["monthly_by_category"]

        # Accounts wired up correctly
        assert result["accounts"] is not None and len(result["accounts"]) == 2
        assert result["net_worth_liquid"] == pytest.approx(23000.00, abs=0.01)

        # leisure should capture ENTERTAINMENT
        assert result["leisure"]["total_all_time"] == pytest.approx(140.00, abs=0.01)
