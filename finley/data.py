import datetime
import hashlib
import json
import os

import pandas as pd


_CACHE_DIR = ".finley_cache"


def _cache_key(*paths: str) -> str:
    """Hash mtime + size of all input files to detect changes."""
    h = hashlib.md5()
    for p in paths:
        if p and os.path.exists(p):
            stat = os.stat(p)
            h.update(f"{p}:{stat.st_mtime}:{stat.st_size}".encode())
    return h.hexdigest()


def _load_cache(key: str) -> dict | None:
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: dict):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ── Sub-functions ─────────────────────────────────────────────────────────────

def _load_transactions(tx_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSV, drop deleted/pending, split into spending and income."""
    df = pd.read_csv(tx_path, low_memory=False)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0.0)

    if "is_deleted" in df.columns:
        df = df[~df["is_deleted"].fillna(False).astype(bool)]
    if "pending" in df.columns:
        df = df[~df["pending"].fillna(False).astype(bool)]

    spending = df[df["amount"] > 0].copy()
    income   = df[df["amount"] < 0].copy()
    return df, spending, income


def _build_basic_aggregates(spending: pd.DataFrame) -> dict:
    """Monthly totals, category breakdown, top merchants."""
    monthly = (
        spending.groupby(spending["transaction_date"].dt.to_period("M").astype(str))["amount"]
        .sum().round(2).to_dict()
    )
    by_category = (
        spending.groupby("primary_category")["amount"]
        .agg(total="sum", count="count", avg="mean")
        .round(2).sort_values("total", ascending=False)
        .to_dict("index")
    )
    top_merchants = (
        spending[spending["merchant_name"].notna() & (spending["merchant_name"] != "NULL")]
        .groupby("merchant_name")["amount"]
        .agg(total="sum", count="count")
        .round(2).sort_values("total", ascending=False)
        .head(20).to_dict("index")
    )
    return {"monthly": monthly, "by_category": by_category, "top_merchants": top_merchants}


def _detect_partial_months(spending: pd.DataFrame) -> tuple[list, str | None, str | None, bool, bool]:
    """Return sorted month list plus first/last month and whether each is partial."""
    spending = spending.copy()
    spending["_month"] = spending["transaction_date"].dt.to_period("M").astype(str)
    all_months = sorted(spending["_month"].unique())

    first_month = all_months[0] if all_months else None
    last_month  = all_months[-1] if all_months else None

    def _days_span(month_str: str) -> int:
        m_data = spending[spending["_month"] == month_str]["transaction_date"]
        return int((m_data.max() - m_data.min()).days) + 1 if len(m_data) > 1 else 1

    first_partial = bool(first_month and _days_span(first_month) < 15)
    last_partial  = bool(last_month  and _days_span(last_month)  < 15)
    return all_months, first_month, last_month, first_partial, last_partial


def _build_monthly_category_breakdown(
    spending: pd.DataFrame,
    all_months: list,
    first_month: str | None,
    last_month: str | None,
    first_partial: bool,
    last_partial: bool,
) -> tuple[dict, str | None]:
    """Per-month spending by category and subcategory. Returns (monthly_by_category, last_complete_month)."""
    spending = spending.copy()
    spending["_month"] = spending["transaction_date"].dt.to_period("M").astype(str)

    monthly_by_category = {}
    for month, grp in spending.groupby("_month"):
        is_partial = (month == first_month and first_partial) or (month == last_month and last_partial)
        by_cat = (
            grp.groupby("primary_category")["amount"]
            .sum().round(2).sort_values(ascending=False).to_dict()
        )
        by_subcat = {
            cat: cat_grp.groupby("sub_category")["amount"].sum().round(2).sort_values(ascending=False).to_dict()
            for cat, cat_grp in grp.groupby("primary_category")
        }
        monthly_by_category[month] = {"is_partial": is_partial, "spending": by_cat, "subcategories": by_subcat}

    last_complete_month = (
        all_months[-2] if (last_partial and len(all_months) >= 2) else last_month
    )
    return monthly_by_category, last_complete_month


def _build_subscription_history(spending: pd.DataFrame, subscriptions: list) -> dict:
    """Per-month actual payment amounts for each active subscription."""
    if not subscriptions:
        return {}
    spending = spending.copy()
    spending["_month"] = spending["transaction_date"].dt.to_period("M").astype(str)
    history = {}
    for sub in subscriptions:
        merchant = str(sub.get("merchant_name") or "").strip()
        desc = str(sub.get("description") or "").strip()
        lookup = merchant or desc
        if not lookup:
            continue
        mask = spending["merchant_name"].str.contains(lookup, case=False, na=False)
        if not mask.any() and desc:
            mask = spending["name"].str.contains(desc, case=False, na=False)
        if not mask.any():
            continue
        monthly = spending[mask].groupby("_month")["amount"].sum().round(2).to_dict()
        history[lookup] = {
            "frequency": sub.get("frequency", ""),
            "average_amount": sub.get("average_amount", 0),
            "monthly_payments": monthly,
        }
    return history


def _build_merchant_monthly(spending: pd.DataFrame, top_n: int = 30) -> dict:
    """Monthly spending per top merchant — answers 'how much did I pay Netflix last month?'"""
    spending = spending.copy()
    spending["_month"] = spending["transaction_date"].dt.to_period("M").astype(str)
    # Exclude transfers — only real merchant spending
    valid = spending[
        spending["merchant_name"].notna() &
        (spending["merchant_name"] != "NULL") &
        (spending["primary_category"] != "TRANSFER_OUT")
    ]
    # Last 12 months only to cap JSON size
    cutoff = (pd.Timestamp.today() - pd.DateOffset(months=12)).to_period("M").strftime("%Y-%m")
    valid = valid[valid["_month"] >= cutoff]
    top = valid.groupby("merchant_name")["amount"].sum().nlargest(top_n).index.tolist()
    return {
        m: valid[valid["merchant_name"] == m].groupby("_month")["amount"].sum().round(2).to_dict()
        for m in top
    }


def _build_top_transactions(spending: pd.DataFrame, last_complete_month: str | None) -> list:
    """Top 15 non-transfer transactions from the most recent complete month."""
    if not last_complete_month:
        return []
    spending = spending.copy()
    spending["_month"] = spending["transaction_date"].dt.to_period("M").astype(str)
    subset = spending[
        (spending["_month"] == last_complete_month) &
        (spending["primary_category"] != "TRANSFER_OUT")
    ].nlargest(15, "amount")
    cols = ["transaction_date", "merchant_name", "primary_category", "sub_category", "amount"]
    if "name" in subset.columns:
        cols.insert(2, "name")
    return (
        subset[cols]
        .assign(transaction_date=lambda x: x["transaction_date"].dt.strftime("%Y-%m-%d"))
        .fillna("").to_dict("records")
    )


def _build_mom_category_delta(
    monthly_by_category: dict,
    all_months: list,
    first_month: str | None,
    last_month: str | None,
    first_partial: bool,
    last_partial: bool,
) -> dict:
    """Month-over-month change per category using last two complete months."""
    complete = [
        m for m in all_months
        if not ((m == first_month and first_partial) or (m == last_month and last_partial))
    ]
    if len(complete) < 2:
        return {}

    prev_m, curr_m = complete[-2], complete[-1]
    prev_cats = monthly_by_category[prev_m]["spending"]
    curr_cats = monthly_by_category[curr_m]["spending"]

    delta = {}
    for cat in set(prev_cats) | set(curr_cats):
        p, c = prev_cats.get(cat, 0), curr_cats.get(cat, 0)
        delta[cat] = {
            "prev_month": prev_m, "curr_month": curr_m,
            "prev_amount": round(p, 2), "curr_amount": round(c, 2),
            "change_usd": round(c - p, 2),
            "change_pct": round(((c - p) / p) * 100, 1) if p else None,
        }
    return dict(sorted(delta.items(), key=lambda x: abs(x[1]["change_usd"]), reverse=True))


def _build_leisure(spending: pd.DataFrame, monthly_by_category: dict, all_months: list) -> dict:
    """All-time and per-month leisure (ENTERTAINMENT + TRAVEL) totals."""
    cats = ["ENTERTAINMENT", "TRAVEL"]
    total = round(float(spending[spending["primary_category"].isin(cats)]["amount"].sum()), 2)
    by_month = {
        m: round(sum(monthly_by_category[m]["spending"].get(c, 0) for c in cats), 2)
        for m in all_months
    }
    breakdown = {}
    for cat in cats:
        sub_df = spending[spending["primary_category"] == cat]
        if not sub_df.empty:
            breakdown[cat] = (
                sub_df.groupby("sub_category")["amount"]
                .sum().round(2).sort_values(ascending=False).to_dict()
            )
    return {"total_all_time": total, "categories_included": cats, "by_month": by_month, "breakdown": breakdown}


def _build_yearly_spending(spending: pd.DataFrame) -> dict:
    yearly = {}
    for year, grp in spending.groupby(spending["transaction_date"].dt.year):
        n_months = grp["transaction_date"].dt.to_period("M").nunique()
        total = round(float(grp["amount"].sum()), 2)
        yearly[int(year)] = {
            "total": total,
            "months_of_data": int(n_months),
            "avg_per_month": round(total / n_months, 2),
        }
    return yearly


def _load_accounts(accounts_path: str | None) -> tuple[list, float | None]:
    if not accounts_path:
        return [], None
    acc = pd.read_csv(accounts_path)
    acc = acc[~acc["is_deleted"]]
    acc = acc.drop_duplicates(subset=["name", "subtype", "current_balance"])
    summary = (
        acc[["name", "subtype", "type", "current_balance", "available_balance"]]
        .rename(columns={"current_balance": "balance", "available_balance": "available"})
        .round(2).to_dict("records")
    )
    net_worth = round(float(acc["current_balance"].sum()), 2)
    return summary, net_worth


def _load_subscriptions(recurring_path: str | None) -> list:
    if not recurring_path:
        return []
    rec = pd.read_csv(recurring_path)
    rec = rec[~rec["is_deleted"]]
    active = rec[rec["is_active"] & ~rec["income_tx"]].copy()
    # Exclude transfers — savings, P2P (Venmo), and loan payments are not subscriptions
    if "primary_category" in active.columns:
        active = active[~active["primary_category"].isin(["TRANSFER_OUT", "LOAN_PAYMENTS"])]
    active = active.drop_duplicates(subset=["description", "frequency", "average_amount"])
    active = active.sort_values("average_amount", ascending=False)
    return (
        active[[
            "merchant_name", "description", "primary_category", "sub_category",
            "frequency", "average_amount", "last_amount",
            "last_date", "predicted_next_date", "status",
        ]].fillna("").to_dict("records")
    )


def _build_subscription_spending_by_month(subscription_history: dict) -> dict:
    """Pre-aggregated total subscription spend per month — sum across all active subscriptions."""
    by_month: dict[str, float] = {}
    for hist in subscription_history.values():
        for month, amt in hist["monthly_payments"].items():
            by_month[month] = round(by_month.get(month, 0) + amt, 2)
    return dict(sorted(by_month.items()))


def _build_subscription_aggregates(subscription_by_month: dict) -> dict:
    """Pre-computed annual and quarterly subscription totals — eliminates arithmetic errors."""
    # Annual
    annual: dict[str, dict] = {}
    for month, amt in subscription_by_month.items():
        year = month[:4]
        entry = annual.setdefault(year, {"total": 0.0, "months": 0})
        entry["total"] = round(entry["total"] + amt, 2)
        entry["months"] += 1
    annual_out = {
        y: {"total": v["total"], "avg_per_month": round(v["total"] / v["months"], 2), "months_of_data": v["months"]}
        for y, v in sorted(annual.items())
    }

    # Quarterly
    quarterly: dict[str, dict] = {}
    for month, amt in subscription_by_month.items():
        year = month[:4]
        q = (int(month[5:7]) - 1) // 3 + 1
        key = f"{year}-Q{q}"
        entry = quarterly.setdefault(key, {"total": 0.0, "months": 0})
        entry["total"] = round(entry["total"] + amt, 2)
        entry["months"] += 1
    quarterly_out = {
        k: {"total": v["total"], "avg_per_month": round(v["total"] / v["months"], 2)}
        for k, v in sorted(quarterly.items())
    }

    return {"annual": annual_out, "quarterly": quarterly_out}


def _build_upcoming_bills(subscriptions: list) -> list:
    today  = datetime.date.today()
    cutoff = today + datetime.timedelta(days=30)
    bills  = []
    for sub in subscriptions:
        raw = sub.get("predicted_next_date", "")
        if not raw:
            continue
        try:
            next_date = datetime.date.fromisoformat(str(raw)[:10])
            if today <= next_date <= cutoff:
                bills.append({
                    "name": sub.get("merchant_name") or sub.get("description", ""),
                    "amount": sub.get("average_amount", 0),
                    "due_date": str(next_date),
                    "frequency": sub.get("frequency", ""),
                })
        except (ValueError, TypeError):
            continue
    return sorted(bills, key=lambda x: x["due_date"])


def _build_safe_to_spend(
    accounts_summary: list,
    upcoming_bills: list,
    monthly_by_category: dict,
) -> dict | None:
    checking = [a for a in accounts_summary if a.get("subtype") == "checking"]
    if not checking:
        return None
    checking_available = round(sum(a.get("available") or a.get("balance") or 0 for a in checking), 2)
    upcoming_total     = round(sum(b["amount"] for b in upcoming_bills), 2)
    current_month      = pd.Timestamp.today().to_period("M").strftime("%Y-%m")
    mtd_cats           = monthly_by_category.get(current_month, {}).get("spending", {})
    mtd_spending       = round(sum(v for k, v in mtd_cats.items() if k != "TRANSFER_OUT"), 2)
    return {
        "checking_available": checking_available,
        "upcoming_bills_30d": upcoming_total,
        "mtd_spending_ex_transfers": mtd_spending,
        "result": round(checking_available - upcoming_total - mtd_spending, 2),
        "note": f"checking balance minus {len(upcoming_bills)} upcoming bills minus month-to-date spending",
    }


def _load_salary(salary_path: str | None, income_df: pd.DataFrame) -> tuple[dict, str | None]:
    if not salary_path:
        return {}, None
    sal = pd.read_csv(salary_path)
    if sal.empty:
        return {}, None
    row = sal.iloc[0]
    stated    = float(row["user_provided"])
    identified = float(row["finley_identified"])
    gap_pct   = round(abs(identified - stated) / max(stated, 0.01) * 100, 1)
    info = {
        "user_stated_monthly": stated,
        "finley_identified_monthly": identified,
        "discrepancy_pct": gap_pct,
        "flag": gap_pct > 20,
    }
    flag_msg = None
    if gap_pct > 20:
        flag_msg = (
            f"User stated ${stated:,.0f}/month but Finley identified "
            f"${identified:,.0f}/month ({gap_pct}% gap). "
            "Resolve before building financial plan."
        )
    return info, flag_msg


# ── Public API ────────────────────────────────────────────────────────────────

def process_transactions(
    tx_path: str,
    accounts_path: str = None,
    recurring_path: str = None,
    salary_path: str = None,
) -> dict:
    """
    Process raw CSV files into a structured JSON summary for LLM consumption.
    Results are cached by file hash — re-runs on unchanged data are instant.
    """
    key = _cache_key(tx_path, accounts_path or "", recurring_path or "", salary_path or "")
    cached = _load_cache(key)
    if cached is not None:
        return cached

    df, spending, income = _load_transactions(tx_path)

    basics     = _build_basic_aggregates(spending)
    all_months, first_month, last_month, first_partial, last_partial = _detect_partial_months(spending)

    monthly_by_category, last_complete_month = _build_monthly_category_breakdown(
        spending, all_months, first_month, last_month, first_partial, last_partial
    )
    top_transactions = _build_top_transactions(spending, last_complete_month)
    mom_delta        = _build_mom_category_delta(
        monthly_by_category, all_months, first_month, last_month, first_partial, last_partial
    )
    leisure          = _build_leisure(spending, monthly_by_category, all_months)
    yearly_spending  = _build_yearly_spending(spending)

    accounts_summary, net_worth_liquid = _load_accounts(accounts_path)
    subscriptions            = _load_subscriptions(recurring_path)
    subscription_history     = _build_subscription_history(spending, subscriptions)
    subscription_by_month    = _build_subscription_spending_by_month(subscription_history)
    subscription_aggregates  = _build_subscription_aggregates(subscription_by_month)
    merchant_monthly         = _build_merchant_monthly(spending)
    upcoming_bills      = _build_upcoming_bills(subscriptions)
    safe_to_spend       = _build_safe_to_spend(accounts_summary, upcoming_bills, monthly_by_category) if accounts_path else None
    salary_info, income_discrepancy = _load_salary(salary_path, income)

    income = income.copy()
    income["_month"] = income["transaction_date"].dt.to_period("M").astype(str)
    monthly_income = income.groupby("_month")["amount"].sum().abs().round(2).to_dict()

    def _subcat(cat: str) -> dict:
        sub = spending[spending["primary_category"] == cat]
        if sub.empty:
            return {}
        return sub.groupby("sub_category")["amount"].sum().round(2).sort_values(ascending=False).to_dict()

    total_spending = round(float(spending["amount"].sum()), 2)
    total_income   = round(float(abs(income["amount"].sum())), 2)

    months = sorted(basics["monthly"].keys())
    mom_change = None
    if len(months) >= 2:
        prev, curr = basics["monthly"][months[-2]], basics["monthly"][months[-1]]
        if prev:
            mom_change = round(((curr - prev) / prev) * 100, 1)

    result = {
        "date_range": {
            "from": str(df["transaction_date"].min().date()),
            "to":   str(df["transaction_date"].max().date()),
        },
        "totals": {
            "total_spending":    total_spending,
            "total_income":      total_income,
            "transaction_count": int(len(spending)),
            "net_cashflow":      round(total_income - total_spending, 2),
        },
        "accounts":               accounts_summary,
        "net_worth_liquid":       net_worth_liquid,
        "salary":                 salary_info,
        "income_discrepancy_flag": income_discrepancy,
        "monthly_spending":       basics["monthly"],
        "yearly_spending":        yearly_spending,
        "month_over_month_change_pct": mom_change,
        "by_category":            basics["by_category"],
        "top_merchants":          basics["top_merchants"],
        "subscriptions":          subscriptions,
        "food_breakdown":          _subcat("FOOD_AND_DRINK"),
        "entertainment_breakdown": _subcat("ENTERTAINMENT"),
        "transportation_breakdown":_subcat("TRANSPORTATION"),
        "loan_payments_breakdown": _subcat("LOAN_PAYMENTS"),
        "rent_utilities_breakdown":_subcat("RENT_AND_UTILITIES"),
        "medical_breakdown":       _subcat("MEDICAL"),
        "personal_care_breakdown": _subcat("PERSONAL_CARE"),
        "general_merchandise_breakdown": _subcat("GENERAL_MERCHANDISE"),
        "general_services_breakdown":    _subcat("GENERAL_SERVICES"),
        "transfer_breakdown":      _subcat("TRANSFER_OUT"),
        "monthly_income":          monthly_income,
        "upcoming_bills":          upcoming_bills,
        "safe_to_spend":           safe_to_spend,
        "last_complete_month":     last_complete_month,
        "monthly_by_category":     monthly_by_category,
        "top_transactions_recent_month": {
            "month": last_complete_month,
            "note": "transfers excluded",
            "transactions": top_transactions,
        },
        "mom_category_delta": mom_delta,
        "leisure":            leisure,
        "subscription_monthly_history": subscription_history,
        "subscription_spending_by_month": subscription_by_month,
        "subscription_aggregates": subscription_aggregates,
        "merchant_monthly":  merchant_monthly,
    }

    _save_cache(key, result)
    return result
