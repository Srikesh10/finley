"""
finley.py — Finley AI Financial Advisor
Layer 2: Process CSV → Layer 3: User Profile → Layer 4: Advice
"""

import os
import re
import json
import time
import pandas as pd
from anthropic import AnthropicBedrock
from dotenv import load_dotenv
load_dotenv()

CSV_FILE = "data/all_transactions.csv"
MODEL    = "us.anthropic.claude-opus-4-6-v1"

AWS_KEY        = os.getenv("AWS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")

# ── Langfuse Tracing (optional — graceful fallback if keys not set) ───────────
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))

_langfuse_enabled = False
_lf_client = None
if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
    try:
        from langfuse import Langfuse
        _lf_client = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
        _langfuse_enabled = _lf_client.auth_check()
    except Exception:
        _langfuse_enabled = False


def _lf_log(name: str, model: str, input_msgs: list, output: str,
            input_tokens: int = 0, output_tokens: int = 0,
            session_id: str = None, metadata: dict = None):
    if not _langfuse_enabled:
        return
    try:
        with _lf_client.start_as_current_observation(
            name=name,
            as_type="generation",
            input=input_msgs,
            output=output,
            model=model,
            usage_details={"input": input_tokens, "output": output_tokens},
            metadata=metadata or {},
        ):
            pass
    except Exception:
        pass


def _lf_score(name: str, value: float):
    if not _langfuse_enabled:
        return
    try:
        _lf_client.score_current_trace(name=name, value=value)
    except Exception:
        pass


def _lf_flush():
    if _langfuse_enabled:
        try:
            _lf_client.flush()
        except Exception:
            pass

# ── Layer 4 System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Finley, a sharp personal financial advisor. You have a structured JSON summary of the user's finances. Answer questions using exact numbers from the data — no fabrication, no hedging.

DATA FIELD GUIDE — where to look for each question type

LAST MONTH questions ("what did I spend last month", "groceries last month"):
  → Use `last_complete_month` (e.g. "2025-04"). Never use a month where is_partial=true as "last month".
  → Spending by category: monthly_by_category[last_complete_month]["spending"]
  → Subcategory detail: monthly_by_category[last_complete_month]["subcategories"][CATEGORY]
  → Example: groceries = monthly_by_category[last_complete_month]["subcategories"]["FOOD_AND_DRINK"]["GROCERIES"]

MONTHLY TREND / AVERAGE questions:
  → Iterate monthly_by_category. Skip months where is_partial=true when computing averages.
  → Month-over-month change: mom_category_delta[CATEGORY] has prev_month, curr_month, change_usd, change_pct (complete months only).

INCOME questions:
  → monthly_income[month] = income deposited that month. Average = sum of complete months ÷ count.

SAFE TO SPEND / CASH AVAILABLE:
  → safe_to_spend.result = pre-computed: checking balance − upcoming bills − month-to-date spending.
  → Fields: checking_available, upcoming_bills_30d, mtd_spending_ex_transfers, result.

UPCOMING BILLS / SUBSCRIPTIONS DUE:
  → upcoming_bills list — each has name, amount, due_date, frequency. Already filtered to next 30 days.
  → Full subscription list: subscriptions.

LARGEST / TOP TRANSACTIONS:
  → top_transactions_recent_month.transactions — transfers (TRANSFER_OUT) already excluded.
  → Each record: transaction_date, merchant_name, name, primary_category, sub_category, amount.

LEISURE / ENTERTAINMENT SPEND:
  → leisure.total_all_time — all-time total across ENTERTAINMENT + TRAVEL.
  → leisure.by_month[month] — leisure spend per month.
  → leisure.breakdown — subcategory detail.

ACCOUNT BALANCES:
  → accounts_summary list — each has name, subtype, balance, available.

ACCURACY RULES
1. Never invent numbers. If a field is absent, say so explicitly and tell the user what you'd need.
2. TRANSFER_OUT is not consumption. Do not include it in spending totals or averages.
3. For "last month", always use last_complete_month — never a partial month.
4. For monthly averages, divide by count of complete months only.
5. If you do any arithmetic, show the numbers: "Total $X over Y months = $Z/month."

FINANCIAL PRIORITIES (apply when giving advice)
1. Capture employer 401k match first — if unknown, tell the user to confirm with HR.
2. Eliminate high-interest debt above 15% APR.
3. Build emergency fund (3-6 months essential expenses).
4. Max HSA if eligible (triple tax advantage). 2025 limits: $4,300 individual / $8,550 family.
5. Roth IRA / retirement contributions. 2025 limits: $7,000 under 50 / $8,000 for 50+.
Never recommend step 3-5 if step 1-2 problems exist.

QUESTIONS YOU CANNOT FULLY ANSWER
Answer what you can from the data, then flag the gap cleanly.

Cannot answer at all → flag + redirect:
  - Investment picks, credit score, tax filing, specific APRs, insurance product picks, legal/estate questions.
  - "I don't have [X] in your transaction data. [One sentence on what you can offer instead]."

Partially answerable → answer the data portion, flag the unknown:
  - "Can I afford a house / quit my job / make a big purchase?" → use safe_to_spend.result and monthly surplus. Flag credit score / down payment if not in data.
  - "Am I on track for retirement?" → use income and monthly surplus. Ask for retirement account balance if not visible.
  - "What's my total debt?" → "I see $X/month in debt payments but not the balances. Tell me the balance and I'll build a payoff plan."

Never say "I'm just an AI." Answer what you can, flag what you can't, and offer the next useful step.

RESPONSE STYLE
- Open with the key number or finding, not a preamble.
- Every dollar figure must come from the data. Label monthly vs annual correctly.
- End with one specific action: what to do, this week.
- Never say "consider" — state what to do."""


# ── Layer 2: Process Data ─────────────────────────────────────────────────────

def process_transactions(
    tx_path: str,
    accounts_path: str = None,
    recurring_path: str = None,
    salary_path: str = None,
) -> dict:
    # ── Transactions ──────────────────────────────────────────────────────────
    df = pd.read_csv(tx_path, low_memory=False)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    # Filter out deleted and pending transactions
    if "is_deleted" in df.columns:
        df = df[df["is_deleted"] == False]
    if "pending" in df.columns:
        df = df[df["pending"] == False]

    spending = df[df["amount"] > 0]
    income   = df[df["amount"] < 0]

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

    months = sorted(monthly.keys())
    mom_change = None
    if len(months) >= 2:
        prev, curr = monthly[months[-2]], monthly[months[-1]]
        if prev:
            mom_change = round(((curr - prev) / prev) * 100, 1)

    def subcategory_breakdown(primary_cat):
        cat_df = spending[spending["primary_category"] == primary_cat]
        if cat_df.empty:
            return {}
        return (
            cat_df.groupby("sub_category")["amount"]
            .sum().round(2).sort_values(ascending=False).to_dict()
        )

    # ── Monthly category breakdown (fixes "last month" questions) ────────────
    spending = spending.copy()
    spending["_month"] = spending["transaction_date"].dt.to_period("M").astype(str)

    # Detect partial months: a month is partial if it has < 15 days of data
    # and is the most or least recent month in the dataset
    all_months_sorted = sorted(spending["_month"].unique())
    first_month = all_months_sorted[0] if all_months_sorted else None
    last_month  = all_months_sorted[-1] if all_months_sorted else None

    def _days_in_month_data(month_str):
        m_data = spending[spending["_month"] == month_str]["transaction_date"]
        return int((m_data.max() - m_data.min()).days) + 1 if len(m_data) > 1 else 1

    first_partial = first_month and _days_in_month_data(first_month) < 15
    last_partial  = last_month  and _days_in_month_data(last_month)  < 15

    monthly_by_category = {}
    for month, grp in spending.groupby("_month"):
        is_partial = (month == first_month and first_partial) or (month == last_month and last_partial)
        by_cat = (
            grp.groupby("primary_category")["amount"]
            .sum().round(2).sort_values(ascending=False).to_dict()
        )
        # Subcategory breakdown within each month (fixes "groceries last month" questions)
        by_subcat = {}
        for cat, cat_grp in grp.groupby("primary_category"):
            by_subcat[cat] = (
                cat_grp.groupby("sub_category")["amount"]
                .sum().round(2).sort_values(ascending=False).to_dict()
            )
        monthly_by_category[month] = {
            "is_partial": is_partial,
            "spending": by_cat,
            "subcategories": by_subcat,
        }

    # "Last month" = most recent COMPLETE month (skip partial tail if present)
    last_complete_month = (
        all_months_sorted[-2] if (last_partial and len(all_months_sorted) >= 2)
        else last_month
    )

    # ── Top transactions — most recent COMPLETE month, non-transfer only ───────
    top_transactions_recent = []
    if last_complete_month:
        non_transfer = spending[
            (spending["_month"] == last_complete_month) &
            (spending["primary_category"] != "TRANSFER_OUT")
        ].nlargest(15, "amount")
        top_transactions_recent = (
            non_transfer[["transaction_date", "merchant_name", "name", "primary_category", "sub_category", "amount"]]
            .assign(transaction_date=lambda x: x["transaction_date"].dt.strftime("%Y-%m-%d"))
            .fillna("")
            .to_dict("records")
        )

    # ── Month-over-month category delta (uses last two complete months) ────────
    mom_category_delta = {}
    complete_months = [m for m in all_months_sorted
                       if not ((m == first_month and first_partial) or (m == last_month and last_partial))]
    if len(complete_months) >= 2:
        prev_m, curr_m = complete_months[-2], complete_months[-1]
        prev_cats = monthly_by_category[prev_m]["spending"]
        curr_cats = monthly_by_category[curr_m]["spending"]
        for cat in set(prev_cats) | set(curr_cats):
            p = prev_cats.get(cat, 0)
            c = curr_cats.get(cat, 0)
            mom_category_delta[cat] = {
                "prev_month": prev_m,
                "curr_month": curr_m,
                "prev_amount": round(p, 2),
                "curr_amount": round(c, 2),
                "change_usd": round(c - p, 2),
                "change_pct": round(((c - p) / p) * 100, 1) if p else None,
            }
        mom_category_delta = dict(
            sorted(mom_category_delta.items(), key=lambda x: abs(x[1]["change_usd"]), reverse=True)
        )

    # ── Leisure / Entertainment mapping (ENTERTAINMENT + TRAVEL → "leisure") ───
    leisure_categories = ["ENTERTAINMENT", "TRAVEL"]
    leisure_total = round(float(
        spending[spending["primary_category"].isin(leisure_categories)]["amount"].sum()
    ), 2)
    leisure_by_month = {
        m: round(sum(monthly_by_category[m]["spending"].get(c, 0) for c in leisure_categories), 2)
        for m in all_months_sorted
    }
    leisure_breakdown = {}
    for cat in leisure_categories:
        sub = subcategory_breakdown(cat)
        if sub:
            leisure_breakdown[cat] = sub

    yearly_spending = {}
    for year, grp in spending.groupby(spending["transaction_date"].dt.year):
        months_in_year = grp["transaction_date"].dt.to_period("M").nunique()
        total = round(float(grp["amount"].sum()), 2)
        yearly_spending[int(year)] = {
            "total": total,
            "months_of_data": int(months_in_year),
            "avg_per_month": round(total / months_in_year, 2),
        }

    # ── Accounts (deduped) ────────────────────────────────────────────────────
    accounts_summary = {}
    net_worth_liquid = None
    if accounts_path:
        acc = pd.read_csv(accounts_path)
        acc = acc[acc["is_deleted"] == False]
        # Deduplicate: same name + subtype + balance = same real account
        acc_deduped = acc.drop_duplicates(subset=["name", "subtype", "current_balance"])
        accounts_summary = (
            acc_deduped[["name", "subtype", "type", "current_balance", "available_balance"]]
            .rename(columns={"current_balance": "balance", "available_balance": "available"})
            .round(2)
            .to_dict("records")
        )
        net_worth_liquid = round(float(acc_deduped["current_balance"].sum()), 2)

    # ── Recurring / Subscriptions (from Plaid-tagged data) ───────────────────
    subscriptions = []
    if recurring_path:
        rec = pd.read_csv(recurring_path)
        rec = rec[rec["is_deleted"] == False]
        active = rec[(rec["is_active"] == True) & (rec["income_tx"] == False)].copy()
        # Deduplicate: same description + frequency + average_amount = same real stream
        active = active.drop_duplicates(subset=["description", "frequency", "average_amount"])
        active = active.sort_values("average_amount", ascending=False)
        subscriptions = (
            active[[
                "merchant_name", "description", "primary_category", "sub_category",
                "frequency", "average_amount", "last_amount",
                "last_date", "predicted_next_date", "status"
            ]]
            .fillna("")
            .to_dict("records")
        )

    # ── Salary / Income ───────────────────────────────────────────────────────
    salary_info = {}
    income_discrepancy = None
    if salary_path:
        sal = pd.read_csv(salary_path)
        if not sal.empty:
            row = sal.iloc[0]
            user_stated        = float(row["user_provided"])
            finley_identified  = float(row["finley_identified"])
            gap_pct = round(abs(finley_identified - user_stated) / max(user_stated, 0.01) * 100, 1)
            salary_info = {
                "user_stated_monthly": user_stated,
                "finley_identified_monthly": finley_identified,
                "discrepancy_pct": gap_pct,
                "flag": gap_pct > 20,
            }
            if gap_pct > 20:
                income_discrepancy = (
                    f"User stated ${user_stated:,.0f}/month but Finley identified "
                    f"${finley_identified:,.0f}/month ({gap_pct}% gap). "
                    f"Resolve before building financial plan."
                )

    total_income = round(float(abs(income["amount"].sum())), 2)
    total_spending = round(float(spending["amount"].sum()), 2)

    # ── Monthly income breakdown ──────────────────────────────────────────────
    income = income.copy()
    income["_month"] = income["transaction_date"].dt.to_period("M").astype(str)
    monthly_income = (
        income.groupby("_month")["amount"].sum().abs().round(2).to_dict()
    )

    # ── Upcoming bills (recurring subscriptions due in next 30 days) ─────────
    upcoming_bills = []
    if subscriptions:
        import datetime
        today = datetime.date.today()
        cutoff = today + datetime.timedelta(days=30)
        for sub in subscriptions:
            next_date_str = sub.get("predicted_next_date", "")
            if not next_date_str:
                continue
            try:
                next_date = datetime.date.fromisoformat(str(next_date_str)[:10])
                if today <= next_date <= cutoff:
                    upcoming_bills.append({
                        "name": sub.get("merchant_name") or sub.get("description", ""),
                        "amount": sub.get("average_amount", 0),
                        "due_date": str(next_date),
                        "frequency": sub.get("frequency", ""),
                    })
            except (ValueError, TypeError):
                continue
        upcoming_bills.sort(key=lambda x: x["due_date"])

    # ── Safe to spend (pre-computed) ──────────────────────────────────────────
    safe_to_spend = None
    checking_available = None
    if accounts_path and accounts_summary:
        checking_accts = [a for a in accounts_summary if a.get("subtype") == "checking"]
        if checking_accts:
            checking_available = round(sum(a.get("available") or a.get("balance") or 0 for a in checking_accts), 2)

    upcoming_total = round(sum(b["amount"] for b in upcoming_bills), 2)

    # MTD spending: current partial month excluding transfers
    current_month_str = pd.Timestamp.today().to_period("M").strftime("%Y-%m")
    mtd_cats = monthly_by_category.get(current_month_str, {}).get("spending", {})
    mtd_spending = round(sum(v for k, v in mtd_cats.items() if k != "TRANSFER_OUT"), 2)

    if checking_available is not None:
        safe_to_spend = {
            "checking_available": checking_available,
            "upcoming_bills_30d": upcoming_total,
            "mtd_spending_ex_transfers": mtd_spending,
            "result": round(checking_available - upcoming_total - mtd_spending, 2),
            "note": f"checking balance minus {len(upcoming_bills)} upcoming bills minus month-to-date spending",
        }

    return {
        "date_range": {
            "from": str(df["transaction_date"].min().date()),
            "to": str(df["transaction_date"].max().date()),
        },
        "totals": {
            "total_spending": total_spending,
            "total_income": total_income,
            "transaction_count": int(len(spending)),
            "net_cashflow": round(total_income - total_spending, 2),
        },
        "accounts": accounts_summary,
        "net_worth_liquid": net_worth_liquid,
        "salary": salary_info,
        "income_discrepancy_flag": income_discrepancy,
        "monthly_spending": monthly,
        "yearly_spending": yearly_spending,
        "month_over_month_change_pct": mom_change,
        "by_category": by_category,
        "top_merchants": top_merchants,
        "subscriptions": subscriptions,
        "food_breakdown": subcategory_breakdown("FOOD_AND_DRINK"),
        "entertainment_breakdown": subcategory_breakdown("ENTERTAINMENT"),
        "transportation_breakdown": subcategory_breakdown("TRANSPORTATION"),
        "loan_payments_breakdown": subcategory_breakdown("LOAN_PAYMENTS"),
        "rent_utilities_breakdown": subcategory_breakdown("RENT_AND_UTILITIES"),
        "medical_breakdown": subcategory_breakdown("MEDICAL"),
        "personal_care_breakdown": subcategory_breakdown("PERSONAL_CARE"),
        "general_merchandise_breakdown": subcategory_breakdown("GENERAL_MERCHANDISE"),
        "general_services_breakdown": subcategory_breakdown("GENERAL_SERVICES"),
        "transfer_breakdown": subcategory_breakdown("TRANSFER_OUT"),
        "monthly_income": monthly_income,
        "upcoming_bills": upcoming_bills,
        "safe_to_spend": safe_to_spend,
        "last_complete_month": last_complete_month,
        "monthly_by_category": monthly_by_category,
        "top_transactions_recent_month": {
            "month": last_complete_month,
            "note": "transfers excluded",
            "transactions": top_transactions_recent,
        },
        "mom_category_delta": mom_category_delta,
        "leisure": {
            "total_all_time": leisure_total,
            "categories_included": leisure_categories,
            "by_month": leisure_by_month,
            "breakdown": leisure_breakdown,
        },
    }


# ── Layer 3: User Profile ─────────────────────────────────────────────────────

def collect_user_profile() -> dict:
    print("\n" + "=" * 60)
    print("  FINLEY — Personal Financial Advisor")
    print("=" * 60)
    print("\nFour quick questions before I analyze your finances.\n")

    age = input("1. How old are you?\n   → ").strip()

    income = input("\n2. What is your monthly take-home pay (after tax)?\n   → $").strip()

    print("\n3. What is your #1 financial goal right now?")
    print("   a) Eliminate debt")
    print("   b) Build emergency fund")
    print("   c) Save for a home down payment")
    print("   d) Build retirement savings")
    print("   e) Just understand where my money goes")
    goal_input = input("   → ").strip().lower()
    goal_map = {
        "a": "eliminate debt",
        "b": "build emergency fund",
        "c": "save for a home down payment",
        "d": "build retirement savings",
        "e": "understand spending patterns",
    }
    goal = goal_map.get(goal_input, goal_input)

    print("\n4. Financial status (y/n):")
    emergency = input("   Do you have 3+ months of expenses saved? → ").strip().lower()
    retirement = input("   Are you contributing to a 401k? → ").strip().lower()
    cc_balance = input("   Do you carry a credit card balance each month? → ").strip().lower()

    return {
        "age": age,
        "monthly_take_home": f"${income}",
        "primary_goal": goal,
        "has_3_month_emergency_fund": emergency == "y",
        "contributing_to_401k": retirement == "y",
        "carries_credit_card_balance": cc_balance == "y",
    }


# ── Evaluator ────────────────────────────────────────────────────────────────

EVAL_SYSTEM = """You are a financial data auditor. You will be given a financial advice response and the actual transaction summary it was based on.

Extract every specific dollar figure, count, or percentage claimed in the advice and verify it against the summary data.

Output ONLY valid JSON in this exact format:
{
  "claims": [
    {
      "claim": "exact short quote from advice",
      "value_claimed": 3421.00,
      "value_actual": 3418.55,
      "delta_pct": 0.07,
      "verdict": "PASS",
      "note": "within rounding tolerance"
    }
  ],
  "summary": {
    "total_claims": 12,
    "passed": 10,
    "failed": 2,
    "unverifiable": 1,
    "accuracy_score": 83.3
  }
}

Verdicts: PASS (within 5% tolerance), FAIL (materially wrong), UNVERIFIABLE (future projections, estimates)."""


def evaluate_advice(advice_text: str, summary: dict, client) -> dict:
    prompt = f"""ADVICE:
{advice_text}

TRANSACTION SUMMARY:
{json.dumps(summary, indent=2)}

Verify every dollar figure, count, and percentage claimed."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=EVAL_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?', '', raw).strip()
    # Find the largest JSON object in the response
    try:
        start = cleaned.index('{')
        end = cleaned.rindex('}') + 1
        return json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Could not parse JSON", "raw": raw}


def print_eval_report(result: dict):
    if "error" in result:
        print(f"  Eval error: {result['error']}")
        print(f"  Raw response: {result.get('raw', '')[:500]}")
        return

    s = result.get("summary", {})
    score = s.get("accuracy_score", "?")
    passed = s.get("passed", "?")
    failed = s.get("failed", "?")
    unverifiable = s.get("unverifiable", "?")
    total = s.get("total_claims", "?")

    print(f"  Claims checked: {total} | Passed: {passed} | Failed: {failed} | Unverifiable: {unverifiable}")
    print(f"  Accuracy score: {score}%")

    failed_claims = [c for c in result.get("claims", []) if c.get("verdict") == "FAIL"]
    if failed_claims:
        print(f"\n  FAILED CLAIMS:")
        for c in failed_claims:
            print(f"    ✗ \"{c.get('claim', '')}\"")
            print(f"      Claimed: {c.get('value_claimed')} | Actual: {c.get('value_actual')} | Off by: {c.get('delta_pct')}%")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_prompt(profile: dict, summary: dict) -> str:
    return f"""Analyze my financial situation and give me your honest assessment.

USER PROFILE:
{json.dumps(profile, indent=2)}

TRANSACTION DATA:
{json.dumps(summary, indent=2)}

Give me your honest financial assessment. Start with what matters most."""


from config import TX_FILE, ACCOUNTS_FILE, RECURRING_FILE, SALARY_FILE


def main():
    if not os.path.exists(TX_FILE):
        print(f"Error: {TX_FILE} not found.")
        return

    print("\nLoading transactions...", end="", flush=True)
    t0 = time.time()
    summary = process_transactions(
        tx_path        = TX_FILE,
        accounts_path  = ACCOUNTS_FILE  if os.path.exists(ACCOUNTS_FILE)  else None,
        recurring_path = RECURRING_FILE if os.path.exists(RECURRING_FILE) else None,
        salary_path    = SALARY_FILE    if os.path.exists(SALARY_FILE)    else None,
    )
    print(f" done ({time.time()-t0:.1f}s)")
    print(f"  {summary['totals']['transaction_count']} transactions")
    print(f"  {summary['date_range']['from']} to {summary['date_range']['to']}")
    print(f"  Total spend: ${summary['totals']['total_spending']:,.2f}")
    if summary.get("income_discrepancy_flag"):
        print(f"  [!] {summary['income_discrepancy_flag']}")
    if summary.get("net_worth_liquid"):
        print(f"  Liquid net worth: ${summary['net_worth_liquid']:,.2f}")

    profile = collect_user_profile()

    print("\n" + "=" * 60)
    print("  Analyzing your finances...")
    print("=" * 60 + "\n")

    client = AnthropicBedrock(
        aws_access_key=AWS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region=AWS_REGION,
    )
    user_message = build_prompt(profile, summary)
    messages = [{"role": "user", "content": user_message}]

    session_id = str(int(time.time()))

    t_request_start = time.time()
    first_token = True

    with client.messages.stream(
        model=MODEL,
        max_tokens=10000,
        thinking={"type": "enabled", "budget_tokens": 8000},
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for event in stream:
            if (
                hasattr(event, "type")
                and event.type == "content_block_delta"
                and hasattr(event, "delta")
                and event.delta.type == "text_delta"
            ):
                if first_token:
                    t_first_token = time.time()
                    print(f"[time to first token: {t_first_token - t_request_start:.1f}s]\n", flush=True)
                    first_token = False
                print(event.delta.text, end="", flush=True)

        final = stream.get_final_message()

    t_total = time.time() - t_request_start
    print("\n")
    usage = final.usage
    print(f"[{usage.input_tokens} in / {usage.output_tokens} out | total: {t_total:.1f}s]")

    # Extract text from content blocks for evaluation and history
    advice_text = ""
    for block in final.content:
        if hasattr(block, "type") and block.type == "text":
            advice_text += block.text

    _lf_log(
        name="initial-advice", model=MODEL,
        input_msgs=messages, output=advice_text,
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        session_id=session_id, metadata={"latency_s": round(t_total, 1)},
    )

    messages.append({"role": "assistant", "content": final.content})

    # Auto-evaluate factual accuracy
    print("\n" + "─" * 60)
    print("  Running accuracy check...")
    print("─" * 60)
    eval_result = evaluate_advice(advice_text, summary, client)
    print_eval_report(eval_result)

    score = eval_result.get("summary", {}).get("accuracy_score")
    if score is not None:
        _lf_score(name="factual-accuracy", value=score / 100)

    print("\n" + "─" * 60)
    print("Ask a follow-up (Enter to exit):")

    followup_idx = 0
    while True:
        followup = input("\nYou → ").strip()
        if not followup:
            break

        followup_idx += 1
        messages.append({"role": "user", "content": followup})

        print("\nFinley → ", end="", flush=True)
        response_text = ""
        t_followup_start = time.time()
        first_followup_token = True

        with client.messages.stream(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                if first_followup_token:
                    print(f"[{time.time() - t_followup_start:.1f}s to first token] ", end="", flush=True)
                    first_followup_token = False
                print(text, end="", flush=True)
                response_text += text

        t_followup_total = time.time() - t_followup_start
        messages.append({"role": "assistant", "content": response_text})
        print(f"\n[followup total: {t_followup_total:.1f}s]")

        _lf_log(
            name=f"followup-{followup_idx}", model=MODEL,
            input_msgs=[{"role": "user", "content": followup}],
            output=response_text, session_id=session_id,
            metadata={"latency_s": round(t_followup_total, 1)},
        )

    _lf_flush()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
