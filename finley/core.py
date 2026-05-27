import json
import os
import re
import time

from finley.client import make_client, MODEL, AWS_KEY, AWS_SECRET_KEY, AWS_REGION
from finley.data import process_transactions
from finley.prompt import SYSTEM_PROMPT, EVAL_SYSTEM, build_prompt
from config import TX_FILE, ACCOUNTS_FILE, RECURRING_FILE, SALARY_FILE


def collect_user_profile() -> dict:
    print("\n" + "=" * 60)
    print("  FINLEY — Personal Financial Advisor")
    print("=" * 60)
    print("\nFour quick questions before I analyze your finances.\n")

    age    = input("1. How old are you?\n   → ").strip()
    income = input("\n2. What is your monthly take-home pay (after tax)?\n   → $").strip()

    print("\n3. What is your #1 financial goal right now?")
    print("   a) Eliminate debt")
    print("   b) Build emergency fund")
    print("   c) Save for a home down payment")
    print("   d) Build retirement savings")
    print("   e) Just understand where my money goes")
    goal_map = {
        "a": "eliminate debt",
        "b": "build emergency fund",
        "c": "save for a home down payment",
        "d": "build retirement savings",
        "e": "understand spending patterns",
    }
    goal = goal_map.get(input("   → ").strip().lower(), "understand spending patterns")

    print("\n4. Financial status (y/n):")
    emergency  = input("   Do you have 3+ months of expenses saved? → ").strip().lower()
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
    raw     = response.content[0].text
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        start = cleaned.index("{")
        end   = cleaned.rindex("}") + 1
        return json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Could not parse JSON", "raw": raw}


def print_eval_report(result: dict):
    if "error" in result:
        print(f"  Eval error: {result['error']}")
        return

    s = result.get("summary", {})
    print(f"  Claims: {s.get('total_claims')} | Passed: {s.get('passed')} | "
          f"Failed: {s.get('failed')} | Unverifiable: {s.get('unverifiable')}")
    print(f"  Accuracy: {s.get('accuracy_score')}%")

    for c in result.get("claims", []):
        if c.get("verdict") == "FAIL":
            print(f"    ✗ \"{c.get('claim')}\"")
            print(f"      Claimed: {c.get('value_claimed')} | Actual: {c.get('value_actual')} | Off: {c.get('delta_pct')}%")


def main():  # noqa: C901
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
    print(f" done ({time.time() - t0:.1f}s)")
    print(f"  {summary['totals']['transaction_count']} transactions")
    print(f"  {summary['date_range']['from']} to {summary['date_range']['to']}")
    print(f"  Total spend: ${summary['totals']['total_spending']:,.2f}")
    if summary.get("income_discrepancy_flag"):
        print(f"  [!] {summary['income_discrepancy_flag']}")
    if summary.get("net_worth_liquid"):
        print(f"  Liquid net worth: ${summary['net_worth_liquid']:,.2f}")

    print("\n" + "=" * 60)
    print("  Analyzing your finances...")
    print("=" * 60 + "\n")

    client  = make_client()
    message = build_prompt({}, summary)

    with client.messages.stream(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
