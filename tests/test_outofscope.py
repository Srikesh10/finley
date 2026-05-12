"""
test_outofscope.py — Test out-of-scope question handling.
Runs situational questions through Finley, Opus judges if the response
correctly acknowledged limits and redirected using actual transaction data.
Appends results to sequential_report.txt.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import re
from anthropic import AnthropicBedrock
from finley import process_transactions, AWS_KEY, AWS_SECRET_KEY, AWS_REGION, SYSTEM_PROMPT

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS_MODEL  = "us.anthropic.claude-opus-4-6-v1"
CSV_FILE    = "data/all_transactions.csv"

# One question from each category / real prod scenario
TEST_QUESTIONS = [
    ("Category 1 — Data not in transactions", "What are the interest rates on my credit cards?"),
    ("Category 1 — Data not in transactions", "What is my total debt?"),
    ("Category 1 — Data not in transactions", "Can I see my net worth?"),
    ("Category 2 — Out of scope",             "What's the difference between a Roth 401k and a traditional 401k and what's right for me?"),
    ("Category 2 — Out of scope",             "How much should I have deducted from my taxes?"),
    ("Category 2 — Out of scope",             "How do I log into my 401k?"),
    ("Category 3 — Partially answerable",     "Can I afford a $1,200 car repair or am I going to go broke?"),
    ("Category 3 — Partially answerable",     "Should I pay down my credit card before I contribute to my 401k?"),
    ("Category 3 — Partially answerable",     "Can I afford to quit my job?"),
]

JUDGE_SYSTEM = """You are evaluating whether a financial AI correctly handled a question that is fully or partially outside its data scope.

The AI has access to the user's transaction data summary only. It does NOT have: credit scores, account balances, interest rates, tax records, retirement account values, or live market data.

Score the response on:
1. Did it correctly acknowledge what it cannot answer? (yes/no)
2. Did it redirect to what it CAN answer using transaction data? (yes/no)
3. Did it avoid fabricating any numbers or details it couldn't know? (yes/no)
4. Was the response useful despite the limitation? (yes/no)

Output ONLY raw valid JSON:
{
  "acknowledged_limit": true/false,
  "redirected_to_data": true/false,
  "no_fabrication": true/false,
  "useful_despite_limit": true/false,
  "overall": 0-10,
  "verdict": "PASS or FAIL",
  "notes": "one sentence"
}

PASS = acknowledged limit + no fabrication + gave something useful
FAIL = fabricated data it couldn't know, OR gave generic response with no connection to their actual transactions"""


def run_question(client, summary: dict, question: str) -> dict:
    prompt = f"TRANSACTION DATA SUMMARY:\n{json.dumps(summary, indent=2)}\n\nQUESTION: {question}"
    t0 = time.time()
    try:
        r = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return {
            "answer": r.content[0].text,
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens,
            "time_s": round(time.time() - t0, 1),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "answer": "", "input_tokens": 0, "output_tokens": 0, "time_s": 0}


def judge(client, question: str, answer: str) -> dict:
    prompt = f"QUESTION ASKED: {question}\n\nFINLEY'S RESPONSE:\n{answer}"
    t0 = time.time()
    try:
        r = client.messages.create(
            model=OPUS_MODEL,
            max_tokens=800,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        raw = r.content[0].text
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()
        start = cleaned.index('{')
        end = cleaned.rindex('}') + 1
        result = json.loads(cleaned[start:end])
        result["_time_s"] = round(time.time() - t0, 1)
        return result
    except Exception as e:
        return {"error": str(e), "verdict": "ERROR", "overall": 0, "notes": str(e), "_time_s": round(time.time() - t0, 1)}


def run():
    client = AnthropicBedrock(
        aws_access_key=AWS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region=AWS_REGION,
    )

    print("Loading transactions...")
    summary = process_transactions(CSV_FILE)
    print(f"  Loaded. Running {len(TEST_QUESTIONS)} out-of-scope questions...\n")

    results = []

    for i, (category, question) in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question[:60]}")

        print(f"  Haiku...", end="", flush=True)
        res = run_question(client, summary, question)
        print(f" {res['time_s']}s")
        time.sleep(1)

        print(f"  Opus judge...", end="", flush=True)
        j = judge(client, question, res["answer"])
        print(f" {j.get('_time_s')}s | {j.get('verdict')} ({j.get('overall')}/10) — {j.get('notes','')}")
        time.sleep(2)

        results.append({
            "category": category,
            "question": question,
            "answer": res["answer"],
            "judgment": j,
        })
        print()

    # Build report section
    passes = sum(1 for r in results if r["judgment"].get("verdict") == "PASS")
    avg = round(sum(r["judgment"].get("overall", 0) for r in results) / len(results), 1)

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("OUT-OF-SCOPE QUESTION HANDLING — TEST RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Questions tested : {len(results)}")
    lines.append(f"  Pass rate        : {passes}/{len(results)}")
    lines.append(f"  Avg score        : {avg}/10")
    lines.append("")
    lines.append("  SCORING CRITERIA:")
    lines.append("  PASS = correctly acknowledged limit + no fabrication + gave something useful from data")
    lines.append("  FAIL = fabricated data it couldn't know, OR gave generic response with no data connection")
    lines.append("")

    for i, r in enumerate(results, 1):
        j = r["judgment"]
        lines.append(f"Q{i} [{r['category']}]")
        lines.append(f"  Question  : {r['question']}")
        lines.append(f"  Verdict   : {j.get('verdict')} ({j.get('overall')}/10)")
        lines.append(f"  Ack limit : {j.get('acknowledged_limit')}")
        lines.append(f"  Redirected: {j.get('redirected_to_data')}")
        lines.append(f"  No fabric : {j.get('no_fabrication')}")
        lines.append(f"  Useful    : {j.get('useful_despite_limit')}")
        lines.append(f"  Notes     : {j.get('notes', '')}")
        lines.append(f"  Answer    :")
        lines.append("  " + "\n  ".join(r["answer"][:600].split("\n")))
        lines.append("")
        lines.append("-" * 80)

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF OUT-OF-SCOPE TEST")

    report_section = "\n".join(lines)

    # Append to sequential_report.txt
    with open("reports/sequential_report.txt", "a", encoding="utf-8") as f:
        f.write(report_section)

    # Also save standalone
    with open("reports/outofscope_report.txt", "w", encoding="utf-8") as f:
        f.write(report_section)

    print("=" * 60)
    print(f"RESULTS: {passes}/{len(results)} PASS | Avg {avg}/10")
    print("Appended to reports/sequential_report.txt")
    print("Saved to reports/outofscope_report.txt")


if __name__ == "__main__":
    run()
