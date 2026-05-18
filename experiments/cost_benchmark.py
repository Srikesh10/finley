"""
cost_benchmark.py — Cost + quality comparison: initial code approach vs current.

Initial: raw CSV + 6-line prompt + Sonnet pricing
Current: JSON summary + full system prompt + Haiku

Opus judges both answers for each question. Side-by-side report.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import time
from anthropic import AnthropicBedrock
from finley import process_transactions, AWS_KEY, AWS_SECRET_KEY, AWS_REGION, SYSTEM_PROMPT

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS_MODEL  = "us.anthropic.claude-opus-4-6-v1"
from config import TX_FILE, ACCOUNTS_FILE, RECURRING_FILE, SALARY_FILE

SONNET_IN  = 3.00   # Sonnet pricing — what initial code used
SONNET_OUT = 15.00
HAIKU_IN   = 1.00
HAIKU_OUT  = 5.00
OPUS_IN    = 15.00  # Opus pricing — used as judge in current pipeline
OPUS_OUT   = 75.00

TEST_QUESTIONS = [
    "Show me my monthly financial review",
    "Show me my recurring subscriptions",
    "What is my savings rate based on my income and spending?",
    "Where did I spend my money on restaurants and food?",
    "Can I afford to invest $500 a month?",
]

INITIAL_SYSTEM = f"""You are a financial data analyst. When provided with CSV data containing financial transactions and a question, analyze the data and answer the question.
1. Analyze the transaction data to understand the financial picture.
2. If the question does not specify a time period, analyze the last 3 months.
3. Today's date is {time.strftime('%Y-%m-%d')}.
4. Print a clear, concise final answer."""

CURRENT_SYSTEM_ADDON = """
---
QUERY MODE: Answer the user's question directly using the data summary. Use exact dollar amounts and counts from the data."""

JUDGE_SYSTEM = """You are a financial data accuracy auditor. You will be given a question, the data used to answer it, and an AI-generated answer.

Evaluate the answer. Output ONLY raw valid JSON, no markdown, no code fences:

{
  "factual_accuracy": 0-10,
  "completeness": 0-10,
  "clarity": 0-10,
  "overall": 0-10,
  "errors": ["list any factual errors or fabricated numbers"],
  "verdict": "PASS or FAIL",
  "notes": "one sentence summary"
}

PASS = overall >= 7 and no critical factual errors
FAIL = overall < 7 or contains wrong numbers"""


def cost_sonnet(inp, out):
    return round((inp / 1_000_000) * SONNET_IN + (out / 1_000_000) * SONNET_OUT, 6)


def cost_haiku(inp, out):
    return round((inp / 1_000_000) * HAIKU_IN + (out / 1_000_000) * HAIKU_OUT, 6)


def run_initial(client, csv_text: str, question: str) -> dict:
    prompt = f"Transaction data:\n\n{csv_text}\n\nQuestion: {question}"
    t0 = time.time()
    try:
        r = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2000,
            system=INITIAL_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            timeout=120,
        )
        inp, out = r.usage.input_tokens, r.usage.output_tokens
        return {
            "answer": r.content[0].text,
            "input_tokens": inp,
            "output_tokens": out,
            "cost_usd": cost_sonnet(inp, out),
            "time_s": round(time.time() - t0, 1),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "time_s": 0}


def run_current(client, summary: dict, question: str) -> dict:
    prompt = f"TRANSACTION DATA SUMMARY:\n{json.dumps(summary, indent=2)}\n\nQUESTION: {question}"
    system = SYSTEM_PROMPT + CURRENT_SYSTEM_ADDON
    t0 = time.time()
    try:
        r = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        inp, out = r.usage.input_tokens, r.usage.output_tokens
        return {
            "answer": r.content[0].text,
            "input_tokens": inp,
            "output_tokens": out,
            "cost_usd": cost_haiku(inp, out),
            "time_s": round(time.time() - t0, 1),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "time_s": 0}


def judge(client, question: str, answer: str, data_context: str) -> dict:
    prompt = f"QUESTION: {question}\n\nDATA USED:\n{data_context}\n\nANSWER TO JUDGE:\n{answer}"
    t0 = time.time()
    try:
        r = client.messages.create(
            model=OPUS_MODEL,
            max_tokens=1500,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            timeout=120,
        )
        raw = r.content[0].text
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()
        start = cleaned.index('{')
        end = cleaned.rindex('}') + 1
        result = json.loads(cleaned[start:end])
        result["_time_s"] = round(time.time() - t0, 1)
        result["_input_tokens"]  = r.usage.input_tokens
        result["_output_tokens"] = r.usage.output_tokens
        result["_cost_usd"] = round(
            (r.usage.input_tokens / 1_000_000) * OPUS_IN +
            (r.usage.output_tokens / 1_000_000) * OPUS_OUT, 6
        )
        return result
    except Exception as e:
        return {"error": str(e), "overall": 0, "verdict": "ERROR", "_time_s": round(time.time() - t0, 1),
                "_input_tokens": 0, "_output_tokens": 0, "_cost_usd": 0}


def run():
    client = AnthropicBedrock(
        aws_access_key=AWS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region=AWS_REGION,
    )

    print("Loading data...")
    csv_text = load_csv_as_text(TX_FILE)
    summary  = process_transactions(
        tx_path        = TX_FILE,
        accounts_path  = ACCOUNTS_FILE,
        recurring_path = RECURRING_FILE,
        salary_path    = SALARY_FILE,
    )
    json_ctx = json.dumps(summary, indent=2)

    csv_chars  = len(csv_text)
    json_chars = len(json_ctx)
    print(f"  CSV: {csv_chars:,} chars | JSON summary: {json_chars:,} chars ({round(csv_chars/json_chars, 1)}x smaller)\n")

    results = []

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {q}")

        print(f"  Initial (raw CSV)...", end="", flush=True)
        ini = run_initial(client, csv_text, q)
        print(f" {ini['time_s']}s | {ini['input_tokens']:,} in / {ini['output_tokens']:,} out | ${ini['cost_usd']:.4f}")
        time.sleep(1)

        print(f"  Current (JSON)  ...", end="", flush=True)
        cur = run_current(client, summary, q)
        print(f" {cur['time_s']}s | {cur['input_tokens']:,} in / {cur['output_tokens']:,} out | ${cur['cost_usd']:.4f}")
        time.sleep(1)

        print(f"  Judging initial...", end="", flush=True)
        ini_j = judge(client, q, ini["answer"], csv_text[:3000])
        print(f" {ini_j.get('_time_s')}s | {ini_j.get('verdict', '?')} ({ini_j.get('overall', '?')}/10)")
        time.sleep(1)

        print(f"  Judging current...", end="", flush=True)
        cur_j = judge(client, q, cur["answer"], json_ctx)
        print(f" {cur_j.get('_time_s')}s | {cur_j.get('verdict', '?')} ({cur_j.get('overall', '?')}/10)")
        time.sleep(2)

        results.append({"question": q, "initial": ini, "current": cur,
                         "initial_judgment": ini_j, "current_judgment": cur_j})
        print()

    # Save raw
    with open("reports/cost_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("COST + QUALITY BENCHMARK — INITIAL CODE vs CURRENT CODE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("METHODOLOGY:")
    lines.append("  Initial : Raw CSV (430K chars) + 4-line prompt → Haiku (token count)")
    lines.append("            Cost calculated at Sonnet 4.6 rates ($3/$15 per 1M)")
    lines.append("            because that is the model the initial code used.")
    lines.append("  Current : JSON summary (6K chars) + 500-line system prompt → Haiku (generate)")
    lines.append("            + Opus 4.6 judges answer before it reaches the user (purify step)")
    lines.append("            Haiku at $1/$5 per 1M | Opus at $15/$75 per 1M")
    lines.append("  This is the full production pipeline cost — not just generation.")
    lines.append("")

    # Aggregate stats
    ini_scores = [r["initial_judgment"].get("overall", 0) for r in results]
    cur_scores = [r["current_judgment"].get("overall", 0) for r in results]
    ini_costs  = [r["initial"]["cost_usd"] for r in results]
    cur_gen_costs   = [r["current"]["cost_usd"] for r in results]
    cur_judge_costs = [r["current_judgment"].get("_cost_usd", 0) for r in results]
    cur_costs = [g + j for g, j in zip(cur_gen_costs, cur_judge_costs)]
    ini_passes = sum(1 for r in results if r["initial_judgment"].get("verdict") == "PASS")
    cur_passes = sum(1 for r in results if r["current_judgment"].get("verdict") == "PASS")

    avg_ini_score  = round(sum(ini_scores) / max(len(ini_scores), 1), 2)
    avg_cur_score  = round(sum(cur_scores) / max(len(cur_scores), 1), 2)
    total_ini_cost = sum(ini_costs)
    total_cur_gen  = sum(cur_gen_costs)
    total_cur_judge= sum(cur_judge_costs)
    total_cur_cost = sum(cur_costs)
    cost_ratio = round(total_ini_cost / max(total_cur_cost, 0.000001), 1)
    score_delta = round(avg_cur_score - avg_ini_score, 2)

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  {'METRIC':<35} {'INITIAL':>10} {'CURRENT':>10} {'DELTA':>10}")
    lines.append(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    lines.append(f"  {'Avg quality score (0-10)':<35} {avg_ini_score:>10} {avg_cur_score:>10} {score_delta:>+10}")
    lines.append(f"  {'Pass rate':<35} {f'{ini_passes}/{len(results)}':>10} {f'{cur_passes}/{len(results)}':>10} {f'{cur_passes-ini_passes:+d}':>10}")
    lines.append(f"  {'Haiku generation cost':<35} {'N/A':>10} ${total_cur_gen:>9.4f}")
    lines.append(f"  {'Opus judge cost':<35} {'N/A':>10} ${total_cur_judge:>9.4f}")
    lines.append(f"  {'Total cost ({} queries)'.format(len(results)):<35} ${total_ini_cost:>9.4f} ${total_cur_cost:>9.4f} {f'{cost_ratio:.1f}x cheaper':>10}")
    lines.append(f"  {'Avg input tokens/query':<35} {round(sum(r['initial']['input_tokens'] for r in results)/len(results)):>10,} {round(sum(r['current']['input_tokens'] for r in results)/len(results)):>10,}")
    lines.append("")

    quality_cost = round(score_delta / max(total_ini_cost - total_cur_cost, 0.0001), 2)
    lines.append(f"  QUALITY vs COST TRADEOFF:")
    lines.append(f"  Score change  : {score_delta:+.2f} points")
    lines.append(f"  Cost change   : {cost_ratio:.1f}x cheaper")
    if score_delta >= 0:
        lines.append(f"  Verdict       : Current code is BOTH cheaper AND higher quality.")
    else:
        lines.append(f"  Verdict       : Current code is {cost_ratio:.1f}x cheaper at a cost of {abs(score_delta):.2f} quality points.")
        lines.append(f"  Cost per quality point sacrificed: ${abs(1/(quality_cost)) if quality_cost != 0 else 'N/A':.4f} saved per 0.1 point drop")
    lines.append("")

    # Per-question breakdown
    lines.append("=" * 80)
    lines.append("PER-QUESTION BREAKDOWN")
    lines.append("=" * 80)

    for i, r in enumerate(results, 1):
        ij = r["initial_judgment"]
        cj = r["current_judgment"]
        ini_s = ij.get("overall", "?")
        cur_s = cj.get("overall", "?")
        delta = (cur_s - ini_s) if isinstance(cur_s, (int, float)) and isinstance(ini_s, (int, float)) else "?"
        cur_full_cost = r["current"]["cost_usd"] + r["current_judgment"].get("_cost_usd", 0)
        cost_x = round(r["initial"]["cost_usd"] / max(cur_full_cost, 0.000001), 1)

        lines.append("")
        lines.append(f"Q{i}: {r['question']}")
        lines.append(f"  Quality : {ini_s}/10 initial → {cur_s}/10 current  ({delta:+} points)" if isinstance(delta, (int, float)) else f"  Quality : {ini_s}/10 → {cur_s}/10")
        lines.append(f"  Cost    : ${r['initial']['cost_usd']:.4f} initial → ${cur_full_cost:.4f} current (Haiku ${r['current']['cost_usd']:.4f} + Opus ${r['current_judgment'].get('_cost_usd',0):.4f})  ({cost_x}x cheaper)")
        lines.append(f"  Tokens  : {r['initial']['input_tokens']:,} in initial → {r['current']['input_tokens']:,} in current")
        lines.append(f"  Verdict : {ij.get('verdict','?')} initial | {cj.get('verdict','?')} current")
        lines.append(f"  Initial notes : {ij.get('notes', '')}")
        lines.append(f"  Current notes : {cj.get('notes', '')}")

        if ij.get("errors"):
            lines.append(f"  Initial errors:")
            for e in ij["errors"]:
                lines.append(f"    - {e}")
        if cj.get("errors"):
            lines.append(f"  Current errors:")
            for e in cj["errors"]:
                lines.append(f"    - {e}")

        lines.append("")
        lines.append("  INITIAL ANSWER:")
        lines.append("  " + "\n  ".join(r["initial"]["answer"][:800].split("\n")))
        lines.append("")
        lines.append("  CURRENT ANSWER:")
        lines.append("  " + "\n  ".join(r["current"]["answer"][:800].split("\n")))
        lines.append("")
        lines.append("-" * 80)

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF BENCHMARK")

    report = "\n".join(lines)
    with open("reports/cost_benchmark.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Avg score  : {avg_ini_score}/10 initial -> {avg_cur_score}/10 current  ({score_delta:+.2f})")
    print(f"  Pass rate  : {ini_passes}/{len(results)} initial -> {cur_passes}/{len(results)} current")
    print(f"  Total cost : ${total_ini_cost:.4f} initial -> ${total_cur_cost:.4f} current  ({cost_ratio}x cheaper)")
    print(f"\nFull report saved to reports/cost_benchmark.txt")


def load_csv_as_text(csv_file: str) -> str:
    with open(csv_file, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    run()
