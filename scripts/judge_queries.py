"""
scripts/judge_queries.py — Add Opus accuracy scores to queries_results.xlsx.

Reads each question + answer, sends to Opus with the transaction summary,
appends three columns: Verdict, Accuracy (0-10), Judge Notes.
Supports checkpointing: interrupted runs resume from where they stopped.
Output: reports/queries_results_judged.xlsx
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from anthropic import AnthropicBedrock
from dotenv import load_dotenv
load_dotenv()

from finley import AWS_KEY, AWS_SECRET_KEY, AWS_REGION, process_transactions
from config import TX_FILE, ACCOUNTS_FILE, RECURRING_FILE, SALARY_FILE
from scripts.utils import (
    DARK, STRIPE, WHITE, PASS_CLR, FAIL_CLR, SUM_CLR, BORDER,
    HDR_FONT, BODY_FONT, NUM_FONT, WRAP_TOP, CENTER_TOP, RIGHT_TOP,
    row_fill,
)

OPUS_MODEL   = "us.anthropic.claude-opus-4-6-v1"
INPUT_FILE   = "reports/queries_results.xlsx"
OUTPUT_FILE  = "reports/queries_results_judged.xlsx"
CHECKPOINT   = "reports/.judge_queries_checkpoint.json"
MAX_RETRIES  = 3
RETRY_DELAY  = 5

JUDGE_SYSTEM = """You are a financial data accuracy auditor. You will be given:
1. A question a user asked about their finances
2. The structured transaction summary the AI used to answer it
3. The AI's answer

Evaluate whether the answer is factually correct and complete based on the data.

Output ONLY raw valid JSON, no markdown, no code fences:
{
  "verdict": "PASS",
  "score": 0-10,
  "notes": "one sentence summary of quality",
  "errors": ["specific error 1", "specific error 2"]
}

CRITICAL: verdict MUST be exactly "PASS" or "FAIL". Never "PARTIAL", never "UNCLEAR", never anything else.
If the answer is mostly correct with minor issues, verdict = "PASS" with a lower score.
If the answer has significant factual errors or is misleading, verdict = "FAIL".

Scoring guide:
9-10: Accurate, complete, well-structured
7-8:  Accurate with minor gaps or one small error
5-6:  Partially correct but missing key info or has a notable error
3-4:  Significant errors or major gaps
0-2:  Wrong or completely unhelpful

For out-of-scope questions (HSA explanation, platform how-tos, talking to a human),
score based on whether the deflection is appropriate and helpful — not on data accuracy.
PASS if it handles gracefully, FAIL if it fabricates or gives bad advice."""


def parse_judgment(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        return json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    # Regex fallback — handles truncated or slightly malformed JSON
    verdict_m = re.search(r'"verdict"\s*:\s*"(PASS|FAIL)"', raw, re.IGNORECASE)
    score_m   = re.search(r'"score"\s*:\s*(\d+)', raw)
    notes_m   = re.search(r'"notes"\s*:\s*"([^"]+)"', raw)
    if verdict_m and score_m:
        return {
            "verdict": verdict_m.group(1).upper(),
            "score":   int(score_m.group(1)),
            "notes":   (notes_m.group(1) if notes_m else "") + " [parsed via regex fallback]",
            "errors":  [],
        }
    return {"verdict": "ERROR", "score": 0, "notes": "Could not parse judge response", "errors": []}


# ── Load data ─────────────────────────────────────────────────────────────────

print("Processing transactions...", end="", flush=True)
summary = process_transactions(
    tx_path        = TX_FILE,
    accounts_path  = ACCOUNTS_FILE  if os.path.exists(ACCOUNTS_FILE)  else None,
    recurring_path = RECURRING_FILE if os.path.exists(RECURRING_FILE) else None,
    salary_path    = SALARY_FILE    if os.path.exists(SALARY_FILE)    else None,
)
summary_json = json.dumps(summary)
print(f" done — {len(summary_json):,} chars")

wb_in = openpyxl.load_workbook(INPUT_FILE)
ws_in = wb_in.active

rows = []
for row in ws_in.iter_rows(min_row=3, max_row=ws_in.max_row - 1, values_only=True):
    if row[0] and isinstance(row[0], int):
        rows.append({"n": row[0], "question": row[1], "answer": row[2],
                     "runtime": row[3], "input_tokens": row[4],
                     "output_tokens": row[5], "cost": row[6]})

print(f"Loaded {len(rows)} questions from {INPUT_FILE}")

client = AnthropicBedrock(
    aws_access_key=AWS_KEY,
    aws_secret_key=AWS_SECRET_KEY,
    aws_region=AWS_REGION,
)

# ── Load checkpoint ──────────────────────────────────────────────────────────

judgments = []
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, encoding="utf-8") as f:
        judgments = json.load(f)
    print(f"Resuming from checkpoint — {len(judgments)}/{len(rows)} already judged")

done_indices = {j["_n"] for j in judgments}
total_opus_cost = sum(j.get("judge_cost", 0) for j in judgments)

# ── Judge each answer ─────────────────────────────────────────────────────────

for r in rows:
    if r["n"] in done_indices:
        continue

    print(f"[{r['n']:02d}/{len(rows)}] {r['question'][:65]}...", end="", flush=True)

    prompt = f"""QUESTION: {r['question']}

TRANSACTION SUMMARY:
{summary_json}

AI ANSWER:
{r['answer']}

Evaluate the answer's accuracy against the transaction summary."""

    inp = out = elapsed = 0
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0       = time.time()
            response = client.messages.create(
                model=OPUS_MODEL,
                max_tokens=400,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = round(time.time() - t0, 1)
            inp     = response.usage.input_tokens
            out     = response.usage.output_tokens
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f" FAILED: {e}")
            else:
                print(f" retry {attempt}...", end="", flush=True)
                time.sleep(RETRY_DELAY * attempt)
    cost = round((inp / 1_000_000 * 15.0) + (out / 1_000_000 * 75.0), 4)
    total_opus_cost += cost

    raw_text = response.content[0].text if response else '{"verdict":"ERROR","score":0,"notes":"No response","errors":[]}'
    j = parse_judgment(raw_text)
    j["judge_cost"] = cost
    j["_n"] = r["n"]
    judgments.append(j)

    # Checkpoint after every judgment
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(judgments, f)

    print(f" {j['verdict']} {j['score']}/10 | {elapsed}s | ${cost:.4f}")
    time.sleep(0.5)

judgments.sort(key=lambda j: j["_n"])

# ── Build output Excel ────────────────────────────────────────────────────────

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Results + Accuracy"

HEADERS = ["#", "Question", "Answer", "Runtime (s)", "Input Tokens",
           "Output Tokens", "Cost ($)", "Verdict", "Accuracy (0-10)", "Judge Notes"]
WIDTHS  = [5, 40, 65, 13, 14, 15, 10, 10, 16, 55]

passes = sum(1 for j in judgments if j.get("verdict") == "PASS")
avg_score = round(sum(j.get("score", 0) for j in judgments) / len(judgments), 1)
total_cost = round(sum(r["cost"] or 0 for r in rows) + total_opus_cost, 4)

# Title
ws.merge_cells("A1:J1")
ws["A1"].value = (f"Finley Query Results — {len(rows)} questions — "
                  f"Haiku generates / Opus judges | "
                  f"{passes}/{len(rows)} PASS | avg {avg_score}/10 | total ${total_cost:.4f}")
ws["A1"].font      = Font(name="Calibri", bold=True, size=12, color="111827")
ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
ws.row_dimensions[1].height = 26

# Headers
for ci, h in enumerate(HEADERS, 1):
    c = ws.cell(row=2, column=ci, value=h)
    c.fill      = DARK
    c.font      = HDR_FONT
    c.border    = BORDER
    c.alignment = CENTER_TOP if ci not in (2, 3, 10) else Alignment(horizontal="left", vertical="center")
ws.row_dimensions[2].height = 20

# Data
for ri, (r, j) in enumerate(zip(rows, judgments), start=3):
    verdict = j.get("verdict", "")
    score   = j.get("score", 0)
    notes   = j.get("notes", "")
    errors  = j.get("errors", [])
    note_full = notes
    if errors:
        note_full += " | Errors: " + "; ".join(errors)

    base_fill = row_fill(ri)

    vals = [r["n"], r["question"], r["answer"], r["runtime"],
            r["input_tokens"], r["output_tokens"], r["cost"],
            verdict, score, note_full]

    for ci, val in enumerate(vals, 1):
        c = ws.cell(row=ri, column=ci, value=val)
        c.border = BORDER

        if ci == 8:  # Verdict
            c.fill = PASS_CLR if verdict == "PASS" else (FAIL_CLR if verdict == "FAIL" else base_fill)
            c.font = Font(name="Calibri", bold=True, size=10,
                          color="065F46" if verdict == "PASS" else "991B1B")
            c.alignment = CENTER_TOP
        elif ci == 9:  # Score
            color = "065F46" if score >= 8 else ("92400E" if score >= 6 else "991B1B")
            c.fill      = base_fill
            c.font      = Font(name="Calibri", bold=True, size=10, color=color)
            c.alignment = CENTER_TOP
        elif ci in (1, 4, 5, 6, 7):
            c.fill      = base_fill
            c.font      = NUM_FONT
            c.alignment = RIGHT_TOP if ci == 7 else CENTER_TOP
        else:
            c.fill      = base_fill
            c.font      = BODY_FONT
            c.alignment = WRAP_TOP

    ws.row_dimensions[ri].height = 90

for ci, w in enumerate(WIDTHS, 1):
    ws.column_dimensions[get_column_letter(ci)].width = w

# Summary row
sr = len(rows) + 3
ws.cell(row=sr, column=1, value="TOTAL").font = NUM_FONT
ws.cell(row=sr, column=1).alignment = CENTER_TOP
ws.cell(row=sr, column=1).fill = SUM_CLR

for ci, val in [(4, f"{round(sum(r['runtime'] or 0 for r in rows)/len(rows),1)}s avg"),
                (5, sum(r['input_tokens'] or 0 for r in rows)),
                (6, sum(r['output_tokens'] or 0 for r in rows)),
                (7, round(sum(r['cost'] or 0 for r in rows), 4)),
                (8, f"{passes}/{len(rows)} PASS"),
                (9, avg_score),
                (10, f"Opus judge cost: ${total_opus_cost:.4f}")]:
    c = ws.cell(row=sr, column=ci, value=val)
    c.font      = NUM_FONT
    c.alignment = RIGHT_TOP if ci == 7 else CENTER_TOP
    c.fill      = SUM_CLR
    c.border    = BORDER

ws.freeze_panes = "A3"
wb.save(OUTPUT_FILE)

# Clean up checkpoint on successful completion
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

print(f"\nDone.")
print(f"  {passes}/{len(rows)} PASS | avg {avg_score}/10")
print(f"  Opus judge cost : ${total_opus_cost:.4f}")
print(f"  Total cost      : ${total_cost:.4f}")
print(f"  Saved to        : {OUTPUT_FILE}")
