"""
scripts/judge_batch.py — Judge all answers in a single Opus call.

Sends all Q+A pairs tagged by number to Opus once, gets back a JSON array
of 51 scores. 40x cheaper than the per-question judge.

Usage:
  python -m scripts.judge_batch --input reports/queries_results_hybrid.xlsx
  python -m scripts.judge_batch --input reports/queries_results_routed.xlsx

Output: replaces _results_ with _judged_ in the input filename.
"""

import sys, os, argparse, json, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="reports/queries_results_hybrid.xlsx")
args = parser.parse_args()

INPUT_FILE  = args.input
OUTPUT_FILE = INPUT_FILE.replace("queries_results", "queries_judged")
OPUS_MODEL  = "us.anthropic.claude-opus-4-6-v1"
MAX_RETRIES = 3

JUDGE_SYSTEM = """You are a financial data accuracy auditor. You will be given:
1. A transaction summary containing the user's real financial data
2. A numbered list of questions and the AI answers given to each

Evaluate every answer for factual correctness and completeness against the data.

Output ONLY a raw JSON array — no markdown, no code fences, no other text.
The array must have exactly one object per question, in order, with this shape:
[
  {
    "n": 1,
    "verdict": "PASS",
    "score": 0-10,
    "notes": "one sentence summary",
    "errors": ["specific error if any"]
  },
  ...
]

CRITICAL rules:
- verdict MUST be exactly "PASS" or "FAIL". Nothing else.
- If mostly correct with minor issues: PASS with lower score.
- If significant factual errors or misleading: FAIL.
- For out-of-scope questions (HSA explanation, platform how-tos, talking to a human):
  PASS if handled gracefully, FAIL if fabricated or bad advice.

Scoring guide:
9-10: Accurate, complete, well-structured
7-8:  Accurate with minor gaps or one small error
5-6:  Partially correct but missing key info or has a notable error
3-4:  Significant errors or major gaps
0-2:  Wrong or completely unhelpful"""


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

header_row = [str(c or "").strip().lower()
              for c in next(ws_in.iter_rows(min_row=2, max_row=2, values_only=True))]

def _col(name, fallback):
    for i, h in enumerate(header_row):
        if name in h:
            return i
    return fallback

ci_n    = _col("#",        0)
ci_q    = _col("question", 1)
ci_a    = _col("answer",   2)
ci_rt   = _col("runtime",  3)
ci_in   = _col("input",    4)
ci_out  = _col("output",   5)
ci_cost = _col("cost",     6)

rows = []
for row in ws_in.iter_rows(min_row=3, max_row=ws_in.max_row - 1, values_only=True):
    if row[ci_n] and isinstance(row[ci_n], int):
        rows.append({
            "n":            row[ci_n],
            "question":     row[ci_q],
            "answer":       row[ci_a],
            "runtime":      row[ci_rt],
            "input_tokens": row[ci_in],
            "output_tokens":row[ci_out],
            "cost":         row[ci_cost],
        })

print(f"Loaded {len(rows)} questions from {INPUT_FILE}")

# ── Build batch prompt ────────────────────────────────────────────────────────

qa_block = ""
for r in rows:
    qa_block += f"[{r['n']}] QUESTION: {r['question']}\nANSWER: {r['answer']}\n\n"

prompt = (
    f"TRANSACTION SUMMARY:\n{summary_json}\n\n"
    f"Evaluate the following {len(rows)} question-answer pairs:\n\n"
    f"{qa_block}"
    f"Return a JSON array of exactly {len(rows)} judgment objects, one per question in order."
)

print(f"Prompt size: {len(prompt):,} chars | Sending 1 Opus call for all {len(rows)} judgments...")

# ── Single Opus call ──────────────────────────────────────────────────────────

client = AnthropicBedrock(
    aws_access_key=AWS_KEY,
    aws_secret_key=AWS_SECRET_KEY,
    aws_region=AWS_REGION,
)

raw_text = None
inp = out = 0
for attempt in range(1, MAX_RETRIES + 1):
    try:
        t0       = time.time()
        response = client.messages.create(
            model=OPUS_MODEL,
            max_tokens=8000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = round(time.time() - t0, 1)
        inp     = response.usage.input_tokens
        out     = response.usage.output_tokens
        raw_text = response.content[0].text.strip()
        print(f"Response received in {elapsed}s | {inp:,} in / {out:,} out")
        break
    except Exception as e:
        if attempt == MAX_RETRIES:
            print(f"FAILED after {MAX_RETRIES} attempts: {e}")
            sys.exit(1)
        print(f"Retry {attempt}...")
        time.sleep(5 * attempt)

opus_cost = round((inp / 1_000_000 * 15.0) + (out / 1_000_000 * 75.0), 4)
print(f"Opus judge cost: ${opus_cost:.4f}")

# ── Parse response ────────────────────────────────────────────────────────────

def parse_batch(text: str, n_expected: int) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        start = cleaned.index("[")
        end   = cleaned.rindex("]") + 1
        items = json.loads(cleaned[start:end])
        if isinstance(items, list) and len(items) == n_expected:
            return items
    except (ValueError, json.JSONDecodeError):
        pass

    # Fallback: extract individual objects by n tag
    items = []
    for m in re.finditer(r'\{[^{}]+\}', cleaned):
        try:
            obj = json.loads(m.group())
            if "verdict" in obj and "score" in obj:
                items.append(obj)
        except json.JSONDecodeError:
            pass
    if items:
        print(f"  Parsed {len(items)}/{n_expected} via fallback object extraction")
        return items

    print("  WARNING: Could not parse batch response — falling back to ERROR entries")
    return [{"n": i+1, "verdict": "ERROR", "score": 0,
             "notes": "Parse failed", "errors": []} for i in range(n_expected)]

judgments = parse_batch(raw_text, len(rows))

# Align by n if present, else by position
if all("n" in j for j in judgments):
    j_by_n = {j["n"]: j for j in judgments}
    judgments = [j_by_n.get(r["n"], {"n": r["n"], "verdict": "ERROR", "score": 0,
                                      "notes": "Missing from response", "errors": []})
                 for r in rows]

passes    = sum(1 for j in judgments if j.get("verdict") == "PASS")
avg_score = round(sum(j.get("score", 0) for j in judgments) / len(judgments), 1)
total_cost = round(sum(r["cost"] or 0 for r in rows) + opus_cost, 4)
model_label = os.path.basename(INPUT_FILE).replace("queries_results_", "").replace(".xlsx", "").upper()

print(f"\nResults: {passes}/{len(rows)} PASS | avg {avg_score}/10")
for j in judgments:
    status = j.get("verdict", "?")
    score  = j.get("score", 0)
    notes  = j.get("notes", "")[:60]
    print(f"  [{j.get('n',0):02d}] {status:<4} {score}/10  {notes}")

# ── Build output Excel ────────────────────────────────────────────────────────

HEADERS = ["#", "Question", "Answer", "Runtime (s)", "Input Tokens",
           "Output Tokens", "Cost ($)", "Verdict", "Accuracy (0-10)", "Judge Notes"]
WIDTHS  = [5, 40, 65, 13, 14, 15, 10, 10, 16, 55]

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Results + Accuracy"

ws.merge_cells("A1:J1")
ws["A1"].value = (f"Finley Query Results — {len(rows)} questions — "
                  f"{model_label} generates / Opus batch-judges | "
                  f"{passes}/{len(rows)} PASS | avg {avg_score}/10 | total ${total_cost:.4f}")
ws["A1"].font      = Font(name="Calibri", bold=True, size=12, color="111827")
ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
ws.row_dimensions[1].height = 26

for ci, h in enumerate(HEADERS, 1):
    c = ws.cell(row=2, column=ci, value=h)
    c.fill      = DARK
    c.font      = HDR_FONT
    c.border    = BORDER
    c.alignment = CENTER_TOP if ci not in (2, 3, 10) else Alignment(horizontal="left", vertical="center")
ws.row_dimensions[2].height = 20

for ri, (r, j) in enumerate(zip(rows, judgments), start=3):
    verdict   = j.get("verdict", "")
    score     = j.get("score", 0)
    notes     = j.get("notes", "")
    errors    = j.get("errors", [])
    note_full = notes + (" | Errors: " + "; ".join(errors) if errors else "")
    base_fill = row_fill(ri)

    vals = [r["n"], r["question"], r["answer"], r["runtime"],
            r["input_tokens"], r["output_tokens"], r["cost"],
            verdict, score, note_full]

    for ci, val in enumerate(vals, 1):
        c = ws.cell(row=ri, column=ci, value=val)
        c.border = BORDER
        if ci == 8:
            c.fill = PASS_CLR if verdict == "PASS" else (FAIL_CLR if verdict == "FAIL" else base_fill)
            c.font = Font(name="Calibri", bold=True, size=10,
                          color="065F46" if verdict == "PASS" else "991B1B")
            c.alignment = CENTER_TOP
        elif ci == 9:
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
                (10, f"Opus batch judge cost: ${opus_cost:.4f}")]:
    c = ws.cell(row=sr, column=ci, value=val)
    c.font      = NUM_FONT
    c.alignment = RIGHT_TOP if ci == 7 else CENTER_TOP
    c.fill      = SUM_CLR
    c.border    = BORDER

ws.freeze_panes = "A3"
wb.save(OUTPUT_FILE)

print(f"\nDone.")
print(f"  {passes}/{len(rows)} PASS | avg {avg_score}/10")
print(f"  Opus batch judge cost : ${opus_cost:.4f}  (was ~$24.65 per-question)")
print(f"  Total cost            : ${total_cost:.4f}")
print(f"  Saved to              : {OUTPUT_FILE}")
