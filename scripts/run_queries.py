"""
scripts/run_queries.py — Run queries through a model and export results to Excel.

Usage:
  python -m scripts.run_queries --model haiku    (default)
  python -m scripts.run_queries --model sonnet
  python -m scripts.run_queries --model gpt4     (requires OPENAI_API_KEY in .env)

Output: reports/queries_results_{model}.xlsx
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time

import openpyxl
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv
load_dotenv()

from finley import process_transactions, SYSTEM_PROMPT, AWS_KEY, AWS_SECRET_KEY, AWS_REGION
from config import TX_FILE, ACCOUNTS_FILE, RECURRING_FILE, SALARY_FILE
from scripts.utils import (
    DARK, STRIPE, WHITE, SUM_CLR, BORDER,
    HDR_FONT, BODY_FONT, NUM_FONT,
    WRAP_TOP, CENTER_TOP, RIGHT_TOP, LEFT_MID,
    row_fill, write_title, write_header_row,
)

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "haiku": {
        "provider":  "anthropic",
        "model_id":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "in_rate":   1.00,   # $ per 1M tokens
        "out_rate":  5.00,
        "label":     "Haiku 4.5",
    },
    "sonnet": {
        "provider":  "anthropic",
        "model_id":  "us.anthropic.claude-sonnet-4-6",
        "in_rate":   3.00,
        "out_rate":  15.00,
        "label":     "Sonnet 4.6",
    },
    "gpt4": {
        "provider":  "azure_openai",
        "model_id":  "gpt-4.1",
        "in_rate":   2.00,
        "out_rate":  8.00,
        "label":     "GPT-4.1",
    },
    "gpt54": {
        "provider":              "azure_openai",
        "model_id":              "gpt-5.4",
        "in_rate":               15.00,   # update from platform.openai.com/pricing
        "out_rate":              60.00,
        "label":                 "GPT-5.4",
        "use_completion_tokens": True,    # GPT-5+ uses max_completion_tokens not max_tokens
    },
}

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODELS.keys()), default="haiku",
                    help="Model to benchmark: " + ", ".join(MODELS.keys()))
args = parser.parse_args()

cfg         = MODELS[args.model]
MODEL_LABEL = cfg["label"]
OUTPUT_FILE = f"reports/queries_results_{args.model}.xlsx"
CHECKPOINT  = f"reports/.run_queries_{args.model}_checkpoint.json"
QUERIES_FILE = "queries_new.txt"
MAX_RETRIES  = 3
RETRY_DELAY  = 5

print(f"Model: {MODEL_LABEL} ({cfg['model_id']})")

# ── Build client ──────────────────────────────────────────────────────────────

if cfg["provider"] == "anthropic":
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(
        aws_access_key=AWS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region=AWS_REGION,
    )
    def call_model(question, summary_json):
        r = client.messages.create(
            model=cfg["model_id"],
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Transaction summary:\n{summary_json}\n\nQuestion: {question}"}],
        )
        return r.content[0].text.strip(), r.usage.input_tokens, r.usage.output_tokens

elif cfg["provider"] in ("openai", "azure_openai"):
    if cfg["provider"] == "azure_openai":
        AZ_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZ_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        if not AZ_KEY or not AZ_ENDPOINT:
            print("ERROR: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set in .env")
            sys.exit(1)
        from openai import AzureOpenAI
        oa_client = AzureOpenAI(
            api_key=AZ_KEY,
            azure_endpoint=AZ_ENDPOINT,
            api_version="2024-12-01-preview",
        )
    else:
        OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
        if not OPENAI_KEY:
            print("ERROR: OPENAI_API_KEY not set in .env — add it and re-run.")
            sys.exit(1)
        from openai import OpenAI
        oa_client = OpenAI(api_key=OPENAI_KEY)

    def call_model(question, summary_json):
        # Newer OpenAI models (GPT-5+) use max_completion_tokens, older use max_tokens
        token_param = "max_completion_tokens" if cfg.get("use_completion_tokens") else "max_tokens"
        r = oa_client.chat.completions.create(
            model=cfg["model_id"],
            **{token_param: 1500},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Transaction summary:\n{summary_json}\n\nQuestion: {question}"},
            ],
        )
        inp = r.usage.prompt_tokens
        out = r.usage.completion_tokens
        return r.choices[0].message.content.strip(), inp, out

# ── Load queries ──────────────────────────────────────────────────────────────

with open(QUERIES_FILE, encoding="utf-8") as f:
    raw = [line.strip() for line in f]
queries = [line for line in raw if line and line.lower() != "question"]
print(f"Loaded {len(queries)} queries from {QUERIES_FILE}")

# ── Load checkpoint ───────────────────────────────────────────────────────────

results = []
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, encoding="utf-8") as f:
        results = json.load(f)
    print(f"Resuming from checkpoint — {len(results)}/{len(queries)} already done")

done_indices = {r["n"] for r in results}

# ── Load transaction data ─────────────────────────────────────────────────────

print("Processing transactions...", end="", flush=True)
t0 = time.time()
summary = process_transactions(
    tx_path        = TX_FILE,
    accounts_path  = ACCOUNTS_FILE  if os.path.exists(ACCOUNTS_FILE)  else None,
    recurring_path = RECURRING_FILE if os.path.exists(RECURRING_FILE) else None,
    salary_path    = SALARY_FILE    if os.path.exists(SALARY_FILE)    else None,
)
summary_json = json.dumps(summary)
print(f" done ({time.time() - t0:.1f}s) — {len(summary_json):,} chars")

# ── Run queries ───────────────────────────────────────────────────────────────

os.makedirs("reports", exist_ok=True)

for i, question in enumerate(queries, start=1):
    if i in done_indices:
        continue

    print(f"[{i:02d}/{len(queries)}] {question[:65]}...", end="", flush=True)

    answer = inp = out = elapsed = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t_start = time.time()
            answer, inp, out = call_model(question, summary_json)
            elapsed = round(time.time() - t_start, 2)
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f" FAILED after {MAX_RETRIES} attempts: {e}")
                answer, inp, out, elapsed = f"ERROR: {e}", 0, 0, 0.0
            else:
                print(f" retry {attempt}...", end="", flush=True)
                time.sleep(RETRY_DELAY * attempt)

    cost = round((inp / 1_000_000 * cfg["in_rate"]) + (out / 1_000_000 * cfg["out_rate"]), 5)
    results.append({
        "n": i, "question": question, "answer": answer,
        "runtime_s": elapsed, "input_tokens": inp, "output_tokens": out, "cost_usd": cost,
    })

    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f" {elapsed}s | {inp}in/{out}out | ${cost:.5f}")

results.sort(key=lambda r: r["n"])

# ── Build Excel ───────────────────────────────────────────────────────────────

HEADERS    = ["#", "Question", "Answer", "Runtime (s)", "Input Tokens", "Output Tokens", "Cost ($)"]
COL_WIDTHS = [5, 42, 72, 13, 14, 15, 11]

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Results"

write_title(ws, f"Finley Query Results — {len(results)} questions — {MODEL_LABEL}", "A1:G1")
write_header_row(ws, HEADERS, left_cols={2})

for ri, r in enumerate(results, start=3):
    fill = row_fill(ri)
    vals = [r["n"], r["question"], r["answer"], r["runtime_s"],
            r["input_tokens"], r["output_tokens"], round(r["cost_usd"], 5)]
    for ci, val in enumerate(vals, 1):
        c = ws.cell(row=ri, column=ci, value=val)
        c.fill   = fill
        c.border = BORDER
        if ci in (1, 4, 5, 6, 7):
            c.font      = NUM_FONT
            c.alignment = RIGHT_TOP if ci == 7 else CENTER_TOP
        else:
            c.font      = BODY_FONT
            c.alignment = WRAP_TOP
    ws.row_dimensions[ri].height = 90

for ci, w in enumerate(COL_WIDTHS, 1):
    ws.column_dimensions[get_column_letter(ci)].width = w

sr = len(results) + 3
total_cost  = round(sum(r["cost_usd"] for r in results), 4)
avg_runtime = round(sum(r["runtime_s"] for r in results) / len(results), 2)
total_in    = sum(r["input_tokens"] for r in results)
total_out   = sum(r["output_tokens"] for r in results)

ws.cell(row=sr, column=1, value="TOTAL").font = NUM_FONT
ws.cell(row=sr, column=1).alignment = CENTER_TOP
ws.cell(row=sr, column=1).fill = SUM_CLR

for ci, val in zip([4, 5, 6, 7], [f"{avg_runtime}s avg", total_in, total_out, total_cost]):
    c = ws.cell(row=sr, column=ci, value=val)
    c.font      = NUM_FONT
    c.alignment = RIGHT_TOP if ci == 7 else CENTER_TOP
    c.fill      = SUM_CLR
    c.border    = BORDER

ws.freeze_panes = "A3"
wb.save(OUTPUT_FILE)

if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

print(f"\nDone.")
print(f"  {len(results)} questions answered")
print(f"  Avg runtime : {avg_runtime}s")
print(f"  Total tokens: {total_in:,} in / {total_out:,} out")
print(f"  Total cost  : ${total_cost:.4f}")
print(f"  Saved to    : {OUTPUT_FILE}")
