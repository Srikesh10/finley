"""
scripts/test_memory.py — Test cross-session semantic memory.

Seeds the memory store with known Q&A pairs, then queries with similar
(but not identical) questions and checks whether the right past exchange
is retrieved above the similarity threshold.

Usage:
  python -m scripts.test_memory
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finley.memory import MemoryStore, _THRESHOLD

# ── Seed data ─────────────────────────────────────────────────────────────────

SEED = [
    {
        "question": "Should I pay off my credit card before contributing to my 401k?",
        "answer":   "With high-interest credit card debt (18%+), pay that off first — the guaranteed 18% return beats typical 401k growth. Once clear, capture any employer match immediately since that's a 50-100% instant return.",
    },
    {
        "question": "Do I have enough saved for emergencies?",
        "answer":   "Your current checking balance covers about 1.5 months of expenses. The standard target is 3-6 months. You're roughly $8,000 short of the minimum 3-month buffer.",
    },
    {
        "question": "How should I tackle my debt?",
        "answer":   "You have two main debts. Avalanche method: hit the highest-interest debt first to minimise total interest paid. Snowball: pay off the smallest balance first for psychological momentum. With your cash flow, the avalanche saves you more.",
    },
    {
        "question": "Can I afford to take a vacation next month?",
        "answer":   "Your safe-to-spend is currently negative. A vacation next month would require cutting $600-800 from discretionary spending first or waiting until after next paycheck clears.",
    },
    {
        "question": "What is my biggest spending problem?",
        "answer":   "Subscriptions and food & drink together account for 38% of your total spend. Your subscription stack has grown 22% year-over-year — several services haven't been used recently.",
    },
]

# ── Test queries (similar but not identical to seeds) ─────────────────────────

TESTS = [
    {
        "query":    "Should I invest in my 401k or clear my credit card debt first?",
        "expected": 0,   # index into SEED
    },
    {
        "query":    "Is my emergency fund big enough?",
        "expected": 1,
    },
    {
        "query":    "What's the best strategy to pay down what I owe?",
        "expected": 2,
    },
    {
        "query":    "I want to go on a trip — can I afford it?",
        "expected": 3,
    },
    {
        "query":    "Where am I overspending the most?",
        "expected": 4,
    },
]

# ── Run ───────────────────────────────────────────────────────────────────────

import shutil

# Use a temp memory dir so tests don't pollute real memory
TEST_DIR = ".finley_memory_test"
import finley.memory as mem_module
_orig_qa  = mem_module._QA_FILE
_orig_emb = mem_module._EMB_FILE
_orig_dir = mem_module._MEMORY_DIR
mem_module._MEMORY_DIR = TEST_DIR
mem_module._QA_FILE    = os.path.join(TEST_DIR, "conversations.json")
mem_module._EMB_FILE   = os.path.join(TEST_DIR, "embeddings.npy")

try:
    store = MemoryStore()

    print("Loading embedding model (first run downloads ~22MB)...")
    for s in SEED:
        store.store(s["question"], s["answer"])
    print(f"Seeded {store.count()} Q&A pairs.\n")

    passed = 0
    for t in TESTS:
        matches = store.search(t["query"])
        top     = matches[0] if matches else None
        expected_q = SEED[t["expected"]]["question"]

        if top and top["question"] == expected_q:
            status = "PASS"
            passed += 1
        elif top:
            status = "WRONG"
        else:
            status = f"NO MATCH (threshold={_THRESHOLD})"

        print(f"[{status}]")
        print(f"  Query   : {t['query']}")
        if top:
            print(f"  Got     : {top['question']}  (score={top['score']:.3f})")
        if status != "PASS":
            print(f"  Expected: {expected_q}")
        print()

    print(f"Results: {passed}/{len(TESTS)} passed")

finally:
    # Restore real paths and clean up test dir
    mem_module._MEMORY_DIR = _orig_dir
    mem_module._QA_FILE    = _orig_qa
    mem_module._EMB_FILE   = _orig_emb
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
