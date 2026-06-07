"""
scripts/test_features.py — Live feature tests for question recommendations
and cross-session semantic memory (vector search).

Runs real API calls and real embedding search. Results are printed and
match what the user sees in the app.

Usage:
  python -m scripts.test_features
"""

import sys, os, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

passed = 0
failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {label}")
        if detail:
            print(f"         {detail}")
        passed += 1
    else:
        print(f"  [FAIL] {label}")
        if detail:
            print(f"         {detail}")
        failed += 1

# -- Test 1: Question Recommendations ----------------------------------------─

print("\n-- TEST 1: Question Recommendations ------------------------------------")
print("   Calls Haiku with a sample Q&A and checks 3 suggestions are returned.\n")

from finley.client import make_client, HAIKU_MODEL

client = make_client()

sample_question = "How much did I spend on food last month?"
sample_answer   = ("You spent $842 on food and drink last month across 34 transactions. "
                   "Your top merchants were Zomato ($210), Swiggy ($175), and Starbucks ($98). "
                   "This is 18% above your 6-month average of $714.")

prompt = (
    f"The user asked: {sample_question}\n\n"
    f"Finley answered: {sample_answer}\n\n"
    "Suggest 3 short follow-up questions the user might ask about their finances. "
    "Each question must be 6-10 words. Return a JSON array of 3 strings only, no explanation."
)

try:
    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    start, end = raw.find("["), raw.rfind("]")
    suggestions = json.loads(raw[start : end + 1])

    check("Returns a list",          isinstance(suggestions, list),        f"type={type(suggestions).__name__}")
    check("Exactly 3 suggestions",   len(suggestions) == 3,                f"got {len(suggestions)}")
    check("All strings",             all(isinstance(s, str) for s in suggestions))
    check("All non-empty",           all(len(s.strip()) > 0 for s in suggestions))
    check("Relevant to food/spend",  any(
        any(w in s.lower() for w in ["food", "spend", "zomato", "swiggy", "coffee",
                                      "restaurant", "dining", "month", "budget", "average"])
        for s in suggestions
    ), "at least one suggestion references the topic")

    print(f"\n   Suggestions returned:")
    for i, s in enumerate(suggestions, 1):
        print(f"     {i}. {s}")

except Exception as e:
    check("API call succeeded", False, str(e))
    check("Returns a list",    False)
    check("Exactly 3 suggestions", False)
    check("All strings",       False)
    check("All non-empty",     False)
    check("Relevant to food/spend", False)

# -- Test 2: Vector Search (cross-session semantic memory) --------------------─

print("\n-- TEST 2: Cross-Session Semantic Memory (Vector Search) ----------------")
print("   Seeds 5 Q&A pairs, queries with paraphrased versions, checks retrieval.\n")

import finley.memory as mem_module

TEST_DIR   = ".finley_memory_test"
_orig_dir  = mem_module._MEMORY_DIR
_orig_qa   = mem_module._QA_FILE
_orig_emb  = mem_module._EMB_FILE
mem_module._MEMORY_DIR = TEST_DIR
mem_module._QA_FILE    = os.path.join(TEST_DIR, "conversations.json")
mem_module._EMB_FILE   = os.path.join(TEST_DIR, "embeddings.npy")

from finley.memory import MemoryStore, _THRESHOLD

SEED = [
    {
        "question": "Should I pay off my credit card before contributing to my 401k?",
        "answer":   "With high-interest credit card debt (18%+), pay that off first.",
    },
    {
        "question": "Do I have enough saved for emergencies?",
        "answer":   "Your balance covers about 1.5 months. Target is 3-6 months.",
    },
    {
        "question": "How should I tackle my debt?",
        "answer":   "Avalanche method: hit the highest-interest debt first.",
    },
    {
        "question": "Can I afford to take a vacation next month?",
        "answer":   "Your safe-to-spend is currently negative.",
    },
    {
        "question": "What is my biggest spending problem?",
        "answer":   "Subscriptions and food together account for 38% of total spend.",
    },
]

TESTS = [
    ("Should I invest in my 401k or clear my credit card debt first?", 0),
    ("Is my emergency fund big enough?",                                1),
    ("What's the best strategy to pay down what I owe?",               2),
    ("I want to go on a trip — can I afford it?",                      3),
    ("Where am I overspending the most?",                              4),
]

try:
    store = MemoryStore()
    print("   Loading embedding model (all-MiniLM-L6-v2)...")
    for s in SEED:
        store.store(s["question"], s["answer"])
    print(f"   Seeded {store.count()} Q&A pairs into temporary store.\n")

    for query, expected_idx in TESTS:
        matches      = store.search(query)
        top          = matches[0] if matches else None
        expected_q   = SEED[expected_idx]["question"]
        correct      = top is not None and top["question"] == expected_q
        score_str    = f"score={top['score']:.3f}" if top else "no match"
        check(
            f'"{query[:55]}..."',
            correct,
            score_str + (f" | expected: {expected_q[:50]}..." if not correct else ""),
        )

    print(f"\n   Similarity threshold : {_THRESHOLD}")
    print(f"   Embedding dimensions : 384")
    print(f"   Storage              : numpy array (no vector DB)")

finally:
    mem_module._MEMORY_DIR = _orig_dir
    mem_module._QA_FILE    = _orig_qa
    mem_module._EMB_FILE   = _orig_emb
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

# -- Summary ------------------------------------------------------------------─

print(f"\n{'-'*70}")
print(f"  RESULTS: {passed}/{passed+failed} PASS")
print(f"{'-'*70}\n")
sys.exit(0 if failed == 0 else 1)
