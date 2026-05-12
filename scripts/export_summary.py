import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finley import process_transactions
from config import TX_FILE, ACCOUNTS_FILE, RECURRING_FILE, SALARY_FILE
import json

summary = process_transactions(
    tx_path        = TX_FILE,
    accounts_path  = ACCOUNTS_FILE,
    recurring_path = RECURRING_FILE,
    salary_path    = SALARY_FILE,
)
with open("data/transaction_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved to data/transaction_summary.json")
print(f"JSON size: {len(json.dumps(summary)):,} chars")
