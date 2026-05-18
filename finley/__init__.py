# Public API — all imports from `finley` work as before
from finley.client import AWS_KEY, AWS_SECRET_KEY, AWS_REGION, MODEL, make_client
from finley.data import process_transactions
from finley.prompt import SYSTEM_PROMPT, EVAL_SYSTEM, build_prompt
from finley.tracing import log as _lf_log, score as _lf_score, flush as _lf_flush
from finley.core import (
    collect_user_profile,
    evaluate_advice,
    print_eval_report,
    main,
)

__all__ = [
    "AWS_KEY", "AWS_SECRET_KEY", "AWS_REGION", "MODEL", "make_client",
    "process_transactions",
    "SYSTEM_PROMPT", "EVAL_SYSTEM", "build_prompt",
    "_lf_log", "_lf_score", "_lf_flush",
    "collect_user_profile", "evaluate_advice", "print_eval_report", "main",
]
