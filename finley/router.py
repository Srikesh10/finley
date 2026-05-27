"""
finley/router.py — Question routing layer.

classify(question) → "python" | "llm"
python_answer(question, summary) → str

Routes deterministic data-lookup questions directly to Python,
skipping the LLM entirely. Everything else goes to the LLM.
"""

import re

# ── Category name mappings ────────────────────────────────────────────────────

_CATEGORY_ALIASES = {
    "food": "FOOD_AND_DRINK",
    "food and drink": "FOOD_AND_DRINK",
    "food & drink": "FOOD_AND_DRINK",
    "dining": "FOOD_AND_DRINK",
    "restaurants": "FOOD_AND_DRINK",
    "groceries": "FOOD_AND_DRINK",
    "grocery": "FOOD_AND_DRINK",
    "healthcare": "MEDICAL",
    "medical": "MEDICAL",
    "health": "MEDICAL",
    "travel": "TRAVEL",
    "transportation": "TRANSPORTATION",
    "transport": "TRANSPORTATION",
    "entertainment": "ENTERTAINMENT",
    "shopping": "GENERAL_MERCHANDISE",
    "merchandise": "GENERAL_MERCHANDISE",
    "utilities": "HOME",
    "home": "HOME",
    "subscriptions": "GENERAL_SERVICES",
    "services": "GENERAL_SERVICES",
    "personal care": "PERSONAL_CARE",
    "education": "EDUCATION",
    "government": "GOVERNMENT_AND_NON_PROFIT",
}

# Terms that mean "this month" (current partial month)
_THIS_MONTH_TERMS = r"this month|current month|so far|mtd"
# Terms that mean "last month" (last complete month)
_LAST_MONTH_TERMS = r"last month|previous month|past month"


# ── Pattern registry ──────────────────────────────────────────────────────────
# Each entry: (compiled_regex, handler_name)
# Evaluated in order — first match wins.

_PATTERNS: list[tuple[re.Pattern, str]] = []

def _register(pattern: str, handler: str):
    _PATTERNS.append((re.compile(pattern, re.IGNORECASE), handler))

# Subscription presence — "am I paying for Netflix?"
_register(
    r"(?:am i|do i|are we|is there|do we)\s+(?:still\s+)?(?:paying for|pay for|subscribed to|have\s+(?:a\s+)?(?:subscription (?:to|for)|account (?:with|on)))\s+(.+?)(?:\?|$)",
    "subscription_presence",
)
# Also catch "is [service] on my subscriptions?"
_register(
    r"^(?:is|does)\s+(\S+(?:\s+\S+)?)\s+(?:show up|appear|on my|in my|still|active)",
    "subscription_presence",
)

# Upcoming bills
_register(
    r"(?:upcoming|due|pending|next)\s+(?:\d+\s+days?\s+)?bills?|bills?\s+(?:due|coming up|next|upcoming)|what\s+(?:bills?|payments?)\s+(?:do i have|are due|are coming)",
    "upcoming_bills",
)

# Safe to spend — must come BEFORE account_balances (queries often contain the word "balance")
_register(
    r"safe.?to.?spend|how much can i (?:safely\s+)?spend|what(?:'s| is)\s+(?:my\s+)?(?:safe|available\s+to\s+spend)|break\s*down\s+(?:my\s+)?safe|what(?:'s| is)\s+included\s+in\s+(?:my\s+)?safe",
    "safe_to_spend",
)

# Account balances
_register(
    r"(?:my\s+)?(?:account\s+)?balances?|how much\s+(?:is\s+)?(?:in\s+)?(?:my\s+)?(?:account|bank|checking|savings)|what(?:'s| is)\s+(?:in\s+)?my\s+(?:account|checking|savings|bank)",
    "account_balances",
)

# Subscription list — simple list request (no trend/details/compare modifier)
_register(
    r"^(?:show\s+(?:me\s+)?(?:my\s+)?|list\s+(?:my\s+)?|what\s+are\s+(?:my\s+)?)(?:active\s+|recurring\s+|current\s+)?subscriptions?(?:\s*[?.]?\s*$)",
    "subscription_list",
)

# Highest/most expensive subscription
_register(
    r"(?:which|what)\s+subscription\s+(?:has\s+the\s+)?(?:highest|most expensive|largest|biggest)\s+(?:monthly\s+)?(?:cost|price|amount|fee)|most expensive\s+subscription|highest.cost\s+subscription",
    "subscription_most_expensive",
)

# Top categories — last month
_register(
    r"(?:top|biggest|largest|highest)\s+(?:spending\s+)?categor(?:y|ies)\s+(?:(?:from|for|in)\s+(?:the\s+)?)?(?:last|previous|past)\s+month|last\s+month(?:'s|s)?\s+(?:top|biggest|largest)\s+(?:spending\s+)?categor",
    "top_categories_last_month",
)

# Category spending — last month (specific category named)
_register(
    r"(?:what\s+did\s+i\s+spend|how\s+much\s+(?:did\s+i\s+spend|have\s+i\s+spent)?)\s+on\s+(\w[\w\s&]+?)\s+last\s+month",
    "category_spending_last_month",
)

# Merchant / coffee spending — this month
_register(
    r"(?:how much have i spent|what(?:'s| is) my spend|what did i spend)\s+on\s+(\w[\w\s]+?)\s+(?:this month|so far|mtd|current month)",
    "merchant_spending_this_month",
)

# Merchant / coffee spending — specific month (e.g. "in May 2025")
_register(
    r"(?:how much have i spent|what(?:'s| is) my spend|what did i spend)\s+on\s+(\w[\w\s]+?)\s+in\s+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}|\d{4}-\d{2})",
    "merchant_spending_month",
)

# Merchant / coffee spending — all time / total
_register(
    r"how much have i spent on\s+(\w[\w\s]+?)(?:\?|$|\s+(?:total|all time|ever|overall))",
    "merchant_spending_total",
)


# ── classify ──────────────────────────────────────────────────────────────────

def classify(question: str) -> str:
    """Return 'python' if answerable from pre-computed JSON, else 'llm'."""
    q = question.strip()
    for pattern, _ in _PATTERNS:
        if pattern.search(q):
            return "python"
    return "llm"


_ADVICE_RE = re.compile(
    r"should i|how should|recommend|advice|advise|"
    r"how do i (?:pay|save|invest|budget|tackle|handle|reduce|manage|build|get out|deal)|"
    r"plan|strategy|best way|worth it|better to|"
    r"pay off|invest|retirement|401k|savings goal|emergency fund|"
    r"can i afford|can i spend|enough to|have enough|"
    r"what if|what would happen|"
    r"analyze|analysis|insight|pattern|overview|review|"
    r"duplicate|overlap|"
    r"trend|over time|annually|yearly|quarterly|"
    r"increase|decrease|grew|dropped|compar",
    re.IGNORECASE,
)

def classify_complexity(question: str) -> str:
    """
    For questions already classified as 'llm', determine model tier.
    Returns 'advice' (needs Sonnet — reasoning/recommendations) or
    'data' (Haiku is fine — retrieval/display with no reasoning required).
    """
    if _ADVICE_RE.search(question.strip()):
        return "advice"
    return "data"


# ── python_answer ─────────────────────────────────────────────────────────────

def python_answer(question: str, summary: dict) -> str | None:
    """
    Generate a direct answer from the pre-computed summary JSON.
    Returns None if no pattern matched (shouldn't happen if classify returned 'python').
    """
    q = question.strip()
    for pattern, handler in _PATTERNS:
        m = pattern.search(q)
        if m:
            fn = _HANDLERS.get(handler)
            if fn:
                return fn(q, m, summary)
    return None


# ── Handlers ──────────────────────────────────────────────────────────────────

def _fmt(n) -> str:
    try:
        return f"${float(n):,.2f}"
    except (TypeError, ValueError):
        return str(n)


def _h_subscription_presence(q: str, m: re.Match, summary: dict) -> str:
    # Extract service name from capture group or from question
    service = (m.group(1) if m.lastindex and m.group(1) else q).strip().lower()
    service = re.sub(r"[?.!]+$", "", service).strip()

    subs = summary.get("subscriptions") or []
    service_words = [w for w in service.split() if len(w) > 3]
    matches = []
    for s in subs:
        name = str(s.get("merchant_name") or s.get("description") or "").lower()
        partial = service_words and any(w in name for w in service_words)
        if service in name or name in service or _fuzzy_match(service, name) or partial:
            matches.append(s)

    if matches:
        s = matches[0]
        name = s.get("merchant_name") or s.get("description") or "Unknown"
        amt = _fmt(s.get("average_amount", 0))
        freq = s.get("frequency", "monthly").lower()
        return f"Yes — {name} is active on your account at {amt}/{freq}. Cancel in the app or directly with the provider."
    else:
        # Check merchant_monthly as fallback
        merchant_monthly = summary.get("merchant_monthly") or {}
        for merchant, months in merchant_monthly.items():
            ml = merchant.lower()
            partial = service_words and any(w in ml for w in service_words)
            if service in ml or _fuzzy_match(service, ml) or partial:
                total = sum(months.values())
                recent = sorted(months.keys())[-1] if months else "unknown"
                return f"{merchant} appears in your transaction history (most recent: {recent}, {_fmt(total)} total) but is not flagged as an active subscription by your bank."
        return f"No — {service.title()} does not appear in your active subscriptions or recent transactions."


def _h_upcoming_bills(q: str, m: re.Match, summary: dict) -> str:
    bills = summary.get("upcoming_bills") or []
    if not bills:
        return "No upcoming bills found in the next 30 days based on your linked accounts."

    lines = [f"**{len(bills)} bills due in the next 30 days:**\n"]
    total = 0.0
    for b in sorted(bills, key=lambda x: x.get("due_date", "")):
        name = b.get("name") or b.get("merchant_name") or "Unknown"
        amt  = float(b.get("amount", 0))
        due  = b.get("due_date", "—")
        freq = b.get("frequency", "")
        total += amt
        lines.append(f"  • {name}: {_fmt(amt)} due {due}" + (f" ({freq})" if freq else ""))

    lines.append(f"\n**Total due: {_fmt(total)}**")

    sts = summary.get("safe_to_spend") or {}
    result = sts.get("result", 0)
    if result and float(result) >= total:
        lines.append(f"Your safe-to-spend is {_fmt(result)} — you can cover all of these.")
    elif result:
        lines.append(f"Your safe-to-spend is {_fmt(result)} — short by {_fmt(total - float(result))}. Review discretionary spending before the due dates.")

    return "\n".join(lines)


def _h_account_balances(q: str, m: re.Match, summary: dict) -> str:
    accounts = summary.get("accounts_summary") or []
    if not accounts:
        return "No account data available. Connect your accounts in the app."

    lines = []
    total_available = 0.0
    for a in accounts:
        name     = a.get("name", "Account")
        subtype  = a.get("subtype", "").replace("_", " ").title()
        balance  = float(a.get("balance") or 0)
        available = float(a.get("available") or balance)
        total_available += available
        lines.append(f"  • {name} ({subtype}): {_fmt(balance)} balance / {_fmt(available)} available")

    header = f"**{len(accounts)} linked account{'s' if len(accounts) != 1 else ''}:**\n"
    footer = f"\n**Total available: {_fmt(total_available)}**"
    return header + "\n".join(lines) + footer


def _h_safe_to_spend(q: str, m: re.Match, summary: dict) -> str:
    sts = summary.get("safe_to_spend") or {}
    result       = float(sts.get("result") or 0)
    checking     = float(sts.get("checking_available") or 0)
    upcoming     = float(sts.get("upcoming_bills_30d") or 0)
    mtd_spending = float(sts.get("mtd_spending_ex_transfers") or 0)

    sign = "+" if result >= 0 else ""
    return (
        f"**Safe to spend: {_fmt(result)}**\n\n"
        f"  Checking available:        {_fmt(checking)}\n"
        f"  - Upcoming bills (30d):   -{_fmt(upcoming)}\n"
        f"  - Month-to-date spend:    -{_fmt(mtd_spending)}\n"
        f"  ----------------------------------\n"
        f"  = Safe to spend:          {sign}{_fmt(result)}\n\n"
        + ("You have a comfortable buffer." if result > 500 else
           "Buffer is tight — hold off on discretionary purchases until next paycheck." if result >= 0 else
           "You're overextended this month. Review upcoming bills and cut non-essentials now.")
    )


def _h_subscription_list(q: str, m: re.Match, summary: dict) -> str:
    subs = summary.get("subscriptions") or []
    if not subs:
        return "No active subscriptions found in your linked accounts."

    subs_sorted = sorted(subs, key=lambda s: float(s.get("average_amount") or 0), reverse=True)
    total = sum(float(s.get("average_amount") or 0) for s in subs_sorted)

    lines = [f"**{len(subs_sorted)} active subscriptions — {_fmt(total)}/month total:**\n"]
    for s in subs_sorted:
        name = s.get("merchant_name") or s.get("description") or "Unknown"
        amt  = _fmt(s.get("average_amount", 0))
        freq = s.get("frequency", "monthly").lower()
        lines.append(f"  • {name}: {amt}/{freq}")

    lines.append(f"\nCancel anything you haven't used in 30 days.")
    return "\n".join(lines)


def _h_subscription_most_expensive(q: str, m: re.Match, summary: dict) -> str:
    subs = summary.get("subscriptions") or []
    if not subs:
        return "No active subscriptions found."

    top = max(subs, key=lambda s: float(s.get("average_amount") or 0))
    name = top.get("merchant_name") or top.get("description") or "Unknown"
    amt  = _fmt(top.get("average_amount", 0))
    freq = top.get("frequency", "monthly").lower()

    runner_up = sorted(subs, key=lambda s: float(s.get("average_amount") or 0), reverse=True)
    lines = [f"**{name}** is your most expensive subscription at {amt}/{freq}."]
    if len(runner_up) > 1:
        r = runner_up[1]
        r_name = r.get("merchant_name") or r.get("description") or "Unknown"
        lines.append(f"Next: {r_name} at {_fmt(r.get('average_amount', 0))}/{r.get('frequency','monthly').lower()}.")
    return " ".join(lines)


def _h_top_categories_last_month(q: str, m: re.Match, summary: dict) -> str:
    lcm = summary.get("last_complete_month")
    mbc = summary.get("monthly_by_category") or {}

    if not lcm or lcm not in mbc:
        return "Last month's category data isn't available."

    spending = mbc[lcm].get("spending") or {}
    if not spending:
        return f"No spending data for {lcm}."

    sorted_cats = sorted(spending.items(), key=lambda x: x[1], reverse=True)
    total = sum(spending.values())

    lines = [f"**Top spending categories — {lcm}:**\n"]
    for cat, amt in sorted_cats[:6]:
        label = cat.replace("_", " ").title()
        pct   = round(amt / total * 100, 1) if total else 0
        lines.append(f"  • {label}: {_fmt(amt)} ({pct}%)")

    lines.append(f"\n**Total: {_fmt(total)}**")
    return "\n".join(lines)


def _h_category_spending_last_month(q: str, m: re.Match, summary: dict) -> str | None:
    term = (m.group(1) if m.lastindex else "").strip().lower()
    lcm  = summary.get("last_complete_month")
    mbc  = summary.get("monthly_by_category") or {}

    if not lcm or lcm not in mbc:
        return None

    spending = mbc[lcm].get("spending") or {}
    subcats  = mbc[lcm].get("subcategories") or {}

    # 1. Check subcategories first (most specific — catches "groceries", "coffee", "restaurants")
    for cat, sub in subcats.items():
        for subcat, amt in sub.items():
            subcat_words = subcat.lower().replace("_", " ")
            if term in subcat_words or subcat_words.startswith(term):
                lines = [f"**{term.title()} spending in {lcm}: {_fmt(amt)}**"]
                parent = spending.get(cat, 0)
                if parent and abs(float(parent) - float(amt)) > 1:
                    lines.append(f"Part of {cat.replace('_', ' ').title()} total: {_fmt(parent)}")
                return "\n".join(lines)

    # 2. Check top-level category alias
    cat_key = _CATEGORY_ALIASES.get(term)
    if cat_key and cat_key in spending:
        total = spending[cat_key]
        sub   = subcats.get(cat_key, {})
        lines = [f"**{term.title()} spending in {lcm}: {_fmt(total)}**"]
        for subcat, amt in sorted(sub.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"  • {subcat.replace('_', ' ').title()}: {_fmt(amt)}")
        return "\n".join(lines)

    # 3. Fuzzy scan top-level categories
    for cat, amt in spending.items():
        if term in cat.lower().replace("_", " "):
            return f"**{term.title()} spending in {lcm}: {_fmt(amt)}**"

    # Can't answer reliably — fall back to LLM
    return None


def _h_merchant_spending_this_month(q: str, m: re.Match, summary: dict) -> str | None:
    term = (m.group(1) if m.lastindex else "").strip().lower()
    mbc  = summary.get("monthly_by_category") or {}
    mm   = summary.get("merchant_monthly") or {}

    current = next((mo for mo, d in sorted(mbc.items()) if d.get("is_partial")), None)
    if not current:
        return None

    # Try merchant_monthly first
    merchant, total = _find_merchant(term, mm, month=current)
    if merchant and total:
        return f"**{term.title()} spending so far this month ({current}): {_fmt(total)}**\nThis is a partial month — the final number will be higher."

    # Try subcategory fallback
    month_data = mbc.get(current, {})
    subcats    = month_data.get("subcategories") or {}
    for cat, sub in subcats.items():
        for subcat, amt in sub.items():
            if term in subcat.lower().replace("_", " "):
                return f"**{term.title()} spending so far this month ({current}): {_fmt(amt)}**\nThis is a partial month — the final number will be higher."

    return None


def _h_merchant_spending_month(q: str, m: re.Match, summary: dict) -> str | None:
    term      = (m.group(1) if m.lastindex and m.lastindex >= 1 else "").strip().lower()
    month_str = (m.group(2) if m.lastindex and m.lastindex >= 2 else "").strip()
    mm        = summary.get("merchant_monthly") or {}
    mbc       = summary.get("monthly_by_category") or {}

    month_key = _parse_month(month_str)
    if not month_key:
        return None

    # Try merchant_monthly first
    merchant, total = _find_merchant(term, mm, month=month_key)
    if merchant and total:
        return f"**{term.title()} spending in {month_key}: {_fmt(total)}**"

    # Try subcategory fallback
    month_data = mbc.get(month_key, {})
    subcats    = month_data.get("subcategories") or {}
    for cat, sub in subcats.items():
        for subcat, amt in sub.items():
            if term in subcat.lower().replace("_", " "):
                return f"**{term.title()} spending in {month_key}: {_fmt(amt)}**"

    return None


def _h_merchant_spending_total(q: str, m: re.Match, summary: dict) -> str | None:
    term = (m.group(1) if m.lastindex else "").strip().lower()
    mm   = summary.get("merchant_monthly") or {}
    mbc  = summary.get("monthly_by_category") or {}

    # Try merchant_monthly first
    merchant, total = _find_merchant(term, mm)
    if merchant and total:
        months = mm.get(merchant, {})
        n   = len([v for v in months.values() if v > 0])
        avg = total / n if n else 0
        return f"**Total {term} spending: {_fmt(total)}** across {n} months ({_fmt(avg)}/month avg).\nData covers the last 12 months."

    # Try summing across subcategories in monthly_by_category
    # Accumulate per (cat, subcat) key across all complete months
    subcat_totals: dict[tuple, float] = {}
    for month, month_data in mbc.items():
        if month_data.get("is_partial"):
            continue
        subcats = month_data.get("subcategories") or {}
        for cat, sub in subcats.items():
            for subcat, amt in sub.items():
                if term in subcat.lower().replace("_", " "):
                    key = (cat, subcat)
                    subcat_totals[key] = subcat_totals.get(key, 0) + float(amt)

    if subcat_totals:
        best_key = max(subcat_totals, key=subcat_totals.get)
        matched_cat, matched_subcat = best_key
        total = round(subcat_totals[best_key], 2)
        # Count months where this subcat appeared
        n = sum(
            1 for md in mbc.values()
            if not md.get("is_partial")
            and matched_subcat in (md.get("subcategories") or {}).get(matched_cat, {})
        )
        avg = round(total / n, 2) if n else 0
        return (f"**Total {term} spending: {_fmt(total)}** across {n} complete months "
                f"({_fmt(avg)}/month avg).\nTracked under "
                f"{matched_cat.replace('_',' ').title()} / {matched_subcat.replace('_',' ').title()}.")

    return None


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fuzzy_match(a: str, b: str) -> bool:
    """Very light fuzzy match — checks if all words in a appear in b."""
    words = [w for w in a.split() if len(w) > 3]
    return bool(words) and all(w in b for w in words)


def _find_merchant(term: str, merchant_monthly: dict, month: str | None = None) -> tuple[str | None, float]:
    """Find best-matching merchant and return (name, total_or_month_amount)."""
    best_name  = None
    best_score = 0

    for merchant in merchant_monthly:
        ml = merchant.lower()
        score = 0
        if term in ml:
            score = len(term)
        elif _fuzzy_match(term, ml):
            score = 1
        if score > best_score:
            best_score = score
            best_name  = merchant

    if not best_name:
        return None, 0.0

    months = merchant_monthly[best_name]
    if month:
        return best_name, months.get(month, 0.0)
    return best_name, round(sum(months.values()), 2)


def _parse_month(s: str) -> str | None:
    """Parse 'May 2025' or '2025-05' → '2025-05'."""
    import datetime
    s = s.strip()
    if re.match(r"^\d{4}-\d{2}$", s):
        return s
    for fmt in ("%B %Y", "%b %Y"):
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    return None


# ── Handler dispatch map ──────────────────────────────────────────────────────

_HANDLERS = {
    "subscription_presence":        _h_subscription_presence,
    "upcoming_bills":               _h_upcoming_bills,
    "account_balances":             _h_account_balances,
    "safe_to_spend":                _h_safe_to_spend,
    "subscription_list":            _h_subscription_list,
    "subscription_most_expensive":  _h_subscription_most_expensive,
    "top_categories_last_month":    _h_top_categories_last_month,
    "category_spending_last_month": _h_category_spending_last_month,
    "merchant_spending_this_month": _h_merchant_spending_this_month,
    "merchant_spending_month":      _h_merchant_spending_month,
    "merchant_spending_total":      _h_merchant_spending_total,
}
