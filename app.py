"""
app.py — Finley UI
Streamlit frontend for the Finley financial advisor pipeline.
"""

import os
import json
import tempfile
import time
import streamlit as st
import pandas as pd
from anthropic import AnthropicBedrock

from finley import (
    process_transactions, build_prompt, SYSTEM_PROMPT,
    AWS_KEY, AWS_SECRET_KEY, AWS_REGION,
    MODEL, SONNET_MODEL, HAIKU_MODEL,
    classify, classify_complexity, python_answer,
)
from finley.memory import MemoryStore

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Finley",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  .stApp {
    background-color: #0f1117;
    color: #e8e8e8;
  }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Card */
  .card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
  }

  /* Stat blocks */
  .stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
  }
  .stat-block {
    flex: 1;
    background: #12151e;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }
  .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
  }
  .stat-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Advice bubble */
  .advice-box {
    background: #12151e;
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 20px 24px;
    margin-bottom: 12px;
    line-height: 1.7;
    color: #d1d5db;
    white-space: pre-wrap;
  }

  /* Chat messages */
  .msg-user {
    background: #1e2030;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    color: #e8e8e8;
    font-size: 0.9rem;
  }
  .msg-finley {
    background: #12151e;
    border: 1px solid #2a2d3a;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 40px 8px 0;
    color: #d1d5db;
    font-size: 0.9rem;
    line-height: 1.6;
    white-space: pre-wrap;
  }

  /* Progress dots */
  .step-dots {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 32px;
  }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #2a2d3a;
  }
  .dot.active { background: #6366f1; }
  .dot.done   { background: #22c55e; }

  /* Section header */
  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
  }

  /* Insight tag */
  .tag {
    display: inline-block;
    background: #1e2030;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #9ca3af;
    margin: 3px;
  }
  .tag.red   { border-color: #ef4444; color: #ef4444; }
  .tag.green { border-color: #22c55e; color: #22c55e; }
  .tag.amber { border-color: #f59e0b; color: #f59e0b; }

  /* Inputs */
  .stTextInput input, .stSelectbox select {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3a !important;
    color: #e8e8e8 !important;
    border-radius: 8px !important;
  }

  /* Buttons */
  .stButton > button {
    background: #6366f1;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    width: 100%;
    transition: background 0.2s;
  }
  .stButton > button:hover {
    background: #5254cc;
  }

  /* Divider */
  hr { border-color: #2a2d3a; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────

DEFAULTS = {
    "step": "upload",          # upload → analysis → chat
    "summary": None,
    "profile": {},
    "messages": [],
    "advice": "",
    "suggestions": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_dollar(n):
    return f"${n:,.0f}"

def get_client():
    if "client" not in st.session_state:
        st.session_state.client = AnthropicBedrock(
            aws_access_key=AWS_KEY,
            aws_secret_key=AWS_SECRET_KEY,
            aws_region=AWS_REGION,
        )
    return st.session_state.client

def get_memory() -> MemoryStore:
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryStore()
    return st.session_state.memory

CHAT_WINDOW = 8  # recent chat messages to keep beyond the first pair

def _generate_suggestions(question: str, answer: str, client) -> list:
    prompt = (
        f"The user asked: {question}\n\n"
        f"Finley answered: {answer}\n\n"
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
        if isinstance(suggestions, list) and len(suggestions) == 3:
            return [str(s) for s in suggestions]
        return []
    except Exception:
        return []

def _build_api_messages(messages: list) -> list:
    """
    Build a clean message list safe for any model:
    - Strips ThinkingBlocks from assistant messages (Haiku/Sonnet reject them)
    - Always keeps the first pair (has the full transaction JSON + profile)
    - Applies a rolling window on the rest so context doesn't grow unboundedly
    """
    def _text_only(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                (b.text if hasattr(b, "text") else b.get("text", ""))
                for b in content
                if (hasattr(b, "type") and b.type == "text") or
                   (isinstance(b, dict) and b.get("type") == "text")
            ]
            return "\n".join(p for p in parts if p)
        return str(content)

    if not messages:
        return []

    clean = [{"role": messages[0]["role"], "content": _text_only(messages[0]["content"])}]
    if len(messages) > 1:
        clean.append({"role": messages[1]["role"], "content": _text_only(messages[1]["content"])})

    rest = messages[2:]
    if len(rest) > CHAT_WINDOW:
        rest = rest[-CHAT_WINDOW:]
    for msg in rest:
        clean.append({"role": msg["role"], "content": _text_only(msg["content"])})

    return clean

def step_dots(current, total=4):
    dots = ""
    for i in range(1, total + 1):
        if i < current:
            dots += '<span class="dot done"></span>'
        elif i == current:
            dots += '<span class="dot active"></span>'
        else:
            dots += '<span class="dot"></span>'
    st.markdown(f'<div class="step-dots">{dots}</div>', unsafe_allow_html=True)

# ── Step 1: Upload ────────────────────────────────────────────────────────────

def render_upload():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## Finley")
        st.markdown('<p style="color:#6b7280;margin-top:-12px;margin-bottom:32px;">Your personal financial advisor</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload your transaction CSV", type="csv", label_visibility="collapsed")

        st.markdown('<p style="color:#6b7280;font-size:0.8rem;text-align:center;margin-top:8px;">or</p>', unsafe_allow_html=True)

        use_demo = st.button("Use demo transactions", use_container_width=True)

        if use_demo:
            demo_path = "data/all_transactions.csv"
            if os.path.exists(demo_path):
                with st.spinner("Loading transactions..."):
                    st.session_state.summary = process_transactions(demo_path)
                st.session_state.step = "analysis"
                st.rerun()
            else:
                st.error(f"{demo_path} not found in project folder.")

        if uploaded:
            with st.spinner("Processing transactions..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                    f.write(uploaded.read())
                    tmp_path = f.name
                st.session_state.summary = process_transactions(tmp_path)
            st.session_state.step = "analysis"
            st.rerun()

# ── Step 2: Analysis + Advice ─────────────────────────────────────────────────

def render_analysis():
    summary = st.session_state.summary
    profile = st.session_state.profile

    # Header
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Your Financial Picture")
    st.markdown(f'<p style="color:#6b7280;font-size:0.85rem;">{summary["date_range"]["from"]} → {summary["date_range"]["to"]}</p>', unsafe_allow_html=True)

    # Stats row
    net = summary["totals"]["net_cashflow"]
    net_color = "#22c55e" if net >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-block">
        <div class="stat-value">{fmt_dollar(summary['totals']['total_spending'])}</div>
        <div class="stat-label">Total Spent</div>
      </div>
      <div class="stat-block">
        <div class="stat-value">{fmt_dollar(summary['totals']['total_income'])}</div>
        <div class="stat-label">Total Income</div>
      </div>
      <div class="stat-block">
        <div class="stat-value" style="color:{net_color}">{fmt_dollar(abs(net))}</div>
        <div class="stat-label">{"Saved" if net >= 0 else "Overspent"}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick signals
    tags = []
    if not profile.get("has_3_month_emergency_fund"):
        tags.append('<span class="tag red">No emergency fund</span>')
    if not profile.get("contributing_to_401k"):
        tags.append('<span class="tag amber">Not contributing to 401k</span>')
    if profile.get("carries_credit_card_balance"):
        tags.append('<span class="tag red">Carrying CC balance</span>')
    if summary.get("month_over_month_change_pct") and summary["month_over_month_change_pct"] > 10:
        tags.append(f'<span class="tag amber">Spending up {summary["month_over_month_change_pct"]}% MoM</span>')
    if tags:
        st.markdown("".join(tags), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Top categories
    st.markdown('<div class="section-label">Top Spending Categories</div>', unsafe_allow_html=True)
    cats = summary.get("by_category", {})
    top_cats = list(cats.items())[:6]
    total_spend = summary["totals"]["total_spending"]
    for cat, data in top_cats:
        pct = round((data["total"] / total_spend) * 100, 1) if total_spend else 0
        bar_width = max(pct, 1)
        st.markdown(f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.85rem;color:#d1d5db;">{cat.replace("_", " ").title()}</span>
            <span style="font-size:0.85rem;color:#9ca3af;">{fmt_dollar(data['total'])} <span style="color:#4b5563">({pct}%)</span></span>
          </div>
          <div style="background:#1a1d27;border-radius:4px;height:4px;">
            <div style="background:#6366f1;width:{bar_width}%;height:4px;border-radius:4px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Advice section
    st.markdown("## Finley's Assessment")

    if not st.session_state.advice:
        if st.button("Get my personalized advice →", use_container_width=True):
            client = get_client()
            user_msg = build_prompt(profile, summary)
            st.session_state.messages = [{"role": "user", "content": user_msg}]

            advice_placeholder = st.empty()
            advice_text = ""
            final = None

            with client.messages.stream(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=st.session_state.messages,
            ) as stream:
                for event in stream:
                    if (
                        hasattr(event, "type")
                        and event.type == "content_block_delta"
                        and hasattr(event, "delta")
                        and event.delta.type == "text_delta"
                    ):
                        advice_text += event.delta.text
                        advice_placeholder.markdown(
                            f'<div class="advice-box">{advice_text}▌</div>',
                            unsafe_allow_html=True,
                        )
                final = stream.get_final_message()

            st.session_state.advice = advice_text
            st.session_state.messages.append({"role": "assistant", "content": final.content})
            advice_placeholder.markdown(
                f'<div class="advice-box">{advice_text}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.step = "chat"
            st.rerun()
    else:
        st.markdown(f'<div class="advice-box">{st.session_state.advice}</div>', unsafe_allow_html=True)
        if st.button("Ask a follow-up →", use_container_width=True):
            st.session_state.step = "chat"
            st.rerun()

# ── Step 4: Chat ──────────────────────────────────────────────────────────────

def render_chat():
    summary = st.session_state.summary
    profile = st.session_state.profile

    # Compact header
    net = summary["totals"]["net_cashflow"]
    net_color = "#22c55e" if net >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="card" style="display:flex;gap:24px;align-items:center;padding:16px 20px;">
      <div><span style="color:#6b7280;font-size:0.75rem;">SPENT</span><br><b>{fmt_dollar(summary['totals']['total_spending'])}</b></div>
      <div><span style="color:#6b7280;font-size:0.75rem;">INCOME</span><br><b>{fmt_dollar(summary['totals']['total_income'])}</b></div>
      <div><span style="color:#6b7280;font-size:0.75rem;">NET</span><br><b style="color:{net_color}">{fmt_dollar(abs(net))}</b></div>
      <div style="margin-left:auto;font-size:0.75rem;color:#6b7280;">Goal: {profile.get('primary_goal','—').title()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Initial advice as first message
    if st.session_state.advice:
        st.markdown(f'<div class="msg-finley">{st.session_state.advice}</div>', unsafe_allow_html=True)

    # Conversation history (skip first user + assistant pair — that's the analysis)
    history = st.session_state.messages[2:] if len(st.session_state.messages) > 2 else []
    for msg in history:
        role = msg["role"]
        content = msg["content"] if isinstance(msg["content"], str) else ""
        if role == "user":
            st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-finley">{content}</div>', unsafe_allow_html=True)

    # Suggestion buttons below most recent Finley response
    if st.session_state.suggestions:
        cols = st.columns(len(st.session_state.suggestions))
        for i, (col, suggestion) in enumerate(zip(cols, st.session_state.suggestions)):
            with col:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.pending_suggestion = suggestion
                    st.session_state.suggestions = []
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    pending = st.session_state.pop("pending_suggestion", None)

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        followup = st.text_input("Ask Finley anything...", key=f"chat_input_{len(st.session_state.messages)}", label_visibility="collapsed", placeholder="Ask Finley anything about your finances...")
    with col2:
        send = st.button("Send", use_container_width=True)

    question = None
    if pending:
        question = pending
    elif send and followup.strip():
        question = followup.strip()

    if question:
        st.session_state.suggestions = []
        st.session_state.messages.append({"role": "user", "content": question})

        route = classify(question)

        if route == "python":
            response_text = python_answer(question, summary) or ""
            if not response_text:
                route = "llm"

        if route == "llm":
            complexity = classify_complexity(question)
            chat_model = HAIKU_MODEL if complexity == "data" else SONNET_MODEL

            client = get_client()
            memory = get_memory()

            # Retrieve relevant past advice exchanges and prepend to system prompt
            system_with_memory = SYSTEM_PROMPT
            if complexity == "advice":
                matches = memory.search(question)
                context = memory.format_context(matches)
                if context:
                    system_with_memory = SYSTEM_PROMPT + "\n\n" + context

            response_text = ""
            resp_placeholder = st.empty()

            with client.messages.stream(
                model=chat_model,
                max_tokens=1000,
                system=system_with_memory,
                messages=_build_api_messages(st.session_state.messages),
            ) as stream:
                for text in stream.text_stream:
                    response_text += text
                    resp_placeholder.markdown(
                        f'<div class="msg-finley">{response_text}▌</div>',
                        unsafe_allow_html=True,
                    )
            resp_placeholder.empty()

            # Only store advice answers — data answers go stale with new transactions
            if complexity == "advice":
                memory.store(question, response_text)

            st.session_state.suggestions = _generate_suggestions(question, response_text, client)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()

# ── Router ────────────────────────────────────────────────────────────────────

step = st.session_state.step

if step == "upload":
    render_upload()
elif step == "analysis":
    render_analysis()
elif step == "chat":
    render_chat()
