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

from finley import process_transactions, build_prompt, SYSTEM_PROMPT, AWS_KEY, AWS_SECRET_KEY, AWS_REGION, MODEL

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
    "step": "upload",          # upload → onboard → analysis → chat
    "summary": None,
    "profile": {},
    "messages": [],
    "advice": "",
    "onboard_step": 1,
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
                st.session_state.step = "onboard"
                st.rerun()
            else:
                st.error(f"{demo_path} not found in project folder.")

        if uploaded:
            with st.spinner("Processing transactions..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                    f.write(uploaded.read())
                    tmp_path = f.name
                st.session_state.summary = process_transactions(tmp_path)
            st.session_state.step = "onboard"
            st.rerun()

# ── Step 2: Onboarding ────────────────────────────────────────────────────────

def render_onboard():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        s = st.session_state.onboard_step

        step_dots(s)

        if s == 1:
            st.markdown("### How old are you?")
            age = st.text_input("Age", placeholder="e.g. 28", label_visibility="collapsed")
            if st.button("Continue →"):
                if age.strip():
                    st.session_state.profile["age"] = age.strip()
                    st.session_state.onboard_step = 2
                    st.rerun()

        elif s == 2:
            st.markdown("### What's your monthly take-home pay?")
            st.markdown('<p style="color:#6b7280;font-size:0.85rem;">After tax, what hits your bank account each month</p>', unsafe_allow_html=True)
            income = st.text_input("Monthly income", placeholder="e.g. 5000", label_visibility="collapsed")
            if st.button("Continue →"):
                if income.strip():
                    st.session_state.profile["monthly_take_home"] = f"${income.strip()}"
                    st.session_state.onboard_step = 3
                    st.rerun()

        elif s == 3:
            st.markdown("### What's your #1 financial goal right now?")
            goal = st.selectbox(
                "Goal",
                [
                    "Eliminate debt",
                    "Build emergency fund",
                    "Save for a home down payment",
                    "Build retirement savings",
                    "Understand where my money goes",
                ],
                label_visibility="collapsed",
            )
            if st.button("Continue →"):
                st.session_state.profile["primary_goal"] = goal.lower()
                st.session_state.onboard_step = 4
                st.rerun()

        elif s == 4:
            st.markdown("### Last few questions")
            st.markdown('<p style="color:#6b7280;font-size:0.85rem;">Yes / No</p>', unsafe_allow_html=True)
            emergency = st.radio("Do you have 3+ months of expenses saved?", ["No", "Yes"], horizontal=True)
            retirement = st.radio("Are you contributing to a 401k?", ["No", "Yes"], horizontal=True)
            cc = st.radio("Do you carry a credit card balance each month?", ["No", "Yes"], horizontal=True)

            if st.button("Analyze my finances →"):
                st.session_state.profile["has_3_month_emergency_fund"] = emergency == "Yes"
                st.session_state.profile["contributing_to_401k"] = retirement == "Yes"
                st.session_state.profile["carries_credit_card_balance"] = cc == "Yes"
                st.session_state.step = "analysis"
                st.rerun()

# ── Step 3: Analysis + Advice ─────────────────────────────────────────────────

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

            with client.messages.stream(
                model=MODEL,
                max_tokens=2000,
                thinking={"type": "enabled", "budget_tokens": 8000},
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

    st.markdown("<br>", unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        followup = st.text_input("Ask Finley anything...", key=f"chat_input_{len(st.session_state.messages)}", label_visibility="collapsed", placeholder="Ask Finley anything about your finances...")
    with col2:
        send = st.button("Send", use_container_width=True)

    if send and followup.strip():
        st.session_state.messages.append({"role": "user", "content": followup.strip()})

        client = get_client()
        response_text = ""
        resp_placeholder = st.empty()

        with client.messages.stream(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=st.session_state.messages,
        ) as stream:
            for text in stream.text_stream:
                response_text += text
                resp_placeholder.markdown(
                    f'<div class="msg-finley">{response_text}▌</div>',
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        resp_placeholder.empty()
        st.rerun()

# ── Router ────────────────────────────────────────────────────────────────────

step = st.session_state.step

if step == "upload":
    render_upload()
elif step == "onboard":
    render_onboard()
elif step == "analysis":
    render_analysis()
elif step == "chat":
    render_chat()
