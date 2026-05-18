"""
scripts/add_routing_column.py — Add Route classification column to queries_results_judged.xlsx.

Route values:
  Python Lookup — answerable by reading a field from the pre-computed JSON summary directly
  LLM Required  — needs reasoning, synthesis, trend analysis, or financial advice
  Out of Scope  — platform how-tos, general finance education unrelated to user data
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

INPUT_FILE  = "reports/queries_results_judged.xlsx"
OUTPUT_FILE = "reports/queries_results_judged.xlsx"

# Route for each question in order (matches queries_new.txt, 51 questions)
ROUTES = [
    "LLM Required",   # Q1  — subscription trend over past year vs other categories
    "LLM Required",   # Q2  — category breakdown by month, past 6 months
    "LLM Required",   # Q3  — avg monthly subscription spend vs last quarter
    "Python Lookup",  # Q4  — largest healthcare transactions last 3 months
    "Python Lookup",  # Q5  — largest transactions this month
    "LLM Required",   # Q6  — analyze spending patterns / overspending
    "LLM Required",   # Q7  — spending breakdown this month with insights
    "LLM Required",   # Q8  — this month vs last month
    "LLM Required",   # Q9  — where am I overspending right now
    "LLM Required",   # Q10 — compared to past few months (context-dependent)
    "Python Lookup",  # Q11 — restaurants and food spending
    "Python Lookup",  # Q12 — break down spending by category
    "Python Lookup",  # Q13 — spending this month
    "LLM Required",   # Q14 — spending per category last month vs previous month
    "LLM Required",   # Q15 — subscription details last 3 months
    "LLM Required",   # Q16 — can I spend $300 on car part without going broke
    "LLM Required",   # Q17 — subscriptions vs other categories this month
    "Python Lookup",  # Q18 — top spending categories last month
    "LLM Required",   # Q19 — monthly financial review
    "Python Lookup",  # Q20 — duplicate subscriptions
    "Python Lookup",  # Q21 — recurring subscriptions list
    "LLM Required",   # Q22 — largest category increase/decrease vs previous month
    "LLM Required",   # Q23 — annual subscription spend
    "Python Lookup",  # Q24 — total coffee spend
    "Python Lookup",  # Q25 — am I paying for Amazon Prime
    "Python Lookup",  # Q26 — coffee spend May 2025
    "Python Lookup",  # Q27 — coffee spend this month
    "Python Lookup",  # Q28 — am I paying for Netflix
    "Python Lookup",  # Q29 — highest monthly cost subscription
    "LLM Required",   # Q30 — which category increased the most
    "Python Lookup",  # Q31 — Uber Eats spend
    "LLM Required",   # Q32 — coffee last month vs same month last year
    "Python Lookup",  # Q33 — groceries last month
    "Python Lookup",  # Q34 — travel transactions
    "Python Lookup",  # Q35 — safe to spend breakdown
    "LLM Required",   # Q36 — how is planned spending goal calculated
    "LLM Required",   # Q37 — do I have enough for a trip next month
    "LLM Required",   # Q38 — how do I pay off my current debt
    "LLM Required",   # Q39 — I have $17K debt, how do I tackle it
    "LLM Required",   # Q40 — credit card vs 401k contribution priority
    "LLM Required",   # Q41 — how long to pay off one credit card
    "Out of Scope",   # Q42 — what is an HSA
    "Out of Scope",   # Q43 — how do I add an account
    "Out of Scope",   # Q44 — why haven't my transactions updated
    "Out of Scope",   # Q45 — how do I change a transaction category
    "Out of Scope",   # Q46 — can I schedule automatic data refreshes
    "Out of Scope",   # Q47 — I hate AI, can I talk with a real person
    "Python Lookup",  # Q48 — what's included in my safe to spend number
    "LLM Required",   # Q49 — Roth 401k vs traditional, which is right for me
    "Python Lookup",  # Q50 — what are the actual transactions
    "Python Lookup",  # Q51 — upcoming bills
]

DARK      = PatternFill("solid", fgColor="1F2937")
LOOKUP_CLR = PatternFill("solid", fgColor="DBEAFE")   # blue-100
LLM_CLR    = PatternFill("solid", fgColor="EDE9FE")   # purple-100
SCOPE_CLR  = PatternFill("solid", fgColor="FEF9C3")   # yellow-100
STRIPE     = PatternFill("solid", fgColor="F9FAFB")
WHITE      = PatternFill("solid", fgColor="FFFFFF")
BORDER = Border(
    left=Side(style="thin", color="E5E7EB"),
    right=Side(style="thin", color="E5E7EB"),
    top=Side(style="thin", color="E5E7EB"),
    bottom=Side(style="thin", color="E5E7EB"),
)

hdr_font  = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
body_font = Font(name="Calibri", size=10)
center_top = Alignment(horizontal="center", vertical="top", wrap_text=True)

wb = openpyxl.load_workbook(INPUT_FILE)
ws = wb.active

# Update title row
title = ws["A1"].value or ""
n_lookup = ROUTES.count("Python Lookup")
n_llm    = ROUTES.count("LLM Required")
n_scope  = ROUTES.count("Out of Scope")
ws["A1"].value = title + f" | Route: {n_lookup} Python Lookup / {n_llm} LLM Required / {n_scope} Out of Scope"
ws.merge_cells("A1:K1")

# Header for column 11
hdr_cell = ws.cell(row=2, column=11, value="Route")
hdr_cell.fill      = DARK
hdr_cell.font      = hdr_font
hdr_cell.border    = BORDER
hdr_cell.alignment = Alignment(horizontal="center", vertical="center")

ws.column_dimensions[get_column_letter(11)].width = 16

# Data rows: rows 3 to 3+51-1 = 53
for i, route in enumerate(ROUTES):
    row = i + 3
    base_fill = STRIPE if row % 2 == 0 else WHITE

    if route == "Python Lookup":
        fill  = LOOKUP_CLR
        color = "1D4ED8"  # blue-700
    elif route == "LLM Required":
        fill  = LLM_CLR
        color = "6D28D9"  # purple-700
    else:
        fill  = SCOPE_CLR
        color = "92400E"  # amber-700

    c = ws.cell(row=row, column=11, value=route)
    c.fill      = fill
    c.font      = Font(name="Calibri", bold=True, size=10, color=color)
    c.border    = BORDER
    c.alignment = center_top

wb.save(OUTPUT_FILE)

print(f"Done. Route column added to {OUTPUT_FILE}")
print(f"  Python Lookup : {n_lookup}")
print(f"  LLM Required  : {n_llm}")
print(f"  Out of Scope  : {n_scope}")
