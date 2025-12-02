# File: modules/tracker.py
import datetime as dt
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import json
import numpy as np
import pandas as pd
import streamlit as st
#from modules.groq_client import clean_groq_output

from jacobs_qa.modules import data_handler as dh
from jacobs_qa.modules.provider import llm_client

# Quality categories mapping and fallback keyword map (same logic as original)
QUALITY_CATEGORIES = [
    "Chat Msg", "CI & Category", "Resolution Note & Code", "Remote session disclaimer",
    "KB Info", "Teams & Service Now integration", "Preferred contact",
    "Closing statements", "Work notes", "Short Description"
]

_keyword_map = [
    (["short description", "sd field"], "Short Description"),
    (["work note", "worknote", "work-notes"], "Work notes"),
    (["closing", "wrap up", "wrap-up"], "Closing statements"),
    (["preferred contact", "contact method"], "Preferred contact"),
    (["teams", "ms teams", "service now integration", "servicenow integration"], "Teams & Service Now integration"),
    (["kb ", "knowledge base", "article"], "KB Info"),
    (["disclaimer", "remote session"], "Remote session disclaimer"),
    (["resolution note", "resolution code", "code block"], "Resolution Note & Code"),
    (["ci ", "configuration item", "category"], "CI & Category"),
    (["chat message", "chat msg", "chat "], "Chat Msg"),
]

# ---------------- Session defaults / storage ----------------
def ensure_session_defaults():
    if "groq_key_index" not in st.session_state:
        st.session_state.groq_key_index = 0
    if "groq_key_limits" not in st.session_state:
        st.session_state.groq_key_limits = defaultdict(bool)
    if "weekly_trackers" not in st.session_state:
        st.session_state.weekly_trackers = []
    if "view" not in st.session_state:
        st.session_state.view = None
    if "tracker" not in st.session_state:
        st.session_state.tracker = pd.DataFrame()

def register_weekly_tracker(tracker_df: pd.DataFrame):
    if tracker_df is None or tracker_df.empty:
        return
    week_copy = tracker_df.copy()
    if "__score__" not in week_copy.columns:
        week_copy["__score__"] = week_copy.apply(_row_perf_score, axis=1)
    lst = st.session_state.get("weekly_trackers", [])
    lst.append(week_copy)
    st.session_state["weekly_trackers"] = lst

# ---------------- Performance helpers ----------------
def _row_perf_score(row: pd.Series) -> float:
    vals = []
    for col in ["Service Now Assessments Average Rating", "Manual Assessments Average Rating"]:
        try:
            v = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(v):
                vals.append(float(v))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")

# ---------------- Groq / summarization helpers ----------------
def _fallback_summarize(text: str) -> str:
    if not text:
        return ""
    parts = [p.strip() for p in pd.Series(text.splitlines()).tolist() if p.strip()]
    # simpler fallback: take sentences containing action words
    parts = []
    for sent in __split_sentences(text):
        if any(k in sent.lower() for k in ["should", "need to", "needs to", "improve", "ensure", "reduce", "follow", "confirm", "probe", "document", "validate", "summarize", "escalat"]):
            parts.append(sent)
    if not parts:
        # fallback to first 2-3 sentences
        parts = __split_sentences(text)[:3]
    out = " ".join(parts)
    if len(out) > 600:
        out = out[:600].rsplit(" ", 1)[0] + "..."
    return out

def __split_sentences(text: str) -> List[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def synthesize_areas_to_improve(all_comments: List[str], employee_name: str, seed_examples: List[str],score: float = None) -> str:
    """
    Summarize improvement areas based on comments.
    - If score == 10 → return 'No improvement required'.
    - Else → up to 3 short bullet points (≤15 words each).
    """
    # Handle perfect score early
    if score is not None and float(score) == 10:
        return "No improvement required"

    try:
        cleaned_bits = [c.strip() for c in (all_comments or []) if _norm(c)]
    except NameError:
        cleaned_bits = [c.strip() for c in (all_comments or []) if isinstance(c, str) and c.strip()]
    if not cleaned_bits:
        return ""
    combined = " ".join(cleaned_bits)

    user_prompt = f"""
    You are preparing 'Areas to Improve' for an agent.

    If the agent's score is 10 → reply exactly: "No improvement required".
    Otherwise:
    Write only 2–3 very short sentences (≤15 words each) describing precise improvement actions or missed steps.
    Do not include strengths, generic best practices, or disclaimers.
    Write them in plain text only — no bullets, dashes, numbering, or line breaks.
    Separate sentences with a period and a space.

    Agent: {employee_name}
    Comments: {combined}

    Respond ONLY with the sentences or "No improvement required".
    """
    return llm_client.chat_completion(user_prompt)

# ---------------- Categorization ----------------
def _fallback_classify(text: str) -> str:
    T = (text or "").lower()
    for keys, label in _keyword_map:
        if any(k in T for k in keys):
            return label
    return "Work notes"

@st.cache_data
def categorize_quality_parameter(areas_text: str) -> str:
    if not areas_text or str(areas_text).strip() == "":
        return ""
    prompt = f"""
You must classify the following feedback into EXACTLY ONE category.

CATEGORIES:
{', '.join(QUALITY_CATEGORIES)}

RULES:
- Always return exactly one category from the list above.
- If the text clearly states there is no improvement needed (e.g., "satisfactory", "done well", "nothing to improve"), return "" (empty string).
- Otherwise, always choose the closest matching category. Do NOT return blank or multiple values.
- Return your answer ONLY as JSON: {{ "category": "<one category or empty string>" }}

TEXT:
{areas_text}
    """.strip()

    result = llm_client.chat_completion(prompt)
    if result:
        try:
            parsed = json.loads(result.strip().split("\n")[0])
            cat = parsed.get("category", "").strip()
            if cat in QUALITY_CATEGORIES or cat == "":
                return cat
        except Exception:
            pass
        return _fallback_classify(areas_text)
    return _fallback_classify(areas_text)

@st.cache_data
def compute_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Quality Parameter Category" not in df.columns:
        df["Quality Parameter Category"] = ""
    for idx, row in df.iterrows():
        cur_val = str(row.get("Quality Parameter Category", "")).strip()
        if cur_val == "" or cur_val.lower() in ["nan", "none"]:
            cat = categorize_quality_parameter(row.get("Areas to Improve", ""))
            df.at[idx, "Quality Parameter Category"] = cat
    return df

# ---------------- Tracker generation (core) ----------------
def generate_tracker(manual_df: pd.DataFrame, sn_df: pd.DataFrame, month_n: int, week_n: int,
                     seed_examples: List[str]) -> pd.DataFrame:
    # follow original flow: aggregate per-employee from both sources, produce per-employee summary
    m_cols = dh.extract_people_fields(manual_df)
    s_cols = dh.extract_people_fields(sn_df)

    m_mask = dh.get_week_filter_mask(manual_df, m_cols.get("week_col"), m_cols.get("date_col"), month_n, week_n)
    s_mask = dh.get_week_filter_mask(sn_df, s_cols.get("week_col"), s_cols.get("date_col"), month_n, week_n)

    mdf = manual_df[m_mask].copy() if len(manual_df) else manual_df.copy()
    sdf = sn_df[s_mask].copy() if len(sn_df) else sn_df.copy()

    manual_names = mdf[m_cols["name_col"]].dropna().astype(str).map(dh.build_employee_key) if m_cols.get("name_col") else pd.Series([], dtype=str)
    sn_names = sdf[s_cols["name_col"]].dropna().astype(str).map(dh.build_employee_key) if s_cols.get("name_col") else pd.Series([], dtype=str)
    all_names = sorted(set(manual_names.dropna()).union(set(sn_names.dropna())))
    #print(mdf.columns.tolist())
    #print(sdf.columns.tolist())

    def per_employee(df: pd.DataFrame, cols: Dict[str, str], ticket_col: Optional[str] = None):
        d = defaultdict(lambda: {"team_leads": [], "assessors": [], "comments": [], "sn_ratings": [], "manual_grades": [], "tickets": []})
        if df is None or df.empty or not cols.get("name_col"):
            return d
        for _, row in df.iterrows():
            name = dh.build_employee_key(row.get(cols["name_col"]))
            if not name:
                continue
            tl = row.get(cols.get("tl_col")) if cols.get("tl_col") else None
            qa = row.get(cols.get("qa_col")) if cols.get("qa_col") else None
            cm = row.get(cols.get("comments_col")) if cols.get("comments_col") else None
            sr = row.get(cols.get("sn_rating_col")) if cols.get("sn_rating_col") else None
            mg = row.get(cols.get("manual_grade_col")) if cols.get("manual_grade_col") else None
            tn = row.get(ticket_col) if ticket_col else None  # <--- dynamic ticket column
            tn = row.get(ticket_col) if ticket_col else None
            if tn:
                tn = str(tn)
                # Remove prefix before colon, if present
                if ":" in tn:
                    tn = tn.split(":", 1)[1].strip()
                d[name]["tickets"].append(tn)
            #if tn: d[name]["tickets"].append(str(tn))
            if tl: d[name]["team_leads"].append(str(tl))
            if qa: d[name]["assessors"].append(str(qa))
            if cm: d[name]["comments"].append(str(cm))
            if sr is not None: d[name]["sn_ratings"].append(sr)
            if mg is not None: d[name]["manual_grades"].append(mg)
        return d

    agg_manual = per_employee(mdf, m_cols, ticket_col="Ticket Number")
    agg_sn = per_employee(sdf, s_cols, ticket_col="Record")  # sn_df ticket column

    records = []
    for name in all_names:
        team_leads = agg_manual[name]["team_leads"] + agg_sn[name]["team_leads"]
        assessors = agg_manual[name]["assessors"] + agg_sn[name]["assessors"]
        comments = agg_manual[name]["comments"] + agg_sn[name]["comments"]
        sn_ratings = pd.to_numeric(pd.Series(agg_sn[name]["sn_ratings"]), errors="coerce").dropna().tolist()
        manual_grades_series = pd.Series(agg_manual[name]["manual_grades"]) if agg_manual[name]["manual_grades"] else pd.Series([], dtype=float)
        tickets = agg_manual[name]["tickets"] + agg_sn[name]["tickets"]
        ticket_str = ", ".join(sorted(set(tickets))) if tickets else ""

        # team lead final
        team_lead_final = None
        if team_leads:
            c = Counter([dh._norm(tl) for tl in team_leads if dh._norm(tl)])
            team_lead_final = c.most_common(1)[0][0] if c else None
        assessors_final = ", ".join(sorted(set([dh._norm(a) for a in assessors if dh._norm(a)]))) or None
        service_now_count = len(sn_ratings)
        service_now_avg = round(float(np.mean(sn_ratings)), 2) if sn_ratings else None
        manual_count = len(agg_manual[name]["manual_grades"]) if agg_manual[name]["manual_grades"] else 0
        manual_avg = dh.average_of_grades(manual_grades_series) if manual_count else None

        # PER-EMPLOYEE summary (critical fix)
        areas_text = synthesize_areas_to_improve(comments, name, seed_examples)
        rec = {
            "Employee Name": name,
            "Emp ID": "",
            "Organization": "",
            "Team Lead": team_lead_final or "",
            "Status": "",
            "Quality Assessors": assessors_final or "",
            "Week": f"{dt.date(1900, month_n, 1).strftime('%b')} - Week {week_n}",
            "Service Now Assessments Count": service_now_count if service_now_count else "",
            "Service Now Assessments Average Rating": service_now_avg if service_now_avg is not None else "",
            "Manual Assessments Count": manual_count if manual_count else "",
            "Manual Assessments Average Rating": manual_avg if manual_avg is not None else "",
            "Areas to Improve": areas_text,
            "Ticket Number": ticket_str,
        }
        records.append(rec)

    tracker = pd.DataFrame.from_records(records)
    col_order = [
        "Ticket Number", "Employee Name", "Emp ID", "Organization", "Team Lead", "Status", "Quality Assessors", "Week",
        "Service Now Assessments Count", "Service Now Assessments Average Rating",
        "Manual Assessments Count", "Manual Assessments Average Rating", "Areas to Improve",
    ]
    tracker = tracker.reindex(columns=col_order)
    return tracker

