# ---------------------------------------------------------------

import os
import io
import re
import math
import datetime as dt
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import streamlit as st

# ========================= UI CONSTANTS ========================= #
APP_TITLE = "Weekly Coaching Assessment Tracker"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
AREAS_TO_IMPROVE_EXAMPLES = [
    "Probe for full issue context before troubleshooting.",
    "Summarize resolution and confirm next steps with the customer.",
    "Reduce hold time by using knowledge base bookmarks.",
    "Follow escalation matrix within SLA when blockers occur.",
    "Ensure ticket notes are complete: symptoms, root cause, and fix.",
]
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ========================= UTILS ========================= #
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if isinstance(s, (int, float)) and not pd.isna(s):
        return str(s).strip()
    if isinstance(s, str):
        return s.strip()
    return None

def pick_first_available_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    def col_to_text(col) -> str:
        if isinstance(col, (list, tuple)):
            col = " ".join(str(x) for x in col)
        return str(col)
    def norm(s: str) -> str:
        return s.lower().strip()
    def norm_simple(s: str) -> str:
        return re.sub(r"[^a-z]", "", norm(s))
    lower_map, simple_map = {}, {}
    for original in df.columns:
        t = col_to_text(original)
        lower_map.setdefault(norm(t), original)
        simple_map.setdefault(norm_simple(t), original)
    for cand in candidates:
        key = norm(str(cand))
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = norm_simple(str(cand))
        if key in simple_map:
            return simple_map[key]
    return None

def week_of_month(date: dt.date) -> int:
    first_day = date.replace(day=1)
    adjusted_dom = date.day + (first_day.weekday())
    return int(math.ceil(adjusted_dom / 7.0))

def to_date(s) -> Optional[dt.date]:
    if pd.isna(s):
        return None
    if isinstance(s, dt.datetime):
        return s.date()
    if isinstance(s, dt.date):
        return s
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None

def try_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def letter_grade_to_score(val: str) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    m = {
        "A+": 5.0, "A": 5.0, "A-": 4.7,
        "B+": 4.3, "B": 4.0, "B-": 3.7,
        "C+": 3.3, "C": 3.0, "C-": 2.7,
        "D+": 2.3, "D": 2.0, "D-": 1.7,
        "E": 1.0, "F": 0.0,
    }
    s = str(val).strip().upper()
    return m.get(s)

def average_of_grades(series: pd.Series) -> Optional[float]:
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        vals = s_num.dropna().tolist()
        return round(float(np.mean(vals)), 2) if vals else None
    mapped = [letter_grade_to_score(v) for v in series]
    mapped = [v for v in mapped if v is not None]
    return round(float(np.mean(mapped)), 2) if mapped else None

# ========================= TEXT CLEANING ========================= #
YESNO_PATTERNS = [
    r":\s*-?\s*Yes\b", r":\s*-?\s*No\b", r":\s*-?\s*Yes/No\b",
    r":\s*-?\s*Resolved\b", r":\s*-?\s*Escalated\b", r":\s*-?\s*Closed\b",
    r":\s*Accurate\b",
]
YESNO_SUFFIX = re.compile("|".join(YESNO_PATTERNS), re.IGNORECASE)
BULLET_LINE = re.compile(r"^\s*(?:\d+[\.\)]|[-*•·])\s+")
CHECKLIST_KEYS = [
    "checked for previous tickets","preference contact method","preferred contact method","contact type",
    "ms teams chat integration","ms teams integration","short description","configuration item","category",
    "computer name documented","error screenshot attached","ticket state","work note documentation",
    "additional assistance","call opening followed","user validation procedure","display empathy & assurance",
    "call holding procedure","professional and positive tone","sufficient probing","proper 3 reminder process",
    "bomgar chat messages attached","computer name documented/ bomgar chat messages attached",
]

def _is_checklist_line(line: str) -> bool:
    L = line.strip().strip('"').lower()
    if not L:
        return True
    if L.startswith("observations") or "observation/areas for improvement" in L:
        return True
    if any(k in L for k in CHECKLIST_KEYS):
        return True
    if YESNO_SUFFIX.search(line):
        return True
    if BULLET_LINE.match(line):
        return True
    return False

def clean_comment(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    raw = raw.replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    kept: List[str] = []
    for ln in lines:
        if _is_checklist_line(ln):
            continue
        if len(ln.strip(" -•*·\t\"'")) < 3:
            continue
        kept.append(ln.strip('"').strip())
    text = " ".join(kept)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_output(text: str) -> str:
    if not text:
        return ""
    prefixes = [
        "Here is a concise summary of Areas to Improve:",
        "Here is a summary of Areas to Improve:",
        "Based on the raw comments,",
        "Here are the areas to improve:",
        "Here are the Areas to Improve:",
        "Here is the summary of Areas to Improve:"
    ]
    for p in prefixes:
        text = text.replace(p, "")
    return text.strip()

load_dotenv(dotenv_path="/teamspace/studios/this_studio/jserviceops/.env")

# ========================= OLLAMA LLM (LOCAL, NO API KEYS) ========================= #
import os
import re
import streamlit as st
from collections import defaultdict
from typing import Dict, List
import ollama  # pip install ollama

SYSTEM_SUMMARY = (
    "You are a QA coaching assistant. Summarize ONLY improvement-relevant text "
    "in 2–3 concise sentences (no bullets). Do not mention strengths, generic advice, "
    "or checklists. Be direct and specific."
)

# ------------------------------------------------------------------ #
# Since Ollama is local, no need for API key management
# ------------------------------------------------------------------ #
def is_rate_limit_error(_):
    return False  # no rate limits in local ollama

def get_next_groq_client():
    """Dummy placeholder to match old structure"""
    return True  # always available (Ollama runs locally)

# ------------------------------------------------------------------ #
# Core Ollama completion
# ------------------------------------------------------------------ #
def _groq_complete(messages: List[Dict[str, str]], max_tokens: int = 220) -> str:
    try:
        response = ollama.chat(
            model="mistral:latest",
            messages=messages,
            options={"temperature": 0.2, "num_predict": max_tokens}
        )
        out = (response.get("message", {}).get("content", "") or "").strip()
        return clean_output(out)
    except Exception as e:
        print(f"(Ollama) completion error: {e}")
        return ""

# ------------------------------------------------------------------ #
# Fallback summarizer (rule-based)
# ------------------------------------------------------------------ #
def _fallback_summarize(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 0]
    prioritized = [p for p in parts if any(k in p.lower() for k in [
        "should", "need to", "needs to", "improve", "ensure", "reduce", "increase",
        "follow", "confirm", "probe", "document", "validate", "summarize", "escalat"
    ])]
    chosen = prioritized[:3] if prioritized else parts[:3]
    summary = " ".join(chosen)
    if len(summary) > 600:
        summary = summary[:600].rsplit(" ", 1)[0] + "..."
    return summary

# ------------------------------------------------------------------ #
# Output cleaner
# ------------------------------------------------------------------ #
def clean_groq_output(text: str) -> str:
    return clean_output(text)

def clean_output(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ------------------------------------------------------------------ #
# Main function for summarization
# ------------------------------------------------------------------ #
def groq_chat_completion(prompt: str, system: str = None) -> str:
    sys_msg = system or (
        "You are a QA coaching assistant.\n"
        "From the raw comments, output only improvement issues.\n"
        "- Each issue: one short phrase or one short sentence.\n"
        "- Write each issue on a new line with no numbers, no bullets.\n"
        "- Merge duplicates; avoid filler and repetition.\n"
        "- Do not output strengths, generic advice, or checklists.\n"
        '- If no improvements, respond exactly: "Analyst performed well. No improvement as of now."'
    )


    try:
        response = ollama.chat(
            model="mistral:latest",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2}
        )

        if response and response.get("message"):
            raw = response["message"].get("content", "")
            return clean_groq_output(raw).strip()
        return ""
    except Exception as e:
        print(f"(Ollama) chat error: {e}")
        return _fallback_summarize(prompt)  # fallback if model fails


def synthesize_areas_to_improve(all_comments: List[str], employee_name: str, seed_examples: List[str]) -> str:
    try:
        cleaned_bits = [c.strip() for c in (all_comments or []) if _norm(c)]
    except NameError:
        cleaned_bits = [c.strip() for c in (all_comments or []) if isinstance(c, str) and c.strip()]
    if not cleaned_bits:
        return ""
    combined = " ".join(cleaned_bits)
    user_prompt = (
        f"Raw observations (cleaned): {combined}\n"
        + "Write only 2–3 sentences focusing on specific improvement actions and missed steps. "
          "Avoid strengths, disclaimers, or generic best practices. Do not include personally identifiable customer data."
    )
    return groq_chat_completion(user_prompt)

# ========================= EXTRACTION LAYER ========================= #
COLUMN_CANDIDATES = {
    "employee_name": ["Employee Name", "Analyst Name", "Trainee", "Analyst", "Name"],
    "trainee_name": ["Trainee", "Trainee Name", "Analyst Name", "Employee Name", "Analyst"],
    "analyst_name": ["Analyst Name", "Employee Name", "Analyst"],
    "team_lead": ["Team Lead", "Team Leader", "TL", "Manager", "Reporting Manager"],
    "manager": ["Manager", "Reporting Manager", "Supervisor"],
    "quality_assessor": ["Assessor", "Quality Assessor", "QA", "Quality Analyst", "Analyst Name", "Coach"],
    "week": ["Week", "Week Number", "Week_of_Month", "WOM"],
    "date": ["Date", "Assessed On", "Created", "Assessment Date"],
    "sn_rating": ["Trainee rating", "Trainee Rating", "Rating"],
    "manual_grade": ["Grade", "Score", "Marks"],
    "comments": ["Comments", "Notes", "QA Comments", "Reviewer Comments"],
}

def build_employee_key(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return re.sub(r"\s+", " ", str(name).strip()).title()

def extract_people_fields(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    out["name_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["employee_name"]) or \
                      pick_first_available_column(df, COLUMN_CANDIDATES["trainee_name"]) or \
                      pick_first_available_column(df, COLUMN_CANDIDATES["analyst_name"])
    out["tl_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["team_lead"]) or \
                    pick_first_available_column(df, COLUMN_CANDIDATES["manager"])
    out["qa_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["quality_assessor"])
    out["week_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["week"])
    out["date_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["date"])
    out["sn_rating_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["sn_rating"])
    out["manual_grade_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["manual_grade"])
    out["comments_col"] = pick_first_available_column(df, COLUMN_CANDIDATES["comments"])
    return out

def get_week_filter_mask(df: pd.DataFrame, week_col: Optional[str], date_col: Optional[str], month_n: int, week_n: int) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([False] * 0)
    mask = pd.Series([True] * len(df))
    if week_col and week_col in df.columns:
        wvals = df[week_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
        mask &= (wvals == week_n)
    elif date_col and date_col in df.columns:
        dates = df[date_col].apply(to_date)
        wom = dates.apply(lambda d: week_of_month(d) if d else np.nan)
        months = dates.apply(lambda d: d.month if d else np.nan)
        mask &= (months == month_n) & (wom == week_n)
    else:
        mask &= True
    return mask.fillna(False)

# ========================= CORE PIPELINE ========================= #
def generate_tracker(manual_df: pd.DataFrame, sn_df: pd.DataFrame, month_n: int, week_n: int,
                     seed_examples: List[str]) -> pd.DataFrame:
    m_cols = extract_people_fields(manual_df)
    s_cols = extract_people_fields(sn_df)
    m_mask = get_week_filter_mask(manual_df, m_cols["week_col"], m_cols["date_col"], month_n, week_n)
    s_mask = get_week_filter_mask(sn_df, s_cols["week_col"], s_cols["date_col"], month_n, week_n)
    mdf = manual_df[m_mask].copy() if len(manual_df) else manual_df.copy()
    sdf = sn_df[s_mask].copy() if len(sn_df) else sn_df.copy()
    manual_names = mdf[m_cols["name_col"]].dropna().astype(str).map(build_employee_key) if m_cols["name_col"] else pd.Series([], dtype=str)
    sn_names = sdf[s_cols["name_col"]].dropna().astype(str).map(build_employee_key) if s_cols["name_col"] else pd.Series([], dtype=str)
    all_names = sorted(set(manual_names.dropna()).union(set(sn_names.dropna())))
    def per_employee(df: pd.DataFrame, cols: Dict[str, str]) -> Dict[str, Dict[str, list]]:
        d = defaultdict(lambda: {"team_leads": [], "assessors": [], "comments": [], "sn_ratings": [], "manual_grades": []})
        if df is None or df.empty or not cols.get("name_col"):
            return d
        for _, row in df.iterrows():
            name = build_employee_key(row.get(cols["name_col"]))
            if not name:
                continue
            tl = row.get(cols["tl_col"]) if cols.get("tl_col") else None
            qa = row.get(cols["qa_col"]) if cols.get("qa_col") else None
            cm = row.get(cols["comments_col"]) if cols.get("comments_col") else None
            sr = row.get(cols["sn_rating_col"]) if cols.get("sn_rating_col") else None
            mg = row.get(cols["manual_grade_col"]) if cols.get("manual_grade_col") else None
            if tl: d[name]["team_leads"].append(str(tl))
            if qa: d[name]["assessors"].append(str(qa))
            if cm: d[name]["comments"].append(str(cm))
            if sr is not None: d[name]["sn_ratings"].append(sr)
            if mg is not None: d[name]["manual_grades"].append(mg)
        return d
    agg_manual = per_employee(mdf, m_cols)
    agg_sn = per_employee(sdf, s_cols)
    records = []
    for name in all_names:
        team_leads = agg_manual[name]["team_leads"] + agg_sn[name]["team_leads"]
        assessors = agg_manual[name]["assessors"] + agg_sn[name]["assessors"]
        comments = agg_manual[name]["comments"] + agg_sn[name]["comments"]
        sn_ratings = pd.to_numeric(pd.Series(agg_sn[name]["sn_ratings"]), errors="coerce").dropna().tolist()
        manual_grades_series = pd.Series(agg_manual[name]["manual_grades"])
        team_lead_final = None
        if team_leads:
            c = Counter([_norm(tl) for tl in team_leads if _norm(tl)])
            team_lead_final = c.most_common(1)[0][0] if c else None
        assessors_final = ", ".join(sorted(set([_norm(a) for a in assessors if _norm(a)]))) or None
        service_now_count = len(sn_ratings)
        service_now_avg = round(float(np.mean(sn_ratings)), 2) if sn_ratings else None
        manual_count = len(agg_manual[name]["manual_grades"]) if agg_manual[name]["manual_grades"] else 0
        manual_avg = average_of_grades(manual_grades_series) if manual_count else None
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
        }
        records.append(rec)
    tracker = pd.DataFrame.from_records(records)
    col_order = [
        "Employee Name", "Emp ID", "Organization", "Team Lead", "Status", "Quality Assessors", "Week",
        "Service Now Assessments Count", "Service Now Assessments Average Rating",
        "Manual Assessments Count", "Manual Assessments Average Rating", "Areas to Improve",
    ]
    tracker = tracker.reindex(columns=col_order)
    return tracker

# ========================= CATEGORY CLASSIFIER ========================= #
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
    system = "You are a strict classifier for ticket QA."
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
    result = groq_chat_completion(prompt, system=system)
    if result:
        try:
            import json
            parsed = json.loads(result.strip().split("\n")[0])
            cat = parsed.get("category", "").strip()
            if cat in QUALITY_CATEGORIES or cat == "":
                return cat
        except Exception:
            pass
        return _fallback_classify(areas_text)
    return _fallback_classify(areas_text)

# ========================= PERFORMANCE HELPERS (NEW) ========================= #
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

@st.cache_data
def compute_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Quality Parameter Category" not in df.columns:
        df["Quality Parameter Category"] = ""
    for idx, row in df.iterrows():
        cur_val = str(row.get("Quality Parameter Category", "")).strip()
        if cur_val == "" or cur_val.lower() in ["nan", "none"]:
            # only compute for blanks -- this triggers Groq only when necessary
            cat = categorize_quality_parameter(row.get("Areas to Improve", ""))
            df.at[idx, "Quality Parameter Category"] = cat
    return df

def _build_month_tracker(manual_df: pd.DataFrame, sn_df: pd.DataFrame, month_n: int, seed_examples: List[str]) -> pd.DataFrame:
    frames = []
    for wk in [1, 2, 3, 4, 5]:
        t = generate_tracker(manual_df, sn_df, month_n, wk, seed_examples)
        if not t.empty:
            t["Quality Parameter Category"] = t["Areas to Improve"].apply(categorize_quality_parameter).astype(str).str.strip()
            frames.append(t)
    if not frames:
        return pd.DataFrame()
    month_tracker = pd.concat(frames, ignore_index=True)
    if not month_tracker.empty:
        month_tracker["__score__"] = month_tracker.apply(_row_perf_score, axis=1)
    return month_tracker

# ========================= UI helpers ========================= #
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
    # keep a copy
    week_copy = tracker_df.copy()
    # compute score for this weekly df
    if "__score__" not in week_copy.columns:
        week_copy["__score__"] = week_copy.apply(_row_perf_score, axis=1)
    # store list
    lst = st.session_state.get("weekly_trackers", [])
    lst.append(week_copy)
    st.session_state["weekly_trackers"] = lst

def render_treemap(tracker_df: pd.DataFrame):
    import plotly.express as px
    _tmp = tracker_df.copy()
    _tmp["Quality Parameter Category"] = _tmp.get("Quality Parameter Category", "").astype(str).str.strip()
    _tmp = _tmp[_tmp["Quality Parameter Category"].notna()]
    _tmp = _tmp[_tmp["Quality Parameter Category"].str.strip().ne("")]
    if _tmp.empty:
        st.info("No missing quality parameters to visualize for this week.")
        return
    category_counts = (
        _tmp["Quality Parameter Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Count")
    )
    category_counts["Label"] = category_counts.apply(lambda r: f"{r['Category']}, {int(r['Count'])}", axis=1)
    fig = px.treemap(category_counts, path=["Label"], values="Count", title="Missing Quality Parameters (Auto-Categorized)")
    fig.update_traces(hovertemplate="<b>%{label}</b><extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

def show_quick_views():
    tracker = st.session_state.get("tracker", pd.DataFrame())
    if tracker is None or tracker.empty:
        st.info("No tracker available. Generate a tracker or upload a saved tracker.")
        return

    # Ensure categories exist, but compute only if blanks (compute_categories is cached)
    if "Quality Parameter Category" not in tracker.columns or tracker["Quality Parameter Category"].isnull().any() or (tracker["Quality Parameter Category"].astype(str).str.strip() == "").any():
        # This will only call groq for rows missing category (per compute_categories implementation)
        tracker = compute_categories(tracker)
        st.session_state["tracker"] = tracker  # persist filled categories

    # Category filter view
    st.markdown("### 📂 View by Quality Parameter Category")
    cats = sorted([c for c in tracker["Quality Parameter Category"].dropna().unique() if str(c).strip() != ""])
    if cats:
        sel_cat = st.selectbox("Select a Quality Parameter Category", ["(all)"] + cats, key="cat_select")
        if sel_cat and sel_cat != "(all)":
            cat_df = tracker[tracker["Quality Parameter Category"] == sel_cat].copy()
            st.subheader(f"🔎 Records for category: {sel_cat}")
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
            st.subheader(f"⚠️ Analysts with Missing Criteria in {sel_cat}")
            missing_df = cat_df[["Employee Name", "Team Lead", "Quality Assessors", "Areas to Improve"]]
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.info("✅ No analysts missing this criteria.")
        else:
            st.dataframe(tracker, use_container_width=True, hide_index=True)
    else:
        st.info("No categorized records found.")
        st.dataframe(tracker, use_container_width=True, hide_index=True)

    # Quick Views buttons
    st.markdown("### Quick Views")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🏆 View Top Performer (Selected Week)", key="btn_top_week"):
            st.session_state["view"] = "top_week"
    with col_b:
        if st.button("🏅 View Top Performer (Selected Month)", key="btn_top_month"):
            st.session_state["view"] = "top_month"
    with col_c:
        if st.button("⚠️ Agents with Missing Criteria (Improvements Needed)", key="btn_improve"):
            st.session_state["view"] = "needs_improve"

    # Render selected view
    view = st.session_state.get("view")
    if view == "top_week":
        wk_df = st.session_state["tracker"].copy()
        if "__score__" not in wk_df.columns:
            wk_df["__score__"] = wk_df.apply(_row_perf_score, axis=1)
        wk_df["_rank"] = wk_df["__score__"].rank(method="min", ascending=False)
        top_score = wk_df["__score__"].max()
        top_week = wk_df[wk_df["__score__"] == top_score].sort_values(["__score__", "Employee Name"], ascending=[False, True])
        st.subheader("🏆 Top Performer — Selected Week")
        st.dataframe(top_week.drop(columns=["_rank"], errors="ignore"), use_container_width=True, hide_index=True)
        if st.button("◀ Back", key="back_from_week"):
            st.session_state["view"] = None

    elif view == "top_month":
        # Build month-to-date tracker from available weekly trackers if present
        month_tracker = None
        # If we have weekly_trackers list built in session, use it
        wk_list = st.session_state.get("weekly_trackers", [])
        if wk_list:
            with st.spinner("Building month-to-date tracker from available weeks…"):
                try:
                    month_tracker = pd.concat(wk_list, ignore_index=True)
                    if "__score__" not in month_tracker.columns:
                        month_tracker["__score__"] = month_tracker.apply(_row_perf_score, axis=1)
                    # aggregate per employee
                    month_tracker = month_tracker.groupby("Employee Name", as_index=False).agg({"__score__": "sum"})
                except Exception as e:
                    st.error(f"Could not build month-to-date tracker from weekly trackers: {e}")
                    month_tracker = None
        elif "uploaded_tracker_df" in st.session_state:
            with st.spinner("Aggregating from uploaded coaching tracker…"):
                month_tracker = st.session_state["uploaded_tracker_df"].copy()
                if "__score__" not in month_tracker.columns:
                    month_tracker["__score__"] = month_tracker.apply(_row_perf_score, axis=1)
                month_tracker = month_tracker.groupby("Employee Name", as_index=False).agg({"__score__": "sum"})
        else:
            # fallback: use current tracker as the only available week
            with st.spinner("Using current weekly tracker as month-to-date (only available week)…"):
                month_tracker = st.session_state["tracker"].copy()
                if "__score__" not in month_tracker.columns:
                    month_tracker["__score__"] = month_tracker.apply(_row_perf_score, axis=1)
                month_tracker = month_tracker.groupby("Employee Name", as_index=False).agg({"__score__": "sum"})

        if month_tracker is not None and not month_tracker.empty:
            top_score = month_tracker["__score__"].max()
            top_month = month_tracker[month_tracker["__score__"] == top_score].sort_values(["__score__", "Employee Name"], ascending=[False, True])
            st.subheader("🏅 Top Performer — Month-to-Date (Based on Available Weeks)")
            st.dataframe(top_month.drop(columns=["__score__"], errors="ignore"), use_container_width=True, hide_index=True)
        else:
            st.info("No data available to compute month-to-date performer. Generate at least one weekly tracker or upload a saved tracker.")
        if st.button("◀ Back", key="back_from_month"):
            st.session_state["view"] = None

    elif view == "needs_improve":
        need_improve = st.session_state["tracker"][
            st.session_state["tracker"]["Quality Parameter Category"].notna()
            & st.session_state["tracker"]["Quality Parameter Category"].astype(str).str.strip().ne("")
            & st.session_state["tracker"]["Quality Parameter Category"].astype(str).str.lower().ne("nan")
        ].copy()
        st.subheader("⚠️ Agents with Improvement Criteria Identified")
        st.dataframe(need_improve, use_container_width=True, hide_index=True)
        if st.button("◀ Back", key="back_from_needs"):
            st.session_state["view"] = None

# ========================= STREAMLIT APP ========================= #
def main():
    ensure_session_defaults()

    if st.button("Back to Homepage", key="heat_back_homepage"):
        st.session_state["current_app"] = "Homepage"
        st.rerun()

    st.title(APP_TITLE)
    st.caption(
        "Upload weekly report → pick Month/Week → Generate tracker → Download as Excel. "
        "Optional: set GROQ_API_KEY in environment for better 'Areas to Improve'. "
        "Or upload an already saved tracker to review."
    )

    with st.sidebar:
        st.header("Inputs")
        weekly_file = st.file_uploader(
            "Upload Weekly Coaching Assessment Report (.xlsx)", type=["xlsx"], accept_multiple_files=False, key="sidebar_weekly"
        )
        st.write(":blue[Sheets required: 'Manual Assessments Data' and 'ServiceNow Coaching Assessment']")
        sample_tracker = st.file_uploader(
            "(Optional) Upload historical Coaching Assessment Tracker for example phrases (.xlsx)",
            type=["xlsx"],
            accept_multiple_files=False,
            help="Used only to seed the style of 'Areas to Improve' (no PII stored).",
            key="sidebar_sample"
        )
        saved_tracker_file = st.file_uploader(
            "📂 (Optional) Upload an already saved Coaching Assessment Tracker (.xlsx)",
            type=["xlsx"],
            accept_multiple_files=False,
            help="Use this if you already have a generated tracker and want to view/analyze it.",
            key="sidebar_saved"
        )
        month_name_to_num = {m: i for i, m in enumerate(
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
        month = st.selectbox("Month", list(month_name_to_num.keys()), index=dt.date.today().month - 1, key="sidebar_month")
        week = st.selectbox("Week of month", [1, 2, 3, 4, 5],
                            index=min(4, week_of_month(dt.date.today()) - 1), key="sidebar_week")
        do_generate = st.button("Generate Tracker", type="primary", key="sidebar_generate")

        # Add API status button
        if st.button("Show API Key Status"):
            status_text = "API Key Status:\n"
            for i, key in enumerate(GROQ_API_KEYS):
                hit_limit = st.session_state.groq_key_limits.get(i, False)
                status = "Active" if not hit_limit and i == st.session_state.groq_key_index else "Available" if not hit_limit else "Rate Limited"
                status_text += f"Key {i + 1}: {status}\n"

            # Show which key is currently active
            active_key = st.session_state.groq_key_index + 1
            status_text += f"\nCurrent Active Key: {active_key}"

            st.sidebar.info(status_text)

        # Add a button to reset all API key limits
        if st.button("Reset All API Key Limits"):
            for i in range(len(GROQ_API_KEYS)):
                st.session_state.groq_key_limits[i] = False
            st.session_state.groq_key_index = 0
            st.sidebar.success("All API key limits have been reset!")
    st.warning("Emp ID, Organization, and Status are intentionally left blank. Please enrich later from HRIS.")

    # ----- Handle uploaded saved tracker FIRST (if present) -----
    if saved_tracker_file:
        try:
            tracker = pd.read_excel(saved_tracker_file, sheet_name="Coaching Tracker").drop(columns=["Unnamed: 0"], errors="ignore")
            # persist uploaded tracker (useful for month aggregation)
            # compute categories only if missing (compute_categories will call groq only for missing rows)
            if "Quality Parameter Category" not in tracker.columns or tracker["Quality Parameter Category"].isnull().any() or (tracker["Quality Parameter Category"].astype(str).str.strip() == "").any():
                tracker = compute_categories(tracker)
            # compute __score__
            if "__score__" not in tracker.columns:
                tracker["__score__"] = tracker.apply(_row_perf_score, axis=1)
            st.session_state["tracker"] = tracker
            st.session_state["uploaded_tracker_df"] = tracker.copy()
            st.success(f"Loaded saved tracker. Rows: {len(tracker)}")
            st.dataframe(tracker, use_container_width=True, hide_index=True)
            # show treemap and quick views (common)
            render_treemap(tracker)
            show_quick_views()
            return
        except Exception as e:
            st.error("Could not read uploaded tracker.")
            st.exception(e)
            st.stop()

    # ----- Handle weekly report generation -----
    if weekly_file:
        try:
            xls = pd.ExcelFile(weekly_file)
            def find_sheet(target: str) -> Optional[str]:
                for s in xls.sheet_names:
                    if target.lower() in s.lower():
                        return s
                return None
            manual_sheet = find_sheet("Manual Assessments Data") or find_sheet("Manual") or find_sheet("Assessments")
            sn_sheet = find_sheet("ServiceNow Coaching Assessment") or find_sheet("ServiceNow") or find_sheet("Coaching")
            if not manual_sheet or not sn_sheet:
                st.error("Could not find required sheets. Ensure workbook has 'Manual Assessments Data' and 'ServiceNow Coaching Assessment'.")
                st.stop()
            manual_df = pd.read_excel(xls, manual_sheet).drop(columns=["Unnamed: 0"], errors="ignore")
            sn_df = pd.read_excel(xls, sn_sheet).drop(columns=["Unnamed: 0"], errors="ignore")
        except Exception as e:
            st.exception(e)
            st.stop()

        seed_examples = AREAS_TO_IMPROVE_EXAMPLES.copy()
        if sample_tracker:
            try:
                st.caption("Using uploaded tracker to seed Areas to Improve examples.")
                df_seed = pd.read_excel(sample_tracker)
                col = pick_first_available_column(df_seed, ["Areas to Improve", "Areas", "Improvements"])
                if col and not df_seed.empty:
                    vals = [str(v) for v in df_seed[col].dropna().astype(str).tolist()][:20]
                    cleaned = []
                    for v in vals:
                        for line in str(v).splitlines():
                            line = line.strip("- • ").strip()
                            if 10 <= len(line) <= 160:
                                cleaned.append(line)
                    if cleaned:
                        seed_examples = cleaned[:20]
            except Exception:
                pass

        month_n = month_name_to_num[month]
        week_n = int(week)

        if do_generate:
            with st.spinner("Generating tracker…"):
                tracker = generate_tracker(manual_df, sn_df, month_n, week_n, seed_examples)
            # compute categories for missing rows only
            if "Quality Parameter Category" not in tracker.columns or tracker["Quality Parameter Category"].isnull().any() or (tracker["Quality Parameter Category"].astype(str).str.strip() == "").any():
                tracker = compute_categories(tracker)
            # compute scores and persist
            if "__score__" not in tracker.columns:
                tracker["__score__"] = tracker.apply(_row_perf_score, axis=1)
            st.session_state["tracker"] = tracker
            # store raw data and weekly tracker for month-to-date combining
            st.session_state["manual_df_raw"] = manual_df.copy()
            st.session_state["sn_df_raw"] = sn_df.copy()
            register_weekly_tracker(tracker)  # append to weekly list (so month aggregation uses available weeks)
            st.success(f"Tracker generated for {month} – Week {week_n}. Rows: {len(tracker)}")
            st.dataframe(tracker, use_container_width=True, hide_index=True)

            # save for download
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                tracker.to_excel(writer, index=False, sheet_name="Coaching Tracker")
            st.download_button(
                label="Download Coaching Assessment Tracker (Excel)",
                data=out.getvalue(),
                file_name=f"Coaching_Assessment_Tracker_{month}_W{week_n}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # visuals + quick views
            render_treemap(tracker)
            show_quick_views()
            return

    # If no files/actions: if any previous tracker in session show it
    current_tracker = st.session_state.get("tracker")
    if isinstance(current_tracker, pd.DataFrame) and not current_tracker.empty:
        st.info("Showing last generated/loaded tracker from this session.")
        st.dataframe(current_tracker, use_container_width=True, hide_index=True)
        render_treemap(current_tracker)
        show_quick_views()
        return

    st.info("Upload a Weekly Coaching Report or an already saved Tracker to begin.")
    return

if __name__ == "__main__":
    main()
