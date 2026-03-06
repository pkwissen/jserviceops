# feedback_app.py
import os
import io
import sys
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Feedback Form", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project root (jserviceops2) is on path so we can import sharepoint client
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from ai_tool_quality.modules.sharepoint_client import (
        download_feedback_records,
        upload_feedback_records,
    )
except Exception:
    download_feedback_records = None
    upload_feedback_records = None


# -----------------------------
# Read helper
# -----------------------------
# Note: local file reads/writes removed — this app uses SharePoint-only storage.


def load_feedback_df_from_sharepoint():
    """Try downloading feedback records from SharePoint. Return DataFrame or empty DF."""
    # Strict SharePoint-only: do not write a local cache. If SharePoint client
    # isn't available or download fails, return None so caller can show an error.
    if download_feedback_records is None:
        return None
    try:
        stream = download_feedback_records()
    except Exception:
        return None
    if not stream:
        return None
    try:
        try:
            stream.seek(0)
        except Exception:
            pass
        return pd.read_excel(io.BytesIO(stream.getbuffer()), dtype=str, engine="openpyxl")
    except Exception:
        return None


def save_feedback_df_to_sharepoint(df: pd.DataFrame):
    """Write DataFrame to SharePoint (and local cache). Returns (uploaded: bool, error_or_None)."""
    # SharePoint-only save: do not persist locally. Return False with error
    # if upload is unavailable or fails.
    if upload_feedback_records is None:
        return False, RuntimeError("SharePoint upload function not available")
    out = io.BytesIO()
    try:
        # ensure BytesIO is clean
        try:
            out.truncate(0)
            out.seek(0)
        except Exception:
            pass
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        try:
            out.seek(0)
        except Exception:
            pass
        # upload to SharePoint (sharepoint_client will handle stream reads)
        upload_feedback_records(out)
        return True, None
    except Exception as e:
        return False, e

# -----------------------------
# Read query parameters
# -----------------------------
params = st.query_params

def qp(k):
    v = params.get(k, "")
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""

ticket = qp("ticket").strip()
analyst = qp("analyst").strip()
response = qp("click") or qp("response")
response = (response or "").strip()
token = (qp("token") or "").strip()   # <--- now optional, not required

# -----------------------------
# HEADER
# -----------------------------
st.title("📋 Feedback Form")
st.write("Please confirm details and submit your comments.")
st.write(f"**Analyst:** {analyst or '-'}")
st.write(f"**Ticket:** {ticket or '-'}")
st.write(f"**You clicked:** {response or '-'}")
st.markdown("---")


# -----------------------------
# LOAD RECORDS (try SharePoint first, then local fallback)
# -----------------------------
df = load_feedback_df_from_sharepoint()
if df is None or df.empty:
    st.error("Could not load feedback records from SharePoint. Please check SharePoint connectivity and try again.")
    st.stop()

df.rename(columns=lambda x: x.strip(), inplace=True)

if "Token_Used" not in df.columns:
    df["Token_Used"] = ""
if "Token" not in df.columns:
    df["Token"] = ""

# -----------------------------
# FIND ROW BY TICKET NUMBER (token NOT required anymore)
# -----------------------------
if "Ticket Number" not in df.columns:
    st.error("Column 'Ticket Number' missing in records.")
    st.stop()

mask_ticket = df["Ticket Number"].astype(str).str.strip() == ticket

if not mask_ticket.any():
    st.error("Ticket not found in records.")
    st.stop()

row_ix = df.index[mask_ticket][0]


# -----------------------------
# CHECK if already responded
# -----------------------------
token_used = str(df.at[row_ix, "Token_Used"] or "").strip().lower()
existing_response = str(df.at[row_ix, "Response"] or "").strip()

already_used = (str(token_used).strip().lower() == "yes") or (str(existing_response).strip().lower() not in ("", "nan", "none"))

if already_used:
    st.warning("⚠️ You have already submitted feedback for this ticket.")
    st.stop()


# -----------------------------
# FEEDBACK FORM
# -----------------------------
comments = st.text_area(
    "Feedback comments (required for 'No')",
    height=160,
    max_chars=1000,
    placeholder="Enter your feedback here..."
)

if st.button("Submit Feedback"):
    if response.lower() == "no" and not comments.strip():
        st.error("Please provide feedback comments.")
        st.stop()

    df.at[row_ix, "Response"] = response
    df.at[row_ix, "Response_Comments"] = comments
    df.at[row_ix, "Feedback_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.at[row_ix, "Token_Used"] = "Yes"

    uploaded, err = save_feedback_df_to_sharepoint(df)
    if uploaded:
        st.success("✅ Thank you — your feedback has been recorded.")
    else:
        st.error(f"Failed to upload feedback: {err}")
