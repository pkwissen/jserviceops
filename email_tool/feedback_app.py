# feedback_app.py
import os
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Feedback Form", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "email", "data")
RECORDS_PATH = os.path.join(DATA_DIR, "feedback_records.xlsx")


# -----------------------------
# Read helper
# -----------------------------
def read_excel(path):
    try:
        return pd.read_excel(path, dtype=str, engine="openpyxl")
    except:
        return pd.DataFrame()


def write_excel(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)


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
# LOAD RECORDS
# -----------------------------
df = read_excel(RECORDS_PATH)

if df.empty:
    st.error("Feedback records file not found or empty.")
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

already_used = (token_used == "yes") or (existing_response != "")

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

    write_excel(df, RECORDS_PATH)

    st.success("✅ Thank you — your feedback has been recorded.")
