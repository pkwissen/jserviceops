# main.py
import streamlit as st
import sys
import os

# ---- Page config: must be FIRST and ONLY once in the whole app ----
st.set_page_config(page_title="ServiceOps Intelligence Suite", layout="wide")

# ---- Ensure project root is importable ----
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ---- Import apps as proper packages (no subfolder sys.path hacks) ----
import homepage
import forecast_app_final.app as forecast_app
import ticket_feedback_dashboard.app as ticket_dashboard
import jacobs_qa.app as jacobs_qa_app
import email_tool.email.app as email_app

# ---- Session state navigation setup ----
if "current_app" not in st.session_state:
    st.session_state["current_app"] = "Homepage"

# ---- Navigation Logic ----
if st.session_state["current_app"] == "Homepage":
    homepage.main()

elif st.session_state["current_app"] == "Forecast_App":
    forecast_app.main()

elif st.session_state["current_app"] == "Ticket_Feedback_Dashboard":
    ticket_dashboard.main()

elif st.session_state["current_app"] == "Jacobs_QA":
    jacobs_qa_app.main()

elif st.session_state["current_app"] in ["Email_Tool", "Email_Automation"]:
    email_app.main()
