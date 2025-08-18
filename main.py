# main.py
import streamlit as st
import sys
import os

# --- Add app folders to Python path ---
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "forecast_app_final"))
sys.path.append(os.path.join(BASE_DIR, "ticket_feedback_dashboard"))

# --- Import apps ---
import homepage
import forecast_app_final.app as forecast_app
import ticket_feedback_dashboard.app as ticket_dashboard

# --- Session state navigation setup ---
if "current_app" not in st.session_state:
    st.session_state["current_app"] = "Homepage"

# --- Navigation Logic ---
if st.session_state["current_app"] == "Homepage":
    homepage.main()

elif st.session_state["current_app"] == "Forecast_App":
    forecast_app.main()

elif st.session_state["current_app"] == "Ticket_Feedback_Dashboard":
    ticket_dashboard.main()
