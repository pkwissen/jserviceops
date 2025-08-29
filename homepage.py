# homepage.py
import streamlit as st

def main():
    # DO NOT call st.set_page_config here.
    st.title("🚀 ServiceOps Intelligence Suite")
    st.markdown("Welcome! Select an application below to explore its features")

    # First row → 2 apps
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Heat Map & Shift Planning")
        st.markdown("Create Heat Map & Shift Planning.")
        if st.button("Go to Heat Map & Shift Planning", key="btn_heatmap"):
            st.session_state["current_app"] = "Forecast_App"
            st.rerun()

    with col2:
        st.subheader("🎫 Ticket Feedback Dashboard")
        st.markdown("Analyze and visualize ticket feedback data.")
        if st.button("Go to Ticket Feedback Dashboard", key="btn_ticket"):
            st.session_state["current_app"] = "Ticket_Feedback_Dashboard"
            st.rerun()

    # Second row → Weekly app directly below the first column
    col3, _ = st.columns([1, 1])

    with col3:
        st.subheader("📅 Weekly Coaching Assessment Tracker")
        st.markdown("Create Weekly Coaching Assessment report based on Manual and ServiceNow assessment.")
        if st.button("Go to Weekly Coaching Assessment Tracker", key="btn_weekly"):
            st.session_state["current_app"] = "Jacobs_QA"
            st.rerun()
