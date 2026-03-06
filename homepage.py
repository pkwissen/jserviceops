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

    # Second row → Weekly app and Email Automation side by side
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📅 Weekly Coaching Assessment Tracker")
        st.markdown("Create Weekly Coaching Assessment report based on Manual and ServiceNow assessment.")
        if st.button("Go to Weekly Coaching Assessment Tracker", key="btn_weekly"):
            st.session_state["current_app"] = "Jacobs_QA"
            st.rerun()

    with col4:
        st.subheader("📧 Email Automation")
        st.markdown("Automate email sending and management tasks.")
        if st.button("Go to Email Automation", key="btn_email_automation"):
            st.session_state["current_app"] = "Email_Automation"
            st.rerun()


    # Third row → AI Tool for Quality Team
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📧 AI Tool for Quality Team")
        st.markdown("QM Form for quality assessment and Email Automation.")
        if st.button("Go to QM Feedback Email Automation Tool", key="btn_ai_tool"):
            st.session_state["current_app"] = "AI_Quality_Tool"
            st.rerun()


    # Fourth row → Intelligent Agent Assist
 
    with col6:
        st.subheader("🤖 Intelligent Agent Assist")
        st.markdown("Assist agents with knowledge retrieval, automation, and guidance.")
        if st.button("Go to Intelligent Agent Assist", key="btn_intelligent_agent"):
            st.session_state["current_app"] = "Intelligent_Agent"
            st.rerun()
