# homepage.py
import streamlit as st

def main():
    st.set_page_config(page_title="ServiceOps Intelligence Suite", layout="wide")
    st.title("🚀 ServiceOps Intelligence Suite")
    st.markdown("Welcome! Select an application below to explore its features")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Heat Map & Shift Planning")
        st.markdown("Create Heat Map & Shift Planning .")
        if st.button("Go to Heat Map & Shift Planning"):
            st.session_state["current_app"] = "Forecast_App"
            st.rerun()

    with col2:
        st.subheader("🎫 Ticket Feedback Dashboard")
        st.markdown("Analyze and visualize ticket feedback data.")
        if st.button("Go to Ticket Feedback Dashboard"):
            st.session_state["current_app"] = "Ticket_Feedback_Dashboard"
            st.rerun()
