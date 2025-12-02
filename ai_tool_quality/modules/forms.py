import streamlit as st
from datetime import datetime
import pandas as pd
from modules.helpers import find_matching_sheets, load_questions_filtered_by_status, append_submission_row
from modules.config import EXCEL_PATH, OUTPUT_PATH, OUTPUT_SHEET
import os

def _load_names_from_sheet(sheet_name: str):
    """Load distinct names from given sheet (expects column 'Name')."""
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, dtype=str)
        if "Name" in df.columns:
            names = df["Name"].dropna().astype(str).str.strip().unique().tolist()
            return names
    except Exception as e:
        st.warning(f"Could not load names from sheet '{sheet_name}': {e}")
    return []

def run_form():
    if not os.path.exists(EXCEL_PATH):
        st.error(f"Excel file not found: {EXCEL_PATH}")
        return

    xls = pd.ExcelFile(EXCEL_PATH)
    sheets = xls.sheet_names

    # Load dropdown values from special sheets
    team_leader_list = _load_names_from_sheet("Team_Leader")
    quality_coach_list = _load_names_from_sheet("Quality_coach")
    analyst_name_list = _load_names_from_sheet("Analyst_Name")

    st.subheader("Ticket / Meta details")
    col1, col2 = st.columns(2)
    ticket_no = col1.text_input("Ticket Number", key="ticket_no")

    # Dropdown for Quality Coach
    if quality_coach_list:
        quality_coach = col2.selectbox("Quality Coach", [""] + quality_coach_list, key="quality_coach")
    else:
        quality_coach = col2.text_input("Quality Coach", key="quality_coach")

    col3, col4 = st.columns(2)

    # Dropdown for Team Leader
    if team_leader_list:
        team_leader = col3.selectbox("Team Leader", [""] + team_leader_list, key="team_leader")
    else:
        team_leader = col3.text_input("Team Leader", key="team_leader")

    # Dropdown for Analyst Name
    if analyst_name_list:
        analyst_name = col4.selectbox("Analyst Name", [""] + analyst_name_list, key="analyst_name")
    else:
        analyst_name = col4.text_input("Analyst Name", key="analyst_name")

    q_number = 1

    st.markdown(f"**{q_number}. Was the Agent handling the original contact?**")
    agent_handling = st.radio("", ["Blank", "Yes", "No"], index=0, key="agent_handling_out")
    st.markdown("---"); q_number += 1

    st.markdown(f"**{q_number}. Contact Type?**")
    contact_type = st.radio("", ["Blank", "Chat", "Phone", "Self-Service"], index=0, key="contact_type_out")
    st.markdown("---"); q_number += 1

    st.markdown(f"**{q_number}. Was the ticket escalated, resolved or cancelled?**")
    ticket_status = st.radio("", ["Resolved", "Escalated", "Cancelled"], index=0, key="ticket_status_out")
    st.markdown("---"); q_number += 1

    matched = find_matching_sheets(contact_type, sheets) if contact_type != "Blank" else []
    if matched:
        if len(matched) == 1:
            sheet_for_contact = matched[0]
            st.success(f"Auto-selected sheet: {sheet_for_contact}")
        else:
            sheet_for_contact = st.selectbox("Multiple matching sheets found — choose one", options=matched, index=0)
    elif contact_type == "Blank":
        sheet_for_contact = None
    else:
        st.info("Could not auto-detect sheet. Choose manually:")
        sheet_for_contact = st.selectbox("Choose sheet", options=sheets, index=0)

    answers = {}
    if sheet_for_contact:
        try:
            q_pairs = load_questions_filtered_by_status(EXCEL_PATH, sheet_for_contact, ticket_status)
        except Exception as e:
            st.error(f"Failed to load sheet '{sheet_for_contact}': {e}")
            q_pairs = []

        st.subheader(f"Checklist questions from: {sheet_for_contact} (filtered by: {ticket_status})")
        if not q_pairs:
            st.warning(f"No valid checklist questions for status '{ticket_status}' found in sheet '{sheet_for_contact}'.")
        else:
            for i, (sr, q_text, opts) in enumerate(q_pairs, start=1):
                st.markdown(f"**{q_number}. {q_text}**")
                radio_options = ["Blank"] + opts
                ans = st.radio("", radio_options, index=0, key=f"q_{sheet_for_contact}_{i}_{sr}")
                answers[q_text] = ans
                st.markdown("---")
                q_number += 1
    else:
        st.info("Contact Type is Blank — no checklist sheet selected.")

    st.markdown(f"**{q_number}. Any general comments (optional)**")
    comments = st.text_area("", height=140, key="comments")
    st.markdown("---")

    if st.button("Submit"):
        missing = []
        if not str(ticket_no).strip():
            missing.append("Ticket Number")
        if not str(analyst_name).strip():
            missing.append("Analyst Name")

        if missing:
            st.error(f"Please fill mandatory fields: {', '.join(missing)}")
        else:
            row = {
                "Date": datetime.now().date().isoformat(),
                "Ticket Number": ticket_no,
                "Quality Coach": quality_coach,
                "Team Leader": team_leader,
                "Analyst Name": analyst_name,
                "Agent handling original contact?": "" if agent_handling == "Blank" else agent_handling,
                "Contact Type": "" if contact_type == "Blank" else contact_type,
                "Was the ticket escalated, resolved or cancelled?": ticket_status,
            }
            for colname, ans in answers.items():
                row[colname] = "" if ans == "Blank" else ans
            row["Comments"] = comments
            row["Timestamp"] = datetime.now().isoformat(timespec='seconds')

            df_row = pd.DataFrame([row])
            try:
                append_submission_row(df_row, OUTPUT_PATH, OUTPUT_SHEET)
                # st.success(f"Saved submission to '{OUTPUT_PATH}' (sheet: {OUTPUT_SHEET}).")
            except Exception as e:
                st.error(f"Failed to save submission: {e}")
