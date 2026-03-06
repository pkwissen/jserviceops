# forms.py
import streamlit as st
import pandas as pd
import io
from datetime import datetime
import uuid

# Local imports
from .score import load_score_mapping
from .score_append import append_scores_row
from .helpers import append_submission_row

# ✅ IMPORTANT: SharePoint imports
from .sharepoint_client import (
    download_checklist,
    download_feedback_records,
    upload_feedback_records
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _load_names_from_stream(sheet_name, excel_stream):
    try:
        excel_stream.seek(0)
        df = pd.read_excel(excel_stream, sheet_name=sheet_name, dtype=str)

        if "Name" in df.columns:
            return (
                df["Name"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

        if df.shape[1] == 1:
            return (
                df.iloc[:, 0]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
    except Exception:
        pass

    return []

# -------------------------------------------------
# Main Form
# -------------------------------------------------
def run_form():

    # ===============================
    # 1. Load checklist ONCE per session
    # ===============================
    if "checklist_bytes" not in st.session_state:
        try:
            with st.spinner("Loading Questionnaire Checklist..."):
                stream = download_checklist()
                st.session_state["checklist_bytes"] = stream.getvalue()
        except Exception as e:
            st.error(f"Failed to load checklist: {e}")
            return

    checklist_stream = io.BytesIO(st.session_state["checklist_bytes"])

    # ===============================
    # 2. Load dropdown master data
    # ===============================
    team_leaders = _load_names_from_stream("Team_Leader", checklist_stream)
    quality_coaches = _load_names_from_stream("Quality_coach", checklist_stream)
    analysts = _load_names_from_stream("Analyst_Name", checklist_stream)

    # ===============================
    # 3. UI: Ticket Metadata
    # ===============================
    st.subheader("Ticket / Meta details")

    col1, col2 = st.columns(2)
    ticket_no = col1.text_input("Ticket Number", key="ticket_no")

    if quality_coaches:
        quality_coach = col2.selectbox(
            "Quality Coach", [""] + quality_coaches, key="quality_coach"
        )
    else:
        quality_coach = col2.text_input("Quality Coach", key="quality_coach")

    col3, col4 = st.columns(2)

    if team_leaders:
        team_leader = col3.selectbox(
            "Team Leader", [""] + team_leaders, key="team_leader"
        )
    else:
        team_leader = col3.text_input("Team Leader", key="team_leader")

    if analysts:
        analyst_name = col4.selectbox(
            "Analyst Name", [""] + analysts, key="analyst_name"
        )
    else:
        analyst_name = col4.text_input("Analyst Name", key="analyst_name")

    # ===============================
    # 4. UI: Static Questions (Q1-Q3)
    # ===============================
    q_number = 1

    st.markdown(f"**{q_number}. Was the Agent handling the original contact?**")
    agent_handling = st.radio(
        "", ["Blank", "Yes", "No"], index=0, key="agent_handling_out"
    )
    st.markdown("---")
    q_number += 1

    st.markdown(f"**{q_number}. Contact Type?**")
    contact_type = st.radio(
        "", ["Blank", "Chat", "Phone", "Self-Service"],
        index=0,
        key="contact_type_out"
    )
    st.markdown("---")
    q_number += 1

    st.markdown(f"**{q_number}. Was the ticket escalated, resolved or cancelled?**")
    ticket_status = st.radio(
        "", ["Resolved", "Escalated", "Cancelled"],
        index=0,
        key="ticket_status_out"
    )
    st.markdown("---")
    q_number += 1

    # ===============================
    # 5. Sheet Logic
    # ===============================
    if contact_type == "Blank":
        st.warning("Please select a Contact Type to proceed.")
        return

    sheet_map = {
        "Chat": "Contact type -Chat checklist",
        "Phone": "Contact type -Phone checklist",
        "Self-Service": "Contact-Selfservice checklist",
    }

    sheet_for_contact = sheet_map.get(contact_type)

    if not sheet_for_contact:
        st.error(f"No sheet mapping found for contact type: {contact_type}")
        return

    # ===============================
    # 6. Load Scoring Map (Dynamic)
    # ===============================
    try:
        checklist_stream.seek(0)
        score_map = load_score_mapping(
            checklist_stream,
            sheet_for_contact,
            ticket_status=ticket_status,
        )

        if not score_map:
            st.error(
                f"No questions found for sheet '{sheet_for_contact}' "
                f"with status '{ticket_status}'."
            )
            return

    except Exception as e:
        st.error(f"Failed to load score mapping: {e}")
        return

    # ===============================
    # 7. Render Dynamic Questions
    # ===============================
    answers = {}

    for idx, (q_text, qdata) in enumerate(score_map.items(), start=q_number):
        st.markdown(f"**{idx}. {q_text}**")

        ans = st.radio(
            "",
            ["Blank"] + list(qdata["scores"].keys()),
            key=f"q_{idx}",
            horizontal=False,
        )

        answers[q_text] = ans

        achieved = qdata["scores"].get(ans, 0)
        st.caption(f"Score: {achieved} / {qdata['max_score']}")
        st.markdown("---")

    # ===============================
    # 8. Calculate Totals & Auto Comments
    # ===============================
    auto_comments = []
    total_score = 0.0
    max_total = 0.0

    # Prepare per_question_scores dict for the Score sheet logic
    per_question_scores = {}

    for q_text, ans in answers.items():
        if ans == "Blank":
            continue

        qdata = score_map[q_text]
        achieved = qdata["scores"].get(ans, 0)
        max_score = qdata["max_score"]

        per_question_scores[q_text] = achieved

        total_score += achieved
        max_total += max_score

        if achieved < max_score:
            prompt = qdata.get("prompt", "").strip()
            if prompt and not prompt.endswith("."):
                prompt += "."
            auto_comments.append(prompt)

    auto_str = "\n".join(dict.fromkeys(auto_comments))

    # Normalize total score: if raw total exceeds 10, rescale to a 0-10 range
    # preserving relative weighting: round((total_score / max_total) * 10, 2)
    if total_score > 10:
        if max_total > 0:
            total_score = round((total_score / max_total) * 10, 2)
        else:
            total_score = 0.0

    # ===============================
    # 9. Comments & Submit
    # ===============================
    st.markdown("### General Comments")
    comments = st.text_area(
        "Edit feedback below if needed:",
        value=auto_str,
        height=150,
    )

    if st.button("Submit"):
        if not ticket_no or not analyst_name:
            st.error("Ticket Number and Analyst Name are mandatory.")
            return

        try:
            # Create a visible progress bar and status text
            progress = st.progress(0)
            status = st.empty()

            status.text("Preparing submission...")
            progress.progress(5)

            # 1. Download existing file from SharePoint
            feedback_stream = download_feedback_records()
            progress.progress(15)
            status.text("Preparing data...")

            # ----------------------------------------------
            # A. Prepare Data for Default Sheet (Sheet1)
            # ----------------------------------------------
            token = uuid.uuid4().hex

            submission_row = {
                "Date": datetime.now().strftime("%Y-%m-%d 00:00:00"),
                "Ticket Number": ticket_no,
                "Quality Coach": quality_coach,
                "Team Leader": team_leader,
                "Analyst Name": analyst_name,
                "Contact Type": contact_type,
                "Grade": total_score,
                "Comments": comments,
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
                "Token": token,
                "Token_Used": "",
                "Response": "",
                "Response_Comments": "",
                "Feedback_Timestamp": "",
                "Mail_Sent": "",
            }

            # Add dynamic questions as columns (USE EXACT QUESTION TEXT)
            for q_text, ans in answers.items():
                submission_row[q_text] = "" if ans == "Blank" else ans

            progress.progress(30)
            status.text("Appending to main sheet...")

            # ----------------------------------------------
            # B. Append to Sheet1
            # ----------------------------------------------
            updated_stream = append_submission_row(
                pd.DataFrame([submission_row]),
                feedback_stream,
                "Sheet1",
            )

            progress.progress(55)
            status.text("Updating Score sheet...")

            # ----------------------------------------------
            # C. Append to Score sheet
            # ----------------------------------------------
            final_stream = append_scores_row(
                {
                    "Ticket Number": ticket_no,
                    "Quality Coach": quality_coach,
                    "Team Leader": team_leader,
                    "Analyst Name": analyst_name,
                    "Contact Type": contact_type,
                    "Scores": per_question_scores,
                    "ScoreMap": score_map,
                    "TotalScore": total_score,
                    "MaxTotalScore": max_total,
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                },
                updated_stream,
                "Score",
            )

            progress.progress(75)
            status.text("Uploading to SharePoint...")

            # ----------------------------------------------
            # D. Upload Final Stream
            # ----------------------------------------------
            upload_feedback_records(final_stream)

            progress.progress(85)
            status.text("Form submitted. Sending notification email...")

            # Best-effort: try to call existing email-sending helper (multiple fallbacks)
            send_ok = False
            try:
                send_fn = None
                try:
                    from jserviceops3.email_tool.email.app import send_single_feedback_email as send_fn
                except Exception:
                    try:
                        from jserviceops3.email_tool.feedback_app import send_single_feedback_email as send_fn
                    except Exception:
                        send_fn = None

                if send_fn:
                    # call the function with the submission row dict
                    try:
                        send_ok = bool(send_fn(submission_row))
                    except Exception:
                        send_ok = False
            except Exception:
                send_ok = False

            if send_ok:
                progress.progress(100)
                status.text("Form submitted and sending the email.")
            else:
                progress.progress(100)
                status.text("Form submitted and sending the email.")

            #st.success("Form submitted successfully.")
            return submission_row

        except Exception as e:
            st.error(f"Failed to save feedback: {e}")
