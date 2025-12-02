import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#print("PYTHONPATH patched:", sys.path[-1])

# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os, uuid

from email_tool.email.modules.utils import format_body, get_email, get_lead_email, get_QC_email, get_fixed_cc
from email_tool.email.modules.mailer import get_sender_name, send_mail
 
from email_tool.email.modules.user_manager import (
    ensure_user_sheets,
    list_user_sheets,
    get_users,
    add_user,
    remove_user,
    SHEET_NAME_HEADER,
    EMAIL_HEADER,
)
 
st.set_page_config(page_title="📧 Feedback Email Automation Tool", layout="centered")

# ---------- Helpers to find project root & shared data dir ----------
def find_project_root(start_dir: str = None) -> str:
    """
    Climb up from the current file directory until a folder containing 'modules'
    is found. Return that folder path. Fallback to the directory of this file.
    This ensures that files placed in subfolders (e.g. email/) and root (email_tool/)
    both resolve to the same project root that contains 'modules'.
    """
    if start_dir is None:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "modules")):
            return cur
        parent = os.path.abspath(os.path.join(cur, ".."))
        if parent == cur:
            # reached filesystem root, return current dir as fallback
            return cur
        cur = parent

PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
RECORDS_PATH = os.path.join(DATA_DIR, "feedback_records.xlsx")
 
# ---------- Shared file paths (both apps must use these) ----------
MAPPING_PATH = os.path.join(DATA_DIR, "Questionnaire_Checklist.xlsx")
FEEDBACK_SOURCE_PATH = os.path.join(DATA_DIR, "last_uploaded_feedback.xlsx")
FEEDBACK_RECORDS_PATH = os.path.join(DATA_DIR, "feedback_records.xlsx")
FEEDBACK_SUBMISSIONS_PATH = os.path.join(DATA_DIR, "feedback_submissions.xlsx")
SUBMISSIONS_PATH = os.path.join(DATA_DIR, "Questionnaire_submissions.xlsx")

# ensure mapping user sheets exist (creates file if necessary)
ensure_user_sheets(MAPPING_PATH)
def main():
    
    # ---------- App UI ----------
    if st.button("Back to Homepage", key="heat_back_homepage"):
        st.session_state["current_app"] = "Homepage"
        st.rerun()  # Navigate back to the homepage
        # ---------- App UI ----------
    st.title("📧 Feedback Email Automation Tool")

    PROJECT_ROOT = find_project_root()
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    RECORDS_PATH = os.path.join(DATA_DIR, "feedback_records.xlsx")
    
    st.markdown("""
    <style>
    /* General tab label (covers different DOM variants) */
    div[data-baseweb="tab"] > button > div,
    div[data-baseweb="tab"] > button > div > span,
    [role="tablist"] [role="tab"] > div,
    [role="tablist"] [role="tab"] {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #374151 !important; /* default color */
        transition: color 0.2s ease, background-color 0.2s ease;
    }

    /* Hover */
    div[data-baseweb="tab"] > button:hover > div,
    div[data-baseweb="tab"] > button:hover > div > span,
    [role="tablist"] [role="tab"]:hover {
        color: #2563eb !important; /* blue on hover */
    }

    /* Active tab: button with aria-selected="true" */
    div[data-baseweb="tab"] > button[aria-selected="true"] > div,
    div[data-baseweb="tab"] > button[aria-selected="true"] > div > span,
    [role="tablist"] [role="tab"][aria-selected="true"],
    [role="tablist"] [role="tab"][aria-selected="true"] > div {
        color: #1d4ed8 !important;   /* blue */
        border-bottom: 3px solid #1d4ed8 !important;
    }

    /* Active tab background shading */
    div[data-baseweb="tab"] > button[aria-selected="true"],
    [role="tablist"] [role="tab"][aria-selected="true"] {
        background-color: rgba(29,78,216,0.06) !important;
        border-radius: 8px 8px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Now create tabs
    tabs = st.tabs(["Manage Users", "Send Emails", "Tracking Dashboard"])

    with tabs[0]:
        st.write("Manage Users content")
    with tabs[1]:
        st.write("Send Emails content")
    with tabs[2]:
        st.write("Tracking Dashboard content")
    manage_tab, send_tab, track_tab = tabs

    # ---------------- Manage Users ----------------
    with manage_tab:
        st.header("Manage Users (Team Leader / Quality Coach / Analyst)")
        st.write("Add or remove users. Each entry stores a Name and Email in the workbook.")
    
        sheet_ids = list_user_sheets()
        sheet_id = st.selectbox("Select user list", options=sheet_ids)
    
        users = get_users(MAPPING_PATH, sheet_id)
        st.subheader(f"Existing users in '{sheet_id}' ({len(users)})")
        if users:
            df_users = pd.DataFrame(users, columns=[SHEET_NAME_HEADER.get(sheet_id, "Name"), EMAIL_HEADER])
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        else:
            st.info("No users found in this list.")
    
        st.markdown("---")
        st.markdown("### Add new user")
        coln, cole = st.columns(2)
        new_name = coln.text_input("Full name", key="new_user_name")
        new_email = cole.text_input("Email", key="new_user_email")
    
        if st.button("Add user"):
            if not new_name.strip():
                st.error("Please enter a name.")
            else:
                try:
                    added = add_user(MAPPING_PATH, sheet_id, new_name.strip(), new_email.strip())
                    if added:
                        st.success(f"Added: {new_name.strip()}  —  {new_email.strip()}")
                    else:
                        st.warning("This user already exists (same name+email).")
                except Exception as e:
                    st.error(f"Failed to add user: {e}")
                st.rerun()
    
        st.markdown("---")
        st.markdown("### Remove user")
        if users:
            selection_items = [f"{n} — {e}" if e else f"{n} — (no email)" for n, e in users]
            to_remove = st.selectbox("Select user to remove", options=selection_items, key="remove_select")
            if st.button("Remove selected user"):
                sel = to_remove.split(" — ")
                name_part = sel[0].strip()
                email_part = sel[1].strip() if len(sel) > 1 else ""
                if email_part == "(no email)":
                    email_part = ""
                try:
                    removed = remove_user(MAPPING_PATH, sheet_id, name_part, email_part)
                    if removed:
                        st.success(f"Removed: {name_part} — {email_part}")
                    else:
                        st.warning("User not found or could not remove.")
                except Exception as e:
                    st.error(f"Failed to remove user: {e}")
                st.rerun()
        else:
            st.info("No users to remove.")

    # ---------------- Send Emails ----------------
    with send_tab:
        st.header("Send Emails")
        st.write("Upload a feedback Excel (it must contain at least: Date, Ticket Number, Quality Coach, Team Leader, Analyst Name).")
    
        uploaded_file = st.file_uploader("Upload Feedback Sheet (Excel)", type=["xlsx"])
        if uploaded_file is None:
            st.info("Upload the feedback sheet to begin.")
        else:
            try:
                df = pd.read_excel(uploaded_file, dtype=str)
            except Exception as e:
                st.error(f"Failed to read uploaded Excel: {e}")
                df = None

            if df is not None:
                st.subheader("Preview of uploaded data")
                st.dataframe(df.head())

                if st.button("Prepare feedback records"):
                
                    try:
                        # Start from full uploaded dataframe (preserve all columns)
                        df_new = df.copy()
                
                        # Ensure meta columns exist in the uploaded copy
                        for col in ["Response", "Response_Comments", "Feedback_Timestamp", "Token", "Token_Used"]:
                            if col not in df_new.columns:
                                df_new[col] = ""
                
                        # Generate tokens for empty Token values in uploaded data
                        def gen_token_if_empty(t):
                            if pd.isna(t) or str(t).strip() == "":
                                return uuid.uuid4().hex
                            return str(t).strip()
                
                        df_new["Token"] = df_new["Token"].apply(gen_token_if_empty)
                        df_new["Token_Used"] = df_new["Token_Used"].fillna("")
                
                        # If no existing records file, simply save df_new (preserves all uploaded columns)
                        if not os.path.exists(RECORDS_PATH):
                            os.makedirs(os.path.dirname(RECORDS_PATH), exist_ok=True)
                            df_new.to_excel(RECORDS_PATH, index=False, engine="openpyxl")
                        else:
                            # Load existing records (preserve their columns)
                            df_existing = pd.read_excel(RECORDS_PATH, dtype=str, engine="openpyxl").fillna("")
                
                            # Ensure meta columns exist in existing file
                            for col in ["Response", "Response_Comments", "Feedback_Timestamp", "Token", "Token_Used"]:
                                if col not in df_existing.columns:
                                    df_existing[col] = ""
                
                            # Build a set of existing keys for quick lookup (ticket, analyst)
                            def norm_key(ticket, analyst):
                                return (str(ticket or "").strip().lower(), str(analyst or "").strip().lower())
                
                            existing_keys = set()
                            if "Ticket Number" in df_existing.columns or "Analyst Name" in df_existing.columns:
                                for _, r in df_existing.iterrows():
                                    existing_keys.add(norm_key(r.get("Ticket Number", ""), r.get("Analyst Name", "")))
                
                            # Prepare list of new rows to append (as dicts) preserving uploaded columns
                            new_rows = []
                            appended = 0
                            for _, r in df_new.iterrows():
                                t_val = str(r.get("Ticket Number", "") or "").strip()
                                a_val = str(r.get("Analyst Name", "") or "").strip()
                                key = norm_key(t_val, a_val)
                
                                # if either Ticket+Analyst match an existing record, skip (do not overwrite)
                                if key in existing_keys:
                                    continue
                
                                # Build row dict preserving all columns from uploaded row
                                new_row = {}
                                for c in df_new.columns:
                                    new_row[c] = r.get(c, "") if c in r.index else ""
                
                                # Ensure meta columns exist in this new row
                                if "Token" not in new_row or str(new_row.get("Token", "")).strip() == "":
                                    new_row["Token"] = uuid.uuid4().hex
                                if "Token_Used" not in new_row:
                                    new_row["Token_Used"] = ""
                                if "Response" not in new_row:
                                    new_row["Response"] = ""
                                if "Response_Comments" not in new_row:
                                    new_row["Response_Comments"] = ""
                                if "Feedback_Timestamp" not in new_row:
                                    new_row["Feedback_Timestamp"] = ""
                
                                new_rows.append(new_row)
                                appended += 1
                                existing_keys.add(key)
                
                            # If there are new rows, append them to existing dataframe
                            if new_rows:
                                df_append = pd.DataFrame(new_rows)
                                # Combine columns: union of existing and new columns (existing order preserved, new columns appended)
                                all_cols = list(df_existing.columns)
                                for c in df_append.columns:
                                    if c not in all_cols:
                                        all_cols.append(c)
                                # Reindex both frames to all_cols before concat (fills missing with "")
                                df_existing = df_existing.reindex(columns=all_cols).fillna("")
                                df_append = df_append.reindex(columns=all_cols).fillna("")
                                df_out = pd.concat([df_existing, df_append], ignore_index=True)
                            else:
                                df_out = df_existing.copy()
                
                            # Save merged output
                            os.makedirs(os.path.dirname(RECORDS_PATH), exist_ok=True)
                            df_out.to_excel(RECORDS_PATH, index=False, engine="openpyxl")
                    except Exception as e:
                        st.error(f"Failed to prepare/append records: {e}")
    
                # Send Emails button
                if st.button("Send Emails"):
                    start_time = datetime.now()
                    st.info(f"Execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    try:
                        sender_name = get_sender_name()
                    except Exception:
                        sender_name = ""

                    if not os.path.exists(RECORDS_PATH):
                        st.error("Prepared feedback records not found. Please click 'Prepare feedback records' first.")
                    else:
                        try:
                            df_records = pd.read_excel(RECORDS_PATH, dtype=str, engine="openpyxl").fillna("")

                            # Ensure Mail_Sent column exists
                            if "Mail_Sent" not in df_records.columns:
                                df_records["Mail_Sent"] = ""

                            total = len(df_records)
                            sent_count = 0
                            skipped_count = 0

                            progress_bar = st.progress(0)
                            counter_placeholder = st.empty()

                            for idx, row in df_records.iterrows():
                                # Skip already sent emails
                                if str(row.get("Mail_Sent", "")).strip().lower() == "yes":
                                    skipped_count += 1
                                    continue

                                analyst_name = row.get("Analyst Name")
                                if pd.isna(analyst_name) or str(analyst_name).strip() == "":
                                    st.warning(f"Row {idx + 2}: 'Analyst Name' is empty — skipping.")
                                    continue

                                to_email = get_email(analyst_name, MAPPING_PATH, sheet_name="Analyst_Name")
                                if not to_email:
                                    st.warning(f"Row {idx + 2}: No email found for '{analyst_name}' — skipping.")
                                    continue

                                lead_name = row.get("Team Leader")
                                QC_name = row.get("Quality Coach")
                                cc_email_lead = get_lead_email(lead_name, MAPPING_PATH, sheet_name="Team_Leader")
                                cc_email_QC = get_QC_email(QC_name, MAPPING_PATH, sheet_name="Quality_coach")
                                fixed_cc = get_fixed_cc(MAPPING_PATH, sheet_name="Fixed_CC")

                                subject = f"Feedback for Analyst {analyst_name}"
                                body_html = format_body(row, sender_name)

                                try:
                                    send_mail(to_email, subject, body_html, cc=[cc_email_lead, cc_email_QC] + fixed_cc)
                                    sent_count += 1
                                    df_records.at[idx, "Mail_Sent"] = "Yes"  # ✅ Mark as sent
                                except Exception as e:
                                    st.error(f"Failed to send to {to_email} (Row {idx + 2}): {e}")

                                # Update progress
                                progress = int(((sent_count + skipped_count) / total) * 100) if total else 0
                                progress_bar.progress(min(progress, 100))
                                counter_placeholder.text(f"📨 Sent {sent_count}/{total} emails (Skipped {skipped_count})")

                            # Save updated records after sending
                            df_records.to_excel(RECORDS_PATH, index=False, engine="openpyxl")

                            end_time = datetime.now()
                            duration = end_time - start_time
                            st.info(f"Execution ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            st.success(f"✅ Total execution time: {duration} | Emails sent: {sent_count} | Skipped: {skipped_count}")

                        except Exception as e:
                            st.error(f"Failed to send emails: {e}")

    # ---------------- Tracking Dashboard ----------------
    with track_tab:
        st.header("Tracking Dashboard — Feedback records")
        st.write("This dashboard reads the feedback-records file and shows responses and counts.")
    
        RECORDS_PATH = globals().get("RECORDS_PATH") or globals().get("FEEDBACK_RECORDS_PATH")
        if not RECORDS_PATH:
            st.error("No records path defined (RECORDS_PATH or FEEDBACK_RECORDS_PATH).")
        else:
            c1, c2, c3 = st.columns([1, 2, 3])
            refresh_clicked = c1.button("🔄 Refresh", key="dashboard_refresh_btn")
            if refresh_clicked:
                st.session_state["dashboard_last_refresh"] = datetime.now().isoformat(timespec="seconds")
    
            last_refresh = st.session_state.get("dashboard_last_refresh")
            if last_refresh:
                c2.markdown(f"**Last refresh:** {last_refresh}")
    
            try:
                mtime = os.path.getmtime(RECORDS_PATH)
                file_mod = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                c3.markdown(f"**File last modified:** {file_mod}")
            except Exception:
                c3.markdown("**File last modified:** —")
    
            st.markdown("---")
    
            if not os.path.exists(RECORDS_PATH):
                st.info("No feedback records found. Upload a feedback sheet and click 'Prepare feedback records' in Send Emails tab first.")
            else:
                try:
                    df_feedback = pd.read_excel(RECORDS_PATH, dtype=str, engine="openpyxl")
                except Exception as e:
                    st.error(f"Failed to read feedback records: {e}")
                    df_feedback = pd.DataFrame()
    
                if df_feedback.empty:
                    st.info("No records to display.")
                else:
                    # create a working copy and add SR No (1-based) as a display column
                    df_display = df_feedback.copy().reset_index(drop=True)
                    df_display.insert(0, "SR No.", df_display.index.map(lambda x: x + 1))
    
                    # --- Format date column if present ---
                    if "Date" in df_display.columns:
                        try:
                            df_display["Date"] = pd.to_datetime(df_display["Date"], errors="coerce").dt.strftime("%d-%m-%Y")
                        except Exception:
                            # leave as-is if formatting fails
                            pass
    
                    # response normalization
                    resp_series = df_display.get("Response", pd.Series([""] * len(df_display))).fillna("").astype(str).str.strip().str.lower()
                    yes_vals, no_vals = {"yes", "y", "true", "1"}, {"no", "n", "false", "0"}
    
                    yes_count = int(resp_series.isin(yes_vals).sum())
                    no_count = int(resp_series.isin(no_vals).sum())
                    not_responded_count = int((~resp_series.isin(yes_vals.union(no_vals))).sum())
    
                    colA, colB, colC, colD = st.columns([1, 1, 1, 2])
                    colA.metric("Total Records", len(df_display))
                    colB.metric("Responded — Yes", yes_count)
                    colC.metric("Responded — No", no_count)
                    colD.metric("Not responded yet", not_responded_count)
    
                    st.markdown("---")

                    # --- Date range filter using Calendar + Quality Coach + Response filter ---
                    if "Date" in df_display.columns:
                        try:
                            df_display["Date"] = pd.to_datetime(df_display["Date"], format="%d-%m-%Y", errors="coerce")

                            min_date = df_display["Date"].min()
                            max_date = df_display["Date"].max()

                            st.write("📅 **Filter by Date Range**")
                            col_start, col_end = st.columns(2)

                            import datetime as dt
                            default_start = min_date.date() if pd.notnull(min_date) else (
                                        dt.date.today() - dt.timedelta(days=30))
                            default_end = max_date.date() if pd.notnull(max_date) else dt.date.today()

                            # Let user pick any range freely
                            start_date = col_start.date_input(
                                "Start date",
                                value=default_start,
                                key="start_date_filter"
                            )
                            end_date = col_end.date_input(
                                "End date",
                                value=default_end,
                                key="end_date_filter"
                            )

                            if start_date > end_date:
                                st.warning("⚠️ Start date cannot be after end date.")
                            else:
                                # --- Filter by date range ---
                                df_filtered = df_display[
                                    (df_display["Date"].dt.date >= start_date) &
                                    (df_display["Date"].dt.date <= end_date)
                                    ].copy()

                                # --- Quality Coach filter (with 'All' option) ---
                                if "Quality Coach" in df_filtered.columns and not df_filtered.empty:
                                    coach_list = sorted(df_filtered["Quality Coach"].dropna().unique().tolist())
                                    coach_list.insert(0, "All")  # Add 'All' option at the top

                                    selected_coach = st.selectbox("🎯 Select Quality Coach", coach_list)

                                    if selected_coach != "All":
                                        df_filtered = df_filtered[df_filtered["Quality Coach"] == selected_coach]

                                # --- Response filter (AFTER date & coach) ---
                                filter_option = st.radio(
                                    "📋 Filter by Response",
                                    options=["All", "Responded — Yes", "Responded — No", "Not responded yet"],
                                    index=0,
                                    horizontal=True,
                                    key="response_filter_radio"
                                )

                                mask_yes = df_filtered["Response"].astype(str).str.lower().isin(yes_vals)
                                mask_no = df_filtered["Response"].astype(str).str.lower().isin(no_vals)
                                mask_not = ~df_filtered["Response"].astype(str).str.lower().isin(
                                    yes_vals.union(no_vals))

                                if filter_option == "Responded — Yes":
                                    df_filtered = df_filtered[mask_yes]
                                elif filter_option == "Responded — No":
                                    df_filtered = df_filtered[mask_no]
                                elif filter_option == "Not responded yet":
                                    df_filtered = df_filtered[mask_not]
                                # else “All” keeps df_filtered unchanged

                                # Reformat back to string for display
                                df_filtered["Date"] = df_filtered["Date"].dt.strftime("%d-%m-%Y")

                                # --- Show filtered table ---
                                display_cols = [
                                    c for c in [
                                        "SR No.", "Date", "Ticket Number", "Quality Coach", "Team Leader",
                                        "Analyst Name", "Response", "Response_Comments",
                                        "Feedback_Timestamp", "Mail_Sent"
                                    ]
                                    if c in df_filtered.columns
                                ]

                                if not df_filtered.empty and display_cols:
                                    st.dataframe(df_filtered[display_cols], use_container_width=True, hide_index=True)
                                else:
                                    st.warning("No data available for the selected filter.")

                        except Exception as e:
                            st.error(f"Date filtering failed: {e}")

                    st.markdown("---")

                    # --- Visualizations Section ---
                    if not df_display.empty:
                        st.markdown("### 📊 Visual Insights")

                        import altair as alt

                        # 1️⃣ Manual Coaching Assessments by Quality Coach (Including Response = No)
                        st.subheader("Manual Coaching Acknowledgements by Quality Coach (Only Response = No)")

                        # Prepare cleaned response column
                        df_display["Response_Clean"] = df_display["Response"].astype(str).str.lower().str.strip()

                        # Define mask for "No" responses
                        no_mask = df_display["Response_Clean"].isin(["no", "responded — no"])

                        # Get all quality coaches (ensuring everyone appears)
                        all_coaches = sorted(df_display["Quality Coach"].dropna().unique())

                        # Initialize DataFrame with all coaches and count = 0
                        coach_counts = pd.DataFrame({"Quality Coach": all_coaches, "Count": [0] * len(all_coaches)})

                        # Calculate actual counts for “No” responses
                        actual_no_counts = (
                            df_display[no_mask]
                            .groupby("Quality Coach")
                            .size()
                            .rename("Count")
                            .to_dict()
                        )

                        # Fill counts where available
                        coach_counts["Count"] = coach_counts["Quality Coach"].map(lambda x: actual_no_counts.get(x, 0))

                        # Create multicolor Altair chart
                        if not coach_counts.empty:
                            chart_coach = alt.Chart(coach_counts).mark_bar().encode(
                                x=alt.X("Quality Coach:N", sort=None, title="Quality Coach"),
                                y=alt.Y("Count:Q", title="Number of Assessments"),
                                color=alt.Color("Quality Coach:N", legend=None),
                                tooltip=["Quality Coach", "Count"]
                            ).properties(
                                width="container",
                                height=400,
                                #title="Manual Coaching Assessments by Quality Coach (Including Response = No)"
                            )

                            st.altair_chart(chart_coach, use_container_width=True)
                        else:
                            st.info("No records found for Response = No")

                        # 2️⃣ Combined Visualization: Manual Coaching Acknowledgements by Team Leader (Yes / No / Not Responded)
                        st.subheader("Manual Coaching Acknowledgements by Team Leads")

                        # Prepare cleaned response column
                        df_display["Response_Clean"] = df_display["Response"].astype(str).str.lower().str.strip()

                        # Define masks
                        yes_mask = df_display["Response_Clean"].isin(["yes", "responded — yes"])
                        no_mask = df_display["Response_Clean"].isin(["no", "responded — no"])
                        not_resp_mask = ~df_display["Response_Clean"].isin(
                            ["yes", "no", "responded — yes", "responded — no"])

                        # Get full set of team leaders
                        all_team_leads = sorted(df_display["Team Leader"].dropna().unique())

                        # Initialize counts for all team leaders and all response types
                        response_types = ["Yes", "No", "Not Responded"]
                        df_team = pd.DataFrame([(tl, rt, 0) for tl in all_team_leads for rt in response_types],
                                               columns=["Team Leader", "Response_Type", "Count"])

                        # Compute actual counts for each category
                        yes_counts = df_display[yes_mask].groupby("Team Leader").size().rename("Yes").to_dict()
                        no_counts = df_display[no_mask].groupby("Team Leader").size().rename("No").to_dict()
                        not_counts = df_display[not_resp_mask].groupby("Team Leader").size().rename("Not Responded").to_dict()

                        # Fill in real counts where available
                        for idx, row in df_team.iterrows():
                            tl, rt = row["Team Leader"], row["Response_Type"]
                            if rt == "Yes" and tl in yes_counts:
                                df_team.at[idx, "Count"] = yes_counts[tl]
                            elif rt == "No" and tl in no_counts:
                                df_team.at[idx, "Count"] = no_counts[tl]
                            elif rt == "Not Responded" and tl in not_counts:
                                df_team.at[idx, "Count"] = not_counts[tl]

                        # User selects which response type to view
                        response_filter = st.radio(
                            "📋 Select Response Type",
                            options=["Yes", "No", "Not Responded"],
                            horizontal=True,
                            key="team_response_filter"
                        )

                        filtered_team = df_team[df_team["Response_Type"] == response_filter]

                        # Create bar chart
                        if not filtered_team.empty:
                            chart_team = alt.Chart(filtered_team).mark_bar().encode(
                                x=alt.X("Team Leader:N", sort=None, title="Team Leads"),
                                y=alt.Y("Count:Q", title="Number of Assessments"),
                                color=alt.Color("Team Leader:N", legend=None),
                                tooltip=["Team Leader", "Count", "Response_Type"]
                            ).properties(
                                width="container",
                                height=400,
                                title=f"Manual Coaching Acknowledgements by Team Leads — {response_filter}"
                            )

                            st.altair_chart(chart_team, use_container_width=True)
                        else:
                            st.info(f"No records found for Response = {response_filter}")

                        st.markdown("<br><hr style='margin:25px 0;border:1px solid #444;'>", unsafe_allow_html=True)
                        # ----------------------------------------------------------
                        # 🧩 Add-ons Section: Top/Bottom Performers + Contact Type Dashboard
                        # ----------------------------------------------------------

                        # --- Top & Bottom Performers Section ---
                        st.markdown("### 🏆 Performance Summary")

                        if not df_filtered.empty:
                            df_temp = df_filtered.copy()

                            # Ensure date is datetime
                            df_temp["Date"] = pd.to_datetime(df_temp["Date"], errors="coerce", dayfirst=True)
                            df_temp = df_temp.dropna(subset=["Date", "Grade", "Analyst Name"])
                            df_temp["Grade"] = pd.to_numeric(df_temp["Grade"], errors="coerce")

                            # Derive Week and Month
                            df_temp["WeekNum"] = df_temp["Date"].dt.isocalendar().week
                            df_temp["Month"] = df_temp["Date"].dt.strftime("%b")  # short month name
                            df_temp["Year"] = df_temp["Date"].dt.year

                            # Create readable week labels (e.g. "Oct - 2nd Week")
                            def week_label(row):
                                # Week of month: difference between current week and first week of that month
                                month_start = pd.Timestamp(year=row["Year"], month=row["Date"].month, day=1)
                                week_of_month = (row["Date"].isocalendar().week -
                                                 month_start.isocalendar().week + 1)
                                suffix = {1: "st", 2: "nd", 3: "rd"}.get(week_of_month, "th")
                                return f"{row['Month']} - {week_of_month}{suffix} Week"

                            df_temp["WeekLabel"] = df_temp.apply(week_label, axis=1)
                            df_temp["MonthLabel"] = df_temp["Date"].dt.strftime("%B %Y")

                            # --- Choose time period ---
                            period_choice = st.radio(
                                "Select Performance Period:",
                                options=["Weekly", "Monthly"],
                                horizontal=True,
                                key="performance_period_choice"
                            )

                            if period_choice == "Weekly":
                                week_options = sorted(df_temp["WeekLabel"].unique())
                                selected_period = st.selectbox("Select Week:", week_options)
                                df_period = df_temp[df_temp["WeekLabel"] == selected_period]
                            else:
                                month_options = sorted(df_temp["MonthLabel"].unique())
                                selected_period = st.selectbox("Select Month:", month_options)
                                df_period = df_temp[df_temp["MonthLabel"] == selected_period]

                            if not df_period.empty:
                                # Compute average + count
                                avg_ratings = (
                                    df_period.groupby("Analyst Name")["Grade"]
                                    .agg(Avg_Grade="mean", Assessments="count")
                                    .reset_index()
                                )

                                # ✅ Sort: First by Avg_Grade (desc), then by Assessments (desc)
                                avg_ratings = avg_ratings.sort_values(
                                    by=["Avg_Grade", "Assessments"],
                                    ascending=[False, False]
                                ).reset_index(drop=True)

                                # --- Top Performer (always shown)
                                top_performer = avg_ratings.iloc[0]

                                # --- Bottom Performer Logic (exclude avg = 10)
                                non_perfect = avg_ratings[avg_ratings["Avg_Grade"] < 10]

                                if not non_perfect.empty:
                                    # ✅ Apply same secondary sort rule to bottom selection
                                    non_perfect_sorted = non_perfect.sort_values(
                                        by=["Avg_Grade", "Assessments"],
                                        ascending=[True, False]
                                    )
                                    bottom_performer = non_perfect_sorted.iloc[0]

                                    bottom_block = f"""
                                    <div style='background-color:#ffebee;padding:15px;border-radius:10px;'>
                                        <b style='color:black;'>⚠️ Bottom Performer:</b> 
                                        <span style='color:black;font-weight:bold;'>{bottom_performer['Analyst Name']}</span> — 
                                        <b style='color:black;'>{bottom_performer['Avg_Grade']:.2f}</b> 
                                        <span style='color:#b71c1c;'>(Assessments: {bottom_performer['Assessments']})</span>
                                    </div>
                                    """
                                else:
                                    bottom_block = """
                                    <div style='background-color:#fff3cd;padding:15px;border-radius:10px;'>
                                        <b style='color:black;'>✅ No Bottom Performer:</b>
                                        <span style='color:black;font-weight:bold;'>All analysts have perfect scores (10.0)</span>
                                    </div>
                                    """

                                # --- Display Top / Bottom Performer Cards ---
                                st.markdown(
                                    f"""
                                    <div style='background-color:#e8f5e9;padding:15px;border-radius:10px;'>
                                        <b style='color:black;'>🏅 Top Performer:</b> 
                                        <span style='color:black;font-weight:bold;'>{top_performer['Analyst Name']}</span> — 
                                        <b style='color:black;'>{top_performer['Avg_Grade']:.2f}</b> 
                                        <span style='color:#1b5e20;'>(Assessments: {top_performer['Assessments']})</span>
                                    </div>
                                    <br>
                                    {bottom_block}
                                    """,
                                    unsafe_allow_html=True
                                )

                                # --- Detailed Table (Now includes Assessments) ---
                                with st.expander("📊 View All Analyst Ratings"):
                                    st.dataframe(
                                        avg_ratings[["Analyst Name", "Assessments", "Avg_Grade"]],
                                        hide_index=True,
                                        use_container_width=True
                                    )
                            else:
                                st.warning("No data available for the selected week or month.")

                        else:
                            st.warning("No data available for performance analysis.")

                        st.markdown("<br><hr style='margin:25px 0;border:1px solid #444;'>", unsafe_allow_html=True)
                        # --------------------
                        # --- Manual Coaching Assessment Dashboard ---
                        import altair as alt

                        st.markdown("### 📞 Manual Coaching Assessment by Contact Type")

                        if not df_filtered.empty and "Contact Type?" in df_filtered.columns:
                            df_ct = df_filtered.copy()
                            df_ct["Grade"] = pd.to_numeric(df_ct["Grade"], errors="coerce")
                            df_ct = df_ct.dropna(subset=["Contact Type?", "Analyst Name", "Grade"])

                            # --- Group by contact type and analyst ---
                            contact_summary = (
                                df_ct.groupby(["Contact Type?", "Analyst Name"])
                                .agg(Assessments=("Grade", "count"), Avg_Grade=("Grade", "mean"))
                                .reset_index()
                            )

                            # --- Overall summary (total assessments per contact type) ---
                            overall_summary = (
                                contact_summary.groupby("Contact Type?")
                                .agg(Total_Assessments=("Assessments", "sum"), Avg_Rating=("Avg_Grade", "mean"))
                                .reset_index()
                            )

                            # --- Bar chart: Total assessments by contact type ---
                            chart_summary = (
                                alt.Chart(overall_summary)
                                .mark_bar(size=45)
                                .encode(
                                    x=alt.X("Contact Type?:N", title="Contact Type",
                                            sort=["Chat", "Phone", "Self Service"]),
                                    y=alt.Y("Total_Assessments:Q", title="Number of Assessments"),
                                    color=alt.Color("Contact Type?:N", scale=alt.Scale(scheme="tealblues")),
                                    tooltip=[
                                        alt.Tooltip("Contact Type?:N", title="Contact Type"),
                                        alt.Tooltip("Total_Assessments:Q", title="Assessments"),
                                        alt.Tooltip("Avg_Rating:Q", format=".2f", title="Average Rating"),
                                    ],
                                )
                                #.properties(title="Manual Coaching Assessments by Contact Type", height=400)
                            )

                            st.altair_chart(chart_summary, use_container_width=True)

                            # --- Dashboard per contact type ---
                            st.markdown("#### Select Contact Type to View Agent Ratings:")
                            contact_types = sorted(contact_summary["Contact Type?"].unique())
                            selected_type = st.selectbox("Contact Type:", contact_types)

                            df_type = contact_summary[contact_summary["Contact Type?"] == selected_type]

                            st.markdown(f"##### Agent-wise Average Rating — {selected_type}")

                            # --- Sort by average grade ---
                            df_type = df_type.sort_values(by="Avg_Grade", ascending=False)

                            # --- Altair horizontal bar chart for agent-wise ratings ---
                            chart_agents = (
                                alt.Chart(df_type)
                                .mark_bar(size=18)
                                .encode(
                                    x=alt.X("Avg_Grade:Q", title="Average Rating", scale=alt.Scale(domain=[0, 10])),
                                    y=alt.Y("Analyst Name:N", sort="-x", title="Analyst Name"),
                                    color=alt.Color("Avg_Grade:Q", scale=alt.Scale(scheme="blues")),
                                    tooltip=[
                                        alt.Tooltip("Analyst Name", title="Analyst"),
                                        alt.Tooltip("Avg_Grade:Q", format=".2f", title="Average Rating"),
                                        alt.Tooltip("Assessments:Q", title="No. of Assessments"),
                                    ],
                                )
                                .properties(title=f"{selected_type} — Agent Average Ratings", height=400)
                            )

                            st.altair_chart(chart_agents, use_container_width=True)

                            # --- Display summary table ---
                            with st.expander(f"📋 View Detailed Data for {selected_type}"):
                                st.dataframe(df_type, use_container_width=True, hide_index=True)

                        else:
                            st.info("No contact type data available for visualization.")

                        st.markdown("<br><hr style='margin:25px 0;border:1px solid #444;'>", unsafe_allow_html=True)
                    #**********************************************************************************************************************************
                    # --- Delete controls (placed after filters) ---
                    st.subheader("Delete records")
                    st.write("You may delete all records or a range of records by SR No. Use with caution.")
    
                    del_col1, del_col2 = st.columns([1, 2])
                    # delete_mode = del_col1.selectbox("Delete option", options=["None", "Delete all records", "Delete by SR No. range"], index=0)
                    delete_mode = del_col1.selectbox("Delete option", options=["None", "Delete by SR No. range"], index=0)
                    
                    # default values
                    sr_min = 1
                    sr_max = len(df_display)
    
                    if delete_mode == "Delete by SR No. range":
                        # show inputs for from/to with safe bounds
                        from_sr = del_col2.number_input("From SR No.", min_value=1, max_value=sr_max, value=1, step=1, key="del_from_sr")
                        to_sr = del_col2.number_input("To SR No.", min_value=1, max_value=sr_max, value=sr_max, step=1, key="del_to_sr")
                    else:
                        from_sr = None
                        to_sr = None
    
                    # Safety confirmation
                    confirm = st.checkbox("I confirm I want to delete the selected records", key="delete_confirm")
    
                    # Delete button
                    if st.button("Delete records", key="delete_records_btn"):
                        if not confirm:
                            st.error("Please confirm deletion by checking the confirmation box.")
                        else:
                            try:
                                # Reload the latest file to avoid stale data
                                df_all = pd.read_excel(RECORDS_PATH, dtype=str, engine="openpyxl").fillna("")
                                df_all = df_all.reset_index(drop=True)
                                # add SR No to this fresh df for reference
                                df_all.insert(0, "SR No.", df_all.index.map(lambda x: x + 1))
    
                                # if delete_mode == "Delete all records":
                                #     # write an empty dataframe with same columns (preserve columns)
                                #     cols = [c for c in df_all.columns if c != "SR No."]  # drop SR No
                                #     empty_df = pd.DataFrame(columns=cols)
                                #     empty_df.to_excel(RECORDS_PATH, index=False, engine="openpyxl")
                                #     st.success("All records deleted.")
                                #     st.session_state["dashboard_last_refresh"] = datetime.now().isoformat(timespec="seconds")
                                # elif delete_mode == "Delete by SR No. range":
                                if delete_mode == "Delete by SR No. range":
                                    # validate inputs
                                    a = int(from_sr)
                                    b = int(to_sr)
                                    if a < 1 or b < a or b > len(df_all):
                                        st.error("Invalid SR No. range.")
                                    else:
                                        # compute zero-based indices to drop
                                        drop_idx = list(range(a - 1, b))
                                        df_after = df_all.drop(index=drop_idx).reset_index(drop=True)
                                        # remove SR No. column before saving
                                        if "SR No." in df_after.columns:
                                            df_after = df_after.drop(columns=["SR No."])
                                        # write back
                                        df_after.to_excel(RECORDS_PATH, index=False, engine="openpyxl")
                                        st.success(f"Deleted records SR No. {a} to {b}.")
                                        st.session_state["dashboard_last_refresh"] = datetime.now().isoformat(timespec="seconds")
                                else:
                                    st.info("No delete action selected.")
                            except Exception as e:
                                st.error(f"Failed to delete records: {e}")
    
                    st.markdown("---")
    
                    # --- Download filtered records ---
                    csv = df_filtered[["SR No.", "Date", "Ticket Number", "Quality Coach", "Team Leader", "Analyst Name", "Response", "Response_Comments", "Feedback_Timestamp", "Mail_Sent"]].to_csv(index=False).encode("utf-8")
                    st.download_button("Download records (CSV)", data=csv, file_name="feedback_records_filtered.csv", mime="text/csv")

if __name__ == "__main__":
    main()
