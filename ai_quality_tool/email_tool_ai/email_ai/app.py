import sys, os
sys.dont_write_bytecode = True
import io  # <--- Added for handling SharePoint streams
import pandas as pd
import streamlit as st
from datetime import datetime
import uuid

# Add the top-level jserviceops folder to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- SharePoint Imports (Added) ---
from ai_quality_tool.ai_tool_quality.modules.sharepoint_client import (
    download_feedback_records,
    upload_feedback_records,
    download_checklist,
    upload_checklist
)

# Now imports will work
from ai_tool_quality.modules.forms import run_form

# app.py
from ai_quality_tool.email_tool_ai.email_ai.modules.utils import format_body, get_email, get_lead_email, get_QC_email, get_fixed_cc
from ai_quality_tool.email_tool_ai.email_ai.modules.mailer import get_sender_name, send_mail
 
from ai_quality_tool.email_tool_ai.email_ai.modules.user_manager import (
    ensure_user_sheets,
    list_user_sheets,
    get_users,
    add_user,
    remove_user,
    SHEET_NAME_HEADER,
    EMAIL_HEADER,
)
 
st.set_page_config(page_title="📧 QM Feedback Email Automation Tool", layout="centered")

# ---------- Helpers to find project root & shared data dir ----------
def find_project_root(start_dir: str = None) -> str:
    """
    Climb up from the current file directory until a folder containing 'modules'
    is found. Return that folder path. Fallback to the directory of this file.
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

# NOTE: We avoid creating a local persistent checklist file. Instead we load the
# checklist into memory (BytesIO) from SharePoint for reads. Writes (add/remove)
# will upload the updated BytesIO back to SharePoint.
def load_checklist_stream():
    """Return a BytesIO stream of the checklist downloaded from SharePoint, or None."""
    try:
        stream = download_checklist()
        if stream:
            try:
                stream.seek(0)
            except Exception:
                pass
            return stream
    except Exception:
        # intentionally silent: failure will be handled by callers
        return None

def main():
    
    # --- Startup: Load checklist from SharePoint into session (read-only cache) ---
    if "checklist_stream" not in st.session_state:
        st.session_state["checklist_stream"] = load_checklist_stream()

    # ---------- App UI ----------
    if st.button("Back to Homepage", key="heat_back_homepage"):
        st.session_state["current_app"] = "Homepage"
        st.rerun()  # Navigate back to the homepage
        # ---------- App UI ----------
    st.title("📧 QM Feedback Email Automation Tool")

    # Re-declare for safety within main scope
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
    tabs = st.tabs(["QM_Form", "Tracking Dashboard", "Manage Users"])

    send_tab, track_tab, manage_tab = tabs

    # ---------------- Manage Users ----------------
    with manage_tab:
        st.header("Manage Users (Team Leader / Quality Coach / Analyst)")
        st.write("Add or remove users. Updates are synced to SharePoint.")
    
        sheet_ids = list_user_sheets()
        sheet_id = st.selectbox("Select user list", options=sheet_ids)
    
        # load latest checklist stream (in-memory) for reads
        checklist_stream = st.session_state.get("checklist_stream") or load_checklist_stream()
        if checklist_stream is not None:
            try:
                checklist_stream.seek(0)
            except Exception:
                pass
        users = get_users(checklist_stream, sheet_id)
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
                    # 1. Load current checklist into a BytesIO buffer
                    s = st.session_state.get("checklist_stream") or load_checklist_stream()
                    if s is not None:
                        try:
                            s.seek(0)
                        except Exception:
                            pass
                        buf = io.BytesIO(s.getbuffer())
                    else:
                        buf = io.BytesIO()

                    added, out_stream = add_user(buf, sheet_id, new_name.strip(), new_email.strip())
                    if added:
                        # 2. Upload change to SharePoint (out_stream is the updated BytesIO)
                        if out_stream is not None:
                            try:
                                out_stream.seek(0)
                            except Exception:
                                pass
                            upload_checklist(out_stream)
                            # update cached stream
                            st.session_state["checklist_stream"] = io.BytesIO(out_stream.getbuffer())
                        st.success(f"Added & Uploaded: {new_name.strip()}  —  {new_email.strip()}")
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
                    # 1. Load current checklist into a BytesIO buffer
                    s = st.session_state.get("checklist_stream") or load_checklist_stream()
                    if s is not None:
                        try:
                            s.seek(0)
                        except Exception:
                            pass
                        buf = io.BytesIO(s.getbuffer())
                    else:
                        buf = io.BytesIO()

                    removed, out_stream = remove_user(buf, sheet_id, name_part, email_part)
                    if removed:
                        # 2. Upload change to SharePoint
                        if out_stream is not None:
                            try:
                                out_stream.seek(0)
                            except Exception:
                                pass
                            upload_checklist(out_stream)
                            st.session_state["checklist_stream"] = io.BytesIO(out_stream.getbuffer())
                        st.success(f"Removed & Uploaded: {name_part} — {email_part}")
                    else:
                        st.warning("User not found or could not remove.")
                except Exception as e:
                    st.error(f"Failed to remove user: {e}")
                st.rerun()
        else:
            st.info("No users to remove.")

        # ---------------- Send Emails ----------------
    with send_tab:
        st.header("Fill Feedback Form")
        # Calling the form UI
        submitted_data = run_form()
        if submitted_data:
            # The `run_form()` already saves Sheet1 + Score to SharePoint.
            # Avoid double-writing here which would create duplicate rows and
            # mismatched columns. Only send the email and show success.
            row = submitted_data  # dictionary returned from form
            send_single_feedback_email(row)

            st.success("Form submitted and email sent successfully.")

    # ---------------- Tracking Dashboard ----------------
    with track_tab:
        st.header("Tracking Dashboard — Feedback records")
        st.write("This dashboard reads the feedback-records file from SharePoint.")
    
        RECORDS_PATH = globals().get("RECORDS_PATH") or globals().get("FEEDBACK_RECORDS_PATH")
        if not RECORDS_PATH:
            st.error("No records path defined (RECORDS_PATH or FEEDBACK_RECORDS_PATH).")
        else:
            c1, c2, c3 = st.columns([1, 2, 3])
            refresh_clicked = c1.button("🔄 Refresh", key="dashboard_refresh_btn")
            if refresh_clicked:
                st.session_state["dashboard_last_refresh"] = datetime.now().isoformat(timespec="seconds")
            
            # --- [Modified] Download Latest Records from SharePoint ---
            try:
                stream = download_feedback_records()
                if stream:
                    with open(RECORDS_PATH, "wb") as f:
                        f.write(stream.getbuffer())
            except Exception as e:
                # If sync fails, we might still have a local file to show
                st.warning(f"Could not sync from SharePoint: {e}")

            last_refresh = st.session_state.get("dashboard_last_refresh")
            if last_refresh:
                c2.markdown(f"**Last refresh:** {last_refresh}")
    
            try:
                if os.path.exists(RECORDS_PATH):
                    mtime = os.path.getmtime(RECORDS_PATH)
                    file_mod = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    c3.markdown(f"**File last modified:** {file_mod}")
            except Exception:
                c3.markdown("**File last modified:** —")
    
            st.markdown("---")
    
            if not os.path.exists(RECORDS_PATH):
                st.info("No feedback records found.")
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
                    # Ensure expected feedback/meta columns exist to avoid KeyErrors
                    for col in ["Response", "Response_Comments", "Feedback_Timestamp", "Token", "Token_Used", "Mail_Sent"]:
                        if col not in df_display.columns:
                            df_display[col] = ""
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
                    # Ensure df_filtered exists as a working copy (used later by visuals)
                    df_filtered = df_display.copy()

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
                            if "Date" not in df_temp.columns:
                                st.warning("Cannot perform performance analysis: no 'Date' column present.")
                                df_temp = pd.DataFrame()
                            else:
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
                                # [Modified] Reload the latest file from SharePoint to avoid stale data
                                s_del = download_feedback_records()
                                if s_del:
                                    s_del.seek(0)
                                    df_all = pd.read_excel(s_del, dtype=str, engine="openpyxl").fillna("")
                                else:
                                    df_all = pd.read_excel(RECORDS_PATH, dtype=str, engine="openpyxl").fillna("")

                                df_all = df_all.reset_index(drop=True)
                                # add SR No to this fresh df for reference
                                df_all.insert(0, "SR No.", df_all.index.map(lambda x: x + 1))

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
                                        
                                        # [Modified] Write back to Stream and Upload
                                        out_s = io.BytesIO()
                                        df_after.to_excel(out_s, index=False, engine="openpyxl")
                                        upload_feedback_records(out_s)
                                        
                                        # Also update local file for immediate UI refresh
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

def append_one_row_to_feedback_records(row_dict):
    import pandas as pd
    import uuid
    import os

    # [Modified] 1. Download existing records from SharePoint
    try:
        stream = download_feedback_records()
        if stream:
            stream.seek(0)
            df = pd.read_excel(stream, dtype=str, engine="openpyxl").fillna("")
        else:
            df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    # Ensure required meta columns exist
    for col in ["Response", "Response_Comments", "Feedback_Timestamp", "Token", "Token_Used", "Mail_Sent"]:
        if col not in df.columns:
            df[col] = ""

    # Prepare new row dictionary following EXACT SAME COLUMNS
    new_row = {}

    for col in df.columns:
        if col in row_dict:
            new_row[col] = row_dict[col]
        else:
            new_row[col] = ""

    # Token if needed
    if not new_row.get("Token"):
        new_row["Token"] = uuid.uuid4().hex

    # Convert into DataFrame
    df_new = pd.DataFrame([new_row])

    # Append
    df = pd.concat([df, df_new], ignore_index=True)

    # [Modified] 2. Save to Stream and Upload to SharePoint
    try:
        out_stream = io.BytesIO()
        df.to_excel(out_stream, index=False, engine="openpyxl")
        upload_feedback_records(out_stream)
        
        # Also save locally for backup/consistency
        df.to_excel(FEEDBACK_RECORDS_PATH, index=False, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to upload records: {e}")


def send_single_feedback_email(row_dict):
    import pandas as pd

    # Convert dict → Series for format_body()
    row = pd.Series(row_dict)

    try:
        sender = get_sender_name()
    except Exception:
        sender = ""

    analyst_name = row.get("Analyst Name", "").strip()
    if not analyst_name:
        st.error("Cannot send email — Analyst Name missing.")
        return False

    # Use checklist from session cache or download fresh from SharePoint
    checklist_stream = st.session_state.get("checklist_stream") or load_checklist_stream()
    if checklist_stream is not None:
        try:
            checklist_stream.seek(0)
        except Exception:
            pass
    to_email = get_email(analyst_name, checklist_stream, sheet_name="Analyst_Name")
    if not to_email:
        st.error(f"No email found for analyst {analyst_name}.")
        return False

    lead_name = row.get("Team Leader", "")
    QC_name = row.get("Quality Coach", "")

    cc_lead = get_lead_email(lead_name, checklist_stream, sheet_name="Team_Leader")
    cc_qc = get_QC_email(QC_name, checklist_stream, sheet_name="Quality_coach")
    fixed_cc = get_fixed_cc(checklist_stream, sheet_name="Fixed_CC")

    subject = f"Feedback for Analyst {analyst_name}"

    # Now this works because row is Series
    body_html = format_body(row, sender)

    send_mail(
        to_email,
        subject,
        body_html,
        cc=[cc_lead, cc_qc] + fixed_cc
    )

    # --- After sending, mark Mail_Sent in feedback records (by Token) ---
    try:
        token = str(row.get("Token", "")).strip()
        # Download latest records
        s = download_feedback_records()
        if s:
            s.seek(0)
            df_all = pd.read_excel(s, dtype=str, engine="openpyxl").fillna("")
        elif os.path.exists(RECORDS_PATH):
            df_all = pd.read_excel(RECORDS_PATH, dtype=str, engine="openpyxl").fillna("")
        else:
            df_all = pd.DataFrame()

        if not df_all.empty and token:
            # find matching row by Token (exact match). Fall back to Ticket Number+Timestamp.
            mask = df_all.get("Token", "").astype(str).str.strip() == token
            if not mask.any():
                # fallback using Ticket Number + Timestamp
                tn = str(row.get("Ticket Number", "")).strip()
                ts = str(row.get("Timestamp", "")).strip()
                mask = (df_all.get("Ticket Number", "").astype(str).str.strip() == tn) & (df_all.get("Timestamp", "").astype(str).str.strip() == ts)

            if mask.any():
                # nowstr = datetime.now().isoformat(timespec="seconds")
                df_all.loc[mask, "Mail_Sent"] = "Yes"
                # ensure Token exists
                if "Token" in df_all.columns:
                    df_all.loc[mask, "Token"] = token

                # write back
                out_s = io.BytesIO()
                df_all.to_excel(out_s, index=False, engine="openpyxl")
                try:
                    out_s.seek(0)
                except Exception:
                    pass
                upload_feedback_records(out_s)
                # also save local copy
                try:
                    df_all.to_excel(RECORDS_PATH, index=False, engine="openpyxl")
                except Exception:
                    pass
    except Exception:
        # non-fatal
        pass

    return True

if __name__ == "__main__":
    main()