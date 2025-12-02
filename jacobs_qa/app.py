# File: app.py
import io
import datetime as dt
import pandas as pd
import streamlit as st

from jacobs_qa.modules import data_handler as dh
from jacobs_qa.modules import tracker as tr
from jacobs_qa.modules import views

APP_TITLE = "Weekly Coaching Assessment Tracker"
AREAS_TO_IMPROVE_EXAMPLES = [
    "Probe for full issue context before troubleshooting.",
    "Summarize resolution and confirm next steps with the customer.",
    "Reduce hold time by using knowledge base bookmarks.",
    "Follow escalation matrix within SLA when blockers occur.",
    "Ensure ticket notes are complete: symptoms, root cause, and fix.",
]

st.set_page_config(page_title=APP_TITLE, layout="wide")


def _find_sheet(xls: pd.ExcelFile, target: str):
    for s in xls.sheet_names:
        if target.lower() in s.lower():
            return s
    return None


def main():
    tr.ensure_session_defaults()

    if st.button("Back to Homepage", key="heat_back_homepage"):
        st.session_state["current_app"] = "Homepage"
        st.rerun()

    st.title(APP_TITLE)
    st.caption(
        "Upload weekly report → pick Month/Week → Generate tracker → Download as Excel. "
        "Optional: set GROQ_API_KEY in environment for better 'Areas to Improve'. "
        "Or upload an already saved tracker to review."
    )

    # month map MUST be available for all branches -> define before sidebar
    month_name_to_num = {m: i for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

    with st.sidebar:
        st.header("Inputs")
        weekly_file = st.file_uploader(
            "Upload Weekly Coaching Assessment Report (.xlsx)",
            type=["xlsx"], accept_multiple_files=False, key="sidebar_weekly",
        )
        st.write(":blue[Sheets required: 'Manual Assessments Data' and 'ServiceNow Coaching Assessment']")

        saved_tracker_file = st.file_uploader(
            "📂 (Optional) Upload an already saved Coaching Assessment Tracker (.xlsx)",
            type=["xlsx"], accept_multiple_files=False,
            help="Use this if you already have a generated tracker and want to view/analyze it.",
            key="sidebar_saved",
        )

        # keep the selectboxes in sidebar and store values in session_state
        month = st.selectbox("Month", list(month_name_to_num.keys()),
                             index=dt.date.today().month - 1, key="sidebar_month")
        week = st.selectbox("Week of month", [1, 2, 3, 4, 5],
                            index=min(4, dh.week_of_month(dt.date.today()) - 1), key="sidebar_week")
        do_generate = st.button("Generate Tracker", type="primary", key="sidebar_generate")

    st.warning("Emp ID, Organization, and Status are intentionally left blank.")

    # ----- Handle uploaded saved tracker FIRST (if present) -----
    if saved_tracker_file:
        try:
            tracker = pd.read_excel(saved_tracker_file, sheet_name="Coaching Tracker").drop(columns=["Unnamed: 0"], errors="ignore")
            st.session_state["tracker"] = tracker
            st.session_state["uploaded_tracker_df"] = tracker.copy()
            st.success(f"Loaded saved tracker. Rows: {len(tracker)}")
            st.dataframe(tracker, use_container_width=True, hide_index=True)
            # show treemap and quick views (common)
            views.render_treemap(tracker)
            views.show_quick_views()
            return
        except Exception as e:
            st.error("Could not read uploaded tracker.")
            st.exception(e)
            st.stop()

    # ----- Handle weekly report generation -----
    if 'weekly_file' in locals() and weekly_file:
        try:
            xls = pd.ExcelFile(weekly_file)
            manual_sheet = _find_sheet(xls, "Manual Assessments Data") or _find_sheet(xls, "Manual") or _find_sheet(xls, "Assessments")
            sn_sheet = _find_sheet(xls, "ServiceNow Coaching Assessment") or _find_sheet(xls, "ServiceNow") or _find_sheet(xls, "Coaching")
            if not manual_sheet or not sn_sheet:
                st.error("Could not find required sheets. Ensure workbook has 'Manual Assessments Data' and 'ServiceNow Coaching Assessment'.")
                st.stop()
            manual_df = pd.read_excel(xls, manual_sheet).drop(columns=["Unnamed: 0"], errors="ignore")
            sn_df = pd.read_excel(xls, sn_sheet).drop(columns=["Unnamed: 0"], errors="ignore")
        except Exception as e:
            st.exception(e)
            st.stop()

        seed_examples = AREAS_TO_IMPROVE_EXAMPLES.copy()
        if 'sample_tracker' in locals() and sample_tracker:
            try:
                st.caption("Using uploaded tracker to seed Areas to Improve examples.")
                df_seed = pd.read_excel(sample_tracker)
                col = dh.pick_first_available_column(df_seed, ["Areas to Improve", "Areas", "Improvements"])
                if col and not df_seed.empty:
                    vals = [str(v) for v in df_seed[col].dropna().astype(str).tolist()][:20]
                    cleaned = []
                    for v in vals:
                        for line in str(v).splitlines():
                            line = line.strip("- • ").strip()
                            if 10 <= len(line) <= 160:
                                cleaned.append(line)
                    if cleaned:
                        seed_examples = cleaned[:20]
            except Exception:
                pass

        # compute numeric month/week values from the sidebar selections
        month_n = month_name_to_num.get(st.session_state.get("sidebar_month"), dt.date.today().month)
        week_n = int(st.session_state.get("sidebar_week", dh.week_of_month(dt.date.today())))

        if do_generate:
            with st.spinner("Generating tracker…"):
                tracker = tr.generate_tracker(manual_df, sn_df, month_n, week_n, seed_examples)
            if ("Quality Parameter Category" not in tracker.columns or
                tracker["Quality Parameter Category"].isnull().any() or
                (tracker["Quality Parameter Category"].astype(str).str.strip() == "").any()):
                tracker = tr.compute_categories(tracker)
            if "__score__" not in tracker.columns:
                tracker["__score__"] = tracker.apply(tr._row_perf_score, axis=1)
            st.session_state["tracker"] = tracker
            st.session_state["manual_df_raw"] = manual_df.copy()
            st.session_state["sn_df_raw"] = sn_df.copy()
            tr.register_weekly_tracker(tracker)
            st.success(f"Tracker generated for {st.session_state.get('sidebar_month')} – Week {week_n}. Rows: {len(tracker)}")
            st.dataframe(tracker, use_container_width=True, hide_index=True)

            views.render_treemap(tracker)
            views.show_quick_views()
            return

    # If no files/actions: show last tracker if present
    current_tracker = st.session_state.get("tracker")
    if isinstance(current_tracker, pd.DataFrame) and not current_tracker.empty:
        st.info("Showing last generated/loaded tracker from this session.")
        st.dataframe(current_tracker, use_container_width=True, hide_index=True)
        views.render_treemap(current_tracker)
        views.show_quick_views()
        # render download button for existing tracker
        # compute safe month/week from session (defaults used if missing)
        safe_month_n = month_name_to_num.get(st.session_state.get("sidebar_month"), dt.date.today().month)
        safe_week_n = int(st.session_state.get("sidebar_week", dh.week_of_month(dt.date.today())))
        views.render_download_button(safe_month_n, safe_week_n)
        return

    st.info("Upload a Weekly Coaching Report or an already saved Tracker to begin.")

    # Always render download button if tracker exists (use safe defaults)
    if "tracker" in st.session_state:
        safe_month_n = month_name_to_num.get(st.session_state.get("sidebar_month"), dt.date.today().month)
        safe_week_n = int(st.session_state.get("sidebar_week", dh.week_of_month(dt.date.today())))
        views.render_download_button(safe_month_n, safe_week_n)

if __name__ == "__main__":
    main()
