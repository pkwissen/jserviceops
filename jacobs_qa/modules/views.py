# File: modules/views.py
import streamlit as st
import pandas as pd
import plotly.express as px
from jacobs_qa.modules import tracker as tr
from io import BytesIO


def render_download_button(month: str, week_n: int):
    """Show a persistent download button if a tracker is present."""
    if "tracker" in st.session_state:
        df = st.session_state["tracker"]
        out = BytesIO()

        # Set the sheet name here
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Coaching Tracker")

        st.download_button(
            label="Download Coaching Assessment Tracker (Excel)",
            data=out.getvalue(),
            file_name=f"Coaching_Assessment_Tracker_{month}_W{week_n}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_tracker"
        )

def render_treemap(tracker_df: pd.DataFrame):
    import plotly.express as px
    _tmp = tracker_df.copy()
    _tmp["Quality Parameter Category"] = _tmp.get("Quality Parameter Category", "").astype(str).str.strip()
    _tmp = _tmp[_tmp["Quality Parameter Category"].notna()]
    _tmp = _tmp[_tmp["Quality Parameter Category"].str.strip().ne("")]
    if _tmp.empty:
        st.info("No missing quality parameters to visualize for this week.")
        return
    category_counts = (
        _tmp["Quality Parameter Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Count")
    )
    category_counts["Label"] = category_counts.apply(lambda r: f"{r['Category']}, {int(r['Count'])}", axis=1)
    fig = px.treemap(category_counts, path=["Label"], values="Count", title="Missing Quality Parameters (Auto-Categorized)")
    fig.update_traces(hovertemplate="<b>%{label}</b><extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

def show_quick_views():
    tracker = st.session_state.get("tracker", pd.DataFrame())
    if tracker is None or tracker.empty:
        st.info("No tracker available. Generate a tracker or upload a saved tracker.")
        return

    # Ensure categories exist (compute only blanks)
    if (
        "Quality Parameter Category" not in tracker.columns
        or tracker["Quality Parameter Category"].isnull().any()
        or (tracker["Quality Parameter Category"].astype(str).str.strip() == "").any()
    ):
        tracker = tr.compute_categories(tracker)
        st.session_state["tracker"] = tracker  # persist filled categories

    # Category filter view
    st.markdown("### 📂 View by Quality Parameter Category")
    cats = sorted([c for c in tracker["Quality Parameter Category"].dropna().unique() if str(c).strip() != ""])
    if cats:
        sel_cat = st.selectbox("Select a Quality Parameter Category", ["(all)"] + cats, key="cat_select")
        if sel_cat and sel_cat != "(all)":
            cat_df = tracker[tracker["Quality Parameter Category"] == sel_cat].copy()
            st.subheader(f"🔎 Records for category: {sel_cat}")
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
            st.subheader(f"⚠️ Analysts with Missing Criteria in {sel_cat}")
            missing_df = cat_df[["Employee Name", "Team Lead", "Quality Assessors", "Areas to Improve"]]
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.info("✅ No analysts missing this criteria.")
        else:
            st.dataframe(tracker, use_container_width=True, hide_index=True)
    else:
        st.info("No categorized records found.")
        st.dataframe(tracker, use_container_width=True, hide_index=True)

    # Quick Views buttons
    st.markdown("### Quick Views")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🏆 View Top Performer (Selected Week)", key="btn_top_week"):
            st.session_state["view"] = "top_week"
    with col_b:
        if st.button("🏅 View Top Performer (Selected Month)", key="btn_top_month"):
            st.session_state["view"] = "top_month"
    with col_c:
        if st.button("⚠️ Agents with Missing Criteria (Improvements Needed)", key="btn_improve"):
            st.session_state["view"] = "needs_improve"

    # Render selected view
    view = st.session_state.get("view")
    if view == "top_week":
        wk_df = st.session_state["tracker"].copy()
        if "__score__" not in wk_df.columns:
            wk_df["__score__"] = wk_df.apply(tr._row_perf_score, axis=1)
        wk_df["_rank"] = wk_df["__score__"].rank(method="min", ascending=False)
        top_score = wk_df["__score__"].max()
        top_week = wk_df[wk_df["__score__"] == top_score].sort_values(["__score__", "Employee Name"], ascending=[False, True])
        st.subheader("🏆 Top Performer — Selected Week")
        st.dataframe(top_week.drop(columns=["_rank"], errors="ignore"), use_container_width=True, hide_index=True)
        if st.button("◀ Back", key="back_from_week"):
            st.session_state["view"] = None

    elif view == "top_month":
        # Build month-to-date tracker from available weekly trackers if present
        month_tracker = None
        wk_list = st.session_state.get("weekly_trackers", [])
        if wk_list:
            with st.spinner("Building month-to-date tracker from available weeks…"):
                try:
                    month_tracker = pd.concat(wk_list, ignore_index=True)
                    if "__score__" not in month_tracker.columns:
                        month_tracker["__score__"] = month_tracker.apply(tr._row_perf_score, axis=1)
                    # aggregate per employee
                    month_tracker = month_tracker.groupby("Employee Name", as_index=False).agg({"__score__": "sum"})
                except Exception as e:
                    st.error(f"Could not build month-to-date tracker from weekly trackers: {e}")
                    month_tracker = None
        elif "uploaded_tracker_df" in st.session_state:
            with st.spinner("Aggregating from uploaded coaching tracker…"):
                month_tracker = st.session_state["uploaded_tracker_df"].copy()
                if "__score__" not in month_tracker.columns:
                    month_tracker["__score__"] = month_tracker.apply(tr._row_perf_score, axis=1)
                month_tracker = month_tracker.groupby("Employee Name", as_index=False).agg({"__score__": "sum"})
        else:
            # fallback: use current tracker as the only available week
            with st.spinner("Using current weekly tracker as month-to-date (only available week)…"):
                month_tracker = st.session_state["tracker"].copy()
                if "__score__" not in month_tracker.columns:
                    month_tracker["__score__"] = month_tracker.apply(tr._row_perf_score, axis=1)
                month_tracker = month_tracker.groupby("Employee Name", as_index=False).agg({"__score__": "sum"})

        if month_tracker is not None and not month_tracker.empty:
            top_score = month_tracker["__score__"].max()
            top_month = month_tracker[month_tracker["__score__"] == top_score].sort_values(["__score__", "Employee Name"], ascending=[False, True])
            st.subheader("🏅 Top Performer — Month-to-Date (Based on Available Weeks)")
            st.dataframe(top_month.drop(columns=["__score__"], errors="ignore"), use_container_width=True, hide_index=True)
        else:
            st.info("No data available to compute month-to-date performer. Generate at least one weekly tracker or upload a saved tracker.")
        if st.button("◀ Back", key="back_from_month"):
            st.session_state["view"] = None

    elif view == "needs_improve":
        tracker_df = st.session_state.get("tracker", pd.DataFrame()).copy()
        if tracker_df is None or tracker_df.empty:
            st.info("No tracker available.")
        else:
            # 1️⃣ Ensure score column exists
            if "__score__" not in tracker_df.columns:
                tracker_df["__score__"] = tracker_df.apply(_row_perf_score, axis=1)
            # 2️⃣ Filter: has category, score not blank, score != 10
            need_improve = tracker_df[
                tracker_df["Quality Parameter Category"].notna()
                & tracker_df["Quality Parameter Category"].astype(str).str.strip().ne("")
                & tracker_df["__score__"].notna()
                & (tracker_df["__score__"] != 10)
                ].copy()
            if need_improve.empty:
                st.success("✅ No agents need improvement.")
            else:
                # 3️⃣ Sort ascending by score so weakest agents are at top
                need_improve = need_improve.sort_values("__score__", ascending=True)
                st.subheader("⚠️ Agents Needing Improvement (Sorted by Score)")
                st.dataframe(
                    need_improve[
                        ["Employee Name", "Team Lead", "Quality Assessors",
                         "Areas to Improve", "Quality Parameter Category", "__score__"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            if st.button("◀ Back", key="back_from_needs"):
                st.session_state["view"] = None
