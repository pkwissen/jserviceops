# app.py
import sys
import os
import subprocess
import streamlit as st
from pathlib import Path
import pandas as pd
# from jacobs_aiops_project.forecast_app_final.modules import shift_avail_calculate
from forecast_app_final.modules import data_handler, forecaster, planner, shift_plan,shift_avail_calculate
import webbrowser
from datetime import date, timedelta

# -----------------------------
# 🧠 Safe base path handling
# -----------------------------
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_PATH = DATA_DIR / "processed_data.xlsx"
TRANSFORMED_PATH = DATA_DIR / "transformed_data.xlsx"
TRANSFORMED_PHONE59_PATH = DATA_DIR / "transformed_Phone59.xlsx"
FORECAST_DATA_PATH = DATA_DIR / "forecast_data.xlsx"

# 👇 Add modules directory to import path (keeps original behaviour)
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "modules"))
sys.path.insert(0, str(BASE_DIR / "prophet"))

# -----------------------------
# Try importing the summary & shift modules (defensive)
# -----------------------------
try:
    from modules.summary_generator import generate_summary_table, save_summary_to_excel, save_summary_with_formatting
except Exception:
    try:
        from modules.summary_generator import generate_summary_table, save_summary_to_excel
        save_summary_with_formatting = None
    except Exception:
        generate_summary_table = None
        save_summary_to_excel = None
        save_summary_with_formatting = None


# -----------------------------
# 🚀 Auto-launch via streamlit run (if frozen)
# -----------------------------
if getattr(sys, 'frozen', False):
    script_path = str(BASE_DIR / 'app.py')
    webbrowser.open("http://localhost:8501")
    subprocess.Popen(["streamlit", "run", script_path])
    sys.exit()

# -----------------------------
# Helper: find latest Summary_*.xlsx from expected folders
# -----------------------------
def find_latest_summary_file() -> Path | None:
    candidates = []
    out_dir = BASE_DIR / "forecast_app" / "output"
    if out_dir.exists():
        candidates.extend(list(out_dir.glob("Summary_*.xls*")))
    if DATA_DIR.exists():
        candidates.extend(list(DATA_DIR.glob("Summary_*.xls*")))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest

# Helper: save DataFrame to a temp summary file for shift module to consume
def save_session_summary_temp(df: pd.DataFrame) -> str:
    out_dir = BASE_DIR / "forecast_app" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "session_summary_for_shift.xlsx"
    df.to_excel(tmp, index=False)
    return str(tmp)

# Helper: find first column matching any of keywords (exact then substring)
def find_col(cols, keywords):
    for k in keywords:
        for c in cols:
            if k.lower() == str(c).strip().lower():
                return c
    for c in cols:
        low = str(c).lower()
        for k in keywords:
            if k.lower() in low:
                return c
    return None

# Excel column letter helper (0-indexed)
def col_idx_to_excel(col_idx: int) -> str:
    """Convert zero-based column index to Excel column letters (0->A, 25->Z, 26->AA)."""
    letters = ""
    i = int(col_idx)
    while i >= 0:
        letters = chr((i % 26) + 65) + letters
        i = i // 26 - 1
    return letters

# small helper to compute column widths for autofit
def compute_column_widths(df: pd.DataFrame):
    widths = []
    for col in df.columns:
        max_len = max(
            len(str(col)),
            *(len(str(v)) for v in df[col].astype(str).values[:1000])  # limit to 1000 for speed
        )
        widths.append(min(max_len + 2, 50))
    return widths

def main():
    if st.button("Back to Homepage", key="heat_back_homepage"):
        st.session_state["current_app"] = "Homepage"
        st.rerun()  # Navigate back to the homepage

    st.set_page_config(page_title="Ticket Volume Heat Map", layout="wide")
    st.title("Ticket Volume Heat Map")

    # --- Option Selection ---
    st.markdown("### Select Option to Create Heatmap")
    heatmap_option = st.selectbox(
        "Choose from the options below:",
        options=[
            "Select option from below to create heatmap",
            "Service Now",
            "Service Now - Five9 together"
        ],
        index=0
    )

    if heatmap_option == "Select option from below to create heatmap":
        st.info("ℹ️ Please select a valid option to proceed.")
        st.stop()

    # --- Ask for data upload ---
    use_new_data = st.radio(
        "Do you want to upload new raw data?",
        options=["Select...", "Yes", "No"],
        index=0
    )

    uploaded_file1 = uploaded_file2 = processed_df = xls = None

    if use_new_data == "Select...":
        st.info("ℹ️ Please select Yes or No to continue.")
        st.stop()

    # Session state setup (keeps previous keys + added summary & shift caches)
    st.session_state.setdefault('forecast', None)
    st.session_state.setdefault('plans', None)
    st.session_state.setdefault('show_shift_summary', False)
    st.session_state.setdefault('show_hourly_distribution', False)

    # NEW: persist summary & shift table across reruns
    st.session_state.setdefault('summary_table', None)
    st.session_state.setdefault('summary_path', None)
    st.session_state.setdefault('shift_table', None)

    # ensure summary/hourly cached in session_state if previously generated
    summary_df = st.session_state.get('summary_df', None)
    hourly_df = st.session_state.get('hourly_df', None)
    filtered_summary_df = st.session_state.get('filtered_summary_df', None)

    # File paths (use Path objects everywhere)
    processed_path = PROCESSED_PATH
    transformed_path = TRANSFORMED_PATH

    # ------------- Handle Uploads Based on Option Selected ---------------

    if use_new_data == "Yes":
        # Always ask for Service Now data
        uploaded_file1 = st.file_uploader("Upload new raw data from Service Now (.xlsx)", type=["xlsx"])
        if uploaded_file1:
            try:
                st.success("✅ File uploaded successfully from Service Now!")
                raw_df = data_handler.transform_new_data(uploaded_file1, TRANSFORMED_PATH)
                processed_df = data_handler.merge_with_existing(transformed_path, processed_path)
                xls = pd.ExcelFile(processed_path)
            except Exception as e:
                st.error(f"❌ Error processing Service Now file: {e}")
                st.stop()
        else:
            st.warning("⚠️ Please upload Service Now file to continue.")
            st.stop()

        # Only if both are selected, ask for Five9 file
        if heatmap_option == "Service Now - Five9 together":
            uploaded_file2 = st.file_uploader("Upload new raw data from Five9 (.csv)", type=["csv"])
            if uploaded_file2:
                try:
                    st.success("✅ File uploaded successfully from Five9!")
                    raw_df = data_handler.transform_data_59(uploaded_file2, TRANSFORMED_PHONE59_PATH)
                    processed_df = data_handler.add_transformed_phone59_sheet(PROCESSED_PATH, TRANSFORMED_PHONE59_PATH, sheet_name="Phone59")
                    xls = pd.ExcelFile(processed_path)
                except Exception as e:
                    st.error(f"❌ Error processing Five9 file: {e}")
                    st.stop()
            else:
                st.warning("⚠️ Please upload Five9 file to continue.")
                st.stop()

    elif use_new_data == "No":
        if not processed_path.exists():
            st.error(f"❌ No processed data found at {processed_path}")
            st.stop()
        try:
            xls = pd.ExcelFile(processed_path)
        except Exception as e:
            st.error(f"❌ Failed to load processed file: {e}")
            st.stop()

    # --- Forecast Period Selection ---
    st.markdown("### Select Forecasting Period")

    latest_data_date = xls.parse("Chat")["Date"].max().date()
    earliest_data_date = xls.parse("Chat")["Date"].min().date()
    today = pd.Timestamp.today().date()
    future_limit = today + pd.Timedelta(days=120)

    min_selectable = min(earliest_data_date, today)
    max_selectable = max(latest_data_date, future_limit)

    default_start = max(min_selectable, today)
    default_end = min(default_start + pd.Timedelta(days=7), max_selectable)

    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=min_selectable,
        max_value=max_selectable
    )
    end_date = st.date_input(
        "End Date",
        value=default_end,
        min_value=start_date,
        max_value=max_selectable
    )

    # save selections globally and to session (so our generator can use them)
    GLOBAL_HEATMAP_OPTION = heatmap_option
    GLOBAL_START_DATE = start_date
    GLOBAL_END_DATE = end_date

    st.session_state['global_heatmap_option'] = GLOBAL_HEATMAP_OPTION
    st.session_state['global_start_date'] = GLOBAL_START_DATE
    st.session_state['global_end_date'] = GLOBAL_END_DATE

    tasks_per_resource = {"Chat": 13, "Phone": 13, "Phone59": 13, "Self-service": 15}

    # Run Forecast Button
    run_forecast = st.button("🚀 Run Forecast & Generate Heat Map", use_container_width=True)

    if run_forecast:
        st.info(f"📊 Forecasting for **{start_date.strftime('%b %d')}** to **{end_date.strftime('%b %d')}**")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        progress = st.progress(0, text="🚀 Starting forecasting...")

        forecast_dfs = []
        # Set channels dynamically based on selected heatmap option
        if heatmap_option == "Service Now":
            channels = ["Chat", "Phone", "Self-service"]
        elif heatmap_option == "Service Now - Five9 together":
            channels = ["Chat", "Phone59", "Self-service"]

        all_dates = pd.date_range(start=start_dt, end=end_dt).date

        for i, channel in enumerate(channels):
            channel_df = xls.parse(channel)
            channel_df["Date"] = pd.to_datetime(channel_df["Date"]).dt.date

            # Check if all required dates are present in the data
            available_dates = set(channel_df["Date"].unique())
            missing_dates = [d for d in all_dates if d not in available_dates]

            if not missing_dates:
                # All dates present, use existing data for this period
                st.success(f"✅ Using existing data for {channel} ({start_date} to {end_date})")
                period_df = channel_df[channel_df["Date"].isin(all_dates)].copy()
                period_df["Channel"] = channel
                forecast_dfs.append(period_df)
            else:
                # Some dates missing, run forecasting for this channel
                progress.progress(int((i + 1) / len(channels) * 80), text=f"⏳ Forecasting {channel}...")
                df = forecaster.run_forecasting(channel_df, start_dt, end_dt)
                df["Channel"] = channel
                forecast_dfs.append(df)

        full_forecast = pd.concat(forecast_dfs, ignore_index=True)
        st.session_state['forecast'] = full_forecast

        progress.progress(90, text="🗕️ Forecasting complete. Generating Heat Map...")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        shift_plans = planner.generate_shift_plan(full_forecast, tasks_per_resource=tasks_per_resource, forecast_start=start_dt, forecast_end=end_dt)
        st.session_state['plans'] = shift_plans
        progress.progress(100, text="✅ All done!")
        planner.apply_coloring_and_download(shift_plans, start_date, end_date, backup_path=FORECAST_DATA_PATH, heatmap_option=heatmap_option)


    # Show saved heatmap if available
    if st.session_state['plans'] is not None and not run_forecast:
        st.markdown("### Showing Heat Map")
        planner.apply_coloring_and_download(st.session_state['plans'], start_date, end_date, heatmap_option=heatmap_option)

    # -------------------------
    # SHIFT-WISE AVAILABILITY (moved to show BEFORE summary)
    # -------------------------
    st.markdown("## 🧭 Shift-Wise Availability Table")

    # Analyst input for shift calculation (re-uses summary input later)
    analyst_daily_task_completion = st.number_input(
        "Analyst Daily Task Completion (tickets per analyst per day)",
        min_value=1, value=12, step=1,
        help="How many tickets one analyst completes in a day (default 10)."
    )
    team_size = st.number_input(
        "Team Size (number of analysts per team)",
        min_value=1, value=15, step=1,
        help="Team size used to compute 'Available' = Teams available X Team Size."
    )
    if 'hourly_df' not in st.session_state or st.session_state['hourly_df'] is None:
                        try:
                            _s, _h = shift_plan.main_pipeline(FORECAST_DATA_PATH)
                            st.session_state['summary_df'] = _s
                            st.session_state['hourly_df'] = _h
                        except Exception:
                            pass

    hourly_df_local = st.session_state.get('hourly_df', None)


    if st.button("🔁 Generate Shift-Wise Availability", use_container_width=True):
        res = generate_summary_table(
            forecast_data_path=str(FORECAST_DATA_PATH),
            heatmap_option=heatmap_option,
            start_dt=start_date,
            end_dt=end_date,
            hourly_df=hourly_df_local,
            analyst_daily_task_completion=int(analyst_daily_task_completion),
            team_size=int(team_size),
            diagnostics=False
        )

        with pd.ExcelWriter(DATA_DIR / "Summary.xlsx", engine="xlsxwriter") as writer:
            res.to_excel(writer, sheet_name="Summary", index=False)

        summary_candidates = list(Path(DATA_DIR).glob("Summary_*.xls*")) + list(Path(DATA_DIR).glob("Summary.xlsx"))
        summary_path = summary_candidates[0] if summary_candidates else None
        shift_avail = shift_avail_calculate.shift_availability(summary_path)

        with pd.ExcelWriter(DATA_DIR / "shift_avail.xlsx", engine="xlsxwriter") as writer:
            shift_avail.to_excel(writer, sheet_name="ShiftWise", index=False)
        
        st.session_state['shift_avail'] = shift_avail  # <-- Add to session state

    # Always show shift_avail if present in session state
    if st.session_state.get('shift_avail') is not None:
        st.dataframe(st.session_state['shift_avail'], use_container_width=True, hide_index=True)

    st.markdown("## 🧭 Summary Table")

    if st.button("📑 Generate Summary Table", use_container_width=True):
        summary_candidates = list(Path(DATA_DIR).glob("Summary_*.xls*")) + list(Path(DATA_DIR).glob("Summary.xlsx"))
        summary_path = summary_candidates[0] if summary_candidates else None
        summary_display = pd.read_excel(summary_path) if summary_path else None
        st.session_state['summary_display'] = summary_display  # <-- Add to session state
       
    # Always show summary_display if present in session state
    if st.session_state.get('summary_display') is not None:
        st.dataframe(st.session_state['summary_display'], use_container_width=True, hide_index=True)

    # -------------------------
    # HeatMap (Single Sheet) - create/download button (existing behavior)
    # -------------------------
    st.markdown("## HeatMap (Single Sheet)")

    if st.button("📑 Create and download HeatMap", use_container_width=True):
        from modules.combine_forecast_tables import combine_forecast_tables
        combined_output_path = DATA_DIR / f"combined_forecast_{heatmap_option.replace(' ', '_')}_{start_date.strftime('%b%d')}_{end_date.strftime('%b%d')}.xlsx"
        try:
            # prepare filtered_summary_df & hourly_df if not present
            if 'filtered_summary_df' in st.session_state and st.session_state['filtered_summary_df'] is not None:
                filtered_summary_df = st.session_state['filtered_summary_df']
                summary_df_local = st.session_state.get('summary_df', None)
                hourly_df = st.session_state.get('hourly_df', None)
            else:
                try:
                    summary_df_local, hourly_df = shift_plan.main_pipeline(FORECAST_DATA_PATH)
                    st.session_state['summary_df'] = summary_df_local
                    st.session_state['hourly_df'] = hourly_df
                    filtered_summary_df = summary_df_local.drop(columns=["shift", "start", "end"], errors='ignore')
                    if heatmap_option == "Service Now":
                        filtered_summary_df = filtered_summary_df.drop(columns=["Phone59"], errors='ignore')
                    elif heatmap_option == "Service Now - Five9 together":
                        filtered_summary_df = filtered_summary_df.drop(columns=["Phone"], errors='ignore')
                    st.session_state['filtered_summary_df'] = filtered_summary_df
                except Exception as e:
                    st.error(f"❌ Failed to generate shift plan (required for HeatMap): {e}")
                    raise

            if st.session_state.get('summary_table') is not None:
                summary_to_pass = st.session_state['summary_table']
            else:
                summary_to_pass = filtered_summary_df if (filtered_summary_df is not None) else None

            combined_df, diag = combine_forecast_tables(
                forecast_data_path=FORECAST_DATA_PATH,
                output_path=combined_output_path,
                heatmap_option=heatmap_option,
                hourly_df=hourly_df,
                summary_df=summary_to_pass
            )

            if combined_df is None or not isinstance(combined_df, pd.DataFrame):
                st.error("combine_forecast_tables did not return a DataFrame. Aborting.")
                return

            # apply Resource renames, compute Resources and write formatted excel (reuse earlier logic)
            modified_combined_df = combined_df.copy()

            # positional renaming logic
            original_cols = list(modified_combined_df.columns)
            new_cols = original_cols.copy()

            def find_col_index(cols_list, keywords):
                for i, c in enumerate(cols_list):
                    low = str(c).strip().lower()
                    for k in keywords:
                        if k.lower() == low:
                            return i
                # substring fallback
                for i, c in enumerate(cols_list):
                    low = str(c).strip().lower()
                    for k in keywords:
                        if k.lower() in low:
                            return i
                return None

            def find_resource_index_to_right(cols_list, start_idx, window=3):
                for j in range(start_idx + 1, min(start_idx + 1 + window, len(cols_list))):
                    low = str(cols_list[j]).strip().lower()
                    if low in ("resource", "resources", "resource.3", "resource 3", "resource. 3"):
                        return j
                return None

            chat_idx = find_col_index(original_cols, ["Chat", "chat"])
            if chat_idx is not None:
                res_idx = find_resource_index_to_right(original_cols, chat_idx, window=3)
                if res_idx is not None:
                    new_cols[res_idx] = "Resource - Chat"

            phone_idx = find_col_index(original_cols, ["Phone59", "Phone", "phone"])
            if phone_idx is not None:
                res_idx = find_resource_index_to_right(original_cols, phone_idx, window=3)
                if res_idx is not None:
                    new_cols[res_idx] = "Resource - Phone"

            self_idx = find_col_index(original_cols, ["Self-service", "Self_service", "Self service", "Self"])
            if self_idx is not None:
                res_idx = find_resource_index_to_right(original_cols, self_idx, window=3)
                if res_idx is not None:
                    new_cols[res_idx] = "Resource - Self-service"

            teams_idx = find_col_index(original_cols, ["Teams", "teams", "Teams available"])
            if teams_idx is not None:
                for j in range(teams_idx + 1, min(teams_idx + 1 + 3, len(original_cols))):
                    low = str(original_cols[j]).strip().lower()
                    if low in ("resources", "resource", "resource.3", "resource 3"):
                        new_cols[j] = "Resource - Total"
                        break

            modified_combined_df.columns = new_cols

            # ensure canonical columns
            canonical_total = "Resource - Total"
            canonical_chat = "Resource - Chat"
            canonical_phone = "Resource - Phone"
            canonical_self = "Resource - Self-service"
            for c in (canonical_total, canonical_chat, canonical_phone, canonical_self):
                if c not in modified_combined_df.columns:
                    modified_combined_df[c] = 0

            teams_col_final = find_col(list(modified_combined_df.columns), ["Teams", "teams", "Teams available"])

            for idx, row in modified_combined_df.iterrows():
                teams_val = None
                if teams_col_final is not None and teams_col_final in modified_combined_df.columns:
                    try:
                        v = row[teams_col_final]
                        if pd.notna(v):
                            teams_val = int(float(v))
                    except Exception:
                        teams_val = None

                if teams_val is None:
                    continue

                total_res = int(team_size) * int(teams_val)
                modified_combined_df.at[idx, canonical_total] = total_res

                base = total_res // 3
                rem = total_res - base * 3
                chat_val = base + (1 if rem > 0 else 0)
                phone_val = base + (1 if rem > 1 else 0)
                self_val = base

                modified_combined_df.at[idx, canonical_chat] = chat_val
                modified_combined_df.at[idx, canonical_phone] = phone_val
                modified_combined_df.at[idx, canonical_self] = self_val

            # write single-sheet excel with formatting (same as earlier)
            out_path = Path(combined_output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
                    sheet_name = "HeatMap_Single_Sheet"
                    modified_combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]

                    header_format = workbook.add_format({
                        "bold": True,
                        "text_wrap": True,
                        "valign": "top",
                        "fg_color": "#D9E1F2",
                        "border": 1
                    })

                    for col_idx, col_name in enumerate(modified_combined_df.columns):
                        worksheet.set_column(col_idx, col_idx, 15)
                        worksheet.write(0, col_idx, str(col_name), header_format)

                    n_rows = len(modified_combined_df)
                    if n_rows <= 0:
                        n_rows = 1

                    avg_col_indices = []
                    for i, col in enumerate(modified_combined_df.columns):
                        if "avg" in str(col).lower() or "average" in str(col).lower():
                            avg_col_indices.append(i)

                    total_col_indices = []
                    cols_list = [str(c) for c in modified_combined_df.columns]
                    for i, cname in enumerate(cols_list):
                        low = cname.strip().lower()
                        if low == "resource - total":
                            total_col_indices.append(i)
                            continue
                        if "resource" in low and "total" in low:
                            total_col_indices.append(i)
                            continue
                        if "resource" in low and ("total" in low.split() or "tot" in low):
                            total_col_indices.append(i)
                            continue
                    if not total_col_indices:
                        for i, cname in enumerate(cols_list):
                            low = cname.strip().lower()
                            if low in ("resources", "resource"):
                                total_col_indices.append(i)

                    color_green = "#63BE7B"
                    color_yellow = "#FFEB84"
                    color_red = "#F8696B"

                    def apply_three_color_scale(col_indices, min_color, mid_color, max_color, num_format_str="#,##0.00"):
                        for col_idx in col_indices:
                            try:
                                ci = int(col_idx)
                            except Exception:
                                continue
                            if ci < 0 or ci >= len(modified_combined_df.columns):
                                continue
                            col_letter = col_idx_to_excel(ci)
                            cell_range = f"{col_letter}2:{col_letter}{n_rows+1}"
                            num_format = workbook.add_format({"num_format": num_format_str})
                            worksheet.set_column(ci, ci, 12, num_format)
                            worksheet.conditional_format(cell_range, {
                                "type": "3_color_scale",
                                "min_type": "min",
                                "mid_type": "percentile",
                                "mid_value": 50,
                                "max_type": "max",
                                "min_color": min_color,
                                "mid_color": mid_color,
                                "max_color": max_color
                            })

                    if avg_col_indices:
                        apply_three_color_scale(avg_col_indices, min_color=color_green, mid_color=color_yellow, max_color=color_red, num_format_str="#,##0.00")
                    if total_col_indices:
                        apply_three_color_scale(total_col_indices, min_color=color_red, mid_color=color_yellow, max_color=color_green, num_format_str="#,##0")

                    worksheet.freeze_panes(1, 0)

                st.success("HeatMap (single-sheet) created with Excel color formatting (download ready).")
            except Exception as e:
                st.error(f"Failed to write combined heatmap file with formatting: {e}")
                raise

            # show preview and download
            st.write(modified_combined_df.to_html(index=False, escape=False), unsafe_allow_html=True)
            with open(combined_output_path, "rb") as f:
                st.download_button(
                    label="📥 Download HeatMap (Single Sheet)",
                    data=f,
                    file_name=combined_output_path.name
                )

        except Exception as e:
            st.error(f"❌ Failed to combine forecast tables: {e}")
            st.exception(e)

    # -------------------------
    # FINAL: Download full workbook (Shift + Summary in Sheet1 side-by-side, HeatMap in Sheet2)
    # -------------------------
    st.markdown("---")
    st.markdown("## 📦 Download Full Plan Workbook (Shift + Summary + HeatMap)")

    if st.button("📥 Download Full Plan Workbook", use_container_width=True):
        # Use shift_avail and summary_display instead of shift_df and summary_df_session
        shift_avail = st.session_state.get('shift_avail', None)
        summary_display = st.session_state.get('summary_display', None)

        # Save to session state if not already present
        if shift_avail is None:
            summary_candidates = list(Path(DATA_DIR).glob("shift_avail.xlsx"))
            shift_avail_path = summary_candidates[0] if summary_candidates else None
            if shift_avail_path:
                shift_avail = pd.read_excel(shift_avail_path)
                st.session_state['shift_avail'] = shift_avail

        if summary_display is None:
            summary_candidates = list(Path(DATA_DIR).glob("Summary.xlsx"))
            summary_path = summary_candidates[0] if summary_candidates else None
            if summary_path:
                summary_display = pd.read_excel(summary_path)
                st.session_state['summary_display'] = summary_display

        # try to generate heatmap_df (modified_combined_df) if missing
        heatmap_df = None
        try:
            combined_output_path = DATA_DIR / f"combined_forecast_{heatmap_option.replace(' ', '_')}_{start_date.strftime('%b%d')}_{end_date.strftime('%b%d')}.xlsx"
            from modules.combine_forecast_tables import combine_forecast_tables

            if st.session_state.get('summary_table') is not None:
                summary_to_pass = st.session_state['summary_table']
            else:
                summary_to_pass = st.session_state.get('filtered_summary_df', None)

            combined_df, diag = combine_forecast_tables(
                forecast_data_path=FORECAST_DATA_PATH,
                output_path=combined_output_path,
                heatmap_option=heatmap_option,
                hourly_df=st.session_state.get('hourly_df', None),
                summary_df=summary_to_pass
            )
            if isinstance(combined_df, pd.DataFrame):
                modified_combined_df = combined_df.copy()

                original_cols = list(modified_combined_df.columns)
                new_cols = original_cols.copy()

                def find_col_index_local(cols_list, keywords):
                    for i, c in enumerate(cols_list):
                        low = str(c).strip().lower()
                        for k in keywords:
                            if k.lower() == low:
                                return i
                    for i, c in enumerate(cols_list):
                        low = str(c).strip().lower()
                        for k in keywords:
                            if k.lower() in low:
                                return i
                    return None

                def find_resource_index_to_right_local(cols_list, start_idx, window=3):
                    for j in range(start_idx + 1, min(start_idx + 1 + window, len(cols_list))):
                        low = str(cols_list[j]).strip().lower()
                        if low in ("resource", "resources", "resource.3", "resource 3", "resource. 3"):
                            return j
                    return None

                chat_idx = find_col_index_local(original_cols, ["Chat", "chat"])
                if chat_idx is not None:
                    res_idx = find_resource_index_to_right_local(original_cols, chat_idx, window=3)
                    if res_idx is not None:
                        new_cols[res_idx] = "Resource - Chat"
                phone_idx = find_col_index_local(original_cols, ["Phone59", "Phone", "phone"])
                if phone_idx is not None:
                    res_idx = find_resource_index_to_right_local(original_cols, phone_idx, window=3)
                    if res_idx is not None:
                        new_cols[res_idx] = "Resource - Phone"
                self_idx = find_col_index_local(original_cols, ["Self-service", "Self_service", "Self service", "Self"])
                if self_idx is not None:
                    res_idx = find_resource_index_to_right_local(original_cols, self_idx, window=3)
                    if res_idx is not None:
                        new_cols[res_idx] = "Resource - Self-service"
                teams_idx = find_col_index_local(original_cols, ["Teams", "teams", "Teams available"])
                if teams_idx is not None:
                    for j in range(teams_idx + 1, min(teams_idx + 1 + 3, len(original_cols))):
                        low = str(original_cols[j]).strip().lower()
                        if low in ("resources", "resource", "resource.3", "resource 3"):
                            new_cols[j] = "Resource - Total"
                            break

                modified_combined_df.columns = new_cols

                for c in ("Resource - Total", "Resource - Chat", "Resource - Phone", "Resource - Self-service"):
                    if c not in modified_combined_df.columns:
                        modified_combined_df[c] = 0

                teams_col_final = find_col(list(modified_combined_df.columns), ["Teams", "teams", "Teams available"])

                for idx, row in modified_combined_df.iterrows():
                    teams_val = None
                    if teams_col_final is not None and teams_col_final in modified_combined_df.columns:
                        try:
                            v = row[teams_col_final]
                            if pd.notna(v):
                                teams_val = int(float(v))
                        except Exception:
                            teams_val = None
                    if teams_val is None:
                        continue
                    total_res = int(team_size) * int(teams_val)
                    modified_combined_df.at[idx, "Resource - Total"] = total_res
                    base = total_res // 3
                    rem = total_res - base * 3
                    chat_val = base + (1 if rem > 0 else 0)
                    phone_val = base + (1 if rem > 1 else 0)
                    self_val = base
                    modified_combined_df.at[idx, "Resource - Chat"] = chat_val
                    modified_combined_df.at[idx, "Resource - Phone"] = phone_val
                    modified_combined_df.at[idx, "Resource - Self-service"] = self_val

                heatmap_df = modified_combined_df.copy()
            else:
                heatmap_df = None
        except Exception as e:
            st.warning(f"Could not generate HeatMap for workbook: {e}")
            heatmap_df = None

        # Final checks
        if (shift_avail is None or not isinstance(shift_avail, pd.DataFrame)) and (summary_display is None or not isinstance(summary_display, pd.DataFrame)) and (heatmap_df is None or not isinstance(heatmap_df, pd.DataFrame)):
            st.error("Nothing available to write to workbook. Generate Shift, Summary or HeatMap first.")
            return

        out_dir = Path("forecast_app/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"Full_Plan_{start_date.strftime('%b%d')}_{end_date.strftime('%b%d')}.xlsx"

        try:
            with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
                workbook = writer.book

                # -------------------------
                # Sheet1: Plan (two tables side-by-side)
                # -------------------------
                sheet1 = "Plan"
                worksheet = workbook.add_worksheet(sheet1)
                writer.sheets[sheet1] = worksheet

                # Title/heading formats and cell formats
                title_format = workbook.add_format({"bold": True, "font_size": 14})
                header_format = workbook.add_format({"bold": True, "border": 1, "align": "center", "valign": "vcenter", "fg_color": "#D9E1F2"})
                body_format = workbook.add_format({"border": 1, "valign": "top"})
                body_format_int = workbook.add_format({"border": 1, "valign": "top", "num_format": "#,##0"})
                body_format_float = workbook.add_format({"border": 1, "valign": "top", "num_format": "#,##0.00"})

                # Write Shift table (left)
                left_col = 0
                top_row = 2  # leave rows 0-1 for top headings / buffer
                if isinstance(shift_avail, pd.DataFrame):
                    worksheet.write(0, left_col, "Shift-Wise Availability", title_format)
                    shift_avail.to_excel(writer, sheet_name=sheet1, index=False, startrow=top_row, startcol=left_col)
                    for c_idx, colname in enumerate(shift_avail.columns):
                        worksheet.write(top_row, left_col + c_idx, str(colname), header_format)
                    shift_widths = compute_column_widths(shift_avail)
                    for i, w in enumerate(shift_widths):
                        col_idx = left_col + i
                        dtype = shift_avail.dtypes[i]
                        if pd.api.types.is_integer_dtype(dtype):
                            worksheet.set_column(col_idx, col_idx, w, body_format_int)
                        elif pd.api.types.is_float_dtype(dtype):
                            worksheet.set_column(col_idx, col_idx, w, body_format_float)
                        else:
                            worksheet.set_column(col_idx, col_idx, w, body_format)
                else:
                    worksheet.write(0, left_col, "Shift-Wise Availability (not available)", title_format)
                    worksheet.write(2, left_col, "No shift data available", body_format)

                gap = 3
                left_table_cols = shift_avail.shape[1] if isinstance(shift_avail, pd.DataFrame) else 3
                right_col = left_col + left_table_cols + gap

                if isinstance(summary_display, pd.DataFrame):
                    worksheet.write(0, right_col, "Summary Table", title_format)
                    summary_display.to_excel(writer, sheet_name=sheet1, index=False, startrow=top_row, startcol=right_col)
                    for c_idx, colname in enumerate(summary_display.columns):
                        worksheet.write(top_row, right_col + c_idx, str(colname), header_format)
                    summary_widths = compute_column_widths(summary_display)
                    for i, w in enumerate(summary_widths):
                        col_idx = right_col + i
                        dtype = summary_display.dtypes[i]
                        if pd.api.types.is_integer_dtype(dtype):
                            worksheet.set_column(col_idx, col_idx, w, body_format_int)
                        elif pd.api.types.is_float_dtype(dtype):
                            worksheet.set_column(col_idx, col_idx, w, body_format_float)
                        else:
                            worksheet.set_column(col_idx, col_idx, w, body_format)
                else:
                    worksheet.write(0, right_col, "Summary Table (not available)", title_format)
                    worksheet.write(2, right_col, "No summary data available", body_format)

                worksheet.freeze_panes(top_row + 1, 0)

                # -------------------------
                # Sheet2: HeatMap (apply same formatting & conditional formatting)
                # -------------------------
                sheet2 = "HeatMap"
                if isinstance(heatmap_df, pd.DataFrame):
                    heatmap_df.to_excel(writer, sheet_name=sheet2, index=False)
                    worksheet2 = writer.sheets[sheet2]
                    header_format2 = workbook.add_format({
                        "bold": True,
                        "text_wrap": True,
                        "valign": "top",
                        "fg_color": "#D9E1F2",
                        "border": 1
                    })
                    for col_idx, col_name in enumerate(heatmap_df.columns):
                        worksheet2.set_column(col_idx, col_idx, 15)
                        worksheet2.write(0, col_idx, str(col_name), header_format2)

                    n_rows = len(heatmap_df)
                    if n_rows <= 0:
                        n_rows = 1

                    avg_col_indices = []
                    for i, col in enumerate(heatmap_df.columns):
                        if "avg" in str(col).lower() or "average" in str(col).lower():
                            avg_col_indices.append(i)

                    total_col_indices = []
                    cols_list = [str(c) for c in heatmap_df.columns]
                    for i, cname in enumerate(cols_list):
                        low = cname.strip().lower()
                        if low == "resource - total":
                            total_col_indices.append(i); continue
                        if "resource" in low and "total" in low:
                            total_col_indices.append(i); continue
                        if "resource" in low and ("total" in low.split() or "tot" in low):
                            total_col_indices.append(i); continue
                    if not total_col_indices:
                        for i, cname in enumerate(cols_list):
                            low = cname.strip().lower()
                            if low in ("resources", "resource"):
                                total_col_indices.append(i)

                    color_green = "#63BE7B"
                    color_yellow = "#FFEB84"
                    color_red = "#F8696B"

                    def apply_three_color_scale_ws(ws, wk, col_indices, min_color, mid_color, max_color, num_format_str="#,##0.00"):
                        for col_idx in col_indices:
                            try:
                                ci = int(col_idx)
                            except Exception:
                                continue
                            if ci < 0 or ci >= len(heatmap_df.columns):
                                continue
                            col_letter = col_idx_to_excel(ci)
                            cell_range = f"{col_letter}2:{col_letter}{n_rows+1}"
                            num_format = wk.add_format({"num_format": num_format_str})
                            ws.set_column(ci, ci, 12, num_format)
                            ws.conditional_format(cell_range, {
                                "type": "3_color_scale",
                                "min_type": "min",
                                "mid_type": "percentile",
                                "mid_value": 50,
                                "max_type": "max",
                                "min_color": min_color,
                                "mid_color": mid_color,
                                "max_color": max_color
                            })

                    if avg_col_indices:
                        apply_three_color_scale_ws(worksheet2, workbook, avg_col_indices, color_green, color_yellow, color_red, num_format_str="#,##0")
                    if total_col_indices:
                        apply_three_color_scale_ws(worksheet2, workbook, total_col_indices, color_green, color_yellow, color_red, num_format_str="#,##0")
                    worksheet2.freeze_panes(1, 0)
                else:
                    # no heatmap: write a simple sheet message
                    ws_note = workbook.add_worksheet("HeatMap")
                    writer.sheets["HeatMap"] = ws_note
                    ws_note.write(0, 0, "HeatMap not available", workbook.add_format({"bold": True}))

            st.success(f"Full plan workbook created: {out_path.name}")
            with open(out_path, "rb") as f:
                st.download_button(label="📥 Download Full Plan Workbook", data=f, file_name=out_path.name)
        except Exception as e:
            st.error(f"Failed to write full plan workbook: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
