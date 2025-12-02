import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil  # ✅ For file copy
import logging
# from modules import shift_plan

def generate_shift_plan(forecast_df, tasks_per_resource=None, forecast_start=None, forecast_end=None):
    if tasks_per_resource is None:
        tasks_per_resource = {"Chat": 13, "Phone": 13, "Phone59": 13, "Self-service": 15}

    result_frames = {}
    for channel, task_count in tasks_per_resource.items():
        df = forecast_df[forecast_df["Channel"] == channel].copy()

        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])

        if forecast_start is not None and forecast_end is not None:
            fs = pd.to_datetime(forecast_start).normalize()
            fe = pd.to_datetime(forecast_end).normalize()
            if fe < fs:
                selected_weekday_mask = df["Date"].dt.weekday < 5
                selected_working_dates = df.loc[selected_weekday_mask, "Date"].dt.date.unique()
            else:
                all_dates = pd.date_range(fs, fe, freq='D')
                sel_weekdays = all_dates[all_dates.weekday < 5].date
                selected_weekday_mask = df["Date"].dt.date.isin(set(sel_weekdays))
                selected_working_dates = sel_weekdays
        else:
            selected_weekday_mask = df["Date"].dt.weekday < 5
            selected_working_dates = df.loc[selected_weekday_mask, "Date"].dt.date.unique()

        num_working_days = len(selected_working_dates)
        if num_working_days == 0:
            continue

        avg_col_name = f"Avg_for_{num_working_days}_working_days"

        planning = []
        for hour in range(1, 25):
            try:
                hourly_values = df.loc[selected_weekday_mask, str(hour)].values
            except KeyError:
                continue

            sum_working_days = np.sum(hourly_values)
            avg_per_working_day = sum_working_days / num_working_days
            min_resources = int(np.ceil(avg_per_working_day / task_count))

            planning.append({
                "Hour": (hour + 5) % 24,
                channel.capitalize(): round(sum_working_days),
                avg_col_name: round(avg_per_working_day)
            })

        result_df = pd.DataFrame(planning)
        result_df.reset_index(drop=True, inplace=True)
        result_frames[channel] = result_df

    return result_frames


def apply_coloring_and_download(all_plans, start_date, end_date, backup_path=None, heatmap_option=None):
    output_dir = Path("forecast_app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "final_output.xlsx"

    # Define which channels to display
    if heatmap_option == "Service Now":
        display_channels = ["Chat", "Phone", "Self-service"]
    elif heatmap_option == "Service Now - Five9 together":
        display_channels = ["Chat", "Phone59", "Self-service"]
    else:
        display_channels = all_plans.keys()

    # Display relevant tables only
    for channel in display_channels:
        if channel not in all_plans:
            continue
        df = all_plans[channel]
        st.subheader(f"{channel} Plan")

        df_display = df.copy()
        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = 'Sr.No.'

        avg_col_name = get_avg_col_name(df_display)
        if avg_col_name:
            styled = df_display.style.background_gradient(
                cmap="RdYlGn_r", subset=[avg_col_name]
            )
        else:
            styled = df_display.style
        st.dataframe(styled, use_container_width=True)

        individual_file = output_dir / f"{channel}_plan.xlsx"
        with pd.ExcelWriter(individual_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=channel, startrow=1, index=False)
            workbook = writer.book
            worksheet = writer.sheets[channel]
            heading = f"Forecast and Heat Map for {channel} for {start_date.strftime('%b %d')}–{end_date.strftime('%b %d')}"
            worksheet.merge_range('A1:F1', heading, workbook.add_format({
                'bold': True, 'align': 'center', 'valign': 'vcenter', 'font_size': 14
            }))
            if avg_col_name:
                avg_col_index = df.columns.get_loc(avg_col_name)
                avg_col_letter = chr(ord('A') + avg_col_index)
                worksheet.conditional_format(
                    f'{avg_col_letter}3:{avg_col_letter}{len(df) + 2}',
                    {
                        'type': '3_color_scale',
                        'min_color': "#63BE7B",
                        'mid_color': "#FFEB84",
                        'max_color': "#F8696B"
                    }
                )

        with open(individual_file, "rb") as f:
            st.download_button(
                label=f"📥 Download {channel} Plan",
                data=f,
                file_name=f"{channel}_Plan_{start_date.strftime('%b%d')}_{end_date.strftime('%b%d')}.xlsx"
            )

    # Write ALL plans to one Excel file
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for channel, df in all_plans.items():
            df.to_excel(writer, sheet_name=channel, startrow=1, index=False)
            workbook = writer.book
            worksheet = writer.sheets[channel]
            heading = f"Forecast and Heat Map for {channel} for {start_date.strftime('%b %d')}–{end_date.strftime('%b %d')}"
            worksheet.merge_range('A1:F1', heading, workbook.add_format({
                'bold': True, 'align': 'center', 'valign': 'vcenter', 'font_size': 14
            }))
            avg_col_name = get_avg_col_name(df)
            if avg_col_name:
                avg_col_index = df.columns.get_loc(avg_col_name)
                avg_col_letter = chr(ord('A') + avg_col_index)
                worksheet.conditional_format(
                    f'{avg_col_letter}3:{avg_col_letter}{len(df) + 2}',
                    {
                        'type': '3_color_scale',
                        'min_color': "#63BE7B",
                        'mid_color': "#FFEB84",
                        'max_color': "#F8696B"
                    }
                )
    # Save backup copy if needed
    if backup_path:
        shutil.copy(output_file, backup_path)

    # Download all plans
    with open(output_file, "rb") as f:
        st.download_button(
            label="📥 Download All Plans",
            data=f,
            file_name=f"Final_Heat_Map_{start_date.strftime('%b%d')}_{end_date.strftime('%b%d')}.xlsx"
        )
# -----------------------------
# NEW: Daily analyst requirement
# -----------------------------
def daily_analyst_requirements(full_forecast: pd.DataFrame, analysts_capacity_per_day: int = 14, include_total: bool = True) -> pd.DataFrame:
    """
    Compute daily analyst requirements using 14 tickets/analyst/day (default).
    Works for whichever channels are present (Chat, Phone, Phone59, Self-service).
    Expects `full_forecast` with columns: Date, Channel, '1'..'24'.
    """
    if full_forecast is None or full_forecast.empty:
        return pd.DataFrame(columns=["Date", "Channel", "Forecasted_Volume", "Analysts_Required", "Capacity_per_Analyst_per_Day"])

    df = full_forecast.copy()
    hour_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]
    if not hour_cols:
        return pd.DataFrame(columns=["Date", "Channel", "Forecasted_Volume", "Analysts_Required", "Capacity_per_Analyst_per_Day"])

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    daily = df.groupby(["Date", "Channel"], as_index=False)[hour_cols].sum()
    daily["Forecasted_Volume"] = daily[hour_cols].sum(axis=1).astype(int)
    daily = daily[["Date", "Channel", "Forecasted_Volume"]]
    daily["Analysts_Required"] = np.ceil(daily["Forecasted_Volume"] / float(analysts_capacity_per_day)).astype(int)
    daily["Capacity_per_Analyst_per_Day"] = int(analysts_capacity_per_day)

    if include_total:
        total = daily.groupby("Date", as_index=False)["Forecasted_Volume"].sum()
        total["Channel"] = "All"
        total["Analysts_Required"] = np.ceil(total["Forecasted_Volume"] / float(analysts_capacity_per_day)).astype(int)
        total["Capacity_per_Analyst_per_Day"] = int(analysts_capacity_per_day)
        total = total[["Date", "Channel", "Forecasted_Volume", "Analysts_Required", "Capacity_per_Analyst_per_Day"]]
        daily = pd.concat([daily, total], ignore_index=True)

    return daily.sort_values(["Date", "Channel"]).reset_index(drop=True)


def get_avg_col_name(df):
    for col in df.columns:
        if col.startswith("Avg_for_") and col.endswith("_working_days"):
            return col
    return None
