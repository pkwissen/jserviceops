import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil  # ✅ For file copy

def generate_shift_plan(forecast_df, tasks_per_resource=None):
    if tasks_per_resource is None:
        tasks_per_resource = {"Chat": 13, "Phone": 13, "Phone59": 13, "Self-service": 15}

    result_frames = {}
    for channel, task_count in tasks_per_resource.items():
        df = forecast_df[forecast_df["Channel"] == channel].copy()

        planning = []
        for hour in range(1, 25):
            try:
                hourly_values = df[str(hour)].values
            except KeyError:
                continue

            sum_7_days = np.sum(hourly_values)
            avg_per_working_day = sum_7_days / 7
            min_resources = int(np.ceil(avg_per_working_day / task_count))

            planning.append({
                "Hour": (hour + 5) % 24,
                channel.capitalize(): round(sum_7_days),
                "Avg_for_7_days": round(avg_per_working_day)
            })

        result_df = pd.DataFrame(planning)
        result_df.reset_index(drop=True, inplace=True)
        result_frames[channel] = result_df

    return result_frames


def apply_coloring_and_download(all_plans, start_date, end_date, backup_path=None, heatmap_option=None):
    output_dir = Path("forecast_app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "final_output.xlsx"

    # Define which channels to display based on selected option
    if heatmap_option == "Service Now":
        display_channels = ["Chat", "Phone", "Self-service"]
    elif heatmap_option == "Service Now - Five9 together":
        display_channels = ["Chat", "Phone59", "Self-service"]
    else:
        display_channels = all_plans.keys()  # fallback in case no option matches

    # Display relevant tables only
    for channel in display_channels:
        if channel not in all_plans:
            continue  # skip if data is not available for this channel

        df = all_plans[channel]
        st.subheader(f"{channel} Plan")

        # Prepare dataframe for display
        df_display = df.copy()
        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = 'Sr.No.'  # Rename index

        if "Avg_for_7_days" in df_display.columns:
            styled = df_display.style.background_gradient(
                cmap="RdYlGn_r", subset=["Avg_for_7_days"]
            )
        else:
            styled = df_display.style

        # Display in Streamlit
        st.dataframe(styled, use_container_width=True)

        # Save individual plan Excel with formatting
        individual_file = output_dir / f"{channel}_plan.xlsx"
        with pd.ExcelWriter(individual_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=channel, startrow=1, index=False)
            workbook = writer.book
            worksheet = writer.sheets[channel]

            heading = f"Forecast and Heat Map for {channel} for {start_date.strftime('%b %d')}–{end_date.strftime('%b %d')}"
            worksheet.merge_range('A1:F1', heading, workbook.add_format({
                'bold': True, 'align': 'center', 'valign': 'vcenter', 'font_size': 14
            }))

            if "Avg_for_7_days" in df.columns:
                avg_col_index = df.columns.get_loc("Avg_for_7_days")
                avg_col_letter = chr(ord('A') + avg_col_index)
                worksheet.conditional_format(f'{avg_col_letter}3:{avg_col_letter}{len(df) + 2}', {
                    'type': '3_color_scale',
                    'min_color': "#63BE7B",
                    'mid_color': "#FFEB84",
                    'max_color': "#F8696B"
                })

        # Download button for each plan
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

            if "Avg_for_7_days" in df.columns:
                avg_col_index = df.columns.get_loc("Avg_for_7_days")
                avg_col_letter = chr(ord('A') + avg_col_index)
                worksheet.conditional_format(f'{avg_col_letter}3:{avg_col_letter}{len(df) + 2}', {
                    'type': '3_color_scale',
                    'min_color': "#63BE7B",
                    'mid_color': "#FFEB84",
                    'max_color': "#F8696B"
                })

    # Save backup copy if backup_path is provided
    if backup_path:
        shutil.copy(output_file, backup_path)

    # Download all plans
    with open(output_file, "rb") as f:
        st.download_button(
            label="📥 Download All Plans",
            data=f,
            file_name=f"Final_Heat_Map_{start_date.strftime('%b%d')}_{end_date.strftime('%b%d')}.xlsx"
        )

