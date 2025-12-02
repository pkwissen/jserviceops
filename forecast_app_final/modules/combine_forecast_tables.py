import re
import math
import pandas as pd
from pathlib import Path
from typing import Optional

# Use this mapping for labels inside shift columns
SHIFT_TIME_LABELS = {
    "shift1": "6 am - 3 pm",
    "shift2": "8 am - 5 pm",
    "shift3": "11 am - 8 pm",
    "shift4": "12 pm - 9 pm",
    "shift5": "2 pm - 11 pm",
    "shift6": "5 pm - 2 am",
    "shift7": "8 pm - 5 am",
    "shift8": "10 pm - 7 am",
}

def rename_avg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any 'Avg_for_*_working_days' column to 'Avg/Day'."""
    df = df.copy()
    df.columns = [
        'Avg/Day' if ('Avg_for_' in str(col) and '_working_days' in str(col)) else str(col)
        for col in df.columns
    ]
    return df

def read_and_prepare_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    Read a sheet using header=1 (so the merged title row is skipped and real headers are used),
    then ensure Avg/Day rename and reset index.
    """
    df = pd.read_excel(xls, sheet_name=sheet_name, header=1)
    df = rename_avg_columns(df)
    df = df.reset_index(drop=True)
    return df

# Helper: parse hour token like "6 am" or "12 pm" -> 0-23 integer
def _parse_hour_token(token: str) -> int:
    token = token.strip().lower().replace('.', '')
    m = re.search(r'(\d{1,2})\s*([ap]m)', token)
    if not m:
        raise ValueError(f"Can't parse hour token: {token}")
    hour = int(m.group(1))
    ampm = m.group(2)
    if ampm == 'am':
        return 0 if hour == 12 else hour
    else:  # pm
        return 12 if hour == 12 else hour + 12

# Helper: from label "6 am - 3 pm" return list of hours included (end exclusive)
def hours_for_shift_label(label: str):
    parts = label.split('-')
    if len(parts) != 2:
        raise ValueError(f"Bad shift label: {label}")
    start_tok, end_tok = parts[0].strip(), parts[1].strip()
    start_h = _parse_hour_token(start_tok)
    end_h = _parse_hour_token(end_tok)
    if start_h < end_h:
        return list(range(start_h, end_h))  # end exclusive
    elif start_h > end_h:
        # wrap midnight
        return list(range(start_h, 24)) + list(range(0, end_h))
    else:
        # start == end -> full day (unlikely)
        return list(range(0, 24))

def combine_forecast_tables(
    forecast_data_path: str,
    output_path: str,
    heatmap_option: str,
    hourly_df: pd.DataFrame = None,
    summary_df: pd.DataFrame = None
):
    xls = pd.ExcelFile(forecast_data_path)
    hourly_df = hourly_df.rename(columns={'total resources': 'Resources'}) if hourly_df is not None else None
    # read channel sheets with header=1 (skip merged title)
    chat_df = read_and_prepare_sheet(xls, "Chat")
    phone_df = read_and_prepare_sheet(xls, "Phone")
    phone59_df = read_and_prepare_sheet(xls, "Phone59")
    self_service_df = read_and_prepare_sheet(xls, "Self-service")

    # choose tables and corresponding channel lookup names
    if heatmap_option == "Service Now":
        tables = [chat_df, phone_df, self_service_df]
        channel_names = ["Chat", "Phone", "Self-service"]
    elif heatmap_option == "Service Now - Five9 together":
        tables = [chat_df, phone59_df, self_service_df]
        channel_names = ["Chat", "Phone59", "Self-service"]
    else:
        raise ValueError("Invalid heatmap_option")

    # ---- Insert Resource column into each of the first three channel tables ----
    prepared_tables = []
    for df, ch_name in zip(tables, channel_names):
        df_local = df.copy()
        resource_vals = [""] * len(df_local)

        if summary_df is not None and hourly_df is not None:
            if "Resources" in hourly_df.columns and "total_resource" in summary_df.columns:
                # map total_resource values to summary rows
                mapping = {}
                for _, srow in summary_df.iterrows():
                    total_val = srow["total_resource"]
                    if pd.notna(total_val):
                        try:
                            mapping[int(total_val)] = srow
                        except Exception:
                            pass

                # now assign per row (match hourly_df rows to summary mapping if possible)
                for ridx in range(min(len(df_local), len(hourly_df))):
                    res_val = hourly_df.loc[ridx, "Resources"]
                    if pd.notna(res_val):
                        try:
                            res_val_int = int(res_val)
                            if res_val_int in mapping:
                                srow = mapping[res_val_int]
                                if ch_name in srow:
                                    resource_vals[ridx] = srow[ch_name]
                        except Exception:
                            # leave blank if conversion fails
                            pass

        # Insert 'Resource' column immediately after 'Avg/Day' if present, else append at end
        insert_pos = None
        for i, c in enumerate(df_local.columns):
            if str(c) == 'Avg/Day':
                insert_pos = i + 1
                break
        if insert_pos is None:
            df_local['Resource'] = resource_vals
        else:
            df_local.insert(insert_pos, 'Resource', resource_vals)

        prepared_tables.append(df_local)

    # Keep reference to teams_table and its length for coloring later
    teams_table = None
    hourly_len = 0

    # If hourly_df is provided, add it as the 4th table
    if hourly_df is not None:
        if all(col in hourly_df.columns for col in ["Hour", "Teams", "Resources"]):
            teams_table = hourly_df[["Hour", "Teams", "Resources"]].copy().reset_index(drop=True)

            # Rotate the hourly table so it starts at Hour == 6 (if 6 exists). This matches your image ordering.
            start_idx = None
            for idx, val in enumerate(teams_table["Hour"].astype(int)):
                if int(val) == 6:
                    start_idx = idx
                    break
            if start_idx is not None and start_idx != 0:
                teams_table = pd.concat([teams_table.iloc[start_idx:], teams_table.iloc[:start_idx]], ignore_index=True)

            # --- Create shift1..shift8 columns and leave blank by default ---
            for s in range(1, 9):
                teams_table[f"shift{s}"] = [""] * len(teams_table)

            # Fill the shift column values (we will color them in Excel later).
            # IMPORTANT CHANGE: write shift NUMBER (string) instead of the time-range label.
            hourly_len = len(teams_table)
            for s in range(1, 9):
                shift_key = f"shift{s}"
                label = SHIFT_TIME_LABELS.get(shift_key, "")
                if label:
                    shift_hours = set(hours_for_shift_label(label))
                    for ridx in range(hourly_len):
                        try:
                            hour_val = int(teams_table.loc[ridx, "Hour"])
                        except Exception:
                            continue
                        if hour_val in shift_hours:
                            # <-- change: write shift number (not time-range)
                            teams_table.at[ridx, shift_key] = str(s)

            prepared_tables.append(teams_table)
        else:
            raise ValueError("hourly_df must contain 'Hour', 'Teams', 'Resources' columns")

    # ---- Pad all tables to same number of rows (with empty strings) ----
    max_rows = max(len(df) for df in prepared_tables)
    padded_tables = []
    for df in prepared_tables:
        df_copy = df.copy()
        rows_to_add = max_rows - len(df_copy)
        if rows_to_add > 0:
            pad = pd.DataFrame([[""] * df_copy.shape[1]] * rows_to_add, columns=df_copy.columns)
            df_copy = pd.concat([df_copy, pad], ignore_index=True)
        padded_tables.append(df_copy)

    # build list to concat with '' as blank column between tables
    pieces = []
    for i, df in enumerate(padded_tables):
        pieces.append(df)
        if i < len(padded_tables) - 1:
            pieces.append(pd.DataFrame({'': [''] * max_rows}))

    combined = pd.concat(pieces, axis=1)

    # If pandas added an index-like unnamed column, remove it (defensive)
    if combined.columns[0] and str(combined.columns[0]).lower() in ['index', 'unnamed: 0']:
        combined = combined.drop(combined.columns[0], axis=1)

    # Now compute avgday column indices and table ranges from the final combined columns
    avgday_cols = [i for i, col in enumerate(combined.columns) if str(col) == "Avg/Day"]

    # compute table ranges by scanning columns and treating '' as separator
    table_ranges = []
    start = None
    for idx, col in enumerate(combined.columns):
        if col != '':
            if start is None:
                start = idx
            end = idx
        else:
            if start is not None:
                table_ranges.append((start, end))
                start = None
    if start is not None:
        table_ranges.append((start, end))

    # Write to Excel and apply formatting (borders + 3-color for Avg/Day + Resources in 4th table + shift cell colors)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        combined.to_excel(writer, index=False, sheet_name="Combined_Forecast")
        workbook = writer.book
        worksheet = writer.sheets["Combined_Forecast"]

        # formats
        border_fmt = workbook.add_format({'border': 1})
        color_fmt = {
            'type': '3_color_scale',
            'min_color': "#63BE7B",  # green
            'mid_color': "#FFEB84",  # yellow
            'max_color': "#F8696B"   # red
        }

        # formats for shift cells (alternating)
        shift_yellow_fmt = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1,
            'text_wrap': True, 'bold': True, 'fg_color': "#FFEB84"
        })
        shift_blue_fmt = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1,
            'text_wrap': True, 'bold': True, 'fg_color': "#9DC3E6"
        })

        # Excel rows: 0 = header, 1..N = data
        header_row = 0
        first_data_row = 1
        last_data_row = combined.shape[0]

        # Apply borders for each table range (include header row)
        for start_col, end_col in table_ranges:
            worksheet.conditional_format(
                header_row, start_col, last_data_row, end_col,
                {'type': 'no_blanks', 'format': border_fmt}
            )

        # Apply color formatting only to Avg/Day columns (data rows only)
        for avg_col in avgday_cols:
            worksheet.conditional_format(
                first_data_row, avg_col, last_data_row, avg_col, color_fmt
            )

        # Find the hourly table by presence of 'Hour' header and apply Resources color + shift cell coloring
        for start_col, end_col in table_ranges:
            table_headers = list(combined.columns[start_col:end_col+1])
            if "Hour" in table_headers:
                # this is the hourly/teams table
                # Color Resources column with 3-color scale if present
                if "Resources" in table_headers:
                    res_idx = start_col + table_headers.index("Resources")
                    # apply only to the actual hourly rows (not padded)
                    worksheet.conditional_format(
                        first_data_row, res_idx, first_data_row + (hourly_len - 1 if hourly_len>0 else 0), res_idx, color_fmt
                    )

                # Color only the respective cells in each shift column — write shift NUMBER (not time label)
                for s in range(1, 9):
                    shift_col_name = f"shift{s}"
                    if shift_col_name in table_headers:
                        col_idx = start_col + table_headers.index(shift_col_name)
                        label = SHIFT_TIME_LABELS.get(shift_col_name, "")
                        if label:
                            try:
                                shift_hours = set(hours_for_shift_label(label))
                            except Exception:
                                shift_hours = set()
                        else:
                            shift_hours = set()

                        # choose alternating color: odd->yellow, even->blue (matches previous look)
                        fmt = shift_yellow_fmt if (s % 2 == 1) else shift_blue_fmt

                        # Overwrite only the hourly rows that match shift_hours with formatted cell (write shift number)
                        if teams_table is not None and hourly_len > 0:
                            for ridx in range(hourly_len):
                                try:
                                    hour_val = int(teams_table.loc[ridx, "Hour"])
                                except Exception:
                                    continue
                                if hour_val in shift_hours:
                                    excel_row = first_data_row + ridx
                                    # write shift NUMBER (e.g., "1") instead of time-range
                                    worksheet.write(excel_row, col_idx, str(s), fmt)

                # --- MERGE header row across shift1..shift8 into single "Shifts" label ---
                if "shift1" in table_headers and "shift8" in table_headers:
                    first_shift_col = start_col + table_headers.index("shift1")
                    last_shift_col = start_col + table_headers.index("shift8")
                    worksheet.merge_range(header_row, first_shift_col, header_row, last_shift_col, "Shifts", shift_blue_fmt)
                    # Do NOT clear the first data row — merging header is sufficient to display "Shifts"

        # column widths
        for col_num in range(combined.shape[1]):
            col_name = combined.columns[col_num]
            # make shift columns a bit narrower
            if str(col_name).startswith("shift"):
                worksheet.set_column(col_num, col_num, 14)
            else:
                worksheet.set_column(col_num, col_num, 16)

    return combined, output_path
