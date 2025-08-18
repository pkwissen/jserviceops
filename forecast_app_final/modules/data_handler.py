import os
import pandas as pd
import time
from openpyxl import load_workbook
# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
PROCESSED_PATH = os.path.join(BASE_DIR, "..", "data", "processed_data.xlsx")
TRANSFORMED_PATH = os.path.join(BASE_DIR, "..", "data", "transformed_data.xlsx")
TRANSFORMED_PHONE59_PATH = os.path.join(BASE_DIR, "..", "data", "transformed_Phone59.xlsx")
def map_hour_to_slot(hour):
    return ((hour - 6) % 24) + 1

def transform_data(df, output_path):
    if 'Created' in df.columns:
        df.rename(columns={'Created': 'sys_created_on'}, inplace=True)
    if 'Contact type' in df.columns:
        df.rename(columns={'Contact type': 'task.contact_type'}, inplace=True)

    df['Date'] = pd.to_datetime(df['sys_created_on']).dt.date
    df['Hour'] = pd.to_datetime(df['sys_created_on']).dt.hour
    df['Slot'] = df['Hour'].apply(map_hour_to_slot)

    grouped = df.groupby(['task.contact_type', 'Date', 'Slot']).size().reset_index(name='Count')

    pivoted_data = {}
    for contact_type in ['Chat', 'Phone', 'Self-service']:
        filtered = grouped[grouped['task.contact_type'] == contact_type]
        pivot = filtered.pivot(index='Date', columns='Slot', values='Count').fillna(0).astype(int)

        for col in range(1, 25):
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[[col for col in range(1, 25)]]
        pivot.columns = [str(col) for col in pivot.columns]

        pivoted_data[contact_type] = pivot

    # ✅ Save all contact type dataframes to different sheets
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        for contact_type, pivot_df in pivoted_data.items():
            pivot_df.to_excel(writer, sheet_name=contact_type)

    return pivoted_data

def transform_new_data(uploaded_file, transformed_path):
    df = pd.read_excel(uploaded_file)
    transform_data(df, transformed_path)
    return transformed_path

def merge_with_existing(new_transformed_path, processed_path):
    if not os.path.exists(new_transformed_path):
        raise FileNotFoundError(f"{new_transformed_path} does not exist.")

    new_xls = pd.ExcelFile(new_transformed_path, engine='openpyxl')

    if os.path.exists(processed_path):
        processed_xls = pd.ExcelFile(processed_path, engine='openpyxl')
        merged_sheets = {}

        all_sheets = set(processed_xls.sheet_names).union(set(new_xls.sheet_names))

        for sheet in all_sheets:
            df_existing = processed_xls.parse(sheet) if sheet in processed_xls.sheet_names else pd.DataFrame()
            df_new = new_xls.parse(sheet) if sheet in new_xls.sheet_names else pd.DataFrame()

            df_merged = pd.concat([df_existing, df_new], ignore_index=True)

            if 'Date' in df_merged.columns:
                df_merged.drop_duplicates(subset='Date', keep='last', inplace=True)

            merged_sheets[sheet] = df_merged

    else:
        os.rename(new_transformed_path, processed_path)
        return pd.read_excel(processed_path, sheet_name=None)

    # ✅ Write merged data to processed.xlsx
    with pd.ExcelWriter(processed_path, engine='openpyxl') as writer:
        for sheet, df in merged_sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

    # # ✅ Safely delete the temporary transformed file
    # try:
    #     os.remove(new_transformed_path)
    # except PermissionError:
    #     time.sleep(1)  # wait briefly and retry
    #     try:
    #         os.remove(new_transformed_path)
    #     except Exception as e:
    #         print(f"⚠️ Could not delete file: {new_transformed_path} due to: {e}")

    return merged_sheets

def load_processed_data(path):
    return pd.read_excel(str(path), sheet_name=None, engine='openpyxl')


import pandas as pd

def transform_data_59(uploaded_file59, transformed_phone59_path):
    # Load raw CSV
    df = pd.read_csv(uploaded_file59)

    # Normalize column names
    df.columns = df.columns.str.strip().str.upper()

    # Filter out CALL TYPE = 'Manual'
    if "CALL TYPE" in df.columns:
        df = df[~df["CALL TYPE"].astype(str).str.strip().str.lower().eq("manual")]

    # Convert DATE to datetime and extract date
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce').dt.date
    else:
        raise ValueError("The input file does not contain a 'DATE' column.")

    # Convert HOUR from "HH:MM" to integer hour
    if "HOUR" in df.columns:
        df["HOUR"] = df["HOUR"].astype(str).str.split(':').str[0].astype(int, errors='ignore')
    else:
        raise ValueError("The input file does not contain a 'HOUR' column.")

    # Drop rows with invalid or missing hours
    df = df[df["HOUR"].between(0, 23)]

    # Hour mapping from 0–23 to 1–24 format
    hour_map = {
        6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7,
        13: 8, 14: 9, 15: 10, 16: 11, 17: 12, 18: 13,
        19: 14, 20: 15, 21: 16, 22: 17, 23: 18,
        0: 19, 1: 20, 2: 21, 3: 22, 4: 23, 5: 24
    }

    # Apply mapping
    df["MAPPED_HOUR"] = df["HOUR"].map(hour_map)

    # Group by date and mapped hour, count occurrences
    grouped = df.groupby(["DATE", "MAPPED_HOUR"]).size().reset_index(name="Count")

    # Pivot to wide format
    pivot = grouped.pivot(index="DATE", columns="MAPPED_HOUR", values="Count").fillna(0).astype(int)

    # Ensure all columns 1–24 are present
    for col in range(1, 25):
        if col not in pivot.columns:
            pivot[col] = 0

    # Reorder columns
    pivot = pivot[sorted(pivot.columns)]

    # Rename index to "Date" (since DATE is the index after pivot)
    pivot.index.name = "Date"

    # Rename columns as strings
    pivot.columns = [str(col) for col in pivot.columns]

    # Save to Excel
    pivot.reset_index().to_excel(str(transformed_phone59_path), index=False)


def add_transformed_phone59_sheet(processed_path, transformed_phone59_path, sheet_name="Phone59"):
    # Load source data
    df = pd.read_excel(str(transformed_phone59_path))

    # If target file exists, append sheet (replace if already exists)
    if os.path.exists(processed_path):
        with pd.ExcelWriter(processed_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Create new file with sheet
        with pd.ExcelWriter(processed_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
