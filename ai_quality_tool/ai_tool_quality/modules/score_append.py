import pandas as pd
import io
import os
import re

# ------------------------------------------------------
# Safe column normalizer (Preserved from your code)
# ------------------------------------------------------
def _safe_col(text: str) -> str:
    """
    Convert question text into a stable Excel-safe column name.
    """
    if not text:
        return ""
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def append_scores_row(score_dict, input_file, sheet_name="Score"):
    """
    Appends one agent score row to the Score sheet using SAFE column names.
    
    Args:
        score_dict (dict): The data to save.
        input_file (str or BytesIO): File path or SharePoint stream.
        sheet_name (str): Target sheet name.
        
    Returns:
        BytesIO: The updated Excel file as a stream (ready for upload).
    """

    scores = score_dict.get("Scores", {})

    # ---------------------------------------------------------
    # 1. Read existing file (Supports both Path string and Stream)
    # ---------------------------------------------------------
    all_sheets = {}
    
    try:
        # If it's a string path (Legacy/Local mode)
        if isinstance(input_file, str):
            if os.path.exists(input_file):
                all_sheets = pd.read_excel(input_file, sheet_name=None, dtype=object)
        
        # If it's a Stream (SharePoint mode)
        elif hasattr(input_file, 'read'):
            input_file.seek(0)
            try:
                all_sheets = pd.read_excel(input_file, sheet_name=None, dtype=object)
            except ValueError:
                # Handle empty stream case
                all_sheets = {}
                
    except Exception:
        # Fallback if file is corrupt or unreadable
        all_sheets = {}

    df = all_sheets.get(sheet_name, pd.DataFrame())

    # ---------------------------------------------------------
    # 2. Build agent row (Your Logic Preserved)
    # ---------------------------------------------------------
    row_data = {
        "Ticket Number": score_dict.get("Ticket Number", ""),
        "Quality Coach": score_dict.get("Quality Coach", ""),
        "Team Leader": score_dict.get("Team Leader", ""),
        "Analyst Name": score_dict.get("Analyst Name", ""),
        "Contact Type": score_dict.get("Contact Type", ""),
        
        # Added these fields from forms.py to ensure they are saved
        "Agent handling original contact?": score_dict.get("Agent handling original contact?", ""),
        "Was the ticket escalated...?": score_dict.get("Was the ticket escalated...?", ""),
        "Comments": score_dict.get("Comments", "")
    }

    # -------- Per-question scores --------
    # For the main submissions sheet (Sheet1) we preserve original question text
    # as column headers so answers are readable. For the 'Score' sheet we use
    # safe normalized column names.
    for q_text, sc in scores.items():
        if not q_text:
            continue
        if sheet_name and str(sheet_name).strip().lower() == "score":
            safe_key = _safe_col(q_text)
            if not safe_key:
                continue
            row_data[safe_key] = sc
        else:
            # preserve original question header for Sheet1 / other sheets
            row_data[str(q_text).strip()] = sc

    # Ensure meta columns for feedback workflow exist on Sheet1
    for meta_col in ["Token", "Token_Used", "Response", "Response_Comments", "Feedback_Timestamp", "Mail_Sent"]:
        if meta_col not in row_data:
            row_data[meta_col] = ""

    # -------- Totals --------
    row_data["total_score"] = score_dict.get("TotalScore", 0)
    row_data["max_total_score"] = score_dict.get("MaxTotalScore", 0)
    row_data["timestamp"] = score_dict.get("Timestamp", "")

    new_df = pd.DataFrame([row_data])

    # ---------------------------------------------------------
    # 3. Merge safely (Your Logic Preserved)
    # ---------------------------------------------------------
    if not df.empty:
        final_cols = list(df.columns)

        for c in new_df.columns:
            if c not in final_cols:
                final_cols.append(c)

        # Reindex to ensure columns align
        df = df.reindex(columns=final_cols)
        new_df = new_df.reindex(columns=final_cols)

        # Concat
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    all_sheets[sheet_name] = df

    # ---------------------------------------------------------
    # 4. Write to Stream (Required for SharePoint)
    # ---------------------------------------------------------
    output_stream = io.BytesIO()
    
    # We use valid Excel writer logic on the memory stream
    with pd.ExcelWriter(output_stream, engine="openpyxl") as writer:
        for sname, sdf in all_sheets.items():
            sdf.to_excel(writer, sheet_name=sname, index=False)
            
    output_stream.seek(0)
    return output_stream