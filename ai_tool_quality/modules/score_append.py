import pandas as pd
import os

def append_scores_row(score_dict, out_path, sheet_name):
    """
    Appends one row to the Score sheet.
    score_dict structure:

    {
        "Ticket Number": "...",
        "Quality Coach": "...",
        "Team Leader": "...",
        "Analyst Name": "...",
        "Contact Type": "...",
        "Scores": { q_text: score_value, ... },
        "TotalScore": int,
        "MaxTotalScore": int,
        "Timestamp": "..."
    }
    """

    scores = score_dict["Scores"]

    # Prepare final row for Excel
    row_data = {
        "Ticket Number": score_dict["Ticket Number"],
        "Quality Coach": score_dict["Quality Coach"],
        "Team Leader": score_dict["Team Leader"],
        "Analyst Name": score_dict["Analyst Name"],
        "Contact Type": score_dict["Contact Type"],
    }

    # Add per-question score columns
    for q_text, sc in scores.items():
        row_data[q_text] = sc

    # Add totals
    row_data["Total Score"] = score_dict["TotalScore"]
    row_data["Max Total Score"] = score_dict["MaxTotalScore"]
    row_data["Timestamp"] = score_dict["Timestamp"]

    df_new = pd.DataFrame([row_data])

    # Load existing Score sheet
    if os.path.exists(out_path):
        try:
            existing = pd.read_excel(out_path, sheet_name=sheet_name)
        except:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    # Merge columns automatically
    if not existing.empty:
        final_cols = list(existing.columns)

        # Add new columns if missing
        for c in df_new.columns:
            if c not in final_cols:
                final_cols.append(c)

        # Reindex both dfs
        existing = existing.reindex(columns=final_cols)
        df_new = df_new.reindex(columns=final_cols)

        final_df = pd.concat([existing, df_new], ignore_index=True)
    else:
        final_df = df_new

    # Save back to Excel
    with pd.ExcelWriter(out_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        final_df.to_excel(writer, sheet_name=sheet_name, index=False)
