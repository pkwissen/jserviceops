import pandas as pd
import re


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def normalize_key(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def find_header_row(df):
    """
    Finds the row index that contains 'Prompt' header.
    This is the real header row in your Excel.
    """
    for i in range(len(df)):
        row = df.iloc[i].astype(str).str.strip().str.lower()
        if "prompt" in row.values:
            return i
    return None


# -------------------------------------------------
# Main loader
# -------------------------------------------------
def load_score_mapping(excel_path, sheet_name, ticket_status):
    """
    Loads score + prompt mapping using real header row.
    Works with merged title rows.
    """

    raw_df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        dtype=object
    ).fillna("")

    header_row = find_header_row(raw_df)
    if header_row is None:
        raise ValueError(f"Header row with 'Prompt' not found in {sheet_name}")

    # Build proper dataframe
    df = raw_df.iloc[header_row + 1:].copy()
    df.columns = raw_df.iloc[header_row].astype(str).str.strip()

    mapping = {}

    # ---- Required columns ----
    required_cols = [
        "Resolved", "Escalated", "Cancelled",
        "Option1", "Option2", "Option3", "Option4",
        "Score1", "Score2", "Score3", "Score4",
        "TScore", "Prompt"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {sheet_name}")

    # ---- Detect Question column (first long-text column) ----
    question_col = None
    for col in df.columns:
        if col in required_cols:
            continue

        sample = df[col].astype(str).str.strip()
        sample = sample[sample != ""]
        if not sample.empty and sample.map(len).mean() > 40:
            question_col = col
            break

    if question_col is None:
        raise ValueError(f"Question column not detected in {sheet_name}")

    status_col = ticket_status  # Resolved / Escalated / Cancelled

    # ---- Build mapping ----
    for _, row in df.iterrows():

        if str(row[status_col]).strip().lower() != "yes":
            continue

        question_text = str(row[question_col]).strip()
        if not question_text:
            continue

        options = []
        scores = {}
        max_score = 0

        for opt_col, sc_col in zip(
            ["Option1", "Option2", "Option3", "Option4"],
            ["Score1", "Score2", "Score3", "Score4"]
        ):
            opt = str(row[opt_col]).strip()
            if not opt:
                continue

            try:
                sc = float(row[sc_col])
            except Exception:
                sc = 0

            options.append(opt)
            scores[opt] = sc
            max_score = max(max_score, sc)

        prompt_text = str(row["Prompt"]).strip()

        key = normalize_key(question_text)

        mapping[key] = {
            "original_question": question_text,
            "options": options,
            "scores": scores,
            "max_score": max_score,
            "prompt": prompt_text
        }

    return mapping
