# modules/score.py

import os
from datetime import datetime
import pandas as pd
from modules.helpers import find_status_column_indices, _find_option_columns


def _find_score_columns(df_raw):
    """
    Detect columns containing 'Score1', 'Score2', ... in any of the first 5 rows.
    Handles merged cells and blank rows.
    """

    score_cols = []

    scan_rows = min(5, df_raw.shape[0])  # scan more rows to avoid merged blanks

    for col_idx in range(df_raw.shape[1]):
        for r in range(scan_rows):
            cell = df_raw.iat[r, col_idx]
            if pd.isna(cell) or cell is None:
                continue

            text = str(cell).strip().lower()

            # MUST match Score1, Score2, Score3, Score4
            if text.startswith("score"):
                score_cols.append(col_idx)
                break

    return score_cols

def _find_tscore_column(df_raw):
    scan_rows = min(5, df_raw.shape[0])
    for col_idx in range(df_raw.shape[1]):
        for r in range(scan_rows):
            cell = df_raw.iat[r, col_idx]
            if pd.isna(cell) or cell is None:
                continue
            text = str(cell).strip().lower()
            if text in ("tscore", "t score", "total score"):
                return col_idx
    return None


def load_score_mapping(path, sheet_name, ticket_status):
    """
    ------------------------------------------
    Reads sheet RAW (header=None), same style as question loader.
    Auto-detects:
      - Option columns
      - Score columns (Score1, Score2, Score3, Score4)
      - TScore column
    Scanning FIRST 5 ROWS to bypass merged header blanks.
    """

    # Load raw sheet
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str)

    # Status column detection
    status_cols = find_status_column_indices(df_raw)
    chosen_idx = None
    sts = (ticket_status or "").strip().lower()

    if sts.startswith("resolved"):
        chosen_idx = status_cols.get("resolved")
    elif sts.startswith("escalated"):
        chosen_idx = status_cols.get("escalated")
    elif sts.startswith("cancel"):
        chosen_idx = status_cols.get("cancelled")

    # 1️⃣ Detect option columns - same as your question loader
    option_cols = _find_option_columns(df_raw)

    # 2️⃣ Detect score columns (Score1–Score4)
    score_cols = []
    scan_rows = min(5, df_raw.shape[0])  # <-- FIXED

    for col in range(df_raw.shape[1]):
        for r in range(scan_rows):
            cell = df_raw.iat[r, col]
            if pd.isna(cell):
                continue
            txt = str(cell).strip().lower()

            # Must match Score1, Score2, ..., but not TScore
            if txt.startswith("score") and "tscore" not in txt:
                score_cols.append(col)
                break

    # 3️⃣ Detect TScore column
    tscore_col = None
    for col in range(df_raw.shape[1]):
        for r in range(scan_rows):
            cell = df_raw.iat[r, col]
            if pd.isna(cell):
                continue
            txt = str(cell).strip().lower()

            if txt in ("tscore", "t score", "total score"):
                tscore_col = col
                break

    mapping = {}

    # 4️⃣ Process each row exactly like question loader
    for ridx in range(df_raw.shape[0]):

        # --- Question text ---
        try:
            q_text = df_raw.iat[ridx, 1]
            q_text = "" if pd.isna(q_text) else str(q_text).strip()
        except:
            continue

        if not q_text or q_text.lower().startswith("contact type -"):
            continue

        # --- Status filtering ---
        include = True
        if chosen_idx is not None and chosen_idx < df_raw.shape[1]:
            st_cell = df_raw.iat[ridx, chosen_idx]
            st_txt = "" if pd.isna(st_cell) else str(st_cell).strip().lower()
            include = (st_txt == "yes")

        if not include:
            continue

        # --- Options ---
        options = []
        for c in option_cols:
            if c < df_raw.shape[1]:
                val = df_raw.iat[ridx, c]
                if val and str(val).strip() != "":
                    options.append(str(val).strip())

        # --- Scores (aligned with options) ---
        scores = {}
        for i, opt in enumerate(options):
            if i < len(score_cols):
                sc_val = df_raw.iat[ridx, score_cols[i]]
                try:
                    s = float(sc_val)
                    s = int(s) if s.is_integer() else s  # support decimals (e.g. 0.5)
                except:
                    s = 0
            else:
                s = 0
            scores[opt] = s

        # --- Max Score (TScore) ---
        if tscore_col is not None:
            ts_val = df_raw.iat[ridx, tscore_col]
            try:
                max_score = float(ts_val)
                max_score = int(max_score) if max_score.is_integer() else max_score
            except:
                max_score = 0
        else:
            max_score = 0

        # --- Save mapping ---
        mapping[q_text] = {
            "options": options,
            "scores": scores,
            "max_score": max_score
        }

    return mapping


def calculate_scores(answers_dict, score_map):
    """
    Calculate scores for submitted answers.

    answers_dict: {question_text: selected_option}
    score_map: from load_score_mapping

    Returns:
        {
            "per_question": { qtext: {"selected_option": x, "score": y, "max_score": z} },
            "total_score": int,
            "max_score": int,
            "percentage": float
        }
    """
    per_question = {}
    total_score = 0
    max_score = 0

    for q_text, ans in answers_dict.items():
        if q_text not in score_map:
            continue

        selected_option = ans
        q_info = score_map[q_text]
        q_max = q_info.get("max_score", 0)
        score_lookup = q_info.get("scores", {})

        if selected_option == "Blank":
            score = 0
        else:
            score = score_lookup.get(selected_option, 0)

        per_question[q_text] = {
            "selected_option": selected_option,
            "score": score,
            "max_score": q_max,
        }

        total_score += score
        max_score += q_max

    percentage = (total_score / max_score * 100) if max_score > 0 else 0.0

    return {
        "per_question": per_question,
        "total_score": total_score,
        "max_score": max_score,
        "percentage": round(percentage, 2),
    }
