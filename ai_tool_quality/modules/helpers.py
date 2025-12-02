import pandas as pd
import os
import re

# ---------- normalization ----------
def _normalize(col_name: str) -> str:
    """Normalize column name for matching: strip, collapse whitespace, lower."""
    if col_name is None:
        return ""
    s = str(col_name).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

# ---------- contact sheet helpers ----------
def keywords_for_contact(contact_type):
    ct = (contact_type or "").lower()
    if "phone" in ct:
        return ["phone"]
    if "chat" in ct:
        return ["chat"]
    if "self" in ct or "service" in ct:
        return ["self", "self-service", "self service"]
    return [ct] if ct else []

def find_matching_sheets(contact_type, sheet_names):
    keys = keywords_for_contact(contact_type)
    matches = []
    for s in sheet_names:
        low = s.lower()
        for k in keys:
            if k in low:
                matches.append(s)
                break
    return matches

# ---------- status/option helpers ----------
def find_status_column_indices(df_raw):
    target = {"resolved": None, "escalated": None, "cancelled": None}
    nrows = min(3, df_raw.shape[0])
    for col_idx in range(df_raw.shape[1]):
        for r in range(nrows):
            try:
                cell = df_raw.iat[r, col_idx]
            except Exception:
                cell = None
            if pd.isna(cell) or cell is None:
                continue
            text = str(cell).strip().lower()
            if text.startswith("resolved"):
                target["resolved"] = col_idx
            if text.startswith("escalated"):
                target["escalated"] = col_idx
            if text.startswith("cancelled") or "canceled" in text:
                target["cancelled"] = col_idx
    return target

def _find_option_columns(df_raw):
    option_cols = []
    nrows = min(3, df_raw.shape[0])
    for col_idx in range(df_raw.shape[1]):
        for r in range(nrows):
            try:
                cell = df_raw.iat[r, col_idx]
            except Exception:
                cell = None
            if pd.isna(cell) or cell is None:
                continue
            txt = str(cell).strip().lower()
            if "option" in txt:
                option_cols.append(col_idx)
                break
    if not option_cols:
        for col_idx in range(2, min(6, df_raw.shape[1])):
            col_series = df_raw.iloc[:, col_idx].astype(str).fillna("")
            if any([s.strip() for s in col_series.tolist()]):
                option_cols.append(col_idx)
    return option_cols

# ---------- question loader ----------
def load_questions_filtered_by_status(path, sheet_name, ticket_status):
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str)
    if df_raw.shape[1] == 0:
        return []

    status_cols = find_status_column_indices(df_raw)
    chosen_idx = None
    sts = (ticket_status or "").strip().lower()
    if sts.startswith("resolved"):
        chosen_idx = status_cols.get("resolved")
    elif sts.startswith("escalated"):
        chosen_idx = status_cols.get("escalated")
    elif sts.startswith("cancel"):
        chosen_idx = status_cols.get("cancelled")

    option_cols = _find_option_columns(df_raw)

    pairs = []
    for ridx in range(df_raw.shape[0]):
        try:
            a = "" if pd.isna(df_raw.iat[ridx, 0]) else str(df_raw.iat[ridx, 0]).strip()
        except Exception:
            a = ""
        try:
            b = "" if pd.isna(df_raw.iat[ridx, 1]) else str(df_raw.iat[ridx, 1]).strip()
        except Exception:
            b = ""
        if not b or b.strip().lower().startswith("contact type -"):
            continue

        include = True
        if chosen_idx is not None and chosen_idx < df_raw.shape[1]:
            cell_val = df_raw.iat[ridx, chosen_idx]
            cell_text = "" if pd.isna(cell_val) or cell_val is None else str(cell_val).strip().lower()
            include = (cell_text == "yes")

        if not include:
            continue

        opts = []
        for cidx in option_cols:
            if cidx < df_raw.shape[1]:
                v = df_raw.iat[ridx, cidx]
                if not pd.isna(v) and str(v).strip() != "" and str(v) not in opts:
                    opts.append(str(v).strip())

        sr = a if a else str(ridx + 1)
        sr = re.sub(r"\.0+$", "", sr)
        pairs.append((sr, b, opts))
    return pairs

# ---------- append to Excel ----------
def append_submission_row(df_row: pd.DataFrame, out_path, sheet_name):
    df_row = df_row.copy()

    def build_target_mapping(existing_cols, incoming_cols):
        norm_to_existing = {}
        if existing_cols:
            for col in existing_cols:
                n = _normalize(col)
                if n not in norm_to_existing:
                    norm_to_existing[n] = col

        incoming_to_target = {}
        new_targets_in_order = []
        for inc in incoming_cols:
            n = _normalize(inc)
            if n in norm_to_existing:
                target = norm_to_existing[n]
            else:
                target = inc
                if target not in new_targets_in_order:
                    new_targets_in_order.append(target)
                norm_to_existing[n] = target
            incoming_to_target[inc] = target
        return incoming_to_target, new_targets_in_order

    def compute_final_columns(existing_cols, new_targets):
        final = list(existing_cols) if existing_cols else []
        for t in new_targets:
            if t not in final:
                final.append(t)
        has_comments = any(_normalize(c) == "comments" for c in final)
        has_timestamp = any(_normalize(c) == "timestamp" for c in final)
        final = [c for c in final if _normalize(c) not in ("comments", "timestamp")]
        if has_comments or any(_normalize(c) == "comments" for c in new_targets):
            final.append("Comments")
        if has_timestamp or any(_normalize(c) == "timestamp" for c in new_targets):
            final.append("Timestamp")
        return final

    incoming_cols = list(df_row.columns)

    if os.path.exists(out_path):
        try:
            existing = pd.read_excel(out_path, sheet_name=sheet_name, dtype=object)
        except Exception:
            existing = None
        if existing is None or existing.empty:
            existing_cols = []
        else:
            existing_cols = list(existing.columns)
    else:
        existing = None
        existing_cols = []

    incoming_to_target, new_targets = build_target_mapping(existing_cols, incoming_cols)
    final_cols = compute_final_columns(existing_cols, new_targets)

    new_row_values = []
    for c in final_cols:
        val = pd.NA
        for inc, target in incoming_to_target.items():
            if target == c:
                val = df_row.iloc[0][inc]
                break
        new_row_values.append(val if not pd.isna(val) else pd.NA)

    new_row_df = pd.DataFrame([new_row_values], columns=final_cols)

    if existing is not None and not existing.empty:
        existing_reordered = existing.reindex(columns=final_cols)
        result = pd.concat([existing_reordered, new_row_df], ignore_index=True)
    else:
        result = new_row_df

    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        result.to_excel(writer, sheet_name=sheet_name, index=False)
