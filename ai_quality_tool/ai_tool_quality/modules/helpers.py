import pandas as pd
import os
import re

# ---------- normalization ----------
def _normalize(col_name: str) -> str:
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


# ---------- append to Excel (STREAM-ONLY, SHAREPOINT SAFE) ----------
def append_submission_row(df_row: pd.DataFrame, excel_stream, sheet_name):
    import io

    df_row = df_row.copy()

    # ---------------- NORMALIZATION HELPERS ----------------
    def build_target_mapping(existing_cols, incoming_cols):
        norm_to_existing = {}
        if existing_cols:
            for col in existing_cols:
                norm = _normalize(col)
                if norm not in norm_to_existing:
                    norm_to_existing[norm] = col

        incoming_to_target = {}
        new_targets = []

        for inc in incoming_cols:
            norm = _normalize(inc)
            if norm in norm_to_existing:
                target = norm_to_existing[norm]
            else:
                target = inc
                if target not in new_targets:
                    new_targets.append(target)
                norm_to_existing[norm] = target

            incoming_to_target[inc] = target

        return incoming_to_target, new_targets

    def compute_final_columns(existing_cols, new_targets):
        final = list(existing_cols) if existing_cols else []

        for t in new_targets:
            if t not in final:
                final.append(t)

        # Ensure Comments & Timestamp always at the end
        final = [c for c in final if _normalize(c) not in ("comments", "timestamp")]

        if any(_normalize(c) == "comments" for c in existing_cols + new_targets):
            final.append("Comments")
        if any(_normalize(c) == "timestamp" for c in existing_cols + new_targets):
            final.append("Timestamp")

        return final

    # ---------------- READ ALL SHEETS ----------------
    excel_stream.seek(0)
    try:
        all_sheets = pd.read_excel(excel_stream, sheet_name=None, dtype=object)
    except Exception:
        raise RuntimeError("Failed to read Excel stream from SharePoint")

    # # 🔒 HARD SAFETY GUARD — DO NOT ALLOW SCORE DELETION
    # if "Score" not in all_sheets:
    #     raise RuntimeError(
    #         "Score sheet missing from workbook. "
    #         "Aborting write to prevent Score deletion."
    #     )

    existing = all_sheets.get(sheet_name)
    existing_cols = list(existing.columns) if existing is not None else []
    incoming_cols = list(df_row.columns)

    incoming_to_target, new_targets = build_target_mapping(
        existing_cols, incoming_cols
    )

    final_cols = compute_final_columns(existing_cols, new_targets)

    # ---------------- BUILD NEW ROW ----------------
    new_row = []
    for col in final_cols:
        value = pd.NA
        for inc, tgt in incoming_to_target.items():
            if tgt == col:
                value = df_row.iloc[0][inc]
                break
        new_row.append(value)

    new_row_df = pd.DataFrame([new_row], columns=final_cols)

    # ---------------- MERGE ----------------
    if existing is not None and not existing.empty:
        existing = existing.reindex(columns=final_cols)
        result = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        result = new_row_df

    all_sheets[sheet_name] = result

    # ---------------- WRITE BACK SAFELY ----------------
    out_stream = io.BytesIO()

    with pd.ExcelWriter(out_stream, engine="openpyxl", mode="w") as writer:
        for sname, sdf in all_sheets.items():
            sdf.to_excel(writer, sheet_name=sname, index=False)

    out_stream.seek(0)
    return out_stream
