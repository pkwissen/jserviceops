# modules/user_manager.py
import os
import pandas as pd
from typing import List, Dict, Tuple

# sheet identifiers we use in the workbook
USER_SHEETS = ["Team_Leader", "Quality_coach", "Analyst_Name", "Fixed_CC"]

# mapping from sheet identifier -> human column header for the "name" column
SHEET_NAME_HEADER = {
    "Team_Leader": "Team Leader",
    "Quality_coach": "Quality Coach",
    "Analyst_Name": "Analyst Name",
    "Fixed_CC" : "Name"
}

EMAIL_HEADER = "Email"


# ---------- low level helpers ----------
def _read_all_sheets(path: str) -> Dict[str, pd.DataFrame]:
    """Read all sheets from path; return dict of sheet_name -> DataFrame.
    If file doesn't exist return empty dict.
    """
    if not os.path.exists(path):
        return {}
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return {}
    sheets: Dict[str, pd.DataFrame] = {}
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=s, dtype=str)
        except Exception:
            df = pd.DataFrame()
        sheets[s] = df
    return sheets


def _write_all_sheets(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    """Overwrite workbook at path with the given sheets dict."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # ensure at least one sheet exists
    if not sheets:
        sheets = {USER_SHEETS[0]: pd.DataFrame(columns=[SHEET_NAME_HEADER[USER_SHEETS[0]], EMAIL_HEADER])}
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        for name, df in sheets.items():
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(columns=[SHEET_NAME_HEADER.get(name, "Name"), EMAIL_HEADER])
            # Guarantee columns exist
            if df.empty:
                df_to_write = pd.DataFrame(columns=[SHEET_NAME_HEADER.get(name, "Name"), EMAIL_HEADER])
            else:
                df_to_write = df
            # ensure column order: name header first, then Email
            name_col = SHEET_NAME_HEADER.get(name, None)
            if name_col and name_col in df_to_write.columns:
                cols = [name_col] + [c for c in df_to_write.columns if c != name_col and c != EMAIL_HEADER] + ([EMAIL_HEADER] if EMAIL_HEADER in df_to_write.columns else [])
                # make sure Email present
                if EMAIL_HEADER not in cols:
                    cols.append(EMAIL_HEADER)
                df_to_write = df_to_write.reindex(columns=cols)
            else:
                # fallback: ensure there is a name-like column and Email
                if EMAIL_HEADER not in df_to_write.columns:
                    df_to_write[EMAIL_HEADER] = ""
            try:
                df_to_write.to_excel(writer, sheet_name=str(name), index=False)
            except Exception:
                # fallback minimal
                pd.DataFrame(columns=[SHEET_NAME_HEADER.get(name, "Name"), EMAIL_HEADER]).to_excel(writer, sheet_name=str(name), index=False)


# ---------- public API ----------
def ensure_user_sheets(path: str) -> None:
    """Ensure the workbook exists and contains the three user sheets with proper headers."""
    sheets = _read_all_sheets(path)
    changed = False
    for sheet in USER_SHEETS:
        if sheet not in sheets:
            # create empty DF with proper headers
            name_col = SHEET_NAME_HEADER.get(sheet, "Name")
            sheets[sheet] = pd.DataFrame(columns=[name_col, EMAIL_HEADER])
            changed = True
        else:
            df = sheets[sheet]
            # If sheet exists but has no columns, create canonical headers
            if df is None or df.shape[1] == 0:
                name_col = SHEET_NAME_HEADER.get(sheet, "Name")
                sheets[sheet] = pd.DataFrame(columns=[name_col, EMAIL_HEADER])
                changed = True
            else:
                # If sheet has no Email column, add it
                if EMAIL_HEADER not in df.columns:
                    df = df.copy()
                    df[EMAIL_HEADER] = ""
                    sheets[sheet] = df
                    changed = True
                # If sheet does not have our expected name header but has some name-like header, keep it.
                # No action required.
    if changed or not os.path.exists(path):
        _write_all_sheets(path, sheets)


def list_user_sheets() -> List[str]:
    """Return user sheet identifiers."""
    return USER_SHEETS.copy()


def _detect_name_column_for_sheet(df: pd.DataFrame, sheet: str) -> str:
    """Return the best candidate column name to use as 'name' for a sheet."""
    if df is None or df.shape[1] == 0:
        return SHEET_NAME_HEADER.get(sheet, "Name")
    # prefer exact header match
    expected = SHEET_NAME_HEADER.get(sheet)
    if expected and expected in df.columns:
        return expected
    # otherwise find a column containing 'name' case-insensitive
    for c in df.columns:
        if isinstance(c, str) and "name" in c.lower():
            return c
    # fallback to first column
    return str(df.columns[0])


def get_users(excel_path: str, sheet: str) -> List[Tuple[str, str]]:
    """
    Return list of (name, email) tuples for a given sheet.
    If the sheet does not exist returns [].
    """
    sheets = _read_all_sheets(excel_path)
    if sheet not in sheets:
        return []
    df = sheets[sheet]
    if df is None or df.shape[0] == 0:
        return []
    name_col = _detect_name_column_for_sheet(df, sheet)
    # Ensure email col exists
    email_col = EMAIL_HEADER if EMAIL_HEADER in df.columns else (df.columns[1] if df.shape[1] > 1 else None)
    if email_col is None:
        return [(str(x).strip(), "") for x in df[name_col].fillna("").astype(str).tolist()]
    rows = []
    for _, r in df.iterrows():
        n = str(r.get(name_col, "")).strip()
        e = str(r.get(email_col, "")).strip()
        if not n and not e:
            continue
        rows.append((n, e))
    return rows


def add_user(excel_path: str, sheet: str, name: str, email: str) -> bool:
    """
    Add a user (name,email) to the specified sheet.
    Returns True if added; False if a duplicate (same name+email) was found.
    """
    name = str(name or "").strip()
    email = str(email or "").strip()
    if not name:
        raise ValueError("Name is required")
    sheets = _read_all_sheets(excel_path)
    if sheet not in sheets:
        # create new DF with expected headers
        df = pd.DataFrame(columns=[SHEET_NAME_HEADER.get(sheet, "Name"), EMAIL_HEADER])
    else:
        df = sheets[sheet].copy()

    name_col = _detect_name_column_for_sheet(df, sheet)
    if name_col not in df.columns:
        # replace with canonical header
        name_col = SHEET_NAME_HEADER.get(sheet, "Name")
        df[name_col] = df.iloc[:, 0].astype(str)

    email_col = EMAIL_HEADER if EMAIL_HEADER in df.columns else (df.columns[1] if df.shape[1] > 1 else EMAIL_HEADER)
    # Normalize existing
    existing = set()
    for _, r in df.iterrows():
        n = str(r.get(name_col, "")).strip().lower()
        e = str(r.get(email_col, "")).strip().lower() if email_col in df.columns else ""
        existing.add((n, e))
    key = (name.lower(), email.lower())
    if key in existing:
        return False

    # Append row using canonical columns
    new_row = {}
    new_row[name_col] = name
    new_row[EMAIL_HEADER] = email
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # update dict and write back
    sheets[sheet] = df
    # ensure other user sheets preserved/created
    for s in USER_SHEETS:
        if s not in sheets:
            sheets[s] = pd.DataFrame(columns=[SHEET_NAME_HEADER.get(s, "Name"), EMAIL_HEADER])
    _write_all_sheets(excel_path, sheets)
    return True


def remove_user(excel_path: str, sheet: str, name: str, email: str) -> bool:
    """
    Remove a user (by name+email) from the sheet. Returns True if removed, False otherwise.
    If multiple identical rows exist, all matching rows are removed.
    """
    name = str(name or "").strip()
    email = str(email or "").strip()
    if not name and not email:
        raise ValueError("Provide at least name or email to remove")

    sheets = _read_all_sheets(excel_path)
    if sheet not in sheets:
        return False
    df = sheets[sheet].copy()

    name_col = _detect_name_column_for_sheet(df, sheet)
    if name_col not in df.columns:
        return False
    # determine email column
    email_col = EMAIL_HEADER if EMAIL_HEADER in df.columns else (df.columns[1] if df.shape[1] > 1 else None)
    if email_col is None:
        # only name column exists: remove rows matching name
        mask_keep = df[name_col].fillna("").astype(str).str.strip().str.lower() != name.lower()
    else:
        mask_keep = ~(
            (df[name_col].fillna("").astype(str).str.strip().str.lower() == name.lower()) &
            (df[email_col].fillna("").astype(str).str.strip().str.lower() == email.lower())
        )
    if mask_keep.all():
        return False
    df2 = df.loc[mask_keep].reset_index(drop=True)
    sheets[sheet] = df2
    # ensure other user sheets present
    for s in USER_SHEETS:
        if s not in sheets:
            sheets[s] = pd.DataFrame(columns=[SHEET_NAME_HEADER.get(s, "Name"), EMAIL_HEADER])
    _write_all_sheets(excel_path, sheets)
    return True
