# modules/utils.py
import pandas as pd
import os
import urllib.parse
import uuid
from io import BytesIO

# ===== PRODUCTION-GRADE VALIDATION =====

MAX_FILE_SIZE_MB = 10  # Limit file uploads to 10MB
REQUIRED_COLUMNS = {"Date", "Ticket Number", "Quality Coach", "Team Leader", "Analyst Name"}

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Production-grade validation for uploaded Excel files.
    Checks file size, format, structure, and content.
    
    Returns: Validated DataFrame
    Raises: ValidationError with descriptive message
    """
    # 1. Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValidationError(f"File too large. Max size: {MAX_FILE_SIZE_MB}MB. Uploaded: {round(uploaded_file.size / 1024 / 1024, 1)}MB")
    
    # 2. Try to read the file
    try:
        df = pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")
    except Exception as e:
        # Check if it's really an Excel file
        raise ValidationError(f"Invalid Excel file or corrupted format: {str(e)[:100]}")
    
    # 3. Check if dataframe is empty
    if df.empty:
        raise ValidationError("Excel file is empty. Please upload a file with data.")
    
    # 4. Check required columns
    df_cols_normalized = {col.strip().lower() for col in df.columns}
    required_normalized = {col.strip().lower() for col in REQUIRED_COLUMNS}
    
    missing_cols = required_normalized - df_cols_normalized
    if missing_cols:
        raise ValidationError(f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}")
    
    # 5. Check for minimum rows
    if len(df) == 0:
        raise ValidationError("No data rows found in Excel file.")
    
    # 6. Check for critical missing values in key columns
    critical_cols = ["Ticket Number", "Analyst Name"]
    for col in critical_cols:
        # Find the actual column name (case-insensitive)
        actual_col = next((c for c in df.columns if c.strip().lower() == col.lower()), col)
        null_count = df[actual_col].isna().sum() + (df[actual_col].astype(str).str.strip() == "").sum()
        if null_count > 0:
            null_pct = (null_count / len(df)) * 100
            if null_pct > 10:  # More than 10% null values
                raise ValidationError(f"Column '{actual_col}' has {null_pct:.1f}% empty values. Max allowed: 10%")
    
    # 7. Sanitize column names (trim whitespace)
    df.columns = df.columns.str.strip()
    
    # 8. Clean data: trim whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
    
    return df

def safe_read_excel(path, sheet_name=0, dtype=None, **kwargs):
    """
    Safely read an Excel file with error recovery.
    If file is corrupted, backs it up and returns empty DataFrame.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    
    try:
        return pd.read_excel(path, sheet_name=sheet_name, dtype=dtype, engine="openpyxl", **kwargs)
    except Exception as e:
        # File is corrupted, try to backup and recover
        try:
            backup_path = path + ".corrupted"
            if not os.path.exists(backup_path):
                os.rename(path, backup_path)
                print(f"⚠️ Corrupted Excel file backed up to: {backup_path}")
        except Exception:
            pass
        return pd.DataFrame()

def load_mapping(path="Questionnaire_Checklist.xlsx"):
    """Load Analyst Name -> Email mapping from Excel."""
    return safe_read_excel(path)

def get_lead_email(lead_name, mapping_path="Questionnaire_Checklist.xlsx", sheet_name="Team_Leader"):
    """Return the email for a given Team Leader name from the mapping Excel (Team_Leader sheet)."""
    if pd.isna(lead_name) or str(lead_name).strip() == "":
        return None

    # Load mapping sheet
    df = safe_read_excel(mapping_path, sheet_name=sheet_name)
    if df.empty:
        return None
    
    df.columns = df.columns.str.strip()   # clean headers
    df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

    # Find the email by matching lead name (case-insensitive)
    mask = df["Team Leader"].astype(str).str.strip().str.lower() == str(lead_name).strip().lower()
    res = df.loc[mask, "Email"]

    if res.empty:
        return None

    return res.iloc[0]  # Return single email

def get_QC_email(lead_name, mapping_path="Questionnaire_Checklist.xlsx", sheet_name="Quality_coach"):
    """Return the email for a given Quality Coach name from the mapping Excel (Quality_coach sheet)."""
    if pd.isna(lead_name) or str(lead_name).strip() == "":
        return None

    # Load mapping sheet
    df = safe_read_excel(mapping_path, sheet_name=sheet_name)
    if df.empty:
        return None
    
    df.columns = df.columns.str.strip()   # clean headers
    df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

    # Find the email by matching lead name (case-insensitive)
    mask = df["Quality Coach"].astype(str).str.strip().str.lower() == str(lead_name).strip().lower()
    res = df.loc[mask, "Email"]

    if res.empty:
        return None

    return res.iloc[0]  # Return single email

def get_fixed_cc(mapping_path="Questionnaire_Checklist.xlsx", sheet_name="Fixed_CC"):
    """
    Return the list of email IDs from the Fixed_CC sheet in the mapping Excel.
    - The sheet should have a column named 'Email'.
    - Returns a list of email addresses or an empty list if none.
    """
    try:
        df = safe_read_excel(mapping_path, sheet_name=sheet_name)
        if df.empty:
            return []
        
        df.columns = df.columns.str.strip()  # clean header names

        if "Email" not in df.columns:
            print(f"⚠️ No 'Email' column found in {sheet_name} sheet.")
            return []

        # Strip and drop blanks
        df["Email"] = df["Email"].astype(str).str.strip()
        emails = [e for e in df["Email"].tolist() if e and e.lower() != "nan"]

        return emails

    except FileNotFoundError:
        print(f"❌ Mapping file not found: {mapping_path}")
        return []
    except Exception as e:
        print(f"❌ Failed to read Fixed_CC emails: {e}")
        return []

def get_email(analyst_name, mapping_path="Questionnaire_Checklist.xlsx", sheet_name="Analyst_Name"):
    """Return the email for a given Analyst name from the Analyst sheet."""
    # Load the Analyst sheet
    df = pd.read_excel(mapping_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()   # clean up headers
    if pd.isna(analyst_name):
        return None
    # Make sure expected columns exist
    if "Analyst Name" not in df.columns or "Email" not in df.columns:
        raise KeyError(f"Expected 'Analyst Name' and 'Email' columns in sheet {sheet_name}, found {df.columns.tolist()}")
    # Match name
    mask = df["Analyst Name"].astype(str).str.strip() == str(analyst_name).strip()
    res = df.loc[mask, "Email"]
    return res.iloc[0] if not res.empty else None

# modules/utils.py (snippet)
def format_body(row, sender_name):
    """
    Return an HTML string containing greeting, table of non-empty fields,
    closing, and a feedback prompt with Yes / No links that open the Streamlit feedback app.
    Links include ticket, analyst, click (Yes/No), and a unique Token for one-time use.
    """
    import os, urllib.parse, pandas as pd

    agent_name = row.get("Analyst Name", "Agent")
    ticket = str(row.get("Ticket Number", "") or "").strip()
    token = str(row.get("Token", "") or "").strip()  # 👈 include the unique token for one-time survey

    # Greeting + intro
    parts = [
        f"<p>Dear {agent_name},</p>",
        "<p>I hope you're doing well.<br>"
        "We have reviewed your recent interactions and would like to share feedback "
        "to help you continue delivering excellent service.</p>"
    ]

    # Preserve original column order but exclude meta columns
    exclude_cols = {"Token", "Token_Used", "Response", "Feedback_Timestamp"}
    col_order = [c for c in row.index if c not in exclude_cols]

    # If Grade exists, move it to the end of the order
    if "Grade" in col_order:
        col_order = [c for c in col_order if c != "Grade"] + ["Grade"]

    # Table of details
    parts.append("<table border='1' cellspacing='0' cellpadding='4' style='border-collapse: collapse;'>")

    for col in col_order:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip() not in ["", "0"]:
            # 🧩 Fix floating number issue (8.649999 → 8.65)
            if isinstance(val, (float, int)):
                val = f"{val:.2f}".rstrip("0").rstrip(".")
            parts.append(
                f"<tr>"
                f"<td style='font-weight:bold; padding:4px; border:1px solid #000;'>{col}</td>"
                f"<td style='padding:4px; border:1px solid #000;'>{val}</td>"
                f"</tr>"
            )
    parts.append("</table>")

    base = "https://8501-01k47bf9vkrm588t5zrmvev9z2.cloudspaces.litng.ai"
    # Build links with token
    def build_link(click_value):
        params = {
            "ticket": ticket,
            "analyst": agent_name,
            "click": click_value,
            "token": token
        }
        q = urllib.parse.urlencode(params)
        return f"{base}/feedback?{q}"

    yes_link = build_link("Yes")
    no_link = build_link("No")

    # Feedback prompt
    parts.append("<p><b>Kindly acknowledge the feedback using the options below:</b></p>")
    parts.append(
        "<p>"
        f"<a href=\"{yes_link}\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"text-decoration:none;\">"
        "<span style='display:inline-block; padding:8px 12px; background:#e6ffed; border-radius:6px;'>✅ Yes, it was helpful</span></a>"
        "&nbsp;&nbsp;"
        f"<a href=\"{no_link}\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"text-decoration:none;\">"
        "<span style='display:inline-block; padding:8px 12px; background:#ffecec; border-radius:6px;'>❌ Not quite</span></a>"
        "</p>"
    )

    # Closing
    parts.append(
        "<p>If you have any questions or would like to discuss this feedback further, "
        "feel free to reach out to your Quality Coach. We’re here to support your growth and success.</p>"
        "<p>Regards,<br>Jacobs GSD - Quality Assessors</p>"
        "<p>⚠️ Please do not reply to this email. This mailbox is not monitored.</p>"
    )

    return "\n".join(parts)
