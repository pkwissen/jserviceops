# modules/utils.py
import pandas as pd
import os
import urllib.parse
import uuid
def load_mapping(path="Questionnaire_Checklist.xlsx"):
    """Load Analyst Name -> Email mapping from Excel."""
    return pd.read_excel(path)

def get_lead_email(lead_name, mapping_path="Questionnaire_Checklist.xlsx", sheet_name="Team_Leader"):
    """Return the email for a given Team Leader name from the mapping Excel (Team_Leader sheet)."""
    if pd.isna(lead_name) or str(lead_name).strip() == "":
        return None

    # Load mapping sheet
    df = pd.read_excel(mapping_path, sheet_name=sheet_name)
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
    df = pd.read_excel(mapping_path, sheet_name=sheet_name)
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
        df = pd.read_excel(mapping_path, sheet_name=sheet_name)
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
