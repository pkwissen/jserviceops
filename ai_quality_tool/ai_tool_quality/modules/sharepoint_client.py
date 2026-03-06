import os
import io
import requests
import msal
import time
from dotenv import load_dotenv

load_dotenv()
# SharePoint Site Configuration
SITE_URL="https://wisseninfotech0.sharepoint.com/sites/jacobs"
TENANT_ID="8a38d5c9-ff2f-479e-8637-d73f6241a4f0"

# App Registration (Client Credentials)
CLIENT_ID="4ca1d37b-a699-4b76-939c-fcb9ea3be85f"
CLIENT_SECRET="H8M8Q~ok5BfaKchn3PZYsuf90SSCZ_px751sDdx6"

USE_SITES_SELECTED="false"
# ================= CONFIG =================
# TENANT_ID = os.getenv("SP_TENANT_ID")
# CLIENT_ID = os.getenv("SP_CLIENT_ID")
# Note: CLIENT_SECRET is NOT needed for this flow (it uses your user login)

SITE_HOST = "wisseninfotech0.sharepoint.com"
SITE_PATH = "/sites/jacobs" 

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# --- File Paths ---
# Checklist (Input)
INPUT_FILE_PATH = "/AI_tool_for_quality_team/data/Questionnaire_Checklist.xlsx"

# Feedback Records (Output 1 - For Dashboard/Emails)
OUTPUT_FILE_PATH = "/AI_tool_for_quality_team/output/feedback_records.xlsx"

# Submissions (Output 2 - For Raw Form Data)
SUBMISSIONS_FILE_PATH = "/AI_tool_for_quality_team/output/Questionnaire_submissions.xlsx"


# The permissions you listed (Delegated)
SCOPES = [
    "Files.Read",
    "Files.Read.All",
    "Sites.Read.All",
    "Files.ReadWrite.All",
    "User.Read"
]

# File to store the login token so you don't have to sign in every time
CACHE_FILE = "token_cache.bin"

# ================= AUTH (Device Code Flow + Persistence) =================
def _get_token():
    # 1. Initialize the Token Cache
    cache = msal.SerializableTokenCache()
    
    # 2. Check if a cache file exists and load it
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache.deserialize(f.read())

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    
    # 3. Create the App with the loaded cache
    app = msal.PublicClientApplication(
        CLIENT_ID, 
        authority=authority,
        token_cache=cache
    )

    # 4. Try to get token SILENTLY (from the cache file)
    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

    # 5. If no token in cache, start Device Code Flow (Manual Login)
    if not result:
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise ValueError("Failed to create device flow. Ensure 'Allow public client flows' is YES in Azure App Registration > Authentication.")

        print("\n" + "="*60)
        print(f"ACTION REQUIRED: {flow['message']}")
        print("="*60 + "\n")

        result = app.acquire_token_by_device_flow(flow)

    # 6. Save the new token back to the cache file
    if "access_token" in result:
        if cache.has_state_changed:
            with open(CACHE_FILE, "w") as f:
                f.write(cache.serialize())
        return result["access_token"]
    else:
        raise Exception(f"Login failed: {result.get('error_description')}")

def _headers(content_type="application/json"):
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": content_type,
    }

# ================= SHAREPOINT LOGIC =================
def _site_id():
    # 1. Find Site ID
    # print(f"Connecting to site: {SITE_HOST}{SITE_PATH}...")
    url = f"{GRAPH_BASE}/sites/{SITE_HOST}:{SITE_PATH}"
    resp = requests.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()["id"]

def _drive_root():
    # 2. Get Drive ID
    site_id = _site_id()
    return f"{GRAPH_BASE}/sites/{site_id}/drive/root:"

# ================= READ OPERATIONS =================

def download_checklist() -> io.BytesIO:
    """Downloads the Checklist file."""
    print(f"Downloading: {INPUT_FILE_PATH}")
    url = f"{_drive_root()}{INPUT_FILE_PATH}:/content"
    
    resp = requests.get(url, headers=_headers())
    if resp.status_code == 404:
        raise FileNotFoundError(f"File not found: {INPUT_FILE_PATH}")
    resp.raise_for_status()
    return io.BytesIO(resp.content)


def download_feedback_records():
    print(f"Checking for: {OUTPUT_FILE_PATH}")
    url = f"{_drive_root()}{OUTPUT_FILE_PATH}:/content"

    resp = requests.get(url, headers=_headers())
    if resp.status_code == 404:
        # 🔥 IMPORTANT FIX
        return io.BytesIO()
    resp.raise_for_status()
    return io.BytesIO(resp.content)


def download_submissions():
    """Downloads Questionnaire Submissions. Returns io.BytesIO() (empty) if 404."""
    print(f"Checking for: {SUBMISSIONS_FILE_PATH}")
    url = f"{_drive_root()}{SUBMISSIONS_FILE_PATH}:/content"
    
    resp = requests.get(url, headers=_headers())
    if resp.status_code == 404:
        # Return empty stream so pandas can create new DF
        return io.BytesIO()
    resp.raise_for_status()
    return io.BytesIO(resp.content)

# ================= WRITE OPERATIONS =================

def upload_feedback_records(stream: io.BytesIO):
    """Uploads to Feedback Records path."""
    print(f"Uploading to: {OUTPUT_FILE_PATH}")
    import io as _io
    import pandas as _pd

    stream.seek(0)
    url = f"{_drive_root()}{OUTPUT_FILE_PATH}:/content"

    # Try to GET the existing remote file so we can merge sheets if needed
    try:
        existing_resp = requests.get(url, headers=_headers())
    except Exception:
        existing_resp = None

    # If there is an existing file, attempt to preserve any sheets not present
    # in the incoming stream (this prevents accidental deletion of sheets
    # such as 'Score' when callers upload a single-sheet workbook).
    if existing_resp is not None and existing_resp.status_code == 200:
        try:
            existing_bytes = existing_resp.content
            existing_sheets = _pd.read_excel(_io.BytesIO(existing_bytes), sheet_name=None, dtype=object)
        except Exception:
            existing_sheets = {}

        try:
            stream.seek(0)
            incoming_sheets = _pd.read_excel(stream, sheet_name=None, dtype=object)
        except Exception:
            incoming_sheets = {}

        # If incoming read failed, just upload the original stream as-is
        if incoming_sheets:
            # Merge sheets: keep incoming sheets, and add any missing from existing
            for sname, sdf in existing_sheets.items():
                if sname not in incoming_sheets:
                    incoming_sheets[sname] = sdf

            # Write merged workbook to a new stream
            out_stream = _io.BytesIO()
            with _pd.ExcelWriter(out_stream, engine="openpyxl") as writer:
                for sname, sdf in incoming_sheets.items():
                    sdf.to_excel(writer, sheet_name=sname, index=False)
            out_stream.seek(0)

            headers = _headers(content_type="application/octet-stream")
            resp = requests.put(url, headers=headers, data=out_stream)
            resp.raise_for_status()
            print("Upload successful (merged with existing sheets).")
            return

    # Fallback: upload the provided stream as-is
    headers = _headers(content_type="application/octet-stream")
    stream.seek(0)
    resp = requests.put(url, headers=headers, data=stream)
    resp.raise_for_status()
    print("Upload successful.")

def upload_checklist(stream: io.BytesIO):
    """Uploads to Checklist path (used by User Manager)."""
    print(f"Uploading to: {INPUT_FILE_PATH}")
    stream.seek(0)
    url = f"{_drive_root()}{INPUT_FILE_PATH}:/content"
    
    headers = _headers(content_type="application/octet-stream")
    
    resp = requests.put(url, headers=headers, data=stream)
    resp.raise_for_status()
    print("Checklist updated successfully.")

def upload_submissions(stream: io.BytesIO):
    """Uploads to Submissions path."""
    print(f"Uploading to: {SUBMISSIONS_FILE_PATH}")
    stream.seek(0)
    url = f"{_drive_root()}{SUBMISSIONS_FILE_PATH}:/content"
    
    headers = _headers(content_type="application/octet-stream")
    
    resp = requests.put(url, headers=headers, data=stream)
    resp.raise_for_status()
    print("Submissions updated successfully.")