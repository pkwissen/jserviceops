import os
import requests
import msal
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ================= LOAD ENV =================
load_dotenv()

TENANT_ID = os.getenv("SP_TENANT_ID")
CLIENT_ID = os.getenv("SP_CLIENT_ID")
SITE_URL = os.getenv("SP_SITE_URL")
TARGET_FOLDER = "Intelligent_Agent_Assist_KB_articles"
CACHE_FILE = "token_cache.bin"

# Use the exact scopes from your working Code 2
SCOPES = ["Sites.Read.All", "Files.Read.All"]

def _timestamped_name(name: str) -> str:
    base, ext = os.path.splitext(name)
    ist = timezone(timedelta(hours=5, minutes=30))
    ts = datetime.now(ist).strftime("%d-%m-%Y_%H-%M-%S")
    return f"{base}_{ts}{ext}"

# ================= AUTH (Delegated) =================

def _get_token():
    cache = msal.SerializableTokenCache()

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache.deserialize(f.read())
            # print("DEBUG: Cache deserialized")

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"

    app = msal.PublicClientApplication(
        CLIENT_ID,
        authority=authority,
        token_cache=cache
    )

    accounts = app.get_accounts()
    result = None

    if accounts:
        # Try to get token silently using the EXACT same scopes
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

    if not result:
        logger.info("🔐 Cache expired or empty. Authentication required...")
        flow = app.initiate_device_flow(scopes=SCOPES)
        
        if "user_code" not in flow:
            raise Exception(f"Failed to create device flow: {flow}")

        logger.info("\n" + "="*60)
        logger.info(flow['message'])
        logger.info("="*60 + "\n")

        result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        raise Exception(f"Login failed: {result.get('error_description')}")

    if cache.has_state_changed:
        with open(CACHE_FILE, "w") as f:
            f.write(cache.serialize())
        # print("DEBUG: Cache updated")

    return result["access_token"]

# ================= SITE ID =================

def _get_site_id(token):
    site_url = SITE_URL.rstrip("/")
    parsed = urlparse(site_url)
    host = parsed.netloc
    path = parsed.path

    url = f"https://graph.microsoft.com/v1.0/sites/{host}:{path}"
    headers = {"Authorization": f"Bearer {token}"}
    
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["id"]

# ================= UPLOAD LOGIC =================

def upload_file_to_sharepoint(file_bytes, filename):
    """
    Unified upload function using your working Reference Scopes.
    """
    try:
        token = _get_token()
        site_id = _get_site_id(token)
        
        # Timestamp the filename for versioning
        new_name = _timestamped_name(filename)
        
        # Note: In Graph, 'Read.All' often allows writing if the user has 
        # personal permissions on the SharePoint site itself.
        upload_url = (
            f"https://graph.microsoft.com/v1.0/sites/{site_id}"
            f"/drive/root:/{TARGET_FOLDER}/{new_name}:/content"
        )

    

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream"
        }

        logger.info(f"📡 Uploading to SharePoint: {new_name}")
        r = requests.put(upload_url, headers=headers, data=file_bytes, timeout=60)
        r.raise_for_status()
        
        logger.info(f"✅ Success! Web URL: {r.json().get('webUrl')}")
        return r.json()

    except Exception as e:
        logger.error(f"❌ SharePoint Error: {e}")
        return None