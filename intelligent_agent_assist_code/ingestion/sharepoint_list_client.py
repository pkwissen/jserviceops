import os
import requests
import msal
from dotenv import load_dotenv
from urllib.parse import urlparse


# ================= LOAD ENV =================

load_dotenv()

TENANT_ID = os.getenv("SP_TENANT_ID")
CLIENT_ID = os.getenv("SP_CLIENT_ID")

SITE_URL = os.getenv("SP_SITE_URL")   # https://wisseninfotech0.sharepoint.com/sites/jacobs

TARGET_FOLDER = "Intelligent_Agent_Assist_KB_articles"

CACHE_FILE = "token_cache.bin"


# ================= AUTH (Delegated) =================

def _get_token():

    cache = msal.SerializableTokenCache()

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache.deserialize(f.read())

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"

    app = msal.PublicClientApplication(
        CLIENT_ID,
        authority=authority,
        token_cache=cache
    )

    accounts = app.get_accounts()

    result = None

    if accounts:
        result = app.acquire_token_silent(
            ["Sites.Read.All", "Files.Read.All"],
            account=accounts[0]
        )

    if not result:

        flow = app.initiate_device_flow(
            scopes=["Sites.Read.All", "Files.Read.All"]
        )

        # Note: flow['message'] contains device code instructions for authentication

        result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        raise Exception("Login failed")

    if cache.has_state_changed:
        with open(CACHE_FILE, "w") as f:
            f.write(cache.serialize())

    return result["access_token"]


# ================= SITE ID =================

def _get_site_id(token):

    site_url = SITE_URL.rstrip("/")

    parsed = urlparse(site_url)

    host = parsed.netloc
    path = parsed.path

    url = f"https://graph.microsoft.com/v1.0/sites/{host}:{path}"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    r = requests.get(url, headers=headers)

    r.raise_for_status()

    return r.json()["id"]


# ================= MAIN =================

def list_kb_files():
    """
    Returns list of KB files from SharePoint
    """

    token = _get_token()

    site_id = _get_site_id(token)

    url = (
        f"https://graph.microsoft.com/v1.0"
        f"/sites/{site_id}"
        f"/drive/root:/{TARGET_FOLDER}:/children"
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    r = requests.get(url, headers=headers)

    r.raise_for_status()

    data = r.json()

    files = []

    for item in data.get("value", []):

        if "file" in item:

            files.append({
                "name": item["name"],
                "size_kb": round(item["size"] / 1024, 2),
                "last_modified": item["lastModifiedDateTime"]
            })

    return files
