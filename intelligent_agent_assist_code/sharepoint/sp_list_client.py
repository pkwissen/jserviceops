# sharepoint/sp_list_client.py
import os
import logging
import requests
import msal
from dotenv import load_dotenv
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

load_dotenv()

TENANT_ID = os.getenv("SP_TENANT_ID")
CLIENT_ID = os.getenv("SP_CLIENT_ID")
SITE_URL = os.getenv("SP_SITE_URL") 
TARGET_FOLDER = "Intelligent_Agent_Assist_KB_articles"
CACHE_FILE = "token_cache.bin"

def _get_token():
    cache = msal.SerializableTokenCache()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache.deserialize(f.read())

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = msal.PublicClientApplication(CLIENT_ID, authority=authority, token_cache=cache)
    
    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(["Sites.Read.All", "Files.Read.All"], account=accounts[0])

    if not result:
        # In a Streamlit production environment, you'd usually use Client Secret (App-Only) 
        # but continuing with your Device Flow logic:
        flow = app.initiate_device_flow(scopes=["Sites.Read.All", "Files.Read.All"])
        if 'message' in flow:
            logger.info(flow['message']) # This will show in your terminal/logs
        result = app.acquire_token_by_device_flow(flow)

    if cache.has_state_changed:
        with open(CACHE_FILE, "w") as f:
            f.write(cache.serialize())
    return result.get("access_token")

def _get_site_id(token):
    parsed = urlparse(SITE_URL.rstrip("/"))
    url = f"https://graph.microsoft.com/v1.0/sites/{parsed.netloc}:{parsed.path}"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["id"]

def list_kb_files():
    """Returns list of KB files from SharePoint TARGET_FOLDER"""
    try:
        token = _get_token()
        if not token: return []
        site_id = _get_site_id(token)
        
        # Accessing the specific folder children
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{TARGET_FOLDER}:/children"
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        
        files = []
        for item in r.json().get("value", []):
            if "file" in item:
                files.append({
                    "File Name": item["name"],
                    "Size (KB)": round(item["size"] / 1024, 2),
                    "Last Modified": item["lastModifiedDateTime"][:10] # Simplified date
                })
        return files
    except Exception as e:
        logger.error(f"SharePoint List Error: {e}")
        return []