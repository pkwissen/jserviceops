import os
import json
import time
import base64
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from pathlib import Path

try:
    import msal
    import requests
except Exception:
    msal = None
    requests = None


def _timestamped_name(name: str) -> str:
    """Generate timestamped filename using Indian Standard Time (IST, UTC+5:30).
    Format: name_DD-MM-YYYY_HH-MM-SS.ext
    """
    base, ext = os.path.splitext(name)
    # IST is UTC+5:30
    ist = timezone(timedelta(hours=5, minutes=30))
    ts = datetime.now(ist).strftime("%d-%m-%Y_%H-%M-%S")
    return f"{base}_{ts}{ext}"


def upload_to_sharepoint(file):
    """Upload `file` to SharePoint (Microsoft Graph). Falls back to local save.

    Expects environment variables (loaded via dotenv elsewhere):
    - SP_SITE_URL, SP_TENANT_ID, SP_CLIENT_ID, SP_CLIENT_SECRET

    `file` should be a file-like object with `.name` and `.read()` (Streamlit UploadedFile works).
    Returns a dict with keys: file_name, file_path, web_url (when available).
    """
    from dotenv import load_dotenv

    load_dotenv()
    site_url = os.getenv("SP_SITE_URL")
    tenant = os.getenv("SP_TENANT_ID")
    client_id = os.getenv("SP_CLIENT_ID")
    client_secret = os.getenv("SP_CLIENT_SECRET")

    new_name = _timestamped_name(file.name)

    # If msal or requests not available or creds missing, fall back to local save
    if not msal or not requests or not (site_url and tenant and client_id and client_secret):
        out_dir = os.path.join(os.getcwd(), "sharepoint_outbox")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, new_name)
        with open(out_path, "wb") as f:
            f.write(file.read())
        return {
            "file_name": new_name,
            "file_path": out_path,
            "web_url": None,
            "note": "saved locally (msal/requests or creds missing)"
        }

    # Acquire token using client credentials for Microsoft Graph
    # Try to reuse an existing token from token_cache.json (if present)
    access_token = None
    token_note = None
    cache_paths = [
        Path(__file__).parent.parent / "token_cache.json",
        Path.cwd() / "token_cache.json",
    ]

    # Try token_cache.json entries first and prefer tokens that include SharePoint scopes
    for cp in cache_paths:
        try:
            if cp.exists():
                with open(cp, "r", encoding="utf-8") as fh:
                    cache = json.load(fh)
                atokens = cache.get("AccessToken", {}) or {}
                now = int(time.time())
                for v in atokens.values():
                    try:
                        expires = int(v.get("expires_on") or v.get("cached_at") or 0)
                    except Exception:
                        expires = 0
                    if expires and expires - 60 > now:
                        tok = v.get("secret")
                        if tok:
                            # Inspect token claims to check for SharePoint scopes/roles
                            try:
                                payload = tok.split(".")[1]
                                payload += "=" * (-len(payload) % 4)
                                claims = json.loads(base64.urlsafe_b64decode(payload))
                                scp = claims.get("scp", "")
                                roles = claims.get("roles", [])
                                if ("Sites.Read" in scp or "Files.ReadWrite" in scp) or any(
                                    r for r in roles if "Sites.Read" in r or "Files.ReadWrite" in r
                                ):
                                    access_token = tok
                                    token_note = f"token loaded from cache {cp} (has SharePoint scopes)"
                                    break
                            except Exception:
                                # If parsing fails, accept token as best-effort
                                access_token = tok
                                token_note = f"token loaded from cache {cp} (unverified claims)"
                                break
                if access_token:
                    break
        except Exception:
            continue

    # Next, try MSAL public client cache (device flow) token_cache.bin
    if not access_token and msal:
        bin_cache = Path(__file__).parent.parent / "token_cache.bin"
        if not bin_cache.exists():
            bin_cache = Path.cwd() / "token_cache.bin"

        delegated_scopes = ["Sites.Read.All", "Files.ReadWrite.All"]
        try:
            if bin_cache.exists():
                cache = msal.SerializableTokenCache()
                with open(bin_cache, "r", encoding="utf-8") as fh:
                    cache.deserialize(fh.read())
                public_app = msal.PublicClientApplication(
                    client_id, authority=f"https://login.microsoftonline.com/{tenant}", token_cache=cache
                )
                accounts = public_app.get_accounts()
                if accounts:
                    result = public_app.acquire_token_silent(delegated_scopes, account=accounts[0])
                    if result and result.get("access_token"):
                        access_token = result.get("access_token")
                        token_note = f"token loaded from device-cache {bin_cache}"
        except Exception:
            pass

        # If still no token, attempt interactive device flow to obtain delegated token
        if not access_token:
            try:
                cache = msal.SerializableTokenCache()
                public_app = msal.PublicClientApplication(
                    client_id, authority=f"https://login.microsoftonline.com/{tenant}", token_cache=cache
                )
                accounts = public_app.get_accounts()
                result = None
                if accounts:
                    result = public_app.acquire_token_silent(delegated_scopes, account=accounts[0])
                if not result:
                    flow = public_app.initiate_device_flow(scopes=delegated_scopes)
                    if "user_code" in flow:
                        # Print instructions to console for interactive completion
                        print(flow.get("message"))
                        result = public_app.acquire_token_by_device_flow(flow)
                if result and result.get("access_token"):
                    access_token = result.get("access_token")
                    token_note = "token acquired via device flow (delegated)"
                    if cache.has_state_changed:
                        try:
                            with open(bin_cache, "w", encoding="utf-8") as fh:
                                fh.write(cache.serialize())
                        except Exception:
                            pass
            except Exception:
                pass

    # Finally, fall back to client credentials flow
    if not access_token:
        authority = f"https://login.microsoftonline.com/{tenant}"
        app = msal.ConfidentialClientApplication(client_id, authority=authority, client_credential=client_secret)
        token_resp = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        access_token = token_resp.get("access_token")
        if not access_token:
            raise RuntimeError(f"Failed to acquire access token: {json.dumps(token_resp)}")
        token_note = "token acquired via MSAL client credentials"

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/octet-stream"}

    # Resolve site and path
    parsed = urlparse(site_url)
    hostname = parsed.netloc
    site_path = parsed.path  # e.g. /sites/jacobs

    # Get site id
    graph_base = "https://graph.microsoft.com/v1.0"
    site_url_endpoint = f"{graph_base}/sites/{hostname}:{site_path}"
    r = requests.get(site_url_endpoint, headers={"Authorization": f"Bearer {access_token}"})
    if r.status_code != 200:
        raise RuntimeError(f"Failed to resolve site: {r.status_code} {r.text}")
    site_info = r.json()
    site_id = site_info.get("id")

    # Upload to a folder within the site's default document library
    # Use path: /drive/root:/Shared Documents/Intelligent_Agent_Assist_KB_articles/{new_name}:/content
    target_folder = "Intelligent_Agent_Assist_KB_articles"
    upload_endpoint = f"{graph_base}/sites/{site_id}/drive/root:/{target_folder}/{new_name}:/content"

    file_bytes = file.read()
    resp = requests.put(upload_endpoint, headers=headers, data=file_bytes)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")

    upload_info = resp.json()
    web_url = upload_info.get("webUrl")

    return {
        "file_name": new_name,
        "file_path": f"{target_folder}/{new_name}",
        "web_url": web_url
    }

