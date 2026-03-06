# modules/mailer.py
import os
import requests
import msal
import json

# ===========================
# 🔧 Azure AD app registration
# ===========================
TENANT_ID = "8a38d5c9-ff2f-479e-8637-d73f6241a4f0"
CLIENT_ID = "eed0b8a6-a784-446e-a6ab-aa026d146e84"

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Mail.Send"]
SENDER_UPN = "email.automation@wisseninfotech.com"
SENDER_DISPLAY_NAME = "Jacobs GSD - Quality Team"
# Prefer a shared token cache at the package root so multiple modules share the same cache.
MODULE_DIR = os.path.dirname(__file__)
TOKEN_CACHE_FILE = os.path.join(MODULE_DIR, "token_cache.json")
SHARED_CACHE_FILE = os.path.abspath(os.path.join(MODULE_DIR, "..", "..", "token_cache.json"))


def _get_graph_token():
    """
    Acquire a delegated token via Device Code flow.
    Automatically refreshes the token silently if it's expired.
    """
    cache = msal.SerializableTokenCache()
    # Try shared cache first, then module-local cache
    loaded_cache_path = None
    for path in (SHARED_CACHE_FILE, TOKEN_CACHE_FILE):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cache.deserialize(f.read())
                loaded_cache_path = path
                break
        except Exception:
            continue

    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY, token_cache=cache)

    # 1. Try silent login (handles expired tokens using the Refresh Token)
    accounts = app.get_accounts()
    if accounts:
        # This will automatically use the refresh token if the access token is expired
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

        if result and "access_token" in result:
            # If the cache was updated (refreshed), save it back to disk(s) immediately
            if cache.has_state_changed:
                for out_path in (loaded_cache_path or TOKEN_CACHE_FILE, TOKEN_CACHE_FILE, SHARED_CACHE_FILE):
                    try:
                        if out_path:
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            with open(out_path, "w", encoding="utf-8") as f:
                                f.write(cache.serialize())
                    except Exception:
                        # ignore write errors and continue
                        pass
            return result["access_token"]

    # 2. Fallback — manual login using device code (only if refresh fails/expires)
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError("❌ Failed to create device flow. Try again.")
    
    print("\n🔐 AUTHENTICATION REQUIRED (Refresh Token Expired):")
    print(flow["message"])

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(f"❌ Failed to acquire delegated token: {result}")

    # Save the new cache after manual login to both shared and module locations (best-effort)
    for out_path in (TOKEN_CACHE_FILE, SHARED_CACHE_FILE):
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cache.serialize())
        except Exception:
            pass

    return result["access_token"]


def get_sender_name():
    return SENDER_DISPLAY_NAME


def send_mail(to, subject, html_body, cc=None):
    token = _get_graph_token()

    message = {
        "message": {
            "subject": subject,
            "body": {"contentType": "HTML", "content": html_body},
            "toRecipients": [{"emailAddress": {"address": to}}],
            "from": {"emailAddress": {"name": SENDER_DISPLAY_NAME, "address": SENDER_UPN}}
        },
        "saveToSentItems": "true",
    }

    if cc:
        if isinstance(cc, str):
            cc = [cc]
        message["message"]["ccRecipients"] = [
            {"emailAddress": {"address": addr}} for addr in cc
        ]

    url = "https://graph.microsoft.com/v1.0/me/sendMail"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=message)

    if resp.status_code not in (200, 202):
        raise RuntimeError(
            f"❌ Graph API sendMail failed: {resp.status_code}, {resp.text}"
        )