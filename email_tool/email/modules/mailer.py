# modules/mailer.py
import os
import requests
import msal
import json

# ===========================
# 🔧 Azure AD app registration
# ===========================
TENANT_ID = "8a38d5c9-ff2f-479e-8637-d73f6241a4f0"
CLIENT_ID = "eed0b8a6-a784-446e-a6ab-aa026d146e84"  # Your registered app client ID

# ---------------------------
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Mail.Send"]
SENDER_UPN = "email.automation@wisseninfotech.com"
SENDER_DISPLAY_NAME = "Jacobs GSD - Quality Team"
# Local token cache file (store access + refresh tokens)
TOKEN_CACHE_FILE = os.path.join(os.path.dirname(__file__), "token_cache.json")


def _get_graph_token():
    """
    Acquire a delegated token via Device Code flow (login once, then reuse token from cache).
    """
    # Load token cache if exists
    cache = msal.SerializableTokenCache()
    if os.path.exists(TOKEN_CACHE_FILE):
        try:
            with open(TOKEN_CACHE_FILE, "r") as f:
                cache.deserialize(f.read())
        except Exception:
            pass

    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY, token_cache=cache)

    # Try silent login using cached refresh token
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            return result["access_token"]

    # Fallback — manual login using device code (only first time)
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError("❌ Failed to create device flow. Try again.")
    print("\n🔐 AUTHENTICATION REQUIRED:")
    print(flow["message"])  # You’ll open this URL and enter code once

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(f"❌ Failed to acquire delegated token: {result}")

    # Save token cache for next time
    with open(TOKEN_CACHE_FILE, "w") as f:
        f.write(cache.serialize())

    return result["access_token"]


def get_sender_name():
    """Return the signed-in user email (UPN)."""
    return SENDER_DISPLAY_NAME


def send_mail(to, subject, html_body, cc=None):
    """Send mail using Microsoft Graph API (delegated permissions)."""
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

    # print(f"✅ Mail sent to {to}" + (f" (cc={cc})" if cc else ""))
