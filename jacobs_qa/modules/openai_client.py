# modules/openai_client.py
import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# --------------------------------------------------------------------
# Config from environment (no hard-coding)
# --------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "wise-azure-gpt-4.1-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://wise-gateway.wisseninfotech.com")
DEFAULT_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIClient:
    """Thin wrapper around the OpenAI SDK (kept same API as before)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        key = api_key or DEFAULT_KEY
        url = base_url or DEFAULT_BASE_URL
        if not key:
            raise RuntimeError("❌ OPENAI_API_KEY is missing in environment.")
        self.client = OpenAI(api_key=key, base_url=url)

    def chat_completion(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 220,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=30,
        )
        return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------
# OpenAI fallback helper (Lightning endpoint) — safe & env-driven
# --------------------------------------------------------------------
def openai_chat_completion(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 220,
) -> str:
    """
    Sends a prompt to an OpenAI-compatible endpoint.
    Returns plain text (never raises).
    """
    try:
        key = DEFAULT_KEY
        url = DEFAULT_BASE_URL
        if not key:
            print("⚠️ OPENAI_API_KEY missing.")
            return ""

        client = OpenAI(api_key=key, base_url=url)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=30,
        )
        if resp and getattr(resp, "choices", None):
            raw = resp.choices[0].message.content or ""
            return raw.strip()
        return ""
    except Exception as e:
        print(f"OpenAI fallback failed: {e}")
        return ""
