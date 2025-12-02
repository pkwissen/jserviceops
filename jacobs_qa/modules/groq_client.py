# modules/groq_client.py
import random
from groq import Groq
from .llm_base import LLMClient

class GroqClient(LLMClient):
    def __init__(self, api_keys: list[str], model: str = "llama-3.1-8b-instant", limit_per_key: int = 4500):
        if not api_keys:
            raise ValueError("No Groq API keys provided")

        self.api_keys = api_keys
        self.model = model
        self.limit_per_key = limit_per_key
        self.usage = {key: 0 for key in api_keys}  # track usage per key

    def _get_next_client(self) -> Groq | None:
        """Pick the next available API key with remaining quota."""
        import re

        for key in self.api_keys:
            clean_key = key.strip().strip('"').strip("'")
            if self.usage[key] < self.limit_per_key:
                # 🔎 DEBUG PRINT
                print("Using Groq key:", re.sub(r".{20}$", "********", clean_key))
                return Groq(api_key=clean_key), key
        return None, None

    def chat_completion(self, prompt: str) -> str:
        """Send a prompt to Groq and return the response text."""
        client, key = self._get_next_client()
        if not client:
            raise RuntimeError("All Groq API keys have reached their usage limits")

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )

        # count usage
        self.usage[key] += 1

        if not resp or not resp.choices:
            return ""

        return resp.choices[0].message.content.strip()

def clean_groq_output(text: str) -> str:
    """Utility to normalize Groq responses into clean text/JSON."""
    if not text:
        return ""
    return (
        text.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )