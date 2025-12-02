# modules/provider.py
import os
from dotenv import load_dotenv
# from .groq_client import GroqClient
from .openai_client import OpenAIClient, openai_chat_completion
# from .ollama_client import ollama_chat_completion

load_dotenv()

# ---- collect Groq keys --------------------------------------------------
api_keys = []

# Format A: comma-separated
env_keys = os.getenv("GROQ_API_KEY", "")
if env_keys:
    api_keys.extend([k.strip().strip('"').strip("'") for k in env_keys.split(",") if k.strip()])

# Format B: numbered vars
i = 1
while True:
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if not key:
        break
    api_keys.append(key.strip().strip('"').strip("'"))
    i += 1

api_keys = list(dict.fromkeys(api_keys))  # deduplicate
llm_client = OpenAIClient(
    api_key="sk-84HmOIqSTNnBlPL4K4Z95A",
    base_url="https://wise-gateway.wisseninfotech.com")
# ---- pick an LLM client -------------------------------------------------
# if api_keys:
#     llm_client = GroqClient(api_keys=api_keys)        # ✅ Groq default
# else:
#     # no Groq keys? fall back to OpenAI (Lightning endpoint)
#     llm_client = OpenAIClient(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         base_url="https://wise-gateway.wisseninfotech.com"
#     )
