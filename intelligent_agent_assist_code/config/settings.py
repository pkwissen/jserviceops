import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
AWS_REGION = os.getenv("AWS_REGION")
INDEX_NAME = os.getenv("INDEX_NAME")

# ===== PERFORMANCE OPTIMIZATIONS =====
# Optimized for speed while maintaining response quality
CHUNK_SIZE = 700
CHUNK_OVERLAP = 200  # ⬆️ Increased from 100 (better context continuity)
TOP_K = 6  # ⚡ Balanced: better results (was 7), faster retrieval (now 6)

# Retrieval speed settings
RETRIEVAL_TIMEOUT = 45  # ⬇️ Reduced from 60s (faster failure detection)
LLM_MAX_TOKENS = None  # No limit - let model generate complete responses
