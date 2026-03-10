# settings.py
import os
import sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from dotenv import load_dotenv

load_dotenv()

# ===== SharePoint Configuration =====
SP_SITE_URL = os.getenv("SP_SITE_URL")
SP_TENANT_ID = os.getenv("SP_TENANT_ID")
SP_CLIENT_ID = os.getenv("SP_CLIENT_ID")
SP_CLIENT_SECRET = os.getenv("SP_CLIENT_SECRET")

# ===== OpenSearch Configuration =====
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "jacobs_vector_v2")

# Validate OpenSearch credentials
if not OPENSEARCH_ENDPOINT:
    raise ValueError("❌ OPENSEARCH_ENDPOINT not configured in .env")
if not AWS_ACCESS_KEY_ID:
    raise ValueError("❌ AWS_ACCESS_KEY_ID not configured in .env")
if not AWS_SECRET_ACCESS_KEY:
    raise ValueError("❌ AWS_SECRET_ACCESS_KEY not configured in .env")

# ===== Azure OpenAI Configuration =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")