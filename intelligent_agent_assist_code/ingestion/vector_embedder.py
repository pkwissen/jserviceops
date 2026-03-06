import time
from openai import OpenAI
from ..config.settings import OPENAI_API_KEY, OPENAI_BASE_URL

class VectorEmbedder:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=120  # 120 second timeout for embeddings
        )
        self.max_retries = 5
        self.chunk_count = 0

    def embed(self, text: str):
        """Generate embeddings with retry logic for network resilience."""
        self.chunk_count += 1
        clean_text = text.replace("\n", " ")
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model="wise-azure-text-embedding-3-small",
                    input=[clean_text]
                )
                if response and response.data and len(response.data) > 0:
                    return response.data[0].embedding
                else:
                    raise ValueError("Empty embedding response")
                    
            except Exception as e:
                error_msg = str(e).lower()
                attempt_num = attempt + 1
                
                # Adjust retry strategy based on error type
                is_retryable = any([
                    "connection" in error_msg,
                    "timeout" in error_msg,
                    "temporary" in error_msg,
                    "429" in error_msg,  # Rate limit
                    "unavailable" in error_msg,
                    "service" in error_msg and "unavailable" in error_msg
                ])
                
                if attempt_num < self.max_retries and is_retryable:
                    # Longer backoff for rate limiting
                    if "429" in error_msg:
                        wait_time = 10 * attempt_num  # 10s, 20s, 30s...
                    else:
                        wait_time = 3 ** attempt_num  # 3s, 9s, 27s...
                    
                    print(f"⏳ Chunk {self.chunk_count} embedding attempt {attempt_num}/{self.max_retries} failed, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    raise ConnectionError(f"{str(e)}")
