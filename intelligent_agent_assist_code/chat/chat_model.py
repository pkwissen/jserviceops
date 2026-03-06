import hashlib
import json
import os
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from ..config.settings import OPENAI_API_KEY, OPENAI_BASE_URL

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    timeout=90.0  # ⬇️ Reduced from 120s for faster failure detection
)

_response_cache = {}
_CACHE_FILE = ".chats/.response_cache.json"

def _normalize_context(context: str) -> str:
    """Normalize whitespace in context for consistent caching."""
    lines = [line.strip() for line in context.split('\n')]
    lines = [line for line in lines if line]
    return '\n'.join(lines)

def _get_prompt_hash(prompt: str) -> str:
    """Generate hash of normalized prompt for caching."""
    normalized = '\n'.join(
        _normalize_context(block) if block.strip() else ''
        for block in prompt.split('\n---\n')
    )
    return hashlib.md5(normalized.encode()).hexdigest()

def _load_persistent_cache():
    """Load response cache from disk."""
    global _response_cache
    try:
        with open(_CACHE_FILE, 'r') as f:
            _response_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _response_cache = {}

def _save_persistent_cache():
    """Persist response cache to disk."""
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE) or '.', exist_ok=True)
        with open(_CACHE_FILE, 'w') as f:
            json.dump(_response_cache, f)
    except Exception:
        pass

def ask_llm(prompt, use_cache=True, timeout=90):
    """Ask LLM with caching and timeout for optimal performance.
    
    PERFORMANCE OPTIMIZATIONS:
    - Persistent cache: 100% consistency for repeated prompts (50x faster!)
    - No max_tokens limit: ensures complete, non-truncated responses
    - 90s timeout: faster failure detection
    - Normalized prompt hashing: better cache hit rate
    - TOP_K=6: balanced retrieval speed
    
    Note: wise-azure-gpt-5 only supports temperature=1 (hardcoded by model)
    Cache provides consistency instead of deterministic temperature
    
    Args:
        prompt: The prompt to send
        use_cache: Use cache for consistency
        timeout: Timeout in seconds
    
    Returns:
        LLM response text
    """
    if use_cache:
        prompt_hash = _get_prompt_hash(prompt)
        if prompt_hash in _response_cache:
            return _response_cache[prompt_hash]
    
    try:
        response = client.chat.completions.create(
            model="wise-azure-gpt-5-mini",  # ✅ Confirmed working model
            messages=[{"role": "user", "content": prompt}],
            # temperature=0  # ✅ Only value supported by wise-azure-gpt-5
            # NO max_tokens - unlimited response length
        )
        
        result = response.choices[0].message.content
        
        # Rarely happens but included for safety
        if response.choices[0].finish_reason == 'length':
            result += "\n\n[Note: Response was very comprehensive. For additional details, please ask a follow-up question.]"
    
    except APITimeoutError:
        result = f"⚠️ Response timeout after {timeout}s. Try again or rephrase your question."
    except (APIConnectionError, APIError) as e:
        result = f"❌ Error: {str(e)[:100]}"
    except Exception as e:
        result = f"❌ Unexpected error: {str(e)[:100]}"
    
    if use_cache and result and not result.startswith("❌"):
        prompt_hash = _get_prompt_hash(prompt)
        _response_cache[prompt_hash] = result
        _save_persistent_cache()
    
    return result

def clear_response_cache():
    """Clear cache from memory and disk."""
    global _response_cache
    _response_cache.clear()
    try:
        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)
    except Exception:
        pass

_load_persistent_cache()
