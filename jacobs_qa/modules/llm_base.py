# modules/llm_base.py
class LLMClient:
    """Base class for any LLM provider (Groq, OpenAI, etc.)."""

    def chat_completion(self, prompt: str) -> str:
        """Send a prompt and return the model response as string."""
        raise NotImplementedError("LLMClient subclasses must implement chat_completion()")
