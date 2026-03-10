import os
import requests
from typing import List, Dict

class AzureCohereReranker:
    def __init__(self):
        # These should be in your .env file
        self.api_key = os.getenv("AZURE_COHERE_API_KEY")
        # Ensure endpoint doesn't have trailing spaces
        self.endpoint = os.getenv("AZURE_COHERE_ENDPOINT", "").strip()
        
        if not self.api_key or not self.endpoint:
            raise ValueError("AZURE_COHERE_API_KEY and AZURE_COHERE_ENDPOINT must be set in the environment.")

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        """
        Reranks a list of strings and returns the top_n most relevant ones.
        """
        if not documents:
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        
        payload = {
            "model": "Cohere-rerank-v4.0-fast",
            "query": query,
            "documents": documents,
            "top_n": top_n
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()
            
            reranked_docs = []
            # API usually returns {"results": [{"index": 0, "relevance_score": 0.99}, ...]}
            for res in results.get("results", []):
                idx = res["index"]
                reranked_docs.append({
                    "content": documents[idx],
                    "index": idx,
                    "relevance_score": res["relevance_score"]
                })
                
            return reranked_docs
            
        except requests.exceptions.HTTPError as e:
            # Re-raise with response body to make debugging easier
            raise RuntimeError(f"Azure Cohere API error: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Reranking failed: {e}") from e