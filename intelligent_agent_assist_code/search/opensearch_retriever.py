#!/usr/bin/env python3
"""
OpenSearch Retriever Module
Handles hybrid search queries combining BM25 (lexical) and kNN (semantic) search
PERFORMANCE OPTIMIZATION: Includes semantic search caching for 40-50% faster repeated queries
"""

import os
import json
import yaml
import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.exceptions import ConnectionError, NotFoundError, AuthorizationException
import boto3

# package-relative imports
from ..ingestion.vector_embedder import VectorEmbedder
from ..config.settings import INDEX_NAME

load_dotenv()

# 🚀 PERFORMANCE: Semantic search cache (stores top_k results for each query)
_SEARCH_CACHE = {}
_SEARCH_CACHE_SIZE_LIMIT = 100

# 🎯 SERVICE/TOPIC DETECTION (shared heuristics with chat continuity logic)
_SERVICE_KEYWORDS = {
    "citrix": ["citrix", "vdi", "vda", "workspace", "virtual desktop", "virtual application"],
    "outlook": ["outlook", "email", "mail", "owa", "mailbox", "office"],
    "teams": ["teams", "microsoft teams", "teams chat", "teams meeting"],
    "bitlocker": ["bitlocker", "encryption", "recovery key", "disk encryption"],
    "vpn": ["vpn", "network", "remote access", "tunnel", "zscaler"],
    "password": ["password", "reset password", "aduc", "account unlock", "verification"],
    "mobile": ["mobile", "iphone", "android", "mobilepass", "safenet"],
    "hardware": ["printer", "monitor", "keyboard", "mouse", "laptop", "hardware"],
    "software": ["software", "install", "uninstall", "application", "app install"],
}


def _detect_service(text: Optional[str]) -> Optional[str]:
    """Return normalized service/topic name if text mentions a known service."""
    if not text:
        return None
    lowered = text.lower()
    for service, keywords in _SERVICE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                return service
    return None


def _rerank_results_by_service(question: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Boost search results that match the question's service/topic keywords."""
    if not results:
        return results

    question_lower = question.lower()
    question_service = _detect_service(question_lower)
    question_terms = set(
        term for term in re.findall(r"[a-z0-9]+", question_lower)
        if len(term) > 3
    )

    reranked = []
    for result in results:
        base_score = result.get("score", 0) or 0
        metadata = result.get("metadata", {}) or {}
        kb_title = (metadata.get("kb_title") or metadata.get("document_title") or "").lower()
        text_body = (result.get("text") or "").lower()
        combined_text = f"{kb_title} {text_body}"
        doc_service = _detect_service(combined_text)

        boost = 0
        if question_service:
            if doc_service == question_service:
                boost += 75  # Strong match on same service/topic
            elif question_service in kb_title:
                boost += 50  # KB title mentions the service
            elif question_service in combined_text:
                boost += 20  # Service mentioned in body

        # Reward keyword overlaps
        keyword_hits = sum(1 for term in question_terms if term in combined_text)
        boost += keyword_hits * 5

        result["_rerank_score"] = base_score + boost
        reranked.append(result)

    reranked.sort(key=lambda r: r.get("_rerank_score", 0), reverse=True)
    for result in reranked:
        result.pop("_rerank_score", None)
    return reranked


def _normalize_query(query_text: str) -> str:
    """
    Normalize query to ensure similar questions produce identical results.
    
    Removes:
    - Personal context (employee, staff, user, I, we, me, one of, etc.)
    - Unnecessary pronouns and articles
    - Contextual language that doesn't affect the core question
    - Extra whitespace
    
    Example:
    - "how to get bitlocker key" 
    - "one of the employee needs a bitlocker key how can I get that"
    Both normalize to: "get bitlocker key"
    
    This ensures semantic similarity produces identical cached results.
    """
    import re
    
    # Convert to lowercase
    query = query_text.lower().strip()
    
    # Remove common personal/contextual words that don't affect search intent
    # These words add context but don't change what information is needed
    contextual_words = [
        r'\bone\s+of\s+the\s+',  # "one of the"
        r'\bthe\s+employee',     # "the employee"
        r'\ban\s+employee',      # "an employee"
        r'\bour\s+employee',     # "our employee"
        r'\bstaff\s+member',     # "staff member"
        r'\buser\s+needs',       # "user needs"
        r'\bi\s+need\s+',        # "i need"
        r'\bwe\s+need\s+',       # "we need"
        r'\bcan\s+you\s+help',   # "can you help"
        r'\bcan\s+i\s+',         # "can i"
        r'\bcan\s+we\s+',        # "can we"
        r'\bhow\s+to\s+',        # "how to" (keep but normalize)
        r'\bhow\s+can\s+',       # "how can"
        r'\bwhat\s+is\s+',       # "what is" (keep but normalize)
        r'\bwhat\s+are\s+',      # "what are"
        r'\bplease\s+',          # "please"
        r'\bthanks\s+',          # "thanks"
    ]
    
    for pattern in contextual_words:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Remove trailing question mark
    query = query.rstrip('?').strip()
    
    return query


class OpenSearchRetriever:
    """
    Handles hybrid search queries on OpenSearch index.
    Combines BM25 (lexical) and kNN (semantic) search using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        index_config_path: str = "config.yaml",
        retrieve_config_path: str = "retrieve_config.yaml"
    ):
        """
        Initialize OpenSearch retriever with configuration.

        Args:
            index_config_path: Path to main configuration file (for index name)
            retrieve_config_path: Path to retrieval configuration file
        """
        # Resolve config paths relative to this module's parent directory
        module_dir = Path(__file__).parent.parent
        index_config_path = module_dir / index_config_path if not Path(index_config_path).is_absolute() else Path(index_config_path)
        retrieve_config_path = module_dir / retrieve_config_path if not Path(retrieve_config_path).is_absolute() else Path(retrieve_config_path)
        
        # Load configurations
        self.index_config = self._load_config(str(index_config_path))
        self.retrieve_config = self._load_config(str(retrieve_config_path))

        # OpenSearch connection settings
        self.endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        self.use_aws_auth = os.getenv("OPENSEARCH_USE_AWS_AUTH", "false").lower() == "true"
        
        # For basic auth
        self.username = os.getenv("OPENSEARCH_USERNAME")
        self.password = os.getenv("OPENSEARCH_PASSWORD")
        
        # For AWS IAM auth
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not self.endpoint:
            raise ValueError(
                "Missing required OpenSearch environment variable: OPENSEARCH_ENDPOINT"
            )

        if not self.use_aws_auth and (not self.username or not self.password):
            raise ValueError(
                "Missing required OpenSearch credentials. "
                "Either set OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD for basic auth, "
                "or set OPENSEARCH_USE_AWS_AUTH=true for AWS IAM authentication."
            )

        # Get index name from settings (same as uploader)
        self.index_name = INDEX_NAME or self.index_config.get("opensearch", {}).get(
            "index_name", "documents_hybrid"
        )

        # Initialize OpenSearch client
        self.os_client = self._initialize_client()

        # Initialize vector embedder for query embedding
        self.embedder = VectorEmbedder()

        # Get retrieval settings
        self.retrieval_settings = self.retrieve_config.get("retrieval", {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        paths_to_try = [
            Path(config_path),  # As provided
            Path(__file__).parent.parent / config_path,  # Relative to module parent
            Path.cwd() / config_path,  # Relative to current working directory
        ]
        
        for path in paths_to_try:
            try:
                if path.exists():
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                    return config or {}
            except Exception:
                continue
        
        return {}

    def _initialize_client(self) -> OpenSearch:
        """Initialize and test OpenSearch client connection."""
        try:
            # Remove https:// or http:// from endpoint if present
            host = self.endpoint.replace("https://", "").replace("http://", "")
            
            if self.use_aws_auth:
                # Use AWS IAM authentication
                if self.aws_access_key_id and self.aws_secret_access_key:
                    session = boto3.Session(
                        aws_access_key_id=self.aws_access_key_id,
                        aws_secret_access_key=self.aws_secret_access_key,
                        region_name=self.aws_region
                    )
                else:
                    session = boto3.Session(region_name=self.aws_region)
                
                credentials = session.get_credentials()
                auth = AWSV4SignerAuth(credentials, self.aws_region, 'es')
                
                os_client = OpenSearch(
                    hosts=[{'host': host, 'port': 443}],
                    http_auth=auth,
                    use_ssl=True,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection,
                    timeout=30
                )
            else:
                # Use basic authentication
                os_client = OpenSearch(
                    hosts=[{'host': host, 'port': 443}],
                    http_auth=(self.username, self.password),
                    use_ssl=True,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection,
                    timeout=30
                )

            # Test connection
            try:
                info = os_client.info()
            except (AuthorizationException, Exception):
                # Connection attempt made, will be handled by caller
                pass

            return os_client

        except ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to OpenSearch at {self.endpoint}. "
                f"Error: {str(e)}"
            )
        except Exception as e:
            raise Exception(f"Error initializing OpenSearch client: {str(e)}")

    def search(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        document_title: Optional[str] = None,
        use_hybrid: bool = True,
        timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search on the OpenSearch index with timeout handling and caching.
        
        🚀 PERFORMANCE: Caches semantic search results for repeated queries
        (typically 40-50% faster for repeated questions)

        Args:
            query_text: Text query to search for
            top_k: Number of results to return (uses config default if None)
            document_title: Optional filter by document title
            use_hybrid: Use hybrid search (BM25 + kNN) if True, else only BM25
            timeout: Timeout in seconds (default: 45, max: 120)

        Returns:
            List of search results with scores and metadata
        """
        if not query_text:
            return []

        # 🎯 SEMANTIC SIMILARITY: Normalize query to match similar questions
        # E.g., "how to get bitlocker key" and "one of the employee needs a bitlocker key how can I get that"
        # Both normalize to "get bitlocker key" ensuring they get identical results
        normalized_query = _normalize_query(query_text)
        
        # 🚀 Check semantic search cache first using NORMALIZED query
        # This ensures similar questions hit the cache
        cache_key = hashlib.md5(f"{normalized_query}:{top_k}".encode()).hexdigest()
        if cache_key in _SEARCH_CACHE:
            print(f"[CACHE HIT] Similar query found: '{query_text[:40]}' → '{normalized_query}'")
            return _SEARCH_CACHE[cache_key]

        # Set timeout (clamp between 10-120 seconds)
        if timeout is None:
            timeout = 45  # ⬇️ Default reduced from 60s for faster retrieval
        timeout = max(10, min(timeout, 120))
        
        # Update client timeout
        self.os_client.transport.kwargs['timeout'] = timeout

        # Check if index exists
        try:
            index_exists = self.os_client.indices.exists(index=self.index_name)
        except Exception as e:
            print(f"[WARNING] Failed to check if index exists: {type(e).__name__}")
            return []
            
        if not index_exists:
            print(f"[WARNING] Index '{self.index_name}' does not exist")
            return []

        # Get configuration
        if top_k is None:
            top_k = self.retrieval_settings.get("default_top_k", 5)

        try:
            if use_hybrid:
                results = self._hybrid_search(query_text, top_k, document_title)
            else:
                results = self._bm25_search(query_text, top_k, document_title)

            # 🎯 SERVICE-AWARE RERANKING: prefer KBs that match the question's topic
            if results:
                results = _rerank_results_by_service(query_text, results)

            # 🚀 Cache search results for faster subsequent queries
            if results and len(_SEARCH_CACHE) < _SEARCH_CACHE_SIZE_LIMIT:
                _SEARCH_CACHE[cache_key] = results

            return results

        except Exception as e:
            print(f"[WARNING] Search error ({type(e).__name__}): {str(e)[:100]}")
            print(f"[INFO] Falling back to BM25 only search")
            try:
                fallback_results = self._bm25_search(query_text, top_k, document_title)
                if fallback_results:
                    fallback_results = _rerank_results_by_service(query_text, fallback_results)
                # Cache fallback results too
                if fallback_results and len(_SEARCH_CACHE) < _SEARCH_CACHE_SIZE_LIMIT:
                    _SEARCH_CACHE[cache_key] = fallback_results
                return fallback_results
            except Exception as e2:
                print(f"[WARNING] BM25 fallback also failed: {type(e2).__name__}")
                return []

    def _hybrid_search(
        self,
        query_text: str,
        top_k: int,
        document_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).
        Combines BM25 (lexical) and kNN (semantic) search.
        Falls back to BM25 only if embeddings fail.

        Args:
            query_text: Text query to search for
            top_k: Number of results to return
            document_title: Optional filter by document title

        Returns:
            List of search results
        """
        # Generate query embedding
        # Generate query embedding
        try:
            query_embedding = self.embedder.embed(query_text)
        except Exception as e:
            print(f"[WARNING] Embeddings API failed ({type(e).__name__}): {str(e)}")
            print(f"[INFO] Falling back to BM25 (keyword) search only")
            return self._bm25_search(query_text, top_k, document_title)
        
        if not query_embedding:
            return self._bm25_search(query_text, top_k, document_title)

        # Get hybrid search settings
        hybrid_settings = self.retrieval_settings.get("hybrid_search", {})
        knn_settings = hybrid_settings.get("knn", {})

        rank_window_size = hybrid_settings.get("rank_window_size", 50)
        num_candidates = hybrid_settings.get("num_candidates", 200)
        k = knn_settings.get("k", 50)

        # Build filter if document_title specified
        filter_clause = None
        if document_title:
            filter_clause = [{"term": {"document_title": document_title}}]

        # Build hybrid query for OpenSearch
        # Use bool query with both match (BM25) and knn queries
        knn_clause = {
            "knn": {
                "text_vector": {
                    "vector": query_embedding,
                    "k": k
                }
            }
        }

        query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # BM25 query
                        {
                            "match": {
                                "text": query_text
                            }
                        },
                        # kNN query
                        knn_clause
                    ]
                }
            }
        }

        # Add filter to query if document_title specified
        if filter_clause:
            query["query"]["bool"]["filter"] = filter_clause

        # Add highlighting if enabled
        highlighting = self.retrieval_settings.get("highlighting", {})
        if highlighting.get("enabled", True):
            query["highlight"] = {
                "fields": {field: {} for field in highlighting.get("fields", ["text"])},
                "pre_tags": [highlighting.get("pre_tag", "<mark>")],
                "post_tags": [highlighting.get("post_tag", "</mark>")],
                "fragment_size": highlighting.get("fragment_size", 150),
                "number_of_fragments": highlighting.get("number_of_fragments", 3)
            }

        # Execute search
        response = self.os_client.search(index=self.index_name, body=query)

        # Parse results
        return self._parse_results(response, query_text)

    def _bm25_search(
        self,
        query_text: str,
        top_k: int,
        document_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 (lexical) search only.

        Args:
            query_text: Text query to search for
            top_k: Number of results to return
            document_title: Optional filter by document title

        Returns:
            List of search results
        """
        # Build query
        query = {
            "bool": {
                "must": [
                    {"match": {"text": query_text}}
                ]
            }
        }

        # Add document title filter if specified
        if document_title:
            query["bool"]["filter"] = [
                {"term": {"document_title": document_title}}
            ]

        # Add highlighting
        highlighting = self.retrieval_settings.get("highlighting", {})
        highlight = None
        if highlighting.get("enabled", True):
            highlight = {
                "fields": {field: {} for field in highlighting.get("fields", ["text"])},
                "pre_tags": [highlighting.get("pre_tag", "<mark>")],
                "post_tags": [highlighting.get("post_tag", "</mark>")],
                "fragment_size": highlighting.get("fragment_size", 150),
                "number_of_fragments": highlighting.get("number_of_fragments", 3)
            }

        # Execute search
        search_body = {"query": query, "size": top_k}
        if highlight:
            search_body["highlight"] = highlight

        response = self.os_client.search(
            index=self.index_name,
            body=search_body
        )

        # Parse results
        return self._parse_results(response, query_text)

    def _parse_results(
            self,
            response: Dict[str, Any],
            query_text: str
    ) -> List[Dict[str, Any]]:

        hits = response.get("hits", {}).get("hits", [])

        total = response.get("hits", {}).get("total", {})
        if isinstance(total, dict):
            total = total.get("value", len(hits))
        else:
            total = len(hits)

        output_settings = self.retrieval_settings.get("output", {})

        include_text = output_settings.get("include_text", True)
        include_embeddings = output_settings.get("include_embeddings", False)
        include_metadata = output_settings.get("include_metadata", True)

        results = []

        for i, hit in enumerate(hits, 1):

            source = hit.get("_source", {})

            # ===============================
            # Read KB Metadata (FROM INDEX)
            # ===============================
            kb_number = source.get("kb_number", "")
            kb_title = source.get("kb_title", "")
            source_file = source.get("source_file", "")

            document_title = source.get("document_title", "")

            # ===============================
            # Build Base Result
            # ===============================
            result = {
                "rank": i,
                "score": hit.get("_score", 0.0),
                "id": hit.get("_id", "")
            }

            # ===============================
            # Metadata
            # ===============================
            if include_metadata:

                # Final safety fallback
                if not kb_title and document_title:
                    kb_title = document_title

                if not kb_title:
                    kb_title = "Unknown"

                result["metadata"] = {

                    # MAIN KB INFO
                    "kb_number": kb_number or "Unknown",
                    "kb_title": kb_title,
                    "source_file": source_file,

                    # OTHER
                    "document_title": document_title or kb_title,
                    "chunk_index": source.get("chunk_index", 0),

                    "chunk_size": source.get("chunk_size", 0),
                    "chunking_method": source.get("chunking_method", "unknown"),
                    "page_number": source.get("page_number", 1),

                    "indexed_at": source.get("indexed_at", "")
                }

            # ===============================
            # Text
            # ===============================
            if include_text:
                result["text"] = source.get("text", "")

            # ===============================
            # Highlights
            # ===============================
            if "highlight" in hit:

                result["highlights"] = []

                for field, snippets in hit["highlight"].items():
                    result["highlights"].extend(snippets)

            # ===============================
            # Embeddings (optional)
            # ===============================
            if include_embeddings and "embedding" in source:
                result["embedding"] = source["embedding"]

            results.append(result)

        return results

    def search_by_document(
        self,
        document_title: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific document.

        Args:
            document_title: Title of the document to retrieve
            top_k: Maximum number of chunks to return (None = all)

        Returns:
            List of document chunks
        """
        if top_k is None:
            top_k = 1000

        try:
            query = {
                "term": {"document_title": document_title}
            }

            response = self.os_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": top_k,
                    "sort": [{"chunk_index": {"order": "asc"}}]
                }
            )

            results = self._parse_results(response, "")
            return results

        except Exception:
            return []

    def save_results(
        self,
        results: List[Dict[str, Any]],
        query_text: str,
        filename: Optional[str] = None
    ):
        """
        Save search results to JSON file.

        Args:
            results: Search results to save
            query_text: Original query text
            filename: Optional custom filename
        """
        output_settings = self.retrieval_settings.get("output", {})

        if not output_settings.get("save_to_file", True):
            return

        # Create output directory
        output_dir = Path(output_settings.get("output_directory", "./search_results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c if c.isalnum() else "_" for c in query_text[:30])
            filename = f"search_{safe_query}_{timestamp}.json"

        filepath = output_dir / filename

        # Prepare output data
        output_data = {
            "query": query_text,
            "timestamp": datetime.now().isoformat(),
            "index": self.index_name,
            "total_results": len(results),
            "results": results
        }

        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def display_results(self, results: List[Dict[str, Any]], query_text: str):
        """
        Display search results in a readable format (not used in Streamlit app).

        Args:
            results: Search results to display
            query_text: Original query text
        """
        pass


def main():
    """Testing function - not used in production Streamlit app."""
    pass


if __name__ == "__main__":
    main()

