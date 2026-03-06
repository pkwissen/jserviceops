import os
from datetime import datetime, timezone
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from ..config.settings import OPENSEARCH_ENDPOINT, AWS_REGION, INDEX_NAME


class VectorOpenSearchUploader:
    def __init__(self):
        # Validate endpoint configuration
        if not OPENSEARCH_ENDPOINT:
            raise ConnectionError("❌ OPENSEARCH_ENDPOINT not configured in .env file")
        
        if not AWS_REGION:
            raise ConnectionError("❌ AWS_REGION not configured in .env file")

        try:
            host = OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "")
            
            # Get AWS credentials
            session = boto3.Session(region_name=AWS_REGION)
            credentials = session.get_credentials()
            
            if not credentials:
                raise ConnectionError(
                    "❌ AWS credentials not found. Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
                )
            
            # Create authentication
            auth = AWSV4SignerAuth(credentials, AWS_REGION, "es")
            
            # Initialize OpenSearch client
            self.client = OpenSearch(
                hosts=[{"host": host, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=10
            )
            
            # Test connection by pinging the cluster
            try:
                self.client.info()
            except Exception as conn_err:
                raise ConnectionError(
                    f"❌ Failed to connect to OpenSearch cluster at {OPENSEARCH_ENDPOINT}. "
                    f"Error: {str(conn_err)}. Check if the cluster is running and credentials are valid."
                )
                
        except ConnectionError:
            raise
        except Exception as e:
            raise ConnectionError(f"❌ OpenSearch connection error: {str(e)}")

    def create_index_if_not_exists(self):
        if self.client.indices.exists(index=INDEX_NAME):
            return

        mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {

                    # ===== KB METADATA =====
                    "kb_number": {"type": "keyword"},
                    "kb_title": {
                        "type": "text",
                        "fields": {"raw": {"type": "keyword"}}
                    },
                    "kb_url": {"type": "keyword"},
                    "source_file": {"type": "keyword"},

                    # ===== EXISTING FIELDS =====
                    "document_title": {
                        "type": "text",
                        "fields": {"raw": {"type": "keyword"}}
                    },
                    "document_key": {"type": "keyword"},
                    "text": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "chunk_start": {"type": "integer"},
                    "chunk_end": {"type": "integer"},
                    "chunk_size": {"type": "integer"},
                    "chunking_method": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "language": {"type": "keyword"},
                    "popularity": {"type": "integer"},
                    "indexed_at": {"type": "date"},
                    "created_at": {"type": "date"},

                    # ===== VECTOR =====
                    "text_vector": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "space_type": "cosinesimil",
                            "parameters": {
                                "m": 16,
                                "ef_construction": 128
                            }
                        }
                    }
                }
            }
        }

        self.client.indices.create(index=INDEX_NAME, body=mapping)

    def upload_chunk(
            self,
            title: str,
            text: str,
            vector: list,
            kb_number: str,
            kb_title: str,
            source_file: str,
            chunk_index: int = None,
            kb_url: str = None,
            chunk_start: int = None,
            chunk_end: int = None,
            chunk_size: int = None,
            chunking_method: str = None,
            page_number: int = None,
            language: str = None
    ):
        doc = {
            "document_title": title,
            "text": text,
            "text_vector": vector,

            # # ===== KB METADATA =====
            "kb_number": kb_number,
            "kb_title": kb_title,
            "kb_url": kb_url,
            "source_file": source_file,

            # ===== OPTIONAL =====
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "chunk_size": chunk_size,
            "chunking_method": chunking_method,
            "page_number": page_number,
            "language": language,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "indexed_at": datetime.now(timezone.utc).isoformat()
        }

        # Use deterministic doc id so each chunk maps back to the KB article
        doc_id = None
        try:
            if kb_number and chunk_index is not None:
                doc_id = f"{kb_number}::{chunk_index}"
        except Exception:
            doc_id = None

        if doc_id:
            self.client.index(index=INDEX_NAME, id=doc_id, body=doc)
        else:
            self.client.index(index=INDEX_NAME, body=doc)

    def refresh_index(self):
        """Refresh the index to make newly uploaded documents searchable."""
        try:
            self.client.indices.refresh(index=INDEX_NAME)
        except (OSError, IOError):
            pass
