# ingestion/ingest_pipeline.py
import hashlib
import logging
import re
from typing import List, Sequence

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.opensearch import OpensearchVectorClient, OpensearchVectorStore

from intelligent_agent_assist_code.config.settings import INDEX_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, OPENSEARCH_ENDPOINT
from intelligent_agent_assist_code.search.opensearch_client import get_opensearch_client

logger = logging.getLogger(__name__)

# Common section headers in GSD KB articles
SECTION_HEADERS = re.compile(
    r"^(Troubleshooting|Resolution|Steps|Workaround|Escalation|Symptoms|Cause|"
    r"Information|Prerequisites|Instructions|Notes|Known\s*(?:Issues|Errors)|"
    r"Issue\s*Diagnosis|Table\s*of\s*Contents)",
    re.IGNORECASE | re.MULTILINE,
)

def _ensure_index(client):
    """Create index with correct knn mapping if missing."""
    if client.indices.exists(index=INDEX_NAME):
        return

    body = {
        "settings": {
            "index": {"knn": True},
            "number_of_shards": 1,
            "number_of_replicas": 1,
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    },
                },
                "text": {"type": "text"},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }

    client.indices.create(index=INDEX_NAME, body=body)
    logger.info("Created OpenSearch index %s with knn mapping", INDEX_NAME)


def _generate_chunk_id(kb_id: str, text: str) -> str:
    """Generate stable chunk ID for deduplication."""
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"{kb_id}_{content_hash}"


def _prepare_metadata(doc: Document) -> Document:
    """Normalize metadata and ensure required fields for retrieval accuracy."""
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}

    # Extract KB ID (e.g., KB0040001) from filename or content
    filename = metadata.get("file_name", "")
    content_snippet = doc.get_content()[:500]
    
    if "kb_id" not in metadata:
        kb_match = re.search(r"(KB\d{5,})", filename, re.IGNORECASE) or \
                   re.search(r"(KB\d{5,})", content_snippet, re.IGNORECASE)
        metadata["kb_id"] = kb_match.group(1).upper() if kb_match else "UNKNOWN"

    # Extract Topic
    if "topic" not in metadata:
        topic_match = re.search(r"GSD[_\s\-]+([A-Za-z0-9_\-]+)", filename, re.IGNORECASE)
        if topic_match:
            metadata["topic"] = topic_match.group(1).replace("_", " ").strip().title()
        else:
            metadata["topic"] = "General Support"

    # Generate clean title
    if "title" not in metadata or not metadata["title"]:
        base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
        title = re.sub(r"^KB\d+[_\s\-]*", "", base_name).replace("_", " ").replace("GSD ", "").strip()
        metadata["title"] = title or filename

    doc.metadata = metadata
    return doc


def _prepend_metadata_to_chunks(nodes: List) -> List:
    """Prepend KB metadata to chunk text for better semantic retrieval overlap."""
    for node in nodes:
        meta = node.metadata or {}
        kb_id = meta.get("kb_id", "UNKNOWN")
        topic = meta.get("topic", "")
        title = meta.get("title", "")

        original_text = node.get_content()
        
        # Prepend context header so vector search finds the KB via its ID or Topic
        prefix = f"[KB: {kb_id}] [Topic: {topic}] [Title: {title}]\n"
        node.set_content(prefix + original_text)

        # Set stable ID for deduplication
        node.id_ = _generate_chunk_id(kb_id, original_text)

    return nodes


def run_ingestion(documents: Sequence[Document]):
    """Ingest documents into OpenSearch with proxy-aware embedding configuration."""
    if not documents:
        raise ValueError("No documents provided for ingestion")

    try:
        client = get_opensearch_client()
        _ensure_index(client)

        # INTEGRATED FIX: Nested kwargs AND batch size 1 for Proxy stability
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_BASE_URL.rstrip("/"),
            embed_batch_size=1, 
            additional_kwargs={
                "extra_body": {
                    "model": "wise-azure-text-embedding-3-small"
                }
            },
        )

        # Chunking: 512 tokens is standard for KB troubleshooting steps
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

        vector_client = OpensearchVectorClient(
            endpoint=OPENSEARCH_ENDPOINT,
            index=INDEX_NAME,
            dim=1536,
            os_client=client,
        )
        vector_store = OpensearchVectorStore(vector_client)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Sanitize documents and prepare metadata
        valid_docs = [doc for doc in documents if doc.get_content().strip()]
        prepared_docs = [_prepare_metadata(doc) for doc in valid_docs]

        # Log document summary
        kb_ids = sorted(list(set(d.metadata.get("kb_id", "UNKNOWN") for d in prepared_docs)))
        logger.info("Processing %s documents covering KBs: %s", len(prepared_docs), kb_ids)

        # Parse into nodes
        nodes = node_parser.get_nodes_from_documents(prepared_docs)

        if not nodes:
            raise ValueError("Chunking produced no nodes; check document content")

        # Enrich chunks with metadata context
        nodes = _prepend_metadata_to_chunks(nodes)

        # Build and store index
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        logger.info(
            "Successfully ingested %s chunks from %s documents into index %s",
            len(nodes),
            len(documents),
            INDEX_NAME,
        )
        return index

    except Exception:
        logger.exception("Ingestion pipeline failed")
        raise