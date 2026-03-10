# document_loader.py
import logging
import os
import re
import tempfile
from typing import Dict, List, Optional

from llama_index.core import Document, SimpleDirectoryReader

logger = logging.getLogger(__name__)

# Regex to extract KB ID from filename (e.g., KB0015259_GSD_Citrix_Troubleshooting.docx)
KB_PATTERN = re.compile(r"(KB\d{5,})", re.IGNORECASE)
# Pattern to extract topic from filename after GSD_
TOPIC_PATTERN = re.compile(r"GSD[_\s\-]+([A-Za-z0-9_\-]+)", re.IGNORECASE)


def _extract_kb_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract KB ID, topic, and title from standardized KB filename."""
    metadata: Dict[str, str] = {}

    # Extract KB ID
    kb_match = KB_PATTERN.search(filename)
    if kb_match:
        metadata["kb_id"] = kb_match.group(1).upper()

    # Extract topic (e.g., Citrix, Outlook, Zscaler)
    topic_match = TOPIC_PATTERN.search(filename)
    if topic_match:
        raw_topic = topic_match.group(1).replace("_", " ").strip()
        metadata["topic"] = raw_topic

    # Generate clean title from filename
    base_name = os.path.splitext(filename)[0]
    # Remove KB prefix and clean up
    title = re.sub(r"^KB\d+[_\s\-]*", "", base_name)
    title = title.replace("_", " ").replace("GSD ", "").strip()
    metadata["title"] = title

    return metadata


def _extract_kb_metadata_from_docx(file_path: str) -> Dict[str, str]:
    """Extract KB ID and title from Word document content."""
    metadata: Dict[str, str] = {}
    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        # Check first 10 paragraphs for KB info
        for para in doc.paragraphs[:10]:
            text = para.text.strip()
            if not text:
                continue

            # Look for KB number in content (e.g., "KB0015259 - Latest Version")
            kb_match = KB_PATTERN.search(text)
            if kb_match and "kb_id" not in metadata:
                metadata["kb_id"] = kb_match.group(1).upper()

            # First non-empty paragraph is often the title (e.g., "GSD - Citrix: Troubleshooting")
            if "doc_title" not in metadata and len(text) > 5:
                metadata["doc_title"] = text

    except ImportError:
        logger.warning("python-docx not installed; skipping content metadata extraction")
    except Exception as e:
        logger.warning("Failed to extract metadata from docx content: %s", e)

    return metadata


def save_upload_to_temp(uploaded_file):
    """Save uploaded file to temp directory and return path."""
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return path, temp_dir


def load_documents_from_file(file_path: str) -> List[Document]:
    """Load documents with enriched KB metadata."""
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    filename = os.path.basename(file_path)
    file_metadata = _extract_kb_metadata_from_filename(filename)

    # For Word docs, also extract from content
    if file_path.lower().endswith((".docx", ".doc")):
        content_metadata = _extract_kb_metadata_from_docx(file_path)
        # Content metadata takes precedence for kb_id if found
        if content_metadata.get("kb_id"):
            file_metadata["kb_id"] = content_metadata["kb_id"]
        if content_metadata.get("doc_title"):
            file_metadata["doc_title"] = content_metadata["doc_title"]

    # Enrich each document with metadata
    for doc in documents:
        doc.metadata = doc.metadata or {}
        doc.metadata.update(file_metadata)
        doc.metadata["file_name"] = filename
        doc.metadata["source"] = file_path

    logger.info("Loaded %s document(s) from %s with metadata: %s", len(documents), filename, file_metadata)
    return documents


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Load all KB documents from a directory with enriched metadata."""
    all_docs: List[Document] = []

    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if not os.path.isfile(file_path):
            continue
        if filename.startswith(".") or filename.startswith("~"):
            continue

        try:
            docs = load_documents_from_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.error("Failed to load %s: %s", filename, e)

    logger.info("Loaded %s total documents from directory %s", len(all_docs), directory_path)
    return all_docs