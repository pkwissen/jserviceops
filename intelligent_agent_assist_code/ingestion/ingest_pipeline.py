from .document_reader import read_document
from .text_cleaner import clean_text
from .chunker import chunk_text
from .vector_embedder import VectorEmbedder
from ..search.opensearch_uploader import VectorOpenSearchUploader
from .sharepoint_uploader import upload_to_sharepoint
from .kb_extractor import extract_kb_info
from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def run_ingestion(file):

    try:
        # --------------------------------------------------
        # Read + Clean
        # --------------------------------------------------
        try:
            raw_text = read_document(file)
            cleaned = clean_text(raw_text)
        except Exception as e:
            raise ValueError(f"Document reading error: {str(e)}")

        # --------------------------------------------------
        # Chunk
        # --------------------------------------------------
        try:
            chunks = chunk_text(
                cleaned,
                CHUNK_SIZE,
                CHUNK_OVERLAP
            )
        except Exception as e:
            raise ValueError(f"Document chunking error: {str(e)}")

        # --------------------------------------------------
        # Init Vector + OpenSearch
        # --------------------------------------------------
        try:
            embedder = VectorEmbedder()
        except Exception as e:
            raise ConnectionError(f"Failed to initialize embedder: {str(e)}")
        
        try:
            uploader = VectorOpenSearchUploader()
        except ConnectionError as e:
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenSearch: {str(e)}")

        try:
            uploader.create_index_if_not_exists()
        except Exception as e:
            raise ConnectionError(f"Failed to create OpenSearch index: {str(e)}")

        # --------------------------------------------------
        # Extract KB Metadata
        # --------------------------------------------------
        try:
            kb_info = extract_kb_info(cleaned, file.name)
            kb_number = kb_info.get("kb_number", "Unknown")
            kb_title = kb_info.get("kb_title", file.name)
            document_title = kb_title
        except Exception as e:
            raise ValueError(f"KB metadata extraction error: {str(e)}")

        # --------------------------------------------------
        # Embed + Upload
        # --------------------------------------------------
        try:
            for idx, chunk in enumerate(chunks):
                chunk_text_data = chunk["chunk_text"]
                
                try:
                    vector = embedder.embed(chunk_text_data)
                except ConnectionError as e:
                    raise ConnectionError(f"Embedding generation failed for chunk {idx}: {str(e)}")
                except Exception as e:
                    raise ConnectionError(f"Embedding generation failed for chunk {idx}: {type(e).__name__}: {str(e)}")
                
                try:
                    uploader.upload_chunk(
                        title=document_title,
                        text=chunk_text_data,
                        vector=vector,
                        kb_number=kb_number,
                        kb_title=kb_title,
                        source_file=file.name,
                        chunk_index=idx
                    )
                except Exception as e:
                    raise ConnectionError(f"Upload failed for chunk {idx}: {str(e)}")
        except Exception as e:
            raise

        # --------------------------------------------------
        # Refresh Index
        # --------------------------------------------------
        try:
            uploader.refresh_index()
        except Exception as e:
            raise ConnectionError(f"Failed to refresh OpenSearch index: {str(e)}")

        # --------------------------------------------------
        # Upload Original File to SharePoint
        # --------------------------------------------------
        try:
            try:
                file.seek(0)
            except Exception:
                pass

            upload_to_sharepoint(file)

        except Exception:
            pass

    except Exception:
        raise
