def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append({
            "chunk_index": idx,
            "chunk_text": chunk
        })

        idx += 1
        start += chunk_size - overlap

    return chunks
