from typing import List, Dict

"""
Split text into overlapping chunks.

Args:
    text: Text to split.
    chunk_size: Maximum number of characters per chunk.
    chunk_overlap: Number of characters shared between chunks.
"""
def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def create_document_chunks(
    pages: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """
    Create chunks while preserving document metadata.
    """

    chunks = []

    for page in pages:

        page_chunks = split_text(
            text=page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(page_chunks):

            chunks.append(
                {
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": chunk_index,
                }
            )

    return chunks