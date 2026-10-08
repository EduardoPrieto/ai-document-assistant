from app.rag.document_loader import extract_text_from_pdf
from app.rag.chunker import create_document_chunks


PDF_PATH = "documents/return_policy.pdf"


pages = extract_text_from_pdf(PDF_PATH)

chunks = create_document_chunks(
    pages,
    chunk_size=500,
    chunk_overlap=100,
)

print(f"\nTotal pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")

for chunk in chunks:

    print("\n" + "=" * 60)

    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Chunk: {chunk['chunk']}")

    print("=" * 60)

    print(chunk["text"])