from app.rag.document_loader import extract_text_from_pdf


PDF_PATH = "documents/return_policy.pdf"


pages = extract_text_from_pdf(PDF_PATH)

print(f"Pages found: {len(pages)}")

for page in pages:
    print("\n" + "=" * 60)
    print(f"Source: {page['source']}")
    print(f"Page: {page['page']}")
    print("=" * 60)
    print(page["text"][:500])