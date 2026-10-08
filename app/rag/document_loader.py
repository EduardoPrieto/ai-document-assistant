from pathlib import Path

from pypdf import PdfReader

"""
Extract info from pdf
"""
def extract_text_from_pdf(file_path: str) -> list[dict]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {file_path}")

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "text": text.strip(),
                "page": page_number,
                "source": path.name,
            }
        )

    return pages