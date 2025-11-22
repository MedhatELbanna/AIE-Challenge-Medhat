# src/pdf_loader.py

from typing import List
from pypdf import PdfReader


def load_pdf_text(path: str) -> str:
    """
    Load a PDF file and return its full text as a single string.
    """
    reader = PdfReader(path)
    pages_text: List[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    return full_text.strip()
