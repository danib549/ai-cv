"""CV file parser — extracts text from PDF and DOCX files."""

from __future__ import annotations

from pathlib import Path


def parse_cv(path: str | Path) -> str:
    """Extract raw text from a CV file (PDF or DOCX).

    Returns the full text content as a string.
    Raises ValueError for unsupported formats.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(path)
    elif suffix == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _parse_pdf(path: Path) -> str:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _parse_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
