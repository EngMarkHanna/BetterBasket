from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from eda_utils import OUTPUT_DIR, ROOT, ensure_output_dir


PDF_PATH = ROOT / "[BetterBasket] Engineering Technical Assessment.pdf"
OUTPUT_PATH = OUTPUT_DIR / "assessment_text.txt"


def clean_text(text: str) -> str:
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    ensure_output_dir()
    reader = PdfReader(str(PDF_PATH))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        pages.append(f"--- Page {page_number} ---\n{text}")
    OUTPUT_PATH.write_text("\n\n".join(pages), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Pages: {len(reader.pages)}")


if __name__ == "__main__":
    main()

