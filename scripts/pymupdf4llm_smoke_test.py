import tempfile
import time
from pathlib import Path

import pymupdf
import pymupdf4llm


def main() -> None:
    expected = "PYMUPDF4LLM_REAL_CONVERSION_OK"

    pdf_path = (
        Path(tempfile.gettempdir())
        / "ai_document_assistant_pymupdf4llm_test.pdf"
    )

    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "PyMuPDF4LLM Integration Test",
        fontsize=16,
    )
    page.insert_text(
        (72, 110),
        f"Expected phrase: {expected}",
        fontsize=11,
    )

    pdf.save(pdf_path)
    pdf.close()

    start = time.perf_counter()

    result = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    elapsed = time.perf_counter() - start

    if isinstance(result, list):
        markdown = "\n\n".join(
            chunk.get("text", "")
            for chunk in result
        )
    else:
        markdown = str(result)

    normalized = markdown.replace(r"\_", "_")

    print(f"Conversion time: {elapsed:.4f} seconds")
    print(f"Extracted characters: {len(markdown)}")
    print("\n--- Extracted Markdown ---")
    print(markdown)

    if expected not in normalized:
        raise SystemExit(
            "\nFAIL: Expected text was not extracted."
        )

    print("\nPASS: Real PyMuPDF4LLM conversion succeeded.")


if __name__ == "__main__":
    main()