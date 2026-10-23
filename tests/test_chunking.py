import pytest

from backend.chunking import create_chunks
from backend.document_parser import ParsedDocument


def test_pdf_chunks_preserve_page_numbers():
    parsed = ParsedDocument(
        source_name="report.pdf",
        file_type="pdf",
        content=(
            "<!-- Page 1 -->\n\n"
            "# First page\n\n"
            "Evidence about authentication and access control.\n\n"
            "<!-- Page 2 -->\n\n"
            "# Second page\n\n"
            "Findings about authorization and monitoring."
        ),
        extraction_method="pymupdf4llm",
        page_count=2,
    )

    chunks = create_chunks(
        parsed,
        chunk_size=45,
        chunk_overlap=5,
    )

    assert chunks
    assert {chunk.metadata["page_number"] for chunk in chunks} == {1, 2}

    assert all(chunk.metadata["source_name"] == "report.pdf" for chunk in chunks)


def test_txt_chunks_do_not_invent_page_numbers():
    parsed = ParsedDocument(
        source_name="notes.txt",
        file_type="txt",
        content="First paragraph.\n\nSecond paragraph.",
        extraction_method="text_reader",
    )

    chunks = create_chunks(parsed)

    assert chunks
    assert all("page_number" not in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["file_type"] == "txt" for chunk in chunks)


def test_invalid_chunk_overlap_is_rejected():
    parsed = ParsedDocument(
        source_name="notes.txt",
        file_type="txt",
        content="Some document text.",
        extraction_method="text_reader",
    )

    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        create_chunks(
            parsed,
            chunk_size=100,
            chunk_overlap=100,
        )


def test_empty_document_is_rejected():
    parsed = ParsedDocument(
        source_name="empty.txt",
        file_type="txt",
        content="   ",
        extraction_method="text_reader",
    )

    with pytest.raises(ValueError, match="empty document"):
        create_chunks(parsed)
