import sys
from pathlib import Path

# Ensure project root is on sys.path for direct invocation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.document_parser import parse_document
from backend.vector_store import index_parsed_document


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: uv run python scripts/index_document.py "path/to/file.pdf"'
        )

    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"Error: File not found: {path}")

    parsed = parse_document(path)
    result = index_parsed_document(parsed)

    print("Document indexed successfully.")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
