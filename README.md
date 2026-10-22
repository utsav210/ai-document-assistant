# AI-Powered Document Assistant

An enterprise-ready Conversational Retrieval-Augmented Generation (RAG) assistant that allows users to upload documents (PDF, TXT) and interact with them via an intuitive conversational interface with strict evidence grounding, OCR support, tabular data preservation, and bounded conversational memory.

---

## 1. Project Overview

The **AI-Powered Document Assistant** processes unstructured and semi-structured documents, indexes their content in a persistent vector database, and generates factually grounded responses using modern Large Language Models (LLMs). The assistant is built with high reliability, session isolation, graceful error handling, and robust source citation.

### Key Capabilities
- **Multi-Format Document Ingestion**: Ingests multiple PDF and TXT files.
- **Multimodal & Tabular Extraction**: Handles text, embedded tables, and image/scanned content via OCR.
- **Semantic Vector Retrieval**: Chunks and embeds document content into a vector store (ChromaDB) with similarity scoring.
- **Strict Grounding & Hallucination Guardrails**: Responds accurately based only on retrieved evidence and clearly indicates when requested information is absent.
- **Context-Aware Conversational Memory**: Remembers the latest 3 conversation turns per session for coherent follow-up questions while preventing contextual drift.
- **Clear Source Attribution**: Reports source document names, page numbers, and chunk references.

---

## 2. Architecture & Directory Structure

```text
ai-document-assistant/
├── .agents/                    # Autonomous agent instructions & quality skills
│   └── skills/
│       ├── ai-document-assistant-engineer/
│       ├── enterprise-development-maintainer/
│       └── code-review/
├── backend/                    # Core application logic & API services
│   ├── api/                    # FastAPI routes and endpoint controllers
│   ├── core/                   # Configuration, logging, and security utilities
│   ├── document_processing/    # Parsers, OCR, table extractors, and chunkers
│   ├── rag/                    # Embeddings, vector store connectors, retrieval
│   └── services/               # Conversational memory and QA orchestration
├── frontend/                   # User interface (Streamlit / Web UI)
├── sample_data/                # Representative evaluation documents
├── tests/                      # Unit and integration test suites
├── .gitignore                  # Git tracking exclusion rules
└── README.md                   # Project documentation
```

---

## 3. Technology Stack

- **Runtime & Environment**: Python 3.12+ managed with `uv`
- **LLM & Embedding Providers**: Google Gemini / OpenAI / Groq / Anthropic (configurable)
- **Vector Database**: ChromaDB (isolated, persistent vector store)
- **Orchestration**: LangChain / LangGraph
- **Backend API**: FastAPI
- **Frontend UI**: Streamlit
- **Document Processing & OCR**: PyMuPDF (`fitz`), pdfplumber, pytesseract / pdf2image

---

## 4. Setup & Installation

### Prerequisites
- Python 3.12+ (or `uv` package manager)
- Tesseract OCR (installed on host system for scanned document OCR)

### Environment Initialization
```bash
# 1. Clone repository
git clone <repository_url>
cd ai-document-assistant

# 2. Set up virtual environment using uv
uv venv --python 3.12
# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Configure environment variables
# Create .env and supply your API credentials (e.g., GROQ_API_KEY)
```

---

## 5. Development Quality & Standards

This project adheres to rigorous enterprise engineering standards:
- **Separation of Concerns**: Modular components for ingestion, vectorization, retrieval, memory, and presentation.
- **Security-First**: Zero hardcoded credentials, input validation, path sanitization, and session isolation.
- **Resilience**: Comprehensive logging, explicit error handling, and graceful failure modes.
- **Automated Quality Verification**: Automated code reviews and testing via dedicated `.agents` skills.
