# AI-Powered Document Assistant

An enterprise-ready Conversational Retrieval-Augmented Generation (RAG) assistant that allows users to upload documents (PDF, DOCX, TXT) and interact with them through an intuitive conversational interface with strict evidence grounding, OCR table/image extraction, bounded conversational memory, and source attribution.

---

## 1. Project Overview

The **AI-Powered Document Assistant** bridges unstructured enterprise documents and modern Large Language Models (LLMs). It ingests multi-format documents, splits and indexes content into an embedded vector store with true cosine semantic similarity scoring, and provides grounded, factual answers to user inquiries.

### Key Capabilities
- **Multi-Format Ingestion**: Ingests multiple PDF, DOCX, and TXT documents per session.
- **Table & OCR Extraction**: Preserves structured tables in Markdown format and handles scanned content via OCR.
- **Strict Grounding & Guardrails**: Enforces structured Pydantic output validation to ensure answers are strictly derived from retrieved evidence, explicitly refusing to answer when context is insufficient.
- **Context-Aware Follow-Up Resolution**: Automatically resolves conversational references (pronouns like "it", "they", "this") into standalone search queries.
- **Bounded 3-Turn Conversational Memory**: Retains strictly the latest 3 conversational exchanges (6 messages) per session to maintain relevant context while preventing contextual drift and token bloat.
- **Exact Source Attribution**: Every response provides citations detailing the source document filename, page number, and chunk index.
- **Multi-Tenant Session Isolation**: Documents and conversational histories are tagged with unique session IDs, preventing cross-session data leakage.

---

## 2. Features

- **Document Parser**:
  - Primary extraction with `PyMuPDF4LLM` for high-fidelity Markdown and page markers (`<!-- Page N -->`).
  - Embedded image and scanned page text extraction via `RapidOCR`.
  - Robust table parsing rendering GitHub-flavored Markdown tables.
  - Secondary fallback to IBM's `Docling` converter on malformed files.
  - Multi-encoding plain text decoder (`utf-8-sig`, `utf-8`, `cp1252`, `latin-1`).
  - Word document parsing via `python-docx` preserving headings and tables.
- **Vector Storage & Retrieval**:
  - `RecursiveCharacterTextSplitter` with `chunk_size=900` and `chunk_overlap=150` to maintain cross-boundary context.
  - ChromaDB persistent storage with metadata filtering by `session_id` and optional `source_name`.
  - True cosine semantic similarity mapping from Chroma squared Euclidean distance: $\text{similarity} = 1.0 - \text{distance} / 2.0$.
  - Tunable relevance threshold (`RETRIEVAL_MIN_RELEVANCE = 0.20`) filtering irrelevant chunks.
- **Conversational Memory**:
  - Bounded memory window utilizing LangChain's `trim_messages` (strictly retaining the latest 3 turns / 6 messages).
  - Standalone query re-writer using LangChain `ChatPromptTemplate` and `MessagesPlaceholder` to resolve conversational pronouns.
- **Dual Interface**:
  - RESTful FastAPI backend with OpenAPI Swagger documentation at `/docs`.
  - Interactive Streamlit frontend providing file upload, real-time chat, expandable source citations, and active memory inspection.

---

## 3. Technology Stack & Tool Selection Rationale

| Layer / Role | Technology Used | Reason & Justification for Selection |
| :--- | :--- | :--- |
| **Language & Runtime** | **Python 3.12** | Industry standard for AI/ML engineering, offering mature library support, native performance enhancements, and modern type annotations. |
| **Backend API** | **FastAPI** | High-performance asynchronous ASGI framework with automated OpenAPI/Swagger documentation, strict request/response data validation via Pydantic v2, and lightweight footprint. |
| **Frontend UI** | **Streamlit** | Rapidly builds functional, responsive user interfaces with native chat widgets (`st.chat_message`, `st.chat_input`), seamless session state management, and zero frontend boilerplate. |
| **LLM Provider** | **Groq API** (`openai/gpt-oss-120b`) | Provides ultra-low inference latency (crucial for responsive multi-turn conversational RAG), strong reasoning fidelity, and robust support for LangChain structured JSON outputs. |
| **Vector Database** | **ChromaDB** | Lightweight, open-source embedded vector database that runs in-process with zero external server infrastructure overhead, persistent on-disk storage, and native metadata filtering for session isolation. |
| **Embedding Model** | **HuggingFace** (`all-MiniLM-L6-v2`) | Compact, high-speed 384-dimensional dense semantic embedding model running locally on CPU with zero API rate limits or inference costs. Generates unit-normalized vectors ideal for cosine similarity. |
| **RAG Orchestration** | **LangChain Core** | Provides production-ready abstractions (`RecursiveCharacterTextSplitter`, `ChatPromptTemplate`, `MessagesPlaceholder`, `trim_messages`, `with_structured_output`) preventing vendor lock-in. |
| **Document Parsing & OCR** | **PyMuPDF4LLM + RapidOCR + Docling** | `PyMuPDF4LLM` delivers structural Markdown formatting with native table extraction; `RapidOCR` extracts text from scanned pages; `Docling` provides resilient fallback for complex documents. |

---

## 4. How to Run

### Prerequisites
- **Python 3.12+** (or `uv` package manager)
- A **Groq API Key** (obtainable from [Groq Console](https://console.groq.com))

### Step 1: Clone the Repository
```bash
git clone https://github.com/utsav210/ai-document-assistant.git
cd ai-document-assistant
```

### Step 2: Set Up Virtual Environment
```bash
# Using standard Python venv:
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root (or copy from `.env.example`):
```env
GROQ_API_KEY="your-groq-api-key-here"
GROQ_MODEL="openai/gpt-oss-120b"
LOG_LEVEL="INFO"
CHROMA_PERSIST_DIRECTORY="./chroma_db"
CHROMA_COLLECTION_NAME="ai_document_assistant"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE=900
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=4
RETRIEVAL_MIN_RELEVANCE=0.20
MAX_UPLOAD_MB=20
```

### Step 5: Start the FastAPI Backend
```bash
uvicorn backend.main:app --reload --port 8000
```
- API will be accessible at: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

### Step 6: Start the Streamlit User Interface
In a new terminal (with `.venv` activated):
```bash
streamlit run frontend/app.py
```
- Web Application will be accessible at: `http://localhost:8501`

### Step 7: Run Automated Tests
```bash
pytest
```
To run tests with verbose logging and code linting:
```bash
ruff check .
pytest -v
```

---

## 5. Implementation Overview & Design Decisions

### Architecture Flow
```text
┌─────────────────┐       ┌─────────────────┐
│  Streamlit UI   │ ◄───► │   FastAPI API   │
│ (frontend/app)  │ HTTP  │ (backend/main)  │
└─────────────────┘       └────────┬────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
    ┌──────────────────────┐               ┌──────────────────────┐
    │  Document Processing │               │     RAG Service      │
    │  - document_parser   │               │  - rag_service       │
    │  - chunking (900/150)│               │  - vector_store      │
    └──────────┬───────────┘               └──────────┬───────────┘
               │                                       │
               ▼                                       ▼
    ┌──────────────────────┐               ┌──────────────────────┐
    │ ChromaDB VectorStore │ ◄─────────────┤   Groq LLM Service   │
    │ (Cosine Similarity)  │               │ (Structured Guardrail│
    └──────────────────────┘               └──────────────────────┘
```

### Key Design Decisions

1. **Separation of Responsibilities**:
   - `backend/document_parser.py`: Ingestion, OCR, and table extraction into normalized Markdown.
   - `backend/chunking.py`: Chunk generation with start index and metadata tags.
   - `backend/vector_store.py`: Embeddings, ChromaDB indexing, and cosine scoring.
   - `backend/rag_service.py`: Memory management, pronoun query rewriting, and grounded response synthesis.
   - `backend/main.py`: REST routes, upload staging, and error handling.
   - `frontend/app.py`: Clean Streamlit UI communicating strictly over HTTP.

2. **Chroma Distance to Cosine Semantic Similarity**:
   ChromaDB returns squared Euclidean distance ($d^2$). For unit-normalized vectors:
   $$d^2 = \|u - v\|^2 = 2 - 2\cos(\theta) \implies \cos(\theta) = 1.0 - \frac{d^2}{2.0}$$
   A custom `relevance_score_fn` was implemented so Chroma scores accurately reflect true cosine similarity in $[0, 1]$, preventing premature discarding of valid document evidence.

3. **Chunk Overlap Context Continuity**:
   Setting `chunk_size=900` and `chunk_overlap=150` ensures that sentence structures and clauses spanning chunk boundaries maintain semantic continuity, preventing context fragmentation during vector search.

4. **Conversational Follow-Up Resolution**:
   Follow-up questions are resolved using a dedicated query re-writer (`REWRITE_PROMPT` + `MessagesPlaceholder`). For example:
   - *Previous Answer*: "The document was created for testing the data and the API."
   - *User Input*: "Will it return output precisely?"
   - *Rewritten Query*: "Does the test document for data and API return precise output?"
   - This ensures vector search retrieves evidence for the real intent rather than ambiguous pronouns.

5. **Anti-Hallucination Guardrail (`GroundedAnswer`)**:
   Instead of relying on unstructured text output, the LLM is prompted via LangChain's `with_structured_output` using a Pydantic schema:
   ```python
   class GroundedAnswer(BaseModel):
       has_sufficient_context: bool
       answer: str
   ```
   If evidence is absent from the indexed chunks, `has_sufficient_context` is marked `False`, triggering a standardized refusal without speculation.

---

## 6. Directory Structure

```text
ai-document-assistant/
├── backend/
│   ├── __init__.py
│   ├── config.py              # Environment configuration & Settings dataclass
│   ├── logging_config.py      # Structured application logging
│   ├── document_parser.py     # PDF, DOCX, TXT parser with OCR & table support
│   ├── chunking.py            # Recursive character chunking & overlap
│   ├── vector_store.py        # ChromaDB connector & cosine similarity function
│   ├── rag_service.py         # 3-turn memory, query re-writer, grounded QA
│   └── main.py                # FastAPI endpoints (/health, /documents, /chat)
├── frontend/
│   └── app.py                 # Streamlit conversational web interface
├── sample_data/
│   └── sample_test.txt        # Sample test document
├── tests/
│   ├── test_config.py
│   ├── test_document_parser.py
│   ├── test_chunking.py
│   ├── test_vector_store.py
│   ├── test_rag_service.py
│   └── test_main.py
├── .env.example               # Template environment configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive project documentation
```

---

## 7. Limitations & Future Improvements

While the application satisfies all operational and functional requirements, the following enhancements could be incorporated in future iterations:
- **Hybrid Search (Dense + Sparse BM25)**: Combining Chroma dense semantic search with BM25 lexical search and a cross-encoder re-ranker to maximize retrieval precision on rare acronyms or numerical IDs.
- **Persistent Session Storage**: Currently, conversational turn memory is maintained in an in-memory thread-safe LRU structure. Transitioning to Redis or PostgreSQL would support horizontal scaling across multiple application instances.
- **Real-Time Streaming Responses**: Implementing server-sent events (`StreamingResponse` in FastAPI and `st.write_stream` in Streamlit) for token-by-token generation.
- **Multimodal VLM Reasoning**: Direct image-to-text visual reasoning using vision language models (e.g., Llama-3.2-Vision or Gemini Flash) for charts and architectural diagrams.

---

## Note

This application was developed with the assistance of LangChain official documentation, ChatGPT, and Gemini 3.8 Flash in Google Antigravity to support streamlined, fast-paced development while maintaining rigorous engineering standards, error handling, security practices, and comprehensive automated test suites.
