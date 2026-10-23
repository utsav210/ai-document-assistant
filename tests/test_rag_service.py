import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend import rag_service
from backend.rag_service import (
    UNKNOWN_ANSWER,
    ask_question,
    clear_conversation_history,
    get_conversation_history,
)


@pytest.fixture(autouse=True)
def clean_memory():
    """Clear memory before and after each test."""
    clear_conversation_history("test-session")
    clear_conversation_history("another-session")
    yield
    clear_conversation_history("test-session")
    clear_conversation_history("another-session")


def test_conversation_history_bounds_to_three_turns():
    session_id = "test-session"

    for i in range(5):
        rag_service._save_exchange(
            session_id,
            f"Question {i}",
            f"Answer {i}",
        )

    history = get_conversation_history(session_id)
    assert len(history) == 3
    assert history[0]["question"] == "Question 2"
    assert history[1]["question"] == "Question 3"
    assert history[2]["question"] == "Question 4"


def test_clear_conversation_history():
    session_id = "test-session"
    rag_service._save_exchange(session_id, "Q1", "A1")
    assert len(get_conversation_history(session_id)) == 1

    clear_conversation_history(session_id)
    assert get_conversation_history(session_id) == []


def test_get_session_messages_trimming():
    session_id = "test-session"
    for i in range(5):
        rag_service._save_exchange(session_id, f"User Q{i}", f"AI A{i}")

    messages = rag_service.get_session_messages(session_id)
    assert len(messages) == 6  # 3 turns = 6 messages
    assert messages[0].content == "User Q2"
    assert messages[1].content == "AI A2"
    assert messages[-1].content == "AI A4"


def test_format_history_with_langchain_messages():
    session_id = "test-session"
    rag_service._save_exchange(session_id, "Hello", "Hi there")
    messages = rag_service.get_session_messages(session_id)

    formatted = rag_service._format_history(messages)
    assert "Human: Hello" in formatted
    assert "AI: Hi there" in formatted


def test_ask_question_validation_errors():
    with pytest.raises(ValueError, match="session_id is required"):
        ask_question("", "Valid question")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        ask_question("test-session", "   ")

    with pytest.raises(ValueError, match="exceeds maximum length"):
        ask_question("test-session", "a" * 4001)

    with pytest.raises(ValueError, match="exceeds maximum length"):
        ask_question("x" * 129, "Valid question")


def test_ask_question_unknown_answer_when_no_chunks(monkeypatch):
    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, *args, **kwargs):
            return []

    monkeypatch.setattr(
        rag_service,
        "get_vector_store",
        lambda: FakeVectorStore(),
    )

    result = ask_question("test-session", "What is the secret policy?")
    assert result["answer"] == UNKNOWN_ANSWER
    assert result["sources"] == []

    history = get_conversation_history("test-session")
    assert len(history) == 1
    assert history[0]["answer"] == UNKNOWN_ANSWER


def test_ask_question_grounded_success(monkeypatch):
    test_doc = Document(
        page_content="The policy requires 2FA on all accounts.",
        metadata={
            "source_name": "security_guide.pdf",
            "page_number": 2,
            "file_type": "pdf",
            "extraction_method": "pymupdf4llm",
            "chunk_index": 0,
        },
    )

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, *args, **kwargs):
            return [(test_doc, 0.85)]

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(
                content="2FA is required on all accounts [security_guide.pdf, page 2]."
            )

    monkeypatch.setattr(
        rag_service,
        "get_vector_store",
        lambda: FakeVectorStore(),
    )
    monkeypatch.setattr(
        rag_service,
        "get_llm",
        lambda: FakeLLM(),
    )

    result = ask_question(
        "test-session",
        "What does the policy say about authentication?",
    )

    assert "2FA is required" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_name"] == "security_guide.pdf"
    assert result["sources"][0]["page_number"] == 2

    # Check that conversational memory was updated
    history = get_conversation_history("test-session")
    assert len(history) == 1
    assert history[0]["question"] == "What does the policy say about authentication?"
    assert "2FA is required" in history[0]["answer"]


def test_ask_question_guardrail_refusal_when_unsupported(monkeypatch):
    """Verify that LangChain guardrail returns UNKNOWN_ANSWER when context is insufficient."""
    test_doc = Document(
        page_content="Unrelated vacation guidelines.",
        metadata={"source_name": "hr_guide.pdf", "page_number": 1},
    )

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, *args, **kwargs):
            return [(test_doc, 0.80)]

    class FakeStructuredLLM:
        def invoke(self, messages):
            return rag_service.GroundedAnswer(
                has_sufficient_context=False,
                answer="No relevant policy found.",
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeStructuredLLM()

        def invoke(self, messages):
            return AIMessage(content="No relevant policy found.")

    monkeypatch.setattr(rag_service, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(rag_service, "get_llm", lambda: FakeLLM())

    result = ask_question("test-session", "What is the encryption key length?")

    assert result["answer"] == UNKNOWN_ANSWER
    assert result["sources"] == []
    history = get_conversation_history("test-session")
    assert history[0]["answer"] == UNKNOWN_ANSWER


def test_ask_question_structured_output_success(monkeypatch):
    """Verify that LangChain structured output guardrail returns grounded answer."""
    test_doc = Document(
        page_content="Encryption keys must be at least 256 bits.",
        metadata={"source_name": "crypto.pdf", "page_number": 3},
    )

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, *args, **kwargs):
            return [(test_doc, 0.90)]

    class FakeStructuredLLM:
        def invoke(self, messages):
            return rag_service.GroundedAnswer(
                has_sufficient_context=True,
                answer="Encryption keys must be at least 256 bits [crypto.pdf, page 3].",
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeStructuredLLM()

        def invoke(self, messages):
            return AIMessage(content="Encryption keys must be at least 256 bits.")

    monkeypatch.setattr(rag_service, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(rag_service, "get_llm", lambda: FakeLLM())

    result = ask_question("test-session", "What is the required key length?")

    assert "256 bits" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_name"] == "crypto.pdf"


def test_retrieve_context_filters(monkeypatch):
    captured_kwargs = []

    class FakeStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            captured_kwargs.append(kwargs)
            return []

    monkeypatch.setattr(rag_service, "get_vector_store", lambda: FakeStore())

    # Neither session_id nor source_name
    rag_service._retrieve_context("query")
    assert "filter" not in captured_kwargs[-1]

    # session_id only
    rag_service._retrieve_context("query", session_id="sess-1")
    assert captured_kwargs[-1]["filter"] == {"session_id": "sess-1"}

    # source_name only
    rag_service._retrieve_context("query", source_name="doc.pdf")
    assert captured_kwargs[-1]["filter"] == {"source_name": "doc.pdf"}

    # Both session_id and source_name
    rag_service._retrieve_context("query", session_id="sess-1", source_name="doc.pdf")
    assert captured_kwargs[-1]["filter"] == {
        "$and": [{"session_id": "sess-1"}, {"source_name": "doc.pdf"}]
    }


def test_rewrite_follow_up_empty_history():
    assert rag_service._rewrite_follow_up("What is 2FA?", []) == "What is 2FA?"


def test_rewrite_follow_up_with_llm(monkeypatch):
    class FakeRewriteLLM:
        def invoke(self, messages):
            return AIMessage(content="How do I configure two-factor authentication?")

    monkeypatch.setattr(rag_service, "get_llm", lambda: FakeRewriteLLM())

    history = [
        rag_service.HumanMessage(content="What is two-factor authentication?"),
        rag_service.AIMessage(content="It is an extra layer of security."),
    ]
    rewritten = rag_service._rewrite_follow_up("How do I configure it?", history)
    assert rewritten == "How do I configure two-factor authentication?"


def test_rewrite_follow_up_fallback_on_error(monkeypatch):
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("API failure")

    monkeypatch.setattr(rag_service, "get_llm", lambda: FailingLLM())

    history = [
        rag_service.HumanMessage(content="What is encryption?"),
        rag_service.AIMessage(content="Securing data at rest."),
    ]
    rewritten = rag_service._rewrite_follow_up("Is it required?", history)
    assert "Securing data at rest" in rewritten
    assert "Human: Is it required?" in rewritten


def test_extract_failed_generation_from_body():
    class DummyGroqError(Exception):
        def __init__(self):
            super().__init__("Tool use failed")
            self.body = {
                "error": {
                    "message": "Tool choice is required, but model did not call a tool",
                    "code": "tool_use_failed",
                    "failed_generation": "| Role | Name |\n|---|---|\n| Lead | Mira |",
                }
            }

    extracted = rag_service._extract_failed_generation(DummyGroqError())
    assert extracted == "| Role | Name |\n|---|---|\n| Lead | Mira |"


def test_extract_failed_generation_from_string():
    err_str = "Error code: 400 - {'error': {'code': 'tool_use_failed', 'failed_generation': 'Pilot team details'}}"
    exc = Exception(err_str)
    extracted = rag_service._extract_failed_generation(exc)
    assert extracted == "Pilot team details"


def test_ask_question_recovers_from_groq_tool_use_failed(monkeypatch):
    test_doc = Document(
        page_content="Team: Mira Shah is Project Owner.",
        metadata={"source_name": "report.docx", "page_number": 1},
    )

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, *args, **kwargs):
            return [(test_doc, 0.9)]

    class DummyGroqToolError(Exception):
        def __init__(self):
            super().__init__("Tool use failed")
            self.body = {
                "error": {
                    "message": "Tool choice is required, but model did not call a tool",
                    "code": "tool_use_failed",
                    "failed_generation": "Mira Shah is the Project Owner [report.docx].",
                }
            }

    class FailingStructuredLLM:
        def invoke(self, messages):
            raise DummyGroqToolError()

    class FakeLLM:
        def with_structured_output(self, schema):
            return FailingStructuredLLM()

        def invoke(self, messages):
            return AIMessage(content="Fallback answer")

    monkeypatch.setattr(rag_service, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(rag_service, "get_llm", lambda: FakeLLM())

    result = ask_question("test-session", "Who is the project owner?")
    assert "Mira Shah is the Project Owner" in result["answer"]
    assert len(result["sources"]) == 1


def test_ask_question_falls_back_to_standard_invoke_when_structured_fails(monkeypatch):
    test_doc = Document(
        page_content="Database port is 5432.",
        metadata={"source_name": "config.txt", "page_number": 1},
    )

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, *args, **kwargs):
            return [(test_doc, 0.9)]

    class FailingStructuredLLM:
        def invoke(self, messages):
            raise ValueError("Schema parse error without failed_generation")

    class FakeLLM:
        def with_structured_output(self, schema):
            return FailingStructuredLLM()

        def invoke(self, messages):
            return AIMessage(content="The database port is 5432 [config.txt].")

    monkeypatch.setattr(rag_service, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(rag_service, "get_llm", lambda: FakeLLM())

    result = ask_question("test-session", "What is the port?")
    assert "5432" in result["answer"]
    assert len(result["sources"]) == 1
