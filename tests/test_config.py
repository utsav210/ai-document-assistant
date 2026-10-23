import pytest

from backend.config import _read_int, get_settings


def test_get_settings_default() -> None:
    """Verify that settings load with valid defaults."""
    settings = get_settings()

    assert settings.app_name == "AI Document Assistant"
    assert settings.app_version == "0.1.0"
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.retrieval_top_k >= 1
    assert settings.retrieval_min_relevance == 0.20
    assert settings.max_upload_mb >= 1
    assert settings.chroma_persist_directory.is_absolute()


def test_get_settings_cached() -> None:
    """Verify that subsequent calls return the cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()

    assert s1 is s2


def test_read_int_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify integer parsing from environment variables."""
    monkeypatch.setenv("TEST_INT_VAR", "42")
    assert _read_int("TEST_INT_VAR", default=10) == 42


def test_read_int_fallback_default() -> None:
    """Verify fallback to default value when environment variable is unset."""
    assert _read_int("NON_EXISTENT_VAR_12345", default=99) == 99


def test_read_int_invalid_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that non-integer values raise a ValueError."""
    monkeypatch.setenv("TEST_INVALID_INT", "not_a_number")
    with pytest.raises(ValueError, match="must be an integer"):
        _read_int("TEST_INVALID_INT", default=10)


def test_read_int_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that values below minimum threshold raise a ValueError."""
    monkeypatch.setenv("TEST_MIN_INT", "0")
    with pytest.raises(ValueError, match="must be at least 1"):
        _read_int("TEST_MIN_INT", default=5, minimum=1)


def test_read_float_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify float parsing and range validation from environment variables."""
    from backend.config import _read_float

    monkeypatch.setenv("TEST_FLOAT_VAR", "0.75")
    assert _read_float("TEST_FLOAT_VAR", default=0.20) == 0.75


def test_read_float_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that float values outside range raise ValueError."""
    from backend.config import _read_float

    monkeypatch.setenv("TEST_FLOAT_VAR", "1.5")
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        _read_float("TEST_FLOAT_VAR", default=0.20, minimum=0.0, maximum=1.0)


def test_read_float_invalid_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that non-float values raise ValueError."""
    from backend.config import _read_float

    monkeypatch.setenv("TEST_FLOAT_VAR", "invalid_float")
    with pytest.raises(ValueError, match="must be a valid float"):
        _read_float("TEST_FLOAT_VAR", default=0.20)
