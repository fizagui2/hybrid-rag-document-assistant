import pytest

from src.generation.llm import _is_daily_limit, chat, get_client
from tests.conftest import requires_groq


def test_get_client_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_client()


# --- _is_daily_limit: pure string-matching logic, no model needed ---
# Real error message text observed from Groq: "...on tokens per day (TPD):
# Limit 200000..." vs "...on tokens per minute (TPM): Limit 8000...".

def test_is_daily_limit_true_for_tpd_message():
    assert _is_daily_limit(Exception("...on tokens per day (TPD): Limit 200000, Used 199523..."))


def test_is_daily_limit_false_for_tpm_message():
    assert not _is_daily_limit(Exception("...on tokens per minute (TPM): Limit 8000, Used 7397..."))


# This test makes a real network call to Groq and needs a real API key, so
# it's skipped automatically unless GROQ_API_KEY is set in the environment
# (loaded from .env via python-dotenv, same as the app itself). Everything
# else in the suite runs with no credentials and no network access.
@requires_groq
def test_chat_returns_a_real_completion():
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer in one short sentence."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    response = chat(messages)

    assert isinstance(response, str)
    assert len(response) > 0
