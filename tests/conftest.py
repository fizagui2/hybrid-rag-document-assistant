"""Shared test fixtures/helpers."""
import os

import pytest

# Importing src.generation.llm triggers its module-level load_dotenv() call,
# so GROQ_API_KEY (if present in .env) is in os.environ by the time this is
# evaluated. Reused by every test module that needs a real Groq API call —
# the rest of the suite runs with no credentials and no network access.
import src.generation.llm  # noqa: F401


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_groq: test makes a real Groq API call")


def requires_groq(func):
    """Two independent reasons a test might need to be excluded, handled
    separately: no GROQ_API_KEY at all (skipif — the test genuinely can't
    run) vs. deliberately avoiding real calls right now, e.g. while rate
    limited (the `requires_groq` marker, filterable with
    `pytest -m "not requires_groq"` regardless of whether a key is
    present). A plain skipif alone can't do the second — it only reacts to
    key presence, and there's no reason to unset a working key just to
    pause live calls temporarily.
    """
    func = pytest.mark.requires_groq(func)
    func = pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set — skipping live Groq API test"
    )(func)
    return func
