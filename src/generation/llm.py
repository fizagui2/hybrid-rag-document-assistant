"""Thin wrapper around the Groq API — the free hosted LLM used for
generation (and, later, LLM-as-judge tasks like citation verification).
Requires a GROQ_API_KEY in .env (see .env.example); get a free one at
https://console.groq.com — no credit card required.

Model choice: llama-3.1-8b-instant and llama-3.3-70b-versatile (the models
originally planned for this project) are both scheduled for deprecation on
2026-08-16. Defaulting instead to openai/gpt-oss-20b, Groq's documented
migration target for the smaller model — fast, cheap, and not about to stop
working. openai/gpt-oss-120b is available for higher-quality answers at a
higher cost, swappable via the model parameter.
"""
import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.0  # deterministic, literal answers — not creative writing

# Groq's free tier has TWO separate limits: a large daily cap (TPD) and a
# much smaller per-minute cap (TPM, 8,000 tokens on this model). A bulk
# operation (e.g. the eval suite, dozens of calls per case) can blow past
# the per-minute cap well within a day's budget just by firing calls faster
# than that bucket refills. Since TPM buckets refill within seconds
# (Groq's own error message suggests ~5s), retrying after a short wait is
# almost always enough — unlike the daily cap, which retrying can't fix.
MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_RETRY_SECONDS = 15


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key "
            "from https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


def _is_daily_limit(exc: RateLimitError) -> bool:
    """RateLimitError doesn't expose a structured field for which limit was
    hit, but Groq's message text distinguishes "tokens per day (TPD)" from
    "tokens per minute (TPM)" — the distinction that actually matters, since
    only the per-minute one recovers on a short retry.
    """
    return "per day" in str(exc)


def chat(messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> str:
    client = get_client()
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            return response.choices[0].message.content
        except RateLimitError as exc:
            if _is_daily_limit(exc) or attempt == MAX_RATE_LIMIT_RETRIES - 1:
                raise
            time.sleep(RATE_LIMIT_RETRY_SECONDS)
