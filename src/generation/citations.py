"""Citation parsing + verification: after generation, check that each
citation the model made actually points to a chunk that supports the claim
it's attached to. "Does this passage support this claim" isn't something
string matching can answer, so each (claim, cited chunk) pair is sent to the
LLM again, this time as a judge rather than a generator.
"""
import re
from dataclasses import dataclass
from typing import Optional

from ..text import split_sentences
from .llm import chat

# Matches both plain ASCII brackets and full-width/CJK brackets — real model
# output has been observed using both for the same citation intent (e.g.
# "[1]" vs "【1】") — plus multi-citation and comma-separated forms:
# [1], [1][2], [1, 2], 【1】, 【1】【2】
_CITATION_RE = re.compile(r"[\[【]\s*(\d+(?:\s*,\s*\d+)*)\s*[\]】]")

JUDGE_SYSTEM_PROMPT = """You are verifying whether a claim is actually supported by a source passage.

Respond with exactly two lines:
Line 1: YES or NO
Line 2: a one-sentence reason"""


@dataclass
class ClaimCitation:
    claim: str
    citation_number: int
    chunk_text: str


@dataclass
class VerifiedCitation:
    claim: str
    citation_number: int
    supported: bool
    reasoning: str


def parse_citations(sentence: str) -> list[int]:
    """Extract every citation number referenced in a sentence, e.g.
    "Paris is the capital[1][2]." -> [1, 2], "no citation here" -> [].
    """
    numbers = []
    for match in _CITATION_RE.finditer(sentence):
        numbers.extend(int(n) for n in match.group(1).split(","))
    return numbers


def extract_claim_citations(answer_text: str, chunks: list) -> list[ClaimCitation]:
    """Split the answer into sentences, and pair each citation number found
    in a sentence with the chunk it actually refers to — using the same
    1-indexed numbering format_context() used to build the original prompt.
    Out-of-range citation numbers (the model citing "[7]" when there were
    only 5 chunks) are silently dropped rather than raising, since that's a
    model mistake to report on, not a reason to crash.
    """
    pairs = []
    for sentence in split_sentences(answer_text):
        for number in parse_citations(sentence):
            if 1 <= number <= len(chunks):
                pairs.append(ClaimCitation(claim=sentence, citation_number=number, chunk_text=chunks[number - 1].text))
    return pairs


def _parse_verdict(response: str) -> tuple[bool, str]:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    verdict = lines[0].upper() if lines else ""
    reasoning = lines[1] if len(lines) > 1 else ""
    return verdict.startswith("YES"), reasoning


def verify_citation(claim_citation: ClaimCitation) -> VerifiedCitation:
    """Ask the judge whether one specific cited passage actually supports
    one specific claim.
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Source passage:\n{claim_citation.chunk_text}\n\n"
                f"Claim: {claim_citation.claim}\n\n"
                "Does the source passage support this claim?"
            ),
        },
    ]
    supported, reasoning = _parse_verdict(chat(messages, temperature=0.0))
    return VerifiedCitation(
        claim=claim_citation.claim,
        citation_number=claim_citation.citation_number,
        supported=supported,
        reasoning=reasoning,
    )


def verify_citations(answer_text: str, chunks: list) -> list[VerifiedCitation]:
    """Verify every citation found in an answer. One LLM call per
    (claim, citation) pair — simple to reason about, though it means an
    answer with many cited sentences makes many calls; worth batching later
    if free-tier rate limits become a real constraint at larger scale.
    """
    return [verify_citation(cc) for cc in extract_claim_citations(answer_text, chunks)]
