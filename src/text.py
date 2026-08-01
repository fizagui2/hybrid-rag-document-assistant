"""Small text utilities shared across pipeline stages that otherwise have
no business depending on each other (e.g. ingestion's chunker and
generation's citation verifier both need to split text into sentences).
"""
import re

# Split on whitespace that follows sentence-ending punctuation, UNLESS what
# follows is a citation bracket ([1], 【1】). Real generation output has been
# observed formatting citations as "claim. [1]" (period before the space),
# which without the negative lookahead splits into two "sentences" — the
# real claim, and a bare "[1]" that gets treated as its own nonsensical
# claim, silently detaching every citation from what it was supposed to
# support.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?![\[【])")


def split_sentences(text: str) -> list[str]:
    """Split on whitespace that follows sentence-ending punctuation. Simple
    and dependency-free; won't handle abbreviations like "Dr." perfectly,
    an acceptable simplification here.
    """
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
