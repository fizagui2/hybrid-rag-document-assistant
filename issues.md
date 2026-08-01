# Issues and Development Notes

This document records notable problems, limitations, observations, and lessons learned throughout the development of the RAG pipeline project. This focuses on issues that really influenced development decisions or showed useful information about AI-assisted software engineering.

---

## AI-Assisted Development Observations

### Excessive automated testing and token consumption

#### Description

Claude frequently attempted to perform extensive testing after relatively small changes. While this approach improved confidence in the implementation, it consumed Groq API tokens at a much faster rate than expected.

On several occasions, only a handful of prompts were enough to consume most of the available tokens because every modification made additional API calls.

#### Resolution

I changed it so that Claude performed only minimal validation after each step. Larger tests were postponed until entire milestones had been completed.

#### Lesson learned

AI systems naturally optimize for correctness and completeness rather than efficiency. Explicit limitations regarding testing frequency, token usage, and cost are often necessary.

---

### Rapid context consumption

#### Description

As the project became larger, the conversation context filled rapidly. Architectural discussions, debugging sessions, prompts, implementation details, and testing results grew quickly.

Toward the end of the project, careful context management became essential.

#### Resolution

Important architectural decisions and milestones were summarized and documented externally rather than relying exclusively on the conversation history.

#### Lesson learned

Long running projects benefit from external documentation because context windows eventually become a limiting factor.

---

## Environment and Setup Issues

### Missing dependencies

At the beginning of the project, Docker and several required libraries were not installed. The environment had to be configured before development could begin.

---

### Windows console encoding problems

A command-line test failed with a `UnicodeEncodeError` because Windows was using the `cp1252` code page while the language model produced Unicode characters.

The problem was resolved by forcing UTF-8 output.

---

### Windows file-permission problems

Temporary directories created during evaluation could not be removed automatically because Chroma maintained active file handles.

Cleanup procedures were modified so that failures would not destroy previously generated results.

---

## Ingestion and Chunking Issues

### File name collisions

Processed documents were initially named using only their file names without extensions.

For example:

- `sample.txt`
- `sample.pdf`
- `sample.md`

all produced `sample.json`, causing files to overwrite one another.

The naming strategy was changed to preserve the full filename.

---

### Semantic chunking edge cases

Single-sentence sections bypassed chunk-size limitations and produced oversized chunks.

---

### Deduplication conflicts

The duplicate detection mechanism incorrectly identified existing documents as duplicates during reindexing.

The solution excluded the current document identifier from similarity comparisons.

---

### Sentence-splitting errors

Citations appearing immediately after periods were interpreted as separate sentences.

The sentence parser was updated to prevent these splits from occurring.

---

## Evaluation and Rate-Limit Issues

### Groq rate limits

The free Groq tier imposed both daily token limits and tokens-per-minute restrictions.

Several evaluation runs were interrupted because these limits were reached.

---

### Misinterpreting quota availability

Successful responses from small test requests were initially interpreted as confirmation that rate limits had been reset.

This assumption proved incorrect.

A single successful request indicates only that some resources remain available, It does not guarantee enough capacity for large scale testing.

---

### Retrying unrecoverable failures

The original retry mechanism treated daily and per-minute limits identically.

This resulted in unnecessary retries and wasted time.

Separate handling procedures were implemented for both situations.

---

## API and Dashboard Issues

### Incorrect form handling

FastAPI silently ignored uploaded form parameters because they were not explicitly declared as form fields.

This issue caused the default configuration to be used unintentionally.

---

### Test contamination

Early tests wrote files directly into production directories.

Dependency injection was introduced to isolate the testing environment.

---

### Browser automation problems

Several browser based tests failed because the automation scripts searched for incorrect UI elements.

Manual verification and server logs were used to identify the problem.

---

## Known Limitations

- Groq rate limits remain a constraint for large-scale evaluations.
- Confidence thresholds may require further tuning.
- Markdown tables require more sophisticated citation parsing.
- Small evaluation datasets can conceal problems that appear only at larger scales.
- Long-running AI conversations require deliberate context management.