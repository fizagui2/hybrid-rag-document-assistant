# Prompt Notes

This document contains noteworthy prompts, prompting strategies, and lessons learned throughout the development of the RAG pipeline project. It is not a complete transcript of every interaction with claude.

The goal of this document is to capture the approaches that proved effective, the prompts that required refinement, and the techniques that improved the overall development process.

---

## Establishing the development workflow

### Objective

Establish a consistent working relationship with the model before writing any code.

### Prompt

"Act as a senior software engineer and mentor. Work incrementally, explain important design decisions before implementing them, avoid unnecessary rewrites, and stop after completing each milestone."

### Outcome

This established the overall structure of the project and significantly reduced the amount of rework later in development.

### Lesson learned

Clearly defining expectations at the beginning of a project produces more consistent results than continuously changing instructions later.

---

## Adapting the project for a zero-cost environment

### Objective

Replace all paid services with free alternatives while preserving the original architecture.

### Prompt

"Review the original project requirements and replace every paid component with a free or local alternative while maintaining equivalent functionality."

### Outcome

Several substitutions were made:

- OpenAI embeddings were replaced with Sentence Transformers.
- GPT-4 and Claude models were replaced with Groq-hosted models.
- Local reranking models were used instead of paid APIs.

### Lesson learned

AI systems are particularly effective at proposing equivalent technologies when clear constraints are provided.

---

## Reducing unnecessary testing

### Objective

Prevent excessive API consumption during development.

### Prompt

"Perform only minimal verification after each step. Comprehensive testing will be performed after the milestone is complete."

### Outcome

Token usage decreased substantially.

### Lesson learned

AI systems often prioritize correctness over efficiency unless explicit limits are established.

---

## Preserving context throughout long conversations

### Objective

Reduce the effects of context-window limitations.

### Prompt

"Summarize important architectural decisions and preserve only information that will influence future work."

### Outcome

The available context was used more efficiently during later stages of the project.

### Lesson learned

Summarization and external documentation become increasingly important as projects grow in complexity.

---

## Verifying assumptions before implementation

### Objective

Ensure that decisions were based on evidence rather than assumptions.

### Prompt

"Verify the behavior experimentally before modifying the implementation."

### Outcome

This strategy prevented unnecessary changes and helped differentiate implementation errors from expected behavior.

Examples included:

- BM25 scoring behavior
- ChromaDB similarity calculations
- Groq rate-limiting behavior
- Citation formatting differences

### Lesson learned

The fastest solution is not always the correct solution. Verification often saves time in the long run.

---

## Combining automated testing with manual validation

### Objective

Avoid relying exclusively on unit tests.

### Prompt

"Run the automated tests, then perform manual end-to-end verification using realistic data."

### Outcome

Multiple issues were discovered that would otherwise have remained hidden.

Examples included:

- Sentence reconstruction problems
- Tokenization errors
- Confidence-scoring failures
- Citation-parsing issues

### Lesson learned

Passing tests does not necessarily imply correct behavior.

---

## Treating failures as research opportunities

### Objective

Document failures rather than immediately hiding or removing them.

### Prompt

"If a limitation or failure is discovered, document it and explain why it occurred."

### Outcome

This approach resulted in a more accurate representation of the strengths and weaknesses of the system.

### Lesson learned

Unexpected behavior can provide valuable insight into both the software and the underlying language model.

---

## Final observations

Several themes emerged repeatedly throughout the project:

- Explicit instructions consistently produced better results than vague requests.
- Smaller, focused prompts outperformed larger, more complicated prompts.
- Human oversight remained essential.
- Thorough testing was necessary despite the model's confidence.
- Documentation significantly reduced the impact of context limitations.
- Cost management became an important consideration when external APIs were involved.