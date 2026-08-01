# Project Timeline

**Project:** RAG Pipeline with Hybrid Search over Internal Documents

**Project duration:** July 15, 2026 – July 29, 2026

**Objective:** Build a complete retrieval-augmented generation (RAG) system capable of ingesting internal documents, retrieving relevant information through hybrid search, generating grounded answers, verifying citations, evaluating performance, and exposing the entire system through an API and web dashboard.

---

## Week 1 (July 15–July 18)

### Environment setup and project planning

Completed tasks:

- Established the project's overall architecture.
- Replaced all paid services with free alternatives.
- Created the repository structure.
- Configured the virtual environment.
- Created the initial project documentation.
- Installed the required libraries and dependencies.

Major decisions:

- Sentence Transformers replaced OpenAI embeddings.
- Groq models replaced GPT-4 and Claude APIs.
- ChromaDB and BM25 were selected for retrieval.
- FastAPI and Streamlit were selected for deployment.

---

### Phase 1 – Ingestion and chunking

Completed tasks:

- Implemented support for text, Markdown, HTML, and PDF files.
- Developed a unified document model.
- Added persistent storage capabilities.
- Implemented fixed-size chunking.
- Implemented structure-aware chunking.
- Implemented semantic chunking.
- Added support for overlap handling.
- Added embedding generation.
- Built the BM25 index.
- Implemented duplicate-detection capabilities.

Major findings:

- File-name collisions resulted in overwritten files.
- Sentence boundaries produced unexpected chunking behavior.
- Passing tests did not always indicate correct results.
- Manual verification proved essential.

---

### Phase 2 – Hybrid retrieval

Completed tasks:

- Implemented dense retrieval.
- Implemented sparse retrieval.
- Added Reciprocal Rank Fusion (RRF).
- Implemented a cross-encoder reranker.
- Performed end-to-end testing.

Major findings:

- BM25 scoring behaved unexpectedly on small datasets.
- Tokenization methods significantly influenced retrieval quality.
- Real-world testing revealed problems that unit tests failed to detect.

---

## Week 2 (July 19–July 22)

### Phase 3 – Generation and citation verification

Completed tasks:

- Implemented grounded answer generation.
- Added citation extraction and parsing.
- Implemented citation verification.
- Added confidence scoring.
- Added graceful refusal responses for low-confidence results.

Major findings:

- Language models produced multiple citation formats.
- Sentence splitting introduced unexpected errors.
- Confidence scores required substantial refinement.

---

### Phase 4 – Evaluation framework

Completed tasks:

- Created a synthetic evaluation corpus.
- Developed a dataset containing more than fifty question-and-answer pairs.
- Implemented correctness metrics.
- Implemented faithfulness metrics.
- Implemented citation metrics.
- Implemented retrieval metrics.

Major findings:

- Larger datasets revealed problems that smaller test datasets concealed.
- Multi-hop questions proved considerably more challenging than expected.
- Markdown tables exposed limitations within the citation-verification system.

---

## Week 3 (July 23–July 26)

### Chunking-strategy evaluation

Completed tasks:

- Compared fixed, structural, and semantic chunking strategies.
- Generated performance reports.
- Performed repeated evaluation runs.

Major findings:

- Groq rate limits became a significant obstacle.
- Daily and per-minute limits required different handling strategies.
- Careful resource management became essential.

---

### AI-assisted development observations

Important lessons learned:

- Extensive automated testing rapidly consumed API tokens.
- Long conversations consumed context much faster than expected.
- Incremental development produced the most reliable results.
- Smaller prompts consistently produced better outcomes.
- Manual validation remained essential throughout the project.

---

## Week 4 (July 27–July 29)

### Phase 5 – API and dashboard development

Completed tasks:

- Developed FastAPI endpoints.
- Created document-ingestion endpoints.
- Implemented question-and-answer endpoints.
- Developed a Streamlit dashboard.
- Performed browser-based testing.
- Implemented Docker support.
- Added automated deployment scripts.

Major findings:

- Dependency injection simplified testing.
- Browser automation revealed issues that unit tests missed.
- Containerization improved deployment consistency.

---

## Final project results

### Overall statistics

- Project duration: 15 days
- Total tests written: 118
- Technologies used: Python, ChromaDB, BM25, FastAPI, Streamlit, Docker, Groq API, Sentence Transformers
- Phases completed: 5