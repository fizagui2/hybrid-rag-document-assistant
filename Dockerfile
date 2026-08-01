FROM python:3.13-slim

WORKDIR /app

# Some dependencies (e.g. chromadb's hnswlib) may need to compile from
# source if no prebuilt wheel matches this exact image; build-essential
# covers that so `pip install` below doesn't fail for a non-obvious reason.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY dashboard/ dashboard/
COPY eval/ eval/
COPY scripts/ scripts/

# Embedding/reranker model weights are NOT baked into the image — they're
# lazily downloaded from Hugging Face on first real use (same as running
# locally), so the container needs internet access on first query, not at
# build time.

# Overridden per-service by docker-compose.yml's `command:`.
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
