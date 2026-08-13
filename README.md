# ESG Repair Search

Standalone RAG model for the existing OpenWebUI that searches Extended Service Goal reports for repairs described in a requested structural zone. It reuses the running Intranet RAG infrastructure but owns its API, runtime data, and Qdrant collection.

## Architecture

```text
network source folder (read-only)
-> MinerU Markdown
-> source_documents (Markdown only)
-> heading-aware sections
-> all heading-aware sections of canonical TR documents
-> SQLite FTS + Qdrant dense index
-> deterministic interval matching
-> optional external reranker
-> evidence-constrained answer
-> OpenAI-compatible API / OpenWebUI
```

All sections of every canonical TR document are indexed as domain-neutral raw text. There is no section boost or chapter filter. Chapter 5 is retained only as heading metadata for future rules. The small local Qwen structures only the user's query; document sections are not classified or parsed by an LLM during ingestion.

## Setup

1. Create `.env`, set `ESG_INPUT_DIR` to the primary absolute source path and optionally `ESG_ARCHIVE_INPUT_DIR` to a second source. Both are mounted read-only. `ESG_SOURCE_LINK_ROOTS` maps the archive prefix to its Windows/UNC root. The current corpus uses `.docx` only. `ESG_INPUT_FILENAME_TOKENS=TR,SR` limits conversion to reports whose basename contains either separate token; `ESG_SEARCH_FILENAME_TOKENS=TR,SR` applies the same alternatives inside both retrieval indexes. Use `.env.example` as a reference.
2. Ensure the host running Docker can read that mounted network path.
3. Start the ESG API container:

```bash
docker compose up -d --build
```

The API is available at `http://localhost:8132`. It joins `intranet-rag_default` and reuses `qdrant`, `ollama`, `reranker-gpu`, and the configured answer LLM. It does not start duplicate infrastructure.

The primary input folder is mounted at `/app/input_documents` and the archive at `/app/input_documents/__s7a`, both read-only. MinerU writes only final Markdown files to the local `source_documents/` directory, preserving the relative source tree. Intermediate MinerU files remain under ignored `runtime/`. Incremental sync tracks original files by SHA-256 and removes the corresponding Markdown and index records when an original disappears.

Only extensions listed in `ESG_INPUT_EXTENSIONS` are discovered. Comma-separated values are supported; Word lock files named `~$*.docx` are always skipped.

DOCX files are extracted directly from their OOXML structure by default (`ESG_DIRECT_DOCX_ENABLED=true`). Word heading styles, paragraphs, lists, and tables are preserved as Markdown without OCR, layout models, or a temporary MinerU API. MinerU with `MINERU_METHOD=txt` remains a fallback only when a DOCX cannot be parsed or contains no extractable text. A fresh derived Markdown file is reused when recovering an interrupted index run; normal incremental decisions still use source metadata and SHA-256 where needed.

Filename token filters are generic metadata filters, not document-content heuristics. Comma-separated configured tokens are alternatives (`OR`), are separated in filenames by punctuation, underscores, or spaces, and are matched case-insensitively. Empty `ESG_INPUT_FILENAME_TOKENS` indexes every configured extension. Empty `ESG_SEARCH_FILENAME_TOKENS` searches all indexed document profiles. The current model indexes and retrieves `TR` or `SR` records before reranking and LLM generation.

Before conversion, source files are grouped by case-insensitive basename. Files with the same name in different folders are treated as copies of one document. Only the copy with the newest modification time is converted and indexed; all paths are retained as aliases. A tie is resolved by the lexicographically greatest relative path. The API shows the canonical path and does not duplicate search results.

There is no document-side LLM extraction during ingestion. Retrieval searches all TR chunks and reranks them. The small local Qwen model parses only the user query; retrieved chunks remain raw and the final answer model checks them against the structured query. Query-time chunk extraction is retained as an experimental, disabled-by-default option controlled by `ESG_CHUNK_EXTRACTION_ENABLED`.

When experimental chunk extraction is enabled, the matcher evaluates every extracted zone independently and never combines coordinates from different repairs. With the default disabled configuration, candidates remain `UNKNOWN` and the final model reviews their original text. The query schema distinguishes frames, stringers, ribs, flaps, slats, spoilers, NLG, and MLG.

The service refuses to start ingestion if the input path overlaps `source_documents` or `runtime`. It never creates, edits, moves, or deletes files under `ESG_INPUT_DIR`.

In the existing OpenWebUI admin settings, add an OpenAI-compatible connection with base URL `http://esg-api:8132/v1` and any non-empty API key. Both containers are on `intranet-rag_default`; the model appears as `esg-repair-search`.

Refresh the source catalog manually and start incremental ingestion:

```bash
curl -X POST http://localhost:8132/api/sources/refresh \
  -H 'Content-Type: application/json' \
  -d '{}'
curl http://localhost:8132/api/ingest/status
```

This is the only operation that walks the configured network directories. It atomically stores the
filtered TR/SR DOCX list in SQLite and then indexes new or changed canonical documents. There is no
scheduled refresh; call this endpoint when files may have been added, changed, or deleted.

Start ingestion from the saved catalog without walking the network directory trees:

```bash
curl -X POST http://localhost:8132/api/ingest/run \
  -H 'Content-Type: application/json' \
  -d '{}'
curl http://localhost:8132/api/ingest/status
```

Force reconversion and reindexing:

```bash
curl -X POST http://localhost:8132/api/ingest/run \
  -H 'Content-Type: application/json' \
  -d '{"force":true}'
```

CLI equivalents:

```bash
python3 -m esg.api ingest
python3 -m esg.api ingest --refresh-sources
python3 -m esg.api ingest --force
python3 -m esg.api ingest-status
python3 -m esg.api sources-status
python3 -m esg.api health
```

## API

- `GET /api/health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /api/ingest/run`
- `GET /api/ingest/status`
- `POST /api/sources/refresh`
- `GET /api/sources/status`

Chat responses include a concise `reasoning_content` activity log. In streaming mode it is emitted as a reasoning delta before the final answer, so OpenWebUI can display query extraction, retrieval, zone filtering, reranking, and evidence-based answer generation progress.

Example:

```bash
curl http://localhost:8132/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"esg-repair-search",
    "messages":[{"role":"user","content":"Есть ли ремонт трещины в обшивке между стрингерами 24 и 28 у шпангоута 34?"}],
    "stream":false
  }'
```

A positive response means that a repair is described in an indexed engineering report in a compatible zone. It does not independently prove physical completion outside that report. A negative response is always scoped to the indexed corpus.

Interval endpoints are inclusive. A query for stringers 4-8 intersects a document zone 6-10 at 6-8. Candidates with explicit conflicts are removed, candidates with missing structural data remain `UNKNOWN`, and exact structural candidates are added independently of the text retrieval limit.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
