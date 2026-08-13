from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _extensions(name: str, default: str) -> tuple[str, ...]:
    values = []
    for item in os.getenv(name, default).split(","):
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        values.append(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return tuple(dict.fromkeys(values))


def _tokens(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip().casefold()
            for item in os.getenv(name, default).split(",")
            if item.strip()
        )
    )


def _source_link_roots() -> tuple[tuple[str, str], ...]:
    raw = os.getenv("ESG_SOURCE_LINK_ROOTS", "").strip()
    if not raw:
        return ()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("ESG_SOURCE_LINK_ROOTS must be a JSON object")
    return tuple((str(prefix).strip("/"), str(root)) for prefix, root in payload.items() if str(prefix).strip("/"))


def filename_tokens(path: str | Path) -> tuple[str, ...]:
    stem = Path(path).stem.casefold()
    return tuple(dict.fromkeys(re.findall(r"[a-zа-яё0-9]+", stem)))


def filename_matches(path: str | Path, required_tokens: tuple[str, ...]) -> bool:
    available = set(filename_tokens(path))
    return not required_tokens or any(token in available for token in required_tokens)


@dataclass(frozen=True, slots=True)
class Settings:
    input_dir: Path = Path(os.getenv("ESG_INPUT_DIR", ROOT / "input_documents"))
    archive_input_dir: Path | None = _optional_path("ESG_ARCHIVE_INPUT_DIR")
    markdown_dir: Path = Path(os.getenv("ESG_MARKDOWN_DIR", ROOT / "source_documents"))
    source_link_root: str = os.getenv("ESG_SOURCE_LINK_ROOT", "")
    source_link_roots: tuple[tuple[str, str], ...] = _source_link_roots()
    input_extensions: tuple[str, ...] = _extensions("ESG_INPUT_EXTENSIONS", ".docx")
    input_filename_tokens: tuple[str, ...] = _tokens("ESG_INPUT_FILENAME_TOKENS")
    search_filename_tokens: tuple[str, ...] = _tokens("ESG_SEARCH_FILENAME_TOKENS")
    runtime_dir: Path = Path(os.getenv("ESG_RUNTIME_DIR", ROOT / "runtime"))
    api_host: str = os.getenv("ESG_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("ESG_API_PORT", "8132"))
    model_id: str = os.getenv("ESG_MODEL_ID", "esg-repair-search")

    qdrant_url: str = os.getenv("ESG_QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_collection: str = os.getenv("ESG_QDRANT_COLLECTION", "esg_document_chunks")
    ollama_url: str = os.getenv("ESG_OLLAMA_URL", "http://127.0.0.1:11434")
    embedding_api: str = os.getenv("ESG_EMBEDDING_API", "ollama").strip().casefold()
    embedding_model: str = os.getenv("ESG_EMBEDDING_MODEL", "bge-m3")
    embedding_batch_size: int = int(os.getenv("ESG_EMBEDDING_BATCH_SIZE", "16"))

    llm_base_url: str = os.getenv("ESG_LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("ESG_LLM_API_KEY", "local")
    llm_model: str = os.getenv("ESG_LLM_MODEL", "")
    llm_timeout_seconds: int = int(os.getenv("ESG_LLM_TIMEOUT_SECONDS", "120"))
    llm_max_tokens: int = int(os.getenv("ESG_LLM_MAX_TOKENS", "4000"))
    query_extraction_enabled: bool = _bool("ESG_QUERY_EXTRACTION_ENABLED", False)
    query_llm_base_url: str = os.getenv("ESG_QUERY_LLM_BASE_URL", "")
    query_llm_model: str = os.getenv("ESG_QUERY_LLM_MODEL", "esg-query-extractor")
    query_llm_timeout_seconds: int = int(os.getenv("ESG_QUERY_LLM_TIMEOUT_SECONDS", "15"))
    chunk_extraction_enabled: bool = _bool("ESG_CHUNK_EXTRACTION_ENABLED", False)
    chunk_extraction_max_chars: int = int(os.getenv("ESG_CHUNK_EXTRACTION_MAX_CHARS", "5000"))
    chunk_extraction_workers: int = int(os.getenv("ESG_CHUNK_EXTRACTION_WORKERS", "2"))

    reranker_url: str = os.getenv("ESG_RERANKER_URL", "")
    reranker_timeout_seconds: int = int(os.getenv("ESG_RERANKER_TIMEOUT_SECONDS", "60"))
    reranker_batch_size: int = int(os.getenv("ESG_RERANKER_BATCH_SIZE", "8"))
    retrieval_top_k: int = int(os.getenv("ESG_RETRIEVAL_TOP_K", "40"))
    final_top_k: int = int(os.getenv("ESG_FINAL_TOP_K", "6"))
    show_retrieved_chunks: bool = _bool("ESG_SHOW_RETRIEVED_CHUNKS", False)
    rrf_k: int = int(os.getenv("ESG_RRF_K", "60"))

    mineru_backend: str = os.getenv("MINERU_BACKEND", "pipeline")
    mineru_method: str = os.getenv("MINERU_METHOD", "auto")
    mineru_lang: str = os.getenv("MINERU_LANG", "cyrillic")
    mineru_timeout_seconds: int = int(os.getenv("MINERU_TIMEOUT_SECONDS", "1800"))
    direct_docx_enabled: bool = _bool("ESG_DIRECT_DOCX_ENABLED", True)
    debug: bool = _bool("ESG_DEBUG", False)

    @property
    def db_path(self) -> Path:
        return self.runtime_dir / "esg.sqlite3"

    @property
    def converted_dir(self) -> Path:
        return self.runtime_dir / "converted"

    @property
    def jobs_dir(self) -> Path:
        return self.runtime_dir / "jobs"

    def ensure_dirs(self) -> None:
        inputs = [self.input_dir, *([self.archive_input_dir] if self.archive_input_dir else [])]
        for input_dir in inputs:
            if not input_dir.is_dir():
                raise RuntimeError(f"ESG input directory does not exist or is not a directory: {input_dir}")
            input_path = input_dir.resolve()
            for output in (self.markdown_dir, self.runtime_dir):
                output_path = output.resolve()
                if input_path == output_path or input_path in output_path.parents or output_path in input_path.parents:
                    raise RuntimeError(f"ESG input directory must not overlap writable output directory: {output}")
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
