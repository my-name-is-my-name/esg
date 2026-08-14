from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import uuid
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import parse_qs, urlsplit

from esg.clients import EmbeddingClient, ExternalReranker, OpenAIClient, SemanticExtractor
from esg.config import Settings
from esg.ingest import IngestionService
from esg.retrieval import NEGATIVE_ANSWER, RetrievalService, normalized_evidence
from esg.storage import SQLiteStore
from esg.vector_index import VectorIndex


class Application:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.store = SQLiteStore(self.settings.db_path)
        self.store.interrupt_running_jobs()
        self.llm = OpenAIClient(self.settings)
        query_llm = OpenAIClient(
            replace(
                self.settings,
                llm_base_url=self.settings.query_llm_base_url or self.settings.llm_base_url,
                llm_model=self.settings.query_llm_model,
                llm_timeout_seconds=self.settings.query_llm_timeout_seconds,
                llm_max_tokens=1200,
            )
        )
        extractor = SemanticExtractor(query_llm)
        embeddings = EmbeddingClient(self.settings)
        vector_index = VectorIndex(self.settings)
        self.vector_index = vector_index
        self.ingestion = IngestionService(self.settings, self.store, embeddings, vector_index)
        self.retrieval = RetrievalService(
            self.settings,
            self.store,
            extractor,
            embeddings,
            vector_index,
            self.llm,
            ExternalReranker(self.settings),
        )

    def health(self) -> dict[str, object]:
        vector = self.vector_index.health()
        return {
            "ok": bool(vector.get("ok")),
            "model": self.settings.model_id,
            "corpus": self.store.counts(),
            "qdrant": vector,
            "llm": {"configured": self.llm.enabled, "base_url": self.settings.llm_base_url},
            "reranker": {"configured": bool(self.settings.reranker_url)},
            "last_ingestion": self.store.latest_job(),
            "sources": self.ingestion.source_status(),
        }


def handler_for(app: Application) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ESGRepairSearch/0.1"

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors_headers()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/api/health":
                self._json(HTTPStatus.OK, app.health())
                return
            if path == "/api/ingest/status":
                result = app.store.latest_job() or {"status": "not_started"}
                result["sources"] = app.ingestion.source_status()
                self._json(HTTPStatus.OK, result)
                return
            if path == "/api/sources/status":
                self._json(HTTPStatus.OK, app.ingestion.source_status())
                return
            if path == "/api/documents/summary":
                self._json(HTTPStatus.OK, app.store.document_registry_summary())
                return
            if path == "/api/documents":
                result = app.store.document_registry(
                    status=str((query.get("status") or [""])[0]),
                    limit=_query_int(query, "limit", 100),
                    offset=_query_int(query, "offset", 0),
                )
                self._json(HTTPStatus.OK, result)
                return
            if path == "/api/documents.csv":
                self._csv(app.store.registry_items())
                return
            if path == "/v1/models":
                self._json(
                    HTTPStatus.OK,
                    {"object": "list", "data": [{"id": app.settings.model_id, "object": "model", "owned_by": "esg"}]},
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not_found"}})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                payload = self._read_json()
                if path == "/api/ingest/run":
                    result = app.ingestion.start(
                        force=bool(payload.get("force", False)),
                        refresh_sources=bool(payload.get("refresh_sources", False)),
                    )
                    self._json(HTTPStatus.ACCEPTED if result.get("accepted") else HTTPStatus.CONFLICT, result)
                    return
                if path == "/api/sources/refresh":
                    result = app.ingestion.start(
                        force=bool(payload.get("force", False)),
                        refresh_sources=True,
                    )
                    self._json(HTTPStatus.ACCEPTED if result.get("accepted") else HTTPStatus.CONFLICT, result)
                    return
                if path == "/api/ingest/cancel":
                    result = app.ingestion.cancel()
                    self._json(HTTPStatus.ACCEPTED if result.get("accepted") else HTTPStatus.CONFLICT, result)
                    return
                if path == "/v1/chat/completions":
                    self._chat(payload)
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not_found"}})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": repr(exc)}})

        def _chat(self, payload: dict[str, Any]) -> None:
            model = str(payload.get("model") or app.settings.model_id)
            if model != app.settings.model_id:
                raise ValueError(f"Unknown model: {model}")
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be an array")
            question = next(
                (str(item.get("content") or "") for item in reversed(messages) if isinstance(item, dict) and item.get("role") == "user"),
                "",
            ).strip()
            if not question:
                raise ValueError("A non-empty user message is required")
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            if bool(payload.get("stream", False)):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self._cors_headers()
                self.end_headers()
                def emit_reasoning(message: str, newline: bool = True) -> None:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "reasoning_content": message + ("\n" if newline else ""),
                            },
                            "finish_reason": None,
                        }],
                    }
                    self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()

                result = app.retrieval.chat(
                    question,
                    progress=emit_reasoning,
                    reasoning_progress=lambda delta: emit_reasoning(delta, newline=False),
                )
                content = append_sources(
                    str(result["answer"]),
                    list(result.get("sources") or []),
                    app.settings.source_link_root,
                    app.settings.source_link_roots,
                    app.settings.show_retrieved_chunks,
                )
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
                }
                self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                end = dict(chunk)
                end["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                self.wfile.write(("data: " + json.dumps(end, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode())
                return
            result = app.retrieval.chat(question)
            content = append_sources(
                str(result["answer"]),
                list(result.get("sources") or []),
                app.settings.source_link_root,
                app.settings.source_link_roots,
                app.settings.show_retrieved_chunks,
            )
            reasoning = "\n".join(str(item) for item in result.get("reasoning", []) if str(item).strip())
            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "sources": result.get("sources", []),
                "warnings": result.get("warnings", []),
            }
            self._json(HTTPStatus.OK, response)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _csv(self, rows: list[dict[str, object]]) -> None:
            fields = [
                "source_path", "status", "reason", "is_canonical", "canonical_source_path",
                "size", "mtime", "markdown_path", "repair_section_count", "repair_count", "updated_at",
            ]
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            body = ("\ufeff" + output.getvalue()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="esg-documents.csv"')
            self.send_header("Content-Length", str(len(body)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return Handler


def _query_int(query: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int((query.get(name) or [str(default)])[0])
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def append_sources(
    answer: str,
    sources: list[dict[str, object]],
    source_link_root: str = "",
    source_link_roots: tuple[tuple[str, str], ...] = (),
    show_on_negative: bool = False,
) -> str:
    if not sources or (answer.strip() == NEGATIVE_ANSWER and not show_on_negative):
        return answer
    lines = [
        answer.rstrip(),
        "",
        "Источники:",
        "| Путь к исходному DOCX | Раздел | Похожий фрагмент |",
        "|---|---|---|",
    ]
    seen: set[str] = set()
    for source in sources:
        source_path = str(source.get("source_path") or "")
        section_heading = str(source.get("section_heading") or "")
        evidence = normalized_evidence(str(source.get("evidence_text") or ""))
        key = evidence or f"{source_path}|{section_heading}"
        if key in seen:
            continue
        seen.add(key)
        document = source_windows_path(source_path, source_link_root, source_link_roots).replace("|", "\\|")
        section = section_heading.replace("|", "\\|")
        excerpt = source_excerpt(str(source.get("evidence_text") or "")).replace("|", "\\|")
        lines.append(f"| `{document}` | {section} | {excerpt} |")
    return "\n".join(lines)


def source_windows_path(
    source_path: str,
    source_link_root: str,
    source_link_roots: tuple[tuple[str, str], ...] = (),
) -> str:
    relative_parts = PurePosixPath(source_path).parts
    if relative_parts:
        mappings = dict(source_link_roots)
        mapped_root = mappings.get(relative_parts[0])
        if mapped_root:
            return str(PureWindowsPath(mapped_root, *relative_parts[1:]))
    if not source_link_root:
        return source_path
    return str(PureWindowsPath(source_link_root, *relative_parts))


def source_excerpt(text: str, limit: int = 500) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="ESG repair search service")
    parser.add_argument(
        "command", nargs="?", default="serve",
        choices=["serve", "ingest", "ingest-status", "sources-status", "health"],
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-sources", action="store_true")
    args = parser.parse_args()
    app = Application()
    if args.command == "ingest":
        print(json.dumps(
            app.ingestion.run(force=args.force, refresh_sources=args.refresh_sources),
            ensure_ascii=False,
            indent=2,
        ))
        return
    if args.command == "ingest-status":
        print(json.dumps(app.store.latest_job() or {"status": "not_started"}, ensure_ascii=False, indent=2))
        return
    if args.command == "sources-status":
        print(json.dumps(app.ingestion.source_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "health":
        print(json.dumps(app.health(), ensure_ascii=False, indent=2))
        return
    host = args.host or app.settings.api_host
    port = args.port or app.settings.api_port
    server = ThreadingHTTPServer((host, port), handler_for(app))
    print(f"ESG repair search listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
