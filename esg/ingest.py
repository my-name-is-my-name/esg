from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from esg.chunking import Section, parse_markdown
from esg.clients import EmbeddingClient
from esg.config import Settings, filename_matches, filename_tokens
from esg.mineru import MinerUConverter
from esg.storage import SQLiteStore
from esg.vector_index import VectorIndex


INDEX_VERSION = 3


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    relative: str
    root_key: str = "primary"
    physical_relative: str = ""
    size: int | None = None
    mtime: float | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        embeddings: EmbeddingClient,
        vector_index: VectorIndex,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embeddings = embeddings
        self.vector_index = vector_index
        self.converter = MinerUConverter(settings)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, force: bool = False, refresh_sources: bool = False) -> dict[str, object]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                latest = self.store.latest_job() or {}
                return {"accepted": False, "reason": "ingestion_already_running", **latest}
            job_id = str(uuid.uuid4())
            started = utc_now()
            payload = self._job_payload(force, refresh_sources)
            self.store.upsert_job(job_id, "running", payload, started, None)
            self._thread = threading.Thread(target=self._run_job, args=(job_id, started, payload), daemon=True)
            self._thread.start()
            return {"accepted": True, "job_id": job_id, "status": "running"}

    def run(self, force: bool = False, refresh_sources: bool = False) -> dict[str, object]:
        job_id = str(uuid.uuid4())
        started = utc_now()
        payload = self._job_payload(force, refresh_sources)
        self.store.upsert_job(job_id, "running", payload, started, None)
        self._run_job(job_id, started, payload)
        return self.store.latest_job() or payload

    def _run_job(self, job_id: str, started: str, payload: dict[str, object]) -> None:
        try:
            self.settings.ensure_dirs()
            discovered = self._sources_for_job(bool(payload["refresh_sources"]))
            groups = group_source_files(discovered)
            payload["scanned"] = len(discovered)
            payload["canonical"] = len(groups)
            payload["duplicates"] = len(discovered) - len(groups)
            self.store.upsert_job(job_id, "running", payload, started, None)
            existing = {str(row["source_path"]): row for row in self.store.documents()}
            seen: set[str] = set()
            for source_name_key, group in groups.items():
                selected = select_canonical_source(group)
                source = selected.path
                relative = selected.relative
                try:
                    previous = existing.get(relative)
                    source_size = selected.size
                    source_mtime = selected.mtime
                    if source_size is None or source_mtime is None:
                        stat = source.stat()
                        source_size = stat.st_size
                        source_mtime = stat.st_mtime
                    metadata_unchanged = bool(
                        previous
                        and int(previous.get("size") or -1) == source_size
                        and float(previous.get("mtime") or -1) == source_mtime
                    )
                    current_version = bool(previous and int(previous.get("index_version") or 1) == INDEX_VERSION)
                    if metadata_unchanged and current_version and not bool(payload["force"]):
                        payload["skipped"] = int(payload["skipped"]) + 1
                        document_id = str(previous["document_id"])
                    elif metadata_unchanged and previous and not bool(payload["force"]):
                        document_id = str(previous["document_id"])
                        removed = self.store.migrate_raw_document(document_id, INDEX_VERSION)
                        self.vector_index.delete_records(removed)
                        payload["migrated"] = int(payload["migrated"]) + 1
                    else:
                        content_hash = sha256_file(source)
                        content_unchanged = bool(previous and previous.get("content_hash") == content_hash)
                        markdown_target = self.converter.markdown_path(relative, source.suffix.lower())
                        existing_markdown_is_fresh = bool(
                            previous is None
                            and markdown_target.is_file()
                            and markdown_target.stat().st_mtime >= source_mtime
                        )
                        evidence_count = self._index_source(
                            source,
                            relative,
                            content_hash,
                            reuse_markdown=(content_unchanged or existing_markdown_is_fresh)
                            and not bool(payload["force"]),
                        )
                        payload["indexed"] = int(payload["indexed"]) + 1
                        payload["evidence"] = int(payload["evidence"]) + evidence_count
                        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, relative))
                    aliases = []
                    for alias in group:
                        alias_size = alias.size
                        alias_mtime = alias.mtime
                        if alias_size is None or alias_mtime is None:
                            stat = alias.path.stat()
                            alias_size = stat.st_size
                            alias_mtime = stat.st_mtime
                        aliases.append({
                            "source_path": alias.relative,
                            "source_name_key": source_name_key,
                            "canonical_document_id": document_id,
                            "size": alias_size,
                            "mtime": alias_mtime,
                        })
                    self.store.replace_aliases(document_id, aliases)
                    seen.add(relative)
                except Exception as exc:
                    payload["failed"] = int(payload["failed"]) + 1
                    errors = payload["errors"]
                    assert isinstance(errors, list)
                    errors.append({"source_path": relative, "error": repr(exc)})
                    seen.update(
                        path for path in existing
                        if Path(path).name.casefold() == source_name_key
                    )
                self.store.upsert_job(job_id, "running", payload, started, None)
            for relative, document in existing.items():
                if relative in seen:
                    continue
                document_id = str(document["document_id"])
                self.vector_index.delete_document(document_id)
                self.store.delete_document(document_id)
                self.converter.delete_markdown(relative)
                payload["deleted"] = int(payload["deleted"]) + 1
            status = "completed" if int(payload["failed"]) == 0 else "completed_with_errors"
            self.store.upsert_job(job_id, status, payload, started, utc_now())
        except Exception as exc:
            errors = payload["errors"]
            assert isinstance(errors, list)
            errors.append({"source_path": "", "error": repr(exc)})
            self.store.upsert_job(job_id, "failed", payload, started, utc_now())

    @staticmethod
    def _job_payload(force: bool, refresh_sources: bool) -> dict[str, object]:
        return {
            "force": force,
            "refresh_sources": refresh_sources,
            "source_catalog": "refresh" if refresh_sources else "cached",
            "scanned": 0,
            "indexed": 0,
            "migrated": 0,
            "evidence": 0,
            "skipped": 0,
            "deleted": 0,
            "failed": 0,
            "errors": [],
        }

    def source_config_signature(self) -> str:
        payload = {
            "version": 1,
            "roots": [
                {"key": key, "path": str(root), "prefix": prefix}
                for key, root, prefix in self._source_roots()
            ],
            "extensions": sorted(self.settings.input_extensions),
            "filename_tokens": sorted(self.settings.input_filename_tokens),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def source_status(self) -> dict[str, object]:
        return self.store.source_sync_status(self.source_config_signature())

    def _sources_for_job(self, refresh_sources: bool) -> list[SourceFile]:
        signature = self.source_config_signature()
        if refresh_sources:
            started = utc_now()
            try:
                discovered = self._discover_sources()
                self.store.replace_source_inventory(
                    [self._inventory_item(source) for source in discovered],
                    signature,
                    started,
                    utc_now(),
                )
                return discovered
            except Exception as exc:
                self.store.record_source_sync_failure(signature, started, utc_now(), repr(exc))
                raise
        inventory = self.store.source_inventory(signature)
        if inventory is None:
            raise RuntimeError(
                "Source catalog is missing or incompatible; run ingestion with refresh_sources=true"
            )
        roots = {key: root for key, root, _ in self._source_roots()}
        return [
            SourceFile(
                path=roots[str(item["root_key"])] / str(item["physical_relative"]),
                relative=str(item["source_path"]),
                root_key=str(item["root_key"]),
                physical_relative=str(item["physical_relative"]),
                size=int(item["size"]),
                mtime=float(item["mtime"]),
            )
            for item in inventory
        ]

    @staticmethod
    def _inventory_item(source: SourceFile) -> dict[str, object]:
        return {
            "source_path": source.relative,
            "root_key": source.root_key,
            "physical_relative": source.physical_relative,
            "source_name_key": source.path.name.casefold(),
            "size": source.size,
            "mtime": source.mtime,
        }

    def _is_source(self, path: Path) -> bool:
        return (
            path.suffix.lower() in self.settings.input_extensions
            and not path.name.startswith("~$")
            and filename_matches(path.name, self.settings.input_filename_tokens)
            and path.is_file()
        )

    def _discover_sources(self) -> list[SourceFile]:
        discovered: list[SourceFile] = []
        for root_key, root, prefix in self._source_roots():
            walk_error: OSError | None = None

            def raise_walk_error(error: OSError) -> None:
                nonlocal walk_error
                walk_error = error
                raise error

            for directory, _, filenames in os.walk(root, onerror=raise_walk_error):
                directory_path = Path(directory)
                for filename in filenames:
                    if filename.startswith("~$"):
                        continue
                    path = directory_path / filename
                    if path.suffix.lower() not in self.settings.input_extensions:
                        continue
                    if not filename_matches(filename, self.settings.input_filename_tokens):
                        continue
                    relative = path.relative_to(root).as_posix()
                    stat = path.stat()
                    discovered.append(SourceFile(
                        path=path,
                        relative=f"{prefix}/{relative}" if prefix else relative,
                        root_key=root_key,
                        physical_relative=relative,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                    ))
            if walk_error is not None:
                raise walk_error
        return sorted(discovered, key=lambda item: item.relative.casefold())

    def _source_roots(self) -> list[tuple[str, Path, str]]:
        roots = [("primary", self.settings.input_dir, "")]
        if self.settings.archive_input_dir:
            roots.append(("archive", self.settings.archive_input_dir, "__s7a"))
        return roots

    def _index_source(
        self,
        source: Path,
        relative: str,
        content_hash: str,
        reuse_markdown: bool = False,
    ) -> int:
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, relative))
        markdown_path = self.converter.convert(source, document_id, relative, reuse_existing=reuse_markdown)
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        sections = parse_markdown(markdown, document_id)
        if not sections:
            raise RuntimeError("Document contains no indexable sections")
        records: list[dict[str, object]] = []
        for section in sections:
            role = "chapter_5" if is_chapter_five(section) else "other"
            records.append(self._raw_record(document_id, relative, section, role))
        vectors = self.embeddings.embed([str(record["search_text"]) for record in records])
        stat = source.stat()
        document = {
            "document_id": document_id,
            "source_path": relative,
            "content_hash": content_hash,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "status": "indexed",
            "error": "",
            "indexed_at": utc_now(),
            "index_version": INDEX_VERSION,
        }
        self.store.replace_document(document, records)
        self.vector_index.replace_document(document_id, list(zip(records, vectors, strict=True)))
        return 0

    @staticmethod
    def _raw_record(
        document_id: str,
        relative: str,
        section: Section,
        section_role: str = "other",
        extraction_status: str = "not_requested",
    ) -> dict[str, object]:
        return _base_record(
            record_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{section.section_id}|raw")),
            document_id=document_id,
            relative=relative,
            section=section,
            record_type="raw_section",
            section_role=section_role,
            extraction_status=extraction_status,
            evidence_text=section.text,
            search_text=section.search_text,
        )

def _base_record(
    *, record_id: str, document_id: str, relative: str, section: Section,
    record_type: str, section_role: str, extraction_status: str,
    defect_type: str = "", repair_description: str = "", frame_start: int | None = None,
    frame_end: int | None = None, stringer_start: int | None = None, stringer_end: int | None = None,
    component: str = "", side: str = "", zone_text: str = "", evidence_text: str, search_text: str,
    structure: str = "", system: str = "", region: str = "", surface: str = "",
    components: list[str] | None = None, elements: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    tokens = filename_tokens(relative)
    return {
        "record_id": record_id, "document_id": document_id, "record_type": record_type,
        "section_id": section.section_id, "section_heading": section.heading,
        "heading_path_json": json.dumps(section.heading_path, ensure_ascii=False),
        "section_role": section_role, "defect_type": defect_type,
        "repair_description": repair_description, "frame_start": frame_start, "frame_end": frame_end,
        "stringer_start": stringer_start, "stringer_end": stringer_end, "component": component,
        "side": side, "structure": structure, "system": system, "region": region, "surface": surface,
        "components": components or ([component] if component else []), "elements": elements or [],
        "zone_text": zone_text, "evidence_text": evidence_text,
        "search_text": search_text, "extraction_status": extraction_status, "source_path": relative,
        "filename_tokens": list(tokens),
        "filename_token_key": "|" + "|".join(tokens) + "|",
    }


def is_chapter_five(section: Section) -> bool:
    for heading in section.heading_path:
        cleaned = re.sub(r"<[^>]+>|[*_`]", "", heading).strip()
        match = re.match(r"^5(?:\.(?:\d+\.)*\d+)?(?:\s|\.|$)", cleaned)
        if match:
            return True
    return False


def group_sources(sources: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for source in sources:
        groups.setdefault(source.name.casefold(), []).append(source)
    return groups


def group_source_files(sources: list[SourceFile]) -> dict[str, list[SourceFile]]:
    groups: dict[str, list[SourceFile]] = {}
    for source in sources:
        groups.setdefault(source.path.name.casefold(), []).append(source)
    return groups


def select_canonical_source(sources: list[SourceFile]) -> SourceFile:
    return max(
        sources,
        key=lambda source: (
            source.mtime if source.mtime is not None else source.path.stat().st_mtime,
            source.relative.casefold(),
        ),
    )


def select_canonical(sources: list[Path], root: Path) -> Path:
    return max(
        sources,
        key=lambda path: (path.stat().st_mtime, path.relative_to(root).as_posix().casefold()),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
