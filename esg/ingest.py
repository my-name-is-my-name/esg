from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from esg.chunking import NamedSection, find_named_sections
from esg.clients import EmbeddingClient
from esg.config import Settings, filename_matches, filename_tokens
from esg.repair_text import append_interval_tokens, repair_section_chunks
from esg.storage import SQLiteStore
from esg.vector_index import VectorIndex


INDEX_VERSION = 7
REPAIR_HEADING = "Описание ремонта"
INDEX_BATCH_DOCUMENTS = 32
INDEX_BATCH_CHUNKS = 32


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


def source_name_key(source: SourceFile) -> str:
    return Path(source.relative).name.casefold()


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
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def start(self, force: bool = False, refresh_sources: bool = False) -> dict[str, object]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                latest = self.store.latest_job() or {}
                return {"accepted": False, "reason": "ingestion_already_running", **latest}
            job_id = str(uuid.uuid4())
            started = utc_now()
            payload = self._job_payload(force, refresh_sources)
            self.store.upsert_job(job_id, "running", payload, started, None)
            self._cancel.clear()
            self._thread = threading.Thread(
                target=self._run_job, args=(job_id, started, payload), daemon=True
            )
            self._thread.start()
            return {"accepted": True, "job_id": job_id, "status": "running"}

    def run(self, force: bool = False, refresh_sources: bool = False) -> dict[str, object]:
        job_id = str(uuid.uuid4())
        started = utc_now()
        payload = self._job_payload(force, refresh_sources)
        self.store.upsert_job(job_id, "running", payload, started, None)
        self._cancel.clear()
        self._run_job(job_id, started, payload)
        return self.store.latest_job() or payload

    def cancel(self) -> dict[str, object]:
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return {"accepted": False, "reason": "ingestion_not_running"}
            self._cancel.set()
            latest = self.store.latest_job() or {}
            return {"accepted": True, "status": "cancelling", "job_id": latest.get("job_id", "")}

    def _run_job(self, job_id: str, started: str, payload: dict[str, object]) -> None:
        try:
            self.settings.ensure_dirs()
            discovered = self._sources_for_job(bool(payload["refresh_sources"]))
            groups = group_source_files(discovered)
            canonical = {name: select_canonical_source(group) for name, group in groups.items()}
            payload.update({
                "phase": "markdown",
                "scanned": len(discovered),
                "canonical": len(groups),
                "duplicates": len(discovered) - len(groups),
                "conversion_total": 0,
                "extraction_total": len(groups),
            })
            self._initialize_registry(discovered, groups, canonical, bool(payload["force"]))
            self.store.upsert_job(job_id, "running", payload, started, None)

            batch: list[tuple[SourceFile, list[SourceFile], str, list[NamedSection], list[dict[str, object]]]] = []
            for source_name_key in sorted(canonical):
                if self._cancelled(job_id, started, payload):
                    return
                selected = canonical[source_name_key]
                try:
                    registry = self.store.registry_item(selected.relative) or {}
                    document = self.store.document_state(selected.relative) or {}
                    status = str(registry.get("status") or "")
                    terminal = status == "no_repair_section" or (
                        status in {"indexed", "indexed_zero_repairs"}
                        and int(document.get("index_version") or 0) == INDEX_VERSION
                    )
                    if terminal and not bool(payload["force"]):
                        payload["skipped"] = int(payload["skipped"]) + 1
                        continue
                    sections, content_hash = self._load_markdown_sections(registry)
                    if sections:
                        records = self._chunk_records(selected, sections)
                        if records:
                            batch.append((selected, groups[source_name_key], content_hash, sections, records))
                            if len(batch) >= INDEX_BATCH_DOCUMENTS or _batch_record_count(batch) >= INDEX_BATCH_CHUNKS:
                                self._index_batch(batch, payload)
                                batch = []
                        else:
                            self._store_empty_document(
                                selected, content_hash, "indexed_zero_repairs",
                                "Раздел «Описание ремонта» не содержит индексируемого текста",
                            )
                            self._mark_indexed(selected, groups[source_name_key], 0, "indexed_zero_repairs", payload)
                    else:
                        payload["no_repair_section"] = int(payload["no_repair_section"]) + 1
                        self._store_empty_document(selected, content_hash, "no_repair_section", "Нет раздела «Описание ремонта»")
                        self.store.update_registry(
                            selected.relative, status="no_repair_section",
                            reason="Нет раздела «Описание ремонта»", repair_count=0, updated_at=utc_now(),
                        )
                    self._replace_aliases(selected, groups[source_name_key])
                except Exception as exc:
                    payload["extraction_failed"] = int(payload["extraction_failed"]) + 1
                    self.store.update_registry(
                        selected.relative,
                        status="extraction_failed",
                        reason=repr(exc),
                        updated_at=utc_now(),
                    )
                    self._append_error(payload, selected.relative, "markdown", exc)
                finally:
                    payload["extraction_processed"] = int(payload["extraction_processed"]) + 1
                    self.store.upsert_job(job_id, "running", payload, started, None)
            if batch:
                self._index_batch(batch, payload)

            failures = int(payload["conversion_failed"]) + int(payload["extraction_failed"])
            status = "completed" if failures == 0 else "completed_with_errors"
            payload["phase"] = "completed"
            self.store.upsert_job(job_id, status, payload, started, utc_now())
        except Exception as exc:
            self._append_error(payload, "", "job", exc)
            self.store.upsert_job(job_id, "failed", payload, started, utc_now())

    def _cancelled(self, job_id: str, started: str, payload: dict[str, object]) -> bool:
        if not self._cancel.is_set():
            return False
        payload["phase"] = "cancelled"
        self.store.upsert_job(job_id, "cancelled", payload, started, utc_now())
        return True

    @staticmethod
    def _job_payload(force: bool, refresh_sources: bool) -> dict[str, object]:
        return {
            "force": force,
            "refresh_sources": refresh_sources,
            "source_catalog": "refresh" if refresh_sources else "cached",
            "phase": "preparing",
            "scanned": 0,
            "canonical": 0,
            "duplicates": 0,
            "conversion_total": 0,
            "conversion_processed": 0,
            "converted": 0,
            "skipped": 0,
            "no_repair_section": 0,
            "conversion_failed": 0,
            "extraction_total": 0,
            "extraction_processed": 0,
            "indexed": 0,
            "indexed_zero_repairs": 0,
            "repairs": 0,
            "chunks": 0,
            "extraction_failed": 0,
            "errors": [],
        }

    def _initialize_registry(
        self,
        discovered: list[SourceFile],
        groups: dict[str, list[SourceFile]],
        canonical: dict[str, SourceFile],
        force: bool,
    ) -> None:
        previous = {str(item["source_path"]): item for item in self.store.registry_items()}
        rows: list[dict[str, object]] = []
        now = utc_now()
        for source in discovered:
            selected = canonical[source_name_key(source)]
            is_canonical = source.relative == selected.relative
            old = previous.get(source.relative)
            unchanged = bool(
                old
                and int(old["size"]) == int(source.size or 0)
                and float(old["mtime"]) == float(source.mtime or 0)
                and str(old["canonical_source_path"]) == selected.relative
            )
            if not is_canonical:
                status = "duplicate"
                reason = f"Дубликат имени; индексируется {selected.relative}"
            elif unchanged and not force:
                status = str(old["status"])
                reason = str(old["reason"])
            else:
                status = "conversion_pending"
                reason = ""
            rows.append({
                "source_path": source.relative,
                "root_key": source.root_key,
                "physical_relative": source.physical_relative,
                "source_name_key": source_name_key(source),
                "size": int(source.size or 0),
                "mtime": float(source.mtime or 0),
                "canonical_source_path": selected.relative,
                "canonical_document_id": document_id_for(selected.relative),
                "is_canonical": int(is_canonical),
                "status": status,
                "reason": reason,
                "markdown_path": self._registry_markdown_path(source, old, unchanged),
                "repair_section_count": int(old.get("repair_section_count") or 0) if unchanged and old else 0,
                "repair_count": int(old.get("repair_count") or 0) if unchanged and old else 0,
                "content_hash": str(old.get("content_hash") or "") if unchanged and old else "",
                "updated_at": now,
            })
        self.store.replace_registry(rows)

    def _registry_markdown_path(
        self, source: SourceFile, old: dict[str, object] | None, unchanged: bool
    ) -> str:
        if self.settings.markdown_only_ingestion:
            return str(source.path)
        if unchanged and old:
            return str(old.get("markdown_path") or "")
        return ""

    def _load_markdown_sections(self, registry: dict[str, object]) -> tuple[list[NamedSection], str]:
        markdown_path = Path(str(registry.get("markdown_path") or ""))
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        sections = find_named_sections(markdown, REPAIR_HEADING)
        content_hash = str(registry.get("content_hash") or "") or sha256_file(markdown_path)
        return sections, content_hash

    def _chunk_records(
        self,
        source: SourceFile,
        sections: list[NamedSection],
    ) -> list[dict[str, object]]:
        document_id = document_id_for(source.relative)
        records: list[dict[str, object]] = []
        for section in sections:
            chunks = repair_section_chunks(section.text, self.settings.repair_chunk_max_chars)
            for chunk_index, chunk in enumerate(chunks, start=1):
                raw_id = f"{document_id}|{section.ordinal}|{chunk_index}|{hashlib.sha256(chunk.encode()).hexdigest()}"
                records.append(_base_record(
                    record_id=str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id)),
                    document_id=document_id,
                    relative=source.relative,
                    record_type="document_repairs",
                    section_id=f"{document_id}-repair-{section.ordinal}-{chunk_index}",
                    section_heading=section.heading,
                    heading_path=[section.heading],
                    section_role="repair_description",
                    extraction_status="deterministic",
                    evidence_text=chunk,
                    search_text=append_interval_tokens(f"{REPAIR_HEADING}\n{chunk}"),
                    repairs=[],
                    zones=[],
                    zone_text=chunk,
                ))
        return records

    def _index_batch(
        self,
        batch: list[tuple[SourceFile, list[SourceFile], str, list[NamedSection], list[dict[str, object]]]],
        payload: dict[str, object],
    ) -> None:
        records = [record for _, _, _, _, document_records in batch for record in document_records]
        vectors = self.embeddings.embed([str(record["search_text"]) for record in records])
        vector_offset = 0
        vector_documents: list[tuple[str, list[tuple[dict[str, object], list[float]]]]] = []
        for source, aliases, content_hash, sections, document_records in batch:
            count = len(document_records)
            document_vectors = vectors[vector_offset : vector_offset + count]
            vector_offset += count
            self.store.replace_document(
                self._document(source, content_hash, "indexed", ""), document_records
            )
            vector_documents.append((
                document_id_for(source.relative), list(zip(document_records, document_vectors, strict=True))
            ))
            self.store.update_registry(
                source.relative, status="indexed", reason="", repair_count=len(sections), updated_at=utc_now()
            )
            self._replace_aliases(source, aliases)
            payload["indexed"] = int(payload["indexed"]) + 1
            payload["repairs"] = int(payload["repairs"]) + len(sections)
            payload["chunks"] = int(payload["chunks"]) + count
        self.vector_index.replace_documents(vector_documents)

    def _mark_indexed(
        self, source: SourceFile, aliases: list[SourceFile], repair_count: int,
        status: str, payload: dict[str, object],
    ) -> None:
        self.store.update_registry(
            source.relative, status=status, reason="", repair_count=repair_count, updated_at=utc_now()
        )
        self._replace_aliases(source, aliases)
        payload[status] = int(payload[status]) + 1

    def _store_empty_document(self, source: SourceFile, content_hash: str, status: str, reason: str) -> None:
        document_id = document_id_for(source.relative)
        self.vector_index.delete_document(document_id)
        self.store.replace_document(self._document(source, content_hash, status, reason), [])

    def _document(self, source: SourceFile, content_hash: str, status: str, error: str) -> dict[str, object]:
        return {
            "document_id": document_id_for(source.relative),
            "source_path": source.relative,
            "content_hash": content_hash,
            "size": int(source.size or source.path.stat().st_size),
            "mtime": float(source.mtime or source.path.stat().st_mtime),
            "status": status,
            "error": error,
            "indexed_at": utc_now(),
            "index_version": INDEX_VERSION,
        }

    def _replace_aliases(self, selected: SourceFile, group: list[SourceFile]) -> None:
        document_id = document_id_for(selected.relative)
        if not self.store.document_state(selected.relative):
            return
        self.store.replace_aliases(document_id, [{
            "source_path": alias.relative,
            "source_name_key": alias.path.name.casefold(),
            "canonical_document_id": document_id,
            "size": int(alias.size or 0),
            "mtime": float(alias.mtime or 0),
        } for alias in group])

    @staticmethod
    def _append_error(payload: dict[str, object], source_path: str, phase: str, exc: Exception) -> None:
        errors = payload["errors"]
        assert isinstance(errors, list)
        if len(errors) < 200:
            errors.append({"source_path": source_path, "phase": phase, "error": repr(exc)})

    def source_config_signature(self) -> str:
        payload = {
            "version": 1,
            "roots": [{"key": key, "path": str(root), "prefix": prefix} for key, root, prefix in self._source_roots()],
            "extensions": sorted(self.settings.input_extensions),
            "filename_tokens": sorted(self.settings.input_filename_tokens),
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def source_status(self) -> dict[str, object]:
        return self.store.source_sync_status(self.source_config_signature())

    def _sources_for_job(self, refresh_sources: bool) -> list[SourceFile]:
        signature = self.source_config_signature()
        if self.settings.markdown_only_ingestion:
            if not refresh_sources:
                inventory = self.store.source_inventory(signature)
                if inventory is not None:
                    return self._sources_from_markdown_inventory(inventory)
            discovered = self._discover_markdown_sources()
            self.store.replace_source_inventory(
                [self._inventory_item(source) for source in discovered],
                signature,
                utc_now(),
                utc_now(),
            )
            return discovered
        if refresh_sources:
            if self.settings.markdown_only_ingestion:
                raise RuntimeError("Source refresh is disabled in Markdown-only ingestion mode")
            started = utc_now()
            try:
                discovered = self._discover_sources()
                self.store.replace_source_inventory(
                    [self._inventory_item(source) for source in discovered], signature, started, utc_now()
                )
                return discovered
            except Exception as exc:
                self.store.record_source_sync_failure(signature, started, utc_now(), repr(exc))
                raise
        inventory = self.store.source_inventory(signature)
        if inventory is None:
            raise RuntimeError("Source catalog is missing or incompatible; run ingestion with refresh_sources=true")
        roots = {key: root for key, root, _ in self._source_roots()}
        return [SourceFile(
            path=roots[str(item["root_key"])] / str(item["physical_relative"]),
            relative=str(item["source_path"]),
            root_key=str(item["root_key"]),
            physical_relative=str(item["physical_relative"]),
            size=int(item["size"]),
            mtime=float(item["mtime"]),
        ) for item in inventory]

    def _sources_from_markdown_inventory(self, inventory: list[dict[str, object]]) -> list[SourceFile]:
        return [SourceFile(
            path=self.settings.markdown_dir / str(item["physical_relative"]),
            relative=str(item["source_path"]),
            root_key=str(item["root_key"]),
            physical_relative=str(item["physical_relative"]),
            size=int(item["size"]),
            mtime=float(item["mtime"]),
        ) for item in inventory]

    @staticmethod
    def _inventory_item(source: SourceFile) -> dict[str, object]:
        return {
            "source_path": source.relative,
            "root_key": source.root_key,
            "physical_relative": source.physical_relative,
            "source_name_key": source_name_key(source),
            "size": source.size,
            "mtime": source.mtime,
        }

    def _discover_markdown_sources(self) -> list[SourceFile]:
        discovered: list[SourceFile] = []
        root = self.settings.markdown_dir
        for path in root.rglob("*.docx.md"):
            if not path.is_file():
                continue
            relative_md = path.relative_to(root).as_posix()
            relative_docx = relative_md[:-3]
            if not filename_matches(Path(relative_docx).name, self.settings.input_filename_tokens):
                continue
            stat = path.stat()
            discovered.append(SourceFile(
                path=path,
                relative=relative_docx,
                root_key="markdown",
                physical_relative=relative_md,
                size=stat.st_size,
                mtime=stat.st_mtime,
            ))
        return sorted(discovered, key=lambda item: item.relative.casefold())

    def _discover_sources(self) -> list[SourceFile]:
        discovered: list[SourceFile] = []
        for root_key, root, prefix in self._source_roots():
            for directory, _, filenames in os.walk(root):
                directory_path = Path(directory)
                for filename in filenames:
                    path = directory_path / filename
                    if filename.startswith("~$") or not self._is_source(path):
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
        return sorted(discovered, key=lambda item: item.relative.casefold())

    def _is_source(self, path: Path) -> bool:
        return (
            path.suffix.lower() in self.settings.input_extensions
            and not path.name.startswith("~$")
            and filename_matches(path.name, self.settings.input_filename_tokens)
            and path.is_file()
        )

    def _source_roots(self) -> list[tuple[str, Path, str]]:
        roots = [("primary", self.settings.input_dir, "")]
        if self.settings.archive_input_dir:
            roots.append(("archive", self.settings.archive_input_dir, "__s7a"))
        return roots


def _base_record(
    *, record_id: str, document_id: str, relative: str, record_type: str,
    section_id: str, section_heading: str, heading_path: list[str], section_role: str,
    extraction_status: str, evidence_text: str, search_text: str, defect_type: str = "",
    repair_description: str = "", repairs: list[dict[str, object]] | None = None,
    zones: list[dict[str, object]] | None = None, zone_text: str = "",
) -> dict[str, object]:
    tokens = filename_tokens(relative)
    return {
        "record_id": record_id,
        "document_id": document_id,
        "record_type": record_type,
        "section_id": section_id,
        "section_heading": section_heading,
        "heading_path_json": json.dumps(heading_path, ensure_ascii=False),
        "section_role": section_role,
        "defect_type": defect_type,
        "repair_description": repair_description,
        "frame_start": None,
        "frame_end": None,
        "stringer_start": None,
        "stringer_end": None,
        "component": "",
        "side": "",
        "structure": "",
        "system": "",
        "region": "",
        "surface": "",
        "components": [],
        "elements": [],
        "zone_text": zone_text,
        "evidence_text": evidence_text,
        "search_text": search_text,
        "extraction_status": extraction_status,
        "source_path": relative,
        "filename_tokens": list(tokens),
        "filename_token_key": "|" + "|".join(tokens) + "|",
        "repairs": repairs or [],
        "zones": zones or [],
    }


def document_id_for(relative: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, relative))


def group_sources(sources: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for source in sources:
        groups.setdefault(source.name.casefold(), []).append(source)
    return groups


def group_source_files(sources: list[SourceFile]) -> dict[str, list[SourceFile]]:
    groups: dict[str, list[SourceFile]] = {}
    for source in sources:
        groups.setdefault(source_name_key(source), []).append(source)
    return groups


def select_canonical_source(sources: list[SourceFile]) -> SourceFile:
    return max(sources, key=lambda source: (
        source.mtime if source.mtime is not None else source.path.stat().st_mtime,
        source.relative.casefold(),
    ))


def _batch_record_count(
    batch: list[tuple[SourceFile, list[SourceFile], str, list[NamedSection], list[dict[str, object]]]]
) -> int:
    return sum(len(records) for _, _, _, _, records in batch)


def select_canonical(sources: list[Path], root: Path) -> Path:
    return max(sources, key=lambda path: (path.stat().st_mtime, path.relative_to(root).as_posix().casefold()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
