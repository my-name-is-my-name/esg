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

from esg.chunking import NamedSection, find_named_sections, split_text
from esg.clients import EmbeddingClient, SemanticExtractor
from esg.config import Settings, filename_matches, filename_tokens
from esg.mineru import MinerUConverter
from esg.models import Repair, Zone
from esg.storage import SQLiteStore
from esg.vector_index import VectorIndex


INDEX_VERSION = 4
REPAIR_HEADING = "Описание ремонта"
EXTRACTION_PROMPT_VERSION = "document-repairs-v1"


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
        document_extractor: SemanticExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embeddings = embeddings
        self.vector_index = vector_index
        self.document_extractor = document_extractor
        self.converter = MinerUConverter(settings)
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
                "phase": "conversion",
                "scanned": len(discovered),
                "canonical": len(groups),
                "duplicates": len(discovered) - len(groups),
                "conversion_total": len(groups),
            })
            self._initialize_registry(discovered, groups, canonical, bool(payload["force"]))
            self.store.upsert_job(job_id, "running", payload, started, None)

            ready: list[tuple[SourceFile, list[SourceFile], str]] = []
            for source_name_key in sorted(canonical):
                if self._cancelled(job_id, started, payload):
                    return
                selected = canonical[source_name_key]
                try:
                    registry = self.store.registry_item(selected.relative) or {}
                    terminal = str(registry.get("status") or "") in {
                        "indexed", "indexed_zero_repairs", "no_repair_section"
                    }
                    markdown_path = Path(str(registry.get("markdown_path") or ""))
                    if terminal and markdown_path.is_file() and not bool(payload["force"]):
                        payload["skipped"] = int(payload["skipped"]) + 1
                        continue
                    sections, content_hash = self._convert_document(selected, bool(payload["force"]))
                    payload["converted"] = int(payload["converted"]) + 1
                    if sections:
                        ready.append((selected, groups[source_name_key], content_hash))
                    else:
                        payload["no_repair_section"] = int(payload["no_repair_section"]) + 1
                        self._store_empty_document(selected, content_hash, "no_repair_section", "Нет раздела «Описание ремонта»")
                    self._replace_aliases(selected, groups[source_name_key])
                except Exception as exc:
                    payload["conversion_failed"] = int(payload["conversion_failed"]) + 1
                    self.store.update_registry(
                        selected.relative,
                        status="conversion_failed",
                        reason=repr(exc),
                        updated_at=utc_now(),
                    )
                    self._append_error(payload, selected.relative, "conversion", exc)
                finally:
                    payload["conversion_processed"] = int(payload["conversion_processed"]) + 1
                    self.store.upsert_job(job_id, "running", payload, started, None)

            payload["phase"] = "extraction"
            payload["extraction_total"] = len(ready)
            self.store.upsert_job(job_id, "running", payload, started, None)
            for selected, aliases, content_hash in ready:
                if self._cancelled(job_id, started, payload):
                    return
                try:
                    markdown_path = self.converter.markdown_path(
                        selected.relative, selected.path.suffix.lower()
                    )
                    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
                    sections = find_named_sections(markdown, REPAIR_HEADING)
                    if not sections:
                        raise RuntimeError("Repair sections disappeared after conversion")
                    repairs = self._extract_repairs(selected, sections, content_hash, bool(payload["force"]))
                    if repairs:
                        self._index_repairs(selected, content_hash, sections, repairs)
                        status = "indexed"
                        reason = ""
                        payload["indexed"] = int(payload["indexed"]) + 1
                        payload["repairs"] = int(payload["repairs"]) + len(repairs)
                    else:
                        self._store_empty_document(
                            selected, content_hash, "indexed_zero_repairs",
                            "LLM не обнаружила подтвержденных ремонтов в разделе «Описание ремонта»",
                        )
                        status = "indexed_zero_repairs"
                        reason = "LLM не обнаружила подтвержденных ремонтов"
                        payload["indexed_zero_repairs"] = int(payload["indexed_zero_repairs"]) + 1
                    self.store.update_registry(
                        selected.relative,
                        status=status,
                        reason=reason,
                        repair_count=len(repairs),
                        updated_at=utc_now(),
                    )
                    self._replace_aliases(selected, aliases)
                except Exception as exc:
                    payload["extraction_failed"] = int(payload["extraction_failed"]) + 1
                    self.store.update_registry(
                        selected.relative,
                        status="extraction_failed",
                        reason=repr(exc),
                        updated_at=utc_now(),
                    )
                    self._append_error(payload, selected.relative, "extraction", exc)
                payload["extraction_processed"] = int(payload["extraction_processed"]) + 1
                self.store.upsert_job(job_id, "running", payload, started, None)

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
            selected = canonical[source.path.name.casefold()]
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
                "source_name_key": source.path.name.casefold(),
                "size": int(source.size or 0),
                "mtime": float(source.mtime or 0),
                "canonical_source_path": selected.relative,
                "canonical_document_id": document_id_for(selected.relative),
                "is_canonical": int(is_canonical),
                "status": status,
                "reason": reason,
                "markdown_path": str(old.get("markdown_path") or "") if unchanged and old else "",
                "repair_section_count": int(old.get("repair_section_count") or 0) if unchanged and old else 0,
                "repair_count": int(old.get("repair_count") or 0) if unchanged and old else 0,
                "content_hash": str(old.get("content_hash") or "") if unchanged and old else "",
                "updated_at": now,
            })
        self.store.replace_registry(rows)

    def _convert_document(self, source: SourceFile, force: bool) -> tuple[list[NamedSection], str]:
        target = self.converter.markdown_path(source.relative, source.path.suffix.lower())
        registry = self.store.registry_item(source.relative)
        reuse = bool(
            target.is_file()
            and registry
            and str(registry.get("status") or "") != "conversion_pending"
            and not force
        )
        content_hash = str(registry.get("content_hash") or "") if registry else ""
        markdown_path = self.converter.convert(
            source.path, document_id_for(source.relative), source.relative, reuse_existing=reuse
        )
        if not content_hash:
            content_hash = sha256_file(markdown_path)
        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        sections = find_named_sections(markdown, REPAIR_HEADING)
        status = "converted" if sections else "no_repair_section"
        reason = "" if sections else "Нет раздела «Описание ремонта»"
        self.store.update_registry(
            source.relative,
            status=status,
            reason=reason,
            markdown_path=str(markdown_path),
            repair_section_count=len(sections),
            content_hash=content_hash,
            updated_at=utc_now(),
        )
        return sections, content_hash

    def _extract_repairs(
        self,
        source: SourceFile,
        sections: list[NamedSection],
        content_hash: str,
        force: bool,
    ) -> list[Repair]:
        self.store.update_registry(
            source.relative, status="extraction_pending", reason="", updated_at=utc_now()
        )
        parts: list[dict[str, str]] = []
        for section in sections:
            text_parts = split_text(section.text, max(1, self.settings.document_extraction_max_chars))
            parts.extend({"heading": section.heading, "text": part} for part in text_parts if part.strip())
        source_text = "\n\n".join(section.text for section in sections)
        repairs: list[Repair] = []
        for part in parts:
            cache_key = hashlib.sha256((
                EXTRACTION_PROMPT_VERSION + "\0" + content_hash + "\0" + part["heading"] + "\0" + part["text"]
            ).encode("utf-8")).hexdigest()
            cached = None if force else self.store.cached_extraction(cache_key)
            if cached and cached.get("status") == "ok":
                values = (cached.get("payload") or {}).get("repairs", [])
                extracted = [Repair.model_validate(item) for item in values]
            else:
                if self.document_extractor is None:
                    raise RuntimeError("Document repair extractor is not configured")
                result = self.document_extractor.extract_document_repairs([part])
                extracted = result.repairs
                invalid = [item for item in extracted if not quote_in_source(item.evidence_text, part["text"])]
                if invalid:
                    raise ValueError(
                        f"LLM evidence is not a verbatim fragment of {REPAIR_HEADING}: "
                        f"{invalid[0].evidence_text[:120]!r}"
                    )
                self.store.cache_extraction(
                    cache_key,
                    {"repairs": [item.model_dump() for item in extracted]},
                    "ok",
                    "",
                    utc_now(),
                )
            for repair in extracted:
                if not quote_in_source(repair.evidence_text, source_text):
                    raise ValueError(
                        f"LLM evidence is not a verbatim fragment of {REPAIR_HEADING}: {repair.evidence_text[:120]!r}"
                    )
                repairs.append(repair)
        return merge_repairs(repairs)

    def _index_repairs(
        self,
        source: SourceFile,
        content_hash: str,
        sections: list[NamedSection],
        repairs: list[Repair],
    ) -> None:
        document_id = document_id_for(source.relative)
        zones = [zone for repair in repairs for zone in repair.zones]
        evidence = "\n\n".join(repair.evidence_text for repair in repairs)
        headings = list(dict.fromkeys(section.heading for section in sections))
        record = _base_record(
            record_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}|document-repairs-v1")),
            document_id=document_id,
            relative=source.relative,
            record_type="document_repairs",
            section_id=f"{document_id}-repair-descriptions",
            section_heading="; ".join(headings),
            heading_path=headings,
            section_role="repair_description",
            extraction_status="ok",
            defect_type=", ".join(dict.fromkeys(item.defect_type for item in repairs if item.defect_type)),
            repair_description=", ".join(dict.fromkeys(item.repair_id for item in repairs if item.repair_id)),
            evidence_text=evidence,
            search_text=repair_search_text(repairs),
            repairs=[item.model_dump() for item in repairs],
            zones=[item.model_dump() for item in zones],
        )
        vector = self.embeddings.embed([str(record["search_text"])])[0]
        document = self._document(source, content_hash, "indexed", "")
        self.store.replace_document(document, [record])
        self.vector_index.replace_document(document_id, [(record, vector)])

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
        if refresh_sources:
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
    zones: list[dict[str, object]] | None = None,
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
        "zone_text": "",
        "evidence_text": evidence_text,
        "search_text": search_text,
        "extraction_status": extraction_status,
        "source_path": relative,
        "filename_tokens": list(tokens),
        "filename_token_key": "|" + "|".join(tokens) + "|",
        "repairs": repairs or [],
        "zones": zones or [],
    }


def repair_search_text(repairs: list[Repair]) -> str:
    parts: list[str] = []
    for repair in repairs:
        parts.extend([repair.repair_id, repair.defect_type, repair.evidence_text])
        for zone in repair.zones:
            parts.extend([zone.structure, zone.system, zone.region, zone.side, zone.surface, *zone.components])
            parts.extend(
                f"{item.kind} {item.start or ''} {item.end or ''} {item.qualifier} {item.role}"
                for item in zone.elements
            )
    return "\n".join(part for part in parts if part).strip()


def merge_repairs(repairs: list[Repair]) -> list[Repair]:
    merged: dict[str, Repair] = {}
    for repair in repairs:
        key = repair.repair_id.casefold() if repair.repair_id else normalize_quote(repair.evidence_text)
        previous = merged.get(key)
        if not previous:
            merged[key] = repair
            continue
        seen = {json.dumps(zone.model_dump(), ensure_ascii=False, sort_keys=True) for zone in previous.zones}
        zones = list(previous.zones)
        for zone in repair.zones:
            encoded = json.dumps(zone.model_dump(), ensure_ascii=False, sort_keys=True)
            if encoded not in seen:
                seen.add(encoded)
                zones.append(zone)
        merged[key] = previous.model_copy(update={"zones": zones})
    return list(merged.values())


def normalize_quote(value: str) -> str:
    value = re.sub(r"<a\b[^>]*>.*?</a>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[*_`]", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def quote_in_source(quote: str, source: str) -> bool:
    normalized = normalize_quote(quote)
    return bool(normalized) and normalized in normalize_quote(source)


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
        groups.setdefault(source.path.name.casefold(), []).append(source)
    return groups


def select_canonical_source(sources: list[SourceFile]) -> SourceFile:
    return max(sources, key=lambda source: (
        source.mtime if source.mtime is not None else source.path.stat().st_mtime,
        source.relative.casefold(),
    ))


def select_canonical(sources: list[Path], root: Path) -> Path:
    return max(sources, key=lambda path: (path.stat().st_mtime, path.relative_to(root).as_posix().casefold()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
