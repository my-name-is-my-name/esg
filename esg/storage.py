from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from esg.config import filename_tokens


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_aliases (
    source_path TEXT PRIMARY KEY,
    source_name_key TEXT NOT NULL,
    canonical_document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_aliases_canonical
ON document_aliases(canonical_document_id);
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    record_type TEXT NOT NULL,
    section_id TEXT NOT NULL,
    section_heading TEXT NOT NULL,
    heading_path_json TEXT NOT NULL,
    section_role TEXT NOT NULL,
    defect_type TEXT NOT NULL,
    repair_description TEXT NOT NULL,
    frame_start INTEGER,
    frame_end INTEGER,
    stringer_start INTEGER,
    stringer_end INTEGER,
    component TEXT NOT NULL,
    side TEXT NOT NULL,
    zone_text TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    search_text TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    source_path TEXT NOT NULL,
    filename_token_key TEXT NOT NULL DEFAULT '|'
);
CREATE INDEX IF NOT EXISTS idx_records_document ON records(document_id);
CREATE INDEX IF NOT EXISTS idx_records_section_role ON records(section_role);
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    record_id UNINDEXED,
    search_text,
    tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS record_elements (
    record_id TEXT NOT NULL REFERENCES records(record_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL,
    start INTEGER,
    end INTEGER,
    qualifier TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (record_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_record_elements_kind_range
ON record_elements(kind, start, end);
CREATE TABLE IF NOT EXISTS extraction_cache (
    section_hash TEXT PRIMARY KEY,
    extraction_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_inventory (
    source_path TEXT PRIMARY KEY,
    root_key TEXT NOT NULL,
    physical_relative TEXT NOT NULL,
    source_name_key TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_inventory_name
ON source_inventory(source_name_key);
CREATE TABLE IF NOT EXISTS source_sync_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    config_signature TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    scanned INTEGER NOT NULL DEFAULT 0,
    canonical INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS document_registry (
    source_path TEXT PRIMARY KEY,
    root_key TEXT NOT NULL,
    physical_relative TEXT NOT NULL,
    source_name_key TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    canonical_source_path TEXT NOT NULL,
    canonical_document_id TEXT NOT NULL,
    is_canonical INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    markdown_path TEXT NOT NULL DEFAULT '',
    repair_section_count INTEGER NOT NULL DEFAULT 0,
    repair_count INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_registry_status
ON document_registry(status);
"""


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            document_columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
            if "index_version" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN index_version INTEGER NOT NULL DEFAULT 1")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(records)")}
            migrations = {
                "filename_token_key": "TEXT NOT NULL DEFAULT '|'",
                "structure": "TEXT NOT NULL DEFAULT ''",
                "system": "TEXT NOT NULL DEFAULT ''",
                "region": "TEXT NOT NULL DEFAULT ''",
                "surface": "TEXT NOT NULL DEFAULT ''",
                "components_json": "TEXT NOT NULL DEFAULT '[]'",
                "elements_json": "TEXT NOT NULL DEFAULT '[]'",
                "repairs_json": "TEXT NOT NULL DEFAULT '[]'",
                "zones_json": "TEXT NOT NULL DEFAULT '[]'",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE records ADD COLUMN {name} {declaration}")
            element_columns = {row[1] for row in connection.execute("PRAGMA table_info(record_elements)")}
            if "role" not in element_columns:
                connection.execute("ALTER TABLE record_elements ADD COLUMN role TEXT NOT NULL DEFAULT ''")
            missing = connection.execute(
                "SELECT record_id, source_path FROM records WHERE filename_token_key = '|'"
            ).fetchall()
            connection.executemany(
                "UPDATE records SET filename_token_key = ? WHERE record_id = ?",
                [(_token_key(str(row["source_path"])), str(row["record_id"])) for row in missing],
            )
            lexical_version = connection.execute(
                "SELECT value FROM metadata WHERE key = 'lexical_index_version'"
            ).fetchone()
            if not lexical_version or lexical_version["value"] != "3":
                rows = connection.execute("SELECT record_id, search_text FROM records").fetchall()
                connection.execute("DELETE FROM records_fts")
                connection.executemany(
                    "INSERT INTO records_fts(record_id, search_text) VALUES (?, ?)",
                    [(row["record_id"], str(row["search_text"])) for row in rows],
                )
                connection.execute(
                    """INSERT INTO metadata(key, value) VALUES ('lexical_index_version', '3')
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value"""
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def document_state(self, source_path: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE source_path = ?", (source_path,)).fetchone()
        return dict(row) if row else None

    def documents(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY source_path").fetchall()
        return [dict(row) for row in rows]

    def replace_source_inventory(
        self,
        sources: list[dict[str, object]],
        config_signature: str,
        started_at: str,
        finished_at: str,
    ) -> None:
        canonical = len({str(item["source_name_key"]) for item in sources})
        with self.connect() as connection:
            connection.execute("DELETE FROM source_inventory")
            connection.executemany(
                """INSERT INTO source_inventory
                (source_path, root_key, physical_relative, source_name_key, size, mtime)
                VALUES (:source_path, :root_key, :physical_relative, :source_name_key, :size, :mtime)""",
                sources,
            )
            connection.execute(
                """INSERT INTO source_sync_state
                (singleton, config_signature, status, started_at, finished_at,
                 scanned, canonical, duplicates, error)
                VALUES (1, ?, 'completed', ?, ?, ?, ?, ?, '')
                ON CONFLICT(singleton) DO UPDATE SET
                config_signature=excluded.config_signature, status=excluded.status,
                started_at=excluded.started_at, finished_at=excluded.finished_at,
                scanned=excluded.scanned, canonical=excluded.canonical,
                duplicates=excluded.duplicates, error=''""",
                (
                    config_signature,
                    started_at,
                    finished_at,
                    len(sources),
                    canonical,
                    len(sources) - canonical,
                ),
            )

    def source_inventory(self, config_signature: str) -> list[dict[str, object]] | None:
        with self.connect() as connection:
            state = connection.execute(
                "SELECT config_signature, status FROM source_sync_state WHERE singleton = 1"
            ).fetchone()
            if not state or state["status"] != "completed" or state["config_signature"] != config_signature:
                return None
            rows = connection.execute(
                "SELECT * FROM source_inventory ORDER BY source_path COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    def source_sync_status(self, config_signature: str | None = None) -> dict[str, object]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM source_sync_state WHERE singleton = 1").fetchone()
        if not row:
            return {"status": "not_refreshed", "catalog_available": False, "scheduled": False}
        result = dict(row)
        result.pop("singleton", None)
        result["catalog_available"] = bool(
            result.get("status") == "completed"
            and (config_signature is None or result.get("config_signature") == config_signature)
        )
        result["scheduled"] = False
        return result

    def record_source_sync_failure(
        self, config_signature: str, started_at: str, finished_at: str, error: str
    ) -> None:
        # A failed refresh must not replace the last known-good inventory or its state.
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM source_sync_state WHERE singleton = 1"
            ).fetchone()
            if not existing:
                connection.execute(
                    """INSERT INTO source_sync_state
                    (singleton, config_signature, status, started_at, finished_at, error)
                    VALUES (1, ?, 'failed', ?, ?, ?)""",
                    (config_signature, started_at, finished_at, error),
                )

    def replace_aliases(self, document_id: str, aliases: list[dict[str, object]]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM document_aliases WHERE canonical_document_id = ?", (document_id,))
            connection.executemany(
                """INSERT INTO document_aliases
                (source_path, source_name_key, canonical_document_id, size, mtime)
                VALUES (:source_path, :source_name_key, :canonical_document_id, :size, :mtime)
                ON CONFLICT(source_path) DO UPDATE SET
                source_name_key=excluded.source_name_key,
                canonical_document_id=excluded.canonical_document_id,
                size=excluded.size,
                mtime=excluded.mtime""",
                aliases,
            )

    def aliases(self, document_id: str | None = None) -> list[dict[str, object]]:
        with self.connect() as connection:
            if document_id is None:
                rows = connection.execute("SELECT * FROM document_aliases ORDER BY source_path").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM document_aliases WHERE canonical_document_id = ? ORDER BY source_path",
                    (document_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def registry_items(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM document_registry ORDER BY source_path COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    def registry_item(self, source_path: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_registry WHERE source_path = ?", (source_path,)
            ).fetchone()
        return dict(row) if row else None

    def replace_registry(self, rows: list[dict[str, object]]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM document_registry")
            connection.executemany(
                """INSERT INTO document_registry (
                source_path, root_key, physical_relative, source_name_key, size, mtime,
                canonical_source_path, canonical_document_id, is_canonical, status, reason,
                markdown_path, repair_section_count, repair_count, content_hash, updated_at
                ) VALUES (
                :source_path, :root_key, :physical_relative, :source_name_key, :size, :mtime,
                :canonical_source_path, :canonical_document_id, :is_canonical, :status, :reason,
                :markdown_path, :repair_section_count, :repair_count, :content_hash, :updated_at
                )""",
                rows,
            )

    def update_registry(self, source_path: str, **values: object) -> None:
        allowed = {
            "status", "reason", "markdown_path", "repair_section_count", "repair_count",
            "content_hash", "updated_at",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE document_registry SET {assignments} WHERE source_path = ?",
                [*updates.values(), source_path],
            )

    def document_registry(
        self, status: str = "", limit: int = 100, offset: int = 0
    ) -> dict[str, object]:
        where = "WHERE status = ?" if status else ""
        parameters: list[object] = [status] if status else []
        with self.connect() as connection:
            total = int(connection.execute(
                f"SELECT count(*) FROM document_registry {where}", parameters
            ).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM document_registry {where} ORDER BY source_path COLLATE NOCASE LIMIT ? OFFSET ?",
                [*parameters, max(1, min(limit, 5000)), max(0, offset)],
            ).fetchall()
        return {"total": total, "limit": limit, "offset": offset, "items": [dict(row) for row in rows]}

    def document_registry_summary(self) -> dict[str, object]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT status, count(*) AS count FROM document_registry GROUP BY status ORDER BY status"
            ).fetchall()
            repairs = int(connection.execute(
                "SELECT coalesce(sum(repair_count), 0) FROM document_registry WHERE is_canonical = 1"
            ).fetchone()[0])
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return {"total": sum(counts.values()), "repairs": repairs, "statuses": counts}

    def replace_document(self, document: dict[str, object], records: list[dict[str, object]]) -> None:
        with self.connect() as connection:
            document_item = dict(document)
            document_item.setdefault("index_version", 1)
            previous = connection.execute(
                "SELECT document_id FROM documents WHERE source_path = ?", (document["source_path"],)
            ).fetchone()
            if previous:
                self._delete_records(connection, str(previous["document_id"]))
                connection.execute("DELETE FROM documents WHERE document_id = ?", (previous["document_id"],))
            connection.execute(
                """INSERT INTO documents
                (document_id, source_path, content_hash, size, mtime, status, error, indexed_at, index_version)
                VALUES (:document_id, :source_path, :content_hash, :size, :mtime, :status, :error, :indexed_at, :index_version)""",
                document_item,
            )
            for record in records:
                item = _record_defaults(record)
                connection.execute(
                    """INSERT INTO records (
                    record_id, document_id, record_type, section_id, section_heading,
                    heading_path_json, section_role, defect_type, repair_description,
                    frame_start, frame_end, stringer_start, stringer_end, component,
                    side, zone_text, evidence_text, search_text, extraction_status, source_path,
                    filename_token_key, structure, system, region, surface,
                    components_json, elements_json, repairs_json, zones_json
                    ) VALUES (
                    :record_id, :document_id, :record_type, :section_id, :section_heading,
                    :heading_path_json, :section_role, :defect_type, :repair_description,
                    :frame_start, :frame_end, :stringer_start, :stringer_end, :component,
                    :side, :zone_text, :evidence_text, :search_text, :extraction_status, :source_path,
                    :filename_token_key, :structure, :system, :region, :surface,
                    :components_json, :elements_json, :repairs_json, :zones_json
                    )""",
                    item,
                )
                connection.execute(
                    "INSERT INTO records_fts(record_id, search_text) VALUES (?, ?)",
                    (item["record_id"], str(item["search_text"])),
                )
                connection.executemany(
                    """INSERT INTO record_elements(record_id, ordinal, kind, start, end, qualifier, role)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            item["record_id"], ordinal, str(element["kind"]).strip().casefold(), element.get("start"),
                            element.get("end"), str(element.get("qualifier", "")).strip().casefold(),
                            str(element.get("role", "")).strip().casefold(),
                        )
                        for ordinal, element in enumerate(item["elements"], start=1)
                    ],
                )

    def delete_document(self, document_id: str) -> None:
        with self.connect() as connection:
            self._delete_records(connection, document_id)
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def migrate_raw_document(self, document_id: str, index_version: int) -> list[str]:
        with self.connect() as connection:
            ids = [
                str(row[0])
                for row in connection.execute(
                    "SELECT record_id FROM records WHERE document_id = ? AND record_type != 'raw_section'",
                    (document_id,),
                )
            ]
            if ids:
                connection.executemany("DELETE FROM records_fts WHERE record_id = ?", [(item,) for item in ids])
                connection.executemany("DELETE FROM records WHERE record_id = ?", [(item,) for item in ids])
            connection.execute(
                "UPDATE documents SET index_version = ?, indexed_at = ? WHERE document_id = ?",
                (index_version, _sqlite_now(), document_id),
            )
        return ids

    @staticmethod
    def _delete_records(connection: sqlite3.Connection, document_id: str) -> None:
        ids = [row[0] for row in connection.execute("SELECT record_id FROM records WHERE document_id = ?", (document_id,))]
        if ids:
            connection.executemany("DELETE FROM records_fts WHERE record_id = ?", [(item,) for item in ids])
        connection.execute("DELETE FROM records WHERE document_id = ?", (document_id,))

    def record(self, record_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
        return _decode_record(row) if row else None

    def records_for_document(self, document_id: str) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM records WHERE document_id = ?", (document_id,)).fetchall()
        return [_decode_record(row) for row in rows]

    def filename_token_groups(self) -> dict[tuple[str, ...], list[str]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT record_id, filename_token_key FROM records").fetchall()
        groups: dict[tuple[str, ...], list[str]] = {}
        for row in rows:
            tokens = tuple(token for token in str(row["filename_token_key"]).split("|") if token)
            groups.setdefault(tokens, []).append(str(row["record_id"]))
        return groups

    def lexical_search(
        self, query: str, limit: int, filename_tokens: tuple[str, ...] = (),
        section_roles: tuple[str, ...] = (), record_types: tuple[str, ...] = (),
    ) -> list[dict[str, object]]:
        expression = _fts_expression(query)
        if not expression:
            return []
        with self.connect() as connection:
            token_filters = (
                " AND (" + " OR ".join("instr(records.filename_token_key, ?) > 0" for _ in filename_tokens) + ")"
                if filename_tokens else ""
            )
            parameters: list[object] = [expression]
            parameters.extend(f"|{token.casefold()}|" for token in filename_tokens)
            role_filter = ""
            if section_roles:
                placeholders = ",".join("?" for _ in section_roles)
                role_filter = f" AND records.section_role IN ({placeholders})"
                parameters.extend(section_roles)
            type_filter = ""
            if record_types:
                placeholders = ",".join("?" for _ in record_types)
                type_filter = f" AND records.record_type IN ({placeholders})"
                parameters.extend(record_types)
            parameters.append(limit)
            rows = connection.execute(
                """SELECT records.*, bm25(records_fts) AS lexical_rank
                FROM records_fts JOIN records USING(record_id)
                WHERE records_fts MATCH ?""" + token_filters + role_filter + type_filter + " ORDER BY lexical_rank LIMIT ?",
                parameters,
            ).fetchall()
        return [_decode_record(row) for row in rows]

    def repair_document_records(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM records WHERE record_type = 'document_repairs'"
            ).fetchall()
        return [_decode_record(row) for row in rows]

    def structural_search(
        self,
        elements: list[dict[str, object]],
        filename_tokens: tuple[str, ...] = (),
        section_roles: tuple[str, ...] = ("chapter_5",),
    ) -> list[dict[str, object]]:
        requested = [item for item in elements if str(item.get("kind") or "").strip()]
        if not requested:
            return []
        clauses: list[str] = []
        parameters: list[object] = []
        for item in requested:
            kind = str(item["kind"]).strip().casefold()
            start = item.get("start")
            end = item.get("end")
            clause = """EXISTS (
                SELECT 1 FROM record_elements element
                WHERE element.record_id = records.record_id AND element.kind = ?"""
            values: list[object] = [kind]
            if start is not None and end is not None:
                clause += " AND element.start IS NOT NULL AND element.end IS NOT NULL AND element.start <= ? AND element.end >= ?"
                values.extend([int(end), int(start)])
            qualifier = str(item.get("qualifier") or "").strip().casefold()
            if qualifier:
                clause += " AND (element.qualifier = '' OR element.qualifier = ?)"
                values.append(qualifier)
            clause += ")"
            clauses.append(clause)
            parameters.extend(values)
        token_filters = (
            " AND (" + " OR ".join("instr(records.filename_token_key, ?) > 0" for _ in filename_tokens) + ")"
            if filename_tokens else ""
        )
        parameters.extend(f"|{token.casefold()}|" for token in filename_tokens)
        role_filter = ""
        if section_roles:
            placeholders = ",".join("?" for _ in section_roles)
            role_filter = f" AND records.section_role IN ({placeholders})"
            parameters.extend(section_roles)
        sql = "SELECT records.* FROM records WHERE " + " AND ".join(clauses) + token_filters + role_filter
        with self.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [_decode_record(row) for row in rows]

    def cached_extraction(self, section_hash: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT extraction_json, status, error FROM extraction_cache WHERE section_hash = ?", (section_hash,)
            ).fetchone()
        if not row:
            return None
        return {"payload": json.loads(row["extraction_json"]), "status": row["status"], "error": row["error"]}

    def cache_extraction(self, section_hash: str, payload: dict[str, object], status: str, error: str, now: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO extraction_cache(section_hash, extraction_json, status, error, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(section_hash) DO UPDATE SET extraction_json=excluded.extraction_json,
                status=excluded.status, error=excluded.error, updated_at=excluded.updated_at""",
                (section_hash, json.dumps(payload, ensure_ascii=False), status, error, now),
            )

    def upsert_job(self, job_id: str, status: str, payload: dict[str, object], started_at: str, finished_at: str | None) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO jobs(job_id, status, payload_json, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,
                payload_json=excluded.payload_json, finished_at=excluded.finished_at""",
                (job_id, status, json.dumps(payload, ensure_ascii=False), started_at, finished_at),
            )

    def latest_job(self) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs ORDER BY started_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        payload.update({"job_id": row["job_id"], "status": row["status"], "started_at": row["started_at"], "finished_at": row["finished_at"]})
        return payload

    def interrupt_running_jobs(self) -> None:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT job_id, payload_json FROM jobs WHERE status IN ('running', 'cancelling')"
            ).fetchall()
            for row in rows:
                payload = json.loads(str(row["payload_json"]))
                payload["phase"] = "interrupted"
                connection.execute(
                    "UPDATE jobs SET status = 'interrupted', payload_json = ?, finished_at = ? WHERE job_id = ?",
                    (json.dumps(payload, ensure_ascii=False), _sqlite_now(), row["job_id"]),
                )

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            documents = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            records = connection.execute("SELECT count(*) FROM records").fetchone()[0]
            repairs = connection.execute("SELECT count(*) FROM records WHERE record_type='document_repairs'").fetchone()[0]
            failed = connection.execute("SELECT count(*) FROM records WHERE extraction_status='extraction_failed'").fetchone()[0]
            elements = connection.execute("SELECT count(*) FROM record_elements").fetchone()[0]
            aliases = connection.execute("SELECT count(*) FROM document_aliases").fetchone()[0]
        return {
            "documents": documents,
            "records": records,
            "repair_records": repairs,
            "zone_elements": elements,
            "source_aliases": aliases,
            "extraction_failed": failed,
        }


def _fts_expression(query: str) -> str:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", query)
    return " OR ".join(f'"{token}"' for token in tokens[:20])


def _token_key(path: str) -> str:
    return "|" + "|".join(filename_tokens(path)) + "|"


def _decode_record(row: sqlite3.Row) -> dict[str, object]:
    item = dict(row)
    item["heading_path"] = json.loads(str(item.pop("heading_path_json")))
    item["filename_tokens"] = [token for token in str(item["filename_token_key"]).split("|") if token]
    item["components"] = json.loads(str(item.get("components_json") or "[]"))
    item["elements"] = json.loads(str(item.get("elements_json") or "[]"))
    item["repairs"] = json.loads(str(item.get("repairs_json") or "[]"))
    item["zones"] = json.loads(str(item.get("zones_json") or "[]"))
    return item


def _record_defaults(record: dict[str, object]) -> dict[str, object]:
    item = dict(record)
    for name in ("structure", "system", "region", "surface"):
        item.setdefault(name, "")
    components = list(item.get("components") or [])
    elements = list(item.get("elements") or [])
    repairs = list(item.get("repairs") or [])
    zones = list(item.get("zones") or [])
    item["components"] = components
    item["elements"] = elements
    item["components_json"] = json.dumps(components, ensure_ascii=False)
    item["elements_json"] = json.dumps(elements, ensure_ascii=False)
    item["repairs_json"] = json.dumps(repairs, ensure_ascii=False)
    item["zones_json"] = json.dumps(zones, ensure_ascii=False)
    return item


def _sqlite_now() -> str:
    return datetime.now(timezone.utc).isoformat()
