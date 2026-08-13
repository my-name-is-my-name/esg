from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from esg.chunking import split_text
from esg.clients import EmbeddingClient, ExternalReranker, OpenAIClient, SemanticExtractor
from esg.config import Settings
from esg.matching import ZoneMatch, best_zone_match
from esg.models import AnswerDecision, QueryExtraction, SearchSource, Zone
from esg.storage import SQLiteStore
from esg.vector_index import VectorIndex


NEGATIVE_ANSWER = "В проиндексированных документах подтверждающий ремонт не найден."


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        extractor: SemanticExtractor,
        embeddings: EmbeddingClient,
        vector_index: VectorIndex,
        llm: OpenAIClient,
        reranker: ExternalReranker,
    ) -> None:
        self.settings = settings
        self.store = store
        self.extractor = extractor
        self.embeddings = embeddings
        self.vector_index = vector_index
        self.llm = llm
        self.reranker = reranker

    def chat(
        self,
        question: str,
        progress: Callable[[str], None] | None = None,
        reasoning_progress: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        warnings: list[str] = []
        reasoning: list[str] = []

        def note(message: str) -> None:
            reasoning.append(message)
            if progress:
                progress(message)

        if self.settings.query_extraction_enabled:
            note("Извлекаю из запроса тип дефекта и координаты конструктивной зоны.")
            try:
                query = self.extractor.extract_query(question)
                note(f"Извлечена зона: {query_summary(query)}.")
                note("Выполняю лексический и векторный поиск по ESG-корпусу.")
            except Exception as exc:
                query = QueryExtraction()
                warnings.append(f"query_extraction_failed: {exc!r}")
                note("Структурированное извлечение запроса не выполнено; продолжаю поиск по исходному тексту запроса.")
        else:
            query = QueryExtraction()
            note("Использую исходный текст запроса без отдельного LLM-разбора координат.")

        if self.settings.search_filename_tokens:
            profile = ", ".join(token.upper() for token in self.settings.search_filename_tokens)
            note(f"Ограничиваю поиск документами профиля: {profile}.")
        retrieval_query = question
        lexical = self.store.lexical_search(
            retrieval_query,
            self.settings.retrieval_top_k,
            self.settings.search_filename_tokens,
            record_types=("document_repairs",),
        )
        vector_rows: list[tuple[str, float]] = []
        try:
            query_vector = self.embeddings.embed([retrieval_query])[0]
            vector_rows = self.vector_index.search(
                query_vector,
                self.settings.retrieval_top_k,
                self.settings.search_filename_tokens,
                record_types=("document_repairs",),
            )
        except Exception as exc:
            warnings.append(f"vector_search_unavailable: {exc!r}")
            note("Векторный поиск недоступен; продолжаю по лексическим кандидатам.")

        note(
            f"Найдено текстовых кандидатов: {len(lexical)}; "
            f"векторных: {len(vector_rows)}."
        )
        candidates = self._fuse(lexical, vector_rows)
        hybrid_ids = {str(row["record_id"]) for row in candidates}
        structured_matches: list[dict[str, object]] = []
        compatible: list[dict[str, object]] = []
        counts = {"MATCH": 0, "UNKNOWN": 0, "CONFLICT": 0}
        for row in self.store.repair_document_records():
            annotated = match_document_record(query.zone, row)
            counts[str(annotated["zone_status"])] += 1
            if annotated["zone_status"] == "MATCH":
                structured_matches.append(annotated)
            elif annotated["zone_status"] == "UNKNOWN" and str(row["record_id"]) in hybrid_ids:
                compatible.append(annotated)
        by_id = {str(row["record_id"]): row for row in [*structured_matches, *compatible]}
        for row in candidates:
            record_id = str(row["record_id"])
            annotated = by_id.get(record_id)
            if annotated:
                annotated["retrieval_score"] = row.get("retrieval_score", 0.0)
                annotated["vector_score"] = row.get("vector_score")
        candidates = list(by_id.values())
        note(
            "Проверена сохраненная структура зон: "
            f"совпадений {counts['MATCH']}, неполных {counts['UNKNOWN']}, "
            f"исключено конфликтов {counts['CONFLICT']}."
        )
        intersections = list(dict.fromkeys(
            detail
            for row in structured_matches
            for detail in list(row.get("zone_details") or [])
        ))
        if intersections:
            note("Подтвержденные пересечения: " + "; ".join(intersections[:8]) + ".")
        if self.reranker.enabled and candidates:
            note("Переранжирую кандидатов локальной cross-encoder моделью.")
        candidates = self._rerank(question, candidates, warnings)
        candidates = self._score(candidates)
        before_deduplication = len(candidates)
        candidates = deduplicate_evidence(candidates)
        removed = before_deduplication - len(candidates)
        if removed:
            note(f"Удалены дубликаты одинаковых фрагментов: {removed}.")
        candidates = candidates[: self.settings.final_top_k]
        note(f"Отобрано источников для проверки ответа: {len(candidates)}.")
        sources = [source_from_record(row) for row in candidates]
        note("Формирую ответ только по подтверждающим фрагментам документов.")
        answer, supporting, found = self._answer(
            question, query, sources, warnings, reasoning_progress
        )
        if self.settings.show_retrieved_chunks:
            note(f"Тестовый режим: показываю отобранные retrieval-чанки: {len(sources)}.")
        elif not found:
            sources = []
        elif supporting:
            selected_sources = [sources[index - 1] for index in supporting if 1 <= index <= len(sources)]
            if selected_sources:
                sources = selected_sources
        return {
            "answer": answer,
            "sources": [source.model_dump() for source in sources],
            "warnings": warnings,
            "query": query.model_dump(),
            "reasoning": reasoning,
        }

    def _fuse(
        self,
        lexical: list[dict[str, object]],
        vector_rows: list[tuple[str, float]],
    ) -> list[dict[str, object]]:
        fused: dict[str, dict[str, object]] = {}
        scores: dict[str, float] = {}
        for rank, row in enumerate(lexical, start=1):
            record_id = str(row["record_id"])
            fused[record_id] = row
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (self.settings.rrf_k + rank)
        for rank, (record_id, vector_score) in enumerate(vector_rows, start=1):
            row = fused.get(record_id) or self.store.record(record_id)
            if not row:
                continue
            fused[record_id] = row
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (self.settings.rrf_k + rank)
            row["vector_score"] = vector_score
        result = []
        for record_id, row in fused.items():
            item = dict(row)
            item["retrieval_score"] = scores.get(record_id, 0.0)
            result.append(item)
        result.sort(key=lambda item: float(item["retrieval_score"]), reverse=True)
        return result[: self.settings.retrieval_top_k]

    def _extract_candidate_zones(
        self,
        candidates: list[dict[str, object]],
    ) -> tuple[list[list[Zone]], list[bool], int, int, list[str]]:
        workers = max(1, self.settings.chunk_extraction_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._zones_for_candidate, row) for row in candidates]
            results = [future.result() for future in futures]
        return (
            [result[0] for result in results],
            [bool(result[3]) for result in results],
            sum(result[1] for result in results),
            sum(result[2] for result in results),
            [error for result in results for error in result[3]],
        )

    def _zones_for_candidate(
        self,
        row: dict[str, object],
    ) -> tuple[list[Zone], int, int, list[str]]:
        text = str(row.get("evidence_text") or row.get("search_text") or "")
        heading_path = [str(value) for value in list(row.get("heading_path") or [])]
        parts = split_text(text, max_chars=max(1, self.settings.chunk_extraction_max_chars))
        zones: list[Zone] = []
        cache_hits = 0
        model_calls = 0
        errors: list[str] = []
        for part in parts:
            cache_key = hashlib.sha256(
                ("retrieved-zone-v6\0" + "\0".join(heading_path) + "\0" + part).encode("utf-8")
            ).hexdigest()
            cached = self.store.cached_extraction(cache_key)
            part_zones: list[Zone] | None = None
            if cached and cached.get("status") == "ok":
                try:
                    payload = cached.get("payload") or {}
                    values = payload.get("zones") if isinstance(payload, dict) else []
                    part_zones = [Zone.model_validate(value) for value in values or []]
                    cache_hits += 1
                except Exception:
                    part_zones = None
            if part_zones is None:
                try:
                    model_calls += 1
                    part_zones = self.extractor.extract_chunk_zones(heading_path, part)
                    self.store.cache_extraction(
                        cache_key,
                        {"zones": [zone.model_dump() for zone in part_zones]},
                        "ok",
                        "",
                        datetime.now(timezone.utc).isoformat(),
                    )
                except Exception as exc:
                    errors.append(f"chunk_extraction_failed:{row.get('record_id')}: {exc!r}")
                    continue
            zones.extend(part_zones)

        unique: dict[str, Zone] = {}
        for zone in zones:
            key = json.dumps(zone.model_dump(), ensure_ascii=False, sort_keys=True)
            unique.setdefault(key, zone)
        return list(unique.values()), cache_hits, model_calls, errors

    def _rerank(
        self,
        question: str,
        candidates: list[dict[str, object]],
        warnings: list[str],
    ) -> list[dict[str, object]]:
        if not candidates or not self.reranker.enabled:
            return candidates
        try:
            scores = self.reranker.rerank(question, [str(row["search_text"]) for row in candidates])
            for row, score in zip(candidates, scores, strict=True):
                row["rerank_score"] = score
            return sorted(candidates, key=lambda row: float(row["rerank_score"]), reverse=True)
        except Exception as exc:
            warnings.append(f"reranker_unavailable: {exc!r}")
            return candidates

    def _score(self, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        result = []
        for row in candidates:
            item = dict(row)
            base = float(item.get("rerank_score", item.get("retrieval_score", 0.0)))
            item["final_score"] = base
            result.append(item)
        tier = {"MATCH": 0, "UNKNOWN": 1}
        return sorted(
            result,
            key=lambda row: (tier.get(str(row.get("zone_status")), 2), -float(row["final_score"])),
        )

    def _answer(
        self,
        question: str,
        query: QueryExtraction,
        sources: list[SearchSource],
        warnings: list[str],
        reasoning_progress: Callable[[str], None] | None = None,
    ) -> tuple[str, list[int], bool]:
        if not sources:
            return NEGATIVE_ANSWER, [], False
        if not self.llm.enabled:
            found = any(
                source.zone_status == "MATCH" and (source.repair_description or source.defect_type)
                for source in sources
            )
            if found:
                return self._fallback_found(sources), [1], True
            return NEGATIVE_ANSWER, [], False
        system = (
            "Ты проверяешь, описан ли ремонт в указанной пользователем зоне конструкции. "
            "Используй только переданные источники. Ремонт считается найденным, если отчет описывает ремонт "
            "в совместимой зоне; отдельная фраза о фактическом выполнении не требуется. "
            "Оглавление, общая методика, инспекция без ремонта и совпадение только соседней зоны не являются подтверждением. "
            "Если подтверждения нет, answer должен быть ровно: "
            "'В проиндексированных документах подтверждающий ремонт не найден.' "
            "При подтверждении начни answer с: "
            "'Да, в проиндексированных документах найден описанный ремонт.' "
            "Верни JSON и укажи номера только действительно подтверждающих источников."
        )
        payload = {
            "question": question,
            "structured_query": query.model_dump(),
            "sources": [
                {
                    "index": index,
                    "document": source.source_path,
                    "section": source.section_heading,
                    "defect_type": source.defect_type,
                    "repair_description": source.repair_description,
                    "zone": source.zone.model_dump(),
                    "zone_status": source.zone_status,
                    "zone_details": source.zone_details,
                    "evidence": source.evidence_text[:5000],
                    "extraction_status": source.extraction_status,
                }
                for index, source in enumerate(sources, start=1)
            ],
            "schema": AnswerDecision.model_json_schema(),
        }
        try:
            request = json.dumps(payload, ensure_ascii=False)
            if reasoning_progress:
                response = self.llm.json_completion_stream(system, request, reasoning_progress)
            else:
                response = self.llm.json_completion(system, request)
            decision = AnswerDecision.model_validate(response)
            if not decision.found:
                return NEGATIVE_ANSWER, [], False
            return decision.answer, decision.supporting_source_indexes, True
        except Exception as exc:
            warnings.append(f"answer_generation_failed: {exc!r}")
            found = any(
                source.zone_status == "MATCH" and (source.repair_description or source.defect_type)
                for source in sources
            )
            if found:
                return self._fallback_found(sources), [1], True
            return NEGATIVE_ANSWER, [], False

    @staticmethod
    def _fallback_found(sources: list[SearchSource]) -> str:
        source = sources[0]
        detail = source.repair_description or source.evidence_text[:300]
        return (
            "Да, в проиндексированных документах найден описанный ремонт.\n\n"
            f"Документ: {source.source_path}\nРаздел: {source.section_heading}\nФрагмент: {detail}"
        )


def source_from_record(row: dict[str, object]) -> SearchSource:
    return SearchSource(
        document_id=str(row["document_id"]),
        source_path=str(row["source_path"]),
        section_heading=str(row["section_heading"]),
        heading_path=list(row.get("heading_path") or []),
        evidence_text=str(row["evidence_text"]),
        defect_type=str(row.get("defect_type") or ""),
        repair_description=str(row.get("repair_description") or ""),
        zone=zone_from_record(row),
        zone_status=str(row.get("zone_status") or "UNKNOWN"),
        zone_details=list(row.get("zone_details") or []),
        retrieval_score=float(row.get("retrieval_score") or 0.0),
        rerank_score=float(row["rerank_score"]) if row.get("rerank_score") is not None else None,
        final_score=float(row.get("final_score") or 0.0),
        extraction_status=str(row.get("extraction_status") or "ok"),
    )


def match_document_record(query: Zone, row: dict[str, object]) -> dict[str, object]:
    item = dict(row)
    choices: list[tuple[ZoneMatch, Zone, dict[str, object]]] = []
    for repair in list(row.get("repairs") or []):
        if not isinstance(repair, dict):
            continue
        zones = [Zone.model_validate(zone) for zone in list(repair.get("zones") or [])]
        match = best_zone_match(query, zones)
        if match.zone_index >= 0:
            choices.append((match, zones[match.zone_index], repair))
        elif zones:
            choices.append((match, zones[0], repair))
    if not choices:
        zones = [Zone.model_validate(zone) for zone in list(row.get("zones") or [])]
        match = best_zone_match(query, zones)
        zone = zones[match.zone_index] if match.zone_index >= 0 else (zones[0] if zones else Zone())
        choices.append((match, zone, {}))
    rank = {"MATCH": 0, "UNKNOWN": 1, "CONFLICT": 2}
    match, zone, repair = min(choices, key=lambda value: rank[value[0].status])
    item["zone_status"] = match.status
    item["zone_details"] = match.details
    item["matched_zone"] = zone.model_dump()
    if repair:
        item["evidence_text"] = str(repair.get("evidence_text") or item.get("evidence_text") or "")
        item["repair_description"] = str(repair.get("repair_id") or "")
        item["defect_type"] = str(repair.get("defect_type") or "")
        item["section_heading"] = str(repair.get("section_heading") or item.get("section_heading") or "")
    return item


def zone_from_record(row: dict[str, object]) -> Zone:
    matched = row.get("matched_zone")
    if isinstance(matched, dict):
        return Zone.model_validate(matched)
    return Zone(
        elements=list(row.get("elements") or []),
        components=list(row.get("components") or []),
        structure=str(row.get("structure") or ""),
        system=str(row.get("system") or ""),
        region=str(row.get("region") or ""),
        side=str(row.get("side") or ""),
        surface=str(row.get("surface") or ""),
        zone_text=str(row.get("zone_text") or ""),
    )


_EMPTY_ANCHOR_RE = re.compile(r"<a\s+id=[\"'][^\"']+[\"']\s*></a>", re.IGNORECASE)


def normalized_evidence(text: str) -> str:
    without_anchors = _EMPTY_ANCHOR_RE.sub(" ", text)
    return re.sub(r"\s+", " ", without_anchors).strip()


def deduplicate_evidence(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for candidate in candidates:
        evidence = normalized_evidence(str(candidate.get("evidence_text") or ""))
        key = evidence or f"{candidate.get('source_path')}|{candidate.get('section_id')}"
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def query_summary(query: QueryExtraction) -> str:
    values: list[str] = []
    if query.defect_type:
        values.append(f"дефект={query.defect_type}")
    for label, value in (
        ("конструкция", query.zone.structure),
        ("система", query.zone.system),
        ("область", query.zone.region),
        ("сторона", query.zone.side),
        ("поверхность", query.zone.surface),
    ):
        if value:
            values.append(f"{label}={value}")
    if query.zone.components:
        values.append("компоненты=" + ",".join(query.zone.components))
    labels = {
        "frame": "шпангоуты",
        "stringer": "стрингеры",
        "rib": "нервюры",
        "flap": "закрылки",
        "slat": "предкрылки",
        "spoiler": "спойлеры",
    }
    for element in query.zone.elements:
        label = labels.get(element.kind, element.kind)
        if element.start is None or element.end is None:
            interval = "без номера"
        else:
            interval = str(element.start) if element.start == element.end else f"{element.start}-{element.end}"
        qualifier = f" {element.qualifier.upper()}" if element.qualifier else ""
        role = f" ({element.role})" if element.role else ""
        values.append(f"{label}={interval}{qualifier}{role}")
    return "; ".join(values) if values else "структурные параметры не указаны"
