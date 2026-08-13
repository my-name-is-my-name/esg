from __future__ import annotations

from esg.config import Settings


class VectorIndex:
    def __init__(self, settings: Settings) -> None:
        from qdrant_client import QdrantClient

        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, timeout=60)

    def ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        names = {item.name for item in self.client.get_collections().collections}
        if self.settings.qdrant_collection not in names:
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        for field in ("document_id", "record_type", "section_role", "extraction_status", "filename_tokens"):
            try:
                self.client.create_payload_index(
                    collection_name=self.settings.qdrant_collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception:
                pass

    def replace_document(self, document_id: str, rows: list[tuple[dict[str, object], list[float]]]) -> None:
        from qdrant_client.models import PointStruct

        self.delete_document(document_id)
        if not rows:
            return
        self.ensure_collection(len(rows[0][1]))
        points = [
            PointStruct(
                id=str(record["record_id"]),
                vector=vector,
                payload={
                    "record_id": record["record_id"],
                    "document_id": record["document_id"],
                    "record_type": record["record_type"],
                    "section_role": record["section_role"],
                    "extraction_status": record["extraction_status"],
                    "filename_tokens": record["filename_tokens"],
                },
            )
            for record, vector in rows
        ]
        self.client.upsert(collection_name=self.settings.qdrant_collection, points=points, wait=True)

    def delete_document(self, document_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        names = {item.name for item in self.client.get_collections().collections}
        if self.settings.qdrant_collection not in names:
            return
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
            wait=True,
        )

    def delete_records(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=PointIdsList(points=record_ids),
            wait=True,
        )

    def sync_filename_tokens(self, groups: dict[tuple[str, ...], list[str]]) -> None:
        names = {item.name for item in self.client.get_collections().collections}
        if self.settings.qdrant_collection not in names:
            return
        for tokens, record_ids in groups.items():
            if not record_ids:
                continue
            self.client.set_payload(
                collection_name=self.settings.qdrant_collection,
                payload={"filename_tokens": list(tokens)},
                points=record_ids,
                wait=True,
            )

    def search(
        self, vector: list[float], limit: int, filename_tokens: tuple[str, ...] = (),
        section_roles: tuple[str, ...] = (), record_types: tuple[str, ...] = (),
    ) -> list[tuple[str, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        names = {item.name for item in self.client.get_collections().collections}
        if self.settings.qdrant_collection not in names:
            return []
        query_filter = None
        if filename_tokens or section_roles or record_types:
            should = [
                FieldCondition(key="filename_tokens", match=MatchValue(value=token.casefold()))
                for token in filename_tokens
            ]
            must = []
            if section_roles:
                must.append(FieldCondition(key="section_role", match=MatchAny(any=list(section_roles))))
            if record_types:
                must.append(FieldCondition(key="record_type", match=MatchAny(any=list(record_types))))
            query_filter = Filter(must=must, should=should)
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            )
            points = result.points
        else:
            points = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            )
        return [(str(point.payload.get("record_id") or point.id), float(point.score)) for point in points]

    def health(self) -> dict[str, object]:
        try:
            names = {item.name for item in self.client.get_collections().collections}
            exists = self.settings.qdrant_collection in names
            count = 0
            if exists:
                count = int(self.client.count(self.settings.qdrant_collection, exact=True).count)
            return {"ok": True, "collection": self.settings.qdrant_collection, "exists": exists, "points": count}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}
