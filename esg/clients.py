from __future__ import annotations

import json
import re
from typing import Any

import requests

from esg.config import Settings
from esg.models import QueryExtraction, Zone, ZoneElement


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._resolved_model = settings.llm_model

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_base_url)

    def model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        response = requests.get(
            f"{self.settings.llm_base_url.rstrip('/')}/models",
            headers=self._headers(),
            timeout=self.settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models:
            raise RuntimeError("LLM endpoint returned no models")
        self._resolved_model = str(models[0]["id"])
        return self._resolved_model

    def json_completion(self, system: str, user: str) -> dict[str, Any]:
        content = self.completion(system, user, response_format={"type": "json_object"})
        return _json_object(content)

    def completion(
        self,
        system: str,
        user: str,
        response_format: dict[str, str] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
        if extra_payload:
            payload.update(extra_payload)
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        response = requests.post(url, headers=self._headers(), json=dict(payload), timeout=self.settings.llm_timeout_seconds)
        if response.status_code == 400 and response_format:
            payload.pop("response_format", None)
            response = requests.post(url, headers=self._headers(), json=dict(payload), timeout=self.settings.llm_timeout_seconds)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"].get("content") or "").strip()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.llm_api_key}", "Content-Type": "application/json"}


class SemanticExtractor:
    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def extract_query(self, question: str) -> QueryExtraction:
        system = (
            "Ответь одной строкой JSON без markdown. Ключи: defect и zone. В zone ровно ключи: "
            "elements,components,structure,system,region,side,surface. elements — массив объектов "
            "{kind,start,end,qualifier,role}. kind — нормализованный тип на английском в единственном числе: "
            "frame, stringer, rib, flap, slat, spoiler или другой явно названный тип. "
            "Номер или диапазон записывай в start/end включительно; для ненумерованного элемента оба null. "
            "qualifier содержит LH/RH только когда он относится именно к элементу. role: target, если поврежден "
            "сам элемент; boundary, если элемент задает границы; reference, если дефект находится у элемента. "
            "components — массив поврежденных компонентов на английском в единственном числе. "
            "Например, skin, frame, rib, panel, spar, flap, slat, spoiler. structure: wing, fuselage или landing_gear. "
            "Нормализация русских терминов: обшивка=skin, шпангоут=frame, стрингер=stringer, нервюра=rib, "
            "закрылок=flap, предкрылок=slat, спойлер=spoiler, фланец=flange. Фланец никогда не является flap. "
            "system: NLG/MLG только явно. side и surface заполняй только явно. Извлекай каждый упомянутый "
            "элемент и каждый диапазон. Не выводи structure из компонента: обшивка бывает у крыла и фюзеляжа. Не додумывай. "
            "Координатные элементы не являются поврежденными компонентами: 'обшивка между нервюрами 1-15 и "
            "стрингерами 2-10' означает components=[skin], rib/stringer role=boundary. "
            "'трещина на нервюре 7' означает components=[rib], rib role=target. "
            "'трещина обшивки у шпангоута 34' означает components=[skin], frame role=reference. "
            "Для 'трещина на самой нервюре 7 между стрингерами 4 и 8 правого крыла': defect=crack, "
            "components=[rib], elements=[rib 7 target, stringer 4-8 boundary], structure=wing, side=right. "
            "Для 'трещина обшивки у шпангоута 34 между стрингерами 24 и 28' без названия конструкции: "
            "defect=crack, components=[skin], elements=[frame 34 reference, stringer 24-28 boundary], structure=''. "
            "Пример для нервюр 1-15, стрингеров 2-10 и спойлера 3: "
            "{\"defect\":\"\",\"zone\":{\"elements\":[{\"kind\":\"rib\",\"start\":1,\"end\":15,\"qualifier\":\"\",\"role\":\"boundary\"},{\"kind\":\"stringer\",\"start\":2,\"end\":10,\"qualifier\":\"\",\"role\":\"boundary\"},{\"kind\":\"spoiler\",\"start\":3,\"end\":3,\"qualifier\":\"\",\"role\":\"target\"}],\"components\":[\"skin\",\"spoiler\"],\"structure\":\"wing\",\"system\":\"\",\"region\":\"console\",\"side\":\"right\",\"surface\":\"upper\"}}."
        )
        content = self.client.completion(
            system,
            question,
            response_format={"type": "json_object"},
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
        payload = _json_object(content)
        return QueryExtraction(
            defect_type=str(payload.get("defect") or ""),
            zone=_zone_payload(payload.get("zone"), question),
        )

    def extract_chunk_zones(self, heading_path: list[str], text: str) -> list[Zone]:
        system = (
            "Верни только JSON {\"zones\":[...]}. Найди все явно описанные зоны ремонта, дефекта или расчета. "
            "Каждое самостоятельное утверждение о расположении дает одну zone; все координаты одного утверждения "
            "остаются в этой zone. Общую зону ремонта и следующую узкую расчетную точку верни отдельно. "
            "Оглавление и методика без зоны дают пустой массив. "
            "Zone содержит elements,components,structure,system,region,side,surface. Element содержит "
            "kind,start,end,qualifier,role. Диапазон включительный. Роль target означает дефект самого элемента, "
            "boundary означает границу 'между', reference означает расположение 'у'. "
            "Строгий словарь: обшивка=skin; шпангоут=frame; стрингер=stringer; нервюра=rib; закрылок=flap; "
            "предкрылок=slat; спойлер=spoiler; фланец=flange. Никогда не заменяй rib на frame или flange на flap. "
            "Если ремонт/дефект расположен на обшивке, components содержит skin. Если повреждена сама нервюра, "
            "components содержит rib и element rib имеет role=target. У обшивки возле шпангоута components=[skin], "
            "а frame имеет role=reference. У обшивки между нервюрами и стрингерами components=[skin], а оба "
            "диапазона имеют role=boundary. structure: wing, fuselage или landing_gear только когда явно сказано; "
            "system: NLG/MLG только когда явно сказано. side: left/right, surface: upper/lower. "
            "Число крепежа, рисунка, листа, позиции или Item не является номером конструктивного элемента. Не додумывай."
        )
        user = json.dumps({"heading_path": heading_path, "text": text}, ensure_ascii=False)
        content = self.client.completion(
            system,
            user,
            response_format={"type": "json_object"},
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
        payload = _json_object(content)
        values = payload.get("zones")
        if not isinstance(values, list):
            raise ValueError("LLM chunk extraction must contain a zones array")
        return [_zone_payload(value, "") for value in values if isinstance(value, dict)]

class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output: list[list[float]] = []
        batch_size = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(texts), batch_size):
            output.extend(self._embed_batch(texts[start : start + batch_size]))
        return output

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        base = self.settings.ollama_url.rstrip("/")
        if getattr(self.settings, "embedding_api", "ollama") == "openai":
            response = requests.post(
                f"{base}/embeddings",
                json={"model": self.settings.embedding_model, "input": texts},
                timeout=180,
            )
            response.raise_for_status()
            data = response.json().get("data") or []
            if len(data) != len(texts):
                raise RuntimeError("OpenAI-compatible endpoint returned an unexpected number of embeddings")
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            return [[float(value) for value in item["embedding"]] for item in ordered]
        response = requests.post(
            f"{base}/api/embed",
            json={"model": self.settings.embedding_model, "input": texts},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings:
            return [[float(value) for value in row] for row in embeddings]
        if len(texts) == 1 and data.get("embedding"):
            return [[float(value) for value in data["embedding"]]]
        raise RuntimeError("Ollama returned no embeddings")


class ExternalReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.reranker_url)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [[query, document] for document in documents]
        scores: list[float] = []
        batch_size = max(1, self.settings.reranker_batch_size)
        for start in range(0, len(pairs), batch_size):
            scores.extend(self._rerank_batch(pairs[start : start + batch_size]))
        return scores

    def _rerank_batch(self, pairs: list[list[str]]) -> list[float]:
        response = requests.post(
            self.settings.reranker_url.rstrip("/") + "/rerank",
            json={"pairs": pairs},
            timeout=self.settings.reranker_timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.RequestException:
            if len(pairs) <= 1:
                raise
            midpoint = len(pairs) // 2
            return self._rerank_batch(pairs[:midpoint]) + self._rerank_batch(pairs[midpoint:])
        data = response.json()
        if isinstance(data.get("scores"), list):
            return [float(value) for value in data["scores"]]
        results = data.get("results") or []
        scores = [0.0] * len(pairs)
        for item in results:
            scores[int(item["index"])] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores


def _json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response contains no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def _zone_payload(value: object, zone_text: str) -> Zone:
    payload = value if isinstance(value, dict) else {}
    raw_elements = payload.get("elements")
    elements: list[ZoneElement] = []
    if isinstance(raw_elements, list):
        for item in raw_elements:
            if not isinstance(item, dict) or not str(item.get("kind") or "").strip():
                continue
            start, end = _element_bounds(item.get("start"), item.get("end"))
            elements.append(
                ZoneElement(
                    kind=str(item["kind"]),
                    start=start,
                    end=end,
                    qualifier=str(item.get("qualifier") or ""),
                    role=str(item.get("role") or ""),
                )
            )
    components = payload.get("components")
    return Zone(
        elements=elements,
        components=[str(item) for item in components] if isinstance(components, list) else [],
        structure=str(payload.get("structure") or ""),
        system=str(payload.get("system") or ""),
        region=str(payload.get("region") or ""),
        side=str(payload.get("side") or ""),
        surface=str(payload.get("surface") or ""),
        zone_text=zone_text,
    )


def _element_bounds(start: object, end: object) -> tuple[int | None, int | None]:
    start_values = re.findall(r"\d+", str(start)) if start not in (None, "") else []
    end_values = re.findall(r"\d+", str(end)) if end not in (None, "") else []
    if not start_values and not end_values:
        return None, None
    first = int(start_values[0] if start_values else end_values[0])
    last = int(end_values[-1] if end_values else start_values[-1])
    return first, last
