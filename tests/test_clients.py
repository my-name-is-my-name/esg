from __future__ import annotations

import unittest
from unittest import mock

from esg.clients import EmbeddingClient, ExternalReranker, OpenAIClient, SemanticExtractor


class Response:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self.payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self) -> dict[str, object]:
        return self.payload


class OpenAIClientTests(unittest.TestCase):
    def test_embeddings_support_openai_compatible_endpoint(self) -> None:
        settings = mock.Mock(
            ollama_url="http://embeddings/v1",
            embedding_api="openai",
            embedding_model="bge-m3",
            embedding_batch_size=16,
        )
        response = Response(200, {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        })
        with mock.patch("esg.clients.requests.post", return_value=response) as post:
            vectors = EmbeddingClient(settings).embed(["one", "two"])

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(post.call_args.args[0], "http://embeddings/v1/embeddings")
        self.assertEqual(post.call_args.kwargs["json"], {"model": "bge-m3", "input": ["one", "two"]})

    def test_json_mode_retries_without_response_format_on_400(self) -> None:
        settings = mock.Mock(
            llm_base_url="http://llm/v1",
            llm_api_key="local",
            llm_model="model",
            llm_timeout_seconds=10,
            llm_max_tokens=4000,
        )
        client = OpenAIClient(settings)
        responses = [
            Response(400, {}),
            Response(200, {"choices": [{"message": {"content": '{"value": 1}'}}]}),
        ]
        with mock.patch("esg.clients.requests.post", side_effect=responses) as post:
            payload = client.json_completion("system", "user")
        self.assertEqual(payload, {"value": 1})
        self.assertIn("response_format", post.call_args_list[0].kwargs["json"])
        self.assertNotIn("response_format", post.call_args_list[1].kwargs["json"])

    def test_reranker_uses_running_server_pairs_contract(self) -> None:
        settings = mock.Mock(reranker_url="http://reranker:9101", reranker_timeout_seconds=10, reranker_batch_size=8)
        response = Response(200, {"scores": [0.9, 0.2]})
        with mock.patch("esg.clients.requests.post", return_value=response) as post:
            scores = ExternalReranker(settings).rerank("query", ["one", "two"])
        self.assertEqual(scores, [0.9, 0.2])
        self.assertEqual(post.call_args.kwargs["json"], {"pairs": [["query", "one"], ["query", "two"]]})

    def test_semantic_extractor_maps_generic_zone_contract(self) -> None:
        client = mock.Mock()
        client.completion.return_value = (
            '{"defect":"трещина","zone":{"elements":['
            '{"kind":"frame","start":34,"end":34,"qualifier":"","role":"reference"},'
            '{"kind":"stringer","start":24,"end":28,"qualifier":"","role":"boundary"},'
            '{"kind":"rib","start":7,"end":7,"qualifier":"RH","role":"target"}],'
            '"components":["rib"],"structure":"fuselage","system":"",'
            '"region":"","side":"left","surface":""}}'
        )
        result = SemanticExtractor(client).extract_query("трещина у шпангоута 34")
        self.assertEqual((result.zone.frames.start, result.zone.frames.end), (34, 34))
        self.assertEqual((result.zone.stringers.start, result.zone.stringers.end), (24, 28))
        self.assertEqual((result.zone.ribs.start, result.zone.ribs.end), (7, 7))
        self.assertEqual(result.zone.structure, "fuselage")
        self.assertEqual(result.zone.side, "left")
        self.assertEqual(result.zone.element("rib").role, "target")
        self.assertEqual(
            client.completion.call_args.kwargs["extra_payload"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(
            client.completion.call_args.kwargs["response_format"],
            {"type": "json_object"},
        )

    def test_semantic_extractor_returns_independent_chunk_zones(self) -> None:
        client = mock.Mock()
        client.completion.return_value = (
            '{"zones":['
            '{"elements":[{"kind":"rib","start":7,"end":7,"qualifier":"","role":"target"}],'
            '"components":["rib"],"structure":"wing","system":"","region":"","side":"right","surface":""},'
            '{"elements":[{"kind":"spoiler","start":3,"end":3,"qualifier":"","role":"target"}],'
            '"components":["spoiler"],"structure":"wing","system":"","region":"","side":"left","surface":"upper"}'
            ']}'
        )

        zones = SemanticExtractor(client).extract_chunk_zones(["5", "Описание ремонта"], "текст")

        self.assertEqual(len(zones), 2)
        self.assertEqual(zones[0].element("rib").role, "target")
        self.assertEqual(zones[1].element("spoiler").start, 3)
        self.assertEqual(
            client.completion.call_args.kwargs["extra_payload"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_semantic_extractor_normalizes_string_interval(self) -> None:
        client = mock.Mock()
        client.completion.return_value = (
            '{"zones":[{"elements":[{"kind":"stringer","start":"2 – 10","end":null,'
            '"qualifier":"","role":"boundary"}],"components":["skin"]}]}'
        )

        zone = SemanticExtractor(client).extract_chunk_zones([], "text")[0]

        self.assertEqual((zone.element("stringer").start, zone.element("stringer").end), (2, 10))

if __name__ == "__main__":
    unittest.main()
