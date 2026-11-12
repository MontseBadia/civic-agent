import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx2
import openai

from agent.config import LLMConfig
from agent.llm import TOP_K_OFF, TOP_P_OFF, GenerationSettings, LLMClient

CONFIG = LLMConfig(base_url="http://h/v1", api_key="k", model="test-model")
SETTINGS = GenerationSettings(temperature=0.0, seed=7, max_tokens=256, top_p=TOP_P_OFF, top_k=TOP_K_OFF)
SAMPLING = GenerationSettings(temperature=1.0, seed=7, max_tokens=256, top_p=0.95, top_k=20)
MESSAGES = [{"role": "user", "content": "hi"}]


def fake_completion(content="ok", tool_calls=None, usage=True):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        model="served-model",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3) if usage else None,
    )


def client_returning(completion):
    fake = MagicMock()
    fake.chat.completions.create.return_value = completion
    return LLMClient(CONFIG, client=fake), fake.chat.completions.create


class RequestTest(unittest.TestCase):
    def test_sends_model_messages_and_settings(self):
        llm, create = client_returning(fake_completion())
        llm.chat(MESSAGES, SETTINGS)
        sent = create.call_args.kwargs
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["messages"], MESSAGES)
        self.assertEqual((sent["temperature"], sent["seed"], sent["max_tokens"]), (0.0, 7, 256))
        self.assertNotIn("tools", sent)
        self.assertNotIn("response_format", sent)

    def test_sends_top_p_and_top_k(self):
        llm, create = client_returning(fake_completion())
        llm.chat(MESSAGES, SAMPLING)
        sent = create.call_args.kwargs
        self.assertEqual(sent["top_p"], 0.95)
        self.assertEqual(sent["extra_body"], {"top_k": 20})

    def test_top_k_reaches_the_http_body_as_a_top_level_field(self):
        # The server ignores unknown or misplaced fields silently, so check the wire.
        sent = {}

        def handler(request):
            sent.update(json.loads(request.content))
            return httpx2.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
            })

        client = openai.OpenAI(base_url="http://h/v1", api_key="k", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        LLMClient(CONFIG, client=client).chat(MESSAGES, SAMPLING)
        self.assertEqual((sent["top_p"], sent["top_k"]), (0.95, 20))
        self.assertNotIn("extra_body", sent)

    def test_response_schema_becomes_json_schema_format(self):
        llm, create = client_returning(fake_completion())
        schema = {"type": "object"}
        llm.chat(MESSAGES, SETTINGS, response_schema=schema)
        self.assertEqual(
            create.call_args.kwargs["response_format"],
            {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        )

    def test_tools_are_passed_through(self):
        llm, create = client_returning(fake_completion())
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        llm.chat(MESSAGES, SETTINGS, tools=tools)
        self.assertEqual(create.call_args.kwargs["tools"], tools)

    def test_tools_and_schema_together_are_refused_before_calling(self):
        llm, create = client_returning(fake_completion())
        with self.assertRaises(ValueError):
            llm.chat(MESSAGES, SETTINGS, tools=[{"type": "function"}], response_schema={})
        create.assert_not_called()

    def test_real_client_has_no_retries_and_timeout(self):
        llm = LLMClient(CONFIG)
        self.assertEqual(llm._client.max_retries, 0)
        self.assertEqual(llm._client.timeout, 300.0)


class SettingsTest(unittest.TestCase):
    def test_all_values_required(self):
        with self.assertRaises(TypeError):
            GenerationSettings(temperature=0.0, seed=7, max_tokens=256)

    def test_invalid_values_refused(self):
        valid = dict(temperature=1.0, seed=7, max_tokens=256, top_p=0.95, top_k=20)
        for field, bad in [("temperature", -0.1), ("top_p", 0.0), ("top_p", 1.5), ("top_k", -1), ("max_tokens", 0)]:
            with self.subTest(field=field, value=bad), self.assertRaisesRegex(ValueError, field):
                GenerationSettings(**{**valid, field: bad})


class ResponseTest(unittest.TestCase):
    def test_fields_are_copied(self):
        llm, _ = client_returning(fake_completion(content="answer"))
        with patch("agent.llm.time.monotonic", side_effect=[10.0, 12.5]):
            r = llm.chat(MESSAGES, SETTINGS)
        self.assertEqual(r.content, "answer")
        self.assertEqual(r.tool_calls, ())
        self.assertEqual(r.finish_reason, "stop")
        self.assertEqual(r.model, "served-model")
        self.assertEqual((r.prompt_tokens, r.completion_tokens), (10, 3))
        self.assertEqual(r.duration_s, 2.5)

    def test_tool_call_arguments_stay_raw(self):
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_evidence", arguments="{not json"))
        llm, _ = client_returning(fake_completion(content=None, tool_calls=[call]))
        r = llm.chat(MESSAGES, SETTINGS)
        self.assertEqual(len(r.tool_calls), 1)
        self.assertEqual((r.tool_calls[0].name, r.tool_calls[0].arguments), ("get_evidence", "{not json"))

    def test_missing_usage_gives_none(self):
        llm, _ = client_returning(fake_completion(usage=False))
        r = llm.chat(MESSAGES, SETTINGS)
        self.assertIsNone(r.prompt_tokens)

    def test_server_errors_propagate(self):
        fake = MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("server down")
        with self.assertRaisesRegex(RuntimeError, "server down"):
            LLMClient(CONFIG, client=fake).chat(MESSAGES, SETTINGS)


if __name__ == "__main__":
    unittest.main()
