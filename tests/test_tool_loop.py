import json
import unittest

from agent.answer import ANSWER_SCHEMA
from agent.llm import GenerationSettings, LLMResponse, ToolCall
from agent.prompts import C_FINAL_INSTRUCTION
from agent.tool_loop import (
    MAX_RESULT_CHARS,
    InvalidArguments,
    Tool,
    ToolError,
    ToolStep,
    run_tool_loop,
    validate_arguments,
)

SETTINGS = GenerationSettings(temperature=0.0, seed=7, max_tokens=256)
MESSAGES = [{"role": "user", "content": "question"}]
FINAL_JSON = '{"final": true}'


def response(content=None, calls=()):
    tool_calls = tuple(ToolCall(id=i, name=n, arguments=a) for i, n, a in calls)
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        model="m",
        prompt_tokens=1,
        completion_tokens=1,
        duration_s=0.1,
    )


class ScriptedLLM:
    """Returns the scripted responses in order and records every call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, settings, tools=None, response_schema=None):
        self.calls.append({
            "messages": list(messages),
            "settings": settings,
            "tools": tools,
            "response_schema": response_schema,
        })
        return self.responses.pop(0)


def echo_tool(calls_log=None):
    def run(text: str):
        if calls_log is not None:
            calls_log.append(text)
        return {"echo": text}

    return Tool(
        name="echo",
        description="Echo text.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        function=run,
    )


def tool_raising(exc):
    def run():
        raise exc

    return Tool(name="boom", description="", parameters={"type": "object", "properties": {}}, function=run)


def tool_steps(result):
    return [s for s in result.trace if isinstance(s, ToolStep)]


def tool_messages(result):
    return [m for m in result.messages if m["role"] == "tool"]


class FlowTest(unittest.TestCase):
    def test_no_tool_calls_goes_straight_to_final_call(self):
        llm = ScriptedLLM(response(content="done"), response(content=FINAL_JSON))
        result = run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS)
        self.assertEqual(result.final.content, FINAL_JSON)
        self.assertEqual(len(llm.calls), 2)
        self.assertFalse(result.limit_reached)

    def test_final_call_uses_json_mode_without_tools(self):
        llm = ScriptedLLM(response(content="done"), response(content=FINAL_JSON))
        run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS)
        final_call = llm.calls[-1]
        self.assertIsNone(final_call["tools"])
        self.assertEqual(final_call["response_schema"], ANSWER_SCHEMA)
        self.assertEqual(final_call["messages"][-1], {"role": "user", "content": C_FINAL_INSTRUCTION})

    def test_tool_turns_offer_tools_without_json_mode(self):
        llm = ScriptedLLM(response(content="done"), response(content=FINAL_JSON))
        run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS)
        self.assertEqual(llm.calls[0]["tools"], [echo_tool().to_api()])
        self.assertIsNone(llm.calls[0]["response_schema"])

    def test_same_settings_on_every_call(self):
        llm = ScriptedLLM(response(calls=[("c1", "echo", '{"text": "a"}')]), response(), response(content=FINAL_JSON))
        run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS)
        self.assertTrue(all(c["settings"] is SETTINGS for c in llm.calls))

    def test_one_tool_call_then_stop(self):
        ran = []
        llm = ScriptedLLM(response(calls=[("c1", "echo", '{"text": "hi"}')]), response(), response(content=FINAL_JSON))
        result = run_tool_loop(llm, MESSAGES, [echo_tool(ran)], SETTINGS)
        self.assertEqual(ran, ["hi"])
        self.assertEqual(tool_messages(result), [{"role": "tool", "tool_call_id": "c1", "content": '{"echo": "hi"}'}])
        step = tool_steps(result)[0]
        self.assertEqual((step.outcome, step.result), ("ok", {"echo": "hi"}))

    def test_assistant_tool_call_message_is_kept_in_history(self):
        llm = ScriptedLLM(response(calls=[("c1", "echo", '{"text": "hi"}')]), response(), response(content=FINAL_JSON))
        result = run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS)
        assistant = result.messages[1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"], {"name": "echo", "arguments": '{"text": "hi"}'})

    def test_several_calls_in_one_turn_all_run_in_order(self):
        calls = [("c1", "echo", '{"text": "a"}'), ("c2", "echo", '{"text": "b"}')]
        llm = ScriptedLLM(response(calls=calls), response(), response(content=FINAL_JSON))
        result = run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS)
        self.assertEqual([m["tool_call_id"] for m in tool_messages(result)], ["c1", "c2"])

    def test_caller_messages_not_modified(self):
        messages = [{"role": "user", "content": "question"}]
        llm = ScriptedLLM(response(content="done"), response(content=FINAL_JSON))
        run_tool_loop(llm, messages, [echo_tool()], SETTINGS)
        self.assertEqual(messages, [{"role": "user", "content": "question"}])

    def test_duplicate_tool_names_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate tool names"):
            run_tool_loop(ScriptedLLM(), MESSAGES, [echo_tool(), echo_tool()], SETTINGS)


class BadCallTest(unittest.TestCase):
    def run_single(self, call, tools):
        llm = ScriptedLLM(response(calls=[call]), response(), response(content=FINAL_JSON))
        return run_tool_loop(llm, MESSAGES, tools, SETTINGS)

    def test_invalid_json_arguments_go_back_to_model(self):
        ran = []
        result = self.run_single(("c1", "echo", "{not json"), [echo_tool(ran)])
        self.assertEqual(ran, [])
        self.assertEqual(tool_steps(result)[0].outcome, "invalid_arguments")
        self.assertIn("not valid JSON", json.loads(tool_messages(result)[0]["content"])["error"])

    def test_schema_violation_goes_back_to_model(self):
        result = self.run_single(("c1", "echo", '{"text": 3}'), [echo_tool()])
        self.assertEqual(tool_steps(result)[0].outcome, "invalid_arguments")

    def test_unknown_tool_goes_back_to_model(self):
        result = self.run_single(("c1", "nope", "{}"), [echo_tool()])
        self.assertEqual(tool_steps(result)[0].outcome, "unknown_tool")
        self.assertIn("available: echo", tool_messages(result)[0]["content"])

    def test_tool_error_goes_back_to_model(self):
        result = self.run_single(("c1", "boom", "{}"), [tool_raising(ToolError("gene not found"))])
        self.assertEqual(tool_steps(result)[0].outcome, "tool_error")
        self.assertEqual(json.loads(tool_messages(result)[0]["content"]), {"error": "gene not found"})

    def test_other_exception_crashes_the_run(self):
        with self.assertRaisesRegex(KeyError, "bug"):
            self.run_single(("c1", "boom", "{}"), [tool_raising(KeyError("bug"))])

    def test_oversized_result_raises(self):
        big = Tool(name="big", description="", parameters={"type": "object", "properties": {}},
                   function=lambda: "x" * MAX_RESULT_CHARS)
        with self.assertRaisesRegex(ValueError, "max 20000"):
            self.run_single(("c1", "big", "{}"), [big])


class LimitTest(unittest.TestCase):
    def test_limit_across_turns_still_gives_final_answer(self):
        ran = []
        turn = lambda i: response(calls=[(f"c{i}", "echo", f'{{"text": "{i}"}}')])
        llm = ScriptedLLM(turn(1), turn(2), turn(3), response(content=FINAL_JSON))
        result = run_tool_loop(llm, MESSAGES, [echo_tool(ran)], SETTINGS, max_tool_calls=2)
        self.assertEqual(ran, ["1", "2"])
        self.assertTrue(result.limit_reached)
        self.assertEqual([s.outcome for s in tool_steps(result)], ["ok", "ok", "over_limit"])
        self.assertEqual(result.final.content, FINAL_JSON)

    def test_too_many_calls_in_one_turn_every_id_answered(self):
        calls = [(f"c{i}", "echo", '{"text": "x"}') for i in range(3)]
        llm = ScriptedLLM(response(calls=calls), response(content=FINAL_JSON))
        result = run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS, max_tool_calls=2)
        self.assertEqual([m["tool_call_id"] for m in tool_messages(result)], ["c0", "c1", "c2"])
        self.assertEqual([s.outcome for s in tool_steps(result)], ["ok", "ok", "over_limit"])
        self.assertTrue(result.limit_reached)

    def test_budget_used_exactly_then_stopping_is_not_limit_reached(self):
        llm = ScriptedLLM(
            response(calls=[("c1", "echo", '{"text": "a"}'), ("c2", "echo", '{"text": "b"}')]),
            response(content="done"),
            response(content=FINAL_JSON),
        )
        result = run_tool_loop(llm, MESSAGES, [echo_tool()], SETTINGS, max_tool_calls=2)
        self.assertFalse(result.limit_reached)


class ToolSchemaTest(unittest.TestCase):
    def make(self, parameters):
        return Tool(name="t", description="", parameters=parameters, function=lambda: None)

    def test_supported_schema_accepted(self):
        self.make({"type": "object",
                   "properties": {"id": {"type": "integer", "description": "profile id"}},
                   "required": ["id"]})

    def test_top_level_must_be_object(self):
        with self.assertRaisesRegex(ValueError, "type object"):
            self.make({"type": "array"})

    def test_unsupported_top_level_key(self):
        with self.assertRaisesRegex(ValueError, "unsupported schema keys: additionalProperties"):
            self.make({"type": "object", "properties": {}, "additionalProperties": False})

    def test_unsupported_property_key(self):
        with self.assertRaisesRegex(ValueError, r"t\.disease: unsupported keys: enum"):
            self.make({"type": "object", "properties": {"disease": {"type": "string", "enum": ["Melanoma"]}}})

    def test_property_without_supported_type(self):
        with self.assertRaisesRegex(ValueError, r"t\.ids: type must be one of"):
            self.make({"type": "object", "properties": {"ids": {"type": "array"}}})

    def test_required_must_be_declared(self):
        with self.assertRaisesRegex(ValueError, "required but not declared: id"):
            self.make({"type": "object", "properties": {}, "required": ["id"]})


class ValidateArgumentsTest(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "disease": {"type": "string"}},
        "required": ["id"],
    }

    def test_valid(self):
        validate_arguments({"id": 12, "disease": "Melanoma"}, self.SCHEMA)

    def test_optional_may_be_omitted(self):
        validate_arguments({"id": 12}, self.SCHEMA)

    def test_not_an_object(self):
        with self.assertRaisesRegex(InvalidArguments, "JSON object"):
            validate_arguments([12], self.SCHEMA)

    def test_missing_required(self):
        with self.assertRaisesRegex(InvalidArguments, "missing arguments: id"):
            validate_arguments({}, self.SCHEMA)

    def test_unknown_argument(self):
        with self.assertRaisesRegex(InvalidArguments, "unknown arguments: gene"):
            validate_arguments({"id": 12, "gene": "BRAF"}, self.SCHEMA)

    def test_wrong_type(self):
        with self.assertRaisesRegex(InvalidArguments, "id must be integer, got str"):
            validate_arguments({"id": "12"}, self.SCHEMA)

    def test_bool_is_not_integer(self):
        with self.assertRaisesRegex(InvalidArguments, "id must be integer, got boolean"):
            validate_arguments({"id": True}, self.SCHEMA)

    def test_null_for_optional_is_rejected(self):
        with self.assertRaisesRegex(InvalidArguments, "disease must be string"):
            validate_arguments({"id": 12, "disease": None}, self.SCHEMA)


if __name__ == "__main__":
    unittest.main()
