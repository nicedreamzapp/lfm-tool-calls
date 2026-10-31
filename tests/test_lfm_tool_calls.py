import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfm_tool_calls import canon_name, has_call, parse, schemas_from_openai_tools, to_openai_tool_calls  # noqa: E402

OFFERED = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]


class Pythonic(unittest.TestCase):
    def test_single(self):
        calls, rest = parse('Let me look.<|tool_call_start|>[Bash(command="ls -la")]<|tool_call_end|>')
        self.assertEqual(calls, [("Bash", {"command": "ls -la"})])
        self.assertEqual(rest, "Let me look.")

    def test_two_calls_types(self):
        calls, _ = parse('<|tool_call_start|>[Read(file_path="/a", limit=20), Grep(pattern="x", multiline=True)]<|tool_call_end|>')
        self.assertEqual(calls, [("Read", {"file_path": "/a", "limit": 20}), ("Grep", {"pattern": "x", "multiline": True})])

    def test_missing_end_and_bracket(self):
        calls, _ = parse('<|tool_call_start|>[Bash(command="echo hi")')
        self.assertEqual(calls, [("Bash", {"command": "echo hi"})])

    def test_json_words(self):
        calls, _ = parse('<|tool_call_start|>[Grep(pattern="a", multiline=true)]<|tool_call_end|>')
        self.assertEqual(calls, [("Grep", {"pattern": "a", "multiline": True})])

    def test_json_mode(self):
        calls, _ = parse('<|tool_call_start|>[{"name": "Bash", "arguments": {"command": "pwd"}}]<|tool_call_end|>')
        self.assertEqual(calls, [("Bash", {"command": "pwd"})])

    def test_never_executes(self):
        calls, _ = parse('<|tool_call_start|>[Bash(command=__import__("os").system("touch /tmp/pwned_lfm"))]<|tool_call_end|>')
        self.assertFalse(os.path.exists("/tmp/pwned_lfm"))
        self.assertEqual(calls[0][0], "Bash")

    def test_think_ignored(self):
        calls, _ = parse('<think>maybe <|tool_call_start|>[Bash(command="rm -rf /")]<|tool_call_end|></think>ok')
        self.assertEqual(calls, [])


class Xml(unittest.TestCase):
    def test_lfm3(self):
        t = "Checking.\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la ~/x\n</parameter>\n</function>\n</tool_call>"
        calls, rest = parse(t)
        self.assertEqual(calls, [("Bash", {"command": "ls -la ~/x"})])
        self.assertEqual(rest, "Checking.")

    def test_numbers_and_multi(self):
        t = ("<tool_call><function=Read><parameter=file_path>/a</parameter><parameter=limit>40</parameter></function></tool_call>"
             "<tool_call><function=Glob><parameter=pattern>**/*.md</parameter></function></tool_call>")
        calls, _ = parse(t)
        self.assertEqual(calls, [("Read", {"file_path": "/a", "limit": 40}), ("Glob", {"pattern": "**/*.md"})])

    def test_truncated(self):
        calls, _ = parse("<tool_call><function=Write><parameter=file_path>/tmp/a.md</parameter><parameter=content>line1\nline2")
        self.assertEqual(calls, [("Write", {"file_path": "/tmp/a.md", "content": "line1\nline2"})])

    def test_has_call(self):
        self.assertTrue(has_call("x <tool_call><function=Bash>"))
        self.assertFalse(has_call("plain answer"))
        self.assertFalse(has_call("<think><tool_call></think>done"))


class Names(unittest.TestCase):
    def test_case_and_alias(self):
        self.assertEqual(canon_name("bash", OFFERED), "Bash")
        self.assertEqual(canon_name("read_file", OFFERED), "Read")
        self.assertEqual(canon_name("shell", OFFERED), "Bash")
        self.assertEqual(canon_name("write_file", OFFERED), "Write")

    def test_unknown_passes_through(self):
        self.assertEqual(canon_name("Teleport", OFFERED), "Teleport")
        self.assertEqual(canon_name("Bash", OFFERED), "Bash")


class QuotedAndUnsafe(unittest.TestCase):
    """A quoted or explained call must never become a real one."""

    def test_fenced_example_is_not_a_call(self):
        t = ("The format looks like this: ```<tool_call><function=Bash><parameter=command>rm -rf ~/tmp"
             "</parameter></function></tool_call>```")
        self.assertEqual(parse(t), ([], t))
        self.assertFalse(has_call(t))

    def test_inline_backticks(self):
        t = "Liquid emits `<|tool_call_start|>[Bash(command=\"rm -rf /\")]<|tool_call_end|>` for calls."
        self.assertFalse(has_call(t))

    def test_bare_function_tag_in_prose(self):
        t = "Inside <function=Bash> you put parameters, then the call ends."
        self.assertEqual(parse(t), ([], t))

    def test_call_mid_explanation_is_not_rescued(self):
        t = ("Example: <tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call> "
             "and that is how the model asks for a shell command.")
        self.assertEqual(parse(t)[0], [])

    def test_nul_byte_does_not_crash(self):
        calls, _ = parse("<|tool_call_start|>[Bash(command=\"a\x00b\")]<|tool_call_end|>")
        self.assertIsInstance(calls, list)

    def test_json_looking_content_stays_string(self):
        t = '<tool_call><function=Write><parameter=file_path>/a.json</parameter><parameter=content>{"a": 1}</parameter></function></tool_call>'
        self.assertEqual(parse(t)[0], [("Write", {"file_path": "/a.json", "content": '{"a": 1}'})])

    def test_schema_decodes_non_string(self):
        t = '<tool_call><function=Q><parameter=ids>[1, 2]</parameter><parameter=note>[1, 2]</parameter></function></tool_call>'
        calls, _ = parse(t, {"Q": {"ids": "array", "note": "string"}})
        self.assertEqual(calls, [("Q", {"ids": [1, 2], "note": "[1, 2]"})])

    def test_function_tag_inside_value_does_not_split(self):
        t = '<tool_call><function=Write><parameter=content>Use <function=Bash> here</parameter></function></tool_call>'
        self.assertEqual(parse(t)[0], [("Write", {"content": "Use <function=Bash> here"})])

    def test_exec_alias_dropped(self):
        self.assertEqual(canon_name("exec", OFFERED), "exec")


class JsonDrift(unittest.TestCase):
    """Liquid4All/docs#130: bare JSON instead of the Pythonic call."""
    S = {"TaskCreate": {"subject": "string"}, "Bash": {"command": "string"}}

    def test_tool_keyed_object(self):
        t = 'Let me make a task. {"TaskCreate": {"subject": "Fix login"}, "activeForm": "Fixing"}'
        calls, rest = parse(t, self.S)
        self.assertEqual(calls, [("TaskCreate", {"subject": "Fix login", "activeForm": "Fixing"})])
        self.assertEqual(rest, "Let me make a task.")

    def test_name_arguments_object(self):
        calls, _ = parse('{"name": "Bash", "arguments": "{\\"command\\": \\"ls\\"}"}', self.S)
        self.assertEqual(calls, [("Bash", {"command": "ls"})])

    def test_unoffered_tool_is_text(self):
        t = '{"DeleteEverything": {"path": "/"}}'
        self.assertEqual(parse(t, self.S), ([], t))

    def test_no_schemas_no_rescue(self):
        t = '{"Bash": {"command": "ls"}}'
        self.assertEqual(parse(t), ([], t))

    def test_json_mid_answer_is_text(self):
        t = 'The config is {"Bash": {"command": "ls"}} and that is all.'
        self.assertEqual(parse(t, self.S), ([], t))

    def test_quoted_json_is_text(self):
        t = 'Example: ```{"Bash": {"command": "rm -rf /"}}```'
        self.assertEqual(parse(t, self.S), ([], t))


class RemainingText(unittest.TestCase):
    def test_garbage_block_returns_text_unchanged(self):
        t = "Hmm <|tool_call_start|>not a call at all<|tool_call_end|>"
        self.assertEqual(parse(t), ([], t))

    def test_code_and_reasoning_before_call_are_kept(self):
        t = '<think>need ls</think>Run `ls` first.\n<|tool_call_start|>[Bash(command="ls")]<|tool_call_end|>'
        calls, rest = parse(t)
        self.assertEqual(calls, [("Bash", {"command": "ls"})])
        self.assertEqual(rest, "<think>need ls</think>Run `ls` first.")

    def test_string_with_apostrophe(self):
        # docs#130 notes the stock template does not escape quotes; the model's own escaped form must read back.
        calls, _ = parse("<|tool_call_start|>[send_message(text='It\\'s fine')]<|tool_call_end|>")
        self.assertEqual(calls, [("send_message", {"text": "It's fine"})])


class OpenAIShape(unittest.TestCase):
    TOOLS = [
        {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "limit": {"type": ["integer", "null"]}, "x": {}}}}},
        {"type": "function", "function": {"name": "Ping"}},
        {"type": "function"},
    ]

    def test_schemas_from_tools(self):
        self.assertEqual(schemas_from_openai_tools(self.TOOLS),
                         {"Read": {"file_path": "string", "limit": "integer", "x": None}, "Ping": {}})

    def test_to_openai(self):
        out = to_openai_tool_calls([("read_file", {"file_path": "/a", "limit": 3})], ["Read", "Ping"])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["id"].startswith("call_"))
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(out[0]["function"]["name"], "Read")
        self.assertEqual(json.loads(out[0]["function"]["arguments"]), {"file_path": "/a", "limit": 3})

    def test_unserialisable_value_becomes_text(self):
        out = to_openai_tool_calls([("F", {"s": {1, 2}})])
        self.assertIsInstance(json.loads(out[0]["function"]["arguments"])["s"], str)


if __name__ == "__main__":
    unittest.main()
