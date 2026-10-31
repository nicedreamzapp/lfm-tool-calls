"""Replies recorded from LFM3-Omni-2B in a browser-agent benchmark (synthetic test shop, no personal data).
Each fixture is one real reply, the tools offered in that run, and the calls a person reads out of it."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lfm_tool_calls import parse

TOOLS = {
    "navigate": {"url": "string"}, "snapshot": {}, "click": {"uid": "string"},
    "type_text": {"uid": "string", "text": "string"}, "scroll": {"direction": "string"},
    "js": {"code": "string"}, "shell": {"cmd": "string", "timeout": "number"},
    "read_file": {"path": "string"}, "write_file": {"path": "string", "content": "string"},
    "ask": {"question": "string"}, "done": {"message": "string"},
}
PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_lfm3.jsonl")


def load():
    with open(PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RealReplies(unittest.TestCase):
    def test_fixtures_present(self):
        self.assertGreaterEqual(len(load()), 10)

    def test_each_reply(self):
        for fx in load():
            with self.subTest(id=fx["id"], note=fx["note"]):
                calls, rest = parse(fx["reply"], TOOLS)
                got = [{"name": n, "arguments": a} for n, a in calls]
                self.assertEqual(got, fx["expected"])
                if not fx["expected"]:
                    self.assertEqual(rest, fx["reply"])  # left as text, unchanged

    def test_no_user_paths_in_fixtures(self):
        with open(PATH, encoding="utf-8") as f:
            raw = f.read()
        self.assertNotIn("/Users/", raw)
        self.assertNotIn("/home/", raw)

    def test_json_drift_needs_an_offered_tool(self):
        self.assertEqual(parse('{"tool": "rm_rf", "args": {}}', TOOLS)[0], [])
        self.assertEqual(parse('{"tool": "click", "args": {"uid": "1"}}')[0], [])

    def test_json_drift_quoted_or_followed_by_prose_is_text(self):
        for t in ('Use `{"tool": "click", "args": {"uid": "1"}}` to click.',
                  '{"tool": "click", "args": {"uid": "1"}} and then I will wait.',
                  '```json\n{"tool": "click", "args": {"uid": "1"}}\n```'):
            self.assertEqual(parse(t, TOOLS)[0], [], t)

    def test_array_with_one_unknown_tool_is_text(self):
        t = '[{"tool": "click", "args": {"uid": "1"}}, {"tool": "nope", "args": {}}]'
        self.assertEqual(parse(t, TOOLS)[0], [])

    def test_array_of_calls(self):
        t = '[{"tool": "type_text", "args": {"uid": "1", "text": "a"}}, {"tool": "click", "args": {"uid": "2"}}]'
        self.assertEqual([c[0] for c in parse(t, TOOLS)[0]], ["type_text", "click"])


if __name__ == "__main__":
    unittest.main()
