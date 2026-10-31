"""Proxy tests against a fake OpenAI-compatible upstream (no model, no network beyond 127.0.0.1)."""
import json
import os
import sys
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import proxy  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {
        "file_path": {"type": "string"}, "limit": {"type": "integer"}}}}},
]


class FakeUpstream(BaseHTTPRequestHandler):
    """Replies with whatever the test put in `reply`: a message dict, as JSON or as an SSE stream."""
    reply = {"role": "assistant", "content": "hi"}
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "lfm-test", "object": "model"}]}).encode()
        self._send(body, "application/json")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUpstream.seen.append(req)
        msg = dict(FakeUpstream.reply)
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        if not req.get("stream"):
            body = {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "lfm-test",
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}
            return self._send(json.dumps(body).encode(), "application/json")
        chunks = [{"role": "assistant", "content": ""}]
        content = msg.get("content") or ""
        chunks += [{"content": content[i:i + 7]} for i in range(0, len(content), 7)]
        for i, tc in enumerate(msg.get("tool_calls") or []):
            chunks.append({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                           "function": {"name": tc["function"]["name"], "arguments": ""}}]})
            chunks.append({"tool_calls": [{"index": i, "function": {"arguments": tc["function"]["arguments"]}}]})
        out = ""
        for d in chunks:
            out += "data: " + json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
                                          "model": "lfm-test",
                                          "choices": [{"index": 0, "delta": d, "finish_reason": None}]}) + "\n\n"
        out += "data: " + json.dumps({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
                                      "model": "lfm-test",
                                      "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n"
        out += "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(out.encode())  # no Content-Length: like a real stream

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(srv):
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


class ProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = _serve(ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream))
        cls.px = _serve(proxy.make_server("http://127.0.0.1:%d/v1" % cls.up.server_address[1], "127.0.0.1", 0))
        cls.base = "http://127.0.0.1:%d" % cls.px.server_address[1]
        cls._stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

    @classmethod
    def tearDownClass(cls):
        sys.stderr.close()
        sys.stderr = cls._stderr
        cls.px.shutdown()
        cls.up.shutdown()

    def chat(self, content=None, stream=False, tools=TOOLS, tool_calls=None):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        FakeUpstream.reply = msg
        body = {"model": "lfm-test", "messages": [{"role": "user", "content": "go"}], "stream": stream}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.headers.get("Content-Type", ""), r.read()

    def direct(self, stream):
        """What the upstream itself returns for the current reply, for byte-for-byte comparisons."""
        body = {"model": "lfm-test", "messages": [], "stream": stream, "tools": TOOLS}
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % self.up.server_address[1],
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read()

    @staticmethod
    def sse(raw):
        events = [ln[6:] for ln in raw.decode().split("\n") if ln.startswith("data: ")]
        assert events[-1] == "[DONE]", events[-1]
        return [json.loads(e) for e in events[:-1]]

    # ---- non-streaming

    def test_pythonic_call(self):
        _, raw = self.chat('Listing.<|tool_call_start|>[Bash(command="ls -la")]<|tool_call_end|>')
        ch = json.loads(raw)["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertEqual(ch["message"]["content"], "Listing.")
        tc = ch["message"]["tool_calls"]
        self.assertEqual(tc[0]["function"]["name"], "Bash")
        self.assertEqual(json.loads(tc[0]["function"]["arguments"]), {"command": "ls -la"})
        self.assertEqual(json.loads(raw)["usage"]["total_tokens"], 12)

    def test_lfm3_xml_call(self):
        _, raw = self.chat("<tool_call>\n<function=Read>\n<parameter=file_path>\n/tmp/a.txt\n</parameter>\n"
                           "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>")
        ch = json.loads(raw)["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertIsNone(ch["message"]["content"])
        args = json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args, {"file_path": "/tmp/a.txt", "limit": 20})

    def test_quoted_example_stays_text(self):
        text = 'A call looks like `<|tool_call_start|>[Bash(command="rm -rf /")]<|tool_call_end|>`.'
        _, raw = self.chat(text)
        self.assertEqual(raw, self.direct(False))
        ch = json.loads(raw)["choices"][0]
        self.assertEqual(ch["message"]["content"], text)
        self.assertNotIn("tool_calls", ch["message"])

    def test_normal_answer_untouched(self):
        _, raw = self.chat("The answer is 4.")
        self.assertEqual(raw, self.direct(False))

    def test_upstream_tool_calls_untouched(self):
        tcs = [{"id": "call_up1", "type": "function", "function": {"name": "Bash", "arguments": "{\"command\": \"pwd\"}"}}]
        _, raw = self.chat("<|tool_call_start|>[Bash(command=\"ls\")]<|tool_call_end|>", tool_calls=tcs)
        self.assertEqual(raw, self.direct(False))
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["tool_calls"], tcs)

    def test_no_tools_in_request_passes_through(self):
        _, raw = self.chat('<|tool_call_start|>[Bash(command="ls")]<|tool_call_end|>', tools=None)
        self.assertNotIn("tool_calls", json.loads(raw)["choices"][0]["message"])

    def test_json_drift_for_offered_tool(self):
        _, raw = self.chat('Checking. {"Bash": {"command": "git status"}}')
        ch = json.loads(raw)["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertEqual(ch["message"]["tool_calls"][0]["function"]["name"], "Bash")

    def test_models_pass_through(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
            self.assertEqual(json.loads(r.read())["data"][0]["id"], "lfm-test")

    # ---- streaming

    def test_stream_pythonic_call(self):
        ctype, raw = self.chat('Listing.<|tool_call_start|>[Bash(command="ls -la"), Read(file_path="/a")]'
                               '<|tool_call_end|>', stream=True)
        self.assertIn("text/event-stream", ctype)
        chunks = self.sse(raw)
        for c in chunks:
            self.assertEqual(c["object"], "chat.completion.chunk")
            self.assertEqual(c["id"], "chatcmpl-1")
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertEqual(deltas[0]["role"], "assistant")
        self.assertEqual("".join(d.get("content") or "" for d in deltas), "Listing.")
        tcs = [d["tool_calls"][0] for d in deltas if d.get("tool_calls")]
        self.assertEqual([t["index"] for t in tcs], [0, 1])
        self.assertEqual([t["function"]["name"] for t in tcs], ["Bash", "Read"])
        self.assertTrue(all(t["id"].startswith("call_") and t["type"] == "function" for t in tcs))
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]), {"command": "ls -la"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1]))

    def test_stream_lfm3_xml_call(self):
        _, raw = self.chat("Reading it.\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/tmp/b.md\n"
                           "</parameter>\n</function>\n</tool_call>", stream=True)
        chunks = self.sse(raw)
        tcs = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]), {"file_path": "/tmp/b.md"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_stream_quoted_example_untouched(self):
        _, raw = self.chat('Example: ```<tool_call><function=Bash><parameter=command>rm -rf ~</parameter>'
                           '</function></tool_call>```', stream=True)
        self.assertEqual(raw, self.direct(True))

    def test_stream_normal_answer_untouched(self):
        _, raw = self.chat("Plain streamed answer, nothing to see.", stream=True)
        self.assertEqual(raw, self.direct(True))

    def test_stream_upstream_tool_calls_untouched(self):
        tcs = [{"id": "call_up2", "type": "function", "function": {"name": "Read", "arguments": "{\"file_path\": \"/x\"}"}}]
        _, raw = self.chat(None, stream=True, tool_calls=tcs)
        self.assertEqual(raw, self.direct(True))

    def test_stream_without_tools_is_relayed(self):
        _, raw = self.chat("streamed", stream=True, tools=None)
        self.assertEqual(raw, self.direct(True))


class Assemble(unittest.TestCase):
    def test_reasoning_and_usage_survive(self):
        raw = (b'data: {"id":"x","created":2,"model":"m","choices":[{"index":0,"delta":{"role":"assistant",'
               b'"reasoning_content":"think "}}]}\n\n'
               b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"<|tool_call_start|>[Bash(command=\\"ls\\")]'
               b'<|tool_call_end|>"}}]}\n\n'
               b'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
               b'data: {"id":"x","choices":[],"usage":{"total_tokens":3}}\n\n'
               b'data: [DONE]\n\n')
        obj = proxy.assemble_stream(raw)
        self.assertTrue(proxy.rewrite_completion(obj, {"Bash": {"command": "string"}}))
        chunks = ProxyTest.sse(proxy.to_sse(obj))
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"reasoning_content": "think "})
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"], {"total_tokens": 3})
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")

    def test_unreadable_stream(self):
        self.assertIsNone(proxy.assemble_stream(b"data: {not json\n\n"))
        self.assertIsNone(proxy.assemble_stream(b""))


if __name__ == "__main__":
    unittest.main()
