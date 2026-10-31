#!/usr/bin/env python3
"""OpenAI-compatible proxy that turns LFM tool calls left in `content` into real `tool_calls`.

Put it between your client and Ollama, llama-server, vLLM or LM Studio:

    UPSTREAM_URL=http://127.0.0.1:11434 PORT=8090 python3 proxy.py

then point the client at http://127.0.0.1:8090/v1.

What it does on POST /v1/chat/completions when the request carries `tools`:
  * forwards the request unchanged,
  * if a reply choice has no tool_calls but its content ends in an LFM tool call (see lfm_tool_calls.py),
    replaces it with proper tool_calls and finish_reason "tool_calls",
  * otherwise returns the upstream bytes untouched.
If the client asked for a stream, the upstream stream is buffered, checked, and sent back as valid
OpenAI SSE chunks (or relayed byte for byte when nothing changed). Requests without tools and every
other path (/v1/models and so on) are passed through.

Environment:
  UPSTREAM_URL      upstream base URL (default http://127.0.0.1:11434); a trailing /v1 is ignored
  PORT              listen port (default 8090)
  HOST              listen address (default 127.0.0.1)
  UPSTREAM_TIMEOUT  seconds (default 600)

Logs one line per request (method, path, status). Message content is never logged.
Standard library only, Python 3.9+.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lfm_tool_calls import parse, schemas_from_openai_tools, to_openai_tool_calls

_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding"}
_CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")


def _base(url):
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


# ---------------------------------------------------------------- reply rewriting

def rewrite_message(msg, schemas):
    """Return (new_message, changed). Only touches a message with no tool_calls and string content."""
    if not isinstance(msg, dict) or msg.get("tool_calls"):
        return msg, False
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        return msg, False
    calls, rest = parse(content, schemas)
    if not calls:
        return msg, False
    new = dict(msg)
    new["content"] = rest or None
    new["tool_calls"] = to_openai_tool_calls(calls, list(schemas))
    return new, True


def rewrite_completion(obj, schemas):
    """Rewrite a non-streamed chat.completion object in place. Returns True when anything changed."""
    changed = False
    for choice in obj.get("choices") or []:
        msg, did = rewrite_message(choice.get("message"), schemas)
        if did:
            choice["message"] = msg
            choice["finish_reason"] = "tool_calls"
            changed = True
    return changed


def assemble_stream(raw):
    """Fold an OpenAI SSE stream (bytes) into a chat.completion-like dict. Returns None if unreadable."""
    head, choices, usage, saw = {}, {}, None, False
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            return None
        if not isinstance(chunk, dict):
            return None
        saw = True
        for k in ("id", "created", "model", "system_fingerprint"):
            if k in chunk and k not in head:
                head[k] = chunk[k]
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            idx = ch.get("index", 0)
            acc = choices.setdefault(idx, {"role": "assistant", "content": "", "reasoning": {}, "tools": {},
                                           "finish_reason": None})
            delta = ch.get("delta") or {}
            if delta.get("role"):
                acc["role"] = delta["role"]
            if isinstance(delta.get("content"), str):
                acc["content"] += delta["content"]
            for rk in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(rk), str):
                    acc["reasoning"][rk] = acc["reasoning"].get(rk, "") + delta[rk]
            for tc in delta.get("tool_calls") or []:
                t = acc["tools"].setdefault(tc.get("index", 0), {"id": None, "type": "function",
                                                                 "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    t["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    t["function"]["name"] += fn["name"]
                if isinstance(fn.get("arguments"), str):
                    t["function"]["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                acc["finish_reason"] = ch["finish_reason"]
    if not saw:
        return None
    out = dict(head)
    out["choices"] = []
    for idx in sorted(choices):
        acc = choices[idx]
        msg = {"role": acc["role"], "content": acc["content"] or None}
        msg.update(acc["reasoning"])
        if acc["tools"]:
            msg["tool_calls"] = [acc["tools"][i] for i in sorted(acc["tools"])]
        out["choices"].append({"index": idx, "message": msg, "finish_reason": acc["finish_reason"]})
    if usage:
        out["usage"] = usage
    return out


def to_sse(obj):
    """Emit a chat.completion-like dict as OpenAI chat.completion.chunk SSE bytes."""
    base = {"id": obj.get("id") or "chatcmpl-proxy", "object": "chat.completion.chunk",
            "created": obj.get("created") or int(time.time()), "model": obj.get("model") or ""}
    if obj.get("system_fingerprint"):
        base["system_fingerprint"] = obj["system_fingerprint"]
    events = []

    def emit(choices, **extra):
        c = dict(base)
        c["choices"] = choices
        c.update(extra)
        events.append("data: " + json.dumps(c, ensure_ascii=False) + "\n\n")

    for ch in obj.get("choices") or []:
        idx, msg = ch.get("index", 0), ch.get("message") or {}
        emit([{"index": idx, "delta": {"role": msg.get("role") or "assistant", "content": ""},
               "finish_reason": None}])
        for rk in ("reasoning_content", "reasoning"):
            if msg.get(rk):
                emit([{"index": idx, "delta": {rk: msg[rk]}, "finish_reason": None}])
        if msg.get("content"):
            emit([{"index": idx, "delta": {"content": msg["content"]}, "finish_reason": None}])
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            emit([{"index": idx, "delta": {"tool_calls": [{
                "index": i, "id": tc.get("id"), "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments", "")}}]},
                "finish_reason": None}])
        emit([{"index": idx, "delta": {}, "finish_reason": ch.get("finish_reason") or "stop"}])
    if obj.get("usage"):
        emit([], usage=obj["usage"])
    events.append("data: [DONE]\n\n")
    return "".join(events).encode("utf-8")


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:11434"
    timeout_s = 600.0
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # request line only, never bodies
        pass

    def _log(self, status):
        sys.stderr.write("%s %s %s\n" % (self.command, self.path.split("?")[0], status))

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _open(self, body):
        """Send the request upstream. Returns (status, headers, response_or_bytes)."""
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
        req = urllib.request.Request(self.upstream + self.path, data=body if self.command != "GET" else None,
                                     headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:  # connection refused, timeout
            msg = json.dumps({"error": {"message": "upstream unreachable: %s" % type(e).__name__,
                                        "type": "proxy_error"}}).encode()
            return 502, {"Content-Type": "application/json"}, msg
        return resp.code, dict(resp.headers.items()), resp

    def _send(self, status, headers, data):
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in _HOP:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self._log(status)

    def _pipe(self, status, headers, resp):
        """Relay a response untouched. Streams it through as it arrives when the length is unknown."""
        if isinstance(resp, bytes):
            return self._send(status, headers, resp)
        length = {k.lower(): v for k, v in headers.items()}.get("content-length")
        if length is not None:
            return self._send(status, headers, resp.read())
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in _HOP:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        read = getattr(resp, "read1", None) or resp.read
        while True:
            chunk = read(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        self._log(status)

    def _relay(self):
        self._pipe(*self._open(self._read_body()))

    do_GET = do_DELETE = do_PUT = do_PATCH = _relay

    def do_POST(self):
        body = self._read_body()
        req = None
        if self.path.split("?")[0] in _CHAT_PATHS:
            try:
                req = json.loads(body or b"{}")
            except ValueError:
                req = None
        tools = req.get("tools") if isinstance(req, dict) else None
        status, headers, resp = self._open(body)
        if not tools or status != 200:
            return self._pipe(status, headers, resp)
        data = resp if isinstance(resp, bytes) else resp.read()
        schemas = schemas_from_openai_tools(tools)
        if req.get("stream"):
            obj = assemble_stream(data)
            if obj is None or not rewrite_completion(obj, schemas):
                return self._send(status, headers, data)
            headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
            headers["Content-Type"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"
            return self._send(200, headers, to_sse(obj))
        try:
            obj = json.loads(data)
        except ValueError:
            return self._send(status, headers, data)
        if not isinstance(obj, dict) or not rewrite_completion(obj, schemas):
            return self._send(status, headers, data)
        headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
        headers["Content-Type"] = "application/json"
        return self._send(200, headers, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def make_server(upstream, host="127.0.0.1", port=8090, timeout=600.0):
    handler = type("LFMProxyHandler", (Handler,), {"upstream": _base(upstream), "timeout_s": float(timeout)})
    return ThreadingHTTPServer((host, port), handler)


def main():
    upstream = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:11434")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8090"))
    srv = make_server(upstream, host, port, float(os.environ.get("UPSTREAM_TIMEOUT", "600")))
    sys.stderr.write("lfm-tool-calls proxy on http://%s:%d/v1 -> %s\n" % (host, port, _base(upstream)))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
