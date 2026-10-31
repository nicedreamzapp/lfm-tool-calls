# lfm-tool-calls

A small parser and an OpenAI-compatible proxy that turn Liquid AI LFM tool calls into proper `tool_calls`, whatever format the model used.

## The problem

LFM models write tool calls in different formats depending on the generation and on how they were prompted. Serving stacks (Ollama, llama-server, vLLM, LM Studio) only recognise some of them, and when a call is not recognised it reaches the client as plain text and the agent loop stops.

Two public reports describe this:

- [Liquid4All/cookbook#72](https://github.com/Liquid4All/cookbook/issues/72): using LFM2 with Ollama and opencode needed a hand-written proxy, because replies came back as "a mix of native, json and even xml".
- [Liquid4All/docs#130](https://github.com/Liquid4All/docs/issues/130): the chat template never states the call format, so the model sometimes drifts to `{"TaskCreate": {...}}` and the call is silently lost.

## Formats it reads

| Format | Example |
|---|---|
| LFM2 / LFM2.5 Pythonic | `<\|tool_call_start\|>[Bash(command="ls")]<\|tool_call_end\|>` |
| LFM2.5 JSON mode | `<\|tool_call_start\|>[{"name": "Bash", "arguments": {"command": "ls"}}]<\|tool_call_end\|>` |
| LFM3 XML | `<tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call>` |
| Bare JSON drift | `{"Bash": {"command": "ls"}}` or `{"name": "Bash", "arguments": {...}}` |

It also handles a missing end tag or closing bracket (a truncated stream), several calls in one block, `true`/`false`/`null` written inside Pythonic calls, and tool names that differ only by case or underscores from the offered tool.

## Quick start

Both files use the Python standard library only (3.9 or newer). Point your client at `http://127.0.0.1:8090/v1`.

With Ollama:

```
ollama serve
UPSTREAM_URL=http://127.0.0.1:11434 python3 proxy.py
```

With llama-server:

```
llama-server -m your-lfm-model.gguf --jinja --port 8080
UPSTREAM_URL=http://127.0.0.1:8080 PORT=8090 python3 proxy.py
```

vLLM and LM Studio work the same way: set `UPSTREAM_URL` to their base URL.

As a library:

```python
from lfm_tool_calls import parse, schemas_from_openai_tools, to_openai_tool_calls

calls, text = parse(reply_text, schemas_from_openai_tools(tools))
tool_calls = to_openai_tool_calls(calls, offered=[t["function"]["name"] for t in tools])
```

`parse` returns a list of `(name, args)` pairs and the text that came before the first call. If nothing qualifies it returns `([], reply_text)` unchanged.

## What the proxy does

For `POST /v1/chat/completions` requests that include `tools`, it forwards the request unchanged. If a reply has no `tool_calls` but its content ends in an LFM tool call, the proxy returns that call as `tool_calls` with `finish_reason: "tool_calls"`. Anything else comes back byte for byte as the upstream sent it, including replies where the upstream already produced `tool_calls`.

When the client asks for a stream, the proxy reads the whole upstream stream first, then either relays it unchanged or sends a rewritten one as standard `chat.completion.chunk` events. That means a tool-using request is not streamed token by token. Requests without `tools`, `/v1/models` and other paths are passed straight through.

Settings: `UPSTREAM_URL` (default `http://127.0.0.1:11434`), `PORT` (default `8090`), `HOST` (default `127.0.0.1`), `UPSTREAM_TIMEOUT` in seconds (default `600`). The proxy logs one line per request (method, path, status) and never logs message content.

## Safety

- It never executes anything. Pythonic calls are read with `ast.parse` and values with `ast.literal_eval`; an expression like `__import__("os").system(...)` is kept as text, not run.
- Calls inside `<think>...</think>`, inside backticks, or inside code fences are ignored, so a model explaining the format does not trigger a call.
- A call only counts when it closes out the reply. A call followed by more prose is treated as an example and left as text.
- XML parameter values stay strings unless the tool schema says the parameter is a number, boolean, array or object, so file content that looks like JSON is not decoded.
- Bare JSON is only accepted when it names a tool that was offered in the request, and only at the end of the reply.

## Tested with

`python3 -m unittest discover -s tests` runs 50 tests: 34 for the parser (including the safety cases above) and 16 for the proxy against a fake upstream, streaming and non-streaming. They pass on Python 3.9 and 3.14.

All test inputs are synthetic, written in the formats listed above. No test replays a recorded model output, and the proxy tests do not talk to a real model server.

## Credits

LFM models are made by [Liquid AI](https://www.liquid.ai/). This project is not affiliated with Liquid AI.

## License

MIT, see [LICENSE](LICENSE).
