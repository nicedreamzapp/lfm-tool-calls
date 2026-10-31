"""lfm_tool_calls: read Liquid AI LFM tool calls out of plain model text.

LFM generations write tool calls in different formats:

  LFM2 / LFM2.5 (Pythonic):  <|tool_call_start|>[Bash(command="ls -la"), Read(file_path="/x")]<|tool_call_end|>
  LFM2.5 (JSON mode):        <|tool_call_start|>[{"name": "Bash", "arguments": {"command": "ls"}}]<|tool_call_end|>
  LFM3 (XML):                <tool_call><function=Bash><parameter=command>ls -la</parameter></function></tool_call>
  Bare JSON drift:           {"Bash": {"command": "ls"}}, {"name": "Bash", "arguments": {...}},
                             {"tool": "Bash", "args": {...}}, or a JSON array of those

When the serving stack does not recognise the format (wrong template, a missing end tag, a call that
follows some reasoning), the client gets text instead of a tool call. parse() turns any of the above
into (calls, remaining_text).

Guarantees:
  * Nothing is ever executed. Pythonic calls are read with ast.parse, values with ast.literal_eval.
  * Calls inside <think>...</think> or inside backticks / code fences are ignored.
  * Calls only count when they close out the reply. A call followed by more prose is an example,
    not a request, and is left as text.
  * XML parameter values stay strings unless the tool's schema says otherwise.
  * Bare JSON is only accepted when it names a tool that was actually offered.

Standard library only, Python 3.9+.
"""
import ast
import json
import re
import uuid

__all__ = ["parse", "has_call", "to_openai_tool_calls", "canon_name", "schemas_from_openai_tools"]

# A call needs the full wrapper (a bare "<function=Bash>" in prose is not a call).
_PY_BLOCK = re.compile(r"<\|tool_call_start\|>(.*?)(?:<\|tool_call_end\|>|\Z)", re.S)
_XML_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)(?:</function>\s*(?:</tool_call>)?|\Z)", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?(?:</parameter>|\Z)", re.S)
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
_FENCE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]*`", re.S)
_TAIL_OK = re.compile(r"^(?:\s|</?tool_call>|<\|tool_call_end\|>)*$")


def _blank(m):
    return " " * (m.end() - m.start())


def _visible(text):
    """Blank out reasoning and quoted code with spaces. Same length as the input, so positions found
    here are positions in the original text."""
    return _FENCE.sub(_blank, _THINK.sub(_blank, text or ""))


def _spans(visible):
    spans = [(m.start(), m.end(), "py", m) for m in _PY_BLOCK.finditer(visible)]
    spans += [(m.start(), m.end(), "xml", m) for m in _XML_CALL.finditer(visible)]
    spans.sort(key=lambda t: t[0])
    if not spans:
        return []
    # Everything after the first call must be more calls or closing tags, never more prose.
    gaps = [visible[a[1]:b[0]] for a, b in zip(spans, spans[1:])] + [visible[spans[-1][1]:]]
    if not all(_TAIL_OK.match(g) for g in gaps):
        return []
    return spans


def has_call(text):
    """True when the text ends in something parse() would treat as a wrapped tool call."""
    return bool(_spans(_visible(text)))


def _value(raw, typ=None):
    """XML params arrive as text. Decode JSON only when the schema says the param is not a string; with no
    schema, decode plain numbers/booleans/null only, so Write(content='{"a": 1}') stays a string."""
    text = raw.strip("\n")
    if typ == "string":
        return text
    s = raw.strip()
    if typ is None and not re.match(r"^(?:-?\d+(?:\.\d+)?|true|false|null)$", s):
        return text
    try:
        return json.loads(s)
    except Exception:
        return text


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        # true/false/null borrowed from JSON, or a bare word meant as a string.
        if isinstance(node, ast.Name):
            return {"true": True, "false": False, "null": None, "None": None}.get(node.id, node.id)
        try:
            return ast.unparse(node)  # source text of the expression, never evaluated
        except Exception:
            return None


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):  # functions.Bash(...) -> Bash
        return func.attr
    return None


def _repair_brackets(s):
    """A truncated stream often loses the closing ) or ]. Close what was opened, outside strings."""
    stack, quote, esc = [], None, False
    for ch in s:
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            stack.append({"(": ")", "[": "]", "{": "}"}[ch])
        elif ch in ")]}" and stack and stack[-1] == ch:
            stack.pop()
    if quote:
        s += quote
    return s + "".join(reversed(stack))


def _parse_pythonic(body):
    body = body.strip()
    if not body:
        return []
    if body[0] in "[{":
        # JSON mode first.
        try:
            v = json.loads(body)
            items = v if isinstance(v, list) else [v]
            out = []
            for it in items:
                if isinstance(it, dict) and it.get("name"):
                    args = it.get("arguments", it.get("parameters", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"_raw": args}
                    out.append((it["name"], args if isinstance(args, dict) else {"_raw": args}))
            if out:
                return out
        except Exception:
            pass
    src = body if body.startswith("[") else "[" + body + "]"
    tree = None
    for cand in (src, _repair_brackets(src)):
        try:
            tree = ast.parse(cand, mode="eval")
            break
        except (SyntaxError, ValueError):  # 3.9 raises ValueError on a NUL byte
            continue
    if tree is None or not isinstance(tree.body, (ast.List, ast.Tuple)):
        return []
    out = []
    for el in tree.body.elts:
        if not isinstance(el, ast.Call):
            continue
        name = _call_name(el.func)
        if not name:
            continue
        args = {kw.arg: _literal(kw.value) for kw in el.keywords if kw.arg}
        if el.args and not args:
            # Positional only: f({"a": 1}) or f("ls"). Keep it instead of dropping the call.
            first = _literal(el.args[0])
            args = first if isinstance(first, dict) else {"_positional": [_literal(a) for a in el.args]}
        out.append((name, args))
    return out


def _parse_xml_call(m, schemas):
    name, inner = m.group(1), m.group(2)
    types = (schemas or {}).get(name) or {}
    params = {k: _value(v, types.get(k)) for k, v in _XML_PARAM.findall(inner)}
    if not params and inner.strip().startswith("{"):
        try:
            params = json.loads(inner.strip())
        except Exception:
            pass
    return [(name, params)]


def _drift_one(obj, schemas):
    """One bare JSON object -> (name, args), or None. Shapes: {"name": T, "arguments": {...}},
    {"tool": T, "args": {...}}, and {T: {...}}. T must be a tool that was offered."""
    offered = schemas
    if not isinstance(obj, dict):
        return None
    for key in ("name", "tool"):
        if isinstance(obj.get(key), str) and obj[key] in offered:
            args = {}
            for ak in ("arguments", "args", "parameters", "input"):
                if ak in obj:
                    args = obj[ak]
                    break
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    # A plain string for a tool with exactly one parameter: {"tool": "ask", "args": "why?"}
                    params = list((schemas.get(obj[key]) or {}).keys())
                    if len(params) != 1:
                        return None
                    args = {params[0]: args}
            return (obj[key], args) if isinstance(args, dict) else None
    name = next(iter(obj)) if obj else None
    if name in offered and isinstance(obj[name], dict):
        args = dict(obj[name])
        args.update({k: v for k, v in obj.items() if k != name})  # stray keys the model put beside it
        return (name, args)
    return None


def _json_drift(text, visible, schemas):
    """Without a format directive in the chat template, LFM models sometimes end a turn with a bare
    {"ToolName": {...}}, {"name": "ToolName", "arguments": {...}} or {"tool": "ToolName", "args": {...}}
    object, or a JSON array of them (Liquid4All/docs#130). Accepted ONLY when it closes out the reply
    and every call names a tool that was actually offered."""
    offered = set((schemas or {}).keys())
    if not offered:
        return None
    dec = json.JSONDecoder()
    tail = visible.rstrip()
    for i in [m.start() for m in re.finditer(r"[{\[]", tail)]:
        try:
            obj, end = dec.raw_decode(tail, i)
        except ValueError:
            continue
        if end != len(tail) or not isinstance(obj, (dict, list)):
            continue
        items = obj if isinstance(obj, list) else [obj]
        calls = [_drift_one(it, schemas) for it in items]
        if not calls or any(c is None for c in calls):
            return None
        return calls, text[:i].strip()
    return None


def parse(text, schemas=None):
    """Return ([(name, args_dict), ...], remaining_text).

    schemas = {tool_name: {param: json_type}} is optional. It keeps string params as strings, decodes
    typed XML params, and is required before bare JSON is accepted as a call. remaining_text is the
    original text before the first call (reasoning included), stripped. When nothing qualifies the
    result is ([], text) with text unchanged."""
    if not text:
        return [], text or ""
    visible = _visible(text)
    spans = _spans(visible)
    if not spans:
        drift = _json_drift(text, visible, schemas)
        return drift if drift else ([], text)
    calls = []
    for _, _, kind, m in spans:
        calls.extend(_parse_pythonic(m.group(1)) if kind == "py" else _parse_xml_call(m, schemas))
    if not calls:
        return [], text
    return calls, text[:spans[0][0]].strip()


def _json_type(prop):
    t = prop.get("type") if isinstance(prop, dict) else None
    if isinstance(t, list):
        t = [x for x in t if x != "null"]
        t = t[0] if len(t) == 1 else None
    return t if isinstance(t, str) else None


def schemas_from_openai_tools(tools):
    """Build the schemas argument for parse() from an OpenAI-style `tools` list."""
    out = {}
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        out[fn["name"]] = {k: _json_type(v) for k, v in (props or {}).items() if isinstance(k, str)}
    return out


def to_openai_tool_calls(calls, offered=None):
    """Turn parse() output into OpenAI `tool_calls` entries. When `offered` (tool names) is given, each
    name goes through canon_name() so read_file maps to Read and so on."""
    out = []
    for name, args in calls:
        if offered:
            name = canon_name(name, offered)
        out.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, default=str)},
        })
    return out


def _squash(n):
    return re.sub(r"[^a-z0-9]", "", str(n).lower())


ALIASES = {"readfile": "read", "writefile": "write", "editfile": "edit", "shell": "bash", "runshell": "bash",
           "runcommand": "bash", "listdir": "glob", "search": "grep", "findfiles": "glob"}


def canon_name(name, offered):
    """Map the name the model used to the exact tool that was offered. An exact match wins; then a
    case/underscore-blind match; then a small alias table (read_file -> Read, shell -> Bash). Only maps
    to a unique hit; otherwise returns the name unchanged so the client can report the mistake."""
    if not name or not offered or name in offered:
        return name
    want = _squash(name)
    hits = [o for o in offered if _squash(o) == want]
    if len(hits) == 1:
        return hits[0]
    alias = ALIASES.get(want)
    if alias:
        hits = [o for o in offered if _squash(o) == alias]
        if len(hits) == 1:
            return hits[0]
    return name
