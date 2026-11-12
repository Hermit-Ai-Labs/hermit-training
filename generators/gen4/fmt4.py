#!/usr/bin/env python3
"""Hermit prompt format v3 (30 tools, tool packs, SYSTEM_AGENT_v3, device word), on top of gen3/fmt3.py.

gen2/ and gen3/ are read only; this module imports them and never edits them.
  * TOOLS / TOOL_NAMES / TOOL_SCHEMA from spec/hermit_tools_v3.json (30 tools; the first 19 are the v2 tools byte for byte)
  * system_agent(bio): spec/SYSTEM_AGENT_v3.txt with {BIO} filled
  * tools_block(pack, bio): the Qwen3 tool block with ONLY the tools of `pack`, in registry order, then
    .replace("Face ID", bio) over the whole block (the app's rule). Same header, same one-JSON-object-per-line format, same
    footer as fmt3 / gen2 hermit_fmt: only the tool lines change.
  * render(messages, pack, family) / render_text: exactly fmt3.render's layout (system + "\n\n" + block, user, assistant with
    the empty think block for family "lite" in the non-trained prefix, tool responses merged into one user turn).
    `pack`: a list of tool names (any order; rendered in registry order), "all" / True for all 30, None / [] / False for a
    chat-mode row (no tool block, system text alone).
  * system_prefix(system_text, pack, bio): the rendered system turn alone (what spec/render_golden_v3.json pins).
  * validate_call: JSON-schema check against the v3 registry (types, enums, nested arrays/objects, unknown keys, required;
    prepare_send may leave out `chain`).
Byte compatibility: render(msgs, <the 19 v2 names>, fam) == fmt3.render(msgs, True, fam) for every v2 row (self test below
and spec/render_golden_v3.json `v2_compat`).
"""
import hashlib, json, os, sys
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
SPEC = os.path.join(LAB, "spec")
for _p in (os.path.join(LAB, "gen3"), os.path.join(LAB, "gen2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hermit_fmt as _H                                   # gen2, read only (THINK_EMPTY, assistant_text, parse_calls, calc)
import fmt3 as _F3                                        # gen3, read only (validate_args, v2 system text)

TOOLS_PATH = os.path.join(SPEC, "hermit_tools_v3.json")
TOOLS = json.load(open(TOOLS_PATH))
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
TOOL_INDEX = {n: i for i, n in enumerate(TOOL_NAMES)}
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TOOL_LINE = {t["function"]["name"]: json.dumps(t, ensure_ascii=False) for t in TOOLS}
V2_NAMES = list(_F3.TOOL_NAMES)                           # the 19 v2 tools
assert TOOL_NAMES[:19] == V2_NAMES, "v3 registry must start with the 19 v2 tools in order"
assert all(json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False) for a, b in zip(TOOLS, _F3.TOOLS)), \
    "the 19 v2 tools must be byte identical in v3"

SYSTEM_TEMPLATE = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert "{BIO}" in SYSTEM_TEMPLATE
SYSTEM_TEMPLATE_V2 = _F3.SYSTEM_TEMPLATE
OLD_SYSTEM_AGENT = _F3.OLD_SYSTEM_AGENT
SYSTEM_CHAT = _H.SYSTEM_CHAT
THINK_EMPTY = _H.THINK_EMPTY
BIOS = ("Face ID", "Touch ID")
REQUIRED_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}

assistant_text = _H.assistant_text
parse_calls = _H.parse_calls
strip_calls = _H.strip_calls
calc = _H.calc
calc_str = _H.calc_str
other_bio = _F3.other_bio
detect_bio = _F3.detect_bio

HEADER = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
          "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
FOOTER = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
          "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")


def system_agent(bio="Face ID"):
    return SYSTEM_TEMPLATE.replace("{BIO}", bio)


def system_agent_v2(bio="Face ID"):
    return SYSTEM_TEMPLATE_V2.replace("{BIO}", bio)


def norm_pack(pack):
    """-> list of tool names in registry order, or [] for no tools. Unknown names raise."""
    if pack is True or pack == "all":
        return list(TOOL_NAMES)
    if not pack:
        return []
    s = set(pack)
    bad = s - set(TOOL_NAMES)
    if bad:
        raise ValueError(f"unknown tools in pack: {sorted(bad)}")
    return [n for n in TOOL_NAMES if n in s]


_BLOCK_CACHE = {}


def tools_block(pack=True, bio="Face ID"):
    names = norm_pack(pack)
    key = (tuple(names), bio)
    if key not in _BLOCK_CACHE:
        txt = HEADER + "".join("\n" + TOOL_LINE[n] for n in names) + FOOTER
        _BLOCK_CACHE[key] = txt.replace("Face ID", bio)
    return _BLOCK_CACHE[key]


def system_prefix(system_text, pack, bio="Face ID"):
    """The rendered system turn: what the app sends before the first user message."""
    names = norm_pack(pack)
    if names:
        sys_txt = (system_text + "\n\n") if system_text else ""
        return "<|im_start|>system\n" + sys_txt + tools_block(names, bio) + "<|im_end|>\n"
    return "<|im_start|>system\n" + system_text + "<|im_end|>\n"


def render(messages, pack, family="4b", add_generation_prompt=False, bio=None):
    """List of (text, trainable). Layout identical to fmt3.render; the tool block holds only the pack's tools."""
    if family not in ("4b", "lite"):
        raise ValueError("family must be 4b or lite")
    bio = bio or detect_bio(messages)
    names = norm_pack(pack)
    segs = []
    think = THINK_EMPTY if family == "lite" else ""
    msgs = list(messages)
    if names:
        sys_txt = (msgs[0]["content"] + "\n\n") if msgs and msgs[0]["role"] == "system" else ""
        segs.append(("<|im_start|>system\n" + sys_txt + tools_block(names, bio) + "<|im_end|>\n", False))
    elif msgs and msgs[0]["role"] == "system":
        segs.append(("<|im_start|>system\n" + msgs[0]["content"] + "<|im_end|>\n", False))
    start = 1 if msgs and msgs[0]["role"] == "system" else 0
    for i in range(start, len(msgs)):
        m = msgs[i]
        r = m["role"]
        if r == "user":
            segs.append(("<|im_start|>user\n" + m["content"] + "<|im_end|>\n", False))
        elif r == "assistant":
            segs.append(("<|im_start|>assistant\n" + think, False))
            segs.append((assistant_text(m) + "<|im_end|>", m.get("train", True)))
            segs.append(("\n", False))
        elif r == "tool":
            pre = "" if (i > start and msgs[i - 1]["role"] == "tool") else "<|im_start|>user"
            post = "" if (i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool") else "<|im_end|>\n"
            segs.append((pre + "\n<tool_response>\n" + m["content"] + "\n</tool_response>" + post, False))
    if add_generation_prompt:
        segs.append(("<|im_start|>assistant\n" + think, False))
    return segs


def render_text(messages, pack, family="4b", add_generation_prompt=False, bio=None):
    return "".join(t for t, _ in render(messages, pack, family, add_generation_prompt, bio))


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- tool-call validation (v3 registry)
def validate_args(args, sch, path=""):
    errs = _F3.validate_args(args, sch, path)
    # nested objects one level deeper than fmt3 walks (object properties that are objects)
    if isinstance(args, dict):
        for k, v in args.items():
            p = (sch.get("properties") or {}).get(k) or {}
            if p.get("type") == "object" and isinstance(v, dict) and p.get("properties"):
                errs += validate_args(v, p, f"{path}{k}.")
            if p.get("type") == "array" and isinstance(v, list):
                it = p.get("items") or {}
                if it.get("enum"):
                    errs += [f"{path}{k}[{i}] not in enum" for i, x in enumerate(v) if x not in it["enum"]]
                if it.get("type") == "object":
                    for i, x in enumerate(v):
                        if isinstance(x, dict):
                            for r in it.get("required", []):
                                if r not in x:
                                    errs.append(f"{path}{k}[{i}] missing required {r}")
    return errs


def validate_call(call):
    name = call.get("name")
    sch = TOOL_SCHEMA.get(name)
    if sch is None:
        return [f"unknown tool {name}"]
    args = call.get("arguments")
    errs = validate_args(args, sch)
    if isinstance(args, dict):
        for r in REQUIRED_OVERRIDE.get(name, sch.get("required", [])):
            if r not in args:
                errs.append(f"missing required {r}")
    return errs


def _selftest(n_rows=400):
    """fmt4 with the 19 v2 tools == fmt3 (v2) byte for byte, both families; blocks are subsets in registry order."""
    p = os.path.join(LAB, "data3", "sft4_train.jsonl")
    n = 0
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            for fam in ("4b", "lite"):
                a = _F3.render_text(r["messages"], r["tools"], fam, bio=r["meta"]["bio"])
                b = render_text(r["messages"], V2_NAMES if r["tools"] else None, fam, bio=r["meta"]["bio"])
                assert a == b, "fmt4 differs from fmt3 on a v2 row"
                assert _F3.render(r["messages"], r["tools"], fam, bio=r["meta"]["bio"]) == \
                    render(r["messages"], V2_NAMES if r["tools"] else None, fam, bio=r["meta"]["bio"])
            n += 1
            if n >= n_rows:
                break
    for bio in BIOS:
        assert tools_block(V2_NAMES, bio) == _F3.tools_block(bio)
        assert tools_block(list(reversed(TOOL_NAMES)), bio) == tools_block(True, bio)
    assert "Touch ID" in tools_block(True, "Touch ID") and "Face ID" not in tools_block(True, "Touch ID")
    return n


if __name__ == "__main__":
    n = _selftest()
    print(f"fmt4 self test ok: {n} sft4 rows x 2 families render byte-identical to fmt3 with the 19 v2 tools")
    print("tools", len(TOOLS), "full block chars", len(tools_block(True)), "system chars", len(system_agent()))
    print(validate_call({"name": "prepare_sell", "arguments": {"token": "WIF", "percent": 50}}))
    print(validate_call({"name": "cancel_order", "arguments": {"coin": "SOL", "kind": "stop"}}))
