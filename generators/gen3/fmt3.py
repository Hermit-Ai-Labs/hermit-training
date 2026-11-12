#!/usr/bin/env python3
"""Hermit prompt format v2 (19 tools, SYSTEM_AGENT_v2, device word), on top of gen2/hermit_fmt.py.

gen2/hermit_fmt.py is read only. This module imports it, never edits it, and adds what the next run needs:
  * TOOLS / TOOL_SCHEMA / TOOL_NAMES from spec/hermit_tools_v2.json (19 tools)
  * system_agent(bio): spec/SYSTEM_AGENT_v2.txt with {BIO} filled
  * tools_block(bio): the Qwen3 tool block for the 19 tools with "Face ID" replaced by the device word, exactly as the
    app does (it rewrites the whole prompt, tool descriptions included)
  * render / render_text (bio aware): same layout as hermit_fmt.render, tool block built here. The device word is read
    from the system message ("Touch ID" in it -> Touch ID, else Face ID) unless `bio` is passed.
  * validate_call: JSON-schema check of one tool call (types, enums, nested recipients, unknown keys). prepare_send does
    not require `chain` (the registry keeps it required because the app's first 15 lines are kept byte for byte; the
    behaviour we train is to leave it out unless the user named a network, see spec/PLAN.md section 0.2).
Everything else (calc, parse_calls, strip_calls, SYSTEM_CHAT, evm_addr, ...) is forwarded to hermit_fmt.
"""
import json, os, sys
sys.dont_write_bytecode = True      # never write .pyc next to the read-only gen2 modules we import

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
SPEC = os.path.join(LAB, "spec")
if os.path.join(LAB, "gen2") not in sys.path:
    sys.path.insert(0, os.path.join(LAB, "gen2"))
import hermit_fmt as _H                                   # gen2, read only

TOOLS_PATH = os.path.join(SPEC, "hermit_tools_v2.json")
TOOLS = json.load(open(TOOLS_PATH))
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
SYSTEM_TEMPLATE = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().strip("\n")
assert "{BIO}" in SYSTEM_TEMPLATE
SYSTEM_AGENT = SYSTEM_TEMPLATE.replace("{BIO}", "Face ID")   # compat name used by the eval harness (v2 text)
OLD_SYSTEM_AGENT = _H.SYSTEM_AGENT                        # the 1.2b-era prompt (12 tools, no price/view sentences)
SYSTEM_CHAT = _H.SYSTEM_CHAT
BIOS = ("Face ID", "Touch ID")
# registry keeps `chain` required on prepare_send (byte-identical to the app); we accept a call without it
REQUIRED_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}


def __getattr__(name):                                    # forward everything else to gen2
    return getattr(_H, name)


def system_agent(bio="Face ID"):
    return SYSTEM_TEMPLATE.replace("{BIO}", bio)


def other_bio(bio):
    return "Touch ID" if bio == "Face ID" else "Face ID"


def detect_bio(messages_or_text):
    if isinstance(messages_or_text, str):
        s = messages_or_text
    else:
        s = messages_or_text[0]["content"] if messages_or_text and messages_or_text[0]["role"] == "system" else ""
    return "Touch ID" if ("Touch ID" in s and "Face ID" not in s) else "Face ID"


def is_agent_system(s):
    """True when a system message is (an extension of) the old or the v2 agent prompt."""
    base = SYSTEM_TEMPLATE.split("{BIO}")[0]
    return s.startswith(OLD_SYSTEM_AGENT) or s.startswith(base)


def _block(bio):
    lines = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in TOOLS)
    txt = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
           "You are provided with function signatures within <tools></tools> XML tags:\n<tools>" + lines +
           "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
           "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
    return txt.replace("Face ID", bio)


_BLOCKS = {b: _block(b) for b in BIOS}


def tools_block(bio="Face ID"):
    return _BLOCKS[bio]


def render(messages, tools, family="4b", add_generation_prompt=False, bio=None):
    """List of (text, trainable). Same layout as hermit_fmt.render, with the v2 tool block for the device word."""
    bio = bio or detect_bio(messages)
    segs = []
    think = _H.THINK_EMPTY if family == "lite" else ""
    msgs = list(messages)
    if tools:
        sys_txt = (msgs[0]["content"] + "\n\n") if msgs and msgs[0]["role"] == "system" else ""
        segs.append(("<|im_start|>system\n" + sys_txt + tools_block(bio) + "<|im_end|>\n", False))
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
            segs.append((_H.assistant_text(m) + "<|im_end|>", m.get("train", True)))
            segs.append(("\n", False))
        elif r == "tool":
            pre = "" if (i > start and msgs[i - 1]["role"] == "tool") else "<|im_start|>user"
            post = "" if (i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool") else "<|im_end|>\n"
            segs.append((pre + "\n<tool_response>\n" + m["content"] + "\n</tool_response>" + post, False))
    if add_generation_prompt:
        segs.append(("<|im_start|>assistant\n" + think, False))
    return segs


def render_text(messages, tools, family="4b", add_generation_prompt=False, bio=None):
    return "".join(t for t, _ in render(messages, tools, family, add_generation_prompt, bio))


# ---------------------------------------------------------------- tool-call validation
def _type_ok(v, p):
    t = p.get("type")
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return True


def validate_args(args, sch, path=""):
    """Return a list of problems (empty = fine)."""
    errs = []
    if not isinstance(args, dict):
        return [path + "arguments is not an object"]
    props = sch.get("properties", {})
    for k, v in args.items():
        p = props.get(k)
        if p is None:
            errs.append(f"{path}unknown argument {k}")
            continue
        if not _type_ok(v, p):
            errs.append(f"{path}{k} should be {p.get('type')}")
            continue
        if p.get("enum") and v not in p["enum"]:
            errs.append(f"{path}{k} not in {p['enum']}")
        if p.get("type") == "array" and isinstance(v, list):
            it = p.get("items") or {}
            for i, x in enumerate(v):
                if it.get("type") == "object":
                    errs += validate_args(x, it, f"{path}{k}[{i}].")
                elif not _type_ok(x, it):
                    errs.append(f"{path}{k}[{i}] should be {it.get('type')}")
    return errs


def validate_call(call):
    """call = {"name":..., "arguments":...}. Returns a list of problems (empty = fine)."""
    name = call.get("name")
    sch = TOOL_SCHEMA.get(name)
    if sch is None:
        return [f"unknown tool {name}"]
    args = call.get("arguments")
    errs = validate_args(args, sch)
    if isinstance(args, dict):
        req = REQUIRED_OVERRIDE.get(name, sch.get("required", []))
        for r in req:
            if r not in args:
                errs.append(f"missing required {r}")
    return errs


if __name__ == "__main__":
    # self test: our block equals gen2's builder pointed at the v2 registry
    saved = (_H.TOOLS, _H.TOOL_NAMES, _H.TOOL_SCHEMA, _H.TOOLS_BLOCK)
    _H.TOOLS = TOOLS
    assert _H.tools_block() == tools_block("Face ID"), "block differs from gen2 builder"
    _H.TOOLS, _H.TOOL_NAMES, _H.TOOL_SCHEMA, _H.TOOLS_BLOCK = saved
    assert "Touch ID" in tools_block("Touch ID") and "Face ID" not in tools_block("Touch ID")
    print("tools", len(TOOLS), "block chars", len(tools_block("Face ID")))
    print(validate_call({"name": "prepare_send", "arguments": {"to_address": "0x1", "amount": 1, "symbol": "USDC"}}))
    print(validate_call({"name": "prepare_send", "arguments": {"to_address": "0x1", "amount": "1", "symbol": "USDC", "x": 1}}))
