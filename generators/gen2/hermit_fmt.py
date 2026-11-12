#!/usr/bin/env python3
"""Shared format for Hermit 1.2 data, training and eval.

One source of truth for:
  * the system prompts (chat-only = the app's current LocalAgent prompt; agent = with tools),
  * the tool block (hermit_tools.json rendered in Qwen3's native layout, a FIXED string that the
    shipped chat_template.jinja embeds verbatim, so no `tojson` is needed in swift-jinja),
  * render(): the exact text the shipped template produces, split into (text, trainable) segments,
  * the safe calculator used by the data generators, the tool simulator and the eval harness,
  * EVM (EIP-55) and Solana (base58) address generators.

Message shape used everywhere (OpenAI-like):
  {"role": "system"|"user"|"assistant"|"tool", "content": str,
   "tool_calls": [{"name": str, "arguments": dict}]   # assistant only, optional
  }
"""
import ast, json, math, operator, os, random, re

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(HERE, "..", "hermit_tools.json")
TOOLS = json.load(open(TOOLS_PATH))
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

# ---------------------------------------------------------------- system prompts
# The app's current chat prompt (LocalAgent.systemPrompt, 2026-11-07). Used for no-tool chat so 1.2
# behaves well inside today's app with no changes.
SYSTEM_CHAT = (
    "You are Hermit, a private AI assistant that runs entirely on this device. Nothing the user types leaves the "
    "device: there is no cloud and no server. You help with questions, writing, explaining, planning and files the user attaches.\n"
    "In chat you can't see the user's wallets or balances, move money, trade, browse the web or read email. If asked, "
    "say so plainly and point them to the Wallets page in the app for their own wallets. Never pretend you did something you can't do.\n"
    "Keep answers short and plain: a few sentences, simple words, no filler. Do not use middle dots as separators. "
    "If you don't know something, say so.")

# The agent prompt to use once the app passes tools (recommended for the app; see results_1.2.md).
SYSTEM_AGENT = (
    "You are Hermit, a private AI assistant that runs entirely on this device. You help with trading, crypto, DeFi, "
    "code, writing and everyday questions.\n"
    "You have tools for live prices, the user's balances and Hyperliquid account, position maths, and for preparing "
    "trades, swaps, sends and deposits. A prepared action opens a review screen and only happens when the user approves "
    "it with Face ID, so never say it already happened.\n"
    "Use the calculator for any arithmetic that isn't trivial. Never state a live price, balance or position you didn't "
    "get from a tool. Copy addresses and amounts exactly. If a required detail is missing, ask one short question.\n"
    "Keep answers short and plain: a few sentences, simple words, no filler. Do not use middle dots as separators. "
    "If you don't know something, say so.")

NAMES = ["Noah", "Maya", "Leo", "Aisha", "Tom", "Yuki", "Sam", "Priya", "Diego", "Nina", "Omar", "Chloe",
         "Ken", "Zara", "Ivan", "Lena", "Arjun", "Sofia", "Ben", "Mei"]
NOTES = ["I trade mostly BTC and ETH perps on Hyperliquid.", "Keep answers very short.", "I'm new to crypto, explain simply.",
         "I use Base and Arbitrum most.", "Never suggest more than 5x leverage.", "I DCA into ETH every Friday.",
         "My risk per trade is 1%.", "I prefer stablecoin yield over farming.", "Use bullet points for steps.",
         "I live in Europe, quote times in CET.", "I hold SOL long term, don't tell me to sell it."]


def with_user(base, rng):
    """Append name / hermit.md notes exactly as LocalAgent.instructions() does (sometimes)."""
    s = base
    r = rng.random()
    if r < 0.25:
        s += f"\nThe user wants to be called {rng.choice(NAMES)}."
    if rng.random() < 0.15:
        notes = rng.sample(NOTES, rng.choice([1, 2]))
        s += "\nThe user's standing notes (hermit.md). Follow them:\n" + "\n".join("- " + n for n in notes)
    return s


# ---------------------------------------------------------------- tool block (fixed)
def tools_block():
    lines = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in TOOLS)
    return ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>" + lines +
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")


TOOLS_BLOCK = tools_block()
THINK_EMPTY = "<think>\n\n</think>\n\n"


def fmt_call(name, args):
    return '<tool_call>\n{"name": "' + name + '", "arguments": ' + json.dumps(args, ensure_ascii=False) + '}\n</tool_call>'


def assistant_text(m):
    content = m.get("content") or ""
    out = content
    for i, tc in enumerate(m.get("tool_calls") or []):
        if (i == 0 and content) or i > 0:
            out += "\n"
        out += fmt_call(tc["name"], tc["arguments"])
    return out


def render(messages, tools, family="4b", add_generation_prompt=False):
    """Return list of (text, trainable). Mirrors export/chat_template_*.jinja exactly.
    family: '4b' (Qwen3-4B-Instruct-2507, no think) or 'lite' (Qwen3-1.7B: empty think block after every
    assistant prefix, which is how the app runs it)."""
    segs = []
    think = THINK_EMPTY if family == "lite" else ""
    msgs = list(messages)
    if tools:
        sys_txt = (msgs[0]["content"] + "\n\n") if msgs and msgs[0]["role"] == "system" else ""
        segs.append(("<|im_start|>system\n" + sys_txt + TOOLS_BLOCK + "<|im_end|>\n", False))
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


def render_text(messages, tools, family="4b", add_generation_prompt=False):
    return "".join(t for t, _ in render(messages, tools, family, add_generation_prompt))


# ---------------------------------------------------------------- tool-call parsing (eval + checks)
TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_calls(text):
    calls, bad, matched = [], 0, False
    for m in TC_RE.finditer(text):
        matched = True
        try:
            o = json.loads(m.group(1))
            calls.append({"name": o.get("name"), "arguments": o.get("arguments") or {}})
        except Exception:
            bad += 1
    if "<tool_call>" in text and not matched:
        bad += 1          # opened a tool call the regex couldn't close/parse
    return calls, bad


def strip_calls(text):
    return TC_RE.sub("", text).strip()


# ---------------------------------------------------------------- safe calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_FUN = {"sqrt": math.sqrt, "ln": math.log, "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
        "min": min, "max": max, "round": round, "pow": pow}


def _ev(n):
    if isinstance(n, ast.Expression):
        return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords:
        return _FUN[n.func.id](*[_ev(a) for a in n.args])
    raise ValueError("unsupported expression")


def calc(expr):
    """Evaluate; returns float/int. Accepts '^' as power, strips $ , % signs sensibly."""
    e = expr.replace("^", "**").replace("$", "").replace("×", "*").replace("÷", "/")
    e = re.sub(r"(?<=\d),(?=\d{3}\b)", "", e)          # 1,234 -> 1234
    return _ev(ast.parse(e, mode="eval"))


def calc_str(v):
    """How the calculator tool reports a result (what the app should mirror)."""
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) < 1e15):
        return str(int(v))
    s = f"{v:.10g}"
    if "e" in s:
        s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s


def calc_result(expr):
    try:
        return json.dumps({"result": calc_str(calc(expr))})
    except ZeroDivisionError:
        return json.dumps({"error": "division by zero"})
    except Exception:
        return json.dumps({"error": "could not evaluate expression"})


# ---------------------------------------------------------------- addresses
from Crypto.Hash import keccak as _keccak  # pycryptodome

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def eip55(addr_hex40):
    a = addr_hex40.lower()
    h = _keccak.new(digest_bits=256, data=a.encode()).hexdigest()
    return "0x" + "".join(c.upper() if c.isalpha() and int(h[i], 16) >= 8 else c for i, c in enumerate(a))


def evm_addr(rng, checksum=None):
    raw = "".join(rng.choice("0123456789abcdef") for _ in range(40))
    if checksum is None:
        checksum = rng.random() < 0.85
    return eip55(raw) if checksum else "0x" + raw


def b58encode(b):
    n = int.from_bytes(b, "big")
    s = ""
    while n:
        n, r = divmod(n, 58)
        s = B58[r] + s
    pad = len(b) - len(b.lstrip(b"\0"))
    return "1" * pad + s


def sol_addr(rng):
    return b58encode(bytes(rng.getrandbits(8) for _ in range(32)))


# ---------------------------------------------------------------- number formatting
def money(x, dp=2):
    return ("-" if x < 0 else "") + "$" + f"{abs(x):,.{dp}f}"


def smart_money(x):
    a = abs(x)
    if a >= 10 or a == 0:
        return money(x, 2)
    dp = 4 if a >= 0.01 else 8
    s = f"{a:.{dp}f}".rstrip("0")
    if len(s.split(".")[1]) < 2:
        s = f"{a:.2f}"
    return ("-" if x < 0 else "") + "$" + s


def num(x, dp=4):
    s = f"{x:,.{dp}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def pct(x, dp=2):
    return num(x, dp) + "%"


if __name__ == "__main__":
    rng = random.Random(0)
    print(eip55("5aaeb6053f3e94c9b9a09f33669435e7ef1beaed"))   # EIP-55 test vector
    assert eip55("5aaeb6053f3e94c9b9a09f33669435e7ef1beaed") == "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
    print(sol_addr(rng), evm_addr(rng))
    print(calc_result("(2410.5 - 2311) * 0.75"), calc_result("sqrt(2.25)"), calc_result("1/3"), calc_result("1e-7*3"))
    print(len(TOOLS_BLOCK), "chars in tool block")
