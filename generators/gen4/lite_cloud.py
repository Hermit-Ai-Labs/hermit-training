#!/usr/bin/env python3
"""lite_cloud generator (sft5 generator 13): Hermit Lite (Qwen3-1.7B) parity rows, long context, copy fidelity, cloud examples.

Part 1 (150 train)  Lite trajectories: 16-tool packs (core 12 + at most one pack), at most 3 steps, answers <= 45 words,
                    90 rows with the tool in the pack and 60 TOOL-ABSENT rows (the tool the user needs is not in the pack:
                    one honest sentence + open_page, never a call to a tool that is not in the block).
Part 2 (60 / 40)    long context: 8 user turns with 450 to 750 token tool results (12 pools, 10 activity rows, 12 spot
                    rows, 16 holdings, 5 tokens) inside the 6,144 token cap; the final answer or call uses the right value.
Part 3 (50 / 60)    copy fidelity: a 42 character EVM address or a 6 decimal amount appears 3 times before the call
                    (user text, an attached file, a tool result); the call copies it character for character.
Part 4 (eval only)  eval4/cloud_examples_v3.jsonl: the 28 worked examples of spec/CLOUD_SYSTEM_v3.md as eval items for
                    eval/eval_openai.py (checks parse_ok, no_unbacked_done, exact_digits), replayed against this sim.

Records are {messages, tools, meta} like gen3, system prompt = spec/SYSTEM_AGENT_v3.txt with {BIO} (Face ID 70%, Touch ID
30%), tool block = the v3 registry restricted to the pack, rendered with family="lite" (empty think block after every
'<|im_start|>assistant\\n', generation prompt included). Every assistant sentence is an f-string filled from tool results,
user text or a calculator result; no LLM text, no number typed by hand.

Token buckets: `tokens` is the whole prompt (system + tool block + messages) counted with the Qwen tokenizer in
base/songt1-bf16; `hist_tokens` is the conversation without system and tool block. The tool block of a 16-tool pack alone
is about 2.2k tokens, so a 450+ token result puts every long row in the 2-4k or 4-6k bucket; there is no long row under 2k.

Uniqueness: the opener of every conversation and every user text of five words or more is unique inside a file, and no such
text of the eval file occurs in the train file; only short follow-up replies ("base", "the first one") may repeat. Eval items use
the held-out fifth of the templates and a quarter of the slot values (names, coins, tokens) that train never draws. Copy rows keep
`meta.copy_targets` (must occur 3 times before the call and be copied exactly) and `meta.copy_also` (copied exactly, e.g. the address
next to a 6 decimal amount). The sim (`W`) implements all 30 tools with the result shapes of spec/hermit_tools_v3.meta.json; the
cloud examples replay the document results through `world.overrides` (a result is served only when name and arguments match).

Usage: python lite_cloud.py [--seed 4] [--n 260] [--eval-seed 9413] [--eval-n 100]
Writes ../data4/lite_cloud.jsonl, ../eval4/lite_cloud.jsonl and ../eval4/cloud_examples_v3.jsonl, after a self-check of
every record (exit 1 and nothing written on any failure). Never edits gen2/ or gen3/ (imported read only).
"""
import argparse, copy, json, os, random, re, sys
from collections import Counter
from decimal import Decimal, InvalidOperation

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
SPEC = os.path.join(LAB, "spec")
for _p in (os.path.join(LAB, "gen2"), os.path.join(LAB, "gen3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hermit_fmt as H                      # noqa: E402  gen2, read only: calc, evm_addr, sol_addr, parse_calls

GEN = "lite_cloud"
# ------------------------------------------------------------------------------------------------ registry and packs
TOOLS = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
META_TOOLS = json.load(open(os.path.join(SPEC, "hermit_tools_v3.meta.json")))
SYSTEM_TEMPLATE = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert len(TOOLS) == 30 and "{BIO}" in SYSTEM_TEMPLATE
THINK = H.THINK_EMPTY
CORE = ["calculator", "get_price", "get_balances", "get_portfolio", "get_address", "get_activity",
        "prepare_send", "prepare_swap", "get_quote", "open_page", "open_app", "open_add_funds"]
LITE_EXTRA = {
    "trade": ["hyperliquid_account", "prepare_trade", "prepare_close", "cancel_order"],
    "solana": ["search_tokens", "prepare_buy", "prepare_sell", "get_positions"],
    "earn": ["get_yields", "prepare_deposit", "prepare_withdraw", "get_positions"],
    "pay": ["prepare_multisend", "add_contact", "schedule_payment", "create_automation"],
}
LITE_MAX = 16
CAP_TOKENS = 6144
RESERVE = 220            # room kept for the reply
BIOS = ("Face ID", "Touch ID")


def pack_names(pack_name):
    """Tools of a Lite pack in registry order. pack_name: 'core' or one of trade|solana|earn|pay."""
    names = set(CORE) | set(LITE_EXTRA.get(pack_name, []))
    out = [n for n in TOOL_NAMES if n in names]
    assert len(out) <= LITE_MAX
    return out


PACK_OF_TOOL = {}
for _pk, _ts in LITE_EXTRA.items():
    for _t in _ts:
        PACK_OF_TOOL.setdefault(_t, []).append(_pk)


def tools_block(pack, bio):
    byn = {t["function"]["name"]: t for t in TOOLS}
    lines = "".join("\n" + json.dumps(byn[n], ensure_ascii=False) for n in pack)
    txt = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
           "You are provided with function signatures within <tools></tools> XML tags:\n<tools>" + lines +
           "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
           "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
    return txt.replace("Face ID", bio)


def system_agent(bio, name=None, notes=None):
    s = SYSTEM_TEMPLATE.replace("{BIO}", bio)
    if name:
        s += f"\nThe user wants to be called {name}."
    if notes:
        s += "\nThe user's standing notes (hermit.md). Follow them:\n" + "\n".join("- " + n for n in notes)
    return s


def render(messages, pack, bio, family="lite", gen_prompt=False):
    """List of (text, trainable). Same layout as gen2 hermit_fmt.render; tool block = the pack; family lite adds the think block."""
    think = THINK if family == "lite" else ""
    segs = []
    msgs = list(messages)
    sys_txt = msgs[0]["content"] + "\n\n"
    segs.append(("<|im_start|>system\n" + sys_txt + tools_block(pack, bio) + "<|im_end|>\n", False))
    for i in range(1, len(msgs)):
        m = msgs[i]
        r = m["role"]
        if r == "user":
            segs.append(("<|im_start|>user\n" + m["content"] + "<|im_end|>\n", False))
        elif r == "assistant":
            segs.append(("<|im_start|>assistant\n" + think, False))
            segs.append((H.assistant_text(m) + "<|im_end|>", True))
            segs.append(("\n", False))
        elif r == "tool":
            pre = "" if msgs[i - 1]["role"] == "tool" else "<|im_start|>user"
            post = "" if (i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool") else "<|im_end|>\n"
            segs.append((pre + "\n<tool_response>\n" + m["content"] + "\n</tool_response>" + post, False))
    if gen_prompt:
        segs.append(("<|im_start|>assistant\n" + think, False))
    return segs


def render_text(messages, pack, bio, family="lite", gen_prompt=False):
    return "".join(t for t, _ in render(messages, pack, bio, family, gen_prompt))


# ------------------------------------------------------------------------------------------------ tokenizer
_TOK = None


def tok_len(text):
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(os.path.join(LAB, "base", "songt1-bf16", "tokenizer.json"))
    return len(_TOK.encode(text).ids)


# ------------------------------------------------------------------------------------------------ small helpers
def norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def jd(r):
    return json.dumps(r, ensure_ascii=False)


def srt(d):
    """Recursively sort dict keys (the app's toolbox results use alphabetical keys)."""
    if isinstance(d, dict):
        return {k: srt(d[k]) for k in sorted(d)}
    if isinstance(d, list):
        return [srt(x) for x in d]
    return d


def plain(x):
    """Shortest plain decimal text of a number (no exponent): 22.0 -> 22, 0.0000098 -> 0.0000098."""
    if isinstance(x, str):
        return x
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        x = int(x)
    d = Decimal(repr(x)) if isinstance(x, float) else Decimal(x)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def short(x):
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def usd(x):
    """$ amount with the exact digits of x (two decimals unless whole), thousands commas."""
    d = Decimal(plain(abs(x)))
    s = format(d, "f")
    ip, _, fp = s.partition(".")
    if fp and len(fp) < 2:
        fp = fp.ljust(2, "0")
    return ("-" if x < 0 else "") + "$" + f"{int(ip):,}" + ("." + fp if fp else "")


def commas(x):
    d = Decimal(plain(x))
    ip, _, fp = format(d, "f").partition(".")
    return f"{int(ip):,}" + ("." + fp if fp else "")


def big(x):
    """$38M / $1.9B / $3,900 style for TVL, liquidity and volume (exact when the value has <= 3 significant digits)."""
    x = float(x)
    if x >= 1e9:
        return "$" + plain(round(x / 1e9, 2)) + "B"
    if x >= 1e6:
        return "$" + plain(round(x / 1e6, 2)) + "M"
    return usd(x)


def approx(x):
    """The app's Units.approx: >= 1000 -> 2 dp; else 3 significant digits, trailing zeros trimmed."""
    x = float(x)
    if x >= 1000:
        return plain(round(x, 2))
    if x == 0:
        return "0"
    from math import floor, log10
    dp = max(0, 2 - int(floor(log10(abs(x)))))
    return plain(round(x, dp))


def join_and(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def cap1(s):
    return s[:1].upper() + s[1:]


CHAIN_NAME = {"base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "solana": "Solana", "bnb": "BNB Chain",
              "bsc": "BNB Chain", "polygon": "Polygon", "near": "NEAR", "monad": "Monad", "hyperliquid": "Hyperliquid"}
CHAIN_WORDS = {"base": ["base"], "arbitrum": ["arbitrum", "arb "], "ethereum": ["ethereum", "mainnet", "ethereum mainnet"],
               "solana": ["solana"], "bnb": ["bnb", "bsc", "bnb chain"], "polygon": ["polygon"]}
EVM_NETS = ["ethereum", "base", "arbitrum", "bnb", "polygon", "monad"]
NET_NAME = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "bnb": "BNB Chain", "polygon": "Polygon",
            "monad": "Monad", "solana": "Solana", "near": "NEAR"}
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
EVM_RE = re.compile(r"0x[0-9a-fA-F]{40}")
B58_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")
FILE_RE = re.compile(r"<file>.*?</file>", re.S)


def strip_files(t):
    return FILE_RE.sub("", t)


def chain_named(text, chain):
    t = " " + text.lower() + " "
    return any(w in t for w in CHAIN_WORDS.get(chain, [chain]))


# ================================================================================================ simulator (all 30 tools)
ASSETS = {"BTC": (62000, 112000, 1), "ETH": (2400, 4600, 2), "SOL": (110, 260, 2), "HYPE": (18, 52, 3), "BNB": (480, 980, 2),
          "DOGE": (0.12, 0.34, 4), "XRP": (1.6, 3.4, 4), "AVAX": (18, 52, 2), "LINK": (11, 28, 3), "SUI": (2, 5.4, 4),
          "ENA": (0.3, 0.95, 4), "ARB": (0.3, 1.1, 4), "OP": (0.5, 2.1, 3), "NEAR": (2.2, 6.5, 3), "INJ": (8, 30, 3),
          "TAO": (200, 560, 2)}
MAXLEV = {"BTC": 40, "ETH": 25, "SOL": 20, "XRP": 20, "DOGE": 10, "HYPE": 10, "BNB": 10, "AVAX": 10, "LINK": 10, "SUI": 10,
          "ENA": 10, "ARB": 10, "OP": 10, "NEAR": 10, "INJ": 10, "TAO": 5}
MEMES = {"WIF": 1.82, "JUP": 0.78, "POPCAT": 0.52, "PUMP": 0.0043, "MOODENG": 0.21, "GOAT": 0.12, "FARTCOIN": 0.41, "PENGU": 0.0081}
NATIVE = {"ethereum": "ETH", "base": "ETH", "arbitrum": "ETH", "solana": "SOL", "bnb": "BNB", "polygon": "POL", "near": "NEAR"}
DAPPS = {"hyperliquid": "https://app.hyperliquid.xyz", "aave": "https://app.aave.com", "pendle": "https://app.pendle.finance",
         "ethena": "https://app.ethena.fi", "jupiter": "https://jup.ag", "uniswap": "https://app.uniswap.org",
         "lido": "https://stake.lido.fi", "morpho": "https://app.morpho.org", "kamino": "https://app.kamino.finance",
         "aerodrome": "https://aerodrome.finance", "polymarket": "https://polymarket.com"}
PAGES = {"home": "Home", "wallets": "Wallets", "trade": "Trade", "earn": "Earn", "payments": "Payments", "explore": "Explore",
         "calendar": "Calendar", "projects": "Projects", "automations": "Automations", "files": "Files",
         "connectors": "Connectors", "settings": "Settings"}
TABS = {"trade": ["markets", "positions"], "earn": ["save", "positions", "explore"],
        "payments": ["schedule", "payees", "transfer", "apis"], "settings": ["profile", "safety", "memory", "models", "cloud"]}
BAD_NOTE = "Nothing was prepared. Tell the user why."
FACE = "Waiting for the user to approve with Face ID."
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TODAY = "2026-11-11"          # a Wednesday


def liq_price(side, entry, lev, mm=0.0):
    if side == "long":
        return entry * (1 - 1 / lev) / (1 - mm)
    return entry * (1 + 1 / lev) / (1 + mm)


def short_addr(a):
    return f"{a[:6]}… {a[-4:]}"


def addr_family(a):
    if re.fullmatch(r"0x[0-9a-fA-F]{40}", a or ""):
        return "evm"
    if re.fullmatch(r"[a-z0-9_-]+\.near", a or "") or re.fullmatch(r"[0-9a-f]{64}", a or ""):
        return "near"
    if re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", a or ""):
        return "solana"
    return None


class W:
    """One user's state plus the app's tool results (exact keys of spec/hermit_tools_v3.meta.json).

    Every handler is a pure function of the state (no rng at call time) so a recorded conversation can be replayed from
    meta.world byte for byte. State that changes during a chat: typed (user text), searched, resolved, fail (one shot)."""

    def __init__(self, d):
        self.d = copy.deepcopy(d)

    @classmethod
    def from_state(cls, d):
        return cls(d)

    def state(self):
        return json.loads(json.dumps(self.d))

    @classmethod
    def fresh(cls, rng, device="iPhone"):
        d = {"today": TODAY, "device": device, "wallet": "Main", "other_wallets": [], "near_account": None,
             "prices": {}, "changes": {}, "holdings": [], "hl": {"value": 0, "positions": [], "orders": []},
             "spot": [], "impact": {}, "tokens": [], "earn": [], "aave_apy": {"ethereum": 3.1, "base": 4.21, "arbitrum": 3.9},
             "yields": [], "markets": [], "contacts": [], "activity": [], "own": {}, "quote": {"seconds": 14, "fee_bps": 35, "spread": 0.07},
             "cap_usd": 5000, "automations": 0, "typed": "", "searched": [], "resolved": [], "fail": {}, "no_route": [],
             "unreachable": [], "overrides": [], "route_buy": "Jupiter via Orca, Raydium", "route_sell": "Jupiter via Orca, Raydium"}
        for s_, (lo, hi, dp) in ASSETS.items():
            d["prices"][s_] = round(rng.uniform(lo, hi), dp)
            d["changes"][s_] = round(rng.uniform(-6.5, 6.5), 2)
        for s_, p in MEMES.items():
            d["prices"][s_] = float(f"{p * rng.uniform(0.85, 1.2):.3g}")
            d["changes"][s_] = round(rng.uniform(-9, 9), 1)
        for s_ in ("USDC", "USDT"):
            d["prices"][s_] = 1.0
        d["prices"]["POL"] = round(rng.uniform(0.2, 0.6), 4)
        d["own"] = {"evm": H.evm_addr(rng), "solana": H.sol_addr(rng), "near": rng.choice(["maya", "noah", "leo7", "nina", "omar"]) + ".near"}
        return cls(d)

    # -------------------------------------------------------------------------------- builders
    def price(self, sym):
        return self.d["prices"].get(sym.upper())

    def add_hold(self, sym, chain, amount):
        amt = plain(amount) if not isinstance(amount, str) else amount
        usd_ = round(float(Decimal(amt)) * self.d["prices"].get(sym, 1.0), 2)
        self.d["holdings"] = [h for h in self.d["holdings"] if not (h["symbol"] == sym and h["chain"] == chain)]
        self.d["holdings"].append({"amount": amt, "chain": chain, "symbol": sym, "usd": usd_})

    def hold(self, sym, chain=None):
        return [h for h in self.d["holdings"] if h["symbol"] == sym and (chain is None or h["chain"] == chain) and float(h["amount"]) > 0]

    def add_position(self, coin, side, size, entry, lev, pnl=None):
        mark = self.d["prices"][coin]
        sgn = 1 if side == "long" else -1
        dp = ASSETS[coin][2]
        upnl = round((mark - entry) * size * sgn, 2) if pnl is None else pnl
        self.d["hl"]["positions"].append({"coin": coin, "side": side, "size": short(size), "entry": short(entry), "leverage": lev,
                                          "liquidation": round(liq_price(side, entry, lev, 0.005), dp),
                                          "unrealised_pnl": short(upnl), "margin": round(entry * size / lev, 2)})

    def add_order(self, oid, coin, kind, price, side, size, whole=False):
        self.d["hl"]["orders"].append({"id": str(oid), "coin": coin, "kind": kind, "price": short(price), "side": side,
                                       "size": short(size), "whole_position": whole})

    def add_spot(self, sym, mint, amount, avg_cost=None, frozen=False, price=None):
        p = price if price is not None else self.d["prices"].get(sym, 1.0)
        amt = plain(amount)
        usd_ = round(float(Decimal(amt)) * p, 2)
        pnl = None if avg_cost is None else round((p / avg_cost - 1) * 100, 1)
        self.d["spot"].append({"symbol": sym, "mint": mint, "amount": amt, "usd": usd_, "avg_cost": avg_cost, "frozen": frozen, "pnl_pct": pnl})

    def pos(self, coin):
        for p in self.d["hl"]["positions"]:
            if p["coin"] == coin:
                return p
        return None

    # -------------------------------------------------------------------------------- dispatch
    def call(self, name, args):
        d = self.d
        if not isinstance(args, dict):
            args = {}
        for i, o in enumerate(d["overrides"]):
            if o["name"] == name and o["args"] == args:
                d["overrides"].pop(i)
                return o["result"]
        if name in d["fail"]:
            f = d["fail"][name]
            if isinstance(f, list):
                msg = f.pop(0)
                if not f:
                    del d["fail"][name]
            else:
                msg = d["fail"].pop(name)
            return {"error": msg}
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError, OverflowError, InvalidOperation) as e:
            return {"error": f"bad arguments: {type(e).__name__}"}

    def typed(self):
        return self.d["typed"]

    # -------------------------------------------------------------------------------- reads
    def t_calculator(self, expression):
        try:
            return {"result": H.calc_str(H.calc(expression))}
        except Exception:
            return {"error": "Couldn't evaluate that expression"}

    def t_get_price(self, symbol):
        s_ = str(symbol).upper().replace("XYZ:", "")
        if s_ not in self.d["prices"] or s_ in ("USDC", "USDT"):
            return {"error": f"No Hyperliquid market for {s_}. Answer from what you know instead, and say there's no live price here."}
        return srt({"change_24h_pct": self.d["changes"].get(s_), "price": self.d["prices"][s_], "source": "Hyperliquid mark price", "symbol": s_})

    def t_get_balances(self, chain=None):
        d = self.d
        c = {"bsc": "bnb"}.get(chain, chain)
        hs = [h for h in d["holdings"] if float(h["amount"]) > 0 and (c is None or h["chain"] == c)]
        r = {"holdings": [srt(h) for h in hs]}
        if d["near_account"]:
            r["near_account"] = d["near_account"]
        if d["unreachable"]:
            r["unreachable"] = list(d["unreachable"])
        if d["other_wallets"]:
            r["note"] = f"Showing the selected wallet ({d['wallet']}). Other wallets: {', '.join(d['other_wallets'])}."
            r["wallet"] = d["wallet"]
        return srt(r)

    def _hl_numbers(self):
        hl = self.d["hl"]
        upnl = sum(p["unrealised_pnl"] for p in hl["positions"])
        value = round(hl["value"] + upnl, 2)
        margin = round(sum(p["margin"] for p in hl["positions"]), 2)
        return value, margin, round(max(0, value - margin), 2)

    def t_hyperliquid_account(self):
        hl = self.d["hl"]
        value, margin, avail = self._hl_numbers()
        trig = any(o["kind"] != "limit" for o in hl["orders"])
        orders = []
        for o in hl["orders"]:
            if not trig:
                orders.append({"coin": o["coin"], "price": o["price"], "side": o["side"], "size": o["size"]})
            elif o["kind"] == "limit":
                orders.append({"coin": o["coin"], "is_trigger": False, "order_type": "Limit", "price": o["price"], "side": o["side"], "size": o["size"]})
            else:
                below = o["kind"] == "stop_loss"
                orders.append({"coin": o["coin"], "is_trigger": True, "order_type": "Stop Market" if below else "Take Profit Market",
                               "price": o["price"], "side": o["side"], "size": o["size"],
                               "trigger_condition": f"Price {'below' if (o['side'] == 'sell') == below else 'above'} {plain(o['price'])}",
                               "whole_position": bool(o["whole_position"])})
        poss = [{k: p[k] for k in ("coin", "entry", "leverage", "liquidation", "side", "size", "unrealised_pnl")} for p in hl["positions"]]
        r = {"account_value": value, "available_usdc": avail, "margin_used": margin, "open_orders": orders, "positions": poss}
        if trig:
            r["account_mode"] = "unified"
        return srt(r)

    def t_position_math(self, side, entry, size_usd, leverage, price=None, maintenance_margin=0):
        liq = liq_price(side, entry, leverage, maintenance_margin)
        ref = price if price is not None else entry
        r = {"distance_to_liq_pct": short(round(abs(ref - liq) / ref * 100, 2)), "entry": short(entry), "leverage": short(leverage),
             "liquidation_price": short(round(liq, 2)), "margin": short(round(size_usd / leverage, 2)),
             "quantity": short(round(size_usd / entry, 6)), "side": side, "size_usd": short(size_usd)}
        if price is not None:
            sgn = 1 if side == "long" else -1
            pnl = (price - entry) / entry * size_usd * sgn
            r["pnl"] = short(round(pnl, 2))
            r["price"] = short(price)
            r["roe_pct"] = short(round(pnl / (size_usd / leverage) * 100, 2))
        return srt(r)

    def t_search_markets(self, query):
        q = str(query).upper()
        rows = [m for m in self.d["markets"] if q in m["name"].upper()]
        return {"markets": [srt(m) for m in rows]}

    def t_get_yields(self, asset=None, chain=None):
        rows = [p for p in self.d["yields"] if (not asset or p["asset"].upper() == str(asset).upper())
                and (not chain or p["chain"].lower() == str(chain).lower())]
        rows = sorted(rows, key=lambda p: -p["apy_pct"])
        return {"pools": [srt(p) for p in rows]}

    def t_get_activity(self, limit=10):
        rows = self.d["activity"][:int(limit)]
        inp = sum(1 for r in rows if r["status"] == "in_progress")
        note = "Reviews the user has not approved yet do not appear here."
        if not self.d["activity"]:
            return {"activity": [], "in_progress": 0, "note": "No activity yet. " + note}
        return srt({"activity": rows, "in_progress": inp, "note": note})

    def t_get_portfolio(self):
        d = self.d
        wallet = round(sum(h["usd"] for h in d["holdings"]), 2)
        spot = round(sum(s_["usd"] for s_ in d["spot"]), 2)
        aave = round(sum(float(e["amount"]) * d["prices"].get(e["symbol"], 1) for e in d["earn"]), 2)
        hlv = self._hl_numbers()[0]
        places = [("Wallet", wallet), ("Spot tokens", spot), ("Aave", aave), ("Hyperliquid", hlv)]
        by = [{"place": p, "usd": u} for p, u in places if u > 0]
        rows = []
        for h in d["holdings"]:
            rows.append({"place": "Wallet", "usd": h["usd"], "what": h["symbol"] if h["symbol"] == NATIVE.get(h["chain"]) and h["chain"] == "solana" else f"{h['symbol']} on {CHAIN_NAME[h['chain']]}"})
        for s_ in d["spot"]:
            rows.append({"place": "Spot tokens", "usd": s_["usd"], "what": s_["symbol"]})
        for e in d["earn"]:
            rows.append({"place": "Aave", "usd": round(float(e["amount"]) * d["prices"].get(e["symbol"], 1), 2), "what": f"{e['symbol']} on {CHAIN_NAME[e['chain']]}"})
        if hlv > 0:
            rows.append({"place": "Hyperliquid", "usd": hlv, "what": "Account value"})
        rows = sorted([r for r in rows if r["usd"] > 0], key=lambda r: -r["usd"])[:3]
        r = {"by_place": by, "top": rows, "total_usd": round(sum(u for _, u in places), 2), "wallet": d["wallet"]}
        if d["unreachable"]:
            r["unreachable"] = list(d["unreachable"])
        return srt(r)

    def t_get_positions(self, kind="all"):
        d = self.d
        spot = [srt({k: s_[k] for k in ("amount", "avg_cost", "frozen", "mint", "pnl_pct", "symbol", "usd")}) for s_ in d["spot"]]
        earn = [srt({"amount": e["amount"], "apy_pct": e["apy_pct"], "chain": e["chain"], "symbol": e["symbol"]}) for e in d["earn"]]
        orders = [srt({k: o[k] for k in ("coin", "id", "kind", "price", "side", "size")} | ({"whole_position": True} if o["whole_position"] else {})) for o in d["hl"]["orders"]]
        poss = [srt({k: p[k] for k in ("coin", "entry", "leverage", "liquidation", "side", "size", "unrealised_pnl")}) for p in d["hl"]["positions"]]
        perps = {"open_orders": orders, "positions": poss}
        full = {"earn": earn, "perps": perps, "spot": spot}
        k = kind or "all"
        r = full if k == "all" else {k: full[k]}
        r["wallet"] = d["wallet"]
        if d["unreachable"]:
            r["unreachable"] = list(d["unreachable"])
        return srt(r)

    def _tok_row(self, t):
        r = {k: t[k] for k in ("change_24h_pct", "liquidity_usd", "mint", "name", "price", "symbol", "tags", "verified", "volume_24h_usd")}
        if t["liquidity_usd"] < 5000:
            r["refusal"] = "Liquidity is under $5K, so Hermit won't buy it."
        elif not t["verified"] and "Freeze on" in t["tags"]:
            r["refusal"] = "It is unverified with freeze authority on, so Hermit won't buy it."
        return srt(r)

    def t_search_tokens(self, query):
        q = str(query).strip().lower()
        hits = [t for t in self.d["tokens"] if q and (q == t["mint"].lower() or q in t["symbol"].lower() or q in t["name"].lower())]
        hits = sorted(hits, key=lambda t: -t["liquidity_usd"])[:5]
        if not hits:
            return {"note": "No Solana token matched. Check the spelling or paste the mint.", "tokens": []}
        for t in hits:
            if t["mint"] not in self.d["searched"]:
                self.d["searched"].append(t["mint"])
        return {"tokens": [self._tok_row(t) for t in hits]}

    def t_get_address(self, contact=None, chain=None):
        d = self.d
        if contact:
            q = str(contact).strip().lower()
            ms = [c for c in d["contacts"] if c["name"].lower() == q] or [c for c in d["contacts"] if c["name"].lower().split()[0] == q or c["name"].lower().startswith(q)]
            if not ms:
                return {"note": f"No saved contact called {contact}. Ask the user for the address.", "status": "not_found"}
            if len(ms) > 1:
                names = [c["name"] for c in ms]
                return srt({"note": "Nothing was returned. Ask the user which one.", "options": names,
                            "question": f"Which {contact}? {' or '.join(names)}.", "status": "needs_choice"})
            c = ms[0]
            if c["address"] not in d["resolved"]:
                d["resolved"].append(c["address"])
            return {"contacts": [srt({"address": c["address"], "family": c["family"], "name": c["name"],
                                      "networks": EVM_NETS if c["family"] == "evm" else [c["family"]]})]}
        fam = None
        if chain:
            ch = {"bsc": "bnb"}.get(chain, chain)
            fam = "solana" if ch == "solana" else "near" if ch == "near" else "evm"
        rows = []
        for f_, nets in (("evm", EVM_NETS), ("solana", ["solana"]), ("near", ["near"])):
            if fam and f_ != fam:
                continue
            rows.append({"address": d["own"][f_], "family": f_, "networks": nets, "wallet": d["wallet"]})
        return srt({"addresses": rows, "note": "Only send tokens on these networks to these addresses. The QR code is in Add funds."})

    def _quote_core(self, fs, fc, ts, tc, amount):
        d = self.d
        fs, ts = fs.upper(), ts.upper()
        pf, pt = d["prices"].get(fs), d["prices"].get(ts)
        if pf is None or pt is None:
            return {"error": "That pair isn't supported on this route."}
        usd_in = round(float(amount) * pf, 2)
        if usd_in < 1:
            return {"error": "Amount is below the minimum for this route (about $1)."}
        q = d["quote"]
        cost = round(usd_in * q["fee_bps"] / 1e4 + q["spread"], 2)
        usd_out = round(usd_in - cost, 2)
        out = usd_out / pt
        out_txt = f"{out:.2f}" if pt == 1.0 else approx(out)
        if pt == 1.0:
            out_txt = plain(round(out, 2))
        return {"amount_in_usd": short(usd_in), "amount_out": out_txt, "amount_out_usd": short(usd_out), "fee_bps": q["fee_bps"],
                "from": f"{plain(amount)} {fs} on {CHAIN_NAME.get(fc, fc)}", "route": "NEAR Intents", "seconds": q["seconds"],
                "signature_verified": True, "to": f"about {out_txt} {ts} on {CHAIN_NAME.get(tc, tc)}", "total_cost_usd": short(cost)}

    def t_get_quote(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        return srt(self._quote_core(from_symbol, from_chain, to_symbol, to_chain, amount))

    # -------------------------------------------------------------------------------- opens
    def t_open_add_funds(self, goal):
        return srt({"goal": goal, "note": "The user taps it to open Add funds.", "screen": "add_funds", "status": "button_shown"})

    def t_open_page(self, page, tab=None):
        if page not in PAGES:
            return {"error": "That page can't be opened from chat."}
        if tab is not None and tab not in TABS.get(page, []):
            return {"error": f"{PAGES[page]} has no tab called {tab}. Tabs: {', '.join(TABS.get(page, [])) or 'none'}."}
        r = {"note": f"The user taps it to open {PAGES[page]}.", "page": page, "screen": page, "status": "button_shown"}
        if tab:
            r["tab"] = tab
        return srt(r)

    def t_open_app(self, name=None, url=None):
        if url:
            if not re.match(r"https?://", url):
                return {"error": "Only http and https addresses can be opened."}
            return srt({"note": "Opened in Hermit's browser. Nothing connects to the wallet until the user approves it.", "opened": url, "status": "opened"})
        key = (name or "").strip().lower()
        if key in DAPPS:
            return srt({"note": "Opened in Hermit's browser. Nothing connects to the wallet until the user approves it.", "opened": DAPPS[key], "status": "opened"})
        return {"note": f"No dApp called {name} in Explore. Ask the user for the web address.", "status": "not_found"}

    # -------------------------------------------------------------------------------- prepares
    def _blocked(self, usd_amt):
        if usd_amt > self.d["cap_usd"]:
            return {"note": BAD_NOTE, "reasons": ["Amount is above your daily limit for the assistant."], "status": "blocked"}
        return None

    def _needs_chain(self, what, sym, hs):
        names = [CHAIN_NAME[h["chain"]] for h in hs]
        det = ", ".join(f"{CHAIN_NAME[h['chain']]}: {h['amount']}" for h in hs)
        return srt({"note": "Nothing was prepared. Ask the user which network, in one short question.", "options": [h["chain"] for h in hs],
                    "prepared": False, "question": f"Which network should the {what} use? You hold {sym} on {join_and(names)} ({det}).", "status": "needs_chain"})

    def _pick(self, sym, chain, families, what, amount):
        """App rules for chain-less send/deposit/payment. Returns (holding, result_or_None)."""
        typed = strip_files(self.d["typed"])
        if chain:
            chain = {"bsc": "bnb"}.get(chain, chain)
            if not chain_named(typed, chain):
                chain = None                                   # ToolGuard.adjusted drops a chain the user never named
        cands = [h for h in self.hold(sym) if h["chain"] in families]
        if chain:
            own = [h for h in cands if h["chain"] == chain]
            if own:
                h = own[0]
            elif len(cands) == 1:
                h = cands[0]
            else:
                return None, self._needs_chain(what, sym, cands)
        else:
            if len(cands) >= 2:
                return None, self._needs_chain(what, sym, cands)
            if not cands:
                return None, srt({"holding": "none", "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"})
            h = cands[0]
        if Decimal(plain(amount)) > Decimal(h["amount"]):
            return None, srt({"holding": f"{h['amount']} {sym} on {CHAIN_NAME[h['chain']]}", "note": "The user holds less than that. Tell them.",
                              "prepared": False, "status": "insufficient"})
        return h, None

    def _addr_ok(self, a):
        return a in strip_files(self.d["typed"]) or a in self.d["resolved"]

    def t_prepare_send(self, to_address, amount, symbol, chain=None):
        d = self.d
        if not self._addr_ok(to_address):
            return {"error": "Refused: the user didn't type that address in this chat. Only send to an address the user gave you themselves. Ask them for it."}
        sym = symbol.upper()
        fam = addr_family(to_address)
        families = {"evm": [c for c in EVM_NETS if c != "monad"], "solana": ["solana"], "near": ["near"]}.get(fam, [])
        h, err = self._pick(sym, chain, families, "send", amount)
        if err:
            return err
        b = self._blocked(float(amount) * d["prices"].get(sym, 1))
        if b:
            return b
        return srt({"amount": f"{plain(amount)} {sym}", "executed": False, "kind": "send", "network": CHAIN_NAME[h["chain"]],
                    "note": FACE, "status": "review_opened"})

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        fc = {"bsc": "bnb"}.get(from_chain, from_chain)
        hs = self.hold(from_symbol.upper(), fc)
        if not hs or Decimal(plain(amount)) > Decimal(hs[0]["amount"]):
            have = hs[0]["amount"] if hs else "0"
            return srt({"holding": f"{have} {from_symbol.upper()} on {CHAIN_NAME.get(fc, fc)}", "note": "The user holds less than that. Tell them.",
                        "prepared": False, "status": "insufficient"})
        q = self._quote_core(from_symbol, from_chain, to_symbol, to_chain, amount)
        if "error" in q:
            return q
        b = self._blocked(q["amount_in_usd"])
        if b:
            return b
        q.update({"note": "Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels.",
                  "policy": "allow", "status": "prepared_for_review"})
        return srt(q)

    def t_prepare_trade(self, coin, side, size_usd, leverage, order_type, limit_price=None, take_profit=None, stop_loss=None):
        c = str(coin).upper()
        if c not in ASSETS:
            return {"error": f"{c} isn't tradable on Hyperliquid"}
        if order_type == "limit" and limit_price is None:
            return {"error": "limit_price is required for a limit order.", "prepared": False}
        lev = min(leverage, MAXLEV[c])
        value, margin, avail = self._hl_numbers()
        need = size_usd / lev
        if avail < need:
            return {"error": f"Not enough margin: {avail:.2f} USDC available, {need:.2f} needed", "prepared": False}
        b = self._blocked(size_usd) and {"note": BAD_NOTE, "reasons": ["This trade is above your daily limit for the assistant."], "status": "blocked"}
        if b:
            return b
        return {"status": "review_opened", "kind": "trade", "executed": False, "note": FACE}

    def t_prepare_close(self, coin, percent=None):
        c = str(coin).upper()
        p = self.pos(c)
        if not p:
            return {"note": f"The user has no open {c} position on Hyperliquid. Tell them.", "prepared": False, "status": "no_position"}
        pc = 100 if percent is None else percent
        size = Decimal(plain(p["size"]))
        part = size * Decimal(plain(pc)) / 100
        who = f"{c} {p['side']}"
        closing = f"all of the {who}, {plain(p['size'])} {c}" if pc == 100 else f"{plain(pc)}% of the {who}, {plain(float(part))} {c}"
        return srt({"closing": closing, "executed": False, "kind": "close", "note": FACE, "order": "Market, reduce only",
                    "status": "review_opened", "unrealised_pnl": short(round(p["unrealised_pnl"] * pc / 100, 2))})

    def _order_text(self, o):
        c = o["coin"]
        if o["kind"] == "stop_loss":
            return f"{c} stop loss at {usd(o['price'])}"
        if o["kind"] == "take_profit":
            return f"{c} take profit at {usd(o['price'])}"
        return f"{c} {o['side']} limit, {plain(o['size'])} {c} at {usd(o['price'])}"

    def t_cancel_order(self, coin, kind=None, order_id=None):
        d = self.d
        orders = d["hl"]["orders"]
        c = str(coin).upper() if str(coin).lower() != "all" else "all"
        kmap = {"order": "limit", "take_profit": "take_profit", "stop_loss": "stop_loss"}
        if order_id:
            ms = [o for o in orders if o["id"] == str(order_id)]
        elif c == "all":
            ms = [o for o in orders if o["kind"] == "limit"]
            if not ms:
                return {"note": "The user has no open limit orders. Tell them.", "prepared": False, "status": "nothing_to_cancel"}
            coins = []
            for o in ms:
                if o["coin"] not in coins:
                    coins.append(o["coin"])
            return srt({"cancelling": f"{len(ms)} open limit orders ({', '.join(coins)})", "executed": False, "kind": "cancel_order",
                        "note": "Waiting for the user to tap Confirm. Nothing is cancelled yet.", "status": "card_opened"})
        else:
            ms = [o for o in orders if o["coin"] == c and (kind is None or o["kind"] == kmap.get(kind))]
        if not ms:
            what = {"stop_loss": "stop loss", "take_profit": "take profit", "order": "limit order"}.get(kind)
            return {"note": f"The user has no open {c} {what} order. Tell them." if what else f"The user has no open {c} orders. Tell them.",
                    "prepared": False, "status": "nothing_to_cancel"}
        if len(ms) > 1 and not order_id and kind is None:
            opts = [f"{o['kind']} at {usd(o['price'])}" for o in ms]
            words = [f"{'a ' + o['kind'].replace('_', ' ')} at {usd(o['price'])}" for o in ms]
            return srt({"note": "Nothing was prepared. Ask the user which one.", "options": opts, "prepared": False,
                        "question": f"Which {c} order? {' or '.join(cap1(w) if i == 0 else w for i, w in enumerate(words))}.", "status": "needs_choice"})
        o = ms[0]
        r = {"cancelling": self._order_text(o), "executed": False, "kind": "cancel_order",
             "note": "Waiting for the user to tap Confirm. Nothing is cancelled yet.", "status": "card_opened"}
        if o["kind"] == "stop_loss":
            p = self.pos(o["coin"])
            r["warning"] = f"Without it the {o['coin']} {p['side'] if p else 'position'} has no stop loss."
        return srt(r)

    def t_set_tp_sl(self, coin, take_profit=None, stop_loss=None):
        d = self.d
        c = str(coin).upper()
        p = self.pos(c)
        if not p:
            return {"note": f"The user has no open {c} position on Hyperliquid. Tell them.", "prepared": False, "status": "no_position"}
        if take_profit is None and stop_loss is None:
            return {"error": "Give a take profit or a stop loss.", "prepared": False}
        mark = d["prices"][c]
        long_ = p["side"] == "long"
        if stop_loss is not None and ((stop_loss >= mark) if long_ else (stop_loss <= mark)):
            return {"error": f"A stop loss for a {p['side']} must be {'below' if long_ else 'above'} the current price ({plain(mark)}).", "prepared": False}
        if take_profit is not None and ((take_profit <= mark) if long_ else (take_profit >= mark)):
            return {"error": f"A take profit for a {p['side']} must be {'above' if long_ else 'below'} the current price ({plain(mark)}).", "prepared": False}
        b = self._blocked(float(p["size"]) * mark)
        if b:
            b["reasons"] = ["This order is above your daily limit for the assistant."]
            return b
        r = {"executed": False, "kind": "tp_sl", "note": FACE, "order": "Trigger, market, reduce only",
             "position": f"{c} {p['side']}, {plain(p['size'])} {c}", "status": "review_opened"}
        old = [o for o in d["hl"]["orders"] if o["coin"] == c and ((o["kind"] == "stop_loss" and stop_loss is not None and take_profit is None) or (o["kind"] == "take_profit" and take_profit is not None and stop_loss is None))]
        if old:
            r["replaces"] = f"{old[0]['kind'].replace('_', ' ')} at {usd(old[0]['price'])}"
        if stop_loss is not None:
            r["stop_loss"] = short(stop_loss)
        if take_profit is not None:
            r["take_profit"] = short(take_profit)
        return srt(r)

    def _resolve_token(self, tok):
        """Spot rows matching a ticker or mint (mint must be typed or searched)."""
        d = self.d
        t = str(tok)
        if re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", t):
            if t not in strip_files(d["typed"]) and t not in d["searched"]:
                return None, {"error": "Refused: that mint isn't in the user's messages or a token search of this chat. Ask the user for it."}
            return [s_ for s_ in d["spot"] if s_["mint"] == t], None
        return [s_ for s_ in d["spot"] if s_["symbol"].upper() == t.upper()], None

    def t_prepare_sell(self, token, percent=None, amount=None, receive=None):
        d = self.d
        if (percent is None) == (amount is None):
            return {"error": "Give percent or amount, not both.", "prepared": False}
        rows, err = self._resolve_token(token)
        if err:
            return err
        sym = str(token).upper() if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", str(token)) else None
        if not rows:
            return {"note": f"The user holds no {sym or 'such token'} on Solana. Tell them.", "prepared": False, "status": "not_held"}
        if len(rows) > 1:
            q = " and ".join(f"{r['amount']} ({r['mint'][:6]}…{r['mint'][-3:]})" for r in rows)
            for r in rows:
                if r["mint"] not in d["searched"]:
                    d["searched"].append(r["mint"])
            return srt({"note": "Nothing was prepared. Ask the user which one.", "options": [{"amount": r["amount"], "mint": r["mint"], "symbol": r["symbol"]} for r in rows],
                        "prepared": False, "question": f"You hold two tokens called {rows[0]['symbol']}: {q}. Which one?", "status": "needs_choice"})
        r0 = rows[0]
        s_ = r0["symbol"]
        if r0["frozen"]:
            return {"note": BAD_NOTE, "reasons": [f"Your {s_} balance is frozen by the token's authority, so it can't be sold."], "status": "blocked"}
        if s_ in d["no_route"]:
            return {"error": "No route to sell that token right now."}
        have = Decimal(r0["amount"])
        if percent is not None:
            sell = (have * Decimal(plain(percent)) / 100)
            lab = "all of your " + s_ if percent == 100 else f"{plain(percent)}% of your {s_}"
        else:
            sell = Decimal(plain(amount))
            if sell > have:
                return srt({"holding": f"{r0['amount']} {s_}", "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"})
            lab = None
        sell_f = float(sell)
        px = d["prices"].get(s_, 1.0)
        usd_val = round(sell_f * px, 2)
        b = self._blocked(usd_val)
        if b:
            return b
        impact = d["impact"].get(s_, 0.04)
        got_usd = usd_val * (1 - impact / 100)
        sell_txt = commas(round(sell_f, 2) if sell_f >= 0.1 else round(sell_f, 6)) if percent is not None else plain(amount)
        if percent is not None and sell_f >= 1000:
            sell_txt = commas(round(sell_f))
        you_sell = f"{sell_txt} {s_}" + (f" ({lab})" if lab else "")
        if receive == "SOL":
            get_txt = f"about {approx(got_usd / d['prices']['SOL'])} SOL on Solana"
        else:
            get_txt = f"about {approx(got_usd)} USDC on Solana"
        res = {"executed": False, "kind": "sell", "note": "Nothing has moved. The Sell review is open with this Jupiter quote; the user approves it with Face ID or cancels.",
               "price_impact_pct": impact, "route": d["route_sell"], "slippage_pct": 0.5 if impact < 3 else 1.0, "status": "review_opened",
               "symbol": s_, "you_get": get_txt, "you_sell": you_sell}
        if impact >= 3:
            res["policy"] = "warn"
            res["warnings"] = [f"Price impact is {plain(impact)}%, so you get less than the market price."]
        return srt(res)

    def t_prepare_buy(self, symbol, amount_usd, pay_with=None):
        d = self.d
        t = str(symbol)
        if re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", t):
            if t not in strip_files(d["typed"]) and t not in d["searched"]:
                return {"error": "Refused: that mint isn't in the user's messages or a token search of this chat. Ask the user for it."}
            cands = [x for x in d["tokens"] if x["mint"] == t]
        else:
            cands = [x for x in d["tokens"] if x["symbol"].upper() == t.upper()]
            if len(cands) > 1:
                cands = sorted(cands, key=lambda x: -x["liquidity_usd"])
                if cands[1]["liquidity_usd"] * 2 >= cands[0]["liquidity_usd"]:
                    for x in cands[:2]:
                        if x["mint"] not in d["searched"]:
                            d["searched"].append(x["mint"])
                    return srt({"note": "Nothing was prepared. Ask the user which one.",
                                "options": [{"liquidity_usd": x["liquidity_usd"], "mint": x["mint"], "symbol": x["symbol"]} for x in cands[:2]],
                                "prepared": False, "question": f"Two tokens are called {cands[0]['symbol']}. Pick the one you mean on Spot or Memes, or paste its mint.",
                                "status": "needs_choice"})
                cands = cands[:1]
        if not cands or not cands[0]["verified"] and not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", t):
            return {"note": BAD_NOTE, "reasons": [f"There's no verified {t.upper()} token on Solana, so Hermit won't buy one."], "status": "blocked"}
        tk = cands[0]
        usdc = sum(float(h["amount"]) for h in self.hold("USDC", "solana"))
        sol_usd = sum(float(h["amount"]) for h in self.hold("SOL", "solana")) * d["prices"]["SOL"]
        pay = pay_with or ("USDC" if usdc >= amount_usd else "SOL")
        have = usdc if pay == "USDC" else sol_usd
        if have < amount_usd:
            parts = []
            if usdc:
                parts.append(f"{plain(usdc)} USDC")
            solh = self.hold("SOL", "solana")
            if solh:
                parts.append(f"{solh[0]['amount']} SOL")
            return srt({"holding": (" and ".join(parts) or "nothing") + " on Solana", "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"})
        b = self._blocked(amount_usd)
        if b:
            return b
        impact = d["impact"].get(tk["symbol"], 0.03)
        got = amount_usd * (1 - impact / 100) / tk["price"]
        res = {"executed": False, "kind": "buy", "note": "Nothing has moved. The Buy review is open with this Jupiter quote; the user approves it with Face ID or cancels.",
               "price_impact_pct": impact, "route": d["route_buy"], "slippage_pct": 0.5, "status": "review_opened", "symbol": tk["symbol"],
               "you_get": f"about {commas(approx(got)) if got >= 1000 else approx(got)} {tk['symbol']}", "you_pay": f"{plain(amount_usd)} {pay} on Solana"}
        if not tk["verified"]:
            res["verified"] = False
            res["warnings"] = ["Unverified token"] + (["Freeze authority is on"] if "Freeze on" in tk["tags"] else [])
            res["slippage_pct"] = 1.0
        return srt(res)

    def t_prepare_deposit(self, symbol, amount, project="aave-v3", chain=None):
        sym = symbol.upper()
        if project and "aave" not in str(project).lower():
            return {"note": "Only Aave v3 deposits can be prepared here. Other pools are in Earn, DeFi, with Ask.", "prepared": False, "status": "unsupported"}
        h, err = self._pick(sym, chain, ["ethereum", "base", "arbitrum"], "deposit", amount)
        if err:
            return err
        b = self._blocked(float(amount) * self.d["prices"].get(sym, 1))
        if b:
            return b
        apy = self.d["aave_apy"].get(h["chain"], 4.0)
        return srt({"amount": f"{plain(amount)} {sym}", "apy_pct": apy, "executed": False, "kind": "deposit",
                    "note": "Waiting for the user to review the deposit and approve it with Face ID.",
                    "pool": f"Aave v3 on {CHAIN_NAME[h['chain']]}", "status": "review_opened"})

    def t_prepare_withdraw(self, symbol, chain=None, amount=None):
        sym = symbol.upper()
        rows = [e for e in self.d["earn"] if e["symbol"] == sym and (chain is None or e["chain"] == chain)]
        if not rows:
            return {"note": f"The user has no {sym} in Aave. Tell them.", "prepared": False, "status": "nothing_to_withdraw"}
        if len(rows) > 1:
            q = ", ".join(f"{e['amount']} {sym} on {CHAIN_NAME[e['chain']]}" for e in rows)
            return {"prepared": False, "question": f"Which deposit? {q}.", "status": "needs_choice", "note": "Nothing was prepared. Ask the user which one."}
        e = rows[0]
        if amount is not None and Decimal(plain(amount)) > Decimal(e["amount"]):
            return {"note": f"Only {e['amount']} {sym} is in Aave on {CHAIN_NAME[e['chain']]}. Tell the user.", "prepared": False, "status": "insufficient",
                    "supplied": f"{e['amount']} {sym}"}
        return {"status": "review_opened", "kind": "withdraw", "executed": False, "note": FACE}

    def t_prepare_multisend(self, symbol, recipients, chain=None):
        typed = self.d["typed"]
        for r in recipients:
            if r["address"] not in typed:
                return {"error": f"Refused: {r['address'][:12]}… isn't in the user's messages or attached file. Only send to addresses the user gave you. Ask them for the list."}
        sym = symbol.upper()
        good = [r for r in recipients if addr_family(r["address"]) == "evm"]
        bad = [i + 1 for i, r in enumerate(recipients) if addr_family(r["address"]) != "evm"]
        total = sum(Decimal(plain(r["amount"])) for r in good)
        hs = self.hold(sym)
        if not hs:
            return {"note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient", "holding": "none"}
        h = hs[0] if not chain else ([x for x in hs if x["chain"] == chain] or hs)[0]
        res = {"executed": False, "fees_up_to": f"{plain(round(0.04 * max(1, len(good)), 2))} {sym}", "kind": "multisend", "network": CHAIN_NAME[h["chain"]],
               "note": "Nothing has moved. The Batch send review lists every recipient; the user approves all of it with one Face ID, or cancels.",
               "recipients": len(good), "status": "review_opened", "total": f"{plain(total)} {sym}"}
        if bad:
            res["left_out"] = [f"row {i}: not an address" for i in bad]
            res["left_out_count"] = len(bad)
        return srt(res)

    # -------------------------------------------------------------------------------- cards
    def t_add_contact(self, name, address, network=None):
        d = self.d
        fam = addr_family(address)
        if fam is None:
            return {"error": "That doesn't look like a valid address."}
        if address not in strip_files(d["typed"]):
            return {"error": "Refused: the user didn't type that address in this chat. Only save an address the user gave you themselves. Ask them for it."}
        for c in d["contacts"]:
            if c["name"].lower() == name.lower():
                if c["address"] == address:
                    return {"note": f"{c['name']} is already saved with that address.", "prepared": False, "status": "exists"}
                return {"note": f"A contact called {c['name']} already exists with another address. Ask the user for a different name.", "prepared": False, "status": "name_taken"}
        return srt({"address": address, "executed": False, "kind": "contact", "name": name, "network": {"evm": "EVM", "solana": "Solana", "near": "NEAR"}[fam],
                    "note": "Waiting for the user to tap Save. The card shows the full address; nothing is sent.", "status": "card_opened"})

    def _parse_when(self, when):
        import datetime as dt
        today = dt.date.fromisoformat(self.d["today"])
        t = str(when).lower().strip()
        hh, mm = 9, 0
        m = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", t) or re.search(r"\b(\d{1,2})\s*(am|pm)\b", t)
        if m:
            g = m.groups()
            hh = int(g[0])
            mm = int(g[1]) if len(g) > 2 and g[1] and g[1].isdigit() else 0
            ap = g[-1]
            if ap == "pm" and hh < 12:
                hh += 12
            if ap == "am" and hh == 12:
                hh = 0
            t = (t[:m.start()] + t[m.end():]).strip()
        day = None
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
        if m:
            day = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if day is None:
            m = re.search(r"in (\d+) (day|days|week|weeks)", t)
            if m:
                day = today + dt.timedelta(days=int(m.group(1)) * (7 if m.group(2).startswith("week") else 1))
        if day is None and "tomorrow" in t:
            day = today + dt.timedelta(days=1)
        if day is None and "today" in t:
            day = today
        if day is None:
            for i, wd in enumerate(WEEKDAYS):
                if wd in t:
                    delta = (i - today.weekday()) % 7 or 7
                    day = today + dt.timedelta(days=delta)
                    break
        if day is None:
            m = re.search(r"\b([a-z]{3})[a-z]*\.? (\d{1,2})\b", t) or re.search(r"\b(\d{1,2})(?:st|nd|rd|th)? ([a-z]{3})[a-z]*\b", t)
            if m:
                a, b = m.group(1), m.group(2)
                mon, dd = (a, b) if a.isalpha() else (b, a)
                if mon[:3] in MONTHS:
                    day = dt.date(today.year, MONTHS[mon[:3]], int(dd))
        if day is None:
            return None
        return dt.datetime(day.year, day.month, day.day, hh, mm)

    def t_schedule_payment(self, payee, amount, symbol, when, chain=None, repeat=None):
        import datetime as dt
        d = self.d
        sym = symbol.upper()
        name = None
        fam = None
        ms = [c for c in d["contacts"] if c["name"].lower() == str(payee).lower()] or [c for c in d["contacts"] if c["name"].lower().split()[0] == str(payee).lower()]
        if addr_family(str(payee)):
            if not self._addr_ok(payee):
                return {"error": "Refused: the user didn't type that address in this chat. Only schedule payments to a saved contact or an address the user gave you themselves. Ask them for it."}
            name, fam = short_addr(payee), addr_family(payee)
        elif not ms:
            return {"note": f"No saved contact called {payee}. Ask the user for the address, or to save the contact first.", "prepared": False, "status": "payee_not_found"}
        elif len(ms) > 1:
            names = [c["name"] for c in ms]
            return srt({"note": "Nothing was prepared. Ask the user which one.", "options": names, "prepared": False,
                        "question": f"Which {payee}? {' or '.join(names)}.", "status": "needs_choice"})
        else:
            name, fam = ms[0]["name"], ms[0]["family"]
        nxt = self._parse_when(when)
        if nxt is None or nxt < dt.datetime.fromisoformat(d["today"] + " 00:00"):
            return {"error": "That date is in the past.", "prepared": False} if nxt is not None else {"error": "I couldn't read that date.", "prepared": False}
        families = {"evm": [c for c in EVM_NETS if c != "monad"], "solana": ["solana"], "near": ["near"]}[fam]
        h, err = self._pick(sym, chain, families, "payment", 0)        # a reminder does not need the balance today
        if err:
            return err
        return srt({"amount": f"{plain(amount)} {sym}", "executed": False, "kind": "scheduled_payment", "network": CHAIN_NAME[h["chain"]],
                    "next_due": nxt.strftime("%Y-%m-%d %H:%M"),
                    "note": "Waiting for the user to tap Schedule. This is a reminder: nothing is sent by itself, and the user approves each payment with Face ID when it is due.",
                    "payee": name, "repeat": repeat or "once", "status": "card_opened"})

    def _parse_schedule(self, text):
        t = str(text).lower().strip()

        def clock(default):
            m = re.search(r"(?:at )?(\d{1,2}):(\d{2})\s*(am|pm)?", t) or re.search(r"(?:at )?\b(\d{1,2})\s*(am|pm)\b", t) or re.search(r"\bat (\d{1,2})\b", t)
            if not m:
                return default
            g = m.groups()
            hh = int(g[0])
            mm = int(g[1]) if len(g) > 1 and g[1] and g[1].isdigit() else 0
            if g[-1] == "pm" and hh < 12:
                hh += 12
            if g[-1] == "am" and hh == 12:
                hh = 0
            return f"{hh:02d}:{mm:02d}"
        m = re.search(r"every (\d+) ?(?:minutes?|mins?)", t)
        if m or "every minute" in t:
            n = int(m.group(1)) if m else 1
            return ("too_frequent", None) if n < 5 else ("ok", f"Every {n} minutes")
        m = re.search(r"every (\d+) hours?", t)
        if m:
            return "ok", f"Every {int(m.group(1))} hours"
        if "every hour" in t or "hourly" in t:
            return "ok", "Every hour"
        if "weekday" in t:
            return "ok", f"Every weekday at {clock('09:00')}"
        for wd in WEEKDAYS:
            if wd in t:
                return "ok", f"Every {wd.capitalize()} at {clock('09:00')}"
        if "morning" in t:
            return "ok", f"Every day at {clock('08:00')}"
        if "evening" in t:
            return "ok", f"Every day at {clock('18:00')}"
        if "night" in t:
            return "ok", f"Every day at {clock('21:00')}"
        if re.search(r"every day|daily|each day|everyday", t):
            return "ok", f"Every day at {clock('09:00')}"
        return "bad", None

    def t_create_automation(self, name, prompt, schedule):
        st, norm_ = self._parse_schedule(schedule)
        if st == "bad":
            return {"error": "I couldn't read that schedule. Use words like 'every morning at 8', 'every 5 minutes' or 'every Friday at 18:00'.", "prepared": False}
        if st == "too_frequent":
            return {"note": BAD_NOTE, "reasons": ["The shortest repeat is every 5 minutes."], "status": "blocked"}
        if self.d["automations"] >= 10:
            return {"note": BAD_NOTE, "reasons": ["You already have 10 automations. Delete one in Automations first."], "status": "blocked"}
        return srt({"executed": False, "kind": "automation", "name": name, "note": "Waiting for the user to tap Create. Nothing is scheduled yet.",
                    "prompt": prompt, "schedule": norm_, "status": "card_opened"})


# ================================================================================================ conversation builder
class Dup(Exception):
    pass


class Conv:
    """Builds one conversation while executing tool calls on the sim. Only tools of the pack may be called."""

    def __init__(self, rng, w, pack_name, bio, used, name=None, notes=None):
        self.rng, self.w, self.bio = rng, w, bio
        self.pack_name = pack_name
        self.pack = pack_names(pack_name)
        self.used = used
        self.world0 = None                              # snapshot taken at the first user turn (scenarios finish the world first)
        self.msgs = [{"role": "system", "content": system_agent(bio, name, notes)}]
        self.expect, self.rounds, self.turn_rounds = [], 0, 0
        self.world_last = w.state()
        self.checkpoints = {}
        self.texts = []
        self.cards = []
        self.name, self.notes = name, notes

    def user(self, text, file_text=None):
        full = text if not file_text else file_text + "\n\n" + text
        if self.world0 is None:
            self.world0 = self.w.state()
        k = norm(full)
        if len(text.split()) > 4 or not self.texts:       # short follow-up replies ("base", "the first one") may repeat; openers and long texts are unique
            if k in self.used or k in self.texts:
                raise Dup(k)
            self.texts.append(k)
        self.msgs.append({"role": "user", "content": full})
        self.w.d["typed"] += "\n" + text
        self.expect = []
        self.turn_rounds = 0
        self.world_last = self.w.state()

    def call(self, *calls, lead=""):
        """calls: (name, args) executed together in one assistant message (parallel). Returns a result or a list."""
        tcs = []
        for n, a in calls:
            assert n in self.pack, f"{n} not in pack {self.pack_name}"
            tcs.append({"name": n, "arguments": a})
        self.msgs.append({"role": "assistant", "content": lead, "tool_calls": tcs})
        self.expect.append({"calls": copy.deepcopy(tcs)})
        self.rounds += 1
        self.turn_rounds += 1
        res = []
        for n, a in calls:
            r = self.w.call(n, copy.deepcopy(a))
            res.append(r)
            self.msgs.append({"role": "tool", "content": jd(r).replace("Face ID", self.bio)})
            if isinstance(r, dict) and r.get("status") == "card_opened":
                self.cards.append(n)
        return res if len(res) > 1 else res[0]

    def say(self, text, ask=False):
        self.msgs.append({"role": "assistant", "content": text})
        self.expect.append({"ask": True} if ask else {"text": True})

    def mutate(self, fn):
        fn(self.w)
        self.checkpoints[len(self.msgs)] = self.w.state()

    def record(self, kind, fam, part, **extra):
        meta = {"source": GEN, "generator": GEN, "part": part, "kind": kind, "family_scenario": fam, "bio": self.bio,
                "family": "lite", "family_ok": ["4b", "lite"], "pack_name": self.pack_name, "pack": list(self.pack),
                "world": self.world0, "world_last": self.world_last, "checkpoints": {str(k): v for k, v in self.checkpoints.items()},
                "expect": self.expect, "rounds_total": self.rounds, "steps_last_turn": self.turn_rounds,
                "steps": [tc["name"] for e in self.expect if "calls" in e for tc in e["calls"]],
                "cards": list(self.cards), "name": self.name, "notes": self.notes}
        meta.update(extra)
        return {"messages": self.msgs, "tools": True, "meta": meta}


# ================================================================================================ phrasing helpers
def reserved_t(i):
    return i % 5 == 4          # one template in five is held out for eval


def reserved_s(i):
    return i % 4 == 3          # one slot value in four is held out for eval


def pick(rng, split, lst):
    """Template choice. Train never sees the held-out fifth; eval draws from it 60% of the time."""
    if len(lst) < 5:
        return rng.choice(lst)
    held = [x for i, x in enumerate(lst) if reserved_t(i)]
    rest = [x for i, x in enumerate(lst) if not reserved_t(i)]
    if split == "train":
        return rng.choice(rest)
    return rng.choice(held) if rng.random() < 0.6 else rng.choice(lst)


def slot(rng, split, lst):
    """Slot value choice (names, tokens, amounts): a quarter of the values never appears in train."""
    if len(lst) < 4:
        return rng.choice(lst)
    held = [x for i, x in enumerate(lst) if reserved_s(i)]
    rest = [x for i, x in enumerate(lst) if not reserved_s(i)]
    if split == "train":
        return rng.choice(rest)
    return rng.choice(held) if rng.random() < 0.5 else rng.choice(lst)


PROTECT = set(WEEKDAYS) | {"friday", "tomorrow", "monthly", "weekly", "morning", "evening", "arbitrum", "ethereum", "solana", "polygon", "profit", "bridge", "deposit", "aave"}


def styl(rng, t):
    """Casual typing: lower case, a typo on a long plain word, a polite opener or ender. Placeholders {x} stay intact."""
    words = t.split(" ")
    if rng.random() < 0.32:
        words = [w if ("{" in w or "}" in w) else (w.lower() if not w.isupper() or len(w) > 3 else w) for w in words]
    if rng.random() < 0.12:
        idx = [i for i, w in enumerate(words) if len(w) >= 6 and w.isalpha() and w.islower() and "{" not in w and w not in PROTECT]
        if idx:
            i = rng.choice(idx)
            w = words[i]
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    out = " ".join(words)
    if rng.random() < 0.10 and not out.endswith("?"):
        out += rng.choice([" pls", " please", " thanks", " asap", " now"])
    if rng.random() < 0.08:
        out = rng.choice(["hey ", "ok ", "yo ", "hi, ", "so ", "quick one: "]) + out
    return out


def U(rng, split, tmpls, **slots):
    return styl(rng, pick(rng, split, tmpls)).format(**slots)


PREFIXES = ["ok so ", "alright, ", "one more thing: ", "also ", "next, ", "right, ", "sorry, ", "hm, ", "and now ", "by the way, ", "ok, ", "wait, ", "last thing: "]


def V(rng, split, tmpls, **slots):
    """U() plus a conversational prefix half of the time (long chats repeat the same kinds of request)."""
    t = styl(rng, pick(rng, split, tmpls)).format(**slots)
    if rng.random() < 0.55:
        t = rng.choice(PREFIXES) + (t[:1].lower() + t[1:] if not t[:2].isupper() else t)
    if rng.random() < 0.3:
        t += rng.choice([" please", " thanks", " when you can", " real quick", " for me", " if possible"])
    return t


def rnd(rng, lo, hi, dp):
    return round(rng.uniform(lo, hi), dp)


NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Karl", "Lena", "Mallory", "Nick", "Olga",
         "Pedro", "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Yara", "Zoe", "Mum", "Landlord"]
USER_NAMES = ["Noah", "Maya", "Leo", "Aisha", "Tom", "Yuki", "Priya", "Diego", "Nina", "Omar", "Chloe", "Sam"]
COINS = ["BTC", "ETH", "SOL", "HYPE", "BNB", "DOGE", "XRP", "AVAX", "LINK", "SUI", "ENA", "ARB", "OP", "NEAR", "INJ", "TAO"]
COIN_SAY = {"BTC": ["btc", "bitcoin"], "ETH": ["eth", "ether"], "SOL": ["sol", "solana"], "HYPE": ["hype"], "BNB": ["bnb"], "DOGE": ["doge"],
            "XRP": ["xrp"], "AVAX": ["avax"], "LINK": ["link"], "SUI": ["sui"], "ENA": ["ena"], "ARB": ["arb"], "OP": ["op"],
            "NEAR": ["near"], "INJ": ["inj"], "TAO": ["tao"]}
MEME_LIST = list(MEMES)
TOKEN_NAME = {"WIF": "dogwifhat", "JUP": "Jupiter", "POPCAT": "Popcat", "PUMP": "Pump", "MOODENG": "Moo Deng", "GOAT": "Goatseus Maximus",
              "FARTCOIN": "Fartcoin", "PENGU": "Pudgy Penguins"}
SUFFIXES = ["", "", "", " pls", " thanks"]


def tap(rng, bio, cap=True):
    v = "tap" if bio == "Face ID" or rng.random() < 0.3 else "click"
    return v.capitalize() if cap else v


def say_coin(rng, c):
    return rng.choice([c, c] + COIN_SAY[c])


# ------------------------------------------------------------------------------------------------ answer snippets
def review_tail(rng, bio):
    return rng.choice([f"Approve it with {bio} or cancel.", f"It is waiting for your {bio}.", f"Review it and approve with {bio}.",
                       f"It waits for your {bio} on the review screen."])


def nothing_moved(rng):
    return rng.choice(["Nothing has moved yet.", "Nothing happens until you approve.", "Nothing is placed until you approve."])


def addr_tail(rng, bio):
    return rng.choice([f"Check the address on the review, then approve with {bio}.", f"Check the address, then approve with {bio}.",
                       f"Look at the address on the review and approve with {bio}."])


# ------------------------------------------------------------------------------------------------ world helpers
def mkw(rng, bio):
    return W.fresh(rng, "Mac" if bio == "Touch ID" else "iPhone")


def mint_for(rng):
    return H.sol_addr(rng)


def mk_token(rng, w, sym, verified=True, liq=None, tags=None, mint=None, price=None):
    p = price if price is not None else w.d["prices"].get(sym, 0.05)
    t = {"symbol": sym, "name": TOKEN_NAME.get(sym, sym.title()), "mint": mint or mint_for(rng), "verified": verified,
         "price": p, "change_24h_pct": round(rng.uniform(-14, 14), 1),
         "liquidity_usd": liq if liq is not None else int(rng.choice([2.4, 3.1, 4.6, 6.2, 9.1, 12.5, 18.4, 27, 38, 52]) * 1e6),
         "volume_24h_usd": int(rng.choice([1.2, 3.4, 7.3, 15.2, 41, 96]) * 1e6), "tags": tags if tags is not None else []}
    w.d["tokens"].append(t)
    return t


def solana_world(rng, bio, held=2, usdc=None, extra_catalog=3):
    w = mkw(rng, bio)
    w.add_hold("USDC", "solana", usdc if usdc is not None else rng.choice([45, 80.5, 120, 250.75, 500, 64.2]))
    w.add_hold("SOL", "solana", rnd(rng, 0.08, 2.4, 3))
    syms = rng.sample(MEME_LIST, held + extra_catalog)
    for i, s_ in enumerate(syms):
        t = mk_token(rng, w, s_)
        if i < held:
            px = w.d["prices"][s_]
            target = rng.choice([8, 12, 25, 40, 75, 120, 300])
            amt = round(target / px, 2)
            if amt >= 100:
                amt = round(amt)
            cost = round(px * rng.uniform(0.6, 1.5), 4 if px < 1 else 2)
            w.add_spot(s_, t["mint"], amt, cost)
    return w, syms[:held], syms[held:]


def trade_world(rng, bio, coins=("SOL",), with_orders=True, value=None):
    w = mkw(rng, bio)
    w.d["hl"]["value"] = value if value is not None else rng.choice([180.0, 250.5, 420, 800, 1500.25, 95.5])
    w.add_hold("USDC", "base", rng.choice([120.5, 300, 75, 540]))
    for i, c in enumerate(coins):
        mark = w.d["prices"][c]
        side = rng.choice(["long", "short"])
        lev = rng.choice([3, 5, 5, 10])
        size = round(rng.choice([20, 50, 100, 250]) / mark, 3 if mark > 50 else 2)
        entry = round(mark * rng.uniform(0.94, 1.06), ASSETS[c][2])
        w.add_position(c, side, size, entry, lev)
        if with_orders:
            p = w.pos(c)
            stop_side = "sell" if side == "long" else "buy"
            sp = round(mark * (0.9 if side == "long" else 1.1), ASSETS[c][2])
            tp = round(mark * (1.12 if side == "long" else 0.88), ASSETS[c][2])
            w.add_order(8100 + i * 10, c, "stop_loss", sp, stop_side, size, True)
            w.add_order(8101 + i * 10, c, "take_profit", tp, stop_side, size, True)
    return w


def amount_say(rng, x):
    return plain(x)


ABSENT_PAGE_TAIL = None


# ================================================================================================ PART 1: scenarios with the tool in the pack
def tag_words(tags):
    out = []
    for t in tags:
        if t == "Freeze on":
            out.append("freeze authority is on, so the creator can freeze your tokens")
        elif t == "Mint on":
            out.append("mint authority is on, so more can be created")
        elif t == "Low liquidity":
            out.append("liquidity is low")
        elif t == "New":
            out.append("it is new")
        elif t == "Unverified":
            continue
        elif t.startswith("Top holders"):
            out.append("the top holders own " + t.split()[-1])
        else:
            out.append(t.lower())
    return out


def ans_sell(rng, r, bio):
    imp = plain(r["price_impact_pct"])
    base = rng.choice([
        f"I've prepared selling {r['you_sell']} for {r['you_get']} through Jupiter. Price impact is {imp}%. Approve with {bio} or cancel. Nothing has been sold yet.",
        f"Prepared: sell {r['you_sell']} for {r['you_get']} through Jupiter, price impact {imp}%. It is waiting for your {bio}. Nothing has moved yet.",
        f"I've prepared the sale of {r['you_sell']}: you get {r['you_get']} through Jupiter, price impact {imp}%. Review it and approve with {bio}."])
    if r.get("warnings"):
        base = base.rsplit(". ", 1)[0] + ". " + r["warnings"][0] + " Approve with " + bio + " only if you accept that."
    return base


PCT_SAY = [(50, ["half", "50%"]), (25, ["a quarter", "25%"]), (75, ["75%", "three quarters"]), (30, ["30%"]), (20, ["20%"]),
           (10, ["10%"]), (40, ["40%"]), (60, ["60%"])]
SELL_PCT_T = ["sell {p} of my {s}", "dump {p} of my {s}", "can you sell {p} of my {s}?", "take some profit, sell {p} of {s}",
              "offload {p} of my {s} bag", "i want to sell {p} of my {s}", "trim my {s}: sell {p}", "sell {p} of my {s} for me"]


def sc_sell_pct(rng, split, used, bio):
    w, held, _ = solana_world(rng, bio)
    c = Conv(rng, w, "solana", bio, used)
    sym = rng.choice(held)
    pct, says = rng.choice(PCT_SAY)
    c.user(U(rng, split, SELL_PCT_T, p=rng.choice(says), s=sym if rng.random() < 0.6 else sym.lower()))
    r = c.call(("prepare_sell", {"token": sym, "percent": pct}))
    c.say(ans_sell(rng, r, bio))
    return c.record("sell_pct", "spot_solana", 1)


SELL_ALL_T = ["sell all my {s}{r}", "get out of {s}{r}", "sell everything in {s}{r}", "exit my {s} position{r}", "dump all my {s}{r}",
              "i'm done with {s}, sell it all{r}"]
RECV = [("", None), (" into SOL", "SOL"), (" to usdc", "USDC"), (" and give me SOL", "SOL"), (" for USDC", "USDC"), ("", None)]


def sc_sell_all(rng, split, used, bio):
    w, held, _ = solana_world(rng, bio)
    c = Conv(rng, w, "solana", bio, used)
    sym = rng.choice(held)
    rs, rv = rng.choice(RECV)
    c.user(U(rng, split, SELL_ALL_T, s=sym, r=rs))
    args = {"token": sym, "percent": 100}
    if rv:
        args["receive"] = rv
    r = c.call(("prepare_sell", args))
    c.say(ans_sell(rng, r, bio))
    return c.record("sell_all", "spot_solana", 1)


SELL_AMT_T = ["sell {a} {s}", "sell {a} of my {s}", "can you sell {a} {s} for me?", "swap {a} {s} back to usdc, sell it", "i want to sell exactly {a} {s}"]


def sc_sell_amount(rng, split, used, bio):
    w, held, _ = solana_world(rng, bio)
    c = Conv(rng, w, "solana", bio, used)
    sym = rng.choice(held)
    have = float(next(s_ for s_ in w.d["spot"] if s_["symbol"] == sym)["amount"])
    a = round(have * rng.choice([0.1, 0.2, 0.35, 0.6]), 2)
    if a <= 0:
        a = 1
    a = short(a)
    c.user(U(rng, split, SELL_AMT_T, a=plain(a), s=sym))
    r = c.call(("prepare_sell", {"token": sym, "amount": a}))
    c.say(ans_sell(rng, r, bio))
    return c.record("sell_amount", "spot_solana", 1)


def sc_sell_mint(rng, split, used, bio):
    w, held, _ = solana_world(rng, bio)
    c = Conv(rng, w, "solana", bio, used)
    sym = rng.choice(held)
    mint = next(s_ for s_ in w.d["spot"] if s_["symbol"] == sym)["mint"]
    pct, says = rng.choice(PCT_SAY)
    c.user(styl(rng, rng.choice(["sell {p} of {m}", "sell {p} of this one: {m}", "i hold {m}, sell {p} of it"])).format(p=rng.choice(says), m=mint))
    r = c.call(("prepare_sell", {"token": mint, "percent": pct}))
    c.say(ans_sell(rng, r, bio))
    return c.record("sell_mint", "spot_solana", 1)


SAFE_T = ["is {s} safe?", "is {s} legit?", "check {s} for me", "what's the liquidity on {s}?", "how risky is {s}?", "can you check the token {s}",
          "is it ok to buy {s}?", "any red flags on {s}?"]
RISKY_SYMS = [("WIFG", "WIF Gold"), ("MOONCAT", "Moon Cat"), ("BABYGOAT", "Baby Goat"), ("ELONPEPE", "Elon Pepe"), ("DOGE2", "Doge 2.0"), ("PUMPX", "Pump X")]
RISK_TAGS = [["Freeze on", "Mint on", "Low liquidity", "New"], ["Unverified", "Top holders 41%", "New"], ["Mint on", "New"],
             ["Unverified", "Freeze on", "Top holders 38%"], ["Low liquidity", "New"]]


def sc_search_safety(rng, split, used, bio):
    w, held, other = solana_world(rng, bio, held=1)
    c = Conv(rng, w, "solana", bio, used)
    if rng.random() < 0.5:
        sym = rng.choice(other)
        t = next(x for x in w.d["tokens"] if x["symbol"] == sym)
        q = sym
    else:
        sym, nm = rng.choice(RISKY_SYMS)
        tags = rng.choice(RISK_TAGS)
        t = mk_token(rng, w, sym, verified=False, liq=rng.choice([3900, 4200, 8800, 12400, 21000, 54000]), tags=tags, price=rng.choice([0.00031, 0.0042, 0.015]))
        t["name"] = nm
        t["change_24h_pct"] = rng.choice([412, 96, -45, 18.5])
        q = sym if rng.random() < 0.6 else t["mint"]
    c.user(U(rng, split, SAFE_T, s=q))
    r = c.call(("search_tokens", {"query": q}))
    row = r["tokens"][0]
    chg = row["change_24h_pct"]
    move = f"{'up' if chg > 0 else 'down'} {plain(abs(chg))}% in 24h"
    if row["verified"]:
        txt = rng.choice([f"{row['name']} ({row['symbol']}) is verified, with {big(row['liquidity_usd'])} of liquidity, {move}. It has no warning tags. I can't call any token safe.",
                          f"{row['symbol']} is verified and has {big(row['liquidity_usd'])} of liquidity, {move}. No warning tags. I can't call any token safe, so size it small."])
    else:
        words = tag_words(row["tags"])
        txt = f"{row['symbol']} is not verified, with {big(row['liquidity_usd'])} of liquidity. " + (cap1(join_and(words)) + ". " if words else "")
        if row.get("refusal"):
            txt += row["refusal"] + " "
        txt += "I can't call any token safe."
    c.say(txt.strip())
    return c.record("search_safety", "spot_solana", 1)


BUY_T = ["buy ${a} of {s}", "buy {s} for ${a}", "ape ${a} into {s}", "get me ${a} worth of {s}", "can you buy ${a} of {s}?", "pick up ${a} of {s}"]


def sc_buy_ticker(rng, split, used, bio):
    w, held, other = solana_world(rng, bio, held=0, usdc=rng.choice([120, 250, 500]))
    c = Conv(rng, w, "solana", bio, used)
    sym = rng.choice(other)
    a = rng.choice([5, 10, 15, 20, 25, 50, 100])
    c.user(U(rng, split, BUY_T, a=a, s=sym if rng.random() < 0.6 else sym.lower()))
    r = c.call(("prepare_buy", {"symbol": sym, "amount_usd": a}))
    imp = plain(r["price_impact_pct"])
    c.say(rng.choice([
        f"I've prepared a buy of {sym}: you pay {r['you_pay']} and get {r['you_get']} through Jupiter. Price impact is {imp}%. Approve with {bio} or cancel. Nothing is bought yet.",
        f"Prepared: {r['you_pay']} for {r['you_get']} through Jupiter, price impact {imp}%. It is waiting for your {bio}. Nothing has moved yet."]))
    return c.record("buy_ticker", "spot_solana", 1)


def sc_buy_refused(rng, split, used, bio):
    w, held, other = solana_world(rng, bio, held=0)
    sym, nm = rng.choice(RISKY_SYMS)
    t = mk_token(rng, w, sym, verified=False, liq=rng.choice([2100, 3900, 4400]), tags=["Freeze on", "Mint on", "Low liquidity", "New"], price=0.00031)
    t["name"] = nm
    c = Conv(rng, w, "solana", bio, used)
    a = rng.choice([10, 20, 25])
    c.user(styl(rng, rng.choice(["buy ${a} of this: {m}", "i found {m}, buy ${a} of it", "ape ${a} in {m}"])).format(a=a, m=t["mint"]))
    r = c.call(("search_tokens", {"query": t["mint"]}))
    row = r["tokens"][0]
    c.say(f"I won't prepare that. {row['symbol']} is not verified, its liquidity is {big(row['liquidity_usd'])} and freeze authority is on. {row['refusal']}")
    return c.record("buy_refused", "spot_solana", 1)


FIND_T = ["what's the mint of {n}?", "find the {n} token", "search {n} on solana", "look up {n} for me", "what's the contract for {n}?"]


def sc_search_find(rng, split, used, bio):
    w, held, other = solana_world(rng, bio, held=0)
    c = Conv(rng, w, "solana", bio, used)
    sym = rng.choice(other)
    t = next(x for x in w.d["tokens"] if x["symbol"] == sym)
    c.user(U(rng, split, FIND_T, n=t["name"].lower()))
    r = c.call(("search_tokens", {"query": t["name"].lower()}))
    row = r["tokens"][0]
    c.say(f"{row['name']} ({row['symbol']}) has the mint {row['mint']}. It is {'verified' if row['verified'] else 'not verified'} with {big(row['liquidity_usd'])} of liquidity.")
    return c.record("search_find", "spot_solana", 1)


# ------------------------------------------------------------------------------------------------ orders_protect
CANCEL_KIND_T = {"stop_loss": ["cancel my {c} stop loss", "remove the stop loss on {c}", "delete my {c} stop", "take off my {c} stop loss pls", "get rid of the {c} sl"],
                 "take_profit": ["cancel my {c} take profit", "remove my {c} tp", "delete the take profit on {c}", "drop the {c} take profit"],
                 "order": ["cancel my {c} limit order", "cancel the {c} limit", "remove my open {c} order", "kill my {c} limit order"]}


def sc_cancel_kind(rng, split, used, bio, kind=None):
    kind = kind or rng.choice(["stop_loss", "stop_loss", "take_profit", "order"])
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    w = trade_world(rng, bio, (coin,), with_orders=kind != "order")
    if kind == "order":
        mark = w.d["prices"][coin]
        w.add_order(8123, coin, "limit", round(mark * 0.95, ASSETS[coin][2]), "buy", round(40 / mark, 3), False)
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, CANCEL_KIND_T[kind], c=say_coin(rng, coin)))
    r = c.call(("cancel_order", {"coin": coin, "kind": kind}))
    t_ = tap(rng, bio)
    txt = f"I've put up a card to cancel your {r['cancelling']}. {t_} Confirm to go ahead. "
    if r.get("warning"):
        txt += r["warning"] + " "
    txt += "Nothing is cancelled until you " + t_.lower() + "."
    c.say(txt)
    return c.record("cancel_" + kind, "orders_protect", 1)


def sc_cancel_all(rng, split, used, bio):
    coins = rng.sample(["ETH", "BTC", "SOL"], 2)
    w = trade_world(rng, bio, (coins[0],), with_orders=True)
    for i, cn in enumerate(coins):
        mark = w.d["prices"][cn]
        w.add_order(8200 + i, cn, "limit", round(mark * 0.93, ASSETS[cn][2]), "buy", round(30 / mark, 3), False)
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, ["cancel all my open orders", "cancel every limit order i have", "clear all my pending orders", "remove all open orders pls", "nuke my open limit orders"]))
    r = c.call(("cancel_order", {"coin": "all"}))
    t_ = tap(rng, bio)
    c.say(f"I've put up a card to cancel {r['cancelling']}. Stop losses and take profits stay. {t_} Confirm to go ahead. Nothing is cancelled until you {t_.lower()}.")
    return c.record("cancel_all", "orders_protect", 1)


def sc_cancel_nokind(rng, split, used, bio):
    coin = rng.choice(["SOL", "ETH", "BTC"])
    w = trade_world(rng, bio, (coin,), with_orders=True)
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, ["cancel my {c} order", "remove my {c} order", "cancel the order on {c}", "get rid of my {c} trigger"], c=coin))
    r = c.call(("cancel_order", {"coin": coin}))
    c.say(r["question"], ask=True)
    pick_k = rng.choice(["stop_loss", "take_profit"])
    c.user(rng.choice(["the stop", "stop loss", "the stop loss one"]) if pick_k == "stop_loss" else rng.choice(["the take profit", "tp", "take profit one"]))
    r2 = c.call(("cancel_order", {"coin": coin, "kind": pick_k}))
    t_ = tap(rng, bio)
    txt = f"I've put up a card to cancel your {r2['cancelling']}. {t_} Confirm to go ahead. "
    if r2.get("warning"):
        txt += r2["warning"] + " "
    c.say(txt + "Nothing is cancelled until you " + t_.lower() + ".")
    return c.record("cancel_nokind", "orders_protect", 1)


def sc_orders_read(rng, split, used, bio):
    coin = rng.choice(["SOL", "ETH", "BTC"])
    has = rng.random() < 0.7
    w = trade_world(rng, bio, (coin,), with_orders=has)
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, ["do I have a stop on {c}?", "is my {c} position protected?", "what orders do I have open on {c}?", "any stop loss on my {c} trade?", "check my {c} stop"], c=coin))
    r = c.call(("hyperliquid_account", {}))
    trig = [o for o in r["open_orders"] if o.get("is_trigger") and o["coin"] == coin]
    stops = [o for o in trig if o["order_type"] == "Stop Market"]
    tps = [o for o in trig if o["order_type"] != "Stop Market"]
    pos = next(p for p in r["positions"] if p["coin"] == coin)
    if stops:
        txt = f"Yes, your {coin} {pos['side']} has a stop loss at {usd(stops[0]['price'])}"
        txt += f" and a take profit at {usd(tps[0]['price'])}." if tps else "."
    else:
        txt = f"No. Your {coin} {pos['side']} of {plain(pos['size'])} {coin} has no stop loss open right now."
    c.say(txt)
    return c.record("orders_read", "orders_protect", 1)


def sc_nothing_to_cancel(rng, split, used, bio):
    coin = rng.choice(["SOL", "ETH", "BTC"])
    w = trade_world(rng, bio, (coin,), with_orders=False)
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, ["cancel my {c} stop loss", "remove my {c} take profit", "cancel my {c} orders"], c=coin))
    kind = None
    t = c.msgs[-1]["content"].lower()
    args = {"coin": coin}
    if "stop" in t:
        args["kind"] = "stop_loss"
    elif "take profit" in t:
        args["kind"] = "take_profit"
    r = c.call(("cancel_order", args))
    c.say(f"You don't have any open {coin} {'stop loss' if args.get('kind') == 'stop_loss' else 'take profit' if args.get('kind') == 'take_profit' else 'order'} to cancel, so there is nothing to do.")
    return c.record("nothing_to_cancel", "orders_protect", 1)


CLOSE_T = ["close my {c} {side}", "close my {c} position", "exit my {c} {side}", "get me out of {c}", "shut my {c} trade"]
CLOSE_PCT_T = ["close {p} of my {c} {side}", "take {p} off my {c} position", "reduce my {c} {side} by {p}", "close {p} of {c}"]


def sc_close(rng, split, used, bio):
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    w = trade_world(rng, bio, (coin,), with_orders=False)
    side = w.pos(coin)["side"]
    c = Conv(rng, w, "trade", bio, used)
    if rng.random() < 0.5:
        c.user(U(rng, split, CLOSE_T, c=say_coin(rng, coin), side=side))
        r = c.call(("prepare_close", {"coin": coin}))
    else:
        pct, says = rng.choice([(50, ["half", "50%"]), (25, ["25%", "a quarter"]), (30, ["30%"]), (75, ["75%"])])
        c.user(U(rng, split, CLOSE_PCT_T, c=say_coin(rng, coin), side=side, p=rng.choice(says)))
        r = c.call(("prepare_close", {"coin": coin, "percent": pct}))
    pnl = r["unrealised_pnl"]
    pn = ("+" + usd(pnl)) if pnl >= 0 else ("-" + usd(abs(pnl)))
    c.say(rng.choice([f"I've prepared closing {r['closing']} at market, reduce only. Unrealised PnL on that part is about {pn}. Approve with {bio} on the review.",
                      f"Prepared: close {r['closing']}, market and reduce only, PnL about {pn}. It is waiting for your {bio}. Nothing has moved yet."]))
    return c.record("close", "orders_protect", 1)


# ------------------------------------------------------------------------------------------------ contacts_pay
def contact_world(rng, bio, names, chain_choices=("base",), amount_usdc=None, fam="evm"):
    w = mkw(rng, bio)
    for ch in chain_choices:
        w.add_hold("USDC", ch, amount_usdc if amount_usdc is not None else rng.choice([120.5, 300, 75.25, 540, 1000]))
    w.add_hold("ETH", "base", rnd(rng, 0.02, 1.5, 4))
    w.d["near_account"] = w.d["own"]["near"]
    for n in names:
        f_ = fam if n != names[-1] or fam != "mix" else "solana"
        if fam == "mix":
            f_ = rng.choice(["evm", "evm", "solana"])
        w.d["contacts"].append({"name": n, "address": H.evm_addr(rng) if f_ == "evm" else H.sol_addr(rng), "family": f_})
    return w


ADDR_OWN_T = ["what's my address?", "where do I receive funds?", "show me my wallet address", "give me my deposit address", "what address do I use to get paid?",
              "my address pls", "need my receiving address"]
ADDR_CHAIN_T = ["what's my address on {c}?", "where do I receive on {c}?", "give me my {c} address", "my {c} deposit address pls", "address for {c}?"]


def sc_addr_own(rng, split, used, bio):
    w = contact_world(rng, bio, ["Alice"])
    c = Conv(rng, w, rng.choice(["pay", "core"]), bio, used)
    ch = rng.choice(["base", "arbitrum", "ethereum", "solana", None, None])
    if ch is None:
        c.user(U(rng, split, ADDR_OWN_T))
        r = c.call(("get_address", {}))
        evm = next(a for a in r["addresses"] if a["family"] == "evm")
        sol = next(a for a in r["addresses"] if a["family"] == "solana")
        c.say(f"Your EVM address is {evm['address']}. It works on {join_and([NET_NAME[n] for n in evm['networks']])}. Your Solana address is {sol['address']}.")
    else:
        c.user(U(rng, split, ADDR_CHAIN_T, c=CHAIN_NAME[ch]))
        r = c.call(("get_address", {"chain": ch}))
        a = r["addresses"][0]
        if a["family"] == "evm":
            c.say(f"Your address on {CHAIN_NAME[ch]} is {a['address']}. The same address works on {join_and([NET_NAME[n] for n in a['networks'] if n != ch])}, so only send tokens on those networks to it.")
        else:
            c.say(f"Your Solana address is {a['address']}. Only send Solana tokens to it.")
    return c.record("addr_own", "contacts_pay", 1)


SEND_NAME_T = ["send {a} {s} to {n}", "pay {n} {a} {s}", "transfer {a} {s} to {n}", "can you send {n} {a} {s}?", "{n} needs {a} {s}, send it", "send {n} {a} {s} pls"]


def sc_send_by_name(rng, split, used, bio):
    names = rng.sample(NAMES, 2)
    w = contact_world(rng, bio, names)
    c = Conv(rng, w, rng.choice(["core", "pay"]), bio, used)
    n = names[0]
    a = rng.choice([5, 10, 20, 25, 40, 75])
    c.user(U(rng, split, SEND_NAME_T, a=a, s=rng.choice(["USDC", "usdc"]), n=n if rng.random() < 0.6 else n.lower()))
    r1 = c.call(("get_address", {"contact": n}))
    addr = r1["contacts"][0]["address"]
    r2 = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC"}))
    c.say(f"I've prepared a send of {r2['amount']} on {r2['network']} to {n} ({addr}). {addr_tail(rng, bio)}")
    return c.record("send_by_name", "contacts_pay", 1)


ADD_T = ["save {n} as {a}", "add {n} to my contacts: {a}", "new contact {n}, address {a}", "save this address as {n}: {a}", "remember {n} = {a}", "add {a} as {n}"]


def sc_add_contact(rng, split, used, bio):
    n = rng.choice(NAMES)
    w = mkw(rng, bio)
    c = Conv(rng, w, "pay", bio, used)
    sol = rng.random() < 0.25
    addr = H.sol_addr(rng) if sol else H.evm_addr(rng)
    c.user(U(rng, split, ADD_T, n=n if rng.random() < 0.6 else n.lower(), a=addr))
    r = c.call(("add_contact", {"name": n, "address": addr}))
    t_ = tap(rng, bio)
    c.say(f"I've put up a card to save {n} with {addr}. Check the full address and {t_.lower()} Save. Nothing is saved until you {t_.lower()}.")
    return c.record("add_contact", "contacts_pay", 1)


SCHED_CASES = [("on friday", "Friday", None), ("on the 1st of every month", "Dec 1", "monthly"), ("every week on monday", "Monday", "weekly"),
               ("tomorrow at 6pm", "tomorrow 18:00", None), ("in 3 days", "in 3 days", None), ("on dec 15", "Dec 15", None),
               ("every month starting dec 5", "Dec 5", "monthly"), ("next friday", "next friday", None), ("every week starting friday", "Friday", "weekly")]
SCHED_T = ["remind me to pay {n} {a} usdc {w}", "schedule a payment of {a} usdc to {n} {w}", "set a reminder: {n} gets {a} usdc {w}", "i need to pay {n} {a} usdc {w}, remind me"]


def sc_schedule(rng, split, used, bio):
    names = rng.sample(NAMES, 2)
    w = contact_world(rng, bio, names)
    c = Conv(rng, w, "pay", bio, used)
    n = names[0]
    a = rng.choice([25, 50, 100, 200, 350])
    ph, when, rep = rng.choice(SCHED_CASES)
    c.user(U(rng, split, SCHED_T, n=n, a=a, w=ph))
    args = {"payee": n, "amount": a, "symbol": "USDC", "when": when}
    if rep:
        args["repeat"] = rep
    r = c.call(("schedule_payment", args))
    t_ = tap(rng, bio)
    kind = {"monthly": "a monthly reminder", "weekly": "a weekly reminder"}.get(rep, "a reminder")
    c.say(f"I've drafted {kind} to pay {n} {r['amount']} on {r['network']}, first due {r['next_due']}. {t_} Schedule on the card. It's only a reminder: nothing is sent by itself, and you approve each payment with {bio} when it's due.")
    return c.record("schedule_" + (rep or "once"), "contacts_pay", 1)


def sc_contact_miss(rng, split, used, bio):
    names = rng.sample(NAMES, 3)
    w = contact_world(rng, bio, names[:2])
    c = Conv(rng, w, rng.choice(["core", "pay"]), bio, used)
    n = names[2]
    a = rng.choice([10, 15, 30, 60])
    c.user(U(rng, split, SEND_NAME_T, a=a, s="usdc", n=n))
    c.call(("get_address", {"contact": n}))
    c.say(f"I don't have a contact called {n}. Paste the address here and I'll prepare the send, or tell me to save it as a contact first.")
    return c.record("contact_miss", "contacts_pay", 1)


def sc_contact_ambiguous(rng, split, used, bio):
    first = rng.choice(["Alice", "Sam", "Maria", "Chris"])
    last = rng.sample(["Smith", "Chen", "Lopez", "Khan", "Park"], 2)
    full = [f"{first} {l}" for l in last]
    w = contact_world(rng, bio, full)
    c = Conv(rng, w, "core", bio, used)
    a = rng.choice([10, 20, 35])
    c.user(U(rng, split, SEND_NAME_T, a=a, s="usdc", n=first))
    r = c.call(("get_address", {"contact": first}))
    c.say(r["question"], ask=True)
    pick_i = rng.randrange(2)
    c.user(rng.choice([f"{full[pick_i]}", f"the {last[pick_i]} one", f"{full[pick_i].lower()}"]))
    r1 = c.call(("get_address", {"contact": full[pick_i]}))
    addr = r1["contacts"][0]["address"]
    r2 = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC"}))
    c.say(f"I've prepared a send of {r2['amount']} on {r2['network']} to {full[pick_i]} ({addr}). {addr_tail(rng, bio)}")
    return c.record("contact_ambiguous", "contacts_pay", 1)


# ------------------------------------------------------------------------------------------------ auto_pages
AUTO_TASKS = [
    ("tell me the {C} price", "{C} price check", "Tell me the {C} price and the 24h change.", "tell you the {C} price and the 24h change"),
    ("tell me my total portfolio value", "Portfolio check", "Tell me my total portfolio value.", "tell you your total portfolio value"),
    ("check my activity and tell me if anything failed", "Activity check", "Check my recent activity and tell me if anything failed.", "check your recent activity and tell you if anything failed"),
    ("show me my wallet balances", "Balance check", "Show me my wallet balances.", "show you your wallet balances"),
    ("tell me the {C} price and my open positions", "{C} and positions", "Tell me the {C} price and my open positions.", "tell you the {C} price and your open positions"),
]
AUTO_SCHED = ["every morning at 8", "every day at 9am", "every evening at 6pm", "every friday at 18:00", "every monday at 10", "weekdays at 9am",
              "every hour", "every 15 minutes", "every night at 10pm", "every day at 7:30am"]
AUTO_T = ["{s} {t}", "{t} {s}", "can you {t} {s}?", "set up an automation: {s}, {t}", "{s}, {t}", "i want you to {t} {s}"]


def lowfirst(s):
    return s[:1].lower() + s[1:]


def sc_auto_create(rng, split, used, bio):
    w = mkw(rng, bio)
    c = Conv(rng, w, "pay", bio, used)
    t_, nm, pr, act = rng.choice(AUTO_TASKS)
    coin = rng.choice(["SOL", "BTC", "ETH", "HYPE"])
    s_ = rng.choice(AUTO_SCHED)
    c.user(U(rng, split, AUTO_T, s=s_, t=t_.format(C=coin)))
    r = c.call(("create_automation", {"name": nm.format(C=coin), "prompt": pr.format(C=coin), "schedule": s_}))
    tp = tap(rng, bio)
    c.say(f"I've drafted \"{r['name']}\": {lowfirst(r['schedule'])} it will {act.format(C=coin)}. {tp} Create on the card to turn it on. It can look things up, but it can't approve anything.")
    return c.record("auto_create", "auto_pages", 1)


def sc_auto_alert(rng, split, used, bio):
    w = mkw(rng, bio)
    c = Conv(rng, w, "pay", bio, used)
    coin, thr_say, thr = rng.choice([("BTC", ["100k", "100000", "100,000"], 100000), ("ETH", ["4k", "4000", "4,000"], 4000), ("SOL", ["200", "$200"], 200), ("BTC", ["120k", "120000"], 120000)])
    c.user(U(rng, split, ["tell me when {c} hits {t}", "alert me if {c} goes above {t}", "ping me when {c} reaches {t}", "let me know as soon as {c} is over {t}", "notify me when {c} crosses {t}"], c=coin.lower(), t=rng.choice(thr_say)))
    r = c.call(("create_automation", {"name": f"{coin} price alert", "prompt": f"Check the {coin} price. If it is above {thr}, tell me.", "schedule": "every 5 minutes"}))
    tp = tap(rng, bio)
    c.say(f"I can't send an instant alert, but I've drafted a check every 5 minutes that tells you if {coin} is above {thr}. {tp} Create on the card to turn it on.")
    return c.record("auto_alert", "auto_pages", 1)


PAGE_ASKS = [
    ("payments", "schedule", ["where are my scheduled payments?", "show me my payment reminders", "take me to scheduled payments", "open the schedule page for payments", "where do I see upcoming payments?"]),
    ("payments", "payees", ["show my payees", "where are my saved contacts?", "take me to my contacts list", "open my address book", "where can I edit my payees?"]),
    ("earn", "positions", ["take me to my earn positions", "where do I see my aave deposits?", "show me my earn positions", "open my deposits page", "where are my deposits?"]),
    ("trade", "positions", ["take me to my open positions", "show my positions page", "where do I see my hyperliquid positions?", "open trade positions", "go to my perp positions"]),
    ("settings", "safety", ["open safety settings", "where do I change my limits?", "take me to settings, safety", "where are the assistant limits?", "show me the safety page"]),
    ("settings", "memory", ["open my memory notes", "where is hermit.md?", "take me to memory in settings", "where can I edit my notes?", "show me the memory page"]),
    ("files", None, ["open my files", "where are my attached files?", "take me to files", "show me the files page", "go to files"]),
    ("projects", None, ["open my projects", "take me to projects", "where are my projects?", "show me the projects page", "go to projects"]),
    ("connectors", None, ["open connectors", "where do I connect binance?", "take me to connectors", "show the connectors page", "where are my integrations?"]),
    ("automations", None, ["open my automations", "where are my automations?", "take me to automations", "show the automations page", "go to automations"]),
    ("wallets", None, ["open my wallets", "take me to wallets", "show me the wallets page", "where are my wallets?", "go to wallets"]),
    ("calendar", None, ["open my calendar", "take me to the calendar", "show me the calendar page", "where is my calendar?", "go to calendar"]),
]
PAGE_TAILS = ["{V} it to open the page.", "{V} it when you are ready.", "It takes you straight there."]


def page_label(page, tab):
    return PAGES[page] + (", " + cap1(tab) if tab else "")


def button_line(rng, page, tab):
    return rng.choice([f"I put a button below that opens {page_label(page, tab)}.", f"I put a button below that opens {page_label(page, tab)}.",
                       f"I put a button below that takes you to {page_label(page, tab)}."])


def sc_page_open(rng, split, used, bio):
    w = mkw(rng, bio)
    page, tab, asks = rng.choice(PAGE_ASKS)
    c = Conv(rng, w, rng.choice(["core", "pay", "trade", "earn", "solana"]), bio, used)
    c.user(styl(rng, pick(rng, split, asks)))
    args = {"page": page}
    if tab:
        args["tab"] = tab
    c.call(("open_page", args))
    c.say(button_line(rng, page, tab) + " " + rng.choice(PAGE_TAILS).format(V=tap(rng, bio)))
    return c.record("page_open", "auto_pages", 1)


POINTERS = [
    ("calendar", None, ["add a meeting with dave tomorrow at 3pm", "put a dentist appointment on my calendar", "schedule a call for friday afternoon", "add a to-do: renew passport"],
     "I can't add calendar items from this chat yet."),
    ("projects", None, ["create a project called taxes", "start a new project for my dapp", "make a project folder for the dao"],
     "I can't create projects from this chat yet."),
    ("settings", "safety", ["raise my daily limit", "turn the daily limit up", "change my assistant spending limit"],
     "I can't change your limits from this chat. You change them yourself."),
    ("connectors", None, ["connect my binance account", "add the binance connector", "link my exchange account"],
     "I can't connect accounts from this chat yet."),
]


def sc_pointer(rng, split, used, bio):
    w = mkw(rng, bio)
    page, tab, asks, sent = rng.choice(POINTERS)
    c = Conv(rng, w, rng.choice(["core", "pay", "trade"]), bio, used)
    c.user(styl(rng, pick(rng, split, asks)))
    args = {"page": page}
    if tab:
        args["tab"] = tab
    c.call(("open_page", args))
    c.say(f"{sent} {button_line(rng, page, tab)}")
    return c.record("pointer", "auto_pages", 1)


# ------------------------------------------------------------------------------------------------ yields
PROTOCOLS = [("Morpho", "no", True), ("Aave v3", "no", True), ("Compound v3", "no", False), ("Spark", "no", False), ("Fluid", "no", False),
             ("Euler", "no", False), ("Pendle", "no", False), ("Moonwell", "no", False), ("Aerodrome", "yes", False), ("Yearn", "no", False),
             ("Ethena", "no", False), ("Kamino", "no", False)]
YCHAINS = ["Base", "Arbitrum", "Ethereum", "Polygon", "Solana"]


def make_yields(rng, w, n, asset="USDC", ensure_aave=("base", "arbitrum", "ethereum")):
    rows, seen = [], set()
    for ch in ensure_aave:
        base = round(rng.uniform(2.2, 4.6), 2)
        rew = round(rng.choice([0, 0, 0.31, 0.2]), 2)
        apy = round(base + rew, 2)
        rows.append({"apy_30d_pct": round(apy - rng.uniform(0.05, 0.4), 2), "apy_base_pct": base, "apy_pct": apy, "apy_reward_pct": rew, "asset": asset,
                     "can_deposit_in_hermit": True, "chain": CHAIN_NAME[ch], "il_risk": "no", "pool_meta": None, "protocol": "Aave v3",
                     "tvl_usd": int(rng.choice([180, 330, 640, 950, 1400]) * 1e6)})
        seen.add(("Aave v3", ch))
        w.d["aave_apy"][ch] = apy
    tries = 0
    while len(rows) < n and tries < 200:
        tries += 1
        pname, il, _ = rng.choice(PROTOCOLS[:])
        if pname == "Aave v3":
            continue
        ch = rng.choice(YCHAINS)
        if (pname, ch.lower()) in seen:
            continue
        seen.add((pname, ch.lower()))
        base = round(rng.uniform(1.8, 8.4), 2)
        rew = round(rng.choice([0, 0, 0.5, 1.2, 0.8]), 2)
        apy = round(base + rew, 2)
        rows.append({"apy_30d_pct": round(apy - rng.uniform(-0.3, 0.6), 2), "apy_base_pct": base, "apy_pct": apy, "apy_reward_pct": rew, "asset": asset,
                     "can_deposit_in_hermit": False, "chain": ch, "il_risk": il, "pool_meta": rng.choice([None, None, "3x"] if il == "yes" else [None]),
                     "protocol": pname, "tvl_usd": int(rng.choice([4.2, 12, 38, 61, 120, 260, 420]) * 1e6)})
    w.d["yields"] = rows
    return rows


# ------------------------------------------------------------------------------------------------ multistep (at most 3 steps)
def sc_ms_bridge_deposit(rng, split, used, bio):
    w = mkw(rng, bio)
    fa, tb = rng.sample(["base", "arbitrum", "ethereum"], 2)
    a = rng.choice([20, 25, 40, 50, 100])
    w.add_hold("USDC", fa, a + rng.choice([5, 30, 120]))
    make_yields(rng, w, 4)
    c = Conv(rng, w, "earn", bio, used)
    c.user(U(rng, split, ["bridge {a} usdc from {f} to {t} then put it in aave", "move {a} usdc from {f} to {t} and then deposit it into aave", "send {a} usdc {f} -> {t}, after that deposit in aave", "i want to bridge {a} usdc to {t} from {f} and earn on it in aave"],
             a=a, f=CHAIN_NAME[fa], t=CHAIN_NAME[tb]))
    r = c.call(("prepare_swap", {"from_symbol": "USDC", "from_chain": fa, "to_symbol": "USDC", "to_chain": tb, "amount": a}))
    c.say(f"I've prepared the bridge: {r['from']} to {r['to']} through NEAR Intents, waiting for your {bio}. The Aave deposit comes after it lands. Tell me when it has and I'll prepare it.")
    out = r["amount_out"]

    def landed(wd):
        wd.add_hold("USDC", fa, plain(Decimal(next(h for h in wd.d["holdings"] if h["chain"] == fa and h["symbol"] == "USDC")["amount"]) - Decimal(plain(a))))
        wd.add_hold("USDC", tb, out)
        wd.d["activity"].insert(0, {"status": "confirmed", "time": "2026-11-11 09:12", "what": f"Swap {a} USDC on {CHAIN_NAME[fa]} to USDC on {CHAIN_NAME[tb]}"})
    c.mutate(landed)
    c.user(rng.choice(["it landed", "ok it's there", "the bridge finished", "arrived, go on", "it landed, now the deposit"]))
    r1, r2 = c.call(("get_activity", {"limit": 5}), ("get_balances", {"chain": tb}))
    amt = float(out)
    r3 = c.call(("prepare_deposit", {"project": "aave-v3", "symbol": "USDC", "chain": tb, "amount": short(amt)}))
    c.say(f"Activity shows the bridge as confirmed, and you have {out} USDC on {CHAIN_NAME[tb]}. I've prepared depositing it into {r3['pool']} at {plain(r3['apy_pct'])}% APY, which can change. Approve with {bio}.")
    return c.record("ms_bridge_deposit", "multistep", 1)


def sc_ms_quote_chain(rng, split, used, bio):
    w = mkw(rng, bio)
    fa, tb = rng.sample(["base", "arbitrum", "ethereum"], 2)
    amt = rng.choice([120.5, 80, 250, 61.5, 340])
    w.add_hold("USDC", fa, amt)
    c = Conv(rng, w, "core", bio, used)
    to = rng.choice(["ETH", "SOL"] if tb != "solana" else ["SOL"])
    to = "ETH"
    c.user(U(rng, split, ["how much {t} on {b} would i get for half my usdc on {f}?", "quote me half of my {f} usdc into {t} on {b}", "what would half my usdc on {f} turn into as {t} on {b}?", "if i swapped half my {f} usdc to {t} on {b}, what do i get?"],
             t=to, b=CHAIN_NAME[tb], f=CHAIN_NAME[fa]))
    c.call(("get_balances", {}))
    half = c.call(("calculator", {"expression": f"{plain(amt)} / 2"}))["result"]
    q = c.call(("get_quote", {"from_symbol": "USDC", "from_chain": fa, "to_symbol": to, "to_chain": tb, "amount": float(half) if "." in half else int(half)}))
    c.say(f"Half of your USDC on {CHAIN_NAME[fa]} is {half}. That gets you {q['to']} through NEAR Intents, with about {usd(q['total_cost_usd'])} in total costs. This is only a quote. Want me to prepare the swap?")
    return c.record("ms_quote_chain", "multistep", 1)


def sc_ms_risk_stop(rng, split, used, bio):
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE", "AVAX"])
    side = rng.choice(["long", "long", "short"])
    w = mkw(rng, bio)
    w.d["hl"]["value"] = rng.choice([250.5, 400, 800])
    c = Conv(rng, w, "trade", bio, used)
    size = rng.choice([50, 100, 200])
    lev = rng.choice([2, 3, 5])
    p = rng.choice([5, 7, 8, 10])
    word = "under" if side == "long" else "above"
    c.user(U(rng, split, ["{sd} {c} ${s} at {l}x with a stop {p}% {w} the price", "open a {l}x {sd} on {c}, ${s}, stop {p}% {w} the current price", "{sd} ${s} of {c}, {l}x leverage, put the stop {p}% {w} where it is now", "i want a {c} {sd}: ${s}, {l}x, market, stop loss {p}% {w} the price"],
             sd=side, c=say_coin(rng, coin), s=size, l=lev, p=p, w=word))
    px = c.call(("get_price", {"symbol": coin}))["price"]
    fac = round((100 - p) / 100 if side == "long" else (100 + p) / 100, 2)
    dp = ASSETS[coin][2]
    res = c.call(("calculator", {"expression": f"round({plain(px)} * {plain(fac)}, {max(dp, 2)})"}))["result"]
    stop = float(res)
    c.call(("prepare_trade", {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market", "stop_loss": short(stop)}))
    c.say(f"I've prepared a {usd(size)} {coin} {side} at {lev}x, market order, with the stop loss {p}% {word} today's {usd(px)} at {usd(stop)}. Approve with {bio}.")
    return c.record("ms_risk_stop", "multistep", 1)


def sc_ms_yields_deposit(rng, split, used, bio):
    w = mkw(rng, bio)
    ch = rng.choice(["base", "arbitrum", "ethereum"])
    w.add_hold("USDC", ch, rng.choice([120.5, 300, 540, 75]))
    rows = make_yields(rng, w, 5)
    c = Conv(rng, w, "earn", bio, used)
    c.user(U(rng, split, ["where can i earn on my usdc?", "best yield for usdc right now?", "what are the top usdc yields?", "show me where usdc earns the most", "which pools pay the most on usdc?"]))
    r = c.call(("get_yields", {"asset": "USDC"}))
    p0, p1 = r["pools"][0], r["pools"][1]
    dep = [p for p in r["pools"] if p["can_deposit_in_hermit"]]
    c.say(f"The top USDC pool is {p0['protocol']} on {p0['chain']} at {plain(p0['apy_pct'])}%, then {p1['protocol']} on {p1['chain']} at {plain(p1['apy_pct'])}%. APYs change. I can prepare Aave v3 deposits only; the others are in Earn.")
    a = rng.choice([25, 50, 100])
    h = w.hold("USDC", ch)[0]
    a = min(a, int(float(h["amount"])))
    c.user(rng.choice([f"put {a} in aave", f"ok deposit {a} usdc into aave", f"deposit {a} into aave pls"]))
    r2 = c.call(("prepare_deposit", {"project": "aave-v3", "symbol": "USDC", "amount": a}))
    c.say(f"I've prepared a deposit of {r2['amount']} into {r2['pool']} at {plain(r2['apy_pct'])}% APY, which can change. Approve with {bio}.")
    return c.record("ms_yields_deposit", "multistep", 1)


def sc_ms_sell_send(rng, split, used, bio):
    w, held, _ = solana_world(rng, bio)
    n = rng.choice(NAMES)
    w.d["contacts"].append({"name": n, "address": H.evm_addr(rng), "family": "evm"})
    c = Conv(rng, w, "solana", bio, used)
    sym = rng.choice(held)
    c.user(U(rng, split, ["sell all my {s} and send the usdc to {n}", "dump my {s} and then send the usdc to {n}", "sell everything in {s}, then pay {n} the usdc", "cash out {s} to usdc and send it to {n}"], s=sym, n=n))
    r = c.call(("prepare_sell", {"token": sym, "percent": 100, "receive": "USDC"}))
    c.say(f"I've prepared selling {r['you_sell']} for {r['you_get']}. Approve it with {bio}. The send to {n} comes next: tell me when the sale has landed and I'll check your balance and prepare it.")
    return c.record("ms_sell_send", "multistep", 1)


def sc_ms_pair(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", 300)
    w.add_hold("USDT", "base", 200)
    a1, a2 = H.evm_addr(rng), H.evm_addr(rng)
    n1, n2 = rng.choice([10, 20, 25]), rng.choice([5, 15, 30])
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["send {n1} USDC to {a1} and {n2} USDT to {a2}", "two sends please: {n1} usdc to {a1}, and {n2} usdt to {a2}", "pay {a1} {n1} usdc and {a2} {n2} usdt"], n1=n1, n2=n2, a1=a1, a2=a2))
    r = c.call(("prepare_send", {"to_address": a1, "amount": n1, "symbol": "USDC"}))
    c.say(f"I've prepared the first send: {r['amount']} on {r['network']} to {a1}. Approve it with {bio}. The second one, {n2} USDT to {a2}, comes next: tell me when you're ready.")
    return c.record("ms_pair", "multistep", 1)


def sc_ms_prices(rng, split, used, bio):
    w = mkw(rng, bio)
    cs = rng.sample(["BTC", "ETH", "SOL", "HYPE", "BNB"], 3)
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["price of {a}, {b} and {c}?", "give me {a}, {b} and {c} prices", "how are {a}, {b} and {c} doing right now?", "quick check on {a} {b} {c}"], a=cs[0].lower(), b=cs[1].lower(), c=cs[2].lower()))
    rs = c.call(*[("get_price", {"symbol": x}) for x in cs])
    parts = [f"{r['symbol']} {usd(r['price'])} ({'up' if r['change_24h_pct'] > 0 else 'down'} {plain(abs(r['change_24h_pct']))}%)" for r in rs]
    c.say(f"{join_and(parts)} over the last 24h.")
    return c.record("ms_prices", "multistep", 1)


# ------------------------------------------------------------------------------------------------ notes_name
def sc_note_leverage(rng, split, used, bio):
    cap = rng.choice([3, 5, 5, 10])
    lev = cap * rng.choice([2, 3]) if cap < 10 else 20
    coin = rng.choice(["ETH", "BTC", "SOL"])
    w = mkw(rng, bio)
    w.d["hl"]["value"] = 800
    nm = rng.choice(USER_NAMES) if rng.random() < 0.5 else None
    line = rng.choice([f"Never suggest more than {cap}x leverage.", f"Max leverage {cap}x, always.", f"Cap my leverage at {cap}x."])
    c = Conv(rng, w, "trade", bio, used, name=nm, notes=[line])
    size = rng.choice([50, 100, 200])
    c.user(U(rng, split, ["long {c} ${s} at {l}x market", "go long {c} with ${s}, {l}x", "open a {l}x long on {c} for ${s}", "{c} long, ${s}, {l}x leverage, market order"], c=say_coin(rng, coin), s=size, l=lev))
    c.call(("prepare_trade", {"coin": coin, "side": "long", "size_usd": size, "leverage": cap, "order_type": "market"}))
    c.say(f"Your notes cap leverage at {cap}x, so I prepared a {usd(size)} {coin} long at {cap}x instead of {lev}x, market order. Approve with {bio}, or tell me to change it.")
    return c.record("note_leverage", "notes_name", 1)


def sc_note_name(rng, split, used, bio, use_name):
    nm = rng.choice(USER_NAMES)
    w = mkw(rng, bio)
    c = Conv(rng, w, "core", bio, used, name=nm)
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    c.user(U(rng, split, ["what's {c} at?", "price of {c}?", "how is {c} doing today?", "{c} price check", "where's {c} trading?"], c=say_coin(rng, coin)))
    r = c.call(("get_price", {"symbol": coin}))
    ch = r["change_24h_pct"]
    base = f"{coin} is {usd(r['price'])}, {'up' if ch > 0 else 'down'} {plain(abs(ch))}% in the last 24h."
    c.say((f"{nm}, " + base[:1].lower() + base[1:] if not base[:2].isupper() and not base[:3].isupper() else f"{nm}, {base}") if use_name else base)
    return c.record("note_name_" + ("used" if use_name else "unused"), "notes_name", 1)


def sc_note_short(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", rng.choice([120.5, 300]))
    w.add_hold("USDC", "arbitrum", rng.choice([40, 75.5]))
    c = Conv(rng, w, "core", bio, used, notes=["Keep answers very short."])
    c.user(U(rng, split, ["what do I have in my wallet?", "show me my balances", "what's in my wallet right now?", "how much do I hold?"]))
    r = c.call(("get_balances", {}))
    parts = [f"{h['amount']} {h['symbol']} on {CHAIN_NAME[h['chain']]}" for h in r["holdings"]]
    c.say(f"You have {join_and(parts)}.")
    return c.record("note_short", "notes_name", 1)


KNOW = [
    ("stop loss", ["what is a stop loss?", "explain stop losses", "what does a stop loss do?", "how does a stop loss work?", "why would i use a stop loss?", "quick question, what is a stop?"],
     "A stop loss closes your position if the price moves against you to a level you pick. It limits your loss, but in a fast market it can fill worse than your price."),
    ("leverage", ["what is leverage?", "explain leverage to me", "how does leverage work?", "what does 5x leverage mean?", "is leverage risky?", "quick question, what is leverage?"],
     "Leverage lets you hold a bigger position than your cash. It boosts gains and losses, and a small move against you can liquidate the position."),
    ("slippage", ["what is slippage?", "what does slippage mean?", "explain slippage", "why did i get a worse price than expected?", "what causes slippage?", "how do i avoid slippage?"],
     "Slippage is the gap between the price you expected and the price you got. It grows on thin markets and big orders."),
    ("liquidation", ["what is liquidation?", "what does getting liquidated mean?", "explain liquidation", "how do i avoid liquidation?", "when do i get liquidated?", "what happens at liquidation?"],
     "Liquidation is when the exchange closes your leveraged position because your margin can't cover the loss. Lower leverage keeps it further from the price."),
    ("funding", ["what is funding on a perp?", "what does funding rate mean?", "explain perp funding", "who pays funding?", "what is a funding rate?", "why is funding negative sometimes?"],
     "Funding is a small payment between longs and shorts that keeps a perp close to the spot price. When funding is positive, longs pay shorts."),
    ("apy", ["what is apy?", "what does apy mean?", "explain apy", "is apy fixed?", "what does 5% apy mean?", "how is apy different from interest?"],
     "APY is the yearly return including compounding. On lending pools it changes with demand, so today's rate is not a promise."),
]
SIMPLE_KNOW = {
    "stop loss": "A stop loss is an automatic exit. You pick a price, and if the coin falls to it, your trade closes so you lose less.",
    "leverage": "Leverage is borrowing to make a bigger bet. You can win more, but you can lose everything faster.",
    "slippage": "Slippage means you got a slightly different price than you saw. It happens more with big orders.",
    "liquidation": "Liquidation is when a leveraged trade is shut down because you ran out of margin. You lose that margin.",
    "funding": "Funding is a small fee that longs and shorts pay each other. It keeps the perp price near the real price.",
    "apy": "APY is how much you earn in a year, with the interest added back in. The rate moves, so it is not fixed.",
}


def sc_note_simple(rng, split, used, bio):
    key, qs, ans = rng.choice(KNOW)
    w = mkw(rng, bio)
    c = Conv(rng, w, "core", bio, used, notes=["I'm new to crypto, explain simply."])
    c.user(styl(rng, pick(rng, split, qs)))
    c.say(SIMPLE_KNOW[key])
    return c.record("note_simple", "notes_name", 1)


def sc_note_addrbook(rng, split, used, bio):
    n = rng.choice(NAMES)
    addr = H.evm_addr(rng)
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", 300)
    c = Conv(rng, w, "core", bio, used, notes=[f"{n}: {addr}"])
    a = rng.choice([10, 20, 50])
    c.user(U(rng, split, SEND_NAME_T, a=a, s="usdc", n=n))
    c.say(f"Your notes list {n} as {addr}. I can only send to an address you type in this chat, so paste it here and I'll prepare the send.")
    return c.record("note_addrbook", "notes_name", 1)


# ------------------------------------------------------------------------------------------------ recovery_errors
def sc_rec_quote_min(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", 100)
    c = Conv(rng, w, "core", bio, used)
    a = rng.choice([0.2, 0.5, 0.8])
    c.user(U(rng, split, ["quote me {a} usdc from base to eth on arbitrum", "how much eth on arbitrum for {a} usdc on base?", "what would {a} usdc on base get me in eth on arbitrum?"], a=a))
    r = c.call(("get_quote", {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "ETH", "to_chain": "arbitrum", "amount": a}))
    c.say("That amount is below the minimum for this route, about $1. Give me a bigger amount and I'll quote it again.")
    return c.record("rec_quote_min", "recovery_errors", 1)


def sc_rec_blocked(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", 1500)
    w.d["cap_usd"] = rng.choice([100, 250, 500])
    c = Conv(rng, w, "core", bio, used)
    a = rng.choice([700, 900, 1200])
    c.user(U(rng, split, ["swap {a} usdc on base to eth on arbitrum", "bridge {a} usdc from base into eth on arbitrum", "turn {a} usdc on base into eth on arbitrum"], a=a))
    r = c.call(("prepare_swap", {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "ETH", "to_chain": "arbitrum", "amount": a}))
    why = lowfirst(r["reasons"][0]).rstrip(".")
    c.say(f"I couldn't prepare that: {why}. A smaller amount might fit. Tell me one and I'll try it.")
    return c.record("rec_blocked", "recovery_errors", 1)


def sc_rec_insufficient(rng, split, used, bio):
    w = mkw(rng, bio)
    ch = rng.choice(["base", "arbitrum"])
    have = rng.choice([20, 35.5, 12.25])
    w.add_hold("USDC", ch, have)
    c = Conv(rng, w, "earn", bio, used)
    a = rng.choice([50, 100, 200])
    c.user(U(rng, split, ["deposit {a} usdc in aave", "put {a} usdc into aave", "lend {a} usdc on aave for me", "i want to earn on {a} usdc with aave"], a=a))
    r = c.call(("prepare_deposit", {"project": "aave-v3", "symbol": "USDC", "amount": a}))
    c.say(f"You only hold {r['holding']}, which is less than {a}. Want me to prepare a deposit of {r['holding'].split(' on ')[0]} instead?")
    return c.record("rec_insufficient", "recovery_errors", 1)


def sc_rec_needs_chain(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", rng.choice([120.5, 300]))
    w.add_hold("USDC", "arbitrum", rng.choice([40, 75]))
    addr = H.evm_addr(rng)
    c = Conv(rng, w, "core", bio, used)
    a = rng.choice([10, 15, 25])
    c.user(U(rng, split, ["send {a} usdc to {x}", "pay {x} {a} usdc", "transfer {a} usdc to {x} pls", "can you send {a} usdc to {x}?"], a=a, x=addr))
    r = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC"}))
    c.say(r["question"], ask=True)
    ch = rng.choice(["base", "arbitrum"])
    c.user(rng.choice([ch, f"on {ch}", f"{ch} please", f"use {ch}"]))
    r2 = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC", "chain": ch}))
    c.say(f"I've prepared a send of {r2['amount']} on {r2['network']} to {addr}. {addr_tail(rng, bio)}")
    return c.record("rec_needs_chain", "recovery_errors", 1)


def sc_rec_timeout(rng, split, used, bio, twice=False):
    w = mkw(rng, bio)
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    w.d["fail"] = {"get_price": ["network timeout, try again"] * (2 if twice else 1)}
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["what's {c} at right now?", "price of {c} pls", "{c} price?", "how much is {c} today?", "current {c} price"], c=say_coin(rng, coin)))
    c.call(("get_price", {"symbol": coin}))
    r = c.call(("get_price", {"symbol": coin}))
    if "error" in r:
        c.say("The price feed isn't answering right now, so I don't have a live price. Try again in a minute.")
        return c.record("rec_timeout_stop", "recovery_errors", 1)
    ch = r["change_24h_pct"]
    c.say(f"{coin} is {usd(r['price'])}, {'up' if ch > 0 else 'down'} {plain(abs(ch))}% in the last 24h. The first try timed out, so I asked again.")
    return c.record("rec_timeout_retry", "recovery_errors", 1)


def sc_rec_margin(rng, split, used, bio):
    w = mkw(rng, bio)
    w.d["hl"]["value"] = rng.choice([12.4, 30.25, 40.0])
    coin = rng.choice(["ETH", "SOL", "BTC"])
    c = Conv(rng, w, "trade", bio, used)
    size = rng.choice([300, 500, 800])
    lev = rng.choice([3, 5])
    c.user(U(rng, split, ["long {c} ${s} at {l}x market", "open a {l}x {c} long with ${s}", "buy ${s} of {c} perp, {l}x"], c=say_coin(rng, coin), s=size, l=lev))
    r = c.call(("prepare_trade", {"coin": coin, "side": "long", "size_usd": size, "leverage": lev, "order_type": "market"}))
    m = re.search(r"([\d.]+) USDC available, ([\d.]+) needed", r["error"])
    c.say(f"There isn't enough margin: you have {m.group(1)} USDC available and this trade needs {m.group(2)}. Try a smaller size or lower leverage, or add funds.")
    return c.record("rec_margin", "recovery_errors", 1)


def sc_rec_sell_err(rng, split, used, bio, mode):
    w, held, other = solana_world(rng, bio, held=1, extra_catalog=2)
    c = Conv(rng, w, "solana", bio, used)
    if mode == "not_held":
        sym = rng.choice(other)
        c.user(U(rng, split, ["sell all my {s}", "dump my {s}", "sell half of my {s}"], s=sym))
        pct = 50 if "half" in c.msgs[-1]["content"] else 100
        c.call(("prepare_sell", {"token": sym, "percent": pct}))
        c.say(f"You don't hold any {sym} on Solana, so there is nothing to sell.")
        return c.record("rec_sell_not_held", "recovery_errors", 1)
    if mode == "frozen":
        sym = held[0]
        w.d["spot"][0]["frozen"] = True
        c.user(U(rng, split, ["sell all my {s}", "get rid of my {s}", "dump everything in {s}"], s=sym))
        r = c.call(("prepare_sell", {"token": sym, "percent": 100}))
        c.say(f"I couldn't prepare that: {lowfirst(r['reasons'][0])} Nothing I can do changes that.")
        return c.record("rec_sell_frozen", "recovery_errors", 1)
    sym = "PUMP"
    w.d["spot"] = []
    m1, m2 = mint_for(rng), mint_for(rng)
    w.add_spot(sym, m1, 120, None, False)
    w.add_spot(sym, m2, 4, None, False)
    c.user(U(rng, split, ["sell half of my pump", "sell 50% of my pump", "dump half my pump"]))
    r = c.call(("prepare_sell", {"token": sym, "percent": 50}))
    c.say(r["question"], ask=True)
    c.user(rng.choice(["the one with 120", "the bigger one", "the 120 one"]))
    r2 = c.call(("prepare_sell", {"token": m1, "percent": 50}))
    c.say(ans_sell(rng, r2, bio))
    return c.record("rec_sell_choice", "recovery_errors", 1)


def sc_rec_no_position(rng, split, used, bio):
    w = mkw(rng, bio)
    w.d["hl"]["value"] = 250.5
    coin = rng.choice(["SOL", "ETH", "BTC"])
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, ["close my {c} position", "close my {c} long", "exit {c}", "shut my {c} trade"], c=say_coin(rng, coin)))
    c.call(("prepare_close", {"coin": coin}))
    c.say(f"You have no open {coin} position on Hyperliquid, so there is nothing to close.")
    return c.record("rec_no_position", "recovery_errors", 1)


# ------------------------------------------------------------------------------------------------ followups_stale
def sc_fu_same_for(rng, split, used, bio):
    c1, c2 = rng.sample(["SOL", "ETH", "BTC", "HYPE", "AVAX"], 2)
    side = rng.choice(["long", "short"])
    size, lev = rng.choice([50, 100, 150]), rng.choice([2, 3, 5])
    w = mkw(rng, bio)
    w.d["hl"]["value"] = 800
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, ["{sd} {c} ${s} {l}x market", "open a {l}x {sd} on {c} for ${s}", "{sd} ${s} {c} at {l}x", "go {sd} {c}, ${s}, {l}x leverage"], sd=side, c=say_coin(rng, c1), s=size, l=lev))
    c.call(("prepare_trade", {"coin": c1, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}))
    c.say(f"I've prepared a {usd(size)} {c1} {side} at {lev}x, market order. Approve with {bio} on the review.")
    c.user(rng.choice([f"do the same for {c2.lower()}", f"same thing on {c2}", f"now the same for {say_coin(rng, c2)}", f"and the same trade for {c2}"]))
    c.call(("prepare_trade", {"coin": c2, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}))
    c.say(f"I've prepared a {usd(size)} {c2} {side} at {lev}x, market order. The {c1} review is still open, so close it if you don't want it. Approve with {bio}.")
    return c.record("fu_same_for", "followups_stale", 1)


def sc_fu_cancel_that(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", 300)
    addr = H.evm_addr(rng)
    a = rng.choice([10, 25, 40])
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["send {a} usdc to {x}", "pay {x} {a} usdc", "send {x} {a} usdc please"], a=a, x=addr))
    r = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC"}))
    c.say(f"I've prepared a send of {r['amount']} on {r['network']} to {addr}. {addr_tail(rng, bio)}")
    c.user(rng.choice(["cancel that", "wait, cancel that", "actually never mind, cancel it", "undo that"]))
    c.say(f"{tap(rng, bio)} Cancel on the review; nothing was sent.")
    return c.record("fu_cancel_that", "followups_stale", 1)


def sc_fu_is_it_there(rng, split, used, bio):
    w = mkw(rng, bio)
    fa, tb = rng.sample(["base", "arbitrum", "ethereum"], 2)
    a = rng.choice([20, 50, 100])
    w.add_hold("USDC", fa, a + 100)
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["bridge {a} usdc from {f} to {t}", "move {a} usdc from {f} to {t}", "send {a} usdc {f} to {t} by bridge"], a=a, f=CHAIN_NAME[fa], t=CHAIN_NAME[tb]))
    r = c.call(("prepare_swap", {"from_symbol": "USDC", "from_chain": fa, "to_symbol": "USDC", "to_chain": tb, "amount": a}))
    c.say(f"I've prepared the bridge: {r['from']} to {r['to']} through NEAR Intents, waiting for your {bio}. {nothing_moved(rng)}")

    def pending(wd):
        wd.d["activity"].insert(0, {"status": "in_progress", "time": "2026-11-11 09:20", "what": f"Swap {a} USDC on {CHAIN_NAME[fa]} to USDC on {CHAIN_NAME[tb]}"})
    c.mutate(pending)
    c.user(rng.choice(["approved it, is it there yet?", "i approved, did it arrive?", "ok approved. has it landed?", "approved, is it done yet?"]))
    r2 = c.call(("get_activity", {"limit": 5}))
    c.say(f"Not confirmed yet. Activity shows the swap of {a} USDC as in progress, and the quote said about {r['seconds']} seconds. Ask me again in a minute.")
    return c.record("fu_is_it_there", "followups_stale", 1)


def sc_fu_balance_after(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", rng.choice([120.5, 300, 75.25]))
    addr = H.evm_addr(rng)
    a = rng.choice([10, 20])
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["send {a} usdc to {x}", "pay {x} {a} usdc"], a=a, x=addr))
    r = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC"}))
    c.say(f"I've prepared a send of {r['amount']} on {r['network']} to {addr}. {addr_tail(rng, bio)}")
    c.user(rng.choice(["approved. what's my balance now?", "ok approved it, how much do I have left?", "i approved, what's my usdc balance?"]))
    r2 = c.call(("get_balances", {}))
    h = r2["holdings"][0]
    c.say(f"Your balance reads {h['amount']} {h['symbol']} on {CHAIN_NAME[h['chain']]}. If you approved just now it may not show yet; Activity has the real status.")
    return c.record("fu_balance_after", "followups_stale", 1)


def sc_fu_make_it(rng, split, used, bio):
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", rng.choice([300, 540, 1000]))
    addr = H.evm_addr(rng)
    a, b = rng.choice([(10, 50), (5, 25), (20, 100)])
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["send {a} usdc to {x}", "pay {x} {a} usdc", "can you send {a} usdc to {x}?"], a=a, x=addr))
    r = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC"}))
    c.say(f"I've prepared a send of {r['amount']} on {r['network']} to {addr}. {addr_tail(rng, bio)}")
    c.user(rng.choice([f"make it {b}", f"actually make that {b}", f"no, {b} not {a}", f"change it to {b} usdc"]))
    r2 = c.call(("prepare_send", {"to_address": addr, "amount": b, "symbol": "USDC"}))
    c.say(f"I've prepared a new send of {r2['amount']} on {r2['network']} to {addr}. The earlier review is still open, so cancel it. Approve this one with {bio}.")
    return c.record("fu_make_it", "followups_stale", 1)


# ------------------------------------------------------------------------------------------------ answers (short, mostly no extra steps)
def sc_ans_price(rng, split, used, bio):
    w = mkw(rng, bio)
    coin = rng.choice(COINS)
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["what's {c} at?", "price of {c}?", "how much is {c} right now?", "{c} price pls", "where is {c} trading?"], c=say_coin(rng, coin)))
    r = c.call(("get_price", {"symbol": coin}))
    ch = r["change_24h_pct"]
    c.say(rng.choice([f"{coin} is {usd(r['price'])}, {'up' if ch > 0 else 'down'} {plain(abs(ch))}% in the last 24h.",
                      f"{coin} trades at {usd(r['price'])}, {'up' if ch > 0 else 'down'} {plain(abs(ch))}% over 24h."]))
    return c.record("ans_price", "answers", 1)


def sc_ans_portfolio(rng, split, used, bio):
    w = solana_world(rng, bio, held=2)[0]
    w.add_hold("USDC", "base", 120.5)
    w.d["hl"]["value"] = rng.choice([250.5, 400])
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, ["how much do I have in total?", "what's my net worth?", "total value of everything i own?", "how much am I worth across everything?"]))
    r = c.call(("get_portfolio", {}))
    t0, t1 = r["top"][0], r["top"][1]
    c.say(f"About {usd(r['total_usd'])} in total in your {r['wallet']} wallet. Most of it is {t0['what']} ({usd(t0['usd'])}), then {t1['what']} ({usd(t1['usd'])}).")
    return c.record("ans_portfolio", "answers", 1)


def sc_ans_knowledge(rng, split, used, bio):
    key, qs, ans = rng.choice(KNOW)
    w = mkw(rng, bio)
    c = Conv(rng, w, rng.choice(["core", "trade"]), bio, used)
    c.user(styl(rng, pick(rng, split, qs)))
    c.say(ans)
    return c.record("ans_knowledge", "answers", 1)


def sc_ans_view(rng, split, used, bio):
    w = mkw(rng, bio)
    w.d["hl"]["value"] = rng.choice([250.5, 400, 800])
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    c = Conv(rng, w, "trade", bio, used)
    c.user(U(rng, split, ["should I long {c} here?", "what do you think of {c} right now?", "give me a view on {c}", "is {c} a buy at this price?"], c=say_coin(rng, coin)))
    rp, ra = c.call(("get_price", {"symbol": coin}), ("hyperliquid_account", {}))
    ch = rp["change_24h_pct"]
    c.say(f"{coin} is {usd(rp['price'])}, {'up' if ch > 0 else 'down'} {plain(abs(ch))}% in 24h. I can't say where it goes next, and a long risks a drop and liquidation at high leverage. You have {usd(ra['available_usdc'])} available. Want me to prepare a small one?")
    return c.record("ans_view", "answers", 1)


# ================================================================================================ PART 1b: tool-absent rows
def _slots_stop(rng, w):
    c = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    mark = w.d["prices"][c]
    p = round(mark * rng.choice([0.88, 0.9, 0.93]), ASSETS[c][2])
    return {"c": say_coin(rng, c), "p": plain(p), "side": rng.choice(["long", "short"]), "_coin": c}


def _slots_tp(rng, w):
    c = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    mark = w.d["prices"][c]
    return {"c": say_coin(rng, c), "p": plain(round(mark * rng.choice([1.1, 1.15, 1.2]), ASSETS[c][2])), "_coin": c}


def _slots_tpsl(rng, w):
    c = rng.choice(["SOL", "ETH", "BTC"])
    mark = w.d["prices"][c]
    dp = ASSETS[c][2]
    return {"c": say_coin(rng, c), "s": plain(round(mark * 0.9, dp)), "p": plain(round(mark * 1.15, dp)), "_coin": c}


def _slots_trade(rng, w):
    return {"c": say_coin(rng, rng.choice(["SOL", "ETH", "BTC", "HYPE", "AVAX"])), "s": rng.choice([50, 100, 200, 500]), "l": rng.choice([2, 3, 5, 10])}


def _slots_coin(rng, w):
    c = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
    return {"c": say_coin(rng, c), "_coin": c}


def _slots_tokbuy(rng, w):
    return {"t": rng.choice(MEME_LIST), "a": rng.choice([5, 10, 20, 25, 50])}


def _slots_tok(rng, w):
    return {"t": rng.choice(MEME_LIST)}


def _slots_dep(rng, w):
    return {"a": rng.choice([20, 50, 100, 250]), "x": rng.choice(["USDC", "ETH"])}


def _slots_pay(rng, w):
    n = rng.choice(NAMES)
    return {"n": n, "a": rng.choice([25, 50, 100, 200]), "x": H.evm_addr(rng), "y": H.evm_addr(rng), "z": H.evm_addr(rng)}


NEEDS = {
    "stop": dict(tool="set_tp_sl", page="trade", tab="positions", slots=_slots_stop,
                 unavail=["I can't set stops in this chat.", "Setting a stop isn't available in this chat.", "Stops can't be set from this chat.", "This chat can't set a stop loss."],
                 users=["put a stop at {p} on my {c}", "set a stop loss on {c} at {p}", "add a stop on my {c} {side} at {p}", "protect my {c} position with a stop at {p}",
                        "stop loss {p} for {c} pls", "can you put a stop under {c} at {p}?", "i want a stop at {p} on {c}", "place a stop loss at {p} for my {c} trade"]),
    "tp": dict(tool="set_tp_sl", page="trade", tab="positions", slots=_slots_tp,
               unavail=["I can't set a take profit in this chat.", "Take profits can't be set from this chat.", "Setting a take profit isn't available in this chat."],
               users=["set a take profit at {p} on {c}", "add a tp at {p} for my {c}", "put a take profit on my {c} long at {p}", "i want to take profit on {c} at {p}", "take profit {p} on {c} pls"]),
    "tpsl": dict(tool="set_tp_sl", page="trade", tab="positions", slots=_slots_tpsl,
                 unavail=["I can't set stops or take profits in this chat.", "Stops and take profits can't be set from this chat."],
                 users=["stop {s} take profit {p} on {c}", "set stop at {s} and tp at {p} for my {c}", "{c}: stop loss {s}, take profit {p}", "protect my {c} long, stop {s}, target {p}"]),
    "newtrade": dict(tool="prepare_trade", page="trade", tab=None, slots=_slots_trade,
                     unavail=["Trading perps isn't available in this chat.", "I can't open trades in this chat.", "Perp trades can't be placed from this chat."],
                     users=["long {c} ${s} {l}x", "short {c} with ${s} at {l}x", "open a {l}x long on {c}, ${s}", "go long {c}, {l}x, ${s}", "i want to short {c} ${s} {l}x market"]),
    "close": dict(tool="prepare_close", page="trade", tab="positions", slots=_slots_coin,
                  unavail=["I can't close positions in this chat.", "Closing a position isn't available in this chat.", "Positions can't be closed from this chat."],
                  users=["close my {c} long", "close my {c} position", "get me out of {c}", "exit my {c} trade", "close half my {c}"]),
    "cancel": dict(tool="cancel_order", page="trade", tab="positions", slots=_slots_coin,
                   unavail=["I can't cancel orders in this chat.", "Cancelling an order isn't available in this chat.", "Orders can't be cancelled from this chat."],
                   users=["cancel my {c} order", "cancel my {c} limit order", "remove my {c} stop loss", "cancel all my open orders", "delete my {c} take profit"]),
    "sell": dict(tool="prepare_sell", page="trade", tab=None, slots=_slots_tok,
                 unavail=["Selling Solana tokens isn't available in this chat.", "I can't sell tokens in this chat.", "Token sales can't be prepared from this chat."],
                 users=["sell half my {t}", "sell all my {t}", "dump my {t}", "sell 25% of my {t}", "get out of {t}"]),
    "buy": dict(tool="prepare_buy", page="trade", tab=None, slots=_slots_tokbuy,
                unavail=["Buying Solana tokens isn't available in this chat.", "I can't buy tokens in this chat.", "Token buys can't be prepared from this chat."],
                users=["buy ${a} of {t}", "ape ${a} into {t}", "get me ${a} worth of {t}", "buy some {t}, ${a}", "pick up ${a} of {t}"]),
    "search": dict(tool="search_tokens", page="trade", tab=None, slots=_slots_tok,
                   unavail=["Token checks aren't available in this chat.", "I can't look up tokens in this chat.", "Token search isn't available in this chat."],
                   users=["is {t} safe?", "check the {t} token for me", "what's the liquidity on {t}?", "any red flags on {t}?", "look up {t}"]),
    "deposit": dict(tool="prepare_deposit", page="earn", tab="save", slots=_slots_dep,
                    unavail=["Deposits aren't available in this chat.", "I can't prepare Aave deposits in this chat.", "Earn deposits can't be prepared from this chat."],
                    users=["deposit {a} usdc in aave", "put {a} usdc into aave", "lend {a} usdc on aave", "earn on {a} usdc with aave", "start earning with {a} usdc"]),
    "withdraw": dict(tool="prepare_withdraw", page="earn", tab="positions", slots=_slots_dep,
                     unavail=["Withdrawals aren't available in this chat.", "I can't prepare Aave withdrawals in this chat.", "Earn withdrawals can't be prepared from this chat."],
                     users=["withdraw my usdc from aave", "take my money out of aave", "pull {a} usdc out of aave", "withdraw all from aave", "get my aave deposit back"]),
    "yields": dict(tool="get_yields", page="earn", tab="explore", slots=_slots_dep,
                   unavail=["Yield lookups aren't available in this chat.", "I can't look up pool yields in this chat.", "Pool yields can't be listed from this chat."],
                   users=["where can I earn on my usdc?", "best yields for usdc?", "what pools pay the most on {x}?", "show me the top apys", "which defi pools pay best right now?"]),
    "multisend": dict(tool="prepare_multisend", page="payments", tab="transfer", slots=_slots_pay,
                      unavail=["Batch sends aren't available in this chat.", "I can't prepare batch sends in this chat.", "Multi-recipient payments can't be prepared from this chat."],
                      users=["send {a} usdc each to {x} and {y} and {z}", "batch send {a} usdc to {x}, {y}, {z}", "split payments: {a} usdc to each of {x} {y}", "pay these three {a} usdc each: {x} {y} {z}"]),
    "contact": dict(tool="add_contact", page="payments", tab="payees", slots=_slots_pay,
                    unavail=["Saving contacts isn't available in this chat.", "I can't save contacts in this chat.", "Contacts can't be added from this chat."],
                    users=["save {n} as {x}", "add {n} to my contacts: {x}", "new contact {n}, address {x}", "remember {x} as {n}", "add {x} as {n} in my payees"]),
    "schedule": dict(tool="schedule_payment", page="payments", tab="schedule", slots=_slots_pay,
                     unavail=["Payment reminders aren't available in this chat.", "I can't schedule payments in this chat.", "Scheduled payments can't be set up from this chat."],
                     users=["remind me to pay {n} {a} usdc on friday", "schedule {a} usdc to {n} every month", "set a payment reminder for {n}, {a} usdc, dec 15", "pay {n} {a} usdc on the 1st of every month", "i need a weekly reminder to pay {n} {a} usdc"]),
    "automation": dict(tool="create_automation", page="automations", tab=None, slots=_slots_coin,
                       unavail=["Automations aren't available in this chat.", "I can't set up automations in this chat.", "Scheduled automations can't be created from this chat."],
                       users=["every morning at 8 tell me the {c} price", "set up an automation to check {c} every hour", "send me my portfolio value every day", "every friday at 18:00 summarize my activity", "run a {c} price check every 15 minutes"]),
    "markets": dict(tool="search_markets", page="trade", tab="markets", slots=_slots_coin,
                    unavail=["Browsing markets isn't available in this chat.", "I can't search markets in this chat.", "Market search isn't available in this chat."],
                    users=["which perp has the highest funding right now?", "show me the top markets by volume", "what's the funding on {c}?", "find me perps with negative funding", "search the markets for {c}"]),
    "pos_spot": dict(tool="get_positions", page="wallets", tab=None, slots=_slots_coin,
                     unavail=["Spot positions aren't available in this chat.", "I can't list your token positions in this chat.", "Token positions can't be read from this chat."],
                     users=["what tokens do I hold on solana?", "show my spot tokens and my pnl", "what's my average cost on my solana tokens?", "list my spot positions", "how are my memecoins doing?"]),
    "pos_earn": dict(tool="get_positions", page="earn", tab="positions", slots=_slots_coin,
                     unavail=["Earn positions aren't available in this chat.", "I can't read your deposits in this chat.", "Deposits can't be listed from this chat."],
                     users=["what do I have in aave?", "show my earn positions", "how much am I earning in aave?", "list my deposits", "what are my aave deposits worth?"]),
}
# (need, pack) for the 60 tool-absent rows; every pack lacks the tool of its need
ABSENT_PLAN = ([("stop", p) for p in ["trade"] * 4 + ["core"] * 2 + ["solana", "earn"]] + [("tp", p) for p in ["trade"] * 3 + ["pay"]] +
               [("tpsl", "trade")] * 2 + [("newtrade", p) for p in ["core", "core", "core", "solana", "earn", "pay"]] +
               [("close", p) for p in ["core", "solana", "earn"]] + [("cancel", p) for p in ["core", "pay", "earn"]] +
               [("sell", p) for p in ["core", "earn", "pay"]] + [("buy", p) for p in ["core", "trade", "earn"]] + [("search", p) for p in ["core", "pay"]] +
               [("deposit", p) for p in ["core", "trade", "solana"]] + [("withdraw", p) for p in ["core", "pay", "solana"]] + [("yields", p) for p in ["core", "trade"]] +
               [("multisend", p) for p in ["core", "trade"]] + [("contact", p) for p in ["core", "trade", "solana"]] + [("schedule", p) for p in ["core", "trade", "solana"]] +
               [("automation", p) for p in ["core", "solana"]] + [("markets", p) for p in ["core", "solana", "earn", "pay"]] +
               [("pos_spot", p) for p in ["core", "pay"]] + [("pos_earn", p) for p in ["core", "trade"]])
assert len(ABSENT_PLAN) == 60


def sc_absent(rng, split, used, bio, need_id, pack):
    nd = NEEDS[need_id]
    assert nd["tool"] not in pack_names(pack), (need_id, pack)
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", rng.choice([120.5, 300, 75]))
    sl = nd["slots"](rng, w)
    coin = sl.pop("_coin", None)
    if coin and need_id in ("stop", "tp", "tpsl", "close", "cancel"):
        w = trade_world(rng, bio, (coin,), with_orders=need_id in ("cancel",))
    c = Conv(rng, w, pack, bio, used)
    c.user(styl(rng, pick(rng, split, nd["users"])).format(**sl))
    args = {"page": nd["page"]}
    if nd["tab"]:
        args["tab"] = nd["tab"]
    r = c.call(("open_page", args))
    assert r["status"] == "button_shown"
    c.say(f"{rng.choice(nd['unavail'])} {button_line(rng, nd['page'], nd['tab'])}")
    rec = c.record("absent_" + need_id, "tool_absent", 1, absent=True, absent_tools=[nd["tool"]], absent_pack=pack)
    return rec


# ================================================================================================ PART 2: long context
EXTRA_MEMES = {"BOME": 0.0068, "MEW": 0.0071, "MYRO": 0.12, "SLERF": 0.21, "GIGA": 0.031, "WEN": 0.0002 * 5, "SAMO": 0.0094, "MICHI": 0.045,
               "DOGINME": 0.0123, "TOSHI": 0.0004 * 10}
ACT_TEMPL = [("Swap {a} USDC on {A} to USDC on {B}", "swap"), ("Sent {a} USDC to {ad}", "send"), ("Bought {b} {sym} with {a} USDC", "buy"),
             ("Sent {a2} ETH to {ad}", "send"), ("Swap {a} USDC on {A} to ETH on {B}", "swap"), ("Deposited {a} USDC into Aave on {A}", "deposit")]


def short_hex(rng):
    h = "".join(rng.choice("0123456789abcdef") for _ in range(4))
    t = "".join(rng.choice("0123456789abcdef") for _ in range(2))
    return f"0x{h[:2]}…{t}"


def build_activity(rng, n=10):
    rows = []
    t_min = 14 * 60 + 41
    day = 10
    chains = ["Base", "Arbitrum", "Ethereum", "Solana"]
    st = ["confirmed"] * (n - 3) + ["failed", "in_progress", "in_progress"]
    rng.shuffle(st)
    for i in range(n):
        tpl, _ = rng.choice(ACT_TEMPL)
        A, B = rng.sample(chains, 2)
        what = tpl.format(a=rng.choice([0.5, 1, 2, 5, 10, 20, 25, 40, 75, 120]), a2=rng.choice([0.01, 0.02, 0.05, 0.1]), b=rng.choice([4.82, 12.5, 98.4, 310]),
                          sym=rng.choice(["WIF", "JUP", "POPCAT"]), A=A, B=B, ad=short_hex(rng))
        rows.append({"status": st[i], "time": f"2026-11-{day:02d} {t_min // 60:02d}:{t_min % 60:02d}", "what": what})
        t_min -= rng.choice([22, 41, 97, 180, 305, 410])
        while t_min < 0:
            t_min += 24 * 60
            day -= 1
    return rows


class LongBuilder:
    """One long-context conversation: 8 user turns, n_long of them are big reads (450+ tokens each), the 8th asks about one."""

    def __init__(self, rng, split, used, bio, pack, kinds, n_long):
        self.rng, self.split, self.bio, self.pack, self.kinds, self.n_long = rng, split, bio, pack, kinds, n_long
        self.w = mkw(rng, bio)
        self.c = None
        self.long = {}            # kind -> info

    # ---------------------------------------------------------------- data per kind
    def prep_world(self):
        rng, w = self.rng, self.w
        for ch in ("base", "arbitrum", "ethereum"):
            w.add_hold("USDC", ch, rng.choice([120.5, 300, 540, 85.25, 1000]))
        w.add_hold("ETH", "base", rnd(rng, 0.05, 1.5, 4))
        w.add_hold("SOL", "solana", rnd(rng, 0.2, 3, 3))
        w.add_hold("USDC", "solana", rng.choice([45, 120]))
        w.d["hl"]["value"] = rng.choice([250.5, 400])
        if "yields" in self.kinds:
            make_yields(rng, w, 6)
        if "activity" in self.kinds:
            w.d["activity"] = build_activity(rng, 10)
        if "spot" in self.kinds:
            for s_, p in EXTRA_MEMES.items():
                w.d["prices"][s_] = p
            allsyms = MEME_LIST + list(EXTRA_MEMES)
            for s_ in rng.sample(allsyms, 7):
                px = w.d["prices"][s_]
                amt = round(rng.choice([8, 12, 25, 40, 75, 120, 300, 900]) * rng.uniform(0.6, 1.4) / px, 2)
                amt = round(amt) if amt >= 100 else amt
                cost = None if rng.random() < 0.2 else round(px * rng.uniform(0.6, 1.5), 4 if px < 1 else 2)
                w.add_spot(s_, mint_for(rng), amt, cost, frozen=(rng.random() < 0.08), price=px)
            top = max(w.d["spot"], key=lambda x: x["usd"])
            top["frozen"] = False
        if "balances" in self.kinds:
            pairs = [("USDC", c_) for c_ in ("base", "arbitrum", "ethereum", "polygon", "solana")] + [("USDT", c_) for c_ in ("base", "arbitrum", "ethereum", "polygon", "bnb")] + \
                    [("ETH", c_) for c_ in ("base", "arbitrum", "ethereum")] + [("SOL", "solana"), ("BNB", "bnb"), ("LINK", "ethereum"), ("ARB", "arbitrum")]
            keep = [h for h in w.d["holdings"] if h["symbol"] == "USDC" and h["chain"] in ("base", "arbitrum", "ethereum")]
            w.d["holdings"] = [{**h, "amount": plain(max(float(h["amount"]), 300.0))} for h in keep]
            for h in w.d["holdings"]:
                h["usd"] = round(float(h["amount"]), 2)
            pairs = [p for p in pairs if not (p[0] == "USDC" and p[1] in ("base", "arbitrum", "ethereum"))]
            for s_, c_ in rng.sample(pairs, 13):
                px = w.d["prices"].get(s_, 1.0)
                usd_t = rng.choice([8, 15, 40, 90, 150, 320, 700])
                w.add_hold(s_, c_, plain(round(usd_t * rng.uniform(0.5, 1.5) / px, 4 if px > 50 else 2)))
        if "tokens" in self.kinds:
            names = rng.sample(["Popcat", "Moon Cat", "Cat in a dogs world", "Cat Mew", "Nyan Cat", "Cat Coin", "Cheshire Cat", "Space Cat"], 5)
            for nm in names:
                sym = "".join(ch for ch in nm.upper() if ch.isalpha())[:8]
                mk_token(rng, w, sym, verified=rng.random() < 0.6, liq=int(rng.choice([0.4, 1.2, 3.4, 6.2, 9.1, 14, 27]) * 1e6),
                         tags=rng.choice([[], [], ["New"], ["Top holders 38%"], ["Mint on"]]), price=rng.choice([0.0031, 0.052, 0.21, 0.88, 1.4]))
                w.d["tokens"][-1]["name"] = nm
                w.d["tokens"][-1]["symbol"] = sym
        w.d["near_account"] = None

    # ---------------------------------------------------------------- long reads: (user text, call, brief answer)
    def do_long(self, kind):
        rng, c, w, split = self.rng, self.c, self.w, self.split
        if kind == "yields":
            c.user(V(rng, split, ["show me the usdc yields", "list every usdc pool you can see", "what are all the usdc pools right now?", "give me the usdc yield list", "all usdc yields pls", "what usdc pools are there?", "i want to see all the usdc yields", "can you list the usdc yields?", "usdc pools, all of them", "pull up the usdc yield table"]))
            r = c.call(("get_yields", {"asset": "USDC"}))
            p0, p1 = r["pools"][0], r["pools"][1]
            c.say(f"There are {len(r['pools'])} USDC pools. The top is {p0['protocol']} on {p0['chain']} at {plain(p0['apy_pct'])}%, then {p1['protocol']} on {p1['chain']} at {plain(p1['apy_pct'])}%. APYs change.")
        elif kind == "activity":
            c.user(V(rng, split, ["what's my recent activity?", "show me my last transactions", "list my latest activity", "what happened in my wallet lately?", "pull up my activity", "what have i done in the wallet recently?", "give me my recent transactions", "any recent movement in my wallet?", "show the latest wallet activity", "what are my last ten transactions?"]))
            r = c.call(("get_activity", {"limit": 10}))
            a0 = r["activity"][0]
            c.say(f"You have {len(r['activity'])} recent items and {r['in_progress']} still in progress. The newest is \"{a0['what']}\", {a0['status'].replace('_', ' ')}.")
        elif kind == "spot":
            c.user(V(rng, split, ["show my solana tokens", "what spot tokens do I hold?", "list my token positions", "what memecoins am I holding?", "show every token i own on solana", "which tokens do i have on solana?", "give me my spot token list", "what is in my token bag?", "list the tokens in my wallet", "my solana holdings please"]))
            r = c.call(("get_positions", {"kind": "spot"}))
            rows = sorted(r["spot"], key=lambda x: -x["usd"])
            c.say(f"You hold {len(rows)} tokens. The biggest are {rows[0]['symbol']} ({usd(rows[0]['usd'])}), {rows[1]['symbol']} ({usd(rows[1]['usd'])}) and {rows[2]['symbol']} ({usd(rows[2]['usd'])}), and {len(rows) - 3} smaller.")
        elif kind == "balances":
            c.user(V(rng, split, ["what do I hold across all chains?", "show me every balance in my wallet", "list all my balances", "what's in my wallet on every network?", "give me my full balance list", "what do i own on every chain?", "balances everywhere please", "list everything in my wallet", "show balances for all networks", "full wallet breakdown pls"]))
            r = c.call(("get_balances", {}))
            rows = sorted(r["holdings"], key=lambda x: -x["usd"])
            c.say(f"The biggest are {rows[0]['amount']} {rows[0]['symbol']} on {CHAIN_NAME[rows[0]['chain']]} ({usd(rows[0]['usd'])}), {rows[1]['amount']} {rows[1]['symbol']} on {CHAIN_NAME[rows[1]['chain']]} ({usd(rows[1]['usd'])}) and {len(rows) - 2} more.")
        else:
            c.user(V(rng, split, ["find cat tokens on solana", "search solana for cat tokens", "look up cat coins for me", "what cat tokens are there on solana?", "show me tokens with cat in the name", "any cat coins on solana?", "list the cat tokens you can find", "i want to see cat tokens", "search for tokens called cat something", "cat memecoins on solana, find them"]))
            r = c.call(("search_tokens", {"query": "cat"}))
            t0, t1 = r["tokens"][0], r["tokens"][1]
            c.say(f"I found {len(r['tokens'])} tokens. {t0['name']} has {big(t0['liquidity_usd'])} of liquidity and {t1['name']} has {big(t1['liquidity_usd'])}. Tell me which one you want to look at.")
        self.long[kind] = r

    # ---------------------------------------------------------------- filler turns
    def do_filler(self):
        rng, c, split = self.rng, self.c, self.split
        k = rng.choice(["price", "calc", "know", "portfolio", "thanks", "price", "calc", "price", "thanks"])
        if k == "price":
            coin = rng.choice(COINS)
            c.user(V(rng, split, ["what's {c} at?", "price of {c}?", "how much is {c} right now?", "{c} price pls", "where is {c} trading?", "is {c} up today?"], c=say_coin(rng, coin)))
            r = c.call(("get_price", {"symbol": coin}))
            ch = r["change_24h_pct"]
            c.say(f"{coin} is {usd(r['price'])}, {'up' if ch > 0 else 'down'} {plain(abs(ch))}% in the last 24h.")
        elif k == "calc":
            p, b = rng.choice([5, 8, 12, 15, 18, 20, 35]), rng.choice([240, 350, 480, 1200, 75, 960])
            c.user(V(rng, split, ["what's {p}% of {b}?", "quick maths: {p}% of {b}", "how much is {p} percent of {b}?", "calc {p}% of {b} for me"], p=p, b=b))
            r = c.call(("calculator", {"expression": f"{b} * {p} / 100"}))
            c.say(f"{p}% of {b} is {r['result']}.")
        elif k == "know":
            key, qs, ans = rng.choice(KNOW)
            q = V(rng, split, qs)
            if norm(q) in self.used or norm(q) in c.texts:
                q = q.rstrip("?") + " again?"
            c.user(q)
            c.say(ans)
        elif k == "portfolio":
            c.user(V(rng, split, ["how much do I have in total?", "what's my net worth?", "total value of everything i own?", "how much am I worth across everything?", "what is my total balance everywhere?", "give me the total of everything i hold", "how much is all my stuff worth?", "sum up my whole portfolio", "overall portfolio value?"]))
            r = c.call(("get_portfolio", {}))
            t0 = r["top"][0]
            c.say(f"About {usd(r['total_usd'])} in total in your {r['wallet']} wallet. The biggest piece is {t0['what']} ({usd(t0['usd'])}).")
        else:
            c.user(rng.choice(["ok thanks", "great thanks", "thanks, got it", "nice, thank you", "cool thanks", "perfect, thanks"]))
            c.say(rng.choice(["Anytime. Tell me what you want to do next.", "Glad it helps. What's next?", "You're welcome. Ask me anything else."]))

    # ---------------------------------------------------------------- final turn
    def final(self, target):
        rng, c, w, split = self.rng, self.c, self.w, self.split
        r = self.long[target]
        chk = {"final": "text", "no_tools": True, "target_kind": target}
        if target == "yields":
            pools = r["pools"]
            mode = rng.choice(["apy", "tvl", "best_chain", "deposit", "apy", "deposit"])
            if mode == "apy":
                p = rng.choice([x for x in pools if not x["can_deposit_in_hermit"]] or pools)
                c.user(V(rng, split, ["what was the apy on {p} {c} again?", "remind me, how much did {p} on {c} pay?", "sorry, what's the apy for {p} on {c}?", "the {p} pool on {c}: what apy was that?", "how much does {p} on {c} pay?"], p=p["protocol"], c=p["chain"]))
                c.say(f"{p['protocol']} on {p['chain']} pays {plain(p['apy_pct'])}%. It can change.")
                chk.update(must=[[plain(p["apy_pct"])], [p["protocol"]]], mode="apy")
            elif mode == "tvl":
                p = max(pools, key=lambda x: x["tvl_usd"])
                c.user(V(rng, split, ["which of those pools is the biggest, and how big?", "which pool has the most tvl and how much is it?", "biggest pool in that list? how much tvl?", "what's the largest pool there by tvl?", "which one holds the most money?"]))
                c.say(f"{p['protocol']} on {p['chain']} is the biggest, with {big(p['tvl_usd'])} of TVL.")
                chk.update(must=[[big(p["tvl_usd"]), usd(p["tvl_usd"]), plain(p["tvl_usd"])], [p["protocol"]]], mode="tvl")
            elif mode == "best_chain":
                ch = rng.choice(sorted({x["chain"] for x in pools if sum(1 for y in pools if y["chain"] == x["chain"]) >= 2}) or ["Base"])
                p = max([x for x in pools if x["chain"] == ch], key=lambda x: x["apy_pct"])
                c.user(V(rng, split, ["which pool pays the most on {c}, and what's the apy?", "best apy on {c} from that list?", "on {c} only, which pool is highest and how much?", "top pool on {c}? what apy?", "of those, what pays best on {c}?"], c=ch))
                c.say(f"On {ch}, {p['protocol']} pays the most at {plain(p['apy_pct'])}%.")
                chk.update(must=[[plain(p["apy_pct"])], [p["protocol"]]], mode="best_chain")
            else:
                ch = rng.choice(["base", "arbitrum", "ethereum"])
                a = rng.choice([20, 25, 50, 75])
                c.user(V(rng, split, ["put {a} usdc in the aave one on {c}", "deposit {a} usdc into aave on {c}", "ok, {a} usdc into the aave pool on {c}", "do the aave deposit on {c} with {a} usdc", "aave on {c}: deposit {a} usdc"], a=a, c=ch))
                rr = c.call(("prepare_deposit", {"project": "aave-v3", "symbol": "USDC", "chain": ch, "amount": a}))
                c.say(f"I've prepared a deposit of {rr['amount']} into {rr['pool']} at {plain(rr['apy_pct'])}% APY, which can change. Approve with {self.bio}.")
                chk.update(final="call", no_tools=False, mode="deposit")
        elif target == "activity":
            rows = r["activity"]
            mode = rng.choice(["count", "failed", "time"])
            if mode == "count":
                c.user(V(rng, split, ["how many of those are still in progress?", "which of these are still pending, how many?", "how many are not confirmed yet?", "count the ones still in progress"]))
                c.say(f"{r['in_progress']} of the {len(rows)} are still in progress.")
                chk.update(must=[[str(r["in_progress"])]], mode="count")
            elif mode == "failed":
                f = next(x for x in rows if x["status"] == "failed")
                c.user(V(rng, split, ["which one failed?", "was anything in that list a failure?", "which of those didn't work?", "did any of them fail? which?", "tell me the failed one"]))
                c.say(f"One failed: \"{f['what']}\" at {f['time']}.")
                chk.update(must=[[f["time"]], [f["what"]]], mode="failed")
            else:
                s_ = next((x for x in rows if x["what"].startswith("Sent")), rows[0])
                c.user(V(rng, split, ["when was the one that says \"{w}\"?", "what time was \"{w}\"?", "give me the time of \"{w}\"", "when did \"{w}\" happen?", "timestamp for \"{w}\" please"], w=s_["what"]))
                c.say(f"\"{s_['what']}\" was at {s_['time']}, status {s_['status'].replace('_', ' ')}.")
                chk.update(must=[[s_["time"]]], mode="time")
        elif target == "spot":
            rows = r["spot"]
            mode = rng.choice(["amount", "cost", "sell_big", "sell_all"] if "prepare_sell" in c.pack else ["amount", "cost"])
            if mode == "amount":
                s_ = rng.choice(rows)
                c.user(V(rng, split, ["how much {s} do I hold?", "what's my {s} balance from that list?", "remind me how many {s} I have", "{s} amount?", "how many {s} again?"], s=s_["symbol"]))
                c.say(f"You hold {commas(s_['amount'])} {s_['symbol']}, worth {usd(s_['usd'])}.")
                chk.update(must=[[s_["amount"], commas(s_["amount"])]], mode="amount")
            elif mode == "cost":
                s_ = rng.choice([x for x in rows if x["avg_cost"] is not None])
                c.user(V(rng, split, ["what did {s} cost me on average?", "my average cost on {s}?", "what's my avg cost for {s}?", "average entry on {s} again?", "avg cost of my {s}?"], s=s_["symbol"]))
                c.say(f"Your average cost on {s_['symbol']} is {usd(s_['avg_cost']) if s_['avg_cost'] >= 1 else plain(s_['avg_cost'])}.")
                chk.update(must=[[plain(s_["avg_cost"])]], mode="cost")
            elif mode == "sell_big":
                ok = [x for x in rows if not x["frozen"]]
                big_ = max(ok, key=lambda x: x["usd"])
                c.user(V(rng, split, ["sell half of the biggest one", "sell 50% of my largest token from that list", "take half off my biggest bag", "trim the biggest holding by half", "sell half of whichever is worth most"]))
                if any(x["usd"] > big_["usd"] and x["frozen"] for x in rows):
                    pass
                rr = c.call(("prepare_sell", {"token": big_["symbol"], "percent": 50}))
                c.say(ans_sell(rng, rr, self.bio))
                chk.update(final="call", no_tools=False, mode="sell_big")
            else:
                cnt = Counter(x["symbol"] for x in rows)
                s_ = rng.choice([x for x in rows if not x["frozen"] and cnt[x["symbol"]] == 1])
                c.user(V(rng, split, ["sell all my {s}", "dump all my {s}", "get out of {s} completely", "exit {s} fully", "sell everything in {s}"], s=s_["symbol"]))
                rr = c.call(("prepare_sell", {"token": s_["symbol"], "percent": 100}))
                c.say(ans_sell(rng, rr, self.bio))
                chk.update(final="call", no_tools=False, mode="sell_all")
        elif target == "balances":
            rows = r["holdings"]
            mode = rng.choice(["one", "most"])
            if mode == "one":
                h = rng.choice(rows)
                c.user(V(rng, split, ["how much {s} do I have on {c}?", "my {s} on {c}, how much?", "what's the {s} balance on {c} from that list?", "{s} on {c} amount?", "remind me, {s} on {c}?"], s=h["symbol"], c=CHAIN_NAME[h["chain"]]))
                c.say(f"You have {h['amount']} {h['symbol']} on {CHAIN_NAME[h['chain']]}, worth {usd(h['usd'])}.")
                chk.update(must=[[h["amount"]]], mode="one")
            else:
                sym = "USDC"
                cand = [x for x in rows if x["symbol"] == sym]
                h = max(cand, key=lambda x: float(x["amount"]))
                c.user(V(rng, split, ["on which chain do I have the most usdc, and how much?", "where is my biggest usdc balance?", "which chain holds most of my usdc? how much?", "biggest usdc balance, where and how much?", "my largest usdc pile is on which chain?"]))
                c.say(f"Your biggest USDC balance is on {CHAIN_NAME[h['chain']]}: {h['amount']} USDC.")
                chk.update(must=[[h["amount"]], [CHAIN_NAME[h["chain"]]]], mode="most")
        else:
            rows = r["tokens"]
            mode = rng.choice(["mint", "liq", "verified"])
            t = rng.choice(rows)
            if mode == "mint":
                c.user(V(rng, split, ["what's the mint of {n} from that list?", "give me the mint address for {n}", "mint of the {n} one please", "i need the contract address of {n}", "copy the mint for {n}"], n=t["name"]))
                c.say(f"The mint of {t['name']} is {t['mint']}.")
                chk.update(must=[[t["mint"]]], mode="mint", copy_targets=[t["mint"]])
            elif mode == "liq":
                top = max(rows, key=lambda x: x["liquidity_usd"])
                c.user(V(rng, split, ["which of those has the most liquidity, and how much?", "most liquid token in that list? how much?", "biggest liquidity there?", "which one is the deepest and by how much?", "who has the highest liquidity of them?"]))
                c.say(f"{top['name']} has the most liquidity, {big(top['liquidity_usd'])}.")
                chk.update(must=[[big(top["liquidity_usd"]), usd(top["liquidity_usd"])], [top["name"]]], mode="liq")
            else:
                c.user(V(rng, split, ["is {n} verified?", "is the {n} one verified or not?", "{n}: verified?", "does {n} have the verified flag?", "check if {n} is verified"], n=t["name"]))
                c.say(f"{t['name']} is {'verified' if t['verified'] else 'not verified'}" + (f", with tags: {', '.join(tag_words(t['tags']))}." if t["tags"] and not t["verified"] else "."))
                chk.update(must=[["not verified" if not t["verified"] else "is verified"], [t["name"]]], mode="verified")
        return chk

    def build(self, with_final=True):
        rng = self.rng
        self.used = self.c_used
        self.prep_world()
        self.c = Conv(rng, self.w, self.pack, self.bio, self.c_used)
        n_filler = 7 - self.n_long
        order = ["L"] * self.n_long + ["F"] * n_filler
        rng.shuffle(order)
        kinds = rng.sample(self.kinds, self.n_long)
        ki = 0
        long_turns = []
        for tag in order:
            if tag == "L":
                self.do_long(kinds[ki])
                long_turns.append(kinds[ki])
                ki += 1
            else:
                self.do_filler()
        target = long_turns[0] if rng.random() < 0.5 else rng.choice(long_turns)
        chk = self.final(target)
        return chk, long_turns


KINDS_BY_PACK = {"earn": ["yields", "activity", "balances", "spot"], "solana": ["tokens", "spot", "activity", "balances"], "core": ["activity", "balances"]}
BUCKETS = [(0, 2000, "<2k"), (2000, 4000, "2-4k"), (4000, 6200, "4-6k")]


def bucket_of(n):
    for lo, hi, nm in BUCKETS:
        if lo <= n < hi:
            return nm
    return ">6k"


def make_long(rng, split, used, bio, want_bucket):
    reasons = Counter()
    for attempt in range(40):
        pack = rng.choice(["earn", "solana", "earn", "solana", "core"]) if want_bucket == "2-4k" else rng.choice(["earn", "solana"])
        kinds = KINDS_BY_PACK[pack]
        n_long = rng.choice([1, 1, 2]) if want_bucket == "2-4k" else rng.choice([3, 3, 4])
        n_long = min(n_long, len(kinds))
        lb = LongBuilder(rng, split, used, bio, pack, kinds, n_long)
        lb.c_used = used
        try:
            chk, long_turns = lb.build()
        except Dup:
            reasons["dup:" + str(__import__("sys").exc_info()[1])[:50]] += 1
            continue
        rec = lb.c.record("long_" + chk["mode"], "long_context", 2, long_kinds=long_turns, n_long=n_long, check=chk)
        toks = tok_len(render_text(rec["messages"], rec["meta"]["pack"], bio, "lite", False))
        if bucket_of(toks) == want_bucket and toks <= CAP_TOKENS - RESERVE:
            return rec
        reasons[f"{bucket_of(toks)}:{n_long}"] += 1
    raise RuntimeError(f"could not build a {want_bucket} long row: {dict(reasons)}")


# ================================================================================================ PART 3: copy fidelity
def six(rng, int_lo, int_hi):
    """A 6 decimal amount whose last digit is not zero, as a string."""
    ip = rng.randint(int_lo, int_hi)
    fp = rng.randint(100, 999999)
    if fp % 10 == 0:
        fp += rng.randint(1, 9)
    s = f"{ip}.{fp:06d}"
    assert plain(float(s)) == s
    return s


def swap_digits(s, rng):
    """A near-identical distractor: two neighbouring digits of the fraction swapped."""
    ip, fp = s.split(".")
    for _ in range(20):
        i = rng.randrange(0, len(fp) - 1)
        if fp[i] != fp[i + 1]:
            f2 = fp[:i] + fp[i + 1] + fp[i] + fp[i + 2:]
            if not f2.endswith("0"):
                return f"{ip}.{f2}"
    return f"{ip}.{fp[:-1]}{(int(fp[-1]) % 9) + 1}"


def lookalike(rng, addr):
    raw = addr[2:8] + "".join(rng.choice("0123456789abcdef") for _ in range(28)) + addr[-6:]
    return H.eip55(raw.lower())


def payee_file(rng, target_name, target_addr, fname="payees.csv"):
    rows = [(target_name, target_addr, "EVM")]
    for _ in range(2):
        rows.append((rng.choice([n for n in NAMES if n != target_name]) + " (old)", lookalike(rng, target_addr), "EVM"))
    for _ in range(rng.randint(1, 2)):
        rows.append((rng.choice(NAMES), H.evm_addr(rng), "EVM"))
    rng.shuffle(rows)
    body = "Name,Address,Network\n" + "\n".join(",".join(r) for r in rows)
    return f"The user attached the file {fname} (table):\n<file>\n{body}\n</file>"


def invoice_file(rng, amt, sym, addr, dist):
    inv = rng.randint(1000, 4999)
    lines = [f"Invoice {inv}", "Item: design work", f"Earlier balance: {dist} {sym}", f"Total due: {amt} {sym}", f"Pay to: {addr}"]
    return f"The user attached the file invoice-{inv}.pdf (PDF, 1 page):\n<file>\n" + "\n".join(lines) + "\n</file>"


def plan_file(rng, text_lines, fname="notes.txt"):
    return f"The user attached the file {fname} (text):\n<file>\n" + "\n".join(text_lines) + "\n</file>"


def copy_final_text(rng, bio, r, addr=None):
    if r.get("kind") == "send":
        return f"I've prepared a send of {r['amount']} on {r['network']} to {addr}. {addr_tail(rng, bio)}"
    return None


COPY_ADDR_SEND_T1 = ["what's {n}'s address?", "can you pull up {n}'s saved address?", "show me the address saved for {n}", "what address do I have for {n}?", "look up {n} in my contacts", "which address is saved under {n}?", "what address is {n} saved with?", "i need {n}'s address, can you fetch it?"]
COPY_ADDR_SEND_T2 = ["send {a} usdc to {n}: {x}", "pay {n} {a} usdc, address {x}, same as in the file", "{n} again, {a} usdc to {x}", "send {a} usdc to {x}, that's {n}'s",
                     "please pay {a} usdc to {x} ({n}, matches the list)", "i'm sending {a} usdc to {n} at {x}, the file has it too", "{a} usdc to {x} for {n}", "can you pay {n} {a} usdc? their address is {x}", "{x} is {n}'s address, send {a} usdc there", "send {a} usdc to {n} at {x}, it matches the file"]


def sc_copy_addr_send(rng, split, used, bio):
    n = slot(rng, split, NAMES)
    addr = H.evm_addr(rng)
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", rng.choice([300, 540, 1000]))
    w.d["contacts"] = [{"name": n, "address": addr, "family": "evm"}]
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, COPY_ADDR_SEND_T1, n=n))
    r1 = c.call(("get_address", {"contact": n}))
    c.say(f"The address for {n} is {addr}. It works on {join_and([NET_NAME[x] for x in r1['contacts'][0]['networks']])}.")
    a = rng.choice([15, 25, 40, 60, 120])
    c.user(U(rng, split, COPY_ADDR_SEND_T2, a=a, n=n, x=addr), file_text=payee_file(rng, n, addr))
    r2 = c.call(("prepare_send", {"to_address": addr, "amount": a, "symbol": "USDC"}))
    c.say(f"I've prepared a send of {r2['amount']} on {r2['network']} to {n} ({addr}). {addr_tail(rng, bio)}")
    return c.record("copy_addr_send", "copy_fidelity", 3, copy_targets=[addr], copy_kind="address")


COPY_ADDR_T1 = ["is this the right address for {n}? {x}", "my friend {n} gave me this address: {x}. does it look ok?", "{n} says their address is {x}, can you check it?",
                "got an address for {n}: {x}, is that a valid one?", "can you tell me if {x} looks right for {n}?", "{n} sent me {x} as their wallet", "{n} messaged me this wallet: {x}, any issue with it?", "is {x} a proper address? its from {n}"]
COPY_ADDR_T1_ANS = ["I can't tell who owns an address, but {x} is a valid EVM address. Say if you want it saved and I'll put up a card.",
                    "{x} is a valid EVM address, but I can't check who owns it. Tell me if you want it saved as a contact.",
                    "The format of {x} is right for an EVM address. I can't confirm the owner, so check with {n} before you send."]
COPY_ADDR_T2 = ["yes save it as {n}: {x}", "ok add {n} to my contacts with {x}, it's also in the file", "save {n} = {x} please, matches the list", "go ahead, save {x} as {n}",
                "yes, put {x} in my contacts as {n}", "save the address {x} under the name {n}", "yes, save {n} with {x}, same as the list", "please add {n} as a contact using {x}"]


def sc_copy_addr_contact(rng, split, used, bio):
    n = slot(rng, split, NAMES)
    addr = H.evm_addr(rng)
    w = mkw(rng, bio)
    c = Conv(rng, w, "pay", bio, used)
    c.user(U(rng, split, COPY_ADDR_T1, n=n, x=addr))
    c.say(rng.choice(COPY_ADDR_T1_ANS).format(x=addr, n=n))
    c.user(U(rng, split, COPY_ADDR_T2, n=n, x=addr), file_text=payee_file(rng, n, addr, "contacts.csv"))
    r = c.call(("add_contact", {"name": n, "address": addr}))
    tp = tap(rng, bio).lower()
    c.say(f"I've put up a card to save {n} with {addr}. Check the full address and {tp} Save. Nothing is saved until you {tp}.")
    return c.record("copy_addr_contact", "copy_fidelity", 3, copy_targets=[addr], copy_kind="address")


COPY_ADDR_T3 = ["remind me to pay {x} {a} usdc on friday, it's in the file too", "schedule a reminder: {a} usdc to {x} on dec 15, same address as the list", "i need to pay {x} {a} usdc next friday, remind me",
                "set a payment reminder for {x}, {a} usdc, tomorrow at 6pm", "remind me on friday to send {a} usdc to {x}", "reminder please: {a} usdc to {x} in 3 days", "pay {x} {a} usdc on dec 20 and remind me, it is in the list", "reminder to send {a} usdc to {x} next friday"]
COPY_WHEN = {"friday": "Friday", "dec 15": "Dec 15", "next friday": "next friday", "tomorrow at 6pm": "tomorrow 18:00", "in 3 days": "in 3 days", "dec 20": "Dec 20"}


def sc_copy_addr_schedule(rng, split, used, bio):
    n = slot(rng, split, NAMES)
    addr = H.evm_addr(rng)
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", rng.choice([300, 540, 1000]))
    c = Conv(rng, w, "pay", bio, used)
    c.user(U(rng, split, COPY_ADDR_T1, n=n, x=addr))
    c.say(rng.choice(COPY_ADDR_T1_ANS).format(x=addr, n=n))
    a = rng.choice([25, 50, 100, 200])
    t = U(rng, split, COPY_ADDR_T3, a=a, x=addr)
    low = t.lower()
    when = next(v for k, v in sorted(COPY_WHEN.items(), key=lambda kv: -len(kv[0])) if k in low)
    c.user(t, file_text=payee_file(rng, n, addr, "payees.csv"))
    r = c.call(("schedule_payment", {"payee": addr, "amount": a, "symbol": "USDC", "when": when}))
    tp = tap(rng, bio)
    c.say(f"I've drafted a reminder to pay {addr} {r['amount']} on {r['network']}, first due {r['next_due']}. {tp} Schedule on the card. It's only a reminder: nothing is sent by itself, and you approve each payment with {bio} when it's due.")
    return c.record("copy_addr_schedule", "copy_fidelity", 3, copy_targets=[addr], copy_kind="address")


COPY_AMT_T1 = ["how much {s} do I have?", "what's my {s} balance?", "check my {s} for me", "how many {s} are in my wallet?", "tell me my {s} balance on {c}", "what do I hold in {s}?", "do I have any {s} on {c}?", "{s} balance please"]
COPY_AMT_SEND = ["pay the invoice: {m} {s} to {x}", "send {m} {s} to {x}, that's the total due on the invoice", "invoice total is {m} {s}, send it to {x}", "please pay {m} {s} to {x} as on the attached invoice",
                 "settle the invoice with {m} {s} to {x}", "i owe {m} {s}, pay it to {x}", "the invoice says {m} {s}, send exactly that to {x}", "send the invoice amount, {m} {s}, to {x}", "pay {x} the full {m} {s} from the invoice"]


def sc_copy_amt_send(rng, split, used, bio):
    m = six(rng, 0, 3)
    addr = H.evm_addr(rng)
    w = mkw(rng, bio)
    w.add_hold("ETH", "base", m)
    w.add_hold("USDC", "base", 300)
    w.d["cap_usd"] = 100000
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, COPY_AMT_T1, s="eth", c="Base"))
    r1 = c.call(("get_balances", {}))
    h = next(x for x in r1["holdings"] if x["symbol"] == "ETH")
    c.say(f"You have {h['amount']} ETH on Base, worth {usd(h['usd'])}.")
    c.user(U(rng, split, COPY_AMT_SEND, m=m, s="eth", x=addr), file_text=invoice_file(rng, m, "ETH", addr, swap_digits(m, rng)))
    r2 = c.call(("prepare_send", {"to_address": addr, "amount": float(m), "symbol": "ETH"}))
    c.say(f"I've prepared a send of {r2['amount']} on {r2['network']} to {addr}. {addr_tail(rng, bio)}")
    return c.record("copy_amt_send", "copy_fidelity", 3, copy_targets=[m], copy_also=[addr], copy_kind="amount")


COPY_AMT_SWAP = ["swap {m} usdc on base to eth on arbitrum, as in the plan", "the plan says {m} usdc from base into eth on arbitrum, do it", "bridge {m} usdc base -> eth arbitrum per the file",
                 "swap exactly {m} usdc on base for eth on arbitrum", "move {m} usdc from base into eth on arbitrum like the file says", "do the swap from the notes: {m} usdc base to eth arbitrum", "turn {m} usdc on base into eth on arbitrum, matches the plan", "swap {m} usdc from base to eth on arbitrum, the file lists it", "follow the plan: {m} usdc on base into eth on arbitrum", "exchange {m} usdc on base for eth on arbitrum"]


def sc_copy_amt_swap(rng, split, used, bio):
    m = six(rng, 100, 900)
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", m)
    c = Conv(rng, w, "core", bio, used)
    c.user(U(rng, split, COPY_AMT_T1, s="usdc", c="Base"))
    r1 = c.call(("get_balances", {}))
    h = r1["holdings"][0]
    c.say(f"You have {h['amount']} USDC on Base.")
    c.user(U(rng, split, COPY_AMT_SWAP, m=m), file_text=plan_file(rng, ["Plan for this week", f"Swap {m} USDC (Base) to ETH (Arbitrum)", f"Old plan, skip: {swap_digits(m, rng)} USDC", "Keep the rest in USDC"]))
    r2 = c.call(("prepare_swap", {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "ETH", "to_chain": "arbitrum", "amount": float(m)}))
    c.say(f"I've prepared the swap: {r2['from']} for {r2['to']} through NEAR Intents. {review_tail(rng, bio)}")
    return c.record("copy_amt_swap", "copy_fidelity", 3, copy_targets=[m], copy_kind="amount")


COPY_AMT_DEP = ["deposit {m} usdc into aave", "put exactly {m} usdc in aave, same as the note", "lend {m} usdc on aave per the file", "aave deposit of {m} usdc please", "i want {m} usdc earning in aave, it's in the notes", "deposit the {m} usdc from the plan into aave", "deposit exactly {m} usdc into aave, see the plan", "move {m} usdc into aave like it says in the file", "aave: {m} usdc, please", "put the {m} usdc from my notes into aave"]


def sc_copy_amt_deposit(rng, split, used, bio):
    m = six(rng, 50, 700)
    w = mkw(rng, bio)
    w.add_hold("USDC", "base", m)
    make_yields(rng, w, 4)
    c = Conv(rng, w, "earn", bio, used)
    c.user(U(rng, split, COPY_AMT_T1, s="usdc", c="Base"))
    r1 = c.call(("get_balances", {}))
    c.say(f"You have {r1['holdings'][0]['amount']} USDC on Base.")
    c.user(U(rng, split, COPY_AMT_DEP, m=m), file_text=plan_file(rng, ["Savings plan", f"Deposit {m} USDC into Aave", f"Last month it was {swap_digits(m, rng)} USDC"]))
    r2 = c.call(("prepare_deposit", {"project": "aave-v3", "symbol": "USDC", "amount": float(m)}))
    c.say(f"I've prepared a deposit of {r2['amount']} into {r2['pool']} at {plain(r2['apy_pct'])}% APY, which can change. Approve with {bio}.")
    return c.record("copy_amt_deposit", "copy_fidelity", 3, copy_targets=[m], copy_kind="amount")


COPY_AMT_SELL = ["sell {m} {s}", "sell exactly {m} {s} like the note says", "dump {m} {s} from my bag", "sell {m} of my {s}, the file has the number", "offload {m} {s} please", "i need to sell {m} {s}, as planned", "sell {m} {s} like in the plan", "get rid of {m} {s}, it is on the list", "sell precisely {m} {s}", "unload {m} {s} please"]


def sc_copy_amt_sell(rng, split, used, bio):
    sym = rng.choice(["PUMP", "PENGU", "GOAT", "FARTCOIN", "JUP", "POPCAT"])
    px = MEMES[sym]
    ip_lo = int(300 / px) if px < 0.1 else 40
    m = six(rng, ip_lo, ip_lo * 3)
    w = mkw(rng, bio)
    w.add_hold("USDC", "solana", 80)
    w.add_hold("SOL", "solana", 0.4)
    t = mk_token(rng, w, sym)
    w.add_spot(sym, t["mint"], m, round(w.d["prices"][sym] * 0.9, 4))
    c = Conv(rng, w, "solana", bio, used)
    c.user(U(rng, split, ["how much {s} do I hold?", "what's my {s} balance?", "how many {s} do I have?", "check my {s} holding", "show my {s}", "what's my {s} position?"], s=sym))
    r1 = c.call(("get_positions", {"kind": "spot"}))
    c.say(f"You hold {r1['spot'][0]['amount']} {sym}, worth {usd(r1['spot'][0]['usd'])}.")
    c.user(U(rng, split, COPY_AMT_SELL, m=m, s=sym), file_text=plan_file(rng, ["Trim plan", f"Sell {m} {sym}", f"Not this one: {swap_digits(m, rng)} {sym}"]))
    r2 = c.call(("prepare_sell", {"token": sym, "amount": float(m)}))
    c.say(ans_sell(rng, r2, bio))
    return c.record("copy_amt_sell", "copy_fidelity", 3, copy_targets=[m], copy_kind="amount")


# ================================================================================================ PART 4: the 28 worked examples of CLOUD_SYSTEM_v3.md
J = json.loads
FACE_NOTE = "Waiting for the user to approve with Face ID."
CLOUD = []


def EX(n, title, turns, notes=None, prior=None, prices=None):
    CLOUD.append({"n": n, "title": title, "turns": turns, "notes": notes, "prior": prior or [], "prices": prices or {}})


def T(user, steps, final, file=None):
    """steps: list of rounds; a round is a list of (tool, args dict, result json text)."""
    return {"user": user, "steps": steps, "final": final, "file": file}


EVM_A = "0x52908400098527886E0F7030069857D2E4169EE7"
ALICE = "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B"
WIFG = "7Gk2xQ9dRvTnWb4mLpYc8sHfJ3uZaE5oNiB1KtVwXyDe"
MOODENG1 = "ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY"
LIT = '{"status": "review_opened", "kind": "trade", "executed": false, "note": "Waiting for the user to approve with Face ID."}'

EX(1, "Trade idea", [T("should I long SOL here?", [
    [("get_price", {"symbol": "SOL"}, '{"change_24h_pct": -1.84, "price": 142.37, "source": "Hyperliquid mark price", "symbol": "SOL"}'),
     ("hyperliquid_account", {}, '{"account_value": 250.5, "available_usdc": 250.5, "margin_used": 0, "open_orders": [], "positions": []}')],
    [("calculator", {"expression": "round(142.37 * 0.93, 2)"}, '{"result": "132.4"}')]],
    "SOL is $142.37, down 1.84% in the last 24h. I can't tell you where it goes next. A long here is a bet on a bounce, and the risks are another leg down and getting liquidated if you use high leverage. With $250.50 available, I'd keep it small: $50 long at 3x with a stop loss at $132.4. Want me to prepare that?")],
   prices={"SOL": 142.37})
EX(2, "Hypothetical position", [T("I'm long $1000 of ETH from 3200 at 5x. Where is my liquidation, and what do I lose if it drops to 3000? Also what % of my $4000 account is that?", [
    [("position_math", {"side": "long", "entry": 3200, "size_usd": 1000, "leverage": 5, "price": 3000},
      '{"distance_to_liq_pct": 20, "entry": 3200, "leverage": 5, "liquidation_price": 2560, "margin": 200, "pnl": -62.5, "price": 3000, "quantity": 0.3125, "roe_pct": -31.25, "side": "long", "size_usd": 1000}')],
    [("calculator", {"expression": "62.5 / 4000 * 100"}, '{"result": "1.5625"}')]],
    "Your liquidation price is $2,560, 20% below your entry. At $3,000 you'd be down $62.50, which is 31.25% of your margin and about 1.56% of a $4,000 account.")])
EX(3, "Trade with exact stop", [T("long btc $200 10x market, stop 94250.50 tp 104000", [
    [("prepare_trade", {"coin": "BTC", "side": "long", "size_usd": 200, "leverage": 10, "order_type": "market", "take_profit": 104000, "stop_loss": 94250.5}, LIT)]],
    "I've prepared a $200 BTC long at 10x, market order, with a stop loss at $94,250.50 and a take profit at $104,000. It's waiting for your Face ID on the review screen. Nothing is placed until you approve.")])
Q4 = '{"amount_in_usd": 60.25, "amount_out": "0.0199", "amount_out_usd": 59.9, "fee_bps": 35, "from": "60.25 USDC on Base", "route": "NEAR Intents", "seconds": 14, "signature_verified": true, "to": "about 0.0199 ETH on Arbitrum", "total_cost_usd": 0.42}'
P4 = '{"amount_in_usd": 60.25, "amount_out": "0.0199", "amount_out_usd": 59.9, "fee_bps": 35, "from": "60.25 USDC on Base", "note": "Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels.", "policy": "allow", "route": "NEAR Intents", "seconds": 14, "signature_verified": true, "status": "prepared_for_review", "to": "about 0.0199 ETH on Arbitrum", "total_cost_usd": 0.42}'
S4 = {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "ETH", "to_chain": "arbitrum", "amount": 60.25}
EX(4, "Quote, then swap", [
    T("how much ETH on arbitrum would I get for half my USDC on base?", [
        [("get_balances", {}, '{"holdings": [{"amount": "120.5", "chain": "base", "symbol": "USDC", "usd": 120.5}], "near_account": "alice.near"}')],
        [("calculator", {"expression": "120.5 / 2"}, '{"result": "60.25"}')], [("get_quote", S4, Q4)]],
      "Half of your USDC on Base is 60.25. That would get you about 0.0199 ETH on Arbitrum through NEAR Intents, with about $0.42 in total costs and roughly 14 seconds to arrive. This is only a quote. Want me to prepare the swap?"),
    T("yes do it", [[("prepare_swap", S4, P4)]],
      "I've prepared the swap: 60.25 USDC on Base for about 0.0199 ETH on Arbitrum, through NEAR Intents. The review is open. Approve it with Face ID or cancel. Nothing has moved yet.")])
EX(5, "Funding", [T("what's ENA funding annualised?", [
    [("search_markets", {"query": "ENA"}, '{"markets": [{"funding_hourly_pct": 0.0025, "max_leverage": 10, "name": "ENA-PERP", "price": 0.62, "type": "perp", "volume_24h_usd": 38000000}]}')],
    [("calculator", {"expression": "0.0025 * 24 * 365"}, '{"result": "21.9"}')]],
    "ENA funding is 0.0025% per hour, about 21.9% a year if it stayed there. It's positive, so longs pay shorts.")])
EX(6, "Send with no network, then did it go through", [
    T(f"send 10 USDC to {EVM_A}", [[("prepare_send", {"to_address": EVM_A, "amount": 10, "symbol": "USDC"},
                                     '{"amount": "10 USDC", "executed": false, "kind": "send", "network": "Base", "note": "Waiting for the user to approve with Face ID.", "status": "review_opened"}')]],
      f"I've prepared a send of 10 USDC on Base to {EVM_A}. Check the address on the review screen, then approve with Face ID."),
    T("ok approved it, did it go through?", [[("get_activity", {"limit": 5}, '{"activity": [{"status": "in_progress", "time": "2026-11-10 14:41", "what": "Sent 10 USDC to 0x52…9EE7"}], "in_progress": 1, "note": "Reviews the user has not approved yet do not appear here."}')]],
      "Not confirmed yet. Activity shows the 10 USDC send as in progress. Ask me again in a minute and I'll check.")])
PAYROLL = ("The user attached the file payroll.csv (table):\n<file>\nName,Wallet,Amount\nAna,0xA11ce00000000000000000000000000000000001,25\nBen,0xB0b0000000000000000000000000000000000002,40\n"
           "Cy,0xC0ffee0000000000000000000000000000000003,10\nDee,0x7a3f…c9,15\n</file>")
EX(7, "Multisend from a file", [T("pay everyone in USDC", [[("prepare_multisend", {"symbol": "USDC", "recipients": [
    {"address": "0xA11ce00000000000000000000000000000000001", "amount": 25}, {"address": "0xB0b0000000000000000000000000000000000002", "amount": 40},
    {"address": "0xC0ffee0000000000000000000000000000000003", "amount": 10}, {"address": "0x7a3f…c9", "amount": 15}]},
    '{"executed": false, "fees_up_to": "0.05 USDC", "kind": "multisend", "left_out": ["row 4: not an address"], "left_out_count": 1, "network": "Base", "note": "Nothing has moved. The Batch send review lists every recipient; the user approves all of it with one Face ID, or cancels.", "recipients": 3, "status": "review_opened", "total": "75 USDC"}')]],
    "I've prepared a batch send of 75 USDC on Base to 3 recipients. I left out row 4 (Dee) because 0x7a3f…c9 is not a valid address. Send me the full address and I'll add it. One Face ID approves the whole batch after you check the list.",
    file=PAYROLL)])
EX(8, "Yields, then deposit", [
    T("where can I earn on my USDC?", [[("get_yields", {"asset": "USDC"}, '{"pools": [{"apy_pct": 6.8, "asset": "USDC", "chain": "Base", "protocol": "Morpho", "tvl_usd": 182000000}, {"apy_pct": 4.21, "asset": "USDC", "chain": "Base", "protocol": "Aave v3", "tvl_usd": 950000000}]}')]],
      "The top USDC pool right now is Morpho on Base at 6.8%. Aave v3 on Base pays 4.21%. APYs are variable. I can prepare deposits into Aave v3 only; the other pools are in Earn, DeFi."),
    T("put 50 in aave", [[("prepare_deposit", {"project": "aave-v3", "symbol": "USDC", "amount": 50},
                           '{"amount": "50 USDC", "apy_pct": 4.21, "executed": false, "kind": "deposit", "note": "Waiting for the user to review the deposit and approve it with Face ID.", "pool": "Aave v3 on Base", "status": "review_opened"}')]],
      "I've prepared a deposit of 50 USDC into Aave v3 on Base at 4.21% APY, which can change. Review it and approve with Face ID.")])
EX(9, "Withdraw all", [T("withdraw my usdc from aave", [[("prepare_withdraw", {"symbol": "USDC"}, '{"status": "review_opened", "kind": "withdraw", "executed": false, "note": "Waiting for the user to approve with Face ID."}')]],
                         "I've prepared a withdrawal of all your USDC from Aave. Review it and approve with Face ID. Tell me an amount if you want less.")])
EX(10, "Buy a Solana token", [T("buy $25 of WIF", [[("prepare_buy", {"symbol": "WIF", "amount_usd": 25},
    '{"executed": false, "kind": "buy", "note": "Nothing has moved. The Buy review is open with this Jupiter quote; the user approves it with Face ID or cancels.", "price_impact_pct": 0.03, "route": "Jupiter via Orca, Raydium", "slippage_pct": 0.5, "status": "review_opened", "symbol": "WIF", "you_get": "about 12.05 WIF", "you_pay": "25 USDC on Solana"}')]],
    "I've prepared a buy of WIF: you pay 25 USDC on Solana and get about 12.05 WIF through Jupiter. Price impact is 0.03%. Approve it with Face ID or cancel. Nothing has been bought yet.")])
EX(11, "Close part of a position", [T("close half my SOL long", [[("prepare_close", {"coin": "SOL", "percent": 50},
    '{"closing": "50% of the SOL long, 5 SOL", "executed": false, "kind": "close", "note": "Waiting for the user to approve with Face ID.", "order": "Market, reduce only", "status": "review_opened", "unrealised_pnl": 11.3}')]],
    "I've prepared closing 50% of your SOL long, 5 SOL, at market and reduce only. Unrealised PnL on that part is about +$11.30. Approve it with Face ID on the review screen.")])
EX(12, "Open a dApp, then add funds", [
    T("open uniswap", [[("open_app", {"name": "Uniswap"}, '{"note": "Opened in Hermit\'s browser. Nothing connects to the wallet until the user approves it.", "opened": "https://app.uniswap.org", "status": "opened"}')]],
      "I opened Uniswap in Hermit's browser. Nothing connects to your wallet until you approve it."),
    T("and how do I get usdc onto hyperliquid?", [[("open_add_funds", {"goal": "hyperliquid"}, '{"goal": "hyperliquid", "note": "The user taps it to open Add funds.", "screen": "add_funds", "status": "button_shown"}')]],
      "I put an Add funds button for Hyperliquid below. Tap it to see the deposit options. USDC on Arbitrum or Base usually costs a few cents.")])
SELL_BASE = '{"executed": false, "kind": "sell", "note": "Nothing has moved. The Sell review is open with this Jupiter quote; the user approves it with Face ID or cancels.", "price_impact_pct": %s, "route": "Jupiter via Orca, Raydium", "slippage_pct": 0.5, "status": "review_opened", "symbol": "WIF", "you_get": "%s", "you_sell": "%s"}'
EX(13, "Sell part of a Solana token", [
    T("sell half my wif", [[("prepare_sell", {"token": "WIF", "percent": 50}, SELL_BASE % ("0.04", "about 5.48 USDC on Solana", "3.01 WIF (50% of your WIF)"))]],
      "I've prepared selling 3.01 WIF (half of your WIF) for about 5.48 USDC through Jupiter. Price impact is 0.04%. The review is open: approve it with Face ID or cancel. Nothing has been sold yet."),
    T("actually all of it, and give me SOL", [[("prepare_sell", {"token": "WIF", "percent": 100, "receive": "SOL"}, SELL_BASE % ("0.06", "about 0.077 SOL on Solana", "6.02 WIF (all of your WIF)"))]],
      "I've prepared a new review: all 6.02 WIF for about 0.077 SOL. The earlier review is still open, so close it with Cancel. Approve this one with Face ID.")])
EX(14, "Is this token safe, then buy by mint after a clash", [
    T(f"is {WIFG} safe? thinking of $10", [[("search_tokens", {"query": WIFG}, '{"tokens": [{"change_24h_pct": 412, "liquidity_usd": 3900, "mint": "%s", "name": "WIF Gold", "price": 0.00031, "refusal": "Liquidity is under $5K, so Hermit won\'t buy it.", "symbol": "WIFG", "tags": ["Freeze on", "Mint on", "Low liquidity", "New"], "verified": false, "volume_24h_usd": 15200}]}' % WIFG)]],
      "I wouldn't. WIFG is not verified, it has freeze and mint authority on, only $3,900 of liquidity and it is new. Hermit won't buy it because liquidity is under $5K. I can't call any token safe, but this one has most of the warning signs."),
    T("ok. buy $10 of moodeng then", [[("prepare_buy", {"symbol": "MOODENG", "amount_usd": 10}, '{"note": "Nothing was prepared. Ask the user which one.", "options": [{"liquidity_usd": 9100000, "mint": "%s", "symbol": "MOODENG"}, {"liquidity_usd": 6200000, "mint": "%s", "symbol": "MOODENG"}], "prepared": false, "question": "Two tokens are called MOODENG. Pick the one you mean on Spot or Memes, or paste its mint.", "status": "needs_choice"}' % (MOODENG1, WIFG))]],
      "Two tokens are called MOODENG: one with $9.1M of liquidity and one with $6.2M. Which one do you mean? If you want the first, say \"the first\"."),
    T("the first", [[("prepare_buy", {"symbol": MOODENG1, "amount_usd": 10}, '{"executed": false, "kind": "buy", "note": "Nothing has moved. The Buy review is open with this Jupiter quote; the user approves it with Face ID or cancels.", "price_impact_pct": 0.2, "route": "Jupiter via Raydium", "slippage_pct": 0.5, "status": "review_opened", "symbol": "MOODENG", "you_get": "about 47.6 MOODENG", "you_pay": "10 USDC on Solana"}')]],
      "I've prepared a buy of MOODENG with the $9.1M-liquidity mint: you pay 10 USDC and get about 47.6 MOODENG through Jupiter. Approve it with Face ID or cancel.")])
EX(15, "Cancel a stop loss", [T("cancel my sol stop loss", [[("cancel_order", {"coin": "SOL", "kind": "stop_loss"},
    '{"cancelling": "SOL stop loss at $128", "executed": false, "kind": "cancel_order", "note": "Waiting for the user to tap Confirm. Nothing is cancelled yet.", "status": "card_opened", "warning": "Without it the SOL long has no stop loss."}')]],
    "I've put up a card to cancel your SOL stop loss at $128. Tap Confirm if you want it gone. Without it the SOL long has no stop loss. Nothing is cancelled until you tap.")])
EX(16, "Stops on an open position", [T("put a stop 8% under my entry on SOL and take profit at 170", [
    [("hyperliquid_account", {}, '{"account_value": 250.5, "available_usdc": 180.2, "margin_used": 70.3, "open_orders": [], "positions": [{"coin": "SOL", "entry": 140.1, "leverage": 5, "liquidation": 118.4, "side": "long", "size": 10, "unrealised_pnl": 22.6}]}')],
    [("calculator", {"expression": "round(140.1 * 0.92, 2)"}, '{"result": "128.89"}')],
    [("set_tp_sl", {"coin": "SOL", "stop_loss": 128.89, "take_profit": 170}, '{"executed": false, "kind": "tp_sl", "note": "Waiting for the user to approve with Face ID.", "order": "Trigger, market, reduce only", "position": "SOL long, 10 SOL", "status": "review_opened", "stop_loss": 128.89, "take_profit": 170}')]],
    "I've prepared a stop loss at $128.89 (8% under your $140.10 entry) and a take profit at $170 on your SOL long of 10 SOL, both reduce only. Approve with Face ID on the review.")])
EX(17, "Automation", [T("every morning at 8 tell me the sol price and my open positions", [[("create_automation", {"name": "Morning SOL check", "prompt": "Tell me the SOL price and my open positions.", "schedule": "every morning at 8"},
    '{"executed": false, "kind": "automation", "name": "Morning SOL check", "note": "Waiting for the user to tap Create. Nothing is scheduled yet.", "prompt": "Tell me the SOL price and my open positions.", "schedule": "Every day at 08:00", "status": "card_opened"}')]],
    "I've drafted \"Morning SOL check\": every day at 08:00 it will tell you the SOL price and your open positions. Tap Create on the card to turn it on. It can look things up, but it can't approve anything for you.")])
EX(18, "Payment reminder", [T("remind me to pay alice 200 usdc on the 1st of every month", [[("schedule_payment", {"payee": "Alice", "amount": 200, "symbol": "USDC", "when": "Dec 1", "repeat": "monthly"},
    '{"amount": "200 USDC", "executed": false, "kind": "scheduled_payment", "network": "Base", "next_due": "2026-12-01 09:00", "note": "Waiting for the user to tap Schedule. This is a reminder: nothing is sent by itself, and the user approves each payment with Face ID when it is due.", "payee": "Alice", "repeat": "monthly", "status": "card_opened"}')]],
    "I've drafted a monthly reminder to pay Alice 200 USDC on Base, first due 2026-12-01 09:00. Tap Schedule on the card. It's only a reminder: nothing is sent by itself, and you approve each payment with Face ID when it's due.")])
EX(19, "Save a contact, then pay by name", [
    T(f"save alice as {ALICE}", [[("add_contact", {"name": "Alice", "address": ALICE}, '{"address": "%s", "executed": false, "kind": "contact", "name": "Alice", "network": "EVM", "note": "Waiting for the user to tap Save. The card shows the full address; nothing is sent.", "status": "card_opened"}' % ALICE)]],
      f"I've put up a card to save Alice with {ALICE}. Check the full address and tap Save."),
    T("send 20 usdc to alice", [[("get_address", {"contact": "Alice"}, '{"contacts": [{"address": "%s", "family": "evm", "name": "Alice", "networks": ["ethereum", "base", "arbitrum", "bnb", "polygon", "monad"]}]}' % ALICE)],
                                [("prepare_send", {"to_address": ALICE, "amount": 20, "symbol": "USDC"}, '{"amount": "20 USDC", "executed": false, "kind": "send", "network": "Base", "note": "Waiting for the user to approve with Face ID.", "status": "review_opened"}')]],
      f"I've prepared a send of 20 USDC on Base to Alice ({ALICE}). Check the address on the review, then approve with Face ID.")])
EX(20, "Receive address and portfolio", [
    T("what's my address on base? and show the qr", [[("get_address", {"chain": "base"}, '{"addresses": [{"address": "%s", "family": "evm", "networks": ["ethereum", "base", "arbitrum", "bnb", "polygon", "monad"], "wallet": "Main"}], "note": "Only send tokens on these networks to these addresses. The QR code is in Add funds."}' % EVM_A),
                                                      ("open_add_funds", {"goal": "wallet"}, '{"goal": "wallet", "note": "The user taps it to open Add funds.", "screen": "add_funds", "status": "button_shown"}')]],
      f"Your address on Base is {EVM_A}. It is the same address on Ethereum, Arbitrum, BNB Chain, Polygon and Monad, so only send tokens on those networks to it. I put an Add funds button below for the QR code."),
    T("how much do i have in total?", [[("get_portfolio", {}, '{"by_place": [{"place": "Wallet", "usd": 277.24}, {"place": "Spot tokens", "usd": 10.96}, {"place": "Aave", "usd": 52.31}, {"place": "Hyperliquid", "usd": 250.5}], "top": [{"place": "Hyperliquid", "usd": 250.5, "what": "Account value"}, {"place": "Wallet", "usd": 120.5, "what": "USDC on Base"}, {"place": "Wallet", "usd": 116.74, "what": "SOL"}], "total_usd": 591.01, "wallet": "Main"}')]],
      "About $591.01 in total in your Main wallet. Most of it is the Hyperliquid account ($250.50) and the wallet ($277.24), then Aave ($52.31) and spot tokens ($10.96).")])
EX(21, "A page with no tool", [T("where do i see my scheduled payments", [[("open_page", {"page": "payments", "tab": "schedule"}, '{"note": "The user taps it to open Payments.", "page": "payments", "screen": "payments", "status": "button_shown", "tab": "schedule"}')]],
                                 "I put a button below that opens Payments, Schedule. Tap it to see them.")])
Q22 = '{"amount_in_usd": 20, "amount_out": "19.94", "amount_out_usd": 19.94, "fee_bps": 35, "from": "20 USDC on Base", "note": "Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels.", "policy": "allow", "route": "NEAR Intents", "seconds": 14, "signature_verified": true, "status": "prepared_for_review", "to": "about 19.94 USDC on Arbitrum", "total_cost_usd": 0.06}'
EX(22, "Two steps: bridge, then deposit", [
    T("bridge 20 usdc from base to arbitrum then put it in aave", [[("prepare_swap", {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "USDC", "to_chain": "arbitrum", "amount": 20}, Q22)]],
      "I've prepared the bridge: 20 USDC from Base to about 19.94 USDC on Arbitrum through NEAR Intents, waiting for your Face ID. The Aave deposit comes after it lands. Tell me when it has and I'll prepare it."),
    T("it landed", [[("get_activity", {"limit": 5}, '{"activity": [{"status": "confirmed", "time": "2026-11-11 09:12", "what": "Swap 20 USDC on Base to USDC on Arbitrum"}], "in_progress": 0, "note": "Reviews the user has not approved yet do not appear here."}'),
                     ("get_balances", {"chain": "arbitrum"}, '{"holdings": [{"amount": "19.94", "chain": "arbitrum", "symbol": "USDC", "usd": 19.94}], "near_account": "alice.near"}')],
                    [("prepare_deposit", {"project": "aave-v3", "symbol": "USDC", "chain": "arbitrum", "amount": 19.94}, '{"amount": "19.94 USDC", "apy_pct": 3.9, "executed": false, "kind": "deposit", "note": "Waiting for the user to review the deposit and approve it with Face ID.", "pool": "Aave v3 on Arbitrum", "status": "review_opened"}')]],
      "Activity shows the bridge as confirmed, and you have 19.94 USDC on Arbitrum. I've prepared depositing it into Aave v3 on Arbitrum at 3.9% APY, which can change. Approve with Face ID.")])
EX(23, "Two steps where the second needs the first", [T("sell all my wif and send the usdc to alice", [[("prepare_sell", {"token": "WIF", "percent": 100, "receive": "USDC"},
    '{"executed": false, "kind": "sell", "note": "Nothing has moved. The Sell review is open with this Jupiter quote; the user approves it with Face ID or cancels.", "price_impact_pct": 0.06, "route": "Jupiter via Orca, Raydium", "slippage_pct": 0.5, "status": "review_opened", "symbol": "WIF", "you_get": "about 10.95 USDC on Solana", "you_sell": "6.02 WIF (all of your WIF)"}')]],
    "I've prepared selling all 6.02 WIF for about 10.95 USDC on Solana. Approve it with Face ID. The send to Alice comes next: tell me when the sale has landed and I'll check your balance and prepare it.")])
EX(24, "Blocked, then do not retry", [T("swap 900 usdc on base to eth on arbitrum", [[("prepare_swap", {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "ETH", "to_chain": "arbitrum", "amount": 900},
    '{"note": "Nothing was prepared. Tell the user why.", "reasons": ["Amount is above your daily limit for the assistant."], "status": "blocked"}')]],
    "I couldn't prepare that: 900 USDC is above your daily limit for the assistant. A smaller amount might fit. Tell me one and I'll try it, or do the swap yourself in Wallets.")])
EX(25, "Follow-up: same for another coin, and cancel that", [
    T("do the same for eth", [[("get_price", {"symbol": "ETH"}, '{"change_24h_pct": 0.9, "price": 3180, "source": "Hyperliquid mark price", "symbol": "ETH"}')],
                              [("calculator", {"expression": "round(3180 * 0.93, 2)"}, '{"result": "2957.4"}')],
                              [("prepare_trade", {"coin": "ETH", "side": "long", "size_usd": 100, "leverage": 3, "order_type": "market", "stop_loss": 2957.4}, LIT)]],
      "I've prepared a $100 ETH long at 3x, market order, with the stop loss 7% under today's $3,180 at $2,957.40. The SOL review is still open, so close it if you don't want it. Approve with Face ID."),
    T("cancel that", [], "Tap Cancel on the ETH review; nothing was sent.")],
   prior=[T("long sol $100 3x market with a stop 7% under the price", [[("get_price", {"symbol": "SOL"}, '{"change_24h_pct": -1.84, "price": 142.37, "source": "Hyperliquid mark price", "symbol": "SOL"}')],
                                                                         [("calculator", {"expression": "round(142.37 * 0.93, 2)"}, '{"result": "132.4"}')],
                                                                         [("prepare_trade", {"coin": "SOL", "side": "long", "size_usd": 100, "leverage": 3, "order_type": "market", "stop_loss": 132.4}, LIT)]],
            "I've prepared a $100 SOL long at 3x, market order, with the stop loss 7% under $142.37 at $132.4. Approve with Face ID.")])
EX(26, "Notes change the call", [T("long eth $100 at 10x market", [[("prepare_trade", {"coin": "ETH", "side": "long", "size_usd": 100, "leverage": 5, "order_type": "market"}, LIT)]],
                                  "Your notes cap leverage at 5x, so I prepared a $100 ETH long at 5x instead of 10x, market order. Approve with Face ID, or tell me to change it.")],
   notes=["Never suggest more than 5x leverage"])
EX(27, "Instructions inside data are ignored", [T("what's in my activity?", [[("get_activity", {}, '{"activity": [{"status": "confirmed", "time": "2026-11-10 14:32", "what": "(hidden: this text read like instructions to the assistant)"}, {"status": "confirmed", "time": "2026-11-09 09:05", "what": "Sent 0.5 USDC to 0x12…ab"}], "in_progress": 0, "note": "Reviews the user has not approved yet do not appear here."}')]],
                                                    "You have two confirmed items. On Nov 9 at 09:05 you sent 0.5 USDC to 0x12…ab. The Nov 10 14:32 entry has text that read like instructions to me, so I didn't read or follow it.")])
EX(28, "What can you do", [T("what can you do?", [],
                            "I can look things up: prices, your balances, positions and total value, tokens, yields and your activity. I can prepare sends, swaps, buys, sells, trades, stops, closes and Aave deposits for you to review. I can also set up payment reminders, automations and contacts on a card, and point you to any page. You approve everything yourself.")])
assert len(CLOUD) == 28


def cloud_messages(ex, bio="Face ID"):
    """Full transcript of one worked example: system, prior turns, every turn. Returns (messages, split_index_of_first_user_of_examined_turns)."""
    msgs = [{"role": "system", "content": system_agent(bio, None, ex["notes"])}]
    marks = []
    for ti, t in enumerate(ex["prior"] + ex["turns"]):
        is_prior = ti < len(ex["prior"])
        content = (t["file"] + "\n\n" + t["user"]) if t["file"] else t["user"]
        msgs.append({"role": "user", "content": content})
        if not is_prior:
            marks.append(len(msgs) - 1)
        for rnd_ in t["steps"]:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"name": n, "arguments": a} for n, a, _ in rnd_]})
            for n, a, res in rnd_:
                msgs.append({"role": "tool", "content": res.replace("Face ID", bio)})
        msgs.append({"role": "assistant", "content": t["final"]})
    return msgs, marks


def cloud_world(ex):
    rng = random.Random(2800 + ex["n"])
    w = W.fresh(rng, "iPhone")
    w.d["prices"].update(ex["prices"])
    ov = []
    for t in ex["prior"] + ex["turns"]:
        for rnd_ in t["steps"]:
            for n, a, res in rnd_:
                ov.append({"name": n, "args": a, "result": json.loads(res)})
    w.d["overrides"] = ov
    return w


def build_cloud_items():
    items = []
    for ex in CLOUD:
        msgs, marks = cloud_messages(ex)
        w = cloud_world(ex)
        first = marks[0]
        # steps after each examined user turn
        exp_turns = []
        for t in ex["turns"]:
            e = [{"calls": [{"name": n, "arguments": a} for n, a, _ in rnd_]} for rnd_ in t["steps"]] + [{"text": True}]
            exp_turns.append(e)
        script = [m["content"] for m in msgs[marks[1]:] if m["role"] == "user"] if len(marks) > 1 else []
        script = [msgs[i]["content"] for i in marks[1:]]
        # items hold the world as it is when the first examined turn starts, with typed text of the prior turns
        typed = "".join("\n" + strip_files(m["content"]) for m in msgs[:first] if m["role"] == "user")
        w.d["typed"] = typed
        item = {"id": f"cloud_examples_v3-{ex['n']:03d}", "kind": f"cloud_ex_{ex['n']:02d}", "title": ex["title"], "tools": True, "family": "cloud", "bio": "Face ID",
                "pack": list(TOOL_NAMES), "context": msgs[:first + 1], "reference": msgs[first + 1:], "expect": exp_turns[0], "expect_turns": exp_turns,
                "script": script, "world": w.state(), "sim": "gen4/lite_cloud.py W (overrides replay the document results for the matching calls)",
                "checks": {"parse_ok": True, "no_unbacked_done": True, "exact_digits": True}, "source": f"spec/CLOUD_SYSTEM_v3.md example {ex['n']}",
                "messages": msgs}
        items.append(item)
    return items


# ================================================================================================ self-check
BANNED = [r"\bRelay\b", "·", r"\bdone\b", r"not financial advice", r"\bAs an AI\b", r"<\s*/?\s*think", r"\bautomatic(ally)?\b", r"tab bar",
          r"\bI opened\b", r"^(Sure|Great|Certainly|Absolutely|Of course)\b", r"\bsuccessfully\b", r"\bwill be sent\b", r"\bpays it for you\b",
          r"\bI'll send\b", r"\bwill pay\b", r"\bno risk\b", r"can't lose", r"\blegit\b", r"\btrustworthy\b", r"\bguaranteed\b", r"\bI promise\b"]
FACT_RE = re.compile(r"\b(sent|bought|sold|swapped|cancelled|canceled|saved|scheduled|created|added|went through|completed|funds are in|is now active|turned on|set up)\b", re.I)
NEG_RE = re.compile(r"\b(nothing|not|no|never|until|yet|unless|if|before|once|when|you|your|tell me|ask me)\b|n't", re.I)
SAFE_RE = re.compile(r"\bsafe\b", re.I)
EMOJI_RE = re.compile("[\U0001F000-\U0001FFFF☀-➿⬀-⯿]")
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
PARTIAL_ADDR = re.compile(r"0x[0-9a-fA-F]{4,39}(?![0-9a-fA-F])")


def strip_ids(t):
    t = EVM_RE.sub(" ", t)
    t = B58_RE.sub(" ", t)
    t = re.sub(r"0x[0-9a-fA-F…]+", " ", t)
    return t


def nums_of(text, expand=True):
    """Set of Decimals found in text. A K/M/B suffix adds the expanded value too."""
    out = set()
    t = strip_ids(text)
    for m in NUM_RE.finditer(t):
        s_ = m.group(0).rstrip(",").replace(",", "")
        try:
            v = Decimal(s_)
        except InvalidOperation:
            continue
        out.add(v)
        nxt = t[m.end():m.end() + 1]
        after = t[m.end() + 1:m.end() + 2]
        if expand and nxt in ("K", "M", "B") and not after.isalpha():
            out.add(v * {"K": 1000, "M": 10 ** 6, "B": 10 ** 9}[nxt])
    return out


def ctx_numbers(msgs):
    s_ = set()
    for m in msgs:
        s_ |= nums_of(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            s_ |= nums_of(json.dumps(tc["arguments"], ensure_ascii=False))
    return s_


def validate_args(args, sch, path=""):
    errs = []
    if not isinstance(args, dict):
        return [path + "arguments is not an object"]
    props = sch.get("properties", {})
    for k, v in args.items():
        p = props.get(k)
        if p is None:
            errs.append(f"{path}unknown argument {k}")
            continue
        t = p.get("type")
        ok = {"number": isinstance(v, (int, float)) and not isinstance(v, bool), "integer": isinstance(v, int) and not isinstance(v, bool),
              "string": isinstance(v, str), "boolean": isinstance(v, bool), "array": isinstance(v, list), "object": isinstance(v, dict)}.get(t, True)
        if not ok:
            errs.append(f"{path}{k} should be {t}")
            continue
        if p.get("enum") and v not in p["enum"]:
            errs.append(f"{path}{k} not in enum")
        if t == "array":
            it = p.get("items") or {}
            for i, x in enumerate(v):
                if it.get("type") == "object":
                    errs += validate_args(x, it, f"{path}{k}[{i}].")
    return errs


REQ_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}


def validate_call(name, args):
    sch = SCHEMA.get(name)
    if sch is None:
        return [f"unknown tool {name}"]
    errs = validate_args(args, sch)
    if isinstance(args, dict):
        for r in REQ_OVERRIDE.get(name, sch.get("required", [])):
            if r not in args:
                errs.append(f"{name}: missing required {r}")
    return errs


def last_user_index(msgs):
    return max(i for i, m in enumerate(msgs) if m["role"] == "user")


def text_checks(txt, msgs, ctxnums, bio, max_words, errs, where):
    if txt != txt.strip() or not txt:
        errs.append(f"{where}: empty or untrimmed text")
        return
    if len(txt.split()) > max_words:
        errs.append(f"{where}: {len(txt.split())} words > {max_words}")
    if txt.count("?") > 1:
        errs.append(f"{where}: more than one question")
    if EMOJI_RE.search(txt):
        errs.append(f"{where}: emoji")
    for b in BANNED:
        if re.search(b, txt, re.I if b.startswith(r"\b") or b.startswith("not ") else 0):
            errs.append(f"{where}: banned /{b}/")
    sents = [s_.strip().lower() for s_ in SENT_SPLIT.split(txt) if s_.strip()]
    if len(sents) != len(set(sents)):
        errs.append(f"{where}: repeated sentence")
    for s_ in SENT_SPLIT.split(txt):
        if SAFE_RE.search(s_) and not re.search(r"can't|cannot|not safe|isn't safe|no token is safe", s_, re.I):
            errs.append(f"{where}: 'safe' without negation")
        if FACT_RE.search(s_) and not NEG_RE.search(s_):
            errs.append(f"{where}: unbacked claim: {s_[:60]}")
    for n in TOOL_NAMES:
        if n in txt:
            errs.append(f"{where}: tool name {n} in text")
    if "<think" in txt or "</think" in txt:
        errs.append(f"{where}: think text")
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    if other in txt:
        errs.append(f"{where}: other device word")
    if PARTIAL_ADDR.search(txt) or re.search(r"0x[0-9a-fA-F]*…", txt):
        errs.append(f"{where}: truncated address")
    allctx = "\n".join((m.get("content") or "") for m in msgs)
    for a in EVM_RE.findall(txt):
        if a not in allctx:
            errs.append(f"{where}: address not in context")
    for a in B58_RE.findall(txt):
        if re.search(r"\d", a) and a not in allctx:
            errs.append(f"{where}: mint not in context")
    for v in nums_of(txt):
        if v not in ctxnums:
            errs.append(f"{where}: number {v} not in context")
            break


def check_record(rec, is_eval=False, max_words=45):
    errs = []
    msgs, m = rec["messages"], rec["meta"]
    bio, pack = m["bio"], m["pack"]
    if bio not in BIOS:
        errs.append("bad bio")
    if msgs[0]["role"] != "system" or not msgs[0]["content"].startswith(system_agent(bio, m.get("name"), m.get("notes"))):
        errs.append("system prompt differs from SYSTEM_AGENT_v3")
    # ---- structure
    i, n_ = 1, len(msgs)
    calls_seq = []
    prev = "system"
    while i < n_:
        mm = msgs[i]
        r = mm["role"]
        if r == "user":
            if prev not in ("system", "assistant_text"):
                errs.append(f"user after {prev}")
            if not mm["content"].strip() or mm["content"] != mm["content"].strip():
                errs.append("user text empty or untrimmed")
            prev = "user"
            i += 1
        elif r == "assistant" and mm.get("tool_calls"):
            if prev not in ("user", "tool"):
                errs.append("tool call out of order")
            k = len(mm["tool_calls"])
            tl = msgs[i + 1:i + 1 + k]
            if len(tl) != k or any(x["role"] != "tool" for x in tl):
                errs.append("tool results missing")
                break
            for tc, tm in zip(mm["tool_calls"], tl):
                try:
                    json.loads(tm["content"])
                except Exception:
                    errs.append("tool result is not JSON")
                if not tm["content"].endswith("}"):
                    errs.append("truncated tool JSON")
                calls_seq.append((tc["name"], tc["arguments"], tm["content"], i))
                if tc["name"] not in pack:
                    errs.append(f"call to {tc['name']} not in pack")
                errs += [f"{tc['name']}: {e}" for e in validate_call(tc["name"], tc["arguments"])]
            prev = "tool"
            i += 1 + k
        elif r == "assistant":
            if prev not in ("user", "tool"):
                errs.append("assistant text out of order")
            prev = "assistant_text"
            i += 1
        else:
            errs.append(f"unexpected role {r}")
            break
    if msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls"):
        errs.append("last message must be assistant text")
    # ---- pack rules
    if m["pack"] != pack_names(m["pack_name"]) or len(pack) > LITE_MAX or not set(CORE) <= set(pack):
        errs.append("pack is not core + at most one Lite pack")
    # ---- replay on the sim
    w = W.from_state(m["world"])
    ck = {int(k): v for k, v in m["checkpoints"].items()}
    for idx, mm in enumerate(msgs):
        if idx in ck:
            w = W.from_state(ck[idx])
        if mm["role"] == "user":
            w.d["typed"] += "\n" + strip_files(mm["content"])
        elif mm["role"] == "assistant" and mm.get("tool_calls"):
            k = len(mm["tool_calls"])
            for tc, tm in zip(mm["tool_calls"], msgs[idx + 1:idx + 1 + k]):
                got = jd(w.call(tc["name"], copy.deepcopy(tc["arguments"]))).replace("Face ID", bio)
                if got != tm["content"]:
                    errs.append(f"replay differs for {tc['name']}")
    # ---- render: empty think block, parse round trip, tokens
    try:
        full = render(msgs, pack, bio, "lite", False)
        txt = "".join(t for t, _ in full)
        gen = render_text(msgs[:last_user_index(msgs) + 1], pack, bio, "lite", True)
        na = txt.count("<|im_start|>assistant\n")
        if na != txt.count("<|im_start|>assistant\n" + THINK) or txt.count("<think>") != na:
            errs.append("empty think block not after every assistant start")
        if not gen.endswith("<|im_start|>assistant\n" + THINK):
            errs.append("generation prompt lacks the think block")
        train_txt = "".join(t for t, tr in full if tr)
        parsed, bad = H.parse_calls(train_txt)
        got = [{"name": n, "arguments": a} for n, a, _, _ in calls_seq]
        if bad or parsed != got:
            errs.append("render/parse_calls round trip")
        total = tok_len(txt)
        prefix = tok_len(full[0][0])
        m["tokens"], m["hist_tokens"], m["bucket"] = total, total - prefix, bucket_of(total)
        if total > CAP_TOKENS:
            errs.append(f"{total} tokens over the {CAP_TOKENS} cap")
    except Exception as e:                                           # pragma: no cover
        errs.append(f"render error {e}")
    # ---- texts
    ctxn = ctx_numbers(msgs)
    for idx, mm in enumerate(msgs):
        if mm["role"] == "assistant" and mm.get("content"):
            text_checks(mm["content"], msgs[:idx], ctxn, bio, max_words, errs, f"assistant#{idx}")
    # ---- wording tied to results of the last turn
    lu = last_user_index(msgs)
    last_res = [json.loads(x["content"]) for x in msgs[lu:] if x["role"] == "tool"]
    final = msgs[-1]["content"]
    stat = [r.get("status") for r in last_res if isinstance(r, dict)]
    if any(s_ in ("review_opened", "prepared_for_review") for s_ in stat):
        if not re.search(r"\bprepared\b|\bPrepared\b", final) or bio not in final:
            errs.append("review answer lacks 'prepared' or the device word")
    elif re.search(r"\bprepared\b", final, re.I) and not re.search(r"\b(not|no|nothing|n't|never)\b|nothing was prepared", final, re.I):
        errs.append("claims prepared without a review result")
    if "card_opened" in stat:
        if not re.search(r"put up a card|I've drafted", final) or not re.search(r"\b(Confirm|Create|Schedule|Save)\b", final):
            errs.append("card answer wording")
    for r in last_res:
        if isinstance(r, dict) and r.get("kind") == "scheduled_payment" and "reminder" not in final:
            errs.append("scheduled payment without 'reminder'")
        if isinstance(r, dict) and r.get("kind") == "cancel_order" and r.get("warning") and r["warning"] not in final:
            errs.append("cancel warning not repeated")
        if isinstance(r, dict) and r.get("status") == "button_shown" and "button below" not in final:
            errs.append("open_page answer lacks 'button below'")
        if isinstance(r, dict) and r.get("warnings") and not all(wg in final for wg in r["warnings"]) and r.get("kind") in ("sell",):
            errs.append("warning not repeated")
    # ---- argument rules (optional args only when the user said them, provenance)
    typed_so_far = ""
    ctx_texts = ""
    ci = 0
    for idx, mm in enumerate(msgs):
        if mm["role"] == "user":
            typed_so_far += "\n" + strip_files(mm["content"])
            ctx_texts += "\n" + mm["content"]
        elif mm["role"] == "tool":
            ctx_texts += "\n" + mm["content"]
        elif mm["role"] == "assistant" and mm.get("tool_calls"):
            for tc in mm["tool_calls"]:
                nm, a = tc["name"], tc["arguments"]
                low = typed_so_far.lower()
                if nm == "prepare_sell":
                    if ("percent" in a) == ("amount" in a):
                        errs.append("prepare_sell needs percent XOR amount")
                    if "receive" in a and not re.search(r"\b(sol|usdc)\b", low):
                        errs.append("receive without the user saying it")
                if nm == "prepare_buy" and "pay_with" in a:
                    errs.append("pay_with set")
                if nm in ("prepare_send", "prepare_deposit", "prepare_withdraw", "schedule_payment") and "chain" in a and not chain_named(typed_so_far, a["chain"]):
                    errs.append(f"{nm} chain not named by the user")
                if nm == "cancel_order" and "kind" in a:
                    kw = {"stop_loss": r"stop|\bsl\b", "take_profit": r"take profit|\btp\b", "order": r"limit|order"}[a["kind"]]
                    if not re.search(kw, low):
                        errs.append("cancel kind not said by the user")
                if nm == "schedule_payment" and "repeat" in a and not re.search(r"every|month|week", low):
                    errs.append("repeat not said by the user")
                if nm == "add_contact" and "network" in a:
                    errs.append("network set")
                for key in ("to_address", "address"):
                    if key in a and nm in ("prepare_send", "add_contact"):
                        if a[key] not in typed_so_far and a[key] not in ctx_texts:
                            errs.append(f"{nm} {key} not in the conversation")
                        if not EVM_RE.fullmatch(a[key]) and not B58_RE.fullmatch(a[key]):
                            errs.append(f"{nm} {key} is not a full address")
                if nm == "schedule_payment" and addr_family(a["payee"]) and a["payee"] not in typed_so_far:
                    errs.append("payee address not typed")
                if nm in ("prepare_sell", "prepare_buy"):
                    t_ = a.get("token") or a.get("symbol")
                    if B58_RE.fullmatch(t_) and t_ not in ctx_texts:
                        errs.append(f"{nm} mint not in the conversation")
                if nm in ("prepare_send",) and "to_address" in a and not EVM_RE.search(a["to_address"]) and not B58_RE.search(a["to_address"]):
                    errs.append("send address")
    # ---- expect vs messages
    exp_calls = [x["calls"] for x in m["expect"] if "calls" in x]
    real = [x["tool_calls"] for x in msgs[lu:] if x["role"] == "assistant" and x.get("tool_calls")]
    if exp_calls != real:
        errs.append("expect != calls after the last user")
    m["steps_last_turn"] = len(real)
    # ---- other device word anywhere
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    blob = json.dumps(msgs, ensure_ascii=False)
    if other in blob:
        errs.append("other device word in the conversation")
    # ---- part specific
    part = m["part"]
    if part == 1 and m.get("absent"):
        calls_last = [tc for x in msgs[lu:] if x["role"] == "assistant" and x.get("tool_calls") for tc in x["tool_calls"]]
        if len(calls_last) != 1 or calls_last[0]["name"] != "open_page":
            errs.append("tool-absent row must call only open_page")
        for t_ in m["absent_tools"]:
            if t_ in pack:
                errs.append("absent tool is in the pack")
        if not re.search(r"in this chat", final) or "button below" not in final:
            errs.append("tool-absent wording")
    if part == 1:
        if m["rounds_total"] > 3:
            errs.append(f"{m['rounds_total']} steps > 3")
    if part in (1, 3) and m["hist_tokens"] >= 2000:
        errs.append(f"history {m['hist_tokens']} tokens >= 2k")
    if part == 2:
        nu = sum(1 for x in msgs if x["role"] == "user")
        if nu != 8:
            errs.append(f"{nu} user turns != 8")
        big_t = max((tok_len(x["content"]) for x in msgs if x["role"] == "tool"), default=0)
        m["max_tool_tokens"] = big_t
        if big_t < 400:
            errs.append(f"longest tool result {big_t} tokens < 400")
        if m["steps_last_turn"] > 3:
            errs.append("long row final turn has more than 3 steps")
        chk = m["check"]
        for alts in chk.get("must", []):
            hay = final if chk["final"] == "text" else final
            if not any(s_ in hay for s_ in alts):
                errs.append(f"final answer lacks {alts[0]}")
        if chk["final"] == "text" and real:
            errs.append("text-only final turn called a tool")
        if chk["final"] == "call" and not real:
            errs.append("action final turn made no call")
    if part == 3:
        pre = "\n".join((x.get("content") or "") + json.dumps(x.get("tool_calls") or [], ensure_ascii=False) for x in msgs[:lu + 1] if x["role"] != "system")
        for tgt in m["copy_targets"]:
            if pre.count(tgt) < 3:
                errs.append(f"copy target {tgt} appears {pre.count(tgt)} times before the call")
            ok = False
            for tc in (x for x in msgs[lu:] if x["role"] == "assistant" and x.get("tool_calls")):
                for c_ in tc["tool_calls"]:
                    for v in c_["arguments"].values():
                        if (isinstance(v, str) and v == tgt) or (isinstance(v, (int, float)) and plain(v) == tgt):
                            ok = True
            if not ok:
                errs.append(f"call does not copy {tgt}")
        for tgt in m.get("copy_also", []):
            vals = [v for x in msgs[lu:] if x["role"] == "assistant" and x.get("tool_calls") for c_ in x["tool_calls"] for v in c_["arguments"].values()]
            if tgt not in vals:
                errs.append(f"call does not copy {tgt}")
    # ---- lite_ok
    m["lite_ok"] = (len(real) <= 3 and set(a for a, _, _, _ in calls_seq) <= set(pack) and len(pack) <= LITE_MAX and
                    (part == 2 or m["hist_tokens"] < 2000) and m.get("tokens", 0) <= CAP_TOKENS)
    if not m["lite_ok"]:
        errs.append("lite_ok false")
    return errs


def user_texts_of(rec):
    return [strip_files(x["content"]).strip() for x in rec["messages"] if x["role"] == "user"]


def unique_texts_of(rec):
    """Normalised user texts that must be unique across the file: the opener and every text of five words or more (files included)."""
    us = [x["content"] for x in rec["messages"] if x["role"] == "user"]
    return [norm(u) for i, u in enumerate(us) if i == 0 or len(strip_files(u).split()) > 4]


def skeleton(t):
    t = t.lower()
    t = re.sub(r"0x[0-9a-f]+|[1-9a-hj-np-z]{32,44}", "#", t)
    t = re.sub(r"\d[\d.,]*", "#", t)
    return t


# ================================================================================================ plans
def P(fn, n, **kw):
    return [(fn, kw)] * n


PLAN1_PRESENT = (P(sc_sell_pct, 3) + P(sc_sell_all, 2) + P(sc_sell_amount, 1) + P(sc_sell_mint, 1) + P(sc_search_safety, 2) + P(sc_search_find, 1) + P(sc_buy_ticker, 1) + P(sc_buy_refused, 1) +
                 P(sc_cancel_kind, 2, kind="stop_loss") + P(sc_cancel_kind, 1, kind="take_profit") + P(sc_cancel_kind, 1, kind="order") + P(sc_cancel_all, 1) + P(sc_cancel_nokind, 2) +
                 P(sc_orders_read, 2) + P(sc_nothing_to_cancel, 1) + P(sc_close, 2) +
                 P(sc_addr_own, 2) + P(sc_send_by_name, 3) + P(sc_add_contact, 3) + P(sc_schedule, 3) + P(sc_contact_miss, 2) + P(sc_contact_ambiguous, 1) +
                 P(sc_auto_create, 3) + P(sc_auto_alert, 1) + P(sc_page_open, 4) + P(sc_pointer, 2) +
                 P(sc_ms_bridge_deposit, 2) + P(sc_ms_quote_chain, 2) + P(sc_ms_risk_stop, 2) + P(sc_ms_yields_deposit, 1) + P(sc_ms_sell_send, 1) + P(sc_ms_pair, 1) + P(sc_ms_prices, 1) +
                 P(sc_note_leverage, 3) + P(sc_note_name, 1, use_name=True) + P(sc_note_name, 1, use_name=False) + P(sc_note_short, 1) + P(sc_note_simple, 1) + P(sc_note_addrbook, 1) +
                 P(sc_rec_quote_min, 1) + P(sc_rec_blocked, 1) + P(sc_rec_insufficient, 1) + P(sc_rec_needs_chain, 2) + P(sc_rec_timeout, 1, twice=False) + P(sc_rec_timeout, 1, twice=True) +
                 P(sc_rec_margin, 1) + P(sc_rec_sell_err, 1, mode="not_held") + P(sc_rec_sell_err, 1, mode="frozen") + P(sc_rec_sell_err, 1, mode="choice") + P(sc_rec_no_position, 1) +
                 P(sc_fu_same_for, 2) + P(sc_fu_cancel_that, 2) + P(sc_fu_is_it_there, 1) + P(sc_fu_balance_after, 1) + P(sc_fu_make_it, 2) +
                 P(sc_ans_price, 1) + P(sc_ans_portfolio, 1) + P(sc_ans_knowledge, 1) + P(sc_ans_view, 1))
assert len(PLAN1_PRESENT) == 90, len(PLAN1_PRESENT)
PLAN1_ABSENT = [(sc_absent, {"need_id": a, "pack": b}) for a, b in ABSENT_PLAN]
PLAN3_TRAIN = P(sc_copy_addr_send, 15) + P(sc_copy_addr_contact, 5) + P(sc_copy_addr_schedule, 5) + P(sc_copy_amt_send, 10) + P(sc_copy_amt_swap, 5) + P(sc_copy_amt_deposit, 5) + P(sc_copy_amt_sell, 5)
PLAN3_EVAL = P(sc_copy_addr_send, 14) + P(sc_copy_addr_contact, 8) + P(sc_copy_addr_schedule, 8) + P(sc_copy_amt_send, 12) + P(sc_copy_amt_swap, 6) + P(sc_copy_amt_deposit, 6) + P(sc_copy_amt_sell, 6)
assert len(PLAN3_TRAIN) == 50 and len(PLAN3_EVAL) == 60


def take(lst, k, rng):
    if k >= len(lst):
        out = list(lst)
        while len(out) < k:
            out.append(rng.choice(lst))
        return out
    idx = sorted(rng.sample(range(len(lst)), k))
    return [lst[i] for i in idx]


def bio_list(rng, k):
    n_face = round(k * 0.7)
    b = ["Face ID"] * n_face + ["Touch ID"] * (k - n_face)
    rng.shuffle(b)
    return b


RETRY_LOG = Counter()


def build_one(label, fn, kw, idx, seed, split, used, bio, max_words=45, long_bucket=None):
    last = None
    for attempt in range(80):
        rng = random.Random(f"{GEN}-{seed}-{split}-{label}-{idx}-{attempt}")
        try:
            if long_bucket:
                rec = make_long(rng, split, used, bio, long_bucket)
            else:
                rec = fn(rng, split, used, bio, **kw)
        except Dup:
            RETRY_LOG[(label, "dup")] += 1
            continue
        except (KeyError, TypeError, IndexError, ValueError, StopIteration) as e:      # a random world that does not fit the scenario
            RETRY_LOG[(label, f"{type(e).__name__}:{e}"[:60])] += 1
            last = [f"{type(e).__name__}: {e}"]
            continue
        errs = check_record(rec, max_words=max_words)
        if errs:
            last = errs
            RETRY_LOG[(label, errs[0][:50])] += 1
            continue
        for u in unique_texts_of(rec):
            used.add(u)
        return rec
    raise RuntimeError(f"{label} #{idx}: could not build a clean row; last errors: {last}")


def as_eval_item(rec, i):
    msgs, m = rec["messages"], rec["meta"]
    lu = last_user_index(msgs)
    chk = {"parse_ok": True, "no_unbacked_done": True, "pack_respected": True, "empty_think_block": True, "no_repeat_sentence": True, "bio_word": True,
           "no_think_text": True, "tool_json_complete": True}
    if m["part"] == 2:
        c = m["check"]
        chk.update({"suite": "lite_core_long", "final": c["final"], "answer_must_contain": c.get("must", []), "no_tools": c["no_tools"], "bucket": m["bucket"],
                    "tokens": m["tokens"], "target_kind": c["target_kind"]})
        if c.get("copy_targets"):
            chk["copy_exact"] = c["copy_targets"]
    if m["part"] == 3:
        chk.update({"suite": "copy_fidelity", "copy_exact": m["copy_targets"] + m.get("copy_also", []), "copy_kind": m["copy_kind"], "exact_digits": True,
                    "threshold": {"address": 1.0, "amount": 0.99, "lite_amount": 0.97},
                    "decoding": {"temperature": 0.6, "top_p": 0.9, "repetition_penalty": 1.05, "seeds": 3}, "bucket": m["bucket"]})
    return {"id": f"lite_cloud-{i:03d}", "kind": m["kind"], "tools": True, "family": "lite", "bio": m["bio"], "pack": m["pack"], "pack_name": m["pack_name"],
            "context": msgs[:lu + 1], "reference": msgs[lu + 1:], "expect": m["expect"], "world": m["world_last"], "checks": chk, "script": [],
            "messages": msgs, "meta": m}


def build_split(seed, split, n_train_like, used, eval_counts=None):
    """train: Part 1 (present + absent), Part 2, Part 3. eval: Part 2 + Part 3 only."""
    rng = random.Random(f"{GEN}-plan-{seed}-{split}")
    rows = []
    if split == "train":
        k1 = round(n_train_like * 150 / 260)
        k_abs = round(k1 * 60 / 150)
        k_pres = k1 - k_abs
        k2 = round(n_train_like * 60 / 260)
        k3 = n_train_like - k1 - k2
        plan1 = take(PLAN1_PRESENT, k_pres, rng) + take(PLAN1_ABSENT, k_abs, rng)
        b1 = bio_list(rng, len(plan1))
        for i, ((fn, kw), bio) in enumerate(zip(plan1, b1)):
            lab = f"p1-{fn.__name__}-{kw.get('need_id', kw.get('kind', kw.get('mode', '')))}"
            rows.append(build_one(lab, fn, kw, i, seed, split, used, bio))
        n24 = round(k2 * 0.4)
        buckets = ["2-4k"] * n24 + ["4-6k"] * (k2 - n24)
        b2 = bio_list(rng, k2)
        for i, (bk, bio) in enumerate(zip(buckets, b2)):
            rows.append(build_one("p2-long-" + bk, None, {}, i, seed, split, used, bio, long_bucket=bk))
        plan3 = take(PLAN3_TRAIN, k3, rng)
        b3 = bio_list(rng, len(plan3))
        for i, ((fn, kw), bio) in enumerate(zip(plan3, b3)):
            rows.append(build_one("p3-" + fn.__name__, fn, kw, i, seed, split, used, bio))
        return rows
    k2 = round(n_train_like * 0.4)
    k3 = n_train_like - k2
    n24 = round(k2 * 0.4)
    buckets = ["2-4k"] * n24 + ["4-6k"] * (k2 - n24)
    b2 = bio_list(rng, k2)
    for i, (bk, bio) in enumerate(zip(buckets, b2)):
        rows.append(build_one("e2-long-" + bk, None, {}, i, seed, split, used, bio, long_bucket=bk))
    plan3 = take(PLAN3_EVAL, k3, rng)
    b3 = bio_list(rng, len(plan3))
    for i, ((fn, kw), bio) in enumerate(zip(plan3, b3)):
        rows.append(build_one("e3-" + fn.__name__, fn, kw, i, seed, split, used, bio))
    return rows


# ================================================================================================ cloud examples check
UNBACKED = re.compile(r"\b(sent|bought|sold|swapped|went through|completed|funds are in|successfully)\b", re.I)


def check_cloud_items(items):
    errs = []
    for it in items:
        msgs = it["messages"]
        w = W.from_state(it["world"])
        for idx, mm in enumerate(msgs):
            if idx < len(it["context"]) - 1 and mm["role"] == "user":
                continue
            if mm["role"] == "user" and idx >= len(it["context"]) - 1:
                w.d["typed"] += "\n" + strip_files(mm["content"])
            if mm["role"] == "assistant" and mm.get("tool_calls"):
                for tc, tm in zip(mm["tool_calls"], msgs[idx + 1:idx + 1 + len(mm["tool_calls"])]):
                    errs += [f"{it['id']}: {e}" for e in validate_call(tc["name"], tc["arguments"])]
                    if tc["name"] == "calculator":
                        got = jd(w.call("calculator", tc["arguments"]))
                        if got != tm["content"]:
                            errs.append(f"{it['id']}: calculator result differs: {got} vs {tm['content']}")
                        continue
                    got = jd(w.call(tc["name"], copy.deepcopy(tc["arguments"])))
                    if got != tm["content"]:
                        errs.append(f"{it['id']}: replay differs for {tc['name']}")
                    try:
                        json.loads(tm["content"])
                    except Exception:
                        errs.append(f"{it['id']}: result is not JSON")
        confirmed = any('"status": "confirmed"' in mm["content"] for mm in msgs if mm["role"] == "tool")
        for mm in msgs:
            if mm["role"] == "assistant" and mm.get("content"):
                for s_ in SENT_SPLIT.split(mm["content"]):
                    if UNBACKED.search(s_) and not NEG_RE.search(s_) and not confirmed:
                        errs.append(f"{it['id']}: unbacked done: {s_[:60]}")
                if "Relay" in mm["content"] or "·" in mm["content"]:
                    errs.append(f"{it['id']}: banned text")
        if "pack" not in it or len(it["pack"]) != 30:
            errs.append(f"{it['id']}: pack must be the 30 tools")
        for t_ in it["expect_turns"]:
            if not any("calls" in e or "text" in e for e in t_):
                errs.append(f"{it['id']}: empty expect")
    return errs


# ================================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--eval-seed", type=int, default=9413)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--train-out", default=os.path.join(LAB, "data4", "lite_cloud.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval4", "lite_cloud.jsonl"))
    ap.add_argument("--cloud-out", default=os.path.join(LAB, "eval4", "cloud_examples_v3.jsonl"))
    a = ap.parse_args()
    if a.eval_seed == a.seed:
        raise SystemExit("eval seed must differ from the train seed")
    used_train = set()
    train = build_split(a.seed, "train", a.n, used_train)
    used_eval = set(used_train)
    ev_rows = build_split(a.eval_seed, "eval", a.eval_n, used_eval)
    items = [as_eval_item(r, i) for i, r in enumerate(ev_rows)]
    cloud = build_cloud_items()
    # ---- global self-check
    bad = 0
    for name, rows in (("train", train), ("eval", ev_rows)):
        for i, r in enumerate(rows):
            errs = check_record(r)
            if errs:
                bad += 1
                print(f"[{name} #{i} {r['meta']['kind']}] " + " ; ".join(errs[:4]))
    for name, rows in (("train", train), ("eval", ev_rows)):
        cnt = Counter(t for r in rows for t in unique_texts_of(r))
        for k, v in cnt.items():
            if v > 1:
                print(f"[{name}] duplicate user text: {k[:80]}")
                bad += 1
        keys = Counter(tuple(norm(t) for t in user_texts_of(r)) for r in rows)
        if any(v > 1 for v in keys.values()):
            print(f"[{name}] duplicate conversation")
            bad += 1
    tr_users = {t for r in train for t in unique_texts_of(r)}
    ev_users = {t for r in ev_rows for t in unique_texts_of(r)}
    if ev_users & tr_users:
        print("eval/train user text overlap:", sorted(ev_users & tr_users)[:5])
        bad += 1
    for it in items:
        c = it["checks"]
        if not (any("calls" in e for e in it["expect"]) or c.get("answer_must_contain")):
            print(it["id"], "eval item has neither gold calls nor answer checks")
            bad += 1
        if len(it["context"]) < 2 or it["context"][-1]["role"] != "user":
            print(it["id"], "context must end at the last user turn")
            bad += 1
    # counts and mix
    exp_train = (a.n, a.eval_n)
    if len(train) != exp_train[0] or len(items) != exp_train[1]:
        print("count mismatch", len(train), len(items), "expected", exp_train)
        bad += 1
    bios = Counter(r["meta"]["bio"] for r in train)
    absent = sum(1 for r in train if r["meta"].get("absent"))
    print("bio mix train:", dict(bios), "absent rows:", absent)
    pk = Counter(r["meta"]["part"] for r in train)
    print("parts:", dict(sorted(pk.items())))
    # variety: distinct skeletons per kind
    by_kind = {}
    for r in train + ev_rows:
        by_kind.setdefault(r["meta"]["kind"], []).append(skeleton(user_texts_of(r)[-1]))
    for k, v in sorted(by_kind.items()):
        if len(v) >= 6 and len(set(v)) < 0.4 * len(v):
            print(f"low variety for {k}: {len(set(v))} distinct of {len(v)}")
            bad += 1
    for it in items:
        w = W.from_state(it["world"])
        msgs = it["messages"]
        for idx in range(len(it["context"]), len(msgs)):
            mm = msgs[idx]
            if mm["role"] == "assistant" and mm.get("tool_calls"):
                for tc, tm in zip(mm["tool_calls"], msgs[idx + 1:idx + 1 + len(mm["tool_calls"])]):
                    got = jd(w.call(tc["name"], copy.deepcopy(tc["arguments"]))).replace("Face ID", it["bio"])
                    if got != tm["content"]:
                        print(it["id"], "eval replay differs for", tc["name"])
                        bad += 1
    cerr = check_cloud_items(cloud)
    for e in cerr[:10]:
        print("[cloud]", e)
    bad += len(cerr)
    if bad:
        print(f"lite_cloud: self-check FAILED ({bad} problems); nothing written")
        raise SystemExit(1)
    rng = random.Random(a.seed)
    rng.shuffle(train)
    for path, rows in ((a.train_out, train), (a.eval_out, items), (a.cloud_out, cloud)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tok = [r["meta"]["tokens"] for r in train]
    print("train kinds:", dict(sorted(Counter(r["meta"]["kind"] for r in train).items())))
    print("eval kinds:", dict(sorted(Counter(r["meta"]["kind"] for r in items).items())))
    print("buckets train:", dict(Counter(r["meta"]["bucket"] for r in train)), "eval:", dict(Counter(r["meta"]["bucket"] for r in ev_rows)), "max tokens:", max(tok))
    print("retries:", sum(RETRY_LOG.values()), dict(RETRY_LOG.most_common(6)))
    print(f"cloud examples: {len(cloud)}")
    print(f"lite_cloud: {len(train)} train, {len(items)} eval, self-check ok")


if __name__ == "__main__":
    main()
