#!/usr/bin/env python3
"""rounding_brief: honest rounding, units, short-by-default answers, detail on request, small tables.

Feeds the derived suite latency_short (spec/PLAN_v3.md 2.10, GAPS_behaviour_v3.md P1-6 and P2-11).

  python rounding_brief.py --seed 4 --n 750 --eval-seed 9404 --eval-n 100

Writes  ../data4/rounding_brief.jsonl   (train, seed 4, about 750 records)
        ../eval4/rounding_brief.jsonl   (100 held-out items: other seed, held-out templates and topics, user texts
                                         disjoint from train and from eval/eval2/eval3, gold calls and answer checks)

RULES taught (every assistant string is an f-string over tool results, calculator results or the user's own text):
  * USD two decimals and thousands comma ($1,234.57); under $1 four significant digits; $1.9B and $38M for TVL/volume.
  * Tokens: up to 4 decimals above 1 (2 for stablecoins and amounts of 1,000 or more), 3 to 4 significant digits below 1.
  * Any raw value (18/24-decimal balance, file cell, dApp text, old-build amount_out) goes through calculator
    round(x,k) and the answer prints the RESULT. Percentages at most 2 decimals; 35 bps = 0.35% (calculator);
    funding annualised with the calculator (hourly*24*365), "about" for estimates.
  * Anything that becomes an ARGUMENT keeps full precision (send all passes the balance, stop digits exact).
  * Lead with the answer plus at most two supporting numbers; 5+ holdings -> top 3 then "and N smaller"; sums via the
    calculator; unreachable chains "may be missing".
  * Plain answers 2 to 4 sentences (cap 70 words; 110 for views, files and tables); "in detail" up to 250 words;
    "tl;dr" one sentence; "shorter" shortens; "more" extends WITHOUT repeating. Tables: 3+ items on 2+ attributes ->
    markdown table of at most 5 rows plus one sentence; one fact stays a sentence; aligned plain lines when
    meta.table_ok is false (those rows carry a standing note saying the chat view shows plain text only).
  * No filler opener, no middle dot, preamble before a call empty or <= 12 words with no number.
Nothing here edits gen2/gen3: their helpers are imported through sys.path and the v3 registry is patched in memory.
"""
import argparse, ast, json, math, os, random, re, sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, getcontext

sys.dont_write_bytecode = True
getcontext().prec = 60
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(LAB, "gen3"), os.path.join(LAB, "gen2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hermit_fmt as H      # gen2: calculator, render, parse_calls, addresses, with_user constants
import fmt3                 # gen3: validate_args (schema walk)

SPEC = os.path.join(LAB, "spec")
V3_TOOLS = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
SYSTEM_V3 = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert "{BIO}" in SYSTEM_V3
TOOL_NAMES = [t["function"]["name"] for t in V3_TOOLS]
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in V3_TOOLS}
REQUIRED_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}      # chain only when the user named it
H.TOOLS, H.TOOL_NAMES, H.TOOL_SCHEMA = V3_TOOLS, TOOL_NAMES, TOOL_SCHEMA
BIOS = ("Face ID", "Touch ID")


def tools_block(bio):
    lines = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in V3_TOOLS)
    txt = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
           "You are provided with function signatures within <tools></tools> XML tags:\n<tools>" + lines +
           "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
           "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
    return txt.replace("Face ID", bio)


def render_text(msgs, bio):
    H.TOOLS_BLOCK = tools_block(bio)
    return H.render_text(msgs, True)


class Retry(Exception):
    pass


# ------------------------------------------------------------------ number helpers
def D(x):
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        return Decimal(repr(x))
    return Decimal(str(x).strip())


def plain(d):
    d = D(d)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def ndec(s):
    s = plain(s)
    return len(s.split(".")[1]) if "." in s else 0


def jn(x):
    d = D(x)
    return int(d) if d == d.to_integral_value() else float(d)


def approx(x):
    """Units.approx of the app: >= 1000 -> 2 dp, else 3 significant digits, trailing zeros trimmed (string)."""
    d = D(x)
    if d >= 1000:
        return plain(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    e = d.adjusted()
    return plain(d.quantize(Decimal(1).scaleb(e - 2), rounding=ROUND_HALF_UP))


def k_usd(v):
    v = abs(D(v))
    if v >= 1 or v == 0:
        return 2
    return 3 - v.adjusted()                  # four significant digits under $1


def k_tok(sym, v, sig=4):
    v = abs(D(v))
    if v == 0:
        return 2
    if v < 1:
        return (sig - 1) - v.adjusted()
    if sym in STABLE or v >= 1000:
        return 2
    return 4


def comma(s):
    """thousands commas on a plain decimal string"""
    d = D(s)
    neg = d < 0
    t = plain(abs(d))
    ip, _, fp = t.partition(".")
    ip = f"{int(ip):,}"
    return ("-" if neg else "") + ip + ("." + fp if fp else "")


def f_usd(s, sign=False):
    d = D(s)
    neg = d < 0
    a = abs(d)
    if a >= 1 or a == 0:
        t = f"{a:,.2f}"
    else:
        t = plain(a)
        if ndec(t) < 2:
            t = f"{a:.2f}"
    pre = "-" if neg else ("+" if sign and a != 0 else "")
    return pre + "$" + t


def f_tok(s):
    return comma(plain(D(s)))


def f_pct(s, sign=False):
    d = D(s)
    t = plain(d)
    return ("+" if sign and d > 0 else "") + t + "%"


STABLE = {"USDC", "USDT"}
DISPLAY = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "bnb": "BNB Chain", "bsc": "BNB Chain",
           "polygon": "Polygon", "monad": "Monad", "solana": "Solana", "near": "NEAR", "bitcoin": "Bitcoin",
           "hyperliquid": "Hyperliquid"}
CHAIN_WORDS = {
    "ethereum": ["ethereum", "Ethereum", "mainnet", "eth mainnet"], "base": ["base", "Base", "base chain"],
    "arbitrum": ["arbitrum", "Arbitrum"], "bnb": ["bnb chain", "BNB Chain", "bsc", "BSC", "bnb"],
    "polygon": ["polygon", "Polygon"], "solana": ["solana", "Solana"], "near": ["near", "NEAR"],
}
HOLD = {"ETH": ["ethereum", "base", "arbitrum"], "USDC": ["ethereum", "base", "arbitrum", "bnb", "polygon", "solana"],
        "USDT": ["ethereum", "arbitrum", "bnb", "solana"], "BNB": ["bnb"], "SOL": ["solana"], "NEAR": ["near"],
        "POL": ["polygon"], "LINK": ["ethereum"], "UNI": ["ethereum"], "AAVE": ["ethereum"], "ARB": ["arbitrum"]}
DEC = {"ETH": 18, "USDC": 6, "USDT": 6, "BNB": 18, "SOL": 9, "NEAR": 24, "POL": 18, "LINK": 18, "UNI": 18, "AAVE": 18,
       "ARB": 18, "HYPE": 18, "BTC": 8, "DOGE": 8}
PRICE_BAND = {"BTC": (62000, 118000), "ETH": (2200, 4800), "SOL": (95, 260), "BNB": (480, 900), "HYPE": (14, 58),
              "NEAR": (2.4, 8.5), "LINK": (10, 28), "ARB": (0.3, 1.2), "POL": (0.2, 0.7), "UNI": (5, 14),
              "AAVE": (120, 330), "DOGE": (0.11, 0.4), "AVAX": (18, 50), "XRP": (0.5, 3.1), "USDC": (1, 1), "USDT": (1, 1)}


def rand_price(rng, sym):
    lo, hi = PRICE_BAND[sym]
    if lo == hi:
        return 1
    p = rng.uniform(lo, hi)
    return round(p, 2) if p >= 10 else round(p, 4)


def raw_amount(rng, sym, usd, price, dec=None, sig15=False):
    """A raw looking token amount (string, many decimals) worth about `usd` at `price`."""
    dec = dec or DEC.get(sym, 18)
    if sym in STABLE:
        a = Decimal(int(usd * 10 ** 6) + rng.randint(1, 9999)) / Decimal(10 ** 6)
    else:
        u = Decimal(int(usd * 100)) / 100 + Decimal(rng.randint(1, 999999)) / Decimal(10 ** 6)
        a = (u / D(price)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    if sig15:
        from decimal import Context
        a = Context(prec=15, rounding=ROUND_HALF_UP).create_decimal(a)
    s = plain(a)
    if a == 0:
        raise Retry()
    return s


def float_exact(s):
    return Decimal(repr(float(s))) == D(s)


def fmt_plain(x):
    return plain(D(x))


def wc(text):
    return len(re.findall(r"\S+", text))


def wc_table(text):
    return len([w for w in re.findall(r"\S+", text) if re.search(r"[A-Za-z0-9$]", w)])


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def norm_text(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def dump(r):
    return json.dumps(r, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ simulator (the v3 shapes this generator needs)
class RBWorld:
    """Deterministic snapshot world: tools return what the snapshot holds, so a recorded conversation replays exactly."""

    def __init__(self, d):
        self.d = d

    @classmethod
    def from_state(cls, st):
        return cls(json.loads(json.dumps(st)))

    def state(self):
        return json.loads(json.dumps(self.d))

    def call(self, name, args):
        if name == "calculator":
            try:
                return json.loads(H.calc_result(str((args or {})["expression"])))
            except Exception as e:
                return {"error": f"bad arguments: {e}"}
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args if isinstance(args, dict) else {}))
        except (TypeError, ValueError, KeyError) as e:
            return {"error": f"bad arguments: {e}"}

    # reads
    def t_get_balances(self, chain=None):
        ch = None if chain is None else ("bnb" if str(chain).lower() == "bsc" else str(chain).lower())
        rows = [dict(r) for r in self.d.get("balances", []) if not ch or r["chain"] == ch]
        rows.sort(key=lambda r: (-D(r["usd"]), r["symbol"], r["chain"]))
        out = {"holdings": rows}
        if any(r["chain"] == "near" for r in rows) and self.d.get("near_account"):
            out["near_account"] = self.d["near_account"]
        if self.d.get("unreachable"):
            out["unreachable"] = list(self.d["unreachable"])
        if self.d.get("wallet"):
            out["wallet"] = self.d["wallet"]
            out["note"] = self.d["wallet_note"]
        return out

    def t_get_portfolio(self):
        return dict(self.d["portfolio"])

    def t_hyperliquid_account(self):
        return json.loads(json.dumps(self.d["hl"]))

    def t_get_positions(self, kind="all"):
        p = self.d["positions"]
        if kind in (None, "all"):
            return json.loads(json.dumps(p))
        out = {kind: p.get(kind), "wallet": p.get("wallet")}
        return json.loads(json.dumps(out))

    def t_get_price(self, symbol):
        row = self.d.get("prices", {}).get(str(symbol).upper())
        if not row:
            return {"error": f"No Hyperliquid market for {symbol}. Answer from what you know instead, and say there's no live price here."}
        return {"change_24h_pct": row["change"], "price": row["price"], "source": "Hyperliquid mark price",
                "symbol": str(symbol).upper()}

    def t_search_markets(self, query):
        toks = [t for t in re.split(r"[\s,/]+", str(query).upper()) if t]
        rows = []
        for m in self.d.get("markets", []):
            hay = m["name"].upper() + " " + m["type"].upper()
            if any(t in hay for t in toks):
                rows.append(dict(m))
        return {"markets": rows}

    def t_get_yields(self, asset=None, chain=None):
        rows = [dict(p) for p in self.d.get("pools", [])
                if (not asset or p["asset"].upper() == str(asset).upper())
                and (not chain or p["chain"].lower() == DISPLAY.get(str(chain).lower(), str(chain)).lower())]
        rows.sort(key=lambda p: (-D(p["apy_pct"]), p["protocol"]))
        return {"pools": rows[:5]}

    def t_get_quote(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        key = json.dumps([str(from_symbol).upper(), str(from_chain).lower(), str(to_symbol).upper(),
                          str(to_chain).lower(), jn(amount)])
        q = self.d.get("quotes", {}).get(key)
        if q is None:
            return {"error": "No route found for this pair."}
        return dict(q)

    # money (prepare only)
    def t_prepare_send(self, to_address, amount, symbol, chain=None):
        sym = str(symbol).upper()
        held = [r for r in self.d.get("balances", []) if r["symbol"] == sym]
        if chain:
            held = [r for r in held if r["chain"] == ("bnb" if str(chain).lower() == "bsc" else str(chain).lower())]
        if not held:
            return {"error": f"You don't hold {sym}."}
        if len(held) > 1:
            return {"note": "Nothing was prepared. Ask the user which network, in one short question.",
                    "options": sorted(r["chain"] for r in held), "prepared": False, "question": "Which network should the send use?",
                    "status": "needs_chain"}
        if D(amount) > D(held[0]["amount"]):
            return {"error": f"Not enough {sym}: you have {held[0]['amount']}."}
        return {"amount": f"{plain(D(amount).quantize(Decimal(1).scaleb(-k_tok(sym, amount, 4)), rounding=ROUND_HALF_UP))} {sym}", "executed": False, "kind": "send",
                "network": DISPLAY[held[0]["chain"]], "note": f"Waiting for the user to approve with {self.d['bio']}.",
                "status": "review_opened"}

    def t_set_tp_sl(self, coin, take_profit=None, stop_loss=None):
        c = str(coin).upper()
        pos = [p for p in self.d["hl"]["positions"] if p["coin"] == c]
        if not pos:
            return {"note": f"The user has no open {c} position on Hyperliquid. Tell them.", "prepared": False,
                    "status": "no_position"}
        if take_profit is None and stop_loss is None:
            return {"error": "Give a take_profit or a stop_loss."}
        p = pos[0]
        mark = D(self.d["prices"][c]["price"])
        for val, lab in ((stop_loss, "stop loss"), (take_profit, "take profit")):
            if val is None:
                continue
            below = (lab == "stop loss") == (p["side"] == "long")
            if (below and D(val) >= mark) or (not below and D(val) <= mark):
                return {"error": f"A {lab} for a {p['side']} must be {'below' if below else 'above'} the current price ({plain(mark)}).",
                        "prepared": False}
        out = {"executed": False, "kind": "tp_sl", "note": f"Waiting for the user to approve with {self.d['bio']}.",
               "order": "Trigger, market, reduce only", "position": f"{c} {p['side']}, {approx(p['size'])} {c}",
               "status": "review_opened"}
        if stop_loss is not None:
            out["stop_loss"] = stop_loss
        if take_profit is not None:
            out["take_profit"] = take_profit
        return out


# ------------------------------------------------------------------ conversation builder
NAMES = ["Noah", "Maya", "Leo", "Aisha", "Tom", "Yuki", "Sam", "Priya", "Diego", "Nina", "Omar", "Chloe", "Ken",
         "Zara", "Ivan", "Lena", "Arjun", "Sofia", "Ben", "Mei"]
SAFE_NOTES = ["I trade mostly BTC and ETH perps on Hyperliquid.", "I use Base and Arbitrum most.",
              "I'm new to crypto, explain simply.", "I prefer stablecoin yield over farming.",
              "I hold SOL long term, don't tell me to sell it.", "I live in Europe, quote times in CET."]
SHORT_NOTE = "Keep answers very short."
NO_TABLE_NOTE = "The chat view cannot show tables: use plain aligned lines."
PRE_PREP = ["Preparing that.", "Setting up the review.", "Opening the review.", "On it."]
PRE_CALC = ["Doing the maths.", "Rounding that.", "Working it out.", "Calculating."]
PRE_OK = ["Checking.", "Looking that up.", "One moment.", "Pulling that now.", "Let me check.", "Getting the numbers.", "Reading that now.", "On it."]


class Conv:
    def __init__(self, rng, world, notes=(), allow_notes=True):
        self.rng, self.w = rng, world
        self.bio = "Face ID" if rng.random() < 0.70 else "Touch ID"
        self.w.d["bio"] = self.bio
        s = SYSTEM_V3.replace("{BIO}", self.bio)
        if rng.random() < 0.22:
            s += f"\nThe user wants to be called {rng.choice(NAMES)}."
        ns = list(notes)
        if allow_notes and not ns and rng.random() < 0.12:
            ns = rng.sample(SAFE_NOTES, rng.choice([1, 1, 2]))
            if rng.random() < 0.15:
                ns.append(SHORT_NOTE)
        if ns:
            s += "\nThe user's standing notes (hermit.md). Follow them:\n" + "\n".join("- " + n for n in ns)
        self.msgs = [{"role": "system", "content": s}]
        self.expect, self.nums, self.table_ok = [], [], True
        self.world0 = world.state()
        self.world_last = world.state()

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        self.expect = []
        self.world_last = self.w.state()

    def call(self, *calls, pre=None):
        tcs = [{"name": n, "arguments": a} for n, a in calls]
        if pre is None:
            names = {n for n, _ in calls}
            pool = PRE_CALC if names == {"calculator"} else (PRE_PREP if any(n.startswith("prepare_") or n == "set_tp_sl" for n in names) else PRE_OK)
            pre = self.rng.choice(pool) if self.rng.random() < 0.15 else ""
        self.msgs.append({"role": "assistant", "content": pre, "tool_calls": tcs})
        self.expect.append({"calls": tcs})
        res = []
        for n, a in calls:
            r = self.w.call(n, a)
            res.append(r)
            self.msgs.append({"role": "tool", "content": dump(r)})
        return res if len(res) > 1 else res[0]

    def say(self, text, ask=False):
        self.msgs.append({"role": "assistant", "content": text})
        self.expect.append({"ask": True} if ask else {"text": True})

    def ex(self, kind, spec):
        meta = {"source": "rounding_brief", "generator": "rounding_brief", "kind": kind, "bio": self.bio,
                "world": self.world_last, "world_final": self.w.state(), "expect": self.expect, "spec": spec,
                "lang": "en", "table_ok": self.table_ok}
        spec["nums"] = self.nums
        spec["gold_calls"] = [dict(tc) for e in self.expect for tc in e.get("calls", []) if tc["name"] != "calculator"]
        return {"messages": self.msgs, "tools": True, "meta": meta}


class Calc:
    """Collects the roundings an answer needs and runs them as ONE parallel calculator turn (identical expressions run once).
    A value that already fits the rule is printed from its source and needs no call."""

    def __init__(self, conv):
        self.c, self.todo, self.val = conv, [], {}

    def rnd(self, key, src, k, force=False):
        s = plain(D(src))
        if abs(D(s) * D(10) ** k * 2) % 2 == 1:           # exact tie: float and decimal rounding could disagree
            raise Retry()
        if not force and ndec(s) <= k:
            self.val[key] = dict(text=s, calc=False, src=s, k=k)
        else:
            self.todo.append((key, f"round({s},{k})", s, k))

    def expr(self, key, expression, src=None, k=None):
        self.todo.append((key, expression, src, k))

    def run(self, pre=None, maxn=8):
        if not self.todo:
            return
        uniq = []
        for _, e, _, _ in self.todo:
            if e not in uniq:
                uniq.append(e)
        if len(uniq) > maxn:
            raise Retry()
        res = self.c.call(*[("calculator", {"expression": e}) for e in uniq], pre=pre)
        res = res if isinstance(res, list) else [res]
        out = dict(zip(uniq, res))
        for key, e, src, k in self.todo:
            if "result" not in out[e]:
                raise Retry()
            self.val[key] = dict(text=out[e]["result"], calc=True, src=src, k=k, expr=e)
        self.todo = []

    def __getitem__(self, key):
        return self.val[key]["text"]

    def rounded(self, key):
        return self.val[key]["calc"]

    def commit(self):
        for key, v in self.val.items():
            self.c.nums.append({"key": key, "shown": v["text"], "src": v["src"], "k": v["k"], "calc": v["calc"]})


# ------------------------------------------------------------------ user phrasing
def pick(rng, pool, ev, mod=5):
    """templates whose index % mod == 0 are held out for eval; the text after '|' is a flag list."""
    cand = [s for i, s in enumerate(pool) if (i % mod == 0) == ev]
    if not cand:
        raise Retry()
    return rng.choice(cand)


def typo_word(rng, w):
    op = rng.choice(["swap", "drop", "dbl"])
    i = rng.randrange(1, len(w) - 1)
    if op == "swap" and w[i] != w[i + 1]:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "drop" and w[i] in "aeiou":
        return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


NO_TYPO = {"ethereum", "arbitrum", "polygon", "solana", "mainnet", "bitcoin", "aave", "morpho", "compound", "spark", "euler", "fluid",
           "pendle", "kamino", "hyperliquid", "binance"}


def finish(rng, text, typo=0.10, protect=()):
    """casual surface noise: lower case, dropped final punctuation, one typo, 'pls'. Slot values in `protect` stay."""
    marks = {}
    for i, p in enumerate(protect):
        if p and p in text:
            tok = f"\x00{i}\x00"
            marks[tok] = p
            text = text.replace(p, tok)
    r = rng.random()
    if r < 0.38:
        text = text.lower()
    elif r < 0.45:
        text = text[:1].upper() + text[1:]
    if rng.random() < typo:
        ws = [m for m in re.finditer(r"\b[a-z]{5,}\b", text) if m.group().lower() not in NO_TYPO]
        if ws:
            m = rng.choice(ws)
            text = text[:m.start()] + typo_word(rng, m.group()) + text[m.end():]
    if rng.random() < 0.05:
        text += rng.choice([" pls", " plz", " rn"])
    if rng.random() < 0.2:
        text = text.rstrip("?.")
    for tok, p in marks.items():
        text = text.replace(tok, p)
    return re.sub(r"[ \t]+", " ", text).strip()


def tk(rng, sym):
    r = rng.random()
    return sym if r < 0.55 else (sym.lower() if r < 0.95 else sym.capitalize())


def cw(rng, chain):
    return rng.choice(CHAIN_WORDS[chain])


WORDS = {5: "five", 10: "ten", 20: "twenty", 50: "fifty", 100: "a hundred"}


def amt_text(rng, v):
    fv = float(v)
    r = rng.random()
    s = plain(v)
    if fv.is_integer() and int(fv) in WORDS and r < 0.06:
        return WORDS[int(fv)]
    if fv >= 1000 and fv % 100 == 0 and r < 0.2:
        return f"{fv / 1000:g}k"
    if fv >= 1000 and fv.is_integer() and r < 0.5:
        return f"{int(fv):,}"
    if 0 < fv < 1 and r < 0.2:
        return s[1:]
    return s


def user_numbers(text):
    out = set()
    t = text.lower()
    for m in re.finditer(r"(\d[\d,]*\.?\d*)\s*k\b", t):
        out.add(D(m.group(1).replace(",", "")) * 1000)
    for m in re.finditer(r"(?<![\w.])\.(\d+)", t):
        out.add(D("0." + m.group(1)))
    for m in re.finditer(r"\d[\d,]*\.?\d*", t):
        try:
            out.add(D(m.group().replace(",", "").rstrip(".")))
        except Exception:
            pass
    for k, v in WORDS.items():
        if re.search(r"\b" + v + r"\b", t):
            out.add(D(k))
    return out


# ------------------------------------------------------------------ world builders
def make_holdings(rng, k, must=(), no_sym=(), allow_long=True, sig15=False):
    """k holdings rows (distinct symbol+chain). must = [(sym, chain)] first."""
    rows, used = [], set()
    pairs = [(s, c) for s, cs in HOLD.items() for c in cs if s not in no_sym]
    weights = {"USDC": 5, "ETH": 4, "USDT": 2, "SOL": 2, "BNB": 1.5, "NEAR": 1, "POL": .6, "LINK": .5, "UNI": .4,
               "AAVE": .4, "ARB": .5}
    px = {}

    def price(sym):
        if sym not in px:
            px[sym] = rand_price(rng, sym) if sym in PRICE_BAND else 1
        return px[sym]

    def add(sym, ch):
        if (sym, ch) in used:
            return False
        usd = rng.choice([rng.uniform(4, 300), rng.uniform(300, 6000), rng.uniform(1, 40)])
        a = raw_amount(rng, sym, usd, price(sym), sig15=sig15)
        val = D(a) * D(price(sym))
        usd_v = jn(round(float(val), 6)) if (allow_long and rng.random() < 0.18) else jn(round(float(val), 2))
        if float(usd_v) <= 0:
            return False
        rows.append({"amount": a, "chain": ch, "symbol": sym, "usd": usd_v})
        used.add((sym, ch))
        return True

    for sym, ch in must:
        add(sym, ch)
    tries = 0
    while len(rows) < k and tries < 200:
        tries += 1
        s, c = rng.choices(pairs, weights=[weights[p[0]] for p in pairs])[0]
        add(s, c)
    return rows, px


def base_world(rng, rows=None, unreachable=False, multi=False):
    d = {"balances": rows or [], "prices": {}}
    if any(r["chain"] == "near" for r in d["balances"]):
        d["near_account"] = rng.choice(["alice", "kaito", "maria", "dev42", "lena", "rohan"]) + ".near"
    if unreachable:
        d["unreachable"] = sorted(rng.sample(["bnb", "polygon", "ethereum", "arbitrum", "base"], rng.choice([1, 1, 2])))
    if multi:
        names = rng.sample(["Savings", "Trading", "Spare", "Work"], 2)
        d["wallet"] = "Main"
        d["wallet_note"] = f"Showing the selected wallet (Main). Other wallets: {names[0]}, {names[1]}."
        d["other_wallets"] = names
    return d


def unreach_text(rng, d):
    ch = [DISPLAY[c] for c in d["unreachable"]]
    if len(ch) == 1:
        return rng.choice([f"{ch[0]} couldn't be reached, so it may be missing.", f"I couldn't reach {ch[0]}, so that may be missing."])
    return f"{ch[0]} and {ch[1]} couldn't be reached, so they may be missing."


def cap(s):
    return s[:1].upper() + s[1:]


def ab(calc_used):
    return "about " if calc_used else ""


# ------------------------------------------------------------------ A. balances with raw long amounts
A1_TOK = [
    "how much {t} do I have?", "what's my {t} balance", "do I have any {t}", "{t} balance?", "how much {t} am I holding",
    "check my {t}", "how many {t} do I have rn", "balance of {t} please", "show my {t}", "what's my {t} sitting at",
    "tell me my {t} balance", "how much {t} is in my wallet", "got any {t}?", "how's my {t}, how much is there",
    "i want to know how much {t} i have", "what's my {t} worth and how much do I have",
]
A1_CHAIN = [
    "how much {t} do I have on {c}?", "my {t} balance on {c}", "{c}: how much {t}", "{t} on {c}?", "how much {t} is on {c}",
    "check {t} on {c}", "what {t} do I hold on {c}", "balance of {t} ({c})", "how much {t} did I leave on {c}",
    "show {t} on {c}", "got any {t} on {c}?", "what's my {c} {t} balance",
]
A2_POOL = [
    "what's in my wallet?", "show my balances", "what do I hold?", "how much do I have?", "balances?",
    "what's my wallet look like", "list my holdings", "what coins do I have", "check my wallet", "what am I holding right now",
    "give me my balances", "wallet summary pls", "how much total do I have in my wallet", "show me everything I own",
    "what tokens do I have", "can you check what I've got", "my holdings?", "what's my balance", "what do I have in here",
    "tell me what's in my wallet", "quick look at my wallet", "how much is in my wallet",
]
A3_POOL = [
    "how much do I have in total?", "what's my total portfolio value", "total net worth across everything", "what's everything worth",
    "portfolio value?", "how much is all my stuff worth", "my total balance including aave and hyperliquid",
    "sum of everything I own", "what's my whole portfolio at", "give me my total", "how much do I have overall",
    "total across wallet, aave and hyperliquid?",
]
A4_POOL = [
    "send all my {t} to {a}", "send everything in {t} to {a}", "move my whole {t} balance to {a}", "transfer all {t} to {a}",
    "{a} send max {t}", "send the full {t} balance to {a}", "please send all of my {t} to {a}", "empty my {t} to {a}",
    "send my entire {t} to {a}", "sweep my {t} to {a}", "send all {t} -> {a}", "{a} <- all my {t}",
]


def sc_balance_single(rng, ev):
    sym = rng.choices(["ETH", "SOL", "BNB", "NEAR", "LINK", "UNI", "AAVE", "ARB", "POL", "USDC", "USDT"],
                      weights=[6, 3, 2, 2, 1, 1, 1, 1, 1, 4, 1])[0]
    chains = HOLD[sym]
    multi_chain = len(chains) > 1 and rng.random() < 0.30
    held_n = 2 if multi_chain else 1
    must = [(sym, c) for c in rng.sample(chains, held_n)]
    absent = rng.random() < 0.08
    unreach = rng.random() < 0.13
    rows, px = make_holdings(rng, rng.choice([0, 1, 2, 3]) + (0 if absent else held_n), must=[] if absent else must,
                             no_sym=(sym,) if absent else ())
    if absent:
        rows = [r for r in rows if r["symbol"] != sym]
    d = base_world(rng, rows, unreachable=unreach)
    if unreach and not absent and any(c in d["unreachable"] for _, c in must):
        raise Retry()
    w = RBWorld(d)
    c = Conv(rng, w)
    named_chain = (not absent) and held_n == 1 and rng.random() < 0.35
    t = tk(rng, sym)
    if named_chain:
        cc = must[0][1]
        text = finish(rng, pick(rng, A1_CHAIN, ev).format(t=t, c=cw(rng, cc)), protect=(t,))
    else:
        text = finish(rng, pick(rng, A1_TOK, ev).format(t=t), protect=(t,))
    c.user(text)
    args = {"chain": must[0][1]} if named_chain else {}
    r = c.call(("get_balances", args))
    mine = [h for h in r["holdings"] if h["symbol"] == sym]
    cl = Calc(c)
    spec = {"shape": "long_balance", "scenario": "single", "symbol": sym, "counts": []}
    if not mine:
        reply = rng.choice([f"You don't hold any {sym} right now.", f"No {sym} in this wallet right now.", f"You have no {sym} at the moment."])
        if r.get("unreachable"):
            reply += " " + unreach_text(rng, d)
        c.say(reply)
        spec["scenario"] = "absent"
        return c.ex("long_balance", spec)
    if len(mine) == 1:
        h = mine[0]
        cl.rnd("amt", h["amount"], k_tok(sym, h["amount"], rng.choice([3, 4])))
        cl.rnd("usd", plain(D(h["usd"])), k_usd(h["usd"]))
        cl.run()
        a, u = f_tok(cl["amt"]), f_usd(cl["usd"])
        a_ab = ab(cl.rounded("amt"))
        reply = rng.choice([
            f"You have {a_ab}{a} {sym} on {DISPLAY[h['chain']]}, worth {u}.",
            f"{cap(a_ab)}{a} {sym} on {DISPLAY[h['chain']]}, worth {u} right now.",
            f"You hold {a_ab}{a} {sym} on {DISPLAY[h['chain']]} ({u}).",
            f"Your {sym} on {DISPLAY[h['chain']]} is {a_ab}{a}, which is {u}.",
        ])
    else:
        h1, h2 = mine[0], mine[1]
        e = f"round({h1['amount']}+{h2['amount']},{k_tok(sym, D(h1['amount']) + D(h2['amount']), 4)})"
        cl.expr("tot", e, src=None, k=k_tok(sym, D(h1['amount']) + D(h2['amount']), 4))
        cl.rnd("a1", h1["amount"], k_tok(sym, h1["amount"], 4))
        cl.rnd("a2", h2["amount"], k_tok(sym, h2["amount"], 4))
        cl.run()
        c1, c2 = DISPLAY[h1["chain"]], DISPLAY[h2["chain"]]
        reply = rng.choice([
            f"You hold about {f_tok(cl['tot'])} {sym} in total: {f_tok(cl['a1'])} on {c1} and {f_tok(cl['a2'])} on {c2}.",
            f"About {f_tok(cl['tot'])} {sym} across {c1} and {c2}, with {f_tok(cl['a1'])} on {c1} and {f_tok(cl['a2'])} on {c2}.",
        ])
    if r.get("unreachable"):
        reply += " " + unreach_text(rng, d)
    c.say(reply)
    cl.commit()
    return c.ex("long_balance", spec)


def top_items(rows, n):
    return rows[:n]


def sc_balance_wallet(rng, ev, more=False):
    k = rng.choices([1, 2, 3, 4, 5, 6, 7, 8, 9], weights=[2, 5, 5, 4, 6, 6, 4, 3, 1])[0] if not more else rng.choice([5, 6, 7, 8])
    unreach = rng.random() < 0.14
    multi = rng.random() < 0.10
    rows, px = make_holdings(rng, k)
    if len(rows) < k:
        raise Retry()
    d = base_world(rng, rows, unreachable=unreach, multi=multi)
    w = RBWorld(d)
    c = Conv(rng, w)
    c.user(finish(rng, pick(rng, A2_POOL, ev)))
    r = c.call(("get_balances", {}))
    hs = r["holdings"]
    n = len(hs)
    shown = hs[:3] if n >= 5 else hs
    cl = Calc(c)
    total_expr = "+".join(plain(D(h["usd"])) for h in hs) if n > 1 else plain(D(hs[0]["usd"]))
    tot_val = sum(D(h["usd"]) for h in hs)
    if n > 1:
        cl.expr("total", f"round({total_expr},{k_usd(tot_val)})", src=None, k=k_usd(tot_val))
    else:
        cl.rnd("total", plain(D(hs[0]["usd"])), k_usd(hs[0]["usd"]))
    for i, h in enumerate(shown):
        cl.rnd(f"a{i}", h["amount"], k_tok(h["symbol"], h["amount"], rng.choice([3, 4])))
        cl.rnd(f"u{i}", plain(D(h["usd"])), k_usd(h["usd"]))
    cl.run()

    def item(i, h):
        return f"{f_tok(cl[f'a{i}'])} {h['symbol']} on {DISPLAY[h['chain']]} ({f_usd(cl[f'u{i}'])})"
    items = [item(i, h) for i, h in enumerate(shown)]
    lead_ab = "about " if cl.rounded("total") else ""
    lead = rng.choice([f"You have {lead_ab}{f_usd(cl['total'])} in your wallet.", f"Your wallet holds {lead_ab}{f_usd(cl['total'])} in total.",
                       f"Total: {lead_ab}{f_usd(cl['total'])}."])
    if len(items) == 1:
        body = f"That is {items[0]}."
    elif n >= 5:
        body = f"The biggest are {items[0]}, {items[1]} and {items[2]}, and {n - 3} smaller."
    elif len(items) == 2:
        body = f"That is {items[0]} and {items[1]}."
    else:
        body = f"That is " + ", ".join(items[:-1]) + f" and {items[-1]}."
    reply = lead + " " + body
    if r.get("unreachable"):
        reply += " " + unreach_text(rng, d)
    if r.get("wallet"):
        o = d["other_wallets"]
        reply += f" This is your Main wallet; {o[0]} and {o[1]} are not included."
    c.say(reply)
    cl.commit()
    spec = {"shape": "long_balance", "scenario": "wallet", "n_holdings": n, "counts": [n - 3] if n >= 5 else [],
            "top3": n >= 5, "table_exempt": True}
    if more:
        return c, hs, shown, spec
    return c.ex("long_balance", spec)


def sc_portfolio(rng, ev):
    places = [("Wallet", None), ("Spot tokens", None), ("Aave", None), ("Hyperliquid", None)]
    chosen = rng.sample([p for p, _ in places], rng.choice([2, 3, 4]))
    by = []
    for p in chosen:
        v = rng.choice([rng.uniform(5, 400), rng.uniform(400, 9000)])
        by.append({"place": p, "usd": jn(round(v, 6 if rng.random() < 0.5 else 2))})
    by.sort(key=lambda x: -x["usd"])
    total_val = sum(D(x["usd"]) for x in by)
    unreach = rng.random() < 0.13
    top = []
    for x in by[:3]:
        what = {"Wallet": "USDC on Base", "Spot tokens": "WIF", "Aave": "USDC on Base", "Hyperliquid": "Account value"}[x["place"]]
        top.append({"place": x["place"], "usd": x["usd"], "what": what})
    port = {"by_place": by, "top": top, "total_usd": jn(round(float(total_val), 4)), "wallet": "Main"}
    if unreach:
        port["unreachable"] = [rng.choice(["bnb", "polygon", "ethereum"])]
    d = {"portfolio": port, "balances": [], "prices": {}}
    if unreach:
        d["unreachable"] = port["unreachable"]
    w = RBWorld(d)
    c = Conv(rng, w)
    c.user(finish(rng, pick(rng, A3_POOL, ev)))
    r = c.call(("get_portfolio", {}))
    cl = Calc(c)
    cl.rnd("total", plain(D(r["total_usd"])), k_usd(r["total_usd"]))
    for i, x in enumerate(by[:2]):
        cl.rnd(f"p{i}", plain(D(x["usd"])), k_usd(x["usd"]))
    cl.run()
    where = {"Wallet": "in your wallet", "Spot tokens": "in spot tokens", "Aave": "in Aave", "Hyperliquid": "on Hyperliquid"}
    ab_ = "about " if cl.rounded("total") else ""
    if len(by) == 1:
        reply = f"Your total is {ab_}{f_usd(cl['total'])}, all {where[by[0]['place']]}."
    else:
        reply = rng.choice([f"Your total is {ab_}{f_usd(cl['total'])}: {f_usd(cl['p0'])} {where[by[0]['place']]} and {f_usd(cl['p1'])} {where[by[1]['place']]}.",
                            f"{cap(ab_)}{f_usd(cl['total'])} in total. The most is {f_usd(cl['p0'])} {where[by[0]['place']]}, then {f_usd(cl['p1'])} {where[by[1]['place']]}."])
        if len(by) > 2:
            reply = reply[:-1] + f", and {len(by) - 2} smaller."
    if r.get("unreachable"):
        reply += " " + unreach_text(rng, d)
    c.say(reply)
    cl.commit()
    spec = {"shape": "long_balance", "scenario": "portfolio", "counts": [len(by) - 2] if len(by) > 2 else [], "table_exempt": True}
    return c.ex("long_balance", spec)


def sc_send_all(rng, ev):
    sym = rng.choices(["ETH", "USDC", "SOL", "BNB", "LINK", "USDT", "ARB"], weights=[5, 4, 2, 1, 1, 1, 1])[0]
    chain = rng.choice(HOLD[sym])
    for _ in range(30):
        rows, px = make_holdings(rng, rng.choice([0, 1, 2]), must=[(sym, chain)], sig15=True)
        mine = [r for r in rows if r["symbol"] == sym]
        if len(mine) == 1 and float_exact(mine[0]["amount"]) and D(mine[0]["amount"]) > 0:
            break
    else:
        raise Retry()
    # a token on one chain only for this wallet
    rows = [r for r in rows if r["symbol"] != sym or r["chain"] == chain]
    d = base_world(rng, rows)
    w = RBWorld(d)
    c = Conv(rng, w)
    addr = H.sol_addr(rng) if chain == "solana" else H.evm_addr(rng)
    t = tk(rng, sym)
    c.user(finish(rng, pick(rng, A4_POOL, ev).format(t=t, a=addr), protect=(addr, t)))
    r = c.call(("get_balances", {}))
    h = [x for x in r["holdings"] if x["symbol"] == sym][0]
    amount = float(h["amount"])
    amount = int(amount) if amount == int(amount) else amount
    pr = c.call(("prepare_send", {"to_address": addr, "amount": amount, "symbol": sym}))
    if pr.get("status") != "review_opened":
        raise Retry()
    reply = rng.choice([
        f"I've prepared sending your full {sym} balance, {pr['amount']}, on {pr['network']}. It's waiting for your {c.bio}, and nothing has moved yet.",
        f"Prepared: {pr['amount']} on {pr['network']}, all of your {sym}. Approve it with {c.bio} on the review screen; nothing moves until then.",
        f"The review for {pr['amount']} on {pr['network']} is open. That is your whole {sym} balance, and it only goes after you approve with {c.bio}.",
    ])
    c.say(reply)
    spec = {"shape": "long_balance", "scenario": "send_all", "counts": [], "exact_amount": h["amount"], "address": addr,
            "table_exempt": True}
    return c.ex("long_balance", spec)


# ------------------------------------------------------------------ B. quotes
Q_TOK = ["ETH", "SOL", "BNB", "NEAR", "LINK", "UNI", "AAVE", "ARB", "POL", "USDC", "USDT"]
ROUTE_TOKENS = {"ETH": ["ethereum", "base", "arbitrum"], "USDC": ["ethereum", "base", "arbitrum", "bnb", "polygon", "solana"],
                "USDT": ["ethereum", "arbitrum", "bnb", "solana"], "SOL": ["solana"], "BNB": ["bnb"], "NEAR": ["near"],
                "POL": ["polygon"], "LINK": ["ethereum"], "UNI": ["ethereum"], "AAVE": ["ethereum"], "ARB": ["arbitrum"]}
AMOUNTS = {"USDC": [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 120, 150, 200, 250, 300, 500, 750, 1000, 1500, 2000, 3000, 12.5, 45.5],
           "ETH": [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 0.0125], "SOL": [0.5, 1, 2, 5, 10, 12.5, 20, 50],
           "BNB": [0.1, 0.25, 0.5, 1, 2, 5], "NEAR": [10, 25, 50, 100, 250, 500], "POL": [50, 100, 250, 500, 1000],
           "LINK": [5, 10, 20, 50], "UNI": [5, 10, 25, 40], "AAVE": [0.1, 0.5, 1, 2], "ARB": [50, 100, 500, 1000]}
AMOUNTS["USDT"] = AMOUNTS["USDC"]


def raw_out(rng, usd_out, price, dec):
    a = (D(usd_out) / D(price)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    # nudge the tail so it looks like a raw value
    tail = Decimal(rng.randint(1, 999)) * Decimal(1).scaleb(-dec)
    return plain(a + tail)


def make_quote(rng, w, fs, fc, ts, tc, amount, mode):
    """Builds and stores a get_quote result in the world snapshot. mode app (3 significant digits) or raw (old build)."""
    px = w.d["prices"]
    for s in (fs, ts):
        if s not in px:
            px[s] = {"price": rand_price(rng, s) if s in PRICE_BAND else 1, "change": round(rng.uniform(-6, 6), 2)}
    pf, pt = px[fs]["price"], px[ts]["price"]
    usd_in = round(float(D(amount) * D(pf)), 2)
    if usd_in < 2:
        raise Retry()
    bps = rng.choice([10, 15, 20, 25, 30, 35, 40, 45, 50])
    cost = max(0.05, round(usd_in * (bps + rng.randint(5, 40)) / 10000, 2))
    cost = round(cost, 2)
    usd_out = round(usd_in - cost, 2)
    if mode == "raw":
        out = raw_out(rng, usd_out, pt, rng.choice([18, 18, 24]))
    else:
        out = approx(D(usd_out) / D(pt))
    secs = rng.randint(6, 55)
    q = {"amount_in_usd": jn(usd_in), "amount_out": out, "amount_out_usd": jn(usd_out), "fee_bps": bps,
         "from": f"{plain(amount)} {fs} on {DISPLAY[fc]}", "route": "NEAR Intents", "seconds": secs,
         "signature_verified": True, "to": f"about {out} {ts} on {DISPLAY[tc]}", "total_cost_usd": jn(cost)}
    key = json.dumps([fs, fc, ts, tc, jn(amount)])
    w.d.setdefault("quotes", {})[key] = q
    return q


NATIVE1 = {s: cs[0] for s, cs in ROUTE_TOKENS.items() if len(cs) == 1}      # tokens that live on one chain only
Q_SWAP = [
    "how much {dst} for {src}?", "quote me {src} to {dst}", "what would I get for {src} into {dst}?", "{src} -> {dst} quote?",
    "how much would {src} turn into as {dst}", "price check: {src} to {dst}", "what's the rate if I swap {src} for {dst}",
    "get me a quote: {src} to {dst}", "just checking, how much {dst} would {src} buy me? don't do anything yet",
    "what do I get converting {src} to {dst}", "how much would I receive in {dst} if I sell {src}", "estimate {src} into {dst}",
    "no swap yet, just show me what {src} gets me in {dst}", "can you quote {src} for {dst}", "need a quote on {src} to {dst}",
    "how many {dst} would I get for {src}", "what's {src} in {dst} right now, quote only",
]
Q_BRIDGE = [
    "what would I get bridging {src} to {tc}?", "quote bridging {src} to {tc}", "how much arrives on {tc} if I bridge {src}?",
    "bridge quote: {src} -> {tc}", "what would it cost to move {src} over to {tc}? just a quote", "how much lands on {tc} if I send {src} across",
    "check the numbers on bridging {src} to {tc}", "if I bridge {src} to {tc} what do I end up with?", "{src} to {tc}, how much lands? quote only",
    "can you quote {src} to {tc}?", "get me a bridge quote for {src} going to {tc}",
]


def src_slot(rng, amt, fs, fc):
    f = tk(rng, fs)
    if NATIVE1.get(fs) == fc and rng.random() < 0.65:
        return f"{amt} {f}"
    fcw = cw(rng, fc)
    return rng.choice([f"{amt} {f} on {fcw}", f"{amt} {f} from {fcw}", f"{amt} {f} ({fcw})", f"{amt} of my {f} on {fcw}",
                       f"{amt} {f} sitting on {fcw}", f"{fcw} {amt} {f}"])


def dst_slot(rng, ts, tc):
    t = tk(rng, ts)
    if NATIVE1.get(ts) == tc and rng.random() < 0.7:
        return t
    tcw = cw(rng, tc)
    return rng.choice([f"{t} on {tcw}", f"{tcw} {t}", f"{t} ({tcw})", f"{t} over on {tcw}"])


def sc_quote(rng, ev):
    bridge = rng.random() < 0.38
    fs = rng.choice(Q_TOK)
    fc = rng.choice(ROUTE_TOKENS[fs])
    if bridge:
        ts = fs
        tcs = [c for c in ROUTE_TOKENS[ts] if c != fc]
        if not tcs:
            raise Retry()
        tc = rng.choice(tcs)
    else:
        ts = rng.choice([s for s in Q_TOK if s != fs])
        tc = rng.choice(ROUTE_TOKENS[ts])
        if (fs, fc) == (ts, tc):
            raise Retry()
    if fc not in CHAIN_WORDS or tc not in CHAIN_WORDS:
        raise Retry()
    amount = rng.choice(AMOUNTS[fs])
    mode = "raw" if rng.random() < 0.45 else "app"
    d = {"prices": {}, "balances": []}
    w = RBWorld(d)
    q = make_quote(rng, w, fs, fc, ts, tc, amount, mode)
    c = Conv(rng, w)
    a_txt = amt_text(rng, amount)
    src = src_slot(rng, a_txt, fs, fc)
    if bridge:
        text = pick(rng, Q_BRIDGE, ev).format(src=src, tc=cw(rng, tc))
    else:
        text = pick(rng, Q_SWAP, ev).format(src=src, dst=dst_slot(rng, ts, tc))
    c.user(finish(rng, text, protect=(src, a_txt)))
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": jn(amount)}
    r = c.call(("get_quote", args))
    if "error" in r:
        raise Retry()
    cl = Calc(c)
    cl.rnd("out", r["amount_out"], k_tok(ts, r["amount_out"], rng.choice([3, 4])))
    cl.rnd("cost", plain(D(r["total_cost_usd"])), k_usd(r["total_cost_usd"]))
    use_pct = rng.random() < 0.5
    if use_pct:
        cl.expr("pct", f"{r['fee_bps']}/100", src=str(r["fee_bps"]), k=2)
    cl.run()
    o, cost = f_tok(cl["out"]), f_usd(cl["cost"])
    fromtxt = f"{plain(amount)} {fs} on {DISPLAY[fc]}"
    if bridge:
        t1 = f"{fromtxt} would arrive as about {o} {ts} on {DISPLAY[tc]} through NEAR Intents."
    else:
        t1 = f"{fromtxt} gets you about {o} {ts} on {DISPLAY[tc]} through NEAR Intents."
    if use_pct:
        t2 = rng.choice([f"The fee is {f_pct(cl['pct'])} and the all-in cost is {cost}.",
                         f"Fee {f_pct(cl['pct'])}, all-in cost {cost}."])
    else:
        t2 = rng.choice([f"All-in cost is {cost}, ready in about {r['seconds']} seconds.",
                         f"It costs {cost} in total and takes about {r['seconds']} seconds.", f"Total cost is {cost}."])
    tail = rng.choice(["", "", " This is only a quote, nothing is prepared."])
    c.say(t1 + " " + t2 + tail)
    cl.commit()
    spec = {"shape": "quotes", "scenario": "bridge" if bridge else "swap", "mode": mode, "counts": [r["seconds"]],
            "route": "NEAR Intents", "must_contain": ["NEAR Intents"], "forbid_tools": ["prepare_swap"], "table_exempt": True}
    return c.ex("quotes", spec)


# ------------------------------------------------------------------ C. funding and yield maths
PERP_COINS = ["BTC", "ETH", "SOL", "HYPE", "BNB", "DOGE", "AVAX", "LINK", "XRP", "NEAR"]
PERP_VOL = {"BTC": (900e6, 3.2e9), "ETH": (600e6, 2.4e9), "SOL": (150e6, 900e6), "HYPE": (80e6, 700e6), "BNB": (20e6, 160e6),
            "DOGE": (30e6, 300e6), "AVAX": (15e6, 120e6), "LINK": (10e6, 90e6), "XRP": (50e6, 400e6), "NEAR": (8e6, 70e6)}
MAXLEV = {"BTC": 40, "ETH": 25, "SOL": 20, "HYPE": 10, "BNB": 10, "DOGE": 10, "AVAX": 10, "LINK": 10, "XRP": 20, "NEAR": 10}


def rand_funding(rng):
    r = rng.random()
    if r < 0.3:
        return round(rng.choice([0.00125, 0.0025, 0.001, 0.0005, 0.00375, 0.00625]) * rng.choice([1, 1, -1]), 5)
    base = rng.uniform(0.0001, 0.009) * rng.choice([1, 1, 1, -1])
    return round(base, rng.choice([5, 6, 8, 10]))


def make_markets(rng, coins):
    out = []
    for cn in coins:
        price = rand_price(rng, cn)
        lo, hi = PERP_VOL[cn]
        vol = int(rng.uniform(lo, hi))
        if rng.random() < 0.4:
            vol = int(round(vol, -6))
        out.append({"funding_hourly_pct": rand_funding(rng), "max_leverage": MAXLEV[cn], "name": f"{cn}-PERP", "price": price,
                    "type": "perp", "volume_24h_usd": vol})
    return out


C1_POOL = [
    "what's {c} funding annualised?", "what is the yearly funding on {c}", "how much is {c} funding per year?", "{c} funding rate as an annual number",
    "annualise {c} funding for me", "what would {c} funding be over a year", "{c} perp funding apr?", "funding on {c}, yearly",
    "convert {c} hourly funding to a yearly rate", "how expensive is {c} funding over a year",
]
C1_DAY = [
    "what's {c} funding per day?", "how much is the {c} funding a day", "daily funding on {c}?", "{c} funding over a day in percent",
    "what does {c} funding come to per day",
]
C1_TWO = [
    "which has higher funding, {c} or {d}? yearly", "compare {c} and {d} funding annualised", "{c} vs {d} funding per year",
    "annual funding for {c} and {d}", "is {c} or {d} funding higher? give yearly numbers",
]
C2_POOL = [
    "I'm {side} ${size} on {c}, what does funding cost me a day?", "what's funding on a ${size} {side} in {c} per day",
    "if I hold a ${size} {c} {side}, what do I pay or get in funding daily", "daily funding for ${size} {side} {c}?",
    "${size} {c} {side}, funding per day and per year?", "how much funding for a {size} usd {c} {side} each day",
    "what will funding cost on my ${size} {c} {side}, per day", "funding impact of a ${size} {side} on {c} daily",
]
C3_BEST = [
    "if I put {amt} {t} in the best yield, how much do I earn a year?", "how much would {amt} {t} earn in a year at the top rate",
    "what does {amt} {t} make per year in the highest paying pool", "best {t} yield and what {amt} would earn me",
    "{amt} {t} in the top yield pool, yearly earnings?", "what could {amt} {t} earn in a year",
]
C3_AAVE = [
    "how much would {amt} {t} earn in Aave on {c} per year?", "if I deposit {amt} {t} in aave on {c}, what do I earn a year",
    "{amt} {t} into aave on {c}, yearly and monthly earnings?", "what's the yearly return on {amt} {t} in aave ({c})",
    "aave {c}: {amt} {t} would make how much in a year", "what would {amt} {t} earn on aave {c} each month",
]
C4_POOL = [
    "is {p1} better than {p2} for {t}?", "{p1} or {p2} for {t}, what's the difference in apy?", "compare {p1} and {p2} {t} yields",
    "how much more does {p1} pay than {p2} on {t}", "{p1} vs {p2} {t} apy gap?",
]
C5_POOL = [
    "the route fee says {bps} bps, what's that in percent?", "what is {bps} bps in %", "{bps} bps on ${amt}, how much is that?",
    "fee is {bps} bps, how much do I pay on ${amt}", "convert {bps} basis points to percent", "if a fee is {bps} bps what does it cost on ${amt}",
    "{bps} bps of ${amt} in dollars?", "how big is a {bps} bps fee on {amt} usd",
]
PROTOS = [("Aave v3", True), ("Morpho", False), ("Compound v3", False), ("Spark", False), ("Euler", False), ("Fluid", False),
          ("Pendle", False), ("Kamino", False)]


def make_pools(rng, asset, n, chain=None, must_aave=False):
    chains_disp = ["Ethereum", "Base", "Arbitrum", "Polygon", "Solana"]
    pools, used = [], set()
    if must_aave:
        ch = chain or rng.choice(["Ethereum", "Base", "Arbitrum"])
        pools.append(("Aave v3", ch))
        used.add(("Aave v3", ch))
    tries = 0
    while len(pools) < n and tries < 100:
        tries += 1
        p = rng.choice(PROTOS)[0]
        ch = chain or rng.choice(chains_disp)
        if (p, ch) in used:
            continue
        if p == "Kamino" and ch != "Solana":
            continue
        if p != "Kamino" and ch == "Solana":
            continue
        pools.append((p, ch))
        used.add((p, ch))
    rows = []
    for p, ch in pools:
        base = round(rng.uniform(1.2, 9.5), rng.choice([2, 2, 4, 6]))
        rew = rng.choice([0, 0, 0.31, 0.5, 1.12])
        apy = round(base + rew, rng.choice([2, 2, 4]))
        tvl = int(rng.choice([rng.uniform(8e6, 90e6), rng.uniform(90e6, 900e6), rng.uniform(1.0e9, 3.9e9)]))
        if rng.random() < 0.5:
            tvl = int(round(tvl, -5))
        rows.append({"apy_30d_pct": round(apy - rng.uniform(-0.4, 0.4), 2), "apy_base_pct": base, "apy_pct": apy,
                     "apy_reward_pct": rew, "asset": asset, "can_deposit_in_hermit": (p == "Aave v3" and ch in ("Ethereum", "Base", "Arbitrum")),
                     "chain": ch, "il_risk": "no", "pool_meta": None, "protocol": p, "tvl_usd": tvl})
    return rows


def sc_funding(rng, ev):
    mode = rng.choices(["year", "day", "two"], weights=[14, 5, 6])[0]
    coins = rng.sample(PERP_COINS, 2 if mode == "two" else 1)
    mk = make_markets(rng, coins + ([rng.choice([x for x in PERP_COINS if x not in coins])] if rng.random() < 0.4 and mode != "two" else []))
    d = {"markets": mk, "prices": {}, "balances": []}
    w = RBWorld(d)
    c = Conv(rng, w)
    cn = coins[0]
    ct = tk(rng, cn)
    if mode == "year":
        text = pick(rng, C1_POOL, ev).format(c=ct)
    elif mode == "day":
        text = pick(rng, C1_DAY, ev).format(c=ct)
    else:
        text = pick(rng, C1_TWO, ev).format(c=ct, d=tk(rng, coins[1]))
    c.user(finish(rng, text, protect=(ct,)))
    q = " ".join(coins)
    r = c.call(("search_markets", {"query": q if mode == "two" else cn + (" perp" if rng.random() < 0.3 else "")}))
    rows = {m["name"].split("-")[0]: m for m in r["markets"]}
    cl = Calc(c)
    for cn_ in coins:
        if cn_ not in rows:
            raise Retry()
        f = plain(D(rows[cn_]["funding_hourly_pct"]))
        if mode == "day":
            cl.expr(f"f_{cn_}", f"round({f}*24,2)", src=f, k=2)
        else:
            cl.expr(f"f_{cn_}", f"round({f}*24*365,2)", src=f, k=2)
    cl.run()

    def who(f):
        return "longs pay shorts" if D(f) > 0 else "shorts pay longs"
    if mode == "year":
        x = cl[f"f_{cn}"]
        reply = rng.choice([f"{cn} funding is about {f_pct(x)} a year, so {who(x)}.",
                            f"About {f_pct(x)} a year on {cn}; {who(x)}.",
                            f"Annualised, {cn} funding is about {f_pct(x)}, and {who(x)}."])
    elif mode == "day":
        x = cl[f"f_{cn}"]
        reply = rng.choice([f"{cn} funding comes to about {f_pct(x)} a day, so {who(x)}.", f"About {f_pct(x)} per day on {cn}; {who(x)}."])
    else:
        a, b = coins
        xa, xb = cl[f"f_{a}"], cl[f"f_{b}"]
        hi = a if D(xa) > D(xb) else b
        if D(xa) == D(xb):
            raise Retry()
        if D(xa) <= 0 and D(xb) <= 0:
            reply = f"{a} is about {f_pct(xa)} a year and {b} about {f_pct(xb)}. Both are negative, so shorts pay longs."
        elif D(xa) > 0 and D(xb) > 0:
            reply = f"{a} is about {f_pct(xa)} a year and {b} about {f_pct(xb)}. {hi} is higher; longs pay shorts on both."
        else:
            reply = f"{a} is about {f_pct(xa)} a year and {b} about {f_pct(xb)}. The sign differs: on {a} {who(xa)}, on {b} {who(xb)}."
    c.say(reply)
    cl.commit()
    return c.ex("funding_yield", {"shape": "funding_yield", "scenario": "funding_" + mode, "counts": [], "table_exempt": True,
                                  "forbid_tools": ["prepare_trade"]})


def sc_funding_cost(rng, ev):
    cn = rng.choice(PERP_COINS)
    mk = make_markets(rng, [cn])
    f = mk[0]["funding_hourly_pct"]
    d = {"markets": mk, "prices": {}, "balances": []}
    w = RBWorld(d)
    c = Conv(rng, w)
    size = rng.choice([500, 1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000, 12000, 20000, 25000, 50000])
    side = rng.choice(["long", "short"])
    st = f"{size:,}" if rng.random() < 0.3 else str(size)
    ct = tk(rng, cn)
    text = pick(rng, C2_POOL, ev).format(side=side, size=st, c=ct)
    c.user(finish(rng, text, protect=(st, ct)))
    r = c.call(("search_markets", {"query": cn}))
    row = [m for m in r["markets"] if m["name"].startswith(cn)][0]
    fs = plain(D(row["funding_hourly_pct"]))
    cl = Calc(c)
    cl.expr("day", f"round(abs({size}*{fs}/100*24),2)", src=fs, k=2)
    year_too = "year" in text.lower()
    if year_too:
        cl.expr("year", f"round(abs({size}*{fs}/100*24*365),2)", src=fs, k=2)
    cl.run()
    pays = (D(fs) > 0) == (side == "long")
    verb = "pays" if pays else "receives"
    body = f"A ${size:,} {cn} {side} {verb} about {f_usd(cl['day'])} a day in funding"
    if year_too:
        body += f", roughly {f_usd(cl['year'])} a year"
    body += rng.choice([".", " at the current rate."]) + " Funding changes every hour."
    c.say(body)
    cl.commit()
    return c.ex("funding_yield", {"shape": "funding_yield", "scenario": "funding_cost", "counts": [], "table_exempt": True,
                                  "forbid_tools": ["prepare_trade"]})


def sc_yield_earn(rng, ev):
    asset = rng.choice(["USDC", "USDC", "USDC", "USDT", "ETH"])
    aave_mode = rng.random() < 0.5
    chain = rng.choice(["Ethereum", "Base", "Arbitrum"])
    rows = make_pools(rng, asset, rng.choice([3, 4, 5]), must_aave=aave_mode or rng.random() < 0.5,
                      chain=None)
    # make sure the Aave row is on `chain` for the named case
    d = {"pools": rows, "prices": {}, "balances": []}
    w = RBWorld(d)
    c = Conv(rng, w)
    amts = {"USDC": [100, 250, 500, 1000, 1500, 2500, 5000, 10000], "USDT": [200, 500, 1000, 3000], "ETH": [0.5, 1, 2, 5, 10]}[asset]
    amount = rng.choice(amts)
    at = amt_text(rng, amount)
    t = tk(rng, asset)
    if aave_mode:
        aave_rows = [p for p in rows if p["protocol"] == "Aave v3" and p["chain"] == chain]
        if not aave_rows:
            for p in rows:
                if p["protocol"] == "Aave v3":
                    p["chain"] = chain
                    p["can_deposit_in_hermit"] = True
            if not [p for p in rows if p["protocol"] == "Aave v3"]:
                raise Retry()
        text = pick(rng, C3_AAVE, ev).format(amt=at, t=t, c=cw(rng, chain.lower()))
        args = {"asset": asset, "chain": chain.lower()}
    else:
        text = pick(rng, C3_BEST, ev).format(amt=at, t=t)
        args = {"asset": asset}
    c.user(finish(rng, text, protect=(at, t)))
    r = c.call(("get_yields", args))
    pools = r["pools"]
    if not pools:
        raise Retry()
    row = [p for p in pools if p["protocol"] == "Aave v3" and p["chain"] == chain][0] if aave_mode else pools[0]
    if aave_mode and not row:
        raise Retry()
    apy = plain(D(row["apy_pct"]))
    cl = Calc(c)
    cl.rnd("apy", apy, 2)
    cl.expr("year", f"round({amount}*{apy}/100,{2 if asset != 'ETH' else 4})", src=apy, k=2)
    monthly = rng.random() < 0.5
    if monthly:
        cl.expr("month", f"round({amount}*{apy}/100/12,{2 if asset != 'ETH' else 4})", src=apy, k=2)
    cl.run()
    unit = lambda x: f_usd(x) if asset != "ETH" else f"{f_tok(x)} ETH"
    pool = f"{row['protocol']} on {row['chain']}"
    reply = f"{pool} pays {f_pct(cl['apy'])} APY on {asset}, so {comma(plain(amount))} {asset} would earn about {unit(cl['year'])} a year"
    reply += f", roughly {unit(cl['month'])} a month." if monthly else "."
    reply += " The rate changes over time."
    if not row["can_deposit_in_hermit"]:
        reply += " Hermit deposits only into Aave v3 on Ethereum, Base and Arbitrum; other pools are in Earn."
    c.say(reply)
    cl.commit()
    return c.ex("funding_yield", {"shape": "funding_yield", "scenario": "yield_earn", "counts": [], "table_exempt": True,
                                  "forbid_tools": ["prepare_deposit"]})


def sc_yield_compare(rng, ev):
    asset = rng.choice(["USDC", "USDT"])
    rows = make_pools(rng, asset, 5, must_aave=True)
    prots = {}
    for p in sorted(rows, key=lambda p: -D(p["apy_pct"])):
        prots.setdefault(p["protocol"], p)
    others = [n for n in prots if n != "Aave v3"]
    if len(rows) < 5 or "Aave v3" not in prots or not others:
        raise Retry()
    n1 = rng.choice(others)
    d = {"pools": rows, "prices": {}, "balances": []}
    w = RBWorld(d)
    c = Conv(rng, w)
    t = tk(rng, asset)
    text = pick(rng, C4_POOL, ev).format(p1=n1.lower() if rng.random() < .4 else n1, p2="aave" if rng.random() < .5 else "Aave", t=t)
    c.user(finish(rng, text, protect=(t,)))
    r = c.call(("get_yields", {"asset": asset}))
    pools = {}
    for p in r["pools"]:
        pools.setdefault(p["protocol"], p)
    if n1 not in pools or "Aave v3" not in pools:
        raise Retry()
    a1, a2 = plain(D(pools[n1]["apy_pct"])), plain(D(pools["Aave v3"]["apy_pct"]))
    if D(a1) == D(a2):
        raise Retry()
    cl = Calc(c)
    cl.rnd("a1", a1, 2)
    cl.rnd("a2", a2, 2)
    cl.expr("gap", f"round(abs({a1}-{a2}),2)", src=a1, k=2)
    cl.run()
    hi = n1 if D(a1) > D(a2) else "Aave v3"
    reply = (f"{n1} on {pools[n1]['chain']} pays {f_pct(cl['a1'])} APY and Aave v3 on {pools['Aave v3']['chain']} pays {f_pct(cl['a2'])}, "
             f"so {hi} is higher by {cl['gap']} points. Rates move, and Hermit deposits only into Aave v3 on Ethereum, Base and Arbitrum.")
    c.say(reply)
    cl.commit()
    return c.ex("funding_yield", {"shape": "funding_yield", "scenario": "yield_compare", "counts": [], "table_exempt": True,
                                  "forbid_tools": ["prepare_deposit"]})


def sc_bps(rng, ev):
    bps = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100])
    amount = rng.choice([100, 250, 500, 1000, 2500, 5000, 10000, 20000])
    w = RBWorld({"prices": {}, "balances": []})
    c = Conv(rng, w)
    tmpl = pick(rng, C5_POOL, ev)
    amt_s = f"{amount:,}" if rng.random() < 0.3 else str(amount)
    c.user(finish(rng, tmpl.format(bps=bps, amt=amt_s), protect=(amt_s,)))
    cl = Calc(c)
    cl.expr("pct", f"{bps}/100", src=str(bps), k=2)
    dollars = "$" in tmpl or "amt" in tmpl
    if "{amt}" in tmpl:
        cl.expr("usd", f"round({amount}*{bps}/10000,2)", src=str(bps), k=2)
    cl.run()
    if "{amt}" in tmpl:
        reply = rng.choice([f"{bps} bps is {f_pct(cl['pct'])}, so the fee on ${amount:,} is {f_usd(cl['usd'])}.",
                            f"The fee is {f_usd(cl['usd'])} on ${amount:,}. That is {bps} bps, or {f_pct(cl['pct'])}."])
    else:
        reply = rng.choice([f"{bps} bps is {f_pct(cl['pct'])}.", f"{bps} basis points equal {f_pct(cl['pct'])}."])
    c.say(reply)
    cl.commit()
    return c.ex("funding_yield", {"shape": "funding_yield", "scenario": "bps", "counts": [], "table_exempt": True})


# ------------------------------------------------------------------ D. attached files and pasted raw amounts
def file_msg(name, detail, body, question):
    return f"The user attached the file {name} ({detail}):\n<file>\n{body}\n</file>\n\n{question}"


def rand_date(rng, i=0):
    return f"2026-{rng.choice([9, 10, 10, 11]):02d}-{rng.randint(1, 28):02d}"


D1_Q = [
    "how much {a} did I {verb} in total?", "total {a} {verb}s?", "add up my {a} {verb}s", "what's the sum of the {a} {verb}s in this file",
    "how much {a} have I {verb}ed here, all together", "give me the total of {a} {verb}s", "sum all {a} {verb}s for me",
    "{a} {verb}s: what's the total", "how many {a} did I {verb} in all", "can you total the {a} {verb}s",
]
D1_BIG = [
    "what's my biggest {a} {verb}?", "which {a} {verb} was the largest", "biggest {a} {verb} in the file?", "largest {a} {verb} and when was it",
    "find the top {a} {verb}", "what was my largest {a} {verb}, and the date",
]


def sc_file_transfers(rng, ev, cut=False):
    asset = rng.choice(["ETH", "ETH", "SOL", "USDC", "BNB", "LINK", "NEAR"])
    other = rng.choice([a for a in ["ETH", "SOL", "USDC", "BNB", "LINK", "NEAR"] if a != asset])
    n_main = rng.randint(2, 5)
    verb = rng.choice(["deposit", "withdrawal"])
    mode = "big" if rng.random() < 0.28 else "sum"
    rows = []
    px = {s: rand_price(rng, s) for s in (asset, other)}

    def raw(sym):
        return raw_amount(rng, sym, rng.uniform(8, 900), px[sym] if sym not in STABLE else 1)
    used_dates = set()
    for i in range(n_main):
        rows.append((rand_date(rng), verb, asset, raw(asset)))
    for i in range(rng.randint(1, 4)):
        rows.append((rand_date(rng), rng.choice(["deposit", "withdrawal"]), asset, raw(asset)))
    for i in range(rng.randint(1, 3)):
        rows.append((rand_date(rng), rng.choice(["deposit", "withdrawal"]), other, raw(other)))
    rng.shuffle(rows)
    rows.sort(key=lambda r: r[0])
    main = [r for r in rows if r[1] == verb and r[2] == asset]
    if len(main) < 2:
        raise Retry()
    if mode == "big":
        amts = sorted((D(r[3]) for r in main), reverse=True)
        if amts[0] == amts[1]:
            raise Retry()
    if len({r[0] for r in rows}) != len(rows):
        raise Retry()
    total_n = len(rows) if not cut else rng.randint(len(rows) + 15, len(rows) + 90)
    body = "Date,Type,Asset,Amount\n" + "\n".join(f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in rows)
    if cut:
        detail = f"table; only the first {len(rows)} of {total_n} rows are included, the file is longer"
    else:
        detail = rng.choice(["table", f"spreadsheet, {len(rows)} rows"])
    name = rng.choice(["transfers.csv", "wallet_export.csv", "exchange-history.csv", "history.csv", "moves.xlsx", "ledger.csv"])
    if name.endswith(".xlsx") and detail == "table":
        detail = f"spreadsheet, {len(rows)} rows"
    pool = D1_BIG if mode == "big" else D1_Q
    q = finish(rng, pick(rng, pool, ev).format(a=tk(rng, asset), verb=verb), protect=(tk(rng, asset),))
    w = RBWorld({"prices": {}, "balances": []})
    c = Conv(rng, w)
    c.user(file_msg(name, detail, body, q))
    cl = Calc(c)
    terms = [r[3] for r in main]
    if mode == "sum":
        s = sum(D(t) for t in terms)
        k = k_tok(asset, s, 4)
        cl.expr("tot", f"round({'+'.join(terms)},{k})", src=None, k=k)
    else:
        big = max(main, key=lambda r: D(r[3]))
        cl.rnd("big", big[3], k_tok(asset, big[3], 4))
    cl.run()
    verbs = verb + "s" if verb == "deposit" else "withdrawals"
    past = "deposited" if verb == "deposit" else "withdrew"
    if mode == "sum":
        reply = f"You {past} about {f_tok(cl['tot'])} {asset} in total, over {len(main)} {verbs}."
        counts = [len(main)]
    else:
        reply = f"Your largest {asset} {verb} was about {f_tok(cl['big'])} {asset}, on {big[0]}."
        counts = []
    if cut:
        what = "total" if mode == "sum" else "largest one"
        reply += f" That covers only the {len(rows)} rows I can see; the file has {total_n}, so the full {what} may be different."
    c.say(reply)
    cl.commit()
    spec = {"shape": "file_amounts", "scenario": ("cut_" if cut else "") + "transfers_" + mode, "counts": counts,
            "cut": cut, "calc_terms": terms if mode == "sum" else [big[3]], "max_words": 110, "table_exempt": True}
    return c.ex("file_amounts", spec)


SPEND = [("Apple developer", 99, 99), ("Rent", 700, 1800), ("Groceries", 40, 220), ("Uber", 8, 45), ("Netflix", 9, 20),
         ("Coffee beans", 12, 30), ("Flights", 150, 900), ("Hotel", 90, 600), ("Phone bill", 25, 80), ("Gym", 20, 60),
         ("Cloud hosting", 15, 140), ("Domain names", 10, 60), ("Insurance", 60, 250), ("Dinner out", 25, 140),
         ("Books", 12, 70), ("Train pass", 30, 160), ("Electricity", 40, 150), ("Software licence", 30, 300)]
SPEND_CAT = {"Apple developer": "Dev", "Rent": "Home", "Groceries": "Food", "Uber": "Travel", "Netflix": "Fun", "Coffee beans": "Food",
             "Flights": "Travel", "Hotel": "Travel", "Phone bill": "Bills", "Gym": "Fun", "Cloud hosting": "Dev", "Domain names": "Dev",
             "Insurance": "Bills", "Dinner out": "Food", "Books": "Fun", "Train pass": "Travel", "Electricity": "Bills", "Software licence": "Dev"}
D4_Q = [
    "what did I spend the most on, and what's the total", "total spend and the biggest item?", "how much did I spend overall and what was the biggest thing",
    "biggest expense and the grand total pls", "sum it up: total and top item", "what's the total here, and which line is the largest",
    "can you tell me the total and my largest expense", "what was my biggest cost and how much in total",
]


def sc_file_spend(rng, ev):
    items = rng.sample(SPEND, rng.randint(5, 10))
    cut = rng.random() < 0.2
    rows = []
    for name, lo, hi in items:
        v = round(rng.uniform(lo, hi), 2)
        r = rng.random()
        if r < 0.45:
            s = plain(D(repr(v)) + Decimal(rng.randint(1, 9)) * Decimal("1E-14"))
        elif r < 0.85:
            s = plain(D(repr(v)) + Decimal(rng.randint(1, 9999)) / Decimal(10 ** rng.choice([4, 5, 6])))
        else:
            s = plain(D(repr(v)))
        rows.append((rand_date(rng), name, SPEND_CAT[name], s))
    rows.sort(key=lambda r: r[0])
    if len({r[0] for r in rows}) != len(rows):
        raise Retry()
    amts = sorted((D(r[3]) for r in rows), reverse=True)
    if amts[0] == amts[1] or len({plain(D(r[3])) for r in rows}) != len(rows):
        raise Retry()
    body = "Date,Item,Category,Amount\n" + "\n".join(",".join(r) for r in rows)
    total_n = len(rows) if not cut else rng.randint(len(rows) + 10, len(rows) + 70)
    if cut:
        detail = f"spreadsheet, {total_n} rows; only the first {len(rows)} of {total_n} rows are included, the file is longer"
    else:
        detail = rng.choice([f"spreadsheet, {len(rows)} rows", "table"])
    name = rng.choice(["expenses.xlsx", "spending.csv", "budget.xlsx", "october-costs.csv", "bills.csv"])
    w = RBWorld({"prices": {}, "balances": []})
    c = Conv(rng, w)
    c.user(file_msg(name, detail, body, finish(rng, pick(rng, D4_Q, ev))))
    big = max(rows, key=lambda r: D(r[3]))
    tot = sum(D(r[3]) for r in rows)
    cl = Calc(c)
    cl.expr("tot", f"round({'+'.join(r[3] for r in rows)},2)", src=None, k=2)
    cl.rnd("big", big[3], 2)
    cl.run()
    reply = f"{'In the rows I can see, you' if cut else 'You'} spent {f_usd(cl['tot'])} in total. The biggest was {big[1]} at {f_usd(cl['big'])}."
    if cut:
        reply += f" The file has {total_n} rows and I only see {len(rows)}, so the real total is higher or different."
    c.say(reply)
    cl.commit()
    spec = {"shape": "file_amounts", "scenario": ("cut_" if cut else "") + "spend", "counts": [],
            "cut": cut, "calc_terms": [r[3] for r in rows], "max_words": 110, "table_exempt": True}
    return c.ex("file_amounts", spec)


D3_ONE = [
    "the dapp says \"You will receive {a} {S}\". what is that in normal numbers?", "my old wallet app shows a balance of {a} {S}, how much is that really",
    "etherscan shows {a} {S} for my address. can you round that for me", "it says {a} {S} on the swap page, what does that mean",
    "this page prints {a} {S} as my balance, make it readable", "what is {a} {S} rounded sensibly?",
    "dapp output: {a} {S}. is that a normal amount, what is it roughly", "some site says I have {a} {S}, convert that to something I can read",
    "the confirm popup says {a} {S}, how much is that", "can you tidy {a} {S} for me",
]
D3_TWO = [
    "the swap page says I pay {a} {S} and get {b} {T}. what are those in plain numbers?", "dapp: input {a} {S}, output {b} {T}. round both for me",
    "it shows {a} {S} going in and {b} {T} coming out, what are the real amounts", "can you make sense of {a} {S} -> {b} {T}",
    "old build gave me {a} {S} and {b} {T}, round them sensibly", "pasted from the app: in {a} {S}, out {b} {T}. readable numbers please",
]


def sc_pasted(rng, ev):
    two = rng.random() < 0.4
    syms = [s for s in ["ETH", "SOL", "BNB", "NEAR", "LINK", "USDC", "UNI", "AAVE", "ARB"]]
    s1 = rng.choice(syms)
    s2 = rng.choice([s for s in syms if s != s1])

    def val(sym):
        p = rand_price(rng, sym) if sym in PRICE_BAND else 1
        return raw_amount(rng, sym, rng.uniform(5, 2500), p, dec=rng.choice([18, 18, 24]) if sym not in STABLE else rng.choice([18, 24]))
    a = val(s1)
    b = val(s2) if two else None
    w = RBWorld({"prices": {}, "balances": []})
    c = Conv(rng, w)
    if two:
        text = pick(rng, D3_TWO, ev).format(a=a, S=s1, b=b, T=s2)
    else:
        text = pick(rng, D3_ONE, ev).format(a=a, S=tk(rng, s1) if rng.random() < .2 else s1)
    c.user(finish(rng, text, protect=(a, b or "")))
    cl = Calc(c)
    cl.rnd("a", a, k_tok(s1, a, rng.choice([3, 4])), force=True)
    if two:
        cl.rnd("b", b, k_tok(s2, b, rng.choice([3, 4])), force=True)
    cl.run()
    if two:
        reply = rng.choice([f"That is about {f_tok(cl['a'])} {s1} going in and about {f_tok(cl['b'])} {s2} coming out.",
                            f"You pay about {f_tok(cl['a'])} {s1} and get about {f_tok(cl['b'])} {s2}."])
    else:
        reply = rng.choice([f"That is about {f_tok(cl['a'])} {s1}.", f"Roughly {f_tok(cl['a'])} {s1}; the long tail of digits is just precision.",
                            f"It is about {f_tok(cl['a'])} {s1}."])
    c.say(reply)
    cl.commit()
    return c.ex("file_amounts", {"shape": "file_amounts", "scenario": "pasted_" + ("two" if two else "one"), "counts": [],
                                 "max_words": 110, "table_exempt": True})


# ------------------------------------------------------------------ E. positions and exact digits
def make_hl(rng, npos, coins=None):
    coins = coins or rng.sample(PERP_COINS, npos)
    positions, prices, upnl_sum = [], {}, Decimal(0)
    for cn in coins:
        mark = rand_price(rng, cn)
        side = rng.choice(["long", "short"])
        entry = round(mark * rng.uniform(0.92, 1.08), 2 if mark >= 10 else 4)
        lev = rng.choice([2, 3, 4, 5, 8, 10])
        notional = rng.choice([rng.uniform(150, 1500), rng.uniform(1500, 25000)])
        size = D(notional) / D(entry)
        size = float(size.quantize(Decimal(1).scaleb(-rng.choice([4, 6, 9])), rounding=ROUND_HALF_UP))
        sign = 1 if side == "long" else -1
        pnl = round(sign * (mark - entry) * size, rng.choice([2, 4, 6]))
        liq = entry * (1 - sign / lev * 0.9)
        liq = round(liq, rng.choice([2, 4])) if rng.random() > 0.06 else None
        positions.append({"coin": cn, "entry": entry, "leverage": lev, "liquidation": liq, "side": side, "size": size,
                          "unrealised_pnl": pnl})
        prices[cn] = {"price": mark, "change": round(rng.uniform(-5, 5), 2)}
        upnl_sum += D(pnl)
    acct = round(rng.uniform(200, 20000), rng.choice([2, 4, 6]))
    used = round(acct * rng.uniform(0.1, 0.5), rng.choice([2, 4]))
    hl = {"account_value": acct, "available_usdc": round(acct - used, 4), "margin_used": used, "open_orders": [], "positions": positions}
    return hl, prices


E1_ALL = [
    "how are my positions doing?", "show my hyperliquid positions", "what's my pnl right now", "am I in profit?", "how close am I to liquidation",
    "what positions do I have open", "check my perps", "how's my trading account", "what's my open pnl and liquidation levels",
    "give me a rundown of my open trades", "my positions?", "how are my trades looking", "any of my positions in trouble?",
]
E1_ONE = [
    "how's my {c} {side} doing?", "what's the pnl on my {c}", "where's my {c} liquidation price", "show my {c} position",
    "how close is my {c} {side} to liquidation", "am I up or down on {c}", "what's my {c} entry and pnl", "status of my {c} trade?",
    "{c} position: pnl and liq?", "how is my {c} perp doing",
]


def sc_positions(rng, ev):
    npos = rng.choices([1, 2, 3, 4], weights=[3, 4, 3, 1])[0]
    hl, prices = make_hl(rng, npos)
    d = {"hl": hl, "prices": prices, "balances": []}
    w = RBWorld(d)
    c = Conv(rng, w)
    one = rng.random() < 0.4
    pos = hl["positions"]
    if one:
        p = rng.choice(pos)
        c.user(finish(rng, pick(rng, E1_ONE, ev).format(c=tk(rng, p["coin"]), side=p["side"]), protect=(tk(rng, p["coin"]),)))
    else:
        c.user(finish(rng, pick(rng, E1_ALL, ev)))
    r = c.call(("hyperliquid_account", {}))
    ps = r["positions"]
    if one:
        ps = [x for x in ps if x["coin"] == p["coin"]]
    shown = ps[:3]
    cl = Calc(c)
    for i, x in enumerate(shown):
        cl.rnd(f"pnl{i}", plain(D(x["unrealised_pnl"])), k_usd(x["unrealised_pnl"]))
        if x["liquidation"] is not None:
            cl.rnd(f"liq{i}", plain(D(x["liquidation"])), k_usd(x["liquidation"]))
        if one:
            cl.rnd(f"ent{i}", plain(D(x["entry"])), k_usd(x["entry"]))
    if not one:
        cl.rnd("acct", plain(D(r["account_value"])), k_usd(r["account_value"]))
    cl.run()
    lines = []
    for i, x in enumerate(shown):
        s = f"{x['coin']} {x['side']} {x['leverage']}x: P&L {f_usd(cl[f'pnl{i}'], sign=True)}"
        if one:
            s += f", entry {f_usd(cl[f'ent{i}'])}"
        s += f", liquidation {f_usd(cl[f'liq{i}'])}" if x["liquidation"] is not None else ", no liquidation price"
        lines.append(s)
    counts = []
    if one:
        reply = f"Your {lines[0]}."
    else:
        counts = [len(ps)]
        nm = "position" if len(ps) == 1 else "positions"
        lead = rng.choice([f"You have {len(ps)} open {nm}, and the account is worth {f_usd(cl['acct'])}.",
                           f"Account value is {f_usd(cl['acct'])} with {len(ps)} open {nm}."])
        reply = lead + " " + ". ".join(lines) + "."
        if len(ps) > 3:
            reply = reply[:-1] + f", and {len(ps) - 3} more."
            counts.append(len(ps) - 3)
    c.say(reply)
    cl.commit()
    return c.ex("positions", {"shape": "positions", "scenario": "view_one" if one else "view_all", "counts": counts,
                              "max_words": 110, "table_exempt": True, "forbid_tools": ["prepare_close", "prepare_trade", "set_tp_sl"]})


E2_POOL = [
    "how are my spot tokens doing?", "what's my pnl on my solana tokens", "show my spot holdings with profit", "how are my coins I bought doing",
    "am I up on my spot bags?", "check my spot positions", "what's my solana token pnl", "how are my memecoins doing",
]
SPOT_SYMS = ["WIF", "BONK", "JUP", "PENGU", "POPCAT", "FARTCOIN", "RAY", "ORCA"]


def sc_spot(rng, ev):
    n = rng.choice([1, 2, 2, 3])
    syms = rng.sample(SPOT_SYMS, n)
    spot = []
    for s in syms:
        price = round(rng.uniform(0.02, 3.5), rng.choice([3, 4]))
        amt_v = rng.uniform(2, 9000)
        amt = plain(D(repr(round(amt_v, 9))) + Decimal(rng.randint(1, 999)) / Decimal(10 ** 9))
        usd = round(float(D(amt) * D(price)), rng.choice([2, 6]))
        avg = round(price * rng.uniform(0.7, 1.4), 5) if rng.random() > 0.15 else None
        pnl = round((price / avg - 1) * 100, rng.choice([2, 6])) if avg else None
        spot.append({"amount": amt, "avg_cost": avg, "frozen": False, "mint": H.sol_addr(rng), "pnl_pct": pnl, "symbol": s, "usd": usd})
    spot.sort(key=lambda x: -x["usd"])
    p = {"spot": spot, "wallet": "Main"}
    w = RBWorld({"positions": p, "prices": {}, "balances": []})
    c = Conv(rng, w)
    c.user(finish(rng, pick(rng, E2_POOL, ev)))
    r = c.call(("get_positions", {"kind": "spot"}))
    rows = r["spot"]
    cl = Calc(c)
    for i, x in enumerate(rows):
        cl.rnd(f"a{i}", x["amount"], k_tok(x["symbol"], x["amount"], 4))
        cl.rnd(f"u{i}", plain(D(x["usd"])), k_usd(x["usd"]))
        if x["pnl_pct"] is not None:
            cl.rnd(f"p{i}", plain(D(x["pnl_pct"])), 2)
    cl.run()
    parts = []
    for i, x in enumerate(rows):
        s = f"{x['symbol']}: {f_tok(cl[f'a{i}'])} tokens worth {f_usd(cl[f'u{i}'])}"
        s += f", {f_pct(cl[f'p{i}'], sign=True)} on your cost" if x["pnl_pct"] is not None else ", no cost basis recorded"
        parts.append(s)
    c.say("; ".join(parts) + ".")
    cl.commit()
    return c.ex("positions", {"shape": "positions", "scenario": "spot", "counts": [], "max_words": 110, "table_exempt": True,
                              "forbid_tools": ["prepare_sell"]})


E3_SL = [
    "set a stop loss at {sl} on my {c}", "put a stop on {c} at {sl}", "{c}: stop loss {sl}", "add a stop at {sl} to my {c} position",
    "stop loss for {c} at {sl} please", "I want a stop at {sl} on my {c}", "protect my {c} with a stop at {sl}",
]
E3_TP = [
    "take profit at {tp} for my {c}", "put a take profit on {c} at {tp}", "{c}: tp {tp}", "add a take profit of {tp} to my {c} position",
    "set my {c} take profit to {tp}",
]
E3_BOTH = [
    "stop {sl} and take profit {tp} on my {c}", "put a stop at {sl} and a tp at {tp} on {c}", "{c}: stop loss {sl}, take profit {tp}",
    "add stop {sl} / tp {tp} to my {c} position", "bracket my {c}: sl {sl} tp {tp}",
]


def typed_price(rng, ref, below):
    """a trigger price typed with few decimals, on the right side of ref."""
    for _ in range(40):
        f = rng.uniform(0.03, 0.12)
        v = ref * (1 - f if below else 1 + f)
        dp = 4 if ref < 1 else (3 if ref < 10 else rng.choice([2, 3]) if ref < 1000 else rng.choice([0, 1, 2]))
        s = f"{v:.{dp}f}"
        if "." in s and s.endswith("0"):
            continue
        if float(s) <= 0 or (below and float(s) >= ref) or (not below and float(s) <= ref):
            continue
        return s
    raise Retry()


def sc_tpsl(rng, ev):
    cn = rng.choice(["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "LINK", "DOGE", "NEAR"])
    hl, prices = make_hl(rng, 1, coins=[cn])
    pos = hl["positions"][0]
    mark = prices[cn]["price"]
    long_ = pos["side"] == "long"
    which = rng.choice(["sl", "sl", "tp", "both"])
    sl = typed_price(rng, mark, long_) if which in ("sl", "both") else None
    tp = typed_price(rng, mark, not long_) if which in ("tp", "both") else None
    d = {"hl": hl, "prices": prices, "balances": []}
    w = RBWorld(d)
    c = Conv(rng, w)
    pool = {"sl": E3_SL, "tp": E3_TP, "both": E3_BOTH}[which]
    text = pick(rng, pool, ev).format(c=tk(rng, cn), sl=sl or "", tp=tp or "")
    c.user(finish(rng, text, protect=tuple(x for x in (sl, tp, tk(rng, cn)) if x)))
    args = {"coin": cn}
    if sl is not None:
        args["stop_loss"] = float(sl)
    if tp is not None:
        args["take_profit"] = float(tp)
    r = c.call(("set_tp_sl", args))
    if r.get("status") != "review_opened":
        raise Retry()

    def usd_exact(s):
        return "$" + comma(plain(D(s)))
    bits = []
    if sl is not None:
        bits.append(f"a stop loss at {usd_exact(sl)}")
    if tp is not None:
        bits.append(f"a take profit at {usd_exact(tp)}")
    what = " and ".join(bits)
    pa, pb = r["position"].split(", ", 1)
    pos_txt = f"{pa} ({pb})"
    reply = rng.choice([
        f"I've prepared {what} on your {pos_txt}. It's waiting for your {c.bio}, and nothing is set until you approve.",
        f"Prepared: {what} for your {pos_txt}. Approve it with {c.bio} on the review screen; nothing changes until then.",
    ])
    c.say(reply)
    spec = {"shape": "positions", "scenario": "tpsl_exact", "counts": [], "exact_args": {"stop_loss": sl, "take_profit": tp},
            "max_words": 110, "table_exempt": True}
    return c.ex("positions", spec)


# ------------------------------------------------------------------ F. TVL and volume
def compact(v):
    v = D(v)
    if v >= 10 ** 9:
        return f"round({plain(v)}/1000000000,1)", "B"
    if v >= 10 ** 7:
        return f"round({plain(v)}/1000000,0)", "M"
    if v >= 10 ** 6:
        return f"round({plain(v)}/1000000,1)", "M"
    if v >= 10 ** 3:
        return f"round({plain(v)}/1000,0)", "K"
    return None, ""


def add_compact(cl, key, v):
    e, suf = compact(v)
    if e is None:
        raise Retry()
    cl.expr(key, e, src=plain(D(v)), k=1 if suf == "B" or (suf == "M" and D(v) < 10 ** 7) else 0)
    return suf


def f_compact(cl, key, suf):
    if D(cl[key]) >= 1000:
        raise Retry()
    return "$" + cl[key] + suf


F1_ONE = [
    "how big is the {p} {t} pool on {c}?", "what's the TVL of {p} {t} on {c}", "how much is locked in {p} for {t} on {c}",
    "{p} {t} pool on {c}, how much TVL", "tvl for {p} {t} ({c})?", "how much money is in the {p} {c} {t} pool",
    "size of the {p} {t} market on {c}", "what's {p}'s {t} TVL on {c} right now",
]
F1_TOP = [
    "which {t} pool has the most TVL?", "biggest {t} yield pool by TVL", "where is the most {t} deposited", "which {t} pool is the largest, and its apy",
    "what's the largest {t} pool right now", "top {t} pool by size?",
]
F2_ONE = [
    "how much volume did {c} do today?", "24h volume on {c}?", "what's the {c} perp volume", "how much has {c} traded in the last day",
    "{c} volume 24h", "how liquid is {c} perp, what's the daily volume",
]
F2_TWO = [
    "compare 24h volume on {c} and {d}", "{c} vs {d} volume today", "which trades more, {c} or {d}? give volume", "volume for {c} and {d}?",
    "how much did {c} and {d} trade today",
]


def sc_tvl(rng, ev):
    asset = rng.choice(["USDC", "USDC", "USDT", "ETH"])
    top = rng.random() < 0.4
    chain = rng.choice(["Ethereum", "Base", "Arbitrum"])
    rows = make_pools(rng, asset, 5 if top else 4, chain=None if top else chain, must_aave=not top)
    d = {"pools": rows, "prices": {}, "balances": []}
    w = RBWorld(d)
    c = Conv(rng, w)
    t = tk(rng, asset)
    if top:
        c.user(finish(rng, pick(rng, F1_TOP, ev).format(t=t), protect=(t,)))
        r = c.call(("get_yields", {"asset": asset}))
        row = max(r["pools"], key=lambda p: p["tvl_usd"])
    else:
        prot = rng.choice([p["protocol"] for p in rows])
        prow = [p for p in rows if p["protocol"] == prot]
        row0 = prow[0]
        chain = row0["chain"]
        c.user(finish(rng, pick(rng, F1_ONE, ev).format(p=prot, t=t, c=chain), protect=(t, prot)))
        r = c.call(("get_yields", {"asset": asset, "chain": chain.lower()}))
        row = [p for p in r["pools"] if p["protocol"] == prot]
        if not row:
            raise Retry()
        row = row[0]
    if len({p["tvl_usd"] for p in r["pools"]}) != len(r["pools"]):
        raise Retry()
    cl = Calc(c)
    suf = add_compact(cl, "tvl", row["tvl_usd"])
    cl.rnd("apy", plain(D(row["apy_pct"])), 2)
    cl.run()
    tv = f_compact(cl, "tvl", suf)
    if top:
        reply = f"{row['protocol']} on {row['chain']} is the biggest {asset} pool, with about {tv} TVL and {f_pct(cl['apy'])} APY."
    else:
        reply = rng.choice([f"{row['protocol']} on {row['chain']} holds about {tv} in {asset}, paying {f_pct(cl['apy'])} APY.",
                            f"About {tv} is deposited in {row['protocol']} {asset} on {row['chain']}, at {f_pct(cl['apy'])} APY."])
    c.say(reply)
    cl.commit()
    return c.ex("tvl_volume", {"shape": "tvl_volume", "scenario": "tvl_top" if top else "tvl_one", "counts": [], "table_exempt": True,
                               "forbid_tools": ["prepare_deposit"]})


def sc_volume(rng, ev):
    two = rng.random() < 0.45
    coins = rng.sample(PERP_COINS, 2 if two else 1)
    mk = make_markets(rng, coins)
    w = RBWorld({"markets": mk, "prices": {}, "balances": []})
    c = Conv(rng, w)
    if two:
        text = pick(rng, F2_TWO, ev).format(c=tk(rng, coins[0]), d=tk(rng, coins[1]))
    else:
        text = pick(rng, F2_ONE, ev).format(c=tk(rng, coins[0]))
    c.user(finish(rng, text, protect=tuple(tk(rng, x) for x in coins)))
    r = c.call(("search_markets", {"query": " ".join(coins) if two else coins[0]}))
    rows = {m["name"].split("-")[0]: m for m in r["markets"]}
    cl = Calc(c)
    sufs = {}
    for cn in coins:
        if cn not in rows:
            raise Retry()
        sufs[cn] = add_compact(cl, f"v_{cn}", rows[cn]["volume_24h_usd"])
    cl.run()
    vs = {cn: f_compact(cl, f"v_{cn}", sufs[cn]) for cn in coins}
    if two:
        a, b = coins
        hi = a if D(rows[a]["volume_24h_usd"]) > D(rows[b]["volume_24h_usd"]) else b
        reply = f"{a} traded about {vs[a]} over the last 24 hours and {b} about {vs[b]}, so {hi} has more volume."
        if rows[a]["volume_24h_usd"] == rows[b]["volume_24h_usd"]:
            raise Retry()
    else:
        a = coins[0]
        reply = rng.choice([f"{a} traded about {vs[a]} over the last 24 hours.", f"About {vs[a]} of {a} perp volume in the last 24 hours."])
    c.say(reply)
    cl.commit()
    return c.ex("tvl_volume", {"shape": "tvl_volume", "scenario": "volume_two" if two else "volume_one", "counts": [],
                               "table_exempt": True})


# ------------------------------------------------------------------ G. short vs detailed pairs: knowledge topics
def T(key, qs, core, extra, tldr, short, nums=()):
    return dict(key=key, qs=qs, core=core, extra=extra, tldr=tldr, short=short, nums=[str(n) for n in nums])


TOPICS = [
    T("funding", ["what is funding on perps?", "how does the funding rate work?", "what does funding mean on hyperliquid", "explain perp funding to me",
                  "why do I pay funding on a perp?"],
      ["Funding is a small payment that passes between longs and shorts on a perp, so its price stays close to the spot price.",
       "When funding is positive, longs pay shorts, and when it is negative, shorts pay longs.", "On Hyperliquid it is paid every hour."],
      ["The rate depends on how far the perp trades from the spot price.", "A crowded long side usually means positive funding.",
       "You only pay or receive it while a position is open.", "The cost grows with the size of your position, not with your margin.",
       "To get a yearly figure, multiply the hourly rate by the hours in a day and the days in a year."],
      "Funding is an hourly payment between longs and shorts that keeps the perp price near spot.",
      "It is an hourly payment between longs and shorts."),
    T("leverage", ["what is leverage?", "how does leverage work in trading", "explain leverage like I'm new", "what does 5x leverage mean",
                   "is high leverage dangerous?"],
      ["Leverage lets you open a position bigger than the money you put in, for example five times bigger at 5x.",
       "It magnifies gains and losses by the same factor.", "Higher leverage also puts your liquidation price closer to your entry."],
      ["The money you put into the trade is called margin.", "At high leverage a small move against you can wipe out the margin.",
       "Lower leverage gives the price more room before liquidation.", "Each coin has its own maximum leverage.",
       "Many traders decide how much they can afford to lose first, then pick the leverage that fits."],
      "Leverage lets you trade a bigger position than your cash, which magnifies both gains and losses.",
      "It lets you trade bigger than your cash, and it magnifies gains and losses.", nums=(5,)),
    T("liquidation", ["what is a liquidation price?", "what does liquidation mean?", "how does liquidation work on perps", "when do I get liquidated",
                      "explain the liquidation price"],
      ["The liquidation price is the price where the exchange closes your leveraged position because your margin can no longer cover the loss.",
       "When that happens you lose the margin in that position.", "Your Hyperliquid account reports it for each open position."],
      ["It sits further from your entry when leverage is lower.", "Adding margin moves it further away.",
       "For a long it is below your entry, and for a short it is above.", "Funding and fees can nudge it over time.",
       "A stop loss placed before that price can close the trade earlier, on your terms."],
      "The liquidation price is where your leveraged position gets force-closed and you lose its margin.",
      "It is the price where the exchange closes your leveraged position and you lose the margin."),
    T("stoploss", ["what is a stop loss?", "how does a stop loss work", "explain stop loss orders", "why use a stop loss", "what's a stop?"],
      ["A stop loss is an order that closes your position if the price moves against you to a level you choose.",
       "It limits how much you can lose on a trade.", "On a long it sits below the current price, and on a short it sits above."],
      ["It usually fills at market once triggered, so the exact price is not guaranteed.", "A fast move can fill it worse than your trigger.",
       "Place it where your trade idea is wrong, not at a random round number.",
       "In Hermit a stop is prepared for review first, and it only takes effect after you approve it with {BIO}.",
       "You can cancel a stop at any time before it triggers."],
      "A stop loss closes your position automatically at a price you pick, to cap your loss.",
      "It is an order that closes your position if the price hits a level you set."),
    T("takeprofit", ["what is a take profit?", "how does take profit work", "explain take profit orders", "what's a tp order", "should I set a take profit?"],
      ["A take profit is an order that closes part or all of a position when the price reaches a target in your favour.",
       "It locks in gains without you watching the screen.", "On a long it sits above the current price, and on a short it sits below."],
      ["It usually fills at market once the target is hit.", "Some traders scale out with several targets.", "A very tight target can cap your upside.",
       "A take profit and a stop loss together bracket a trade.", "Both are reduce only, so they cannot open a new position."],
      "A take profit closes your position at a target price so you lock in gains.",
      "It closes your position when the price reaches your target."),
    T("limitmarket", ["what's the difference between a limit and a market order?", "limit order vs market order?", "explain limit orders",
                      "when should I use a limit order", "what is a market order"],
      ["A market order fills right away at the best available price.",
       "A limit order waits until the price you set is reached, so you control the price but it may never fill.",
       "Market orders are faster, and limit orders can be cheaper."],
      ["Market orders can fill worse than expected when the book is thin.", "Limit orders that rest on the book can pay lower fees.",
       "A limit buy sits below the current price, and a limit sell sits above it.", "You can cancel an unfilled limit order at any time.",
       "For a big size, a limit order avoids pushing the price against yourself."],
      "A market order fills now at the going price, while a limit order waits for the price you set.",
      "A market order fills now, and a limit order waits for your price."),
    T("slippage", ["what is slippage?", "what does slippage mean on a swap", "why did I get less than the quote", "explain slippage simply",
                   "how do I avoid slippage"],
      ["Slippage is the gap between the price you expected and the price you actually got.",
       "It happens because prices move while your trade is processed and because large orders use up the best prices.",
       "Thin markets and big sizes have more of it."],
      ["A slippage limit sets the worst price you accept, and beyond it the trade fails instead.", "Setting it too tight makes trades fail often.",
       "Setting it too loose invites bad fills.", "A quote shows the expected amount, and the final amount can differ a little.",
       "Popular, deep pairs have the least slippage."],
      "Slippage is the difference between the price you expected and the price you got.",
      "It is the gap between the expected price and the price you got."),
    T("apyapr", ["what's the difference between APY and APR?", "APY vs APR?", "explain apy", "is apy guaranteed", "apr or apy, which is better"],
      ["APR is the yearly rate without compounding, and APY is the yearly rate with compounding included.",
       "For the same underlying rate, APY is a bit higher than APR.", "Rates on DeFi pools change often, so both are estimates, not promises."],
      ["Compounding means your earnings start earning too.", "The gap grows with higher rates and more frequent compounding.",
       "A pool APY can include reward tokens that may not hold their value.", "Check whether the rate is base yield or includes rewards.",
       "Hermit's yield data lists the base and reward parts when it has them."],
      "APY includes compounding and APR does not, and neither is a promise.",
      "APY counts compounding, APR does not."),
    T("tvl", ["what is TVL?", "what does total value locked mean", "is a high TVL good?", "explain TVL in defi", "why does tvl matter"],
      ["TVL is total value locked: the dollar value of assets deposited in a protocol or pool.",
       "A bigger TVL usually means more people use it and it is easier to move large sums.", "It is not a safety guarantee."],
      ["TVL moves with token prices, so it can fall just because prices fall.", "A high yield on a small TVL can be a warning sign.",
       "Compare TVL across similar pools, not across different kinds of protocols.", "Audits and track record matter as much as TVL.",
       "Check how long a pool has been running."],
      "TVL is the dollar value deposited in a protocol, a rough sign of size, not of safety.",
      "It is the dollar value of assets deposited in a protocol."),
    T("bps", ["what's a basis point?", "what does bps mean?", "explain bps", "how do I read a fee in bps", "bps to percent?"],
      ["A basis point, or bps, is one hundredth of a percent.", "So 100 bps is 1%, and 35 bps is 0.35%.",
       "Fees and rates are often quoted in bps to avoid confusion."],
      ["To turn bps into a percent, divide by 100.", "To get the fee on an amount, multiply the amount by the bps and divide by ten thousand.",
       "A small bps fee still adds up on large trades.", "Route quotes in Hermit show a fee in bps and the total cost in dollars.",
       "Compare the total cost in dollars, since spread counts too."],
      "One basis point is 0.01%, so 100 bps is 1%.", "One bps is a hundredth of a percent.", nums=(100, 1, 35, 0.35, 0.01)),
    T("bridge", ["what is a bridge in crypto?", "how does bridging work?", "what does it mean to bridge tokens", "explain bridges", "is bridging safe"],
      ["A bridge moves your tokens from one blockchain to another.",
       "You send tokens in on the first chain and receive the same kind of token on the second.",
       "It costs a fee and usually takes from seconds to a few minutes."],
      ["Bridging is a swap across chains, so a quote shows what you will receive.",
       "Hermit uses NEAR Intents for swaps and bridges and shows a review first.", "Check the destination chain before you approve.",
       "Sending to the wrong chain can lose funds.", "Keep a little of the native token on the new chain for gas."],
      "A bridge moves tokens from one blockchain to another for a fee.", "It moves your tokens from one chain to another."),
    T("gas", ["what is gas?", "what are gas fees", "why do I pay gas", "explain gas in crypto", "why is gas so high sometimes"],
      ["Gas is the fee you pay the network to process a transaction.",
       "It is paid in the chain's own token, such as ETH on Ethereum and Base, or SOL on Solana.", "The price changes with how busy the network is."],
      ["Layer twos like Base and Arbitrum are usually much cheaper than Ethereum itself.", "A failed transaction can still cost gas.",
       "Complex actions cost more gas than simple sends.", "Keep a small balance of the native token on each chain you use.",
       "Solana fees are usually tiny."],
      "Gas is the network fee for processing a transaction, paid in the chain's own token.",
      "It is the fee the network charges to process your transaction."),
    T("stablecoin", ["what is a stablecoin?", "explain stablecoins", "what's the point of USDC", "how do stablecoins stay at a dollar",
                     "are stablecoins safe"],
      ["A stablecoin is a token designed to stay at one dollar.", "USDC and USDT are the common ones, each backed by reserves held by the issuer.",
       "They are used to hold value, trade and pay without price swings."],
      ["They can still lose the peg in a crisis.", "Issuers can freeze addresses on some stablecoins.", "Not all stablecoins are backed the same way.",
       "Yield on stablecoins comes from lending or trading fees, not from the token itself.",
       "A stablecoin on one chain is separate from the same coin on another chain until you bridge it."],
      "A stablecoin is a token built to hold a steady dollar value.", "It is a token designed to stay at one dollar."),
    T("selfcustody", ["what does self-custody mean?", "what is a self custody wallet", "explain self custody", "who holds my keys",
                      "is self-custody safer than an exchange"],
      ["Self-custody means you hold the keys to your wallet yourself, so nobody else can move your funds.",
       "There is no company to call if you lose access.", "That freedom comes with the duty to protect your recovery phrase."],
      ["Hermit is a self-custody wallet, and approvals happen on your device with {BIO}.", "Exchanges hold your keys for you, which is custody by them.",
       "Back up your recovery phrase offline.", "Never type it into a website or send it to anyone.", "Nobody from support will ever need it."],
      "Self-custody means only you hold the keys, so only you can move your funds.", "You hold the keys, so only you can move your funds."),
    T("seedphrase", ["what is a seed phrase?", "what's a recovery phrase for", "should I share my seed phrase", "explain recovery phrases",
                     "where should I keep my seed phrase"],
      ["A seed phrase, or recovery phrase, is a list of words that can rebuild your wallet and every key in it.", "Anyone who has it can take your funds.",
       "Never share it, and never type it into a website."],
      ["Write it on paper and keep it somewhere safe and private.", "Do not store it in photos, notes apps or cloud drives.",
       "Hermit will never ask you for it in chat.", "If you lose the phone but have the phrase, you can restore the wallet.",
       "If you lose both, the funds cannot be recovered."],
      "Your seed phrase is the master key to your wallet, so keep it offline and never share it.",
      "It is a list of words that can rebuild your wallet, so keep it secret."),
    T("perp", ["what is a perpetual future?", "what's a perp?", "explain perps simply", "how do perps differ from futures", "what does perpetual mean in trading"],
      ["A perpetual future, or perp, is a contract that tracks a price and never expires.",
       "You can go long or short with leverage without owning the coin.", "A funding payment between longs and shorts keeps it near the spot price."],
      ["You post margin, not the full value of the position.", "Profit and loss depend on the price move times your size.",
       "Your position can be liquidated if the price moves too far against you.", "Hyperliquid is the exchange Hermit uses for perps.",
       "Stock perps from trade.xyz track a stock and trade all day and night."],
      "A perp is a contract that tracks a price forever, letting you go long or short with leverage.",
      "It is a contract on a price that never expires."),
    T("il", ["what is impermanent loss?", "explain impermanent loss", "why do liquidity providers lose money", "what's IL in a pool",
             "is providing liquidity risky"],
      ["Impermanent loss is the value you give up by providing liquidity to a pool compared with just holding the two tokens.",
       "It appears when the prices of the two tokens move apart.", "Trading fees earned by the pool may or may not make up for it."],
      ["It is called impermanent because it shrinks if prices move back.", "It becomes permanent when you withdraw while prices are apart.",
       "Pools of two stablecoins have very little of it.", "Volatile pairs have more.", "Check the fees earned before you judge a pool by its APY."],
      "Impermanent loss is what you lose by providing liquidity instead of simply holding when prices move apart.",
      "It is the value you lose in a pool versus just holding the tokens."),
    T("aave", ["what is Aave?", "how does aave work", "explain aave lending", "is aave safe", "what happens when I deposit in aave"],
      ["Aave is a lending protocol where you deposit tokens to earn interest, and others borrow against their collateral.",
       "The rate changes with how much of the pool is borrowed.", "Hermit can deposit into Aave v3 on Ethereum, Base and Arbitrum."],
      ["Deposits can be withdrawn when the pool has enough free cash.", "Smart contract risk exists even for big protocols.",
       "Your deposit earns a variable rate, not a fixed one.", "Hermit prepares the deposit and you approve it with {BIO}.", "Other pools are listed in Earn."],
      "Aave pays interest on deposits that others borrow against, at a rate that changes with demand.",
      "It is a lending protocol where deposits earn interest.", nums=(3,)),
    T("nearintents", ["what is NEAR Intents?", "how does near intents work", "why does hermit use near intents", "explain the swap route in hermit",
                      "what's near intents"],
      ["NEAR Intents is the swap and bridge route Hermit uses to move value between tokens and chains.",
       "You say what you want, and solvers compete to fill it at a good price.", "Hermit shows a quote first, and nothing moves until you approve the review."],
      ["The quote lists the amount you get and the total cost.", "Most routes take seconds to a couple of minutes.",
       "Bitcoin is supported as a destination only.", "Very small amounts can fall under the minimum for a route.", "Check the destination chain before you approve."],
      "NEAR Intents is the route Hermit uses for swaps and bridges, with a quote before anything moves.",
      "It is the route Hermit uses for swaps and bridges."),
    T("marketcap", ["what is market cap?", "explain market cap", "does a low price mean a coin is cheap", "how is market cap calculated",
                    "what's fully diluted value"],
      ["Market cap is the price of a token times the number of tokens in circulation.", "It tells you how big a project is, not how much money is in it.",
       "A low price does not mean a coin is cheap, because supply matters."],
      ["Fully diluted value uses the total future supply instead.", "Small caps can move fast in both directions.",
       "Compare market caps, not prices, when you compare two coins.", "A big gap between market cap and fully diluted value means many tokens are still to come.",
       "Thin trading makes a market cap easy to distort."],
      "Market cap is price times circulating supply, a measure of size, not of value for money.", "It is price times the number of coins in circulation."),
    T("dex", ["what is a DEX?", "how does a decentralised exchange work", "dex vs exchange?", "explain dex", "is a dex safer than binance"],
      ["A DEX, or decentralised exchange, lets you trade tokens straight from your wallet through smart contracts.",
       "There is no company holding your funds or asking you to sign up.", "Prices come from liquidity pools or order books on chain."],
      ["Uniswap and Jupiter are well known examples.", "You pay network gas plus a small trading fee.", "Fake tokens can be listed by anyone, so check the contract.",
       "A swap through Hermit shows a review before anything happens.", "Large trades in small pools move the price against you."],
      "A DEX lets you trade from your own wallet through smart contracts, with no company holding your funds.",
      "It is an exchange you use straight from your wallet."),
    T("memecoin", ["are memecoins risky?", "what should I know about memecoins", "why do memecoins crash", "explain memecoin risk", "is it safe to buy a new memecoin"],
      ["Memecoins are tokens with no cash flow, so their price comes from attention alone.", "They can rise fast and fall faster.",
       "Only use money you can afford to lose."],
      ["Many have very thin liquidity, so selling can be hard.", "New tokens are often copies with the same name.",
       "Check that the token address is the real one before buying.", "Large wallets can sell and crash the price.",
       "Hermit only buys Jupiter-verified tokens on Solana."],
      "Memecoins run on attention alone, so they can crash fast and you should risk only what you can lose.",
      "They run on attention alone and can fall as fast as they rise."),
    T("airdrop", ["what is an airdrop?", "how do airdrops work", "are airdrops safe to claim", "explain crypto airdrops", "got a random token in my wallet, is it an airdrop?"],
      ["An airdrop is a free distribution of tokens to wallets that meet some rule, such as using a protocol early.",
       "Teams use them to reward users and spread ownership.", "Scammers copy the idea to trick people."],
      ["Never connect your wallet to a link you got by message.", "Real airdrops do not ask for your recovery phrase.",
       "Unknown tokens that appear in your wallet can be traps, so do not approve or sell them blindly.", "Check the official site and announcements.",
       "Claiming can cost a small gas fee."],
      "An airdrop is a free token handout to wallets that qualify, and scammers imitate it, so verify the source.",
      "It is a free token handout to qualifying wallets."),
    T("mev", ["what is MEV?", "what's a sandwich attack", "explain mev", "can bots front-run my swap", "what is front-running in crypto"],
      ["MEV is extra profit that block builders or bots earn by choosing the order of transactions.",
       "A common form is a sandwich attack, where a bot trades just before and after your swap to push your price worse.",
       "A tight slippage limit reduces the damage."],
      ["It mostly hits large swaps on busy pools.", "Private transaction routes can hide your trade from bots.",
       "Smaller trades are rarely worth a bot's effort.", "Splitting a big trade can help.", "You see the effect as a worse price than the quote."],
      "MEV is profit bots take by reordering transactions, for example by sandwiching your swap.",
      "Bots profit by ordering transactions around yours."),
    T("rugpull", ["what is a rug pull?", "how do I spot a rug pull", "explain rug pulls", "what is a honeypot token", "how do rugs happen"],
      ["A rug pull is when the creators of a token drain the liquidity or sell their large holdings and leave buyers with a worthless token.",
       "Warning signs include hidden owner powers, blocked sells and anonymous teams with big allocations.",
       "Checking a token before buying lowers the risk, but never removes it."],
      ["Look at who holds the supply and whether liquidity is locked.", "A contract that blocks selling is called a honeypot.",
       "Hype and a guaranteed return are red flags.", "Start with a small test buy if you decide to try.",
       "Search the token in Hermit before buying to see if it is verified."],
      "A rug pull is when creators drain a token's liquidity and leave buyers with nothing.",
      "It is when creators take the money and leave buyers with a dead token."),
    T("ondevice", ["what does on-device AI mean in Hermit?", "is my chat private?", "does hermit send my chats to a server", "how private is hermit",
                   "what runs on my phone and what doesn't"],
      ["Hermit's on-device models run on your phone or Mac, so your chats are processed there and are not sent to a server.",
       "Chat and files work offline.", "Live prices and trades need a connection, because they need the network."],
      ["Smaller phones use Hermit Lite, a smaller model.", "Private Cloud is a separate option for bigger models.",
       "The model only prepares actions, and you approve them with {BIO}.", "Your wallet keys stay on your device.", "Files you attach are read on the device."],
      "On-device models keep your chats on your phone or Mac, though live data and trades still need the network.",
      "Your chats stay on your device, but live prices need a connection."),
    T("review", ["why do I have to approve with {BIO}?", "what does the review screen do", "can hermit send money without me", "what happens after hermit prepares a trade",
                 "did my trade go through if hermit prepared it?"],
      ["When Hermit prepares a trade, swap or send, it opens a review screen and nothing moves yet.", "You check the details and approve with {BIO}, or cancel.",
       "I can prepare actions but I never see the approval, so I cannot confirm the outcome."],
      ["Activity in Wallets shows what really happened.", "Always check the amount, the network and the address on the review.",
       "A reminder or automation card needs only a tap, since it does not move money by itself.", "Cancel on the review if anything looks off.",
       "Nothing is sent until you approve."],
      "Hermit only prepares actions, and nothing moves until you approve the review with {BIO}.",
      "Nothing moves until you approve the review with {BIO}."),
    T("spotperp", ["spot vs perp, what's the difference?", "should I buy spot or trade perps", "what's the difference between spot and futures", "explain spot trading",
                   "can a spot position be liquidated"],
      ["Spot means you buy and own the actual coin.", "A perp is a contract on the price, so you do not own the coin and can go short.",
       "Spot cannot be liquidated, but a leveraged perp can."],
      ["Spot gains and losses follow the coin one to one.", "A perp has funding payments while it is open.", "A perp lets you size larger with margin.",
       "Spot can be sent to other wallets, and a perp cannot.", "Pick spot to hold and perps to trade short term or with leverage."],
      "Spot is owning the coin, while a perp is a leveraged contract on its price.", "Spot means you own the coin, and a perp is a contract on its price."),
    T("pnl", ["what's unrealized vs realized pnl?", "what does unrealised profit mean", "explain pnl", "what is cost basis", "is my profit real before I close?"],
      ["Unrealised profit or loss is what a position would make or lose if you closed it now.", "Realised profit or loss is locked in after you close.",
       "Unrealised numbers change with every price tick."],
      ["Cost basis, or average cost, is the price you paid on average.", "P&L is the difference between the current price and your cost, times your size.",
       "Fees and funding reduce your final result.", "Closing a position turns unrealised into realised.",
       "Tax rules usually care about realised results, so check your local rules."],
      "Unrealised P&L is on paper until you close, and realised P&L is locked in.", "Unrealised is on paper, and realised is locked in."),
    T("approval", ["what is a token approval?", "why does a dapp ask me to approve a token", "are unlimited approvals dangerous", "explain token allowances", "how do I revoke approvals"],
      ["A token approval lets a contract spend a set amount of one of your tokens.", "DeFi apps need it before they can swap or deposit for you.",
       "Unlimited approvals stay active until you revoke them."],
      ["Approve only the amount you need when the app lets you.", "A malicious contract you approved can drain that token.",
       "You can revoke approvals with a block explorer tool.", "Read every signing prompt before you confirm.",
       "In Hermit, every connect and sign request goes through an approval sheet."],
      "A token approval lets a contract spend your token, so keep amounts small and revoke old ones.",
      "It lets a contract spend a token of yours."),
]
assert len(TOPICS) == 30, len(TOPICS)
EVAL_TOPICS = [t for i, t in enumerate(TOPICS) if i % 6 == 0]
TRAIN_TOPICS = [t for i, t in enumerate(TOPICS) if i % 6 != 0]

PAIR_FORMS = ["plain", "detail", "tldr", "shorter", "plain2", "detail2"]
DETAIL_W = ["{q} in detail", "{q} give me the details", "give me the long version: {q}", "{q} go deep", "walk me through it in detail: {q}",
            "in detail: {q}", "{q} full explanation please", "{q} i want the detailed answer"]
TLDR_W = ["tl;dr {q}", "{q} tl;dr", "tldr: {q}", "{q} (tldr)", "{q} give me the tl;dr", "tl;dr version: {q}", "{q} tldr pls"]
SHORTER_U = ["shorter", "make it shorter", "shorter pls", "can you shorten that", "too long, shorter", "a bit shorter please", "shorten it", "shorter version"]
MORE_U = ["more", "more please", "tell me more", "go on", "elaborate", "can you say more?", "and?", "keep going", "give me more", "more detail on that"]
QUICK_W = ["quick question: {q}", "{q} just curious", "hey, {q}", "{q} thanks", "so {q}"]


def bio_fill(s, bio):
    return s.replace("{BIO}", bio)


def knowledge_conv(rng, notes_ok=True):
    w = RBWorld({"prices": {}, "balances": []})
    return Conv(rng, w, allow_notes=notes_ok)


def sc_pair(rng, ev, topic, form):
    detail = form.startswith("detail")
    c = knowledge_conv(rng, notes_ok=not detail)
    qs = topic["qs"]
    qi = {"plain": 0, "plain2": 1, "detail": 2, "detail2": 3, "tldr": 4, "shorter": 1}[form]
    q = bio_fill(qs[qi], c.bio)
    if form == "plain2" and rng.random() < 0.5:
        q = rng.choice(QUICK_W).format(q=bio_fill(qs[0], c.bio))
    core = [bio_fill(s, c.bio) for s in topic["core"]]
    extra = [bio_fill(s, c.bio) for s in topic["extra"]]
    if form in ("plain", "plain2"):
        c.user(finish(rng, q))
        ans = " ".join(core)
        c.say(ans)
        mw = 70
    elif form in ("detail", "detail2"):
        c.user(finish(rng, rng.choice(DETAIL_W).format(q=q)))
        ans = " ".join(core) + "\n\n" + " ".join(extra)
        c.say(ans)
        mw = 250
    elif form == "tldr":
        c.user(finish(rng, rng.choice(TLDR_W).format(q=q)))
        ans = bio_fill(topic["tldr"], c.bio)
        c.say(ans)
        mw = 30
    else:
        c.user(finish(rng, q))
        c.say(" ".join(core))
        c.user(rng.choice(SHORTER_U) if rng.random() < 0.7 else finish(rng, rng.choice(SHORTER_U)))
        ans = bio_fill(topic["short"], c.bio)
        c.say(ans)
        mw = 30
    spec = {"shape": "pairs", "scenario": form, "topic": topic["key"], "counts": [], "max_words": mw, "topic_nums": topic["nums"],
            "plain_core": " ".join(core), "table_exempt": True,
            "sentence_range": [2, 4] if form in ("plain", "plain2") else None}
    return c.ex("pairs", spec)


def sc_more_knowledge(rng, ev, topic):
    c = knowledge_conv(rng)
    q = bio_fill(topic["qs"][0], c.bio)
    c.user(finish(rng, rng.choice(QUICK_W).format(q=q)))
    core = [bio_fill(s, c.bio) for s in topic["core"]]
    extra = [bio_fill(s, c.bio) for s in topic["extra"]]
    c.say(" ".join(core))
    c.user(rng.choice(MORE_U))
    n = rng.choice([3, 4, 5])
    ans = " ".join(extra[:n])
    c.say(ans)
    spec = {"shape": "more", "scenario": "more_knowledge", "topic": topic["key"], "counts": [], "max_words": 110,
            "topic_nums": topic["nums"], "table_exempt": True, "prev_answer": " ".join(core)}
    return c.ex("more", spec)


def sc_more_holdings(rng, ev):
    c, hs, shown, spec0 = sc_balance_wallet(rng, ev, more=True)
    rest = hs[3:]
    if len(rest) < 2:
        raise Retry()
    prev = c.msgs[-1]["content"]
    c.user(rng.choice(MORE_U))
    listed = rest[:5]
    cl = Calc(c)
    for i, h in enumerate(listed):
        cl.rnd(f"a{i}", h["amount"], k_tok(h["symbol"], h["amount"], rng.choice([3, 4])))
        cl.rnd(f"u{i}", plain(D(h["usd"])), k_usd(h["usd"]))
    cl.run(maxn=10)
    items = [f"{f_tok(cl[f'a{i}'])} {h['symbol']} on {DISPLAY[h['chain']]} ({f_usd(cl[f'u{i}'])})" for i, h in enumerate(listed)]
    if len(items) == 1:
        body = items[0]
    else:
        body = ", ".join(items[:-1]) + f" and {items[-1]}"
    ans = rng.choice([f"The rest are {body}.", f"The other holdings are {body}."])
    left = len(rest) - len(listed)
    if left > 0:
        ans = ans[:-1] + f", plus {left} tiny ones."
    c.say(ans)
    cl.commit()
    spec = {"shape": "more", "scenario": "more_holdings", "counts": [len(rest)] + ([left] if left > 0 else []), "max_words": 110,
            "table_exempt": True, "prev_answer": prev, "n_holdings": len(hs)}
    return c.ex("more", spec)


# ------------------------------------------------------------------ H. tables
def render_table(headers, rows, mode):
    if mode == "md":
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    allr = [headers] + rows
    widths = [max(len(r[i]) for r in allr) for i in range(len(headers))]
    return "\n".join("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))).rstrip() for r in allr)


def table_reply(headers, rows, mode, sentence):
    return render_table(headers, rows, mode) + "\n\n" + sentence


H1_POOL = [
    "show me the top {t} yields", "best {t} yields right now", "top yield pools for {t}", "compare {t} pools for me", "list {t} yield options",
    "where can I earn on {t}? show a few", "what are the best {t} rates at the moment", "{t} yields, top few?", "which pools pay the most on {t}",
    "give me an overview of {t} yields",
]
H1_CHAIN = [
    "top {t} yields on {c}", "best {t} pools on {c}", "show {t} yield options on {c}", "{c} {t} yields, top few", "where do I earn most on {t} on {c}",
    "compare {t} pools on {c} please",
]
H2_POOL = [
    "prices of {cs}?", "give me {cs} prices", "what are {cs} at", "quick price check: {cs}", "show me {cs} prices and 24h change",
    "how are {cs} doing today", "{cs} prices pls", "price and move for {cs}", "check {cs} for me", "what's {cs} trading at",
]
H3_POOL = [
    "compare fees to move {amt} {fs} from {fc} to {tcs}", "what's the cheapest route for {amt} {fs} off {fc}: {tcs}?",
    "quotes for {amt} {fs} from {fc} to {tcs}", "bridge fee comparison: {amt} {fs}, {fc} to {tcs}", "how much would {amt} {fs} cost to send from {fc} to each of {tcs}",
    "which is cheaper for {amt} {fs} from {fc}: {tcs}?", "show me quotes for {amt} {fs} on {fc} going to {tcs}",
]
H4_POOL = [
    "show funding and volume for {cs}", "funding and 24h volume on {cs}?", "compare {cs} perps: yearly funding and volume", "give me funding and volume for {cs}",
    "{cs} perp funding annualised and volume", "table of funding and volume for {cs} pls",
]
H5_ONE = [
    "price of {c}?", "what's {c} at", "{c} price", "how much is {c} right now", "{c} price and 24h move", "check {c} for me", "where's {c} trading",
]
H5_TWO = [
    "price of {c} and {d}?", "what are {c} and {d} at", "{c} and {d} prices", "quick check on {c} and {d}", "how are {c} and {d} doing today",
]
H5_POOL = [
    "what does the top {t} pool pay?", "best {t} rate right now", "highest {t} apy?", "top {t} yield, what is it", "who pays the most on {t}?",
]


def join_names(xs):
    return ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]


def table_conv(rng, w, plain_lines):
    if plain_lines:
        c = Conv(rng, w, notes=[NO_TABLE_NOTE])
        c.table_ok = False
    else:
        c = Conv(rng, w)
    return c


def sc_table_yields(rng, ev, plain_lines=False):
    asset = rng.choice(["USDC", "USDC", "USDT", "ETH"])
    chain = rng.choice(["Ethereum", "Base", "Arbitrum"]) if rng.random() < 0.3 else None
    rows_all = make_pools(rng, asset, rng.choice([3, 4, 5, 5]), chain=chain, must_aave=rng.random() < 0.5)
    if len(rows_all) < 3:
        raise Retry()
    if len({p["tvl_usd"] for p in rows_all}) != len(rows_all) or len({plain(D(p["apy_pct"])) for p in rows_all}) != len(rows_all):
        raise Retry()
    w = RBWorld({"pools": rows_all, "prices": {}, "balances": []})
    c = table_conv(rng, w, plain_lines)
    t = tk(rng, asset)
    if chain:
        text = pick(rng, H1_CHAIN, ev).format(t=t, c=cw(rng, chain.lower()))
        args = {"asset": asset, "chain": chain.lower()}
    else:
        text = pick(rng, H1_POOL, ev).format(t=t)
        args = {"asset": asset}
    c.user(finish(rng, text, protect=(t,)))
    r = c.call(("get_yields", args))
    pools = r["pools"][:5]
    if len(pools) < 3:
        raise Retry()
    cl = Calc(c)
    sufs = []
    for i, p in enumerate(pools):
        cl.rnd(f"apy{i}", plain(D(p["apy_pct"])), 2)
        sufs.append(add_compact(cl, f"tvl{i}", p["tvl_usd"]))
    cl.run(maxn=10)
    rows = [[f"{p['protocol']} on {p['chain']}", f_pct(cl[f"apy{i}"]), f_compact(cl, f"tvl{i}", sufs[i])] for i, p in enumerate(pools)]
    top = pools[0]
    dep = [p for p in pools if p["can_deposit_in_hermit"]]
    sent = f"{top['protocol']} on {top['chain']} pays the most" + (", and only Aave v3 deposits open from Hermit." if dep else ".")
    mode = "plain" if plain_lines else "md"
    c.say(table_reply(["Pool", "APY", "TVL"], rows, mode, sent))
    cl.commit()
    return c.ex("tables", {"shape": "tables", "scenario": "yields", "mode": mode, "n_items": len(pools), "counts": [], "max_words": 110,
                           "max_rows": 5, "forbid_tools": ["prepare_deposit"]})


def sc_table_prices(rng, ev, plain_lines=False):
    k = rng.choice([3, 3, 4, 5])
    coins = rng.sample(["BTC", "ETH", "SOL", "BNB", "HYPE", "LINK", "DOGE", "AVAX", "XRP", "NEAR"], k)
    prices = {}
    for cn in coins:
        p = rand_price(rng, cn)
        p = round(p + rng.choice([0, 0, 0.001234, 0.0087, 0.000456]) * (1 if p >= 10 else 0.01), rng.choice([2, 4, 6]) if p >= 10 else rng.choice([4, 6]))
        ch = round(rng.uniform(-7, 7), rng.choice([2, 2, 4, 6]))
        if ch == 0:
            ch = 0.01
        prices[cn] = {"price": p, "change": ch}
    if len({abs(D(v["change"])) for v in prices.values()}) != len(coins):
        raise Retry()
    w = RBWorld({"prices": prices, "balances": []})
    c = table_conv(rng, w, plain_lines)
    names = [tk(rng, x) for x in coins]
    c.user(finish(rng, pick(rng, H2_POOL, ev).format(cs=join_names(names)), protect=tuple(names)))
    rs = c.call(*[("get_price", {"symbol": x}) for x in coins])
    rs = rs if isinstance(rs, list) else [rs]
    cl = Calc(c)
    for i, r in enumerate(rs):
        cl.rnd(f"p{i}", plain(D(r["price"])), k_usd(r["price"]))
        cl.rnd(f"c{i}", plain(D(r["change_24h_pct"])), 2)
    cl.run(maxn=10)
    rows = [[coins[i], f_usd(cl[f"p{i}"]), f_pct(cl[f"c{i}"], sign=True)] for i in range(len(coins))]
    mv = max(range(len(coins)), key=lambda i: abs(D(cl[f"c{i}"])))
    sent = rng.choice([f"{coins[mv]} moved the most today.", f"The biggest move is {coins[mv]}."])
    mode = "plain" if plain_lines else "md"
    c.say(table_reply(["Coin", "Price", "24h"], rows, mode, sent))
    cl.commit()
    return c.ex("tables", {"shape": "tables", "scenario": "prices", "mode": mode, "n_items": len(coins), "counts": [], "max_words": 110,
                           "max_rows": 5, "forbid_tools": ["prepare_trade"]})


def sc_table_routes(rng, ev, plain_lines=False):
    fs = rng.choice(["USDC", "USDC", "USDT"])
    fc = rng.choice(ROUTE_TOKENS[fs])
    dests = [x for x in ["ethereum", "base", "arbitrum", "bnb", "polygon", "solana"] if x != fc and x in ROUTE_TOKENS[fs]]
    if len(dests) < 3:
        raise Retry()
    k = min(len(dests), rng.choice([3, 3, 4]))
    tcs = rng.sample(dests, k)
    amount = rng.choice([50, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000])
    w = RBWorld({"prices": {}, "balances": []})
    mode_q = "raw" if rng.random() < 0.4 else "app"
    qs = [make_quote(rng, w, fs, fc, fs, tc, amount, mode_q) for tc in tcs]
    costs = [D(q["total_cost_usd"]) for q in qs]
    if len(set(costs)) != len(costs) or len({q["seconds"] for q in qs}) != len(qs):
        raise Retry()
    c = table_conv(rng, w, plain_lines)
    at = amt_text(rng, amount)
    names = [cw(rng, x) for x in tcs]
    if fc not in CHAIN_WORDS:
        raise Retry()
    c.user(finish(rng, pick(rng, H3_POOL, ev).format(amt=at, fs=tk(rng, fs), fc=cw(rng, fc), tcs=join_names(names)), protect=(at,) + tuple(names)))
    rs = c.call(*[("get_quote", {"from_symbol": fs, "from_chain": fc, "to_symbol": fs, "to_chain": tc, "amount": jn(amount)}) for tc in tcs])
    rs = rs if isinstance(rs, list) else [rs]
    cl = Calc(c)
    for i, r in enumerate(rs):
        cl.rnd(f"o{i}", r["amount_out"], k_tok(fs, r["amount_out"], 4))
        cl.rnd(f"c{i}", plain(D(r["total_cost_usd"])), k_usd(r["total_cost_usd"]))
    cl.run(maxn=10)
    rows = [[DISPLAY[tcs[i]], f"{f_tok(cl[f'o{i}'])} {fs}", f_usd(cl[f"c{i}"]), f"{rs[i]['seconds']}s"] for i in range(len(tcs))]
    best = min(range(len(tcs)), key=lambda i: D(cl[f"c{i}"]))
    sent = f"{DISPLAY[tcs[best]]} is the cheapest at {f_usd(cl[f'c{best}'])}, all through NEAR Intents."
    mode = "plain" if plain_lines else "md"
    c.say(table_reply(["To", "You get", "Cost", "Time"], rows, mode, sent))
    cl.commit()
    return c.ex("tables", {"shape": "tables", "scenario": "routes", "mode": mode, "n_items": len(tcs), "counts": [], "max_words": 110,
                           "max_rows": 5, "forbid_tools": ["prepare_swap"], "must_contain": ["NEAR Intents"]})


def sc_table_funding(rng, ev, plain_lines=False):
    k = rng.choice([3, 3, 4])
    coins = rng.sample(PERP_COINS[:8], k)
    mk = make_markets(rng, coins)
    fs = [D(m["funding_hourly_pct"]) for m in mk]
    if len(set(fs)) != len(fs):
        raise Retry()
    w = RBWorld({"markets": mk, "prices": {}, "balances": []})
    c = table_conv(rng, w, plain_lines)
    names = [tk(rng, x) for x in coins]
    c.user(finish(rng, pick(rng, H4_POOL, ev).format(cs=join_names(names)), protect=tuple(names)))
    r = c.call(("search_markets", {"query": " ".join(coins)}))
    rows_m = {m["name"].split("-")[0]: m for m in r["markets"]}
    if any(cn not in rows_m for cn in coins):
        raise Retry()
    cl = Calc(c)
    sufs = {}
    for cn in coins:
        f = plain(D(rows_m[cn]["funding_hourly_pct"]))
        cl.expr(f"f_{cn}", f"round({f}*24*365,2)", src=f, k=2)
        sufs[cn] = add_compact(cl, f"v_{cn}", rows_m[cn]["volume_24h_usd"])
    cl.run(maxn=10)
    rows = [[cn, f_pct(cl[f"f_{cn}"], sign=True), f_compact(cl, f"v_{cn}", sufs[cn])] for cn in coins]
    hi = max(coins, key=lambda cn: D(cl[f"f_{cn}"]))
    sent = f"{hi} has the highest funding; when it is positive, longs pay shorts."
    mode = "plain" if plain_lines else "md"
    c.say(table_reply(["Market", "Funding a year", "Volume 24h"], rows, mode, sent))
    cl.commit()
    return c.ex("tables", {"shape": "tables", "scenario": "funding", "mode": mode, "n_items": len(coins), "counts": [], "max_words": 110,
                           "max_rows": 5, "forbid_tools": ["prepare_trade"]})


def sc_single_fact(rng, ev):
    kind = rng.choices(["price", "two", "pool"], weights=[8, 4, 4])[0]
    if kind in ("price", "two"):
        n = 1 if kind == "price" else 2
        coins = rng.sample(["BTC", "ETH", "SOL", "BNB", "HYPE", "LINK", "DOGE", "AVAX", "XRP", "NEAR"], n)
        prices = {}
        for cn in coins:
            p = rand_price(rng, cn)
            p = round(p + rng.choice([0, 0.001234, 0.0087]) * (1 if p >= 10 else 0.01), rng.choice([2, 4, 6]) if p >= 10 else rng.choice([4, 6]))
            prices[cn] = {"price": p, "change": round(rng.uniform(-7, 7), rng.choice([2, 2, 4])) or 0.01}
        w = RBWorld({"prices": prices, "balances": []})
        c = Conv(rng, w)
        names = [tk(rng, x) for x in coins]
        if n == 1:
            c.user(finish(rng, pick(rng, H5_ONE, ev).format(c=names[0]), protect=tuple(names)))
        else:
            c.user(finish(rng, pick(rng, H5_TWO, ev).format(c=names[0], d=names[1]), protect=tuple(names)))
        rs = c.call(*[("get_price", {"symbol": x}) for x in coins])
        rs = rs if isinstance(rs, list) else [rs]
        cl = Calc(c)
        for i, r in enumerate(rs):
            cl.rnd(f"p{i}", plain(D(r["price"])), k_usd(r["price"]))
            cl.rnd(f"c{i}", plain(D(r["change_24h_pct"])), 2)
        cl.run()
        parts = [f"{coins[i]} is {f_usd(cl[f'p{i}'])} ({f_pct(cl[f'c{i}'], sign=True)} in 24 hours)" for i in range(n)]
        reply = (parts[0] if n == 1 else f"{parts[0]} and {parts[1]}") + "."
        c.say(reply)
        cl.commit()
        return c.ex("tables", {"shape": "tables", "scenario": "single_" + kind, "mode": "none", "n_items": n, "counts": [], "max_words": 70,
                               "forbid_tools": ["prepare_trade"]})
    asset = rng.choice(["USDC", "USDT", "ETH"])
    rows = make_pools(rng, asset, rng.choice([2, 3, 4]))
    w = RBWorld({"pools": rows, "prices": {}, "balances": []})
    c = Conv(rng, w)
    t = tk(rng, asset)
    c.user(finish(rng, pick(rng, H5_POOL, ev).format(t=t), protect=(t,)))
    r = c.call(("get_yields", {"asset": asset}))
    row = r["pools"][0]
    cl = Calc(c)
    cl.rnd("apy", plain(D(row["apy_pct"])), 2)
    suf = add_compact(cl, "tvl", row["tvl_usd"])
    cl.run()
    c.say(f"{row['protocol']} on {row['chain']} pays the most on {asset}, {f_pct(cl['apy'])} APY with about {f_compact(cl, 'tvl', suf)} TVL.")
    cl.commit()
    return c.ex("tables", {"shape": "tables", "scenario": "single_pool", "mode": "none", "n_items": 1, "counts": [], "max_words": 70,
                           "forbid_tools": ["prepare_deposit"]})


# ------------------------------------------------------------------ plan and generation
SUBS = [
    # name, base count, function(rng, ev, i, ctx)
    ("bal_single", 44, lambda rng, ev, i, x: sc_balance_single(rng, ev)),
    ("bal_wallet", 50, lambda rng, ev, i, x: sc_balance_wallet(rng, ev)),
    ("bal_portfolio", 12, lambda rng, ev, i, x: sc_portfolio(rng, ev)),
    ("send_all", 14, lambda rng, ev, i, x: sc_send_all(rng, ev)),
    ("quote", 60, lambda rng, ev, i, x: sc_quote(rng, ev)),
    ("funding", 25, lambda rng, ev, i, x: sc_funding(rng, ev)),
    ("funding_cost", 15, lambda rng, ev, i, x: sc_funding_cost(rng, ev)),
    ("yield_earn", 20, lambda rng, ev, i, x: sc_yield_earn(rng, ev)),
    ("yield_compare", 10, lambda rng, ev, i, x: sc_yield_compare(rng, ev)),
    ("bps", 10, lambda rng, ev, i, x: sc_bps(rng, ev)),
    ("file_transfers", 25, lambda rng, ev, i, x: sc_file_transfers(rng, ev)),
    ("file_cut", 15, lambda rng, ev, i, x: sc_file_transfers(rng, ev, cut=True)),
    ("pasted", 20, lambda rng, ev, i, x: sc_pasted(rng, ev)),
    ("file_spend", 20, lambda rng, ev, i, x: sc_file_spend(rng, ev)),
    ("positions", 35, lambda rng, ev, i, x: sc_positions(rng, ev)),
    ("spot", 10, lambda rng, ev, i, x: sc_spot(rng, ev)),
    ("tpsl", 15, lambda rng, ev, i, x: sc_tpsl(rng, ev)),
    ("tvl", 25, lambda rng, ev, i, x: sc_tvl(rng, ev)),
    ("volume", 25, lambda rng, ev, i, x: sc_volume(rng, ev)),
    ("pairs", 150, lambda rng, ev, i, x: sc_pair(rng, ev, *x["pairs"][i % len(x["pairs"])])),
    ("table_yields", 28, lambda rng, ev, i, x: sc_table_yields(rng, ev, plain_lines=(i % 6 == 5))),
    ("table_prices", 26, lambda rng, ev, i, x: sc_table_prices(rng, ev, plain_lines=(i % 6 == 5))),
    ("table_routes", 18, lambda rng, ev, i, x: sc_table_routes(rng, ev, plain_lines=(i % 6 == 5))),
    ("table_funding", 12, lambda rng, ev, i, x: sc_table_funding(rng, ev, plain_lines=(i % 4 == 3))),
    ("single_fact", 16, lambda rng, ev, i, x: sc_single_fact(rng, ev)),
    ("more_knowledge", 25, lambda rng, ev, i, x: sc_more_knowledge(rng, ev, x["topics"][i % len(x["topics"])])),
    ("more_holdings", 25, lambda rng, ev, i, x: sc_more_holdings(rng, ev)),
]
BASE_TOTAL = sum(c for _, c, _ in SUBS)
assert BASE_TOTAL == 750, BASE_TOTAL


def plan(n):
    """largest-remainder scaling of the base counts to n rows, at least one row per sub when n allows."""
    raw = {name: cnt * n / BASE_TOTAL for name, cnt, _ in SUBS}
    out = {k: max(1 if n >= len(SUBS) else 0, int(v)) for k, v in raw.items()}
    rem = sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    i = 0
    while sum(out.values()) < n:
        out[rem[i % len(rem)]] += 1
        i += 1
    while sum(out.values()) > n:
        k = max(out, key=lambda k: out[k])
        out[k] -= 1
    return out


def pair_tasks(rng, topics, count):
    """round robin over the six forms, every topic once per form (when count allows)."""
    per_form = {f: rng.sample(topics, len(topics)) for f in PAIR_FORMS}
    return [(per_form[PAIR_FORMS[k % 6]][(k // 6) % len(topics)], PAIR_FORMS[k % 6]) for k in range(count)]


ALLOW_REPEAT = {norm_text(x) for x in SHORTER_U + MORE_U}


def user_texts(ex):
    return [norm_text(m["content"]) for m in ex["messages"] if m["role"] == "user"]


def old_eval_texts():
    out = set()
    for d in ("eval", "eval2", "eval3", "data3"):
        p = os.path.join(LAB, d)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(p, fn)) as f:
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    ms = (o.get("context") or []) + (o.get("reference") or []) + (o.get("messages") or [])
                    for m in ms:
                        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                            out.add(norm_text(m["content"]))
    return out


def other_texts(dirs):
    """user texts of the OTHER generators' files in data4/eval4 (never this one), so no text is shared across suites."""
    out = set()
    for d in dirs:
        p = os.path.join(LAB, d)
        for fn in (sorted(os.listdir(p)) if os.path.isdir(p) else []):
            if fn.endswith(".jsonl") and fn != "rounding_brief.jsonl":
                with open(os.path.join(p, fn)) as f:
                    for line in f:
                        try:
                            o = json.loads(line)
                        except Exception:
                            continue
                        for m in (o.get("messages") or []) + (o.get("context") or []) + (o.get("reference") or []):
                            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                                out.add(norm_text(m["content"]))
    return out - ALLOW_REPEAT


def generate(seed, n, ev, avoid):
    rng = random.Random(seed)
    counts = plan(n)
    topics = EVAL_TOPICS if ev else TRAIN_TOPICS
    ctx = {"topics": topics, "pairs": pair_tasks(random.Random(seed + 7), topics, counts["pairs"])}
    rows, seen = [], set(avoid)
    for name, _, fn in SUBS:
        made, tries = 0, 0
        while made < counts[name]:
            tries += 1
            if tries > 400 + counts[name] * 120:
                raise RuntimeError(f"cannot generate {counts[name]} unique '{name}' rows (got {made})")
            crng = random.Random(rng.getrandbits(64))
            try:
                ex = fn(crng, ev, made, ctx)
            except Retry:
                continue
            us = user_texts(ex)
            first = us[0]
            if first in seen or len(us) != len(set(us)) and not all(u in ALLOW_REPEAT for u in us[1:]):
                continue
            if any(u in seen for u in us if u not in ALLOW_REPEAT):
                continue
            seen.update(u for u in us if u not in ALLOW_REPEAT)
            ex["meta"]["sub"] = name
            rows.append(ex)
            made += 1
    random.Random(seed + 1).shuffle(rows)
    return rows


# ------------------------------------------------------------------ eval records
def eval_record(ex, i):
    msgs = ex["messages"]
    lu = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    ref = msgs[lu + 1:]
    calls = [tc for m in ref for tc in (m.get("tool_calls") or [])]
    req = [dict(tc) for tc in calls if tc["name"] != "calculator"]
    final = [m for m in ref if m["role"] == "assistant" and not m.get("tool_calls")][-1]["content"]
    sp = ex["meta"]["spec"]
    shape = sp["shape"]
    checks = ["rounded_ok", "len_ok", "preamble_ok", "no_filler"]
    if shape == "tables":
        checks.append("table_rules")
    if shape == "pairs":
        checks.append("form_len")
    if shape == "more":
        checks.append("no_repeat")
    if req:
        checks += ["right_tool", "exact_digits"]
    if any(c["name"] in ("prepare_send", "set_tp_sl") for c in req):
        checks += ["no_unbacked_done", "bio_word"]
    if sp.get("route"):
        checks.append("route_text")
    results = [json.loads(m["content"]) for m in ref if m["role"] == "tool"]
    gold = {"calls": req, "optional_calls": ["calculator"] if any(c["name"] == "calculator" for c in calls) else [],
            "forbid_tools": sp.get("forbid_tools", []), "max_words": sp.get("max_words", 70), "table": {
                "mode": sp.get("mode", "none"), "max_rows": 5, "n_items": sp.get("n_items"), "table_ok": ex["meta"]["table_ok"]},
            "numbers": [x for x in sp["nums"] if x["calc"]] if "nums" in sp else [], "form": sp.get("scenario"),
            "sentence_range": sp.get("sentence_range"), "must_contain": sp.get("must_contain", []),
            "must_not_contain": ["Relay"], "exact_args": sp.get("exact_args"), "reference_answer": final,
            "latency_short": sp.get("max_words", 70) <= 70 and shape in ("pairs", "long_balance", "quotes", "tvl_volume", "funding_yield", "tables")}
    if shape == "more":
        gold["prev_answer"] = sp.get("prev_answer")
    meta = ex["meta"]
    return {"id": f"rounding_brief-{i:03d}", "kind": meta["kind"], "tools": True, "bio": meta["bio"], "context": msgs[:lu + 1],
            "reference": ref, "expect": meta["expect"], "world": meta["world"], "checks": checks, "gold": gold,
            "meta": {k: meta[k] for k in ("source", "generator", "kind", "bio", "spec", "lang", "table_ok", "sub")}}


# ------------------------------------------------------------------ self-check
BANNED = [r"^\s*(sure|great question|good question|i'd be happy|i would be happy|certainly|absolutely|of course|happy to|as an ai|"
          r"here's|here is|okay|ok\b|no problem|glad to)", r"\bRelay\b", r"successfully", r"not financial advice", "[·•]",
          r"went through", r"has been (sent|swapped|bought|deposited|withdrawn|closed|bridged)", r"\bI(?:'ve| have)? (sent|bought|swapped|deposited|withdrew|closed|bridged)\b",
          r"funds (are|have) (in|arrived)", r"^\s*done\b", r"\bcompleted\b", r"\bexecuted\b", r"!", r"i hope", r"feel free", r"let me know if",
          r"hope this helps", r"as a (language|large)", r"what('s| is) (the )?(current )?price", r"your budget", "—"]
BANNED_RE = [re.compile(p, re.I) for p in BANNED]
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
NUMRX = re.compile(r"(?<![A-Za-z0-9_.])([+-]?)(\$?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
CONSTS = {D(x) for x in (24, 365, 100, 12, 10000, 1000, 1000000, 1000000000)}


def nums_in(text):
    out = []
    for m in NUMRX.finditer(text):
        s = m.group(3).replace(",", "") + (m.group(4) or "")
        after = text[m.end():m.end() + 8]
        out.append(dict(v=D(s), s=s, dollar=bool(m.group(2)), pct=after.startswith("%"), after=after, comma="," in m.group(3),
                        dec=len(m.group(4)) - 1 if m.group(4) else 0, start=m.start()))
    return out


def json_numbers(o, acc):
    if isinstance(o, bool) or o is None:
        return
    if isinstance(o, (int, float)):
        acc.add(abs(D(o)).normalize())
    elif isinstance(o, str):
        for x in nums_in(o):
            acc.add(x["v"].normalize())
    elif isinstance(o, dict):
        for k, v in o.items():
            json_numbers(v, acc)
    elif isinstance(o, list):
        for v in o:
            json_numbers(v, acc)


def sig_digits(s):
    t = s.replace(".", "").lstrip("0")
    return len(t.rstrip("0")) if t else 0


def dec_eval(node):
    """independent Decimal evaluation of the expressions this generator writes (+ - * / abs round)."""
    if isinstance(node, ast.Expression):
        return dec_eval(node.body)
    if isinstance(node, ast.Constant):
        return D(repr(node.value)) if isinstance(node.value, float) else D(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -dec_eval(node.operand)
    if isinstance(node, ast.BinOp):
        a, b = dec_eval(node.left), dec_eval(node.right)
        return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b}[type(node.op)]()
    if isinstance(node, ast.Call) and node.func.id == "abs":
        return abs(dec_eval(node.args[0]))
    if isinstance(node, ast.Call) and node.func.id == "round":
        return dec_eval(node.args[0]).quantize(Decimal(1).scaleb(-int(dec_eval(node.args[1]))), rounding=ROUND_HALF_UP)
    raise ValueError("unsupported")


def sentence_overlap(a, b):
    sa = {norm_text(s) for s in sentences(a)}
    sb = {norm_text(s) for s in sentences(b)}
    return sa & sb


def ngrams(t, k=5):
    w = norm_text(t).split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def check_row(row, idx):
    errs = []
    E = lambda m: errs.append(f"[{idx}] {m}")
    if json.loads(json.dumps(row, ensure_ascii=False)) != row:
        E("json roundtrip")
    if "messages" in row:
        msgs, meta = row["messages"], row["meta"]
        bio, world = meta["bio"], meta["world"]
    else:
        msgs, meta, bio, world = row["context"] + row["reference"], row["meta"], row["bio"], row["world"]
    sp = meta["spec"]
    if meta.get("source") != "rounding_brief" or row.get("tools") is not True:
        E("source/tools flag")
    if bio not in BIOS:
        E("bio")
    if msgs[0]["role"] != "system" or not msgs[0]["content"].startswith(SYSTEM_V3.replace("{BIO}", bio)):
        E("system prompt is not SYSTEM_AGENT_v3")
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    for m in msgs:
        t = m.get("content") or ""
        if other in t or "{BIO}" in t:
            E(f"wrong or unsubstituted device word in {m['role']}")
    sysm = msgs[0]["content"]
    if (NO_TABLE_NOTE in sysm) != (meta["table_ok"] is False):
        E("table_ok flag and standing note disagree")
    users = [m["content"] for m in msgs if m["role"] == "user"]
    if not users:
        E("no user turn")
    # structure
    for k, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            j, nres = k + 1, 0
            while j < len(msgs) and msgs[j]["role"] == "tool":
                nres += 1
                j += 1
            if nres != len(m["tool_calls"]):
                E("tool results count")
            pre = m.get("content") or ""
            if pre and (wc(pre) > 12 or re.search(r"\d", pre)):
                E(f"preamble not ok: {pre!r}")
        if m["role"] == "tool":
            try:
                json.loads(m["content"])
            except Exception:
                E("tool result not json")
    if msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls"):
        E("last message must be a final assistant text")
    # tool calls: registry, schema, replay
    all_calls = []
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            all_calls.append(tc)
            nm = tc["name"]
            if nm not in TOOL_NAMES:
                E(f"tool {nm} not in the v3 registry")
                continue
            sch = TOOL_SCHEMA[nm]
            for p in fmt3.validate_args(tc["arguments"], sch):
                E(f"{nm}: {p}")
            for rq in REQUIRED_OVERRIDE.get(nm, sch.get("required", [])):
                if rq not in tc["arguments"]:
                    E(f"{nm}: missing {rq}")
    w = RBWorld.from_state(world)
    ci = 0
    for k, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for t_i, tc in enumerate(m["tool_calls"]):
                res = dump(w.call(tc["name"], tc["arguments"]))
                if res != msgs[k + 1 + t_i]["content"]:
                    E(f"replay mismatch for {tc['name']}: {res[:80]} vs {msgs[k + 1 + t_i]['content'][:80]}")
    # exact arguments against the scenario
    lu_i = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    got = [dict(tc) for m in msgs[lu_i:] for tc in (m.get("tool_calls") or []) if tc["name"] != "calculator"]
    want = sp.get("gold_calls", [])
    if "gold_calls" in sp and [json.dumps(g, sort_keys=True) for g in got] != [json.dumps(g, sort_keys=True) for g in want]:
        E("calls differ from scenario gold calls")
    utext = "\n".join(users)
    for tc in got:
        a, nm = tc["arguments"], tc["name"]
        if nm == "prepare_send":
            if a["to_address"] not in utext:
                E("address not copied from the user text")
            if "chain" in a:
                E("send chain invented")
            bal = sp.get("exact_amount")
            if Decimal(repr(a["amount"])) != D(bal):
                E("send amount is not the full balance string")
        if nm == "set_tp_sl":
            ex_a = sp.get("exact_args") or {}
            for key in ("stop_loss", "take_profit"):
                typed = ex_a.get(key)
                if (typed is None) != (key not in a):
                    E(f"{key} presence")
                if typed is not None and (typed not in utext or float(typed) != a[key]):
                    E(f"{key} digits differ from the typed value {typed}")
        if nm in ("get_quote",):
            lt = utext.lower()
            for kk in ("from_chain", "to_chain"):
                ch = a[kk]
                alias = {"ethereum": r"ethereum|mainnet", "base": r"\bbase\b", "arbitrum": r"arbitrum|\barb\b", "bnb": r"bnb|\bbsc\b",
                         "polygon": r"polygon", "solana": r"solana|sol chain", "near": r"\bnear\b"}[ch]
                sym = a["from_symbol" if kk == "from_chain" else "to_symbol"]
                if not re.search(alias, lt) and len(ROUTE_TOKENS.get(sym, [])) != 1:
                    E(f"chain {ch} neither named nor implied")
            if D(a["amount"]) not in {x.normalize() for x in user_numbers(utext)} and D(a["amount"]).normalize() not in {x.normalize() for x in user_numbers(utext)}:
                E("quote amount not typed by the user")
            for kk in ("from_symbol", "to_symbol"):
                if not re.search(r"\b" + a[kk].lower() + r"\b", utext.lower()):
                    E(f"symbol {a[kk]} not in user text")
        if nm in ("get_balances", "get_yields") and "chain" in a:
            if not re.search({"ethereum": r"ethereum|mainnet", "base": r"\bbase\b", "arbitrum": r"arbitrum|\barb\b", "bnb": r"bnb|\bbsc\b",
                              "polygon": r"polygon", "solana": r"solana|sol chain", "near": r"\bnear\b"}[a["chain"]], utext.lower()):
                E(f"{nm}: chain {a['chain']} not named by the user")
        for sym_key in ("symbol", "coin", "asset"):
            if sym_key in a and not re.search(r"\b" + str(a[sym_key]).lower() + r"\b", utext.lower()) and nm not in ("get_price",):
                E(f"{nm}.{sym_key}={a[sym_key]} not in user text")
    # universe of backed numbers (sign-insensitive)
    uni = set()
    for u in users:
        uni |= {x.normalize() for x in user_numbers(u)}
        for x in nums_in(u):
            uni.add(x["v"].normalize())
    call_args = set()
    for tc in all_calls:
        if tc["name"] != "calculator":
            json_numbers(tc["arguments"], call_args)
    for m in msgs:
        if m["role"] == "tool":
            json_numbers(json.loads(m["content"]), uni)
    # calculator provenance and arithmetic
    seen_calc = set()
    for k, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for t_i, tc in enumerate(m["tool_calls"]):
                if tc["name"] != "calculator":
                    continue
                expr = tc["arguments"]["expression"]
                try:
                    tree = ast.parse(expr, mode="eval")
                except SyntaxError:
                    E(f"calculator syntax {expr!r}")
                    continue
                for lit in re.findall(r"\d+\.?\d*", expr):
                    v = D(lit).normalize()
                    if v not in uni and v not in {c.normalize() for c in CONSTS} and v not in {D(i).normalize() for i in range(0, 9)}:
                        E(f"calculator constant {lit} not from a source: {expr!r}")
                mm = re.fullmatch(r"round\((-?[\d.]+),(\d+)\)", expr)
                res = json.loads(msgs[k + 1 + t_i]["content"])
                if "result" not in res:
                    E(f"calculator error for {expr!r}")
                    continue
                try:
                    dv = dec_eval(tree)
                    k_ = 2
                    mk = re.search(r",(\d+)\)$", expr)
                    unit = Decimal(1).scaleb(-int(mk.group(1))) if mk else Decimal("0.0001")
                    if abs(dv - D(res["result"])) > unit:
                        E(f"independent evaluation {dv} differs from the calculator result {res['result']} for {expr[:60]!r}")
                except (ValueError, KeyError, AttributeError):
                    E(f"cannot independently evaluate {expr[:60]!r}")
                if mm:
                    q = D(mm.group(1)).quantize(Decimal(1).scaleb(-int(mm.group(2))), rounding=ROUND_HALF_UP)
                    if abs(D(res["result"]) - q) > Decimal(1).scaleb(-int(mm.group(2))) / 2:
                        E(f"round result {res['result']} != {q} for {expr!r}")
                seen_calc.add(res["result"])
                uni.add(abs(D(res["result"])).normalize())
    # file sums use exactly the rows of the scenario
    if sp.get("calc_terms"):
        for tc in all_calls:
            if tc["name"] == "calculator" and tc["arguments"]["expression"].startswith("round(") and "+" in tc["arguments"]["expression"]:
                terms = re.findall(r"\d[\d.]*", tc["arguments"]["expression"].rsplit(",", 1)[0])
                if sorted(terms) != sorted(sp["calc_terms"]):
                    E("sum terms differ from the file rows")
    counts = {D(c).normalize() for c in sp.get("counts", [])}
    topic_nums = {D(c).normalize() for c in sp.get("topic_nums", [])}
    # assistant texts
    finals = [(k, m) for k, m in enumerate(msgs) if m["role"] == "assistant" and not m.get("tool_calls")]
    for fi, (k, m) in enumerate(finals):
        txt = m["content"]
        last = fi == len(finals) - 1
        cap = sp.get("max_words", 70) if last else 110
        n_words = wc_table(txt) if "|" in txt else wc(txt)
        if n_words > cap:
            E(f"too long ({n_words} > {cap} words): {txt[:60]!r}")
        for rx in BANNED_RE:
            if rx.search(txt):
                E(f"banned phrase {rx.pattern!r} in {txt[:80]!r}")
        if EMOJI_RE.search(txt):
            E("emoji")
        if re.search(r"\d\.\d{9,}", txt):
            E(f"raw long decimal in answer: {txt[:80]!r}")
        if re.search(r"Face ID|Touch ID", txt) and bio not in txt:
            E("wrong device word")
        for x in nums_in(txt):
            v = x["v"].normalize()
            after = x["after"]
            if re.match(r"\s?(hours?|h\b)", after) and v == D(24):
                continue
            exact_arg = v in call_args
            if v not in uni and v not in counts and v not in topic_nums and not exact_arg:
                E(f"number {x['s']} not from user/results/calculator: {txt[:90]!r}")
            if exact_arg:
                continue
            val = x["v"]
            if x["dollar"]:
                if val >= 1 and x["dec"] > 2:
                    E(f"USD with {x['dec']} decimals: {x['s']}")
                if 0 < val < 1 and sig_digits(x["s"]) > 4:
                    E(f"USD under 1 with more than 4 significant digits: {x['s']}")
            elif x["pct"]:
                if x["dec"] > 2:
                    E(f"percent with {x['dec']} decimals: {x['s']}")
            else:
                if val >= 1 and x["dec"] > 4:
                    E(f"token with {x['dec']} decimals: {x['s']}")
                if 0 < val < 1 and sig_digits(x["s"]) > 4:
                    E(f"token under 1 with more than 4 significant digits: {x['s']}")
            if x["comma"] and val < 1000:
                E("comma in a number below 1000")
            if x["dollar"] and val >= 1000 and not x["comma"]:
                E(f"USD without thousands comma: {x['s']}")
    # form rules on the last answer
    final = finals[-1][1]["content"]
    prior_finals = [m["content"] for _, m in finals[:-1]]
    sc = sp.get("scenario")
    mode = sp.get("mode", "none")
    has_pipe = "|" in final
    if sp["shape"] == "tables":
        n_items = sp["n_items"]
        paras = [p for p in final.split("\n\n") if p.strip()]
        if mode == "md":
            if meta["table_ok"] is not True:
                E("markdown table in a table_ok false row")
            lines = [l for l in final.splitlines() if l.startswith("|")]
            if n_items < 3 or len(lines) != min(n_items, 5) + 2:
                E(f"table rows {len(lines) - 2} for {n_items} items")
            if min(n_items, 5) > 5:
                E("more than 5 rows")
            if not re.fullmatch(r"\|(---\|)+", lines[1] if len(lines) > 1 else ""):
                E("table separator")
            cols = {l.count("|") for l in lines}
            if len(cols) != 1 or list(cols)[0] < 3:
                E("table columns")
            if len(paras) != 2 or len(sentences(paras[1])) != 1:
                E("table needs one sentence after it")
        elif mode == "plain":
            if has_pipe:
                E("plain lines row has a pipe")
            if meta["table_ok"] is not False:
                E("plain lines row must have table_ok false")
            lines = paras[0].splitlines()
            if len(lines) != min(n_items, 5) + 1 or any(len(re.split(r"\s{2,}", l.strip())) < 2 for l in lines):
                E("plain lines not aligned columns")
            if len(paras) != 2 or len(sentences(paras[1])) != 1:
                E("plain lines need one sentence after")
        else:
            if has_pipe or "\n" in final.strip():
                E("single fact must stay a sentence")
            if n_items >= 3:
                E("3+ items need a table")
    elif has_pipe:
        E("pipe outside a table row")
    if sp["shape"] != "tables" and meta["table_ok"] is not True:
        E("table_ok false only for table rows")
    if sc in ("plain", "plain2"):
        ns = len(sentences(final))
        if not 2 <= ns <= 4:
            E(f"plain answer has {ns} sentences")
    if sc in ("detail", "detail2"):
        if wc(final) < 70 or wc(final) < 1.8 * wc(sp["plain_core"]):
            E("detail answer not longer than plain")
    if sc == "tldr" and (len(sentences(final)) != 1 or wc(final) > 30):
        E("tl;dr must be one short sentence")
    if sc == "shorter":
        if not prior_finals or wc(final) >= 0.7 * wc(prior_finals[-1]) or len(sentences(final)) != 1:
            E("shorter is not shorter")
    if sp["shape"] == "more":
        prev = sp["prev_answer"]
        if prev != prior_finals[-1]:
            E("prev_answer mismatch")
        if sentence_overlap(final, prev) or ngrams(final) & ngrams(prev):
            E("more repeats the previous answer")
        if wc(final) < 12:
            E("more is too thin")
        if sc == "more_holdings":
            items_prev = set(re.findall(r"([\d,.]+ [A-Z]{2,5} on [A-Za-z ]+?) \(", prev))
            items_now = set(re.findall(r"([\d,.]+ [A-Z]{2,5} on [A-Za-z ]+?) \(", final))
            if len(items_prev) != 3 or not items_now:
                E("more_holdings items not parsed")
            if {re.sub(r"^[\d,.]+ ", "", x) for x in items_prev} & {re.sub(r"^[\d,.]+ ", "", x) for x in items_now}:
                E("more repeats a holding already listed")
            if not final.startswith(("The rest", "The other")):
                E("more_holdings wording")
    if sc == "wallet" and sp.get("top3"):
        if len(re.findall(r"\([^)]*\$[\d,.]+\)", final)) != 3 or f"and {sp['n_holdings'] - 3} smaller" not in final:
            E("5+ holdings must show the top 3 then 'and N smaller'")
    # prepare answers
    if any(tc["name"] in ("prepare_send", "set_tp_sl") for tc in all_calls):
        low = final.lower()
        if "waiting" not in low and "approve" not in low:
            E("prepare answer does not say it waits for approval")
        if bio not in final:
            E("prepare answer misses the device word")
        if re.search(r"\b(sent|done|set up|is live|are live)\b", low.replace("nothing has been sent", "").replace("nothing is sent", "")):
            E("prepare answer claims completion")
    if sp.get("route") and "NEAR Intents" not in final:
        E("quote answer misses the route")
    for t in ("prepare_swap", "prepare_deposit", "prepare_trade", "prepare_sell", "prepare_close", "prepare_buy"):
        if any(tc["name"] == t for tc in all_calls) and t in sp.get("forbid_tools", []):
            E(f"forbidden tool {t}")
    # render round trip
    try:
        text = render_text([dict(m) for m in msgs], bio)
        back = []
        for part in text.split("<|im_start|>assistant\n")[1:]:
            cs, bad = H.parse_calls(part.split("<|im_end|>")[0])
            if bad:
                E("render: bad call")
            back += cs
        if back != [dict(name=c["name"], arguments=c["arguments"]) for c in all_calls]:
            E("render round trip calls")
        if other in text:
            E("device word leaks into the rendered prompt")
    except Exception as e:
        E(f"render failed: {e}")
        text = ""
    return errs, len(text)


def self_check(train, ev_rows, avoid):
    errs = []
    maxlen = 0
    for i, r in enumerate(train):
        e, ln = check_row(r, f"train {i}")
        errs += e
        maxlen = max(maxlen, ln)
    for i, r in enumerate(ev_rows):
        e, ln = check_row(r, f"eval {i}")
        errs += e
        maxlen = max(maxlen, ln)

    def utexts(r):
        ms = r["messages"] if "messages" in r else r["context"] + r["reference"]
        return [norm_text(m["content"]) for m in ms if m["role"] == "user"]
    seen = {}
    for name, rows in (("train", train), ("eval", ev_rows)):
        for i, r in enumerate(rows):
            us = utexts(r)
            for u in us:
                if u in ALLOW_REPEAT:
                    continue
                if u in seen:
                    errs.append(f"duplicate user text {u!r}: {seen[u]} and {name} {i}")
                seen[u] = f"{name} {i}"
                if u in avoid:
                    errs.append(f"{name} {i} user text collides with an older eval set: {u!r}")
    # realism and variety
    firsts = []
    for r in train:
        ms = r["messages"]
        firsts.append(next(m["content"] for m in ms if m["role"] == "user"))
    short_first = [f for f in firsts if not f.startswith("The user attached")]
    low = sum(1 for f in short_first if f == f.lower()) / max(1, len(short_first))
    if not 0.2 <= low <= 0.7 and len(train) >= 300:
        errs.append(f"casual lower case share {low:.2f} outside 0.2..0.7")
    if len(train) >= 300:
        if len({re.sub(r"[\d.,]+", "#", f.lower())[:40] for f in short_first}) < 150:
            errs.append("too little variety in user language")
        bios = Counter(r["meta"]["bio"] for r in train)
        fid = bios["Face ID"] / len(train)
        if not 0.6 <= fid <= 0.8:
            errs.append(f"Face ID share {fid:.2f} not about 0.70")
    subs_tr = Counter(r["meta"]["sub"] for r in train)
    return errs, maxlen, subs_tr


def utexts_of(r):
    ms = r["messages"] if "messages" in r else r["context"] + r["reference"]
    return [norm_text(m["content"]) for m in ms if m["role"] == "user"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--eval-seed", type=int, default=9404)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(LAB, "data4", "rounding_brief.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval4", "rounding_brief.jsonl"))
    a = ap.parse_args()
    old = old_eval_texts()
    ev_ex = generate(a.eval_seed, a.eval_n, True, old | other_texts(("data4", "eval4")))
    ev_rows = [eval_record(ex, i) for i, ex in enumerate(ev_ex)]
    ev_texts = {t for ex in ev_ex for t in user_texts(ex) if t not in ALLOW_REPEAT}
    train = generate(a.seed, a.n, False, old | ev_texts | other_texts(("eval4",)))
    errs, maxlen, subs = self_check(train, ev_rows, old)
    if errs:
        print(f"SELF-CHECK FAILED: {len(errs)} problems", file=sys.stderr)
        for e in errs[:60]:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for r in ev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train subs:", dict(sorted(subs.items())), file=sys.stderr)
    print("train shapes:", dict(Counter(r["meta"]["kind"] for r in train)), file=sys.stderr)
    print("eval shapes:", dict(Counter(r["kind"] for r in ev_rows)), file=sys.stderr)
    words = sorted(wc_table(m["content"]) if "|" in m["content"] else wc(m["content"]) for r in train
                   for m in r["messages"][-1:])
    print(f"final answer words: median {words[len(words) // 2]}, p95 {words[int(len(words) * .95)]}, max {words[-1]}; "
          f"max rendered chars {maxlen} (~{maxlen // 3.5:.0f} tokens)", file=sys.stderr)
    other = set()
    for d in ("data4", "eval4"):
        p = os.path.join(LAB, d)
        for fn in (os.listdir(p) if os.path.isdir(p) else []):
            if fn.endswith(".jsonl") and fn != "rounding_brief.jsonl":
                for line in open(os.path.join(p, fn)):
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    for m in (o.get("messages") or []) + (o.get("context") or []):
                        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                            other.add(norm_text(m["content"]))
    mine = {t for r in train for t in utexts_of(r)} | {t for r in ev_rows for t in utexts_of(r)}
    clash = (mine - ALLOW_REPEAT) & other
    if clash:
        print(f"note: {len(clash)} user texts also appear in other data4/eval4 files (e.g. {sorted(clash)[:2]})", file=sys.stderr)
    print(f"rounding_brief: {len(train)} train, {len(ev_rows)} eval, self-check ok")


if __name__ == "__main__":
    main()
