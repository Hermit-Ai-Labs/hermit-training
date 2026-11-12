#!/usr/bin/env python3
"""stocks_price: stocks and indexes through get_price (trade.xyz perps), honest about what the number is.

Writes
  data3/stocks_price.jsonl   training rows   (--seed, --n; default seed 3, n 160)
  eval3/stocks_price.jsonl   held-out items  (--eval-seed, --n-eval; default 9308, 24)
and then runs the self-check over both files (exit code 1 on any failure).

App truth (checked in wt-near, read only): prepare_trade looks the coin up with hl.asset(coin) on keys such as
"xyz:NVDA" after .uppercased(), and reads the mark from the MAIN dex contexts only, so a stock can never be
prepared as a perp. Stocks are priced with get_price (HermitToolbox.price strips "XYZ:") and traded on trade.xyz
through open_app. So no xyz: coins are added anywhere and no prepare_* call is ever generated here.

Record shape (same as gen2 Conv.ex, meta extended):
  {"messages": [system, user, assistant{tool_calls}, tool, ..., assistant], "tools": true,
   "meta": {"source": "stocks_price", "kind", "bio", "slots", "gold_calls", "world", "world_final", "expect", "checks", "tpl", ...}}
Eval item: {id, kind, tools, bio, context, reference, expect, world, checks{...}, meta{source, kind}}.
  world = {"stocks": {SYM: {"price", "change_24h_pct"}}}; StockWorld.from_state(world) replays get_price and open_app.
  Any symbol not in world.stocks answers "No Hyperliquid market for X. Answer from what you know instead, and say there's no live price here."

Every assistant sentence is a Python f-string filled from tool results or the user's text. No numbers are typed by hand.
gen2 helpers are imported from ../gen2 (Conv, S), never copied or edited.
"""
import argparse, json, os, random, re, sys
from collections import Counter
from decimal import Decimal

sys.dont_write_bytecode = True          # never write __pycache__ into gen2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "gen2"))
import hermit_fmt as H                  # gen2 (render helpers, parse_calls)
from build_agent import Conv, S         # gen2

SPEC = os.path.join(ROOT, "spec")
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
SYSTEM_TPL = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read()
assert SYSTEM_TPL.count("{BIO}") == 1
OUT_TRAIN = os.path.join(ROOT, "data3", "stocks_price.jsonl")
OUT_EVAL = os.path.join(ROOT, "eval3", "stocks_price.jsonl")

STOCK_SOURCE = "trade.xyz perp on Hyperliquid (tracks the stock)"
OPEN_NOTE = "Opened in Hermit's browser. Nothing connects to the wallet until the user approves it."
SHARES_FIGS = re.compile(r"\$")


# ============================================================ entities
def _e(name, kind, band, tr, ev):
    return {"name": name, "kind": kind, "band": band, "tr": tr, "ev": ev}


STOCKS = {
    "NVDA": _e("Nvidia", "stock", (95, 210), ["NVDA", "nvda", "Nvidia", "nvidia", "NVIDIA", "nvdia"], ["nvidea", "$nvda", "Nvdia"]),
    "TSLA": _e("Tesla", "stock", (180, 480), ["TSLA", "tsla", "Tesla", "tesla", "TESLA", "telsa"], ["tesal", "$TSLA", "Tsla"]),
    "AAPL": _e("Apple", "stock", (170, 290), ["AAPL", "aapl", "Apple", "apple", "appl"], ["appel", "$aapl", "Aapl"]),
    "MSFT": _e("Microsoft", "stock", (350, 560), ["MSFT", "msft", "Microsoft", "microsoft", "Msft"], ["microsft", "$MSFT", "Micro soft"]),
    "META": _e("Meta", "stock", (480, 820), ["META", "meta", "Meta", "Facebook", "facebook"], ["$META", "fb", "Meta Platforms"]),
    "AMZN": _e("Amazon", "stock", (150, 260), ["AMZN", "amzn", "Amazon", "amazon"], ["amazn", "$AMZN", "Amzn"]),
    "GOOGL": _e("Google", "stock", (140, 260), ["GOOGL", "googl", "Google", "google", "Alphabet", "GOOG"], ["googel", "$GOOGL", "alphabet"]),
    "COIN": _e("Coinbase", "stock", (180, 420), ["COIN", "Coinbase", "coinbase", "coinbse"], ["$COIN", "Coinbse", "COINBASE"]),
    "MSTR": _e("MicroStrategy", "stock", (150, 520), ["MSTR", "mstr", "MicroStrategy", "Microstrategy", "microstrategy"], ["microstrat", "$MSTR", "mircostrategy"]),
    "HOOD": _e("Robinhood", "stock", (25, 140), ["HOOD", "Robinhood", "robinhood", "robinhod"], ["$HOOD", "Hood", "ROBINHOOD"]),
    "SPX": _e("S&P 500", "index", (5200, 7200), ["SPX", "spx", "S&P 500", "the S&P 500", "the S&P", "s&p", "the s&p", "S&P500", "sp500"],
              ["the SPX", "S&P", "the S&P500", "sp 500", "the sp500"]),
}
STOCK_ORDER = list(STOCKS)

UNKNOWN = {
    "BRK.B": {"name": "Berkshire Hathaway", "private": False, "tr": ["Berkshire", "Berkshire Hathaway", "BRK.B", "brk.b", "berkshire hathaway"],
              "ev": ["Buffett's Berkshire", "$BRK.B"],
              "fact": "Berkshire Hathaway is the holding company Warren Buffett built, and its B shares are listed in the US."},
    "NTDOY": {"name": "Nintendo", "private": False, "tr": ["Nintendo", "nintendo", "NTDOY"], "ev": ["nintedo", "$NTDOY"],
              "fact": "Nintendo is the Japanese game company behind Mario and the Switch, listed in Tokyo and carried by US brokers as NTDOY."},
    "OPENAI": {"name": "OpenAI", "private": True, "tr": ["OpenAI", "openai", "Open AI"], "ev": ["open ai", "OPENAI"],
               "fact": "OpenAI is a private company, so it has no public share price."},
    "TM": {"name": "Toyota", "private": False, "tr": ["Toyota", "toyota", "TM"], "ev": ["toyotta", "$TM"],
           "fact": "Toyota is the Japanese carmaker, listed in Tokyo and in the US under the ticker TM."},
    "SPACEX": {"name": "SpaceX", "private": True, "tr": ["SpaceX", "spacex", "Space X"], "ev": ["space x", "$SPACEX"],
               "fact": "SpaceX is a private company, so it has no public share price."},
    "TCEHY": {"name": "Tencent", "private": False, "tr": ["Tencent", "tencent"], "ev": ["tencnt"],
              "fact": "Tencent is the Chinese tech and gaming company, listed in Hong Kong and traded over the counter in the US."},
    # eval-only entities
    "SHOP": {"name": "Shopify", "private": False, "tr": ["Shopify", "shopify", "SHOP"], "ev": [], "eval_only": True,
             "fact": "Shopify is the Canadian e-commerce software company, listed in Toronto and the US."},
    "STRIPE": {"name": "Stripe", "private": True, "tr": ["Stripe", "stripe"], "ev": [], "eval_only": True,
               "fact": "Stripe is a private payments company, so it has no public share price."},
    "BABA": {"name": "Alibaba", "private": False, "tr": ["Alibaba", "alibaba", "BABA"], "ev": [], "eval_only": True,
             "fact": "Alibaba is the Chinese e-commerce group, listed in Hong Kong and the US."},
    "SSNLF": {"name": "Samsung", "private": False, "tr": ["Samsung", "samsung"], "ev": [], "eval_only": True,
              "fact": "Samsung Electronics is the Korean electronics maker, listed in Seoul."},
}
UNK_TRAIN = [k for k, v in UNKNOWN.items() if not v.get("eval_only")]
UNK_EVAL_ONLY = [k for k, v in UNKNOWN.items() if v.get("eval_only")]


# ============================================================ simulator (one shared shape for get_price and open_app)
class StockWorld:
    """Replays get_price / open_app for the trade.xyz set. Anything else is an error (no prepare_* here)."""

    def __init__(self, rng=None, stocks=None):
        if stocks is not None:
            self.stocks = json.loads(json.dumps(stocks))
            return
        self.stocks = {}
        for sym in STOCK_ORDER:
            lo, hi = STOCKS[sym]["band"]
            while True:
                p = round(rng.uniform(lo, hi), 2)
                if p != int(p):
                    break
            while True:
                ch = round(max(-9.0, min(9.0, rng.gauss(0, 2.2))), 2)
                if abs(ch) >= 0.05:
                    break
            self.stocks[sym] = {"price": p, "change_24h_pct": ch}

    def null_change(self, sym):
        self.stocks[sym]["change_24h_pct"] = None

    def state(self):
        return {"stocks": json.loads(json.dumps(self.stocks))}

    @classmethod
    def from_state(cls, st):
        return cls(stocks=st["stocks"])

    def call(self, name, args):
        if name == "get_price":
            sym = str(args.get("symbol", "")).upper().strip().replace("XYZ:", "")
            if sym in self.stocks:
                s = self.stocks[sym]
                return {"change_24h_pct": s["change_24h_pct"], "price": s["price"], "source": STOCK_SOURCE, "symbol": sym}
            return {"error": f"No Hyperliquid market for {sym}. Answer from what you know instead, and say there's no live price here."}
        if name == "open_app":
            if str(args.get("name", "")).lower() == "trade.xyz":
                return {"note": OPEN_NOTE, "opened": "https://trade.xyz", "status": "opened"}
            return {"error": "unsupported in stocks_price sim"}
        return {"error": f"unsupported in stocks_price sim: {name}"}


# ============================================================ user phrasing
# (template, ...) strings use {w} and {w2}; templates in the *_EV lists are used for eval only.
# "S|" = stock-only (not the index).
PRICE_TR = [
    "price of {w}", "what's {w} at?", "how's {w} doing", "{w}?", "{w} price", "how much is {w}", "check {w} for me",
    "yo whats {w} at rn", "current {w} price pls", "{w} price now", "can you pull up the price of {w}",
    "tell me where {w} is at", "what's {w} worth right now", "{w} today?", "give me the {w} price",
    "got a price on {w}?", "S|{w} stock price", "S|how much is one share of {w}", "show me {w}", "and {w}?",
    "price check on {w}", "where's {w} trading?", "I wanna know the {w} price", "whats the number on {w}",
    "{w} rn", "what are we at on {w}", "{w} price please", "any move in {w} today?",
    "just checking {w}, where is it", "tell me the {w} price and how it moved today", "price for {w} when you get a sec",
    "hey what's {w} sitting at", "what's {w} trading at", "how much is {w} right now",
    "my friend wont shut up about {w}, what's it at?", "watching {w} this week, where is it now",
    "S|what's the {w} share price", "quick, {w} price?", "{w} quote?", "what does {w} look like right now",
]
PRICE_EV = [
    "what is {w} trading at right now", "quick one, where's {w} right now?", "how is {w} looking today",
    "what's the latest on {w}", "ok what about {w}", "how much is {w} up or down", "Hey Hermit, how's {w}?",
    "S|what's one {w} share going for", "can you tell me {w}'s price",
]
UPQ_TR = [  # (template, mode)
    ("is {w} up today", "up"), ("is {w} up or down today?", "either"), ("{w} green or red today?", "either"),
    ("is {w} down today", "down"), ("did {w} go up today", "up"), ("is {w} having a good day?", "up"),
    ("how's {w} moving today, up or down?", "either"), ("is {w} up rn", "up"), ("are we up or down on {w}", "either"),
    ("is {w} pumping today", "up"), ("is {w} red today?", "down"), ("is {w} dumping?", "down"),
]
UPQ_EV = [("is {w} in the green today?", "up"), ("has {w} moved much in the last day?", "either"), ("up or down: {w}?", "either")]
BUY_TR = [
    "should I buy {w}", "should i buy {w}?", "can I buy {w} here", "can I buy {w} on Hermit?", "is {w} a buy right now",
    "thinking about buying {w}, thoughts?", "can i get some {w} in here?", "want to buy {w}, how do i do that here",
    "should I get into {w}", "is it a good time to buy {w}", "buy {w}?", "{w}, worth buying?", "can I trade {w} with this",
    "how do I buy {w}", "I wanna buy some {w}", "should I buy {w} today", "can I buy {w} in this app",
    "is {w} worth buying right now", "my cousin says buy {w}, should I?", "been eyeing {w} for weeks, buy it?",
    "can u buy {w} for me", "do you think I should buy {w}", "can I invest in {w} here", "S|can I buy {w} stock here",
    "I want some exposure to {w}, can I do that here?", "{w} dip, should I buy?", "is {w} a good buy",
    "could I get into {w} through Hermit", "should I buy {w} now or wait", "tell me about buying {w}",
]
BUY_EV = [
    "is {w} worth getting into", "any chance I can buy {w} through you", "wanna grab some {w}", "{w} buy or nah",
    "could I invest in {w} here?", "S|how do I get {w} shares here",
]
MULTI_TR = [  # (template, cmp)
    ("{w} and {w2} prices?", False), ("price of {w} and {w2}", False), ("what are {w} and {w2} at", False),
    ("{w} vs {w2} today", True), ("check {w} and {w2} for me", False), ("how are {w} and {w2} doing", False),
    ("{w}, {w2}?", False), ("which is up more today, {w} or {w2}?", True), ("{w} or {w2}, who's winning today", True),
    ("give me {w} and {w2}", False),
]
MULTI_EV = [("where are {w} and {w2} trading", False), ("is {w} beating {w2} today", True), ("{w} and {w2} rn pls", False)]
FOLLOW_YES_TR = [
    "yes", "yes please", "yeah open it", "sure", "ok open trade.xyz", "open it", "yep go ahead", "do it", "sure, open it up",
    "yes pls", "ok take me there", "go for it", "yeah let's see it", "ok sure open it", "yup", "ye open it",
    "open trade.xyz so I can look at {w}", "yes open trade.xyz", "sure, take me to trade.xyz", "ok show me {w} there",
    "alright open it", "please do",
]
FOLLOW_YES_EV = ["sounds good, open it", "yes let me see", "k open it up", "yep take me there"]
FOLLOW_NO_TR = ["no thanks", "nah just wanted the price", "no that's all", "nope, thanks", "not now", "nah im good", "just the price, thx"]
FOLLOW_NO_EV = ["no, that's all I needed", "nah"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&$.\s]", "", t.lower())).strip()


def typo(chunk, rng):
    ws = [m for m in re.finditer(r"[A-Za-z]{4,}", chunk) if m.group(0).lower() not in ("pls", "plz", "xyz", "trade")]
    if not ws:
        return chunk
    m = rng.choice(ws)
    w = m.group(0)
    op = rng.choice(["swap", "drop", "dbl"])
    i = rng.randrange(1, len(w) - 1)
    if op == "swap":
        w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif op == "drop":
        vs = [j for j, ch in enumerate(w) if ch in "aeiou" and 0 < j]
        j = rng.choice(vs) if vs else i
        w2 = w[:j] + w[j + 1:]
    else:
        w2 = w[:i] + w[i] + w[i:]
    return chunk[:m.start()] + w2 + chunk[m.end():]


def build_user(rng, tpl, slots):
    """Fill a template. Typos and casing only touch the template text, never a slot (ticker, name, address)."""
    parts = re.split(r"(\{w2?\})", tpl)
    text_idx = [i for i, p in enumerate(parts) if not p.startswith("{w") and p.strip()]
    lower = rng.random() < 0.35
    did_typo = False
    if rng.random() < 0.15 and text_idx:
        k = rng.choice(text_idx)
        new = typo(parts[k], rng)
        did_typo = new != parts[k]
        parts[k] = new
    out = []
    for p in parts:
        if p == "{w}":
            out.append(slots[0])
        elif p == "{w2}":
            out.append(slots[1])
        else:
            out.append(p.lower() if lower else p)
    s = "".join(out)
    if s.endswith("?") and rng.random() < 0.25:
        s = s[:-1]
    if rng.random() < 0.02:
        s = s.upper()
    return s, did_typo


def pick_tpl(rng, pool_tr, pool_ev, mode, stock_only_ok):
    pool = pool_ev if mode == "eval" else pool_tr
    for _ in range(50):
        i = rng.randrange(len(pool))
        t = pool[i]
        raw = t[0] if isinstance(t, tuple) else t
        if raw.startswith("S|") and not stock_only_ok:
            continue
        return i, t
    raise RuntimeError("no template")


def strip_tag(t):
    return t[2:] if t.startswith("S|") else t


def pick_word(rng, sym, mode):
    if sym in STOCKS:
        e = STOCKS[sym]
        if mode == "eval" and rng.random() < 0.6:
            return rng.choice(e["ev"])
        return rng.choice(e["tr"])
    e = UNKNOWN[sym]
    if mode == "eval" and e["ev"] and rng.random() < 0.6:
        return rng.choice(e["ev"])
    return rng.choice(e["tr"])


# ============================================================ answer writing (all numbers from tool results)
def money_stock(p):
    return S(p)                      # "$131.40", "$1,234.50"


def fmt_price(sym, p):
    if STOCKS[sym]["kind"] == "index":
        return f"{p:,.2f}"          # index level, not dollars
    return money_stock(p)


def fmt_chg(c):
    return f"{abs(c):g}"


def dirw(c):
    return "up" if c > 0 else "down"


def label(sym, rng, start=True):
    e = STOCKS[sym]
    if e["kind"] == "index":
        opts = ["SPX", "The S&P 500", "The S&P 500 (SPX)"] if start else ["SPX", "the S&P 500"]
    else:
        opts = [sym, sym, e["name"], f"{e['name']} ({sym})"]
    return rng.choice(opts)


def label_l(sym, rng):
    return label(sym, rng, start=False)


def thing(sym):
    return "index" if STOCKS[sym]["kind"] == "index" else "stock"


def official(sym):
    return "official index level" if STOCKS[sym]["kind"] == "index" else "exchange price"


def wording(rng, sym):
    t, o = thing(sym), official(sym)
    return rng.choice([
        f"That's a trade.xyz perp on Hyperliquid that tracks the {t}, so it trades 24/7 and can differ from the {o}.",
        f"It's a trade.xyz perp on Hyperliquid that tracks the {t}, not the exchange quote, so it trades around the clock and can drift from the official price.",
        f"This comes from the trade.xyz perp on Hyperliquid, which tracks the {t} and trades 24/7, so it can be off from the {o}, especially when the market is closed.",
        f"Note it's a perp on trade.xyz (Hyperliquid) that tracks the {t}. It trades 24/7 and can differ from the {o}.",
        f"That's the trade.xyz perp on Hyperliquid, which tracks the {t} around the clock, so it may not match the {o} exactly.",
        f"The number is from a trade.xyz perp on Hyperliquid that tracks the {t}. Perps trade 24/7, so it can differ from the {o}.",
    ])


def wording_short(rng, sym):
    t, o = thing(sym), official(sym)
    return rng.choice([
        f"That's a trade.xyz perp on Hyperliquid that tracks the {t}, so it trades 24/7.",
        f"It's a trade.xyz perp that tracks the {t}, not the exchange quote.",
        f"That's the trade.xyz perp on Hyperliquid, which tracks the {t} and trades 24/7.",
        f"It's a perp on trade.xyz that tracks the {t} and can differ from the {o}.",
        f"This is a trade.xyz perp on Hyperliquid, so it tracks the {t} around the clock.",
        f"The price is from a perp on trade.xyz that tracks the {t}.",
    ])


def wording_multi(rng, syms):
    kinds = {thing(s) for s in syms}
    ph = "stocks" if kinds == {"stock"} else "stock and the index"
    return rng.choice([
        f"Both are trade.xyz perps on Hyperliquid that track the {ph}, so they trade 24/7 and can differ from exchange prices.",
        f"These are trade.xyz perps on Hyperliquid, which track the {ph} around the clock, so they can differ from the official prices.",
        f"Both numbers come from trade.xyz perps on Hyperliquid that track the {ph}. Perps trade 24/7, so they can drift from the exchange prices.",
        f"Note both are perps on trade.xyz that track the {ph}, so they trade 24/7 and may not match the exchange exactly.",
    ])


def honest(rng, sym):
    if STOCKS[sym]["kind"] == "index":
        return rng.choice([
            "I can't buy the index for you in Hermit, and the perp uses leverage if you trade it.",
            "Hermit can't hold the index itself, and trading the perp uses leverage, so losses can move fast.",
            "You can't own the index in Hermit. The perp tracks it but uses leverage when you trade it.",
            "There's no index fund in Hermit, only the perp, and that uses leverage.",
            "Hermit doesn't hold the index for you. If you trade the perp instead, leverage is involved, so it can go against you quickly.",
            "I can't prepare an index buy here. The perp is the way in, and it uses leverage.",
        ])
    return rng.choice([
        "I can't buy shares for you in Hermit, and the perp uses leverage if you trade it.",
        "Hermit can't hold shares, and trading the perp uses leverage, so losses can move fast.",
        "You can't own the actual shares in Hermit. The perp tracks the stock but uses leverage when you trade it.",
        "There are no real shares in Hermit, only the perp, and that uses leverage.",
        "Hermit doesn't hold shares for you. If you trade the perp instead, leverage is involved, so it can go against you quickly.",
        "I can't prepare a stock buy here, since Hermit has no shares. The perp is the way in, and it uses leverage.",
    ])


def offer(rng):
    return rng.choice([
        "Want me to open trade.xyz?",
        "I can open trade.xyz in Hermit's browser if you want to look at it there.",
        "trade.xyz is where you'd trade it, and I can open it for you.",
        "Say the word and I'll open trade.xyz.",
        "If you want to trade it, I can open trade.xyz for you.",
        "Want me to take you to trade.xyz?",
    ])


def head_price(rng, sym, r):
    L, l = label(sym, rng, True), label(sym, rng, False)
    P = fmt_price(sym, r["price"])
    c = r["change_24h_pct"]
    if c is None:
        return rng.choice([f"{L} is {P} right now.", f"{L} is at {P}.", f"{L} is trading at {P} right now.",
                           f"Right now {l} is {P}.", f"It's {P} for {l} right now."])
    d, cs = dirw(c), fmt_chg(c)
    return rng.choice([
        f"{L} is {P}, {d} {cs}% in 24h.",
        f"{L} is at {P}, {d} {cs}% over the last 24 hours.",
        f"{L} is trading at {P} right now, {d} {cs}% in 24h.",
        f"{L} is {d} {cs}% in 24h, now at {P}.",
        f"Right now {l} is {P} ({d} {cs}% in 24h).",
        f"{L} is sitting at {P}, {d} {cs}% in the last 24 hours.",
    ])


def head_upq(rng, sym, r, mode):
    L, l = label(sym, rng, True), label(sym, rng, False)
    P = fmt_price(sym, r["price"])
    c = r["change_24h_pct"]
    if c is None:
        return rng.choice([
            f"I don't have a 24h change for {l} right now, so I can't say if it's up or down. The price is {P}.",
            f"I can't see a 24h change for {l} at the moment, only the price, which is {P}.",
            f"There's no 24h change available for {l} right now, so I can't tell you if it's up or down. It's at {P}."])
    d, cs = dirw(c), fmt_chg(c)
    if mode == "either":
        return rng.choice([f"{L} is {d} {cs}% in 24h, now at {P}.", f"{d.capitalize()} {cs}% in 24h. {L} is at {P}.",
                           f"{L} is {d} {cs}% over the last 24 hours, at {P}."])
    yes = (c > 0) if mode == "up" else (c < 0)
    pre = "Yes" if yes else "No"
    return rng.choice([f"{pre}, {l} is {d} {cs}% in 24h, now at {P}.", f"{pre}. {L} is {d} {cs}% over the last 24 hours, at {P}.",
                       f"{pre}, it's {d} {cs}% in 24h. {L} is at {P}."])


def head_multi(rng, syms, rs, cmp_):
    Ls = [label(syms[0], rng, True), label(syms[1], rng, False)]
    Lcap = [label(syms[0], rng, True), label(syms[1], rng, True)]
    Ps = [fmt_price(s, r["price"]) for s, r in zip(syms, rs)]

    def cpart(r, full):
        c = r["change_24h_pct"]
        if c is None:
            return ""
        return f", {dirw(c)} {fmt_chg(c)}%" + (" in 24h" if full else "")
    style = rng.randrange(3)
    if style == 0:
        s = f"{Lcap[0]} is {Ps[0]}{cpart(rs[0], True)}, and {Ls[1]} is {Ps[1]}{cpart(rs[1], False)}."
    elif style == 1:
        s = f"{Lcap[0]}: {Ps[0]}{cpart(rs[0], True)}. {Lcap[1]}: {Ps[1]}{cpart(rs[1], False)}."
    else:
        s = f"{Lcap[0]} is at {Ps[0]}{cpart(rs[0], True)}. {Lcap[1]} is at {Ps[1]}{cpart(rs[1], False)}."
    if cmp_:
        c0, c1 = rs[0]["change_24h_pct"], rs[1]["change_24h_pct"]
        if c0 is None or c1 is None:
            miss = syms[0] if c0 is None else syms[1]
            s += f" I don't have a 24h change for {label(miss, rng, False)}, so I can't say which is doing better."
        else:
            win = 0 if c0 >= c1 else 1
            s += rng.choice([f" {label(syms[win], rng, True)} is doing better over 24h.", f" {label(syms[win], rng, True)} has the better 24h move."])
    return s


def head_err(rng, sym):
    u = UNKNOWN[sym]
    L, fact = u["name"], u["fact"]
    tail = rng.choice(["So there's no market price to look up.", "There's nothing live I could show you for it.",
                       "So there's no live market price for it."]) if u["private"] else \
        rng.choice(["A brokerage app or a finance site will have a live quote.", "Your broker or a finance site will show the current price.",
                    "For a live number, check a brokerage app or a finance site."])
    pre = rng.choice([
        f"I don't have a live price for {L} here. From what I know, ",
        f"I don't have a live price for {L} here, so I won't guess a number. From what I know, ",
        f"There's no live price for {L} here. From what I know, ",
        f"I can't get a live price for {L} here. From what I know, ",
        f"Hermit has no live price for {L}, and I won't make one up. From what I know, ",
    ])
    return pre + fact + " " + tail


def err_buy_tail(rng, sym):
    if UNKNOWN[sym]["private"]:
        return rng.choice(["There are no public shares to buy.", "You can't buy shares of it on a public market."])
    return rng.choice(["Hermit can't buy shares either, so a broker is the place for that.", "You can't buy shares in Hermit, so you'd use a broker."])


OPEN_DONE = [
    "Opened trade.xyz in Hermit's browser. Nothing connects to your wallet until you approve it.",
    "I opened trade.xyz in Hermit's browser. Nothing connects to your wallet until you approve it.",
    "trade.xyz is open in Hermit's browser now. It won't connect to your wallet unless you approve it.",
    "Opened. trade.xyz is in Hermit's browser, and nothing connects to your wallet until you approve it.",
    "trade.xyz is open now. Nothing connects to your wallet until you approve it.",
    "Opened trade.xyz for you. Your wallet stays unconnected until you approve it.",
]
NO_REPLY = ["No problem. Ask me any time.", "Okay. I'm here if you want another price.", "Sure. Ask me if you want to check anything else.",
            "Okay, no trade.xyz then. Just ask if you want another price."]


# ============================================================ scenarios
def make_system(rng):
    bio = "Face ID" if rng.random() < 0.70 else "Touch ID"
    return SYSTEM_TPL.replace("{BIO}", bio), bio


def new_conv(rng, w):
    sysm, bio = make_system(rng)
    c = Conv(rng, w, system=sysm, tools=True)
    return c, bio


def finish(c, kind, bio, extra):
    ex = c.ex(kind)
    ex["meta"] = {"source": "stocks_price", "kind": kind, "bio": bio, "lang": "en", "world": c.world0, "world_final": c.w.state(),
                  "expect": c.expect, **extra}
    return ex


def sc_price(rng, mode, f):
    w = StockWorld(rng)
    sym = f["sym"]
    if f["null"]:
        w.null_change(sym)
    c, bio = new_conv(rng, w)
    ti, t = pick_tpl(rng, PRICE_TR, PRICE_EV, mode, STOCKS[sym]["kind"] == "stock")
    word = pick_word(rng, sym, mode)
    u, ty = build_user(rng, strip_tag(t), [word])
    c.user(u)
    r = c.call(("get_price", {"symbol": sym}))
    c.say(head_price(rng, sym, r) + " " + wording(rng, sym))
    return finish(c, "price", bio, {"slots": [{"word": word, "symbol": sym}], "gold_calls": [{"name": "get_price", "arguments": {"symbol": sym}}],
                                    "change_null": f["null"], "typo": ty, "tpl": f"{mode[:2]}:{ti}", "checks": ["no_invented_price", "right_tool", "decimals_ok", "perp_wording"]})


def sc_upq(rng, mode, f):
    w = StockWorld(rng)
    sym = f["sym"]
    if f["null"]:
        w.null_change(sym)
    c, bio = new_conv(rng, w)
    pool_tr, pool_ev = UPQ_TR, UPQ_EV
    pool = pool_ev if mode == "eval" else pool_tr
    ti = rng.randrange(len(pool))
    t, qmode = pool[ti]
    word = pick_word(rng, sym, mode)
    u, ty = build_user(rng, t, [word])
    c.user(u)
    r = c.call(("get_price", {"symbol": sym}))
    c.say(head_upq(rng, sym, r, qmode) + " " + wording(rng, sym))
    return finish(c, "price_direction", bio, {"slots": [{"word": word, "symbol": sym}], "gold_calls": [{"name": "get_price", "arguments": {"symbol": sym}}],
                                              "change_null": f["null"], "typo": ty, "tpl": f"{mode[:2]}:{ti}", "q_mode": qmode,
                                              "checks": ["no_invented_price", "right_tool", "decimals_ok", "perp_wording"]})


def sc_multi(rng, mode, f):
    w = StockWorld(rng)
    a, b = rng.sample(STOCK_ORDER, 2)
    if f["null"]:
        w.null_change(rng.choice([a, b]))
    c, bio = new_conv(rng, w)
    pool = MULTI_EV if mode == "eval" else MULTI_TR
    ti = rng.randrange(len(pool))
    t, cmp_ = pool[ti]
    wa, wb = pick_word(rng, a, mode), pick_word(rng, b, mode)
    u, ty = build_user(rng, t, [wa, wb])
    c.user(u)
    ra, rb = c.call(("get_price", {"symbol": a}), ("get_price", {"symbol": b}))
    c.say(head_multi(rng, [a, b], [ra, rb], cmp_) + " " + wording_multi(rng, [a, b]))
    return finish(c, "price_multi", bio, {"slots": [{"word": wa, "symbol": a}, {"word": wb, "symbol": b}],
                                          "gold_calls": [{"name": "get_price", "arguments": {"symbol": a}}, {"name": "get_price", "arguments": {"symbol": b}}],
                                          "change_null": f["null"], "typo": ty, "tpl": f"{mode[:2]}:{ti}", "cmp": cmp_,
                                          "checks": ["no_invented_price", "right_tool", "decimals_ok", "perp_wording"]})


def sc_buy(rng, mode, f):
    w = StockWorld(rng)
    sym = f["sym"]
    if f["null"]:
        w.null_change(sym)
    c, bio = new_conv(rng, w)
    ti, t = pick_tpl(rng, BUY_TR, BUY_EV, mode, STOCKS[sym]["kind"] == "stock")
    word = pick_word(rng, sym, mode)
    u, ty = build_user(rng, strip_tag(t), [word])
    c.user(u)
    r = c.call(("get_price", {"symbol": sym}))
    c.say(head_price(rng, sym, r) + " " + wording_short(rng, sym) + " " + honest(rng, sym) + " " + offer(rng))
    gold = [{"name": "get_price", "arguments": {"symbol": sym}}]
    kind, tpl2 = "buy_question", None
    follow = f["follow"]
    if follow:
        fpool = (FOLLOW_YES_EV if follow == "yes" else FOLLOW_NO_EV) if mode == "eval" else (FOLLOW_YES_TR if follow == "yes" else FOLLOW_NO_TR)
        fi = rng.randrange(len(fpool))
        u2, ty2 = build_user(rng, fpool[fi], [word])
        c.user(u2)
        ty = ty or ty2
        tpl2 = f"{mode[:2]}:f{fi}"
        if follow == "yes":
            c.call(("open_app", {"name": "trade.xyz"}))
            c.say(rng.choice(OPEN_DONE))
            gold = [{"name": "open_app", "arguments": {"name": "trade.xyz"}}]
            kind = "buy_open_app"
        else:
            c.say(rng.choice(NO_REPLY))
            gold = []
            kind = "buy_decline"
    checks = ["no_invented_price", "right_tool", "decimals_ok", "perp_wording", "no_prepare_for_stocks"] if not follow else ["right_tool", "no_prepare_for_stocks"]
    return finish(c, kind, bio, {"slots": [{"word": word, "symbol": sym}], "gold_calls": gold, "first_turn_gold": [{"name": "get_price", "arguments": {"symbol": sym}}],
                                 "change_null": f["null"], "typo": ty, "tpl": f"{mode[:2]}:{ti}", "tpl2": tpl2, "follow": follow, "checks": checks})


def sc_err(rng, mode, f):
    w = StockWorld(rng)
    c, bio = new_conv(rng, w)
    if mode == "eval" and rng.random() < 0.7:
        sym = rng.choice(UNK_EVAL_ONLY)
    else:
        sym = rng.choice(UNK_TRAIN)
    buy = f["buy"]
    if buy:
        ti, t = pick_tpl(rng, BUY_TR, BUY_EV, mode, True)
    else:
        ti, t = pick_tpl(rng, PRICE_TR, PRICE_EV, mode, True)
    word = pick_word(rng, sym, mode)
    u, ty = build_user(rng, strip_tag(t), [word])
    c.user(u)
    c.call(("get_price", {"symbol": sym}))
    ans = head_err(rng, sym)
    if buy:
        ans += " " + err_buy_tail(rng, sym)
    c.say(ans)
    return finish(c, "price_error_buy" if buy else "price_error", bio,
                  {"slots": [{"word": word, "symbol": sym}], "gold_calls": [{"name": "get_price", "arguments": {"symbol": sym}}],
                   "change_null": False, "typo": ty, "tpl": f"{mode[:2]}:{ti}", "unknown": True,
                   "checks": ["no_invented_price", "right_tool", "no_prepare_for_stocks", "no_dollar_in_error"]})


SC = {"price": sc_price, "price_direction": sc_upq, "price_multi": sc_multi, "buy": sc_buy, "err": sc_err}


# ============================================================ planning
def alloc(n, weights):
    tot = sum(weights.values())
    raw = {k: n * v / tot for k, v in weights.items()}
    cnt = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[: n - sum(cnt.values())]:
        cnt[k] += 1
    return cnt


TRAIN_W = {"price": 0.38, "price_direction": 0.10, "price_multi": 0.07, "buy": 0.28, "err": 0.17}
EVAL_COUNTS = {"price": 8, "price_direction": 3, "price_multi": 2, "buy": 6, "err": 5}


def make_plan(rng, counts, null_n, yes_n, no_n, errbuy_n):
    plan = []
    for k, n in counts.items():
        plan += [{"fn": k, "null": False, "follow": None, "buy": False} for _ in range(n)]
    succ = [p for p in plan if p["fn"] != "err"]
    for p in rng.sample(succ, null_n):
        p["null"] = True
    buys = [p for p in plan if p["fn"] == "buy"]
    rng.shuffle(buys)
    for p in buys[:yes_n]:
        p["follow"] = "yes"
    for p in buys[yes_n:yes_n + no_n]:
        p["follow"] = "no"
    errs = [p for p in plan if p["fn"] == "err"]
    for p in rng.sample(errs, errbuy_n):
        p["buy"] = True
    # ticker deck so every symbol shows up evenly
    deck = []
    for p in plan:
        if p["fn"] in ("price", "price_direction", "buy"):
            if not deck:
                deck = STOCK_ORDER[:]
                rng.shuffle(deck)
            p["sym"] = deck.pop()
    rng.shuffle(plan)
    return plan


def build_rows(seed, mode, counts, nulls, yes_n, no_n, errbuy_n, exclude):
    rng = random.Random(seed)
    plan = make_plan(rng, counts, nulls, yes_n, no_n, errbuy_n)
    seen = set(exclude)
    rows = []
    for p in plan:
        for _ in range(400):
            ex = SC[p["fn"]](random.Random(rng.getrandbits(64)), mode, p)
            us = [norm(m["content"]) for m in ex["messages"] if m["role"] == "user"]
            if len(set(us)) == len(us) and not (set(us) & seen):
                seen.update(us)
                rows.append(ex)
                break
        else:
            raise RuntimeError("could not find a unique user text for " + p["fn"])
    return rows


def eval_item(ex, i):
    msgs = ex["messages"]
    last_u = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    m = ex["meta"]
    gold = m["gold_calls"]
    results = [json.loads(x["content"]) for x in msgs[last_u + 1:] if x["role"] == "tool"]
    answer_free = not gold and m["kind"] == "buy_decline"
    checks = {
        "scorers": m["checks"],
        "gold_calls": gold,                         # calls expected in the last turn (empty list = no call)
        "forbid_tools": ["prepare_trade", "prepare_swap", "prepare_buy", "prepare_send", "prepare_close", "prepare_multisend",
                         "prepare_deposit", "prepare_withdraw"],
        "bio": m["bio"],
    }
    if m["kind"] in ("price", "price_direction", "price_multi", "buy_question"):
        ok = [r for r in results if "price" in r]
        checks["price_strings"] = [fmt_price(r["symbol"], r["price"]) for r in ok]       # exact strings allowed in the answer
        checks["price_values"] = [r["price"] for r in ok]
        checks["change_values"] = [r["change_24h_pct"] for r in ok]                      # null means: no % figure allowed
        checks["max_decimals"] = 2
        checks["perp_wording_required"] = True
        checks["offer_open_app"] = m["kind"] == "buy_question"                           # answer must offer trade.xyz, not call it
    elif m["kind"] in ("price_error", "price_error_buy"):
        checks["no_dollar_figure"] = True
        checks["no_digits"] = True
        checks["error_tool_result"] = results[0]["error"]
        checks["must_say_no_live_price"] = True
    elif m["kind"] == "buy_open_app":
        checks["answer_mentions"] = ["trade.xyz"]
        checks["no_dollar_figure"] = True
    elif answer_free:
        checks["no_dollar_figure"] = True
        checks["no_tool_call"] = True
    return {"id": f"{m['kind']}-s{i:02d}", "kind": m["kind"], "tools": True, "bio": m["bio"], "context": msgs[:last_u + 1],
            "reference": msgs[last_u + 1:], "expect": m["expect"], "world": m["world"], "checks": checks,
            "meta": {"source": "stocks_price", "kind": m["kind"], "tpl": m["tpl"], "tpl2": m.get("tpl2"), "slots": m["slots"],
                     "first_turn_gold": m.get("first_turn_gold")}}


def msgs_of(rec):
    if "messages" in rec:
        return rec["messages"]
    return rec["context"] + rec["reference"]


# ============================================================ self-check
BANNED = [r"[·•]", r"\bRelay\b", r"prepare_\w+", r"can't see the future", r"\bas an ai\b", r"great question", r"financial advice",
          r"what(?:'s| is) (?:the )?(?:current )?price", r"your budget", r"how much (?:do you|are you)", r"went through", r"successfully",
          r"\bI(?:'ve| have) (?:prepared|sent|bought|swapped)", r"funds (?:are|have)", r"\byou should buy\b", r"\bI'd buy\b", r"\bdon't buy\b",
          r"guarantee", r"\bxyz:", r"[\U0001F300-\U0001FAFF☀-➿]", r"^\s*done\b"]
BANNED_RE = [re.compile(p, re.I) for p in BANNED]
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def dec(s):
    return Decimal(s.replace(",", "").rstrip("."))


def check_args(tc):
    assert tc["name"] in TOOL_SCHEMA, ("unknown tool", tc["name"])
    sch = TOOL_SCHEMA[tc["name"]]
    args = tc["arguments"]
    assert isinstance(args, dict), tc
    for req in sch.get("required", []):
        assert req in args, ("missing arg", tc, req)
    for k, v in args.items():
        assert k in sch["properties"], ("unknown arg", tc, k)
        t = sch["properties"][k]["type"]
        assert (t == "string" and isinstance(v, str)) or (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)), (tc, k, v)
        if "enum" in sch["properties"][k]:
            assert v in sch["properties"][k]["enum"], (tc, k, v)


def check_record(rec, is_eval, errs, tag):
    def bad(msg):
        errs.append(f"{tag}: {msg}")
    msgs = msgs_of(rec)
    meta = rec["meta"]
    if is_eval:
        bio = rec["bio"]
        world = rec["world"]
        kind = rec["kind"]
        gold = rec["checks"]["gold_calls"]
    else:
        bio = meta["bio"]
        world = meta["world"]
        kind = meta["kind"]
        gold = meta["gold_calls"]
        if rec.get("tools") is not True:
            bad("tools flag")
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    # system prompt
    if msgs[0]["role"] != "system" or not msgs[0]["content"].startswith(SYSTEM_TPL.replace("{BIO}", bio)):
        bad("system prompt is not SYSTEM_AGENT_v2 with the bio word")
    for m in msgs:
        if other in m["content"] or "{BIO}" in m["content"]:
            bad("wrong or unsubstituted bio word")
    if msgs[-1]["role"] != "assistant" or not msgs[-1]["content"].strip() or msgs[-1].get("tool_calls"):
        bad("does not end with an assistant text")
    # structure + tool names, args
    sim = StockWorld.from_state(world)
    pending, calls_all, user_texts = [], [], []
    last_u = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    last_calls = []
    for j, m in enumerate(msgs):
        if m["role"] == "user":
            user_texts.append(m["content"])
        if m["role"] == "assistant" and m.get("tool_calls"):
            if pending:
                bad("tool calls before results returned")
            for tc in m["tool_calls"]:
                try:
                    check_args(tc)
                except AssertionError as e:
                    bad(f"args: {e}")
                if tc["name"].startswith("prepare_"):
                    bad("prepare_* call in stocks_price")
                if tc["name"] not in ("get_price", "open_app"):
                    bad("unexpected tool " + tc["name"])
                pending.append(tc)
                calls_all.append(tc)
                if j > last_u:
                    last_calls.append(tc)
            rt, bd = H.parse_calls(H.assistant_text(m))
            if bd or [{"name": t["name"], "arguments": t["arguments"]} for t in m["tool_calls"]] != rt:
                bad("render round trip of tool calls")
        elif m["role"] == "tool":
            if not pending:
                bad("tool result without call")
                continue
            tc = pending.pop(0)
            replay = sim.call(tc["name"], tc["arguments"])
            if json.loads(m["content"]) != replay or m["content"] != json.dumps(replay, ensure_ascii=False):
                bad("tool result != replay")
        elif m["role"] == "assistant" and pending:
            bad("assistant text while results pending")
    if pending:
        bad("unanswered tool calls")
    # gold match (last turn) and slots
    if [{"name": t["name"], "arguments": t["arguments"]} for t in last_calls] != gold:
        bad("last-turn calls != gold")
    slots = meta["slots"]
    first_calls = [tc for tc in calls_all if tc["name"] == "get_price"]
    if [tc["arguments"] for tc in first_calls] != [{"symbol": s["symbol"]} for s in slots]:
        bad("get_price symbols != slots")
    ut0 = user_texts[0]
    for s in slots:
        if s["word"] not in ut0 and s["word"].upper() not in ut0 and s["word"].lower() not in ut0:
            bad("slot word not in user text")
        if s["symbol"] != s["symbol"].upper() or ":" in s["symbol"]:
            bad("symbol not plain uppercase ticker")
        if s["symbol"] in STOCKS or s["symbol"] in UNKNOWN:
            aliases = (STOCKS[s["symbol"]]["tr"] + STOCKS[s["symbol"]]["ev"]) if s["symbol"] in STOCKS else \
                (UNKNOWN[s["symbol"]]["tr"] + UNKNOWN[s["symbol"]]["ev"])
            if s["word"] not in aliases:
                bad("slot word not an alias of the symbol")
        else:
            bad("unknown symbol in slots")
    if any(re.search(r"0x[0-9a-fA-F]{6,}", t) for t in user_texts):
        bad("unexpected address in user text")
    # per assistant text
    results = {}
    for m in msgs:
        if m["role"] == "tool":
            r = json.loads(m["content"])
            if "price" in r:
                results[r["symbol"]] = r
    for j, m in enumerate(msgs):
        if m["role"] != "assistant" or m.get("tool_calls"):
            continue
        a = m["content"]
        for rx in BANNED_RE:
            if rx.search(a):
                bad(f"banned phrase /{rx.pattern}/ in: {a}")
        if len(a.split()) > 80:
            bad("answer longer than 80 words")
        if "\n" in a:
            bad("newline in answer")
        # which price answer is this? (the first assistant text after the get_price results)
        is_price_answer = any(x["role"] == "tool" and "price" in x["content"] or "No Hyperliquid" in x["content"] for x in msgs[:j]) and \
            msgs[j - 1]["role"] == "tool" and msgs[j - 1]["content"].startswith("{") and "opened" not in msgs[j - 1]["content"]
        if not is_price_answer:
            continue
        # tool results of this answer
        k = j - 1
        rs = []
        while msgs[k]["role"] == "tool":
            rs.append(json.loads(msgs[k]["content"]))
            k -= 1
        rs.reverse()
        if all("error" in r for r in rs):
            if SHARES_FIGS.search(a) or re.search(r"\d", a):
                bad(f"error branch has a number: {a}")
            if "live price" not in a or "From what I know" not in a:
                bad("error branch wording")
            if re.search(r"%", a):
                bad("error branch %")
            continue
        assert all("price" in r for r in rs)
        allowed = set()
        pct_text = a
        for r in rs:
            sym = r["symbol"]
            ps = fmt_price(sym, r["price"])
            if ps not in a:
                bad(f"price {ps} missing from answer: {a}")
            allowed.add(Decimal(str(r["price"])))
            c = r["change_24h_pct"]
            if c is not None:
                allowed.add(abs(Decimal(str(c))))
        if not re.search(r"\bperps?\b", a, re.I) or not re.search(r"\btracks?\b", a, re.I):
            bad(f"perp wording missing: {a}")
        # $ figures must be in a result
        for mm in re.finditer(r"\$([\d,]+\.?\d*)", a):
            if dec(mm.group(1)) not in allowed:
                bad(f"$ figure {mm.group(0)} not in tool result: {a}")
        scrub = re.sub(r"S&P ?500|sp500|24h|24 hours|24/7|trade\.xyz", "", a, flags=re.I)
        for mm in NUM_RE.finditer(scrub):
            v = dec(mm.group(0))
            if v not in allowed and not any(NUM_RE.search(t) and mm.group(0) in t for t in user_texts):
                bad(f"number {mm.group(0)} not backed by a result: {a}")
        # decimals: every printed price has exactly the result value, 2 dp
        for mm in NUM_RE.finditer(scrub):
            if "." in mm.group(0) and len(mm.group(0).rstrip(".").split(".")[1]) > 2:
                bad("more than 2 decimals: " + mm.group(0))
        # % and direction
        seen_pct = []
        for mm in re.finditer(r"\b(up|down) ([\d.]+)%", a, re.I):
            seen_pct.append((mm.group(1).lower(), dec(mm.group(2))))
        for mm in re.finditer(r"(?<![\d.])([\d.]+)%", a):
            if not any(dec(mm.group(1)) == v for _, v in seen_pct):
                bad("percent without up/down: " + a)
        for r in rs:
            c = r["change_24h_pct"]
            if c is None:
                continue
            if (dirw(c), abs(Decimal(str(c)))) not in [(d.lower(), v) for d, v in seen_pct] and kind != "price_multi":
                bad(f"direction/percent mismatch for {r['symbol']}: {a}")
        for d, v in seen_pct:
            if not any(abs(Decimal(str(r["change_24h_pct"]))) == v and dirw(r["change_24h_pct"]) == d for r in rs if r["change_24h_pct"] is not None):
                bad(f"up/down figure not from a result: {a}")
        if all(r["change_24h_pct"] is None for r in rs) and "%" in a:
            bad("percent shown although change is null")
        if kind == "buy_question" or (kind in ("buy_open_app", "buy_decline") and j < last_u):
            if "trade.xyz" not in a.split(".")[-2] + a[-60:] and "trade.xyz" not in a[-110:]:
                bad("buy answer does not offer trade.xyz: " + a)
            if not re.search(r"leverage", a):
                bad("buy answer without the leverage sentence: " + a)
    if kind == "buy_open_app":
        if msgs[-1]["content"].find("trade.xyz") < 0 or "approve" not in msgs[-1]["content"]:
            bad("open_app answer")
    return user_texts


def self_check(train_path, eval_path):
    errs, all_norm, kinds = [], {}, Counter()
    stats = {}
    for path, is_eval in ((train_path, False), (eval_path, True)):
        recs = []
        with open(path) as f:
            for i, line in enumerate(f):
                try:
                    recs.append(json.loads(line))
                except Exception as e:
                    errs.append(f"{path}:{i + 1}: invalid JSON {e}")
        names = "eval" if is_eval else "train"
        stats[names] = len(recs)
        tpls = Counter()
        nulls = errs_n = 0
        for i, rec in enumerate(recs):
            tag = f"{names}[{i}]"
            try:
                uts = check_record(rec, is_eval, errs, tag)
            except Exception as e:                      # a crash in the checker is a failure too
                errs.append(f"{tag}: checker crashed {type(e).__name__}: {e}")
                continue
            meta = rec["meta"]
            for t in uts:
                n = norm(t)
                if n in all_norm:
                    errs.append(f"{tag}: duplicate user text {t!r} (also {all_norm[n]})")
                all_norm[n] = tag
            kinds[(names, meta["kind"])] += 1
            tp = meta["tpl"]
            if (is_eval and not tp.startswith("ev:")) or (not is_eval and not tp.startswith("tr:")):
                errs.append(f"{tag}: template pool leak {tp}")
            if is_eval:
                for k in ("id", "kind", "context", "reference", "expect", "world", "checks", "bio"):
                    if k not in rec:
                        errs.append(f"{tag}: eval item lacks {k}")
                if not rec["checks"]["gold_calls"] and meta["kind"] != "buy_decline":
                    errs.append(f"{tag}: eval item without gold calls")
            if meta.get("change_null"):
                nulls += 1
        stats[names + "_null"] = nulls
        stats[names + "_bio_touch"] = sum(1 for r in recs if (r["bio"] if is_eval else r["meta"]["bio"]) == "Touch ID")
    return errs, stats, kinds


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--eval-seed", type=int, default=9308)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--out", default=OUT_TRAIN)
    ap.add_argument("--out-eval", default=OUT_EVAL)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if not a.check_only:
        # eval first (held-out templates, held-out entities), then train with every eval text excluded
        ec = EVAL_COUNTS if a.n_eval == 24 else alloc(a.n_eval, EVAL_COUNTS)
        eval_rows = build_rows(a.eval_seed, "eval", ec, nulls=max(2, round(0.15 * (a.n_eval - ec["err"]))), yes_n=1, no_n=1,
                               errbuy_n=min(2, ec["err"]), exclude=set())
        excl = {norm(m["content"]) for r in eval_rows for m in r["messages"] if m["role"] == "user"}
        counts = alloc(a.n, TRAIN_W)
        succ = a.n - counts["err"]
        buys = counts["buy"]
        train_rows = build_rows(a.seed, "train", counts, nulls=round(0.15 * succ), yes_n=round(0.30 * buys), no_n=round(0.08 * buys),
                                errbuy_n=round(0.3 * counts["err"]), exclude=excl)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        os.makedirs(os.path.dirname(a.out_eval), exist_ok=True)
        write_jsonl(a.out, train_rows)
        write_jsonl(a.out_eval, [eval_item(r, i) for i, r in enumerate(eval_rows)])
    errs, stats, kinds = self_check(a.out, a.out_eval)
    if errs:
        for e in errs[:60]:
            print("FAIL", e, file=sys.stderr)
        print(f"stocks_price: self-check FAILED ({len(errs)} problems)")
        sys.exit(1)
    print("kinds:", dict(sorted(kinds.items())), "stats:", stats, file=sys.stderr)
    print(f"stocks_price: {stats['train']} train, {stats['eval']} eval, self-check ok")


if __name__ == "__main__":
    main()
