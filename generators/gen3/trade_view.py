#!/usr/bin/env python3
"""trade_view: trade ideas and views for Hermit (next run, data3 / eval3).

Purpose: the user asks for a trade idea or a view on a coin. The model calls get_price and
hyperliquid_account IN ONE assistant message, never asks the user for a price or budget, then answers
with the live price and 24h move, a measured view with one or two risks, and the order it would prepare.
It does NOT call prepare_trade on the first turn. 25% of rows have a second user turn
(ok do it -> prepare_trade, make it 5x -> position_math, what's my loss if the stop hits -> calculator).

Everything is deterministic Python (no LLM text). Every number in an answer comes from a tool result
(calculator results for derived numbers) or from the user's text.

Usage:
  python3 trade_view.py [--seed 3] [--n 600] [--eval-seed 9303] [--eval-n 60]
Writes  ../data3/trade_view.jsonl  and  ../eval3/trade_view.jsonl , then runs the self-check on every
record (in memory and again from the files on disk). Exit code 1 if anything fails.

Reuses gen2 helpers by import (hermit_fmt, sim, build_agent.Conv); gen2 files are not modified or copied.
The tool schema and the system prompt come from ../spec (v2 registry, 19 tools; SYSTEM_AGENT_v2.txt).
"""
import argparse, json, math, os, random, re, sys
from collections import Counter
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(HERE, ".."))
GEN2 = os.path.join(TRAIN_DIR, "gen2")
SPEC = os.path.join(TRAIN_DIR, "spec")
sys.path.insert(0, GEN2)

import hermit_fmt as H          # gen2 helpers (calculator, formatting, renderer)
import sim                      # gen2 simulator (price bands, liquidation maths)
from build_agent import Conv    # gen2 conversation builder

# ----------------------------------------------------------------------------- v2 registry
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
# make the shared gen2 module see the 19-tool registry (in memory only; gen2 files are untouched)
H.TOOLS, H.TOOL_NAMES, H.TOOL_SCHEMA = TOOLS_V2, list(SCHEMA), SCHEMA
H.TOOLS_BLOCK = H.tools_block()
SYSTEM_V2 = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().rstrip("\n")
assert "{BIO}" in SYSTEM_V2

COINS = ["BTC", "ETH", "SOL", "HYPE", "BNB", "XRP", "DOGE", "AVAX", "LINK", "SUI", "ARB", "OP", "APT", "NEAR", "TIA", "INJ"]
COIN_W = [16, 16, 16, 9, 5, 6, 6, 4, 4, 5, 3, 3, 3, 3, 3, 3]
BANDS = {c: sim.ASSETS[c] for c in COINS if c in sim.ASSETS}
BANDS["TIA"] = (1.0, 6.5, 3)
MEMES = ["PEPE", "BONK", "SHIB", "FLOKI", "BRETT", "TURBO", "MOG", "BABYDOGE"]
ALIAS = {
    "BTC": ["BTC", "btc", "bitcoin", "Bitcoin", "$BTC"], "ETH": ["ETH", "eth", "ether", "ethereum", "Ethereum", "$ETH"],
    "SOL": ["SOL", "sol", "solana", "Solana", "$SOL"], "HYPE": ["HYPE", "hype", "$HYPE"], "BNB": ["BNB", "bnb", "$BNB"],
    "XRP": ["XRP", "xrp", "ripple", "$XRP"], "DOGE": ["DOGE", "doge", "dogecoin", "Dogecoin", "$DOGE"],
    "AVAX": ["AVAX", "avax", "avalanche"], "LINK": ["LINK", "link", "chainlink", "Chainlink"], "SUI": ["SUI", "sui"],
    "ARB": ["ARB", "arb", "arbitrum"], "OP": ["OP", "op", "optimism"], "APT": ["APT", "apt", "aptos"],
    "NEAR": ["NEAR", "near"], "TIA": ["TIA", "tia", "celestia"], "INJ": ["INJ", "inj", "injective"],
    "PEPE": ["PEPE", "pepe", "$PEPE"], "BONK": ["BONK", "bonk", "$BONK"], "SHIB": ["SHIB", "shib", "shiba"],
    "FLOKI": ["FLOKI", "floki"], "BRETT": ["BRETT", "brett"], "TURBO": ["TURBO", "turbo"], "MOG": ["MOG", "mog"],
    "BABYDOGE": ["BABYDOGE", "babydoge"],
}
IDEA_COINS = ["BTC", "ETH", "SOL"]

# ----------------------------------------------------------------------------- user phrasing
# (text, side) ; side in neutral / long / short / ask. Index % 5 == 0 -> held out for eval only.
NEUTRAL = [
    "what do you think of {c} here", "thoughts on {c}", "{c} thoughts?", "how does {c} look right now", "what's your view on {c}",
    "give me a trade idea on {c}", "any trade on {c} today", "what would you do with {c} right now", "is {c} worth trading rn",
    "what about {c}", "what's the price of {c} and what's your budget", "any setup on {c}", "talk me through {c} right now",
    "tell me about {c} as a trade", "what's the move on {c}", "how are you feeling about {c}", "{c} right now, worth it or not",
    "should i be doing anything with {c}", "gimme your take on {c}", "{c} looks interesting, what do you reckon",
    "do you like {c} at this price", "is there a trade in {c}", "I keep seeing {c} everywhere, is it a trade", "view on {c}?",
    "{c} trade idea pls", "what's happening with {c} and is it tradeable",
]
LONG = [
    "should I long {c}", "is {c} a buy", "is it a good time to buy {c}", "worth going long {c}?", "thinking of buying {c}, good idea?",
    "{c} long here?", "should i buy {c} now", "is {c} a good entry", "can I long {c} here", "buy {c} or wait",
    "long {c}, yes or no?", "Is {c} a good buy right now?", "would you buy {c} here", "tempted to long {c}, talk me out of it or in",
    "is now the time to long {c}", "i wanna go long {c}, thoughts", "is this a good price to buy {c}", "{c} buy or pass",
]
SHORT = [
    "worth shorting {c}", "should I short {c}", "{c} short here?", "is {c} a good short", "thinking about shorting {c}, thoughts",
    "would you short {c} right now", "can I short {c} here", "short {c}, yes or no?", "is it time to short {c}",
    "tempted to short {c} rn", "{c} short or pass", "should i bet against {c}",
]
ASK = [
    "{c} long or short?", "long or short {c}", "{c} long or short rn", "do I long or short {c} here", "which way on {c}, long or short",
    "if you had to pick long or short on {c}?", "{c}: long or short", "long {c} or short {c}",
]
STORY = [
    ("been watching {c} all week, it bounced around and now i dont know if i should get in. what do you think", "neutral"),
    ("my friend keeps telling me to buy {c} but i only trade perps on hyperliquid, is it worth a long right now", "long"),
    ("i got stopped out on {c} yesterday and want back in, thoughts on a trade here", "neutral"),
    ("woke up and {c} is moving. should i jump in or wait", "neutral"),
    ("just got paid and want to put something into a {c} trade, what do you reckon", "neutral"),
    ("ok so i missed the last move on {c} and im kinda fomo-ing. should i long it now", "long"),
    ("everyone on my timeline is bearish on {c} lately, worth a short or is that just noise", "short"),
    ("i've been holding {c} spot for months, thinking about a leveraged long on top of it. good idea or no", "long"),
    ("been up all night staring at {c}, someone tell me what to do with it", "neutral"),
    ("first time trading perps. {c} seems popular, is it a decent first trade", "neutral"),
    ("had a rough week, want to make it back with {c}. is a trade there worth it right now", "neutral"),
    ("a guy in my discord says {c} is going to dump. should i short it", "short"),
    ("lunch break, 10 minutes, one quick view on {c} please and what you'd actually do", "neutral"),
    ("been thinking about {c} all day, long, short or sit out", "ask"),
    ("ok be honest, is {c} a trade right now or am i just bored", "neutral"),
]
NOCOIN = [
    "give me a trade idea", "any good trade today", "pick me something", "I have {amt} what should I do", "what should I trade today",
    "got any trade ideas", "what's good right now", "find me a trade", "what's worth trading today",
    "i wanna make some money today what do i trade", "bored, give me something to trade", "any setups worth taking",
    "I have {amt} to play with, ideas?", "trade idea?", "what would you trade right now", "need a trade for today, whatcha got",
    "anything look good to you rn", "got {amt} on hyperliquid, what's the play", "what's the best trade right now",
    "surprise me with a trade", "hey hermit, what would you trade today", "give me one trade idea, keep it simple",
    "I'm new, what should I try trading first", "got {amt} sitting there doing nothing, put it to work?",
    "what's moving today, anything worth a trade", "morning. any trade ideas for today", "what do you like today",
    "should i be in a trade right now", "ideas for a quick trade", "what's your best idea today", "ok i have {amt}, pick something for me",
]
PREFIX = ["hey", "yo", "gm", "ok so", "hermit,", "quick one:", "hi", "hmm", "alright", "so", "ok", "hey hermit,"]
SUFFIX = ["pls", "please", "thanks", "lol", "asap", "be honest", "keep it short", "no bs", "thx"]
LEV_SUFFIX = [" at 5x", " with 5x leverage", " using 5x", " 5x"]
FU_DO = ["ok do it", "yes", "yep prepare it", "go for it", "do it", "ok prepare it", "sure go ahead", "yeah set it up", "ye do it pls",
         "lets go", "prepare it", "ok", "alright do it", "yes please", "sounds good, set it up", "go"]
FU_5X = ["make it 5x", "can we do 5x", "5x instead", "bump it to 5x", "make it 5x pls", "use 5x", "what about 5x", "ok but 5x",
         "change it to 5x", "make that 5x"]
FU_LOSS = ["what's my loss if the stop hits", "how much do i lose if the stop gets hit", "whats the loss if stop triggers",
           "what do i lose if it hits the stop", "if the stop hits how much is that", "how much would i lose at the stop",
           "what's the max loss on that", "and my loss if it stops me out"]


def norm_text(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def amt_text(rng, v):
    f = rng.choice(["${v}", "{v} usd", "{v}$", "{v} bucks", "${v}", "{v} dollars"])
    s = f"{v:,}" if v >= 1000 and rng.random() < 0.5 else str(v)
    if v == 1000 and f == "${v}" and rng.random() < 0.3:
        return "$1k"
    return f.format(v=s)


def typo(rng, parts):
    """parts: list of [text, protected]. Applies ONE typo to an unprotected word (never an address, number or ticker slot)."""
    ops = ["swap", "vowel", "double", "abbr"]
    rng.shuffle(ops)
    for op in ops:
        idxs = [i for i, (t, p) in enumerate(parts) if not p and re.search(r"[a-zA-Z]{4}", t)]
        if op == "abbr":
            for i in idxs:
                t = parts[i][0]
                for a, b in (("please", rng.choice(["pls", "plz"])), ("right now", "rn"), (" you ", " u "), ("thoughts", "thots"),
                             ("thanks", "thx"), ("about", "abt")):
                    if a in t:
                        parts[i][0] = t.replace(a, b, 1)
                        return True
            continue
        rng.shuffle(idxs)
        for i in idxs:
            words = list(re.finditer(r"[a-zA-Z]{4,}", parts[i][0]))
            if not words:
                continue
            m = rng.choice(words)
            w = m.group(0)
            if op == "swap":
                k = rng.randrange(1, len(w) - 1)
                w2 = w[:k] + w[k + 1] + w[k] + w[k + 2:]
            elif op == "vowel":
                vs = [k for k, ch in enumerate(w) if ch in "aeiou" and k > 0]
                if not vs:
                    continue
                k = rng.choice(vs)
                w2 = w[:k] + w[k + 1:]
            else:
                k = rng.randrange(1, len(w))
                w2 = w[:k] + w[k] + w[k:]
            if w2 != w:
                parts[i][0] = parts[i][0][:m.start()] + w2 + parts[i][0][m.end():]
                return True
    return False


def render_user(rng, tpl, alias=None, amt=None, lev_suffix=None):
    pieces = re.split(r"(\{c\}|\{amt\})", tpl)
    parts = []
    for p in pieces:
        if p == "{c}":
            parts.append([alias, True])
        elif p == "{amt}":
            parts.append([amt, True])
        elif p:
            parts.append([p, False])
    # filler words
    if rng.random() < 0.22:
        parts.insert(0, [rng.choice(PREFIX) + " ", False])
    if lev_suffix:
        # lev suffix goes before a trailing question mark / at the end
        if parts and not parts[-1][1] and parts[-1][0].endswith("?"):
            parts[-1][0] = parts[-1][0][:-1]
        parts.append([lev_suffix, True])
    if rng.random() < 0.22:
        parts.append([" " + rng.choice(SUFFIX), False])
    mode = rng.choices(["keep", "lower", "cap"], [0.5, 0.4, 0.1])[0]
    for pc in parts:
        if not pc[1]:
            if mode == "lower":
                pc[0] = pc[0].lower()
    if mode == "cap" and parts and not parts[0][1]:
        parts[0][0] = parts[0][0][:1].upper() + parts[0][0][1:]
    if rng.random() < 0.15:
        typo(rng, parts)
    s = "".join(p[0] for p in parts).strip()
    r = rng.random()
    if r < 0.2:
        s = s.rstrip("?.")
    elif r < 0.27 and not s.endswith("?"):
        s += "?"
    if rng.random() < 0.02:
        s = s.upper()
    return re.sub(r"\s+", " ", s)


# ----------------------------------------------------------------------------- numbers
def nn(x):
    """Normalise a number the way the app prints it: int when integral, else shortest float."""
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        return int(x)
    return x


def ns(x):
    return json.dumps(nn(x))


def commas(s):
    m = re.match(r"^(-?)(\d+)(\.\d+)?$", s)
    if not m:
        return s
    return m.group(1) + f"{int(m.group(2)):,}" + (m.group(3) or "")


def dollars(s):
    return "$" + commas(s if isinstance(s, str) else ns(s))


def stop_dp(price):
    return 0 if price >= 1000 else (2 if price >= 10 else 4)


def parse_calc(s):
    return float(s) if "." in s else int(s)


def view_of(c):
    if abs(c) < 1.5:
        return "flat"
    if c > 5:
        return "up"
    if c < -5:
        return "down"
    return "mild"


VIEW_PHRASE = {"flat": "no clear edge, so I'd keep it small", "up": "already extended, chasing is the risk",
               "down": "a bounce is possible but it can keep falling", "mild": "lean"}

# ----------------------------------------------------------------------------- world (v2 tool shapes)
NO_MARKET_ERR = "No Hyperliquid market for {s}. Answer from what you know instead, and say there's no live price here."


def draw_change(rng):
    b = rng.choices(["flat", "mild", "up", "down"], [0.26, 0.28, 0.23, 0.23])[0]
    if b == "flat":
        v = rng.uniform(0.05, 1.45) * rng.choice([-1, 1])
    elif b == "mild":
        v = rng.uniform(1.55, 4.95) * rng.choice([-1, 1])
    elif b == "up":
        v = rng.uniform(5.05, 13)
    else:
        v = -rng.uniform(5.05, 13)
    v = round(v, 2)
    return nn(v if v != 0 else 0.1)


def draw_price(rng, sym):
    lo, hi, dp = BANDS[sym]
    p = round(rng.uniform(lo, hi), dp)
    return nn(float(p))


class TVWorld:
    """Replayable mini simulator for the tools this generator uses, in the app's exact result shapes."""

    def __init__(self, prices, change, no_market, account, bio, device):
        self.prices, self.change, self.no_market = prices, change, list(no_market)
        self.account, self.bio, self.device = account, bio, device

    def state(self):
        return json.loads(json.dumps({"prices": self.prices, "change": self.change, "no_market": self.no_market,
                                      "account": self.account, "bio": self.bio, "device": self.device}))

    @classmethod
    def from_state(cls, st):
        return cls(st["prices"], st["change"], st["no_market"], st["account"], st["bio"], st["device"])

    def call(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args if isinstance(args, dict) else {}))
        except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as e:
            return {"error": f"bad arguments: {e}"}

    def t_calculator(self, expression):
        try:
            return {"result": H.calc_str(H.calc(str(expression)))}
        except ZeroDivisionError:
            return {"error": "Couldn't evaluate that expression"}
        except Exception:
            return {"error": "Couldn't evaluate that expression"}

    def t_get_price(self, symbol):
        s = str(symbol).upper().replace("$", "").strip()
        if s in self.no_market or s not in self.prices:
            return {"error": NO_MARKET_ERR.format(s=s)}
        return {"change_24h_pct": self.change[s], "price": self.prices[s], "source": "Hyperliquid mark price", "symbol": s}

    def t_hyperliquid_account(self):
        return json.loads(json.dumps(self.account))

    def t_position_math(self, side, entry, size_usd, leverage, price=None, maintenance_margin=None):
        if side not in ("long", "short"):
            return {"error": "side must be long or short"}
        out = sim.position_math(side, float(entry), float(size_usd), float(leverage),
                                None if price is None else float(price), float(maintenance_margin or 0))
        out["liquidation_price"] = out.pop("liquidation_price")
        res = {}
        for k in sorted(out):
            res[k if k != "liquidation_price" else "liquidation_price"] = nn(out[k]) if isinstance(out[k], float) else out[k]
        return res

    def t_prepare_trade(self, coin, side, size_usd, leverage, order_type, limit_price=None, take_profit=None, stop_loss=None):
        if "error" in self.account:
            return {"error": f"No wallet on this {self.device} to trade with"}
        need = float(size_usd) / float(leverage)
        avail = float(self.account["available_usdc"])
        if need > avail:
            return {"error": f"Not enough margin: {avail:.2f} USDC available, {need:.2f} needed", "prepared": False}
        return {"status": "review_opened", "kind": "trade", "executed": False,
                "note": f"Waiting for the user to approve with {self.bio}."}


def make_position(rng, coin, price, side=None):
    lo, hi, dp = BANDS[coin]
    side = side or rng.choice(["long", "short"])
    lev = rng.choice([2, 3, 5, 5, 10])
    entry = nn(float(round(price * (1 + rng.uniform(-0.06, 0.06)), dp)))
    margin = rng.uniform(20, 800)
    notional = margin * lev
    sz_dp = 4 if price >= 1000 else (3 if price >= 100 else (2 if price >= 10 else (1 if price >= 1 else 0)))
    size = nn(round(notional / entry, sz_dp))
    if not size:
        size = nn(round(10 ** -sz_dp, sz_dp)) if sz_dp else 1
    sgn = 1 if side == "long" else -1
    liq = None if rng.random() < 0.05 else nn(round(sim.liq_price(side, entry, lev, 0.005), dp))
    pos = {"coin": coin, "entry": entry, "leverage": lev, "liquidation": liq, "side": side, "size": size,
           "unrealised_pnl": nn(round(sgn * (price - entry) * size, 2))}
    return pos, round(size * entry / lev, 2)


def make_account(rng, mode, prices, device, pos_coin=None, pos_side=None):
    """mode: ok | zero_empty | zero_full | error. Returns the hyperliquid_account result dict."""
    if mode == "error":
        return {"error": f"No wallet on this {device} yet"}
    positions, margin = [], 0.0
    if mode == "zero_empty":
        return {"account_value": 0, "available_usdc": 0, "margin_used": 0, "open_orders": [], "positions": []}
    coins = []
    if pos_coin:
        coins.append((pos_coin, pos_side))
    if (mode == "zero_full" and not coins) or rng.random() < 0.25:
        other = rng.choice([c for c in COINS if c != pos_coin])
        coins.append((other, None))
    for c, sd in coins:
        p, m = make_position(rng, c, prices[c], sd)
        positions.append(p)
        margin += m
    margin = round(margin, 2)
    if mode == "zero_full":
        value, avail = margin, 0
    else:
        a = rng.choices([rng.uniform(30, 300), rng.uniform(300, 3000), rng.uniform(3000, 20000)], [0.3, 0.45, 0.25])[0]
        avail = round(a, 2)
        value = round(margin + avail, 2)
    orders = []
    if rng.random() < 0.15:
        c = rng.choice(COINS)
        lo, hi, dp = BANDS[c]
        orders.append({"coin": c, "price": nn(float(round(prices[c] * rng.choice([0.95, 0.97, 1.03]), dp))), "side": rng.choice(["buy", "sell"]),
                       "size": nn(round(rng.uniform(1, 40) / max(prices[c], 1) * 100, 2) or 1)})
    return {"account_value": nn(value), "available_usdc": nn(avail), "margin_used": nn(margin), "open_orders": orders,
            "positions": positions}


# ----------------------------------------------------------------------------- answer text
def pick(rng, xs):
    return rng.choice(xs)


def price_line(rng, coin, p, c):
    d, x = ("up" if c >= 0 else "down"), ns(abs(c))
    return pick(rng, [
        f"{coin} is at {dollars(p)}, {d} {x}% in the last 24 hours.",
        f"{coin} is trading at {dollars(p)}, {d} {x}% over 24h.",
        f"Right now {coin} is {dollars(p)}, {d} {x}% on the day.",
        f"Live price: {coin} at {dollars(p)}, {d} {x}% in 24h.",
        f"{coin} sits at {dollars(p)} and is {d} {x}% over the last 24 hours.",
        f"Hyperliquid has {coin} at {dollars(p)}, {d} {x}% in 24h.",
    ])


def idea_intro(rng, coin, p, c, moves):
    lst = ", ".join(f"{s} is {'up' if m >= 0 else 'down'} {ns(abs(m))}%" for s, m in moves)
    d, x = ("up" if c >= 0 else "down"), ns(abs(c))
    return pick(rng, [
        f"I checked BTC, ETH and SOL: {lst} over 24h. {coin} moved the most, at {dollars(p)}, {d} {x}%.",
        f"Looking at BTC, ETH and SOL over 24h, {lst}. {coin} has the biggest move, {dollars(p)} and {d} {x}%.",
        f"I pulled BTC, ETH and SOL. {lst} in 24h. The one to talk about is {coin}, at {dollars(p)}.",
        f"Quick scan of BTC, ETH and SOL over 24h: {lst}. {coin} is the mover, {dollars(p)}, {d} {x}%.",
    ])


def view_line(rng, c):
    v = view_of(c)
    if v == "flat":
        return pick(rng, [
            "I can't predict the next move, and there's no clear edge, so I'd keep it small.",
            "Honestly there's no clear edge, so I'd keep it small, and nobody can predict a quiet market.",
            "It's quiet, so there's no clear edge, so I'd keep it small. I can't predict which way it breaks.",
            "No call from me: there's no clear edge, so I'd keep it small, since I can't predict direction.",
            "Nobody can predict this, but right now there's no clear edge, so I'd keep it small.",
            "My read: no clear edge, so I'd keep it small. I can't predict where it goes next.",
        ])
    if v == "up":
        return pick(rng, [
            "It's already extended, chasing is the risk, and I can't predict whether it keeps running.",
            "After a move like that it looks already extended, chasing is the risk. Nobody can predict the next leg.",
            "My read: already extended, chasing is the risk, though I can't predict when it turns.",
            "It's up a lot, so it's already extended, chasing is the risk. I can't predict the top.",
            "I'd be careful: already extended, chasing is the risk, and I can't predict what comes next.",
            "Honestly, already extended, chasing is the risk. I can't predict a top or a bottom.",
        ])
    if v == "down":
        return pick(rng, [
            "My read: a bounce is possible but it can keep falling, and I can't predict which.",
            "After a drop like that, a bounce is possible but it can keep falling. I can't predict where it stops.",
            "I'd say a bounce is possible but it can keep falling, and nobody can predict the bottom.",
            "Honestly, a bounce is possible but it can keep falling. I can't predict it either way.",
            "Hard to say: a bounce is possible but it can keep falling, and I can't predict the timing.",
            "Here a bounce is possible but it can keep falling, and I can't call the bottom or predict it.",
        ])
    bull = c > 0
    w1, w2 = ("bullish", "upside") if bull else ("bearish", "downside")
    return pick(rng, [
        f"If I had to lean, it's slightly {w1}, but only slightly, and I can't predict it.",
        f"There's a slight lean to the {w2}, but it's weak and I can't predict the next move.",
        f"I'd lean a little {w1} here, though it's a weak lean and nobody can predict this.",
        f"My lean is mildly {w1}, nothing strong, and I can't predict where it goes.",
        f"A hedged lean to the {w2}: it's slight, and I can't predict the next move.",
    ])


RISKS = {
    "lev": ["Leverage makes gains and losses bigger.", "Leverage cuts both ways.", "Leverage magnifies a wrong move as much as a right one."],
    "liq": ["A sharp move can hit liquidation.", "If it moves against you far enough you get liquidated.",
            "Liquidation is the risk if the move goes against you."],
    "slip": ["A stop can fill worse than set when the market moves fast.", "Stops can slip in a fast market.",
             "A stop order can fill at a worse price in a fast move."],
    "vol": ["{c} can swing hard in a day.", "{c} is volatile and can move fast.", "Volatility can flip this quickly."],
}


def risk_line(rng, coin):
    kinds = rng.sample(list(RISKS), 2 if rng.random() < 0.65 else 1)
    return " ".join(pick(rng, RISKS[k]).format(c=coin) for k in kinds)


def pos_line(rng, p):
    side, coin, size = p["side"], p["coin"], ns(p["size"])
    if p["liquidation"] is None:
        return pick(rng, [
            f"You already have a {side} on {coin}: {size} {coin}. I don't see a liquidation price for it.",
            f"First, you have an open {coin} {side} of {size} {coin}, with no liquidation price shown.",
            f"You're already {side} {size} {coin}, and I don't see a liquidation price for it.",
        ])
    liq = dollars(ns(p["liquidation"]))
    return pick(rng, [
        f"You already have a {side} on {coin}: {size} {coin}, liquidation at {liq}.",
        f"First, you have an open {coin} {side} of {size} {coin} with liquidation at {liq}.",
        f"Heads up, your account holds a {side} of {size} {coin}, liquidation {liq}.",
        f"You're already {side} {size} {coin}, and its liquidation price is {liq}.",
        f"Your account already has a {coin} {side}, {size} {coin}, liquidation at {liq}.",
    ])


def order_line(side, size_s, coin, lev, stop_s):
    return f"The order I'd prepare: {side} {dollars(size_s)} {coin} at {lev}x with a stop at {dollars(stop_s)}. Want me to prepare it?"


def nothing_line(rng, mode, device):
    if mode == "error":
        return pick(rng, [
            f"I can't see a Hyperliquid account on this {device} yet, so there's nothing to trade with. Want me to open Add funds?",
            f"There's no Hyperliquid account on this {device} yet, so there's nothing to trade with. Want me to open Add funds?",
            f"I couldn't find a Hyperliquid account on this {device}, so you have nothing to trade with. Want me to open Add funds?",
        ])
    return pick(rng, [
        "Your Hyperliquid account has nothing to trade with right now, so there's no order to size yet. Want me to open Add funds?",
        "There's nothing to trade with in your Hyperliquid account, so no order to prepare yet. Want me to open Add funds?",
        "Your account has no free USDC, so you have nothing to trade with at the moment. Want me to open Add funds?",
        "You have nothing to trade with on Hyperliquid right now, so I can't size an order. Want me to open Add funds?",
    ])


def opposite_note(rng, p):
    return pick(rng, [f"Note that's the opposite side of your {p['side']} on {p['coin']}.",
                      f"That's the other side from your open {p['coin']} {p['side']}.",
                      f"Keep in mind your open {p['coin']} is a {p['side']}, so this goes the other way."])


def nomarket_text(rng, coin):
    s1 = pick(rng, [f"There's no live price here for {coin}, Hyperliquid has no market for it, so I can't give you a number.",
                    f"I don't have a live price for {coin}: there's no live price here because Hyperliquid has no market for it.",
                    f"Hyperliquid has no market for {coin}, so there's no live price here and I won't guess one.",
                    f"There's no live price here for {coin}, so I can't quote you a number or a daily move.",
                    f"No live price here for {coin}, it isn't listed on Hyperliquid, so no number from me."])
    s2 = pick(rng, ["Memecoins can swing hard and fast, nobody can predict them, and leverage plus liquidation make that worse.",
                    "I can't predict where a memecoin goes, and the volatility is a real risk, more so with leverage and liquidation.",
                    "Memecoins are very volatile and I can't predict them, so leverage could liquidate you quickly.",
                    "Nobody can predict a memecoin, and a fast swing with leverage can mean liquidation or a stop that slips."])
    s3 = pick(rng, ["Without a live price I can't size an order for it.", "So I can't prepare an order for it here.",
                    "That also means I have no order to propose for it.", "Without a live market there's no order I can prepare."])
    s4 = pick(rng, ["I can give a view on a coin with a live market instead, like BTC, ETH or SOL.",
                    "I can look at a coin that has a live market, like BTC, ETH or SOL, when you want.",
                    "BTC, ETH or SOL have live markets here, so I can give you a proper view on those."])
    return " ".join([s1, s2, s3] + ([s4] if rng.random() < 0.7 else []))


def wc(t):
    return len(t.split())


# ----------------------------------------------------------------------------- conversation
class TVConv(Conv):
    def __init__(self, rng, world, bio):
        self.rng, self.w, self.tools = rng, world, True
        self.world0 = world.state()
        sysm = SYSTEM_V2.replace("{BIO}", bio)
        if rng.random() < 0.15:
            sysm += f"\nThe user wants to be called {rng.choice(H.NAMES)}."
        self.msgs = [{"role": "system", "content": sysm}]
        self.expect = []

    def ex(self, kind):
        return {"messages": self.msgs, "tools": True,
                "meta": {"source": "trade_view", "generator": "trade_view", "kind": kind, "world": self.world0,
                         "world_final": self.w.state(), "expect": self.expect, "lang": "en"}}


def sc_trade_view(rng, eval_mode, force=None):
    """Build one example. Returns (example, user_text)."""
    force = force or {}
    bio = "Face ID" if rng.random() < 0.7 else "Touch ID"
    device = "iPhone" if bio == "Face ID" else ("Mac" if rng.random() < 0.9 else "iPhone")
    # ---- class
    r = rng.random()
    cls = force.get("cls") or ("meme" if r < 0.08 else ("idea" if r < 0.20 else "coin"))
    pools = {"neutral": NEUTRAL, "long": LONG, "short": SHORT, "ask": ASK, "story": STORY}
    def pick_tpl(pool):
        idx = [i for i in range(len(pool)) if (i % 5 == 0) == eval_mode]
        i = rng.choice(idx)
        return i, pool[i]
    user_side, tpl, tpl_id = None, None, None
    coin = None
    amt_s = None
    if cls == "idea":
        i, tpl = pick_tpl(NOCOIN)
        tpl_id = f"nocoin{i}"
        if "{amt}" in tpl:
            amt_s = amt_text(rng, rng.choice([100, 200, 250, 300, 500, 750, 1000, 1500, 2000, 5000]))
    else:
        kind = rng.choices(["neutral", "long", "short", "ask", "story"], [33, 22, 14, 8, 23])[0]
        i, t = pick_tpl(pools[kind])
        tpl_id = f"{kind}{i}"
        if kind == "story":
            tpl, user_side = t
            user_side = None if user_side in ("neutral", "ask") else user_side
        else:
            tpl = t
            user_side = kind if kind in ("long", "short") else None
        coin = rng.choice(MEMES) if cls == "meme" else rng.choices(COINS, COIN_W)[0]
    # ---- coin / world
    prices = {c: draw_price(rng, c) for c in COINS}
    change = {c: draw_change(rng) for c in COINS}
    if cls == "idea":
        while True:
            ch = [abs(change[c]) for c in IDEA_COINS]
            if len({round(x, 1) for x in ch}) == 3:
                break
            for c in IDEA_COINS:
                change[c] = draw_change(rng)
        coin = max(IDEA_COINS, key=lambda c: abs(change[c]))
    no_market = [coin] if cls == "meme" else []
    am = force.get("acct") or rng.choices(["ok", "zero_empty", "zero_full", "error"], [0.88, 0.04, 0.03, 0.05])[0]
    if cls == "meme":
        am = "ok" if rng.random() < 0.8 else am
    pos_coin = coin if (cls != "meme" and rng.random() < 0.18 and am in ("ok", "zero_full")) else None
    if am == "zero_full" and not pos_coin:
        pos_coin = None
    acct = make_account(rng, am, prices, device, pos_coin)
    world = TVWorld(prices, change, no_market, acct, bio, device)
    # ---- user text
    alias = rng.choices(ALIAS[coin], [45] + [20] * (len(ALIAS[coin]) - 1))[0] if cls != "idea" else None
    lev_suffix = None
    user_lev = None
    if cls != "idea" and cls != "meme" and rng.random() < 0.07 and "?" not in tpl[-2:]:
        lev_suffix = rng.choice(LEV_SUFFIX)
        user_lev = 5
    text = render_user(rng, tpl, alias, amt_s, lev_suffix)
    c = TVConv(rng, world, bio)
    c.user(text)
    gold = {"mode": {"idea": "idea", "meme": "nomarket", "coin": "coin"}[cls], "focus": coin, "account": am}
    # ---- tools, turn 1
    syms = IDEA_COINS if cls == "idea" else [coin]
    calls = [("get_price", {"symbol": s}) for s in syms] + [("hyperliquid_account", {})]
    res = c.call(*calls)
    gold["symbols"] = syms
    if cls == "meme":
        c.say(nomarket_text(rng, coin))
        gold["price_str"] = None
        return finish(c, rng, gold, "trade_view_nomarket", tpl_id, text, bio, alias, None)
    pr = {s: res[i] for i, s in enumerate(syms)}
    p, ch = pr[coin]["price"], pr[coin]["change_24h_pct"]
    acct_r = res[-1]
    view = view_of(ch)
    pos = None
    if "error" not in acct_r:
        pos = next((x for x in acct_r["positions"] if x["coin"] == coin), None)
    nothing = ("error" in acct_r) or acct_r["available_usdc"] <= 0
    side = user_side or (pos["side"] if pos else None) or ("short" if view == "up" else "long" if view == "down" else ("long" if ch >= 0 else "short"))
    lev = user_lev or (2 if abs(ch) > 6 else 3)
    gold.update({"price_str": ns(p), "change": ch, "view": view, "view_phrase": VIEW_PHRASE[view], "side": side, "lev": lev,
                 "user_side": user_side, "user_lev": user_lev, "position": pos, "nothing_to_trade": nothing})
    size_s = stop_s = None
    if not nothing:
        avail = ns(acct_r["available_usdc"])
        e_size = f"max(10, round(round({avail} * 0.15, 0) / 5 - 0.4, 0) * 5)"
        dp = stop_dp(p)
        e_stop = f"round({ns(p)} * {'0.93' if side == 'long' else '1.07'}, {dp})"
        rs, rt = c.call(("calculator", {"expression": e_size}), ("calculator", {"expression": e_stop}))
        size_s, stop_s = rs["result"], rt["result"]
        gold.update({"size": parse_calc(size_s), "stop": parse_calc(stop_s), "size_expr": e_size, "stop_expr": e_stop})
    # ---- text (retry until it fits the word cap)
    for _ in range(40):
        parts = []
        if pos:
            parts.append(pos_line(rng, pos))
        if cls == "idea":
            parts.append(idea_intro(rng, coin, p, ch, [(s, pr[s]["change_24h_pct"]) for s in syms]))
        else:
            parts.append(price_line(rng, coin, p, ch))
        parts.append(view_line(rng, ch))
        parts.append(risk_line(rng, coin))
        if nothing:
            parts.append(nothing_line(rng, "error" if "error" in acct_r else "zero", device))
        else:
            if pos and pos["side"] != side:
                parts.append(opposite_note(rng, pos))
            parts.append(order_line(side, size_s, coin, lev, stop_s))
        text1 = " ".join(parts)
        if wc(text1) <= 110:
            break
    else:
        raise RuntimeError("could not fit the word cap")
    c.say(text1)
    # ---- second turn (25% of rows that offer an order)
    kind = "trade_view_idea" if cls == "idea" else "trade_view_coin"
    if not nothing and rng.random() < (force.get("fu_p", 0.25)):
        opts = ["do", "loss"] + (["x5"] if lev != 5 else [])
        fu = force.get("fu") or rng.choices(opts, [45, 25, 30][:len(opts)] if len(opts) == 3 else [60, 40])[0]
        if fu == "x5" and lev == 5:
            fu = "do"
        gold["followup"] = fu
        followup(c, rng, fu, gold, coin, side, lev, size_s, stop_s, p, bio, ch)
        kind += "_fu_" + fu
    return finish(c, rng, gold, kind, tpl_id, text, bio, alias, None)


def followup(c, rng, fu, gold, coin, side, lev, size_s, stop_s, p, bio, ch):
    rng2 = rng
    if fu == "do":
        c.user(typo_short(rng2, pick(rng2, FU_DO)))
        args = {"coin": coin, "side": side, "size_usd": parse_calc(size_s), "leverage": lev, "order_type": "market",
                "stop_loss": parse_calc(stop_s)}
        c.call(("prepare_trade", args))
        gold["fu_args"] = args
        c.say(pick(rng2, [
            f"I've prepared a {lev}x {side} on {coin} for {dollars(size_s)} with a stop at {dollars(stop_s)}. It's waiting for your {bio}, nothing happens until you approve it.",
            f"The {side} on {coin} is prepared: {dollars(size_s)} at {lev}x with a stop at {dollars(stop_s)}. Review it and approve with {bio} when you're ready.",
            f"Prepared: {side} {dollars(size_s)} {coin} at {lev}x, stop at {dollars(stop_s)}. It's on the review screen and waits for your {bio}.",
            f"Your {coin} {side} is prepared, {dollars(size_s)} at {lev}x with the stop at {dollars(stop_s)}. Nothing happens until you approve it with {bio}.",
            f"I've prepared it as a market order: {side} {dollars(size_s)} {coin} at {lev}x, stop {dollars(stop_s)}. Approve it with {bio} on the review screen if it looks right.",
            f"It's prepared and waiting for {bio}: a {lev}x {side} on {coin}, {dollars(size_s)}, stop at {dollars(stop_s)}.",
        ]))
    elif fu == "x5":
        c.user(typo_short(rng2, pick(rng2, FU_5X)))
        args = {"side": side, "entry": p, "size_usd": parse_calc(size_s), "leverage": 5}
        r = c.call(("position_math", args))
        gold["fu_args"] = args
        liq, dist = ns(r["liquidation_price"]), ns(r["distance_to_liq_pct"])
        mg = ns(r["margin"])
        c.say(pick(rng2, [
            f"At 5x the order would be: {side} {dollars(size_s)} {coin} at 5x with a stop at {dollars(stop_s)}. It needs {dollars(mg)} of margin and liquidation would be about {dollars(liq)}, {dist}% from the current price. That's farther than your stop, but a fast move can slip past a stop. Want me to prepare it?",
            f"With 5x it becomes: {side} {dollars(size_s)} {coin} at 5x, stop at {dollars(stop_s)}. Margin is {dollars(mg)} and liquidation is around {dollars(liq)}, {dist}% away. The stop is closer, but stops can slip when the market moves fast. Want me to prepare it?",
            f"5x works like this: {side} {dollars(size_s)} {coin} with a stop at {dollars(stop_s)}, {dollars(mg)} of margin, liquidation near {dollars(liq)}, which is {dist}% from the price. Higher leverage means less room, so it's riskier. Want me to prepare it?",
            f"At 5x the same {dollars(size_s)} {side} needs {dollars(mg)} margin, with the stop still at {dollars(stop_s)}. Liquidation would be about {dollars(liq)}, {dist}% from the current price. Leverage cuts both ways. Want me to prepare it?",
        ]))
        gold["lev_after"] = 5
    else:
        c.user(typo_short(rng2, pick(rng2, FU_LOSS)))
        e = (f"round({size_s} * ({ns(p)} - {stop_s}) / {ns(p)}, 2)" if side == "long"
             else f"round({size_s} * ({stop_s} - {ns(p)}) / {ns(p)}, 2)")
        r = c.call(("calculator", {"expression": e}))
        loss = r["result"]
        gold["fu_expr"], gold["fu_loss"] = e, parse_calc(loss)
        c.say(pick(rng2, [
            f"If the stop at {dollars(stop_s)} hits, you'd lose about {dollars(loss)} on that {dollars(size_s)} position, before fees. A stop can fill worse than set when the market moves fast, so the real loss could be bigger. Want me to prepare it?",
            f"The stop at {dollars(stop_s)} would cost you about {dollars(loss)} on the {dollars(size_s)} position, before fees. Stops can slip in a fast market, so it could be more. Want me to prepare it?",
            f"About {dollars(loss)} lost if the stop at {dollars(stop_s)} is hit, on {dollars(size_s)} of {coin}, before fees. Slippage on a stop can make it larger. Want me to prepare it?",
            f"If it stops you out at {dollars(stop_s)}, that's roughly {dollars(loss)} on the {dollars(size_s)} position, not counting fees. A fast move can fill the stop worse. Want me to prepare it?",
        ]))


def typo_short(rng, s):
    if rng.random() < 0.12:
        parts = [[s, False]]
        typo(rng, parts)
        s = parts[0][0]
    if rng.random() < 0.15:
        s = s.upper() if rng.random() < 0.1 else s.capitalize()
    return s


def finish(c, rng, gold, kind, tpl_id, text, bio, alias, _):
    ex = c.ex(kind)
    m = ex["meta"]
    m["bio"] = bio
    m["gold"] = gold
    m["template"] = tpl_id
    m["slots"] = [gold["mode"], gold.get("focus"), alias, gold.get("account"), gold.get("followup")]
    m["checks"] = ["no_ask_for_data", "no_invented_price", "decimals_ok", "right_tool"] + (
        ["exact_digits"] if gold.get("followup") == "do" else [])
    return ex, text


# ----------------------------------------------------------------------------- self-check
BANNED = [
    r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|deposited|withdrawn|closed|placed|executed)",
    r"\bI(?:'ve| have)? (just )?(sent|bought|sold|swapped|deposited|withdrew|closed|placed|executed|opened)\b", r"successfully",
    r"funds (are|have) (in|arrived)", r"^\s*done\b", r"not financial advice", r"[·•]", r"what('s| is) (the )?(current )?price",
    r"your budget", r"how much (do you|are you|would you)", r"can't see the future", r"guarantee", r"sure thing", r"risk[- ]free",
    r"will (definitely |certainly )?(go up|moon|pump|dump|rally)", r"\bsent\b", r"\bopened\b", r"as an ai",
    r"tell me (your|the) (budget|price|size|amount|holdings|balance)", r"consult a (financial|professional)",
    r"\b(RSI|MACD|moving average|support|resistance|funding rate|whale|news)\b",
]
BANNED_RE = [re.compile(b, re.I) for b in BANNED]
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
PREDICT_RE = re.compile(r"(can't|cannot|nobody can|no one can|can not) (really )?(predict|call)", re.I)
ADDR_RE = re.compile(r"0x[0-9a-fA-F]{40}|\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")


def D(s):
    return Decimal(s.replace(",", ""))


def nums_in(s):
    out = set()
    for m in NUM_RE.findall(s):
        try:
            out.add(D(m).normalize())
        except Exception:
            pass
    return out


def check_args(tc):
    name, args = tc["name"], tc["arguments"]
    assert name in SCHEMA, f"unknown tool {name}"
    sch = SCHEMA[name]
    for req in sch.get("required", []):
        assert req in args, (name, "missing", req)
    for k, v in args.items():
        assert k in sch["properties"], (name, "unknown arg", k)
        t = sch["properties"][k].get("type")
        if t == "number":
            assert isinstance(v, (int, float)) and not isinstance(v, bool), (name, k, v)
        elif t == "integer":
            assert isinstance(v, int) and not isinstance(v, bool), (name, k, v)
        elif t == "string":
            assert isinstance(v, str), (name, k, v)
        if "enum" in sch["properties"][k]:
            assert v in sch["properties"][k]["enum"], (name, k, v)
        if isinstance(v, float):
            assert not v.is_integer(), (name, k, "float with .0", v)


_TOK = None


def count_tokens(text):
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained("base/Qwen3-4B-Instruct-2507")
        except Exception:
            _TOK = False
    return len(_TOK(text)["input_ids"]) if _TOK else int(len(text) / 3.2)


def selfcheck(rec, maxtok=[0]):
    """Raises AssertionError with a message on the first problem."""
    rec = json.loads(json.dumps(rec, ensure_ascii=False))          # valid JSON round trip
    msgs, meta, g = rec["messages"], rec["meta"], rec["meta"]["gold"]
    assert rec["tools"] is True and meta["source"] == "trade_view"
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    full = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs)
    assert other not in full and "{BIO}" not in full, "biometric word mismatch"
    assert bio in msgs[0]["content"]
    # structure
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert msgs[0]["content"].startswith(SYSTEM_V2.replace("{BIO}", bio).split("\n")[0])
    assert SYSTEM_V2.replace("{BIO}", bio) in msgs[0]["content"]
    # calls: schema, replay against the world, recorded results identical
    w = TVWorld.from_state(meta["world"])
    i = 2
    users = [m["content"] for m in msgs if m["role"] == "user"]
    all_calls = []
    tool_texts = []
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            tcs = m["tool_calls"]
            assert m["content"] == ""
            for tc in tcs:
                check_args(tc)
            j = i + 1
            for tc in tcs:
                assert msgs[j]["role"] == "tool", "tool result missing"
                rr = w.call(tc["name"], tc["arguments"])
                assert msgs[j]["content"] == json.dumps(rr, ensure_ascii=False), ("replay mismatch", tc, msgs[j]["content"], rr)
                tool_texts.append(msgs[j]["content"])
                all_calls.append(tc)
                j += 1
            i = j
        else:
            i += 1
    # first turn
    first = msgs[2]
    assert first["role"] == "assistant" and first.get("tool_calls")
    names = [t["name"] for t in first["tool_calls"]]
    exp_syms = g["symbols"]
    assert names == ["get_price"] * len(exp_syms) + ["hyperliquid_account"], names
    assert [t["arguments"] for t in first["tool_calls"]] == [{"symbol": s} for s in exp_syms] + [{}]
    assert len(first["tool_calls"]) == (4 if g["mode"] == "idea" else 2)
    # final first-turn text = first assistant message with no tool_calls
    k = next(n for n, m in enumerate(msgs) if n > 2 and m["role"] == "assistant" and not m.get("tool_calls"))
    t1 = msgs[k]["content"]
    texts = [m["content"] for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
    # prepare_trade never in the first turn
    assert all(tc["name"] != "prepare_trade" for tc in all_calls[:len(first["tool_calls"]) + 2]) or g.get("followup") == "do"
    first_turn_calls = [tc["name"] for m in msgs[2:k] if m.get("tool_calls") for tc in m["tool_calls"]]
    assert "prepare_trade" not in first_turn_calls, "prepare_trade on turn one"
    # banned phrases, bio, '?' rule, words
    for t in texts:
        for rx in BANNED_RE:
            assert not rx.search(t), (rx.pattern, t)
        assert not re.search(r"[\U0001F300-\U0001FAFF]", t)
        assert not ADDR_RE.search(t) or ADDR_RE.search(t).group(0) in " ".join(users), "address not copied"
    assert wc(t1) <= 110, ("too long", wc(t1))
    q = t1.count("?")
    # numbers
    allowed = set()
    for tt in tool_texts + users:
        allowed |= nums_in(tt)
    allowed |= {Decimal(24), Decimal(2), Decimal(3), Decimal(5)}
    blob = " ".join(tool_texts).replace(",", "")
    for t in texts:
        for m in NUM_RE.findall(t):
            d = D(m).normalize()
            assert d in allowed, ("number not from a tool or the user", m, t)
            if "." in m:
                assert m.replace(",", "") in blob, ("decimal not in a tool result", m)
    # per mode
    if g["mode"] == "nomarket":
        assert "no live price here" in t1.lower(), t1
        assert q == 0 and not re.search(r"\d", t1), "memecoin answer must have no number and no question"
        assert json.loads(msgs[3]["content"])["error"].startswith("No Hyperliquid market for")
        assert len(texts) == 1
        return
    pr = {}
    for n, s in enumerate(exp_syms):
        pr[s] = json.loads(msgs[3 + n]["content"])
        assert "error" not in pr[s]
    focus = g["focus"]
    p, ch = pr[focus]["price"], pr[focus]["change_24h_pct"]
    assert ns(p).replace(",", "") in t1.replace(",", ""), ("price string missing", ns(p), t1)
    assert (("up " if ch >= 0 else "down ") + ns(abs(ch)) + "%") in t1, "24h move missing"
    if g["mode"] == "idea":
        assert focus == max(IDEA_COINS, key=lambda s: abs(pr[s]["change_24h_pct"])), "not the biggest mover"
        for s in IDEA_COINS:
            assert (ns(abs(pr[s]["change_24h_pct"])) + "%") in t1
    view = view_of(ch)
    assert view == g["view"]
    assert VIEW_PHRASE[view].lower() in t1.lower(), ("view phrase", view, t1)
    if view == "mild":
        assert "lean" in t1.lower()
    assert PREDICT_RE.search(t1), "no cannot-predict clause"
    assert any(re.search(x, t1, re.I) for x in ("leverage", "liquidat", "stop", "volatil", "swing")), "no risk named"
    acct = json.loads(msgs[3 + len(exp_syms)]["content"])
    pos = None if "error" in acct else next((x for x in acct["positions"] if x["coin"] == focus), None)
    assert pos == g["position"]
    if pos:
        first_sentence = re.split(r"(?<=[.!?])\s", t1)[0]
        assert pos["side"] in first_sentence and ns(pos["size"]) in first_sentence, "position must come first"
        if pos["liquidation"] is not None:
            assert commas(ns(pos["liquidation"])) in first_sentence, "liquidation must come first"
    nothing = ("error" in acct) or acct["available_usdc"] <= 0
    assert nothing == g["nothing_to_trade"]
    if nothing:
        assert "nothing to trade with" in t1 and "Add funds" in t1 and t1.endswith("Want me to open Add funds?")
        assert q == 1
        assert [t["name"] for m in msgs[2:k] if m.get("tool_calls") for t in m["tool_calls"]] == names
        assert not g.get("followup")
    else:
        assert q == 1 and t1.endswith("Want me to prepare it?"), t1
        calc_msgs = msgs[4 + len(exp_syms)]
        assert [t["name"] for t in calc_msgs["tool_calls"]] == ["calculator", "calculator"]
        size = json.loads(msgs[5 + len(exp_syms)]["content"])["result"]
        stop = json.loads(msgs[6 + len(exp_syms)]["content"])["result"]
        # independent computation of the figures
        av = acct["available_usdc"]
        r0 = round(av * 0.15)
        size_ind = max(10, (r0 // 5) * 5)
        assert float(size) == size_ind and size_ind % 5 == 0 and size_ind >= 10, ("size", size, size_ind)
        side = g["side"]
        stop_ind = round(p * (0.93 if side == "long" else 1.07), stop_dp(p))
        assert float(stop) == stop_ind, ("stop", stop, stop_ind)
        assert g["size"] == float(size) and g["stop"] == float(stop)
        want_lev = g["user_lev"] or (2 if abs(ch) > 6 else 3)
        assert g["lev"] == want_lev
        assert acct["available_usdc"] >= size_ind / want_lev, "order needs more margin than available"
        assert order_line(side, size, focus, want_lev, stop) in t1, "order sentence"
        if g["user_side"]:
            assert side == g["user_side"]
        assert side in ("long", "short")
    # follow-up
    fu = g.get("followup")
    if fu:
        assert len(users) == 2 and len(texts) == 2
        t2 = texts[1]
        assert re.search(r"[A-Za-z]", users[1])
        lastcalls = [tc for m in msgs[k + 2:] if m.get("tool_calls") for tc in m["tool_calls"]]
        if fu == "do":
            assert [t["name"] for t in lastcalls] == ["prepare_trade"]
            a = lastcalls[0]["arguments"]
            assert a == {"coin": focus, "side": side, "size_usd": float(size) if "." in size else int(size), "leverage": want_lev,
                         "order_type": "market", "stop_loss": float(stop) if "." in stop else int(stop)}, a
            assert Decimal(str(a["stop_loss"])) == D(stop) and Decimal(str(a["size_usd"])) == D(size)
            assert "prepared" in t2.lower() and bio in t2 and "?" not in t2
            assert dollars(size) in t2 and dollars(stop) in t2
        elif fu == "x5":
            assert [t["name"] for t in lastcalls] == ["position_math"]
            a = lastcalls[0]["arguments"]
            assert a == {"side": side, "entry": p, "size_usd": float(size) if "." in size else int(size), "leverage": 5}, a
            assert "5x" in t2 and dollars(size) in t2 and dollars(stop) in t2 and t2.endswith("Want me to prepare it?")
            res = json.loads(msgs[-2]["content"])
            assert dollars(ns(res["liquidation_price"])) in t2
            assert t2.count("?") == 1
        else:
            assert [t["name"] for t in lastcalls] == ["calculator"]
            res = json.loads(msgs[-2]["content"])["result"]
            sgn = (p - float(stop)) if side == "long" else (float(stop) - p)
            assert abs(float(res) - round(float(size) * sgn / p, 2)) < 1e-9, ("loss", res)
            assert dollars(res) in t2 and t2.endswith("Want me to prepare it?") and t2.count("?") == 1
        assert wc(t2) <= 110
    else:
        assert len(users) == 1 and len(texts) == 1
    # render round trip (19-tool block, biometric word applied like the app does)
    rendered = H.render_text(msgs, True).replace("Face ID", bio)
    calls_back, bad = H.parse_calls(rendered.split("<|im_start|>user\n", 1)[1])
    assert bad == 0 and [c["name"] for c in calls_back] == [tc["name"] for tc in all_calls], "render/parse mismatch"
    for a, b in zip(calls_back, all_calls):
        assert a["arguments"] == b["arguments"], "render/parse args mismatch"
    ntok = count_tokens(rendered)
    maxtok[0] = max(maxtok[0], ntok)
    assert ntok <= 6144, ("too many tokens", ntok)


def first_user(rec):
    return next(m["content"] for m in rec["messages"] if m["role"] == "user")


def eval2_users():
    seen = set()
    d = os.path.join(TRAIN_DIR, "eval2")
    for fn in os.listdir(d):
        for l in open(os.path.join(d, fn)):
            try:
                r = json.loads(l)
            except Exception:
                continue
            for m in r.get("context", []) or []:
                if m.get("role") == "user":
                    seen.add(norm_text(m["content"]))
            for k in ("prompt", "question"):
                if isinstance(r.get(k), str):
                    seen.add(norm_text(r[k]))
            for t in r.get("turns", []) or []:
                if isinstance(t, str):
                    seen.add(norm_text(t))
    return seen


# ----------------------------------------------------------------------------- build
def build(seed, n, eval_mode, forbid):
    rng = random.Random(seed)
    rows, seen = [], set()
    tries = 0
    while len(rows) < n:
        tries += 1
        assert tries < n * 60, "cannot make enough unique rows"
        sub = random.Random(rng.getrandbits(64))
        force = {}
        if eval_mode and len(rows) % 10 == 3:       # make sure the eval has enough "make it 5x" items
            force = {"cls": "coin", "acct": "ok", "fu_p": 1.0, "fu": "x5"}
        ex, text = sc_trade_view(sub, eval_mode, force)
        key = norm_text(text)
        if key in seen or key in forbid:
            continue
        seen.add(key)
        rows.append(ex)
    return rows


def to_eval_item(ex, i):
    msgs = ex["messages"]
    last_u = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    m = ex["meta"]
    g = m["gold"]
    return {"id": f"trade_view-{i}", "kind": m["kind"], "tools": True, "bio": m["bio"], "context": msgs[:last_u + 1],
            "reference": msgs[last_u + 1:], "expect": m["expect"], "world": m["world"], "checks": m["checks"], "gold": g,
            "template": m["template"], "lang": "en"}


def check_eval_item(it):
    """Rebuild a full record from an eval item (context + reference) and run the same self-check."""
    rec = {"messages": it["context"] + it["reference"], "tools": True,
           "meta": {"source": "trade_view", "bio": it["bio"], "world": it["world"], "gold": it["gold"], "kind": it["kind"]}}
    selfcheck(rec)
    exp = it["expect"]
    assert exp and all(("calls" in s) or ("text" in s) or ("ask" in s) for s in exp)
    last = it["context"][-1]
    assert last["role"] == "user"
    ref_calls = [[{"name": t["name"], "arguments": t["arguments"]} for t in mm["tool_calls"]] for mm in it["reference"] if mm.get("tool_calls")]
    exp_calls = [s["calls"] for s in exp if "calls" in s]
    assert ref_calls == exp_calls, "expect does not match the reference calls"
    assert exp[-1].get("text"), "last step must be text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--eval-seed", type=int, default=9303)
    ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(TRAIN_DIR, "data3", "trade_view.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(TRAIN_DIR, "eval3", "trade_view.jsonl"))
    a = ap.parse_args()
    assert a.seed != a.eval_seed
    forbid = eval2_users()
    train = build(a.seed, a.n, False, forbid)
    train_keys = {norm_text(first_user(r)) for r in train}
    ev = build(a.eval_seed, a.eval_n, True, forbid | train_keys)
    items = [to_eval_item(r, i) for i, r in enumerate(ev)]
    for r in train:
        selfcheck(r)
    for r in ev:
        selfcheck(r)
    for it in items:
        check_eval_item(it)
    # file-level checks
    tk = [norm_text(first_user(r)) for r in train]
    assert len(tk) == len(set(tk)), "duplicate user texts in train"
    ek = {norm_text(it["context"][1]["content"]) for it in items}
    assert len(ek) == len(items) and not (ek & set(tk)), "eval user texts overlap train"
    for d in (os.path.dirname(a.out), os.path.dirname(a.eval_out)):
        os.makedirs(d, exist_ok=True)
    with open(a.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # re-read from disk and check again (valid JSON on disk, same checks)
    n1 = n2 = 0
    for l in open(a.out):
        selfcheck(json.loads(l)); n1 += 1
    for l in open(a.eval_out):
        check_eval_item(json.loads(l)); n2 += 1
    assert (n1, n2) == (len(train), len(items))
    # report
    kinds = Counter(r["meta"]["kind"] for r in train)
    ekinds = Counter(it["kind"] for it in items)
    print("train kinds:", dict(kinds), file=sys.stderr)
    print("eval kinds:", dict(ekinds), file=sys.stderr)
    print("bio:", Counter(r["meta"]["bio"] for r in train), "followups:", sum(1 for r in train if r["meta"]["gold"].get("followup")), file=sys.stderr)
    print("accounts:", Counter(r["meta"]["gold"]["account"] for r in train), file=sys.stderr)
    print("max tokens:", selfcheck.__defaults__[0][0], file=sys.stderr)
    print(f"trade_view: {n1} train, {n2} eval, self-check ok")


if __name__ == "__main__":
    main()
