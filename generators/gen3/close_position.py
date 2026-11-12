#!/usr/bin/env python3
"""close_position generator (sft4 generator 12): prepare_close for full and percent closes.

Teaches: prepare_close instead of an opposite prepare_trade or "tap Close"; ambiguity handling
(hyperliquid_account first, one short question); no_position / no wallet / blocked results; negatives
(PnL question, "should I close", cancel order, "short SOL" with a long open, leverage, how-to).

Reuses gen2 helpers by import (hermit_fmt, sim, build_agent.Conv); nothing in gen2 is copied or edited.
Tool registry, result shapes and system prompt come from ../spec (hermit_tools_v2.json,
hermit_tools_v2.meta.json, SYSTEM_AGENT_v2.txt). Every assistant sentence is a Python f-string filled from
tool results or user text; no number is typed by hand.

Usage: python close_position.py --seed 3 --n 350 [--eval-seed 9312 --eval-n 40]
Writes ../data3/close_position.jsonl and ../eval3/close_position.jsonl, then runs the self-check on every
record (exit 1 on any failure).
"""
import argparse, json, os, random, re, sys
from collections import Counter
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gen2"))
import hermit_fmt as H          # noqa: E402  (gen2, read only)
import sim                      # noqa: E402
import build_agent as BA        # noqa: E402  (Conv)

SPEC = os.path.join(HERE, "..", "spec")
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
SYSTEM_V2 = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().rstrip("\n")
assert "{BIO}" in SYSTEM_V2
SYSTEM_FACE = SYSTEM_V2.replace("{BIO}", "Face ID")
GEN_NAME = "close_position"

# ------------------------------------------------------------------ small helpers
def short(x):
    """Shortest JSON number: 22.0 -> 22."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def money(x):
    """Dollar amount with the exact digits of x (at least 2 decimals unless whole)."""
    d = Decimal(str(abs(x)))
    ip, _, fp = format(d, "f").partition(".")
    fp = fp.rstrip("0")
    if fp and len(fp) < 2:
        fp = fp.ljust(2, "0")
    if not fp and Decimal(str(x)) != Decimal(str(x)).to_integral_value():
        fp = "00"
    return "$" + f"{int(ip):,}" + ("." + fp if fp else "")


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def sdec(x, dp):
    return short(float(Decimal(str(x)).quantize(Decimal(1).scaleb(-dp))))


def art(coin):
    return "an" if coin[0] in "AEIOX" else "a"


def pos_phrase(p):
    return f"{art(p['coin'])} {p['coin']} {p['side']}"


def join_and(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


NUMW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# ------------------------------------------------------------------ coins
COIN_WORDS = {
    "BTC": ["BTC", "btc", "bitcoin", "Bitcoin"], "ETH": ["ETH", "eth", "ether", "ethereum", "Ethereum"],
    "SOL": ["SOL", "sol", "solana", "Solana"], "HYPE": ["HYPE", "hype"], "DOGE": ["DOGE", "doge", "dogecoin"],
    "XRP": ["XRP", "xrp", "ripple"], "SUI": ["SUI", "sui"], "AVAX": ["AVAX", "avax", "avalanche"],
    "LINK": ["LINK", "chainlink"], "BNB": ["BNB", "bnb"], "ENA": ["ENA", "ena"], "TAO": ["TAO", "tao", "bittensor"],
    "INJ": ["INJ", "inj"], "ARB": ["ARB", "arb", "arbitrum"], "NEAR": ["NEAR", "near"], "OP": ["OP", "Optimism"],
    "APT": ["APT", "apt", "aptos"],
}
WORD2COIN = {w.lower(): t for t, ws in COIN_WORDS.items() for w in ws}
EVAL_ONLY_COINS = ["INJ", "TAO", "ARB", "NEAR", "APT"]
TRAIN_COINS = ["BTC", "ETH", "SOL", "HYPE", "DOGE", "XRP", "SUI", "AVAX", "LINK", "BNB", "ENA", "OP"]
COIN_WEIGHT = {"BTC": 5, "ETH": 5, "SOL": 5, "HYPE": 3}
assert all(c in sim.ASSETS for c in COIN_WORDS)


def coin_pool(split):
    return TRAIN_COINS if split == "train" else TRAIN_COINS + EVAL_ONLY_COINS


def pick_coins(rng, split, k, exclude=()):
    pool = [c for c in coin_pool(split) if c not in exclude]
    out = []
    while len(out) < k:
        if split == "eval" and rng.random() < 0.35:
            c = rng.choice([x for x in EVAL_ONLY_COINS if x not in exclude and x not in out] or pool)
        else:
            ws = [COIN_WEIGHT.get(c, 1) for c in pool]
            c = rng.choices(pool, ws)[0]
        if c not in out:
            out.append(c)
    return out


def coin_text(rng, c):
    ws = COIN_WORDS[c]
    return ws[0] if rng.random() < 0.5 else rng.choice(ws)

# ------------------------------------------------------------------ world
DEVICE_BLOCKED = ["This trade is above your daily limit for the assistant.",
                  "The assistant has reached its limit for today.",
                  "This close is above the per-trade limit for the assistant."]


class CloseWorld(sim.World):
    """gen2 World with the app's result shapes (hermit_tools_v2.meta.json) for the tools this generator uses."""
    STATE_KEYS = sim.World.STATE_KEYS + ("device", "no_wallet", "blocked")

    def __init__(self, rng, device="iPhone"):
        super().__init__(rng, empty_hl=True, empty_wallet=False)
        self.device, self.no_wallet, self.blocked = device, False, None
        self.hl_positions, self.hl_orders = [], []

    def add_position(self, rng, coin, side, pnl_sign=None):
        mark = self.prices[coin]
        dp = sim.ASSETS[coin][2]
        sgn = 1 if side == "long" else -1
        for _ in range(60):
            lev = rng.choice([2, 3, 5, 5, 10, 10, 15, 20])
            d = rng.uniform(0.012, min(0.09, 0.7 / lev))
            ps = pnl_sign if pnl_sign else rng.choice([1, -1])
            entry = round(mark * (1 - sgn * ps * d), dp)
            entry = short(float(entry)) if dp == 0 else entry
            if entry <= 0:
                continue
            size_usd = rng.uniform(0.05, 0.28) * self.hl_value * lev
            sdp = 4 if mark >= 1000 else 2 if mark >= 10 else 1 if mark >= 1 else 0
            size = round(size_usd / entry, sdp)
            if size <= 0:
                continue
            pnl = round(sgn * (mark - entry) * size, 2)
            if pnl == 0 or (pnl_sign and (pnl > 0) != (pnl_sign > 0)):
                continue
            liq = round(sim.liq_price(side, entry, lev, 0.005), dp or 0)
            self.hl_positions.append({"coin": coin, "side": side, "size": short(size), "entry": short(entry), "mark": mark,
                                      "liq": short(liq), "leverage": lev, "unrealized_pnl": pnl,
                                      "margin": round(size * entry / lev, 2)})
            return
        raise RuntimeError("could not build position")

    # ---- app-shaped tools
    def t_get_price(self, symbol):
        s = str(symbol).upper().strip()
        if s not in sim.ASSETS:
            return {"error": f"No Hyperliquid market for {s}. Answer from what you know instead, and say there's no live price here."}
        return {"change_24h_pct": short(self.change[s]), "price": short(self.prices[s]),
                "source": "Hyperliquid mark price", "symbol": s}

    def t_hyperliquid_account(self):
        if self.no_wallet:
            return {"error": f"No wallet on this {self.device} yet"}
        used = round(sum(p["margin"] for p in self.hl_positions), 2)
        upnl = round(sum(p["unrealized_pnl"] for p in self.hl_positions), 2)
        value = round(self.hl_value + upnl, 2)
        return {"account_value": short(value), "available_usdc": short(round(max(value - used, 0), 2)),
                "margin_used": short(used),
                "open_orders": [{"coin": o["coin"], "price": short(o["price"]), "side": o["side"], "size": short(o["size"])}
                                for o in self.hl_orders],
                "positions": [{"coin": p["coin"], "entry": short(p["entry"]), "leverage": p["leverage"],
                               "liquidation": short(p["liq"]), "side": p["side"], "size": short(p["size"]),
                               "unrealised_pnl": short(p["unrealized_pnl"])} for p in self.hl_positions]}

    def t_prepare_close(self, coin, percent=None):
        if self.no_wallet:
            return {"error": f"No wallet on this {self.device} to trade with"}
        if self.blocked:
            return {"note": "Nothing was prepared. Tell the user why.", "reasons": [self.blocked], "status": "blocked"}
        c = str(coin).upper().strip()
        p = next((x for x in self.hl_positions if x["coin"] == c), None)
        if p is None:
            return {"note": f"The user has no open {c} position on Hyperliquid. Tell them.", "prepared": False,
                    "status": "no_position"}
        if percent is not None and not (isinstance(percent, (int, float)) and 0 < percent <= 100):
            return {"error": "percent must be between 1 and 100"}
        sdp = 4 if p["mark"] >= 1000 else 2 if p["mark"] >= 10 else 1 if p["mark"] >= 1 else 0
        if percent is None or percent == 100:
            closing = f"all of the {c} {p['side']}, {p['size']} {c}"
            upnl = p["unrealized_pnl"]
        else:
            qty = sdec(Decimal(str(p["size"])) * Decimal(str(percent)) / 100, sdp + 2)
            closing = f"{percent}% of the {c} {p['side']}, {qty} {c}"
            upnl = sdec(Decimal(str(p["unrealized_pnl"])) * Decimal(str(percent)) / 100, 2)
        return {"closing": closing, "executed": False, "kind": "close", "note": "Waiting for the user to approve with Face ID.",
                "order": "Market, reduce only", "status": "review_opened", "unrealised_pnl": short(upnl)}


def new_world(rng, split, specs, bio, no_wallet=False, blocked=None, orders=False):
    """specs: list of (coin, side, pnl_sign or None)."""
    w = CloseWorld(rng, "Mac" if (bio == "Touch ID" and rng.random() < 0.9) else "iPhone")
    w.hl_value = round(rng.uniform(400, 25000), 2)
    for coin, side, ps in specs:
        w.add_position(rng, coin, side, ps)
    if orders and rng.random() < 0.3:
        coin = rng.choice(["BTC", "ETH", "SOL"])
        w.hl_orders.append({"coin": coin, "side": rng.choice(["buy", "sell"]), "type": "limit",
                            "price": round(w.prices[coin] * rng.choice([0.95, 0.97, 1.03, 1.05]), sim.ASSETS[coin][2]),
                            "size": round(rng.uniform(0.01, 2), 3)})
    w.no_wallet, w.blocked = no_wallet, blocked
    return w


def rand_specs(rng, split, k, must=None, exclude=(), signs=None):
    """k positions; `must` = (coin, side, pnl_sign) forced in (position 0 before shuffle)."""
    exc = list(exclude) + ([must[0]] if must else [])
    coins = pick_coins(rng, split, k - (1 if must else 0), exc)
    specs = [(c, rng.choice(["long", "short"]), None) for c in coins]
    if must:
        specs.append(must)
    rng.shuffle(specs)
    if len(specs) >= 2 and len({s[1] for s in specs}) == 1 and rng.random() < 0.5 and not (must and False):
        i = rng.randrange(len(specs))
        if not must or specs[i][0] != must[0]:
            specs[i] = (specs[i][0], "short" if specs[i][1] == "long" else "long", specs[i][2])
    return specs

# ------------------------------------------------------------------ phrase pools
# tag|text. Tags: L long only, S short only, H half only, W word-percent only (half/quarter/third), N numeric only.
FULL = [
    "ok close my {c} {s} now", "close my {c} {s}", "close {c}", "exit {c}", "exit my {c} {s}",
    "close out my {c} {s}", "close my {c} position", "get me out of {c}", "get me out of my {c} {s}", "close the {c} trade",
    "shut my {c} {s}", "i want out of {c}", "i wanna close my {c} {s}", "can you close my {c} {s}?", "pls close my {c} position",
    "{c} {s}, close it", "close {c} {s} rn", "take me out of my {c} trade", "end my {c} {s}", "close out {c}",
    "bail on my {c} {s}", "exit the {c} position", "I'd like to close my {c} position please", "hey, close my {c} {s} for me",
    "close my {c} perp", "close the {c} perp position", "close my {c} on hyperliquid", "{c}: close", "close that {c} {s} please",
    "I'm done with {c}, close it", "close my {c} {s} and get me back to USDC", "flatten my {c}", "zero out my {c} {s}",
    "close my entire {c} {s}", "close all of my {c} {s}", "L|sell my {c} long", "L|sell off my {c} position",
    "S|buy back my {c} short", "S|cover my {c} short", "close the whole {c} position", "Close my {c} {s}. Thanks",
    "need to get out of my {c} {s} asap", "kill the {c} {s}", "exit {c} pls", "{c} {s} needs to go, close it",
    "close 100% of my {c} {s}", "100% close on {c}", "dump my {c} {s}", "exit my whole {c} {s}",
]
TP = [
    "take profit on my {c}", "take profits on {c}", "lock in profit on my {c} {s}", "cash out my {c} {s}", "book profits on {c}",
    "I'm up on {c}, take profit", "secure my gains on {c}", "take the win on {c}", "cash in my {c}", "tp my {c} {s}",
    "take profit on {c} {s} now", "close my {c} {s} and take the profit", "I'm in profit on {c}, close it",
    "lock the gains on my {c} position", "bank the profit on {c}", "my {c} {s} is green, close it", "time to take profit on {c}",
    "let's take profit on the {c} {s}", "take my profit on {c}", "realize profit on my {c}", "take profit on my {c} {s} please",
    "I'm happy with the {c} gain, close it", "get me out of {c} while I'm up", "sell my {c} winner", "exit {c}, I'm green",
    "cash out of {c}", "close the {c} {s}, taking profit", "profit time on {c}, close it", "take the money off {c}",
]
CUT = [
    "cut my loss on {c}", "cut my losses on {c} {s}", "stop the bleeding on {c}", "get me out of this {c} loss",
    "I'm down on {c}, close it", "bail on my losing {c} {s}", "exit {c} before it gets worse", "{c} is going against me, close it",
    "take the loss on {c}", "close my losing {c} {s}", "I'm underwater on {c}, get me out", "my {c} {s} is red, close it",
    "cut the {c} {s}", "cut {c}", "this {c} trade is bleeding, close it", "{c} {s} is hurting, close it out",
    "I give up on my {c} {s}, close it", "close the {c} {s}, it's not working", "get me out of this losing {c} trade",
    "take the L on {c}", "accept the loss on {c} and close it", "exit my {c} {s} at a loss", "cut my {c} {s} loss",
    "limit the damage on {c}, close it", "{c} dumped on me, close the {s}", "stop the loss on {c}, close it now",
]
PARTIAL = [
    "close {p} of my {c} {s}", "close {p} of my {c} position", "take {p} off my {c} {s}", "trim my {c} {s} by {p}",
    "reduce my {c} {s} by {p}", "reduce my {c} position by {p}", "cut my {c} {s} by {p}", "L|sell {p} of my {c}",
    "close {p} of {c}", "partial close on {c}, {p}", "close {p} on my {c} {s}", "scale out of {c} {s}, {p}",
    "take {p} of my {c} {s} off the table", "can you close {p} of my {c} {s}?", "lighten my {c} {s} by {p}",
    "{c} {s}: close {p}", "pls close {p} of my {c} position", "close just {p} of my {c} {s}", "only close {p} of {c}",
    "shave {p} off my {c} {s}", "bring my {c} {s} down by {p}", "downsize my {c} {s} by {p}", "close {p} of my {c} {s} please",
    "I need to close {p} of my {c} position", "free up margin, close {p} of {c}", "W|close {p} my {c} position",
    "W|close {p} my {c} {s}", "H|cut my {c} {s} in half", "H|halve my {c} {s}", "H|half my {c} {s} please",
    "H|close my {c} {s} in half", "close {p} of that {c} {s}", "take {p} off {c}", "S|cover {p} of my {c} short",
    "N|close {p} of my {c} {s}, keep the rest", "keep the {c} {s} open but close {p} of it",
]
AMBIG = [
    "close my position", "close it", "close my trade", "exit my position", "get me out", "close the position",
    "close position", "i want to close my trade", "can you close my position?", "exit my trade pls", "close my perp",
    "close out my position", "end the trade", "close my hyperliquid position", "get me out of my trade",
    "close my open position", "shut it", "close that", "flatten", "just close it", "close it out", "take me out",
    "bail", "exit", "close the trade", "i'm done, close my position", "close my position pls", "need to close my trade rn",
    "hey can you close my position for me", "close my position please", "kill my position", "exit my perp position",
]
AMBIG_PARTIAL = [
    "close {p} of my position", "close {p} my position", "take {p} off", "trim {p}", "reduce my position by {p}",
    "close {p} of it", "close {p} of my trade", "partial close, {p}", "cut my position by {p}", "scale out {p}",
    "can you close {p} of my position?", "pls take {p} off my position", "lighten my position by {p}",
    "N|close {p} of my open position", "W|close {p} of the trade", "H|cut my position in half", "H|halve my position",
    "reduce my trade by {p}", "close {p} of my perp", "take {p} off the table",
]
SIDE_ONLY = [
    "close my {s}", "close the {s}", "exit my {s} position", "get me out of my {s}", "close my {s} position",
    "close my {s} please", "exit the {s}", "i want to close my {s}", "shut my {s}", "close out the {s}",
    "close my {s} trade", "end my {s}",
]
EVERY = [
    "get me out of everything", "close everything", "close all my positions", "close all", "flatten everything",
    "exit all positions", "close all my perps", "get me out of all my trades", "close out all of it", "panic close everything",
    "I want to be flat", "go flat", "close all positions on hyperliquid", "exit everything rn", "close my positions",
    "close all my trades", "shut everything down", "get me out of all my positions", "close everything pls",
    "close every position", "exit all my trades", "I want out of everything", "flat everything", "close the lot",
]
EVERY_PARTIAL = [
    "close {p} of everything", "take {p} off all my positions", "reduce all my positions by {p}", "trim everything by {p}",
    "close {p} of each position", "cut all my positions by {p}",
]
ASK_REPLY = [
    "{c}", "the {c} one", "{c} {s}", "my {c}", "the {c}", "{c} please", "the {c} {s}", "{c} one", "it's the {c}",
    "close the {c}", "the {c} one pls", "{c} position", "{c} pls", "the {c} position",
]
ASK_REPLY_SIDE = ["the {s}", "the {s} one", "{s}", "my {s}", "the {s} pls"]
ASK_REPLY_ALL2 = ["both", "both of them", "both pls", "the two of them"]
ASK_REPLY_ALL3 = ["all of them", "all three", "all of them pls", "every one"]
AFTER_NOPOS = ["what do I have open then?", "what positions do I have?", "show my positions", "what's open?", "what am I in right now",
               "which positions are open", "so what do I have open", "list my open positions"]
SHORTLONG_REQ = [  # {c} {w} (short/long) ; optional size
    "{w} {c}", "{w} {c} please", "I want to {w} {c}", "go {w} on {c}", "open a {w} on {c}", "{w} {c} for me", "can you {w} {c}?",
    "let's {w} {c}", "{w} some {c}", "{w} {c} with $200", "{w} {c} 5x", "{w} {c} $100 3x", "place a {w} on {c}", "i wanna {w} {c}",
    "{w} {c} now", "{w} {c} rn", "open {c} {w}", "put on a {w} in {c}", "{w} {c} market", "get me {w} {c}",
]
SHORTLONG_REPLY = ["close it", "close the {s}", "close my {c} {s}", "close it please", "close the {c}", "the {s}, close it",
                   "close that one", "close the {c} {s} first", "close it out", "just close the {s}"]
PNL = [
    "what's my PnL", "how am I doing", "how are my positions", "what's my pnl right now", "show my pnl", "am I up or down",
    "how are my trades doing", "pnl?", "how's my account doing", "what's my unrealised pnl", "how much am I up",
    "how much am I down", "am I in profit", "check my positions", "how are my perps doing", "what's my profit",
    "how's my hyperliquid doing", "what's the damage", "give me my pnl", "where am I at on my trades",
]
PNL_COIN = [
    "what's my PnL on {c}", "how's my {c} {s} doing", "am I up on {c}", "how is the {c} trade", "what's my {c} pnl",
    "how's my {c} position", "am I green on {c}", "how much am I up on {c}", "is my {c} {s} in profit", "check my {c} {s}",
]
SHOULD = [
    "should I close my {c} {s}", "should I close my {c} {s}?", "should i close {c}", "do you think I should close my {c} {s}",
    "is it time to close my {c} {s}?", "should I exit my {c} position", "worth closing my {c} {s} here?",
    "thinking about closing my {c} {s}, what do you think", "should I take profit on {c} or hold", "should I cut my {c} {s}",
    "close my {c} {s} or hold?", "good time to exit {c}?", "should I keep my {c} {s} open", "is it smart to close {c} now",
    "what do you think, close my {c} {s}?", "i'm not sure if i should close my {c} {s}", "hold or close my {c} {s}",
]
CANCEL = [
    "cancel my limit order", "cancel my {c} order", "remove my open order", "delete my limit order", "kill my pending order",
    "cancel that order on {c}", "pull my resting order", "cancel all my orders", "cancel my open orders", "can you cancel my {c} limit order",
    "cancel the order I placed", "get rid of my limit order", "cancel my {c} limit", "pls cancel my order", "cancel the {c} buy order",
    "cancel the {c} sell order", "i want to cancel my limit order", "undo my limit order on {c}",
]
LEVERAGE = [
    "reduce my leverage", "lower my leverage", "lower my {c} leverage", "can you reduce my leverage on {c}", "drop my leverage",
    "I want less leverage on my {c} {s}", "change my {c} leverage to 3x", "set my leverage to 2x", "decrease the leverage on my position",
    "reduce leverage on my {c} {s}", "my leverage is too high, lower it", "bring my leverage down", "cut my leverage on {c}",
]
HOWTO = [
    "how do I close a position", "how can I close my trade", "how do I exit a perp position", "how do i close my position?",
    "what's the way to close a trade", "how do I close part of a position", "how to close my position",
]

# eval-only share: index % 5 == 0 (plus a few shared templates in eval so basics are tested too)
def pool_for(pool, split, rng):
    idx = list(range(len(pool)))
    ev = [i for i in idx if i % 5 == 0]
    tr = [i for i in idx if i % 5 != 0]
    if split == "train":
        use = tr
    else:
        use = ev if rng.random() < 0.7 else idx
    return use


def pick_tpl(rng, pool, split, side=None, pk=None, avoid=()):
    use = pool_for(pool, split, rng)
    for _ in range(80):
        t = pool[rng.choice(use)]
        tags, text = (t.split("|", 1) if "|" in t and t.split("|", 1)[0].isalpha() and t.split("|", 1)[0].isupper() else ("", t))
        if "L" in tags and side != "long":
            continue
        if "S" in tags and side != "short":
            continue
        if "H" in tags and pk != "half":
            continue
        if "W" in tags and pk not in ("half", "quarter", "third"):
            continue
        if "N" in tags and pk != "num":
            continue
        return text
    raise RuntimeError("no template")


PHRASE_PROTECT = set(WORD2COIN) | {"long", "short", "half", "halve", "quarter", "third", "percent", "pct", "all", "both", "everything", "every"}


def typo_layer(rng, text):
    words = text.split(" ")
    cand = [i for i, w in enumerate(words)
            if len(re.sub(r"\W", "", w)) >= 4 and not re.search(r"\d", w) and re.sub(r"\W", "", w).lower() not in PHRASE_PROTECT]
    ops = ["swap", "drop", "double", "abbr"]
    for _ in range(4):
        op = rng.choice(ops)
        if op == "abbr":
            for a, b in (("please", rng.choice(["pls", "plz"])), ("pls", "plz"), ("right now", "rn"), ("you", "u"), ("for me", "4me")):
                if re.search(r"\b" + a + r"\b", text, re.I):
                    return re.sub(r"\b" + a + r"\b", b, text, count=1, flags=re.I)
            continue
        if not cand:
            continue
        i = rng.choice(cand)
        w = words[i]
        core = re.sub(r"^\W+|\W+$", "", w)
        s = w.index(core)
        if op == "swap":
            j = rng.randrange(len(core) - 1)
            core2 = core[:j] + core[j + 1] + core[j] + core[j + 2:]
        elif op == "drop":
            vs = [k for k, ch in enumerate(core) if ch in "aeiou" and k > 0]
            if not vs:
                continue
            k = rng.choice(vs)
            core2 = core[:k] + core[k + 1:]
        else:
            k = rng.randrange(1, len(core))
            core2 = core[:k] + core[k] + core[k:]
        if core2 == core:
            continue
        words[i] = w[:s] + core2 + w[s + len(core):]
        return " ".join(words)
    return text


def finish_text(rng, text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.?!:])", r"\1", text)
    if rng.random() < 0.15:
        text = typo_layer(rng, text)
    r = rng.random()
    if r < 0.25:
        text = text.lower()
    elif r < 0.27:
        text = text.upper()
    elif r < 0.33:
        text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        r = rng.random()
        text += "." if r < 0.18 else "!" if r < 0.24 else ""
    return text


def fill(rng, tpl, c=None, s=None, p=None, w=None, side_prob=0.75):
    sd = s if (s and rng.random() < side_prob) else ""
    return tpl.format(c=coin_text(rng, c) if c else "", s=sd, p=p or "", w=w or "")


# ---- percent kinds
TRAIN_NUM = [5, 10, 20, 30, 33, 40, 60, 70, 75, 80, 90]
EVAL_NUM_ONLY = [15, 35, 45, 55, 65, 85, 12.5]
WORD_NUM = {10: "ten", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def pick_pct(rng, split):
    r = rng.random()
    if r < (0.30 if split == "train" else 0.2):
        kind, val = "half", 50
    elif r < 0.44:
        kind, val = "quarter", 25
    elif r < 0.52:
        kind, val = "third", 33.33
    else:
        kind = "num"
        if split == "train":
            val = rng.choice(TRAIN_NUM)
        else:
            val = rng.choice(EVAL_NUM_ONLY) if rng.random() < 0.6 else rng.choice([10, 20, 30, 60, 75])
    return kind, val


def pct_phrase(rng, kind, val):
    if kind == "half":
        return rng.choice(["half", "half", "half", "50%", "50 percent", "fifty percent", "1/2"])
    if kind == "quarter":
        return rng.choice(["a quarter", "a quarter", "25%", "25 percent", "one quarter", "1/4"])
    if kind == "third":
        return rng.choice(["a third", "a third", "one third", "1/3", "33.33%"])
    v = f"{val:g}"
    if val in WORD_NUM and rng.random() < 0.1:
        return f"{WORD_NUM[val]} percent"
    return rng.choice([f"{v}%", f"{v}%", f"{v} percent", f"{v} pct", f"{v}%"])

# ------------------------------------------------------------------ answer texts
def approve(rng, bio="Face ID"):
    return rng.choice([f"Review it and approve with {bio}.", f"Check the review, then approve with {bio}.",
                       f"It's waiting for your {bio} approval.", f"Approve it with {bio} when you're ready.",
                       f"Nothing happens until you approve it with {bio}.", f"Review it and approve with {bio}, or cancel.",
                       f"The review is open. Approve it with {bio} to go ahead."])


def pnl_phrase(rng, v):
    a = money(v)
    if v == 0:
        return rng.choice(["Unrealised PnL on that is about $0.", "It's flat right now, about $0 unrealised."])
    if v > 0:
        return rng.choice([f"Unrealised PnL on that is about +{a}.", f"That position is up about {a} unrealised.",
                           f"It's sitting at about +{a} unrealised.", f"Unrealised PnL there is roughly +{a}.",
                           f"You're up about {a} on it right now.", f"Unrealised, that's about +{a}."])
    return rng.choice([f"Unrealised PnL on that is about -{a}.", f"That position is down about {a} unrealised.",
                       f"It's sitting at about -{a} unrealised.", f"Unrealised PnL there is roughly -{a}.",
                       f"You're down about {a} on it right now.", f"Unrealised, that's about -{a}."])


def order_l(order):
    return order[0].lower() + order[1:]


def lead_text(rng, closing, order):
    o = order_l(order)
    return rng.choice([f"I've prepared closing {closing}, {o}.", f"I've prepared closing {closing}: {o}.",
                       f"Prepared: closing {closing}, {o}.", f"Closing {closing} is prepared, {o}.",
                       f"I've prepared a close for {closing}, {o}.", f"I've set up closing {closing}, {o}, for your review."])


def single_ok_text(rng, r):
    return " ".join([lead_text(rng, r["closing"], r["order"]), pnl_phrase(rng, r["unrealised_pnl"]), approve(rng)])


def multi_ok_text(rng, coins, rs):
    n = len(rs)
    closings = "; ".join(r["closing"] for r in rs)
    same = len({r["order"] for r in rs}) == 1
    lead = rng.choice([f"I've prepared {NUMW[n]} closes: {closings}.", f"Prepared {NUMW[n]} closes: {closings}.",
                       f"I've prepared closing all {NUMW[n]} positions: {closings}." if (len(coins) == n and all(r["closing"].startswith("all of") for r in rs)) else f"I've prepared {NUMW[n]} closes: {closings}."])
    orders = (rng.choice([f"All are {order_l(rs[0]['order'])}.", f"Each one is {order_l(rs[0]['order'])}."]) if same else "")
    parts = [f"{money(r['unrealised_pnl']).replace('$', ('+$' if r['unrealised_pnl'] > 0 else '-$') if r['unrealised_pnl'] else '$')} on the {c}"
             for c, r in zip(coins, rs)]
    pnl = rng.choice(["Unrealised PnL is about ", "Unrealised, that's about ", "PnL so far is about "]) + join_and(parts) + "."
    return " ".join(x for x in [lead, orders, pnl, rng.choice(["Review them and approve with Face ID.", "Check each review, then approve with Face ID.",
                                                               "They're waiting for your Face ID approval.", "Approve them with Face ID when you're ready."])] if x)


def ask_text(rng, plist, pct_given=False):
    lst = join_and([pos_phrase(p) for p in plist])
    n = NUMW[len(plist)]
    return rng.choice([f"You have {lst}. Which one?", f"You've got {lst} open. Which one should I close?",
                       f"Which one do you mean? You have {lst}.", f"I see {lst}. Which one do you want to close?",
                       f"You have {lst} open. Which should I close?", f"{n.capitalize()} positions are open: {lst}. Which one?",
                       f"You have {lst}. Which one are you after?"])


def ask_many_text(rng, plist):
    lst = join_and([pos_phrase(p) for p in plist])
    n = NUMW[len(plist)]
    return rng.choice([f"You have {n} open positions: {lst}. That's a lot to close at once, so which ones should I close?",
                       f"You have {n} positions open: {lst}. Which ones do you want to close?",
                       f"I see {n} open positions: {lst}. Which of them should I close?",
                       f"You've got {n} positions: {lst}. Tell me which ones to close."])


def no_pos_text(rng, coin):
    base = rng.choice([f"You don't have an open {coin} position on Hyperliquid, so there's nothing to close.",
                       f"There's no open {coin} position on your Hyperliquid account, so I haven't prepared anything.",
                       f"I don't see an open {coin} position on Hyperliquid, so there's nothing to close.",
                       f"No open {coin} position on Hyperliquid, so nothing was prepared.",
                       f"You have no open {coin} position on Hyperliquid, so there's nothing for me to close."])
    if rng.random() < 0.5:
        base += " " + rng.choice(["Want me to check what you do have open?", "Should I look at your open positions?",
                                  "Want me to pull up your open positions?"])
    return base


def none_open_text(rng):
    return rng.choice(["You have no open positions on Hyperliquid, so there's nothing to close.",
                       "There are no open positions on your Hyperliquid account, so nothing to close.",
                       "I don't see any open positions on Hyperliquid, so there's nothing to close.",
                       "Nothing is open on your Hyperliquid account right now, so there's nothing to close."])


def wallet_text(rng, err):
    dev = re.search(r"this (iPhone|Mac)", err).group(1)
    return rng.choice([f"I can't do that yet because there's no wallet on this {dev} to trade with. Set one up in Wallets, then ask me again.",
                       f"There's no wallet on this {dev} yet, so I can't prepare a close. Create or import one in Wallets and try again.",
                       f"I can't prepare that because this {dev} has no wallet to trade with. Add one in Wallets first.",
                       f"No wallet on this {dev} to trade with, so nothing was prepared. Set one up in Wallets and ask again.",
                       f"This {dev} has no wallet yet, so I can't close anything. Set one up in Wallets first.",
                       f"I need a wallet on this {dev} to do that. Set one up in Wallets, then ask me again."])


def blocked_text(rng, reasons):
    rs = reasons[0]
    return rng.choice([f"I couldn't prepare that. {rs} Nothing was prepared.", f"That was blocked. {rs} Nothing was prepared.",
                       f"Nothing was prepared. {rs}", f"I can't prepare that close right now. {rs}",
                       f"The close was blocked. {rs} Nothing is waiting for approval.", f"I couldn't set that up. {rs} Nothing was prepared."])


def acct_list_text(rng, positions):
    lst = join_and([pos_phrase(p) for p in positions])
    return rng.choice([f"Right now you have {lst} open.", f"You have {lst} open.", f"Open on Hyperliquid: {lst}.",
                       f"You're in {lst}.", f"Your open positions are {lst}."])


def updown(v):
    return "up" if v > 0 else "down" if v < 0 else "flat at"


def pnl_line_text(rng, positions):
    parts = []
    for p in positions:
        v = p["unrealised_pnl"]
        parts.append(f"your {p['coin']} {p['side']} is {updown(v)} {money(v)}" if v else f"your {p['coin']} {p['side']} is flat at $0")
    s = join_and(parts)
    s = s[0].upper() + s[1:]
    tail = ""
    if len(positions) == 1:
        p = positions[0]
        tail = " " + rng.choice([f"Entry {money(p['entry'])}, {p['leverage']}x.", f"It's a {p['leverage']}x from {money(p['entry'])}.",
                                 f"Entry was {money(p['entry'])} at {p['leverage']}x."])
    return s + "." + tail


def view_text(rng, coin, side, price_r, pos):
    chg = price_r["change_24h_pct"]
    pr = price_r["price"]
    v = pos["unrealised_pnl"]
    move = f"{coin} is at {money(pr)}, {'up' if chg >= 0 else 'down'} {abs(chg):g}% over the past day."
    mine = rng.choice([
        f"Your {side} from {money(pos['entry'])} is {updown(v)} {money(v)}, with liquidation at {money(pos['liquidation'])}.",
        f"You're {pos['leverage']}x {side} from {money(pos['entry'])}, {updown(v)} {money(v)}, liquidation {money(pos['liquidation'])}."]) \
        if v and pos.get("liquidation") else f"Your {side} is {pos['leverage']}x from {money(pos['entry'])}."
    against = (side == "long" and chg < 0) or (side == "short" and chg > 0)
    if abs(chg) < 1.5:
        lean = rng.choice(["The move is small, so there's no clear edge either way.", "Not much happening, so there's no clear edge to close or hold."])
    elif chg >= 5:
        lean = ("It has run up hard, so taking some off is reasonable, though it can keep going." if side == "long"
                else "It's pushing against your short, and a squeeze can run further.")
    elif chg <= -5:
        lean = ("It's sliding against your long and could keep falling." if side == "long"
                else "It's moving your way, but a bounce can come fast.")
    else:
        lean = ("It's drifting against you, so the risk is worth a look." if against else "It's moving your way, but that can turn.")
    nopred = rng.choice(["I can't predict the next move, so it comes down to your risk.", "Nobody can call the next move, so decide on the risk you want.",
                         "I can't know where it goes next, so it's about how much risk you want."])
    risk = rng.choice([f"At {pos['leverage']}x a sharp move hurts quickly.", f"{pos['leverage']}x leverage means a fast move can cut deep.",
                       "Closing locks in the result; holding keeps the risk on."])
    offer = rng.choice(["Want me to prepare closing it?", "Should I prepare the close?", "Want me to set up the close?",
                        "Shall I prepare closing it for you?"])
    return " ".join([move, mine, lean, nopred, risk, offer])


def shortlong_text(rng, p, want):
    other = "short" if want == "long" else "long"
    pp = pos_phrase(p)
    return rng.choice([f"You already have {pp} open. Do you want to close it, or add {art(want)} {want} anyway?",
                       f"You have {pp} open. Should I close it, or add {art(want)} {want} on top?",
                       f"There's already {pp} open. Close it, or add {art(want)} {want}?",
                       f"You're in {pp} right now. Do you want me to close that, or add {art(want)} {want}?",
                       f"I see {pp} open. Do you want to close it or add {art(want)} {want}?",
                       f"You hold {pp} at the moment. Close it, or open {art(want)} {want} as well?"])


def cancel_text(rng):
    return rng.choice([
        "I can't cancel orders from chat. Open Trade, go to Positions, and tap Cancel next to the order under Open orders. You approve it with Face ID.",
        "I can't cancel orders from here. In Trade, open Positions, find the order under Open orders and tap Cancel, then approve with Face ID.",
        "Cancelling an order isn't something I can do from chat. Go to Trade, then Positions, and tap Cancel on it under Open orders. It asks for Face ID.",
        "I can't cancel orders from chat. You can do it in Trade > Positions: tap Cancel next to the order in Open orders and approve with Face ID.",
        "That's not something I can do from chat. Use Cancel next to the order under Open orders in Trade > Positions, and approve it with Face ID.",
        "I can't cancel an order from here. Open Trade, Positions, then tap Cancel on the order under Open orders. You'll approve with Face ID."])


def leverage_text(rng):
    return rng.choice([
        "I can't change leverage from chat. Closing part of the position also cuts your risk, and I can prepare that. Want me to?",
        "I can't change leverage from here. A partial close shrinks the position and the risk, and I can prepare one for you. Want that?",
        "Changing leverage isn't something I can do from chat. If you want less risk, I can prepare closing some of the position. Should I?",
        "I can't set leverage from chat. What I can do is prepare a partial close, which lowers your exposure. Want me to do that?",
        "Leverage can't be changed from chat. Closing a share of the position cuts the risk, and I can prepare it. Want to do that?",
        "I can't adjust leverage from here. I can prepare closing part of the position instead if you want less risk."])


def howto_text(rng):
    return rng.choice([
        "Just tell me which position, like 'close my SOL long', and I'll prepare it. You review it and approve with Face ID.",
        "Tell me the coin, and a share if you only want part, like 'close half my ETH short'. I'll prepare it and you approve with Face ID.",
        "Ask me here, for example 'close my BTC long'. I'll open the review and you approve it with Face ID.",
        "Say which position to close, all or a share of it. I'll prepare the order and you approve it with Face ID.",
        "You can ask me, like 'close 25% of my SOL'. I prepare it, you check the review and approve with Face ID."])

# ------------------------------------------------------------------ conversation plumbing
class CConv(BA.Conv):
    def __init__(self, rng, world):
        super().__init__(rng, world, system=SYSTEM_FACE)


def pick_bio(rng):
    return "Face ID" if rng.random() < 0.7 else "Touch ID"


def rec(c, kind, bio, gold, checks, slots):
    ex = {"messages": c.msgs, "tools": True,
          "meta": {"source": GEN_NAME, "generator": GEN_NAME, "kind": kind, "bio": bio, "lang": "en",
                   "tools_registry": "spec/hermit_tools_v2.json", "world": c.world0, "world_final": c.w.state(),
                   "expect": c.expect, "gold": gold, "checks": checks, "slots": slots}}
    return ex


def apply_bio(ex):
    bio = ex["meta"]["bio"]
    if bio == "Face ID":
        return ex
    for m in ex["messages"]:
        if isinstance(m.get("content"), str):
            m["content"] = m["content"].replace("Face ID", bio)
    return ex


def pos_by_coin(r):
    return {p["coin"]: p for p in r["positions"]}


CHK_BASE = ["right_tool_new", "no_unbacked_done", "no_prepare_trade"]


def gold_of(c, forbid=("prepare_trade",), **kw):
    g = {"steps": c.expect, "forbidden_tools": list(forbid)}
    g.update(kw)
    return g


class Retry(Exception):
    pass

# ------------------------------------------------------------------ positive scenarios
def finish_single(rng, c, r, coin):
    if r.get("status") == "review_opened":
        c.say(single_ok_text(rng, r))
    else:
        raise Retry


def sc_full_named(rng, split, flavor):
    bio = pick_bio(rng)
    k = rng.choice([1, 1, 2, 2, 3])
    ps = {"tp": 1, "cut": -1}.get(flavor)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    specs = rand_specs(rng, split, k, must=(coin, side, ps))
    w = new_world(rng, split, specs, bio, orders=True)
    c = CConv(rng, w)
    pool = {"plain": FULL, "tp": TP, "cut": CUT}[flavor]
    tpl = pick_tpl(rng, pool, split, side=side)
    c.user(finish_text(rng, fill(rng, tpl, c=coin, s=side)))
    r = c.call(("prepare_close", {"coin": coin}))
    finish_single(rng, c, r, coin)
    return rec(c, f"full_named_{flavor}", bio, gold_of(c, coin=coin, percent=None, must_ask=False),
               CHK_BASE + ["asked_when_needed"], {"flavor": flavor})


def sc_partial_named(rng, split):
    bio = pick_bio(rng)
    k = rng.choice([1, 1, 2, 2, 3])
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    specs = rand_specs(rng, split, k, must=(coin, side, None))
    w = new_world(rng, split, specs, bio, orders=True)
    c = CConv(rng, w)
    kind, val = pick_pct(rng, split)
    tpl = pick_tpl(rng, PARTIAL, split, side=side, pk=kind)
    c.user(finish_text(rng, fill(rng, tpl, c=coin, s=side, p=pct_phrase(rng, kind, val))))
    r = c.call(("prepare_close", {"coin": coin, "percent": val}))
    finish_single(rng, c, r, coin)
    return rec(c, "partial_named", bio, gold_of(c, coin=coin, percent=val, must_ask=False),
               CHK_BASE + ["exact_digits", "asked_when_needed"], {"pct_kind": kind})


def sc_one_unnamed(rng, split):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = new_world(rng, split, [(coin, side, None)], bio, orders=True)
    c = CConv(rng, w)
    partial = rng.random() < 0.32
    kind, val = pick_pct(rng, split) if partial else (None, None)
    if partial:
        tpl = pick_tpl(rng, AMBIG_PARTIAL, split, pk=kind)
        txt = tpl.format(p=pct_phrase(rng, kind, val))
    else:
        txt = pick_tpl(rng, AMBIG, split)
    c.user(finish_text(rng, txt))
    a = c.call(("hyperliquid_account", {}))
    args = {"coin": a["positions"][0]["coin"]}
    if partial:
        args["percent"] = val
    r = c.call(("prepare_close", args))
    finish_single(rng, c, r, coin)
    return rec(c, "one_position_unnamed" + ("_partial" if partial else ""), bio,
               gold_of(c, coin=coin, percent=val, must_ask=False),
               CHK_BASE + ["asked_when_needed"] + (["exact_digits"] if partial else []), {"pct_kind": kind})


def reply_text(rng, split, target, plist):
    sides = [p["side"] for p in plist]
    if sides.count(target["side"]) == 1 and rng.random() < 0.18:
        txt = rng.choice(ASK_REPLY_SIDE).format(s=target["side"])
    else:
        tpl = pick_tpl(rng, ASK_REPLY, split)
        txt = tpl.format(c=coin_text(rng, target["coin"]), s=target["side"] if rng.random() < 0.3 else "")
    return finish_text(rng, txt)


def sc_ambiguous(rng, split):
    bio = pick_bio(rng)
    k = rng.choice([2, 2, 3])
    specs = rand_specs(rng, split, k)
    w = new_world(rng, split, specs, bio, orders=True)
    c = CConv(rng, w)
    partial = rng.random() < 0.3
    kind, val = pick_pct(rng, split) if partial else (None, None)
    if partial:
        txt = pick_tpl(rng, AMBIG_PARTIAL, split, pk=kind).format(p=pct_phrase(rng, kind, val))
    else:
        txt = pick_tpl(rng, AMBIG, split)
    c.user(finish_text(rng, txt))
    a = c.call(("hyperliquid_account", {}))
    plist = a["positions"]
    c.say(ask_text(rng, plist), ask=True)
    first_gold = gold_of(c, must_ask=True, asked_listing=[p["coin"] for p in plist], percent=val)
    first = rec(c, "ambiguous_ask", bio, first_gold, ["right_tool_new", "asked_when_needed", "no_unbacked_done", "no_prepare_trade"],
                {"pct_kind": kind, "second_turn": False})
    if rng.random() > (0.4 if split == "train" else 0.5):
        return first
    # second turn: user answers
    everyone = rng.random() < 0.12
    if everyone:
        c.user(finish_text(rng, rng.choice(ASK_REPLY_ALL2 if k == 2 else ASK_REPLY_ALL3)))
        calls = [("prepare_close", ({"coin": p["coin"], "percent": val} if partial else {"coin": p["coin"]})) for p in plist]
        rs = c.call(*calls)
        rs = rs if isinstance(rs, list) else [rs]
        c.say(multi_ok_text(rng, [p["coin"] for p in plist], rs))
        return rec(c, "ambiguous_ask_all", bio, gold_of(c, must_ask=False, coins=[p["coin"] for p in plist], percent=val),
                   CHK_BASE + ["asked_when_needed"] + (["exact_digits"] if partial else []), {"pct_kind": kind, "second_turn": True})
    target = rng.choice(plist)
    c.user(reply_text(rng, split, target, plist))
    args = {"coin": target["coin"]}
    if partial:
        args["percent"] = val
    r = c.call(("prepare_close", args))
    finish_single(rng, c, r, target["coin"])
    return rec(c, "ambiguous_ask_answered", bio, gold_of(c, coin=target["coin"], percent=val, must_ask=False),
               CHK_BASE + ["asked_when_needed"] + (["exact_digits"] if partial else []), {"pct_kind": kind, "second_turn": True})


def sc_side_only(rng, split):
    bio = pick_bio(rng)
    mode = rng.choice(["one", "one", "two", "none"])
    side = rng.choice(["long", "short"])
    opp = "short" if side == "long" else "long"
    if mode == "one":
        n_same, n_opp = 1, rng.choice([0, 1, 2])
    elif mode == "two":
        n_same, n_opp = 2, rng.choice([0, 1])
    else:
        n_same, n_opp = 0, rng.choice([1, 2])
    coins = pick_coins(rng, split, n_same + n_opp)
    specs = [(cn, side, None) for cn in coins[:n_same]] + [(cn, opp, None) for cn in coins[n_same:]]
    rng.shuffle(specs)
    w = new_world(rng, split, specs, bio)
    c = CConv(rng, w)
    c.user(finish_text(rng, pick_tpl(rng, SIDE_ONLY, split).format(s=side)))
    a = c.call(("hyperliquid_account", {}))
    same = [p for p in a["positions"] if p["side"] == side]
    if mode == "one":
        r = c.call(("prepare_close", {"coin": same[0]["coin"]}))
        finish_single(rng, c, r, same[0]["coin"])
        return rec(c, "side_only_one", bio, gold_of(c, coin=same[0]["coin"], percent=None, must_ask=False),
                   CHK_BASE + ["asked_when_needed"], {})
    if mode == "two":
        c.say(ask_text(rng, same), ask=True)
        return rec(c, "side_only_ask", bio, gold_of(c, must_ask=True, asked_listing=[p["coin"] for p in same], percent=None),
                   ["right_tool_new", "asked_when_needed", "no_unbacked_done", "no_prepare_trade"], {})
    lst = join_and([pos_phrase(p) for p in a["positions"]])
    c.say(rng.choice([f"You don't have an open {side} right now. You have {lst}.", f"I don't see an open {side} on Hyperliquid. You have {lst}.",
                      f"No {side} is open at the moment. You have {lst}.", f"There's no {side} open on your account. You have {lst}."]))
    return rec(c, "side_only_none", bio, gold_of(c, must_ask=False, percent=None, nocall_prepare=True),
               ["right_tool_new", "no_unbacked_done", "no_prepare_trade"], {})


def sc_everything(rng, split, mode):
    bio = pick_bio(rng)
    k = {"few": rng.choice([1, 2, 2, 3, 3]), "many": rng.choice([4, 4, 5]), "zero": 0, "partial": rng.choice([2, 3])}[mode]
    specs = rand_specs(rng, split, k) if k else []
    w = new_world(rng, split, specs, bio, orders=True)
    c = CConv(rng, w)
    kind, val = pick_pct(rng, split) if mode == "partial" else (None, None)
    if mode == "partial":
        txt = pick_tpl(rng, EVERY_PARTIAL, split, pk=kind).format(p=pct_phrase(rng, kind, val))
    else:
        txt = pick_tpl(rng, EVERY, split)
    c.user(finish_text(rng, txt))
    a = c.call(("hyperliquid_account", {}))
    plist = a["positions"]
    if not plist:
        c.say(none_open_text(rng))
        return rec(c, "everything_none", bio, gold_of(c, must_ask=False, percent=None, nocall_prepare=True),
                   ["right_tool_new", "no_unbacked_done", "no_prepare_trade"], {})
    if len(plist) > 3:
        c.say(ask_many_text(rng, plist), ask=True)
        return rec(c, "everything_many_ask", bio, gold_of(c, must_ask=True, asked_listing=[p["coin"] for p in plist], percent=None),
                   ["right_tool_new", "asked_when_needed", "no_unbacked_done", "no_prepare_trade"], {})
    calls = [("prepare_close", ({"coin": p["coin"], "percent": val} if val else {"coin": p["coin"]})) for p in plist]
    rs = c.call(*calls)
    if len(plist) == 1:
        finish_single(rng, c, rs, plist[0]["coin"])
    else:
        c.say(multi_ok_text(rng, [p["coin"] for p in plist], rs))
    return rec(c, "everything_" + ("partial" if val else "full"), bio,
               gold_of(c, coins=[p["coin"] for p in plist], percent=val, must_ask=False, parallel_closes=len(plist) > 1),
               CHK_BASE + ["asked_when_needed"] + (["exact_digits"] if val else []), {"n_positions": len(plist)})


def sc_no_position(rng, split):
    bio = pick_bio(rng)
    k = rng.choice([0, 1, 1, 2])
    coin = pick_coins(rng, split, 1)[0]
    held = pick_coins(rng, split, k, exclude=[coin]) if k else []
    specs = [(h, rng.choice(["long", "short"]), None) for h in held]
    w = new_world(rng, split, specs, bio)
    c = CConv(rng, w)
    side = rng.choice(["long", "short"])
    partial = rng.random() < 0.3
    kind, val = pick_pct(rng, split) if partial else (None, None)
    if partial:
        txt = fill(rng, pick_tpl(rng, PARTIAL, split, side=side, pk=kind), c=coin, s=side, p=pct_phrase(rng, kind, val))
    else:
        txt = fill(rng, pick_tpl(rng, FULL, split, side=side), c=coin, s=side)
    c.user(finish_text(rng, txt))
    args = {"coin": coin, "percent": val} if partial else {"coin": coin}
    r = c.call(("prepare_close", args))
    if r.get("status") != "no_position":
        raise Retry
    c.say(no_pos_text(rng, coin))
    if rng.random() < 0.15:
        c.user(finish_text(rng, rng.choice(AFTER_NOPOS)))
        a = c.call(("hyperliquid_account", {}))
        c.say(acct_list_text(rng, a["positions"]) if a["positions"] else none_open_text(rng).replace(", so there's nothing to close", "").replace(", so nothing to close", ""))
        return rec(c, "no_position_then_list", bio, gold_of(c, must_ask=False, percent=None, nocall_prepare=True),
                   ["right_tool_new", "no_unbacked_done", "no_prepare_trade"], {})
    return rec(c, "no_position", bio, gold_of(c, coin=coin, percent=val, must_ask=False, expect_status="no_position"),
               CHK_BASE + ["asked_when_needed"] + (["exact_digits"] if partial else []), {"pct_kind": kind})


def sc_no_wallet(rng, split):
    bio = pick_bio(rng)
    named = rng.random() < 0.6
    coin = pick_coins(rng, split, 1)[0]
    specs = rand_specs(rng, split, rng.choice([1, 2]), must=(coin, rng.choice(["long", "short"]), None))
    w = new_world(rng, split, specs, bio, no_wallet=True)
    c = CConv(rng, w)
    side = [s for s in specs if s[0] == coin][0][1]
    if named:
        c.user(finish_text(rng, fill(rng, pick_tpl(rng, FULL, split, side=side), c=coin, s=side)))
        r = c.call(("prepare_close", {"coin": coin}))
    else:
        c.user(finish_text(rng, pick_tpl(rng, AMBIG, split)))
        r = c.call(("hyperliquid_account", {}))
    if "error" not in r:
        raise Retry
    c.say(wallet_text(rng, r["error"]))
    return rec(c, "no_wallet" + ("_named" if named else "_account"), bio, gold_of(c, coin=coin if named else None, percent=None, must_ask=False,
                                                                                  nocall_prepare=not named),
               CHK_BASE, {})


def sc_blocked(rng, split):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    specs = rand_specs(rng, split, rng.choice([1, 2]), must=(coin, side, None))
    w = new_world(rng, split, specs, bio, blocked=rng.choice(DEVICE_BLOCKED))
    c = CConv(rng, w)
    c.user(finish_text(rng, fill(rng, pick_tpl(rng, FULL, split, side=side), c=coin, s=side)))
    r = c.call(("prepare_close", {"coin": coin}))
    if r.get("status") != "blocked":
        raise Retry
    c.say(blocked_text(rng, r["reasons"]))
    return rec(c, "blocked", bio, gold_of(c, coin=coin, percent=None, must_ask=False), CHK_BASE, {})

# ------------------------------------------------------------------ negative scenarios
def sc_neg_pnl(rng, split):
    bio = pick_bio(rng)
    k = rng.choice([0, 1, 1, 2, 3])
    specs = rand_specs(rng, split, k) if k else []
    w = new_world(rng, split, specs, bio, orders=True)
    c = CConv(rng, w)
    coin_q = None
    if specs and rng.random() < 0.4:
        coin_q = rng.choice(specs)
    elif rng.random() < 0.08:
        coin_q = (pick_coins(rng, split, 1, exclude=[s[0] for s in specs])[0], "long", None)
    if coin_q:
        c.user(finish_text(rng, fill(rng, pick_tpl(rng, PNL_COIN, split), c=coin_q[0], s=coin_q[1])))
    else:
        c.user(finish_text(rng, pick_tpl(rng, PNL, split)))
    a = c.call(("hyperliquid_account", {}))
    pl = a["positions"]
    if coin_q:
        hit = [p for p in pl if p["coin"] == coin_q[0]]
        if hit:
            c.say(pnl_line_text(rng, hit))
        elif pl:
            c.say(f"I don't see an open {coin_q[0]} position. " + acct_list_text(rng, pl))
        else:
            c.say(f"I don't see an open {coin_q[0]} position, and nothing else is open on Hyperliquid either.")
    elif pl:
        c.say(pnl_line_text(rng, pl))
    else:
        c.say(rng.choice(["You have no open positions on Hyperliquid, so there's no PnL to show.", "Nothing is open on your Hyperliquid account right now.",
                          "I don't see any open positions on Hyperliquid, so no PnL right now."]))
    return rec(c, "neg_pnl", bio, gold_of(c, only_tools=["hyperliquid_account"], must_ask=False), ["right_tool_new", "no_unbacked_done", "no_prepare_trade"], {})


def sc_neg_should(rng, split):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    missing = rng.random() < 0.1
    k = rng.choice([1, 1, 2])
    if missing:
        specs = rand_specs(rng, split, k, exclude=[coin])
    else:
        specs = rand_specs(rng, split, k, must=(coin, side, None))
    w = new_world(rng, split, specs, bio)
    c = CConv(rng, w)
    c.user(finish_text(rng, fill(rng, pick_tpl(rng, SHOULD, split), c=coin, s=side)))
    a, pr = c.call(("hyperliquid_account", {}), ("get_price", {"symbol": coin}))
    pos = [p for p in a["positions"] if p["coin"] == coin]
    if pos:
        c.say(view_text(rng, coin, pos[0]["side"], pr, pos[0]))
    else:
        lst = acct_list_text(rng, a["positions"]) if a["positions"] else "Nothing else is open either."
        c.say(f"I don't see an open {coin} position on Hyperliquid, so there's nothing to close. {lst}")
    return rec(c, "neg_should_close", bio, gold_of(c, only_tools=["hyperliquid_account", "get_price"], parallel_unordered=True,
                                                    must_ask=False), ["right_tool_new", "no_unbacked_done", "no_prepare_trade", "no_ask_for_data"], {})


def sc_neg_cancel(rng, split):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    w = new_world(rng, split, rand_specs(rng, split, rng.choice([0, 1, 2])) if rng.random() < 0.8 else [], bio, orders=True)
    c = CConv(rng, w)
    c.user(finish_text(rng, fill(rng, pick_tpl(rng, CANCEL, split), c=coin)))
    c.say(cancel_text(rng))
    return rec(c, "neg_cancel_order", bio, gold_of(c, only_tools=[], must_ask=False), ["restraint", "no_unbacked_done", "no_prepare_trade"], {})


def sc_neg_shortlong(rng, split):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    have = rng.choice(["long", "short"])
    want = "short" if have == "long" else "long"
    specs = rand_specs(rng, split, rng.choice([1, 1, 2]), must=(coin, have, None))
    w = new_world(rng, split, specs, bio)
    c = CConv(rng, w)
    c.user(finish_text(rng, fill(rng, pick_tpl(rng, SHORTLONG_REQ, split), c=coin, w=want)))
    a = c.call(("hyperliquid_account", {}))
    p = [x for x in a["positions"] if x["coin"] == coin][0]
    c.say(shortlong_text(rng, p, want), ask=True)
    first = rec(c, "neg_short_with_long_open", bio, gold_of(c, must_ask=True, asked_listing=[coin], percent=None,
                                                             only_tools=["hyperliquid_account"]),
                ["right_tool_new", "asked_when_needed", "no_unbacked_done", "no_prepare_trade"], {"second_turn": False})
    if rng.random() > (0.28 if split == "train" else 0.4):
        return first
    c.user(finish_text(rng, rng.choice([t for t in SHORTLONG_REPLY if "{c}" not in t or True]).format(
        c=coin_text(rng, coin), s=have)))
    r = c.call(("prepare_close", {"coin": coin}))
    finish_single(rng, c, r, coin)
    return rec(c, "neg_short_with_long_open_close", bio, gold_of(c, coin=coin, percent=None, must_ask=False),
               CHK_BASE + ["asked_when_needed"], {"second_turn": True})


def sc_neg_leverage(rng, split):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = new_world(rng, split, rand_specs(rng, split, rng.choice([1, 2]), must=(coin, side, None)), bio)
    c = CConv(rng, w)
    c.user(finish_text(rng, fill(rng, pick_tpl(rng, LEVERAGE, split), c=coin, s=side)))
    c.say(leverage_text(rng))
    return rec(c, "neg_reduce_leverage", bio, gold_of(c, only_tools=[], must_ask=False), ["restraint", "no_unbacked_done", "no_prepare_trade"], {})


def sc_neg_howto(rng, split):
    bio = pick_bio(rng)
    w = new_world(rng, split, rand_specs(rng, split, rng.choice([0, 1, 2])) if rng.random() < 0.8 else [], bio)
    c = CConv(rng, w)
    c.user(finish_text(rng, pick_tpl(rng, HOWTO, split)))
    c.say(howto_text(rng))
    return rec(c, "neg_howto_close", bio, gold_of(c, only_tools=[], must_ask=False), ["restraint", "no_unbacked_done", "no_prepare_trade"], {})

# ------------------------------------------------------------------ quotas
POS_TRAIN = [("full_plain", 36), ("full_tp", 17), ("full_cut", 17), ("partial", 58), ("one_unnamed", 24), ("ambiguous", 36),
             ("side_only", 12), ("every_few", 17), ("every_many", 5), ("every_zero", 3), ("every_partial", 5),
             ("no_position", 24), ("no_wallet", 8), ("blocked", 8)]
NEG_TRAIN = [("neg_pnl", 18), ("neg_should", 20), ("neg_cancel", 14), ("neg_shortlong", 16), ("neg_leverage", 8), ("neg_howto", 4)]
POS_EVAL = [("full_plain", 4), ("full_tp", 2), ("full_cut", 2), ("partial", 6), ("one_unnamed", 3), ("ambiguous", 4), ("side_only", 1),
            ("every_few", 2), ("every_many", 1), ("no_position", 3), ("no_wallet", 1), ("blocked", 1)]
NEG_EVAL = [("neg_pnl", 2), ("neg_should", 3), ("neg_cancel", 2), ("neg_shortlong", 2), ("neg_leverage", 1)]

SCEN = {
    "full_plain": lambda r, s: sc_full_named(r, s, "plain"), "full_tp": lambda r, s: sc_full_named(r, s, "tp"),
    "full_cut": lambda r, s: sc_full_named(r, s, "cut"), "partial": sc_partial_named, "one_unnamed": sc_one_unnamed,
    "ambiguous": sc_ambiguous, "side_only": sc_side_only, "every_few": lambda r, s: sc_everything(r, s, "few"),
    "every_many": lambda r, s: sc_everything(r, s, "many"), "every_zero": lambda r, s: sc_everything(r, s, "zero"),
    "every_partial": lambda r, s: sc_everything(r, s, "partial"), "no_position": sc_no_position, "no_wallet": sc_no_wallet,
    "blocked": sc_blocked, "neg_pnl": sc_neg_pnl, "neg_should": sc_neg_should, "neg_cancel": sc_neg_cancel,
    "neg_shortlong": sc_neg_shortlong, "neg_leverage": sc_neg_leverage, "neg_howto": sc_neg_howto,
}


def scale_quota(quota, total):
    base = sum(n for _, n in quota)
    out = [(k, max(1, round(n * total / base))) for k, n in quota]
    diff = total - sum(n for _, n in out)
    i = 0
    while diff:
        k, n = out[i % len(out)]
        step = 1 if diff > 0 else -1
        if n + step >= 1:
            out[i % len(out)] = (k, n + step)
            diff -= step
        i += 1
    return out


def user_key(ex):
    return " | ".join(norm(m["content"]) for m in ex["messages"] if m["role"] == "user")


def build(seed, n, split, n_neg, forbid_users=()):
    rng = random.Random(seed)
    pos_q = POS_TRAIN if split == "train" else POS_EVAL
    neg_q = NEG_TRAIN if split == "train" else NEG_EVAL
    plan = ([(k, c) for k, c in scale_quota(pos_q, n - n_neg)] + [(k, c) for k, c in scale_quota(neg_q, n_neg)])
    seen_all = set(forbid_users)
    seen_first = set()
    out = []
    for name, cnt in plan:
        for _ in range(cnt):
            for attempt in range(400):
                r = random.Random(rng.getrandbits(64))
                try:
                    ex = SCEN[name](r, split)
                except Retry:
                    continue
                users = [norm(m["content"]) for m in ex["messages"] if m["role"] == "user"]
                key = user_key(ex)
                if key in seen_first or any(u in seen_all for u in users if len(u) > 12 or len(users) == 1):
                    continue
                if any(u in seen_all for u in users):
                    continue
                seen_first.add(key)
                out.append(apply_bio(ex))
                break
            else:
                raise SystemExit(f"could not generate a unique example for {name}")
    rng.shuffle(out)
    return out

# ------------------------------------------------------------------ self-check
BANNED = [r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|deposited|withdrawn|closed)",
          r"\bI(?:'ve| have)? (sent|bought|swapped|deposited|withdrew|closed|sold)\b", r"successfully",
          r"funds (are|have) (in|arrived)", r"^\s*done\b", r"not financial advice", r"[·•]",
          r"what('s| is) (the )?(current )?price", r"your budget", r"how much (do you|are you)", r"\bclosed\b", r"tap close",
          r"\b(is|was|are|were) (now )?(closed|sold|exited)\b", r"all set", r"should have arrived", r"\bas an ai\b",
          r"[^\x00-\x7f]"]
NUM_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?\$?\d[\d,]*(?:\.\d+)?")
WORDPCT = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def parse_pct(text):
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)", t)
    v = None
    if m:
        v = float(m.group(1))
    else:
        m = re.search(r"\b(ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+percent", t)
        if m:
            v = float(WORDPCT[m.group(1)])
        elif re.search(r"\bhalf\b|\bhalve\b|\b1/2\b|\bin half\b", t):
            v = 50.0
        elif re.search(r"\bquarter\b|\b1/4\b", t):
            v = 25.0
        elif re.search(r"\bthird\b|\b1/3\b", t):
            v = 33.33
    if v is None or v == 100:
        return None
    return short(v)


def parse_coins(text):
    return {WORD2COIN[w] for w in re.findall(r"[a-z]+", text.lower()) if w in WORD2COIN}


def nums_in(obj, acc):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        acc.add(Decimal(str(obj)).copy_abs())
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            acc.add(Decimal(m.replace("$", "").replace(",", "").replace("+", "")).copy_abs())
    elif isinstance(obj, dict):
        for v in obj.values():
            nums_in(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            nums_in(v, acc)


def check_args(tc):
    name, args = tc["name"], tc["arguments"]
    errs = []
    if name not in SCHEMA:
        return [f"unknown tool {name}"]
    sch = SCHEMA[name]
    for req in sch.get("required", []):
        if req not in args:
            errs.append(f"{name}: missing {req}")
    for k, v in args.items():
        if k not in sch["properties"]:
            errs.append(f"{name}: unknown arg {k}")
            continue
        t = sch["properties"][k]["type"]
        ok = {"number": isinstance(v, (int, float)) and not isinstance(v, bool), "string": isinstance(v, str),
              "integer": isinstance(v, int) and not isinstance(v, bool), "array": isinstance(v, list),
              "object": isinstance(v, dict)}[t]
        if not ok:
            errs.append(f"{name}: {k} wrong type {v!r}")
        if "enum" in sch["properties"][k] and v not in sch["properties"][k]["enum"]:
            errs.append(f"{name}: {k} bad enum {v!r}")
        if isinstance(v, float) and v.is_integer():
            errs.append(f"{name}: {k} should be an int {v!r}")
    return errs


def self_check_record(ex, is_eval):
    errs = []
    meta, msgs = ex["meta"], ex["messages"]
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    # 1 json validity
    try:
        json.loads(json.dumps(ex, ensure_ascii=False))
        for m in msgs:
            if m["role"] == "tool":
                json.loads(m["content"])
    except Exception as e:
        return [f"json: {e}"]
    if ex["tools"] is not True or meta["source"] != GEN_NAME:
        errs.append("record shape")
    sysm = msgs[0]
    if sysm["role"] != "system" or "{BIO}" in sysm["content"] or SYSTEM_V2.split("{BIO}")[0] not in sysm["content"]:
        errs.append("system prompt")
    expected_sys = SYSTEM_V2.replace("{BIO}", bio)
    if not sysm["content"].startswith(expected_sys):
        errs.append("system prompt not SYSTEM_AGENT_v2 with BIO")
    blob = json.dumps(msgs, ensure_ascii=False)
    if other in blob or "{BIO}" in blob:
        errs.append(f"bio word mix ({other} in {bio} example)")
    # 2 calls: schema, scenario
    calls = [tc for m in msgs for tc in (m.get("tool_calls") or [])]
    for tc in calls:
        errs += check_args(tc)
        if tc["name"] == "prepare_trade":
            errs.append("prepare_trade present")
        if tc["name"] == "prepare_close" and (list(tc["arguments"]) not in (["coin"], ["coin", "percent"])):
            errs.append(f"prepare_close arg keys {list(tc['arguments'])}")
    # tool message pairing
    i = 0
    n_calls_msgs = 0
    for j, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            if [x["role"] for x in msgs[j + 1:j + 1 + k]] != ["tool"] * k:
                errs.append("tool message count")
            n_calls_msgs += 1
    # 3 replay
    w0 = CloseWorld.from_state(meta["world"], seed=0)
    tool_msgs = [m["content"] for m in msgs if m["role"] == "tool"]
    results = []
    for tc in calls:
        r = w0.call(tc["name"], tc["arguments"])
        results.append(r)
    replay = [json.dumps(r, ensure_ascii=False).replace("Face ID", bio) for r in results]
    if replay != tool_msgs:
        errs.append("replay mismatch")
    wk = {k: v for k, v in meta["world"].items()}
    pcoins = [p["coin"] for p in meta["world"]["hl_positions"]]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    utext = " ".join(users)
    # 4 scenario exactness for prepare_close
    gold = meta["gold"]
    closes = [(tc, r) for tc, r in zip(calls, results) if tc["name"] == "prepare_close"]
    pct_user = parse_pct(utext)
    ucoins = parse_coins(utext)
    for tc, r in closes:
        a = tc["arguments"]
        if r.get("status") == "review_opened":
            full = r["closing"].startswith("all of the")
            if ("percent" not in a) != full:
                errs.append("percent absent iff full close violated")
            if a["coin"] not in pcoins:
                errs.append("coin not a world position")
            if "percent" in a:
                if a["percent"] != pct_user:
                    errs.append(f"percent {a['percent']} != user text {pct_user}")
            elif pct_user is not None:
                errs.append("user gave a percent but call is full")
        if ucoins and a["coin"] not in ucoins and gold.get("coins") is None:
            errs.append(f"coin {a['coin']} not in user text {ucoins}")
        if r.get("status") == "no_position" and a["coin"] in pcoins:
            errs.append("no_position for a held coin")
    # kind-specific call patterns
    kind = meta["kind"]
    names = [tc["name"] for tc in calls]
    if kind.startswith("full_named") or kind in ("partial_named", "blocked"):
        if names != ["prepare_close"]:
            errs.append(f"{kind}: calls {names}")
    if kind.startswith("one_position_unnamed") and names != ["hyperliquid_account", "prepare_close"]:
        errs.append(f"{kind}: calls {names}")
    if kind in ("ambiguous_ask", "side_only_ask", "everything_many_ask", "neg_short_with_long_open") and names != ["hyperliquid_account"]:
        errs.append(f"{kind}: calls {names}")
    if kind == "ambiguous_ask_answered" and names != ["hyperliquid_account", "prepare_close"]:
        errs.append(f"{kind}: calls {names}")
    if kind.startswith("everything_") and kind not in ("everything_none", "everything_many_ask"):
        want = ["hyperliquid_account"] + ["prepare_close"] * len(gold["coins"])
        if names != want:
            errs.append(f"{kind}: calls {names}")
        elif len(gold["coins"]) > 1:
            first = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")][1]
            if len(first["tool_calls"]) != len(gold["coins"]):
                errs.append("closes not in one assistant message")
        if sorted(c["arguments"]["coin"] for c in calls[1:]) != sorted(pcoins):
            errs.append("everything: not every position closed")
    if kind == "neg_pnl" and names != ["hyperliquid_account"]:
        errs.append("neg_pnl calls")
    if kind == "neg_should_close":
        first = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")][0]
        if sorted(t["name"] for t in first["tool_calls"]) != ["get_price", "hyperliquid_account"] or len(names) != 2:
            errs.append("neg_should_close calls")
    if kind in ("neg_cancel_order", "neg_reduce_leverage", "neg_howto_close") and names:
        errs.append(f"{kind} must not call tools")
    # ask lists every candidate position, no invention
    for m in msgs:
        if m["role"] == "assistant" and not m.get("tool_calls") and gold.get("asked_listing") is not None and m is msgs[-1]:
            for cn in gold["asked_listing"]:
                if not re.search(r"\b" + cn + r"\b", m["content"]):
                    errs.append(f"ask text misses {cn}")
    # 5 assistant text: banned, caps, numbers, invention
    src = set()
    for m in msgs:
        if m["role"] == "tool":
            nums_in(json.loads(m["content"]), src)
        if m["role"] == "user":
            nums_in(m["content"], src)
    for tc in calls:
        nums_in(tc["arguments"], src)
    tickers_src = set(re.findall(r"\b[A-Z]{2,5}\b", " ".join(tool_msgs) + " " + utext + json.dumps(calls)))
    tickers_src |= parse_coins(utext)
    texts = [m["content"] for m in msgs if m["role"] == "assistant" and m.get("content")]
    for t in texts:
        for pat in BANNED:
            if re.search(pat, t, re.I | re.M):
                errs.append(f"banned /{pat}/ in: {t[:80]}")
        words = len(t.split())
        cap = 110 if kind == "neg_should_close" else 70
        if words > cap:
            errs.append(f"too long ({words}) {t[:50]}")
        for tok in NUM_RE.findall(t):
            d = Decimal(tok.replace("$", "").replace(",", "").replace("+", "")).copy_abs()
            if d not in src:
                errs.append(f"number {tok} not backed in: {t[:90]}")
        for tk in set(re.findall(r"\b[A-Z]{2,5}\b", t)) & set(sim.ASSETS):
            if tk not in tickers_src and kind != "neg_howto_close":   # how-to answers quote example commands
                errs.append(f"ticker {tk} invented")
        if t.count("?") > 1:
            errs.append("more than one question")
    # prepared claims need a review_opened result of this example
    final = msgs[-1]
    if final["role"] != "assistant" or final.get("tool_calls"):
        errs.append("last message must be assistant text")
    if re.search(r"\bprepared\b", final["content"], re.I) and not any(isinstance(r, dict) and r.get("status") == "review_opened" for r in results):
        if kind not in ("blocked",) and "Nothing was prepared" not in final["content"] and "nothing was prepared" not in final["content"].lower() \
                and "haven't prepared" not in final["content"] and "can prepare" not in final["content"] and "I can prepare" not in final["content"]:
            errs.append("claims prepared without review_opened")
    if any(isinstance(r, dict) and r.get("status") == "review_opened" for r in results):
        if not re.search(r"\b(Face ID|Touch ID)\b", final["content"]):
            errs.append("prepared answer lacks bio word")
        if "prepared" not in final["content"].lower() and "set up" not in final["content"].lower():
            errs.append("answer lacks 'prepared'")
        # PnL in the answer equals a result's unrealised_pnl
        for r in results:
            if isinstance(r, dict) and r.get("status") == "review_opened":
                if money(r["unrealised_pnl"]) not in final["content"] and r["unrealised_pnl"] != 0:
                    errs.append("unrealised pnl not stated")
                if r["closing"] not in final["content"]:
                    errs.append("closing text not copied")
    # 6 render round trip
    try:
        txt = "".join(t for t, trainable in H.render(msgs, True) if trainable)   # assistant segments only (tool block has a <tool_call> example)
        got = []
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                got.append({"name": tc["name"], "arguments": tc["arguments"]})
        parsed, bad = H.parse_calls(txt)
        if bad or parsed != got:
            errs.append("render/parse_calls round trip")
    except Exception as e:
        errs.append(f"render error {e}")
    # 7 expect vs messages
    last_u = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    exp_calls = [x for x in meta["expect"] if "calls" in x]
    real_calls = [m["tool_calls"] for m in msgs[last_u:] if m["role"] == "assistant" and m.get("tool_calls")]
    if [x["calls"] for x in exp_calls] != real_calls:
        errs.append("expect != calls after last user")
    if is_eval:
        for key in ("context", "reference", "gold", "checks", "world", "id"):
            if key not in ex:
                errs.append(f"eval item lacks {key}")
    return errs


def as_eval_item(ex, i):
    msgs = ex["messages"]
    last_u = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    m = ex["meta"]
    gold = dict(m["gold"])
    gold["steps"] = m["expect"]
    return {"id": f"close_position-{i:03d}", "kind": m["kind"], "tools": True, "bio": m["bio"], "context": msgs[:last_u + 1],
            "reference": msgs[last_u + 1:], "expect": m["expect"], "world": m["world"], "checks": m["checks"], "gold": gold,
            "messages": msgs, "meta": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--eval-seed", type=int, default=9312)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--train-out", default=os.path.join(HERE, "..", "data3", "close_position.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(HERE, "..", "eval3", "close_position.jsonl"))
    a = ap.parse_args()
    if a.eval_seed == a.seed:
        raise SystemExit("eval seed must differ from train seed")
    n_neg = round(a.n * 80 / 350)
    train = build(a.seed, a.n, "train", n_neg)
    train_users = {norm(m["content"]) for ex in train for m in ex["messages"] if m["role"] == "user"}
    ev_neg = round(a.eval_n * 10 / 40)
    ev = build(a.eval_seed, a.eval_n, "eval", ev_neg, forbid_users=train_users)
    items = [as_eval_item(ex, i) for i, ex in enumerate(ev)]
    # ---- self-check
    bad = 0
    for name, rows, is_eval in (("train", train, False), ("eval", items, True)):
        keys = Counter(user_key(r) for r in rows)
        for k, v in keys.items():
            if v > 1:
                print(f"[{name}] duplicate user text: {k[:80]}")
                bad += 1
        for i, r in enumerate(rows):
            errs = self_check_record(r, is_eval)
            if errs:
                bad += 1
                print(f"[{name} #{i} {r['meta']['kind']}] " + " ; ".join(errs[:4]))
    eval_users = {norm(m["content"]) for ex in items for m in ex["messages"] if m["role"] == "user"}
    if eval_users & train_users:
        print("eval/train user text overlap:", sorted(eval_users & train_users)[:5])
        bad += 1
    neg_train = sum(1 for r in train if r["meta"]["kind"].startswith("neg_"))
    neg_eval = sum(1 for r in items if r["meta"]["kind"].startswith("neg_"))
    if len(train) != a.n or len(items) != a.eval_n:
        print("count mismatch", len(train), len(items))
        bad += 1
    if (neg_train, neg_eval) != (n_neg, ev_neg):
        print("negative counts", neg_train, neg_eval, "expected", n_neg, ev_neg)
        bad += 1
    sides = Counter(p["side"] for r in train for p in r["meta"]["world"]["hl_positions"])
    if min(sides.values()) < 50 or len(sides) < 2:
        print("positions not both sides", sides)
        bad += 1
    npos = Counter(len(r["meta"]["world"]["hl_positions"]) for r in train)
    print("positions per world:", dict(sorted(npos.items())), "sides:", dict(sides))
    if bad:
        print(f"close_position: self-check FAILED ({bad} problems); nothing written")
        raise SystemExit(1)
    for path, rows in ((a.train_out, train), (a.eval_out, items)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train kinds:", dict(sorted(Counter(r["meta"]["kind"] for r in train).items())))
    print("eval kinds:", dict(sorted(Counter(r["meta"]["kind"] for r in items).items())))
    print(f"close_position: {len(train)} train, {len(items)} eval, self-check ok")


if __name__ == "__main__":
    main()
