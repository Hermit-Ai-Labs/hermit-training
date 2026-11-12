#!/usr/bin/env python3
"""route_rounding: quotes and swaps that repeat the tool's route and its rounded amounts.

Fixes (NEXT_RUN.md section 2, PLAN.md 2.5): "Relay" instead of NEAR Intents, 24-decimal amount_out,
quote vs prepare, and the 12 broken swap_all eval items (the balance the question assumes is written into
the world snapshot and every record is replayed against the simulator).

  python3 route_rounding.py --seed 3 --n 500 --eval-seed 9305 --eval-n 60

Writes  ../data3/route_rounding.jsonl   (train, seed 3)
        ../eval3/route_rounding.jsonl   (held-out: other seed, held-out templates/route pairs/amounts,
                                         user texts disjoint from train, eval2 and eval)
Every assistant string is an f-string over tool results, calculator results or the user's own text.
Nothing here edits gen2: its helpers (hermit_fmt, sim) are imported; the v2 tool block is patched in
at runtime, in memory only.
"""
import argparse, json, os, random, re, sys, zlib
from collections import Counter
from decimal import Decimal, Context, ROUND_HALF_UP

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen2"))
import hermit_fmt as H      # gen2 helpers (calc, render, parse_calls, with_user)
import sim                  # gen2 simulator (World: prices, wallet; subclassed below)

SPEC = os.path.join(LAB, "spec")
V2_TOOLS = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
SYSTEM_V2 = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().strip("\n")
# runtime patch (memory only): render and validate against the 19-tool registry
H.TOOLS = V2_TOOLS
H.TOOL_NAMES = [t["function"]["name"] for t in V2_TOOLS]
H.TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in V2_TOOLS}
H.TOOLS_BLOCK = H.tools_block()

ROUTE = "NEAR Intents"

# ------------------------------------------------------------------ chains, tokens
DISPLAY = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "bsc": "BNB Chain", "polygon": "Polygon",
           "monad": "Monad", "solana": "Solana", "near": "NEAR", "bitcoin": "Bitcoin", "hyperliquid": "Hyperliquid"}
CHAIN_WORDS = {
    "ethereum": ["ethereum", "Ethereum", "mainnet", "eth mainnet", "ethereum mainnet"],
    "base": ["base", "Base", "base", "Base chain"],
    "arbitrum": ["arbitrum", "Arbitrum", "arb", "arbitrum one"],
    "bsc": ["bsc", "BSC", "BNB Chain", "bnb chain", "bnb", "binance smart chain"],
    "polygon": ["polygon", "Polygon", "polygon pos"],
    "monad": ["monad", "Monad"],
    "solana": ["solana", "Solana"],
    "near": ["near", "NEAR"],
    "bitcoin": ["bitcoin", "Bitcoin"],
    "hyperliquid": ["hyperliquid", "Hyperliquid", "HL"],
}
ALIAS_RE = {"ethereum": r"ethereum|mainnet|eth mainnet", "base": r"\bbase\b", "arbitrum": r"arbitrum|\barb\b",
            "bsc": r"\bbsc\b|bnb|binance", "polygon": r"polygon", "monad": r"monad", "solana": r"solana",
            "near": r"\bnear\b", "bitcoin": r"bitcoin|\bbtc\b", "hyperliquid": r"hyperliquid|\bhl\b"}
TOKEN_CHAINS = {
    "ETH": ["ethereum", "base", "arbitrum"],
    "USDC": ["ethereum", "base", "arbitrum", "bsc", "polygon", "monad", "solana", "near", "hyperliquid"],
    "USDT": ["ethereum", "arbitrum", "bsc", "solana"],
    "BNB": ["bsc"], "SOL": ["solana"], "NEAR": ["near"], "BTC": ["bitcoin"], "POL": ["polygon"], "MON": ["monad"],
    "LINK": ["ethereum"], "UNI": ["ethereum"], "AAVE": ["ethereum"], "ARB": ["arbitrum"], "PEPE": ["ethereum"],
}
NATIVE = {"BNB": "bsc", "SOL": "solana", "NEAR": "near", "BTC": "bitcoin", "MON": "monad", "POL": "polygon"}
DEST_ONLY_TOKENS = {"BTC", "PEPE"}
NO_SRC_CHAINS = {"bitcoin", "hyperliquid"}
STABLE = {"USDC", "USDT"}
TOKEN_W = {"USDC": 5, "ETH": 4, "USDT": 2, "SOL": 2, "BNB": 1.5, "NEAR": 1, "POL": .6, "MON": .6, "LINK": .4,
           "UNI": .3, "AAVE": .3, "ARB": .4, "BTC": 1.5, "PEPE": .5}
SRC_TOKENS = [t for t in TOKEN_CHAINS if t not in DEST_ONLY_TOKENS]
DST_TOKENS = list(TOKEN_CHAINS)
MULTI = ["ETH", "USDC", "USDT"]
AMOUNTS = {
    "USDC": [5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 120, 150, 200, 250, 300, 400, 500, 750, 1000, 1200, 1500,
             2000, 2500, 3000, 12.5, 45.5, 99.99, 7.25],
    "ETH": [0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 0.003, 0.0125, 0.04],
    "SOL": [0.5, 1, 2, 3, 5, 10, 12.5, 20, 50, 0.25], "BNB": [0.1, 0.25, 0.5, 1, 2, 5, 0.05, 3],
    "NEAR": [10, 25, 50, 100, 250, 500, 1000, 75], "POL": [50, 100, 250, 500, 1000, 300, 2000, 75],
    "MON": [100, 500, 1000, 5000, 250, 2500, 800, 10000], "LINK": [5, 10, 20, 50, 8, 15, 100, 2.5],
    "UNI": [5, 10, 25, 40, 100, 12, 8, 60], "AAVE": [0.1, 0.5, 1, 2, 0.25, 3, 5, 0.75],
    "ARB": [50, 100, 500, 250, 1000, 75, 2000, 30], "BTC": [0.001, 0.005, 0.01, 0.02, 0.05, 0.003, 0.0025, 0.1],
}
AMOUNTS["USDT"] = AMOUNTS["USDC"]


class Retry(Exception):
    pass


# ------------------------------------------------------------------ number helpers
def fmt_plain(x):
    d = Decimal(repr(float(x))) if not isinstance(x, (int, Decimal)) else Decimal(x)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def jn(x):
    """shortest JSON number: ints stay ints."""
    x = float(x)
    return int(x) if x == int(x) else x


def approx(x):
    """Units.approx: >= 1000 -> 2 dp, else 3 significant digits, trailing zeros trimmed (string)."""
    d = Decimal(repr(float(x)))
    if d >= 1000:
        d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        d = Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(d)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def valid_approx(s):
    d = Decimal(s)
    if d >= 1000:
        return d == d.quantize(Decimal("0.01"))
    return Context(prec=3, rounding=ROUND_HALF_UP).create_decimal(d) == d


def cents_ok(x):
    c = round(abs(x) * 100) % 100
    return c == 0 or c % 10 != 0


def norm_chain(c):
    c = str(c).strip().lower()
    return "bsc" if c == "bnb" else c


# ------------------------------------------------------------------ simulator (v2 shapes for the 3 tools used here)
class RRWorld(sim.World):
    """gen2 World with the app's real get_balances / get_quote / prepare_swap result shapes (spec meta file).
    Deterministic given the state, so a recorded conversation replays exactly."""
    STATE_KEYS = sim.World.STATE_KEYS + ("policy", "near_account")
    policy = None
    near_account = None

    def __init__(self, rng):
        super().__init__(rng)
        self.balances = []
        self.policy = None
        self.near_account = rng.choice(["alice", "kaito", "maria", "dev42", "lena", "rohan"]) + ".near"

    # -- tools
    def t_get_balances(self, chain=None):
        ch = None if chain is None else norm_chain(chain)
        rows = []
        for b in self.balances:
            if ch and b["chain"] != ch:
                continue
            rows.append({"amount": b["amount"], "chain": "bnb" if b["chain"] == "bsc" else b["chain"],
                         "symbol": b["symbol"], "usd": jn(b["usd"])})
        rows.sort(key=lambda r: (-r["usd"], r["symbol"], r["chain"]))
        out = {"holdings": rows}
        if any(r["chain"] == "near" for r in rows):
            out["near_account"] = self.near_account
        return out

    def _have(self, sym, chain):
        return sum(Decimal(b["amount"]) for b in self.balances if b["symbol"] == sym and b["chain"] == chain)

    def _route(self, from_symbol, from_chain, to_symbol, to_chain, amount, prepare):
        fs, ts = str(from_symbol).upper(), str(to_symbol).upper()
        fc, tc = norm_chain(from_chain), norm_chain(to_chain)
        amount = float(amount)
        if amount <= 0:
            return {"error": "Amount must be greater than zero."}
        if fc == "bitcoin":
            return {"error": "Bitcoin is supported as a destination only."}
        if fc == "hyperliquid":
            return {"error": "Hyperliquid is supported as a USDC destination only."}
        if tc == "hyperliquid" and ts != "USDC":
            return {"error": "Hyperliquid only accepts USDC as a destination."}
        if (fc not in TOKEN_CHAINS.get(fs, [])) or (tc not in TOKEN_CHAINS.get(ts, [])) or (fs, fc) == (ts, tc):
            return {"error": "No route found for this pair."}
        pf, pt = self.prices[fs], self.prices[ts]
        raw_usd = amount * pf
        if raw_usd < 1:
            return {"error": "Amount is below the minimum for this route (about $1)."}
        r = random.Random(zlib.crc32(json.dumps([fs, fc, ts, tc, amount, pf, pt]).encode()))
        usd_in = round(raw_usd, 2)
        bps = r.choice([10, 15, 20, 25, 30, 35, 40, 50])
        cost = max(0.01, round(usd_in * (bps + r.randint(5, 40)) / 10000, 2))
        while not (cents_ok(cost) and cents_ok(round(usd_in - cost, 2))):
            cost = round(cost + 0.01, 2)
        usd_out = round(usd_in - cost, 2)
        out = approx(usd_out / pt)
        if tc == "bitcoin":
            secs = r.choice([60, 75, 90, 120, 150, 180])
        elif tc == "hyperliquid":
            secs = r.randint(10, 30)
        else:
            secs = r.randint(6, 55)
        res = {"amount_in_usd": jn(usd_in), "amount_out": out, "amount_out_usd": jn(usd_out), "fee_bps": bps,
               "from": f"{fmt_plain(amount)} {fs} on {DISPLAY[fc]}", "route": ROUTE, "seconds": secs,
               "signature_verified": True, "to": f"about {out} {ts} on {DISPLAY[tc]}", "total_cost_usd": jn(cost)}
        if not prepare:
            return res
        have = self._have(fs, fc)
        if have < Decimal(repr(amount)):
            return {"error": f"Not enough {fs} on {DISPLAY[fc]}: you have {fmt_plain(have)}."}
        pol = self.policy or {}
        if pol.get("block"):
            return {"note": "Nothing was prepared. Tell the user why.", "reasons": list(pol["block"]), "status": "blocked"}
        res["note"] = ("Nothing has moved. The Swap review is open with this NEAR Intents quote; "
                       "the user approves it with Face ID or cancels.")
        res["policy"] = "warn" if pol.get("warn") else "allow"
        res["status"] = "prepared_for_review"
        if pol.get("warn"):
            res["warnings"] = list(pol["warn"])
        return res

    def t_get_quote(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        return self._route(from_symbol, from_chain, to_symbol, to_chain, amount, False)

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        return self._route(from_symbol, from_chain, to_symbol, to_chain, amount, True)


def dump(r):
    return json.dumps(r, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ world building
def dp_for(sym):
    return 2 if sym in STABLE or sym in ("SOL", "NEAR", "POL", "MON", "LINK", "UNI", "ARB") else 4


def rand_balance(rng, w, sym, lo_usd=5, hi_usd=6000):
    p = w.prices[sym]
    usd = rng.choice([rng.uniform(lo_usd, 300), rng.uniform(300, hi_usd)])
    amt = round(usd / p, dp_for(sym))
    return amt if amt > 0 else None


def put_holding(w, sym, chain, amt):
    amt = float(amt)
    s = fmt_plain(amt)
    for b in w.balances:
        if b["symbol"] == sym and b["chain"] == chain:
            b["amount"], b["usd"] = s, round(amt * w.prices[sym], 2)
            return
    w.balances.append({"symbol": sym, "chain": chain, "amount": s, "usd": round(amt * w.prices[sym], 2)})


def src_pairs():
    return [(s, c) for s in SRC_TOKENS for c in TOKEN_CHAINS[s] if c not in NO_SRC_CHAINS]


def make_world(rng, holds=(), extras=None, avoid_syms=(), w=None):
    w = w or RRWorld(rng)
    for sym, ch, amt in holds:
        put_holding(w, sym, ch, amt)
    k = rng.choice([0, 1, 1, 2, 2, 3]) if extras is None else extras
    pairs = [p for p in src_pairs() if p[0] not in avoid_syms]
    tries = 0
    while k > 0 and tries < 60:
        tries += 1
        sym, ch = rng.choice(pairs)
        if any(b["symbol"] == sym and b["chain"] == ch for b in w.balances):
            continue
        amt = rand_balance(rng, w, sym)
        if amt:
            put_holding(w, sym, ch, amt)
            k -= 1
    return w


# ------------------------------------------------------------------ conversation builder
class Conv:
    def __init__(self, rng, world):
        self.rng, self.w = rng, world
        self.bio = "Face ID" if rng.random() < 0.70 else "Touch ID"
        self.world0 = world.state()
        self.world_last = world.state()
        sysm = H.with_user(SYSTEM_V2.replace("{BIO}", "Face ID"), rng)
        self.msgs = [{"role": "system", "content": sysm}]
        self.expect = []

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        self.expect = []
        self.world_last = self.w.state()

    def call(self, *calls):
        tcs = [{"name": n, "arguments": a} for n, a in calls]
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
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
        msgs = self.msgs
        if self.bio != "Face ID":
            msgs = [dict(m, content=m["content"].replace("Face ID", self.bio)) if m.get("content") else m for m in msgs]
        meta = {"source": "route_rounding", "generator": "route_rounding", "kind": kind, "bio": self.bio,
                "world": self.world_last, "world_final": self.w.state(), "expect": self.expect, "spec": spec,
                "lang": "en"}
        return {"messages": msgs, "tools": True, "meta": meta}


# ------------------------------------------------------------------ user phrasing
def split_pick(rng, pool, ev, ok=lambda flags: True, mod=5):
    """items whose index % mod == 0 are held out for eval; flags ride after '|'."""
    cand = []
    for i, s in enumerate(pool):
        t, _, fl = s.partition("|")
        if ((i % mod == 0) == ev) and ok(fl):
            cand.append(t)
    if not cand:
        raise Retry()
    return rng.choice(cand)


def tk(rng, sym):
    r = rng.random()
    return sym if r < 0.55 else (sym.lower() if r < 0.95 else sym.capitalize())


def cw(rng, chain):
    return rng.choice(CHAIN_WORDS[chain])


WORDS = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 100: "a hundred"}


def amt_text(rng, v, sym):
    r = rng.random()
    s = fmt_plain(v)
    fv = float(v)
    if fv.is_integer() and int(fv) in WORDS and r < 0.07:
        return WORDS[int(fv)]
    if fv >= 1000 and fv % 100 == 0 and r < 0.20:
        return f"{fv / 1000:g}k"
    if fv >= 1000 and fv.is_integer() and r < 0.45:
        return f"{int(fv):,}"
    if sym in STABLE and 0.5 <= r < 0.65:
        return "$" + s
    if 0 < fv < 1 and r < 0.20:
        return s[1:]
    return s


def slots(rng, fs, fc, ts, tc, a):
    amt = amt_text(rng, a, fs)
    f, t = tk(rng, fs), tk(rng, ts)
    fcw, tcw = cw(rng, fc), cw(rng, tc)
    if NATIVE.get(fs) == fc and rng.random() < 0.65:
        src = f"{amt} {f}"
    else:
        src = rng.choice([f"{amt} {f} on {fcw}", f"{amt} {f} from {fcw}", f"{amt} {f} ({fcw})", f"{fcw} {amt} {f}",
                          f"{amt} of my {f} on {fcw}", f"{amt} {f} on {fcw}", f"{amt} {f} sitting on {fcw}"])
    if (ts, tc) == ("USDC", "hyperliquid"):
        dst = rng.choice([f"{t} on {tcw}", f"{tcw} {t}", f"my {tcw} account", f"{tcw}", f"{t} on {tcw}"])
    elif NATIVE.get(ts) == tc and rng.random() < 0.7:
        dst = t
    else:
        dst = rng.choice([f"{t} on {tcw}", f"{tcw} {t}", f"{t} ({tcw})", f"{t} over on {tcw}", f"{t} on {tcw}",
                          f"{t} on {tcw}"])
    return dict(amt=amt, fs=f, fc=fcw, ts=t, tc=tcw, src=src, dst=dst)


def typo_word(rng, w):
    ops = rng.choice(["swap", "drop", "dbl"])
    i = rng.randrange(1, len(w) - 1)
    if ops == "swap" and w[i] != w[i + 1]:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if ops == "drop" and w[i] in "aeiou":
        return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


def render_user(rng, tmpl, S, typo=0.15):
    parts = re.split(r"(\{\w+\})", tmpl)
    lit = [i for i, p in enumerate(parts) if not p.startswith("{") and p]
    r = rng.random()
    if r < 0.40:
        for i in lit:
            parts[i] = parts[i].lower()
    elif r < 0.48 and lit and parts[lit[0]] == parts[0]:
        parts[0] = parts[0][:1].upper() + parts[0][1:]
    if rng.random() < typo:
        words = [(i, m) for i in lit for m in re.finditer(r"\b[a-z]{4,}\b", parts[i])]
        if words:
            i, m = rng.choice(words)
            parts[i] = parts[i][:m.start()] + typo_word(rng, m.group()) + parts[i][m.end():]
    out = "".join(S[p[1:-1]] if p.startswith("{") else p for p in parts)
    if rng.random() < 0.06:
        out += rng.choice([" pls", " plz", " rn"])
    if rng.random() < 0.2:
        out = out.rstrip("?.")
    if rng.random() < 0.02:
        out = out.upper()
    return re.sub(r"\s+", " ", out).strip()


QUOTE_X = [
    "how much {dst} for {src}?", "quote me {src} to {dst}", "what would I get for {src} into {dst}?",
    "{src} -> {dst} quote?", "how much would {src} turn into as {dst}", "price check on {src} to {dst}",
    "what's the rate if I swap {src} for {dst}", "if I swapped {src} to {dst}, what do I end up with?",
    "can you quote {src} for {dst}", "just checking, how much {dst} would {src} buy me? don't do anything yet",
    "get me a quote: {src} to {dst}", "what do I get converting {src} to {dst}", "{src} to {dst}, how much comes out",
    "estimate {src} into {dst}", "curious what {src} would be worth in {dst}", "how many {dst} would I get for {src}",
    "I want to see numbers before I swap. {src} to {dst}", "no swap yet, just show me what {src} gets me in {dst}",
    "what's {src} in {dst} right now", "ok so {src} into {dst}, what's the damage",
    "how much do I come away with if I trade {src} for {dst}", "rate check: {src} for {dst}",
    "yo what would {src} fetch in {dst}", "give me the price to move {src} into {dst}", "need a quote on {src} to {dst}",
    "tell me how much {dst} I'd get from {src}, don't prepare anything", "{dst} for {src}, what's the quote?",
    "would {src} get me a decent amount of {dst}? how much exactly", "show me a quote for {src} going to {dst}",
    "how much will I receive in {dst} if I sell {src}", "what are the costs and the amount if I go {src} to {dst}? quote only",
    "can I see what {src} converts to in {dst}?",
]
QUOTE_B = [
    "what would I get bridging {src} to {tc}?", "quote bridging {src} to {tc}", "how much arrives on {tc} if I bridge {src}?",
    "bridge quote: {src} -> {tc}", "what would it cost to move {src} over to {tc}? just a quote",
    "how much would I receive on {tc} for {src}", "check the numbers on bridging {src} to {tc}",
    "get me a quote to bridge {src} to {tc}", "if I bridge {src} to {tc} what do I end up with?",
    "{src} to {tc}, how much lands", "price for moving {src} to {tc}, don't prepare it",
    "how much {fs} on {tc} do I get for {src}", "what's the damage bridging {src} to {tc}",
    "estimate the bridge from {fc} to {tc} for {amt} {fs}", "can you quote {amt} {fs} from {fc} to {tc}?",
    "how much would I lose bridging {amt} {fs} from {fc} to {tc}", "what comes out on {tc} if I send {src} across?",
    "bridging {src} to {tc}: what's the quote", "tell me what I'd get on {tc} for {src}, no action yet",
    "quote {amt} {fs} {fc} to {tc}",
]
QUOTE_S = [
    "how much {ts} for {amt} {fs} on {fc}?", "quote {amt} {fs} to {ts} on {fc}", "what would {amt} {fs} get me in {ts} on {fc}",
    "on {fc}, how much {ts} do I get for {amt} {fs}?", "{fc} swap quote: {amt} {fs} to {ts}",
    "price check {amt} {fs} into {ts} ({fc})", "how much {ts} would I get if I swap {amt} {fs} on {fc}",
    "can you quote me {amt} {fs} -> {ts} on {fc}", "what's {amt} {fs} worth in {ts} on {fc} right now? quote only",
    "just a quote please: {amt} {fs} for {ts}, {fc}", "estimate {amt} {fs} into {ts} on {fc}",
    "I'm thinking of swapping {amt} {fs} for {ts} on {fc}, how much would I get?",
]
SWAP_X = [
    "swap {src} to {dst}", "swap {src} for {dst}", "convert {src} into {dst}", "trade {src} for {dst}",
    "exchange {src} for {dst}", "can you swap {src} to {dst}?", "I want to swap {src} for {dst}", "turn {src} into {dst}",
    "flip {src} into {dst}", "change {src} to {dst}", "swap {src} over to {dst}", "please convert {src} to {dst}",
    "set up a swap: {src} to {dst}", "prepare a swap of {src} for {dst}", "I'd like {dst}, paying with {src}",
    "get me {dst} using {src}", "buy {dst} with {src}|buy", "sell {src} for {dst}", "{src} to {dst} pls",
    "move {src} into {dst}", "rebalance: {src} into {dst}", "need {dst}, swap {src}", "exchange my {src} for {dst}",
    "do a swap {src} -> {dst}", "swap {src} to {dst} for me", "i wanna swap {src} to {dst}", "ok swap {src} into {dst}",
    "put {src} into {dst}", "{src} into {dst} please, open the review", "let's go {src} to {dst}",
    "swap it: {src} to {dst}", "grab me {dst}, use {src}|buy",
]
SWAP_S = [
    "swap {amt} {fs} for {ts} on {fc}", "{amt} {fs} to {ts} on {fc}", "convert {amt} {fs} to {ts} ({fc})",
    "on {fc}, swap {amt} {fs} into {ts}", "trade {amt} {fs} for {ts} on {fc}", "swap {amt} {fs} to {ts}, {fc}",
    "I want {ts} on {fc}, swap {amt} {fs}", "{fc}: {amt} {fs} to {ts}", "turn {amt} {fs} into {ts} on {fc}",
    "exchange {amt} {fs} for {ts} on {fc}", "swap {amt} {fs} to {ts} on {fc} please", "can you change {amt} {fs} to {ts} on {fc}?",
    "buy {ts} on {fc} with {amt} {fs}|buy", "flip {amt} {fs} to {ts} on {fc}", "convert {amt} {fs} to {ts} on {fc}, thanks",
    "get me {ts} on {fc} for {amt} {fs}|buy", "just swap {amt} {fs} for {ts} on {fc}", "need to swap {amt} {fs} to {ts} on {fc}",
    "sell {amt} {fs} for {ts} on {fc}", "{amt} {fs} -> {ts} on {fc}",
]
BRIDGE = [
    "bridge {src} to {tc}", "bridge {amt} {fs} from {fc} to {tc}", "move {src} to {tc}", "take {src} over to {tc}",
    "get {amt} {fs} from {fc} to {tc}", "cross-chain {src} to {tc}", "bridge over {src} to {tc}",
    "I need my {fs} on {tc}, move {amt} from {fc}", "can you bridge {src} to {tc}?", "{src} -> {tc}, bridge it",
    "bridge {amt} {fs} {fc} to {tc}", "please move {amt} {fs} from {fc} to {tc}", "I want {amt} {fs} on {tc} instead of {fc}",
    "shift {src} to {tc}", "bridge {amt} {fs} to {tc} from {fc}", "set up a bridge: {src} to {tc}",
    "prepare a bridge of {src} to {tc}", "bridge {amt} {fs} from {fc} over to {tc}", "{fc} to {tc}: bridge {amt} {fs}",
    "relocate {amt} {fs} from {fc} to {tc}", "bridge {amt} {fs}, {fc} to {tc}", "get my {amt} {fs} off {fc} and onto {tc}",
    "{amt} {fs} {fc} -> {tc}", "bridge it: {src} to {tc}", "let's move {src} to {tc}", "bridge some {fs}: {amt} from {fc} to {tc}",
]
QUOTE_NOSRC = [
    "how much {dst} for {amt} {fs}?", "quote {amt} {fs} to {dst}", "what would {amt} {fs} get me in {dst}",
    "{amt} {fs} to {dst}, what's the quote", "how much {dst} would I get for {amt} {fs}", "price check {amt} {fs} -> {dst}",
    "can you quote {amt} {fs} into {dst}", "what's {amt} {fs} in {dst}? just a quote", "estimate {amt} {fs} to {dst}",
    "show me what {amt} {fs} turns into as {dst}",
]
QUOTE_NOSRC_B = [
    "quote bridging {amt} {fs} to {tc}", "how much arrives on {tc} if I bridge {amt} {fs}", "what would I get on {tc} for {amt} {fs}",
    "bridge quote for {amt} {fs} to {tc}", "how much {fs} lands on {tc} if I move {amt}", "price check: {amt} {fs} over to {tc}",
]
SWAP_NOSRC = [
    "swap {amt} {fs} to {dst}", "swap {amt} {fs} for {dst}", "convert {amt} {fs} into {dst}", "trade {amt} {fs} for {dst}",
    "turn {amt} {fs} into {dst}", "can you swap {amt} {fs} to {dst}?", "I want {dst} for {amt} {fs}", "exchange {amt} {fs} for {dst}",
    "get me {dst}, use {amt} {fs}", "{amt} {fs} to {dst} pls", "flip {amt} {fs} into {dst}", "swap my {amt} {fs} into {dst}",
]
BRIDGE_NOSRC = [
    "bridge {amt} {fs} to {tc}", "move {amt} {fs} to {tc}", "I need {amt} {fs} on {tc}", "take {amt} {fs} over to {tc}",
    "bridge {amt} {fs} over to {tc}, please", "get {amt} {fs} onto {tc}", "bridge {amt} {fs} to {tc} for me",
    "shift {amt} {fs} to {tc}", "can you bridge {amt} {fs} to {tc}?",
]
QUOTE_NODST = [
    "how much {ts} for {src}?", "quote {src} to {ts}", "what would {src} get me in {ts}", "{src} into {ts}, what's the quote",
    "price check {src} -> {ts}", "can you quote {src} for {ts}", "how much {ts} would I get for {src}",
    "what's {src} in {ts} right now? just a quote",
]
SWAP_NODST = [
    "swap {src} to {ts}", "swap {src} for {ts}", "convert {src} into {ts}", "trade {src} for {ts}", "turn {src} into {ts}",
    "I want {ts}, swap {src}", "exchange {src} for {ts}", "{src} to {ts} pls", "swap {src} into {ts} for me", "flip {src} to {ts}",
]
BRIDGE_NODST = [
    "bridge {src}", "bridge {amt} {fs} from {fc}", "bridge {src} over", "I want to bridge {src}", "move {src} to another chain",
    "bridge my {amt} {fs} off {fc}", "can you bridge {src}?",
]
QUOTE_NODST_B = [
    "what would I get bridging {src}?", "quote bridging {src}", "how much would arrive if I bridge {src}?", "bridge quote for {src}",
    "price check on bridging {src}", "what does bridging {src} cost? just a quote",
]
ALLX = [
    "swap all my {fs} on {fc} to {dst}", "convert my whole {fs} balance on {fc} into {dst}",
    "move everything I have in {fs} on {fc} to {dst}", "put all of my {fs} ({fc}) into {dst}", "all my {fs} on {fc} to {dst}",
    "swap my entire {fs} balance on {fc} for {dst}", "trade all {fs} I have on {fc} for {dst}", "sell all my {fs} on {fc} into {dst}",
    "max swap {fs} on {fc} to {dst}", "swap the full {fs} balance ({fc}) to {dst}",
]
ALLB = [
    "bridge all my {fs} from {fc} to {tc}", "all my {fs} on {fc} to {tc}", "move everything in {fs} on {fc} over to {tc}",
    "bridge my whole {fs} balance on {fc} to {tc}", "take all my {fs} from {fc} to {tc}", "move my entire {fs} on {fc} to {tc}",
    "send all my {fs} on {fc} across to {tc}", "bridge the full {fs} balance on {fc} to {tc}",
]
HALFX = [
    "swap half of my {fs} on {fc} to {dst}", "move half my {fs} on {fc} into {dst}", "50% of my {fs} on {fc} to {dst}",
    "convert half of what I have in {fs} on {fc} to {dst}", "swap half my {fs} ({fc}) for {dst}",
    "trade half of my {fs} on {fc} for {dst}", "half of my {fs} on {fc} into {dst}",
]
HALFB = [
    "bridge half of my {fs} from {fc} to {tc}", "move half my {fs} on {fc} over to {tc}", "half of my {fs} on {fc} to {tc}",
    "bridge 50% of my {fs} on {fc} to {tc}", "take half of my {fs} from {fc} to {tc}", "move half of my {fs} ({fc}) to {tc}",
]
QUOTE_ALL = [
    "how much {dst} would I get for all my {fs} on {fc}?", "quote my whole {fs} balance on {fc} into {dst}",
    "what would all my {fs} on {fc} turn into as {dst}", "if I swap everything in {fs} on {fc} to {dst}, what do I get",
    "price check: all my {fs} on {fc} to {dst}", "how much {dst} for my entire {fs} balance on {fc}? quote only",
]
QUOTE_ALLB = [
    "how much arrives on {tc} if I bridge all my {fs} from {fc}?", "quote bridging my whole {fs} balance on {fc} to {tc}",
    "what would I get on {tc} bridging all my {fs} from {fc}", "price check on moving all my {fs} on {fc} to {tc}",
    "quote: everything I have in {fs} on {fc} over to {tc}", "how much would I get on {tc} for my full {fs} balance on {fc}? just a quote",
]
WHICH = [
    "which bridge is that?|open", "is that Relay?|relay", "is it relay?|relay", "what route is that?|open",
    "who's providing this swap?|open", "which provider is that|open", "is it NEAR Intents?|near", "so is this going through relay?|relay",
    "what's it routed through|open", "which protocol is that one|open", "what service is doing this swap|open",
    "so that's near intents?|near", "which route did you get|open", "wait is this relay or something else|relay",
    "what bridge does this use?|open", "this is relay right?|relay", "is this using near intents|near",
    "what's the route here|open", "what does this go through?|open", "who fills this|open", "NEAR Intents right?|near",
    "which bridge are you using for this|open", "so is it relay doing this|relay", "which dex or bridge is this|open",
    "is it Relay or Across?|relay", "what's behind this route|open", "tell me the provider on this|open", "is that through near intents?|near",
    "relay?|relay", "which one is routing it|open",
]
RATE = [
    "so what's that per {t}?", "what rate is that", "what price am I getting for {t}", "what's the effective {t} price on that",
    "ok and what's that per {t} all in", "what does that work out to per {t}", "per {t}, what rate is that?",
    "what's my price per {t} there", "how much per {t} is that", "give me the per {t} price on that quote",
    "so how much is each {t} costing me", "and the rate per {t}?",
]
DO_IT = [
    "do it", "ok do it", "yes, swap it", "go ahead", "prepare it", "ok set it up", "yep", "yes please", "sure, go for it",
    "let's do it", "ok prepare that", "looks good, do it", "yeah do that one", "go on then", "swap it", "bridge it", "yes", "ok",
    "alright, prepare it", "sounds fine, go", "make it happen", "ok go", "set it up", "okay open the review", "yes open the review",
    "fine, do it", "perfect, do it", "do that", "yes do it", "ok let's go", "nice, do it", "go for it", "sure", "yea do it",
]
SRC_ANS = ["{c}", "on {c}", "it's on {c}", "the one on {c}", "i have it on {c}", "use {c}", "from {c}", "my {c} one", "the {c} one", "{c} is the one", "take it from {c}", "{c} side", "the {c} balance"]
DST_ANS = ["{c}", "to {c}", "on {c}", "send it to {c}", "over to {c}", "the {c} one", "{c} chain",
           "put it on {c}", "land it on {c}", "I want it on {c}", "make it {c}", "{c} is fine", "to {c} then", "{c} side", "{c} works"]
TAILS = ["", "", "", "", " pls", " please", "!", " now", " thanks", ", thanks", " :)", " then", " rn", " yeah", ".", " ok"]


def short_turn(rng, pool, ev, **kw):
    t = split_pick(rng, pool, ev).format(**kw)
    t = t + rng.choice(TAILS)
    r = rng.random()
    if r < 0.3:
        t = t.lower()
    elif r < 0.4:
        t = t[:1].upper() + t[1:]
    if rng.random() < 0.1:
        t = t.rstrip("!.")
    return t


# ------------------------------------------------------------------ route / amount picking
def wchoice(rng, items):
    return rng.choices(items, [TOKEN_W[i] for i in items])[0]


def ev_pair(route):
    return zlib.crc32("|".join(route).encode()) % 4 == 0


def route_kind(fs, fc, ts, tc):
    return "bridge" if fs == ts else ("same" if fc == tc else "swap")


def pick_src(rng, exclude=()):
    s = wchoice(rng, [t for t in SRC_TOKENS if t not in exclude])
    return s, rng.choice([c for c in TOKEN_CHAINS[s] if c not in NO_SRC_CHAINS])


def pick_route(rng, ev, kind="swap", mode="any"):
    for _ in range(400):
        if mode == "btc":
            fs, fc = pick_src(rng, exclude=("BTC",)); ts, tc = "BTC", "bitcoin"
        elif mode == "hl":
            fs, fc = pick_src(rng); ts, tc = "USDC", "hyperliquid"
        elif mode == "bsc":
            if rng.random() < 0.5:
                fs, fc = rng.choice([("BNB", "bsc"), ("USDT", "bsc"), ("USDC", "bsc")])
                ts = wchoice(rng, [t for t in DST_TOKENS if t != fs])
                tc = rng.choice(TOKEN_CHAINS[ts])
            else:
                fs, fc = pick_src(rng)
                ts = rng.choice(["BNB", "USDT", "USDC"]); tc = "bsc"
                if fc == "bsc":
                    continue
        elif kind == "bridge":
            fs = wchoice(rng, MULTI)
            fc = rng.choice([c for c in TOKEN_CHAINS[fs] if c not in NO_SRC_CHAINS])
            ts = fs
            tc = rng.choice([c for c in TOKEN_CHAINS[fs] if c != fc])
        elif kind == "same":
            fs, fc = pick_src(rng)
            ts = wchoice(rng, [t for t in DST_TOKENS if t != fs and fc in TOKEN_CHAINS[t]] or ["USDC"])
            tc = fc
        else:
            fs, fc = pick_src(rng)
            ts = wchoice(rng, [t for t in DST_TOKENS if t != fs])
            tc = rng.choice(TOKEN_CHAINS[ts])
        r = (fs, fc, ts, tc)
        if (fs, fc) == (ts, tc) or fc not in TOKEN_CHAINS[fs] or tc not in TOKEN_CHAINS[ts]:
            continue
        if kind == "same" and tc != fc:
            continue
        if kind == "swap" and (fs == ts or fc == tc) and mode == "any":
            continue
        if ev_pair(r) != ev:
            continue
        return r
    raise Retry()


def pick_amount(rng, sym, ev):
    pool = AMOUNTS[sym]
    cand = [v for i, v in enumerate(pool) if (i % 4 == 0) == ev]
    return rng.choice(cand)


# ------------------------------------------------------------------ answers
def usd(x):
    return fmt_plain(x)


def noun(r):
    return "bridge" if r.get("_bridge") else "swap"


def quote_answer(rng, r):
    f, t, rt = r["from"], r["to"], r["route"]
    cost, sec, out = usd(r["total_cost_usd"]), r["seconds"], usd(r["amount_out_usd"])
    body = rng.choice([
        f"{f} would get you {t} through {rt}, about ${cost} in total costs and roughly {sec} seconds.",
        f"Through {rt}, {f} gets you {t}. Total costs are about ${cost} and it takes roughly {sec} seconds.",
        f"The quote is {t} for {f}, through {rt}. That's about ${out} out, with about ${cost} in total costs and roughly {sec} seconds.",
        f"{rt} quotes {f} at {t}. All in it costs about ${cost}, and it takes roughly {sec} seconds.",
        f"{f} comes to {t} via {rt}, about ${cost} in total costs, roughly {sec} seconds.",
        f"You'd get {t} for {f}, routed through {rt}. Total costs about ${cost}, roughly {sec} seconds.",
        f"I got a quote: {f} for {t}, through {rt}. It's about ${cost} in total costs and roughly {sec} seconds.",
        f"{f} into {t} through {rt}. Costs come to about ${cost} in total, and it should take roughly {sec} seconds.",
    ])
    only = rng.choice(["This is only a quote.", "Nothing is prepared yet.", "It's just a quote for now.", "That's only a quote."])
    ask = rng.choice(["Want me to prepare it?", "Should I prepare it?", "Say the word and I'll prepare it.",
                      "Want me to set it up for review?", "Shall I prepare it?"])
    return f"{body} {only} {ask}"


def swap_answer(rng, r, bridge):
    f, t, rt = r["from"], r["to"], r["route"]
    cost = usd(r["total_cost_usd"])
    n = "bridge" if bridge else "swap"
    return rng.choice([
        f"I've prepared the {n}: {f} for {t}, through {rt}. The review is open. Approve with Face ID or cancel. Nothing has moved yet.",
        f"The {n} is prepared: {f} for {t}, through {rt}. Review it on screen and approve with Face ID or cancel. Nothing has moved yet.",
        f"I've set up the {n} for review: {f} for {t}, through {rt}, with about ${cost} in total costs. Approve with Face ID or cancel, nothing has moved yet.",
        f"Review is open for {f} to {t}, through {rt}. Approve with Face ID when you're ready, or cancel. Nothing has moved yet.",
        f"I've prepared it: {f} for {t} through {rt}. It's waiting for your Face ID, and nothing has moved yet.",
        f"Prepared: {f} for {t}, through {rt}. Total costs are about ${cost}. Approve with Face ID or cancel, nothing has moved.",
        f"{n.capitalize()} prepared through {rt}: {f} for {t}. The review screen is open, and nothing has moved until you approve with Face ID.",
        f"I opened the review for {f} to {t}, through {rt}. Approve it with Face ID or cancel; nothing has moved yet.",
    ])


def which_answer(rng, r, qk, prepared):
    rt = r["route"]
    what = "swap" if prepared else "quote"
    if qk == "relay":
        return rng.choice([f"No, this one goes through {rt}, as the result shows.", f"Not that one. The route here is {rt}.",
                           f"It's {rt}. That's the route the {what} came back with.",
                           f"This {what} is routed through {rt}, according to the result."])
    if qk == "near":
        return rng.choice([f"Yes, the route is {rt}.", f"Yes, this {what} goes through {rt}.",
                           f"That's right, the result says {rt}.", f"Yes. {rt} is the route on this {what}."])
    return rng.choice([f"It goes through {rt}, as the result says.", f"The route is {rt}.",
                       f"{rt}. That's the route this {what} came back with.", f"This {what} is routed through {rt}.",
                       f"The result names {rt} as the route."])


def min_of(err):
    m = re.search(r"about \$(\d+(?:\.\d+)?)", err)
    return m.group(1) if m else None


def error_answer(rng, err, intent, a, fs, fc, ts, tc):
    verb = "quote" if intent == "quote" else "prepare"
    m = min_of(err)
    if m:
        return rng.choice([
            f"That's below the minimum for this route, about ${m}, so I can't {verb} it. Try a larger amount.",
            f"The amount is under the minimum of about ${m} for this route, so there's nothing to {verb}.",
            f"This route needs at least about ${m}, and that amount is below it, so I couldn't {verb} it.",
            f"I can't {verb} that: it's below the route's minimum of about ${m}."])
    if "Bitcoin is supported as a destination only" in err:
        return rng.choice([f"Bitcoin only works as a destination here, so I can't {verb} a swap from it.",
                           f"I can't {verb} that because Bitcoin can only be the destination, not the source.",
                           f"Bitcoin is destination only for these swaps, so there's nothing I can {verb} from BTC."])
    if "Hyperliquid only accepts USDC" in err or "Hyperliquid is supported" in err:
        return rng.choice([f"Hyperliquid only takes USDC as a destination, so I can't {verb} {ts} into it.",
                           f"I couldn't {verb} that: Hyperliquid accepts USDC only.",
                           f"Only USDC can go to Hyperliquid, so I can't {verb} {ts} there."])
    return rng.choice([f"There's no route for that pair right now, so I can't {verb} it.",
                       f"I couldn't {verb} that: no route was found for {fs} on {DISPLAY[fc]} to {ts} on {DISPLAY[tc]}.",
                       f"No route is available for that pair, so I have no number to give you.",
                       f"The tool found no route for that pair, so I can't {verb} it."])


def ask_from(rng, sym, chains):
    names = [DISPLAY[c] if c != "bnb" else "BNB Chain" for c in chains]
    ch = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    return rng.choice([f"Which chain should I use for your {sym}: {ch}?", f"You hold {sym} on {ch}, which one do you want to use?",
                       f"Your {sym} is on {ch}. Which one?", f"Which one should I take the {sym} from, {ch}?",
                       f"I see {sym} on {ch}. Which chain should it come from?", f"Which chain, {ch}? That's where your {sym} is."])


def ask_to(rng, ts, bridge):
    if bridge:
        return rng.choice(["Which chain should I bridge it to?", "Which chain should it go to?", "Bridge to which chain?",
                           "Which chain do you want to move it to?", "Which chain should it land on?", "Which network should it go to?"])
    return rng.choice([f"Which chain should the {ts} go to?", f"Which chain do you want the {ts} on?", "Which network should it end up on?",
                       f"Which chain for the {ts}?", f"On which chain do you want the {ts}?", f"Where should the {ts} land, which chain?"])


# ------------------------------------------------------------------ shared scenario pieces
def user_text(rng, ev, intent, route, a, **over):
    fs, fc, ts, tc = route
    kind = route_kind(fs, fc, ts, tc)
    S = slots(rng, fs, fc, ts, tc, a)
    S.update(over)
    pool = {("quote", "swap"): QUOTE_X, ("quote", "bridge"): QUOTE_B, ("quote", "same"): QUOTE_S,
            ("swap", "swap"): SWAP_X, ("swap", "bridge"): BRIDGE, ("swap", "same"): SWAP_S}[(intent, kind)]
    ok = lambda fl: (fl != "buy") or (intent == "swap" and ts in ("ETH", "SOL", "BNB", "NEAR") and fs in STABLE | {"ETH", "SOL", "BNB"})
    return render_user(rng, split_pick(rng, pool, ev, ok), S)


def args_of(route, a):
    fs, fc, ts, tc = route
    return {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": jn(a)}


def finish(c, kind, intent, route, a, scenario, **extra):
    spec = {"scenario": scenario, "intent": intent, "route": list(route), "amount": jn(a) if a is not None else None}
    spec.update(extra)
    return c.ex(kind, spec)


def do_quote_turn(rng, c, ev, route, a):
    """user asks for a quote, tool, answer. Returns result."""
    c.user(user_text(rng, ev, "quote", route, a))
    r = c.call(("get_quote", args_of(route, a)))
    assert "error" not in r, r
    c.say(quote_answer(rng, r))
    return r


def do_swap_turn(rng, c, ev, route, a):
    c.user(user_text(rng, ev, "swap", route, a))
    r = c.call(("prepare_swap", args_of(route, a)))
    assert r.get("status") == "prepared_for_review", r
    fs, fc, ts, tc = route
    c.say(swap_answer(rng, r, route_kind(*route) == "bridge"))
    return r


def funded_world(rng, route, a, mult=None, w=None):
    fs, fc, ts, tc = route
    m = mult or rng.uniform(1.1, 4)
    amt = float(a) * m
    amt = round(amt, dp_for(fs) + 2)
    return make_world(rng, [(fs, fc, amt)], w=w)


# ------------------------------------------------------------------ scenarios
def sc_quote(rng, ev):
    kind = rng.choices(["swap", "bridge", "same"], [.5, .3, .2])[0]
    route = pick_route(rng, ev, kind)
    a = pick_amount(rng, route[0], ev)
    c = Conv(rng, make_world(rng))
    do_quote_turn(rng, c, ev, route, a)
    return finish(c, "quote", "quote", route, a, "quote")


def sc_swap(rng, ev):
    kind = rng.choices(["swap", "bridge", "same"], [.4, .4, .2])[0]
    route = pick_route(rng, ev, kind)
    a = pick_amount(rng, route[0], ev)
    c = Conv(rng, funded_world(rng, route, a))
    do_swap_turn(rng, c, ev, route, a)
    return finish(c, "swap", "swap", route, a, "swap")


def sc_special(rng, ev, mode):
    kind = "any"
    route = pick_route(rng, ev, "swap", mode)
    a = pick_amount(rng, route[0], ev)
    intent = rng.choice(["quote", "swap"])
    c = Conv(rng, funded_world(rng, route, a))
    (do_quote_turn if intent == "quote" else do_swap_turn)(rng, c, ev, route, a)
    return finish(c, "special_" + mode, intent, route, a, "special_" + mode)


def sc_quote_then_do(rng, ev):
    kind = rng.choices(["swap", "bridge", "same"], [.45, .35, .2])[0]
    route = pick_route(rng, ev, kind, rng.choices(["any", "btc", "bsc"], [.85, .07, .08])[0] if kind == "swap" else "any")
    a = pick_amount(rng, route[0], ev)
    c = Conv(rng, funded_world(rng, route, a))
    do_quote_turn(rng, c, ev, route, a)
    c.user(short_turn(rng, DO_IT, ev))
    r = c.call(("prepare_swap", args_of(route, a)))
    assert r.get("status") == "prepared_for_review"
    c.say(swap_answer(rng, r, route_kind(*route) == "bridge"))
    return finish(c, "quote_then_prepare", "swap", route, a, "quote_then_do")


def bal_amount(rng, w, sym, half):
    p = w.prices[sym]
    usd_v = rng.choice([rng.uniform(15, 400), rng.uniform(400, 9000)])
    dp = 2 if sym in STABLE else 4 if p > 500 else 2
    for _ in range(50):
        v = round(usd_v / p, dp)
        s = fmt_plain(v)
        if v > 0 and not s.endswith("0") and (not half or Decimal(s) / 2 == (Decimal(s) / 2).quantize(Decimal(10) ** -(dp + 1))):
            return v
        usd_v *= 1.013
    raise Retry()


def sc_all_half(rng, ev, mode):
    intent = rng.choices(["swap", "quote"], [.8, .2])[0]
    if intent == "quote":
        mode = "all"
    kind = rng.choices(["swap", "bridge", "same"], [.4, .45, .15])[0]
    route = pick_route(rng, ev, kind)
    fs, fc, ts, tc = route
    if mode == "all" and fs in ("ETH", "SOL", "BNB", "NEAR", "POL", "MON"):
        raise Retry()
    w = RRWorld(rng)
    bal = bal_amount(rng, w, fs, mode == "half")
    w = make_world(rng, [(fs, fc, bal)], avoid_syms=())
    # same token on another chain sometimes (the user's chain word matters)
    if rng.random() < 0.3 and len(TOKEN_CHAINS[fs]) > 1:
        oc = rng.choice([x for x in TOKEN_CHAINS[fs] if x != fc and x not in NO_SRC_CHAINS] or [None])
        if oc:
            put_holding(w, fs, oc, bal_amount(rng, w, fs, False))
    c = Conv(rng, w)
    S = slots(rng, fs, fc, ts, tc, 1)
    if intent == "quote":
        pool = QUOTE_ALLB if kind == "bridge" else QUOTE_ALL
    else:
        pool = (ALLB if kind == "bridge" else ALLX) if mode == "all" else (HALFB if kind == "bridge" else HALFX)
    S["fc"] = cw(rng, fc)
    c.user(render_user(rng, split_pick(rng, pool, ev), S))
    rb = c.call(("get_balances", {}))
    row = [h for h in rb["holdings"] if h["symbol"] == fs and norm_chain(h["chain"]) == fc]
    assert len(row) == 1, rb
    amount = jn(row[0]["amount"])
    if mode == "half":
        rc = c.call(("calculator", {"expression": f"{row[0]['amount']} / 2"}))
        amount = jn(rc["result"])
    args = args_of(route, amount)
    if intent == "quote":
        r = c.call(("get_quote", args))
        assert "error" not in r, r
        c.say(quote_answer(rng, r))
    else:
        r = c.call(("prepare_swap", args))
        assert r.get("status") == "prepared_for_review", r
        c.say(swap_answer(rng, r, kind == "bridge"))
    return finish(c, "swap_" + mode, intent, route, amount, "all_half", mode=mode, balance=row[0]["amount"])


def sc_nosrc_one(rng, ev):
    intent = rng.choices(["swap", "quote"], [.6, .4])[0]
    kind = rng.choices(["swap", "bridge"], [.55, .45])[0]
    for _ in range(100):
        route = pick_route(rng, ev, kind)
        if route[0] in MULTI:
            break
    else:
        raise Retry()
    fs, fc, ts, tc = route
    if fs not in MULTI:
        raise Retry()
    a = pick_amount(rng, fs, ev)
    w = make_world(rng, [(fs, fc, round(float(a) * rng.uniform(1.2, 4), 4))], avoid_syms=(fs,))
    put_holding(w, fs, fc, round(float(a) * rng.uniform(1.2, 4), 4))
    c = Conv(rng, w)
    S = slots(rng, fs, fc, ts, tc, a)
    if kind == "bridge":
        pool = QUOTE_NOSRC_B if intent == "quote" else BRIDGE_NOSRC
    else:
        pool = QUOTE_NOSRC if intent == "quote" else SWAP_NOSRC
    c.user(render_user(rng, split_pick(rng, pool, ev), S))
    rb = c.call(("get_balances", {}))
    row = [h for h in rb["holdings"] if h["symbol"] == fs]
    assert len(row) == 1
    chain = norm_chain(row[0]["chain"])
    route = (fs, chain, ts, tc)
    args = args_of(route, a)
    if intent == "quote":
        r = c.call(("get_quote", args)); assert "error" not in r, r
        c.say(quote_answer(rng, r))
    else:
        r = c.call(("prepare_swap", args)); assert r.get("status") == "prepared_for_review", r
        c.say(swap_answer(rng, r, kind == "bridge"))
    return finish(c, "nosrc_held_one", intent, route, a, "nosrc_one")


def sc_nosrc_many(rng, ev):
    intent = rng.choices(["swap", "quote"], [.6, .4])[0]
    kind = rng.choices(["swap", "bridge"], [.6, .4])[0]
    fs = wchoice(rng, MULTI)
    held = rng.sample([ch for ch in TOKEN_CHAINS[fs] if ch not in NO_SRC_CHAINS], rng.choice([2, 2, 3]))
    for _ in range(100):
        ts = fs if kind == "bridge" else wchoice(rng, [t for t in DST_TOKENS if t != fs])
        tcs = [x for x in TOKEN_CHAINS[ts] if x not in held or kind == "swap"]
        if tcs:
            tc = rng.choice(tcs)
            break
    else:
        raise Retry()
    chosen = rng.choice(held)
    route = (fs, chosen, ts, tc)
    if route_kind(*route) == "same":
        raise Retry()
    # eval/train split on the (token, destination, amount) tuple
    if ev_pair((fs, "x", ts, tc)) != ev:
        raise Retry()
    a = pick_amount(rng, fs, ev)
    w = make_world(rng, [(fs, ch, round(float(a) * rng.uniform(1.2, 4), 4)) for ch in held], avoid_syms=(fs,))
    for ch in held:
        put_holding(w, fs, ch, round(float(a) * rng.uniform(1.2, 4), 4))
    c = Conv(rng, w)
    S = slots(rng, fs, chosen, ts, tc, a)
    if kind == "bridge":
        pool = QUOTE_NOSRC_B if intent == "quote" else BRIDGE_NOSRC
    else:
        pool = QUOTE_NOSRC if intent == "quote" else SWAP_NOSRC
    c.user(render_user(rng, split_pick(rng, pool, ev), S))
    rb = c.call(("get_balances", {}))
    chains = [h["chain"] for h in rb["holdings"] if h["symbol"] == fs]
    c.say(ask_from(rng, fs, chains), ask=True)
    cont = rng.random() < 0.4
    if not cont:
        return finish(c, "nosrc_held_many_ask", intent, route, a, "nosrc_many", ends_with_question=True)
    word = cw(rng, chosen)
    c.user(short_turn(rng, SRC_ANS, ev, c=word))
    args = args_of(route, a)
    if intent == "quote":
        r = c.call(("get_quote", args)); assert "error" not in r, r
        c.say(quote_answer(rng, r))
    else:
        r = c.call(("prepare_swap", args)); assert r.get("status") == "prepared_for_review", r
        c.say(swap_answer(rng, r, kind == "bridge"))
    return finish(c, "nosrc_held_many_then_call", intent, route, a, "nosrc_many", ends_with_question=False)


def sc_nodst(rng, ev):
    intent = rng.choices(["swap", "quote"], [.6, .4])[0]
    bridge = rng.random() < 0.35
    fs, fc = pick_src(rng, exclude=("PEPE",))
    if bridge and fs not in MULTI:
        raise Retry()
    ts = fs if bridge else rng.choice([t for t in MULTI if t != fs])
    opts = [x for x in TOKEN_CHAINS[ts] if x not in NO_SRC_CHAINS and (x != fc or not bridge)]
    if ts == "USDC":
        opts = opts + []
    tc = rng.choice(opts)
    route = (fs, fc, ts, tc)
    if (fs, fc) == (ts, tc):
        raise Retry()
    if ev_pair((fs, fc, ts, "x")) != ev:
        raise Retry()
    a = pick_amount(rng, fs, ev)
    w = funded_world(rng, route, a)
    c = Conv(rng, w)
    S = slots(rng, fs, fc, ts, tc, a)
    if bridge:
        pool = QUOTE_NODST_B if intent == "quote" else BRIDGE_NODST
    else:
        pool = QUOTE_NODST if intent == "quote" else SWAP_NODST
    c.user(render_user(rng, split_pick(rng, pool, ev), S))
    c.say(ask_to(rng, ts, bridge), ask=True)
    if rng.random() >= 0.4:
        return finish(c, "nodst_ask", intent, route, a, "nodst", ends_with_question=True)
    c.user(short_turn(rng, DST_ANS, ev, c=cw(rng, tc)))
    args = args_of(route, a)
    if intent == "quote":
        r = c.call(("get_quote", args)); assert "error" not in r, r
        c.say(quote_answer(rng, r))
    else:
        r = c.call(("prepare_swap", args)); assert r.get("status") == "prepared_for_review", r
        c.say(swap_answer(rng, r, bridge))
    return finish(c, "nodst_then_call", intent, route, a, "nodst", ends_with_question=False)


def sc_none_held(rng, ev):
    fs = wchoice(rng, MULTI)
    kind = rng.choice(["swap", "bridge"])
    ts = fs if kind == "bridge" else rng.choice([t for t in DST_TOKENS if t != fs])
    tc = rng.choice([x for x in TOKEN_CHAINS[ts] if x not in NO_SRC_CHAINS] if kind == "bridge" else TOKEN_CHAINS[ts])
    fc = next(x for x in TOKEN_CHAINS[fs] if x not in NO_SRC_CHAINS and x != tc)
    route = (fs, fc, ts, tc)
    if ev_pair((fs, "x", ts, tc)) != ev:
        raise Retry()
    a = pick_amount(rng, fs, ev)
    w = make_world(rng, avoid_syms=(fs,), extras=rng.choice([0, 1, 2]))
    c = Conv(rng, w)
    S = slots(rng, fs, fc, ts, tc, a)
    pool = BRIDGE_NOSRC if kind == "bridge" else SWAP_NOSRC
    c.user(render_user(rng, split_pick(rng, pool, ev), S))
    c.call(("get_balances", {}))
    c.say(rng.choice([f"I don't see any {fs} in your wallet, so there's nothing to {'bridge' if kind == 'bridge' else 'swap'}.",
                      f"There's no {fs} in your wallet right now, so I can't prepare that.",
                      f"Your wallet has no {fs}, so there's nothing to {'bridge' if kind == 'bridge' else 'swap'}.",
                      f"I checked your balances and there's no {fs} there, so I can't prepare it."]))
    return finish(c, "nosrc_none_held", "swap", route, a, "none_held")


def sc_error(rng, ev, sub):
    intent = rng.choice(["quote", "swap"])
    if sub == "min":
        kind = rng.choice(["swap", "bridge", "same"])
        route = None
        for _ in range(200):
            r = pick_route(rng, ev, kind)
            if r[0] in AMOUNTS:
                route = r
                break
        if not route:
            raise Retry()
        fs, fc, ts, tc = route
        w0 = RRWorld(rng)
        p = w0.prices[fs]
        a = float(f"{rng.uniform(0.1, 0.95) / p:.2g}")
        if a * p >= 1 or a <= 0:
            raise Retry()
        a = jn(a)
    elif sub == "noroute":
        bad = [("USDT", "base"), ("USDT", "polygon"), ("PEPE", "base"), ("BNB", "ethereum"), ("SOL", "base"), ("NEAR", "arbitrum"),
               ("MON", "bsc"), ("AAVE", "solana"), ("ARB", "base"), ("LINK", "polygon")]
        i = rng.randrange(len(bad))
        if (i % 4 == 0) != ev:
            raise Retry()
        s, ch = bad[i]
        if rng.random() < 0.5:
            fs, fc = rng.choice([("USDC", x) for x in ("base", "arbitrum", "solana", "ethereum")]); ts, tc = s, ch
        else:
            fs, fc = s, ch; ts, tc = rng.choice([("USDC", "base"), ("ETH", "arbitrum"), ("USDC", "ethereum")])
        if fs in AMOUNTS:
            a = pick_amount(rng, fs, ev)
        else:
            a = rng.choice([10, 25, 50, 100, 250])
        route = (fs, fc, ts, tc)
    elif sub == "btcsrc":
        fs, fc = "BTC", "bitcoin"
        ts = rng.choice(["ETH", "USDC", "SOL"]); tc = rng.choice(TOKEN_CHAINS[ts])
        route = (fs, fc, ts, tc)
        if ev_pair(route) != ev:
            raise Retry()
        a = pick_amount(rng, "BTC", ev)
    else:
        fs = rng.choice(["ETH", "USDT", "SOL", "BNB"])
        fc = rng.choice(TOKEN_CHAINS[fs])
        ts, tc = fs, "hyperliquid"
        if rng.random() < 0.5:
            ts = rng.choice(["ETH", "SOL"]); ts = ts if ts != fs else "BNB"
        route = (fs, fc, ts, tc)
        if ev_pair(route) != ev:
            raise Retry()
        a = pick_amount(rng, fs, ev)
    w0 = locals().get("w0")
    w = funded_world(rng, route, a, w=w0) if fs in SRC_TOKENS and fc in TOKEN_CHAINS.get(fs, []) and fc not in NO_SRC_CHAINS else make_world(rng, w=w0)
    c = Conv(rng, w)
    c.user(user_text_err(rng, ev, intent, route, a))
    tool = "get_quote" if intent == "quote" else "prepare_swap"
    r = c.call((tool, args_of(route, a)))
    assert "error" in r, (r, route, a)
    c.say(error_answer(rng, r["error"], intent, a, fs, fc, ts, tc))
    return finish(c, "error_" + sub, intent, route, a, "error_" + sub, error=r["error"])


def user_text_err(rng, ev, intent, route, a):
    fs, fc, ts, tc = route
    kind = route_kind(fs, fc, ts, tc)
    if fc == tc and fs != ts:
        kind = "same"
    return user_text(rng, ev, intent, route, a)


def sc_policy(kind_pol):
    def fn(rng, ev):
        k = rng.choices(["swap", "bridge", "same"], [.45, .4, .15])[0]
        route = None
        for _ in range(100):
            r = pick_route(rng, ev, k)
            if r[0] in STABLE | {"ETH"}:
                route = r
                break
        if not route:
            raise Retry()
        fs = route[0]
        a = rng.choice([900, 1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000, 15000, 25000]) if fs in STABLE \
            else rng.choice([1, 2, 3, 5, 8, 10, 15])
        w = RRWorld(rng)
        c_amt = float(a) * rng.uniform(1.15, 1.8)
        w = make_world(rng, [(fs, route[1], round(c_amt, 4))])
        put_holding(w, fs, route[1], round(c_amt, 4))
        if kind_pol == "warn":
            w.policy = {"warn": rng.choice([["This is more than half of your balance."],
                                           ["This is more than half of your balance.", "The price impact looks higher than usual for this size."],
                                           ["The price impact looks higher than usual for this size."],
                                           ["This is a large amount for one move."]])}
        else:
            w.policy = {"block": rng.choice([["Amount is above your daily limit for the assistant."],
                                            ["Amount is above the per-move limit for the assistant."],
                                            ["Amount is above your daily limit for the assistant.", "Amount is above the per-move limit for the assistant."]])}
        c = Conv(rng, w)
        c.user(user_text(rng, ev, "swap", route, a))
        r = c.call(("prepare_swap", args_of(route, a)))
        bridge = route_kind(*route) == "bridge"
        if kind_pol == "warn":
            assert r.get("policy") == "warn" and r.get("status") == "prepared_for_review", r
            ws = " ".join(r["warnings"])
            c.say(swap_answer(rng, r, bridge) + rng.choice([f" One warning: {ws}", f" Heads up: {ws}", f" Note: {ws}"]))
        else:
            assert r.get("status") == "blocked", r
            rs = " ".join(r["reasons"])
            c.say(rng.choice([f"I couldn't prepare it. {rs} Nothing was prepared.", f"It was blocked: {rs} So nothing was prepared.",
                              f"I can't prepare that one. {rs} Nothing is waiting for approval.",
                              f"That swap wasn't prepared. {rs} Nothing is waiting for approval."]))
        return finish(c, "policy_" + kind_pol, "swap", route, a, "policy_" + kind_pol)
    return fn


def sc_which(rng, ev):
    prepared = rng.random() < 0.5
    kind = rng.choices(["swap", "bridge", "same"], [.45, .4, .15])[0]
    route = pick_route(rng, ev, kind, "any")
    a = pick_amount(rng, route[0], ev)
    c = Conv(rng, funded_world(rng, route, a))
    r = (do_swap_turn if prepared else do_quote_turn)(rng, c, ev, route, a)
    q = split_pick(rng, WHICH, ev, mod=5)
    flag = next(s.partition("|")[2] for s in WHICH if s.partition("|")[0] == q)
    q2 = q + rng.choice(["", "", "", "?", " ok", " thanks"])
    if rng.random() < 0.3:
        q2 = q2.lower()
    if q2.endswith("??"):
        q2 = q2[:-1]
    c.user(q2)
    c.say(which_answer(rng, r, flag, prepared))
    return finish(c, "which_route", "swap" if prepared else "quote", route, a, "which_route", question_kind=flag)


def sc_rate(rng, ev):
    for _ in range(200):
        route = pick_route(rng, ev, rng.choice(["swap", "swap", "bridge"]))
        if route[2] in ("ETH", "BTC", "SOL", "BNB", "NEAR", "AAVE", "LINK", "UNI") and route[2] != route[0]:
            break
    else:
        raise Retry()
    a = pick_amount(rng, route[0], ev)
    c = Conv(rng, make_world(rng))
    r = do_quote_turn(rng, c, ev, route, a)
    ts = route[2]
    t = tk(rng, ts)
    c.user(short_turn(rng, RATE, ev, t=t))
    rc = c.call(("calculator", {"expression": f"round({fmt_plain(r['amount_in_usd'])} / {r['amount_out']}, 2)"}))
    val = rc["result"]
    c.say(rng.choice([f"That works out to about ${val} per {ts}, with costs included.",
                      f"About ${val} per {ts} all in, using the quote's dollar value and the amount out.",
                      f"Roughly ${val} for each {ts}, costs included. It's still only a quote.",
                      f"It comes to about ${val} per {ts} once costs are counted.",
                      f"At this quote you pay about ${val} per {ts}, all in."]))
    return finish(c, "rate_calc", "quote", route, a, "rate")


SCEN = {
    "quote": (sc_quote, 62), "swap": (sc_swap, 90), "special_btc": (lambda r, e: sc_special(r, e, "btc"), 17), "special_hl": (lambda r, e: sc_special(r, e, "hl"), 14),
    "special_bsc": (lambda r, e: sc_special(r, e, "bsc"), 14), "quote_then_do": (sc_quote_then_do, 45),
    "all": (lambda r, e: sc_all_half(r, e, "all"), 28), "half": (lambda r, e: sc_all_half(r, e, "half"), 22), "nosrc_one": (sc_nosrc_one, 28), "nosrc_many": (sc_nosrc_many, 18),
    "nodst": (sc_nodst, 20), "none_held": (sc_none_held, 6), "error_min": (lambda r, e: sc_error(r, e, "min"), 14), "error_noroute": (lambda r, e: sc_error(r, e, "noroute"), 8),
    "error_btcsrc": (lambda r, e: sc_error(r, e, "btcsrc"), 5), "error_hldst": (lambda r, e: sc_error(r, e, "hldst"), 5), "warn": (sc_policy("warn"), 18),
    "block": (sc_policy("block"), 18), "which": (sc_which, 25), "rate": (sc_rate, 15),
}


def plan(n, minimum):
    tot = sum(w for _, w in SCEN.values())
    raw = {k: w * n / tot for k, (_, w) in SCEN.items()}
    cnt = {k: max(minimum, int(v)) for k, v in raw.items()}
    order = sorted(raw, key=lambda k: -(raw[k] - int(raw[k])))
    i = 0
    while sum(cnt.values()) < n:
        cnt[order[i % len(order)]] += 1; i += 1
    while sum(cnt.values()) > n:
        k = max(cnt, key=lambda k: cnt[k]); cnt[k] -= 1
    return cnt


def norm_text(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def generate(seed, n, ev, avoid):
    rng = random.Random(seed)
    rows, seen = [], set(avoid)
    for name, cnt in plan(n, 1 if ev else 0).items():
        fn = SCEN[name][0]
        made, tries = 0, 0
        while made < cnt:
            tries += 1
            if tries > 3000 + cnt * 300:
                raise RuntimeError(f"cannot generate {cnt} unique '{name}' rows (got {made})")
            crng = random.Random(rng.getrandbits(64))
            try:
                ex = fn(crng, ev)
            except Retry:
                continue
            us = [norm_text(m["content"]) for m in ex["messages"] if m["role"] == "user"]
            if any(u in seen for u in us) or len(set(us)) != len(us):
                continue
            seen.update(us)
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
    opt = [tc["name"] for tc in calls if tc["name"] == "calculator"]
    final = [m for m in ref if m["role"] == "assistant" and not m.get("tool_calls")][-1]["content"]
    sp = ex["meta"]["spec"]
    ask = bool(ex["meta"]["expect"] and ex["meta"]["expect"][-1].get("ask"))
    results = [json.loads(m["content"]) for m in ref if m["role"] == "tool"]
    has_route = any("route" in r for r in results)
    # the route result may come from an earlier turn (which_route): look in the whole context too
    ctx_results = [json.loads(m["content"]) for m in msgs[:lu + 1] if m["role"] == "tool"]
    has_route = has_route or (any("route" in r for r in ctx_results) and sp["scenario"] == "which_route")
    intent = sp["intent"]
    checks = ["decimals_ok", "no_unbacked_done"]
    if has_route:
        checks.append("route_text")
    if req:
        checks += ["right_tool", "exact_digits"]
    if ask:
        checks.append("asked_when_needed")
    if sp["scenario"] in ("which_route",):
        checks += ["right_tool"]
    forbid = ["prepare_swap"] if intent == "quote" and sp["scenario"] != "which_route" else []
    if sp["scenario"] == "which_route":
        forbid = ["get_quote", "prepare_swap", "get_balances", "calculator"]
    if ask and not req:
        forbid = ["get_quote", "prepare_swap"]
    if ask and req:
        forbid = ["get_quote", "prepare_swap"]       # look at balances, then ask; no money call yet
    if sp["scenario"] == "none_held":
        forbid = ["prepare_swap", "get_quote"]
    if sp["scenario"].startswith("error_") and intent == "quote":
        forbid = ["prepare_swap"]
    gold = {"calls": req, "optional_calls": opt, "forbid_tools": sorted(set(forbid)),
            "ask": ask, "route": ROUTE if has_route else None, "must_not_contain": ["Relay"],
            "scenario": sp["scenario"], "reference_answer": final}
    if has_route and results:
        r = next((r for r in results if "to" in r), None)
        if r:
            gold["must_contain"] = [r["route"], r["to"], r["from"]] if sp["scenario"] not in ("which_route",) else [r["route"]]
            gold["amount_out"] = r["amount_out"]
    if sp["scenario"] == "which_route":
        r = next(r for r in ctx_results if "route" in r)
        gold["must_contain"] = [r["route"]]
    ctx = msgs[:lu + 1]
    meta = dict(ex["meta"])
    return {"id": f"route_rounding-{i:03d}", "kind": ex["meta"]["kind"], "tools": True, "bio": ex["meta"]["bio"],
            "context": ctx, "reference": ref, "expect": ex["meta"]["expect"], "world": ex["meta"]["world"], "checks": checks,
            "gold": gold, "meta": {k: meta[k] for k in ("source", "generator", "kind", "bio", "spec", "lang")}}


# ------------------------------------------------------------------ self-check
BANNED = [r"\bRelay\b", r"\bswapped\b", r"\bbridged\b", r"went through", r"has been (sent|swapped|bought|deposited|withdrawn|closed|bridged)",
          r"\bI(?:'ve| have)? (sent|bought|swapped|deposited|withdrew|closed|bridged)\b", r"successfully", r"funds (are|have) (in|arrived)",
          r"^\s*done\b", r"not financial advice", "[·•]", r"what('s| is) (the )?(current )?price", r"your budget",
          r"how much (do you|are you)", r"can't see the future", r"\bcompleted\b", r"\bexecuted\b"]
BANNED_RE = [re.compile(p, re.I) for p in BANNED]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
ADDR_RE = re.compile(r"0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44}")


def user_numbers(text):
    """numbers a user text can legitimately stand for (typed forms)"""
    out = set()
    t = text.lower()
    for m in re.finditer(r"(\d[\d,]*\.?\d*)\s*k\b", t):
        out.add(Decimal(m.group(1).replace(",", "")) * 1000)
    for m in re.finditer(r"(?<![\w.])\.(\d+)", t):
        out.add(Decimal("0." + m.group(1)))
    for m in re.finditer(r"\d[\d,]*\.?\d*", t):
        try:
            out.add(Decimal(m.group().replace(",", "").rstrip(".")))
        except Exception:
            pass
    for k, v in WORDS.items():
        if re.search(r"\b" + v + r"\b", t):
            out.add(Decimal(k) if v != "a hundred" else Decimal(100))
    return out


def dnorm(s):
    return Decimal(s).normalize()


def check_row(row, idx, replay=True):
    errs = []
    E = lambda m: errs.append(f"[{idx}] {m}")
    s = json.dumps(row, ensure_ascii=False)
    if json.loads(s) != row:
        E("json roundtrip")
    if "messages" in row:
        msgs, meta, tools_flag = row["messages"], row["meta"], row["tools"]
        bio = meta["bio"]
        world = meta["world"]
    else:
        msgs, meta, tools_flag = row["context"] + row["reference"], row["meta"], row["tools"]
        bio, world = row["bio"], row["world"]
    sp = meta["spec"]
    if tools_flag is not True:
        E("tools flag")
    if meta.get("source") != "route_rounding":
        E("source")
    if msgs[0]["role"] != "system":
        E("system first")
    else:
        base = SYSTEM_V2.replace("{BIO}", bio)
        if not msgs[0]["content"].startswith(base):
            E("system prompt not from SYSTEM_AGENT_v2")
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    for m in msgs:
        if other in (m.get("content") or ""):
            E(f"{other} in a {bio} example")
        if "{BIO}" in (m.get("content") or ""):
            E("unsubstituted {BIO}")
    # structure + tool call validity
    users = [m["content"] for m in msgs if m["role"] == "user"]
    utext = " \n ".join(users)
    all_calls = []
    for k, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            nres = 0
            j = k + 1
            while j < len(msgs) and msgs[j]["role"] == "tool":
                nres += 1; j += 1
            if nres != len(m["tool_calls"]):
                E("tool results count")
            for tc in m["tool_calls"]:
                all_calls.append(tc)
                if tc["name"] not in H.TOOL_NAMES:
                    E(f"unknown tool {tc['name']}"); continue
                sch = H.TOOL_SCHEMA[tc["name"]]
                for req in sch.get("required", []):
                    if req not in tc["arguments"]:
                        E(f"missing arg {req}")
                for kk, v in tc["arguments"].items():
                    if kk not in sch["properties"]:
                        E(f"unknown arg {kk}"); continue
                    t = sch["properties"][kk]["type"]
                    if t == "number" and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
                        E(f"type {kk}")
                    if t == "string" and not isinstance(v, str):
                        E(f"type {kk}")
    # scenario args exactly (independent of the generator's call code)
    mc = [tc for tc in all_calls if tc["name"] in ("get_quote", "prepare_swap")]
    scen = sp["scenario"]
    fs, fc, ts, tc_ = sp["route"]
    want = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc_, "amount": sp["amount"]}
    if scen in ("nosrc_many", "nodst", "none_held") and sp.get("ends_with_question") is not False and scen != "none_held":
        if mc:
            E("call before the question was answered")
    elif scen == "none_held":
        if mc:
            E("none_held must not call a money tool")
    else:
        if not mc:
            E("no quote/prepare call")
        for c in mc:
            if c["arguments"] != want:
                E(f"args {c['arguments']} != scenario {want}")
        # right tool
        names = {c["name"] for c in mc}
        if scen == "which_route":
            pass
        if sp["intent"] == "quote" and "prepare_swap" in names and scen != "quote_then_do":
            E("quote question prepared")
        if sp["intent"] == "swap" and names != {"prepare_swap"} and scen != "quote_then_do":
            E("swap request did not prepare")
        if scen == "quote_then_do":
            if [c["name"] for c in mc] != ["get_quote", "prepare_swap"] or mc[0]["arguments"] != mc[1]["arguments"]:
                E("quote_then_do args differ")
        if scen == "which_route" and sum(1 for m in msgs if m["role"] == "assistant" and m.get("tool_calls")) != 1:
            E("which_route extra calls")
    for c in mc:
        a = c["arguments"]
        for kk in ("from_chain", "to_chain"):
            if a[kk] != a[kk].lower() or a[kk] not in DISPLAY:
                E(f"chain id {a[kk]}")
        for kk in ("from_symbol", "to_symbol"):
            if a[kk] != a[kk].upper():
                E(f"ticker case {a[kk]}")
        if a["amount"] <= 0 or isinstance(a["amount"], bool):
            E("amount")
        # amount: typed by the user, or from balances / calculator
        nums = set()
        for u in users:
            nums |= user_numbers(u)
        derived = {Decimal(repr(float(x))) for x in []}
        for m in msgs:
            if m["role"] == "tool":
                r = json.loads(m["content"])
                for h in r.get("holdings", []):
                    nums.add(Decimal(h["amount"]))
                if "result" in r:
                    nums.add(Decimal(r["result"]))
        if Decimal(repr(a["amount"])) not in nums and Decimal(str(a["amount"])).normalize() not in {n.normalize() for n in nums}:
            E(f"amount {a['amount']} not typed/derived")
        # symbols and chains mentioned (alias) or implied/from results
        lt = utext.lower()
        for sym in (a["from_symbol"], a["to_symbol"]):
            ok = re.search(r"\b" + sym.lower() + r"\b", lt) or (sym == "BTC" and "bitcoin" in lt) or (sym == "BNB" and "bnb" in lt)\
                or (sym == "USDC" and a["to_chain"] == "hyperliquid" and re.search(ALIAS_RE["hyperliquid"], lt))
            if not ok:
                E(f"symbol {sym} not in user text")
        balres = " ".join(m["content"] for m in msgs if m["role"] == "tool" and "holdings" in m["content"])
        for kk, symk in (("from_chain", "from_symbol"), ("to_chain", "to_symbol")):
            ch, sym = a[kk], a[symk]
            if re.search(ALIAS_RE[ch], lt):
                continue
            if NATIVE.get(sym) == ch:
                continue
            if kk == "from_chain" and f'"chain": "{"bnb" if ch == "bsc" else ch}"' in balres:
                continue
            E(f"chain {ch} not named, implied or from balances")
    # addresses: none expected anywhere
    for m in msgs:
        if m["role"] != "system" and m.get("content") and ADDR_RE.search(m["content"]):
            if m["role"] == "user" or m["role"] == "assistant":
                E("address-like string")
    # results + assistant text
    allowed = set()
    nums_user = set()
    for u in users:
        nums_user |= user_numbers(u)
    last_route_res = None
    seen_results_numbers = set()
    for k, m in enumerate(msgs):
        if m["role"] == "tool":
            r = json.loads(m["content"])
            for x in NUM_RE.findall(m["content"]):
                seen_results_numbers.add(dnorm(x))
            if "route" in r and r["route"] != ROUTE:
                E("route not NEAR Intents")
            if "relay" in m["content"].lower():
                E("Relay in tool result")
            if "amount_out" in r:
                if not valid_approx(r["amount_out"]):
                    E(f"amount_out digits {r['amount_out']}")
                if r["to"] != f"about {r['amount_out']} {r['to'].split()[2]} on {r['to'].split(' on ')[1]}":
                    E("to string")
                if Decimal(repr(r["amount_in_usd"])) - Decimal(repr(r["total_cost_usd"])) != Decimal(repr(r["amount_out_usd"])):
                    E("usd arithmetic")
            last_route_res = r if "route" in r else last_route_res
        if m["role"] == "assistant":
            txt = m.get("content") or ""
            if not txt:
                continue
            for rx in BANNED_RE:
                if rx.search(txt):
                    E(f"banned phrase {rx.pattern!r} in {txt!r}")
            if EMOJI_RE.search(txt):
                E("emoji")
            if len(txt.split()) > 70:
                E(f"too long ({len(txt.split())} words)")
            for x in NUM_RE.findall(txt):
                d = dnorm(x)
                if d not in seen_results_numbers and d.normalize() not in {n.normalize() for n in nums_user}:
                    E(f"number {x} in answer not from results/user: {txt!r}")
            if "Face ID" in txt or "Touch ID" in txt:
                if bio not in txt:
                    E("wrong bio word")
    # every quote/prepare answer names the route; non-route answers do not
    for k, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("content") and k > 0:
            prev_tools = []
            j = k - 1
            while j > 0 and msgs[j]["role"] == "tool":
                prev_tools.append(json.loads(msgs[j]["content"])); j -= 1
            named = ROUTE in m["content"]
            if any("route" in r for r in prev_tools) and not named:
                E(f"answer misses the route: {m['content']!r}")
            if any(("error" in r or "holdings" in r or r.get("status") == "blocked") for r in prev_tools) and named:
                E("route named without a route result")
    if scen == "which_route":
        if ROUTE not in msgs[-1]["content"]:
            E("which_route answer misses route")
    # no done claims without result: prepared texts say nothing moved yet
    for k, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("content") and k > 0 and "prepared" in m["content"].lower() and "nothing" not in m["content"].lower() \
                and "waiting" not in m["content"].lower():
            E(f"'prepared' without nothing-moved/waiting: {m['content']!r}")
    # replay
    if replay:
        w = RRWorld.from_state(world)
        ci = 0
        first_user_world = None
        for k, m in enumerate(msgs):
            if m["role"] == "assistant" and m.get("tool_calls"):
                for t_i, c in enumerate(m["tool_calls"]):
                    res = dump(w.call(c["name"], c["arguments"])).replace("Face ID", bio)
                    rec = msgs[k + 1 + t_i]["content"]
                    if res != rec:
                        E(f"replay mismatch for {c['name']}: {res[:90]} vs {rec[:90]}")
    # render round trip
    plain = [dict(m) for m in msgs]
    text = H.render_text(plain, True)
    calls_back = []
    for part in text.split("<|im_start|>assistant\n")[1:]:
        cs, bad = H.parse_calls(part.split("<|im_end|>")[0])
        if bad:
            E("render: bad call")
        calls_back += cs
    if calls_back != [dict(name=c["name"], arguments=c["arguments"]) for c in all_calls]:
        E("render round trip calls")
    return errs, len(text)


def self_check(train, ev_rows, avoid_extra):
    errs = []
    maxlen = 0
    for i, r in enumerate(train):
        e, ln = check_row(r, f"train {i}")
        errs += e; maxlen = max(maxlen, ln)
    for i, r in enumerate(ev_rows):
        e, ln = check_row(r, f"eval {i}")
        errs += e; maxlen = max(maxlen, ln)
    # duplicates
    def utexts(r):
        ms = r["messages"] if "messages" in r else r["context"] + r["reference"]
        return [norm_text(m["content"]) for m in ms if m["role"] == "user"]
    seen = {}
    for name, rows in (("train", train), ("eval", ev_rows)):
        for i, r in enumerate(rows):
            for u in utexts(r):
                if u in seen:
                    errs.append(f"duplicate user text {u!r}: {seen[u]} and {name} {i}")
                seen[u] = f"{name} {i}"
                if u in avoid_extra:
                    errs.append(f"{name} {i} user text collides with eval2/eval: {u!r}")
    # eval disjoint templates: exact user text overlap already covered; report held-out route pairs
    tr_pairs = {tuple(r["meta"]["spec"]["route"]) for r in train}
    ev_pairs = {tuple(r["meta"]["spec"]["route"]) for r in ev_rows}
    return errs, maxlen, len(tr_pairs & ev_pairs)


def old_eval_texts():
    out = set()
    for d in ("eval2", "eval"):
        p = os.path.join(LAB, d)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if not fn.endswith(".jsonl"):
                continue
            for line in open(os.path.join(p, fn)):
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                ms = o.get("context", []) + o.get("reference", []) + o.get("messages", [])
                for m in ms:
                    if isinstance(m, dict) and m.get("role") == "user":
                        out.add(norm_text(m.get("content") or ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--eval-seed", type=int, default=9305)
    ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(LAB, "data3", "route_rounding.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval3", "route_rounding.jsonl"))
    a = ap.parse_args()
    old = old_eval_texts()
    ev_ex = generate(a.eval_seed, a.eval_n, True, old)
    ev_rows = [eval_record(ex, i) for i, ex in enumerate(ev_ex)]
    ev_texts = {norm_text(m["content"]) for ex in ev_ex for m in ex["messages"] if m["role"] == "user"}
    train = generate(a.seed, a.n, False, old | ev_texts)
    errs, maxlen, shared = self_check(train, ev_rows, old)
    if errs:
        print(f"SELF-CHECK FAILED: {len(errs)} problems", file=sys.stderr)
        for e in errs[:40]:
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
    kinds = Counter(r["meta"]["kind"] for r in train)
    print("train kinds:", dict(sorted(kinds.items())), file=sys.stderr)
    print("eval kinds:", dict(sorted(Counter(r["kind"] for r in ev_rows).items())), file=sys.stderr)
    print(f"max rendered chars {maxlen} (~{maxlen // 3.5:.0f} tokens), route pairs shared train/eval: {shared}", file=sys.stderr)
    print(f"route_rounding: {len(train)} train, {len(ev_rows)} eval, self-check ok")


if __name__ == "__main__":
    main()
