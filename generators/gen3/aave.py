#!/usr/bin/env python3
"""Aave v3 prepare_deposit / prepare_withdraw data + eval for the next Hermit run (gen3).

Writes   ../data3/aave.jsonl   (train, seed 3, about 500 deposit/withdraw rows + 60 negatives)
         ../eval3/aave.jsonl   (60 held-out items, different seed, user texts disjoint from train)

Rules followed (see spec/NOTES_*.md):
  * gen2 helpers are imported, not copied (Conv, amt_text, hermit_fmt). hermit_fmt is pointed at
    spec/hermit_tools_v2.json after import (module globals), gen2 itself is untouched.
  * Every number in an assistant text comes from a tool result or from the user text; derived numbers
    (half of a balance) go through the calculator tool.
  * Tool results use the app's real shapes (sorted keys, Python style JSON, literal prepared strings).
  * {BIO} is rendered as Face ID (~70%) or Touch ID (~30%) in the system prompt, tool notes and answers.
Usage: python3 aave.py --seed 3 --n 500 [--eval-seed 303]
"""
import argparse, json, os, random, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GEN2 = os.path.join(HERE, "..", "gen2")
SPEC = os.path.join(HERE, "..", "spec")
sys.path.insert(0, GEN2)

import hermit_fmt as H                                    # gen2 helper, patched below (not edited)
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
H.TOOLS = TOOLS_V2
H.TOOL_NAMES = [t["function"]["name"] for t in TOOLS_V2]
H.TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
H.TOOLS_BLOCK = H.tools_block()

import sim                                                # gen2 helper
from build_agent import Conv, amt_text                    # gen2 helper

SYS_TEMPLATE = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().strip()
assert "{BIO}" in SYS_TEMPLATE

AAVE_CHAINS = ["ethereum", "base", "arbitrum"]
CHAIN_NAME = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "polygon": "Polygon",
              "solana": "Solana", "bnb": "BNB Chain"}
SYM_CHAINS = {"USDC": ["ethereum", "base", "arbitrum"], "ETH": ["ethereum", "base", "arbitrum"],
              "USDT": ["ethereum", "arbitrum"], "DAI": ["ethereum"]}
STABLE = {"USDC", "USDT", "DAI", "USDe"}
SYM_WORDS = {"USDC": ["USDC", "usdc", "USDC", "Usdc", "usd coin"], "ETH": ["ETH", "eth", "ETH", "Eth", "ether"],
             "USDT": ["USDT", "usdt", "tether"], "DAI": ["DAI", "dai"], "USDe": ["USDe"], "SOL": ["SOL", "sol"]}
SYM_ALIAS_RE = {"USDC": r"\b(usdc|usd coin)\b", "ETH": r"\b(eth|ether)\b", "USDT": r"\b(usdt|tether)\b",
                "DAI": r"\bdai\b", "USDe": r"\busde\b", "SOL": r"\bsol\b"}
CHAIN_WORDS = {"ethereum": ["ethereum", "Ethereum", "mainnet", "eth mainnet"], "base": ["base", "Base"],
               "arbitrum": ["arbitrum", "Arbitrum", "arb"]}
CHAIN_RE = {"ethereum": r"\b(ethereum|mainnet)\b", "base": r"\bbase\b", "arbitrum": r"\b(arbitrum|arb)\b"}
OTHER_CHAINS = [("Polygon", ["polygon", "Polygon", "polygon pos"]), ("Solana", ["solana", "Solana", "sol chain"]),
                ("BNB Chain", ["bnb chain", "BNB chain", "bsc", "BSC"]), ("Avalanche", ["avalanche", "avax", "Avalanche"]),
                ("Optimism", ["optimism", "Optimism", "op"]), ("Monad", ["monad", "Monad"]),
                ("Linea", ["linea", "Linea"]), ("zkSync", ["zksync", "zkSync"])]
OTHER_CHAIN_RE = r"\b(polygon|solana|sol chain|bnb chain|bsc|avalanche|avax|optimism|op|monad|linea|zksync)\b"

BLOCK_REASONS = ["This is above your daily limit.", "Your spending policy doesn't allow this right now.",
                 "Aave has paused this asset for now.", "The wallet is locked by a pending review. Approve or cancel it first."]

# ------------------------------------------------------------------------ number helpers
def fmt(x):
    """Shortest plain decimal text, no exponent, no trailing .0 (how the app prints amounts)."""
    x = float(x)
    s = repr(round(x, 10))
    if "e" in s or "E" in s:
        s = f"{x:.10f}".rstrip("0").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def num(x):
    x = float(x)
    return int(x) if x.is_integer() else x


NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*\.?\d*|(?<![\w])\.\d+")


def numbers_in(text):
    out = set()
    for m in NUM_RE.findall(text):
        t = m.replace(",", "").rstrip(".")
        try:
            out.add(float(t))
        except ValueError:
            pass
    return out


# ------------------------------------------------------------------------ simulator
class World3(sim.World):
    """gen2 World + app-shaped balances/yields and Aave v3 deposit/withdraw. Pure function of the state."""
    STATE_KEYS = sim.World.STATE_KEYS + ("aave", "pools", "bio", "block")

    # ---- deterministic failure text (so the eval harness reproduces it)
    def call(self, name, args):
        if self.fail_next == name:
            self.fail_next = None
            return {"error": "network timeout, try again"}
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args if isinstance(args, dict) else {}))
        except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as e:
            return {"error": f"bad arguments: {e}"}

    def t_get_balances(self, chain=None):
        rows = [b for b in self.balances if b["amount"] > 0 and (chain is None or b["chain"] == str(chain).lower())]
        return {"holdings": [{"amount": fmt(b["amount"]), "chain": b["chain"], "symbol": b["symbol"], "usd": b["usd"]} for b in rows]}

    def t_get_yields(self, asset=None, chain=None):
        a = None if asset is None else str(asset).upper().replace("USDE", "USDe")
        c = None if chain is None else str(chain).lower()
        rows = [p for p in self.pools if (a is None or p["asset"].upper() == a.upper()) and (c is None or p["chain"].lower() == c)]
        rows = sorted(rows, key=lambda p: -p["apy_pct"])[:5]
        return {"pools": [dict(sorted(p.items())) for p in rows]}

    def _holdings(self, symbol, chain=None):
        return [b for b in self.balances if b["symbol"].upper() == symbol.upper() and b["chain"] in AAVE_CHAINS
                and b["amount"] > 0 and (chain is None or b["chain"] == chain)]

    def _blocked(self):
        return {"note": "Nothing was prepared. Tell the user why.", "reasons": [self.block], "status": "blocked"}

    def _apy(self, sym, chain):
        for p in self.pools:
            if p["protocol"] == "Aave v3" and p["asset"] == sym and p["chain"].lower() == chain:
                return p["apy_pct"]
        return None

    def t_prepare_deposit(self, symbol, amount, project=None, chain=None):
        proj = str(project or "aave-v3").lower().replace(" ", "-")
        if proj not in ("aave-v3", "aave"):
            return {"note": "Only Aave v3 deposits can be prepared here. Other pools are in Earn, DeFi, with Ask.",
                    "prepared": False, "status": "unsupported"}
        sym = str(symbol).upper().replace("USDE", "USDe")
        amount = float(amount)
        if amount <= 0:
            return {"error": "amount must be positive"}
        c = None if chain is None else str(chain).lower()
        if c is not None and c not in AAVE_CHAINS:
            return {"note": "Nothing was prepared. Tell the user why.", "reasons": [f"Aave v3 can't be used on {c} here. It is Ethereum, Base and Arbitrum only."],
                    "status": "blocked"}
        if self.block:
            return self._blocked()
        cands = self._holdings(sym, c)
        if not cands:
            return {"holding": f"0 {sym}", "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"}
        if len(cands) > 1 and c is None:
            cands = sorted(cands, key=lambda b: -b["amount"])
            names = [CHAIN_NAME[b["chain"]] for b in cands]
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            detail = ", ".join(f"{CHAIN_NAME[b['chain']]}: {fmt(b['amount'])}" for b in cands)
            return {"note": "Nothing was prepared. Ask the user which network, in one short question.",
                    "options": [b["chain"] for b in cands], "prepared": False,
                    "question": f"Which network should the deposit use? You hold {sym} on {joined} ({detail}).",
                    "status": "needs_chain"}
        b = cands[0]
        if amount > b["amount"] + 1e-12:
            return {"holding": f"{fmt(b['amount'])} {sym} on {CHAIN_NAME[b['chain']]}",
                    "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"}
        return {"amount": f"{fmt(amount)} {sym}", "apy_pct": self._apy(sym, b["chain"]), "executed": False, "kind": "deposit",
                "note": f"Waiting for the user to review the deposit and approve it with {self.bio}.",
                "pool": f"Aave v3 on {CHAIN_NAME[b['chain']]}", "status": "review_opened"}

    def t_prepare_withdraw(self, symbol, chain=None, amount=None):
        sym = str(symbol).upper().replace("USDE", "USDe")
        c = None if chain is None else str(chain).lower()
        if c is not None and c not in AAVE_CHAINS:
            return {"note": "Nothing was prepared. Tell the user why.", "reasons": [f"Aave v3 can't be used on {c} here. It is Ethereum, Base and Arbitrum only."],
                    "status": "blocked"}
        if self.block:
            return self._blocked()
        pos = [p for p in self.aave if p["symbol"].upper() == sym.upper() and p["amount"] > 0 and (c is None or p["chain"] == c)]
        if not pos:
            return {"note": f"The user has no {sym} in Aave. Tell them.", "prepared": False, "status": "nothing_to_withdraw"}
        if len(pos) > 1:
            q = ", ".join(f"{fmt(p['amount'])} {sym} on {CHAIN_NAME[p['chain']]}" for p in pos)
            return {"note": "Nothing was prepared. Ask the user which one.", "prepared": False,
                    "question": f"Which deposit? {q}.", "status": "needs_choice"}
        p = pos[0]
        if amount is not None:
            amount = float(amount)
            if amount <= 0:
                return {"error": "amount must be positive"}
            if amount > p["amount"] + 1e-12:
                return {"note": f"Only {fmt(p['amount'])} {sym} is in Aave on {CHAIN_NAME[p['chain']]}. Tell the user.",
                        "prepared": False, "status": "insufficient", "supplied": f"{fmt(p['amount'])} {sym}"}
        return {"status": "review_opened", "kind": "withdraw", "executed": False,
                "note": f"Waiting for the user to approve with {self.bio}."}


PROTO = [("Aave v3", {"Ethereum": ["USDC", "USDT", "DAI", "ETH"], "Base": ["USDC", "ETH"], "Arbitrum": ["USDC", "USDT", "ETH"]}),
         ("Morpho", {"Ethereum": ["USDC", "ETH", "DAI"], "Base": ["USDC", "ETH"]}),
         ("Compound v3", {"Ethereum": ["USDC", "ETH"], "Base": ["USDC", "ETH"], "Arbitrum": ["USDC"]}),
         ("Spark", {"Ethereum": ["DAI", "USDC"]}), ("Fluid", {"Ethereum": ["USDC", "ETH"], "Arbitrum": ["USDC"]}),
         ("Kamino", {"Solana": ["USDC"]}), ("Lido", {"Ethereum": ["ETH"]}), ("Pendle", {"Ethereum": ["USDC"], "Arbitrum": ["ETH"]})]
BASE_APY = {"USDC": 4.2, "USDT": 4.0, "DAI": 4.4, "ETH": 2.1}


def make_pools(rng):
    rows = []
    for proto, chains in PROTO:
        for ch, assets in chains.items():
            for a in assets:
                lo, hi = (0.6, 1.5) if proto == "Aave v3" else (0.7, 2.1)
                apy = int(BASE_APY[a] * rng.uniform(lo, hi) * 100) / 100
                if round(apy * 10, 6) == int(round(apy * 10, 6)):
                    apy = round(apy + 0.03, 2)            # never a round number: the text must repeat 2 decimals
                tvl = int(round(rng.uniform(2e7, 9.5e8) if proto == "Aave v3" else rng.uniform(4e6, 4e8), -3))
                rows.append({"apy_pct": apy, "asset": a, "chain": ch, "protocol": proto, "tvl_usd": tvl})
    return rows


DISTRACT = [("SOL", "solana", 0.2, 12), ("BNB", "bnb", 0.05, 3), ("POL", "polygon", 20, 900), ("USDT", "polygon", 10, 400),
            ("USDC", "solana", 10, 500), ("ETH", "base", 0.01, 0.6), ("USDC", "polygon", 10, 700), ("ETH", "arbitrum", 0.01, 0.5),
            ("ETH", "ethereum", 0.01, 0.8), ("USDT", "ethereum", 10, 500), ("DAI", "ethereum", 10, 500)]


def new_world(rng, bio):
    w = World3(rng)
    w.bio = bio
    w.block = None
    w.balances = []
    w.aave = []
    w.pools = make_pools(rng)
    return w


def set_balances(w, rng, held, focus=()):
    """held = [(symbol, chain, amount)]. Distractors never share a symbol with `focus` and never sit on the same (symbol, chain)."""
    w.balances = []
    for sym, ch, amt in held:
        w.balances.append({"symbol": sym, "chain": ch, "amount": amt, "usd": round(amt * w.prices[sym], 2)})
    taken = {(s, c) for s, c, _ in held}
    for sym, ch, lo, hi in rng.sample(DISTRACT, rng.choice([0, 1, 1, 2, 3])):
        if sym in focus or (sym, ch) in taken:
            continue
        amt = round(rng.uniform(lo, hi), 2 if hi > 5 else 4)
        if amt <= 0:
            continue
        taken.add((sym, ch))
        w.balances.append({"symbol": sym, "chain": ch, "amount": amt, "usd": round(amt * w.prices[sym], 2)})
    rng.shuffle(w.balances)


def set_aave(w, rng, positions, extra_other=True):
    w.aave = [{"symbol": s, "chain": c, "amount": a} for s, c, a in positions]
    if extra_other and rng.random() < 0.3:
        s = rng.choice(["DAI", "USDT", "ETH"])
        if not any(p["symbol"] == s for p in w.aave):
            c = rng.choice(SYM_CHAINS[s])
            w.aave.append({"symbol": s, "chain": c, "amount": round(rng.uniform(5, 300), 2) if s != "ETH" else round(rng.uniform(0.02, 1.5), 4)})


# ------------------------------------------------------------------------ user text generation
WRAPS = ["{s}", "{s}", "{s}", "{s}", "{s} please", "{s} pls", "please {s}", "can you {s}", "can you {s}?", "could you {s}?",
         "hey {s}", "hey can you {s}", "ok {s}", "i want to {s}", "i wanna {s}", "i'd like to {s}", "lets {s}", "{s} for me",
         "{s} thanks", "{s} asap", "quick, {s}", "yo {s}", "hi, {s}", "need to {s}", "{s} when you can", "{s} real quick",
         "can u {s}", "{s} plz", "just {s}", "hmm {s}", "{s} rn"]
EV_WRAPS = ["would you {s}?", "{s}, thx", "go ahead and {s}", "i need you to {s}", "{s} now", "ok so {s}", "time to {s}",
            "alright {s}", "{s} if you can", "so {s}", "pls {s}", "{s} ty"]
LIGHT_WRAPS = ["{s}", "{s}", "{s}", "hey {s}", "ok {s}", "{s} pls", "yo {s}", "hi, {s}", "{s} thanks", "quick q: {s}", "so {s}", "hmm {s}", "{s} please"]
EV_LIGHT = ["alright {s}", "{s} thx", "ok so {s}", "one question: {s}", "{s} ty"]
VERBS = {"deposit", "put", "supply", "lend", "add", "move", "park", "stake", "send", "get", "start", "make", "invest", "withdraw", "take",
         "pull", "cash", "remove", "redeem", "unlend", "stop", "bring", "exit", "close", "free", "slide", "drop", "throw", "load", "commit",
         "lock", "yank", "undo", "split", "grow", "earn", "swap", "sell", "convert", "trade", "bridge"}
TYPOS = {"deposit": ["depsoit", "deposti", "diposit"], "withdraw": ["widthdraw", "withdrawl", "withdaw"],
         "please": ["plz", "pleae"], "supply": ["suply"], "everything": ["everyting"], "aave": ["aavee", "aave"]}


def dress(rng, ev, stem):
    raw = stem.startswith("=")
    if raw:
        stem = stem[1:]
        t = stem
    else:
        if stem.split(" ")[0] in VERBS:
            pool, evp = WRAPS, EV_WRAPS
        else:                                   # questions and "i want ..." stems only take light wrappers
            pool, evp = LIGHT_WRAPS, EV_LIGHT
        t = (rng.choice(evp) if ev and rng.random() < 0.4 else rng.choice(pool)).format(s=stem)
    if rng.random() < 0.07:
        for k, v in TYPOS.items():
            if re.search(r"\b" + k + r"\b", t):
                t = re.sub(r"\b" + k + r"\b", rng.choice(v), t, count=1)
                break
    r = rng.random()
    if r < 0.38:
        t = t.lower() if not re.search(r"[A-Z]{3,}", t) else t
    elif r < 0.82:
        t = t[:1].upper() + t[1:]
    if not t.endswith("?"):
        r = rng.random()
        t += "." if r < 0.12 else ("!" if r < 0.16 else "")
    return t


def pick_stem(rng, ev, stems, extra):
    if ev and extra and rng.random() < 0.4:
        return rng.choice(extra)
    return rng.choice(stems)


def aave_word(rng):
    return rng.choice(["aave", "aave", "Aave", "Aave", "aave v3", "Aave v3", "AAVE"])


def sym_word(rng, sym):
    return rng.choice(SYM_WORDS[sym])


def chain_phrase(rng, chain):
    n = rng.choice(CHAIN_WORDS[chain])
    return rng.choice([f" on {n}", f" on {n}", f" over on {n}", f" ({n})", f" in {n}", f" {n}", f" using {n}", f" via {n}"]), n


def rand_sym(rng, weights=(("USDC", 0.64), ("ETH", 0.16), ("USDT", 0.12), ("DAI", 0.08))):
    r, acc = rng.random(), 0
    for s, wt in weights:
        acc += wt
        if r < acc:
            return s
    return weights[0][0]


ETH_AMOUNTS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 8, 10]
USD_EXTRA = [35, 60, 80, 120, 175, 350, 600, 800, 1200, 2000, 3500, 12.5, 37.5, 99.99, 42, 15, 8, 5, 45, 90, 130, 1800]


def rand_amt(rng, sym):
    if sym == "ETH":
        return rng.choice(ETH_AMOUNTS)
    r = rng.random()
    if r < 0.55:
        return rng.choice([10, 20, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2500, 5000])
    if r < 0.9:
        return rng.choice(USD_EXTRA)
    return round(rng.uniform(5, 900), 2)


def amt_forms(rng, sym, x):
    """(plain amount text, dollar-ish text for 'of' templates)."""
    t, _ = amt_text(rng, x)
    au = ("$" + t) if sym in STABLE and rng.random() < 0.7 else t
    return t, au


def hold_amount(rng, sym, need):
    dp = 4 if sym == "ETH" else 2
    return round(need * rng.uniform(1.1, 4) + (0.05 if sym == "ETH" else 5), dp)


def rand_supplied(rng, sym):
    return round(rng.uniform(0.05, 3.2), 4) if sym == "ETH" else round(rng.uniform(8, 4200), 2)


DEP_STEMS = [
    "deposit {a} {s} into {A}{c}", "deposit {a} {s} in {A}{c}", "deposit {a} {s} to {A}{c}", "put {a} {s} into {A}{c}",
    "put {a} {s} in {A}{c}", "supply {a} {s} to {A}{c}", "supply {a} {s} on {A}{c}", "lend {a} {s} on {A}{c}",
    "lend out {a} {s} on {A}{c}", "add {a} {s} to {A}{c}", "move {a} {s} into {A}{c}", "park {a} {s} in {A}{c}",
    "stake {a} {s} in {A}{c}", "send {a} {s} to {A}{c}", "put {a} {s} to work{c}", "put {a} {s} to work in {A}{c}",
    "get {a} {s} earning in {A}{c}", "start lending {a} {s} on {A}{c}", "deposit {au} of {s} into {A}{c}",
    "put {au} of {s} in {A}{c}", "supply {au} {s} to {A}{c}", "=aave deposit {a} {s}{c}", "=deposit {a} {s}{c} into {A}",
    "={a} {s} into {A}{c}", "=supply {s} {a} to {A}{c}", "make {a} {s} earn in {A}{c}", "invest {a} {s} in {A}{c}",
    "put some {s} in {A}{c}: {a}", "deposit {a} {s} into the {A} pool{c}", "supply {a} {s} to {A} v3{c}"]
DEP_EXTRA = ["slide {a} {s} into {A}{c}", "drop {a} {s} in {A}{c}", "throw {a} {s} into {A}{c}", "load {a} {s} into {A}{c}",
             "commit {a} {s} to {A}{c}", "=can u deposit {a} {s} to {A}{c}", "={A}: {a} {s} in{c}", "lock {a} {s} in {A}{c} so it earns",
             "put {au} {s} to work on {A}{c}"]

ALL_STEMS = ["deposit all my {s} into {A}", "put all my {s} in {A}", "deposit everything i have in {s} to {A}", "supply my whole {s} balance to {A}",
             "deposit my entire {s} balance into {A}", "put all of my {s} to work", "lend all my {s} on {A}", "deposit all {s} into {A}",
             "move all my {s} into {A}", "=max deposit {s} into {A}", "stake all my {s} in {A}", "put my full {s} balance in {A}",
             "supply all my {s} to {A}", "=deposit all my {s}", "get all my {s} earning in {A}", "deposit the whole lot of {s} in {A}"]
ALL_EXTRA = ["send all my {s} into {A}", "=all my {s} into {A}", "put every last {s} into {A}"]

FRAC_STEMS = [("half", 2, "deposit half of my {s} into {A}"), ("half", 2, "put half my {s} in {A}"), ("half", 2, "supply half my {s} to {A}"),
              ("half", 2, "deposit 50% of my {s} into {A}"), ("half", 2, "=deposit half my {s}"), ("half", 2, "lend half of my {s} on {A}"),
              ("half", 2, "put half of my {s} to work"), ("half", 2, "move half my {s} into {A}"),
              ("quarter", 4, "put a quarter of my {s} into {A}"), ("quarter", 4, "deposit 25% of my {s} into {A}"),
              ("quarter", 4, "supply a quarter of my {s} to {A}"), ("quarter", 4, "=deposit a quarter of my {s}")]
FRAC_EXTRA = [("half", 2, "split my {s} in two and put one half in {A}"), ("quarter", 4, "put 1/4 of my {s} in {A}")]

EARN_STEMS = ["earn on my {s}", "i want to earn on my {s}", "make my {s} earn something", "put my {s} to work", "can my {s} earn interest",
              "where can i earn on my {s}", "=earn on {s}", "get some yield on my {s}", "my {s} is sitting idle, earn something with it",
              "=any way to earn on my {s}?", "what can my {s} earn", "earn with my {s}", "i have {s} doing nothing, earn", "grow my {s}"]
EARN_EXTRA = ["=how do i make my {s} earn", "make my idle {s} work", "yield on my {s}?"]
EARN_YES_AMT = ["yes, {a}", "ok do {a}", "yeah put {a} in", "sure, deposit {a}", "yes please, {a} {s}", "go for it with {a}", "ok {a} {s}", "yep {a}"]
EARN_YES = ["yes", "yes please", "sure", "ok do it", "yeah", "go ahead", "yep let's do it", "sounds good"]
AMT_REPLY = ["{a}", "{a} {s}", "{au}", "make it {a}", "{a} please", "just {a}", "{a} {s} pls", "let's say {a}"]

OTHER_PROTOS = [("Morpho", "morpho", ["USDC", "ETH"]), ("Pendle", "pendle", ["USDC", "ETH"]), ("Lido", "lido", ["ETH"]),
                ("Compound", "compound", ["USDC", "ETH"]), ("Spark", "spark", ["DAI", "USDC"]), ("Fluid", "fluid", ["USDC", "ETH"]),
                ("Yearn", "yearn", ["USDC", "ETH"]), ("Kamino", "kamino", ["USDC"]), ("Ethena", "ethena", ["USDe"])]
OTHER_STEMS = ["deposit {a} {s} into {P}", "put {a} {s} in {P}", "supply {a} {s} to {P}", "lend {a} {s} on {P}",
               "stake {a} {s} on {P}", "deposit {a} {s} to {P}", "={P} deposit {a} {s}", "move {a} {s} into {P}", "get {a} {s} into {P}"]

WD_ALL = ["withdraw my {s} from {A}{c}", "take my {s} out of {A}{c}", "pull my {s} out of {A}{c}", "get my {s} out of {A}{c}",
          "get my {s} back from {A}{c}", "withdraw all my {s} from {A}{c}", "withdraw all {s} from {A}{c}", "cash out my {s} from {A}{c}",
          "remove my {s} from {A}{c}", "take out my {s} from {A}{c}", "i want my {s} back from {A}{c}", "close my {A} {s} position{c}",
          "exit my {s} position in {A}{c}", "redeem my {s} from {A}{c}", "bring my {s} back from {A}{c}", "pull all my {s} out of {A}{c}",
          "stop lending my {s} on {A}{c}", "withdraw the {s} i put in {A}{c}", "withdraw my whole {s} balance from {A}{c}",
          "take all my {s} out of {A}{c}", "=aave withdraw {s}{c}", "=withdraw {s} from {A}{c}", "=my {s} out of {A}{c}",
          "=all my {s} out of {A}{c}", "withdraw everything {s} from {A}{c}", "pull everything {s} out of {A}{c}",
          "get my {s} out of the {A} pool{c}", "unlend my {s} from {A}{c}", "withdraw {s} from {A}{c}"]
WD_ALL_EXTRA = ["undo my {s} deposit in {A}{c}", "yank my {s} out of {A}{c}", "=need my {s} back from {A}{c}", "take my {s} home from {A}{c}",
                "=stop earning on my {s} in {A}{c}"]
WD_AMT = ["withdraw {a} {s} from {A}{c}", "take {a} {s} out of {A}{c}", "pull {a} {s} from {A}{c}", "pull {a} {s} out of {A}{c}",
          "get {a} {s} out of {A}{c}", "remove {a} {s} from {A}{c}", "withdraw {au} of {s} from {A}{c}", "take out {a} {s} from {A}{c}",
          "=aave withdraw {a} {s}{c}", "=withdraw {a} {s}{c}", "cash out {a} {s} from {A}{c}", "redeem {a} {s} from {A}{c}",
          "=take {au} of {s} out of {A}{c}"]
WD_AMT_EXTRA = ["free up {a} {s} from {A}{c}", "bring {a} {s} back from {A}{c}", "={a} {s} out of {A}{c}"]
WD_ASK = ["pull everything from {A}", "get my money out", "withdraw from {A}", "take my money out of {A}", "i want to withdraw",
          "withdraw everything", "cash out of {A}", "get my funds out of {A}", "take it all out of {A}", "pull my money out of {A}",
          "withdraw all from {A}", "get everything out of {A}", "=withdraw", "=aave withdraw", "get my money out of {A}",
          "take everything out of {A}", "i need my money out"]
WD_ASK_EXTRA = ["=out of {A} please, all of it", "need to exit {A}", "get me out of {A}"]
WD_ASK_AMT = ["withdraw {a} from {A}", "take {a} out of {A}", "pull {au} out of {A}", "get {au} out of {A}", "=withdraw {au}",
              "take {a} out of my {A}", "remove {au} from {A}"]
WD_PART = ["withdraw half of my {s} from {A}{c}", "take out half my {s} from {A}{c}", "pull a quarter of my {s} out of {A}{c}",
           "withdraw some of my {s} from {A}{c}", "take a bit of my {s} out of {A}{c}", "get part of my {s} out of {A}{c}",
           "withdraw half my {s} from {A}{c}", "=half my {s} out of {A}{c}", "take part of my {s} out of {A}{c}",
           "pull out half of my {s} from {A}{c}", "withdraw a portion of my {s} from {A}{c}", "take out some {s} from {A}{c}"]
WD_PART_EXTRA = ["=just some of my {s} out of {A}{c}", "withdraw a third of my {s} from {A}{c}"]
TOKEN_REPLY = ["{s}", "{s}", "my {s}", "the {s}", "{s} pls", "{s} one", "{s}, all of it", "{s} thanks", "it's {s}", "{s} please"]
CHAIN_REPLY = ["{n}", "{n}", "the {n} one", "on {n}", "{n} pls", "{n}, all of it", "{n} please", "the one on {n}"]
CHAIN_REPLY_AMT = ["{n}", "the {n} one", "on {n}", "{n} pls", "{n} please", "the one on {n}"]

FOLLOW_DEP = ["did it go through?", "is it done?", "ok approved, is it in aave now?", "so it's in aave now?", "it's the right one, so my money is in aave?",
              "approved. all good?", "did that work", "is the deposit complete?", "confirmed on my side, did it go through", "so is it deposited?",
              "done?", "it's the right one so the funds are earning now?"]
FOLLOW_WD = ["did it go through?", "is it done?", "ok approved, is my money back in my wallet?", "did the withdrawal finish?",
             "it's the right one, so the funds are back?", "approved. all good?", "did that work", "so did it come out?", "is it complete?",
             "confirmed on my side, did it go through", "is it back in my wallet now?"]
FOLLOW_ANS = ["I can't see approvals from here, so I don't know if you've approved it. The review screen is where you approve it with {BIO}, and Activity in Wallets shows what actually happened.",
              "I can't tell from here. Nothing happens until you approve it with {BIO} on the review screen. Once you have, Activity in Wallets shows the result.",
              "All I know is that I prepared it. I don't see approvals or results, so check Activity in Wallets to see what happened.",
              "I can't see that. If you approved it with {BIO}, Activity in Wallets shows the result. If not, it's still waiting on the review screen."]


def say_pool(rng, stems):
    return rng.choice(stems)


# ------------------------------------------------------------------------ answer texts
def dep_ok_text(rng, r, bio):
    amt, pool, apy = r["amount"], r["pool"], f"{r['apy_pct']:g}"
    return rng.choice([
        "I've prepared a deposit of {amt} into {pool} at {apy}% APY, which can change. Review it and approve with {bio}.",
        "I've prepared a deposit of {amt} into {pool} at {apy}% APY, which can change. Review it and approve with {bio}.",
        "Your deposit of {amt} into {pool} is ready to review. The APY is {apy}% right now and can change. Approve it with {bio} if it looks right.",
        "I've set up {amt} going into {pool} at {apy}% APY. That rate is variable. Open the review and approve with {bio} to go ahead.",
        "Prepared: {amt} into {pool}, {apy}% APY, which can change. Nothing happens until you approve it with {bio}.",
    ]).format(amt=amt, pool=pool, apy=apy, bio=bio)


def wd_ok_text(rng, sym, bio, amount=None, chain=None):
    on = f" on {CHAIN_NAME[chain]}" if chain else ""
    if amount is None:
        return rng.choice([
            "I've prepared a withdrawal of all your {s} from Aave{on}. Review it and approve with {bio}. Tell me an amount if you want less.",
            "I've prepared a withdrawal of all your {s} from Aave{on}. Review it and approve with {bio}. Tell me an amount if you want less.",
            "Your withdrawal of all your {s} from Aave{on} is ready to review. Approve it with {bio} when you're ready. Tell me an amount if you want less.",
            "I've opened the withdrawal for all of your {s} in Aave{on}. Review it and approve with {bio}. Tell me an amount if you want less.",
            "Prepared: a withdrawal of all your {s} from Aave{on}. Nothing moves until you approve it with {bio}. If you want less, tell me an amount.",
        ]).format(s=sym, on=on, bio=bio)
    return rng.choice([
        "I've prepared a withdrawal of {a} {s} from Aave{on}. Review it and approve with {bio}.",
        "I've prepared a withdrawal of {a} {s} from Aave{on}. Review it and approve with {bio}.",
        "Your withdrawal of {a} {s} from Aave{on} is ready to review. Approve it with {bio} when you're ready.",
        "Prepared: {a} {s} out of Aave{on}. Nothing moves until you approve it with {bio}.",
    ]).format(a=fmt(amount), s=sym, on=on, bio=bio)


def err_text(rng, r):
    return rng.choice([f"I couldn't prepare that, it hit an error: {r['error']}. Ask me again in a moment and I'll retry.",
                       f"That didn't work: {r['error']}. Try again in a moment and I'll prepare it again."])


def blocked_text(rng, r, what):
    reason = r["reasons"][0]
    return rng.choice([f"I couldn't prepare that {what}. {reason} Nothing was prepared.",
                       f"I can't prepare that {what} right now. {reason} Nothing was prepared, so nothing will happen."])


def insuff_dep_text(rng, r, amount, sym):
    return rng.choice([
        f"You only hold {r['holding']}, which is less than the {fmt(amount)} {sym} you asked for. Want me to prepare a smaller amount?",
        f"I can't prepare {fmt(amount)} {sym}, you hold {r['holding']}. Want me to deposit what you have, or a smaller amount?"])


def needs_chain_text(rng, r):
    q = r["question"].replace("Which network should the deposit use? You hold", "Which network? You hold")
    return q if rng.random() < 0.6 else q.replace("Which network? You hold", "Which network should I use? You hold")


def choice_text(rng, r):
    m = re.match(r"Which deposit\? (.+?)\.?$", r["question"])
    parts = m.group(1).split(", ")
    return "Which one, " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]) + "?"


def yield_text(rng, rows, sym, offer):
    best = rows[0]
    aave = next((p for p in rows if p["protocol"] == "Aave v3"), None)
    tail = ["APYs are variable.", "These rates can change.", "APY moves around, so treat it as a snapshot."]
    if aave is best:
        s = f"Aave v3 on {best['chain']} pays the most for {sym} right now, {best['apy_pct']:g}% APY. {rng.choice(tail)}"
    elif aave:
        s = (f"The best {sym} rate I see is {best['protocol']} on {best['chain']} at {best['apy_pct']:g}%. "
             f"I can only prepare Aave v3 deposits, and Aave v3 on {aave['chain']} pays {aave['apy_pct']:g}%. {rng.choice(tail)}")
    else:
        s = (f"The best {sym} rate I see is {best['protocol']} on {best['chain']} at {best['apy_pct']:g}%. "
             f"I can only prepare Aave v3 deposits. {rng.choice(tail)}")
    if offer:
        s += " " + rng.choice(["Want me to prepare an Aave deposit? Tell me how much.", "Want me to deposit into Aave? Tell me an amount.",
                               "Should I prepare an Aave v3 deposit for you? How much?"])
    return s


# ------------------------------------------------------------------------ scenario plumbing
def begin(rng, bio=None):
    bio = bio or ("Face ID" if rng.random() < 0.7 else "Touch ID")
    w = new_world(rng, bio)
    return w, bio


def conv(rng, w, bio):
    return Conv(rng, w, system=SYS_TEMPLATE.replace("{BIO}", bio))


def fin(c, kind, family, bio, calls, **extra):
    ex = c.ex(kind)
    ex["meta"].update({"source": "aave", "family": family, "bio": bio,
                       "intent": {"calls": [[n, a] for n, a in calls], **extra}})
    return ex


def dep_args(sym, amount, chain=None, project="aave-v3"):
    a = {"project": project, "symbol": sym}
    if chain:
        a["chain"] = chain
    a["amount"] = num(amount)
    return a


def wd_args(sym, chain=None, amount=None):
    a = {"symbol": sym}
    if chain:
        a["chain"] = chain
    if amount is not None:
        a["amount"] = num(amount)
    return a


def user_dep(rng, ev, sym, amt, chain=None, stems=DEP_STEMS, extra=DEP_EXTRA, **kw):
    a, au = amt_forms(rng, sym, amt)
    cp, _ = chain_phrase(rng, chain) if chain else ("", None)
    st = pick_stem(rng, ev, stems, extra)
    return dress(rng, ev, st.format(a=a, au=au, s=sym_word(rng, sym), A=aave_word(rng), c=cp, **kw))


def setup_single(rng, w, sym, need, chain, focus=None):
    ch = chain or rng.choice(SYM_CHAINS[sym])
    set_balances(w, rng, [(sym, ch, hold_amount(rng, sym, need))], focus=(sym,))
    return ch


# ---------------------------------------------------------------------- deposit scenarios
def deposit_finish(rng, c, w, bio, sym, amount, chain, r, kind, calls):
    if "error" in r:
        c.say(err_text(rng, r))
    elif r["status"] == "review_opened":
        c.say(dep_ok_text(rng, r, bio))
    else:
        raise AssertionError(r)


def sc_dep_direct(rng, ev, chain_named=False):
    w, bio = begin(rng)
    sym = rand_sym(rng)
    amt = rand_amt(rng, sym)
    hold_chain = rng.choice(SYM_CHAINS[sym])
    named = hold_chain if chain_named else None
    setup_single(rng, w, sym, amt, hold_chain)
    c = conv(rng, w, bio)
    c.user(user_dep(rng, ev, sym, amt, named))
    args = dep_args(sym, amt, named)
    r = c.call(("prepare_deposit", args))
    deposit_finish(rng, c, w, bio, sym, amt, named, r, "dep", None)
    return fin(c, "dep_chain" if chain_named else "dep_direct", "deposit", bio, [("prepare_deposit", args)], no_ask=True, outcome=r.get("status"))


def sc_dep_chain(rng, ev):
    return sc_dep_direct(rng, ev, chain_named=True)


def sc_dep_error(rng, ev):
    w, bio = begin(rng)
    sym = rand_sym(rng)
    amt = rand_amt(rng, sym)
    ch = setup_single(rng, w, sym, amt, None)
    w.fail_next = "prepare_deposit"
    c = conv(rng, w, bio)
    c.user(user_dep(rng, ev, sym, amt))
    args = dep_args(sym, amt)
    r = c.call(("prepare_deposit", args))
    c.say(err_text(rng, r))
    return fin(c, "dep_error", "deposit", bio, [("prepare_deposit", args)], no_ask=True, outcome="error")


def sc_dep_blocked(rng, ev):
    w, bio = begin(rng)
    sym = rand_sym(rng)
    amt = rand_amt(rng, sym)
    named = rng.choice(SYM_CHAINS[sym]) if rng.random() < 0.3 else None
    setup_single(rng, w, sym, amt, named)
    w.block = rng.choice(BLOCK_REASONS)
    c = conv(rng, w, bio)
    c.user(user_dep(rng, ev, sym, amt, named))
    args = dep_args(sym, amt, named)
    r = c.call(("prepare_deposit", args))
    assert r["status"] == "blocked"
    c.say(blocked_text(rng, r, "deposit"))
    return fin(c, "dep_blocked", "deposit", bio, [("prepare_deposit", args)], no_ask=True, outcome="blocked")


def sc_dep_insuff(rng, ev):
    w, bio = begin(rng)
    sym = rng.choice(["USDC", "USDC", "USDC", "ETH", "USDT"])
    ch = rng.choice(SYM_CHAINS[sym])
    have = round(rng.uniform(0.01, 0.4), 4) if sym == "ETH" else round(rng.uniform(5, 180), 2)
    need = rng.choice([x for x in (ETH_AMOUNTS if sym == "ETH" else [200, 250, 300, 500, 750, 1000, 2000, 5000]) if x > have * 1.4])
    set_balances(w, rng, [(sym, ch, have)], focus=(sym,))
    named = ch if rng.random() < 0.3 else None
    c = conv(rng, w, bio)
    c.user(user_dep(rng, ev, sym, need, named))
    args = dep_args(sym, need, named)
    r = c.call(("prepare_deposit", args))
    assert r["status"] == "insufficient"
    c.say(insuff_dep_text(rng, r, need, sym))
    return fin(c, "dep_insufficient", "deposit", bio, [("prepare_deposit", args)], no_ask=True, outcome="insufficient")


def sc_dep_needs_chain(rng, ev):
    w, bio = begin(rng)
    sym = rng.choice(["USDC", "USDC", "USDC", "ETH"])
    chs = rng.sample(SYM_CHAINS[sym], 2)
    amt = rand_amt(rng, sym)
    set_balances(w, rng, [(sym, ch, hold_amount(rng, sym, amt)) for ch in chs], focus=(sym,))
    c = conv(rng, w, bio)
    c.user(user_dep(rng, ev, sym, amt))
    a1 = dep_args(sym, amt)
    r = c.call(("prepare_deposit", a1))
    assert r["status"] == "needs_chain"
    c.say(needs_chain_text(rng, r), ask=True)
    pick = rng.choice(chs)
    c.user(dress(rng, ev, "=" + rng.choice(CHAIN_REPLY_AMT).format(n=rng.choice(CHAIN_WORDS[pick]))))
    a2 = dep_args(sym, amt, pick)
    r2 = c.call(("prepare_deposit", a2))
    assert r2["status"] == "review_opened"
    c.say(dep_ok_text(rng, r2, bio))
    return fin(c, "dep_needs_chain", "deposit", bio, [("prepare_deposit", a1), ("prepare_deposit", a2)], outcome="review_opened")


def sc_dep_unsup_chain(rng, ev):
    w, bio = begin(rng)
    sym = rand_sym(rng)
    amt = rand_amt(rng, sym)
    setup_single(rng, w, sym, amt, None)
    disp, words = rng.choice(OTHER_CHAINS)
    word = rng.choice(words)
    a, au = amt_forms(rng, sym, amt)
    st = rng.choice(["deposit {a} {s} into {A} on {w}", "put {a} {s} in {A} on {w}", "supply {a} {s} to {A} {w}", "lend {a} {s} on {A} {w}",
                     "=deposit {a} {s} into {A} on {w}", "=aave on {w}: deposit {a} {s}", "put {au} of {s} to work on {w} in {A}"])
    c = conv(rng, w, bio)
    c.user(dress(rng, ev, st.format(a=a, au=au, s=sym_word(rng, sym), A=aave_word(rng), w=word)))
    c.say(rng.choice([
        f"Aave deposits only work on Ethereum, Base and Arbitrum here, so I can't prepare one on {disp}. Pick one of those three and I'll prepare it.",
        f"I can only prepare Aave v3 deposits on Ethereum, Base or Arbitrum, not {disp}. Tell me which of those you want.",
        f"Aave v3 is only available on Ethereum, Base and Arbitrum in Hermit, so {disp} won't work. Which of those should I use?"]))
    return fin(c, "dep_unsupported_chain", "deposit", bio, [], no_ask=False, outcome="no_call", other_chain=disp)


def sc_dep_other(rng, ev):
    w, bio = begin(rng)
    name, slug, syms = rng.choice(OTHER_PROTOS)
    sym = rng.choice(syms)
    amt = rand_amt(rng, sym if sym in ("ETH", "USDC", "USDT", "DAI") else "USDC")
    if sym in ("ETH", "USDC", "USDT", "DAI"):
        setup_single(rng, w, sym, amt, None)
    else:
        set_balances(w, rng, [(sym, "ethereum", hold_amount(rng, "USDC", amt))], focus=(sym,))
        w.prices.setdefault(sym, 1.0)
    c = conv(rng, w, bio)
    a, au = amt_forms(rng, sym if sym in STABLE or sym == "ETH" else "USDC", amt)
    c.user(dress(rng, ev, rng.choice(OTHER_STEMS).format(a=a, s=sym_word(rng, sym) if sym in SYM_WORDS else sym, P=name)))
    args = dep_args(sym, amt, None, project=slug)
    r = c.call(("prepare_deposit", args))
    assert r["status"] == "unsupported", r
    c.say(rng.choice([
        f"Aave v3 is the only pool I can prepare a deposit for. {name} and the other pools are in Earn, DeFi. I can prepare the same amount into Aave v3 if you like.",
        f"I can only prepare deposits into Aave v3. {name} and the other pools are in Earn, DeFi. Want me to prepare an Aave v3 deposit instead?",
        f"Aave v3 is the only pool I can prepare. {name} is in Earn, DeFi along with the others. Say the word and I'll prepare an Aave v3 deposit."]))
    return fin(c, "dep_other_protocol", "deposit", bio, [("prepare_deposit", args)], no_ask=False, outcome="unsupported", other=name)


def sc_dep_all(rng, ev):
    w, bio = begin(rng)
    sym = rng.choice(["USDC"] * 6 + ["USDT", "DAI"])
    empty = rng.random() < 0.1
    ch = rng.choice(SYM_CHAINS[sym])
    cents = rng.choice([rng.randint(800, 60000), rng.randint(60000, 4000000)])
    amt = round(cents / 100, 2)
    set_balances(w, rng, [] if empty else [(sym, ch, amt)], focus=(sym,))
    c = conv(rng, w, bio)
    c.user(dress(rng, ev, pick_stem(rng, ev, ALL_STEMS, ALL_EXTRA).format(s=sym_word(rng, sym), A=aave_word(rng))))
    rb = c.call(("get_balances", {}))
    h = next((x for x in rb["holdings"] if x["symbol"] == sym and x["chain"] in AAVE_CHAINS), None)
    if h is None:
        c.say(rng.choice([f"I don't see any {sym} in your wallet, so there's nothing to deposit. You can add funds from Wallets > Add funds.",
                          f"Your wallet has no {sym} right now, so I can't prepare a deposit. Add some from Wallets > Add funds first."]))
        return fin(c, "dep_all_empty", "deposit", bio, [], no_ask=True, first_tool="get_balances", outcome="no_holding")
    amount = num(h["amount"])
    args = dep_args(sym, amount)
    r = c.call(("prepare_deposit", args))
    c.say(dep_ok_text(rng, r, bio)) if r.get("status") == "review_opened" else c.say(err_text(rng, r))
    return fin(c, "dep_all", "deposit", bio, [("prepare_deposit", args)], no_ask=True, first_tool="get_balances", outcome=r.get("status"))


def sc_dep_frac(rng, ev):
    w, bio = begin(rng)
    sym = rng.choice(["USDC"] * 6 + ["USDT", "ETH", "ETH"])
    word, div, stem = rng.choice(FRAC_STEMS + (FRAC_EXTRA if ev and rng.random() < 0.4 else []))
    ch = rng.choice(SYM_CHAINS[sym])
    if sym == "ETH":
        amt = rng.randint(1, 4000) * 4 / 10000 + 0.04
        amt = round(amt, 4)
    else:
        amt = round(rng.randint(40, 600000) * 4 / 100 + 0.04, 2)
    set_balances(w, rng, [(sym, ch, amt)], focus=(sym,))
    c = conv(rng, w, bio)
    c.user(dress(rng, ev, stem.format(s=sym_word(rng, sym), A=aave_word(rng))))
    rb = c.call(("get_balances", {}))
    h = next(x for x in rb["holdings"] if x["symbol"] == sym and x["chain"] in AAVE_CHAINS)
    expr = rng.choice([f"{h['amount']} / {div}", f"{h['amount']} * {1 / div:g}"])
    rc = c.call(("calculator", {"expression": expr}))
    amount = num(rc["result"])
    args = dep_args(sym, amount)
    r = c.call(("prepare_deposit", args))
    assert r["status"] == "review_opened", r
    c.say(dep_ok_text(rng, r, bio))
    return fin(c, "dep_fraction", "deposit", bio, [("prepare_deposit", args)], no_ask=True, first_tool="get_balances", outcome="review_opened")


def sc_dep_earn(rng, ev):
    w, bio = begin(rng)
    sym = rng.choice(["USDC"] * 7 + ["USDT", "ETH"])
    # pick the chain of the best Aave row so the offer matches where the user holds the token
    rows = sorted([p for p in w.pools if p["asset"] == sym], key=lambda p: -p["apy_pct"])[:5]
    aave_rows = [p for p in rows if p["protocol"] == "Aave v3"]
    hold_chain = aave_rows[0]["chain"].lower() if aave_rows else rng.choice(SYM_CHAINS[sym])
    mode = rng.random()
    amt = rand_amt(rng, sym)
    set_balances(w, rng, [(sym, hold_chain, hold_amount(rng, sym, amt))], focus=(sym,))
    c = conv(rng, w, bio)
    c.user(dress(rng, ev, pick_stem(rng, ev, EARN_STEMS, EARN_EXTRA).format(s=sym_word(rng, sym))))
    ya = {"asset": sym}
    ry = c.call(("get_yields", ya))
    c.say(yield_text(rng, ry["pools"], sym, offer=True))
    if mode < 0.4:
        return fin(c, "dep_earn_offer", "deposit", bio, [], no_ask=True, first_tool="get_yields", outcome="offer")
    a, au = amt_forms(rng, sym, amt)
    if mode < 0.75:
        c.user(dress(rng, ev, "=" + rng.choice(EARN_YES_AMT).format(a=a, s=sym_word(rng, sym))))
    else:
        c.user(dress(rng, ev, "=" + rng.choice(EARN_YES)))
        c.say(rng.choice([f"How much {sym} should I deposit?", f"How much {sym} do you want to put in?"]), ask=True)
        c.user(dress(rng, ev, "=" + rng.choice(AMT_REPLY).format(a=a, au=au, s=sym_word(rng, sym))))
    args = dep_args(sym, amt)
    r = c.call(("prepare_deposit", args))
    c.say(dep_ok_text(rng, r, bio)) if r.get("status") == "review_opened" else c.say(err_text(rng, r))
    return fin(c, "dep_earn_agree", "deposit", bio, [("prepare_deposit", args)], no_ask=True, first_tool="get_yields", outcome=r.get("status"),
               earlier_get_yields=True)


def sc_dep_follow(rng, ev):
    w, bio = begin(rng)
    sym = rand_sym(rng)
    amt = rand_amt(rng, sym)
    setup_single(rng, w, sym, amt, None)
    c = conv(rng, w, bio)
    c.user(user_dep(rng, ev, sym, amt))
    args = dep_args(sym, amt)
    r = c.call(("prepare_deposit", args))
    assert r["status"] == "review_opened"
    c.say(dep_ok_text(rng, r, bio))
    c.user(dress(rng, ev, "=" + rng.choice(FOLLOW_DEP)))
    c.say(rng.choice(FOLLOW_ANS).format(BIO=bio))
    return fin(c, "dep_followup", "deposit", bio, [("prepare_deposit", args)], no_ask=True, final_no_call=True, outcome="followup")


# ---------------------------------------------------------------------- withdraw scenarios
def wd_world(rng, sym, chain, supplied=None):
    w, bio = begin(rng)
    supplied = supplied if supplied is not None else rand_supplied(rng, sym)
    set_aave(w, rng, [(sym, chain, supplied)])
    set_balances(w, rng, [], focus=(sym,))
    return w, bio, supplied


def wd_user(rng, ev, stems, extra, sym, chain=None, amt=None, **kw):
    cp, _ = chain_phrase(rng, chain) if chain else ("", None)
    a, au = amt_forms(rng, sym, amt) if amt is not None else ("", "")
    st = pick_stem(rng, ev, stems, extra)
    return dress(rng, ev, st.format(s=sym_word(rng, sym), A=aave_word(rng), c=cp, a=a, au=au, **kw))


def wd_finish(rng, c, bio, sym, r, amount=None, chain=None):
    if "error" in r:
        c.say(err_text(rng, r)); return r.get("status", "error")
    st = r["status"]
    if st == "review_opened":
        c.say(wd_ok_text(rng, sym, bio, amount, chain))
    elif st == "nothing_to_withdraw":
        on = f" on {CHAIN_NAME[chain]}" if chain else ""
        c.say(rng.choice([f"You don't have any {sym} in Aave{on}, so there's nothing to withdraw. Earn > Positions shows what you have there.",
                          f"I don't see any {sym} in Aave{on} for you. You can check Earn > Positions to see your deposits."]))
    elif st == "insufficient":
        m = re.search(r"in Aave on (\w+)", r["note"])
        c.say(rng.choice([f"You only have {r['supplied']} in Aave on {m.group(1)}. Want me to withdraw all of it?",
                          f"I can't take out that much, only {r['supplied']} is in Aave on {m.group(1)}. Should I withdraw all of it, or a smaller amount?"]))
    elif st == "blocked":
        c.say(blocked_text(rng, r, "withdrawal"))
    else:
        raise AssertionError(r)
    return st


def sc_wd_all(rng, ev, chain_named=False):
    sym = rand_sym(rng)
    chain = rng.choice(SYM_CHAINS[sym])
    w, bio, _ = wd_world(rng, sym, chain)
    named = chain if chain_named else None
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_ALL, WD_ALL_EXTRA, sym, named))
    args = wd_args(sym, named)
    r = c.call(("prepare_withdraw", args))
    st = wd_finish(rng, c, bio, sym, r, None, named)
    return fin(c, "wd_all_chain" if chain_named else "wd_all", "withdraw", bio, [("prepare_withdraw", args)], no_ask=True, all=True, outcome=st)


def sc_wd_all_chain(rng, ev):
    return sc_wd_all(rng, ev, True)


def sc_wd_asktoken(rng, ev):
    sym = rand_sym(rng)
    chain = rng.choice(SYM_CHAINS[sym])
    w, bio, supplied = wd_world(rng, sym, chain)
    with_amt = rng.random() < 0.3
    amt = rand_amt(rng, sym) if with_amt else None
    if with_amt:
        amt = min(amt, max(round(supplied * 0.5, 2 if sym != "ETH" else 4), 0.01))
        amt = num(round(amt, 4 if sym == "ETH" else 2)) if amt else amt
    c = conv(rng, w, bio)
    if with_amt:
        a, au = amt_forms(rng, sym, amt)
        c.user(dress(rng, ev, pick_stem(rng, ev, WD_ASK_AMT, []).format(a=a, au=au, A=aave_word(rng))))
    else:
        c.user(dress(rng, ev, pick_stem(rng, ev, WD_ASK, WD_ASK_EXTRA).format(A=aave_word(rng))))
    c.say("Which token?" if rng.random() < 0.85 else rng.choice(["Which token do you want out?", "Which token should I withdraw?"]), ask=True)
    c.user(dress(rng, ev, "=" + rng.choice(TOKEN_REPLY).format(s=sym_word(rng, sym))))
    args = wd_args(sym, None, amt if with_amt else None)
    r = c.call(("prepare_withdraw", args))
    st = wd_finish(rng, c, bio, sym, r, amt if with_amt else None, None)
    return fin(c, "wd_ask_token", "withdraw", bio, [("prepare_withdraw", args)], no_ask=False, ask_first="Which token?", outcome=st)


def sc_wd_amount(rng, ev, chain_named=False):
    sym = rand_sym(rng)
    chain = rng.choice(SYM_CHAINS[sym])
    w, bio, supplied = wd_world(rng, sym, chain)
    cap = supplied * 0.9
    cands = [x for x in (ETH_AMOUNTS if sym == "ETH" else [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 500, 1000, 12.5, 35, 60]) if x <= cap]
    amt = rng.choice(cands) if cands else round(supplied / 2, 2)
    named = chain if chain_named else None
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_AMT, WD_AMT_EXTRA, sym, named, amt))
    args = wd_args(sym, named, amt)
    r = c.call(("prepare_withdraw", args))
    st = wd_finish(rng, c, bio, sym, r, amt, named)
    return fin(c, "wd_amount_chain" if chain_named else "wd_amount", "withdraw", bio, [("prepare_withdraw", args)], no_ask=True, outcome=st)


def sc_wd_amount_chain(rng, ev):
    return sc_wd_amount(rng, ev, True)


def sc_wd_part_ask(rng, ev):
    sym = rng.choice(["USDC"] * 6 + ["ETH", "USDT"])
    chain = rng.choice(SYM_CHAINS[sym])
    w, bio, supplied = wd_world(rng, sym, chain)
    named = chain if rng.random() < 0.25 else None
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_PART, WD_PART_EXTRA, sym, named))
    c.say(f"How much {sym} should I take out?" if rng.random() < 0.75 else rng.choice([f"How much {sym} do you want to take out?", f"How much {sym} should I withdraw?"]), ask=True)
    cands = [x for x in (ETH_AMOUNTS if sym == "ETH" else [5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200, 12.5, 35, 60]) if x <= supplied * 0.9]
    amt = rng.choice(cands) if cands else round(supplied / 2, 2)
    a, au = amt_forms(rng, sym, amt)
    c.user(dress(rng, ev, "=" + rng.choice(AMT_REPLY).format(a=a, au=au, s=sym_word(rng, sym))))
    args = wd_args(sym, named, amt)
    r = c.call(("prepare_withdraw", args))
    st = wd_finish(rng, c, bio, sym, r, amt, named)
    return fin(c, "wd_partial_ask", "withdraw", bio, [("prepare_withdraw", args)], no_ask=False, ask_first=f"How much {sym} should I take out?", outcome=st)


def sc_wd_nothing(rng, ev):
    sym = rand_sym(rng)
    w, bio = begin(rng)
    other = rng.choice([s for s in ["DAI", "USDT", "ETH"] if s != sym])
    mode = rng.random()
    if mode < 0.55:
        w.aave = []
    else:
        w.aave = [{"symbol": other, "chain": rng.choice(SYM_CHAINS[other]), "amount": rand_supplied(rng, other)}]
    set_balances(w, rng, [], focus=(sym,))
    named = rng.choice(SYM_CHAINS[sym]) if rng.random() < 0.2 else None
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_ALL, WD_ALL_EXTRA, sym, named))
    args = wd_args(sym, named)
    r = c.call(("prepare_withdraw", args))
    assert r["status"] == "nothing_to_withdraw"
    wd_finish(rng, c, bio, sym, r, None, named)
    return fin(c, "wd_nothing", "withdraw", bio, [("prepare_withdraw", args)], no_ask=True, all=True, outcome="nothing_to_withdraw")


def sc_wd_choice(rng, ev):
    sym = rng.choice(["USDC"] * 4 + ["ETH", "USDT"])
    chs = rng.sample(SYM_CHAINS[sym], 2)
    w, bio = begin(rng)
    sup = [rand_supplied(rng, sym), rand_supplied(rng, sym)]
    while sup[0] == sup[1]:
        sup[1] = rand_supplied(rng, sym)
    set_aave(w, rng, [(sym, chs[0], sup[0]), (sym, chs[1], sup[1])], extra_other=False)
    set_balances(w, rng, [], focus=(sym,))
    with_amt = rng.random() < 0.25
    amt = None
    if with_amt:
        amt = round(min(sup) * 0.4, 4 if sym == "ETH" else 2)
        amt = num(amt)
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_AMT if with_amt else WD_ALL, WD_AMT_EXTRA if with_amt else WD_ALL_EXTRA, sym, None, amt))
    a1 = wd_args(sym, None, amt)
    r = c.call(("prepare_withdraw", a1))
    assert r["status"] == "needs_choice", r
    c.say(choice_text(rng, r), ask=True)
    pick = rng.choice(chs)
    c.user(dress(rng, ev, "=" + rng.choice(CHAIN_REPLY_AMT if with_amt else CHAIN_REPLY).format(n=rng.choice(CHAIN_WORDS[pick]))))
    a2 = wd_args(sym, pick, amt)
    r2 = c.call(("prepare_withdraw", a2))
    st = wd_finish(rng, c, bio, sym, r2, amt, pick)
    return fin(c, "wd_needs_choice", "withdraw", bio, [("prepare_withdraw", a1), ("prepare_withdraw", a2)], no_ask=False, outcome=st)


def sc_wd_insuff(rng, ev):
    sym = rng.choice(["USDC"] * 5 + ["ETH", "USDT"])
    chain = rng.choice(SYM_CHAINS[sym])
    w, bio, supplied = wd_world(rng, sym, chain)
    pool = [x for x in (ETH_AMOUNTS if sym == "ETH" else [100, 150, 200, 250, 300, 500, 750, 1000, 2000, 5000, 10000]) if x > supplied * 1.15]
    amt = rng.choice(pool) if pool else int(supplied * 2) + 1
    named = chain if rng.random() < 0.25 else None
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_AMT, WD_AMT_EXTRA, sym, named, amt))
    args = wd_args(sym, named, amt)
    r = c.call(("prepare_withdraw", args))
    assert r["status"] == "insufficient"
    wd_finish(rng, c, bio, sym, r, amt, named)
    return fin(c, "wd_insufficient", "withdraw", bio, [("prepare_withdraw", args)], no_ask=True, outcome="insufficient")


def sc_wd_blocked(rng, ev):
    sym = rand_sym(rng)
    chain = rng.choice(SYM_CHAINS[sym])
    w, bio, _ = wd_world(rng, sym, chain)
    w.block = rng.choice(BLOCK_REASONS)
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_ALL, WD_ALL_EXTRA, sym))
    args = wd_args(sym)
    r = c.call(("prepare_withdraw", args))
    assert r["status"] == "blocked"
    wd_finish(rng, c, bio, sym, r)
    return fin(c, "wd_blocked", "withdraw", bio, [("prepare_withdraw", args)], no_ask=True, all=True, outcome="blocked")


def sc_wd_unsup_chain(rng, ev):
    sym = rand_sym(rng)
    w, bio, _ = wd_world(rng, sym, rng.choice(SYM_CHAINS[sym]))
    disp, words = rng.choice(OTHER_CHAINS)
    word = rng.choice(words)
    st = rng.choice(["withdraw my {s} from {A} on {w}", "take my {s} out of {A} on {w}", "pull my {s} from {A} {w}", "get my {s} out of {A} on {w}",
                     "=withdraw all my {s} on {w} from {A}", "=my {s} on {w} out of {A}"])
    c = conv(rng, w, bio)
    c.user(dress(rng, ev, st.format(s=sym_word(rng, sym), A=aave_word(rng), w=word)))
    c.say(rng.choice([
        f"Aave only works on Ethereum, Base and Arbitrum here, so I can't withdraw on {disp}. Tell me which of those three you mean.",
        f"I can only prepare Aave v3 withdrawals on Ethereum, Base or Arbitrum, not {disp}. Which of those should I use?",
        f"Aave v3 is only available on Ethereum, Base and Arbitrum in Hermit, so {disp} won't work. Which of those do you want?"]))
    return fin(c, "wd_unsupported_chain", "withdraw", bio, [], no_ask=False, outcome="no_call", other_chain=disp)


def sc_wd_follow(rng, ev):
    sym = rand_sym(rng)
    w, bio, _ = wd_world(rng, sym, rng.choice(SYM_CHAINS[sym]))
    c = conv(rng, w, bio)
    c.user(wd_user(rng, ev, WD_ALL, WD_ALL_EXTRA, sym))
    args = wd_args(sym)
    r = c.call(("prepare_withdraw", args))
    assert r["status"] == "review_opened"
    c.say(wd_ok_text(rng, sym, bio))
    c.user(dress(rng, ev, "=" + rng.choice(FOLLOW_WD)))
    c.say(rng.choice(FOLLOW_ANS).format(BIO=bio))
    return fin(c, "wd_followup", "withdraw", bio, [("prepare_withdraw", args)], no_ask=True, all=True, final_no_call=True, outcome="followup")


# ---------------------------------------------------------------------- negatives
NEG_YIELD = ["what's the {s} yield", "{s} apy?", "how much does {s} earn on aave", "what's aave paying on {s}", "current yield on {s}",
             "best {s} rate right now", "=what are the best {s} yields", "how much can i earn on {s}", "={s} yield on{c}?", "what's the going rate for {s}{c}",
             "what apy am i looking at for {s}", "=whats the apy on {s}"]
NEG_YIELD_EXTRA = ["how are {s} rates looking{c}", "={s} rates?"]
NEG_SAFE = ["is aave safe", "is aave safe to use?", "can i trust aave", "is aave legit", "=aave safe?", "what are the risks of putting money in aave",
            "could i lose my money on aave", "is it risky to lend on aave", "is depositing in aave safe", "how safe is aave really",
            "=is aave a scam", "should i worry about aave getting hacked"]
NEG_SAFE_EXTRA = ["=any risk with aave?", "is aave ok for a beginner"]
NEG_SAFE_ANS = [
    "Aave is a large lending protocol: you supply a token and borrowers pay you interest. It's not risk free. The main risks are smart-contract bugs, bad loans if markets crash, a stablecoin losing its peg, and a variable APY that can drop. Only deposit what you can afford to have locked up in a protocol.",
    "Fairly established, but not risk free. Aave is a lending protocol, so the risks are a bug in the smart contracts, bad debt in a sharp market crash, and a stablecoin depegging. The APY is variable too. No one guarantees your deposit, so start small.",
    "Aave is one of the better known lending protocols, but your funds sit in smart contracts, so there is always smart-contract risk. Rates are variable and a market crash or a depegged stablecoin can hurt. It's not a bank account and nothing is insured."]
NEG_HOWTO_WD = ["how do i withdraw from aave", "how do i get my money out of aave", "how can i take my usdc out of aave", "=how to withdraw from aave?",
                "where do i withdraw my aave deposit", "what's the way to pull money out of aave", "how does withdrawing from aave work"]
NEG_HOWTO_DEP = ["how do i deposit into aave", "how do i put usdc in aave", "=how to lend on aave", "how can i start earning with aave here",
                 "what do i do to deposit in aave", "how does depositing in aave work here"]
NEG_HOWTO_EXTRA = ["=how would i exit aave", "what's the process to deposit on aave"]
NEG_HOWTO_WD_ANS = [
    "Just ask me, for example 'withdraw my USDC from Aave', and I'll prepare it. You then review it and approve with {BIO}. You can also do it from Earn > Positions.",
    "Tell me what to take out, like 'withdraw my USDC from Aave', or add an amount if you want less, and I'll prepare it for you to approve with {BIO}. Earn > Positions works too."]
NEG_HOWTO_DEP_ANS = [
    "Tell me the token and amount you want to deposit, like 'deposit USDC into Aave', and I'll prepare it. You review it and approve with {BIO}. Aave v3 on Ethereum, Base and Arbitrum is the only pool I can prepare.",
    "Just ask me with a token and an amount, like 'deposit my USDC into Aave', and I'll prepare the deposit. Nothing happens until you approve it with {BIO}. You can also deposit from Earn."]
NEG_SWAP = ["swap my aUSDC", "swap my aUSDC for eth", "sell my ausdc", "convert my aUSDC to usdt", "=swap aUSDC to USDC", "trade my aeth for usdc",
            "swap my aBasUSDC", "=can you swap my ausdc", "bridge my aUSDC to base"]
NEG_SWAP_ANS = [
    "I can't swap aUSDC. It's the receipt token Aave gives you for supplying USDC, not something I can trade. I can withdraw your USDC from Aave, and then swap that if you want.",
    "aUSDC is what Aave issues when you supply USDC, so I don't swap it. If you want out, I can prepare a withdrawal of your USDC from Aave first, then we can swap the USDC.",
    "I don't swap aTokens like aUSDC. They're Aave's receipt for your deposit. Ask me to withdraw your USDC from Aave and I'll prepare it, then you can swap the USDC."]
NEG_AMM = ["do i get impermanent loss on aave", "is aave a dex", "is aave an amm?", "=does aave have impermanent loss", "which pool does aave use, like uniswap?",
           "is depositing in aave the same as providing liquidity", "=is aave a liquidity pool"]
NEG_AMM_EXTRA = ["=impermanent loss when i lend on aave?"]
NEG_AMM_ANS = [
    "No. Aave is a lending protocol, not an AMM. You supply one token and earn interest paid by borrowers, so there's no impermanent loss like in a swap pool. The risks are smart-contract risk, bad loans in a crash, and a variable APY.",
    "Aave isn't a DEX or an AMM. It's a lending market: you supply a single token and borrowers pay interest. Impermanent loss only happens when you provide two assets to a swap pool, so it doesn't apply to a plain Aave deposit.",
    "No, depositing in Aave is lending, not providing liquidity to a swap pool. You supply one token and earn a variable APY, so there's no impermanent loss. You still carry smart-contract risk."]
NEG_WHATIS = ["what is aave", "=explain aave to me", "what does aave do", "what's aave", "=aave, what is it"]
NEG_WHATIS_ANS = [
    "Aave is a lending protocol. People supply tokens and earn interest, and others borrow against their own collateral. In Hermit I can prepare Aave v3 deposits and withdrawals on Ethereum, Base and Arbitrum, and you approve each one with {BIO}.",
    "Aave is a decentralised lending market. You supply a token like USDC and earn a variable APY paid by borrowers. Hermit can prepare Aave v3 deposits and withdrawals on Ethereum, Base and Arbitrum for you to approve with {BIO}."]


def sc_neg_yield(rng, ev):
    w, bio = begin(rng)
    sym = rng.choice(["USDC"] * 7 + ["ETH", "USDT", "DAI"])
    chain = rng.choice(["base", "arbitrum", "ethereum"]) if rng.random() < 0.2 else None
    cp = ""
    if chain:
        cp, _ = chain_phrase(rng, chain)
    set_balances(w, rng, [], focus=(sym,))
    c = conv(rng, w, bio)
    c.user(dress(rng, ev, pick_stem(rng, ev, NEG_YIELD, NEG_YIELD_EXTRA).format(s=sym_word(rng, sym), c=cp)))
    ya = {"asset": sym}
    if chain:
        ya["chain"] = chain
    ry = c.call(("get_yields", ya))
    if not ry["pools"]:
        c.say(f"I don't see any {sym} pools{' on ' + CHAIN_NAME[chain] if chain else ''} right now.")
    else:
        c.say(yield_text(rng, ry["pools"], sym, offer=rng.random() < 0.4))
    return fin(c, "neg_yield", "neg", bio, [], no_ask=True, first_tool="get_yields", outcome="yields")


def _plain(kind, stems, extra, answers, fmt_bio=True):
    def f(rng, ev):
        w, bio = begin(rng)
        c = conv(rng, w, bio)
        c.user(dress(rng, ev, pick_stem(rng, ev, stems, extra)))
        c.say(rng.choice(answers).format(BIO=bio))
        return fin(c, kind, "neg", bio, [], no_ask=True, outcome="plain")
    return f


sc_neg_safe = _plain("neg_safe", NEG_SAFE, NEG_SAFE_EXTRA, NEG_SAFE_ANS)
sc_neg_swap = _plain("neg_swap_aave", NEG_SWAP, [], NEG_SWAP_ANS)
sc_neg_amm = _plain("neg_amm", NEG_AMM, NEG_AMM_EXTRA, NEG_AMM_ANS)
sc_neg_whatis = _plain("neg_whatis", NEG_WHATIS, [], NEG_WHATIS_ANS)


def sc_neg_howto(rng, ev):
    w, bio = begin(rng)
    c = conv(rng, w, bio)
    if rng.random() < 0.5:
        c.user(dress(rng, ev, pick_stem(rng, ev, NEG_HOWTO_WD, NEG_HOWTO_EXTRA)))
        c.say(rng.choice(NEG_HOWTO_WD_ANS).format(BIO=bio))
    else:
        c.user(dress(rng, ev, pick_stem(rng, ev, NEG_HOWTO_DEP, NEG_HOWTO_EXTRA)))
        c.say(rng.choice(NEG_HOWTO_DEP_ANS).format(BIO=bio))
    return fin(c, "neg_howto", "neg", bio, [], no_ask=True, outcome="plain")


# ---------------------------------------------------------------------- plans
TRAIN_PLAN = [(sc_dep_direct, 60), (sc_dep_chain, 40), (sc_dep_earn, 30), (sc_dep_all, 25), (sc_dep_frac, 25), (sc_dep_other, 20),
              (sc_dep_insuff, 14), (sc_dep_needs_chain, 14), (sc_dep_blocked, 8), (sc_dep_unsup_chain, 10), (sc_dep_error, 4), (sc_dep_follow, 10),
              (sc_wd_all, 70), (sc_wd_all_chain, 20), (sc_wd_asktoken, 30), (sc_wd_amount, 40), (sc_wd_amount_chain, 10), (sc_wd_part_ask, 14),
              (sc_wd_nothing, 14), (sc_wd_choice, 16), (sc_wd_insuff, 10), (sc_wd_blocked, 4), (sc_wd_unsup_chain, 6), (sc_wd_follow, 6),
              (sc_neg_yield, 14), (sc_neg_safe, 12), (sc_neg_howto, 12), (sc_neg_swap, 10), (sc_neg_amm, 8), (sc_neg_whatis, 4)]
EVAL_PLAN = [(sc_dep_direct, 6), (sc_dep_chain, 3), (sc_dep_earn, 3), (sc_dep_all, 2), (sc_dep_frac, 2), (sc_dep_other, 2), (sc_dep_insuff, 1),
             (sc_dep_needs_chain, 1), (sc_dep_blocked, 1), (sc_dep_unsup_chain, 2), (sc_dep_follow, 1),
             (sc_wd_all, 8), (sc_wd_all_chain, 2), (sc_wd_asktoken, 3), (sc_wd_amount, 4), (sc_wd_amount_chain, 1), (sc_wd_part_ask, 2),
             (sc_wd_nothing, 1), (sc_wd_choice, 2), (sc_wd_insuff, 1), (sc_wd_unsup_chain, 1), (sc_wd_follow, 1),
             (sc_neg_yield, 3), (sc_neg_safe, 2), (sc_neg_howto, 2), (sc_neg_swap, 2), (sc_neg_amm, 1)]


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def first_user(ex):
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def build(seed, plan, ev, scale=1.0, forbid=frozenset()):
    rng = random.Random(seed)
    rows, seen = [], set(forbid)
    for fn, n in plan:
        n = max(1, round(n * scale)) if scale != 1.0 else n
        made, tries = 0, 0
        while made < n:
            tries += 1
            if tries > 400 * n + 400:
                raise RuntimeError(f"cannot make {n} unique rows for {fn.__name__}")
            ex = fn(random.Random(rng.getrandbits(64)), ev)
            k = norm(first_user(ex))
            if k in seen:
                continue
            seen.add(k)
            rows.append(ex)
            made += 1
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------- eval conversion
def chains_named(text):
    t = text.lower()
    return sorted(c for c, rx in CHAIN_RE.items() if re.search(rx, t))


BANNED_CLAIMS = [r"\bdeposited\b", r"\bwithdrew\b", r"\bwithdrawn\b", r"has been (sent|deposited|withdrawn)", r"went through",
                 r"successfully", r"funds are in", r"is now in aave", r"\bit's done\b", r"\bdone!"]


def to_eval(ex, i):
    msgs = ex["messages"]
    last_u = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    users = " ".join(m["content"] for m in msgs[:last_u + 1] if m["role"] == "user")
    tail = msgs[last_u + 1:]
    tail_calls = [tc for m in tail for tc in (m.get("tool_calls") or [])]
    final_text = next((m["content"] for m in reversed(tail) if m["role"] == "assistant" and m["content"]), "")
    meta = ex["meta"]
    intent = meta["intent"]
    expect = meta["expect"]
    prepare = [tc for tc in tail_calls if tc["name"].startswith("prepare_")]
    ask_turn = any(e.get("ask") for e in expect)
    gold_args = [tc["arguments"] for tc in prepare]
    answer_digits = sorted({m for m in re.findall(r"\d+(?:\.\d+)?", re.sub(r"\bv3\b", "", final_text))})
    checks = {
        "gold_calls": [{"name": tc["name"], "arguments": tc["arguments"]} for tc in tail_calls],
        "required_first_call": (tail_calls[0]["name"] if tail_calls else None),
        "exact_digits": {"args": [fmt(a["amount"]) for a in gold_args if "amount" in a], "answer": answer_digits},
        "no_chain_invented": {"allowed_chains": chains_named(users), "gold_chains": [a.get("chain") for a in gold_args],
                              "forbid_unnamed_chain_arg": True},
        "asked_when_needed": {"must_ask_first": bool(ask_turn and not tail_calls) or bool(intent.get("ask_first") and False),
                              "must_not_ask": bool(intent.get("no_ask")), "final_turn_asks": ask_turn,
                              "withdraw_all_must_not_ask_how_much": bool(intent.get("all"))},
        "no_unbacked_done": {"forbid_regex": BANNED_CLAIMS, "only_prepared_text": True},
        "facts_ok": {"forbid_regex": [r"\bAave\b[^.]{0,40}\bis an? (AMM|DEX|exchange)\b", r"pool of two", r"\bRelay\b"],
                     "aave_is_lending": meta["kind"] == "neg_amm"},
        "forbid_calls": ["open_app"] + ([] if prepare else ["prepare_deposit", "prepare_withdraw"]),
        "chain_names_in_answer_must_come_from_user_or_result": meta["kind"] not in ("dep_unsupported_chain", "wd_unsupported_chain"),
    }
    return {"id": f"aave-{meta['kind']}-{i}", "kind": meta["kind"], "tools": True, "context": msgs[:last_u + 1], "reference": tail,
            "expect": expect, "world": meta["world"], "checks": checks,
            "meta": {"source": "aave", "kind": meta["kind"], "family": meta["family"], "bio": meta["bio"], "intent": intent}}


# ---------------------------------------------------------------------- self-check
TOOL_NAMES = set(H.TOOL_NAMES)
ASSIST_BANNED = [r"\bdeposited\b", r"\bwithdrew\b", r"\bwithdrawn\b", r"has been (sent|deposited|withdrawn)", r"went through",
                 r"successfully", r"funds are in", r"\bit's done\b", r"\bdone!", "·", r"\bRelay\b", r"\bI can't see the future\b",
                 r"\bimpermanent\b(?!\s+loss)"]
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
ADDR_RE = re.compile(r"0x[0-9a-fA-F]{40}|\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")


def check_args(tc, errs, ctx):
    sch = H.TOOL_SCHEMA[tc["name"]]
    a = tc["arguments"]
    for req in sch.get("required", []):
        if req not in a:
            errs.append(f"{ctx}: missing {req}")
    for k, v in a.items():
        if k not in sch["properties"]:
            errs.append(f"{ctx}: unknown arg {k}"); continue
        t = sch["properties"][k].get("type")
        if t == "number" and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            errs.append(f"{ctx}: {k} not a number {v!r}")
        if t == "string" and not isinstance(v, str):
            errs.append(f"{ctx}: {k} not a string {v!r}")
    for k, v in a.items():
        if isinstance(v, float) and v.is_integer():
            errs.append(f"{ctx}: float {v!r} should be an int")
        if isinstance(v, float) and ("e" in repr(v)):
            errs.append(f"{ctx}: exponent number {v!r}")


def selfcheck_row(msgs, meta, world_state, errs, ctx):
    # roles + system prompt
    bio = meta["bio"]
    sysm = msgs[0]
    if sysm["role"] != "system" or not sysm["content"].startswith(SYS_TEMPLATE.replace("{BIO}", bio)):
        errs.append(f"{ctx}: system prompt is not SYSTEM_AGENT_v2 with {bio}")
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    blob = json.dumps(msgs, ensure_ascii=False)
    if other in blob:
        errs.append(f"{ctx}: mixes {other} with {bio}")
    if "{BIO}" in blob:
        errs.append(f"{ctx}: unrendered {{BIO}}")
    # replay every call against the simulator, compare recorded results exactly
    w = World3.from_state(world_state)
    users_so_far, tool_nums = [], set()
    prepare_calls, recorded = [], []
    i = 1
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "user":
            users_so_far.append(m["content"]); tool_nums |= numbers_in(m["content"]); i += 1; continue
        if m["role"] == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                results = msgs[i + 1:i + 1 + len(tcs)]
                if len(results) != len(tcs) or any(r["role"] != "tool" for r in results):
                    errs.append(f"{ctx}: tool results do not follow the call"); return
                for tc, rm in zip(tcs, results):
                    if tc["name"] not in TOOL_NAMES:
                        errs.append(f"{ctx}: unknown tool {tc['name']}")
                        continue
                    check_args(tc, errs, ctx + ":" + tc["name"])
                    try:
                        got = json.dumps(w.call(tc["name"], tc["arguments"]), ensure_ascii=False)
                    except Exception as e:                       # noqa
                        errs.append(f"{ctx}: replay crashed {e}"); continue
                    if got != rm["content"]:
                        errs.append(f"{ctx}: replay mismatch {tc['name']}: {got[:120]} vs {rm['content'][:120]}")
                    tool_nums |= numbers_in(rm["content"])
                    tool_nums |= numbers_in(json.dumps(tc["arguments"]))
                    if tc["name"] in ("prepare_deposit", "prepare_withdraw"):
                        prepare_calls.append((tc, " ".join(users_so_far)))
                        recorded.append([tc["name"], tc["arguments"]])
                    if tc["name"] == "open_app":
                        errs.append(f"{ctx}: open_app used")
                i += 1 + len(tcs)
                continue
            # a spoken assistant turn
            txt = m["content"]
            for rx in ASSIST_BANNED:
                if re.search(rx, txt, re.I):
                    errs.append(f"{ctx}: banned phrase {rx} in: {txt[:80]}")
            if EMOJI.search(txt):
                errs.append(f"{ctx}: emoji")
            if meta["kind"] != "neg_amm" and re.search(r"\bAMM\b", txt):
                errs.append(f"{ctx}: AMM mentioned outside neg_amm")
            if meta["kind"] == "neg_amm" and "lending" not in txt:
                errs.append(f"{ctx}: neg_amm answer lacks 'lending'")
            if bio == "Touch ID" and "Face ID" in txt:
                errs.append(f"{ctx}: Face ID in Touch ID row")
            # numbers must come from the user text, tool results or the calculator
            ns = numbers_in(re.sub(r"\bv3\b", "", txt))
            bad = [n for n in ns if n not in tool_nums]
            if bad:
                errs.append(f"{ctx}: numbers {bad} in answer not backed by user/tool text: {txt[:90]}")
            # chain names must come from the user or the results
            if meta["kind"] not in ("dep_unsupported_chain", "wd_unsupported_chain") and meta["family"] != "neg":
                seen = " ".join(users_so_far).lower() + " " + " ".join(x["content"].lower() for x in msgs[:i] if x["role"] == "tool")
                for cn, rx in (("ethereum", r"\bethereum\b"), ("base", r"\bbase\b"), ("arbitrum", r"\barbitrum\b")):
                    if re.search(rx, txt, re.I) and not re.search(CHAIN_RE[cn], seen):
                        errs.append(f"{ctx}: chain {cn} invented in answer: {txt[:90]}")
            if ADDR_RE.search(txt) and not any(ADDR_RE.search(u) for u in users_so_far):
                errs.append(f"{ctx}: address in answer")
            if "?" in txt and meta["intent"].get("no_ask") and len([x for x in msgs[:i] if x["role"] == "assistant" and x.get("tool_calls")]) == 0 \
                    and meta["intent"].get("first_tool") is None:
                errs.append(f"{ctx}: asked a question where the scenario forbids it: {txt[:80]}")
        i += 1
    # the prepare calls are exactly the scenario intent
    intent = meta["intent"]
    if recorded != intent["calls"]:
        errs.append(f"{ctx}: prepare calls differ from intent: {recorded} vs {intent['calls']}")
    fam = meta["family"]
    for tc, users in prepare_calls:
        a = tc["arguments"]
        name = tc["name"]
        sym = a["symbol"]
        if name == "prepare_deposit":
            if meta["kind"] == "dep_other_protocol":
                if a.get("project") == "aave-v3" or not a.get("project"):
                    errs.append(f"{ctx}: other-protocol call must name the protocol")
            elif a.get("project") != "aave-v3":
                errs.append(f"{ctx}: deposit project is not aave-v3: {a}")
            if fam != "deposit":
                errs.append(f"{ctx}: deposit call in {fam} row")
        else:
            if "project" in a:
                errs.append(f"{ctx}: withdraw has project")
            if intent.get("all") and "amount" in a:
                errs.append(f"{ctx}: withdraw-all call has an amount: {a}")
            if fam != "withdraw":
                errs.append(f"{ctx}: withdraw call in {fam} row")
        # chain only if the user named one (up to this call), and exactly the one named
        named = chains_named(users)
        if "chain" in a:
            if a["chain"] not in named:
                errs.append(f"{ctx}: chain {a['chain']} not named by user ({named})")
            if a["chain"] not in AAVE_CHAINS:
                errs.append(f"{ctx}: chain {a['chain']} not an Aave chain")
        elif len(named) == 1 and len(set(named)) == 1 and meta["kind"] not in ("dep_needs_chain", "wd_needs_choice"):
            errs.append(f"{ctx}: user named {named} but call has no chain")
        # symbol must be a token the user typed (or tool holdings for dep_all/frac)
        if not re.search(SYM_ALIAS_RE.get(sym, re.escape(sym.lower())), users.lower()):
            errs.append(f"{ctx}: symbol {sym} not in user text")
        # amount: from user text, or from balances/calculator in the same conversation
        if "amount" in a:
            v = float(a["amount"])
            if v not in tool_nums:
                errs.append(f"{ctx}: amount {a['amount']} appears in no user text or tool result")
            if a["amount"] != num(v):
                errs.append(f"{ctx}: amount not canonical")
    # kind-specific
    k = meta["kind"]
    calls_all = [tc["name"] for m in msgs for tc in (m.get("tool_calls") or [])]
    if k.startswith("neg_") and any(n.startswith("prepare_") for n in calls_all):
        errs.append(f"{ctx}: negative made a prepare call")
    if k in ("neg_safe", "neg_howto", "neg_swap_aave", "neg_amm", "neg_whatis") and calls_all:
        errs.append(f"{ctx}: negative {k} must not call tools")
    if k == "neg_yield" and calls_all != ["get_yields"]:
        errs.append(f"{ctx}: neg_yield must call get_yields once")
    if k in ("dep_all", "dep_fraction") and calls_all[0] != "get_balances":
        errs.append(f"{ctx}: {k} must start with get_balances")
    if k == "dep_fraction" and calls_all[:3] != ["get_balances", "calculator", "prepare_deposit"]:
        errs.append(f"{ctx}: dep_fraction chain wrong {calls_all}")
    if k in ("dep_earn_offer", "dep_earn_agree") and calls_all[0] != "get_yields":
        errs.append(f"{ctx}: earn must start with get_yields")
    if k == "dep_earn_offer" and len(calls_all) != 1:
        errs.append(f"{ctx}: earn offer must not prepare")
    if k in ("dep_unsupported_chain", "wd_unsupported_chain") and calls_all:
        errs.append(f"{ctx}: unsupported chain must not call tools")
    if k.endswith("followup") and msgs[-1].get("tool_calls"):
        errs.append(f"{ctx}: follow-up must be text only")
    # last message must be an assistant text
    if msgs[-1]["role"] != "assistant" or not msgs[-1]["content"]:
        errs.append(f"{ctx}: does not end with an assistant answer")
    # in withdraw-all rows no assistant question precedes the call
    if intent.get("all"):
        for j, m in enumerate(msgs):
            if m["role"] == "assistant" and not m.get("tool_calls") and "?" in m["content"]:
                if not any(x["role"] == "assistant" and x.get("tool_calls") for x in msgs[:j]):
                    errs.append(f"{ctx}: withdraw-all asked before the call")


def selfcheck(train, ev, train_path, eval_path):
    errs = []
    # re-read what was written
    for path, rows, is_eval in ((train_path, train, False), (eval_path, ev, True)):
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        if lines != rows:
            errs.append(f"{path}: file does not round-trip")
    seen = {}
    for i, ex in enumerate(train):
        ctx = f"train[{i}]"
        if json.loads(json.dumps(ex, ensure_ascii=False)) != ex:
            errs.append(f"{ctx}: json roundtrip")
        if ex.get("tools") is not True or ex["meta"].get("source") != "aave":
            errs.append(f"{ctx}: wrong record shape")
        selfcheck_row(ex["messages"], ex["meta"], ex["meta"]["world"], errs, ctx)
        k = norm(first_user(ex))
        if k in seen:
            errs.append(f"{ctx}: duplicate user text of train[{seen[k]}]: {first_user(ex)!r}")
        seen[k] = i
    seen_ev = {}
    for i, r in enumerate(ev):
        ctx = f"eval[{i}]"
        if json.loads(json.dumps(r, ensure_ascii=False)) != r:
            errs.append(f"{ctx}: json roundtrip")
        for key in ("id", "kind", "tools", "context", "reference", "expect", "world", "checks", "meta"):
            if key not in r:
                errs.append(f"{ctx}: missing {key}")
        msgs = r["context"] + r["reference"]
        selfcheck_row(msgs, r["meta"], r["world"], errs, ctx)
        k = norm(next(m["content"] for m in r["context"] if m["role"] == "user"))
        if k in seen:
            errs.append(f"{ctx}: user text also in train: {k!r}")
        if k in seen_ev:
            errs.append(f"{ctx}: duplicate eval user text")
        seen_ev[k] = i
        # all user texts of the eval conversation must differ from every opening train text too
        for m in r["context"]:
            if m["role"] == "user" and norm(m["content"]) in seen and m is not None and len(norm(m["content"])) > 12:
                errs.append(f"{ctx}: a user turn equals a train opening text: {m['content']!r}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=500, help="deposit+withdraw rows (negatives are 12% on top)")
    ap.add_argument("--eval-seed", type=int, default=303)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data3", "aave.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(HERE, "..", "eval3", "aave.jsonl"))
    a = ap.parse_args()
    assert a.eval_seed != a.seed, "eval must use a different seed"
    scale = a.n / 500.0
    train = build(a.seed, TRAIN_PLAN, False, scale)
    train_keys = {norm(first_user(r)) for r in train}
    ev_rows = build(a.eval_seed, EVAL_PLAN, True, 1.0, forbid=train_keys)
    ev = [to_eval(r, i) for i, r in enumerate(ev_rows)]
    for p in (a.out, a.eval_out):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errs = selfcheck(train, ev, a.out, a.eval_out)
    fam = Counter(r["meta"]["family"] for r in train)
    bios = Counter(r["meta"]["bio"] for r in train)
    print("train kinds:", dict(sorted(Counter(r["meta"]["kind"] for r in train).items())), file=sys.stderr)
    print("train families:", dict(fam), "bio:", dict(bios), file=sys.stderr)
    print("eval kinds:", dict(sorted(Counter(r["kind"] for r in ev).items())), file=sys.stderr)
    if errs:
        print(f"{len(errs)} self-check errors, first 25:", file=sys.stderr)
        for e in errs[:25]:
            print("  " + e, file=sys.stderr)
        print(f"aave: {len(train)} train, {len(ev)} eval, self-check FAILED ({len(errs)} errors)")
        sys.exit(1)
    print(f"aave: {len(train)} train, {len(ev)} eval, self-check ok")


if __name__ == "__main__":
    main()
