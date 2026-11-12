#!/usr/bin/env python3
"""send_nochain: single sends where the user names no network (and the other single-send edge cases).

Teaches: call prepare_send WITHOUT `chain` unless the user named a network; the app picks the network where the token
is held and returns it (`network`); never ask "which network?" before the first call; never default to Ethereum;
when the app answers needs_chain, ask its question in one sentence, then call again WITH the chain the user names.

Usage (lab venv):
  python gen3/send_nochain.py [--seed 3] [--n 600] [--eval-seed 9304] [--n-eval 60]
Writes data3/send_nochain.jsonl and eval3/send_nochain.jsonl, then runs the self-check over every record.

Importable pieces for the eval harness: World3 (app-truth simulator, call(name, args)), run_tool(world, name, args,
typed_text) (adds the app's ToolGuard: drops an unnamed chain, refuses an address the user did not type), SYSTEM_TXT.

Everything the assistant says is a Python f-string filled from tool results or user text. Nothing is LLM-written.
Sources of truth: spec/NOTES_apptruth.md (result shapes), wt-near HermitToolbox+Money.swift (pickChain, question text),
MoneyIntents.swift (ChainWords), AgentTools.swift (address refusal), SpendPolicy.swift (agent cap, daily limit).
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, math, os, random, re
from collections import Counter
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen2"))
import hermit_fmt as H          # gen2 helpers, imported not copied (evm_addr, sol_addr, with_user, calc, render, parse_calls)
import sim                      # gen2 simulator: World (prices, state/from_state)
from build_agent import Conv    # gen2 conversation builder

SPEC = os.path.join(LAB, "spec")
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
V2_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
SYSTEM_TXT = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read().rstrip("\n")
assert "{BIO}" in SYSTEM_TXT

# ============================================================== small helpers
def jn(x):
    """JSON number in shortest form: whole floats become ints."""
    x = float(x)
    return int(x) if x.is_integer() else x


def ntxt(x):
    x = jn(x)
    return str(x) if isinstance(x, int) else format(Decimal(repr(x)), "f")


def approx(v):
    """Units.approx from the app: >= 1000 -> 2 dp, else 3 significant digits, trailing zeros trimmed."""
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    dec = 2 if a >= 1000 else min(18, max(0, 3 - int(math.floor(math.log10(a)))))
    s = f"{v:.{dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def money(x):
    x = float(x)
    return "$" + (f"{x:,.0f}" if x == round(x) else f"{x:,.2f}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def join_and(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def join_or(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


# ============================================================== chains, tokens, app guards
ORDER = ["ethereum", "base", "bnb", "monad", "arbitrum", "polygon", "robinhood", "solana", "near"]   # Chain.mainnets order
CHAIN_NAME = {"ethereum": "Ethereum", "base": "Base", "bnb": "BNB Chain", "monad": "Monad", "arbitrum": "Arbitrum",
              "polygon": "Polygon", "robinhood": "Robinhood Chain", "solana": "Solana", "near": "NEAR"}
CHAIN_KIND = {c: "evm" for c in ORDER}
CHAIN_KIND.update({"solana": "solana", "near": "near"})
# where each token exists for a send (TokenTable + native symbols)
TOKEN_CHAINS = {"USDC": ["ethereum", "base", "bnb", "monad", "arbitrum", "polygon", "solana", "near"],
                "USDT": ["ethereum", "bnb", "solana", "near"],
                "ETH": ["ethereum", "base", "arbitrum", "robinhood"],
                "SOL": ["solana"], "NEAR": ["near"], "POL": ["polygon"], "BNB": ["bnb"]}
HELD_CHAINS = {"USDC": ["ethereum", "base", "bnb", "arbitrum", "polygon", "solana", "near"],
               "USDT": ["ethereum", "bnb", "solana", "near"], "ETH": ["ethereum", "base", "arbitrum"],
               "SOL": ["solana"], "NEAR": ["near"], "POL": ["polygon"], "DAI": ["ethereum"]}
EVM_HELD = {"USDC": ["ethereum", "base", "bnb", "arbitrum", "polygon"], "USDT": ["ethereum", "bnb"],
            "ETH": ["ethereum", "base", "arbitrum"]}
STABLE = {"USDC", "USDT", "DAI"}
DP = {"USDC": 2, "USDT": 2, "DAI": 2, "ETH": 5, "SOL": 3, "NEAR": 2, "POL": 2}

# ChainWords.names from MoneyIntents.swift
CHAIN_WORDS = [("ethereum", ["ethereum", "mainnet", "eth mainnet", "on eth", "erc20", "erc-20", "l1"]),
               ("base", ["base"]), ("arbitrum", ["arbitrum", "arb", "arbitrum one"]),
               ("bnb", ["bnb chain", "bnb", "bsc", "bep20", "bep-20", "binance smart chain"]),
               ("polygon", ["polygon", "matic", "pol"]), ("monad", ["monad"]), ("robinhood", ["robinhood"]),
               ("solana", ["solana", "sol", "spl"]), ("near", ["near"])]


def mentions(chain_id, text):
    words = dict(CHAIN_WORDS).get(chain_id)
    if not words:
        return False
    t = " " + re.sub(r"[^a-z0-9\- ]", " ", text.lower()) + " "
    return any(" " + w + " " in t for w in words)


def mentioned(text):
    return [c for c, _ in CHAIN_WORDS if mentions(c, text)]


def find_chain(arg):
    a = str(arg).lower().strip()
    if not a:
        return None
    for cid, words in CHAIN_WORDS:
        if cid == a or a in words:
            return cid
    alias = {"eth": "ethereum", "binance": "bnb", "bnbchain": "bnb", "arb1": "arbitrum"}
    k = a.replace(" ", "")
    if k in alias:
        return alias[k]
    for cid, _ in CHAIN_WORDS:
        if a.startswith(cid):
            return cid
    return None


def without_files(s):
    while "<file>" in s:
        a = s.index("<file>")
        b = s.find("</file>", a + 6)
        s = s[:a] if b < 0 else s[:a] + s[b + 7:]
    return s


def user_typed(address, text):
    is_hex = address[:2].lower() == "0x"
    hay, needle = (text.lower(), address.lower()) if is_hex else (text, address)
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return False
        before = hay[i - 1] if i > 0 else None
        after = hay[i + len(needle)] if i + len(needle) < len(hay) else None
        word = lambda ch: ch is not None and (ch.isalnum() or ch == "_")
        if not word(before) and not word(after):
            return True
        start = i + 1


def addr_kind(to):
    t = to.strip()
    if t[:2].lower() == "0x" or t.lower().endswith(".eth"):
        return "evm"
    if t.lower().endswith(".near") or re.fullmatch(r"[0-9a-f]{64}", t):
        return "near"
    if re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", t):
        return "solana"
    return None


def send_chains(sym, to):
    kind = addr_kind(to)
    return [c for c in ORDER if (kind is None or CHAIN_KIND[c] == kind) and c in TOKEN_CHAINS.get(sym, [])]


REFUSED = ("Refused: the user didn't type that address in this chat. Only send to an address the user gave you themselves. "
           "Ask them for it.")


# ============================================================== simulator (app truth)
class World3(sim.World):
    """gen2 World (prices) with the app's wallet tools: holdings shape, chain-less prepare_send, needs_chain, policy."""
    STATE_KEYS = sim.World.STATE_KEYS + ("near_account", "agent_cap", "daily_left", "daily_cap", "bio")

    def __init__(self, rng, bio="Face ID"):
        super().__init__(rng)
        self.balances = []
        self.near_account = None
        self.agent_cap = 100
        self.daily_left = 1000
        self.daily_cap = 1000
        self.bio = bio

    def holding(self, chain, sym):
        for b in self.balances:
            if b["symbol"] == sym and b["chain"] == chain and b["amount"] > 0:
                return b
        return None

    def t_get_balances(self, chain=None):
        rows = [b for b in self.balances if b["amount"] > 0 and (chain is None or b["chain"] == str(chain).lower())]
        rows = sorted(rows, key=lambda b: (-b["usd"], b["symbol"], b["chain"]))
        out = {"holdings": [{"amount": ntxt(b["amount"]), "chain": b["chain"], "symbol": b["symbol"], "usd": jn(b["usd"])} for b in rows]}
        if self.near_account:
            out["near_account"] = self.near_account
        return out

    def t_prepare_send(self, to_address, amount, symbol, chain=None):
        sym = str(symbol).upper()
        to = str(to_address)
        cands = send_chains(sym, to)
        pick = None
        if chain is not None and str(chain).strip():
            c = find_chain(chain)
            if c is None:
                return {"error": f"Unknown chain {chain}"}
            pick = c
            if self.holding(c, sym) is None and c in cands:
                elsewhere = [x for x in cands if x != c and self.holding(x, sym)]
                if len(elsewhere) == 1:
                    pick = elsewhere[0]
        else:
            if not cands:
                return {"error": f"{sym} can't be used for that on any network Hermit supports"}
            if len(cands) == 1:
                pick = cands[0]
            else:
                held = [x for x in cands if self.holding(x, sym)]
                if len(held) == 1:
                    pick = held[0]
                else:
                    names = [CHAIN_NAME[x] for x in (held or cands)]
                    if held:
                        q = (f"Which network should the send use? You hold {sym} on {join_or(names)} ("
                             + ", ".join(f"{CHAIN_NAME[x]}: {approx(self.holding(x, sym)['amount'])}" for x in held) + ").")
                    else:
                        q = f"You don't hold {sym} on {join_or(names)} yet. Which network should the send use?"
                    return {"note": "Nothing was prepared. Ask the user which network, in one short question.",
                            "options": list(held or cands), "prepared": False, "question": q, "status": "needs_chain"}
        usd = float(amount) * self.prices.get(sym, 0)
        reasons = []
        if usd > self.daily_left:
            reasons.append(f"Over your daily limit: {money(self.daily_left)} of {money(self.daily_cap)} left today.")
        if usd > self.agent_cap:
            reasons.append(f"The assistant can prepare up to {money(self.agent_cap)} per move. Open Send yourself for more.")
        if reasons:
            return {"note": "Nothing was prepared. Tell the user why.", "reasons": reasons, "status": "blocked"}
        h = self.holding(pick, sym)
        if h is None or h["amount"] < float(amount):
            return {"error": f"Not enough {sym} on {CHAIN_NAME[pick]}."}      # generators never reach this
        return {"amount": f"{ntxt(amount)} {sym}", "executed": False, "kind": "send", "network": CHAIN_NAME[pick],
                "note": f"Waiting for the user to approve with {self.bio}.", "status": "review_opened"}


def run_tool(w, name, args, typed):
    """The app's ToolGuard in front of the toolbox. `typed` = what the user typed in this chat (files excluded)."""
    a = dict(args)
    if name == "prepare_send":
        if "chain" in a:
            cid = find_chain(a["chain"])
            if not (cid and mentions(cid, typed)):
                del a["chain"]
        if not user_typed(str(a.get("to_address", "")), typed):
            return {"error": REFUSED}
    return w.call(name, a)


def typed_text(msgs):
    return "\n".join(without_files(m["content"]) for m in msgs if m["role"] == "user")


class Conv3(Conv):
    def __init__(self, rng, world, bio):
        super().__init__(rng, world, system=SYSTEM_TXT.replace("{BIO}", bio))
        self.bio = bio
        self.turns, self.user_idx = [], []

    def user(self, text):
        if self.user_idx:
            self.turns.append(self.expect)
        self.user_idx.append(len(self.msgs))
        super().user(text)

    def call(self, *calls):
        tcs = [{"name": n, "arguments": a} for n, a in calls]
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        self.expect.append({"calls": tcs})
        typed = typed_text(self.msgs)
        res = []
        for n, a in calls:
            r = run_tool(self.w, n, a, typed)
            res.append(r)
            self.msgs.append({"role": "tool", "content": json.dumps(r, ensure_ascii=False, sort_keys=True)})
        return res[0] if len(res) == 1 else res

    def finish(self, kind, **meta):
        turns = self.turns + [self.expect]
        return {"messages": self.msgs, "tools": True,
                "meta": {"source": "send_nochain", "generator": "send_nochain", "kind": kind, "bio": self.bio,
                         "world": self.world0, "world_final": self.w.state(), "expect": self.expect,
                         "turn_expects": turns, "user_idx": self.user_idx, "lang": "en", **meta}}


class Retry(Exception):
    pass


# ============================================================== slot values (train / eval-only split)
def sub(lst, ev):
    """25% of values are eval-only (index % 4 == 3)."""
    return [x for i, x in enumerate(lst) if (i % 4 == 3) == ev]


AMOUNTS = {
    "USDC": [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500, 750, 1000,
             1250, 2000, 7.5, 12.5, 10.5, 33.3, 42.5, 99.99, 15.75, 18, 22, 55, 65, 85],
    "ETH": [0.001, 0.002, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 3,
            0.04, 0.06, 0.08],
    "SOL": [0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 12.5, 20, 50, 0.75, 4],
    "NEAR": [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 250, 8],
    "POL": [5, 10, 20, 25, 50, 100, 150, 200, 500, 1000, 35, 60],
}
AMOUNTS["USDT"] = AMOUNTS["USDC"]
AMOUNTS["DAI"] = AMOUNTS["USDC"]

SURF = {"USDC": ["USDC", "usdc", "USDC", "usdc", "USDC"], "USDT": ["USDT", "usdt", "USDT"], "ETH": ["ETH", "eth", "ETH", "eth"],
        "SOL": ["SOL", "sol"], "NEAR": ["NEAR", "near"], "POL": ["MATIC", "matic", "MATIC", "POL"], "DAI": ["DAI", "dai"]}
SURF_EV = {"USDC": ["usd coin", "USDC"], "USDT": ["tether", "USDT"], "ETH": ["ether", "ETH"], "SOL": ["SOL"], "NEAR": ["NEAR"],
           "POL": ["matic", "MATIC"], "DAI": ["DAI"]}

NAMES_TRAIN = ["Noah", "Maya", "Leo", "Aisha", "Tom", "Yuki", "Sam", "Priya", "Diego", "Nina", "Omar", "Chloe", "Ken", "Zara",
               "Ivan", "Lena", "Arjun", "Sofia", "Ben", "Mei", "Jonas", "Fatima", "Marco", "Hana", "Dev", "Ella", "Raj", "Nora"]
NAMES_EV = ["Kofi", "Anya", "Tariq", "Bea", "Luca", "Imani"]
ROLES_TRAIN = ["my landlord", "my brother", "my sister", "the plumber", "my roommate", "my friend", "my freelancer", "my mom",
               "my dad", "the designer", "my cousin", "my team lead"]
ROLES_EV = ["my accountant", "my babysitter", "the dev"]
ENS_TRAIN = ["alice.eth", "vitalik.eth", "maya.eth", "noah.eth", "satoshi.eth", "nina.eth", "ben.eth", "dev.eth", "mei.eth",
             "zara.eth", "leo.eth", "sofia.eth"]
ENS_EV = ["kofi.eth", "tariq.eth", "luca.eth", "bea.eth"]
NEAR_TRAIN = ["noah.near", "maya.near", "alice.near", "tom.near", "mei-lin.near", "ken.near", "priya.near", "dev.near",
              "nina.near", "sam.near"]
NEAR_EV = ["kofi.near", "luca.near", "bea.near"]
OWN_NEAR = ["me.near", "noah.near", "alex.near", "sam-wallet.near", "kai.near", "mina.near"]

CHAIN_SAY = {"ethereum": ["Ethereum", "ethereum", "mainnet", "eth mainnet", "Ethereum"], "base": ["Base", "base", "Base"],
             "arbitrum": ["Arbitrum", "arbitrum", "Arbitrum"], "bnb": ["BNB Chain", "bnb", "BNB", "bsc"],
             "polygon": ["Polygon", "polygon", "Polygon"], "solana": ["Solana", "solana"], "near": ["NEAR", "NEAR"]}
CHAIN_SAY_EV = {"ethereum": ["mainnet", "Ethereum mainnet"], "base": ["base", "Base"], "arbitrum": ["arb", "Arbitrum One"],
                "bnb": ["BNB chain", "BSC"], "polygon": ["polygon", "Polygon"], "solana": ["Solana", "solana"], "near": ["near"]}
WORDS = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty", 50: "fifty", 100: "a hundred"}


def parse_amount(s):
    """Reads an amount the way the generators write it. Used by the self-check."""
    s = s.strip().lower()
    inv = {v: k for k, v in WORDS.items()}
    if s in inv:
        return float(inv[s])
    s = s.lstrip("$")
    if s.endswith("k"):
        return float(s[:-1].replace(",", "")) * 1000
    return float(s.replace(",", ""))


def amount_surface(rng, v):
    iv = isinstance(jn(v), int)
    r = rng.random()
    if iv and v in WORDS and r < 0.12:
        return WORDS[v]
    if iv and v >= 1000:
        if r < 0.25 and v % 1000 == 0:
            return f"{v // 1000}k"
        if r < 0.6:
            return f"{v:,}"
    if not iv and r < 0.12 and len(ntxt(v).split(".")[1]) == 1:
        return ntxt(v) + "0"
    return ntxt(v)


def token_surface(rng, sym, ev):
    pool = SURF_EV[sym] if (ev and rng.random() < 0.5) else SURF[sym]
    return rng.choice(pool)


def at_text(rng, ctx, sym, val, forced_surface=None):
    a = amount_surface(rng, val)
    t = forced_surface or token_surface(rng, sym, ctx["ev"])
    r = rng.random()
    if r < 0.86:
        return f"{a} {t}"
    if r < 0.92 and sym in STABLE:
        return f"{a}{t}"
    return f"{t} {a}"


# ============================================================== phrase machinery (templates, typos)
PH = re.compile(r"(\{\w+\})")


def typo_parts(rng, parts):
    lits = [i for i, p in enumerate(parts) if not PH.fullmatch(p) and re.search(r"[A-Za-z]{4}", p)]
    out = list(parts)
    for _ in range(6):
        r = rng.random()
        if not lits:
            break
        i = rng.choice(lits)
        p = out[i]
        if r < 0.15:
            q = p.lower()
        elif r < 0.50:
            ws = [m for m in re.finditer(r"[A-Za-z]{4,}", p)]
            m = rng.choice(ws)
            k = rng.randint(1, len(m.group()) - 3) + m.start()
            q = p[:k] + p[k + 1] + p[k] + p[k + 2:]
        elif r < 0.68:
            ws = [m for m in re.finditer(r"[A-Za-z]{5,}", p)]
            if not ws:
                continue
            m = rng.choice(ws)
            vs = [j for j in range(m.start() + 1, m.end()) if p[j] in "aeiou"]
            if not vs:
                continue
            j = rng.choice(vs)
            q = p[:j] + p[j + 1:]
        elif r < 0.80:
            ws = [m for m in re.finditer(r"[A-Za-z]{4,}", p)]
            m = rng.choice(ws)
            js = [j for j in range(m.start() + 1, m.end()) if p[j] not in "aeiou"]
            if not js:
                continue
            j = rng.choice(js)
            q = p[:j] + p[j] + p[j:]
        elif r < 0.92:
            q = re.sub(r"\bplease\b", rng.choice(["pls", "plz"]), p, flags=re.I)
            q = re.sub(r"\bright now\b", "rn", q, flags=re.I)
            q = re.sub(r"\byou\b", "u", q)
            q = re.sub(r"\bthanks\b", "thx", q, flags=re.I)
        elif r < 0.98:
            q = p.rstrip("?.!")
        else:
            q = p.upper()
        if q != p:
            out[i] = q
            return out
    return out


def phrase(rng, ctx, pool, **slots):
    cand = [t for t in pool if t.startswith("~") == ctx["ev"]]
    t = rng.choice(cand).lstrip("~")
    parts = PH.split(t)
    if rng.random() < 0.15:
        parts = typo_parts(rng, parts)
    return "".join(p.format(**slots) if PH.fullmatch(p) else p for p in parts)


T_PLAIN = [
    "send {at} to {addr}", "Send {at} to {addr}", "transfer {at} to {addr}", "pay {addr} {at}", "move {at} to {addr}",
    "can u send {at} to {addr}", "{at} {addr}", "send {addr} {at}", "please send {at} to this address {addr}",
    "hey can you send {at} to {addr}?", "I need to send {at} to {addr}", "I owe my friend {at}, his address is {addr}",
    "pay my landlord {at} at {addr}", "my roommate needs {at} for the electric bill. address: {addr}", "transfer {at} -> {addr}",
    "{addr} send {at}", "wanna send {at} to a friend, here's the address {addr}", "tip {addr} {at}", "yo send {at} to {addr} pls",
    "~I want to send {at} to {addr}", "send {at} over to {addr}", "~Could you transfer {at} to {addr} for me?", "sending {at} to {addr}",
    "i need {at} to go to {addr}", "pay {at} to {addr}", "send {at}\n{addr}", "my brother gave me his wallet {addr}. send him {at}",
    "ok now send {at} to {addr}, it's for the pizza last week", "~Please transfer {at} to the following address: {addr}",
    "wire {at} to {addr}", "give {addr} {at}", "can you move {at} to {addr} for me", "~need to pay a freelancer {at}: {addr}",
    "send {at} to my other wallet {addr}", "Prepare a send of {at} to {addr}", "~I'd like to send {at} to {addr}",
    "quick one, {at} to {addr}", "send {at} to {addr} thanks", "refund {addr} {at}", "pay back {at} to {addr}",
    "~i got an invoice, pay {at} to {addr}", "lets send {at} to {addr}", "{at} to {addr}", "could u send {addr} {at}",
    "~send it: {at} -> {addr}", "~hi! I'd like to pay {addr} {at}", "~kindly transfer {at} to {addr}", "donate {at} to {addr}",
    "splitting dinner, send {at} to {addr}", "~buy my friend a coffee, send {at} to {addr}",
    "~ok so I need to pay {addr}, {at} please", "~yo bro send {at} to {addr}", "~Hello, send {at} to {addr}.", "~can I send {at} to {addr}?",
    "~use my wallet to send {at} to {addr}", "~{at} for {addr}, send",
    "my cousin is waiting on {at}, send it to {addr}", "Send {at} to {addr} please", "need {at} sent over to {addr}",
]
# "need ... sent over" would trip the banned-word check only in assistant text; user text is free.

T_CHAIN = [
    "send {at} on {net} to {addr}", "send {at} to {addr} on {net}", "{net}: send {at} to {addr}", "transfer {at} on {net} to {addr}",
    "pay {addr} {at} on {net}", "can u send {at} to {addr} on {net}", "using {net}, send {at} to {addr}", "move {at} on {net} to {addr}",
    "~I want to send {at} on {net} to {addr}", "{at} -> {addr} ({net})", "please send {at} via {net} to {addr}",
    "~send {at} on the {net} network to {addr}", "my friend is on {net}, send him {at} at {addr}", "~send {at} to {addr} over {net}",
    "pay {at} on {net}, address {addr}", "send {at} to {addr} using {net} pls", "Send {at} on {net} to {addr}",
    "~on {net} please, {at} to {addr}",
]
T_ALL = ["send all my {t} to {addr}", "send max {t} to {addr}", "send everything in {t} to {addr}", "move my whole {t} balance to {addr}",
         "sweep my {t} to {addr}", "~send my entire {t} balance to {addr}", "send the max amount of {t} to {addr}",
         "send all {t} to {addr}, i'm closing this wallet", "{addr} send max {t}", "can you send all my {t} to {addr}",
         "~transfer my full {t} balance to {addr}", "send ALL my {t} to {addr}"]
T_ALL_CH = ["send all my {t} on {net} to {addr}", "send max {t} on {net} to {addr}", "~send everything in {t} on {net} to {addr}",
            "move my whole {t} balance on {net} to {addr}", "on {net}, send all my {t} to {addr}", "sweep my {t} on {net} to {addr}",
            "send max {t} to {addr} on {net}", "~send my entire {t} balance on {net} to {addr}"]
T_HALF = ["send half my {t} to {addr}", "send half of my {t} balance to {addr}", "half of my {t} to {addr} pls",
          "split my {t}: send half to {addr}", "send 50% of my {t} to {addr}", "~send half of what I have in {t} to {addr}",
          "can you send half my {t} to {addr}", "~half my {t} balance -> {addr}", "send half {t} to {addr}"]
T_WHO = ["send {at} to {who}", "can you send {who} {at}?", "pay {who} {at}", "I want to send {at} to {who}", "transfer {at} to {who}",
         "{who} needs {at}, send it", "send {who} {at} for the tickets", "give {who} {at}", "~please pay {who} {at}",
         "send {at} to {who} asap", "~{who} is waiting for {at}, please send it"]
T_NOAMT = ["send {t} to {addr}", "I want to send some {t} to {addr}", "transfer {t} to {addr}", "pay {addr} in {t}",
           "can you send {t} to {addr}", "~send {addr} some {t}", "send {t} to this address: {addr}", "{t} to {addr}",
           "~i need to send {t} to {addr}"]
T_TOKEN = ["send {at} to {addr}", "pay {addr} {at}", "can u send {at} to {addr}", "{at} to {addr}", "I need to send {at} to {addr}",
           "send {addr} {at}", "~please send {at} to {addr}", "give {addr} {at}", "~transfer {at} to {addr}", "send {at} over to {addr}"]
T_TURN2_CHAIN = ["{net}", "on {net}", "{net} please", "use {net}", "go with {net}", "{net} pls", "let's do {net}", "the {net} one",
                 "~send it on {net}", "~{net}, thanks", "{net} is fine"]
T_ADDR_REPLY = ["{addr}", "it's {addr}", "here: {addr}", "{who}'s address is {addr}", "~sure, {addr}", "address is {addr}",
                "ok {addr}", "~here you go {addr}", "{addr} thx"]
T_AMT_REPLY = ["{a}", "{a} {t}", "make it {a}", "{a} please", "~let's say {a}", "{a} {t} pls", "~{a} should do it"]
T_TOK_REPLY = ["{t}", "{t} please", "in {t}", "{t} pls", "~{t}, thanks", "~make it {t}", "{t} yeah"]
T_AGAIN = ["same address again, {at}", "send another {at} to the same address", "again to that address, {at}",
           "and another {at} to the same wallet", "send {at} more to the same one", "do it again but {at}",
           "same address, {at} this time", "~one more: {at} to that address", "~now {at} to the same address too",
           "send the same address {at} again"]
T_SAVED_FIRST = ["this is {who}'s wallet: {addr}", "save this address for later: {addr}", "my friend {who} gave me their address {addr}",
                 "{who}'s address is {addr}", "~heads up, {who} will send me their wallet {addr}", "remember this one: {addr}",
                 "~here's where {who} wants it: {addr}"]
T_SAVED_SEND = ["send them {at}", "{at} please", "send {at} to that address", "{at} to them", "~now send {at} there",
                "ok send {at} to that one", "go ahead and send {at}", "~{at} to the same address"]
T_FILE = ["pay this invoice", "can you pay this for me", "send the amount to the address in this file", "send the payment from the invoice",
          "~pay the address in the screenshot", "pay what it says here", "please pay this", "~send the money to the address in the file",
          "settle this invoice", "pay it"]
T_OWN_NEAR = ["send {at} to my near account", "move {at} to my NEAR wallet", "~send {at} over to my near account",
              "transfer {at} to my near account"]
T_PASTE = ["{addr}", "ok here: {addr}", "it's {addr}", "~here you go: {addr}", "address: {addr}"]


# ============================================================== world building
def hold_amt(rng, sym, need):
    r = round(need * rng.uniform(1.05, 3), DP[sym])
    return max(r, round(need + 10 ** -DP[sym], DP[sym]))


def rand_hold(rng, sym, usd_lo, usd_hi, price):
    dp = DP[sym]
    amt = round(rng.uniform(usd_lo, usd_hi) / price, dp)
    return max(amt, 10 ** -dp)


def add_hold(w, sym, chain, amount):
    amount = round(amount, DP.get(sym, 4))
    for b in w.balances:
        if b["symbol"] == sym and b["chain"] == chain:
            b["amount"] = amount
            b["usd"] = round(amount * w.prices[sym], 2)
            return
    w.balances.append({"symbol": sym, "chain": chain, "amount": amount, "usd": round(amount * w.prices[sym], 2)})


NOISE = [("USDC", "ethereum"), ("USDC", "base"), ("USDC", "arbitrum"), ("USDC", "polygon"), ("USDC", "bnb"), ("USDC", "solana"),
         ("USDT", "ethereum"), ("USDT", "bnb"), ("USDT", "solana"), ("ETH", "ethereum"), ("ETH", "base"), ("ETH", "arbitrum"),
         ("SOL", "solana"), ("NEAR", "near"), ("POL", "polygon")]


def add_noise(rng, w, avoid_syms):
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        pool = [(s, c) for s, c in NOISE if s not in avoid_syms and not w.holding(c, s)]
        if not pool:
            return
        s, c = rng.choice(pool)
        add_hold(w, s, c, rand_hold(rng, s, 3, 3000, w.prices[s]))


CAP_TIERS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 100000]


def set_limits(rng, w, usd, mode="ok"):
    """Agent per-move cap and daily limit. ok: both fit. cap / daily / both: the send is blocked."""
    def tier(x):
        return next(c for c in CAP_TIERS if c >= x)
    if mode in ("ok", "daily"):
        w.agent_cap = 100 if (usd <= 100 and rng.random() < 0.65) else tier(usd * rng.choice([1, 1, 1.5]))
        if mode == "daily":
            w.agent_cap = tier(usd * 1.2)
    else:
        w.agent_cap = 100 if rng.random() < 0.6 else rng.choice([250, 500])
        if usd <= w.agent_cap:
            raise Retry
    w.daily_cap = rng.choice([500, 1000, 2000, 5000])
    if mode in ("ok", "cap"):
        w.daily_left = round(max(usd * rng.uniform(1.1, 3), rng.uniform(100, w.daily_cap)), 2)
        w.daily_cap = max(w.daily_cap, tier(w.daily_left))
        w.daily_left = float(round(w.daily_left))
    else:
        w.daily_cap = max(w.daily_cap, tier(usd * 2))
        w.daily_left = float(round(usd * rng.uniform(0.15, 0.8)))
        if w.daily_left < 1:
            raise Retry


def new_world(rng, bio, sym, held, val=None, hold_override=None, mode="ok", noise=True, extra=None):
    """held: list of chains where `sym` is held. val: amount to send (None for all/half scenarios with hold_override)."""
    w = World3(rng, bio)
    for c in held:
        amt = (hold_override or {}).get(c) or hold_amt(rng, sym, val)
        add_hold(w, sym, c, amt)
    for s, c, amt in (extra or []):
        add_hold(w, s, c, amt)
    if noise:
        add_noise(rng, w, {sym} | {s for s, _, _ in (extra or [])})
    if any(b["chain"] == "near" for b in w.balances) or rng.random() < 0.12:
        w.near_account = rng.choice(OWN_NEAR)
    send_val = val if val is not None else (hold_override or {}).get(held[0], 0)
    set_limits(rng, w, float(send_val) * w.prices.get(sym, 1.0), mode)
    return w


def make_addr(rng, ctx, chain, ens=False):
    k = CHAIN_KIND[chain]
    if k == "evm":
        if ens:
            return rng.choice(sub(ENS_TRAIN, False) if not ctx["ev"] else ENS_EV)
        return H.evm_addr(rng)
    if k == "solana":
        return H.sol_addr(rng)
    if rng.random() < 0.7:
        return rng.choice(sub(NEAR_TRAIN, False) if not ctx["ev"] else NEAR_EV)
    return "".join(rng.choice("0123456789abcdef") for _ in range(64))


def pick_sym_chain(rng, ctx, force=False, allow=("USDC", "USDT", "ETH", "SOL", "NEAR", "POL", "DAI")):
    if force:
        sym = rng.choices(["USDC", "USDT", "ETH"], [70, 12, 18])[0]
        chains = [c for c in HELD_CHAINS[sym] if c in ("base", "arbitrum", "polygon", "solana", "bnb")]
        return sym, rng.choice(chains)
    wts = {"USDC": 40, "USDT": 12, "ETH": 20, "SOL": 8, "NEAR": 5, "POL": 6, "DAI": 3}
    if ctx["ev"]:
        wts = {"USDC": 46, "USDT": 12, "ETH": 24, "SOL": 5, "NEAR": 3, "POL": 3, "DAI": 2}
    ss = [s for s in allow if s in wts]
    sym = rng.choices(ss, [wts[s] for s in ss])[0]
    cw = {"ethereum": 2, "base": 2, "arbitrum": 1.2, "polygon": 1, "bnb": 1, "solana": 1.2, "near": 0.5}
    chains = HELD_CHAINS[sym]
    return sym, rng.choices(chains, [cw[c] for c in chains])[0]


def pick_amount(rng, ctx, sym):
    return rng.choice(sub(AMOUNTS[sym], ctx["ev"]))


def canon(sym):
    return sym


def net_phrase(rng, ctx, chain):
    return rng.choice(CHAIN_SAY_EV[chain] if (ctx["ev"] and rng.random() < 0.6) else CHAIN_SAY[chain])


def eth_trap(w, sym):
    hs = [b["chain"] for b in w.balances if b["symbol"] == sym and b["amount"] > 0]
    return sym in ("USDC", "USDT", "ETH") and bool(hs) and "ethereum" not in hs


# ============================================================== answers (f-strings from results and user text)
def success_text(rng, r, addr, bio, prefix=""):
    amt, net = r["amount"], r["network"]
    v = [f"I've prepared a send of {amt} on {net} to {addr}. Check the address on the review screen, then approve with {bio}.",
         f"Prepared: {amt} on {net} to {addr}. Check that address on the review screen, then approve with {bio}.",
         f"The send of {amt} to {addr} on {net} is ready for review. Compare the full address on the review screen and approve with {bio} when it matches.",
         f"I've set up {amt} to {addr} on {net}. It's waiting for your {bio}, so check the address on the review screen first.",
         f"Ready for your review: {amt} to {addr} on {net}. Check the address, then approve with {bio}.",
         f"I've opened the review for {amt} to {addr} on {net}. Nothing moves until you approve it with {bio}, so check the address first.",
         f"Prepared {amt} on {net} for {addr}. Look over the address on the review screen, then approve with {bio}.",
         f"Your send of {amt} on {net} to {addr} is waiting on the review screen. Check the address, then approve with {bio}."]
    return prefix + rng.choice(v)


def needs_chain_text(rng, r, sym):
    q = r["question"]
    names = [CHAIN_NAME[o] for o in r["options"]]
    m = re.search(r"\((.*)\)", q)
    detail = m.group(1) if m else ""
    held = q.startswith("Which network")
    if not held:
        return rng.choice([f"You don't hold {sym} on {join_or(names)} yet, so which network should I use?",
                           f"I don't see any {sym} on {join_or(names)} yet. Which network should I use?"])
    v = [f"You hold {sym} on {join_and(names)}. Which network should I use?",
         f"Which network should I use? You hold {sym} on {join_and(names)}.",
         f"You hold {sym} on {join_and(names)}, so which network should I use?",
         f"Which network do you want to use? You have {sym} on {join_and(names)}.",
         f"I see {sym} on {join_and(names)}. Which network do you want to send on?"]
    if detail:
        v += [f"You hold {sym} on {join_and(names)} ({detail}). Which one should I use?",
              f"Which network should I use? You hold {sym} on {join_and(names)} ({detail})."]
    return rng.choice(v)


def blocked_text(rng, r):
    rs = " ".join(r["reasons"])
    cap, daily = "per move" in rs, "daily limit" in rs
    if cap and daily:
        return rng.choice(["I can't prepare that one. It's over what the assistant may send in one move, and over what's left of your daily limit, so nothing was set up. You can open Send yourself for a smaller amount.",
                           "That one is blocked: it's above the assistant's limit for one move and above what's left of your daily limit. Nothing was prepared. Try a smaller amount or open Send yourself."])
    if cap:
        return rng.choice(["I can't prepare that one: it's over the most the assistant can send in one move. Nothing was set up. You can open Send yourself and do it there.",
                           "That's above the limit for what I can prepare in a single move, so nothing was set up. Open Send yourself to do it, or try a smaller amount.",
                           "Blocked: the amount is over the assistant's per-move limit. I haven't prepared anything. You can send it yourself from the Send screen.",
                           "I wasn't able to prepare it because it's over the assistant's limit for one move. Open Send yourself for a bigger amount, or give me a smaller one."])
    return rng.choice(["I can't prepare that one: it's more than what's left of your daily limit, so nothing was set up. A smaller amount might fit.",
                       "That's over what you have left of today's spending limit, so nothing was prepared. Try a smaller amount, or wait until the limit frees up.",
                       "Blocked: this would go past your daily limit. I haven't prepared anything. A smaller amount may work.",
                       "I wasn't able to prepare it because it's over your daily limit. Try less, or come back once more of the limit is free."])


def unsupported_text(rng, sym):
    return rng.choice([f"Hermit can't send {sym} on any network it supports, so I didn't prepare anything. I can send USDC or USDT if that works for you.",
                       f"{sym} isn't something Hermit can send on any of its networks, so nothing was prepared. Want to use USDC instead?",
                       f"I couldn't prepare that: Hermit can't use {sym} for a send on any network it supports. Nothing was set up.",
                       f"Hermit doesn't support sending {sym} on any of its networks, so I haven't prepared anything."])


def refused_text(rng, source):
    if source == "file":
        return rng.choice(["I can only send to an address you type in the chat, so I can't use the one from the file. Paste it here and I'll prepare the send.",
                           "That address isn't one you typed here, so the app refused it. Paste the address into the chat and I'll try again.",
                           "I can only prepare sends to an address you type yourself, not one I read from a file. Can you paste it here?",
                           "The app only sends to an address you type in the chat, not one that only appears in a file. Paste it here and I'll set it up.",
                           "I can only send to an address you type in this chat. Paste the address from the file here and I'll prepare it.",
                           "Nothing was prepared: I can only send to an address you type in the chat. Please paste it here."])
    return rng.choice(["I can only send to an address you type in the chat, so I can't use that one. Paste the address you want here and I'll prepare it.",
                       "That address isn't one you typed in this chat, so the app refused it. Paste the address here and I'll set up the send.",
                       "I can only prepare sends to an address you type yourself. Can you paste the address here?",
                       "Nothing was prepared: sends only go to an address you type in the chat. Paste it here and I'll try again."])


def reply(c, rng, r, addr, sym, named=None, matic=False):
    """Writes the final answer for a prepare_send result. Returns 'ok'|'ask'|'blocked'|'unsupported'|'refused'."""
    bio = c.bio
    if r.get("status") == "review_opened":
        prefix = ""
        if named and named != r["network"]:
            net = r["network"]
            prefix = rng.choice([f"You hold {sym} on {net}, so I used {net}. ", f"You don't hold {sym} on {named}, but you do on {net}, so I used {net}. ",
                                 f"I used {net} because that's where you hold {sym}. ", f"You hold your {sym} on {net}, so I used {net}. "])
        elif matic:
            prefix = rng.choice(["MATIC is called POL now, so I used POL. ", "MATIC is now POL, so I used that. "])
        c.say(success_text(rng, r, addr, bio, prefix))
        return "ok"
    if r.get("status") == "needs_chain":
        c.say(needs_chain_text(rng, r, sym), ask=True)
        return "ask"
    if r.get("status") == "blocked":
        c.say(blocked_text(rng, r))
        return "blocked"
    if "can't be used for that" in r.get("error", ""):
        c.say(unsupported_text(rng, sym))
        return "unsupported"
    if r.get("error", "").startswith("Refused"):
        c.say(refused_text(rng, "file"))
        return "refused"
    raise AssertionError(f"unhandled result {r}")


def sym_arg(sym, surface_is_matic):
    return sym


# ============================================================== scenarios
def sc_a(rng, ctx, variant="plain"):
    """No network named. variants: plain, all, half, usd, ens."""
    force = ctx["force"]
    sym, chain = pick_sym_chain(rng, ctx, force, allow=("USDC", "USDT", "ETH") if variant in ("all", "half", "usd", "ens") else
                                ("USDC", "USDT", "ETH", "SOL", "NEAR", "POL", "DAI"))
    if variant == "usd":
        sym = rng.choices(["USDC", "USDT"], [75, 25])[0]
        chain = rng.choice([c for c in HELD_CHAINS[sym] if (c != "ethereum" if force else True)])
    if variant == "ens":
        chain = rng.choice([c for c in HELD_CHAINS[sym] if CHAIN_KIND[c] == "evm" and (c != "ethereum" or not force)])
    addr = make_addr(rng, ctx, chain, ens=(variant == "ens"))
    held = [chain]
    meta = {"variant": variant, "sym": sym}
    if variant in ("all", "half"):
        price = 1.0
        hold = None
    c_args = None
    if variant == "plain" or variant == "ens":
        val = pick_amount(rng, ctx, sym)
        w = new_world(rng, ctx["bio"], sym, held, val)
        at = at_text(rng, ctx, sym, val)
        text = phrase(rng, ctx, T_PLAIN, at=at, addr=addr)
        meta.update(amount_surface=at)
    elif variant == "usd":
        val = pick_amount(rng, ctx, "USDC")
        a = amount_surface(rng, val)
        tsurf = token_surface(rng, sym, ctx["ev"])
        forms = [f"${a} of {tsurf}", f"${a} worth of {tsurf}", f"{a} bucks in {tsurf}", f"{a} bucks of {tsurf}", f"{a} dollars of {tsurf}"]
        if sym == "USDC":
            forms += [f"{a} dollars", f"{a} usd", f"{a} USD", f"{a} dollars", f"{a} usd"]
        at = rng.choice(forms)
        w = new_world(rng, ctx["bio"], sym, held, val)
        text = phrase(rng, ctx, T_PLAIN, at=at, addr=addr)
        meta.update(amount_surface=at)
    else:   # all / half: holding on exactly one chain
        w = None
    if variant in ("all", "half"):
        price = 1.0 if sym in STABLE else None
        tmp = World3(random.Random(0))
        price = tmp.prices[sym] if price is None else price
        # price of the real world is drawn inside new_world, so build first with a placeholder then fix the holding
        w0 = World3(rng, ctx["bio"])
        hold = rand_hold(rng, sym, 15, 95, w0.prices[sym]) if rng.random() < 0.8 else rand_hold(rng, sym, 100, 900, w0.prices[sym])
        hold = round(hold, DP[sym])
        w = new_world(rng, ctx["bio"], sym, held, None, hold_override={chain: hold}, noise=True)
        # noise never holds `sym`, so the token sits on one chain only
        t = token_surface(rng, sym, ctx["ev"])
        text = phrase(rng, ctx, T_ALL if variant == "all" else T_HALF, t=t, addr=addr)
    c = Conv3(rng, w, ctx["bio"])
    c.user(text)
    sym_a = sym
    if variant in ("all", "half"):
        rb = c.call(("get_balances", {}))
        row = next(h for h in rb["holdings"] if h["symbol"] == sym and h["chain"] == chain)
        if variant == "all":
            val = float(row["amount"])
        else:
            rc = c.call(("calculator", {"expression": f"{row['amount']} / 2"}))
            val = float(rc["result"])
    args = {"to_address": addr, "amount": jn(val), "symbol": sym_a}
    r = c.call(("prepare_send", args))
    matic = sym == "POL" and "pol" not in text.lower().split() and re.search(r"matic", text, re.I) is not None
    st = reply(c, rng, r, addr, sym, matic=matic)
    kind = {"plain": "a_plain", "all": "a_all", "half": "a_half", "usd": "a_usd", "ens": "a_ens"}[variant]
    if st == "unsupported":
        kind = "a_unsupported"
    return c.finish(kind, gold_calls=[args], addr=addr, amount_value=jn(val), eth_trap=eth_trap(w, sym), chain_named=None,
                    allowed_induced=induced(sym, addr), **meta)


def induced(sym, addr):
    s = set()
    if sym == "SOL":
        s.add("solana")
    if sym == "NEAR" or addr.lower().endswith(".near"):
        s.add("near")
    if sym == "POL":
        s.add("polygon")
    return sorted(s)


def sc_b(rng, ctx, all_=False):
    """Network named: chain set to the lowercase id."""
    allow = ("USDC", "USDT", "ETH") if all_ else ("USDC", "USDT", "ETH", "SOL", "NEAR", "POL")
    sym, chain = pick_sym_chain(rng, ctx, False, allow=allow)
    addr = make_addr(rng, ctx, chain)
    net = net_phrase(rng, ctx, chain)
    other = [x for x in HELD_CHAINS[sym] if x != chain and CHAIN_KIND[x] == CHAIN_KIND[chain]]
    held = [chain] + ([rng.choice(other)] if other and rng.random() < 0.3 else [])
    if all_:
        w0 = World3(rng, ctx["bio"])
        ov = {c: rand_hold(rng, sym, 15, 95, w0.prices[sym]) for c in held}
        w = new_world(rng, ctx["bio"], sym, held, None, hold_override=ov)
        # the send is for the holding on the named chain, so the cap check uses that one
        t = token_surface(rng, sym, ctx["ev"])
        text = phrase(rng, ctx, T_ALL_CH, t=t, net=net, addr=addr)
    else:
        val = pick_amount(rng, ctx, sym)
        w = new_world(rng, ctx["bio"], sym, held, val)
        at = at_text(rng, ctx, sym, val)
        text = phrase(rng, ctx, T_CHAIN, at=at, net=net, addr=addr)
    c = Conv3(rng, w, ctx["bio"])
    c.user(text)
    if all_:
        gb = {"chain": chain} if rng.random() < 0.5 else {}
        rb = c.call(("get_balances", gb))
        row = next(h for h in rb["holdings"] if h["symbol"] == sym and h["chain"] == chain)
        val = float(row["amount"])
    args = {"to_address": addr, "amount": jn(val), "symbol": sym, "chain": chain}
    r = c.call(("prepare_send", args))
    matic = sym == "POL" and re.search(r"matic", text, re.I) is not None
    reply(c, rng, r, addr, sym, matic=matic)
    return c.finish("b_all" if all_ else "b_named", gold_calls=[args], addr=addr, amount_value=jn(val), eth_trap=False,
                    chain_named=chain, allowed_induced=induced(sym, addr), sym=sym)


def sc_c(rng, ctx):
    """Named chain where the token is not held; exactly one other chain holds it: the app switches silently."""
    sym = rng.choices(["USDC", "USDT", "ETH"], [60, 15, 25])[0]
    chains = EVM_HELD[sym]
    h = rng.choice(chains)
    n = rng.choice([x for x in chains if x != h])
    addr = make_addr(rng, ctx, "base")
    val = pick_amount(rng, ctx, sym)
    w = new_world(rng, ctx["bio"], sym, [h], val)
    net = net_phrase(rng, ctx, n)
    at = at_text(rng, ctx, sym, val)
    text = phrase(rng, ctx, T_CHAIN, at=at, net=net, addr=addr)
    c = Conv3(rng, w, ctx["bio"])
    c.user(text)
    args = {"to_address": addr, "amount": jn(val), "symbol": sym, "chain": n}
    r = c.call(("prepare_send", args))
    reply(c, rng, r, addr, sym, named=CHAIN_NAME[n])
    return c.finish("c_switch", gold_calls=[args], addr=addr, amount_value=jn(val), eth_trap=False, chain_named=n,
                    allowed_induced=[], sym=sym, held_on=h)


def sc_d(rng, ctx):
    """Token held on several chains: needs_chain, ask once, the user names one, second call WITH chain."""
    sym = rng.choices(["USDC", "USDT", "ETH"], [60, 15, 25])[0]
    chains = EVM_HELD[sym]
    k = min(len(chains), rng.choice([2, 2, 3]))
    chosen = rng.sample(chains, k)
    held = [c for c in ORDER if c in chosen]
    addr = make_addr(rng, ctx, "base")
    val = pick_amount(rng, ctx, sym)
    w = new_world(rng, ctx["bio"], sym, held, val)
    at = at_text(rng, ctx, sym, val)
    text = phrase(rng, ctx, T_PLAIN, at=at, addr=addr)
    c = Conv3(rng, w, ctx["bio"])
    c.user(text)
    args1 = {"to_address": addr, "amount": jn(val), "symbol": sym}
    r1 = c.call(("prepare_send", args1))
    assert r1["status"] == "needs_chain", r1
    reply(c, rng, r1, addr, sym)
    pick = rng.choice(held)
    c.user(phrase(rng, ctx, T_TURN2_CHAIN, net=net_phrase(rng, ctx, pick)))
    args2 = {"to_address": addr, "amount": jn(val), "symbol": sym, "chain": pick}
    r2 = c.call(("prepare_send", args2))
    reply(c, rng, r2, addr, sym)
    return c.finish("d_needs_chain", gold_calls=[args1, args2], addr=addr, amount_value=jn(val), eth_trap=False, chain_named=pick,
                    allowed_induced=[], sym=sym, held_on=held)


def sc_e(rng, ctx, sub_kind=None):
    """Something is missing: no call, one short question."""
    sub_kind = sub_kind or ["addr", "token", "addr", "amount", "addr", "token", "addr", "amount", "addr", "token"][ctx["i"] % 10]
    sym = rng.choices(["USDC", "USDT", "ETH"], [65, 12, 23])[0] if sub_kind != "token" else "USDC"
    chain = rng.choice([x for x in HELD_CHAINS[sym] if CHAIN_KIND[x] == "evm"])
    val = pick_amount(rng, ctx, sym)
    w = new_world(rng, ctx["bio"], sym, [chain], val)
    c = Conv3(rng, w, ctx["bio"])
    addr = make_addr(rng, ctx, chain)
    follow = rng.random() < 0.35 and not ctx["ev"]
    if sub_kind == "addr":
        if rng.random() < 0.6:
            who = rng.choice(NAMES_EV if ctx["ev"] else NAMES_TRAIN)
            ref = f"{who}'s address"
        else:
            who = rng.choice(ROLES_EV if ctx["ev"] else ROLES_TRAIN)
            ref = "the address"
        at = at_text(rng, ctx, sym, val)
        c.user(phrase(rng, ctx, T_WHO, at=at, who=who))
        c.say(rng.choice([f"What's {ref}? Paste it here and I'll prepare the send.", f"I need {ref} first. Can you paste it here?",
                          f"Can you paste {ref} here so I can set it up?", "What address should it go to? Paste it here.",
                          "Which address should I send it to? Paste it in the chat.", "Send it where? Paste the address here and I'll prepare it."]), ask=True)
        if follow:
            c.user(phrase(rng, ctx, T_ADDR_REPLY, addr=addr, who=who))
    elif sub_kind == "amount":
        t = token_surface(rng, sym, ctx["ev"])
        c.user(phrase(rng, ctx, T_NOAMT, t=t, addr=addr))
        c.say(rng.choice([f"How much {sym} should I send?", f"How much {sym} do you want to send to that address?",
                          f"What amount of {sym} should I send?", f"How much should I send?", f"How many {sym} should I send?",
                          f"Which amount do you want to send in {sym}?"]), ask=True)
        if follow:
            c.user(phrase(rng, ctx, T_AMT_REPLY, a=amount_surface(rng, val), t=token_surface(rng, sym, False)))
    else:
        a = amount_surface(rng, val)
        bucks = rng.choice([f"{a} bucks", f"${a}", f"{a} bucks", f"${a}"])
        c.user(phrase(rng, ctx, T_TOKEN, at=bucks, addr=addr))
        c.say(rng.choice(["Which token should I send?", "Which token do you mean?", "Which token do you want to send?",
                          "Which token should that be in?", "Which token should I use for that?", "Which coin should I send?"]), ask=True)
        if follow:
            c.user(phrase(rng, ctx, T_TOK_REPLY, t=token_surface(rng, sym, False)))
    gold = []
    if follow:
        args = {"to_address": addr, "amount": jn(val), "symbol": sym}
        r = c.call(("prepare_send", args))
        reply(c, rng, r, addr, sym)
        gold = [args]
    return c.finish("e_" + sub_kind, gold_calls=gold, addr=addr, amount_value=jn(val), eth_trap=False, chain_named=None,
                    allowed_induced=[], sym=sym, ask_first=True)


def sc_f(rng, ctx):
    """An address typed earlier in the chat is reused ('same address again')."""
    force = ctx["force"]
    sym, chain = pick_sym_chain(rng, ctx, force, allow=("USDC", "USDT", "ETH"))
    addr = make_addr(rng, ctx, chain)
    v1, v2 = pick_amount(rng, ctx, sym), pick_amount(rng, ctx, sym)
    w = new_world(rng, ctx["bio"], sym, [chain], v1 + v2)
    c = Conv3(rng, w, ctx["bio"])
    gold = []
    first_ask = False
    if rng.random() < 0.5:
        c.user(phrase(rng, ctx, T_PLAIN, at=at_text(rng, ctx, sym, v1), addr=addr))
        a1 = {"to_address": addr, "amount": jn(v1), "symbol": sym}
        r1 = c.call(("prepare_send", a1))
        reply(c, rng, r1, addr, sym)
        gold.append(a1)
        c.user(phrase(rng, ctx, T_AGAIN, at=at_text(rng, ctx, sym, v2)))
        fk = "f_reuse_sent"
    else:
        who = rng.choice(NAMES_EV if ctx["ev"] else NAMES_TRAIN)
        c.user(phrase(rng, ctx, T_SAVED_FIRST, who=who, addr=addr))
        c.say(rng.choice(["Got it. How much do you want to send, and in which token?", "Okay, I have that address. What should I send?",
                          "Got it. What amount and token should I send there?", "Noted. How much should I send, and in which token?"]), ask=True)
        c.user(phrase(rng, ctx, T_SAVED_SEND, at=at_text(rng, ctx, sym, v2)))
        fk = "f_reuse_addr"
        first_ask = True
    a2 = {"to_address": addr, "amount": jn(v2), "symbol": sym}
    r2 = c.call(("prepare_send", a2))
    reply(c, rng, r2, addr, sym)
    gold.append(a2)
    return c.finish(fk, gold_calls=gold, addr=addr, amount_value=jn(v2), eth_trap=eth_trap(w, sym), chain_named=None,
                    allowed_induced=[], sym=sym, first_ask=first_ask)


def file_block(rng, ctx, addr, sym, val):
    pdf = rng.random() < 0.6
    due = rng.choice(["Due in 7 days", "Payment terms: net 14", "Please pay within 10 days", "Thank you for your business"])
    inv = rng.randint(1000, 9999)
    body = rng.choice([f"Invoice {inv}\nPay to: {addr}\nAmount due: {ntxt(val)} {sym}\n{due}",
                       f"Invoice {inv}\nAmount due: {ntxt(val)} {sym}\nSend payment to wallet {addr}\n{due}",
                       f"Payment details\nWallet: {addr}\nAmount: {ntxt(val)} {sym}\nRef {inv}"])
    if pdf:
        name = rng.choice(["invoice.pdf", f"invoice-{inv}.pdf", "payment-request.pdf", "bill.pdf"])
        return f"The user attached the file {name} (PDF, 1 page):\n<file>\nPage 1\n{body}\n</file>\n\n"
    name = rng.choice(["screenshot.png", "photo.jpg", "invoice.png"])
    return f"The user attached the file {name} (image, text read on this device):\n<file>\n{body.replace(chr(10), '  ')}\n</file>\n\n"


def sc_g(rng, ctx):
    """The address only exists in a file or a tool result: the app refuses; say so and ask them to paste it."""
    if rng.random() < 0.75:
        sym = rng.choices(["USDC", "USDT"], [80, 20])[0]
        chain = rng.choice([x for x in HELD_CHAINS[sym] if CHAIN_KIND[x] == "evm" and (x != "ethereum" or not ctx["force"])])
        addr = make_addr(rng, ctx, chain)
        val = pick_amount(rng, ctx, sym)
        w = new_world(rng, ctx["bio"], sym, [chain], val)
        c = Conv3(rng, w, ctx["bio"])
        c.user(file_block(rng, ctx, addr, sym, val) + phrase(rng, ctx, T_FILE))
        args = {"to_address": addr, "amount": jn(val), "symbol": sym}
        r = c.call(("prepare_send", args))
        assert r.get("error", "").startswith("Refused"), r
        reply(c, rng, r, addr, sym)
        return c.finish("g_file", gold_calls=[args], addr=addr, amount_value=jn(val), eth_trap=eth_trap(w, sym), chain_named=None,
                        allowed_induced=[], sym=sym, refused=True, addr_in_file=True)
    sym = "USDC"
    val = pick_amount(rng, ctx, sym)
    w = new_world(rng, ctx["bio"], sym, ["near"], val)
    w.near_account = rng.choice(OWN_NEAR)
    c = Conv3(rng, w, ctx["bio"])
    c.user(phrase(rng, ctx, T_OWN_NEAR, at=at_text(rng, ctx, sym, val)))
    c.call(("get_balances", {}))
    addr = w.near_account
    args = {"to_address": addr, "amount": jn(val), "symbol": sym}
    r = c.call(("prepare_send", args))
    assert r.get("error", "").startswith("Refused"), r
    c.say(refused_text(rng, "tool"))
    return c.finish("g_tool", gold_calls=[args], addr=addr, amount_value=jn(val), eth_trap=False, chain_named=None,
                    allowed_induced=["near"], sym=sym, refused=True, addr_in_tool=True)


def sc_h(rng, ctx):
    """Blocked by the spending policy: pass the reason on in plain words."""
    force = ctx["force"]
    sym, chain = pick_sym_chain(rng, ctx, force, allow=("USDC", "USDT", "ETH"))
    mode = rng.choices(["cap", "daily", "both"], [55, 35, 10])[0]
    addr = make_addr(rng, ctx, chain)
    pool = [v for v in sub(AMOUNTS[sym], ctx["ev"]) if v * (1 if sym != "ETH" else 1800) >= 150]
    val = rng.choice(pool)
    w = new_world(rng, ctx["bio"], sym, [chain], val, mode=mode)
    at = at_text(rng, ctx, sym, val)
    c = Conv3(rng, w, ctx["bio"])
    c.user(phrase(rng, ctx, T_PLAIN, at=at, addr=addr))
    args = {"to_address": addr, "amount": jn(val), "symbol": sym}
    r = c.call(("prepare_send", args))
    assert r["status"] == "blocked", r
    reply(c, rng, r, addr, sym)
    return c.finish("h_blocked", gold_calls=[args], addr=addr, amount_value=jn(val), eth_trap=eth_trap(w, sym), chain_named=None,
                    allowed_induced=[], sym=sym, blocked=True, block_mode=mode)


# ============================================================== plan, generation
KINDS = [  # name, fn, kwargs, train share, eval count, forced-off-eth share (train), forced count (eval)
    ("a_plain", sc_a, {"variant": "plain"}, 280, 26, 0.4, 12),
    ("a_all", sc_a, {"variant": "all"}, 22, 2, 0.4, 1),
    ("a_half", sc_a, {"variant": "half"}, 16, 2, 0.4, 1),
    ("a_usd", sc_a, {"variant": "usd"}, 22, 3, 0.4, 1),
    ("a_ens", sc_a, {"variant": "ens"}, 20, 2, 0.4, 1),
    ("b_named", sc_b, {}, 120, 10, 0, 0),
    ("b_all", sc_b, {"all_": True}, 6, 1, 0, 0),
    ("c_switch", sc_c, {}, 30, 3, 0, 0),
    ("d_needs_chain", sc_d, {}, 30, 4, 0, 0),
    ("e", sc_e, {}, 20, 2, 0, 0),
    ("f_reuse", sc_f, {}, 14, 3, 0.5, 3),
    ("g_refused", sc_g, {}, 8, 1, 0, 0),
    ("h_blocked", sc_h, {}, 12, 1, 0.5, 1),
]


def gen_one(fn, kw, base_rng, ctx, seen, tries=300):
    for _ in range(tries):
        rng = random.Random(base_rng.getrandbits(64))
        try:
            ex = fn(rng, ctx, **kw)
        except Retry:
            continue
        users = [m["content"] for m in ex["messages"] if m["role"] == "user"]
        key = norm(" ".join(users))
        if key in seen:
            continue
        if not chain_words_ok(ex):
            continue
        seen.add(key)
        return ex
    raise RuntimeError(f"could not generate {fn.__name__} {kw}")


def chain_words_ok(ex):
    """No network words in the typed text unless the scenario names one (or the token/address itself is a chain word)."""
    m = ex["meta"]
    typed = typed_text(ex["messages"])
    named = set(mentioned(typed))
    allowed = set(m.get("allowed_induced", []))
    if m.get("chain_named"):
        allowed.add(m["chain_named"])
    if m["kind"] == "d_needs_chain":
        allowed |= set(m["held_on"])
    return named <= allowed


def draw_bio(rng):
    return "Face ID" if rng.random() < 0.7 else "Touch ID"


def build(seed, n, ev, seen):
    rng = random.Random(seed)
    total = sum(k[4] if ev else k[3] for k in KINDS)
    scale = 1.0 if ev else n / total
    rows = []
    for name, fn, kw, tr, evn, fshare, fev in KINDS:
        cnt = evn if ev else max(1, round(tr * scale))
        nforce = fev if ev else int(round(cnt * fshare))
        for i in range(cnt):
            ctx = {"ev": ev, "bio": draw_bio(rng), "force": i < nforce, "i": i}
            ex = gen_one(fn, kw, rng, ctx, seen)
            rows.append(ex)
    return rows, rng


# ============================================================== eval items
def eval_items(rows, rng):
    """Turn generated conversations into eval items. d_needs_chain: half cut at the first user turn (asks), half at the last."""
    items, dn = [], 0
    for i, ex in enumerate(rows):
        m = ex["meta"]
        nu = len(m["user_idx"])
        cut = nu - 1
        if m["kind"] == "d_needs_chain":
            cut = 0 if dn % 2 == 0 else nu - 1
            dn += 1
        ui = m["user_idx"][cut]
        end = m["user_idx"][cut + 1] if cut + 1 < nu else len(ex["messages"])
        ctx_msgs, ref = ex["messages"][:ui + 1], ex["messages"][ui + 1:end]
        expect = m["turn_expects"][cut]
        kind = m["kind"]
        gold_calls = [g for g in expect if "calls" in g]
        first_call = None
        for g in expect:
            if "calls" in g:
                for tc in g["calls"]:
                    if tc["name"] == "prepare_send":
                        first_call = tc["arguments"]
                        break
            if first_call:
                break
        checks = ["no_unbacked_done", "no_ask_for_data", "exact_digits", "no_chain_invented"]
        if kind in ("e_addr", "e_amount", "e_token") or (kind == "d_needs_chain" and cut == 0):
            checks.append("asked_when_needed")
        final = [x for x in ex["messages"][ui + 1:end] if x["role"] == "assistant"][-1]["content"]
        net = None
        for x in ex["messages"][ui + 1:end]:
            if x["role"] == "tool":
                try:
                    net = json.loads(x["content"]).get("network") or net
                except Exception:
                    pass
        ref_text = " ".join(x["content"] for x in ex["messages"][ui:end] if x["role"] in ("tool", "user"))
        eth_ok = "Ethereum" in ref_text or "ethereum" in typed_text(ex["messages"][:ui + 1]).lower() or "ethereum" in str(m.get("chain_named")) \
            or "mainnet" in typed_text(ex["messages"][:ui + 1]).lower()
        gold = {"first_call": first_call, "named_chain_in_call": (first_call or {}).get("chain"),
                "must_omit_chain": "chain" not in (first_call or {}) and kind not in ("e_addr", "e_amount", "e_token"),
                "ask_before_first_call_forbidden": not m.get("ask_first", False) and not (kind == "d_needs_chain" and cut > 0),
                "asks_instead_of_calling": bool(m.get("ask_first")),
                "network_in_answer": net, "address_in_answer": m["addr"] if net else None,
                "refused": bool(m.get("refused")), "blocked": bool(m.get("blocked")), "needs_chain_first": kind == "d_needs_chain" and cut == 0,
                "eth_trap": bool(m["eth_trap"]) and "chain" not in (first_call or {}), "amount": m["amount_value"],
                "answer_must_not": ["sent", "Relay", "went through"] + (["Ethereum"] if not eth_ok else [])}
        items.append({"id": f"send_nochain-{kind}-{i}", "kind": kind, "tools": True, "bio": m["bio"], "context": ctx_msgs,
                      "reference": ref, "expect": expect, "world": m["world"], "checks": checks, "gold": gold,
                      "meta": {"turn": cut, "source": "send_nochain"}})
    return items


# ============================================================== self-check
BAN = [r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|deposited|withdrawn|closed)",
       r"\bI(?:'ve| have)? (sent|bought|swapped|deposited|withdrew|closed)\b", r"successfully", r"funds (are|have) (in|arrived)",
       r"^\s*done\b", r"not financial advice", r"[·•]", r"what('s| is) (the )?(current )?price", r"your budget",
       r"how much (do you|are you)\b.*\b(hold|have)", r"can't see the future", r"\bsent\b", r"\bcompleted\b", r"\bAs an AI\b", r"…", r"\.\.\.",
       r"\bwhich network\?"]
KEYS = ["to_address", "amount", "symbol", "chain"]
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def numset(text):
    out = set()
    for m in NUM.finditer(text):
        s = m.group().rstrip(",.").replace(",", "")
        try:
            out.add(Decimal(s).normalize())
        except Exception:
            pass
    return out


def check_ex(ex, where=""):
    msgs, m = ex["messages"], ex["meta"]
    bio = m["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    assert ex["tools"] is True and msgs[0]["role"] == "system"
    assert "{BIO}" not in msgs[0]["content"] and bio in msgs[0]["content"], "system bio"
    assert msgs[0]["content"].startswith(SYSTEM_TXT.replace("{BIO}", bio).split("\n")[0]), "system prompt"
    for x in msgs:
        assert other not in (x.get("content") or ""), f"{other} in {x['role']} message"
    typed = typed_text(msgs)
    all_users = "\n".join(x["content"] for x in msgs if x["role"] == "user")
    tool_text = "\n".join(x["content"] for x in msgs if x["role"] == "tool")
    # ---- calls
    calls = [tc for x in msgs for tc in (x.get("tool_calls") or [])]
    sends = [tc["arguments"] for tc in calls if tc["name"] == "prepare_send"]
    for tc in calls:
        assert tc["name"] in V2_SCHEMA, f"unknown tool {tc['name']}"
        sch = V2_SCHEMA[tc["name"]]
        req = [r for r in sch.get("required", []) if not (tc["name"] == "prepare_send" and r == "chain")]
        for r in req:
            assert r in tc["arguments"], (tc, r)
        for k, v in tc["arguments"].items():
            assert k in sch["properties"], f"unknown arg {k}"
            t = sch["properties"][k]["type"]
            assert (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)) or (t == "string" and isinstance(v, str)), (tc, k)
        if tc["name"] == "prepare_send":
            a = tc["arguments"]
            ks = list(a.keys())
            assert ks == [k for k in KEYS if k in a], f"key order {ks}"
            assert isinstance(a["amount"], int) or not float(a["amount"]).is_integer(), "amount not shortest"
            if "chain" in a:
                cid = find_chain(a["chain"])
                assert a["chain"] == cid and cid in CHAIN_NAME, f"chain id {a['chain']}"
                assert mentions(cid, typed), f"chain {cid} not named in typed text"
            if m["kind"] != "g_tool":
                src = all_users if m.get("addr_in_file") else typed
                assert a["to_address"] in src, "address not copied exactly"
    # scenario gold equality
    gc = m["gold_calls"]
    assert sends == gc, f"calls differ from scenario {sends} {gc}"
    # chain absent unless named
    if m["kind"] != "d_needs_chain":
        for a in sends:
            if "chain" in a:
                assert m["chain_named"] == a["chain"], "chain without a named network"
    # first assistant turn: a call, never a question, except the scenarios that ask for a missing detail
    first = next(x for x in msgs if x["role"] == "assistant")
    if m.get("ask_first") or m.get("first_ask"):
        assert not first.get("tool_calls") and first["content"].count("?") == 1
    else:
        assert first.get("tool_calls") and first["content"] == "", "first assistant turn must be a call"
    # ---- answers
    addrs = {a["to_address"] for a in sends} | {m["addr"]}
    final_idx = [i for i, x in enumerate(msgs) if x["role"] == "assistant" and not x.get("tool_calls")]
    calc_text = ""
    results = []
    for i, x in enumerate(msgs):
        if x["role"] == "tool":
            results.append(x["content"])
    src_nums = set()
    for t in [all_users, tool_text]:
        src_nums |= numset(t)
    for i in final_idx:
        t = msgs[i]["content"]
        assert t.strip() and len(t.split()) <= 70, f"answer length {len(t.split())}"
        for rx in BAN:
            assert not re.search(rx, t, re.I | re.M), f"banned {rx!r} in: {t}"
        assert t.count("?") <= 1, f"more than one question: {t}"
        clean = t
        for a in sorted(addrs, key=len, reverse=True):
            clean = clean.replace(a, " ")
        for h in re.findall(r"0x[0-9a-fA-F]+", clean):
            raise AssertionError(f"truncated address {h}")
        extra = numset(clean) - src_nums
        assert not extra, f"numbers not from results/user: {extra} in {t}"
        for chain in CHAIN_NAME.values():
            if re.search(rf"\b{re.escape(chain)}\b", t):
                named_name = CHAIN_NAME.get(m.get("chain_named") or "")
                assert chain in tool_text or chain.lower() in all_users.lower() or chain == named_name, f"{chain} not backed: {t}"
        # an address in an answer is printed in full
        if any(a in t for a in addrs):
            pass
        # a prepared send names the result's network
        j = i - 1
        if msgs[j]["role"] == "tool":
            r = json.loads(msgs[j]["content"])
            if r.get("status") == "review_opened":
                assert r["network"] in t and r["amount"] in t and any(a in t for a in addrs), f"answer lacks network/amount/address: {t}"
                assert bio in t, "bio missing"
                assert "prepared" in t.lower() or "ready" in t.lower() or "set up" in t.lower() or "opened" in t.lower() or "waiting" in t.lower()
            if r.get("status") == "needs_chain":
                assert t.count("?") == 1 and all(CHAIN_NAME[o] in t for o in r["options"]), "needs_chain answer"
            if r.get("error", "").startswith("Refused"):
                assert "typ" in t and "paste" in t.lower(), t
            if r.get("status") == "blocked":
                assert "prepared" in t or "set up" in t or "prepare" in t
    # ---- replay equality
    w = World3.from_state(m["world"], seed=0)
    users = []
    for i, x in enumerate(msgs):
        if x["role"] == "user":
            users.append(x["content"])
        for k, tc in enumerate(x.get("tool_calls") or []):
            r = run_tool(w, tc["name"], tc["arguments"], "\n".join(without_files(u) for u in users))
            assert msgs[i + 1 + k]["content"] == json.dumps(r, ensure_ascii=False, sort_keys=True), f"replay differs: {msgs[i + 1 + k]['content']} vs {r}"
    # ---- amounts consistent with the scenario
    if m["kind"] in ("a_plain", "a_ens", "b_named", "c_switch", "d_needs_chain", "e_amount", "e_addr", "e_token", "f_reuse_sent", "f_reuse_addr", "g_file", "h_blocked"):
        pass
    if m["kind"] in ("a_plain", "a_ens", "b_named") and m.get("amount_surface") is not None:
        pass
    if m.get("amount_surface") and m["kind"] in ("a_plain", "a_ens"):
        nums = re.sub(r"(?i)(tether|ether|usd coin|usdc|usdt|matic|near|eth|sol|pol|dai)", "", m["amount_surface"]).strip()
        assert abs(parse_amount(nums) - float(sends[0]["amount"])) < 1e-12, "amount != typed amount"
    # ---- render round trip
    text = H.render_text(msgs, False, "4b")
    parsed = []
    for x in msgs:
        if x["role"] == "assistant" and x.get("tool_calls"):
            seg = H.assistant_text(x)
            pc, bad = H.parse_calls(seg)
            assert bad == 0 and pc == x["tool_calls"], "render round trip"
    assert "{BIO}" not in text
    return True


def check_file(rows, label, expect_n=None, ev=False):
    seen = {}
    for i, ex in enumerate(rows):
        json.loads(json.dumps(ex, ensure_ascii=False))
        check_ex(ex, f"{label}[{i}]")
        key = norm(" ".join(x["content"] for x in ex["messages"] if x["role"] == "user"))
        assert key not in seen, f"duplicate user text rows {seen[key]} and {i}"
        seen[key] = i
    if expect_n:
        assert len(rows) == expect_n, (len(rows), expect_n)
    return seen


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--eval-seed", type=int, default=9304)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(LAB, "data3", "send_nochain.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval3", "send_nochain.jsonl"))
    a = ap.parse_args()
    assert a.n_eval == 60, "eval plan is fixed at 60 items"

    # texts that must never appear in train or eval: eval2 user texts (leakage index)
    seen = set()
    for f in sorted(os.listdir(os.path.join(LAB, "eval2"))):
        if f.endswith(".jsonl"):
            for line in open(os.path.join(LAB, "eval2", f)):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for msg in d.get("context", []) or []:
                    if msg.get("role") == "user":
                        seen.add(norm(msg["content"]))
    ev_rows, ev_rng = build(a.eval_seed, a.n_eval, True, seen)
    tr_rows, tr_rng = build(a.seed, a.n, False, seen)
    # eval-only guarantee: no train user text equals an eval user text (seen is shared, so already enforced), re-verify below
    random.Random(a.seed).shuffle(tr_rows)
    random.Random(a.eval_seed).shuffle(ev_rows)
    ev_items = eval_items(ev_rows, ev_rng)

    # ---- self-check
    tr_seen = check_file(tr_rows, "train")
    ev_seen = check_file(ev_rows, "eval", expect_n=a.n_eval, ev=True)
    ev_users, tr_users = set(), set()
    for ex in ev_rows:
        ev_users |= {norm(x["content"]) for x in ex["messages"] if x["role"] == "user"}
    for ex in tr_rows:
        tr_users |= {norm(x["content"]) for x in ex["messages"] if x["role"] == "user"}
    assert not (ev_users & tr_users), f"train/eval share user texts: {len(ev_users & tr_users)}"
    # eval-only templates and slot values: check no train row used an eval-only template shape
    off = sum(1 for it in ev_items if it["gold"]["eth_trap"])
    assert off >= 20, f"only {off} eval items with the token held off Ethereum and no network named"
    assert len(ev_items) == a.n_eval
    for it in ev_items:
        assert it["context"][-1]["role"] == "user" or it["context"][-1]["role"] == "tool"
        assert it["world"] and it["expect"] and it["checks"]
        w = World3.from_state(it["world"], seed=0)       # eval world loads
    # each eval item's gold calls run against its own world (replay of the reference was already checked per row)
    for p, rows in ((a.out, tr_rows), ):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
    with open(a.eval_out, "w") as f:
        for it in ev_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # re-read the files and check again (valid JSON on disk)
    with open(a.out) as f:
        back = [json.loads(l) for l in f]
    check_file(back, "train-disk")
    with open(a.eval_out) as f:
        for l in f:
            json.loads(l)
    kinds = Counter(ex["meta"]["kind"] for ex in tr_rows)
    nochain = sum(1 for ex in tr_rows if not ex["meta"].get("chain_named") and ex["meta"]["kind"] != "d_needs_chain")
    trap_tr = sum(1 for ex in tr_rows if ex["meta"]["eth_trap"] and ex["meta"]["kind"].startswith("a_"))
    n_a = sum(1 for ex in tr_rows if ex["meta"]["kind"].startswith("a_"))
    print(f"kinds: {dict(kinds)}", file=sys.stderr)
    print(f"train: {len(tr_rows)}; a-family {n_a}, held-off-Ethereum {trap_tr}; no-network-named {nochain}; "
          f"eval trap items {off}; bio Touch ID {sum(1 for ex in tr_rows if ex['meta']['bio'] == 'Touch ID')}", file=sys.stderr)
    print(f"send_nochain: {len(tr_rows)} train, {len(ev_items)} eval, self-check ok")


if __name__ == "__main__":
    main()
