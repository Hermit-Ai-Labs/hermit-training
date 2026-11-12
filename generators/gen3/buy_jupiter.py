#!/usr/bin/env python3
"""buy_jupiter: prepare_buy for Solana tokens via Jupiter, plus the negatives routed elsewhere.

Positives: prepare_buy {symbol, amount_usd[, pay_with]} and every result branch (success, unverified,
insufficient, policy blocked, high price impact warning, no wallet), ask for a missing amount.
Negatives: ETH/SOL/BNB -> prepare_swap, HYPE/BTC -> open_app Hyperliquid, PEPE/SHIB -> open_app Uniswap,
price question -> get_price, "should I buy" -> short view (get_price, no prepare), "sell my X" -> no call,
"buy X" with no amount -> ask "How much, in dollars?".

Every assistant string is a Python f-string filled from tool results, tool arguments or user text.
Nothing is typed by hand as a number. Reuses gen2 helpers by import (Conv, chg, hermit_fmt); gen2 is not edited.

Usage: python3 buy_jupiter.py [--seed 3] [--n 350] [--eval-seed 9311] [--eval-n 40]
Writes ../data3/buy_jupiter.jsonl and ../eval3/buy_jupiter.jsonl, then self-checks every record.
"""
import sys
sys.dont_write_bytecode = True
import argparse, copy, json, os, random, re
from collections import Counter
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(LAB, "gen2"))
import hermit_fmt as H                      # gen2 helper (read only): with_user, parse_calls, assistant_text
from build_agent import Conv, chg           # gen2 helpers (read only)

SPEC = os.path.join(LAB, "spec")
TOOLS_V2 = json.load(open(os.path.join(SPEC, "hermit_tools_v2.json")))
SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V2}
SYSTEM_TPL = open(os.path.join(SPEC, "SYSTEM_AGENT_v2.txt")).read()
GEN = "buy_jupiter"

# ================================================================================ numbers
def approx(x):
    """The app's Units.approx: >=1000 -> 2 dp, else 3 significant digits, trailing zeros trimmed."""
    d = Decimal(repr(float(x)))
    if d >= 1000:
        q = d.quantize(Decimal("0.01"))
    else:
        q = d.quantize(Decimal(1).scaleb(d.adjusted() - 2))
    s = format(q, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def jnum(x):
    """Number as the app prints it: ints stay ints, floats shortest."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return round(x, 8) if isinstance(x, float) else x


def jstr(x):
    return json.dumps(jnum(x))


def usd(x):
    """$20 or $12.50 (value comes from tool arguments)."""
    x = jnum(x)
    if isinstance(x, int):
        return f"${x:,}"
    s = format(Decimal(repr(x)), "f")
    if len(s.split(".")[1]) < 2:
        s += "0"
    return "$" + s


def price_txt(p):
    s = format(Decimal(repr(float(p))), "f")
    if "." not in s:
        s += ".00"
    elif len(s.split(".")[1]) < 2:
        s += "0"
    return "$" + s


def sort_d(d):
    return dict(sorted(d.items()))


# ================================================================================ tokens
BANDS = {k: v for k, v in __import__("sim").ASSETS.items()}
BANDS.update({"PENGU": (0.008, 0.06, 5), "POPCAT": (0.2, 1.6, 4), "PYTH": (0.08, 0.5, 4), "JTO": (0.8, 3.5, 3),
              "RAY": (1.2, 6.5, 3), "ORCA": (1.0, 4.5, 3), "MEW": (0.002, 0.012, 6), "MOODENG": (0.05, 0.5, 4),
              "SHIB": (0.000007, 0.00003, 9)})
SHARED = ["WIF", "BONK", "PENGU", "JUP", "POPCAT", "XRP", "DOGE", "TRUMP", "FARTCOIN"]
SHARED_W = [24, 17, 10, 8, 7, 6, 7, 5, 5]
TRAIN_ONLY = ["PYTH", "JTO", "RAY"]
EVAL_ONLY = ["MEW", "MOODENG", "ORCA"]
VERIFIED = SHARED + TRAIN_ONLY + EVAL_ONLY
UNV_TRAIN = ["ZORP", "MOONX", "BLORB", "GRIMLY", "WOOFY", "KIRBO", "SNEK", "NUBZ", "SAFEMOON", "BABYWIF", "PUMPIT", "FLOKIZ", "CATGPT", "DEGEN2"]
UNV_EVAL = ["QUACKY", "DEGENX", "BOZO9", "LUNARX"]
NO_MARKET = ["BONK", "PEPE", "SHIB"]                      # no Hyperliquid market for these (get_price errors)
PRICEABLE = ["WIF", "DOGE", "XRP", "JUP", "TRUMP", "FARTCOIN", "PENGU", "POPCAT"]
SELL_TOK = ["WIF", "BONK", "PENGU", "JUP", "POPCAT", "DOGE", "XRP", "TRUMP", "FARTCOIN", "PYTH", "JTO"]
DEX = ["Orca", "Raydium", "Meteora", "Phoenix", "Lifinity"]
CHAIN_NAME = {"solana": "Solana", "base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "bsc": "BNB Chain", "polygon": "Polygon"}
CHAIN_SAY = {"ethereum": ["Ethereum", "ethereum", "mainnet"], "base": ["Base", "base"], "arbitrum": ["Arbitrum", "arbitrum", "arb"]}
ALIAS = {"WIF": ["dogwifhat"], "DOGE": ["dogecoin"], "XRP": ["ripple"]}
APPS = {"hyperliquid": "https://app.hyperliquid.xyz", "uniswap": "https://app.uniswap.org", "aave": "https://app.aave.com",
        "jupiter": "https://jup.ag"}
TRAIN_AMTS = [2.5, 5, 7.5, 10, 10, 10, 12.5, 15, 20, 20, 20, 25, 25, 30, 40, 50, 50, 50, 75, 100, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500]
EVAL_AMTS = [17.5, 35, 45, 60, 80, 120, 175, 350, 600, 1200]
BIG_TRAIN = [300, 400, 500, 750, 1000, 1500, 2000]
BIG_EVAL = [350, 600, 1200]
WORDS = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty", 40: "forty", 50: "fifty", 100: "a hundred"}


class BuyWorld:
    """One user's state for this generator. Deterministic: every result is a pure function of the state."""
    KEYS = ("prices", "change", "wallet", "balances", "daily_left", "move_cap", "liq", "route", "verified",
            "no_market", "device", "bio", "seconds", "spread_bps")

    def __init__(self, st):
        for k in self.KEYS:
            setattr(self, k, copy.deepcopy(st[k]))

    @classmethod
    def from_state(cls, st):
        return cls(st)

    def state(self):
        return {k: copy.deepcopy(getattr(self, k)) for k in self.KEYS}

    @classmethod
    def new(cls, rng, bio, device):
        prices, change = {}, {}
        for s, (lo, hi, dp) in BANDS.items():
            p = rng.uniform(lo, hi)
            prices[s] = round(p, dp) if dp else float(round(p))
            c = round(rng.uniform(-9, 9), 2)
            change[s] = c if c != 0 else 0.1
        liq = {s: float(int(rng.uniform(1.5e6, 40e6))) for s in VERIFIED}
        route = {s: "Jupiter via " + ", ".join(rng.sample(DEX, rng.choice([1, 2, 2, 3]))) for s in VERIFIED}
        return cls({"prices": prices, "change": change, "wallet": True, "balances": [], "daily_left": 100000.0,
                    "move_cap": 100000.0, "liq": liq, "route": route, "verified": list(VERIFIED),
                    "no_market": list(NO_MARKET), "device": device, "bio": bio, "seconds": rng.choice([8, 12, 14, 18, 20, 25]),
                    "spread_bps": rng.choice([5, 10, 15, 20, 30])})

    # ---- holdings helpers (setup, not tools)
    def hold(self, symbol, chain, amount):
        self.balances = [b for b in self.balances if not (b["symbol"] == symbol and b["chain"] == chain)]
        usd_v = round(amount * (1.0 if symbol == "USDC" else self.prices[symbol]), 2)
        self.balances.append({"symbol": symbol, "chain": chain, "amount": jnum(amount), "usd": jnum(usd_v)})

    def sol_hold(self):
        usdc = sum(b["usd"] for b in self.balances if b["symbol"] == "USDC" and b["chain"] == "solana")
        sol = sum(b["usd"] for b in self.balances if b["symbol"] == "SOL" and b["chain"] == "solana")
        return float(usdc), float(sol)

    # ---- tools
    def call(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args if isinstance(args, dict) else {}))
        except (TypeError, ValueError, KeyError) as e:
            return {"error": f"bad arguments: {e}"}

    def t_get_price(self, symbol):
        s = str(symbol).upper()
        if s in self.no_market or s not in self.prices:
            return {"error": f"No Hyperliquid market for {s}. Answer from what you know instead, and say there's no live price here."}
        return sort_d({"change_24h_pct": self.change[s], "price": self.prices[s], "source": "Hyperliquid mark price", "symbol": s})

    def t_get_balances(self, chain=None):
        rows = [b for b in self.balances if b["amount"] > 0 and (chain is None or b["chain"] == chain)]
        return {"holdings": [sort_d({"amount": jstr(b["amount"]), "chain": b["chain"], "symbol": b["symbol"], "usd": b["usd"]}) for b in rows]}

    def t_open_add_funds(self, goal):
        if goal not in ("wallet", "hyperliquid"):
            return {"error": "bad arguments: goal"}
        return sort_d({"goal": goal, "note": "The user taps it to open Add funds.", "screen": "add_funds", "status": "button_shown"})

    def t_open_app(self, name=None, url=None):
        if url:
            if not re.match(r"https?://", url):
                return {"error": "Only http and https addresses can be opened."}
            return sort_d({"note": "Opened in Hermit's browser. Nothing connects to the wallet until the user approves it.", "opened": url, "status": "opened"})
        u = APPS.get((name or "").strip().lower())
        if not u:
            return sort_d({"note": f"No dApp called {name} in Explore. Ask the user for the web address.", "status": "not_found"})
        return sort_d({"note": "Opened in Hermit's browser. Nothing connects to the wallet until the user approves it.", "opened": u, "status": "opened"})

    def _blocked(self, why):
        return sort_d({"note": "Nothing was prepared. Tell the user why.", "reasons": [why], "status": "blocked"})

    def t_prepare_buy(self, symbol, amount_usd, pay_with=None):
        if not self.wallet:
            return {"error": f"No wallet on this {self.device} to buy with"}
        sym = str(symbol).upper()
        if sym not in self.verified:
            return self._blocked(f"There's no verified {sym} token on Solana, so Hermit won't buy one.")
        if not isinstance(amount_usd, (int, float)) or isinstance(amount_usd, bool) or amount_usd <= 0:
            return {"error": "amount_usd must be a number above 0"}
        if pay_with not in (None, "USDC", "SOL"):
            return {"error": "pay_with must be USDC or SOL"}
        if amount_usd > self.daily_left:
            return self._blocked("Amount is above your daily limit for the assistant.")
        if amount_usd > self.move_cap:
            return self._blocked("Amount is above the per-move cap for the assistant.")
        usdc, sol_usd = self.sol_hold()
        pay = pay_with or ("USDC" if usdc >= amount_usd else "SOL")
        have = usdc if pay == "USDC" else sol_usd
        if have + 1e-9 < amount_usd:
            parts = []
            for sy in ("USDC", "SOL"):
                a = sum(b["amount"] for b in self.balances if b["symbol"] == sy and b["chain"] == "solana")
                if a > 0:
                    parts.append(f"{jstr(a)} {sy}")
            hold = (" and ".join(parts) + " on Solana") if parts else "no USDC or SOL on Solana"
            return sort_d({"holding": hold, "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"})
        impact = round(0.02 + 100.0 * amount_usd / self.liq[sym], 2)
        qty = amount_usd / self.prices[sym] * (1 - impact / 100.0)
        you_pay = f"{jstr(amount_usd)} USDC on Solana" if pay == "USDC" else f"{approx(amount_usd / self.prices['SOL'])} SOL on Solana"
        r = {"executed": False, "kind": "buy",
             "note": f"Nothing has moved. The Buy review is open with this Jupiter quote; the user approves it with {self.bio} or cancels.",
             "price_impact_pct": jnum(impact), "route": self.route[sym], "slippage_pct": 0.5, "status": "review_opened",
             "symbol": sym, "you_get": f"about {approx(qty)} {sym}", "you_pay": you_pay}
        if impact >= 1:
            r["warnings"] = [f"High price impact ({jstr(impact)}%): you may get noticeably less than the price shown."]
        return sort_d(r)

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        if not self.wallet:
            return {"error": f"No wallet on this {self.device} to swap with"}
        if amount > self.daily_left:
            return self._blocked("Amount is above your daily limit for the assistant.")
        have = sum(b["amount"] for b in self.balances if b["symbol"] == from_symbol and b["chain"] == from_chain)
        if have + 1e-9 < amount:
            return {"error": f"insufficient balance: {from_symbol} on {from_chain} is {jstr(have)}"}
        fee_bps = 35
        cost = round(amount * (fee_bps + self.spread_bps) / 10000.0, 2)
        out_usd = round(amount - cost, 2)
        out = approx(out_usd / self.prices[to_symbol])
        return sort_d({"amount_in_usd": jnum(amount), "amount_out": out, "amount_out_usd": jnum(out_usd), "fee_bps": fee_bps,
                       "from": f"{jstr(amount)} {from_symbol} on {CHAIN_NAME[from_chain]}",
                       "note": f"Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with {self.bio} or cancels.",
                       "policy": "allow", "route": "NEAR Intents", "seconds": self.seconds, "signature_verified": True,
                       "status": "prepared_for_review", "to": f"about {out} {to_symbol} on {CHAIN_NAME[to_chain]}",
                       "total_cost_usd": jnum(cost)})


# ================================================================================ phrase machinery
def split_tpls(pool, ev):
    return [t for i, t in enumerate(pool) if (i % 5 == 0) == ev]


TYPO_SKIP = {"with", "using", "usdc", "sol", "usd", "bucks", "dollars"}


def typo_word(rng, w):
    i = rng.randrange(1, len(w) - 1)
    k = rng.random()
    if k < 0.4:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:] if i + 2 <= len(w) else w
    if k < 0.7:
        return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


def render_tpl(rng, tpl, slots, typo_p=0.15):
    parts = re.split(r"(\{[A-Za-z]+\})", tpl)
    if rng.random() < typo_p:
        cands = [(pi, m) for pi, p in enumerate(parts) if not p.startswith("{") for m in re.finditer(r"[a-z]{4,}", p) if m.group() not in TYPO_SKIP]
        if cands:
            pi, m = rng.choice(cands)
            parts[pi] = parts[pi][:m.start()] + typo_word(rng, m.group()) + parts[pi][m.end():]
    out = "".join(slots[p[1:-1]] if p.startswith("{") else p for p in parts)
    r = rng.random()
    if r < 0.20:
        out = out.lower()
    elif r < 0.22:
        out = out.upper()
    r = rng.random()
    if r < 0.05 and not out.endswith("?"):
        out += rng.choice([" pls", " plz", " rn"])
    elif r < 0.10:
        out = out.rstrip(".?!")
    return out


def num_text(rng, v, words_ok=True):
    if words_ok and v in WORDS and rng.random() < 0.15:
        return WORDS[v], True
    if v >= 1000 and v % 1000 == 0 and rng.random() < 0.4:
        return f"{int(v // 1000)}k", False
    if v >= 1000 and rng.random() < 0.4:
        return f"{int(v):,}", False
    if isinstance(v, float) and not v.is_integer():
        return rng.choice([f"{v:g}", f"{v:.2f}"]), False
    return str(int(v)), False


def amt_slots(rng, v):
    n, is_word = num_text(rng, v)
    r = rng.random()
    if is_word:
        a = n + rng.choice([" bucks", " dollars"])
    elif r < 0.45:
        a = "$" + n
    elif r < 0.60:
        a = n + " bucks"
    elif r < 0.72:
        a = n + " dollars"
    elif r < 0.82:
        a = n + rng.choice([" usd", " USD"])
    else:
        a = "$" + n
    return {"A": a, "N": n, "U": n + rng.choice([" USDC", " usdc", " USDC"])}


def sym_text(rng, sym, aliases=True):
    r = rng.random()
    if aliases and sym in ALIAS and r < 0.18:
        return rng.choice(ALIAS[sym])
    if r < 0.55:
        return sym
    if r < 0.80:
        return sym.lower()
    return "$" + sym


def pick_amount(rng, ev, big=False):
    if big:
        return rng.choice(BIG_EVAL if ev else BIG_TRAIN)
    return rng.choice(EVAL_AMTS if ev else TRAIN_AMTS)


def pick_symbol(rng, ev):
    if ev and rng.random() < 0.4:
        return rng.choice(EVAL_ONLY)
    if not ev and rng.random() < 0.12:
        return rng.choice(TRAIN_ONLY)
    return rng.choices(SHARED, SHARED_W)[0]


def pick_unverified(rng, ev):
    return rng.choice(UNV_EVAL if ev else UNV_TRAIN)


def pick_bio(rng):
    bio = "Face ID" if rng.random() < 0.70 else "Touch ID"
    dev = "iPhone" if bio == "Face ID" else ("Mac" if rng.random() < 0.9 else "iPhone")
    return bio, dev


# ---- positive templates: {A} amount with unit, {S} ticker text. Index % 5 == 0 -> eval only.
BUY_TPL = [
    "I want {A} of {S}", "buy {A} of {S}", "buy {S} {A}", "ape {A} into {S}", "get me some {S}, {A}",
    "get {A} of {S} asap", "can you buy {A} worth of {S}?", "grab {A} of {S} for me", "pick up {A} of {S}", "{S} {A} buy",
    "snipe {A} of {S}", "buy me {A} of {S} please", "yo buy {S}, {A}", "I'd like to buy {A} of {S} on solana", "purchase {A} of {S}",
    "buy {A} {S}", "put {A} into {S}", "load up on {S}, {A}", "buy some {S}. {A}", "need {S} bad, buy {A}",
    "ok i've been watching {S} all day, buy {A} worth before it pumps", "long story short I think {S} runs, can you buy {A} of it", "friend told me about {S}. buy {A} of it", "throw {A} at {S}", "set me up with {A} of {S}",
    "pls buy {S}, only {A}", "{A} of {S}", "buy {A} of {S} on jupiter", "add {A} of {S} to my bag", "buy the dip on {S}, {A}",
    "i wanna buy {S}. {A} is enough", "ape into {S} for {A}", "buy {S} for {A}", "just got paid, buy {A} of {S}", "hey hermit, {A} of {S} please",
    "go buy {A} of {S}", "open a {A} {S} position, spot", "need {A} of {S} in my wallet", "{S}: buy {A}", "treat myself to {A} of {S}",
]
# ---- swap-style buys (spend USDC for a Solana token; this is a buy, no pay_with)
SWAPBUY_TPL = [
    "swap {U} for {S}", "swap {U} to {S}", "convert {U} into {S}", "trade {U} for {S}", "exchange {U} for {S}",
    "swap {U} into {S}", "can you swap {U} for {S}?", "turn {U} into {S}", "swap {U} for some {S}", "change {U} to {S}",
]
# ---- explicit payment token: {P} = SOL/USDC text
PAY_TPL = [
    "buy {A} of {S} with {P}", "buy {S} with {P}, {A}", "buy {A} of {S} using {P}", "get me {A} worth of {S} paying with {P}", "{A} of {S} with {P} pls",
    "ape {A} into {S} using {P}", "buy {S} with {P}: {A}", "i wanna buy {A} {S} with my {P}", "put {A} into {S} with {P}", "buy {A} of {S}, use the {P} one",
    "can you buy {A} of {S} using {P}?", "with {P}, buy {A} of {S}", "grab {A} of {S} with {P}", "get {S} for {A} using {P}", "purchase {A} of {S} with {P}",
    "buy {S} {A} with {P}", "ape {S}, {A}, using {P}", "I'd like {A} of {S} bought with {P}", "buy {A} worth of {S} with {P} thx", "load {A} of {S}, pay with {P}",
]
PAYRE = re.compile(r"\b(?:with|using)\s+(?:my\s+|the\s+)?(usdc|sol)\b", re.I)
PAY_TPL = [t.replace("use the {P} one", "with the {P} one").replace("pay with {P}", "with {P}") for t in PAY_TPL]
# ---- no amount: ask
ASK_TPL = [
    "buy {S}", "buy some {S}", "ape into {S}", "i want {S}", "get me {S}",
    "get me some {S} pls", "buy {S} for me", "can you buy {S}?", "grab some {S}", "i wanna buy {S}",
    "pick up some {S}", "buy {S} now", "need some {S}", "load up on {S}", "{S} buy",
    "hey, buy {S}", "let's buy {S}", "buy me some {S}", "get {S} in my wallet", "i'm buying {S}",
]
ASK_PAY_TPL = [
    "buy {S} with {P}", "buy some {S} with {P}", "ape into {S} using {P}", "get me {S} with {P}", "i want {S}, pay with {P}",
    "buy {S} using {P}", "grab {S} with {P}", "buy me some {S} with {P}", "get {S} using {P}", "buy {S} with my {P}",
]
ASK_PAY_TPL = [t.replace("pay with {P}", "with {P}") for t in ASK_PAY_TPL]
FOLLOW_TPL = ["{A}", "just {N}", "make it {A}", "{N} bucks", "uh {N}", "let's say {A}", "{A} pls", "{N}", "{A} is fine", "go with {A}",
              "i'll do {A}", "say {A}", "{N} dollars", "about {A}", "{A} worth"]
FUNDS_TPL = ["how do I add funds", "add funds", "top up my wallet", "how can i deposit", "where do i deposit", "ok how do I fund it",
             "i need to add money, how", "show me how to deposit", "how do i put money in", "fund my wallet"]
# ---- negatives
NEG_BUY_TPL = [   # ETH, SOL, BNB, HYPE, BTC, PEPE, SHIB with an amount
    "buy {A} of {S}", "get me {A} of {S}", "buy {S}, {A}", "i want {A} worth of {S}", "pick up {A} of {S}",
    "buy some {S}, {A}", "can you buy {A} of {S}", "stack {A} of {S}", "put {A} into {S}", "grab {A} of {S}",
    "{A} of {S} please", "buy {A} {S}", "ape {A} into {S}", "I'd like {A} of {S}", "buy {S} for {A}",
    "purchase {A} of {S}", "get {A} worth of {S}", "need {A} of {S}", "buy {A} of {S} for me", "load {A} of {S}",
]
NEG_NOAMT_TPL = ["buy {S}", "get me some {S}", "i want to buy {S}", "buy some {S}", "can I buy {S} here?", "how do I buy {S}", "i wanna get {S}", "buy me {S}"]
NEG_PRICE_TPL = [
    "what's {S} at?", "{S} price", "how much is {S} right now", "where is {S} trading", "price of {S}?",
    "{S} price check", "is {S} up today?", "what's {S} doing today", "current {S} price pls", "check {S}",
    "what is {S} worth now", "{S}?", "how's {S} priced", "quick {S} price", "{S} price rn", "what's {S} trading at",
]
NEG_VIEW_TPL = [
    "should I buy {S}?", "is {S} a buy", "is it a good time to buy {S}", "thoughts on {S}?", "what do you think of {S} here",
    "worth buying {S} rn?", "should i ape {S}", "is {S} worth it", "would you buy {S}", "{S} buy or wait?",
    "is {S} too late to buy", "thinking of buying {S}, good idea?", "should i get some {S}", "is {S} a good buy right now", "any reason not to buy {S}?",
    "is it smart to buy {S} today", "{S}, buy now or wait for a dip?", "what's your view on {S}",
]
NEG_SELL_TPL = [
    "sell my {S}", "sell all my {S}", "dump my {S}", "sell {A} of {S}", "cash out my {S}",
    "i want to sell {S}", "get rid of my {S}", "sell half my {S}", "can you sell my {S}?", "exit my {S} bag",
    "sell {S} for me", "offload my {S}", "sell my {S} pls", "swap my {S} back to usdc", "take profit on my {S}",
    "liquidate my {S}", "i need to sell {S}", "sell some {S}",
]


# ================================================================================ answer text
def say_success(rng, r, bio):
    sym, pay, get, route = r["symbol"], r["you_pay"], r["you_get"], r["route"]
    imp, slip = jstr(r["price_impact_pct"]), jstr(r["slippage_pct"])
    intro = rng.choice([
        f"I've prepared a buy of {sym}: you pay {pay} and get {get} through {route}.",
        f"Your {sym} buy is prepared. You pay {pay} and get {get}, routed through {route}.",
        f"Prepared: a {sym} buy where you pay {pay} and get {get}. The route is {route}.",
        f"The {sym} buy is ready for review: {pay} in, {get} out, through {route}.",
        f"I've set up the {sym} buy. You pay {pay} and you get {get} through {route}.",
        f"The Buy review for {sym} is open: you pay {pay} and get {get} through {route}.",
    ])
    closer = rng.choice([
        f"Approve with {bio} or cancel. Nothing has been bought yet.",
        f"Nothing is bought until you approve it with {bio}.",
        f"It only happens if you approve with {bio}, and you can cancel instead. Nothing has been bought yet.",
        f"Review it and approve with {bio}, or cancel. Nothing has been bought yet.",
        f"Waiting for your {bio}. Nothing is bought until you approve.",
        f"Approve it with {bio} when you're ready. Nothing has been bought yet.",
    ])
    if r.get("warnings"):
        warn = r["warnings"][0]
        mid = rng.choice([f"The review warns: {warn}", f"Heads up: {warn}", f"One warning: {warn}", f"Note the warning: {warn}",
                          f"Be careful, {warn}"])
        tail = rng.choice([f"Approve with {bio} only if you're fine with that, or cancel. Nothing has been bought yet.",
                           f"A smaller amount would lower it. Approve with {bio} only if you're fine with that. Nothing has been bought yet.",
                           f"Only approve with {bio} if that's okay, otherwise cancel. Nothing is bought until you approve.",
                           f"If that's too much, cancel and try a smaller amount. Otherwise approve with {bio}. Nothing has been bought yet."])
        return f"{intro} {mid} {tail}"
    mid = rng.choice([f"Price impact is {imp}%.", f"The price impact is {imp}%.", f"Price impact: {imp}%.",
                      f"Expected price impact is {imp}%, with {slip}% slippage.", f"That's {imp}% price impact and {slip}% slippage.",
                      f"Price impact comes to {imp}%."])
    return f"{intro} {mid} {closer}"


def say_unverified(rng, sym):
    return rng.choice([
        f"Hermit won't buy {sym}: there's no verified token for it on Solana.",
        f"Hermit won't buy {sym}: there's no verified token for it on Solana. Nothing was prepared.",
        f"I can't prepare that. There's no verified {sym} token on Solana, so Hermit won't buy one.",
        f"There's no verified {sym} token on Solana, so Hermit won't buy one. Nothing was prepared.",
        f"{sym} isn't a verified token on Solana, so Hermit won't buy it. Nothing was prepared.",
        f"No luck: Hermit won't buy {sym}, there's no verified token for it on Solana.",
    ])


def say_insufficient(rng, r, sym, amt):
    h, a = r["holding"], usd(amt)
    return rng.choice([
        f"I couldn't prepare that: you hold {h}, which isn't enough for a {a} buy of {sym}. Nothing was prepared.",
        f"That's more than you have: you hold {h}.",
        f"You hold {h}, so I can't prepare a {a} {sym} buy.",
        f"Not enough funds for {a} of {sym}. You hold {h}.",
        f"Your wallet doesn't cover {a}: you hold {h}. Nothing was prepared.",
        f"I can't prepare it. For {sym} you'd need {a}, and you hold {h}.",
    ])


def say_blocked(rng, r):
    why = r["reasons"][0]
    return rng.choice([
        f"I couldn't prepare that. {why} Nothing was prepared.",
        f"Hermit blocked that buy. {why}",
        f"That buy was blocked. {why} Nothing was prepared.",
        f"I can't prepare it. {why}",
        f"Blocked: {why} Nothing was prepared.",
        f"No review was opened. {why}",
    ])


def say_nowallet(rng, r):
    dev = r["error"].split("this ")[1].split(" to")[0]
    return rng.choice([
        f"I can't prepare a buy yet: there's no wallet on this {dev}. Set one up in Wallets, then ask me again.",
        f"There's no wallet on this {dev} to buy with. Set one up in Wallets and I'll prepare it.",
        f"No wallet on this {dev} yet, so I can't prepare that. Set one up in Wallets first.",
        f"I need a wallet on this {dev} to buy with, and there isn't one. Set one up in Wallets, then try again.",
        f"Nothing was prepared because there's no wallet on this {dev}. Set one up in Wallets first.",
    ])


def say_addfunds(rng):
    return rng.choice([
        "I put an Add funds button below. Tap it to see the deposit address and pick a network.",
        "There's an Add funds button below. Tap it to deposit; nothing opens by itself.",
        "I've put an Add funds button in the chat. Tap it to get your deposit address and choose a network.",
        "Tap the Add funds button below to deposit. You can pick the network there.",
        "The Add funds button is below. Tap it when you're ready to deposit.",
    ])


def say_swap(rng, r, bio):
    c = usd(r["total_cost_usd"])
    return rng.choice([
        f"I've prepared the swap: {r['from']} for {r['to']}, through {r['route']}. Total cost is about {c}. Approve with {bio} or cancel. Nothing has moved yet.",
        f"Your swap is prepared: {r['from']} for {r['to']}, through {r['route']}. All-in cost is about {c}. Approve with {bio} or cancel. Nothing has moved yet.",
        f"Prepared: {r['from']} for {r['to']} through {r['route']}, about {c} in total costs. Nothing has moved until you approve with {bio}.",
        f"The swap review is open. You give {r['from']} and get {r['to']}, through {r['route']}, for about {c} in costs. Approve with {bio} or cancel. Nothing has moved yet.",
        f"I've set up {r['from']} for {r['to']} through {r['route']}. Costs come to about {c}. Review it and approve with {bio}, or cancel. Nothing has moved yet.",
        f"Swap prepared through {r['route']}: {r['from']} for {r['to']}. Total costs are about {c}. It only moves if you approve with {bio}.",
    ])


def say_view(rng, sym, r):
    p, ch = price_txt(r["price"]), r["change_24h_pct"]
    if abs(ch) < 1.5:
        view = rng.choice(["There's no clear edge here, so I'd wait or go small.", "No clear edge right now, so I'd wait or keep it small."])
    elif ch > 5:
        view = rng.choice(["It's already extended after that run, so chasing it is the risk.", "That's a big run already, and chasing is the main risk."])
    elif ch < -5:
        view = rng.choice(["A bounce is possible, but it can keep falling.", "It could bounce, but it can also keep falling."])
    elif ch > 0:
        view = rng.choice(["Mild momentum, but nothing I'd lean on hard.", "It's drifting up, but that's not a strong signal."])
    else:
        view = rng.choice(["A small dip, not a clear signal either way.", "It's slightly down, which isn't a clear signal."])
    risk = rng.choice(["It can swing hard in either direction, so only buy what you can afford to lose.",
                       "Tokens like this can drop fast and liquidity can be thin, so keep it small.",
                       "Expect sharp swings, and the price can move before you approve.",
                       "The risk is a sharp drop, so size it small.",
                       "It's volatile, so keep the amount small."])
    cant = rng.choice(["I can't predict where it goes.", "Nobody can call the next move.", "I can't tell you where it heads next."])
    offer = rng.choice(["If you want it, tell me the amount in dollars and I'll prepare the buy.",
                        "Want me to prepare a buy? Just give me the amount in dollars.",
                        "I can prepare a buy if you give me an amount in dollars.",
                        "Say the word and an amount in dollars, and I'll prepare a buy.",
                        "If you'd like to buy, give me an amount in dollars and I'll prepare it."])
    head = rng.choice([f"{sym} is at {p}, {chg(ch)} in the last 24 hours.", f"{sym} is trading at {p}, {chg(ch)} over 24 hours.",
                       f"Right now {sym} is {p}, {chg(ch)} in 24 hours."])
    return f"{head} {view} {cant} {risk} {offer}"


def say_price(rng, sym, r):
    p, ch = price_txt(r["price"]), r["change_24h_pct"]
    base = rng.choice([f"{sym} is at {p}, {chg(ch)} in the last 24 hours.", f"{sym} is trading at {p}, {chg(ch)} over 24 hours.",
                       f"{sym} is {p} right now, {chg(ch)} in 24 hours.", f"Right now {sym} is {p}, {chg(ch)} today.",
                       f"{p} for {sym}, {chg(ch)} in the last 24 hours.", f"{sym} is at {p} ({chg(ch)} over 24 hours)."])
    if rng.random() < 0.3:
        base += " " + rng.choice(["I can prepare a buy if you want one.", "Want me to prepare a buy?", "Tell me an amount in dollars if you want to buy."])
    return base


def say_no_market(rng, sym):
    return rng.choice([
        f"I can't get a live {sym} price here, there's no Hyperliquid market for it, so I won't guess a number.",
        f"There's no live price for {sym} here because Hyperliquid has no market for it. I won't guess one.",
        f"I don't have a live price for {sym}: there's no Hyperliquid market for it, and I won't make one up.",
        f"No live {sym} price from here, since there's no Hyperliquid market for it. I'd rather not guess a number.",
        f"Sorry, no live price for {sym}. Hyperliquid has no market for it, so I can't check.",
    ])


def say_open(rng, sym, dapp, r):
    return rng.choice([
        f"Hermit buys {sym} through {dapp}, so I've opened it in Hermit's browser. Nothing connects to your wallet until you approve it.",
        f"{sym} goes through {dapp} in Hermit, so I've opened it in the browser. Nothing connects to your wallet until you approve it.",
        f"For {sym}, Hermit goes through {dapp}. I've opened it in Hermit's browser, and nothing connects to your wallet until you approve it.",
        f"Hermit buys {sym} through {dapp}. I've opened it in Hermit's browser; nothing connects until you approve it.",
        f"You buy {sym} through {dapp} in Hermit. It's open in Hermit's browser now, and nothing connects to your wallet until you approve it.",
        f"I opened {dapp} in Hermit's browser, since that's how Hermit buys {sym}. Nothing connects to your wallet until you approve it.",
    ])


def say_sell(rng, sym):
    return rng.choice([
        "I can only prepare buys here. Use Swap in Wallets to sell.",
        "I can only prepare buys here. Use Swap in Wallets to sell.",
        f"I can only prepare buys here. To sell {sym}, use Swap in Wallets.",
        f"I can only prepare buys here. Use Swap in Wallets to sell your {sym}.",
    ])


def say_swap_blocked(rng, r):
    return rng.choice([f"I couldn't prepare that swap. {r['reasons'][0]} Nothing was prepared.", f"That swap was blocked. {r['reasons'][0]}",
                       f"I can't prepare it. {r['reasons'][0]} Nothing was prepared."])


# ================================================================================ scenarios
def system_for(bio):
    return SYSTEM_TPL.replace("{BIO}", bio)


def finish(c, kind, bio, intent, expected_tools, checks):
    ex = c.ex(kind)
    ex["meta"] = {"source": GEN, "generator": GEN, "kind": kind, "bio": bio, "world": c.world0, "world_final": c.w.state(),
                  "expect": c.expect, "checks": checks, "lang": "en", "intent": intent, "expected_tools": expected_tools}
    return ex


def make_world(rng, ev, bio, dev, amt, pay=None, mode="ok", sym=None):
    w = BuyWorld.new(rng, bio, dev)
    w.daily_left = float(int(amt * rng.uniform(1.3, 9) + 50))
    w.move_cap = float(int(amt * rng.uniform(1.5, 9) + 100))
    sp = w.prices["SOL"]
    # a bit of unrelated holdings on other chains
    if rng.random() < 0.5:
        w.hold("USDC", rng.choice(["base", "arbitrum", "ethereum"]), round(rng.uniform(5, 800), 2))
    if rng.random() < 0.25:
        w.hold("ETH", rng.choice(["base", "arbitrum"]), round(rng.uniform(0.001, 0.4), 4))
    usdc_ok = round(amt * rng.uniform(1.05, 6), 2)
    sol_ok = round(amt * rng.uniform(1.05, 6) / sp, 4)
    if mode == "insufficient":
        if rng.random() < 0.7:
            w.hold("USDC", "solana", round(amt * rng.uniform(0.05, 0.8), 2))
        w.hold("SOL", "solana", round(max(0.0005, amt * rng.uniform(0.02, 0.7) / sp), 4))
        return w
    if mode == "nowallet":
        w.wallet = False
        return w
    if mode == "blocked":
        w.daily_left = float(int(amt * rng.uniform(0.3, 0.8)))
        if rng.random() < 0.3:
            w.daily_left = 100000.0
            w.move_cap = float(int(amt * rng.uniform(0.3, 0.8)))
    if pay == "USDC":
        w.hold("USDC", "solana", usdc_ok)
        if rng.random() < 0.4:
            w.hold("SOL", "solana", round(rng.uniform(0.01, 3), 4))
    elif pay == "SOL":
        w.hold("SOL", "solana", sol_ok)
        if rng.random() < 0.4:
            w.hold("USDC", "solana", usdc_ok)
    elif rng.random() < 0.65:
        w.hold("USDC", "solana", usdc_ok)
        if rng.random() < 0.5:
            w.hold("SOL", "solana", round(rng.uniform(0.01, 3), 4))
    else:
        w.hold("SOL", "solana", sol_ok)
        if rng.random() < 0.5:
            w.hold("USDC", "solana", round(amt * rng.uniform(0.05, 0.9), 2))
    if mode == "impact":
        t = rng.uniform(1.1, 7.0)
        w.liq[sym] = float(int(amt * 100.0 / (t - 0.02)))
    return w


def sc_buy(rng, ev, mode="ok", style="plain", pay=None, followup=False):
    bio, dev = pick_bio(rng)
    unv = mode == "unverified"
    sym = pick_unverified(rng, ev) if unv else pick_symbol(rng, ev)
    amt = pick_amount(rng, ev, big=mode in ("impact", "blocked"))
    if style == "swap":
        pay = None
    w = make_world(rng, ev, bio, dev, amt, pay=pay, mode=mode, sym=sym)
    if style == "swap" and mode == "ok":
        w.hold("USDC", "solana", round(amt * rng.uniform(1.05, 6), 2))
    c = Conv(rng, w, system=system_for(bio))
    sl = amt_slots(rng, amt)
    sl["S"] = sym_text(rng, sym, aliases=not unv)
    if pay:
        sl["P"] = pay if rng.random() < 0.6 else pay.lower()
        tpl = rng.choice(split_tpls(PAY_TPL, ev))
    elif style == "swap":
        tpl = rng.choice(split_tpls(SWAPBUY_TPL, ev))
    else:
        tpl = rng.choice(split_tpls(BUY_TPL, ev))
    c.user(render_tpl(rng, tpl, sl))
    args = {"symbol": sym, "amount_usd": amt}
    if pay:
        args["pay_with"] = pay
    r = c.call(("prepare_buy", args))
    tools = ["prepare_buy"]
    checks = ["right_tool_new", "exact_digits", "no_unbacked_done"]
    if "error" in r:
        c.say(say_nowallet(rng, r)); kind = "buy_nowallet"
    elif r["status"] == "blocked" and unv:
        c.say(say_unverified(rng, sym)); kind = "buy_unverified"
    elif r["status"] == "blocked":
        c.say(say_blocked(rng, r)); kind = "buy_blocked"
    elif r["status"] == "insufficient":
        c.say(say_insufficient(rng, r, sym, amt)); kind = "buy_insufficient"
        if followup:
            c.user(render_tpl(rng, rng.choice(split_tpls(FUNDS_TPL, ev)), {}))
            c.call(("open_add_funds", {"goal": "wallet"}))
            c.say(say_addfunds(rng)); tools.append("open_add_funds")
    else:
        c.say(say_success(rng, r, bio)); checks.append("route_text")
        kind = "buy_impact" if r.get("warnings") else ("buy_pay" if pay else ("buy_swapstyle" if style == "swap" else "buy_ok"))
    intent = {"type": "buy", "symbol": sym, "amount_usd": amt, "pay_with": pay}
    return finish(c, kind + ("_funds" if len(tools) > 1 else ""), bio, intent, tools, checks)


def sc_ask_followup(rng, ev):
    bio, dev = pick_bio(rng)
    sym = pick_symbol(rng, ev)
    pay = rng.choice(["SOL", "USDC"]) if rng.random() < 0.3 else None
    amt = pick_amount(rng, ev)
    w = make_world(rng, ev, bio, dev, amt, pay=pay)
    c = Conv(rng, w, system=system_for(bio))
    sl = {"S": sym_text(rng, sym)}
    if pay:
        sl["P"] = pay if rng.random() < 0.6 else pay.lower()
    c.user(render_tpl(rng, rng.choice(split_tpls(ASK_PAY_TPL if pay else ASK_TPL, ev)), sl))
    c.say("How much, in dollars?", ask=True)
    a = amt_slots(rng, amt)
    c.user(render_tpl(rng, rng.choice(split_tpls(FOLLOW_TPL, ev)), a, typo_p=0))
    args = {"symbol": sym, "amount_usd": amt}
    if pay:
        args["pay_with"] = pay
    r = c.call(("prepare_buy", args))
    assert r.get("status") == "review_opened", r
    c.say(say_success(rng, r, bio))
    return finish(c, "buy_ask_followup", bio, {"type": "buy", "symbol": sym, "amount_usd": amt, "pay_with": pay}, ["prepare_buy"],
                  ["right_tool_new", "exact_digits", "asked_when_needed", "no_unbacked_done", "route_text"])


def neg_world(rng, ev, bio, dev, amt, chain):
    """USDC on exactly one chain, enough for amt."""
    w = BuyWorld.new(rng, bio, dev)
    w.daily_left = float(int(amt * rng.uniform(1.3, 9) + 50))
    w.hold("USDC", chain, round(amt * rng.uniform(1.05, 6), 2))
    if rng.random() < 0.4:
        w.hold("SOL", "solana", round(rng.uniform(0.01, 2), 4))
    return w


def sc_neg_swap(rng, ev):
    bio, dev = pick_bio(rng)
    asset = rng.choice(["ETH", "ETH", "SOL", "SOL", "BNB"])
    amt = pick_amount(rng, ev)
    named, ask_first = None, False
    if asset == "ETH":
        if rng.random() < 0.6:
            named = rng.choice(["base", "arbitrum", "ethereum"])
        else:
            ask_first = True
    if named:
        src = named
    elif ask_first:
        src = rng.choice(["base", "arbitrum", "ethereum"])          # the network the user will answer with
    elif asset == "BNB":
        src = rng.choice(["bsc", "base", "ethereum", "arbitrum"])
    else:
        src = rng.choice(["base", "arbitrum", "ethereum", "solana", "polygon", "solana"])
    w = neg_world(rng, ev, bio, dev, amt, src)
    if asset != "ETH" and rng.random() < 0.05:
        w.daily_left = float(int(amt * rng.uniform(0.3, 0.8)))      # policy blocked
    c = Conv(rng, w, system=system_for(bio))
    st = {"ETH": ["ETH", "eth", "ether"], "SOL": ["SOL", "sol", "solana"], "BNB": ["BNB", "bnb"]}[asset]
    sl = amt_slots(rng, amt)
    sl["S"] = rng.choice(st)
    text = render_tpl(rng, rng.choice(split_tpls(NEG_BUY_TPL, ev)), sl)
    if named:
        text += rng.choice([" on {C}", " on {C} pls", " ({C})", ", {C}"]).replace("{C}", rng.choice(CHAIN_SAY[named]))
    c.user(text)
    to_chain = {"ETH": src, "SOL": "solana", "BNB": "bsc"}[asset]
    checks = ["right_tool_new", "exact_digits", "no_unbacked_done"]
    tools = ["prepare_swap"]
    if ask_first:
        c.say(rng.choice(["Which network should the ETH be on: Ethereum, Base or Arbitrum?", "Which network do you want the ETH on, Ethereum, Base or Arbitrum?",
                          "Which network should I buy the ETH on: Ethereum, Base or Arbitrum?", "Which network for the ETH: Ethereum, Base or Arbitrum?"]), ask=True)
        if rng.random() < 0.5:
            return finish(c, "neg_swap_ask", bio, {"type": "swap_ask", "asset": asset, "amount": amt}, [], checks + ["asked_when_needed"])
        c.user(rng.choice(["{C}", "on {C}", "{C} pls", "{C} is fine", "go with {C}", "{C}, thanks"]).replace("{C}", rng.choice(CHAIN_SAY[src])))
        named = src
    elif not named:
        br = c.call(("get_balances", {}))
        src = [h["chain"] for h in br["holdings"] if h["symbol"] == "USDC"][0]
        tools = ["get_balances", "prepare_swap"]
    args = {"from_symbol": "USDC", "from_chain": src, "to_symbol": asset, "to_chain": to_chain, "amount": amt}
    r = c.call(("prepare_swap", args))
    if r.get("status") == "blocked":
        c.say(say_swap_blocked(rng, r))
    else:
        assert r["status"] == "prepared_for_review", r
        c.say(say_swap(rng, r, bio)); checks.append("route_text")
    return finish(c, "neg_swap", bio, {"type": "swap", "asset": asset, "amount": amt, "chain": named}, tools, checks)


def sc_neg_open(rng, ev, dapp):
    bio, dev = pick_bio(rng)
    syms = {"Hyperliquid": ["HYPE", "BTC"], "Uniswap": ["PEPE", "SHIB"]}[dapp]
    asset = rng.choice(syms)
    text_forms = {"HYPE": ["HYPE", "hype", "$HYPE"], "BTC": ["BTC", "btc", "bitcoin", "Bitcoin"], "PEPE": ["PEPE", "pepe", "$PEPE"],
                  "SHIB": ["SHIB", "shib", "$SHIB", "shiba inu"]}[asset]
    w = BuyWorld.new(rng, bio, dev)
    amt = pick_amount(rng, ev)
    if rng.random() < 0.5:
        w.hold("USDC", rng.choice(["solana", "base", "arbitrum"]), round(amt * rng.uniform(1.05, 6), 2))
    c = Conv(rng, w, system=system_for(bio))
    sl = amt_slots(rng, amt)
    sl["S"] = rng.choice(text_forms)
    if not ev and rng.random() < 0.2:
        text = render_tpl(rng, rng.choice(NEG_NOAMT_TPL), sl)
    else:
        text = render_tpl(rng, rng.choice(split_tpls(NEG_BUY_TPL, ev)), sl)
    c.user(text)
    r = c.call(("open_app", {"name": dapp}))
    assert r["status"] == "opened", r
    c.say(say_open(rng, asset, dapp, r))
    return finish(c, "neg_open_" + dapp.lower(), bio, {"type": "open_app", "asset": asset, "dapp": dapp}, ["open_app"],
                  ["right_tool_new", "no_unbacked_done"])


def sc_neg_price(rng, ev):
    bio, dev = pick_bio(rng)
    err = rng.random() < 0.15
    sym = rng.choice(NO_MARKET) if err else rng.choice(PRICEABLE)
    w = BuyWorld.new(rng, bio, dev)
    c = Conv(rng, w, system=system_for(bio))
    c.user(render_tpl(rng, rng.choice(split_tpls(NEG_PRICE_TPL, ev)), {"S": sym_text(rng, sym)}))
    r = c.call(("get_price", {"symbol": sym}))
    if "error" in r:
        c.say(say_no_market(rng, sym))
    else:
        c.say(say_price(rng, sym, r))
    return finish(c, "neg_price", bio, {"type": "price", "symbol": sym}, ["get_price"], ["right_tool_new", "no_unbacked_done"])


def sc_neg_view(rng, ev):
    bio, dev = pick_bio(rng)
    sym = rng.choice(PRICEABLE)
    w = BuyWorld.new(rng, bio, dev)
    c = Conv(rng, w, system=system_for(bio))
    c.user(render_tpl(rng, rng.choice(split_tpls(NEG_VIEW_TPL, ev)), {"S": sym_text(rng, sym)}))
    r = c.call(("get_price", {"symbol": sym}))
    c.say(say_view(rng, sym, r))
    return finish(c, "neg_view", bio, {"type": "view", "symbol": sym}, ["get_price"], ["right_tool_new", "no_unbacked_done"])


def sc_neg_sell(rng, ev):
    bio, dev = pick_bio(rng)
    sym = rng.choice(SELL_TOK)
    w = BuyWorld.new(rng, bio, dev)
    if rng.random() < 0.5:
        w.hold(sym if sym in w.prices else "SOL", "solana", round(rng.uniform(1, 500), 2))
    c = Conv(rng, w, system=system_for(bio))
    amt = pick_amount(rng, ev)
    sl = amt_slots(rng, amt)
    sl["S"] = sym_text(rng, sym, aliases=False)
    c.user(render_tpl(rng, rng.choice(split_tpls(NEG_SELL_TPL, ev)), sl))
    c.say(say_sell(rng, sym))
    return finish(c, "neg_sell", bio, {"type": "sell", "symbol": sym}, [], ["right_tool_new", "no_unbacked_done"])


def sc_neg_ask(rng, ev):
    bio, dev = pick_bio(rng)
    sym = pick_symbol(rng, ev)
    pay = rng.choice(["SOL", "USDC"]) if rng.random() < 0.25 else None
    w = BuyWorld.new(rng, bio, dev)
    c = Conv(rng, w, system=system_for(bio))
    sl = {"S": sym_text(rng, sym)}
    if pay:
        sl["P"] = pay if rng.random() < 0.6 else pay.lower()
    c.user(render_tpl(rng, rng.choice(split_tpls(ASK_PAY_TPL if pay else ASK_TPL, ev)), sl))
    c.say("How much, in dollars?", ask=True)
    return finish(c, "neg_ask", bio, {"type": "ask", "symbol": sym, "pay_with": pay}, [], ["right_tool_new", "asked_when_needed", "no_unbacked_done"])


# (name, fn, train weight, eval count)
PLAN = [
    ("buy_ok",           lambda r, e: sc_buy(r, e, "ok"),                                 100, 8),
    ("buy_pay",          lambda r, e: sc_buy(r, e, "ok", pay=r.choice(["SOL", "USDC"])),   40, 4),
    ("buy_swapstyle",    lambda r, e: sc_buy(r, e, "ok", style="swap"),                    14, 2),
    ("buy_ask_followup", sc_ask_followup,                                                 16, 2),
    ("buy_impact",       lambda r, e: sc_buy(r, e, "impact"),                              20, 3),
    ("buy_unverified",   lambda r, e: sc_buy(r, e, "unverified"),                          20, 3),
    ("buy_insufficient", lambda r, e: sc_buy(r, e, "insufficient", followup=r.random() < 0.25), 16, 3),
    ("buy_blocked",      lambda r, e: sc_buy(r, e, "blocked"),                              8, 2),
    ("buy_nowallet",     lambda r, e: sc_buy(r, e, "nowallet"),                             6, 1),
    ("neg_swap",         sc_neg_swap,                                                      26, 3),
    ("neg_open_hl",      lambda r, e: sc_neg_open(r, e, "Hyperliquid"),                    18, 2),
    ("neg_open_uni",     lambda r, e: sc_neg_open(r, e, "Uniswap"),                        12, 1),
    ("neg_price",        sc_neg_price,                                                     14, 2),
    ("neg_view",         sc_neg_view,                                                      12, 1),
    ("neg_sell",         sc_neg_sell,                                                      12, 1),
    ("neg_ask",          sc_neg_ask,                                                       16, 2),
]
assert sum(p[2] for p in PLAN) == 350 and sum(p[3] for p in PLAN) == 40


def allocate(total, weights):
    s = sum(weights)
    raw = [total * w / s for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[:total - sum(out)]:
        out[i] += 1
    return out


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$.% ]", "", t.lower())).strip()


def build(seed, counts, ev, forbid_users=()):
    rng = random.Random(seed)
    rows, seen = [], set(forbid_users)
    for (name, fn, _, _), n in zip(PLAN, counts):
        made = 0
        while made < n:
            ex = fn(random.Random(rng.getrandbits(64)), ev)
            users = [norm(m["content"]) for m in ex["messages"] if m["role"] == "user"]
            if users[0] in seen or any(u in seen and len(u) > 12 for u in users):
                continue                       # duplicate user text: draw again
            seen.add(users[0])
            ex["meta"]["plan"] = name
            rows.append(ex)
            made += 1
    rng.shuffle(rows)
    return rows


# ================================================================================ self-check
BANNED = [r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|deposited|withdrawn|closed)",
          r"\bI(?:'ve| have)? (sent|bought|swapped|deposited|withdrew|closed|purchased)\b", r"successfully",
          r"funds (are|have) (in|arrived)", r"^\s*done\b", r"not financial advice", r"[·•]",
          r"what('s| is) (the )?(current )?price", r"your budget", r"how much (do you|are you)", r"\b(bought|swapped|purchased)\b",
          r"\bAs an AI\b", r"\b10 dollars by default\b"]
ALLOW_NEG = [r"nothing has been bought yet", r"nothing is bought until you approve"]
NUM_RE = re.compile(r"\d[\d,]*\.?\d*(?:e-?\d+)?")
ADDR_RE = re.compile(r"0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44}")


def nums(text):
    out = set()
    for m in NUM_RE.findall(text):
        m = m.replace(",", "").rstrip(".")
        try:
            out.add(Decimal(m).normalize())
        except Exception:
            pass
    return out


def amount_in_text(v, text):
    t = text.lower().replace(",", "")
    forms = {jstr(v), f"{float(v):.2f}", f"{float(v):g}"}
    if float(v) % 1000 == 0:
        forms.add(f"{int(v // 1000)}k")
    if v in WORDS:
        forms.add(WORDS[v])
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d]|\.\d)", t) for f in forms)


def check(ex, ev=False):
    meta, msgs = ex["meta"], ex["messages"]
    json.loads(json.dumps(ex, ensure_ascii=False))
    assert ex["tools"] is True and meta["source"] == GEN
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    blob = json.dumps(msgs, ensure_ascii=False)
    assert other not in blob, "other biometric word"
    assert "{BIO}" not in blob and "{" not in " ".join(m["content"] for m in msgs if m["role"] == "assistant"), "template leftover"
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(SYSTEM_TPL.split("{BIO}")[0]) and bio in msgs[0]["content"]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    user_blob = " ".join(users)
    # tool calls
    called, results = [], []
    for i, m in enumerate(msgs):
        for tc in m.get("tool_calls") or []:
            sch = SCHEMA[tc["name"]]                        # KeyError if the tool does not exist
            a = tc["arguments"]
            for req in sch.get("required", []):
                assert req in a, (tc, req)
            for k, v in a.items():
                assert k in sch["properties"], ("unknown arg", tc, k)
                t = sch["properties"][k]["type"]
                assert (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)) or (t == "string" and isinstance(v, str)), (tc, k, v)
                if "enum" in sch["properties"][k]:
                    assert v in sch["properties"][k]["enum"], (tc, k, v)
            called.append(tc["name"])
            for v in a.values():
                if isinstance(v, str) and ADDR_RE.fullmatch(v):
                    assert v in user_blob, "address not copied"
            if tc["name"] == "prepare_buy":
                intent = meta["intent"]
                assert a["symbol"] == a["symbol"].upper() and a["symbol"] == intent["symbol"], a
                assert isinstance(a["amount_usd"], (int, float)) and a["amount_usd"] == intent["amount_usd"] and a["amount_usd"] > 0
                assert amount_in_text(a["amount_usd"], user_blob), ("amount not in user text", a, user_blob)
                assert re.search(r"(?<![a-z])" + a["symbol"].lower() + r"(?![a-z])", user_blob.lower()) or any(
                    al in user_blob.lower() for al in ALIAS.get(a["symbol"], [])), ("ticker not in user text", a)
                m_pay = PAYRE.search(user_blob)
                if "pay_with" in a:
                    assert m_pay and m_pay.group(1).upper() == a["pay_with"] and a["pay_with"] == intent["pay_with"], ("pay_with without user saying so", a, user_blob)
                else:
                    assert not m_pay, ("user said pay token but pay_with missing", user_blob)
                    assert intent["pay_with"] is None
            if tc["name"] == "prepare_swap":
                assert a["from_symbol"] == "USDC" and a["to_symbol"] in ("ETH", "SOL", "BNB") and a["amount"] == meta["intent"]["amount"]
                assert amount_in_text(a["amount"], user_blob)
                assert a["to_chain"] in ("ethereum", "base", "arbitrum", "solana", "bsc")
                chain_words = {"ethereum": r"ethereum|mainnet", "base": r"\bbase\b", "arbitrum": r"arbitrum|\barb\b"}
                if a["to_symbol"] == "ETH":
                    assert re.search(chain_words[a["to_chain"]], user_blob.lower()), "ETH chain not named by user"
            if tc["name"] == "open_app":
                assert a == {"name": meta["intent"]["dapp"]}
    assert called == meta["expected_tools"], (called, meta["expected_tools"])
    assert (not called) or called[0] != "prepare_buy" or meta["kind"] != "buy_ask_followup" or len(users) == 2
    # replay: recorded results equal a fresh world's results
    w = BuyWorld.from_state(meta["world"])
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    calls = [tc for m in msgs for tc in (m.get("tool_calls") or [])]
    assert len(tool_msgs) == len(calls)
    for tc, tm in zip(calls, tool_msgs):
        assert json.dumps(w.call(tc["name"], tc["arguments"]), ensure_ascii=False) == tm["content"], ("replay mismatch", tc)
        results.append(tm["content"])
    assert BuyWorld.from_state(meta["world_final"]).state() == meta["world_final"]
    # assistant text
    allowed = set()
    for t in results + users + [json.dumps(tc["arguments"]) for tc in calls]:
        allowed |= nums(t)
    allowed |= {Decimal(24)}                                    # "24 hours"
    answers = [m["content"] for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
    assert answers and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls")
    for t in answers:
        assert t.isascii(), t
        low = t
        for al in ALLOW_NEG:
            low = re.sub(al, "", low, flags=re.I)
        for b in BANNED:
            assert not re.search(b, low, flags=re.I | re.M), (b, t)
        assert len(t.split()) <= (110 if meta["kind"] == "neg_view" else 70), ("too long", len(t.split()), t)
        for n in nums(t):
            assert n in allowed, ("number not backed", n, t)
        for a in ADDR_RE.findall(t):
            assert a in user_blob, "address not from user"
        assert "Touch ID" not in t or bio == "Touch ID"
    final = answers[-1]
    last = results[-1] if results else ""
    kind = meta["kind"]
    if kind in ("buy_ok", "buy_pay", "buy_swapstyle", "buy_impact", "buy_ask_followup"):
        r = json.loads(last)
        assert r["status"] == "review_opened" and r["route"] in final and "Jupiter" in final and r["you_pay"] in final and r["you_get"] in final
        assert (("Approve" in final or "approve" in final) and bio in final) and re.search(r"[Nn]othing (has been|is) bought", final), final
        assert str(r["price_impact_pct"]) in final.replace("%", "") or jstr(r["price_impact_pct"]) in final
        if r["price_impact_pct"] >= 1:
            assert r["warnings"][0] in final, "high impact warning not repeated"
        else:
            assert "warnings" not in r
        if kind == "buy_impact":
            assert r["price_impact_pct"] >= 1
        else:
            assert r["price_impact_pct"] < 1
    elif kind == "buy_unverified":
        r = json.loads(last)
        assert r["status"] == "blocked" and "verified" in final and meta["intent"]["symbol"] in final
    elif kind == "buy_blocked":
        assert json.loads(last)["reasons"][0] in final
    elif kind.startswith("buy_insufficient"):
        pr = json.loads(results[0])
        assert pr["status"] == "insufficient" and pr["holding"] in answers[0]
        if not kind.endswith("_funds"):
            assert not re.search(r"add funds|deposit", final, re.I), "offered add funds unasked"
        else:
            assert "button" in final.lower() and "opened" not in final.lower()
    elif kind == "buy_nowallet":
        assert "No wallet" in last and "wallet" in final
    elif kind == "neg_swap":
        if called:
            r = json.loads(last)
            if r["status"] == "prepared_for_review":
                assert r["from"] in final and r["to"] in final and r["route"] == "NEAR Intents" and "NEAR Intents" in final
            else:
                assert r["reasons"][0] in final
    elif kind.startswith("neg_open"):
        assert meta["intent"]["dapp"] in final and "Nothing connects" in final or "nothing connects" in final
    elif kind == "neg_price":
        r = json.loads(last)
        if "error" in r:
            assert re.search(r"no Hyperliquid market|Hyperliquid has no market|Hyperliquid has no", final) and "$" not in final
        else:
            assert price_txt(r["price"]) in final
    elif kind == "neg_view":
        r = json.loads(last)
        assert price_txt(r["price"]) in final and "prepare" in final
    elif kind == "neg_sell":
        assert "I can only prepare buys here." in final and "Use Swap in Wallets" in final or "use Swap in Wallets" in final
        assert not called
    elif kind in ("neg_ask",):
        assert final == "How much, in dollars?" and not called
    if kind == "buy_ask_followup":
        assert answers[0] == "How much, in dollars?" and msgs[2]["role"] == "assistant" and "tool_calls" not in msgs[2]
    # expect block
    exp = meta["expect"]
    assert exp and all(("calls" in e) or ("ask" in e) or ("text" in e) for e in exp)
    for m in msgs:                                               # tool-call text round trip
        if m.get("tool_calls"):
            got, bad = H.parse_calls(H.assistant_text(m))
            assert bad == 0 and got == m["tool_calls"], "call round trip"
    return True


def check_all(rows, ev, other_users=()):
    seen_first, seen_all = set(), set()
    for ex in rows:
        check(ex, ev)
        users = [norm(m["content"]) for m in ex["messages"] if m["role"] == "user"]
        assert users[0] not in seen_first, ("duplicate user text", users[0])
        seen_first.add(users[0])
        for u in users:
            assert u not in other_users or len(u) <= 4, ("user text also in the other split", u)
            seen_all.add(u)
    return seen_all


def eval_item(ex, i):
    msgs = ex["messages"]
    last_u = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    meta = ex["meta"]
    calls = [tc for m in msgs[last_u + 1:] for tc in (m.get("tool_calls") or [])]
    final = [m["content"] for m in msgs[last_u + 1:] if m["role"] == "assistant" and not m.get("tool_calls")][-1]
    kind, intent = meta["kind"], meta["intent"]
    last_res = [json.loads(m["content"]) for m in msgs[last_u + 1:] if m["role"] == "tool"]
    r = last_res[-1] if last_res else {}
    must, ask = [], bool(meta["expect"] and "ask" in meta["expect"][-1])
    if kind.startswith("buy_") and r.get("status") == "review_opened":
        must = [r["route"], r["you_pay"], r["you_get"], "Jupiter", meta["bio"]] + r.get("warnings", [])
    elif kind == "buy_unverified":
        must = [intent["symbol"], "verified"]
    elif kind == "buy_blocked":
        must = r["reasons"]
    elif kind.startswith("buy_insufficient") and r.get("holding"):
        must = [r["holding"]]
    elif kind == "neg_swap" and r.get("status") == "prepared_for_review":
        must = [r["from"], r["to"], "NEAR Intents"]
    gold = {"calls": calls, "expects_prepare_buy": any(c["name"] == "prepare_buy" for c in calls), "ask": ask,
            "no_call": not calls, "must_contain": must,
            "must_not_match": [r"\bRelay\b", r"\bbought\b(?!\s+yet)", r"went through", r"\bswapped\b", r"successfully"]}
    if ask:
        gold["answer_exact"] = "How much, in dollars?" if kind in ("neg_ask",) else None
    if intent["type"] == "buy":
        gold["args"] = {"symbol": intent["symbol"], "amount_usd": intent["amount_usd"], "pay_with": intent["pay_with"]}
    return {"id": f"{GEN}-{kind}-{i}", "kind": kind, "tools": True, "bio": meta["bio"], "context": msgs[:last_u + 1],
            "reference": msgs[last_u + 1:], "expect": meta["expect"], "world": meta["world"], "checks": meta["checks"], "gold": gold,
            "meta": {"source": GEN, "generator": GEN, "lang": "en"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--eval-seed", type=int, default=9311)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(LAB, "data3", "buy_jupiter.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval3", "buy_jupiter.jsonl"))
    a = ap.parse_args()
    ev_rows = build(a.eval_seed, allocate(a.eval_n, [p[3] for p in PLAN]), True)
    ev_users = set()
    for ex in ev_rows:
        for m in ex["messages"]:
            if m["role"] == "user":
                ev_users.add(norm(m["content"]))
    tr_rows = build(a.seed, allocate(a.n, [p[2] for p in PLAN]), False, forbid_users=ev_users)
    tr_users = check_all(tr_rows, False, other_users=ev_users)
    check_all(ev_rows, True, other_users=tr_users)
    with open(a.out, "w") as f:
        for ex in tr_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for i, ex in enumerate(ev_rows):
            f.write(json.dumps(eval_item(ex, i), ensure_ascii=False) + "\n")
    print("kinds train:", dict(Counter(e["meta"]["kind"] for e in tr_rows)))
    print("bio train:", dict(Counter(e["meta"]["bio"] for e in tr_rows)))
    print(f"buy_jupiter: {len(tr_rows)} train, {len(ev_rows)} eval, self-check ok")


if __name__ == "__main__":
    main()
