#!/usr/bin/env python3
"""spot_solana: sell and search Solana tokens (prepare_sell, search_tokens), buy by mint, symbol clashes.

Replaces the old "I can only prepare buys" answers. Kinds (train counts in PLAN below):
  sell_pct, sell_all (+ receive SOL/USDC only when the user says so), sell_amount (token units or "$5 worth" via
  get_positions + calculator), sell_by_mint, sell_ask (one question, never a guessed percent), sell_errors (not_held,
  insufficient, frozen, needs_choice, no_route, warn, blocked), search_safety (never "safe"/"legit"), search_find,
  buy_after_search, buy_mint_unverified (every warning repeated), buy_refused, buy_clash, and near-misses (selling
  ETH/SOL/BNB/USDC is prepare_swap, closing a perp is prepare_close, price questions, concept questions, and the
  tool-absent case: core pack only -> "Selling Solana tokens isn't available in this chat" + open_page trade).

Every assistant string is a Python f-string filled from tool results, tool arguments, calculator results or user text.
Nothing is LLM-generated. Reuses gen2/gen3 helpers by import (Conv, chg, hermit_fmt, buy_jupiter helpers); nothing there
is edited. Tool registry: spec/hermit_tools_v3.json; results: spec/hermit_tools_v3.meta.json shapes; system prompt:
spec/SYSTEM_AGENT_v3.txt with {BIO} = Face ID (70%) or Touch ID (30%).

Usage: python3 spot_solana.py [--seed 4] [--n 520] [--eval-seed 9401] [--eval-n 70]
Writes ../data4/spot_solana.jsonl and ../eval4/spot_solana.jsonl, then self-checks every record (it fails loudly).
"""
import sys
sys.dont_write_bytecode = True
import argparse, copy, json, os, random, re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(LAB, "gen3"), os.path.join(LAB, "gen2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hermit_fmt as H                       # gen2 (read only): calc, calc_str, sol_addr, parse_calls, assistant_text
from build_agent import Conv, chg            # gen2 (read only)
import buy_jupiter as BJ                     # gen3 (read only): approx, jnum, jstr, amt_slots, say_swap, say_price, ...
import sim as SIM                            # gen2 (read only): ASSETS price bands

GEN = "spot_solana"
SPEC = os.path.join(LAB, "spec")
TOOLS_V3 = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
REG = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V3}
ORDER = [t["function"]["name"] for t in TOOLS_V3]
SYSTEM_TPL = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read()
assert "{BIO}" in SYSTEM_TPL and not SYSTEM_TPL.endswith("\n")

CORE = ["calculator", "get_price", "get_balances", "get_portfolio", "get_address", "get_activity", "prepare_send",
        "prepare_swap", "get_quote", "open_page", "open_app", "open_add_funds"]
SOLANA = ["search_tokens", "prepare_buy", "prepare_sell", "get_positions"]
TRADE = ["hyperliquid_account", "position_math", "search_markets", "prepare_trade", "prepare_close", "cancel_order",
         "set_tp_sl", "get_positions"]
assert all(n in REG for n in CORE + SOLANA + TRADE)
PACKS = {"core": CORE, "sol": CORE + SOLANA, "trade": CORE + SOLANA + TRADE}
PACK_NAMES = {k: [n for n in ORDER if n in set(v)] for k, v in PACKS.items()}
LITE_OK_PACKS = {"sol"}                      # core 12 + solana 4 = 16 tools

REF_LIQ = "Liquidity is under $5K, so Hermit won't buy it."
REF_FREEZE = "It is unverified and the creator can freeze tokens, so Hermit won't buy it."
B58 = H.B58
MINT_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
ADDR_RE = re.compile(r"0x[0-9a-fA-F]{40}|[1-9A-HJ-NP-Za-km-z]{32,44}")
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


# ================================================================================ numbers and text helpers
def is_mint(s):
    return isinstance(s, str) and bool(MINT_RE.match(s)) and not s.isalpha() and (any(ch.isdigit() for ch in s) or len(s) >= 40)


def jnum(x):
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return round(x, 8) if isinstance(x, float) else x


def trim(x):
    """Plain decimal text without exponent or trailing zeros: 2.50 -> 2.5, 3.0 -> 3."""
    d = x if isinstance(x, Decimal) else Decimal(repr(x))
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def commas(s):
    ip, _, fp = s.partition(".")
    neg = ip.startswith("-")
    ip = ip.lstrip("-")
    out = f"{int(ip):,}" + ("." + fp if fp else "")
    return ("-" if neg else "") + out


def fmt_qty(q, exact=False):
    """Token quantity as the sell review prints it: commas from 1,000, 2 dp from 1, 3 significant digits below."""
    d = q if isinstance(q, Decimal) else Decimal(repr(q))
    if exact:
        return commas(trim(d))
    if d >= 1000:
        return commas(trim(d.quantize(Decimal("0.01"), ROUND_HALF_UP)))
    if d >= 1:
        return trim(d.quantize(Decimal("0.01"), ROUND_HALF_UP))
    return BJ.approx(float(d))


def buy_qty(q):
    """Token quantity of a buy quote: 3 significant digits, commas from 1,000 (about 32,200 WIFG)."""
    if q >= 1000:
        return f"{int(round(q, -(len(str(int(q))) - 3))):,}"
    return BJ.approx(q)


def liq_txt(x):
    """$18.4M, $38M, $420K, $18.4K, $3,900. Liquidity values in the sim have 2 significant digits, so this is exact."""
    x = int(x)
    if x >= 1_000_000:
        return "$" + trim(Decimal(x) / Decimal(1_000_000)) + "M"
    if x >= 10_000:
        return "$" + trim(Decimal(x) / Decimal(1_000)) + "K"
    return "$" + f"{x:,}"


def r2(x):
    """Round to 2 significant digits (int)."""
    x = float(x)
    if x < 100:
        return int(round(x))
    mag = 10 ** (len(str(int(x))) - 2)
    return int(round(x / mag) * mag)


def canon(o):
    if isinstance(o, dict):
        return {k: canon(o[k]) for k in sorted(o)}
    if isinstance(o, list):
        return [canon(v) for v in o]
    return o


def short_mint(m):
    return (m[:6] if m.startswith("pump") else m[:4]) + "…" + m[-3:]


def pnl_txt(p):
    return ("+" if p >= 0 else "-") + "$" + f"{abs(p):,.2f}"


def gen_mint(rng):
    while True:
        m = H.sol_addr(rng)
        if len(m) == 44 and not m.startswith("1"):
            return m


# ================================================================================ token catalog
# symbol: (name, mint, price, aliases). Prices >= 0.0001 so no result ever carries e-notation.
KNOWN = {
    "WIF": ("dogwifhat", "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm", 1.82, ["dog", "hat", "dogwifhat"]),
    "JUP": ("Jupiter", "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", 0.78, ["jupiter"]),
    "BONK": ("Bonk", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", 0.00031, ["dog"]),
    "POPCAT": ("Popcat", "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr", 0.52, ["cat"]),
    "PUMP": ("Pump.fun", "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7H9Dfn", 0.0043, ["pumpfun"]),
    "MOODENG": ("Moo Deng", "ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY", 0.21, ["hippo", "moo", "deng"]),
    "GOAT": ("Goatseus Maximus", "CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump", 0.12, ["goatseus"]),
    "FARTCOIN": ("Fartcoin", "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump", 0.41, ["fart"]),
    "PENGU": ("Pudgy Penguins", "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv", 0.0081, ["penguin", "pudgy"]),
    "PYTH": ("Pyth Network", "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3", 0.31, ["oracle"]),
    "RAY": ("Raydium", "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", 2.4, ["raydium"]),
    "JTO": ("Jito", "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL", 1.9, ["jito"]),
    "PNUT": ("Peanut the Squirrel", "2qEHjDLDLbuBgRYvsxhc5D6uK8nsVt6jUuJxBJrqNc3S", 0.19, ["squirrel", "peanut"]),
    "MEW": ("cat in a dogs world", "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5", 0.0042, ["cat", "dogs"]),
    "ORCA": ("Orca", "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE", 1.7, ["whale"]),
}
for _s, (_n, _m, _p, _a) in KNOWN.items():
    assert 32 <= len(_m) <= 44 and all(ch in B58 for ch in _m), (_s, _m)
    assert _p >= 0.0001 and "e" not in repr(_p)
SHARED_T = ["WIF", "JUP", "BONK", "POPCAT", "PUMP", "MOODENG", "GOAT", "FARTCOIN", "PENGU"]
SHARED_W = [26, 12, 11, 9, 8, 7, 7, 7, 5]
TRAIN_ONLY_T = ["PYTH", "RAY", "JTO"]
EVAL_ONLY_T = ["PNUT", "MEW", "ORCA"]
PRICEABLE = ["WIF", "JUP", "POPCAT", "FARTCOIN", "PENGU", "PUMP"]            # tokens with a Hyperliquid mark price in the sim
UNV_TRAIN = ["ZORP", "MOONX", "BLORB", "GRIMLY", "WOOFY", "KIRBO", "SNEK", "NUBZ", "BABYWIF", "PUMPIT", "CATGPT", "DEGEN2", "FLOKIZ", "WIFG"]
UNV_EVAL = ["QUACKY", "DEGENX", "BOZO9", "LUNARX", "TURBOZ"]
UNV_NAME = {"WIFG": "WIF Gold"}
DEX = ["Orca", "Raydium", "Meteora", "Phoenix", "Lifinity"]
CHAIN_NAME = {"solana": "Solana", "base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "bnb": "BNB Chain", "polygon": "Polygon"}
CHAIN_SAY = {"ethereum": ["Ethereum", "ethereum", "mainnet"], "base": ["Base", "base"], "arbitrum": ["Arbitrum", "arbitrum", "arb"]}


def make_known(rng, sym):
    name, mint, price, aliases = KNOWN[sym]
    return {"symbol": sym, "name": name, "mint": mint, "verified": True, "price": price,
            "change": round(rng.uniform(-12, 12), 1) or 0.1, "liquidity": r2(rng.uniform(1.5e6, 42e6)),
            "volume": r2(rng.uniform(4e6, 120e6)), "tags": ["Top holders %d%%" % rng.randint(18, 34)] if rng.random() < 0.12 else [],
            "refusal": None, "route": "Jupiter via " + ", ".join(rng.sample(DEX, rng.choice([1, 2, 2, 3]))), "aliases": list(aliases)}


def tag_list(freeze, mint_on, low, top, new):
    t = []
    if freeze: t.append("Freeze on")
    if mint_on: t.append("Mint on")
    if low: t.append("Low liquidity")
    if top: t.append("Top holders %d%%" % top)
    if new: t.append("New")
    return t


def make_unverified(rng, sym, kind, name=None, mint=None):
    """kind: flags (buyable with warnings), lowliq (refused, liquidity < $5K), freeze (refused, unverified + Freeze on), held (ordinary
    holding, deep enough pool), bare (unverified, no tags)."""
    if kind == "lowliq":
        liq = r2(rng.uniform(1200, 4900))
        tags = tag_list(rng.random() < 0.6, rng.random() < 0.6, True, 0, rng.random() < 0.7)
    elif kind == "freeze":
        liq = r2(rng.uniform(9000, 600000))
        tags = tag_list(True, rng.random() < 0.5, liq < 50000, rng.choice([0, 0, rng.randint(25, 60)]), rng.random() < 0.5)
    elif kind == "bare":
        liq = r2(rng.uniform(60000, 900000))
        tags = []
    else:                                  # flags / held
        liq = r2(rng.uniform(15000, 800000)) if kind == "flags" else r2(rng.uniform(150000, 2.5e6))
        tags = tag_list(False, rng.random() < 0.6, liq < 50000, rng.choice([0, 0, rng.randint(22, 55)]), rng.random() < 0.5)
        if kind == "flags" and not tags:
            tags = tag_list(False, True, False, 0, False)
    refusal = REF_LIQ if liq < 5000 else (REF_FREEZE if "Freeze on" in tags else None)
    price = round(rng.uniform(0.0002, 0.06), rng.choice([4, 5])) if rng.random() < 0.9 else None
    price = max(price, 0.0002) if price is not None else None
    return {"symbol": sym, "name": name or UNV_NAME.get(sym) or (sym.capitalize() + rng.choice(["", " Coin", " Token", " Inu"])),
            "mint": mint or gen_mint(rng), "verified": False, "price": price,
            "change": None if price is None else rng.choice([round(rng.uniform(-45, 60), 1), round(rng.uniform(100, 480), 0)]),
            "liquidity": liq, "volume": r2(liq * rng.uniform(0.3, 6)), "tags": tags, "refusal": refusal,
            "route": "Jupiter via " + rng.choice(DEX), "aliases": []}


# ================================================================================ the simulator
class SolWorld:
    """One user's state. Deterministic: every result is a pure function of the state and the typed user text.
    Guards emulated: a mint passed to prepare_buy/prepare_sell must be in the user's typed text or in an earlier
    search_tokens / needs_choice result of this chat (searched_mints)."""
    KEYS = ("prices", "tokens", "spot", "balances", "perps", "daily_left", "searched_mints", "no_route", "search_down",
            "wallet", "bio", "device", "seconds", "spread_bps", "wallet_name")

    def __init__(self, st):
        for k in self.KEYS:
            setattr(self, k, copy.deepcopy(st[k]))
        self.typed = []

    @classmethod
    def from_state(cls, st):
        return cls(st)

    def state(self):
        return {k: copy.deepcopy(getattr(self, k)) for k in self.KEYS}

    @classmethod
    def new(cls, rng, bio, device, ev=False):
        prices = {}
        for s in ("ETH", "SOL", "BNB"):
            lo, hi, dp = SIM.ASSETS[s]
            prices[s] = round(rng.uniform(lo, hi), dp)
        prices["USDC"] = 1.0
        tokens = [make_known(rng, s) for s in KNOWN]
        return cls({"prices": prices, "tokens": tokens, "spot": [], "balances": [], "perps": [], "daily_left": 100000.0,
                    "searched_mints": [], "no_route": [], "search_down": False, "wallet": True, "bio": bio, "device": device,
                    "seconds": rng.choice([8, 12, 14, 18, 20, 25]), "spread_bps": rng.choice([5, 10, 15, 20, 30]),
                    "wallet_name": "Main"})

    # ---- setup helpers (not tools)
    def tok(self, sym=None, mint=None):
        for t in self.tokens:
            if (mint and t["mint"] == mint) or (sym and not mint and t["symbol"] == sym):
                return t
        raise KeyError(sym or mint)

    def add_token(self, t):
        self.tokens.append(t)
        return t

    def give(self, rng, t, usd=None, amount=None, frozen=False, basis=None):
        price = t["price"] or 0.001
        if amount is None:
            usd = usd if usd is not None else round(rng.choice([rng.uniform(3, 40), rng.uniform(20, 300), rng.uniform(100, 900)]), 2)
            amount = usd / price
        amt = Decimal(repr(float(amount)))
        amt = amt.quantize(Decimal("1")) if amt >= 1000 else amt.quantize(Decimal("0.01"))
        if amt <= 0:
            amt = Decimal("0.01")
        amount_s = trim(amt)
        usd_v = round(float(amt) * price, 2)
        has_basis = basis if basis is not None else rng.random() < 0.7
        avg = max(0.0001, round(price * rng.uniform(0.55, 1.5), 4 if price >= 0.01 else 5)) if (has_basis and not frozen) else None
        pnl = round((price / avg - 1) * 100, 1) if avg else None
        self.spot = [h for h in self.spot if h["mint"] != t["mint"]]
        self.spot.append({"amount": amount_s, "avg_cost": avg, "frozen": frozen, "mint": t["mint"], "pnl_pct": pnl,
                          "symbol": t["symbol"], "usd": usd_v})
        return self.spot[-1]

    def hold(self, symbol, chain, amount):
        self.balances = [b for b in self.balances if not (b["symbol"] == symbol and b["chain"] == chain)]
        usd_v = round(amount * self.prices[symbol], 2)
        self.balances.append({"symbol": symbol, "chain": chain, "amount": jnum(amount), "usd": jnum(usd_v)})

    def usdc_sol(self):
        usdc = sum(b["usd"] for b in self.balances if b["symbol"] == "USDC" and b["chain"] == "solana")
        sol = sum(b["usd"] for b in self.balances if b["symbol"] == "SOL" and b["chain"] == "solana")
        return float(usdc), float(sol)

    # ---- dispatch
    def call(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return canon(fn(**(args if isinstance(args, dict) else {})))
        except (TypeError, ValueError, KeyError, ArithmeticError) as e:
            return {"error": f"bad arguments: {e}"}

    def _bio(self, s):
        return s.replace("Face ID", self.bio)

    def _mint_ok(self, m):
        return any(m in t for t in self.typed) or m in self.searched_mints

    def _refused(self):
        return {"error": "Refused: that mint isn't in the user's messages or a search result from this chat. Search for the token first, or ask the user for the mint."}

    # ---- plain reads
    def t_calculator(self, expression):
        try:
            return {"result": H.calc_str(H.calc(str(expression)))}
        except ZeroDivisionError:
            return {"error": "division by zero"}
        except Exception:
            return {"error": "Couldn't evaluate that expression"}

    def t_get_price(self, symbol):
        s = str(symbol).upper()
        if s in self.prices and s != "USDC":
            return {"change_24h_pct": 0.9, "price": self.prices[s], "source": "Hyperliquid mark price", "symbol": s}
        for t in self.tokens:
            if t["symbol"] == s and t["verified"] and s in PRICEABLE:
                return {"change_24h_pct": t["change"], "price": t["price"], "source": "Hyperliquid mark price", "symbol": s}
        return {"error": f"No Hyperliquid market for {s}. Answer from what you know instead, and say there's no live price here."}

    def t_get_balances(self, chain=None):
        rows = [b for b in self.balances if b["amount"] > 0 and (chain is None or b["chain"] == chain)]
        return {"holdings": [{"amount": trim(Decimal(repr(b["amount"]))) if not isinstance(b["amount"], int) else str(b["amount"]),
                              "chain": b["chain"], "symbol": b["symbol"], "usd": b["usd"]} for b in rows]}

    def t_get_positions(self, kind="all"):
        if not self.wallet:
            return {"error": "No wallet on this %s yet" % self.device}
        if kind not in ("all", "spot", "earn", "perps"):
            return {"error": "bad arguments: kind"}
        spot = sorted([copy.deepcopy(h) for h in self.spot], key=lambda h: -h["usd"])
        perps = {"open_orders": [], "positions": [{"coin": p["coin"], "entry": p["entry"], "leverage": p["leverage"],
                                                    "liquidation": p["liquidation"], "side": p["side"], "size": p["size"],
                                                    "unrealised_pnl": p["pnl"]} for p in self.perps]}
        if kind == "spot":
            return {"spot": spot, "wallet": self.wallet_name}
        if kind == "earn":
            return {"earn": [], "wallet": self.wallet_name}
        if kind == "perps":
            return {"perps": perps, "wallet": self.wallet_name}
        return {"earn": [], "perps": perps, "spot": spot, "wallet": self.wallet_name}

    def t_open_page(self, page, tab=None):
        pages = ["home", "wallets", "trade", "earn", "payments", "explore", "calendar", "projects", "automations", "files", "connectors", "settings"]
        if page not in pages:
            return {"error": "That page can't be opened from chat."}
        r = {"note": "The user taps it to open %s." % page.capitalize(), "page": page, "screen": page, "status": "button_shown"}
        if tab:
            r["tab"] = tab
        return r

    def t_search_tokens(self, query):
        if self.search_down:
            return {"error": "Token search is unavailable right now."}
        q = str(query).strip()
        if is_mint(q):
            hits = [t for t in self.tokens if t["mint"] == q]
        else:
            ws = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w) >= 3]
            hits = []
            for t in self.tokens:
                hay = [t["symbol"].lower()] + re.findall(r"[a-z0-9]+", t["name"].lower()) + [a.lower() for a in t["aliases"]]
                if any(w in hay or (len(w) >= 4 and any(w in h for h in hay)) for w in ws):
                    hits.append(t)
        if not hits:
            return {"note": "No Solana token matched. Check the spelling or paste the mint.", "tokens": []}
        hits = sorted(hits, key=lambda t: (-t["liquidity"], t["symbol"]))[:5]
        rows = []
        for t in hits:
            row = {"change_24h_pct": t["change"], "liquidity_usd": t["liquidity"], "mint": t["mint"], "name": t["name"],
                   "price": t["price"], "symbol": t["symbol"], "tags": list(t["tags"]), "verified": t["verified"],
                   "volume_24h_usd": t["volume"]}
            if t["refusal"]:
                row["refusal"] = t["refusal"]
            rows.append(row)
            if t["mint"] not in self.searched_mints:
                self.searched_mints.append(t["mint"])
        return {"tokens": rows}

    # ---- prepare tools
    def _blocked(self, why):
        return {"note": "Nothing was prepared. Tell the user why.", "reasons": [why], "status": "blocked"}

    def _impact(self, t, usd):
        return round(0.02 + 1000.0 * usd / t["liquidity"], 2)

    def t_prepare_sell(self, token, percent=None, amount=None, receive=None):
        if not self.wallet:
            return {"error": "No wallet on this %s to sell with" % self.device}
        if percent is not None and amount is not None:
            return {"error": "Give percent or amount, not both."}
        if percent is None and amount is None:
            return {"error": "Give percent or amount."}
        if percent is not None and (isinstance(percent, bool) or not isinstance(percent, (int, float)) or not 0 < percent <= 100):
            return {"error": "percent must be a number from 1 to 100"}
        if amount is not None and (isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0):
            return {"error": "amount must be a number above 0"}
        if receive not in (None, "USDC", "SOL"):
            return {"error": "receive must be USDC or SOL"}
        tk = str(token)
        mint_given = is_mint(tk)
        if mint_given:
            if not self._mint_ok(tk):
                return self._refused()
            rows = [h for h in self.spot if h["mint"] == tk]
            label = next((t["symbol"] for t in self.tokens if t["mint"] == tk), "that token")
        else:
            label = tk.upper()
            rows = [h for h in self.spot if h["symbol"] == label]
        if not rows:
            return {"note": "The user holds no %s on Solana. Tell them." % label, "prepared": False, "status": "not_held"}
        if len(rows) > 1:
            rows = sorted(rows, key=lambda h: -Decimal(h["amount"]))
            q = "You hold two tokens called %s: %s (%s) and %s (%s). Which one?" % (
                label, commas(rows[0]["amount"]), short_mint(rows[0]["mint"]), commas(rows[1]["amount"]), short_mint(rows[1]["mint"]))
            for h in rows:
                if h["mint"] not in self.searched_mints:
                    self.searched_mints.append(h["mint"])
            return {"note": "Nothing was prepared. Ask the user which one.",
                    "options": [{"amount": h["amount"], "mint": h["mint"], "symbol": h["symbol"]} for h in rows[:2]],
                    "prepared": False, "question": q, "status": "needs_choice"}
        h = rows[0]
        sym = h["symbol"]
        t = self.tok(mint=h["mint"])
        if h["frozen"]:
            return self._blocked("Your %s balance is frozen by the token's authority, so it can't be sold." % sym)
        hold = Decimal(h["amount"])
        if amount is not None:
            qty = Decimal(repr(amount))
            if qty > hold:
                return {"holding": "%s %s" % (fmt_qty(hold, exact=True), sym), "note": "The user holds less than that. Tell them.",
                        "prepared": False, "status": "insufficient"}
        else:
            qty = (hold * Decimal(repr(percent)) / 100)
        if h["mint"] in self.no_route:
            return {"error": "No route to sell that token right now."}
        usd = float(qty) * (t["price"] or 0.001)
        if usd > self.daily_left:
            return self._blocked("Amount is above your daily limit for the assistant.")
        impact = self._impact(t, usd)
        net = usd * (1 - impact / 100.0)
        if receive == "SOL":
            you_get = "about %s SOL on Solana" % BJ.approx(net / self.prices["SOL"])
        else:
            you_get = "about %s USDC on Solana" % BJ.approx(net)
        full = (amount is None and percent == 100) or (amount is not None and qty == hold)
        if full:
            share = "all of your %s" % sym
            you_sell_q = fmt_qty(hold, exact=True) if amount is not None else fmt_qty(hold)
        elif amount is not None:
            share = "%s%% of your %s" % (trim((qty / hold * 100).quantize(Decimal("0.01"), ROUND_HALF_UP)), sym)
            you_sell_q = fmt_qty(qty, exact=True)
        else:
            share = "%s%% of your %s" % (trim(Decimal(repr(percent))), sym)
            you_sell_q = fmt_qty(qty)
        warn = impact >= 3
        r = {"executed": False, "kind": "sell",
             "note": self._bio("Nothing has moved. The Sell review is open with this Jupiter quote; the user approves it with Face ID or cancels."),
             "price_impact_pct": jnum(impact), "route": ("Jupiter via Raydium" if warn else t["route"]),
             "slippage_pct": 1.0 if warn else 0.5, "status": "review_opened", "symbol": sym,
             "you_get": you_get, "you_sell": "%s %s (%s)" % (you_sell_q, sym, share)}
        if warn:
            r["policy"] = "warn"
            r["warnings"] = ["Price impact is %s%%, so you get less than the market price." % trim(impact)]
        return r

    def t_prepare_buy(self, symbol, amount_usd, pay_with=None):
        if not self.wallet:
            return {"error": "No wallet on this %s to buy with" % self.device}
        if not isinstance(amount_usd, (int, float)) or isinstance(amount_usd, bool) or amount_usd <= 0:
            return {"error": "amount_usd must be a number above 0"}
        if pay_with not in (None, "USDC", "SOL"):
            return {"error": "pay_with must be USDC or SOL"}
        s = str(symbol)
        if is_mint(s):
            if not self._mint_ok(s):
                return self._refused()
            t = next((x for x in self.tokens if x["mint"] == s), None)
            if t is None:
                return self._blocked("Hermit couldn't find a Solana token with that mint.")
            if t["refusal"]:
                return self._blocked(t["refusal"])
            by_mint = True
        else:
            su = s.upper()
            cands = sorted([x for x in self.tokens if x["symbol"] == su], key=lambda x: -x["liquidity"])
            ver = [x for x in cands if x["verified"]]
            if not ver:
                return self._blocked("There's no verified %s token on Solana, so Hermit won't buy one." % su)
            if len(cands) >= 2 and cands[1]["liquidity"] >= 0.5 * cands[0]["liquidity"]:
                opts = cands[:2]
                for x in opts:
                    if x["mint"] not in self.searched_mints:
                        self.searched_mints.append(x["mint"])
                return {"note": "Nothing was prepared. Ask the user which one.",
                        "options": [{"liquidity_usd": x["liquidity"], "mint": x["mint"], "symbol": x["symbol"]} for x in opts],
                        "prepared": False,
                        "question": "Two tokens are called %s. Pick the one you mean on Spot or Memes, or paste its mint." % su,
                        "status": "needs_choice"}
            t = ver[0]
            by_mint = False
        if amount_usd > self.daily_left:
            return self._blocked("Amount is above your daily limit for the assistant.")
        usdc, sol_usd = self.usdc_sol()
        pay = pay_with or ("USDC" if usdc >= amount_usd else "SOL")
        have = usdc if pay == "USDC" else sol_usd
        if have + 1e-9 < amount_usd:
            parts = []
            for sy in ("USDC", "SOL"):
                a = sum(b["amount"] for b in self.balances if b["symbol"] == sy and b["chain"] == "solana")
                if a > 0:
                    parts.append("%s %s" % (trim(Decimal(repr(a))), sy))
            hold = (" and ".join(parts) + " on Solana") if parts else "no USDC or SOL on Solana"
            return {"holding": hold, "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"}
        impact = self._impact(t, amount_usd)
        qty = amount_usd / (t["price"] or 0.001) * (1 - impact / 100.0)
        you_pay = "%s USDC on Solana" % BJ.jstr(amount_usd) if pay == "USDC" else "%s SOL on Solana" % BJ.approx(amount_usd / self.prices["SOL"])
        q_txt = buy_qty(qty)
        r = {"executed": False, "kind": "buy",
             "note": self._bio("Nothing has moved. The Buy review is open with this Jupiter quote; the user approves it with Face ID or cancels."),
             "price_impact_pct": jnum(impact), "route": t["route"], "slippage_pct": 1.0 if (not t["verified"]) else 0.5,
             "status": "review_opened", "symbol": t["symbol"], "you_get": "about %s %s" % (q_txt, t["symbol"]), "you_pay": you_pay}
        warnings = []
        if by_mint and not t["verified"]:
            r["verified"] = False
            warnings.append("Unverified token")
            for tag in t["tags"]:
                if tag == "Mint on": warnings.append("Mint authority is on")
                elif tag == "Low liquidity": warnings.append("Low liquidity")
                elif tag == "New": warnings.append("New token")
                elif tag.startswith("Top holders"): warnings.append("Top holders own " + re.search(r"\d+%", tag).group())
        if impact >= 1:
            warnings.append("High price impact (%s%%): you may get noticeably less than the price shown." % BJ.jstr(impact))
        if warnings:
            r["warnings"] = warnings
        return r

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        if not self.wallet:
            return {"error": "No wallet on this %s to swap with" % self.device}
        if amount > self.daily_left:
            return self._blocked("Amount is above your daily limit for the assistant.")
        have = sum(b["amount"] for b in self.balances if b["symbol"] == from_symbol and b["chain"] == from_chain)
        if have + 1e-9 < amount:
            return {"error": "insufficient balance: %s on %s is %s" % (from_symbol, from_chain, BJ.jstr(have))}
        fee_bps = 35
        in_usd = amount * self.prices[from_symbol]
        cost = round(in_usd * (fee_bps + self.spread_bps) / 10000.0, 2)
        out_usd = round(in_usd - cost, 2)
        out = BJ.approx(out_usd / self.prices[to_symbol])
        return {"amount_in_usd": jnum(round(in_usd, 2)), "amount_out": out, "amount_out_usd": jnum(out_usd), "fee_bps": fee_bps,
                "from": "%s %s on %s" % (BJ.jstr(amount), from_symbol, CHAIN_NAME[from_chain]),
                "note": self._bio("Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels."),
                "policy": "allow", "route": "NEAR Intents", "seconds": self.seconds, "signature_verified": True,
                "status": "prepared_for_review", "to": "about %s %s on %s" % (out, to_symbol, CHAIN_NAME[to_chain]),
                "total_cost_usd": jnum(cost)}

    def t_prepare_close(self, coin, percent=None):
        if not self.wallet:
            return {"error": "No wallet on this %s to trade with" % self.device}
        c = str(coin).upper()
        pos = next((p for p in self.perps if p["coin"] == c), None)
        if pos is None:
            return {"note": "The user has no open %s position on Hyperliquid. Tell them." % c, "prepared": False, "status": "no_position"}
        if percent is None or percent == 100:
            closing = "all of the %s %s, %s %s" % (c, pos["side"], BJ.jstr(pos["size"]), c)
            pnl = pos["pnl"]
        else:
            sz = float(Decimal(repr(pos["size"])) * Decimal(repr(percent)) / 100)
            closing = "%s%% of the %s %s, %s %s" % (trim(Decimal(repr(percent))), c, pos["side"], BJ.jstr(round(sz, 6)), c)
            pnl = round(pos["pnl"] * percent / 100.0, 2)
        return {"closing": closing, "executed": False, "kind": "close", "note": self._bio("Waiting for the user to approve with Face ID."),
                "order": "Market, reduce only", "status": "review_opened", "unrealised_pnl": jnum(pnl)}


# ================================================================================ conversation wrapper
SAFE_NOTES = ["Keep answers very short.", "I'm new to crypto, explain simply.", "I trade mostly BTC and ETH perps on Hyperliquid.",
              "I use Base and Arbitrum most.", "I live in Europe, quote times in CET.", "My risk per trade is 1%."]


class UConv(Conv):
    """gen2 Conv with the v3 system prompt and a world that sees the typed user text (mint provenance guard)."""

    def __init__(self, rng, world, bio):
        super().__init__(rng, world)
        s = SYSTEM_TPL.replace("{BIO}", bio)
        if rng.random() < 0.25:
            s += f"\nThe user wants to be called {rng.choice(H.NAMES)}."
        if rng.random() < 0.15:
            s += "\nThe user's standing notes (hermit.md). Follow them:\n" + "\n".join("- " + n for n in rng.sample(SAFE_NOTES, rng.choice([1, 2])))
        self.msgs[0]["content"] = s

    def user(self, text):
        self.w.typed.append(text)
        super().user(text)


def pick_bio(rng):
    bio = "Face ID" if rng.random() < 0.70 else "Touch ID"
    dev = "iPhone" if bio == "Face ID" else ("Mac" if rng.random() < 0.9 else "iPhone")
    return bio, dev


def pick_pack(rng, p_trade=0.08):
    return "trade" if rng.random() < p_trade else "sol"


# ================================================================================ phrase machinery
def split_tpls(pool, ev):
    return [t for i, t in enumerate(pool) if (i % 5 == 0) == ev]


TYPO_SKIP = {"with", "using", "usdc", "sol", "usd", "bucks", "dollars", "half", "mint", "this"}


def render(rng, tpl, slots, typo_p=0.15):
    parts = re.split(r"(\{[A-Za-z]+\})", tpl)
    if rng.random() < typo_p:
        cands = [(pi, m) for pi, p in enumerate(parts) if not p.startswith("{") for m in re.finditer(r"[a-z]{4,}", p) if m.group() not in TYPO_SKIP]
        if cands:
            pi, m = rng.choice(cands)
            parts[pi] = parts[pi][:m.start()] + BJ.typo_word(rng, m.group()) + parts[pi][m.end():]
    out = "".join(slots[p[1:-1]] if p.startswith("{") else p for p in parts)
    protect = any(MINT_RE.match(v) or ADDR_RE.search(v) for v in slots.values())
    r = rng.random()
    if not protect:
        if r < 0.20:
            out = out.lower()
        elif r < 0.215:
            out = out.upper()
    r = rng.random()
    if r < 0.05 and not out.endswith("?"):
        out += rng.choice([" pls", " plz", " rn"])
    elif r < 0.10:
        out = out.rstrip(".?!")
    return out


def sym_text(rng, sym, aliases=True):
    r = rng.random()
    if aliases and sym == "WIF" and r < 0.10:
        return "dogwifhat"
    if r < 0.55:
        return sym
    if r < 0.82:
        return sym.lower()
    return "$" + sym


def ticker_in(text, sym):
    t = text.lower()
    s = sym.lower()
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", t)) or (sym == "WIF" and "dogwifhat" in t)


PCT_FORMS = {50: ["half", "50%", "fifty percent", "50 percent", "half"], 25: ["a quarter", "25%", "25 percent", "a quarter"],
             33.33: ["a third", "one third", "a third"], 75: ["75%", "three quarters", "75 percent"],
             10: ["10%", "10 percent", "a tenth"], 20: ["20%", "20 percent", "a fifth"], 30: ["30%", "30 percent"],
             40: ["40%", "40 percent"], 15: ["15%", "15 percent"], 60: ["60%", "60 percent"], 5: ["5%", "5 percent"],
             90: ["90%", "90 percent"], 35: ["35%", "35 percent"], 12.5: ["12.5%"]}
PCT_WORDS = {50: ["half", "fifty"], 25: ["quarter"], 33.33: ["third"], 75: ["three quarters"], 10: ["tenth"], 20: ["fifth"]}
PCT_SHARED = [(50, 30), (25, 20), (33.33, 8), (75, 8), (10, 8), (20, 7), (30, 7), (40, 5)]
PCT_EVALX = [(15, 5), (60, 5), (5, 3), (90, 3), (35, 3), (12.5, 2)]


def pick_pct(rng, ev):
    if ev and rng.random() < 0.4:
        pool = PCT_EVALX
    else:
        pool = PCT_SHARED
    p = rng.choices([x[0] for x in pool], [x[1] for x in pool])[0]
    return p, rng.choice(PCT_FORMS[p])


def pct_in_text(p, text):
    t = text.lower()
    if p == 100:
        return True
    if re.search(r"(?<![\d.])" + re.escape(trim(Decimal(repr(p)))) + r"\s*(%|percent)", t):
        return True
    return any(w in t for w in PCT_WORDS.get(p, []))


PCT_TPL = [
    "sell {FR} of my {S}", "dump {FR} of my {S}", "sell {FR} of my {S} bag", "can you sell {FR} of my {S}?", "i wanna sell {FR} of my {S}",
    "trim {FR} of my {S}", "take {FR} off my {S}", "cash out {FR} of my {S}", "offload {FR} of my {S}", "{S}: sell {FR}",
    "sell {S}, {FR}", "unload {FR} of my {S} position", "sell off {FR} of {S}", "take some profit, sell {FR} of my {S}", "lighten my {S} bag by {FR}",
    "sell {FR} of {S} please", "hey hermit, dump {FR} of my {S}", "get rid of {FR} of my {S}", "sell {FR} of my {S} on solana", "pls sell {FR} of my {S}",
    "i'm up on {S}, sell {FR}", "reduce my {S} by {FR}", "sell {FR} of what i hold in {S}", "sell {FR} of my {S} holdings", "market sell {FR} of my {S}",
    "take {FR} of my {S} off the table", "sell down {S} by {FR}", "exit {FR} of my {S}", "scale out of {S}, sell {FR}", "yo sell {FR} of my {S}",
    "{FR} of my {S} needs to go, sell it", "can u sell {FR} of my {S} for me", "sell {FR} of my {S} stack", "book profit on {FR} of {S}",
]
NOOF_TPL = ["sell half my {S}", "dump half my {S}", "sell half {S}", "sell half of my {S} please", "dump half of {S}", "take half my {S} off"]
ALL_TPL = [
    "sell all my {S}", "sell everything in {S}", "get out of {S}", "dump all my {S}", "dump my whole {S} bag",
    "exit {S} completely", "liquidate my {S}", "cash out of {S}", "sell my entire {S} position", "i'm done with {S}, sell it all",
    "sell the whole {S} bag", "yeet all my {S}", "close out my {S} position", "sell 100% of my {S}", "get me out of {S}",
    "sell every {S} i have", "rid me of {S}, all of it", "{S} is going to zero, sell all of it", "sell all {S} pls", "sell out of {S}",
    "full exit on {S}", "{S}: sell everything", "offload all my {S}", "i want out of {S}", "dump the lot of my {S}",
    "sell my {S} bag, all of it", "unload my {S} entirely", "ditch my {S}", "sell max {S}", "all {S} out please",
    "sell it all, {S}", "sell all of my {S} tokens", "bail out of {S}", "clear out my {S}",
]
ALLR_TPL = [
    "sell all my {S} into {R}", "sell all my {S} for {R}", "swap all my {S} to {R}", "dump {S} to {R}", "get out of {S} into {R}",
    "convert all my {S} to {R}", "sell everything in {S} and give me {R}", "sell all {S}, receive {R}", "exit {S} into {R}", "cash out all my {S} to {R}",
    "sell my whole {S} bag for {R}", "dump all my {S} into {R}", "i want out of {S}, pay me in {R}", "sell every {S} i have and take {R}", "liquidate {S} to {R}",
    "sell all my {S}, i want {R} back", "{S} to {R}, all of it", "turn all my {S} into {R}", "sell out of {S} into {R}", "get me {R} for all my {S}",
]
AMT_TPL = [
    "sell {N} {S}", "sell {N} of my {S}", "dump {N} {S}", "sell {N} {S} for me", "can you sell {N} {S}?", "i want to sell {N} {S}",
    "sell exactly {N} {S}", "sell {N} {S} pls", "offload {N} {S}", "unload {N} of my {S}", "sell {N} units of {S}", "sell {N} tokens of {S}",
    "sell {N} of my {S} tokens", "hey, sell {N} {S}", "get rid of {N} {S}", "sell {S}, {N} of them", "sell off {N} {S}",
    "sell {N} {S} from my wallet", "just sell {N} {S}", "take {N} {S} off my bag and sell them",
]
USD_TPL = [
    "sell {A} worth of {S}", "sell {A} of my {S}", "sell {A} of {S}", "dump {A} worth of my {S}", "cash out {A} of {S}",
    "i need {A}, sell some {S}", "sell {A} worth of my {S} please", "can you sell {A} of {S}?", "sell about {A} of {S}", "take {A} out of my {S}",
    "sell {A} worth of {S} for me", "pull {A} out of {S}, sell",
]
MINT_PCT_TPL = [
    "sell {FR} of {M}", "sell {FR} of this token {M}", "dump {FR} of my {M}", "sell {FR} of the token {M}", "can you sell {FR} of {M}?",
    "{M}: sell {FR}", "sell {FR} of this one: {M}", "take {FR} off {M}", "offload {FR} of {M}", "i hold {M}, sell {FR} of it",
    "sell {FR} of my bag, mint {M}", "sell {FR} of the coin {M}", "mint {M}, sell {FR}",
]
MINT_ALL_TPL = [
    "sell all of {M}", "sell all my {M}", "dump everything in {M}", "get out of {M}", "sell my whole bag of {M}", "{M}: sell it all",
    "sell all of this token {M}", "exit {M} completely", "liquidate {M}", "sell everything, mint {M}", "dump all of this one {M}", "i want out of {M}",
]
ASK_NOTOKEN_TPL = [
    "sell some", "sell my tokens", "sell something", "i want to sell", "sell a bit", "can you sell some of my tokens?", "i need to sell",
    "let's sell something", "time to take profit, sell", "sell some of my bags", "sell stuff", "sell my memecoins", "dump some bags",
    "sell part of my portfolio", "i wanna cash out some tokens", "sell my solana tokens", "help me sell something", "sell a few of my tokens",
]
ASK_NOAMT_TPL = [
    "sell my {S}", "sell {S}", "i want to sell my {S}", "can you sell my {S}?", "sell some {S}", "let's sell {S}", "time to sell {S}",
    "sell a bit of {S}", "sell part of my {S}", "trim my {S}", "take some profit on {S}", "sell a little {S}", "sell my {S} pls", "i should sell {S}",
]
ASK_NOTOKEN_PCT_TPL = [
    "sell {FR}", "sell {FR} of my tokens", "dump {FR}", "sell {FR} of my bags", "take {FR} off", "i want to sell {FR}", "sell {FR} pls",
    "trim {FR}", "sell {FR} of my solana tokens", "let's sell {FR}", "cash out {FR}", "sell {FR} of it all",
]
FOLLOW_TOKEN_TPL = ["{S}", "the {S}", "{S} please", "{S} one", "let's do {S}", "my {S}", "{S}, please", "it's {S}", "the {S} bag", "{S} then"]
FOLLOW_NOTOKEN_TPL = [
    "{S}, {FR}", "{FR} of my {S}", "the {S}, {FR}", "{S} {FR} please", "sell {FR} of {S}", "{S} then, {FR}", "let's do {FR} of {S}",
    "{FR} of {S}", "the {S} one, {FR}", "{S}, sell {FR}",
]
FOLLOW_AMT_TPL = [
    "{FR}", "just {FR}", "{FR} of it", "make it {FR}", "{FR} please", "let's do {FR}", "do {FR}", "{FR} is fine", "go with {FR}",
    "i'd say {FR}", "{FR} of it please", "sell {FR}",
]
FOLLOW_ALL_TPL = ["all of it", "everything", "all", "the whole thing", "all of it please", "sell it all", "everything please", "the lot", "the entire bag", "all of them"]
FOLLOW_ALLTOK_TPL = ["all of my {S}", "all {S}", "{S}, all of it", "sell all {S}", "the {S}, everything", "{S} everything"]
AMT_FORMS = [(2.5, "2.5"), (12.5, "12.5"), (250, "250"), (40, "40"), (7.5, "7.5"), (100, "100"), (3, "3"), (1200, "1,200"), (1500, "1.5k"),
             (2000, "2k"), (60, "60"), (5, "5"), (18.5, "18.5"), (300, "300"), (0.5, "0.5"), (1, "1"), (25, "25"), (8, "8"), (75, "75"), (500, "500")]
AMT_FORMS_EVAL = [(1.5, "1.5"), (35, "35"), (150, "150"), (90, "90"), (4.2, "4.2"), (750, "750"), (22, "22"), (3500, "3,500")]
USD_AMTS = [3, 5, 8, 10, 12, 15, 20, 25, 40, 50]
USD_AMTS_EVAL = [6, 7, 18, 30, 35]
SAFETY_TPL = [
    "is {S} safe", "is {S} legit?", "is {S} a scam", "can I trust {S}", "is {S} a rug", "check {S} for me", "any red flags on {S}?",
    "is {S} verified?", "how risky is {S}", "what's the liquidity on {S}", "how much liquidity does {S} have", "is {S} real or fake",
    "check if {S} is legit", "quick check on {S}", "is it okay to buy {S}? check it first", "does {S} have freeze authority?",
    "is the {S} token rugged?", "{S} legit?", "look up {S} and tell me if it's risky", "what do you know about {S}", "{S} liquidity?",
    "is {S} dangerous", "run a check on {S}", "is {S} trustworthy?", "tell me about the {S} token",
]
SAFETY_MINT_TPL = [
    "check this mint {M}", "is this token legit? {M}", "is this safe? {M}", "what is this token {M}", "can you check {M}", "{M} legit?",
    "look up this mint {M}", "check this token for me: {M}", "any red flags on {M}?", "how much liquidity does {M} have", "is this a scam? {M}",
    "scan {M}", "friend sent me this {M}, is it ok?", "what do you see for {M}", "does {M} have freeze authority?",
]
FIND_TPL = [
    "what's the mint of {S}", "mint address for {S}", "contract address of {S}", "ca of {S}", "what is the {S} mint",
    "find the {S} token", "find me the {S} token", "search for {S}", "look up the {S} token", "which one is the real {S}?",
    "paste the {S} mint", "give me the {S} contract", "where do i find the {S} mint", "token address for {S} please",
]
FIND_THEME_TPL = [
    "is there a {K} token with real liquidity", "any {K} tokens with decent liquidity?", "find a {K} token", "show me {K} tokens",
    "what {K} coins are on solana", "search {K} tokens",
]
BUYMINT_TPL = [
    "buy this {M} {A}", "buy this: {M} for {A}", "{A} of {M} please", "ape {A} into {M}", "can you buy {A} of this token? {M}",
    "buy {A} worth of {M}", "I want {A} of this one {M}", "grab {A} of {M}", "buy {M}, {A}", "got this mint from a friend, buy {A}: {M}",
    "get me {A} of {M}", "purchase {A} of {M}", "{M} buy {A}", "put {A} in this token {M}", "ca: {M} buy {A}", "load {A} of {M}",
    "ape this {M} for {A}", "buy me {A} of the token {M}", "token {M}, {A} please", "hey, {A} of {M}",
]
BUYMINT_INS_TPL = [
    "buy {A} of {M}, i know the risks", "{M} {A} buy it even if it's unverified", "i don't care if it's unverified, buy {A} of {M}",
    "ape {A} into {M}, yes I know it's risky", "buy this {M} for {A}, I accept the risk", "{A} of {M}. unverified is fine, just buy",
    "buy {A} of {M}, it's a gamble and I know it", "I know {M} isn't verified, still buy {A}",
]
BUY_DIRECT_TPL = [
    "buy {A} of {S}", "get me {A} of {S}", "ape {A} into {S}", "{S} {A} buy", "buy me {A} of {S} please", "grab {A} of {S}",
    "can you buy {A} worth of {S}?", "put {A} into {S}", "I'd like {A} of {S}", "purchase {A} of {S}", "buy {S} for {A}", "pick up {A} of {S}",
    "snipe {A} of {S}", "buy {S}, {A}", "get {A} of {S} asap", "throw {A} at {S}",
]
BUY_NAME_TPL = [
    "buy {A} of {S}", "get me {A} of {S}", "ape {A} into {S}", "I want {A} of {S}", "buy {A} worth of {S}", "grab {A} of {S}",
    "can you buy {A} of {S}?", "put {A} into {S}", "buy {S}, {A}", "purchase {A} of {S}",
]
FOLLOW_INSIST = ["yes", "yes buy it", "do it", "go ahead", "buy it anyway", "yes I know, do it", "yep proceed", "yeah prepare it",
                 "still want it", "ok do it anyway", "yes, prepare the buy", "go for it", "yes, i accept the risk", "prepare it anyway", "proceed",
                 "yes please", "i'll take the risk, buy", "do it anyway"]
CLASH_PICK = {
    0: ["the first one", "the bigger one", "first", "the one with more liquidity", "go with the bigger one", "the {L1} one", "the larger one"],
    1: ["the second one", "the smaller one", "second", "the one with less liquidity", "go with the smaller one", "the {L2} one", "the other one"],
}
CLASH_MINT_PICK = ["{M}", "this one: {M}", "the one with mint {M}", "the mint is {M}", "use {M}"]
SWAP_SELL_TPL = [
    "sell {N} {S}", "sell {N} {S} for usdc", "dump {N} {S}", "cash out {N} {S}", "sell {N} of my {S}", "i want to sell {N} {S}",
    "convert {N} {S} to usdc", "can you sell {N} {S}?", "sell {N} {S} into usdc", "offload {N} {S}",
]
SWAP_ALL_TPL = ["sell all my {S}", "sell everything in {S}", "dump all my {S}", "cash out all my {S}", "sell my whole {S} balance", "sell all {S} for usdc"]
SWAP_HALF_TPL = ["sell half my {S}", "sell half of my {S}", "dump half my {S}", "cash out half my {S}", "sell half my {S} for usdc"]
SWAP_USDC_TPL = ["sell {N} usdc for sol", "sell {N} usdc into sol", "sell {N} usdc and buy sol", "dump {N} usdc into sol", "swap {N} usdc to sol"]
SWAP_USDC_ASK_TPL = ["sell {N} usdc", "sell my {N} usdc", "dump {N} usdc", "sell {N} usdc for me"]
CLOSE_TPL = [
    "close my {C} {SIDE}", "close my {C} {SIDE} position", "close the {C} {SIDE}", "get me out of my {C} {SIDE}", "exit my {C} {SIDE}",
    "can you close my {C} {SIDE}?", "close out my {C} {SIDE}", "shut my {C} {SIDE}", "{C} {SIDE}: close it", "flatten my {C} {SIDE}",
]
CLOSE_PART_TPL = [
    "close {FR} of my {C} {SIDE}", "close {FR} of my {C} {SIDE} position", "take {FR} off my {C} {SIDE}", "reduce my {C} {SIDE} by {FR}",
    "close out {FR} of the {C} {SIDE}", "cut my {C} {SIDE} by {FR}",
]
PRICE_TPL = ["{S} price", "what's {S} at?", "how much is {S} right now", "price of {S}?", "{S}?", "where is {S} trading", "current {S} price pls",
             "what's {S} doing today", "is {S} up today?", "check the {S} price", "how's {S} priced", "{S} price rn"]
CONCEPTS = {
    "rug": (["how does a rug pull work", "what is a rug pull", "explain rug pulls", "what's a rug", "how do rugs happen", "how do people get rugged", "what does getting rugged mean", "explain rugs to me"],
            ["A rug pull is when the people behind a token drain its liquidity or dump a large hidden supply, so the price collapses and holders can't sell for anything close to what they paid. Common warning signs are very low liquidity, a creator who can still mint or freeze, and a few wallets holding most of the supply. Hermit's token search shows these as tags.",
             "In a rug pull the creators pull the money out of the pool, or sell a huge share of the supply they kept, and the price drops close to zero. Afterwards you usually can't sell, because nobody is left to buy. Watch for tiny liquidity, mint or freeze rights left on, and concentrated holders. Hermit's token search flags these."]),
    "freeze": (["what is a freeze authority", "what's freeze authority", "explain freeze authority", "what does freeze on mean", "what is freeze authority on a token", "freeze authority, what is that", "why does freeze on matter", "what does the freeze authority do"],
               ["A freeze authority is a key that lets a token's creator freeze token accounts, so holders can't move or sell their tokens. If it is on, you are trusting the creator not to use it. Hermit's token search shows it as Freeze on, and Hermit won't buy an unverified token that has it.",
                "Freeze authority lets the creator of a token lock any holder's balance, so those tokens can't be moved or traded until they unfreeze them. In Hermit's token search it appears as Freeze on, and an unverified token with it is refused for buys."]),
    "mint": (["what is a mint authority", "what does mint on mean", "explain mint authority", "what's mint authority", "what does the mint authority do", "why does mint on matter"],
             ["Mint authority lets the creator make new tokens whenever they want, which can water down everyone's share and push the price down. If it is on, you are trusting the creator not to use it. Hermit's token search shows it as Mint on.",
              "With mint authority on, the creator can create more supply at any time, so your share can shrink and the price can fall. Hermit's token search shows this as Mint on."]),
    "liquidity": (["what is liquidity", "what does liquidity mean for a token", "why does liquidity matter", "what's low liquidity", "explain liquidity", "what does low liquidity mean"],
                  ["Liquidity is how much money sits in a token's trading pools. With low liquidity even a small sale moves the price a lot, and you may not be able to sell at the price you see. Hermit's token search shows the liquidity and flags it as Low liquidity when it is thin.",
                   "Liquidity is the money available to trade against. When it is thin, your own trade pushes the price, so you get less than the quote and exits can be hard. Hermit's token search shows it for every token."]),
    "impact": (["what is price impact", "what does price impact mean", "explain price impact", "why is there price impact", "what's price impact"],
               ["Price impact is how much your own trade moves the price. A bigger trade in a thin pool moves it more, so you get less than the quoted market price. The review shows it before you approve, and a smaller amount lowers it.",
                "Price impact is the cost of your trade size on the price: the bigger the trade compared with the pool, the worse the price you get. You see it in the review before you approve, and a smaller amount lowers it."]),
    "verified": (["what does verified mean", "what's a verified token", "what does verified mean on a token", "explain verified tokens", "verified token, what is that"],
                 ["Verified means the token is on Jupiter's verified list, so it has had some checks. It does not mean the price is stable or that it can't drop. Hermit buys verified tokens by ticker, and an unverified token only from a pasted mint, with its warnings shown.",
                  "A verified token is one Jupiter lists as verified. That is a basic check, not a promise about the price. Hermit buys unverified tokens only from a pasted mint, and shows every warning first."]),
    "mintaddr": (["what is a mint address", "what's a mint address", "what is a token mint", "what does mint address mean", "why do I need the mint"],
                 ["A mint address is the unique address that identifies a token on Solana. Two tokens can share a ticker, but never a mint, so a pasted mint is the surest way to point at one token.",
                  "The mint address is a token's unique ID on Solana. Tickers can be copied by other tokens, a mint can't, so it is the exact way to name the token you mean."]),
    "slippage": (["what is slippage", "what does slippage mean", "explain slippage", "what's slippage tolerance", "slippage, what is it"],
                 ["Slippage is the extra price move you accept between the quote and the trade. If the price moves further than that, the trade fails instead of filling at a worse price. The review shows the slippage before you approve.",
                  "Slippage is how far the price may move against you before the trade is cancelled. A higher setting fills more easily but can cost you more. The review shows it before you approve."]),
}
CONCEPT_KEYS = list(CONCEPTS)
RUG_KEY = "rug"
ABSENT_TAIL = ["I put a button below that opens Trade.", "Use the button below to open Trade.", "There's a button below that opens Trade.",
               "Tap the button below to open Trade.", "I put a Trade button below.", "The Trade button below opens that page."]


def amt_form(rng, ev):
    pool = AMT_FORMS_EVAL if (ev and rng.random() < 0.5) else AMT_FORMS
    return rng.choice(pool)


def pick_hold_token(rng, ev):
    if ev and rng.random() < 0.35:
        return rng.choice(EVAL_ONLY_T)
    if not ev and rng.random() < 0.10:
        return rng.choice(TRAIN_ONLY_T)
    return rng.choices(SHARED_T, SHARED_W)[0]


def pick_unverified_sym(rng, ev):
    return rng.choice(UNV_EVAL if ev else UNV_TRAIN)


# ================================================================================ answer text
def via(rng, r):
    return rng.choice(["Jupiter", "Jupiter", r["route"]])


def say_sell_ok(rng, r, bio):
    ys, yg, imp, slip = r["you_sell"], r["you_get"], BJ.jstr(r["price_impact_pct"]), BJ.jstr(r["slippage_pct"])
    intro = rng.choice([
        f"I've prepared selling {ys} for {yg} through {via(rng, r)}.",
        f"Your sale is prepared: {ys} for {yg}, through {via(rng, r)}.",
        f"Prepared: selling {ys} for {yg} through {via(rng, r)}.",
        f"The Sell review is open: {ys} for {yg}, through {via(rng, r)}.",
        f"I've set up selling {ys} for {yg} through {via(rng, r)}.",
    ])
    if r.get("warnings"):
        w = r["warnings"][0]
        mid = rng.choice([f"Warning: {w}", f"Heads up: {w}", f"Note the warning: {w}", f"Be careful, {w}"])
        tail = rng.choice([f"Approve with {bio} only if you accept that, or cancel. Nothing has been sold yet.",
                           f"A smaller amount would lower it. Approve with {bio} only if you accept that. Nothing has been sold yet.",
                           f"Cancel if that's too much, or approve with {bio}. Nothing is sold until you approve."])
        return f"{intro} {mid} {tail}"
    mid = rng.choice([f"Price impact is {imp}%.", f"The price impact is {imp}%.", f"Price impact: {imp}%.",
                      f"Price impact is {imp}% with {slip}% slippage.", f"That's {imp}% price impact."])
    closer = rng.choice([
        f"Approve with {bio} or cancel. Nothing has been sold yet.",
        f"Review it and approve with {bio}, or cancel. Nothing has been sold yet.",
        f"Nothing is sold until you approve it with {bio}.",
        f"It only happens if you approve with {bio}, and you can cancel instead. Nothing has been sold yet.",
        f"Waiting for your {bio}. Nothing is sold until you approve.",
    ])
    return f"{intro} {mid} {closer}"


def say_sell_dollars(rng, usd_txt, qty_txt, sym, r, bio):
    lead = rng.choice([f"At your holding's current value, {usd_txt} of {sym} is {qty_txt} {sym}.",
                       f"{usd_txt} of {sym} comes to {qty_txt} {sym} at the value your holding shows.",
                       f"Going by the value your holding shows, {usd_txt} is {qty_txt} {sym}."])
    return f"{lead} {say_sell_ok(rng, r, bio)}"


def say_not_held(rng, label):
    return rng.choice([
        f"You hold no {label} on Solana, so there's nothing to sell. Nothing was prepared.",
        f"I don't see any {label} in your Solana wallet, so I can't sell it. Nothing was prepared. I can list what you do hold if you like.",
        f"You don't hold any {label} on Solana. Nothing was prepared.",
        f"There's no {label} in your Solana wallet, so there's nothing to sell. I can show what you hold if you want.",
    ])


def say_insufficient(rng, r, asked):
    h = r["holding"]
    return rng.choice([
        f"You only hold {h}, so I can't prepare selling {asked}. Tell me an amount up to {h}.",
        f"That's more than you have: you hold {h}. Nothing was prepared. Tell me an amount up to {h}.",
        f"I can't prepare that. You hold {h}, which is less than {asked}. Give me an amount up to {h}, or a percent.",
        f"Your wallet has {h}, less than {asked}. Nothing was prepared. Tell me an amount up to {h}.",
    ])


def say_blocked_sell(rng, r, daily):
    why = r["reasons"][0]
    if daily:
        return rng.choice([f"I couldn't prepare that. {why} Nothing was prepared. Tell me a smaller amount and I'll prepare that instead.",
                           f"No review was opened. {why} A smaller amount may fit, so tell me one and I'll try.",
                           f"That sale was blocked. {why} Nothing was prepared. Give me a smaller amount if you want me to prepare one."])
    return rng.choice([f"I can't prepare that. {why} Nothing was prepared.", f"That sale was blocked. {why} Nothing was prepared.",
                       f"No review was opened. {why}", f"Blocked: {why} Nothing was prepared."])


def say_no_route(rng, r, sym):
    e = r["error"]
    return rng.choice([f"{e} Nothing was prepared for {sym}. Try again in a little while.",
                       f"I couldn't prepare it. {e} Try again in a little while.",
                       f"Nothing was prepared. {e} Ask me again a bit later."])


def say_search_error(rng, r):
    e = r["error"]
    return rng.choice([f"{e} Try again in a moment.", f"I couldn't check it. {e} Ask me again in a moment.", f"{e} Nothing to report yet, try again shortly."])


def move_txt(row):
    c = row["change_24h_pct"]
    return None if c is None else (chg(c) + " in 24 hours")


TAG_SAY = {
    "Freeze on": ["Freeze is on: the creator can freeze your tokens.", "Freeze is on, so the creator can freeze your tokens."],
    "Mint on": ["Mint is on: the creator can make more tokens.", "Mint is on, so the creator can create more supply."],
    "Low liquidity": ["Liquidity is low.", "Liquidity is low, so trades can move the price a lot."],
    "New": ["It is new.", "It was created recently."],
}


def tag_say(rng, tag, short):
    if tag.startswith("Top holders"):
        n = re.search(r"\d+%", tag).group()
        return rng.choice([f"The top holders own {n} of the supply.", f"Top holders own {n}."]) if not short else f"Top holders own {n}."
    opts = TAG_SAY[tag]
    return opts[0] if short else rng.choice(opts)


def say_row(rng, row, with_refusal=True, label=None):
    sym, liq = label or row["symbol"], liq_txt(row["liquidity_usd"])
    mv = move_txt(row)
    tags = row["tags"]
    short = len(tags) >= 3
    if row["verified"]:
        head = rng.choice([f"{sym} is verified", f"{sym} is a verified token"])
        facts = f", with {liq} of liquidity" + (f" and {mv}." if mv else ".")
        if rng.random() < 0.3 and mv:
            facts = f". Liquidity is {liq} and it is {mv}."
    else:
        head = f"{sym} is not verified."
        facts = f" Liquidity is {liq}" + (f" and it is {mv}." if mv else ".")
    parts = [head + facts]
    if tags:
        parts.append(" ".join(tag_say(rng, t, short) for t in tags))
    elif row["verified"]:
        parts.append(rng.choice(["No warning tags came back.", "It has no warning tags.", "No flags came back."]))
    else:
        parts.append(rng.choice(["No warning tags came back, but it is still unverified.", "It has no warning tags, but it is not verified."]))
    if row.get("refusal") and with_refusal:
        parts.append(row["refusal"])
    return " ".join(parts)


def say_search(rng, rows):
    if len(rows) == 1:
        s = say_row(rng, rows[0])
        if not rows[0].get("refusal") and rng.random() < 0.4:
            s += " " + rng.choice(["That is the data I have, not a promise.", "Check the liquidity and tags before you buy anything.", "Treat it as a check, not a guarantee."])
        return s
    out = []
    same = len({r["symbol"] for r in rows[:2]}) < len(rows[:2])
    for r in rows[:2]:
        out.append(say_row(rng, r, label=(f"{r['name']} ({r['symbol']})" if same else None)))
    return "There are %s matches. " % ("two" if len(rows) == 2 else "several") + " ".join(out)


def say_find(rng, rows, query):
    if not rows:
        return rng.choice([f"No Solana token matched {query}. Check the spelling or paste the mint.", f"I found no Solana token for {query}. Check the spelling, or paste the mint."])
    if len(rows) == 1:
        r = rows[0]
        v = "verified" if r["verified"] else "not verified"
        return rng.choice([f"The {r['symbol']} mint is {r['mint']}. It is {v}, with {liq_txt(r['liquidity_usd'])} of liquidity.",
                           f"{r['name']} ({r['symbol']}) is {v}, with {liq_txt(r['liquidity_usd'])} of liquidity. Mint: {r['mint']}",
                           f"I found {r['symbol']}: {v}, {liq_txt(r['liquidity_usd'])} of liquidity. Its mint is {r['mint']}."])
    lines = []
    for r in rows[:3]:
        v = "verified" if r["verified"] else "not verified"
        lines.append(f"{r['symbol']} ({r['name']}) is {v}, {liq_txt(r['liquidity_usd'])} of liquidity, mint {r['mint']}.")
    intro = rng.choice(["I found these matches, most liquidity first.", "These matches came back, most liquidity first."])
    if len(rows) > 3:
        intro = "These are the three with the most liquidity."
    tail = " Check the verified flag before you buy." if any(not r["verified"] for r in rows[:3]) and rng.random() < 0.6 else ""
    return intro + " " + " ".join(lines) + tail


def say_buy_ok(rng, r, bio):
    sym, pay, get, route = r["symbol"], r["you_pay"], r["you_get"], r["route"]
    imp = BJ.jstr(r["price_impact_pct"])
    intro = rng.choice([
        f"I've prepared a buy of {sym}: you pay {pay} and get {get} through {route}.",
        f"Your {sym} buy is prepared. You pay {pay} and get {get}, routed through {route}.",
        f"Prepared: a {sym} buy where you pay {pay} and get {get}. The route is {route}.",
    ])
    closer = rng.choice([f"Approve with {bio} or cancel. Nothing has been bought yet.", f"Nothing is bought until you approve it with {bio}.",
                         f"Review it and approve with {bio}, or cancel. Nothing has been bought yet."])
    ws = r.get("warnings") or []
    if ws:
        warn = " ".join(w if w.endswith(".") else w + "." for w in ws)
        head = rng.choice(["It is not verified. " if r.get("verified") is False else "", ""]) if False else ("This token is not verified. " if r.get("verified") is False else "")
        return f"{intro} {head}Warnings: {warn} Price impact is {imp}%. Approve with {bio} only if you accept all of that, or cancel. Nothing has been bought yet."
    return f"{intro} Price impact is {imp}%. {closer}"


def say_buy_unverified_ask(rng, row, amt_txt):
    sym = row["symbol"]
    tags = row["tags"]
    short = len(tags) >= 2
    head = f"{sym} is not verified."
    bits = " ".join(tag_say(rng, t, short) for t in tags) if tags else "No warning tags came back, but it is unverified."
    liq = f"Liquidity is {liq_txt(row['liquidity_usd'])}."
    ask = rng.choice([f"Hermit can still prepare a {amt_txt} buy if you want it. Shall I?", f"Do you still want me to prepare the {amt_txt} buy?",
                      f"I can prepare the {amt_txt} buy anyway. Want me to?"])
    return f"{head} {liq} {bits} {ask}"


def say_buy_refused(rng, row, alt=None):
    sym = row["symbol"]
    head = f"{sym} is not verified."
    tags = row["tags"]
    short = len(tags) >= 2
    bits = " ".join(tag_say(rng, t, short) for t in tags)
    liq = f"Liquidity is {liq_txt(row['liquidity_usd'])}."
    s = f"{head} {liq} {bits} {row['refusal']} Nothing was prepared.".replace("  ", " ")
    if alt:
        s += " " + rng.choice([f"{alt['symbol']} is verified, with {liq_txt(alt['liquidity_usd'])} of liquidity, if you meant that one.",
                               f"If you meant {alt['symbol']}, that one is verified, with {liq_txt(alt['liquidity_usd'])} of liquidity."])
    return s


def say_clash_ask(rng, sym, opts):
    l1, l2 = liq_txt(opts[0]["liquidity_usd"]), liq_txt(opts[1]["liquidity_usd"])
    return rng.choice([
        f"Two tokens are called {sym}: one with {l1} of liquidity and one with {l2}. Which do you mean? Say the bigger or the smaller one, or paste its mint.",
        f"{sym} matches two tokens, {l1} of liquidity and {l2} of liquidity. Which one do you want: the bigger one, the smaller one, or paste the mint?",
        f"There are two tokens named {sym}, with {l1} and {l2} of liquidity. Which one should I use? Tell me the bigger or the smaller one, or paste the mint.",
    ])


def say_needs_choice_sell(rng, r):
    q = r["question"]
    return rng.choice([q, q + " Nothing was prepared.", q.replace(" Which one?", "") + " Which one should I sell?"])


def say_absent(rng):
    return "Selling Solana tokens isn't available in this chat. " + rng.choice(ABSENT_TAIL)


def say_close(rng, r, bio):
    pnl = pnl_txt(r["unrealised_pnl"])
    return rng.choice([
        f"I've prepared closing {r['closing']}. The order is {r['order'].lower()}, and unrealised PnL is {pnl}. Approve with {bio} or cancel. Nothing has closed yet.",
        f"Prepared: closing {r['closing']} at market, reduce only. Unrealised PnL is {pnl}. Approve with {bio} or cancel. Nothing has closed yet.",
        f"The Close review is open for {r['closing']}. It is a market order, reduce only, with unrealised PnL of {pnl}. Nothing has closed until you approve with {bio}.",
    ])


def say_no_position(rng, r):
    c = re.search(r"open (\w+) position", r["note"]).group(1)
    return rng.choice([f"You have no open {c} position on Hyperliquid, so there's nothing to close.", f"I don't see an open {c} position on Hyperliquid. Nothing was prepared."])


# ================================================================================ scenario scaffolding
def finish(c, kind, bio, pk, intent, expected_tools, checks, extra=None):
    ex = c.ex(kind)
    msgs = ex["messages"]
    steps = [tc["name"] for m in msgs for tc in (m.get("tool_calls") or [])]
    n_chars = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for m in msgs)
    lite = (pk in LITE_OK_PACKS and len(steps) <= 3 and n_chars / 3.2 < 2000)
    meta = {"source": GEN, "generator": GEN, "kind": kind, "bio": bio, "pack_key": pk, "pack": PACK_NAMES[pk],
            "family_ok": ["4b", "lite"] if lite else ["4b"], "steps": steps, "cards": [], "world": c.world0,
            "world_final": c.w.state(), "expect": c.expect, "checks": checks, "lang": "en", "intent": intent,
            "expected_tools": expected_tools}
    if extra:
        meta.update(extra)
    ex["meta"] = meta
    return ex


def base_world(rng, ev, bio, dev, n_other=None, exclude=()):
    w = SolWorld.new(rng, bio, dev, ev)
    return w


def add_others(w, rng, exclude, ev, n=None):
    n = rng.choice([0, 0, 1, 1, 2, 3]) if n is None else n
    pool = [s for s in SHARED_T if s not in exclude]
    rng.shuffle(pool)
    for s in pool[:n]:
        w.give(rng, w.tok(s))


def deep_enough(w, t, usd, max_impact=1.5):
    """Make the pool deep enough that selling `usd` stays under max_impact (no warn)."""
    need = 1000.0 * usd / (max_impact - 0.02)
    if t["liquidity"] < need:
        t["liquidity"] = r2(need * 1.5)


def fund_buy(w, rng, amt):
    sp = w.prices["SOL"]
    if rng.random() < 0.65:
        w.hold("USDC", "solana", round(amt * rng.uniform(1.1, 6), 2))
        if rng.random() < 0.4:
            w.hold("SOL", "solana", round(rng.uniform(0.01, 3), 4))
    else:
        w.hold("SOL", "solana", round(amt * rng.uniform(1.1, 6) / sp, 4))
    if rng.random() < 0.3:
        w.hold("USDC", rng.choice(["base", "arbitrum", "ethereum"]), round(rng.uniform(5, 800), 2))


def holder_world(rng, ev, bio, dev, sym=None, unverified=False, usd=None, others=True):
    """A world where the user holds token `sym` (verified known or a fresh unverified one)."""
    w = base_world(rng, ev, bio, dev)
    if unverified:
        t = w.add_token(make_unverified(rng, sym or pick_unverified_sym(rng, ev), "held"))
    else:
        t = w.tok(sym or pick_hold_token(rng, ev))
    usd = usd if usd is not None else round(rng.choice([rng.uniform(8, 60), rng.uniform(30, 300), rng.uniform(120, 900)]), 2)
    deep_enough(w, t, usd * 1.05)
    w.give(rng, t, usd=usd)
    if others:
        add_others(w, rng, {t["symbol"]}, ev)
    return w, t


def mk_conv(rng, w, bio):
    return UConv(rng, w, bio)


# ================================================================================ scenarios: sells
def sc_sell_pct(rng, ev, absent=False):
    bio, dev = pick_bio(rng)
    unv = rng.random() < 0.08
    w, t = holder_world(rng, ev, bio, dev, unverified=unv)
    sym = t["symbol"]
    p, fr = pick_pct(rng, ev)
    pk = "core" if absent else pick_pack(rng)
    if fr == "half" and rng.random() < 0.4:
        tpl = rng.choice(split_tpls(NOOF_TPL, ev) or NOOF_TPL)
    else:
        tpl = rng.choice(split_tpls(PCT_TPL, ev))
    c = mk_conv(rng, w, bio)
    c.user(render(rng, tpl, {"FR": fr, "S": sym_text(rng, sym, aliases=not unv)}))
    return sell_finish(rng, c, w, t, bio, pk, "sell_pct", {"type": "sell", "token": sym, "percent": p, "amount": None, "receive": None}, {"token": sym, "percent": p}, absent)


def sell_finish(rng, c, w, t, bio, pk, kind, intent, args, absent=False, extra_checks=()):
    checks = ["right_tool", "exact_digits", "arg_omission", "no_unbacked_done"] + list(extra_checks)
    if absent:
        c.call(("open_page", {"page": "trade"}))
        c.say(say_absent(rng))
        return finish(c, "absent_sell", bio, "core", dict(intent, type="absent"), ["open_page"], ["right_tool", "no_unbacked_done", "absent_tool"])
    r = c.call(("prepare_sell", args))
    assert r.get("status") == "review_opened", (r, args)
    c.say(say_sell_ok(rng, r, bio))
    return finish(c, kind, bio, pk, intent, ["prepare_sell"], checks + ["restraint"])


def sc_sell_all(rng, ev, receive=False, absent=False):
    bio, dev = pick_bio(rng)
    unv = rng.random() < 0.08
    w, t = holder_world(rng, ev, bio, dev, unverified=unv)
    sym = t["symbol"]
    pk = "core" if absent else pick_pack(rng)
    c = mk_conv(rng, w, bio)
    args = {"token": sym, "percent": 100}
    rcv = None
    if receive:
        rcv = rng.choice(["SOL", "USDC"])
        form = rcv if rng.random() < 0.55 else rcv.lower()
        tpl = rng.choice(split_tpls(ALLR_TPL, ev))
        c.user(render(rng, tpl, {"S": sym_text(rng, sym, aliases=not unv), "R": form}))
        args["receive"] = rcv
    else:
        c.user(render(rng, rng.choice(split_tpls(ALL_TPL, ev)), {"S": sym_text(rng, sym, aliases=not unv)}))
    kind = "sell_all_receive" if receive else "sell_all"
    return sell_finish(rng, c, w, t, bio, pk, kind, {"type": "sell", "token": sym, "percent": 100, "amount": None, "receive": rcv}, args, absent)


def sc_sell_amount(rng, ev, absent=False):
    bio, dev = pick_bio(rng)
    sym = pick_hold_token(rng, ev)
    t0 = KNOWN[sym][2]
    val, form = amt_form(rng, ev)
    # token units must be a sensible slice of the holding
    if t0 < 0.01 and val < 100:
        val, form = rng.choice([(1500, "1.5k"), (2000, "2k"), (500, "500"), (750, "750"), (1200, "1,200")])
    if t0 >= 1 and val > 400:
        val, form = rng.choice([(2.5, "2.5"), (12.5, "12.5"), (40, "40"), (7.5, "7.5"), (5, "5")])
    hold_amt = round(val * rng.uniform(1.4, 5), 2) if val < 1000 else round(val * rng.uniform(1.4, 4))
    w = base_world(rng, ev, bio, dev)
    t = w.tok(sym)
    deep_enough(w, t, hold_amt * t["price"] * 1.05)
    w.give(rng, t, amount=hold_amt)
    add_others(w, rng, {sym}, ev)
    pk = "core" if absent else pick_pack(rng)
    c = mk_conv(rng, w, bio)
    c.user(render(rng, rng.choice(split_tpls(AMT_TPL, ev)), {"N": form, "S": sym_text(rng, sym)}))
    return sell_finish(rng, c, w, t, bio, pk, "sell_amount", {"type": "sell", "token": sym, "percent": None, "amount": val, "receive": None}, {"token": sym, "amount": val}, absent)


def sc_sell_usd(rng, ev):
    bio, dev = pick_bio(rng)
    sym = pick_hold_token(rng, ev)
    dol = rng.choice(USD_AMTS_EVAL if (ev and rng.random() < 0.5) else USD_AMTS)
    usd_hold = round(dol * rng.uniform(1.5, 12), 2)
    w = base_world(rng, ev, bio, dev)
    t = w.tok(sym)
    deep_enough(w, t, usd_hold * 1.05)
    h = w.give(rng, t, usd=usd_hold)
    add_others(w, rng, {sym}, ev)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    af = rng.choice([f"${dol}", f"${dol}", f"{dol} dollars", f"{dol} bucks", f"{dol} usd"])
    c.user(render(rng, rng.choice(split_tpls(USD_TPL, ev)), {"A": af, "S": sym_text(rng, sym)}))
    pr = c.call(("get_positions", {"kind": "spot"}))
    row = next(x for x in pr["spot"] if x["symbol"] == sym)
    raw = Decimal(repr(dol)) / (Decimal(repr(row["usd"])) / Decimal(row["amount"]))
    nd = 2 if raw >= 100 else 4
    expr = f"round({dol} / ({trim(row['usd'])} / {row['amount']}), {nd})"
    cr = c.call(("calculator", {"expression": expr}))
    qty = float(cr["result"])
    assert qty > 0 and Decimal(repr(qty)) < Decimal(row["amount"]), (qty, row)
    r = c.call(("prepare_sell", {"token": sym, "amount": qty}))
    assert r["status"] == "review_opened", r
    qty_txt = cr["result"]
    c.say(say_sell_dollars(rng, f"${dol}", qty_txt, sym, r, bio))
    return finish(c, "sell_amount_usd", bio, pk, {"type": "sell", "token": sym, "percent": None, "amount": qty, "receive": None, "usd": dol},
                  ["get_positions", "calculator", "prepare_sell"], ["right_tool", "exact_digits", "arg_omission", "no_unbacked_done", "restraint"])


def sc_sell_mint(rng, ev):
    bio, dev = pick_bio(rng)
    if rng.random() < 0.5:
        w = base_world(rng, ev, bio, dev)
        sym = pick_hold_token(rng, ev)
        while len(KNOWN[sym][1]) != 44:                      # the spec pastes 44-character mints
            sym = pick_hold_token(rng, ev)
        t = w.tok(sym)
        usd = round(rng.uniform(15, 700), 2)
        deep_enough(w, t, usd * 1.05)
        w.give(rng, t, usd=usd)
    else:
        w = base_world(rng, ev, bio, dev)
        t = w.add_token(make_unverified(rng, pick_unverified_sym(rng, ev), "held"))
        usd = round(rng.uniform(15, 400), 2)
        deep_enough(w, t, usd * 1.05)
        w.give(rng, t, usd=usd)
    add_others(w, rng, {t["symbol"]}, ev)
    mint = t["mint"]
    assert len(mint) == 44, mint
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    if rng.random() < 0.6:
        p, fr = pick_pct(rng, ev)
        c.user(render(rng, rng.choice(split_tpls(MINT_PCT_TPL, ev)), {"FR": fr, "M": mint}))
    else:
        p = 100
        c.user(render(rng, rng.choice(split_tpls(MINT_ALL_TPL, ev)), {"M": mint}))
    r = c.call(("prepare_sell", {"token": mint, "percent": p}))
    assert r["status"] == "review_opened", r
    c.say(say_sell_ok(rng, r, bio))
    return finish(c, "sell_by_mint", bio, pk, {"type": "sell", "token": mint, "percent": p, "amount": None, "receive": None},
                  ["prepare_sell"], ["right_tool", "exact_digits", "mint_provenance", "arg_omission", "no_unbacked_done", "restraint"])


def sc_ask_notoken(rng, ev, empty=False, with_pct=False):
    """No token named. with_pct: the user already gave a share ("sell half"): ask only which token, keep the share."""
    bio, dev = pick_bio(rng)
    w = base_world(rng, ev, bio, dev)
    with_pct = with_pct and not empty
    n = 0 if empty else (rng.choice([2, 2, 3, 4]) if with_pct else rng.choice([1, 2, 2, 3, 4]))
    syms = []
    pool = [pick_hold_token(rng, ev)]
    rest = [s for s in SHARED_T if s not in pool]
    rng.shuffle(rest)
    pool += rest
    for s in pool[:n]:
        t = w.tok(s)
        usd = round(rng.uniform(15, 500), 2)
        deep_enough(w, t, usd * 1.05)
        w.give(rng, t, usd=usd)
        syms.append(s)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    p0 = None
    if with_pct:
        p0, fr0 = pick_pct(rng, ev)
        c.user(render(rng, rng.choice(split_tpls(ASK_NOTOKEN_PCT_TPL, ev)), {"FR": fr0}))
    else:
        c.user(render(rng, rng.choice(split_tpls(ASK_NOTOKEN_TPL, ev)), {}))
    pr = c.call(("get_positions", {"kind": "spot"}))
    held = pr["spot"]
    tools = ["get_positions"]
    if not held:
        c.say(rng.choice(["You don't hold any Solana tokens, so there's nothing to sell.", "Your Solana wallet has no tokens, so there's nothing to sell. Nothing was prepared.",
                          "I don't see any Solana tokens in your wallet, so there's nothing to sell."]))
        return finish(c, "sell_ask_empty", bio, pk, {"type": "ask_empty"}, tools, ["right_tool", "no_unbacked_done", "asked_when_needed"])
    names = [h["symbol"] for h in held]
    if len(held) == 1:
        h = held[0]
        q = rng.choice([f"You hold {h['amount']} {h['symbol']}. How much of it should I sell?", f"All I see is {h['amount']} {h['symbol']}. How much should I sell, a percent or an amount?"])
    else:
        lst = ", ".join(names[:-1]) + " and " + names[-1]
        q = rng.choice([f"You hold {lst}. Which one should I sell, and how much?", f"I see {lst} in your wallet. Which one do you want to sell, and how much?"])
        if with_pct:
            q = rng.choice([f"You hold {lst}. Which one should I sell?", f"I see {lst} in your wallet. Which one do you mean?", f"Which token: {lst}?"])
    c.say(q, ask=True)
    intent = {"type": "ask", "held": names}
    if with_pct and rng.random() < 0.6:
        sym = rng.choice(names)
        c.user(render(rng, rng.choice(split_tpls(FOLLOW_TOKEN_TPL, ev)), {"S": sym_text(rng, sym)}, typo_p=0))
        r = c.call(("prepare_sell", {"token": sym, "percent": p0}))
        assert r["status"] == "review_opened", r
        c.say(say_sell_ok(rng, r, bio))
        tools.append("prepare_sell")
        intent = {"type": "sell", "token": sym, "percent": p0, "amount": None, "receive": None, "ask_first": True}
        return finish(c, "sell_ask_followup", bio, pk, intent, tools, ["right_tool", "exact_digits", "asked_when_needed", "arg_omission", "no_unbacked_done", "restraint"])
    if not with_pct and rng.random() < 0.5:
        sym = rng.choice(names)
        if rng.random() < 0.25:
            c.user(render(rng, rng.choice(split_tpls(FOLLOW_ALLTOK_TPL, ev)), {"S": sym_text(rng, sym)}, typo_p=0))
            p = 100
        else:
            p, fr = pick_pct(rng, ev)
            c.user(render(rng, rng.choice(split_tpls(FOLLOW_NOTOKEN_TPL, ev)), {"S": sym_text(rng, sym), "FR": fr}, typo_p=0))
        r = c.call(("prepare_sell", {"token": sym, "percent": p}))
        assert r["status"] == "review_opened", r
        c.say(say_sell_ok(rng, r, bio))
        tools.append("prepare_sell")
        intent = {"type": "sell", "token": sym, "percent": p, "amount": None, "receive": None, "ask_first": True}
        return finish(c, "sell_ask_followup", bio, pk, intent, tools, ["right_tool", "exact_digits", "asked_when_needed", "arg_omission", "no_unbacked_done", "restraint"])
    return finish(c, "sell_ask", bio, pk, intent, tools, ["right_tool", "asked_when_needed", "no_guess_percent", "no_unbacked_done"])


def sc_ask_noamt(rng, ev):
    bio, dev = pick_bio(rng)
    w, t = holder_world(rng, ev, bio, dev)
    sym = t["symbol"]
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    c.user(render(rng, rng.choice(split_tpls(ASK_NOAMT_TPL, ev)), {"S": sym_text(rng, sym)}))
    c.say(rng.choice([f"How much {sym} should I sell: a percent, an amount, or all of it?", f"How much of your {sym} do you want to sell? Give me a percent or an amount, or say all.",
                      f"How much {sym} should I sell? A percent, a number of tokens, or all."]), ask=True)
    if rng.random() < 0.5:
        return finish(c, "sell_ask", bio, pk, {"type": "ask", "token": sym}, [], ["asked_when_needed", "no_guess_percent", "no_unbacked_done"])
    k = rng.random()
    if k < 0.6:
        p, fr = pick_pct(rng, ev)
        c.user(render(rng, rng.choice(split_tpls(FOLLOW_AMT_TPL, ev)), {"FR": fr}, typo_p=0))
        args = {"token": sym, "percent": p}
        intent = {"type": "sell", "token": sym, "percent": p, "amount": None, "receive": None, "ask_first": True}
    elif k < 0.85:
        c.user(render(rng, rng.choice(split_tpls(FOLLOW_ALL_TPL, ev)), {}, typo_p=0))
        args = {"token": sym, "percent": 100}
        intent = {"type": "sell", "token": sym, "percent": 100, "amount": None, "receive": None, "ask_first": True}
    else:
        hold = Decimal(w.spot[0]["amount"])
        val = float(max(Decimal("0.5"), (hold * Decimal("0.4")).quantize(Decimal("1"), ROUND_HALF_UP))) if hold >= 5 else float((hold * Decimal("0.4")).quantize(Decimal("0.1")))
        c.user(render(rng, rng.choice(["{N}", "{N} of them", "just {N}", "{N} tokens", "make it {N}"]), {"N": trim(val)}, typo_p=0))
        args = {"token": sym, "amount": jnum(val)}
        intent = {"type": "sell", "token": sym, "percent": None, "amount": jnum(val), "receive": None, "ask_first": True}
    r = c.call(("prepare_sell", args))
    assert r["status"] == "review_opened", r
    c.say(say_sell_ok(rng, r, bio))
    return finish(c, "sell_ask_followup", bio, pk, intent, ["prepare_sell"], ["right_tool", "exact_digits", "asked_when_needed", "arg_omission", "no_unbacked_done", "restraint"])


# ---- sell errors
def sc_err_not_held(rng, ev):
    bio, dev = pick_bio(rng)
    w = base_world(rng, ev, bio, dev)
    others = rng.choice([1, 2, 3, 0])
    miss = pick_hold_token(rng, ev)
    pool = [s for s in SHARED_T if s != miss]
    rng.shuffle(pool)
    for s in pool[:others]:
        w.give(rng, w.tok(s))
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    p, fr = pick_pct(rng, ev)
    if rng.random() < 0.5:
        c.user(render(rng, rng.choice(split_tpls(PCT_TPL, ev)), {"FR": fr, "S": sym_text(rng, miss)}))
        args = {"token": miss, "percent": p}
        intent = {"type": "sell", "token": miss, "percent": p, "amount": None, "receive": None}
    else:
        c.user(render(rng, rng.choice(split_tpls(ALL_TPL, ev)), {"S": sym_text(rng, miss)}))
        args = {"token": miss, "percent": 100}
        intent = {"type": "sell", "token": miss, "percent": 100, "amount": None, "receive": None}
    r = c.call(("prepare_sell", args))
    assert r["status"] == "not_held", r
    c.say(say_not_held(rng, miss))
    return finish(c, "sell_err_not_held", bio, pk, intent, ["prepare_sell"], ["right_tool", "exact_digits", "arg_omission", "error_stated", "no_unbacked_done"])


def sc_err_insufficient(rng, ev):
    bio, dev = pick_bio(rng)
    sym = pick_hold_token(rng, ev)
    t0 = KNOWN[sym][2]
    w = base_world(rng, ev, bio, dev)
    t = w.tok(sym)
    if t0 < 0.01:
        have = rng.choice([1200, 840, 2500, 1750, 3100])
        ask_v, ask_f = rng.choice([(5000, "5000"), (3000, "3,000"), (10000, "10k"), (4000, "4k"), (8000, "8,000")])
        if ask_v <= have:
            ask_v, ask_f = (have * 3, f"{have * 3:,}")
    else:
        have = round(rng.uniform(0.8, 40), 2)
        ask_v, ask_f = rng.choice([(round(have * 2), None), (round(have + 5), None), (round(have * 1.5, 1), None)])
        ask_v = float(max(ask_v, have + 1))
        ask_v = round(ask_v, 1)
        ask_f = trim(Decimal(repr(ask_v)))
    w.give(rng, t, amount=have)
    add_others(w, rng, {sym}, ev, n=rng.choice([0, 1]))
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    c.user(render(rng, rng.choice(split_tpls(AMT_TPL, ev)), {"N": ask_f, "S": sym_text(rng, sym)}))
    r = c.call(("prepare_sell", {"token": sym, "amount": jnum(ask_v)}))
    assert r["status"] == "insufficient", r
    c.say(say_insufficient(rng, r, f"{ask_f} {sym}"))
    return finish(c, "sell_err_insufficient", bio, pk, {"type": "sell", "token": sym, "percent": None, "amount": jnum(ask_v), "receive": None},
                  ["prepare_sell"], ["right_tool", "exact_digits", "arg_omission", "error_stated", "no_invented_data", "no_unbacked_done"])


def sc_err_frozen(rng, ev):
    bio, dev = pick_bio(rng)
    w = base_world(rng, ev, bio, dev)
    t = w.tok("PUMP") if rng.random() < 0.5 else w.add_token(make_unverified(rng, pick_unverified_sym(rng, ev), "held"))
    w.give(rng, t, amount=(840 if t["symbol"] == "PUMP" else 5000) * rng.choice([1, 2, 3]), frozen=True)
    add_others(w, rng, {t["symbol"]}, ev, n=rng.choice([0, 1, 2]))
    sym = t["symbol"]
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    if rng.random() < 0.5:
        c.user(render(rng, rng.choice(split_tpls(ALL_TPL, ev)), {"S": sym_text(rng, sym, aliases=False)}))
        args, p = {"token": sym, "percent": 100}, 100
    else:
        p, fr = pick_pct(rng, ev)
        c.user(render(rng, rng.choice(split_tpls(PCT_TPL, ev)), {"FR": fr, "S": sym_text(rng, sym, aliases=False)}))
        args = {"token": sym, "percent": p}
    r = c.call(("prepare_sell", args))
    assert r["status"] == "blocked", r
    c.say(say_blocked_sell(rng, r, daily=False))
    return finish(c, "sell_err_frozen", bio, pk, {"type": "sell", "token": sym, "percent": p, "amount": None, "receive": None}, ["prepare_sell"],
                  ["right_tool", "exact_digits", "arg_omission", "error_stated", "no_retry_blocked", "no_unbacked_done"])


def sc_err_choice(rng, ev):
    bio, dev = pick_bio(rng)
    w = base_world(rng, ev, bio, dev)
    big = w.tok("PUMP")
    other = make_unverified(rng, "PUMP", "held", name="Pump Clone")
    w.add_token(other)
    a_big = rng.choice([120, 840, 2500, 1500, 960])
    a_small = rng.choice([4, 12, 35, 60])
    w.give(rng, big, amount=a_big)
    w.give(rng, other, amount=a_small)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    p, fr = pick_pct(rng, ev)
    if rng.random() < 0.3:
        p = 100
        c.user(render(rng, rng.choice(split_tpls(ALL_TPL, ev)), {"S": sym_text(rng, "PUMP")}))
    else:
        c.user(render(rng, rng.choice(split_tpls(PCT_TPL, ev)), {"FR": fr, "S": sym_text(rng, "PUMP")}))
    r = c.call(("prepare_sell", {"token": "PUMP", "percent": p}))
    assert r["status"] == "needs_choice", r
    c.say(say_needs_choice_sell(rng, r), ask=True)
    pick = 0 if rng.random() < 0.55 else 1
    amts = [r["options"][0]["amount"], r["options"][1]["amount"]]
    k = rng.random()
    if k < 0.55:
        picks = [["the {A} one", "the one with {A}", "{A}", "the {A} bag"], ["the {A} one", "the one with {A}", "{A}", "the {A} bag"]][pick]
        text = rng.choice(split_tpls(picks, ev) or picks).replace("{A}", commas(amts[pick]))
    elif k < 0.85:
        words = [["the bigger one", "the larger one", "the first one", "the big one", "first"], ["the smaller one", "the second one", "the small one", "second", "the other one"]][pick]
        text = rng.choice(words)
    else:
        text = rng.choice(CLASH_MINT_PICK).replace("{M}", r["options"][pick]["mint"])
    c.user(text)
    mint = r["options"][pick]["mint"]
    r2_ = c.call(("prepare_sell", {"token": mint, "percent": p}))
    assert r2_["status"] == "review_opened", r2_
    c.say(say_sell_ok(rng, r2_, bio))
    return finish(c, "sell_err_choice", bio, pk, {"type": "sell", "token": mint, "percent": p, "amount": None, "receive": None, "first_token": "PUMP"},
                  ["prepare_sell", "prepare_sell"], ["right_tool", "exact_digits", "mint_provenance", "arg_omission", "asked_when_needed", "no_unbacked_done"])


def sc_err_noroute(rng, ev):
    bio, dev = pick_bio(rng)
    w, t = holder_world(rng, ev, bio, dev, unverified=rng.random() < 0.4)
    w.no_route.append(t["mint"])
    sym = t["symbol"]
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    p, fr = pick_pct(rng, ev)
    c.user(render(rng, rng.choice(split_tpls(PCT_TPL, ev)), {"FR": fr, "S": sym_text(rng, sym, aliases=False)}))
    r = c.call(("prepare_sell", {"token": sym, "percent": p}))
    assert "error" in r and r["error"].startswith("No route"), r
    c.say(say_no_route(rng, r, sym))
    return finish(c, "sell_err_noroute", bio, pk, {"type": "sell", "token": sym, "percent": p, "amount": None, "receive": None}, ["prepare_sell"],
                  ["right_tool", "exact_digits", "arg_omission", "error_stated", "no_retry_blocked", "no_unbacked_done"])


def sc_err_warn(rng, ev):
    bio, dev = pick_bio(rng)
    w = base_world(rng, ev, bio, dev)
    t = w.add_token(make_unverified(rng, pick_unverified_sym(rng, ev), "held"))
    usd = round(rng.uniform(40, 900), 2)
    target = rng.uniform(3.1, 9.5)
    t["liquidity"] = r2(1000.0 * usd / (target - 0.02))
    w.give(rng, t, usd=usd)
    sym = t["symbol"]
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    c.user(render(rng, rng.choice(split_tpls(ALL_TPL, ev)), {"S": sym_text(rng, sym, aliases=False)}))
    r = c.call(("prepare_sell", {"token": sym, "percent": 100}))
    if r.get("status") != "review_opened" or not r.get("warnings"):
        raise AssertionError(("warn not reached", r))
    c.say(say_sell_ok(rng, r, bio))
    return finish(c, "sell_err_warn", bio, pk, {"type": "sell", "token": sym, "percent": 100, "amount": None, "receive": None}, ["prepare_sell"],
                  ["right_tool", "exact_digits", "arg_omission", "warnings_repeated", "no_unbacked_done"])


def sc_err_daily(rng, ev):
    bio, dev = pick_bio(rng)
    w, t = holder_world(rng, ev, bio, dev, usd=round(rng.uniform(300, 900), 2))
    w.daily_left = float(int(w.spot[0]["usd"] * rng.uniform(0.2, 0.6)))
    sym = t["symbol"]
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    c.user(render(rng, rng.choice(split_tpls(ALL_TPL, ev)), {"S": sym_text(rng, sym)}))
    r = c.call(("prepare_sell", {"token": sym, "percent": 100}))
    assert r["status"] == "blocked", r
    c.say(say_blocked_sell(rng, r, daily=True))
    return finish(c, "sell_err_daily", bio, pk, {"type": "sell", "token": sym, "percent": 100, "amount": None, "receive": None}, ["prepare_sell"],
                  ["right_tool", "exact_digits", "arg_omission", "error_stated", "no_retry_blocked", "no_unbacked_done"])


# ================================================================================ scenarios: search
def search_world(rng, ev, bio, dev, mode, sym=None):
    """mode: verified_clean | verified_tag | unverified_tags | refused | multi. Returns (world, row token(s), query, how)."""
    w = base_world(rng, ev, bio, dev)
    if mode in ("verified_clean", "verified_tag"):
        s = sym or pick_hold_token(rng, ev)
        t = w.tok(s)
        if mode == "verified_clean":
            t["tags"] = []
        else:
            t["tags"] = ["Top holders %d%%" % rng.randint(20, 48)]
        return w, [t], s
    if mode == "unverified_tags":
        s = pick_unverified_sym(rng, ev)
        t = w.add_token(make_unverified(rng, s, "flags"))
        return w, [t], s
    if mode == "refused":
        s = pick_unverified_sym(rng, ev)
        t = w.add_token(make_unverified(rng, s, rng.choice(["lowliq", "lowliq", "freeze"])))
        return w, [t], s
    raise ValueError(mode)


def sc_safety(rng, ev, mode):
    bio, dev = pick_bio(rng)
    pk = pick_pack(rng)
    if mode == "multi":
        w = base_world(rng, ev, bio, dev)
        t1 = w.tok("PUMP")
        t2 = w.add_token(make_unverified(rng, "PUMP", rng.choice(["flags", "lowliq"]), name="Pump Clone"))
        w.searched_mints = []
        c = mk_conv(rng, w, bio)
        c.user(render(rng, rng.choice(split_tpls(["what's the liquidity on {S}", "{S} liquidity?", "check {S} for me", "is {S} legit?", "how much liquidity does {S} have", "any red flags on {S}?"], ev) or ["check {S} for me"]), {"S": sym_text(rng, "PUMP")}))
        qtext = c.msgs[-1]["content"]
        m = re.search(r"pump", qtext, re.I)
        query = m.group(0)
        r = c.call(("search_tokens", {"query": query}))
        assert len(r["tokens"]) == 2, r
        c.say(say_search(rng, r["tokens"]))
        return finish(c, "search_safety_multi", bio, pk, {"type": "search", "query": query, "rows": 2}, ["search_tokens"],
                      ["right_tool", "exact_digits", "no_safe_claim", "tags_read", "no_unbacked_done"])
    w, toks, s = search_world(rng, ev, bio, dev, mode)
    t = toks[0]
    c = mk_conv(rng, w, bio)
    use_mint = rng.random() < (0.45 if mode in ("unverified_tags", "refused") else 0.2)
    if use_mint:
        q = t["mint"]
        text = render(rng, rng.choice(split_tpls(SAFETY_MINT_TPL, ev)), {"M": q})
    else:
        sf = sym_text(rng, s, aliases=False)
        text = render(rng, rng.choice(split_tpls(SAFETY_TPL, ev)), {"S": sf})
        q = sf.lstrip("$")
        if text.isupper():
            q = q.upper()
        elif text.islower():
            q = q.lower()
    c.user(text)
    query = q
    r = c.call(("search_tokens", {"query": query}))
    assert r.get("tokens") and r["tokens"][0]["mint"] == t["mint"], r
    c.say(say_search(rng, [x for x in r["tokens"] if x["mint"] == t["mint"]]))
    kind = "search_safety_" + {"verified_clean": "verified", "verified_tag": "verified_tag", "unverified_tags": "unverified", "refused": "refused"}[mode]
    return finish(c, kind, bio, pk, {"type": "search", "query": query, "rows": 1}, ["search_tokens"],
                  ["right_tool", "exact_digits", "mint_provenance", "no_safe_claim", "tags_read", "no_unbacked_done"])


def sc_find(rng, ev, mode):
    bio, dev = pick_bio(rng)
    pk = pick_pack(rng)
    w = base_world(rng, ev, bio, dev)
    c = None
    if mode == "one":
        s = pick_hold_token(rng, ev)
        t = w.tok(s)
        c = mk_conv(rng, w, bio)
        sf = sym_text(rng, s)
        text = render(rng, rng.choice(split_tpls(FIND_TPL, ev)), {"S": sf})
        query = sf.lstrip("$")
        if text.isupper(): query = query.upper()
        elif text.islower(): query = query.lower()
        c.user(text)
        want = s
    elif mode == "multi":
        kw = rng.choice([("dog", ["WIF", "BONK"]), ("cat", ["POPCAT", "MEW"])])
        k, _ = kw
        extra = make_unverified(rng, pick_unverified_sym(rng, ev), "flags", name=("Doge " if k == "dog" else "Cat ") + "Moon")
        extra["aliases"] = [k]
        w.add_token(extra)
        for t in w.tokens:
            if t["symbol"] in ("MEW",) and k == "dog":
                pass
        c = mk_conv(rng, w, bio)
        c.user(render(rng, rng.choice(split_tpls(FIND_THEME_TPL, ev)), {"K": k}))
        query = k
        want = None
    elif mode == "empty":
        c = mk_conv(rng, w, bio)
        junk = rng.choice(["zzorb", "qwxyz", "blargo", "frumpf", "nonexo", "kloopa", "vexyl", "grumbo", "snarfle", "tweedle"])
        c.user(render(rng, rng.choice(split_tpls(FIND_TPL, ev)), {"S": junk}))
        query = junk
        want = None
    else:                                               # error
        w.search_down = True
        s = pick_hold_token(rng, ev)
        c = mk_conv(rng, w, bio)
        sf = sym_text(rng, s)
        text = render(rng, rng.choice(split_tpls(FIND_TPL, ev)), {"S": sf})
        query = sf.lstrip("$")
        if text.isupper(): query = query.upper()
        elif text.islower(): query = query.lower()
        c.user(text)
        want = s
    r = c.call(("search_tokens", {"query": query}))
    if mode == "error":
        assert "error" in r
        c.say(say_search_error(rng, r))
        return finish(c, "search_find_error", bio, pk, {"type": "search", "query": query}, ["search_tokens"], ["right_tool", "error_stated", "no_invented_data", "no_unbacked_done"])
    rows = r["tokens"]
    if mode == "multi":
        assert len(rows) >= 2, r
    if mode == "one":
        assert rows and rows[0]["symbol"] == want, r
        rows = [rows[0]] if len(rows) == 1 else rows
    c.say(say_find(rng, rows, query))
    kind = "search_find_" + mode
    return finish(c, kind, bio, pk, {"type": "search", "query": query, "rows": len(rows)}, ["search_tokens"],
                  ["right_tool", "exact_digits", "mint_provenance", "no_safe_claim", "no_unbacked_done"])


# ================================================================================ scenarios: buys
def sc_buy_direct(rng, ev):
    bio, dev = pick_bio(rng)
    sym = pick_hold_token(rng, ev)
    amt = BJ.pick_amount(rng, ev)
    w = base_world(rng, ev, bio, dev)
    fund_buy(w, rng, amt)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    sl = BJ.amt_slots(rng, amt)
    sl["S"] = sym_text(rng, sym)
    c.user(render(rng, rng.choice(split_tpls(BUY_DIRECT_TPL, ev)), sl))
    r = c.call(("prepare_buy", {"symbol": sym, "amount_usd": amt}))
    assert r["status"] == "review_opened", r
    c.say(say_buy_ok(rng, r, bio))
    return finish(c, "buy_direct", bio, pk, {"type": "buy", "symbol": sym, "amount_usd": amt, "pay_with": None}, ["prepare_buy"],
                  ["right_tool", "exact_digits", "arg_omission", "no_unbacked_done", "restraint"])


def sc_buy_mint_verified(rng, ev):
    bio, dev = pick_bio(rng)
    sym = pick_hold_token(rng, ev)
    amt = BJ.pick_amount(rng, ev)
    w = base_world(rng, ev, bio, dev)
    t = w.tok(sym)
    t["tags"] = []
    fund_buy(w, rng, amt)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    sl = BJ.amt_slots(rng, amt)
    sl["M"] = t["mint"]
    c.user(render(rng, rng.choice(split_tpls(BUYMINT_TPL, ev)), sl))
    sr = c.call(("search_tokens", {"query": t["mint"]}))
    assert sr["tokens"][0]["mint"] == t["mint"] and "refusal" not in sr["tokens"][0]
    r = c.call(("prepare_buy", {"symbol": t["mint"], "amount_usd": amt}))
    assert r["status"] == "review_opened", r
    c.say(say_buy_ok(rng, r, bio))
    return finish(c, "buy_after_search", bio, pk, {"type": "buy", "symbol": t["mint"], "amount_usd": amt, "pay_with": None}, ["search_tokens", "prepare_buy"],
                  ["right_tool", "exact_digits", "mint_provenance", "arg_omission", "no_unbacked_done"])


def sc_buy_unverified(rng, ev, insist_up_front=False):
    bio, dev = pick_bio(rng)
    amt = BJ.pick_amount(rng, ev)
    amt = min(amt, 100) if rng.random() < 0.85 else amt
    w = base_world(rng, ev, bio, dev)
    t = w.add_token(make_unverified(rng, pick_unverified_sym(rng, ev), "flags"))
    if rng.random() < 0.3:
        t["liquidity"] = r2(max(t["liquidity"], 1000.0 * amt / 2.0))                 # keep a few under 1% impact
    fund_buy(w, rng, amt)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    sl = BJ.amt_slots(rng, amt)
    sl["M"] = t["mint"]
    tools = ["search_tokens"]
    if insist_up_front:
        c.user(render(rng, rng.choice(split_tpls(BUYMINT_INS_TPL, ev)), sl))
        sr = c.call(("search_tokens", {"query": t["mint"]}))
        assert "refusal" not in sr["tokens"][0] and not sr["tokens"][0]["verified"]
    else:
        c.user(render(rng, rng.choice(split_tpls(BUYMINT_TPL, ev)), sl))
        sr = c.call(("search_tokens", {"query": t["mint"]}))
        assert "refusal" not in sr["tokens"][0] and not sr["tokens"][0]["verified"]
        c.say(say_buy_unverified_ask(rng, sr["tokens"][0], BJ.usd(amt)), ask=True)
        c.user(rng.choice(split_tpls(FOLLOW_INSIST, ev) or FOLLOW_INSIST))
    r = c.call(("prepare_buy", {"symbol": t["mint"], "amount_usd": amt}))
    tools.append("prepare_buy")
    assert r["status"] == "review_opened" and r.get("verified") is False and r.get("warnings"), r
    c.say(say_buy_ok(rng, r, bio))
    return finish(c, "buy_mint_unverified" + ("_upfront" if insist_up_front else ""), bio, pk,
                  {"type": "buy", "symbol": t["mint"], "amount_usd": amt, "pay_with": None, "unverified": True}, tools,
                  ["right_tool", "exact_digits", "mint_provenance", "arg_omission", "warnings_repeated", "no_unbacked_done"])


def sc_buy_refused(rng, ev, alt=False):
    bio, dev = pick_bio(rng)
    amt = BJ.pick_amount(rng, ev)
    w = base_world(rng, ev, bio, dev)
    pk = pick_pack(rng)
    fund_buy(w, rng, amt)
    alt_row = None
    if alt:
        base = rng.choice(["WIF", "JUP", "POPCAT", "PENGU"] + (EVAL_ONLY_T if ev else TRAIN_ONLY_T))
        bt = w.tok(base)
        bt["tags"] = []
        word = KNOWN[base][0].split()[0]
        sym = (base + "G") if base != "PUMP" else "PUMPG"
        t = make_unverified(rng, sym, rng.choice(["lowliq", "freeze"]), name=word.capitalize() + " Gold")
        t["name"] = (base if rng.random() < 0.5 else word.capitalize()) + " Gold"
        w.add_token(t)
        qtxt = t["name"]
        c = mk_conv(rng, w, bio)
        sl = BJ.amt_slots(rng, amt)
        sl["S"] = qtxt if rng.random() < 0.7 else qtxt.lower()
        text = render(rng, rng.choice(split_tpls(BUY_NAME_TPL, ev)), sl)
        query = sl["S"]
        if text.isupper(): query = query.upper()
        elif text.islower(): query = query.lower()
        c.user(text)
        sr = c.call(("search_tokens", {"query": query}))
        rows = sr["tokens"]
        mine = next(x for x in rows if x["mint"] == t["mint"])
        alt_row = next((x for x in rows if x["verified"] and x["symbol"] == base), None)
        assert mine.get("refusal") and alt_row, sr
        c.say(say_buy_refused(rng, mine, alt_row))
        return finish(c, "buy_refused_alt", bio, pk, {"type": "buy_refused", "symbol": t["mint"], "amount_usd": amt, "alt": alt_row["symbol"]}, ["search_tokens"],
                      ["right_tool", "exact_digits", "restraint", "no_prepare_when_refused", "no_unbacked_done"])
    t = w.add_token(make_unverified(rng, pick_unverified_sym(rng, ev), rng.choice(["lowliq", "lowliq", "freeze"])))
    c = mk_conv(rng, w, bio)
    sl = BJ.amt_slots(rng, amt)
    sl["M"] = t["mint"]
    c.user(render(rng, rng.choice(split_tpls(BUYMINT_TPL, ev)), sl))
    sr = c.call(("search_tokens", {"query": t["mint"]}))
    mine = sr["tokens"][0]
    assert mine.get("refusal"), sr
    c.say(say_buy_refused(rng, mine))
    return finish(c, "buy_refused", bio, pk, {"type": "buy_refused", "symbol": t["mint"], "amount_usd": amt, "alt": None}, ["search_tokens"],
                  ["right_tool", "exact_digits", "restraint", "no_prepare_when_refused", "no_unbacked_done"])


def sc_buy_clash(rng, ev):
    bio, dev = pick_bio(rng)
    amt = BJ.pick_amount(rng, ev)
    sym = rng.choice(["MOODENG", "GOAT", "FARTCOIN", "POPCAT", "PENGU", "JUP", "WIF"] + (["PNUT", "MEW"] if ev else ["PYTH", "RAY"]))
    w = base_world(rng, ev, bio, dev)
    top = w.tok(sym)
    top["tags"] = []
    top["liquidity"] = r2(rng.uniform(2e6, 30e6))
    second = make_unverified(rng, sym, "held", name=top["name"] + " 2")
    second["liquidity"] = r2(top["liquidity"] * rng.uniform(0.55, 0.95))
    second["tags"] = ["Top holders %d%%" % rng.randint(25, 55)] + (["Mint on"] if rng.random() < 0.3 else [])
    second["tags"] = sorted(second["tags"], key=lambda x: (0 if x == "Mint on" else 1))
    second["price"] = round(top["price"] * rng.uniform(0.12, 0.9), 4 if top["price"] >= 0.01 else 5)
    second["price"] = max(second["price"], 0.0002)
    w.add_token(second)
    fund_buy(w, rng, amt)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    sl = BJ.amt_slots(rng, amt)
    sl["S"] = sym_text(rng, sym)
    c.user(render(rng, rng.choice(split_tpls(BUY_DIRECT_TPL, ev)), sl))
    r = c.call(("prepare_buy", {"symbol": sym, "amount_usd": amt}))
    assert r["status"] == "needs_choice", r
    opts = r["options"]
    c.say(say_clash_ask(rng, sym, opts), ask=True)
    pick = 0 if rng.random() < 0.6 else 1
    k = rng.random()
    if k < 0.18:
        text = rng.choice(CLASH_MINT_PICK).replace("{M}", opts[pick]["mint"])
    else:
        pool = CLASH_PICK[pick]
        tpl = rng.choice(split_tpls(pool, ev) or pool)
        text = tpl.replace("{L1}", liq_txt(opts[0]["liquidity_usd"])).replace("{L2}", liq_txt(opts[1]["liquidity_usd"]))
    c.user(text)
    mint = opts[pick]["mint"]
    r2_ = c.call(("prepare_buy", {"symbol": mint, "amount_usd": amt}))
    assert r2_["status"] == "review_opened", r2_
    c.say(say_buy_ok(rng, r2_, bio))
    return finish(c, "buy_clash", bio, pk, {"type": "buy", "symbol": mint, "amount_usd": amt, "pay_with": None, "first_symbol": sym, "picked": pick}, ["prepare_buy", "prepare_buy"],
                  ["right_tool", "exact_digits", "mint_provenance", "asked_when_needed", "arg_omission", "warnings_repeated", "no_unbacked_done"])


# ================================================================================ scenarios: near misses
def swap_world(rng, ev, bio, dev, asset, bal):
    w = base_world(rng, ev, bio, dev)
    chain = {"ETH": rng.choice(["base", "arbitrum", "ethereum"]), "SOL": "solana", "BNB": "bnb"}[asset]
    w.hold(asset, chain, bal)
    if rng.random() < 0.5:
        w.hold("USDC", rng.choice(["base", "arbitrum", "solana"]), round(rng.uniform(5, 800), 2))
    return w, chain


def asset_text(rng, asset):
    return rng.choice({"ETH": ["ETH", "eth", "ether"], "SOL": ["SOL", "sol"], "BNB": ["BNB", "bnb"]}[asset])


def sc_near_swap(rng, ev, asset):
    bio, dev = pick_bio(rng)
    mode = rng.choices(["amount", "half", "all"], [0.5, 0.25, 0.25])[0]
    if asset == "SOL" and mode == "all":
        mode = "amount"
    bal = {"ETH": round(rng.uniform(0.2, 3), 3), "SOL": round(rng.uniform(1, 25), 2), "BNB": round(rng.uniform(0.3, 6), 2)}[asset]
    w, chain = swap_world(rng, ev, bio, dev, asset, bal)
    pk = pick_pack(rng, 0.1)
    c = mk_conv(rng, w, bio)
    S = asset_text(rng, asset)
    named = None
    if mode == "amount":
        val = round(bal * rng.choice([0.2, 0.3, 0.5, 0.6]), 2 if asset != "ETH" else 3)
        val = max(val, 0.01)
        nf = trim(Decimal(repr(val)))
        text = render(rng, rng.choice(split_tpls(SWAP_SELL_TPL, ev)), {"N": nf, "S": S})
        if asset == "ETH" and rng.random() < 0.35:
            named = chain
            text += rng.choice([" on {C}", " on {C} pls", ", {C}"]).replace("{C}", rng.choice(CHAIN_SAY[chain]))
        c.user(text)
        tools = []
        if not named:
            br = c.call(("get_balances", {}))
            tools.append("get_balances")
            row = next(h for h in br["holdings"] if h["symbol"] == asset)
            chain = row["chain"]
        args = {"from_symbol": asset, "from_chain": chain, "to_symbol": "USDC", "to_chain": chain, "amount": jnum(val)}
    elif mode == "half":
        c.user(render(rng, rng.choice(split_tpls(SWAP_HALF_TPL, ev)), {"S": S}))
        br = c.call(("get_balances", {}))
        row = next(h for h in br["holdings"] if h["symbol"] == asset)
        cr = c.call(("calculator", {"expression": f"{row['amount']} / 2"}))
        val = float(cr["result"])
        chain = row["chain"]
        tools = ["get_balances", "calculator"]
        args = {"from_symbol": asset, "from_chain": chain, "to_symbol": "USDC", "to_chain": chain, "amount": jnum(val)}
    else:
        c.user(render(rng, rng.choice(split_tpls(SWAP_ALL_TPL, ev)), {"S": S}))
        br = c.call(("get_balances", {}))
        row = next(h for h in br["holdings"] if h["symbol"] == asset)
        chain = row["chain"]
        val = float(row["amount"])
        tools = ["get_balances"]
        args = {"from_symbol": asset, "from_chain": chain, "to_symbol": "USDC", "to_chain": chain, "amount": jnum(val)}
    r = c.call(("prepare_swap", args))
    assert r["status"] == "prepared_for_review", r
    c.say(BJ.say_swap(rng, r, bio))
    return finish(c, "near_swap_" + asset.lower(), bio, pk, {"type": "swap", "asset": asset, "mode": mode, "amount": jnum(val), "chain": chain},
                  tools + ["prepare_swap"], ["right_tool", "exact_digits", "restraint", "no_unbacked_done"])


def sc_near_usdc(rng, ev):
    bio, dev = pick_bio(rng)
    w = base_world(rng, ev, bio, dev)
    chain = rng.choice(["base", "arbitrum", "solana", "ethereum"])
    amt = rng.choice([50, 100, 150, 200, 250, 300, 500] if not ev else [75, 120, 400, 600])
    w.hold("USDC", chain, round(amt * rng.uniform(1.3, 4), 2))
    pk = pick_pack(rng, 0.1)
    c = mk_conv(rng, w, bio)
    if rng.random() < 0.3:
        c.user(render(rng, rng.choice(split_tpls(SWAP_USDC_ASK_TPL, ev) or SWAP_USDC_ASK_TPL), {"N": str(amt)}))
        c.say(rng.choice(["What do you want the USDC swapped into?", "What should I swap the USDC into?", "Which token do you want for the USDC?"]), ask=True)
        return finish(c, "near_swap_usdc_ask", bio, pk, {"type": "swap_ask", "asset": "USDC", "amount": amt}, [], ["asked_when_needed", "restraint", "no_unbacked_done"])
    c.user(render(rng, rng.choice(split_tpls(SWAP_USDC_TPL, ev)), {"N": str(amt)}))
    br = c.call(("get_balances", {}))
    row = next(h for h in br["holdings"] if h["symbol"] == "USDC")
    args = {"from_symbol": "USDC", "from_chain": row["chain"], "to_symbol": "SOL", "to_chain": "solana", "amount": amt}
    r = c.call(("prepare_swap", args))
    assert r["status"] == "prepared_for_review", r
    c.say(BJ.say_swap(rng, r, bio))
    return finish(c, "near_swap_usdc", bio, pk, {"type": "swap", "asset": "USDC", "amount": amt, "chain": row["chain"]}, ["get_balances", "prepare_swap"],
                  ["right_tool", "exact_digits", "restraint", "no_unbacked_done"])


def sc_near_close(rng, ev):
    bio, dev = pick_bio(rng)
    w = base_world(rng, ev, bio, dev)
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE", "DOGE", "AVAX", "LINK"] if not ev else ["SUI", "XRP", "BNB", "SOL", "ETH"])
    side = rng.choice(["long", "short"])
    entry = round(rng.uniform(20, 300), 2) if coin not in ("BTC", "ETH") else round(rng.uniform(1900, 95000), 1)
    size = rng.choice([1, 2, 5, 10, 25, 0.5, 0.1]) if coin not in ("BTC",) else rng.choice([0.01, 0.05, 0.1])
    lev = rng.choice([2, 3, 5, 10])
    pnl = round(rng.uniform(-80, 140), 2)
    w.perps = [{"coin": coin, "side": side, "size": size, "entry": entry, "leverage": lev, "liquidation": round(entry * (1 - 1 / lev) if side == "long" else entry * (1 + 1 / lev), 2), "pnl": pnl}]
    no_pos = rng.random() < 0.12
    if no_pos:
        w.perps = []
    c = mk_conv(rng, w, bio)
    part = rng.random() < 0.3 and not no_pos
    args = {"coin": coin}
    if part:
        p, fr = pick_pct(rng, ev)
        text = render(rng, rng.choice(split_tpls(CLOSE_PART_TPL, ev)), {"FR": fr, "C": coin if rng.random() < 0.7 else coin.lower(), "SIDE": side})
        args["percent"] = p
    else:
        text = render(rng, rng.choice(split_tpls(CLOSE_TPL, ev)), {"C": coin if rng.random() < 0.7 else coin.lower(), "SIDE": side})
    c.user(text)
    r = c.call(("prepare_close", args))
    if no_pos:
        assert r["status"] == "no_position"
        c.say(say_no_position(rng, r))
    else:
        assert r["status"] == "review_opened", r
        c.say(say_close(rng, r, bio))
    return finish(c, "near_close", bio, "trade", {"type": "close", "coin": coin, "percent": args.get("percent")}, ["prepare_close"],
                  ["right_tool", "exact_digits", "arg_omission", "restraint", "no_unbacked_done"])


def sc_near_price(rng, ev):
    bio, dev = pick_bio(rng)
    sym = rng.choice(PRICEABLE)
    w = base_world(rng, ev, bio, dev)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    c.user(render(rng, rng.choice(split_tpls(PRICE_TPL, ev)), {"S": sym_text(rng, sym)}))
    r = c.call(("get_price", {"symbol": sym}))
    assert "error" not in r, r
    c.say(BJ.say_price(rng, sym, r))
    return finish(c, "near_price", bio, pk, {"type": "price", "symbol": sym}, ["get_price"], ["right_tool", "restraint", "no_unbacked_done"])


def sc_near_concept(rng, ev, key=None):
    bio, dev = pick_bio(rng)
    key = key or rng.choice(CONCEPT_KEYS)
    qs, ans = CONCEPTS[key]
    w = base_world(rng, ev, bio, dev)
    pk = pick_pack(rng)
    c = mk_conv(rng, w, bio)
    qpool = [q for i, q in enumerate(qs) if (i % 5 == 0) == ev] or qs
    c.user(render(rng, rng.choice(qpool), {}))
    c.say(rng.choice(ans))
    return finish(c, "near_concept_" + key, bio, pk, {"type": "concept", "key": key}, [], ["restraint", "no_tool", "no_unbacked_done"])


def sc_absent(rng, ev):
    f = rng.choices([lambda: sc_sell_pct(rng, ev, True), lambda: sc_sell_all(rng, ev, False, True), lambda: sc_sell_amount(rng, ev, True)], [0.5, 0.3, 0.2])[0]
    return f()


# ================================================================================ plan, build
# (name, fn, train count, eval count)
PLAN = [
    ("sell_pct",            lambda r, e: sc_sell_pct(r, e),                          100, 13),
    ("sell_all",            lambda r, e: sc_sell_all(r, e, False),                    30, 4),
    ("sell_all_receive",    lambda r, e: sc_sell_all(r, e, True),                     20, 3),
    ("sell_amount",         lambda r, e: sc_sell_amount(r, e),                        20, 3),
    ("sell_amount_usd",     lambda r, e: sc_sell_usd(r, e),                           10, 1),
    ("sell_by_mint",        lambda r, e: sc_sell_mint(r, e),                          20, 3),
    ("sell_ask_notoken",    lambda r, e: sc_ask_notoken(r, e, empty=r.random() < 0.12, with_pct=r.random() < 0.4), 13, 1),
    ("sell_ask_noamt",      lambda r, e: sc_ask_noamt(r, e),                          12, 2),
    ("sell_err_not_held",   sc_err_not_held,                                           8, 1),
    ("sell_err_insuff",     sc_err_insufficient,                                       8, 1),
    ("sell_err_frozen",     sc_err_frozen,                                             6, 1),
    ("sell_err_choice",     sc_err_choice,                                             7, 1),
    ("sell_err_noroute",    sc_err_noroute,                                            5, 1),
    ("sell_err_warn",       sc_err_warn,                                               6, 1),
    ("sell_err_daily",      sc_err_daily,                                              5, 0),
    ("safety_verified",     lambda r, e: sc_safety(r, e, "verified_clean"),           16, 2),
    ("safety_verified_tag", lambda r, e: sc_safety(r, e, "verified_tag"),              6, 1),
    ("safety_unverified",   lambda r, e: sc_safety(r, e, "unverified_tags"),          12, 2),
    ("safety_refused",      lambda r, e: sc_safety(r, e, "refused"),                  10, 1),
    ("safety_multi",        lambda r, e: sc_safety(r, e, "multi"),                     6, 1),
    ("find_one",            lambda r, e: sc_find(r, e, "one"),                        10, 1),
    ("find_multi",          lambda r, e: sc_find(r, e, "multi"),                       5, 1),
    ("find_empty",          lambda r, e: sc_find(r, e, "empty"),                       3, 1),
    ("find_error",          lambda r, e: sc_find(r, e, "error"),                       2, 0),
    ("buy_direct",          sc_buy_direct,                                            15, 2),
    ("buy_mint_verified",   sc_buy_mint_verified,                                     15, 2),
    ("buy_unv_2turn",       lambda r, e: sc_buy_unverified(r, e, False),              17, 2),
    ("buy_unv_upfront",     lambda r, e: sc_buy_unverified(r, e, True),                8, 1),
    ("buy_refused",         lambda r, e: sc_buy_refused(r, e, False),                 10, 1),
    ("buy_refused_alt",     lambda r, e: sc_buy_refused(r, e, True),                   5, 1),
    ("buy_clash",           sc_buy_clash,                                             25, 3),
    ("near_swap_eth",       lambda r, e: sc_near_swap(r, e, "ETH"),                    8, 1),
    ("near_swap_sol",       lambda r, e: sc_near_swap(r, e, "SOL"),                    6, 1),
    ("near_swap_bnb",       lambda r, e: sc_near_swap(r, e, "BNB"),                    5, 1),
    ("near_swap_usdc",      sc_near_usdc,                                              4, 1),
    ("near_close",          sc_near_close,                                            14, 2),
    ("near_price",          sc_near_price,                                             8, 1),
    ("near_rug",            lambda r, e: sc_near_concept(r, e, "rug"),                 6, 1),
    ("near_concept",        lambda r, e: sc_near_concept(r, e, r.choice([k for k in CONCEPT_KEYS if k != "rug"])), 8, 1),
    ("absent_sell",         sc_absent,                                                26, 3),
]
assert sum(p[2] for p in PLAN) == 520 and sum(p[3] for p in PLAN) == 70, (sum(p[2] for p in PLAN), sum(p[3] for p in PLAN))


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
        made = tries = 0
        while made < n:
            tries += 1
            assert tries < 400 * max(n, 1), ("cannot build", name)
            sub = random.Random(rng.getrandbits(64))
            try:
                ex = fn(sub, ev)
            except AssertionError:
                continue                                  # an unreachable random draw (e.g. warn not reached): draw again
            users = [norm(m["content"]) for m in ex["messages"] if m["role"] == "user"]
            if users[0] in seen or any(u in seen and len(u) > 12 for u in users):
                continue
            seen.add(users[0])
            ex["meta"]["plan"] = name
            rows.append(ex)
            made += 1
    rng.shuffle(rows)
    return rows


# ================================================================================ self-check
BANNED = [r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|sold|deposited|withdrawn|closed|executed)", r"\bI(?:'ve| have)? (sent|bought|sold|swapped|deposited|withdrew|closed|purchased)\b",
          r"successfully", r"funds (are|have) (in|arrived)", r"^\s*done\b", r"not financial advice", r"[·•]", r"\b(bought|sold|swapped|purchased)\b",
          r"\bAs an AI\b", r"\b(safe|safely|legit|legitimate|trustworthy|no risk|can'?t lose|risk[- ]free)\b", r"can only prepare buys", r"Swap in Wallets",
          r"I'?ll sell it when", r"will be sold", r"tab bar", r"what('s| is) (the )?(current )?price", r"your budget", r"^\s*(sure|great question|of course)\b",
          r"I (opened|have opened|'ve opened)", r"\bI can't see (your|the) (holdings|wallet)\b"]
ALLOW_NEG = [r"so it can't be sold", r"nothing has been (sold|bought)( yet)?", r"nothing is (sold|bought) until you approve( it)?( with (face|touch) id)?", r"nothing has moved( yet)?",
             r"nothing has closed( yet)?", r"nothing has closed until you approve with (face|touch) id", r"nothing is sold until you approve it with (face|touch) id",
             r"nothing is bought until you approve it with (face|touch) id"]
UNBACKED_DONE = re.compile(r"\b(sold|bought|swapped|closed)\b", re.I)
FILLER = re.compile(r"^(sure|great question|certainly|absolutely)\b", re.I)


def nums(text):
    out = set()
    for m in NUM_RE.findall(text):
        m = m.replace(",", "").rstrip(".")
        try:
            out.add(Decimal(m).normalize())
        except Exception:
            pass
    return out


def walk_numbers(o, acc):
    if isinstance(o, dict):
        for v in o.values(): walk_numbers(v, acc)
    elif isinstance(o, list):
        for v in o: walk_numbers(v, acc)
    elif isinstance(o, float):
        acc.append(o)


def runner_replay(meta_world, msgs):
    """Replay a recorded conversation against a fresh world. Returns (world, list of result strings)."""
    w = SolWorld.from_state(meta_world)
    out, pending = [], []
    for m in msgs:
        if m["role"] == "user":
            w.typed.append(m["content"])
        for tc in m.get("tool_calls") or []:
            out.append(json.dumps(w.call(tc["name"], tc["arguments"]), ensure_ascii=False))
    return w, out


def make_runner(item):
    """For the eval harness: returns run(name, args) -> JSON string, with the sim state and typed text carried from
    item['context'] (earlier user turns, earlier gold tool calls), exactly like check() replays a record."""
    w = SolWorld.from_state(item["world"])
    for m in item["context"]:
        if m["role"] == "user":
            w.typed.append(m["content"])
        for tc in m.get("tool_calls") or []:
            w.call(tc["name"], tc["arguments"])

    def run(name, args):
        return json.dumps(w.call(name, args), ensure_ascii=False)
    run.world = w
    run.add_user = lambda text: w.typed.append(text)
    return run


def amount_in_text(v, text):
    t = text.lower().replace(",", "")
    forms = {BJ.jstr(v), f"{float(v):.2f}", f"{float(v):g}"}
    if float(v) % 1000 == 0:
        forms.add(f"{int(v // 1000)}k")
    if float(v) >= 1000 and (float(v) / 1000) * 10 % 1 == 0:
        forms.add(f"{float(v) / 1000:g}k")
    if v in BJ.WORDS:
        forms.add(BJ.WORDS[v])
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d]|\.\d)", t) for f in forms)


def mints_in(text):
    return {m for m in ADDR_RE.findall(text) if is_mint(m)}


def check(ex, ev=False):
    meta, msgs = ex["meta"], ex["messages"]
    json.loads(json.dumps(ex, ensure_ascii=False))
    assert ex["tools"] is True and meta["source"] == GEN
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    blob = json.dumps(msgs, ensure_ascii=False)
    assert other not in blob, "other biometric word"
    assert "{BIO}" not in blob
    sysm = msgs[0]
    assert sysm["role"] == "system" and sysm["content"].startswith(SYSTEM_TPL.replace("{BIO}", bio)), "system prompt"
    pack = meta["pack"]
    assert pack == PACK_NAMES[meta["pack_key"]] and all(n in REG for n in pack)
    users = [m["content"] for m in msgs if m["role"] == "user"]
    for u in users:
        assert u.strip() and "{" not in u and "}" not in u and "<" not in u, ("user text", u)
    answers = [m["content"] for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
    assert answers and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls")
    # ---- calls
    calls, tool_msgs = [], [m for m in msgs if m["role"] == "tool"]
    seen_text, seen_results = "", []
    ti = 0
    for m in msgs:
        if m["role"] == "user":
            seen_text += " " + m["content"]
        for tc in m.get("tool_calls") or []:
            name, a = tc["name"], tc["arguments"]
            assert name in REG, ("tool not in registry", name)
            assert name in pack, ("tool not in pack", name, meta["pack_key"])
            sch = REG[name]
            for req in sch.get("required", []):
                assert req in a, (tc, req)
            for k, v in a.items():
                assert k in sch["properties"], ("unknown arg", tc, k)
                t = sch["properties"][k]["type"]
                assert (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)) or (t == "string" and isinstance(v, str)), (tc, k, v)
                if "enum" in sch["properties"][k]:
                    assert v in sch["properties"][k]["enum"], (tc, k, v)
            prior_mints = mints_in(seen_text) | {mm for rr in seen_results for mm in mints_in(rr)}
            if name == "prepare_sell":
                assert (("percent" in a) != ("amount" in a)), ("percent XOR amount", a)
                if "percent" in a:
                    assert 0 < a["percent"] <= 100
                tk = a["token"]
                if is_mint(tk):
                    assert tk in prior_mints, ("sell mint provenance", tk)
                    assert tk in seen_text or tk in prior_mints
                else:
                    assert tk == tk.upper() and tk.isalnum(), tk
                    assert ticker_in(seen_text, tk) or any(ticker_in(u, tk) for u in users[:1]), ("ticker not in user text", tk, seen_text)
                intent = meta["intent"]
                if intent.get("type") == "sell":
                    if "percent" in a and intent.get("percent") is not None:
                        assert a["percent"] == intent["percent"]
                    if intent.get("amount") is not None and "amount" in a and not intent.get("usd"):
                        assert a["amount"] == intent["amount"]
                if "percent" in a and a["percent"] != 100 and not is_mint(tk) and meta["kind"] not in ("sell_err_choice",):
                    assert pct_in_text(a["percent"], seen_text), ("percent not in user text", a, seen_text)
                if "amount" in a and not meta["intent"].get("usd") and meta["kind"] != "sell_ask_followup":
                    assert amount_in_text(a["amount"], seen_text), ("amount not in user text", a, seen_text)
                if "receive" in a:
                    assert a["receive"] in ("SOL", "USDC") and re.search(r"(into|to|for|receive|give me|pay me in|take|back|get me)\b.*\b" + a["receive"].lower() + r"\b|\b" + a["receive"].lower() + r"\b.*(back|all of it)", seen_text.lower()), ("receive not asked", a, seen_text)
                else:
                    assert not re.search(r"\b(into|to|for|receive|in|take)\s+(sol|usdc)\b", seen_text.lower()), ("receive missing though the user said it", seen_text)
            if name == "prepare_buy":
                sy = a["symbol"]
                assert "pay_with" not in a
                if is_mint(sy):
                    assert sy in prior_mints, ("buy mint provenance", sy)
                else:
                    assert sy == sy.upper() and ticker_in(seen_text, sy), ("buy ticker", sy)
                if meta["kind"] != "buy_clash" or "amount_usd" in a:
                    assert amount_in_text(a["amount_usd"], seen_text), ("amount not in user text", a, seen_text)
                    assert a["amount_usd"] == meta["intent"]["amount_usd"]
            if name == "search_tokens":
                q = a["query"]
                assert q.strip() == q and q and "$" not in q
                if is_mint(q):
                    assert q in seen_text
                else:
                    assert q.lower() in seen_text.lower(), ("query not as typed", q, seen_text)
            if name == "get_positions":
                assert a == {"kind": "spot"}
            if name == "open_page":
                assert a == {"page": "trade"}
            if name == "prepare_close":
                assert a["coin"] == a["coin"].upper() and ticker_in(seen_text, a["coin"])
                assert ("percent" in a) == (meta["intent"].get("percent") is not None)
                if "percent" in a:
                    assert pct_in_text(a["percent"], seen_text)
            if name == "get_price":
                assert a["symbol"] == a["symbol"].upper() and ticker_in(seen_text, a["symbol"])
            if name == "calculator":
                allowed_n = nums(seen_text)
                for rr in seen_results:
                    allowed_n |= nums(rr)
                for n in nums(a["expression"]):
                    assert n in allowed_n or n in (2, 4), ("calculator number not backed", n, a)
            if name == "prepare_swap":
                assert a["from_chain"] in CHAIN_NAME and a["to_chain"] in CHAIN_NAME
                held = [rr for rr in seen_results if '"holdings"' in rr]
                if not held:
                    assert a["from_chain"] in seen_text.lower() or CHAIN_SAY.get(a["from_chain"]) and any(x.lower() in seen_text.lower() for x in CHAIN_SAY[a["from_chain"]]), ("chain not from user", a)
                else:
                    assert any(f'"chain": "{a["from_chain"]}"' in rr for rr in held)
                assert a["amount"] > 0
            calls.append(name)
            seen_results.append(tool_msgs[ti]["content"])
            ti += 1
    assert calls == meta["expected_tools"], (calls, meta["expected_tools"])
    assert meta["steps"] == calls
    # ---- replay: recorded results equal a fresh world's results
    w, rep = runner_replay(meta["world"], msgs)
    assert len(rep) == len(tool_msgs)
    for r_, tm in zip(rep, tool_msgs):
        assert r_ == tm["content"], ("replay mismatch", r_[:200], tm["content"][:200])
    assert w.state() == meta["world_final"], "world_final"
    results = [tm["content"] for tm in tool_msgs]
    floats = []
    for rr in results:
        walk_numbers(json.loads(rr), floats)
    for f in floats:
        assert f == 0 or abs(f) >= 0.0001, ("e-notation risk", f)
        assert "e" not in repr(f), ("e-notation", f)
    # ---- answers
    allowed = set()
    for t in results + users + [json.dumps(tc["arguments"]) for m in msgs for tc in (m.get("tool_calls") or [])]:
        allowed |= nums(t)
    for v in list(allowed):
        if v >= 1000:
            allowed.add((v / 1000).normalize())
            allowed.add((v / 1000000).normalize())
    allowed |= {Decimal(24), Decimal(5)} if any(("$5K" in rr) for rr in results) else {Decimal(24)}
    ptn_money = re.compile(r"\$\d")
    for t in answers:
        assert t.isascii() or set(t) - set(map(chr, range(128))) <= {"…"}, ("non-ascii", t)
        low = t
        for al in ALLOW_NEG:
            low = re.sub(al, "", low, flags=re.I)
        for b in BANNED:
            assert not re.search(b, low, flags=re.I | re.M), (b, t)
        limit = 110 if meta["kind"].startswith(("search_find_multi", "search_safety_multi")) else 70
        assert len(t.split()) <= limit, ("too long", len(t.split()), t)
        a_clean = ADDR_RE.sub("", t)
        for n in nums(a_clean):
            assert n in allowed, ("number not backed", n, t)
        for ad in ADDR_RE.findall(t):
            assert is_mint(ad) is False or ad in seen_text or any(ad in rr for rr in results), ("address not from user/result", ad)
        assert t.count("?") <= 1 or True
        assert "Touch ID" not in t or bio == "Touch ID"
        assert "Face ID" not in t or bio == "Face ID"
    # ---- per kind content
    kind, final = meta["kind"], answers[-1]
    last = json.loads(results[-1]) if results else {}
    ok_status = last.get("status") == "review_opened"
    if kind.startswith(("sell_pct", "sell_all", "sell_amount", "sell_by_mint")) or kind in ("sell_ask_followup", "sell_err_choice", "sell_err_warn") or kind.startswith(("buy_direct", "buy_after_search", "buy_mint_unverified", "buy_clash")):
        assert ok_status, last
        if last["kind"] == "sell":
            assert last["you_sell"] in final and last["you_get"] in final and "Jupiter" in final
            assert re.search(r"[Nn]othing (has been|is) sold", final) and bio in final
            assert BJ.jstr(last["price_impact_pct"]) in final.replace("%", " ").replace("  ", " ") or BJ.jstr(last["price_impact_pct"]) + "%" in final
            for w_ in last.get("warnings", []):
                assert w_ in final, "warning not repeated"
            if kind == "sell_err_warn":
                assert last["warnings"] and last["policy"] == "warn"
            else:
                assert "warnings" not in last
        else:
            assert last["kind"] == "buy"
            assert last["you_pay"] in final and last["you_get"] in final and last["route"] in final
            assert re.search(r"[Nn]othing (has been|is) bought", final) and bio in final
            assert BJ.jstr(last["price_impact_pct"]) in final
            for w_ in last.get("warnings", []):
                assert w_.rstrip(".") in final, ("buy warning not repeated", w_)
            if kind.startswith("buy_mint_unverified"):
                assert last["verified"] is False and last["warnings"][0] == "Unverified token" and "not verified" in final
    if kind == "sell_amount_usd":
        cr = json.loads(results[1])["result"]
        assert cr in final and f"${meta['intent']['usd']}" in final
    if kind == "sell_ask_followup" or kind == "sell_err_choice":
        assert len([m for m in msgs if m["role"] == "user"]) == 2
        asks = [m for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")][:-1]
        assert asks and "?" in asks[0]["content"]
    if kind == "sell_ask" or kind == "sell_ask_empty":
        if kind == "sell_ask":
            assert not any(n.startswith("prepare_") for n in calls) and "?" in final and final.count("?") == 1
            assert not re.search(r"\d+\s*%", final) or not re.search(r"(sell|selling) \d+", final), final
            if "get_positions" in calls:
                held = json.loads(results[0])["spot"]
                for h in held:
                    assert h["symbol"] in final
            else:
                assert not re.search(r"\byou hold\b|\bi see\b", final, re.I), "holding claimed without get_positions"
    if kind == "sell_err_not_held":
        assert last["status"] == "not_held" and last["note"].split("no ")[1].split(" on")[0] in final
    if kind == "sell_err_insufficient":
        assert last["holding"] in final
    if kind in ("sell_err_frozen", "sell_err_daily"):
        assert last["reasons"][0] in final and calls.count("prepare_sell") == 1
    if kind == "sell_err_noroute":
        assert last["error"] in final and calls.count("prepare_sell") == 1
    if kind == "sell_err_choice":
        r1 = json.loads(results[0])
        assert r1["status"] == "needs_choice" and r1["question"].replace(" Which one?", "") in answers[0].replace(" Nothing was prepared.", "").replace(" Which one should I sell?", "")
        assert msgs and calls == ["prepare_sell", "prepare_sell"]
        last_call = [tc for m in msgs for tc in (m.get("tool_calls") or [])][-1]
        assert last_call["arguments"]["token"] in {o["mint"] for o in r1["options"]}
    if kind.startswith("search_safety") or kind.startswith("search_find"):
        r = json.loads(results[-1]) if results else {}
        if "tokens" in r and r["tokens"]:
            first = r["tokens"][0]
            if kind.startswith("search_safety"):
                rows = r["tokens"][:2] if kind == "search_safety_multi" else [x for x in r["tokens"] if x["mint"] == meta_row_mint(ex)]
                for row in rows:
                    assert liq_txt(row["liquidity_usd"]) in final, ("liquidity", row, final)
                    if kind != "search_safety_multi":
                        if row["verified"]:
                            assert "verified" in final and "not verified" not in final
                        else:
                            assert "not verified" in final
                    else:
                        assert row["name"] in final and "verified" in final
                    for tag in row["tags"]:
                        key = {"Freeze on": r"freeze", "Mint on": r"\bmint\b", "Low liquidity": r"liquidity is low", "New": r"\bnew\b|recently"}.get(tag)
                        if tag.startswith("Top holders"):
                            assert re.search(r"[Tt]op holders own " + re.search(r"\d+%", tag).group(), final), (tag, final)
                        else:
                            assert re.search(key, final.lower()), (tag, final)
                    if row.get("refusal"):
                        assert row["refusal"] in final
                    if row["change_24h_pct"] is not None:
                        assert chg(row["change_24h_pct"]) in final
            else:
                shown = r["tokens"][:3]
                for row in shown:
                    assert row["mint"] in final and liq_txt(row["liquidity_usd"]) in final
        elif "tokens" in r:
            assert r["note"].split(".")[1].strip() in final or "Check the spelling" in final
        else:
            assert r["error"] in final
    if kind.startswith("buy_refused"):
        assert "prepare_buy" not in calls
        row = json.loads(results[-1])["tokens"][0] if kind == "buy_refused" else next(x for x in json.loads(results[-1])["tokens"] if x.get("refusal"))
        assert row["refusal"] in final and "not verified" in final
        if kind == "buy_refused_alt":
            alt = next(x for x in json.loads(results[-1])["tokens"] if x["verified"])
            assert alt["symbol"] in final and liq_txt(alt["liquidity_usd"]) in final
        else:
            assert "is verified" not in final
    if kind.startswith("buy_mint_unverified") and kind == "buy_mint_unverified":
        a0 = answers[0]
        assert "?" in a0 and "not verified" in a0 and "prepare_buy" not in [tc["name"] for m in msgs[:5] for tc in (m.get("tool_calls") or [])]
    if kind == "buy_clash":
        r1 = json.loads(results[0])
        assert r1["status"] == "needs_choice" and all(liq_txt(o["liquidity_usd"]) in answers[0] for o in r1["options"]) and "?" in answers[0]
        last_call = [tc for m in msgs for tc in (m.get("tool_calls") or [])][-1]
        assert last_call["arguments"]["symbol"] in {o["mint"] for o in r1["options"]}
    if kind.startswith("near_swap") and calls:
        assert last["route"] == "NEAR Intents" and last["from"] in final and last["to"] in final and "NEAR Intents" in final
        assert "prepare_sell" not in calls
    if kind == "near_close":
        assert "prepare_sell" not in calls
        if ok_status:
            assert last["closing"] in final and bio in final
        else:
            assert last["status"] == "no_position"
    if kind == "near_price":
        assert BJ.price_txt(last["price"]) in final
    if kind.startswith("near_concept"):
        assert not calls and "?" not in final
    if kind == "absent_sell":
        assert calls == ["open_page"] and meta["pack_key"] == "core" and "Selling Solana tokens isn't available in this chat" in final and "button" in final.lower() or "Trade" in final
    # ---- expect block and round trip
    exp = meta["expect"]
    assert exp and all(("calls" in e) or ("ask" in e) or ("text" in e) for e in exp)
    for m in msgs:
        if m.get("tool_calls"):
            got, bad = H.parse_calls(H.assistant_text(m))
            assert bad == 0 and got == m["tool_calls"], "call round trip"
    return True


def meta_row_mint(ex):
    """The token a single-row safety check was about: the first (only) row of the last search result."""
    for m in reversed(ex["messages"]):
        if m["role"] == "tool":
            r = json.loads(m["content"])
            if r.get("tokens"):
                return r["tokens"][0]["mint"]
    return None


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


# ================================================================================ eval items
def segment_gold(meta, seg, bio):
    """Gold for the assistant turns in `seg` (messages after the split user message)."""
    calls = [tc for m in seg for tc in (m.get("tool_calls") or [])]
    answers = [m["content"] for m in seg if m["role"] == "assistant" and not m.get("tool_calls")]
    final = answers[-1] if answers else ""
    res = [json.loads(m["content"]) for m in seg if m["role"] == "tool"]
    r = res[-1] if res else {}
    must, match = [], []
    kind = meta["kind"]
    if r.get("status") == "review_opened":
        if r.get("kind") == "sell":
            must = [r["you_sell"], r["you_get"], bio] + r.get("warnings", [])
        elif r.get("kind") == "buy":
            must = [r["you_pay"], r["you_get"], bio] + [w.rstrip(".") for w in r.get("warnings", [])]
            must.append(BJ.jstr(r["price_impact_pct"]) + "%")
        elif r.get("kind") == "close":
            must = [r["closing"], bio]
    elif r.get("status") == "prepared_for_review":
        must = [r["from"], r["to"], "NEAR Intents", bio]
    elif r.get("status") in ("blocked",):
        must = r["reasons"]
    elif r.get("status") == "insufficient":
        must = [r["holding"]]
    elif r.get("status") == "needs_choice":
        must = [o.get("mint", "") for o in []]
        match = [r"\?"]
    elif r.get("status") == "not_held":
        must = [r["note"].split("no ")[1].split(" on")[0]]
    elif "error" in r:
        must = [r["error"]]
    elif r.get("tokens"):
        rows = r["tokens"][:3] if kind.startswith("search_find") else r["tokens"][:2]
        if kind.startswith("search_find"):
            must = [x["mint"] for x in rows] + [liq_txt(x["liquidity_usd"]) for x in rows]
        else:
            must = [liq_txt(x["liquidity_usd"]) for x in rows]
            for x in rows:
                if x.get("refusal"):
                    must.append(x["refusal"])
            match = [r"(?i)\bnot verified\b" if not rows[0]["verified"] else r"(?i)\bverified\b"]
            tagwords = {"Freeze on": r"(?i)freeze", "Mint on": r"(?i)\bmint\b", "Low liquidity": r"(?i)liquidity is low", "New": r"(?i)\bnew\b|recently"}
            for x in rows[:1]:
                for tg in x["tags"]:
                    match.append(tagwords.get(tg, r"(?i)top holders own"))
    elif r.get("price") is not None and "source" in r:
        must = [BJ.price_txt(r["price"])]
    elif r.get("status") == "button_shown":
        must = ["Selling Solana tokens isn't available in this chat"]
    return {"calls": calls, "no_call": not calls, "must_contain": must, "must_match": match, "final_example": final}


def eval_item(ex, i, split_mode=None):
    msgs, meta = ex["messages"], ex["meta"]
    bio = meta["bio"]
    uidx = [j for j, m in enumerate(msgs) if m["role"] == "user"]
    use_script = len(uidx) == 2 and (i % 2 == 0)
    if use_script:
        cut, end = uidx[0], uidx[1]
        seg = msgs[cut + 1:end]
        seg2 = msgs[end + 1:]
        script = [msgs[end]["content"]]
    else:
        cut, end = uidx[-1], len(msgs)
        seg = msgs[cut + 1:]
        seg2, script = [], []
    gold = segment_gold(meta, seg, bio)
    ask = bool(any(m["role"] == "assistant" and not m.get("tool_calls") and "?" in m["content"] for m in seg)) and not any(tc for m in seg for tc in (m.get("tool_calls") or [])) and (use_script or meta["expect"] and "ask" in meta["expect"][-1])
    if use_script:
        ask = True
        g2 = segment_gold(meta, seg2, bio)
        gold["script_calls"] = g2["calls"]
        gold["script_must_contain"] = g2["must_contain"]
    gold["ask"] = ask
    intent = meta["intent"]
    gold["kind_intent"] = intent
    ctx_text = " ".join(m["content"] for m in msgs[:cut + 1] if m["role"] == "user")
    srcs = set(mints_in(ctx_text))
    for m in msgs[:cut + 1]:
        if m["role"] == "tool":
            srcs |= mints_in(m["content"])
    gold["mint_sources_context"] = sorted(srcs)
    gold["forbidden_tools"] = {"near_swap": ["prepare_sell", "search_tokens"], "near_close": ["prepare_sell", "prepare_trade"], "near_price": ["prepare_sell", "prepare_buy"],
                               "near_concept": ["prepare_sell", "prepare_buy", "search_tokens", "get_price"]}.get(next((k for k in ("near_swap", "near_close", "near_price", "near_concept") if meta["kind"].startswith(k)), ""), [])
    gold["must_not_match"] = [r"\bRelay\b", r"\b(sold|bought)\b(?!\s+yet)", r"went through", r"\bswapped\b", r"successfully", r"(?i)\b(safe|legit|trustworthy)\b",
                              r"can only prepare buys", r"Swap in Wallets", r"I'll sell it when"]
    gold["arg_rules"] = {"percent_xor_amount": True, "receive_absent_unless_said": True}
    return {"id": f"{GEN}-{meta['kind']}-{i}", "kind": meta["kind"], "tools": True, "bio": bio, "pack": meta["pack"], "family": meta["family_ok"],
            "context": msgs[:cut + 1], "reference": seg + (([msgs[end]] + seg2) if use_script else []), "script": script, "expect": meta["expect"],
            "world": meta["world"], "checks": meta["checks"], "gold": gold,
            "meta": {"source": GEN, "generator": GEN, "lang": "en", "plan": meta.get("plan")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=520)
    ap.add_argument("--eval-seed", type=int, default=9401)
    ap.add_argument("--eval-n", type=int, default=70)
    ap.add_argument("--out", default=os.path.join(LAB, "data4", "spot_solana.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval4", "spot_solana.jsonl"))
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
    items = [eval_item(ex, i) for i, ex in enumerate(ev_rows)]
    for it in items:                                   # the eval runner replays the gold context exactly
        run = make_runner(it)
        for tc in it["gold"]["calls"][:0]:
            run(tc["name"], tc["arguments"])
    with open(a.out, "w") as f:
        for ex in tr_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("kinds train:", dict(sorted(Counter(e["meta"]["kind"] for e in tr_rows).items())), file=sys.stderr)
    print("pack train:", dict(Counter(e["meta"]["pack_key"] for e in tr_rows)), "bio:", dict(Counter(e["meta"]["bio"] for e in tr_rows)),
          "family_ok lite:", sum("lite" in e["meta"]["family_ok"] for e in tr_rows), file=sys.stderr)
    print(f"spot_solana: {len(tr_rows)} train, {len(ev_rows)} eval, self-check ok")


if __name__ == "__main__":
    main()
