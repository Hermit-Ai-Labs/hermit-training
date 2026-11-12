#!/usr/bin/env python3
"""multistep: multi-step requests for the next Hermit run (suite chain_multi, PLAN_v3 section 2.6, GAPS_behaviour P0-1).

Writes (one run, both files):
  data4/multistep.jsonl   train rows, --seed (default 4), --n (default 700)
  eval4/multistep.jsonl   held-out eval items, --eval-seed (default 9404), --eval-n (default 120)

Money sequences (about 80% of train, 60 eval items) teach: the app opens ONE review at a time and the model never sees
approvals, so a dependent plan is prepared one step at a time. Step 1 only; the answer names step 2 as next and never
narrates it as done. On "it landed" the model reads get_activity{limit:5} and get_balances in ONE message; a confirmed row
for step 1 plus the balance string size step 2 (calculator for fractions). in_progress: say the quoted seconds and stop.
Blocked/failed step 1: step 2 is not prepared. Something only the user knows (Alice's address): ask once, after the steps
that need nothing. The quote's amount_out is never the step 2 amount.
Read chains (140 train, 60 eval): balances -> quote -> prepare_swap; account + price + calculator -> prepare_trade;
parallel price reads; yields -> calculator -> prepare_deposit; account -> partial prepare_close; search_markets ->
annualised funding; 4- and 5-round chains; a result that changes the next step; the redundant-read trap.

Nothing is LLM written. Every assistant text is a Python f-string over tool results, user text and calculator results; the
simulator below returns exactly the shapes of spec/hermit_tools_v3.meta.json (sorted keys, Python-style JSON, rounded
amounts like the app's Units.approx). Record shape = gen3: {messages, tools, meta}; eval items add context/reference/
expect/world/checks/answer_checks plus `script` (later user turns) and `pack`.

Held-out rule: every phrase pool is split by index % 5 (== 0 only in eval); eval slot values come from the eval seed; no
normalised user text of eval occurs in train.

Run:  python gen4/multistep.py
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, OrderedDict
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP

sys.dont_write_bytecode = True          # never write .pyc next to gen2/gen3
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(HERE, ".."))
SPEC = os.path.join(TRAIN_DIR, "spec")
for p in (os.path.join(TRAIN_DIR, "gen2"), os.path.join(TRAIN_DIR, "gen3")):
    sys.path.insert(0, p)
import hermit_fmt as H                  # noqa: E402  (gen2, read only)

TOOLS_V3 = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
META_V3 = json.load(open(os.path.join(SPEC, "hermit_tools_v3.meta.json")))
SYSTEM_V3 = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert "{BIO}" in SYSTEM_V3
TOOL_NAMES = [t["function"]["name"] for t in TOOLS_V3]
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V3}
assert len(TOOLS_V3) == 30
# render with the v3 registry (runtime patch of the module globals; gen2 files stay untouched)
H.TOOLS = TOOLS_V3
H.TOOL_NAMES = TOOL_NAMES
H.TOOL_SCHEMA = TOOL_SCHEMA
H.TOOLS_BLOCK = H.tools_block()

SRC = "multistep"
ROUTE = "NEAR Intents"
WALLET_NOTE = "Reviews the user has not approved yet do not appear here."

# packs (APP_TOOLS_v3.md section 4)
PACK_CORE = ["calculator", "get_price", "get_balances", "get_portfolio", "get_address", "get_activity", "prepare_send",
             "prepare_swap", "get_quote", "open_page", "open_app", "open_add_funds"]
PACK_TRADE = ["hyperliquid_account", "position_math", "search_markets", "prepare_trade", "prepare_close", "cancel_order",
              "set_tp_sl", "get_positions"]
PACK_SOLANA = ["search_tokens", "prepare_buy", "prepare_sell", "get_positions"]
PACK_EARN = ["get_yields", "prepare_deposit", "prepare_withdraw", "get_positions"]
PACK_PAY = ["prepare_multisend", "add_contact", "schedule_payment", "create_automation"]
PACKS = OrderedDict([("trade", PACK_TRADE), ("solana", PACK_SOLANA), ("earn", PACK_EARN), ("pay", PACK_PAY)])
MONEY_TOOLS = {"prepare_send", "prepare_swap", "prepare_deposit", "prepare_withdraw", "prepare_sell", "prepare_close",
               "prepare_trade", "prepare_multisend", "prepare_buy", "set_tp_sl"}
READ_TOOLS = {"calculator", "get_price", "get_balances", "hyperliquid_account", "search_markets", "get_yields", "get_quote",
              "get_activity", "get_address", "get_positions", "get_portfolio", "search_tokens"}


def pack_for(used):
    """Tool names the app would send for a conversation that uses these tools (core + the packs they need)."""
    names = set(PACK_CORE) | set(used)
    packs = []
    for pk, lst in PACKS.items():
        if any(u in lst and u not in PACK_CORE for u in used if not (u == "get_positions")):
            packs.append(pk)
    if "get_positions" in used and not any(pk in packs for pk in ("trade", "solana", "earn")):
        packs.append("earn")
    allowed = set(PACK_CORE)
    for pk in packs:
        allowed |= set(PACKS[pk])
    assert names <= allowed, (names - allowed)
    return [n for n in TOOL_NAMES if n in allowed], packs


# ------------------------------------------------------------------ small helpers
def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def D(x):
    return x if isinstance(x, Decimal) else Decimal(str(x))


def dstr(x):
    """Shortest plain decimal text of a number or Decimal (no exponent, no trailing zeros)."""
    d = D(x) if not isinstance(x, float) else Decimal(repr(x))
    s = format(d.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def jn(x):
    """JSON number the way the app prints it: whole numbers stay ints, floats are shortest round trip."""
    if isinstance(x, Decimal):
        x = float(x)
    x = float(x)
    return int(x) if x == int(x) else x


def num_json(x):
    """Number for a tool-call argument: int when whole, else float of the shortest decimal."""
    d = D(x)
    return int(d) if d == d.to_integral_value() else float(d)


def approx(v):
    """HermitWallet Units.approx: >= 1000 -> 2 dp, else 3 - floor(log10 a) decimals; trailing zeros trimmed."""
    import math
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    dec = 2 if a >= 1000 else min(18, max(0, 3 - int(math.floor(math.log10(a)))))
    s = f"{v:.{dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def clean(o):
    """ToolResult.clean for generated results: whole floats become ints; key order kept (Lit) or sorted at dump time."""
    if isinstance(o, Lit):
        return Lit((k, clean(v)) for k, v in o.items())
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, Decimal):
        return jn(o)
    if isinstance(o, float):
        return jn(o)
    return o


class Lit(OrderedDict):
    """A result whose keys keep the app's literal order (prepared strings), not sorted."""


def dumps_result(res, bio="Face ID"):
    s = json.dumps(res, ensure_ascii=False, sort_keys=not isinstance(res, Lit))
    return s.replace("Face ID", bio)


CHAIN_NAME = {"ethereum": "Ethereum", "base": "Base", "arbitrum": "Arbitrum", "solana": "Solana", "polygon": "Polygon",
              "bnb": "BNB Chain"}
CHAIN_RE = {"ethereum": r"\b(ethereum|mainnet|eth mainnet)\b", "base": r"\bbase\b", "arbitrum": r"\b(arbitrum|arb)\b",
            "solana": r"\bsolana\b", "polygon": r"\bpolygon\b", "bnb": r"\b(bnb chain|bsc|bnb)\b"}
AAVE_CHAINS = ["ethereum", "base", "arbitrum"]
AAVE_SUPPORT = {"ethereum": ["USDC", "USDT", "ETH", "DAI"], "base": ["USDC", "ETH"], "arbitrum": ["USDC", "USDT", "ETH"]}
STABLES = {"USDC", "USDT", "DAI"}
TOKEN_CHAINS = {"USDC": ["ethereum", "base", "arbitrum", "polygon", "solana"], "USDT": ["ethereum", "arbitrum", "polygon"],
                "ETH": ["ethereum", "base", "arbitrum"], "SOL": ["solana"], "DAI": ["ethereum"]}
QUOTE_CHAINS = {"ethereum", "base", "arbitrum", "polygon", "solana", "bnb"}
PRICE_BAND = {"ETH": (2300, 4300), "SOL": (95, 260), "BTC": (62000, 112000), "HYPE": (18, 55), "BNB": (480, 880),
              "AVAX": (18, 52), "LINK": (11, 27), "DOGE": (0.11, 0.34), "SUI": (2.0, 5.2), "XRP": (1.7, 3.6),
              "WIF": (0.45, 2.4), "JUP": (0.45, 1.3), "POPCAT": (0.3, 1.1), "PUMP": (0.0021, 0.0095),
              "PENGU": (0.0052, 0.021), "MOODENG": (0.12, 0.45), "FARTCOIN": (0.2, 0.9), "GOAT": (0.06, 0.3)}
SPOT_TOKENS = ["WIF", "JUP", "POPCAT", "PUMP", "PENGU", "MOODENG", "FARTCOIN", "GOAT"]
PERP_MAX_LEV = {"BTC": 40, "ETH": 25, "SOL": 20, "HYPE": 10, "BNB": 10, "AVAX": 10, "LINK": 10, "DOGE": 10, "SUI": 10, "XRP": 20}
FUND_NAMES = ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "LINK", "DOGE", "SUI", "XRP"]


def pretty_hold(amount_str, sym, chain):
    return f"{amount_str} {sym} on {CHAIN_NAME[chain]}"


def named_chain(chain, typed):
    return re.search(CHAIN_RE[chain], typed, re.I) is not None


def addr_family(a):
    if a.startswith("0x") and len(a) == 42:
        return "evm"
    if a.endswith(".near"):
        return "near"
    return "solana"


# ------------------------------------------------------------------ simulator
class World:
    """One user's state. Static fields live in self.s (a JSON dict); typed text and resolved addresses are dynamic."""

    def __init__(self, st):
        self.s = json.loads(json.dumps(st))
        self.typed = []
        self.resolved = set()

    @classmethod
    def from_state(cls, st, typed=None, resolved=None):
        w = cls(st)
        w.typed = list(typed or [])
        w.resolved = set(resolved or [])
        return w

    def state(self):
        return json.loads(json.dumps(self.s))

    # ---- holdings
    def hold(self, sym, chain):
        for h in self.s["holdings"]:
            if h["symbol"] == sym and h["chain"] == chain:
                return D(h["amount"])
        return Decimal(0)

    def price(self, sym):
        if sym in STABLES:
            return 1.0
        return self.s["prices"].get(sym, 1.0)

    def set_hold(self, sym, chain, amount):
        amount = D(amount)
        hs = [h for h in self.s["holdings"] if not (h["symbol"] == sym and h["chain"] == chain)]
        if amount > 0:
            hs.append({"amount": dstr(amount), "chain": chain, "symbol": sym,
                       "usd": round(float(amount) * self.price(sym), 2)})
        order = {"USDC": 0, "USDT": 1, "ETH": 2, "SOL": 3}
        hs.sort(key=lambda h: (list(CHAIN_NAME).index(h["chain"]), order.get(h["symbol"], 9), h["symbol"]))
        self.s["holdings"] = hs

    def add_hold(self, sym, chain, delta):
        self.set_hold(sym, chain, self.hold(sym, chain) + D(delta))

    def aave_amt(self, sym, chain):
        for a in self.s["aave"]:
            if a["symbol"] == sym and a["chain"] == chain:
                return D(a["amount"])
        return Decimal(0)

    def set_aave(self, sym, chain, amount):
        amount = D(amount)
        rest = [a for a in self.s["aave"] if not (a["symbol"] == sym and a["chain"] == chain)]
        if amount > 0:
            rest.append({"amount": dstr(amount), "apy_pct": self.apy(chain, sym), "chain": chain, "symbol": sym})
        self.s["aave"] = rest

    def apy(self, chain, sym):
        return self.s["apy"].get(f"{chain}|{sym}", 3.9)

    def add_activity(self, what, status, time):
        self.s["activity"].insert(0, {"status": status, "time": time, "what": what})

    def set_activity_status(self, what, status):
        for r in self.s["activity"]:
            if r["what"] == what:
                r["status"] = status
                return
        raise KeyError(what)

    # ---- quotes (deterministic in the arguments and the world's fee table)
    def quote(self, fs, fc, ts, tc, amount):
        if fc not in QUOTE_CHAINS or tc not in QUOTE_CHAINS:
            return {"error": "That network isn't supported for quotes."}
        usd_in = float(amount) * self.price(fs)
        if usd_in < 1:
            return {"error": "Amount is below the minimum for this route (about $1)."}
        bps = self.s["fee_bps"]
        spread = self.s["spread"] if fc != tc else self.s["spread"] / 2
        cost = round(usd_in * bps / 10000 + spread, 2)
        usd_out = round(usd_in - cost, 2)
        out_units = usd_out / self.price(ts)
        out = approx(out_units)
        return {"amount_in_usd": round(usd_in, 2), "amount_out": out, "amount_out_usd": usd_out, "fee_bps": bps,
                "from": f"{dstr(amount)} {fs} on {CHAIN_NAME[fc]}", "route": ROUTE, "seconds": self.s["seconds"],
                "signature_verified": True, "to": f"about {out} {ts} on {CHAIN_NAME[tc]}", "total_cost_usd": cost,
                "_out_units": out_units}

    # ---- dispatch
    def call(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        if not isinstance(args, dict):
            return {"error": "bad arguments"}
        forced = self.s.get("force", {}).get(name)
        if forced is not None and name in MONEY_TOOLS:
            return clean(forced)
        try:
            r = fn(**args)
        except TypeError as e:
            r = {"error": f"bad arguments: {e}"}
        if isinstance(r, dict):
            r = {k: v for k, v in r.items() if not k.startswith("_")} if not isinstance(r, Lit) else r
        return clean(r)

    # ---- reads
    def t_calculator(self, expression):
        try:
            return {"result": H.calc_str(H.calc(expression))}
        except Exception:
            return {"error": "Couldn't evaluate that expression"}

    def t_get_price(self, symbol):
        sym = symbol.upper()
        if sym in STABLES:
            return {"change_24h_pct": 0.0, "price": 1.0, "source": "Hyperliquid mark price", "symbol": sym}
        if sym not in self.s["prices"]:
            return {"error": f"No Hyperliquid market for {sym}. Answer from what you know instead, and say there's no live price here."}
        return {"change_24h_pct": self.s["change"].get(sym), "price": self.s["prices"][sym],
                "source": "Hyperliquid mark price", "symbol": sym}

    def t_get_balances(self, chain=None):
        hs = self.s["holdings"]
        if chain:
            hs = [h for h in hs if h["chain"] == chain]
        r = {"holdings": [dict(h) for h in hs]}
        if self.s.get("near"):
            r["near_account"] = self.s["near"]
        return r

    def t_hyperliquid_account(self):
        hl = self.s["hl"]
        pos = []
        for p in hl["positions"]:
            q = dict(p)
            mark = self.s["prices"][p["coin"]]
            sgn = 1 if p["side"] == "long" else -1
            q["unrealised_pnl"] = round((mark - p["entry"]) * p["size"] * sgn, 2)
            pos.append(q)
        margin = round(sum(p["size"] * p["entry"] / p["leverage"] for p in hl["positions"]), 2)
        upnl = sum(q["unrealised_pnl"] for q in pos)
        return {"account_value": round(hl["available"] + margin + upnl, 2), "available_usdc": hl["available"],
                "margin_used": margin, "open_orders": hl["orders"], "positions": pos}

    def t_search_markets(self, query):
        q = query.upper()
        rows = [m for m in self.s["markets"] if m["name"].split("-")[0] in q.replace("/", " ").split()
                or m["name"].split("-")[0] in re.findall(r"[A-Z0-9]+", q)]
        return {"markets": rows}

    def t_get_yields(self, asset=None, chain=None):
        rows = self.s["yields"]
        if asset:
            rows = [r for r in rows if r["asset"].upper() == asset.upper()]
        if chain:
            rows = [r for r in rows if r["chain"].lower() == chain.lower()]
        return {"pools": rows}

    def t_get_quote(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        return self.quote(from_symbol, from_chain, to_symbol, to_chain, amount)

    def t_get_activity(self, limit=None):
        rows = self.s["activity"][: (10 if limit is None else int(limit))]
        out = [dict(r) for r in rows]
        note = WALLET_NOTE if out else "No activity yet. " + WALLET_NOTE
        return {"activity": out, "in_progress": sum(1 for r in out if r["status"] == "in_progress"), "note": note}

    def t_get_address(self, contact=None, chain=None):
        if contact:
            hits = [c for c in self.s["contacts"] if contact.lower() in c["name"].lower()]
            exact = [c for c in hits if c["name"].lower() == contact.lower()]
            hits = exact or hits
            if not hits:
                return {"note": f"No saved contact called {contact}. Ask the user for the address.", "status": "not_found"}
            if len(hits) > 1:
                names = [c["name"] for c in hits]
                return {"note": "Nothing was returned. Ask the user which one.", "options": names,
                        "question": f"Which {contact}? " + " or ".join(names) + ".", "status": "needs_choice"}
            c = hits[0]
            self.resolved.add(c["address"])
            nets = (["ethereum", "base", "arbitrum", "bnb", "polygon", "monad"] if c["family"] == "evm" else ["solana"])
            return {"contacts": [{"address": c["address"], "family": c["family"], "name": c["name"], "networks": nets}]}
        return {"error": "get_address without a contact is not used in this suite"}

    def t_get_positions(self, kind=None):
        r = {"wallet": "Main"}
        if kind in (None, "all", "earn"):
            r["earn"] = [{"amount": a["amount"], "apy_pct": a["apy_pct"], "chain": a["chain"], "symbol": a["symbol"]}
                         for a in self.s["aave"]]
        if kind in (None, "all", "spot"):
            r["spot"] = []
        return r

    # ---- prepares
    def _guard_chain(self, chain):
        if chain and not named_chain(chain, "\n".join(self.typed)):
            return None
        return chain

    def _insufficient(self, sym, chain=None):
        hs = [h for h in self.s["holdings"] if h["symbol"] == sym and (chain is None or h["chain"] == chain)]
        if not hs:
            hold = f"no {sym}" + (f" on {CHAIN_NAME[chain]}" if chain else "")
        else:
            hold = " and ".join(pretty_hold(h["amount"], sym, h["chain"]) for h in hs)
        return {"holding": hold, "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"}

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        if self.hold(from_symbol, from_chain) < D(amount):
            return self._insufficient(from_symbol, from_chain)
        q = self.quote(from_symbol, from_chain, to_symbol, to_chain, amount)
        if "error" in q:
            return q
        q["note"] = ("Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it "
                     "with Face ID or cancels.")
        total_usd = sum(h["usd"] for h in self.s["holdings"])
        if q["amount_in_usd"] > 0.5 * total_usd and self.s.get("warn_big", True):
            q["policy"] = "warn"
            q["warnings"] = ["This is more than half of your balance."]
        else:
            q["policy"] = "allow"
        q["status"] = "prepared_for_review"
        return q

    def t_prepare_send(self, to_address, amount, symbol, chain=None):
        typed = "\n".join(self.typed)
        if to_address not in typed and to_address not in self.resolved:
            return {"error": "Refused: the user didn't type that address in this chat. Only send to an address the user "
                             "gave you themselves. Ask them for it."}
        chain = self._guard_chain(chain)
        fam = addr_family(to_address)
        cands = [h["chain"] for h in self.s["holdings"] if h["symbol"] == symbol and
                 (addr_family(to_address) == ("solana" if h["chain"] == "solana" else "evm"))]
        if chain is None:
            if not cands:
                return self._insufficient(symbol)
            if len(cands) > 1:
                return self._needs_chain("send", symbol, cands)
            chain = cands[0]
        if self.hold(symbol, chain) < D(amount):
            return self._insufficient(symbol, chain)
        return {"amount": f"{dstr(amount)} {symbol}", "executed": False, "kind": "send", "network": CHAIN_NAME[chain],
                "note": "Waiting for the user to approve with Face ID.", "status": "review_opened"}

    def _needs_chain(self, what, sym, cands):
        parts = [f"{CHAIN_NAME[c]}: {self.hold(sym, c) and dstr(self.hold(sym, c))}" for c in cands]
        names = " and ".join(CHAIN_NAME[c] for c in cands)
        return {"note": "Nothing was prepared. Ask the user which network, in one short question.", "options": list(cands),
                "prepared": False, "question": f"Which network should the {what} use? You hold {sym} on {names} ({', '.join(parts)}).",
                "status": "needs_chain"}

    def t_prepare_multisend(self, symbol, recipients, chain=None):
        typed = "\n".join(self.typed)
        for r in recipients:
            if r["address"] not in typed and r["address"] not in self.resolved:
                return {"error": f"Refused: {r['address'][:10]}… isn't in the user's messages or attached file. Only send to "
                                 f"addresses the user gave you. Ask them for the list."}
        chain = self._guard_chain(chain)
        cands = [h["chain"] for h in self.s["holdings"] if h["symbol"] == symbol]
        if chain is None:
            if not cands:
                return self._insufficient(symbol)
            if len(cands) > 1:
                return self._needs_chain("send", symbol, cands)
            chain = cands[0]
        total = sum(D(r["amount"]) for r in recipients)
        if self.hold(symbol, chain) < total:
            return self._insufficient(symbol, chain)
        fees = "0.12" if symbol in STABLES else "0.0004"
        return {"executed": False, "fees_up_to": f"{fees} {symbol}", "kind": "multisend", "network": CHAIN_NAME[chain],
                "note": "Nothing has moved. The Batch send review lists every recipient; the user approves all of it with one "
                        "Face ID, or cancels.", "recipients": len(recipients), "status": "review_opened",
                "total": f"{dstr(total)} {symbol}"}

    def t_prepare_deposit(self, symbol, amount, project=None, chain=None):
        if project not in (None, "aave-v3"):
            return {"note": "Only Aave v3 deposits can be prepared here. Other pools are in Earn, DeFi, with Ask.",
                    "prepared": False, "status": "unsupported"}
        chain = self._guard_chain(chain)
        cands = [c for c in AAVE_CHAINS if symbol in AAVE_SUPPORT[c] and self.hold(symbol, c) > 0]
        if chain is None:
            if not cands:
                return self._insufficient(symbol)
            if len(cands) > 1:
                return self._needs_chain("deposit", symbol, cands)
            chain = cands[0]
        if chain not in AAVE_CHAINS or symbol not in AAVE_SUPPORT[chain]:
            return {"note": "Only Aave v3 deposits can be prepared here. Other pools are in Earn, DeFi, with Ask.",
                    "prepared": False, "status": "unsupported"}
        if self.hold(symbol, chain) < D(amount):
            return self._insufficient(symbol, chain)
        return {"amount": f"{dstr(amount)} {symbol}", "apy_pct": self.apy(chain, symbol), "executed": False, "kind": "deposit",
                "note": "Waiting for the user to review the deposit and approve it with Face ID.",
                "pool": f"Aave v3 on {CHAIN_NAME[chain]}", "status": "review_opened"}

    def t_prepare_withdraw(self, symbol, amount=None, chain=None):
        chain = self._guard_chain(chain)
        rows = [a for a in self.s["aave"] if a["symbol"] == symbol and (chain is None or a["chain"] == chain)]
        if not rows:
            return {"note": f"The user has no {symbol} in Aave. Tell them.", "prepared": False, "status": "nothing_to_withdraw"}
        if len(rows) > 1:
            parts = ", ".join(f"{a['amount']} {symbol} on {CHAIN_NAME[a['chain']]}" for a in rows)
            return {"note": "Nothing was prepared. Ask the user which one.", "prepared": False,
                    "question": f"Which deposit? {parts}.", "status": "needs_choice"}
        a = rows[0]
        if amount is not None and D(amount) > D(a["amount"]):
            return {"note": f"Only {a['amount']} {symbol} is in Aave on {CHAIN_NAME[a['chain']]}. Tell the user.",
                    "prepared": False, "status": "insufficient", "supplied": f"{a['amount']} {symbol}"}
        return Lit([("status", "review_opened"), ("kind", "withdraw"), ("executed", False),
                    ("note", "Waiting for the user to approve with Face ID.")])

    def t_prepare_close(self, coin, percent=None):
        pos = [p for p in self.s["hl"]["positions"] if p["coin"] == coin]
        if not pos:
            return {"note": f"The user has no open {coin} position on Hyperliquid. Tell them.", "prepared": False,
                    "status": "no_position"}
        p = pos[0]
        mark = self.s["prices"][coin]
        sgn = 1 if p["side"] == "long" else -1
        upnl = (mark - p["entry"]) * p["size"] * sgn
        frac = (percent or 100) / 100
        size = p["size"] * frac
        closing = (f"all of the {coin} {p['side']}, {dstr(p['size'])} {coin}" if percent in (None, 100)
                   else f"{dstr(percent)}% of the {coin} {p['side']}, {dstr(round(size, 8))} {coin}")
        return {"closing": closing, "executed": False, "kind": "close", "note": "Waiting for the user to approve with Face ID.",
                "order": "Market, reduce only", "status": "review_opened", "unrealised_pnl": round(upnl * frac, 2)}

    def t_prepare_sell(self, token, percent=None, amount=None, receive=None):
        rows = [t for t in self.s["spot"] if t["symbol"].upper() == token.upper()]
        if not rows:
            return {"note": f"The user holds no {token.upper()} on Solana. Tell them.", "prepared": False, "status": "not_held"}
        t = rows[0]
        if t.get("frozen"):
            return {"note": "Nothing was prepared. Tell the user why.",
                    "reasons": [f"Your {t['symbol']} balance is frozen by the token's authority, so it can't be sold."],
                    "status": "blocked"}
        have = D(t["amount"])
        if percent is not None:
            sell = have * D(percent) / 100
            label = "all of your" if percent == 100 else f"{dstr(percent)}% of your"
        else:
            sell = D(amount)
            if sell > have:
                return {"holding": f"{t['amount']} {t['symbol']}", "note": "The user holds less than that. Tell them.",
                        "prepared": False, "status": "insufficient"}
            label = None
        sell, got = self.sell_calc(token, percent, amount, receive)
        rcv = receive or "USDC"
        sell_txt = approx(sell)
        you_sell = f"{sell_txt} {t['symbol']} (" + (label + f" {t['symbol']})" if label else "as you asked)")
        if label is None:
            you_sell = f"{sell_txt} {t['symbol']}"
        return {"executed": False, "kind": "sell", "note": "Nothing has moved. The Sell review is open with this Jupiter quote; "
                "the user approves it with Face ID or cancels.", "price_impact_pct": t.get("impact", 0.04),
                "route": "Jupiter via Orca, Raydium", "slippage_pct": 0.5, "status": "review_opened", "symbol": t["symbol"],
                "you_get": f"about {approx(got)} {rcv} on Solana", "you_sell": you_sell}

    def sell_calc(self, token, percent, amount, receive):
        t = [x for x in self.s["spot"] if x["symbol"].upper() == token.upper()][0]
        have = D(t["amount"])
        sell = have * D(percent) / 100 if percent is not None else D(amount)
        px = self.s["prices"][t["symbol"]]
        usd_v = float(sell) * px
        got = usd_v * 0.998 if (receive or "USDC") == "USDC" else usd_v * 0.998 / self.s["prices"]["SOL"]
        return sell, got

    def sell_apply(self, token, sell):
        for t in self.s["spot"]:
            if t["symbol"].upper() == token.upper():
                left = D(t["amount"]) - sell
                t["amount"] = dstr(left)
                t["usd"] = round(float(left) * self.s["prices"][t["symbol"]], 2)
        self.s["spot"] = [t for t in self.s["spot"] if D(t["amount"]) > 0]

    def t_prepare_trade(self, coin, side, size_usd, leverage, order_type, limit_price=None, take_profit=None, stop_loss=None):
        if coin not in self.s["prices"] or coin not in PERP_MAX_LEV:
            return {"error": f"{coin} isn't tradable on Hyperliquid"}
        lev = min(leverage, PERP_MAX_LEV[coin])
        need = size_usd / lev
        avail = self.s["hl"]["available"]
        if need > avail:
            return {"error": f"Not enough margin: {avail:.2f} USDC available, {need:.2f} needed", "prepared": False}
        return Lit([("status", "review_opened"), ("kind", "trade"), ("executed", False),
                    ("note", "Waiting for the user to approve with Face ID.")])

    def t_open_page(self, page, tab=None):
        res = {"note": f"The user taps it to open {page.capitalize()}.", "page": page, "screen": page, "status": "button_shown"}
        if tab:
            res["tab"] = tab
            res["note"] = f"The user taps it to open {page.capitalize()}."
        return res


# ------------------------------------------------------------------ world builder
def rprice(rng, sym):
    lo, hi = PRICE_BAND[sym]
    x = rng.uniform(lo, hi)
    return round(x, 2) if x >= 10 else (round(x, 3) if x >= 1 else round(x, 4))


YIELD_OTHERS = ["Morpho", "Compound v3", "Spark", "Fluid", "Euler"]


def make_yields(rng, apy):
    rows = []
    for asset in ("USDC", "USDT", "ETH"):
        for ch in AAVE_CHAINS:
            if asset not in AAVE_SUPPORT[ch]:
                continue
            base = apy[f"{ch}|{asset}"]
            rew = 0
            rows.append({"apy_30d_pct": round(base + rng.uniform(-0.4, 0.3), 2), "apy_base_pct": base,
                         "apy_pct": round(base + rew, 2), "apy_reward_pct": rew, "asset": asset, "can_deposit_in_hermit": True,
                         "chain": CHAIN_NAME[ch], "il_risk": "no", "pool_meta": None, "protocol": "Aave v3",
                         "tvl_usd": rng.choice([95, 180, 320, 560, 950]) * 1000000})
        for pr in rng.sample(YIELD_OTHERS, 2):
            ch = rng.choice(AAVE_CHAINS)
            base = round(apy.get(f"{ch}|{asset}", 3.0) + rng.uniform(0.5, 3.2), 2)
            rows.append({"apy_30d_pct": round(base - rng.uniform(0, 0.5), 2), "apy_base_pct": base, "apy_pct": base,
                         "apy_reward_pct": 0, "asset": asset, "can_deposit_in_hermit": False, "chain": CHAIN_NAME[ch],
                         "il_risk": "no", "pool_meta": None, "protocol": pr, "tvl_usd": rng.choice([12, 40, 85, 182]) * 1000000})
    rows.sort(key=lambda r: -r["apy_pct"])
    return rows


def make_markets(rng, prices):
    rows = []
    for c in FUND_NAMES:
        rows.append({"funding_hourly_pct": round(rng.uniform(-0.0045, 0.0125), 5), "max_leverage": PERP_MAX_LEV[c],
                     "name": f"{c}-PERP", "price": prices[c], "type": "perp",
                     "volume_24h_usd": rng.choice([42, 95, 180, 410, 960, 1840]) * 1000000})
    return rows


def new_world(rng):
    prices = {k: rprice(rng, k) for k in PRICE_BAND}
    change = {k: round(rng.uniform(-8, 8), 2) for k in PRICE_BAND}
    apy = {}
    for ch in AAVE_CHAINS:
        for sym in AAVE_SUPPORT[ch]:
            apy[f"{ch}|{sym}"] = round(rng.uniform(1.4, 6.4), 2) if sym != "ETH" else round(rng.uniform(0.8, 2.4), 2)
    st = {"prices": prices, "change": change, "holdings": [], "aave": [], "activity": [], "spot": [], "contacts": [],
          "hl": {"available": 0, "orders": [], "positions": []}, "apy": apy,
          "fee_bps": rng.choice([10, 15, 20, 25, 30, 35]), "seconds": rng.choice([8, 11, 14, 16, 20, 24, 30, 42]),
          "spread": round(rng.choice([0.01, 0.02, 0.03, 0.04, 0.06, 0.08]), 2), "markets": make_markets(rng, prices),
          "yields": make_yields(rng, apy), "force": {}, "warn_big": rng.random() < 0.3}
    if rng.random() < 0.3:
        st["near"] = rng.choice(["alice.near", "mm-wallet.near", "kai.near", "nina.near"])
    return World(st)


def fresh_times(rng):
    """Clock for one scenario on 2026-11-11."""
    return {"m": rng.randint(7 * 60, 16 * 60)}


def tstr(m, day=11):
    return f"2026-11-{day:02d} {m // 60:02d}:{m % 60:02d}"


def short_addr(a):
    return a[:4] + "…" + a[-2:] if a.startswith("0x") else a[:4] + "…" + a[-4:]


def old_rows(rng, w, n, avoid=()):
    """Older confirmed/failed distractor rows (before today), newest first."""
    rows = []
    d = 10
    m = rng.randint(9 * 60, 22 * 60)
    for i in range(n):
        k = rng.choice(["send", "send", "swap", "bridge", "buy", "deposit"])
        amt = rng.choice(["5", "10", "12.5", "25", "40", "60", "75", "150", "250"])
        if k == "send":
            what = f"Sent {amt} USDC to {short_addr(H.evm_addr(rng))}"
        elif k == "swap":
            what = f"Swap {amt} USDC to ETH on {CHAIN_NAME[rng.choice(['base', 'ethereum'])]}"
        elif k == "bridge":
            c1, c2 = rng.sample(["base", "arbitrum", "ethereum", "polygon"], 2)
            what = f"Swap {amt} USDC on {CHAIN_NAME[c1]} to USDC on {CHAIN_NAME[c2]}"
        elif k == "buy":
            what = f"Bought {rng.choice(['4.82', '31.4', '120'])} WIF with {amt} USDC"
        else:
            what = f"Deposited {amt} USDC into Aave v3 on {CHAIN_NAME[rng.choice(AAVE_CHAINS)]}"
        if what in avoid or any(what == r["what"] for r in rows):
            continue
        rows.append({"status": "failed" if rng.random() < 0.15 else "confirmed", "time": tstr(m, d), "what": what})
        m -= rng.randint(30, 600)
        if m < 60:
            m += 1300
            d -= 1
    return rows


def add_noise_holdings(rng, w, avoid):
    """Extra holdings of symbols the scenario does not use (so nothing becomes ambiguous)."""
    syms = {a[0] if isinstance(a, tuple) else a for a in avoid}
    opts = [("ETH", "ethereum"), ("ETH", "base"), ("ETH", "arbitrum"), ("SOL", "solana"), ("USDC", "solana"),
            ("USDT", "ethereum"), ("USDT", "polygon"), ("USDC", "polygon")]
    rng.shuffle(opts)
    n = rng.choice([0, 1, 1, 2])
    used = set()
    for sym, ch in opts:
        if n == 0:
            break
        if sym in syms or sym in used:
            continue
        px = w.price(sym)
        usd_v = rng.choice([8, 15, 30, 60, 120, 250, 400])
        amt = D(str(round(usd_v / px, 4 if px > 50 else 2)))
        if amt <= 0:
            continue
        w.set_hold(sym, ch, amt)
        used.add(sym)
        n -= 1


# ------------------------------------------------------------------ phrasing machinery
def P(rng, split, lst):
    """Pick from the train part (index % 5 != 0) or the eval part (index % 5 == 0) of a phrase pool."""
    pool = [x for i, x in enumerate(lst) if (i % 5 == 0) == (split == "eval")]
    assert pool, "empty pool"
    return rng.choice(pool)


TYPOS = {"bridge": ["brigde", "birdge", "bridg"], "deposit": ["depsoit", "deposti", "desposit"],
         "withdraw": ["widthdraw", "withdrw", "witdraw"], "please": ["pls", "plz"], "then": ["thn", "tehn"],
         "after": ["afer", "aftr"], "swap": ["swpa", "sawp"], "close": ["clsoe", "colse"], "position": ["postion", "positon"],
         "approved": ["aproved", "approvd"], "convert": ["convrt", "conver"], "money": ["moeny"], "first": ["frist", "firts"],
         "send": ["snd", "sned"], "half": ["hlaf", "haf"], "landed": ["landd", "lnaded"], "sell": ["sel", "slel"],
         "next": ["nxt", "netx"], "continue": ["contiue", "continu"], "transfer": ["tranfser", "transfr"],
         "proceeds": ["procedes"], "everything": ["everyting", "evrything"], "check": ["chekc", "chek"],
         "tell": ["tel"], "think": ["thnk"], "about": ["abuot", "abt"], "would": ["wuld"], "need": ["nede"]}


def typo(rng, text):
    ws = re.findall(r"[A-Za-z]+", text)
    cands = [w for w in ws if w.lower() in TYPOS]
    if not cands:
        return text
    w = rng.choice(cands)
    rep = rng.choice(TYPOS[w.lower()])
    if w[0].isupper():
        rep = rep.capitalize()
    return re.sub(r"\b" + re.escape(w) + r"\b", rep, text, count=1)


PRE = ["hey ", "hi, ", "ok so ", "pls ", "quick one: ", "yo ", "ok ", "can you help me. ", "hello, "]
POST = [" thanks", " pls", " thx", " asap", " :)", " please", ", thanks"]


def decorate(rng, text, first=True):
    r = rng.random()
    if first and r < 0.07:
        text = rng.choice(PRE) + text[0].lower() + text[1:]
    r = rng.random()
    if first and r < 0.07 and not text.endswith(("?", ".")):
        text += rng.choice(POST)
    if rng.random() < (0.2 if first else 0.1):
        text = typo(rng, text)
    if first and rng.random() < 0.06:
        text = text.rstrip(".?!")
    if first and rng.random() < 0.05:
        text = text.upper() if len(text) < 40 else text
    return text


def fill(tpl, **kw):
    return tpl.format(**kw)


THING_WORDS = {"bridge": ["the bridge", "the transfer", "it", "that"], "swap": ["the swap", "it", "the conversion"],
               "withdraw": ["the withdrawal", "it", "the withdraw"], "sell": ["the sale", "the sell", "it"],
               "close": ["the close", "it"], "send": ["the send", "it", "the transfer"], "deposit": ["the deposit", "it"]}

# generic "step 1 is in" follow-ups; index % 5 == 0 is eval only
LANDED_T = [
    "its in now", "it landed", "done", "ok {thing} is done", "approved it, should be there now",
    "{thing} landed, go on", "ok next", "yep {thing} went through, whats next", "ok now the next step",
    "approved, now what", "landed. continue", "done, continue", "{thing} is in, do the next one", "ok done pls continue",
    "i approved {thing}", "that one went through", "ok done. next?", "all approved on my side", "has {thing} landed yet?",
    "did {thing} land?", "ok {thing} is confirmed on my side", "landed", "approved and done, next please",
    "it should be there now, check", "go ahead with the next one, {thing} is done", "okay it's in", "ok {thing} finished",
    "done! now {nextv} it", "ok now {nextv} it", "{thing} landed, now {nextv}",
]
# user reports that it is done again, after an in-progress answer
AGAIN_T = [
    "ok now it landed", "try again", "its done now", "check again", "ok should be there now", "landed now",
    "now? it says done on my side", "ok check once more", "still waiting.. check now", "any update?",
    "ok it's there now", "give it another look", "refresh", "ok it landed now", "how about now",
]
FINAL_T = [
    "done", "approved", "approved it", "ok done", "its approved", "done, did it go through?", "ok approved, check",
    "approved. all good?", "ok that one is approved too", "finished", "ok done with that one", "approved, whats the status",
    "did it work?", "check it pls", "all done, is it confirmed?",
]
OK_T = [
    "ok", "yes", "yes do it", "ok do it", "go ahead", "yep", "sure", "yes please", "ok prepare it", "do it", "alright, go",
    "yes, that works", "ok sounds good", "go on then", "ok prepare that",
]


def user_followup(rng, split, pool, **kw):
    t = P(rng, split, pool)
    return decorate(rng, t.format(**kw), first=False)


# amounts as a user types them
AMTS_BIG = [10, 12, 15, 20, 25, 30, 40, 45, 50, 60, 75, 80, 100, 120, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600,
            750, 800, 1000, 1200, 1500, 2000, 2500]


def amt_user(rng, x):
    s = dstr(x)
    if D(x) >= 1000 and rng.random() < 0.35:
        return f"{int(D(x)):,}"
    return s


SYM_WORDS = {"USDC": ["USDC", "usdc", "USDC", "usdc", "usd coin"], "USDT": ["USDT", "usdt", "tether", "USDT"],
             "ETH": ["ETH", "eth", "ETH", "ether"], "SOL": ["SOL", "sol"], "DAI": ["DAI", "dai"]}
CHAIN_WORDS = {"ethereum": ["ethereum", "Ethereum", "mainnet", "ethereum"], "base": ["base", "Base", "base"],
               "arbitrum": ["arbitrum", "Arbitrum", "arbitrum"], "solana": ["solana", "Solana", "solana"],
               "polygon": ["polygon", "Polygon"], "bnb": ["bnb chain"]}
# chain words the regex of ToolGuard (CHAIN_RE) must see; "arb" also works but is avoided for clarity


# ------------------------------------------------------------------ conversation builder
def tick(rng, clock, lo=2, hi=18):
    clock["m"] = min(clock["m"] + rng.randint(lo, hi), 23 * 60 + 40)
    return tstr(clock["m"])


class Retry(Exception):
    pass


class Conv:
    def __init__(self, rng, w, bio, split):
        self.rng, self.w, self.bio, self.split = rng, w, bio, split
        sysmsg = SYSTEM_V3.replace("{BIO}", bio)
        if rng.random() < 0.1:
            sysmsg += f"\nThe user wants to be called {rng.choice(H.NAMES)}."
        self.sys = sysmsg
        self.msgs = [{"role": "system", "content": sysmsg}]
        self.stages, self.users, self.rounds, self.steps = [], [], [], []
        self.used = set()
        self.cut = None                     # index of the user message the eval context ends at (default: first user)

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        self.w.typed.append(text)
        self.stages.append(self.w.state())
        self.users.append(text)
        self.rounds.append([])

    def call(self, *calls):
        self.msgs.append({"role": "assistant", "content": "",
                          "tool_calls": [{"name": n, "arguments": a} for n, a in calls]})
        out, log = [], []
        for n, a in calls:
            r = self.w.call(n, a)
            out.append(r)
            log.append((n, a, r))
            self.used.add(n)
        for r in out:
            self.msgs.append({"role": "tool", "content": dumps_result(r, self.bio)})
        self.rounds[-1].append(log)
        return out

    def call1(self, name, args):
        return self.call((name, args))[0]

    def say(self, text):
        text = text.replace("{BIO}", self.bio)
        self.msgs.append({"role": "assistant", "content": text})
        self.rounds[-1].append(("text", text))

    def step(self, sid, name, args, needs=None, blocked=False):
        st = {"id": sid, "turn": len(self.users) - 1, "name": name, "arguments": args, "needs": needs}
        if blocked:
            st["blocked"] = True
        self.steps.append(st)

    def money_by_turn(self):
        out = []
        for rounds in self.rounds:
            ms = []
            for r in rounds:
                if isinstance(r, tuple):
                    continue
                for n, a, res in r:
                    if n in MONEY_TOOLS:
                        ms.append({"name": n, "arguments": a})
            out.append(ms)
        return out


def last_row_status(act, what):
    for r in act.get("activity", []):
        if r["what"] == what:
            return r["status"]
    return None


def Th(rng, kind):
    return {"bridge": "the bridge", "swap": "the swap", "withdraw": "the withdrawal", "sell": "the sale",
            "deposit": "the deposit", "send": "the send", "close": "the close"}.get(kind, kind)


def cap(s):
    return s[0].upper() + s[1:]


# ------------------------------------------------------------------ answer building blocks
def usd(x):
    return f"${float(x):,.2f}"


def after_line(rng, nxt):
    return rng.choice([
        f"{cap(nxt)} comes after it lands. Tell me when it has.",
        f"Once it lands, tell me and I'll check Activity and prepare {nxt}.",
        f"Next is {nxt}, but only once this one lands. Tell me when it has.",
        f"Tell me when it lands and I'll check, then prepare {nxt}.",
        f"{cap(nxt)} needs this one to land first. Tell me when it has.",
        f"I'll prepare {nxt} once you tell me this one has landed and Activity agrees.",
    ])


def warn_tail(res):
    return "".join(f" The review warns: {w}" for w in res.get("warnings", []))


def announce_swap(rng, res, kind, bio, cost=True):
    frm = res["from"]
    to = res["to"]
    verb = "the bridge" if kind == "bridge" else "the swap"
    s = rng.choice([f"I've prepared {verb}: {frm} to {to} through {ROUTE}, waiting for your {bio}.",
                    f"{cap(verb)} is prepared: {frm} to {to} through {ROUTE}. It's waiting for your {bio}.",
                    f"I've prepared {verb} through {ROUTE}: {frm} to {to}. Approve it with {bio}."])
    if cost and rng.random() < 0.5:
        s += f" The total cost is about {usd(res['total_cost_usd'])}."
    return s + warn_tail(res)


def reasons_text(res):
    return " ".join(res.get("reasons", []))


def blocked_reason_pool(kind):
    return ["Amount is above your daily limit for the assistant.", "This is above your daily limit for the assistant.",
            "Your spending policy doesn't allow this right now."] if kind != "send" else \
        ["Recipient flagged as a known scam address.", "Amount is above your daily limit for the assistant."]


def blocked_result(rng, kind="x"):
    return {"note": "Nothing was prepared. Tell the user why.", "reasons": [rng.choice(blocked_reason_pool(kind))],
            "status": "blocked"}


def ans_blocked(rng, kind, res, stopped_nxt):
    reason = reasons_text(res)
    return rng.choice([
        f"I couldn't prepare {Th(rng, kind)}: {reason} So {stopped_nxt} isn't prepared either.",
        f"{cap(Th(rng, kind))} was not prepared. {reason} {cap(stopped_nxt)} depends on it, so I stopped there.",
        f"Step one stopped: {reason} Nothing is waiting for your {{BIO}}, and {stopped_nxt} waits until this one works.",
    ])


def ans_pending(rng, kind, seconds, nxt):
    th = Th(rng, kind)
    if seconds:
        return rng.choice([
            f"{cap(th)} still shows as in progress in Activity. The quote said about {seconds} seconds, so give it a moment. Tell me when to check again.",
            f"{cap(th)} is still in progress. It was quoted at about {seconds} seconds, so I'm holding off on {nxt}. Tell me when to check again.",
            f"Activity still has {th} in progress (the quote said about {seconds} seconds). I haven't prepared {nxt}; tell me when to look again.",
        ])
    return rng.choice([
        f"{cap(th)} still shows as in progress in Activity, so I haven't prepared {nxt}. Give it a minute and tell me when to check again.",
        f"It's still in progress in Activity. I'll prepare {nxt} once it shows as confirmed; tell me when to check again.",
        f"Activity has {th} in progress. I'm holding off on {nxt} until it is confirmed. Tell me when to look again.",
    ])


def ans_missing(rng, kind, bio, nxt):
    th = Th(rng, kind)
    return rng.choice([
        f"I don't see {th} in Activity yet. Reviews you haven't approved don't show up there, so it may still be waiting for your {bio}. Approve it, then tell me and I'll check again.",
        f"Activity has no row for {th} yet, so I can't tell it was approved. Reviews that wait for {bio} don't appear there. Once you approve it, tell me and I'll check, then prepare {nxt}.",
        f"Nothing in Activity matches {th}. If the review is still open, approve it with {bio}; I haven't prepared {nxt}.",
    ])


def ans_failed(rng, kind, nxt):
    th = Th(rng, kind)
    return rng.choice([
        f"Activity shows {th} as failed, so I haven't prepared {nxt}. Check your balances in Wallets to see where the funds are.",
        f"{cap(th)} failed according to Activity. That stops the plan at step one and {nxt} is not prepared. Your balances in Wallets show where the funds are.",
        f"Activity lists {th} as failed, so {nxt} stays unprepared. Look at your balances in Wallets, then tell me how you want to go on.",
    ])


def confirmed_clause(rng, kind):
    th = Th(rng, kind)
    return rng.choice([f"Activity shows {th} as confirmed", f"{cap(th)} shows as confirmed in Activity",
                       f"I can see {th} as confirmed in Activity"])


def bal_clause(rng, amount, sym, chain):
    return rng.choice([f"you have {amount} {sym} on {CHAIN_NAME[chain]}", f"{CHAIN_NAME[chain]} now holds {amount} {sym} for you",
                       f"your {CHAIN_NAME[chain]} balance is {amount} {sym}"])


def landing_amount(rng, quote, sym, w):
    """What the balance shows after a swap/bridge landed: not always the quoted, rounded amount_out."""
    out = D(repr(quote["_out_units"]))
    mode = rng.random()
    if sym in STABLES:
        if mode < 0.5:
            return D(quote["amount_out"])
        if mode < 0.75:
            return D(quote["amount_out"]) + rng.choice([D("-0.01"), D("0.01"), D("-0.02")])
        return out.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    if mode < 0.5:
        return D(quote["amount_out"])
    return out.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def row_swap(kind, amt, sym, src, to_sym, dst):
    if kind == "bridge":
        return f"Swap {dstr(amt)} {sym} on {CHAIN_NAME[src]} to {to_sym} on {CHAIN_NAME[dst]}"
    return f"Swap {dstr(amt)} {sym} to {to_sym} on {CHAIN_NAME[src]}"


def report_world(rng, w, clock, leg, outcome, first=True):
    """Move the world to what the user's report means. leg = dict(kind,row,debit=(sym,chain,amt),credit=(sym,chain,amt),aave=...)."""
    t = tick(rng, clock, 3, 14)
    if outcome == "missing":
        return
    if first:
        status = {"landed": "confirmed", "pending": "in_progress", "failed": "failed"}[outcome]
        w.add_activity(leg["row"], status, t)
        if outcome in ("landed", "pending"):
            leg["apply_debit"](w)
        if outcome == "landed":
            leg["apply_credit"](w)
    else:                                   # the row exists already as in_progress
        w.set_activity_status(leg["row"], "confirmed" if outcome == "landed" else "failed")
        if outcome == "landed":
            leg["apply_credit"](w)


# ------------------------------------------------------------------ the dependent-chain engine
def find_bal(bal_res, sym, chain):
    for h in bal_res.get("holdings", []):
        if h["symbol"] == sym and h["chain"] == chain:
            return h["amount"]
    return None


def ok_status(res):
    return res.get("status") in ("review_opened", "prepared_for_review")


def mention_bal(rng, bal_str, sym, chain):
    return bal_clause(rng, bal_str, sym, chain)


NEXTV = {"prepare_deposit": "deposit", "prepare_send": "send", "prepare_withdraw": "withdraw", "prepare_sell": "sell",
         "prepare_close": "close"}


def progress_clause(rng, th):
    return rng.choice([f"Activity shows {th} in progress, so it was approved", f"{cap(th)} is in progress in Activity, which means it was approved"])


def run_chain(c, rng, split, legs, seq, final_confirm, clock, indep=False):
    """legs[0] is prepared in the first assistant turn (the user message is already in c); every other leg only after
    a report of the previous leg. seq[k-1] = outcome list for the report(s) on leg k-1, e.g. ['pending', 'landed'].
    indep: the legs do not feed each other (only the one-review-at-a-time rule links them)."""
    bio, w = c.bio, c.w
    n = len(legs)
    for i, l in enumerate(legs):
        l["after_noun"] = legs[i + 1]["noun"] if i + 1 < n else "the next step"
    line = indep_line if indep else after_line
    out = legs[0]["prep"](c, None)
    if not out["ok"]:
        if out.get("args") and out["res"] is not None and out["res"].get("status") == "blocked":
            c.step("s1", legs[0]["tool"], out["args"], blocked=True)
        c.say(out["fail_text"])
        return "blocked0"
    legs[0]["res"] = out["res"]
    c.step("s1", legs[0]["tool"], out["args"])
    if legs[0].get("tail"):
        tail = legs[0]["tail"](rng)
    else:
        tail = line(rng, legs[1]["noun"]) if n > 1 else ""
    c.say((out["text"] + " " + out.get("extra", "") + " " + tail).replace("  ", " ").strip())
    for k in range(1, n):
        prev, cur = legs[k - 1], legs[k]
        pth = prev.get("th", prev["kind"])
        added = False
        for j, oc in enumerate(seq[k - 1]):
            if oc != "missing":
                report_world(rng, w, clock, prev, oc, first=not added)
                added = True
            if (oc == "landed" or (indep and oc == "pending")) and cur.get("force"):
                w.s["force"][cur["tool"]] = cur["force"]
            kw = dict(thing=rng.choice(THING_WORDS.get(prev["kind"], ["it"])) if "th" not in prev else "it",
                      nextv=cur.get("nextv") or NEXTV.get(cur["tool"]) or ("bridge" if cur["kind"] == "bridge" else "swap"), addr=cur.get("late_addr", ""), name=cur.get("name", ""))
            tpl_pool = (cur.get("report_pool") or LANDED_T) if j == 0 else AGAIN_T
            c.user(user_followup(rng, split, tpl_pool, **kw))
            if indep:
                act = c.call1("get_activity", {"limit": 5})
                bal = None
            else:
                bal_args = {"chain": cur["bal_chain"]} if cur.get("bal_chain") else {}
                act, bal = c.call(("get_activity", {"limit": 5}), ("get_balances", bal_args))
            row_state = last_row_status(act, prev["row"]) if "row" in prev else None
            proceed = oc == "landed" or (indep and oc == "pending")
            if proceed:
                if oc == "landed":
                    assert row_state == "confirmed", (row_state, prev["row"])
                else:
                    assert row_state == "in_progress", (row_state, prev["row"])
                out = cur["prep"](c, {"act": act, "bal": bal})
                intro = confirmed_clause(rng, pth) if oc == "landed" else progress_clause(rng, pth)
                if cur.get("bal_text") and out.get("ok"):
                    intro += ", and " + cur["bal_text"](rng, out)
                if not out["ok"]:
                    if out.get("args") and out["res"] is not None and out["res"].get("status") == "blocked":
                        c.step("s%d" % (k + 1), cur["tool"], out["args"], needs={"row": prev["row"], "status": "confirmed|in_progress" if indep else "confirmed"}, blocked=True)
                    c.say(intro + ". " + out["fail_text"])
                    if out.get("ask") and cur.get("short_bs") and rng.random() < 0.8:
                        c.user(decorate(rng, P(rng, split, OK_T), first=False))
                        a2 = D(cur["short_bs"])
                        args = {"to_address": cur["addr"], "amount": num_json(a2), "symbol": cur["sym"]}
                        res = c.call1("prepare_send", args)
                        assert ok_status(res), res
                        c.step("s%d" % (k + 1), "prepare_send", args, needs={"row": prev["row"], "status": "confirmed"})
                        c.say(f"I've prepared sending {res['amount']} to {cur['addr']} on {res['network']}, waiting for your {bio}.")
                    return "blocked%d" % k
                st = "confirmed|in_progress" if indep else "confirmed"
                c.step("s%d" % (k + 1), cur["tool"], out["args"], needs={"row": prev["row"], "status": st})
                tail = line(rng, legs[k + 1]["noun"]) if k + 1 < n else ""
                if cur.get("tail") and k + 1 < n:
                    tail = cur["tail"](rng)
                c.say((intro + ". " + out["text"] + " " + out.get("extra", "") + " " + tail).replace("  ", " ").strip())
                cur["res"] = out["res"]
                break
            if oc == "pending":
                assert row_state == "in_progress"
                c.say(ans_pending(rng, pth, prev.get("seconds"), cur["noun"]))
            elif oc == "missing":
                assert row_state is None
                c.say(ans_missing(rng, pth, bio, cur["noun"]))
            elif oc == "failed":
                assert row_state == "failed"
                if indep:
                    c.say(rng.choice([f"Activity shows {pth} as failed, so I haven't prepared {cur['noun']}. Do you want me to prepare it anyway, since it doesn't depend on the first?",
                                      f"{cap(pth)} failed according to Activity. I stopped there and {cur['noun']} is not prepared. Want me to go on with it anyway?"]))
                    if rng.random() < 0.6:
                        c.user(decorate(rng, P(rng, split, OK_T), first=False))
                        out = cur["prep"](c, {"act": act, "bal": None})
                        if out["ok"]:
                            c.step("s%d" % (k + 1), cur["tool"], out["args"], needs=None)
                            tail = line(rng, legs[k + 1]["noun"]) if k + 1 < n else ""
                            c.say((out["text"] + " " + tail).strip())
                else:
                    c.say(ans_failed(rng, pth, cur["noun"]))
                return "failed%d" % (k - 1)
        else:
            return "open%d" % (k - 1)       # the last outcome was not a go-ahead: the conversation ends here
    if final_confirm:
        last = legs[-1]
        oc = final_confirm
        t = tick(rng, clock, 3, 14)
        w.add_activity(last["row"], "confirmed" if oc == "landed" else "in_progress", t)
        last["apply_debit"](w)
        if oc == "landed":
            last["apply_credit"](w)
        c.user(user_followup(rng, split, FINAL_T))
        act = c.call1("get_activity", {"limit": 5})
        lth = last.get("th", last["kind"])
        if oc == "landed":
            c.say(last["final_text"](rng, act))
        else:
            c.say(rng.choice([f"{cap(Th(rng, lth))} still shows as in progress in Activity. Give it a minute and tell me when to check again.",
                              "It's still in progress in Activity, so I can't call it confirmed yet. Tell me when to look again.",
                              f"Activity has {Th(rng, lth)} in progress, not confirmed. Ask me again in a moment and I'll check."]))
    return "full"


# ------------------------------------------------------------------ leg constructors
def swap_leg(rng, bio, kind, sym, src, to_sym, dst, amt, amount_fn=None, bal_chain=None, noun=None, size_from_bal=None,
             frac=None, user_named_dst=True, nextv=None):
    """A prepare_swap leg. amt: fixed Decimal (leg 0 or typed). size_from_bal: (sym, chain) -> size from the balance string."""
    leg = {"kind": kind, "tool": "prepare_swap", "noun": noun or ("the bridge" if kind == "bridge" else "the swap"),
           "bal_chain": bal_chain, "nextv": nextv}

    def prep(c, check):
        a = amt
        calc_note = None
        if size_from_bal:
            bs = find_bal(check["bal"], size_from_bal[0], size_from_bal[1])
            assert bs is not None
            a = D(bs)
            if frac is not None and frac != 1:
                r = c.call1("calculator", {"expression": f"{bs} * {dstr(frac)}"})
                a = D(r["result"])
        args = {"from_symbol": sym, "from_chain": src, "to_symbol": to_sym, "to_chain": dst, "amount": num_json(a)}
        res = c.call1("prepare_swap", args)
        if not ok_status(res):
            return {"ok": False, "fail_text": fail_text(rng, c, kind, res, leg), "res": res, "args": args}
        q = c.w.quote(sym, src, to_sym, dst, a)
        landed = landing_amount(rng, q, to_sym, c.w)
        leg["row"] = row_swap(kind, a, sym, src, to_sym, dst)
        leg["seconds"] = res["seconds"]
        leg["landed"] = landed
        leg["amt"] = a
        leg["apply_debit"] = lambda w: w.add_hold(sym, src, -a)
        leg["apply_credit"] = lambda w: w.add_hold(to_sym, dst, landed)
        leg["final_text"] = lambda rng_, act: (f"{confirmed_clause(rng_, kind)}. That finishes everything you asked for.")
        return {"ok": True, "res": res, "args": args, "text": announce_swap(rng, res, kind, bio)}
    leg["prep"] = prep
    return leg


def fail_text(rng, c, kind, res, leg):
    st = res.get("status")
    if st == "blocked":
        return ans_blocked(rng, kind, res, leg.get("after_noun", "the next step")).replace("{BIO}", c.bio)
    if st == "insufficient":
        return (f"I couldn't prepare {Th(rng, kind)}: you hold {res['holding']}, which is less than that. "
                f"Nothing is waiting for your {c.bio}, and the next step is not prepared either.")
    if st in ("needs_chain", "needs_choice"):
        return res["question"]
    if st == "nothing_to_withdraw":
        return "You have nothing in Aave to withdraw, so there is nothing to prepare."
    return res.get("error", "That couldn't be prepared.")


def deposit_leg(rng, bio, sym, chain, size_bal=True, frac=None, amt=None, named_chain=True, noun="the Aave deposit"):
    leg = {"kind": "deposit", "tool": "prepare_deposit", "noun": noun, "bal_chain": chain}

    def prep(c, check):
        if size_bal:
            bs = find_bal(check["bal"], sym, chain)
            assert bs is not None
            a = D(bs)
            leg["bal_str"] = bs
            if frac is not None and frac != 1:
                r = c.call1("calculator", {"expression": f"{bs} * {dstr(frac)}"})
                a = D(r["result"])
                leg["calc"] = r["result"]
        else:
            a = amt
        args = {"project": "aave-v3", "symbol": sym}
        if named_chain:
            args["chain"] = chain
        args["amount"] = num_json(a)
        res = c.call1("prepare_deposit", args)
        if not ok_status(res):
            return {"ok": False, "fail_text": fail_text(rng, c, "deposit", res, leg), "res": res, "args": args}
        leg["row"] = f"Deposited {dstr(a)} {sym} into Aave v3 on {CHAIN_NAME[chain]}"
        leg["apply_debit"] = lambda w: w.add_hold(sym, chain, -a)
        leg["apply_credit"] = lambda w: w.set_aave(sym, chain, w.aave_amt(sym, chain) + a)
        apy = res["apy_pct"]
        leg["final_text"] = lambda rng_, act: rng_.choice([
            f"{confirmed_clause(rng_, 'deposit')}. Your {res['amount']} is now in {res['pool']}, so every step you asked for is confirmed.",
            f"{confirmed_clause(rng_, 'deposit')}, and the earlier steps show as confirmed too. {cap(res['amount'])} is now in {res['pool']} at {apy}% APY, which can change.",
        ])
        what = rng.choice([f"I've prepared depositing {res['amount']} into {res['pool']} at {apy}% APY, which can change. Approve with {bio}.",
                           f"I've prepared a deposit of {res['amount']} into {res['pool']} at {apy}% APY, which can change. It's waiting for your {bio}.",
                           f"Next, I've prepared depositing {res['amount']} into {res['pool']} (APY {apy}%, it can change). Approve it with {bio}."])
        if frac is not None and frac != 1:
            what = f"Half of that is {leg['calc']}. " + what
        return {"ok": True, "res": res, "args": args, "text": what}

    def bal_text(rng_, out):
        return bal_clause(rng_, leg["bal_str"], sym, chain) if size_bal and "bal_str" in leg else ""
    leg["prep"] = prep
    leg["bal_text"] = bal_text if size_bal else None
    return leg


def withdraw_leg(rng, bio, sym, chain_arg, amt_arg, aave_chain, noun="the withdrawal", size_exact=None):
    """A prepare_withdraw leg. amt_arg None = withdraw all. The activity row shows the amount that came out."""
    leg = {"kind": "withdraw", "tool": "prepare_withdraw", "noun": noun, "bal_chain": None}

    def prep(c, check):
        args = {"symbol": sym}
        if chain_arg:
            args["chain"] = chain_arg
        if amt_arg is not None:
            args["amount"] = num_json(amt_arg)
        res = c.call1("prepare_withdraw", args)
        if not ok_status(res):
            return {"ok": False, "fail_text": fail_text(rng, c, "withdraw", res, leg), "res": res, "args": args}
        supplied = c.w.aave_amt(sym, aave_chain)
        out_amt = D(amt_arg) if amt_arg is not None else supplied
        leg["out_amt"] = out_amt
        leg["row"] = f"Withdrew {approx(out_amt)} {sym} from Aave v3 on {CHAIN_NAME[aave_chain]}"
        leg["apply_debit"] = lambda w: w.set_aave(sym, aave_chain, w.aave_amt(sym, aave_chain) - out_amt)
        leg["apply_credit"] = lambda w: w.add_hold(sym, aave_chain, out_amt)
        leg["final_text"] = lambda rng_, act: f"{confirmed_clause(rng_, 'withdraw')}. That finishes everything you asked for."
        where = f" on {CHAIN_NAME[aave_chain]}" if chain_arg else ""
        what_amt = f"all of your {sym}" if amt_arg is None else f"{dstr(amt_arg)} {sym}"
        text = rng.choice([f"I've prepared withdrawing {what_amt} from Aave{where}, waiting for your {bio}.",
                           f"The withdrawal of {what_amt} from Aave{where} is prepared. Approve it with {bio}.",
                           f"I've prepared taking {what_amt} out of Aave{where}. It's waiting for your {bio}."])
        return {"ok": True, "res": res, "args": args, "text": text}
    leg["prep"] = prep
    return leg


def send_leg(rng, bio, sym, chain_arg, addr, amt=None, size_bal_chain=None, frac=None, noun=None, resolved_name=None,
             nextv="send", pool=None):
    """A prepare_send leg. amt fixed, or sized from the balance of (sym, size_bal_chain)."""
    leg = {"kind": "send", "tool": "prepare_send", "noun": noun, "bal_chain": size_bal_chain, "nextv": nextv}
    if pool:
        leg["report_pool"] = pool

    leg["addr"], leg["sym"] = addr, sym

    def prep(c, check):
        a = amt
        if amt is None:
            bs = find_bal(check["bal"], sym, size_bal_chain)
            assert bs is not None
            a = D(bs)
            leg["bal_str"] = bs
            if frac is not None and frac != 1:
                r = c.call1("calculator", {"expression": f"{bs} * {dstr(frac)}"})
                a = D(r["result"])
                leg["calc"] = r["result"]
        elif size_bal_chain and check and check.get("bal") is not None:
            bs = find_bal(check["bal"], sym, size_bal_chain)
            if bs is not None and D(bs) < amt:
                leg["short_bs"] = bs
                return {"ok": False, "ask": True, "res": None,
                        "fail_text": rng.choice([
                            f"You have {bs} {sym} on {CHAIN_NAME[size_bal_chain]}, which is less than the {dstr(amt)} you asked me to send. I haven't prepared the send. Want me to send {bs} {sym} instead?",
                            f"Your {CHAIN_NAME[size_bal_chain]} balance is {bs} {sym}, short of {dstr(amt)}, so the send is not prepared. Shall I send {bs} {sym} instead?"])}
        args = {"to_address": addr, "amount": num_json(a), "symbol": sym}
        if chain_arg:
            args["chain"] = chain_arg
        res = c.call1("prepare_send", args)
        if not ok_status(res):
            return {"ok": False, "fail_text": fail_text(rng, c, "send", res, leg), "res": res, "args": args}
        net_chain = [k for k, v in CHAIN_NAME.items() if v == res["network"]][0]
        leg["row"] = f"Sent {dstr(a)} {sym} to {short_addr(addr)}"
        leg["apply_debit"] = lambda w: w.add_hold(sym, net_chain, -a)
        leg["apply_credit"] = lambda w: None
        leg["final_text"] = lambda rng_, act: rng_.choice([
            f"{confirmed_clause(rng_, 'send')}. That finishes everything you asked for.",
            f"{confirmed_clause(rng_, 'send')}: {res['amount']} to {addr}. Every step is confirmed.",
        ])
        who = f"{resolved_name} ({addr})" if resolved_name else addr
        text = rng.choice([f"I've prepared sending {res['amount']} to {who} on {res['network']}. Approve it with {bio}.",
                           f"The send of {res['amount']} to {who} on {res['network']} is prepared, waiting for your {bio}.",
                           f"I've prepared {res['amount']} to {who} on {res['network']}. It's waiting for your {bio}."])
        if frac is not None and frac != 1 and "calc" in leg:
            text = f"That fraction of {leg['bal_str']} is {leg['calc']}. " + text
        return {"ok": True, "res": res, "args": args, "text": text}

    def bal_text(rng_, out):
        return bal_clause(rng_, leg["bal_str"], sym, size_bal_chain) if amt is None and "bal_str" in leg else ""
    leg["prep"] = prep
    leg["bal_text"] = bal_text if amt is None else None
    return leg


# ------------------------------------------------------------------ more legs, outcome sequences, finishing
def sell_leg(rng, bio, tok, percent, amount, receive, noun="the sale", extra_calls=None):
    leg = {"kind": "sell", "tool": "prepare_sell", "noun": noun, "bal_chain": "solana"}

    def prep(c, check):
        args = {"token": tok}
        if percent is not None:
            args["percent"] = percent
        else:
            args["amount"] = num_json(amount)
        if receive:
            args["receive"] = receive
        outs = c.call(("prepare_sell", args), *(extra_calls or []))
        res = outs[0]
        leg["extra_res"] = outs[1:]
        if not ok_status(res):
            return {"ok": False, "fail_text": fail_text(rng, c, "sell", res, leg), "res": res, "args": args}
        sell, got = c.w.sell_calc(tok, percent, amount, receive)
        rcv = receive or "USDC"
        leg["row"] = f"Sold {approx(sell)} {tok} for {approx(got)} {rcv}"
        mode = rng.random()
        gd = D(approx(got)) if mode < 0.5 else (D(approx(got)) + rng.choice([D("-0.01"), D("0.01")]) if mode < 0.75
                                               else D(repr(got)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
        if rcv == "SOL":
            gd = D(repr(got)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        leg["apply_debit"] = lambda w: w.sell_apply(tok, sell)
        leg["apply_credit"] = lambda w: w.add_hold(rcv, "solana", gd)
        leg["final_text"] = lambda rng_, act: f"{confirmed_clause(rng_, 'sell')}. That finishes everything you asked for."
        text = rng.choice([
            f"I've prepared selling {res['you_sell']} for {res['you_get']} through Jupiter, waiting for your {bio}.",
            f"The sale is prepared: {res['you_sell']} for {res['you_get']} through Jupiter. Approve it with {bio}.",
            f"I've prepared selling {res['you_sell']}. You'd get {res['you_get']} (price impact {res['price_impact_pct']}%). Waiting for your {bio}."])
        return {"ok": True, "res": res, "args": args, "text": text}
    leg["prep"] = prep
    return leg


def pick_seq(rng, n_trans, weights=None):
    """Outcome lists per transition; stops after a transition that does not end in landed."""
    opts = [(["landed"], 62), (["pending", "landed"], 14), (["missing", "landed"], 8), (["pending"], 5), (["failed"], 6),
            (["missing"], 5)]
    seq = []
    for _ in range(n_trans):
        r = rng.random() * sum(w for _, w in opts)
        acc = 0
        for o, wgt in opts:
            acc += wgt
            if r <= acc:
                break
        seq.append(list(o))
        if o[-1] != "landed":
            break
    return seq


def seq_complete(seq, n_trans):
    return len(seq) == n_trans and seq[-1][-1] == "landed"


def pick_final(rng):
    r = rng.random()
    if r < 0.38:
        return "landed"
    if r < 0.45:
        return "pending"
    return None


def finish_money(c, shape, tag, extra=None):
    """Turn the conversation into a record."""
    used = set(c.used)
    pack, packs = pack_for(used)
    # the model's own calls only (contacts / reads included)
    n_user = len(c.users)
    n_calls = sum(1 for r in c.msgs if r["role"] == "assistant" and r.get("tool_calls"))
    fam = ["4b", "lite"] if (n_user <= 3 and n_calls <= 5 and len(packs) <= 1) else ["4b"]
    meta = {"source": SRC, "suite": "chain_multi", "kind": shape, "shape": shape, "tag": tag, "bio": c.bio,
            "device": "Mac" if c.bio == "Touch ID" else "iPhone", "pack": pack, "packs": packs, "family_ok": fam,
            "steps": c.steps, "n_user_turns": n_user, "script": c.users[1:], "world": {"stages": c.stages},
            "money_by_turn": c.money_by_turn(), "template_split": c.split,
            "expect": expect_for(c, 0), "cut": 0}
    if extra:
        meta.update(extra)
    return {"messages": c.msgs, "tools": True, "meta": meta}


def expect_for(c, turn):
    """The assistant decisions after user turn `turn`: calls rounds, then a text or an ask."""
    out = []
    for r in c.rounds[turn]:
        if isinstance(r, tuple):
            out.append({"ask": True} if r[1].rstrip().endswith("?") and not any(
                n in MONEY_TOOLS for rd in c.rounds[turn] if not isinstance(rd, tuple) for n, _, _ in rd) else {"text": True})
        else:
            out.append({"calls": [{"name": n, "arguments": a} for n, a, _ in r]})
    return out


def pick_bio(rng):
    return "Touch ID" if rng.random() < 0.3 else "Face ID"


def new_conv(rng, split, bio):
    w = new_world(rng)
    c = Conv(rng, w, bio, split)
    c.clock = fresh_times(rng)
    return w, c


def amt_dec(sym, rng):
    if sym == "ETH":
        return D(str(rng.choice([0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2])))
    if sym == "SOL":
        return D(str(rng.choice([1, 2, 3, 5, 8, 10, 15, 20, 30])))
    return D(str(rng.choice(AMTS_BIG)))


def hold_amt(rng, sym, amt):
    v = float(amt) * rng.choice([1, 1.1, 1.3, 1.6, 2.0, 2.7, 3.5])
    return D(str(round(v, 4 if sym in ("ETH", "SOL") else 2)))


def sym_word(rng, sym):
    return rng.choice(SYM_WORDS[sym])


def chain_word(rng, ch):
    return rng.choice(CHAIN_WORDS[ch])


# ------------------------------------------------------------------ shape 1: bridge, then deposit into Aave
B1S = [
    "bridge {a} {s} from {src} to {dst} then deposit it in aave",
    "move {a} {s} from {src} to {dst} and then put it into aave",
    "can you bridge {a} {s} {src} to {dst}, then deposit it into aave after",
    "i want to earn on aave on {dst}. bridge {a} {s} over from {src} first",
    "send {a} {s} from {src} over to {dst}, then supply it to aave v3",
    "bridge {a} {s} from {src} to {dst} and then lend it on aave",
    "take {a} {s} on {src}, bridge it to {dst}, deposit in aave there",
    "{a} {s} {src} to {dst} bridge, then aave deposit",
    "get {a} {s} from {src} onto {dst} and into aave",
    "step by step please: bridge {a} {s} from {src} to {dst}, then aave deposit",
    "i have {s} on {src} but aave pays better on {dst}. bridge {a} over and deposit it",
    "Bridge {a} {s} from {src} to {dst}, then deposit it to Aave.",
    "hey can u move {a} {s} from {src} to {dst} and park it in aave",
    "bridge then deposit: {a} {s}, {src} to {dst}, aave v3",
    "put {a} {s} from {src} into aave on {dst} (bridge it first)",
    "first bridge {a} {s} from {src} to {dst}, after that deposit it in aave",
]
B1N = [
    "bridge {a} {s} to {dst} then deposit it in aave",
    "move {a} {s} to {dst} and put it in aave",
    "can you bridge {a} {s} to {dst} and then deposit it into aave",
    "bridge {a} {s} to {dst}, then lend it on aave",
    "i want {a} {s} earning on aave on {dst}. bridge it there first",
    "get {a} {s} over to {dst} then aave deposit",
    "bridge {a} {s} to {dst} and supply it to aave v3",
    "put {a} {s} into aave on {dst}, bridge first",
    "bridge {a} {s} to {dst} then deposit in aave pls",
    "send {a} {s} to {dst} and then stake it in aave",
    "Bridge {a} {s} to {dst}, then deposit into Aave.",
    "bridge+deposit: {a} {s} to {dst} aave",
    "hey, {a} {s} to {dst} via bridge and then into aave pls",
    "need {a} {s} on {dst} first, then deposit it into aave",
    "bridge {a} {s} over to {dst}, once its there deposit it to aave",
]
B1F = [
    "bridge {a} {s} to {dst} then deposit half of it in aave",
    "move {a} {s} to {dst}, then put half of it into aave",
    "bridge {a} {s} from {src} to {dst} and deposit half of what arrives in aave",
    "send {a} {s} over to {dst} then lend half of it on aave",
    "bridge {a} {s} to {dst}, after that half of it goes into aave",
    "get {a} {s} onto {dst}, deposit half of it in aave",
    "bridge {a} {s} {src} -> {dst} then half into aave",
    "Bridge {a} {s} to {dst}, then deposit half of it to Aave.",
    "bridge {a} {s} to {dst} and then half of it into aave v3",
    "{a} {s} to {dst} via bridge, then supply half to aave",
]
ANS_FROM = ["from {src}", "{src}", "the {src} one", "from {src} pls", "use {src}", "{src} please", "take it from {src}",
            "{src}, the bigger one", "the one on {src}", "from my {src} balance"]


def sc_bridge_deposit(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    sym = rng.choices(["USDC", "USDT", "ETH"], [72, 10, 18])[0]
    dst = rng.choice([ch for ch in AAVE_CHAINS if sym in AAVE_SUPPORT[ch]])
    src = rng.choice([ch for ch in TOKEN_CHAINS[sym] if ch != dst and ch in QUOTE_CHAINS])
    amt = amt_dec(sym, rng)
    w.set_hold(sym, src, hold_amt(rng, sym, amt))
    add_noise_holdings(rng, w, {(sym, src), (sym, dst)})
    frac = rng.random() < 0.14
    named_src = rng.random() < 0.55
    ambiguous = (not named_src) and rng.random() < 0.1
    other = None
    if ambiguous:
        others = [ch for ch in TOKEN_CHAINS[sym] if ch not in (dst, src) and ch in QUOTE_CHAINS]
        if others:
            other = rng.choice(others)
            w.set_hold(sym, other, hold_amt(rng, sym, amt))
        else:
            ambiguous = False
    blocked0 = rng.random() < 0.08
    kw = dict(a=amt_user(rng, amt), s=sym_word(rng, sym), src=chain_word(rng, src), dst=chain_word(rng, dst))
    pool = B1F if frac else (B1S if named_src else B1N)
    if frac and not named_src:
        pool = [t for t in B1F if "{src}" not in t]
    first = P(rng, split, pool).format(**kw)
    if blocked0:
        w.s["force"]["prepare_swap"] = blocked_result(rng, "swap")
    old = old_rows(rng, w, rng.randint(1, 4))
    w.s["activity"] = old
    c.user(decorate(rng, first))
    if not named_src:
        bal = c.call1("get_balances", {})
        if ambiguous:
            a_, b_ = src, other
            q = (f"Which network should I bridge from? You hold {sym} on {CHAIN_NAME[a_]} ({dstr(w.hold(sym, a_))}) "
                 f"and {CHAIN_NAME[b_]} ({dstr(w.hold(sym, b_))}).")
            c.say(q)
            ans = rng.choice(ANS_FROM).format(src=chain_word(rng, src))
            c.user(decorate(rng, ans, first=False))
    th_blocked = blocked0
    swap = swap_leg(rng, bio, "bridge", sym, src, sym, dst, amt)
    dep = deposit_leg(rng, bio, sym, dst, size_bal=True, frac=0.5 if frac else None)
    legs = [swap, dep]
    seq = pick_seq(rng, 1)
    if rng.random() < 0.04 and not blocked0 and seq[0] == ["landed"]:
        dep["force"] = blocked_result(rng, "deposit")
    fin = pick_final(rng) if seq_complete(seq, 1) and not dep.get("force") else None
    tag = run_chain(c, rng, split, legs, seq, fin, c.clock)
    if tag == "blocked0" and rng.random() < 0.6:
        w.s["force"].pop("prepare_swap", None)
        if src in AAVE_CHAINS and sym in AAVE_SUPPORT[src] and rng.random() < 0.5:
            c.user(decorate(rng, rng.choice(["ok then just deposit it into aave on {src}", "fine, deposit it on {src} then",
                                             "ok skip the bridge, put it in aave on {src}"]).format(src=CHAIN_NAME[src].lower()), first=False))
            args = {"project": "aave-v3", "symbol": sym, "chain": src, "amount": num_json(amt)}
            res = c.call1("prepare_deposit", args)
            if ok_status(res):
                c.step("s1r", "prepare_deposit", args)
                c.say(f"I've prepared depositing {res['amount']} into {res['pool']} at {res['apy_pct']}% APY, which can change. "
                      f"Approve with {bio}. This skips the bridge, as you asked.")
        else:
            half = D(str(round(float(amt) / 2, 4 if sym == "ETH" else 2)))
            c.user(decorate(rng, rng.choice(["ok try {h}", "what about {h} then", "try with {h} instead", "ok do {h} only"]).format(h=dstr(half)), first=False))
            args = {"from_symbol": sym, "from_chain": src, "to_symbol": sym, "to_chain": dst, "amount": num_json(half)}
            res = c.call1("prepare_swap", args)
            if ok_status(res):
                c.step("s1r", "prepare_swap", args)
                c.say(announce_swap(rng, res, "bridge", bio) + " " + after_line(rng, "the Aave deposit"))
    return finish_money(c, "bridge_deposit", tag)


# ------------------------------------------------------------------ recipients: typed address, saved contact, unknown contact
CONTACT_NAMES = ["Alice", "Bob", "Maya", "Dad", "Priya", "Leo", "Sam", "Ken", "Noah", "Zara", "Diego", "Nina", "Omar"]
LATE_ADDR_T = [
    "ok {thing} is in. {name} is {addr}", "done, {name}'s address is {addr}", "landed. here is {name}: {addr}",
    "{thing} went through, address for {name} is {addr}", "approved, {name} = {addr}", "its in. send to {addr}",
    "done! {name}: {addr}", "ok approved. use {addr} for {name}", "{thing} landed, their address {addr}",
    "done, {addr} is {name}", "approved it. {name} -> {addr}", "all good on my side, {name}'s wallet is {addr}",
    "ok landed, the address is {addr}", "approved, {addr}", "yes {thing} is in, {name} is at {addr}",
]


def recipient_plan(rng, split, family="evm"):
    """Returns dict(mode, name, addr, say): how the user refers to the recipient."""
    r = rng.random()
    addr = H.evm_addr(rng) if family == "evm" else H.sol_addr(rng)
    if r < 0.62:
        return {"mode": "addr", "name": None, "addr": addr, "say": addr}
    name = rng.choice(CONTACT_NAMES)
    if r < 0.82:
        return {"mode": "found", "name": name, "addr": addr, "say": name}
    return {"mode": "missing", "name": name, "addr": addr, "say": name}


def plan_world(w, plan, family):
    if plan["mode"] == "found":
        w.s["contacts"].append({"name": plan["name"], "address": plan["addr"], "family": family})
    elif plan["mode"] == "missing":
        # a decoy contact with another name so a not_found is plausible
        w.s["contacts"].append({"name": rng_decoy(plan["name"]), "address": plan["addr"][:-1] + ("1" if plan["addr"][-1] != "1" else "2"),
                                "family": family})


def rng_decoy(name):
    return {"Alice": "Anna", "Bob": "Ben", "Maya": "Mia", "Dad": "Mum", "Priya": "Pooja", "Leo": "Liam", "Sam": "Sara",
            "Ken": "Kim", "Noah": "Ravi", "Zara": "Zoe", "Diego": "Dani", "Nina": "Noor", "Omar": "Owen"}.get(name, "Zed")


def contact_round(c, plan):
    """get_address for a saved name. Returns the result (found / not_found)."""
    return c.call1("get_address", {"contact": plan["name"]})


def late_tail(rng, plan, thing_noun):
    name = plan["name"]
    return rng.choice([
        f"I don't have an address saved for {name}. Paste it in your next message and I'll prepare {thing_noun} after this one lands.",
        f"There is no saved contact called {name}, so I need the address from you. Send it when you tell me this one has landed.",
        f"I can't find {name} in your contacts. Give me the address with your next message, after this one lands, and I'll do the send.",
    ])


# ------------------------------------------------------------------ shape 2: swap, then send
S2A = [
    "swap {a} {s} to {t} and send {k} of it to {addr}",
    "convert {a} {s} into {t}, then send {k} {t} to {addr}",
    "i need to pay {addr} {k} {t}. swap {a} {s} to {t} first",
    "swap {a} {s} for {t} then send {k} to {addr}",
    "turn {a} {s} into {t} and send {k} of the {t} to {addr}",
    "can you swap {a} {s} to {t} then transfer {k} to {addr}",
    "first swap {a} {s} to {t}, after that send {k} {t} to {addr}",
    "sell {a} {s} for {t} and send {k} {t} to {addr}",
    "{a} {s} -> {t}, then {k} {t} to {addr}",
    "swap {a} {s} to {t}, then send {k} {t} over to {addr} pls",
    "I want to swap {a} {s} into {t} and send {k} {t} to {addr}.",
    "swap {a} {s} to {t}{on}, once thats done send {k} to {addr}",
    "get me {t} for {a} {s} and then pay {addr} {k} {t}",
    "please swap {a} {s} to {t} and then send {k} {t} to {addr}",
    "exchange {a} {s} for {t}, then {addr} gets {k} {t}",
]
S2B = [
    "swap half my {s} to {t} and send {k} of it to {addr}",
    "convert half of my {s} to {t} then send {k} {t} to {addr}",
    "swap 50% of my {s} into {t}, then {k} of that to {addr}",
    "half my {s} to {t} and then send {k} to {addr}",
    "i want half my {s} swapped to {t}, then send {k} {t} to {addr}",
    "swap half of my {s} for {t} and send {addr} {k} of it",
    "can you turn half my {s} into {t} and send {k} {t} to {addr}",
    "first swap half my {s} to {t}, then send {k} of the {t} to {addr}",
    "swap half my {s} to {t}{on}, then send {k} to {addr}",
    "Convert half my {s} to {t}, then send {k} {t} to {addr}.",
]
S2C = [
    "swap {a} {s} to {t} and send all of it to {addr}",
    "convert {a} {s} into {t}, then send everything to {addr}",
    "swap {a} {s} for {t} then send all the {t} to {addr}",
    "turn {a} {s} into {t} and forward all of it to {addr}",
    "swap {a} {s} to {t}, then the whole amount goes to {addr}",
    "swap {a} {s} to {t}{on}, then send it all to {addr}",
    "{a} {s} to {t}, then send all of the {t} to {addr}",
    "swap {a} {s} to {t} and send the proceeds to {addr}",
    "Swap {a} {s} for {t}, then send all of it to {addr}.",
    "exchange {a} {s} for {t} and send the full amount to {addr} after",
]


def sc_swap_send(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    variant = rng.choices(["eth_usdc", "usdc_eth", "sol_usdc"], [64, 20, 16])[0]
    if variant == "eth_usdc":
        fs, ts, chain, fam = "ETH", "USDC", rng.choice(["base", "arbitrum", "ethereum"]), "evm"
    elif variant == "usdc_eth":
        fs, ts, chain, fam = "USDC", "ETH", rng.choice(["base", "arbitrum", "ethereum"]), "evm"
    else:
        fs, ts, chain, fam = "SOL", "USDC", "solana", "solana"
    style = rng.choices(["A", "B", "C"], [50, 25, 25])[0]
    plan = recipient_plan(rng, split, fam)
    plan_world(w, plan, fam)
    short_case = style == "A" and rng.random() < 0.22
    # amounts
    if style == "B":
        have = D(str(round(float(amt_dec(fs, rng)) * 2, 4 if fs != "USDC" else 2)))
        w.set_hold(fs, chain, have)
        swap_amt = None
        in_amt = have / 2
    elif short_case:
        k_ = rng.choice([100, 150, 200, 250, 300, 400, 500, 750, 1000] if ts == "USDC" else [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2])
        px_ = w.price(fs) / w.price(ts)            # units of ts per unit of fs: fs -> ts
        in_amt = D(str(round(k_ / px_ * rng.uniform(0.97, 0.993), 3 if fs == "ETH" else (2 if fs == "SOL" else 0))))
        if in_amt <= 0:
            raise Retry("zero input")
        w.set_hold(fs, chain, hold_amt(rng, fs, in_amt))
        swap_amt = in_amt
    else:
        in_amt = amt_dec(fs, rng)
        w.set_hold(fs, chain, hold_amt(rng, fs, in_amt))
        swap_amt = in_amt
    # the target token must not be held on the swap chain (so "all of it" is exactly the proceeds); no twin on others either
    add_noise_holdings(rng, w, {fs, ts})
    q = w.quote(fs, chain, ts, chain, in_amt)
    out_units = D(repr(q["_out_units"]))
    named_chain = rng.random() < 0.4
    # send amount
    if style == "C":
        send_amt, k_txt = None, None
    elif short_case:
        send_amt, k_txt = None, None
    else:
        lim = float(D(q["amount_out"]) * D("0.9"))
        nice = [x for x in ([5, 10, 15, 20, 25, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500] if ts == "USDC"
                            else [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1]) if x <= lim]
        if not nice:
            raise Retry("no nice send amount")
        send_amt = D(str(rng.choice(nice[-4:])))
        k_txt = dstr(send_amt)
    if short_case:
        send_amt = D(str(k_))
        k_txt = dstr(send_amt)
        if D(q["amount_out"]) >= send_amt - D("0.011"):
            raise Retry("not short")
    kw = dict(a=amt_user(rng, in_amt), s=sym_word(rng, fs), t=sym_word(rng, ts), k=k_txt or "", addr=plan["say"],
              on=(" on " + chain_word(rng, chain)) if named_chain else "")
    pool = {"A": S2A, "B": S2B, "C": S2C}[style]
    first = P(rng, split, pool).format(**kw)
    blocked0 = rng.random() < 0.07
    if blocked0:
        w.s["force"]["prepare_swap"] = blocked_result(rng, "swap")
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 4))
    c.user(decorate(rng, first))
    found = None
    if plan["mode"] in ("found", "missing"):
        found = contact_round(c, plan)
    # leg 0 inputs: half of the holding needs balances + calculator
    if style == "B" or (not named_chain and rng.random() < 0.0):
        pass
    leg0 = swap_leg(rng, bio, "swap", fs, chain, ts, chain, swap_amt if style != "B" else None, noun="the swap")
    inner = leg0["prep"]

    if style == "B" or not named_chain:
        def prep0(c_, check, inner=inner):
            bal = c_.call1("get_balances", {})
            have_s = find_bal(bal, fs, chain)
            a = D(have_s)
            if style == "B":
                r = c_.call1("calculator", {"expression": f"{have_s} / 2"})
                a = D(r["result"])
            elif swap_amt is not None:
                a = swap_amt
            args = {"from_symbol": fs, "from_chain": chain, "to_symbol": ts, "to_chain": chain, "amount": num_json(a)}
            res = c_.call1("prepare_swap", args)
            if not ok_status(res):
                return {"ok": False, "fail_text": fail_text(rng, c_, "swap", res, leg0), "res": res, "args": args}
            qq = c_.w.quote(fs, chain, ts, chain, a)
            landed = landing_amount(rng, qq, ts, c_.w)
            leg0["row"] = row_swap("swap", a, fs, chain, ts, chain)
            leg0["seconds"] = res["seconds"]
            leg0["apply_debit"] = lambda w_: w_.add_hold(fs, chain, -a)
            leg0["apply_credit"] = lambda w_: w_.add_hold(ts, chain, landed)
            return {"ok": True, "res": res, "args": args, "text": announce_swap(rng, res, "swap", bio)}
        leg0["prep"] = prep0
    # leg 1
    if plan["mode"] == "found":
        res_c = found
        addr = res_c["contacts"][0]["address"]
        who = plan["name"]
    elif plan["mode"] == "missing":
        addr, who = plan["addr"], plan["name"]
    else:
        addr, who = plan["addr"], None
    noun = f"the send to {who} ({addr})" if (plan["mode"] == "found") else (
        f"the send to {addr}" if plan["mode"] == "addr" else "the send")
    send = send_leg(rng, bio, ts, None, addr, amt=send_amt, size_bal_chain=chain, noun=noun, resolved_name=who if plan["mode"] == "found" else None,
                    nextv="send")
    if plan["mode"] == "missing":
        send["report_pool"] = LATE_ADDR_T
        send["late_addr"] = addr
        send["name"] = plan["name"]
        leg0["tail"] = lambda rng_: late_tail(rng_, plan, "the send")
    legs = [leg0, send]
    seq = pick_seq(rng, 1)
    if short_case:
        seq = [["landed"]]
    if rng.random() < 0.04 and not blocked0 and seq[0] == ["landed"] and not short_case:
        send["force"] = blocked_result(rng, "send")
    fin = pick_final(rng) if seq_complete(seq, 1) and not send.get("force") and not short_case else None
    tag = run_chain(c, rng, split, legs, seq, fin, c.clock)
    return finish_money(c, "swap_send", tag, {"contact": plan["mode"]})


# ------------------------------------------------------------------ shape 3: withdraw from Aave, then send or bridge
S3S = [
    "take my {s} out of aave and send it to {addr}",
    "withdraw my {s} from aave and then send it to {addr}",
    "pull everything out of aave{on} and send the {s} to {addr}",
    "i need my {s} from aave sent to {addr}",
    "withdraw all {s} from aave then transfer it to {addr}",
    "get my {s} out of aave and forward it to {addr}",
    "aave withdraw {s}{on}, then send it on to {addr}",
    "Withdraw my {s} from Aave and send it to {addr}.",
    "close my aave {s} position{on} and send the money to {addr}",
    "cash out my {s} from aave, then send it to {addr}",
    "pls withdraw {s} from aave and send to {addr} after",
    "empty my aave {s}{on} then send it all to {addr}",
    "first withdraw my {s} from aave, then send it to {addr}",
    "my {s} in aave should go to {addr}. withdraw then send",
    "move my aave {s} to {addr}: withdraw, then send",
]
S3P = [
    "take {a} {s} out of aave and send it to {addr}",
    "withdraw {a} {s} from aave{on}, then send it to {addr}",
    "pull {a} {s} from aave and send that to {addr}",
    "i need {a} {s} from aave sent to {addr}",
    "withdraw {a} {s} from aave then transfer it to {addr}",
    "get {a} {s} out of aave and forward it to {addr}",
    "Withdraw {a} {s} from Aave and send it to {addr}.",
    "aave: take out {a} {s}{on}, then send it to {addr}",
    "withdraw just {a} {s} from aave, then send those to {addr}",
    "pls take {a} {s} out of aave and send to {addr} after",
]
S3B = [
    "take my {s} out of aave and bridge it to {dst}",
    "withdraw my {s} from aave{on} then bridge it to {dst}",
    "get all my {s} out of aave and move it to {dst}",
    "withdraw {s} from aave and bridge everything to {dst}",
    "pull my {s} from aave, then bridge it over to {dst}",
    "Withdraw my {s} from Aave and bridge it to {dst}.",
    "aave withdraw {s}{on}, then bridge the lot to {dst}",
    "i want my aave {s} on {dst}. withdraw then bridge it",
    "cash out my {s} from aave and send it across to {dst}",
    "withdraw {s} from aave then bridge to {dst} pls",
]
ANS_WHICH = ["the {c} one", "{c}", "from {c}", "{c} pls", "the one on {c}", "{c}, please", "take it from {c}",
             "my {c} deposit", "{c} one", "go with {c}"]


def sc_withdraw_then(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    sym = rng.choices(["USDC", "USDT", "ETH"], [76, 10, 14])[0]
    ach = rng.choice([ch for ch in AAVE_CHAINS if sym in AAVE_SUPPORT[ch]])
    supplied = D(str(round(rng.choice([21.4, 40, 52.31, 75.5, 120, 180.27, 250, 410.6, 640, 1203.77, 1500, 2300.5])
                           * (1 if sym != "ETH" else 0.0008), 6 if sym == "ETH" else 6)))
    if sym == "ETH":
        supplied = D(str(rng.choice([0.05, 0.12, 0.25, 0.4, 0.75, 1.2, 2.5])))
    # accrued interest: the exact amount has more decimals than the row text
    if sym != "ETH" and rng.random() < 0.6:
        supplied = supplied + D(str(rng.choice([0.000413, 0.001872, 0.004219, 0.012345, 0.0000071])))
    w.set_aave(sym, ach, supplied)
    two_deposits = rng.random() < 0.08
    other = None
    if two_deposits:
        oc = [ch for ch in AAVE_CHAINS if ch != ach and sym in AAVE_SUPPORT[ch]]
        if oc:
            other = rng.choice(oc)
            w.set_aave(sym, other, D(str(rng.choice([10, 25, 40, 90]))))
        else:
            two_deposits = False
    to_bridge = rng.random() < 0.4
    partial = (not two_deposits) and (not to_bridge) and rng.random() < 0.28
    named = two_deposits is False and rng.random() < 0.35
    fam = "evm"
    plan = recipient_plan(rng, split, fam)
    if to_bridge:
        plan = {"mode": "addr", "name": None, "addr": None, "say": None}
    else:
        plan_world(w, plan, fam)
    dst = rng.choice([ch for ch in ["base", "arbitrum", "ethereum", "polygon"] if ch != ach]) if to_bridge else None
    add_noise_holdings(rng, w, {sym})
    amt = None
    if partial:
        cap_ = min(float(supplied), 5000.0)
        amt = D(str(rng.choice([x for x in ([10, 15, 20, 25, 40, 50, 75, 100, 150, 200, 500] if sym != "ETH" else [0.02, 0.05, 0.1, 0.2])
                                if x < cap_ * 0.9] or [10])))
    kw = dict(s=sym_word(rng, sym), addr=plan["say"] or "", dst=chain_word(rng, dst) if dst else "",
              on=(" on " + chain_word(rng, ach)) if named else "", a=amt_user(rng, amt) if amt else "")
    pool = S3B if to_bridge else (S3P if partial else S3S)
    first = P(rng, split, pool).format(**kw)
    blocked0 = rng.random() < 0.06 and not two_deposits
    if blocked0:
        w.s["force"]["prepare_withdraw"] = blocked_result(rng, "withdraw")
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 4))
    c.user(decorate(rng, first))
    chain_arg = ach if named else None
    if two_deposits:
        res = c.call1("prepare_withdraw", {"symbol": sym})
        assert res["status"] == "needs_choice", res
        c.say(res["question"])
        pick_ch = rng.choice([ach, other])
        ach = pick_ch
        c.user(decorate(rng, P(rng, split, ANS_WHICH).format(c=chain_word(rng, ach)), first=False))
        chain_arg = ach
    found = None
    if plan["mode"] in ("found", "missing"):
        found = contact_round(c, plan)
    wd = withdraw_leg(rng, bio, sym, chain_arg, amt, ach)
    if to_bridge:
        nxt = swap_leg(rng, bio, "bridge", sym, ach, sym, dst, None, bal_chain=ach, size_from_bal=(sym, ach), noun="the bridge")
    else:
        if plan["mode"] == "found":
            addr, who = found["contacts"][0]["address"], plan["name"]
        else:
            addr, who = plan["addr"], plan["name"]
        noun = f"the send to {who} ({addr})" if plan["mode"] == "found" else (f"the send to {addr}" if plan["mode"] == "addr" else "the send")
        nxt = send_leg(rng, bio, sym, None, addr, amt=None, size_bal_chain=ach, noun=noun,
                       resolved_name=who if plan["mode"] == "found" else None)
        if plan["mode"] == "missing":
            nxt["report_pool"] = LATE_ADDR_T
            nxt["late_addr"] = addr
            nxt["name"] = plan["name"]
            wd["tail"] = lambda rng_: late_tail(rng_, plan, "the send")
    legs = [wd, nxt]
    seq = pick_seq(rng, 1)
    if rng.random() < 0.04 and not blocked0 and seq[0] == ["landed"]:
        nxt["force"] = blocked_result(rng, "send" if not to_bridge else "swap")
    fin = pick_final(rng) if seq_complete(seq, 1) and not nxt.get("force") else None
    tag = run_chain(c, rng, split, legs, seq, fin, c.clock)
    return finish_money(c, "withdraw_then", tag, {"second": "bridge" if to_bridge else "send", "contact": plan["mode"]})


# ------------------------------------------------------------------ shape 5: sell a Solana token, then send the USDC
S5 = [
    "sell {q} {tok} and send the usdc to {addr}",
    "sell {q} {tok} for usdc, then send it to {addr}",
    "cash out {q} {tok} into usdc and send that to {addr}",
    "dump {q} {tok} and send the usdc to {addr}",
    "i want to sell {q} {tok} and send the usdc to {addr}",
    "sell {q} {tok}, send the usdc proceeds to {addr}",
    "Sell {q} {tok} for USDC and send it to {addr}.",
    "swap {q} {tok} to usdc then send the usdc to {addr}",
    "sell {q} {tok}, then pay {addr} with the usdc",
    "turn {q} {tok} into usdc and send it to {addr} after",
    "pls sell {q} {tok} to usdc and send to {addr}",
    "first sell {q} {tok} for usdc, after that send the usdc to {addr}",
    "take profit on {q} {tok}: sell into usdc and send the usdc to {addr}",
    "sell {q} {tok} -> usdc -> {addr}",
    "get rid of {q} {tok}, usdc goes to {addr}",
]


def sc_sell_send(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    tok = rng.choice(SPOT_TOKENS)
    px = w.s["prices"][tok]
    usd_val = rng.choice([40, 80, 150, 300, 600, 1200, 2500])
    have = D(str(round(usd_val / px, 2 if px >= 0.5 else 0)))
    if have <= 0:
        raise Retry("zero holding")
    w.s["spot"].append({"amount": dstr(have), "avg_cost": round(px * rng.uniform(0.7, 1.4), 4), "frozen": False,
                        "impact": rng.choice([0.03, 0.04, 0.06, 0.09, 0.12]), "symbol": tok, "usd": round(float(have) * px, 2)})
    add_noise_holdings(rng, w, {"USDC", "SOL"})
    w.set_hold("SOL", "solana", D("0.35"))
    mode = rng.choices(["all", "half", "quarter", "pct", "amount"], [38, 24, 8, 10, 20])[0]
    percent, amount = None, None
    if mode == "all":
        percent, q = 100, rng.choice(["all my", "all of my", "my whole", "everything I have in", "the whole bag of"])
    elif mode == "half":
        percent, q = 50, rng.choice(["half my", "half of my", "50% of my"])
    elif mode == "quarter":
        percent, q = 25, rng.choice(["a quarter of my", "25% of my"])
    elif mode == "pct":
        percent = rng.choice([10, 20, 30, 60, 75, 80])
        q = f"{percent}% of my"
    else:
        amt_ = D(str(round(float(have) * rng.choice([0.2, 0.3, 0.5, 0.6]), 0 if float(have) > 30 else 2)))
        if amt_ <= 0:
            raise Retry("zero amount")
        amount = amt_
        q = f"{dstr(amt_)}"
    plan = recipient_plan(rng, split, "solana")
    plan_world(w, plan, "solana")
    blocked0 = rng.random() < 0.07
    if blocked0:
        w.s["spot"][0]["frozen"] = True
    kw = dict(q=q, tok=tok if rng.random() < 0.7 else tok.lower(), addr=plan["say"])
    first = P(rng, split, S5).format(**kw)
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 4))
    c.user(decorate(rng, first))
    extra = [("get_address", {"contact": plan["name"]})] if plan["mode"] in ("found", "missing") else None
    sl = sell_leg(rng, bio, tok, percent, amount, "USDC", extra_calls=extra)
    inner = sl["prep"]

    def prep0(c_, check, inner=inner):
        out = inner(c_, check)
        return out
    if plan["mode"] == "found":
        addr, who = plan["addr"], plan["name"]
    else:
        addr, who = plan["addr"], plan["name"]
    noun = f"the send to {who} ({addr})" if plan["mode"] == "found" else (f"the send to {addr}" if plan["mode"] == "addr" else "the send")
    send = send_leg(rng, bio, "USDC", None, addr, amt=None, size_bal_chain="solana", noun=noun,
                    resolved_name=who if plan["mode"] == "found" else None)
    if plan["mode"] == "missing":
        send["report_pool"] = LATE_ADDR_T
        send["late_addr"] = addr
        send["name"] = plan["name"]
        sl["tail"] = lambda rng_: late_tail(rng_, plan, "the send")
    legs = [sl, send]
    seq = pick_seq(rng, 1)
    fin = pick_final(rng) if seq_complete(seq, 1) else None
    tag = run_chain(c, rng, split, legs, seq, fin, c.clock)
    return finish_money(c, "sell_send", tag, {"contact": plan["mode"]})


# ------------------------------------------------------------------ shape 7: three steps
S7A = [
    "withdraw my {s} from aave on {c1}, bridge it to {c2}, then deposit it again there",
    "take my {s} out of aave on {c1}, bridge to {c2} and deposit it back into aave",
    "move my aave {s} from {c1} to {c2}: withdraw, bridge, deposit again",
    "i want my aave {s} on {c2} instead of {c1}. withdraw it, bridge it over, then deposit",
    "withdraw {s} from aave {c1}, bridge it to {c2}, and put it in aave again",
    "Withdraw my {s} from Aave on {c1}, bridge to {c2}, deposit it again.",
    "aave {c1} -> aave {c2} for my {s}: withdraw, bridge, deposit",
    "relocate my aave {s} from {c1} to {c2} (withdraw, then bridge, then deposit)",
    "withdraw {s} on {c1}, bridge it over to {c2}, re-deposit in aave",
    "pls move all my {s} in aave from {c1} to {c2}, 3 steps: withdraw, bridge, deposit",
]
S7B = [
    "swap {a} {fs} to {s} on {c1}, bridge the {s} to {c2}, then deposit it in aave",
    "convert {a} {fs} into {s} on {c1}, then bridge to {c2} and put it into aave",
    "i want {a} {fs} earning on aave {c2}: swap to {s} on {c1}, bridge, then deposit",
    "swap {a} {fs} for {s}, bridge to {c2}, deposit in aave there. i'm on {c1}",
    "{a} {fs} on {c1} -> {s}, bridge to {c2}, then aave deposit",
    "Swap {a} {fs} to {s} on {c1}, bridge it to {c2}, then deposit it to Aave.",
    "turn {a} {fs} on {c1} into {s}, move it to {c2} and lend it on aave",
    "first swap {a} {fs} to {s} on {c1}, then bridge to {c2}, then deposit in aave",
    "do these in order: swap {a} {fs} to {s} on {c1}, bridge to {c2}, deposit into aave",
    "get {a} {fs} from {c1} into aave on {c2}: swap, bridge, deposit",
]
S7C = [
    "withdraw my {s} from aave on {c1}, swap it to eth, then send the eth to {addr}",
    "take my {s} out of aave on {c1}, swap to eth and send that to {addr}",
    "aave {c1} withdraw {s}, swap to eth, send the eth to {addr}",
    "Withdraw my {s} from Aave on {c1}, swap it to ETH, send the ETH to {addr}.",
    "i need eth sent to {addr}: pull my {s} from aave on {c1}, swap it, then send",
    "withdraw {s} from aave ({c1}), convert to eth, forward the eth to {addr}",
    "get my {s} out of aave on {c1}, turn it into eth and send it to {addr}",
    "withdraw, swap, send: my aave {s} on {c1} to eth to {addr}",
    "first withdraw {s} from aave on {c1}, then swap to eth, then send that eth to {addr}",
    "pls take my {s} from aave on {c1}, swap it to eth and send to {addr}",
]


def sc_three_steps(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    variant = rng.choices(["a", "b", "c"], [50, 28, 22])[0]
    sym = "USDC" if variant != "a" or rng.random() < 0.85 else "USDT"
    if variant == "c":
        sym = "USDC"
    c1 = rng.choice(["base", "arbitrum", "ethereum"] if sym == "USDC" else ["arbitrum", "ethereum"])
    c2 = rng.choice([x for x in AAVE_CHAINS if x != c1 and sym in AAVE_SUPPORT[x]])
    if variant == "c":
        c2 = None
    kw = dict(s=sym_word(rng, sym), c1=chain_word(rng, c1), c2=chain_word(rng, c2) if c2 else "")
    blocked0 = rng.random() < 0.05
    if variant == "a":
        supplied = D(str(rng.choice([35, 52.31, 80, 120.5, 240, 410, 760, 1203.77]))) + D(str(rng.choice([0, 0.000413, 0.002871, 0.01])))
        w.set_aave(sym, c1, supplied)
        add_noise_holdings(rng, w, {sym})
        first = P(rng, split, S7A).format(**kw)
        wd = withdraw_leg(rng, bio, sym, c1, None, c1)
        br = swap_leg(rng, bio, "bridge", sym, c1, sym, c2, None, bal_chain=c1, size_from_bal=(sym, c1), noun="the bridge")
        dep = deposit_leg(rng, bio, sym, c2, size_bal=True)
        legs = [wd, br, dep]
        w.s["force"].update({"prepare_withdraw": blocked_result(rng, "withdraw")} if blocked0 else {})
    elif variant == "b":
        fs = "ETH"
        a_ = D(str(rng.choice([0.2, 0.3, 0.5, 0.75, 1, 1.5, 2])))
        w.set_hold(fs, c1, hold_amt(rng, fs, a_))
        add_noise_holdings(rng, w, {"ETH", sym})
        kw.update(a=dstr(a_), fs=sym_word(rng, fs))
        first = P(rng, split, S7B).format(**kw)
        sw = swap_leg(rng, bio, "swap", fs, c1, sym, c1, a_, noun="the swap")
        br = swap_leg(rng, bio, "bridge", sym, c1, sym, c2, None, bal_chain=c1, size_from_bal=(sym, c1), noun="the bridge")
        dep = deposit_leg(rng, bio, sym, c2, size_bal=True)
        legs = [sw, br, dep]
        if blocked0:
            w.s["force"]["prepare_swap"] = blocked_result(rng, "swap")
    else:
        supplied = D(str(rng.choice([150, 240, 410, 760, 1203.77, 2500]))) + D(str(rng.choice([0, 0.001872, 0.004219])))
        w.set_aave(sym, c1, supplied)
        add_noise_holdings(rng, w, {sym, "ETH"})
        addr = H.evm_addr(rng)
        kw.update(addr=addr)
        first = P(rng, split, S7C).format(**kw)
        wd = withdraw_leg(rng, bio, sym, c1, None, c1)
        sw = swap_leg(rng, bio, "swap", sym, c1, "ETH", c1, None, bal_chain=c1, size_from_bal=(sym, c1), noun="the swap to ETH")
        sd = send_leg(rng, bio, "ETH", None, addr, amt=None, size_bal_chain=c1, noun=f"the send of the ETH to {addr}")
        legs = [wd, sw, sd]
        if blocked0:
            w.s["force"]["prepare_withdraw"] = blocked_result(rng, "withdraw")
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
    c.user(decorate(rng, first))
    seq = pick_seq(rng, 2)
    long_run = rng.random() < 0.12 and not blocked0
    if long_run:
        seq = [["pending", "landed"], rng.choice([["missing", "landed"], ["pending", "landed"]])]
    if len(seq) == 2 and seq[1] == ["landed"] and rng.random() < 0.05 and not blocked0 and not long_run:
        legs[2]["force"] = blocked_result(rng, "deposit" if variant != "c" else "send")
    complete = seq_complete(seq, 2)
    fin = pick_final(rng) if complete and not legs[2].get("force") else None
    if long_run:
        fin = "landed"
    tag = run_chain(c, rng, split, legs, seq, fin, c.clock)
    return finish_money(c, "three_steps", tag, {"variant": variant})


# ------------------------------------------------------------------ Hyperliquid helpers
HL_COINS = ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "LINK", "DOGE", "SUI", "XRP"]


def make_position(rng, w, coin, side, notional, lev):
    px = w.s["prices"][coin]
    entry = round(px * rng.uniform(0.94, 1.06), 2 if px >= 10 else (4 if px >= 0.1 else 5))
    size = float(f"{notional / entry:.3g}")
    liq = round(entry * (1 - 0.9 / lev) if side == "long" else entry * (1 + 0.9 / lev), 2 if px >= 10 else 4)
    return {"coin": coin, "entry": entry, "leverage": lev, "liquidation": liq, "side": side, "size": size}


def add_hl(rng, w, n_pos, coins=None, avail=None):
    coins = coins or rng.sample(HL_COINS, n_pos)
    for coin in coins[:n_pos]:
        side = rng.choice(["long", "short"])
        lev = rng.choice([2, 3, 5, 5, 10])
        w.s["hl"]["positions"].append(make_position(rng, w, coin, side, rng.choice([300, 600, 1000, 2000, 3500, 5000]), min(lev, PERP_MAX_LEV[coin])))
    w.s["hl"]["available"] = avail if avail is not None else round(rng.uniform(40, 1500), 2)


def pos_margin_pnl(w, p):
    mark = w.s["prices"][p["coin"]]
    sgn = 1 if p["side"] == "long" else -1
    return p["size"] * p["entry"] / p["leverage"], (mark - p["entry"]) * p["size"] * sgn


def close_in_world(w, coin, percent):
    for p in w.s["hl"]["positions"]:
        if p["coin"] == coin:
            frac = (percent or 100) / 100
            margin, pnl = pos_margin_pnl(w, p)
            w.s["hl"]["available"] = round(w.s["hl"]["available"] + (margin + pnl) * frac, 2)
            if frac >= 1:
                w.s["hl"]["positions"].remove(p)
            else:
                p["size"] = float(f"{p['size'] * (1 - frac):.6g}")
            return


S4 = [
    "close my {coin} {side} and withdraw the money from hyperliquid",
    "close {coin} and then withdraw everything from hyperliquid",
    "i want out of hyperliquid: close my {coin} position, then withdraw",
    "close my {coin} {side} then move the usdc back to my wallet",
    "close the {coin} trade and withdraw to my wallet",
    "shut my {coin} {side}, then withdraw from hyperliquid",
    "Close my {coin} position and withdraw my funds from Hyperliquid.",
    "take profit: close {coin} and pull the usdc out of hyperliquid",
    "close my {coin} {side}, after that withdraw the usdc from hyperliquid please",
    "exit my {coin} position and cash out of hyperliquid",
    "can you close {coin} and withdraw from hl",
    "first close my {coin} {side}, then withdraw the balance from hyperliquid to my wallet",
    "close {coin} then withdraw. thats the plan",
    "get me out of my {coin} {side} and send the usdc from hyperliquid back to my wallet",
    "close out {coin} and withdraw all usdc from hyperliquid",
]
S4V = [
    "close my biggest position and withdraw from hyperliquid",
    "close my {side} position and then withdraw the usdc from hyperliquid",
    "close the position with the biggest loss, then withdraw from hyperliquid",
    "close my most profitable position and withdraw the cash",
    "shut my largest trade and pull the money out of hyperliquid",
    "close whatever is up the most, then withdraw from hyperliquid",
    "Close my biggest position, then withdraw from Hyperliquid.",
    "close my {side} and withdraw from hl",
    "close my largest position then withdraw the usdc to my wallet",
    "take my best performing position off and withdraw after",
]
S4P = [
    "close half my {coin} {side} and withdraw from hyperliquid",
    "close {pct}% of my {coin} {side}, then withdraw the usdc",
    "reduce my {coin} {side} by {pct}% and then withdraw from hyperliquid",
    "take {pct}% off my {coin} position and withdraw the proceeds from hyperliquid",
    "close a quarter of my {coin} {side} then withdraw",
    "Close {pct}% of my {coin} position and withdraw from Hyperliquid.",
    "close {pct} percent of {coin}, then pull that out of hyperliquid",
    "trim my {coin} {side} by {pct}%, then withdraw the cash",
    "close half of {coin} then withdraw from hl",
    "cut my {coin} {side} in half and withdraw the usdc after",
]
CLOSED_T = [
    "closed", "its closed", "closed it, now the withdraw", "done, position is closed", "ok closed. withdraw next",
    "approved the close", "ok closed, now withdraw", "the close went through", "done closing", "position closed, what now",
    "ok close is done", "approved, now withdraw it", "closed!", "close done", "ok the close is approved",
]
WD_ASK = [
    "ok withdraw it for me", "can you withdraw {w} usdc now", "so how do i withdraw", "withdraw it pls", "ok now withdraw",
    "go ahead and withdraw", "withdraw the usdc to my wallet", "do the withdrawal", "now take the money out",
    "send it to my wallet",
]


def sc_close_withdraw(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    mode = rng.choices(["named", "vague", "partial"], [60, 25, 15])[0]
    n_pos = rng.choice([1, 1, 2, 3])
    if mode == "vague":
        n_pos = max(n_pos, 2)
    add_hl(rng, w, n_pos)
    pos = w.s["hl"]["positions"]
    add_noise_holdings(rng, w, set())
    # choose the target
    hint = None
    pred = None
    if mode == "vague":
        pnl = {p["coin"]: pos_margin_pnl(w, p)[1] for p in pos}
        notional = {p["coin"]: p["size"] * p["entry"] for p in pos}
        hint = rng.choice(["biggest", "loss", "profit", "side"])
        if hint == "biggest":
            target = max(pos, key=lambda p: notional[p["coin"]])
            pred = lambda t: "biggest" in t.lower() or "largest" in t.lower()
        elif hint == "loss":
            target = min(pos, key=lambda p: pnl[p["coin"]])
            pred = lambda t: "loss" in t
        elif hint == "profit":
            target = max(pos, key=lambda p: pnl[p["coin"]])
            pred = lambda t: "profitable" in t or "up the most" in t or "best performing" in t
        else:
            sides = Counter(p["side"] for p in pos)
            uniq = [sd for sd, cnt in sides.items() if cnt == 1]
            if not uniq:
                raise Retry("no unique side")
            target = [p for p in pos if p["side"] == uniq[0]][0]
            pred = lambda t: "{side}" in t
        if len({round(v, 2) for v in pnl.values()}) < len(pnl) or len({round(v) for v in notional.values()}) < len(notional):
            raise Retry("tie")
    else:
        target = rng.choice(pos)
    coin, side = target["coin"], target["side"]
    percent = None
    kw = dict(coin=coin if rng.random() < 0.75 else coin.lower(), side=side)
    if mode == "named":
        tpl = P(rng, split, S4)
    elif mode == "partial":
        tpl = P(rng, split, S4P)
        if "half" in tpl:
            percent = 50
        elif "quarter" in tpl:
            percent = 25
        else:
            percent = rng.choice([20, 30, 40, 60, 75])
    else:
        cand = [t for i, t in enumerate(S4V) if (i % 5 == 0) == (split == "eval") and pred(t)]
        if not cand:
            raise Retry("no vague template for this hint")
        tpl = rng.choice(cand)
    kw["pct"] = percent
    first = tpl.format(**kw)
    blocked0 = False
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
    c.user(decorate(rng, first))
    shown = rng.random() < 0.5
    lead = ""
    if mode == "vague":
        acct = c.call1("hyperliquid_account", {})
        if hint == "biggest":
            c.call(*[("calculator", {"expression": f"{dstr(q['size'])} * {dstr(q['entry'])}"}) for q in acct["positions"]])
            lead = f"The {coin} {side} is your biggest position by value. "
        elif hint == "loss":
            lp = [q for q in acct["positions"] if q["coin"] == coin][0]
            lead = f"The {coin} {side} has the biggest loss ({usd(-lp['unrealised_pnl'])}). " if lp["unrealised_pnl"] < 0 else f"The {coin} {side} is your weakest position (up {usd(lp['unrealised_pnl'])}). "
        elif hint == "profit":
            lp = [q for q in acct["positions"] if q["coin"] == coin][0]
            lead = f"The {coin} {side} is your most profitable position (up {usd(lp['unrealised_pnl'])}). " if lp["unrealised_pnl"] >= 0 else f"The {coin} {side} is your least bad position (down {usd(-lp['unrealised_pnl'])}). "
        else:
            lead = f"Your only {side} is the {coin}. "
    args = {"coin": coin}
    if percent:
        args["percent"] = percent
    calls = [("prepare_close", args)] + ([("open_page", {"page": "wallets"})] if shown else [])
    outs = c.call(*calls)
    res = outs[0]
    assert ok_status(res)
    c.step("s1", "prepare_close", args)
    pnl = res["unrealised_pnl"]
    pnl_txt = ""
    if rng.random() < 0.45 and hint not in ("loss", "profit"):
        pnl_txt = (f" It carries an unrealised profit of {usd(pnl)}." if pnl >= 0 else f" It carries an unrealised loss of {usd(-pnl)}.")
    base = rng.choice([
        f"I've prepared closing {res['closing']} ({res['order']}), waiting for your {bio}.",
        f"The close is prepared: {res['closing']} ({res['order']}). Approve it with {bio}.",
        f"I've prepared closing {res['closing']} at market, reduce only. It's waiting for your {bio}."]) + pnl_txt
    if shown:
        nxt = rng.choice([
            "I can't prepare the Hyperliquid withdrawal myself, there is no tool for it. After the close is approved, use the Wallets button below for that step.",
            "The withdrawal from Hyperliquid isn't something I can prepare. I put a button below that opens Wallets, where you do it once the close is approved.",
            "I have no tool for the Hyperliquid withdrawal, so that step is yours: the button below opens Wallets. Do it after the close is approved."])
    else:
        nxt = rng.choice([
            "I can't prepare the Hyperliquid withdrawal myself. Tell me once the close is approved and I'll check the account and point you to Wallets.",
            "There is no tool for withdrawing from Hyperliquid. Tell me when the close is approved, I'll check the account, then show you where to withdraw.",
            "The withdrawal is a step I can't prepare. Once the close is approved, tell me and I'll check your account and put a Wallets button here."])
    c.say(lead + base + " " + nxt)
    # report turn
    oc = rng.choices(["closed", "notyet", "notyet_closed"], [64, 16, 20])[0]
    if oc != "closed":
        pass
    for j, oo in enumerate(["notyet", "closed"] if oc == "notyet_closed" else [oc]):
        if oo == "closed":
            close_in_world(w, coin, percent)
        c.user(user_followup(rng, split, CLOSED_T if j == 0 else AGAIN_T, thing="the close"))
        acct = c.call1("hyperliquid_account", {})
        if oo == "notyet":
            p = [q for q in acct["positions"] if q["coin"] == coin][0]
            c.say(rng.choice([
                f"I still see your {coin} {p['side']} open at {dstr(p['size'])} {coin}, so the close hasn't been approved yet. Approve the review with {bio}, then tell me. I'm not pointing you to the withdrawal until it is closed.",
                f"The {coin} {p['side']} is still open ({dstr(p['size'])} {coin}). The close needs your {bio} first; tell me when you've approved it.",
                f"Your account still shows the {coin} position at {dstr(p['size'])} {coin}. Nothing has closed yet, so there's nothing to withdraw. Approve the review, then tell me."]))
            continue
        rem = [q for q in acct["positions"] if q["coin"] == coin]
        avail = acct["available_usdc"]
        if not rem:
            state = rng.choice([f"Your {coin} {side} is gone from the account", f"The {coin} position no longer shows in your account"])
        else:
            state = f"Your {coin} {side} is down to {dstr(rem[0]['size'])} {coin}"
        if shown:
            tail = rng.choice([f"{avail} USDC is available on Hyperliquid. I have no tool to withdraw it, so use the Wallets button I showed above.",
                               f"You have {avail} USDC available there. The withdrawal itself isn't something I can prepare; the Wallets button above is where you do it."])
            c.say(f"{state}, and {tail}")
        else:
            c.call1("open_page", {"page": "wallets"})
            c.say(f"{state}, and {avail} USDC is available on Hyperliquid. I can't prepare the withdrawal, so I put a button below that opens Wallets, where you do it.")
    if oc != "notyet" and rng.random() < 0.3:
        c.user(decorate(rng, P(rng, split, WD_ASK).format(w=rng.choice([100, 200, 300, 50])), first=False))
        c.call1("open_page", {"page": "wallets"})
        c.say(rng.choice([
            "I can't withdraw from Hyperliquid, there is no tool for it. I put a button below that opens Wallets, where you can do it.",
            "Withdrawing from Hyperliquid is not something I can prepare. The button below opens Wallets; you start it there.",
            "That step has no tool on my side. Tap the button below to open Wallets and withdraw from there."]))
    tag = "full" if oc != "notyet" else "open0"
    return finish_money(c, "close_withdraw", tag, {"mode": mode, "pointer_first": shown})


# ------------------------------------------------------------------ shape 6: independent steps
S6A = [
    "send {a1} {t1} to {x1} and {a2} {t2} to {x2}",
    "i need to pay two people: {a1} {t1} to {x1}, and {a2} {t2} to {x2}",
    "send {a1} {t1} to {x1} then {a2} {t2} to {x2}",
    "pay {x1} {a1} {t1} and {x2} {a2} {t2}",
    "transfer {a1} {t1} to {x1} and also {a2} {t2} to {x2}",
    "Send {a1} {t1} to {x1} and {a2} {t2} to {x2}.",
    "two sends pls: {a1} {t1} -> {x1}, {a2} {t2} -> {x2}",
    "{x1} gets {a1} {t1}, {x2} gets {a2} {t2}. send both",
    "can you send {a1} {t1} to {x1} and {a2} {t2} to {x2} for me",
    "send {a1} {t1} to {x1}. also send {a2} {t2} to {x2}",
]
S6M = [
    "send {a1} {t1} to {x1} and {a2} {t1} to {x2}",
    "pay {x1} {a1} {t1} and {x2} {a2} {t1}",
    "{a1} {t1} to {x1} and {a2} {t1} to {x2}",
    "send {a1} {t1} to {x1}, {a2} {t1} to {x2}",
    "Send {a1} {t1} to {x1} and {a2} {t1} to {x2}.",
    "i want to send {a1} {t1} to {x1} and {a2} {t1} to {x2}",
    "split it: {x1} gets {a1} {t1}, {x2} gets {a2} {t1}",
    "transfer {a1} {t1} to {x1} plus {a2} {t1} to {x2}",
    "pls send {x1} {a1} {t1} and {x2} {a2} {t1}",
    "can you send {a1} {t1} to {x1} and {a2} {t1} to {x2}",
]
S6S = [  # send then swap
    "send {a1} {t1} to {x1} and swap {a2} {t2} to {t3}",
    "send {x1} {a1} {t1}, and also swap {a2} {t2} for {t3}",
    "i need to send {a1} {t1} to {x1} and swap {a2} {t2} to {t3}",
    "Send {a1} {t1} to {x1} and swap {a2} {t2} to {t3}.",
    "pay {x1} {a1} {t1} and convert {a2} {t2} into {t3}",
    "swap {a2} {t2} to {t3} and send {a1} {t1} to {x1}",
    "send {a1} {t1} to {x1}. then swap {a2} {t2} to {t3}",
    "two things: send {a1} {t1} to {x1}, swap {a2} {t2} to {t3}",
    "{a1} {t1} to {x1}, and {a2} {t2} -> {t3} swap",
    "please send {a1} {t1} to {x1} and swap {a2} {t2} into {t3}",
]
S6D = [  # deposit and send
    "deposit {a2} {t1} into aave and send {a1} {t1} to {x1}",
    "send {a1} {t1} to {x1} and put {a2} {t1} in aave",
    "put {a2} {t1} into aave, and send {a1} {t1} to {x1}",
    "Deposit {a2} {t1} into Aave and send {a1} {t1} to {x1}.",
    "lend {a2} {t1} on aave and pay {x1} {a1} {t1}",
    "send {x1} {a1} {t1} and deposit {a2} {t1} in aave",
    "aave deposit {a2} {t1}, and {a1} {t1} to {x1}",
    "i want to send {a1} {t1} to {x1} and also deposit {a2} {t1} into aave",
    "{a1} {t1} to {x1} and {a2} {t1} into aave",
    "please deposit {a2} {t1} in aave and send {a1} {t1} to {x1}",
]
S63 = [
    "send {a1} {t1} to {x1}, {a2} {t2} to {x2} and {a3} {t3} to {x3}",
    "three payments: {a1} {t1} to {x1}, {a2} {t2} to {x2}, {a3} {t3} to {x3}",
    "pay {x1} {a1} {t1}, {x2} {a2} {t2} and {x3} {a3} {t3}",
    "Send {a1} {t1} to {x1}, {a2} {t2} to {x2}, and {a3} {t3} to {x3}.",
    "{a1} {t1} to {x1}, {a2} {t2} to {x2}, {a3} {t3} to {x3}. send them all",
    "i need three sends: {x1} {a1} {t1}, {x2} {a2} {t2}, {x3} {a3} {t3}",
    "send {a1} {t1} to {x1} and {a2} {t2} to {x2} and {a3} {t3} to {x3}",
    "send {a1} {t1} to {x1}, then {a2} {t2} to {x2}, then {a3} {t3} to {x3}",
    "pls pay: {a1} {t1} to {x1}; {a2} {t2} to {x2}; {a3} {t3} to {x3}",
    "can you send {a1} {t1} to {x1}, {a2} {t2} to {x2} and {a3} {t3} to {x3}",
]
SEND_AMTS = {"USDC": [5, 10, 15, 20, 25, 40, 50, 75, 100, 150, 200], "USDT": [5, 10, 20, 25, 30, 50, 80, 100, 120],
             "ETH": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15], "DAI": [10, 20, 50, 100]}
SEND_CHAIN = {"USDC": "base", "USDT": "arbitrum", "ETH": "base", "DAI": "ethereum"}


def indep_line(rng, nxt):
    return rng.choice([
        f"The app shows one review at a time, so {nxt} comes next. Tell me when this one is approved.",
        f"Next is {nxt}. I'll prepare it once you tell me this one is approved and Activity shows it.",
        f"{cap(nxt)} is next; the app opens one review at a time. Tell me when you've approved this one.",
        f"I'm doing them one at a time. After you approve this, tell me and I'll prepare {nxt}.",
    ])


def sc_independent(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    variant = rng.choices(["two", "multi", "send_swap", "dep_send", "three"], [34, 16, 16, 16, 18])[0]
    syms = rng.sample(["USDC", "USDT", "ETH"], 3)
    addrs = [H.evm_addr(rng) for _ in range(3)]

    def amt_for(sym):
        return D(str(rng.choice(SEND_AMTS[sym])))

    legs, steps_txt = [], None
    kw = {}
    blocked0 = False
    if variant in ("two", "three"):
        n = 2 if variant == "two" else 3
        use = syms[:n]
        amts = [amt_for(s) for s in use]
        for s, a in zip(use, amts):
            w.set_hold(s, SEND_CHAIN[s], hold_amt(rng, s, a))
        for i in range(n):
            kw.update({f"a{i + 1}": dstr(amts[i]), f"t{i + 1}": sym_word(rng, use[i]), f"x{i + 1}": addrs[i]})
        pool = S6A if n == 2 else S63
        first = P(rng, split, pool).format(**kw)
        for i in range(n):
            legs.append(send_leg(rng, bio, use[i], None, addrs[i], amt=amts[i], size_bal_chain=None,
                                 noun=f"the send of {dstr(amts[i])} {use[i]} to {addrs[i]}"))
            legs[-1]["th"] = ["the first send", "the second send", "the third send"][i]
    elif variant == "multi":
        s = rng.choice(["USDC", "USDC", "USDT"])
        a1, a2 = amt_for(s), amt_for(s)
        w.set_hold(s, SEND_CHAIN[s], hold_amt(rng, s, a1 + a2) + a1)
        kw.update(a1=dstr(a1), a2=dstr(a2), t1=sym_word(rng, s), x1=addrs[0], x2=addrs[1])
        first = P(rng, split, S6M).format(**kw)
    elif variant == "send_swap":
        s1, s2 = "USDC", "ETH"
        a1 = amt_for(s1)
        a2 = D(str(rng.choice([0.02, 0.05, 0.1, 0.15, 0.2])))
        w.set_hold(s1, "base", hold_amt(rng, s1, a1))
        w.set_hold(s2, "base", hold_amt(rng, s2, a2) + a2)
        kw.update(a1=dstr(a1), t1=sym_word(rng, s1), x1=addrs[0], a2=dstr(a2), t2=sym_word(rng, s2), t3="usdc")
        first = P(rng, split, S6S).format(**kw)
        swap_first = first.startswith("swap") or first.lower().startswith("swap")
        sl = send_leg(rng, bio, s1, None, addrs[0], amt=a1, size_bal_chain=None, noun=f"the send of {dstr(a1)} {s1} to {addrs[0]}")
        sl["th"] = "the send"
        wl = swap_leg(rng, bio, "swap", s2, "base", "USDC", "base", a2, noun=f"the swap of {dstr(a2)} {s2} to USDC")
        wl["th"] = "the swap"
        legs = [wl, sl] if swap_first else [sl, wl]
    else:  # dep_send
        s = "USDC"
        a1, a2 = amt_for(s), D(str(rng.choice([20, 25, 40, 50, 75, 100, 150, 200])))
        w.set_hold(s, "base", a1 + a2 + D(str(rng.choice([0, 5, 20, 60]))))
        kw.update(a1=dstr(a1), a2=dstr(a2), t1=sym_word(rng, s), x1=addrs[0])
        first = P(rng, split, S6D).format(**kw)
        dep_first = first.lower().startswith(("deposit", "put", "lend", "aave"))
        dl = deposit_leg(rng, bio, s, "base", size_bal=False, amt=a2, named_chain=False, noun=f"the Aave deposit of {dstr(a2)} {s}")
        dl["th"] = "the deposit"
        sl = send_leg(rng, bio, s, None, addrs[0], amt=a1, size_bal_chain=None, noun=f"the send of {dstr(a1)} {s} to {addrs[0]}")
        sl["th"] = "the send"
        legs = [dl, sl] if dep_first else [sl, dl]
    add_noise_holdings(rng, w, set(syms) | {"USDC"})
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
    blocked0 = rng.random() < 0.06 and variant != "multi"
    if blocked0:
        t0 = legs[0]["tool"]
        w.s["force"][t0] = blocked_result(rng, "send" if t0 == "prepare_send" else "swap")
    c.user(decorate(rng, first))
    if variant == "multi":
        recips = [{"address": addrs[0], "amount": num_json(a1)}, {"address": addrs[1], "amount": num_json(a2)}]
        args = {"symbol": s, "recipients": recips}
        res = c.call1("prepare_multisend", args)
        assert res["status"] == "review_opened", res
        c.step("s1", "prepare_multisend", args)
        c.say(rng.choice([
            f"These two are the same token, so I put them in one batch: {res['total']} to {res['recipients']} recipients on {res['network']}, with fees up to {res['fees_up_to']}. One review, one {bio}. Review it and approve it.",
            f"I've prepared one batch send of {res['total']} to {res['recipients']} recipients on {res['network']} (fees up to {res['fees_up_to']}). It's a single review that waits for your {bio}.",
            f"Same token for both, so it's a single batch: {res['total']} across {res['recipients']} recipients on {res['network']}. Approve the one review with {bio}."]))
        if rng.random() < 0.3:
            for r_ in recips:
                pass
            row = f"Sent {dstr(a1)} {s} to {short_addr(addrs[0])}"
            w.add_activity(f"Batch send {res['total']} to {res['recipients']} recipients", "confirmed", tick(rng, c.clock, 3, 14))
            c.user(user_followup(rng, split, FINAL_T))
            act = c.call1("get_activity", {"limit": 5})
            c.say(f"Activity shows the batch send of {res['total']} to {res['recipients']} recipients as confirmed.")
        return finish_money(c, "independent", "full", {"variant": variant})
    n_trans = len(legs) - 1
    seq = []
    for _ in range(n_trans):
        r = rng.random()
        o = (["landed"] if r < 0.55 else ["pending"] if r < 0.75 else ["missing", "landed"] if r < 0.87 else ["failed"])
        seq.append(o)
        if o[-1] == "failed":
            break
    fin = pick_final(rng) if seq[-1][-1] in ("landed", "pending") and len(seq) == n_trans and rng.random() < 0.6 else None
    if fin == "pending":
        fin = None
    tag = run_chain(c, rng, split, legs, seq, fin, c.clock, indep=True)
    return finish_money(c, "independent", tag, {"variant": variant})


# ------------------------------------------------------------------ shape 8: planning questions
S8A = [
    "what would it take to bridge {a} {s} from {src} to {dst} and deposit it in aave? what will it cost",
    "how would you move {a} {s} from {src} into aave on {dst}, and what are the fees",
    "plan: bridge {a} {s} {src} -> {dst}, then aave. steps and cost?",
    "walk me through bridging {a} {s} to {dst} and depositing in aave, how much will it cost me",
    "before i do anything: what are the steps and costs to get {a} {s} from {src} into aave on {dst}",
    "how much does it cost to bridge {a} {s} from {src} to {dst} and then deposit into aave? just tell me, dont prepare anything",
    "Explain the steps to move {a} {s} from {src} to aave on {dst}, with the bridge cost.",
    "if i wanted {a} {s} in aave on {dst} coming from {src}, what happens and what does it cost?",
    "dont do anything yet. how would bridge + aave deposit work for {a} {s} {src} to {dst} and what's the fee",
    "quote me the bridge for {a} {s} {src} to {dst} and tell me what comes after for aave",
]
S8B = [
    "if i swap {a} {s} to usdc on {src} and then send some to a friend, what are the steps and the cost?",
    "what will it cost to convert {a} {s} into usdc on {src} and send it out? just the plan",
    "how would swapping {a} {s} to usdc on {src} and then sending it work, and what's the fee",
    "plan only: swap {a} {s} for usdc on {src}, then send the usdc to someone. cost?",
    "Walk me through swapping {a} {s} to USDC on {src} and sending it on, with the swap cost.",
    "dont prepare anything. {a} {s} -> usdc on {src}, then send it: what are the steps and fees",
    "before i do it: what does swapping {a} {s} to usdc on {src} cost, and what comes after for the send",
    "steps and cost to turn {a} {s} on {src} into usdc and pay someone with it?",
    "how much would it cost to swap {a} {s} into usdc on {src} first and send after",
    "tell me the plan for swapping {a} {s} to usdc on {src} then a send, and the swap cost, but dont start",
]
S8C = [
    "how would i move my aave {s} from {src} to {dst}, and what will it cost",
    "what are the steps and fees to take my {s} out of aave on {src} and get it to {dst}",
    "plan: withdraw my aave {s} on {src}, bridge to {dst}. cost?",
    "walk me through getting my aave {s} from {src} over to {dst}, and the bridge fee",
    "dont do anything yet: aave {s} on {src} to {dst}, what are the steps and the cost",
    "Explain how to move my {s} from aave on {src} to {dst}, with costs.",
    "what would it cost to withdraw my {s} from aave on {src} and bridge it to {dst}",
    "if i pull my {s} out of aave on {src} and bridge to {dst}, what happens and what does it cost?",
    "just the plan and the fees for moving my aave {s} on {src} to {dst}",
    "how do i get my aave {s} on {src} to {dst}? steps and bridge cost please",
]
GOAHEAD = ["ok do it", "ok start with the first step", "yes start", "go ahead with step one", "alright, prepare the first one",
           "ok lets do it", "sounds good, start", "yes, go", "do the first step", "ok prepare it", "fine, begin", "yep start it",
           "ok go for it", "start with the first one pls", "let's go"]


def sc_planning(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    variant = rng.choices(["A", "B", "C"], [46, 27, 27])[0]
    go = rng.random() < 0.38
    if variant == "A":
        sym = rng.choices(["USDC", "USDT", "ETH"], [78, 8, 14])[0]
        dst = rng.choice([ch for ch in AAVE_CHAINS if sym in AAVE_SUPPORT[ch]])
        src = rng.choice([ch for ch in TOKEN_CHAINS[sym] if ch != dst and ch in QUOTE_CHAINS])
        amt = amt_dec(sym, rng)
        w.set_hold(sym, src, hold_amt(rng, sym, amt))
        add_noise_holdings(rng, w, {sym})
        first = P(rng, split, S8A).format(a=amt_user(rng, amt), s=sym_word(rng, sym), src=chain_word(rng, src), dst=chain_word(rng, dst))
        w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
        c.user(decorate(rng, first))
        qargs = {"from_symbol": sym, "from_chain": src, "to_symbol": sym, "to_chain": dst, "amount": num_json(amt)}
        q = c.call1("get_quote", qargs)
        assert "error" not in q, q
        c.say(rng.choice([
            f"It takes two steps, each with its own {bio} review, and the app shows one at a time. First the bridge: {q['from']} to {q['to']} through {ROUTE}, about {q['seconds']} seconds, total cost about {usd(q['total_cost_usd'])}. Once it lands, I prepare the Aave deposit on {CHAIN_NAME[dst]}. Nothing is prepared yet. Want me to start with the bridge?",
            f"Two steps. First, the bridge through {ROUTE}: {q['from']} to {q['to']}, about {q['seconds']} seconds and about {usd(q['total_cost_usd'])} in total cost. Second, after it lands, the Aave deposit on {CHAIN_NAME[dst]}, sized from your balance there. Each needs your {bio}. I haven't prepared anything. Shall I begin with the bridge?",
            f"Here is the plan: bridge first ({q['from']} to {q['to']}, {ROUTE}, about {q['seconds']} seconds, total cost about {usd(q['total_cost_usd'])}), then deposit into Aave on {CHAIN_NAME[dst]} once the funds show up. You approve each step with {bio}. Nothing is prepared yet; say the word and I start with the bridge."]))
        if go:
            c.user(decorate(rng, P(rng, split, GOAHEAD), first=False))
            res = c.call1("prepare_swap", qargs)
            assert ok_status(res)
            c.step("s1", "prepare_swap", qargs)
            c.say(announce_swap(rng, res, "bridge", bio) + " " + after_line(rng, "the Aave deposit"))
    elif variant == "B":
        sym = rng.choice(["ETH", "ETH", "SOL"]) if False else "ETH"
        chain = rng.choice(["base", "arbitrum", "ethereum"])
        amt = amt_dec(sym, rng)
        w.set_hold(sym, chain, hold_amt(rng, sym, amt))
        add_noise_holdings(rng, w, {sym})
        first = P(rng, split, S8B).format(a=amt_user(rng, amt), s=sym_word(rng, sym), src=chain_word(rng, chain))
        w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
        c.user(decorate(rng, first))
        qargs = {"from_symbol": sym, "from_chain": chain, "to_symbol": "USDC", "to_chain": chain, "amount": num_json(amt)}
        q = c.call1("get_quote", qargs)
        assert "error" not in q, q
        c.say(rng.choice([
            f"Two steps, one review at a time. First the swap: {q['from']} to {q['to']} through {ROUTE}, total cost about {usd(q['total_cost_usd'])}, about {q['seconds']} seconds. After it lands, I read your real USDC balance and prepare the send; I need the recipient's address for that. Nothing is prepared yet. Start with the swap?",
            f"The swap comes first: {q['from']} to {q['to']} ({ROUTE}, about {q['seconds']} seconds, total cost about {usd(q['total_cost_usd'])}). The send comes second and only after the swap lands, because I size it from your balance. Each step needs your {bio}. Want me to prepare the swap?",
            f"Plan: swap now, send later. The swap is {q['from']} to {q['to']} through {ROUTE}, costing about {usd(q['total_cost_usd'])} in total. Once it lands I prepare the send (give me the address then). Nothing is prepared yet."]))
        if go:
            c.user(decorate(rng, P(rng, split, GOAHEAD), first=False))
            res = c.call1("prepare_swap", qargs)
            assert ok_status(res)
            c.step("s1", "prepare_swap", qargs)
            c.say(announce_swap(rng, res, "swap", bio) + " " + after_line(rng, "the send") + " I'll need the address then.")
    else:
        sym = rng.choices(["USDC", "USDT"], [85, 15])[0]
        src = rng.choice([ch for ch in AAVE_CHAINS if sym in AAVE_SUPPORT[ch]])
        dst = rng.choice([ch for ch in ["base", "arbitrum", "ethereum", "polygon"] if ch != src])
        supplied = D(str(rng.choice([120.5, 250, 410.6, 640, 1203.77, 1500, 2300.5, 52.31])))
        w.set_aave(sym, src, supplied)
        add_noise_holdings(rng, w, {sym})
        first = P(rng, split, S8C).format(s=sym_word(rng, sym), src=chain_word(rng, src), dst=chain_word(rng, dst))
        w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
        c.user(decorate(rng, first))
        pos = c.call1("get_positions", {"kind": "earn"})
        row = [r for r in pos["earn"] if r["symbol"] == sym and r["chain"] == src][0]
        qargs = {"from_symbol": sym, "from_chain": src, "to_symbol": sym, "to_chain": dst, "amount": num_json(D(row["amount"]))}
        q = c.call1("get_quote", qargs)
        assert "error" not in q, q
        c.say(rng.choice([
            f"You have {row['amount']} {sym} in Aave on {CHAIN_NAME[src]}. Step one is the withdrawal, which I can prepare. Step two is the bridge, which I prepare after the withdrawal lands, sized from your balance. For reference, bridging {q['from']} would give {q['to']} through {ROUTE}, about {q['seconds']} seconds, total cost about {usd(q['total_cost_usd'])}. Nothing is prepared yet. Start with the withdrawal?".replace("Step one", "First").replace("Step two", "Then"),
            f"Your Aave deposit on {CHAIN_NAME[src]} is {row['amount']} {sym}. First I prepare withdrawing it. Once it lands, I check your balance and prepare the bridge to {CHAIN_NAME[dst]}. The bridge quote right now: {q['from']} to {q['to']}, {ROUTE}, total cost about {usd(q['total_cost_usd'])}, about {q['seconds']} seconds. Nothing is prepared yet; each step needs your {bio}.",
            f"Two steps, one at a time: withdraw {row['amount']} {sym} from Aave on {CHAIN_NAME[src]}, then bridge it to {CHAIN_NAME[dst]}. A bridge of that size costs about {usd(q['total_cost_usd'])} in total through {ROUTE} and takes about {q['seconds']} seconds. I haven't prepared anything. Want me to start with the withdrawal?"]))
        if go:
            c.user(decorate(rng, P(rng, split, GOAHEAD), first=False))
            args = {"symbol": sym, "chain": src}
            res = c.call1("prepare_withdraw", args)
            assert ok_status(res), res
            c.step("s1", "prepare_withdraw", args)
            c.say(f"I've prepared withdrawing all of your {sym} from Aave on {CHAIN_NAME[src]}, waiting for your {bio}. The bridge to {CHAIN_NAME[dst]} comes after it lands. Tell me when it has.")
    return finish_money(c, "planning", "plan_go" if go else "plan", {"variant": variant})


# ------------------------------------------------------------------ read chains (suite chain_multi, kinds R1..R9)
def pnum(x):
    """Price text from a result: thousands commas, no exponent."""
    d = D(repr(x)) if isinstance(x, float) else D(x)
    s = format(d, "f")
    ip, _, fp = s.partition(".")
    ip = f"{int(ip):,}"
    return ip + ("." + fp if fp else "")


def chg_txt(x):
    if x is None:
        return "no 24h change available"
    return f"{'up' if x >= 0 else 'down'} {dstr(abs(x))}%"


R1T = [
    "bridge all my {s} on {src} to {dst} if the cost is under ${cap}",
    "swap half my {s} on {src} to {t}, check the cost first and only go ahead if it's under ${cap}",
    "move everything i have in {s} on {src} over to {dst} but only if it costs less than ${cap}",
    "get me a quote and if the total cost is below ${cap}, bridge my whole {s} balance from {src} to {dst}",
    "if it costs under ${cap} total, swap half of my {s} on {src} for {t}",
    "Bridge all my {s} on {src} to {dst} if the fee is under ${cap}.",
    "how much to bridge all my {s} from {src} to {dst}? if it's under ${cap}, prepare it",
    "check the cost of swapping half my {s} on {src} into {t}. under ${cap}? then prepare it",
    "send my full {s} balance on {src} over to {dst}, unless the cost is more than ${cap}",
    "bridge my {s} on {src} to {dst}, whole balance, cap the cost at ${cap}",
    "if the total cost stays below ${cap}, put half my {s} from {src} into {t}",
    "look at what bridging all my {s} from {src} to {dst} costs, and prepare it if it's less than ${cap}",
]


def rc_swap_quote(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    sym = rng.choices(["USDC", "USDT", "ETH"], [60, 12, 28])[0]
    src = rng.choice([ch for ch in TOKEN_CHAINS[sym] if ch in QUOTE_CHAINS and ch != "polygon"])
    mode = rng.choice(["all", "half"])
    bridge = rng.random() < 0.55
    if bridge:
        dst, to_sym = rng.choice([ch for ch in ["base", "arbitrum", "ethereum"] if ch != src]), sym
    else:
        dst, to_sym = src, ("ETH" if sym != "ETH" else "USDC")
    amt = amt_dec(sym, rng)
    w.set_hold(sym, src, hold_amt(rng, sym, amt) * 2 if mode == "half" else hold_amt(rng, sym, amt))
    add_noise_holdings(rng, w, {sym})
    bs = dstr(w.hold(sym, src))
    use = D(bs) if mode == "all" else D(bs) / 2
    use = D(str(round(float(use), 6)))
    q = w.quote(sym, src, to_sym, dst, use)
    if "error" in q:
        raise Retry("quote")
    cost = q["total_cost_usd"]
    ok = rng.random() < 0.7
    caps = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30]
    cap_ = [x for x in caps if (x > cost)] if ok else [x for x in caps if x < cost]
    if not cap_:
        raise Retry("cap")
    cap_v = rng.choice(cap_[:4]) if ok else rng.choice(cap_[-4:])
    tpl = P(rng, split, [t for t in R1T] if bridge else R1T)
    if (not bridge and "{dst}" in tpl) or (bridge and "{t}" in tpl and "{dst}" not in tpl):
        raise Retry("template mismatch")
    if mode == "half" and ("all" in tpl or "whole" in tpl or "everything" in tpl or "full" in tpl):
        raise Retry("mode mismatch")
    if mode == "all" and "half" in tpl:
        raise Retry("mode mismatch")
    first = tpl.format(s=sym_word(rng, sym), src=chain_word(rng, src), dst=chain_word(rng, dst), t=sym_word(rng, to_sym), cap=dstr(cap_v))
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
    c.user(decorate(rng, first))
    bal = c.call1("get_balances", {"chain": src})
    have = find_bal(bal, sym, src)
    a = D(have)
    if mode == "half":
        r = c.call1("calculator", {"expression": f"{have} / 2"})
        a = D(r["result"])
    qargs = {"from_symbol": sym, "from_chain": src, "to_symbol": to_sym, "to_chain": dst, "amount": num_json(a)}
    quote = c.call1("get_quote", qargs)
    assert "error" not in quote, quote
    thing = "bridge" if bridge else "swap"
    if quote["total_cost_usd"] < cap_v:
        res = c.call1("prepare_swap", qargs)
        assert ok_status(res)
        c.step("s1", "prepare_swap", qargs)
        c.say(rng.choice([
            f"The total cost is {usd(quote['total_cost_usd'])}, under your ${dstr(cap_v)}. I've prepared the {thing}: {res['from']} to {res['to']} through {ROUTE}, waiting for your {bio}." + warn_tail(res),
            f"Quote: {usd(quote['total_cost_usd'])} all in, below your limit of ${dstr(cap_v)}. So I prepared it: {res['from']} to {res['to']} ({ROUTE}). Approve with {bio}." + warn_tail(res),
            f"It costs {usd(quote['total_cost_usd'])} in total, which fits your ${dstr(cap_v)} cap. The {thing} is prepared: {res['from']} to {res['to']}, about {res['seconds']} seconds. Approve it with {bio}." + warn_tail(res)]))
    else:
        c.say(rng.choice([
            f"The total cost is {usd(quote['total_cost_usd'])}, above your ${dstr(cap_v)} limit, so I haven't prepared anything. The quote was {quote['from']} to {quote['to']} through {ROUTE}. Tell me if you want it anyway.",
            f"That would cost {usd(quote['total_cost_usd'])} in total, more than the ${dstr(cap_v)} you set. I stopped at the quote: {quote['from']} to {quote['to']}. Nothing is prepared; say so if you want to go ahead.",
            f"{quote['from']} to {quote['to']} costs {usd(quote['total_cost_usd'])} all in, over your ${dstr(cap_v)} cap. I haven't prepared it. Want it at that price?"]))
    return finish_read(c, "swap_quote", {"cost_ok": quote["total_cost_usd"] < cap_v})


def finish_read(c, kind, extra=None, cut_last=False):
    rec = finish_money(c, kind, "read", extra)
    rec["meta"]["suite_kind"] = "read"
    if cut_last:
        rec["meta"]["cut"] = len(c.users) - 1
        rec["meta"]["expect"] = expect_for(c, len(c.users) - 1)
        rec["meta"]["script"] = []
    return rec


R2T = [
    "go {side} {coin} {lev}x, risk {risk}% of my account, stop at {stop}",
    "i want to {side} {coin} with {lev}x. risk {risk}% of my hyperliquid account and put the stop at {stop}",
    "size a {coin} {side} so i lose {risk}% of my account if it hits {stop}, {lev}x leverage, market order",
    "open a {coin} {side}, {lev}x, stop loss {stop}, risking {risk}% of my account",
    "{side} {coin} {lev}x risking {risk}% stop {stop}",
    "Open a {side} on {coin} at {lev}x. Risk {risk}% of my account, stop at {stop}.",
    "what size should a {coin} {side} be if i risk {risk}% with a stop at {stop}? {lev}x, then set it up",
    "set up a {coin} {side} with a {stop} stop, {lev}x, and size it to risk {risk}% of my account",
    "risk {risk}% on a {coin} {side}: {lev}x leverage, stop at {stop}. prepare the trade",
    "{coin} {side} {lev}x, stop {stop}, position sized to {risk}% account risk",
    "put on a {side} in {coin}, {lev}x, with the stop at {stop} and only {risk}% of my account at risk",
]


def rc_risk_trade(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE", "AVAX", "LINK", "SUI", "BNB"])
    side = rng.choice(["long", "short"])
    px = w.s["prices"][coin]
    short_margin = rng.random() < 0.2
    if short_margin:
        risk, dist, lev = 5, 0.02, 2
    else:
        risk = rng.choice([0.5, 1, 1, 1.5, 2, 2, 2.5, 3])
        dist = rng.choice([0.03, 0.04, 0.05, 0.06, 0.08])
        lev = min(rng.choice([2, 3, 5, 5, 10]), PERP_MAX_LEV[coin])
    stop_v = px * (1 - dist) if side == "long" else px * (1 + dist)
    dec = 2 if px >= 10 else (3 if px >= 1 else 4)
    stop_v = round(stop_v, 0 if px >= 1000 else dec)
    stop = dstr(stop_v)
    acct_v = rng.choice([180.5, 250.5, 420, 640.25, 1200, 2480.75, 5200, 9100])
    w.s["hl"]["available"] = acct_v
    w.s["hl"]["positions"] = []
    first = P(rng, split, R2T).format(side=side, coin=coin if rng.random() < 0.8 else coin.lower(), lev=lev, risk=dstr(risk), stop=stop)
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 2))
    c.user(decorate(rng, first))
    acc, pr = c.call(("hyperliquid_account", {}), ("get_price", {"symbol": coin}))
    av = acc["account_value"]
    price = pr["price"]
    expr = f"{dstr(av)} * {dstr(risk)} / 100 / (abs({dstr(price)} - {dstr(stop_v)}) / {dstr(price)})"
    r = c.call1("calculator", {"expression": expr})
    size = round(float(r["result"]), 2)
    avail_now = acc["available_usdc"]
    if size / lev > avail_now:
        r2 = c.call1("calculator", {"expression": f"{dstr(size)} / {lev}"})
        c.say(rng.choice([
            f"{coin} is at ${pnum(price)}. Risking {dstr(risk)}% of your ${pnum(av)} account with a stop at {stop} sizes the {side} at about ${pnum(size)}. At {lev}x that needs {pnum(r2['result'])} USDC of margin and you have {pnum(avail_now)} available, so I haven't prepared it. A higher leverage, a tighter stop or adding funds would fit.",
            f"The sized {side} comes to about ${pnum(size)} ({coin} at ${pnum(price)}, stop {stop}, {dstr(risk)}% of ${pnum(av)}). At {lev}x it needs {pnum(r2['result'])} USDC margin, but only {pnum(avail_now)} is available. Nothing is prepared; tell me if you want a smaller size or more leverage."]))
        return finish_read(c, "risk_trade", {"margin_short": True})
    args = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market", "stop_loss": num_json(D(stop))}
    res = c.call1("prepare_trade", args)
    assert res.get("status") == "review_opened", res
    c.step("s1", "prepare_trade", args)
    c.say(rng.choice([
        f"{coin} is at ${pnum(price)} and your account is ${pnum(av)}. Risking {dstr(risk)}% with the stop at {stop} sizes the {side} at about ${pnum(size)}. I've prepared it as a market order at {lev}x with that stop, waiting for your {bio}.",
        f"I've prepared a {side} on {coin}: ${pnum(size)} at {lev}x, market, stop loss {stop}. That size risks {dstr(risk)}% of your ${pnum(av)} account with {coin} at ${pnum(price)}. Approve it with {bio}.",
        f"Sized from your ${pnum(av)} account and the {stop} stop, the {side} is about ${pnum(size)} at {lev}x ({coin} is ${pnum(price)}). The order is prepared, market with a stop at {stop}. It waits for your {bio}."]))
    return finish_read(c, "risk_trade", {})


R3T = [
    "price of {a}, {b} and {c}", "what are {a}, {b} and {c} at", "{a} {b} {c} prices pls", "how are {a}, {b} and {c} doing today",
    "give me {a}, {b} and {c} prices and which moved most", "quick check on {a} / {b} / {c}", "Whats the price of {a}, {b} and {c}?",
    "compare {a}, {b} and {c} right now", "show me {a}, {b}, {c}", "where are {a}, {b} and {c} trading",
    "is {a} or {b} or {c} up the most today", "prices for {a}, {b} and {c} please",
]
SHARE_T = [
    "what share of my wallet is {sym}", "how much of my total wallet is in {sym}, as a percent", "what percent of my balances is {sym}",
    "{sym} is what % of my wallet", "tell me the {sym} share of my wallet holdings", "Which part of my wallet is {sym}? give a percent",
    "how big is my {sym} bag compared to the rest of the wallet, in %", "percentage of my wallet in {sym}?", "what % of my holdings are {sym}",
    "work out how much of my wallet is {sym}",
]


def rc_parallel_prices(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    if rng.random() < 0.72:
        coins = rng.sample(["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "LINK", "DOGE", "SUI", "XRP"], 3)
        first = P(rng, split, R3T).format(a=coins[0], b=coins[1], c=coins[2]) if rng.random() < 0.5 else \
            P(rng, split, R3T).format(a=coins[0].lower(), b=coins[1].lower(), c=coins[2].lower())
        w.s["activity"] = old_rows(rng, w, 1)
        c.user(decorate(rng, first))
        rs = c.call(*[("get_price", {"symbol": k}) for k in coins])
        parts = [f"{r['symbol']} ${pnum(r['price'])} ({chg_txt(r['change_24h_pct'])})" for r in rs]
        tail = ""
        if re.search(r"most|moved", first):
            worst = max(rs, key=lambda r: abs(r["change_24h_pct"]))
            tail = f" {worst['symbol']} moved the most, {chg_txt(worst['change_24h_pct'])}."
        c.say(rng.choice([", ".join(parts[:2]) + " and " + parts[2] + ".", "Right now: " + "; ".join(parts) + "."]) + tail)
        return finish_read(c, "parallel_prices", {})
    # share of wallet
    n = rng.choice([2, 3])
    pairs = rng.sample([("USDC", "base"), ("ETH", "base"), ("ETH", "arbitrum"), ("SOL", "solana"), ("USDT", "arbitrum"), ("USDC", "arbitrum")], n)
    for s, ch in pairs:
        usd_v = rng.choice([60, 120, 250, 400, 800, 1500])
        w.set_hold(s, ch, D(str(round(usd_v / w.price(s), 4 if w.price(s) > 50 else 2))))
    sym = rng.choice([p[0] for p in pairs])
    first = P(rng, split, SHARE_T).format(sym=sym_word(rng, sym))
    w.s["activity"] = old_rows(rng, w, 1)
    c.user(decorate(rng, first))
    bal = c.call1("get_balances", {})
    tot_terms = " + ".join(dstr(h["usd"]) for h in bal["holdings"])
    mine = " + ".join(dstr(h["usd"]) for h in bal["holdings"] if h["symbol"] == sym)
    r, tot = c.call(("calculator", {"expression": f"({mine}) / ({tot_terms}) * 100"}), ("calculator", {"expression": tot_terms}))
    share = round(float(r["result"]), 1)
    c.say(rng.choice([
        f"{sym} is about {dstr(share)}% of your wallet, out of ${pnum(tot['result'])} in total.",
        f"About {dstr(share)}% of your wallet is {sym}; the whole wallet is ${pnum(tot['result'])}.",
        f"{sym} makes up roughly {dstr(share)}% of your ${pnum(tot['result'])} wallet."]))
    return finish_read(c, "parallel_prices", {"share": True})


R4T = [
    "compare aave usdc on {c1} and {c2}, deposit {a} usdc into the better one and tell me what it earns in a year",
    "which pays more on aave for usdc, {c1} or {c2}? put {a} usdc in the higher one",
    "deposit {a} usdc in aave on whichever of {c1} or {c2} has the better apy, and what would i make in a year",
    "i have usdc on {c1} and {c2}. use the better aave rate for {a} usdc, show me the yearly earnings",
    "Pick the higher aave usdc apy between {c1} and {c2} and deposit {a} usdc there.",
    "aave usdc: {c1} vs {c2}. deposit {a} where it pays more and tell me the yearly yield",
    "what would {a} usdc earn in a year in aave on {c1}, then deposit it",
    "check the aave usdc apy on {c1} and put {a} usdc in. how much per year?",
    "deposit {a} usdc into aave on {c1} and tell me what that earns over a year",
    "Whats the aave usdc rate on {c1}? deposit {a} usdc and give me the yearly amount",
    "estimate a year of interest on {a} usdc in aave {c1}, and deposit it",
    "use the best aave usdc rate of {c1} and {c2} for {a} usdc, then tell me the annual interest",
]


def rc_yield_deposit(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    sym = "USDC"
    a = D(str(rng.choice([100, 150, 200, 250, 300, 500, 750, 1000, 1500, 2000, 5000])))
    c1, c2 = rng.sample(["base", "arbitrum", "ethereum"], 2)
    tpl = P(rng, split, R4T)
    two = "{c2}" in tpl
    chains = [c1, c2] if two else [c1]
    for ch in chains:
        w.set_hold(sym, ch, hold_amt(rng, sym, a) + a)
    add_noise_holdings(rng, w, {sym})
    # make the rates differ clearly
    if two:
        w.s["apy"][f"{c1}|USDC"] = round(rng.uniform(2.0, 4.0), 2)
        w.s["apy"][f"{c2}|USDC"] = round(w.s["apy"][f"{c1}|USDC"] + rng.choice([-1.6, -0.8, 0.9, 1.7]), 2)
        w.s["yields"] = make_yields(rng, w.s["apy"])
    first = tpl.format(a=amt_user(rng, a), c1=chain_word(rng, c1), c2=chain_word(rng, c2))
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 2))
    c.user(decorate(rng, first))
    y = c.call1("get_yields", {"asset": "USDC"})
    aave_rows = [r for r in y["pools"] if r["protocol"] == "Aave v3" and r["asset"] == "USDC" and r["chain"] in [CHAIN_NAME[x] for x in chains]]
    best = max(aave_rows, key=lambda r: r["apy_pct"])
    bch = [k for k, v in CHAIN_NAME.items() if v == best["chain"]][0]
    top_other = [r for r in y["pools"] if not r["can_deposit_in_hermit"] and r["apy_pct"] > best["apy_pct"]]
    r = c.call1("calculator", {"expression": f"{dstr(a)} * {dstr(best['apy_pct'])} / 100"})
    args = {"project": "aave-v3", "symbol": sym, "chain": bch, "amount": num_json(a)}
    res = c.call1("prepare_deposit", args)
    assert ok_status(res), res
    c.step("s1", "prepare_deposit", args)
    extra = ""
    if top_other and two and rng.random() < 0.6:
        t = top_other[0]
        extra = f" {t['protocol']} on {t['chain']} shows {t['apy_pct']}%, but I can only prepare Aave v3 deposits; that pool is in Earn."
    cmp_txt = ""
    if two:
        other = [r_ for r_ in aave_rows if r_ is not best][0]
        cmp_txt = f" Aave v3 pays {best['apy_pct']}% on {best['chain']} against {other['apy_pct']}% on {other['chain']}."
    c.say(rng.choice([
        f"I've prepared depositing {res['amount']} into {res['pool']} at {res['apy_pct']}% APY, which can change.{cmp_txt} At that rate it earns about ${pnum(r['result'])} in a year. Approve with {bio}.{extra}",
        f"{cap(res['amount'])} into {res['pool']} is prepared, waiting for your {bio}.{cmp_txt} At {res['apy_pct']}% that is about ${pnum(r['result'])} a year, and the APY can change.{extra}"]))
    return finish_read(c, "yield_deposit", {})


R5T = [
    "close half of my most profitable position", "close 30% of my biggest loser", "take 25% off my best performing trade",
    "close 40% of the position that is up the most", "close half my worst position", "reduce my biggest winner by 50%",
    "Close 20% of my most profitable position.", "trim 30% off the position with the biggest loss",
    "close 75% of whichever trade is up the most", "cut my losing position in half", "close a quarter of my best winner",
    "close 60% of my biggest loser", "take half off the trade that's up the most", "scale out 50% of my most profitable one",
]
R5LOCK = [
    "close enough of my {coin} to lock in ${x} of profit", "take ${x} profit on my {coin} position by closing part of it",
    "lock in about ${x} from my {coin}: close just enough of it", "close the part of {coin} that is worth ${x} of profit",
    "how much of my {coin} do i close to bank ${x}? do it", "Close enough of {coin} to take ${x} profit.",
    "bank ${x} of profit on {coin}, close only as much as needed", "partially close {coin} so ${x} of the profit is locked in",
]


def rc_partial_close(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    n = rng.choice([2, 3])
    add_hl(rng, w, n)
    add_noise_holdings(rng, w, set())
    pos = w.s["hl"]["positions"]
    pnl = {p["coin"]: pos_margin_pnl(w, p)[1] for p in pos}
    lock = rng.random() < 0.4
    if lock:
        winners = [p for p in pos if pnl[p["coin"]] > 30]
        if not winners:
            raise Retry("no winner")
        tgt = rng.choice(winners)
        coin = tgt["coin"]
        have = round(pnl[coin], 2)
        x = rng.choice([x for x in [10, 15, 20, 25, 40, 50, 75, 100, 150, 200] if x < have * 0.9] or [5])
        first = P(rng, split, R5LOCK).format(coin=coin, x=x)
    else:
        tpl = P(rng, split, R5T)
        crit = ("winner" if re.search(r"profitable|up the most|best|winner", tpl) else "loser")
        pct = 50 if re.search(r"half|50%", tpl) else 33 if "third" in tpl else 25 if re.search(r"quarter|25%", tpl) else int(re.search(r"(\d+)%", tpl).group(1)) if re.search(r"(\d+)%", tpl) else 50
        tgt = max(pos, key=lambda p: pnl[p["coin"]]) if crit == "winner" else min(pos, key=lambda p: pnl[p["coin"]])
        if len({round(v, 2) for v in pnl.values()}) < len(pnl):
            raise Retry("tie")
        first = tpl
        coin = tgt["coin"]
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 2))
    c.user(decorate(rng, first))
    acc = c.call1("hyperliquid_account", {})
    p_ = [q for q in acc["positions"] if q["coin"] == coin][0]
    if lock:
        r = c.call1("calculator", {"expression": f"{x} / {dstr(p_['unrealised_pnl'])} * 100"})
        percent = int(round(float(r["result"])))
        percent = max(1, min(100, percent))
        args = {"coin": coin, "percent": percent}
        res = c.call1("prepare_close", args)
        assert ok_status(res)
        c.step("s1", "prepare_close", args)
        c.say(rng.choice([
            f"Your {coin} {p_['side']} has ${pnum(p_['unrealised_pnl'])} of unrealised profit, so ${x} is about {percent}% of it. I've prepared closing {res['closing']} ({res['order']}), waiting for your {bio}. The result is approximate, as the price moves.",
            f"To bank ${x} from ${pnum(p_['unrealised_pnl'])} of unrealised profit, I rounded to {percent}%. I've prepared closing {res['closing']} at market, reduce only. Approve with {bio}; the exact profit depends on the fill price."]))
        return finish_read(c, "partial_close", {"lock": True})
    args = {"coin": coin, "percent": pct}
    res = c.call1("prepare_close", args)
    assert ok_status(res)
    c.step("s1", "prepare_close", args)
    why = ("is your most profitable position" if crit == "winner" else "has your biggest loss")
    c.say(rng.choice([
        f"Your {coin} {p_['side']} {why} ({'+' if p_['unrealised_pnl'] >= 0 else '-'}${pnum(abs(p_['unrealised_pnl']))}). I've prepared closing {res['closing']} ({res['order']}), waiting for your {bio}.",
        f"The {coin} {p_['side']} {why}. I've prepared closing {res['closing']} at market, reduce only. Approve it with {bio}.",
        f"That is the {coin} {p_['side']}, unrealised {usd(p_['unrealised_pnl']) if p_['unrealised_pnl'] >= 0 else '-' + usd(-p_['unrealised_pnl'])}. Closing {res['closing']} is prepared and waits for your {bio}."]))
    return finish_read(c, "partial_close", {"lock": False})


R6T = [
    "whats the annualised funding on {a}", "annualized funding rate for {a}? is it expensive to hold a long",
    "how much funding would i pay per year on {a}", "{a} funding rate, annualised please",
    "what is {a}'s funding if you scale it to a year", "Is funding on {a} high? give me the yearly number",
    "yearly funding on a {a} perp?", "what does {a} funding work out to over a year",
    "funding for {a} per year, long or short which side pays",
]
R6C = [
    "compare the annualised funding on {a} and {b}", "which has higher yearly funding, {a} or {b}",
    "{a} vs {b}: annualised funding", "annualised funding for {a} and {b} please",
    "Funding on {a} and {b} scaled to a year, which is more expensive for a long?",
    "yearly funding {a} {b}, who pays more", "work out the annual funding rate for both {a} and {b}",
    "is {a} or {b} funding richer on a yearly basis",
]


def rc_funding(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    two = rng.random() < 0.3
    coins = rng.sample(FUND_NAMES, 2 if two else 1)
    tpl = P(rng, split, R6C if two else R6T)
    first = tpl.format(a=coins[0], b=coins[1] if two else "")
    w.s["activity"] = old_rows(rng, w, 1)
    c.user(decorate(rng, first))
    ms = c.call(*[("search_markets", {"query": f"{k} perp"}) for k in coins])
    rows = [m["markets"][0] for m in ms]
    cs = c.call(*[("calculator", {"expression": f"{dstr(r['funding_hourly_pct'])} * 24 * 365"}) for r in rows])
    anns = [r["result"] for r in cs]

    def side_txt(v):
        v = float(v)
        return "longs pay shorts" if v > 0 else "shorts pay longs"
    if not two:
        r0, a0 = rows[0], anns[0]
        c.say(rng.choice([
            f"{r0['name']} funding is {dstr(r0['funding_hourly_pct'])}% per hour, which scales to about {a0}% a year. At that rate {side_txt(a0)}. Funding changes all the time, so treat it as a snapshot.",
            f"Hourly funding on {r0['name']} is {dstr(r0['funding_hourly_pct'])}%. Times 24 and 365 that is about {a0}% a year, so {side_txt(a0)}. It moves, this is only the current rate.",
            f"About {a0}% a year: {r0['name']} funds at {dstr(r0['funding_hourly_pct'])}% per hour right now, and {side_txt(a0)}. The rate changes every hour."]))
    else:
        hi = max(range(2), key=lambda i: float(anns[i]))
        lo = 1 - hi
        c.say(rng.choice([
            f"{rows[0]['name']} funds at {dstr(rows[0]['funding_hourly_pct'])}% per hour, about {anns[0]}% a year. {rows[1]['name']} is {dstr(rows[1]['funding_hourly_pct'])}% per hour, about {anns[1]}% a year. {rows[hi]['name']} is the higher rate, so a long pays more there (or earns less). Both move hourly.",
            f"Annualised: {rows[0]['name']} {anns[0]}% and {rows[1]['name']} {anns[1]}% (from {dstr(rows[0]['funding_hourly_pct'])}% and {dstr(rows[1]['funding_hourly_pct'])}% per hour). {rows[hi]['name']} is higher. Positive means longs pay shorts; the rates change all the time."]))
    return finish_read(c, "funding_annualised", {"two": two})


# ------------------------------------------------------------------ R7: 4- and 5-round chains
R7A = [
    "swap ${usd} worth of my {s} to usdc", "sell ${usd} of my {s} into usdc", "convert ${usd} worth of {s} to usdc",
    "turn ${usd} of my {s} into usdc", "i want ${usd} in usdc, swap that much {s} for it", "swap about ${usd} of {s} to usdc",
    "Swap ${usd} worth of {s} to USDC.", "exchange ${usd} worth of my {s} for usdc", "get me ${usd} of usdc by swapping {s}",
    "put ${usd} of my {s} into usdc", "change ${usd} of {s} to usdc",
]
R7B = [
    "swap ${usd} of my biggest non-stable holding to usdc", "sell ${usd} worth of my largest crypto holding into usdc",
    "convert ${usd} of whatever volatile coin i hold the most of into usdc", "swap ${usd} worth of my biggest non-stablecoin to usdc",
    "take ${usd} out of my largest non-usdc holding and swap it to usdc", "Swap ${usd} of my biggest volatile holding to USDC.",
    "put ${usd} of my biggest coin (not stables) into usdc", "my largest non-stable holding: swap ${usd} of it to usdc",
    "turn ${usd} of my most valuable non-stable token into usdc", "swap ${usd} worth of the coin i hold most of, except stablecoins, to usdc",
    "dollar value ${usd} of my top non-stable bag, swap it to usdc",
]
R7C = [
    "swap ${usd} worth of my {s} to usdc, but only if the total cost is under ${cap}",
    "sell ${usd} of my {s} for usdc if the cost is below ${cap}, otherwise just tell me the cost",
    "convert ${usd} worth of {s} to usdc and check the cost first, go ahead only under ${cap}",
    "swap ${usd} of {s} into usdc, cost must be under ${cap}",
    "Swap ${usd} worth of {s} to USDC if it costs less than ${cap}.",
    "get a quote for ${usd} of my {s} into usdc and prepare it only if the cost is under ${cap}",
    "if swapping ${usd} of my {s} to usdc costs less than ${cap}, do it",
    "i want ${usd} of my {s} in usdc, but not if the cost is over ${cap}",
    "check the cost to swap ${usd} worth of {s} to usdc. prepare it if under ${cap}",
    "swap ${usd} of {s} to usdc, cap the total cost at ${cap}",
]


def token_amount(units, sym):
    q = Decimal("0.000001") if sym in ("ETH", "SOL") else Decimal("0.01")
    return D(repr(float(units))).quantize(q, rounding=ROUND_HALF_UP)


def rc_long_chain(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    variant = rng.choices(["a", "b", "c"], [34, 33, 33])[0]
    usd_v = rng.choice([100, 150, 200, 250, 300, 400, 500, 600, 800])
    if variant == "b":
        # several non-stable holdings, one clearly the biggest in usd
        choices = rng.sample([("ETH", "base"), ("ETH", "arbitrum"), ("SOL", "solana")], 2)
        big_sym, big_ch = choices[0]
        small_sym, small_ch = choices[1]
        if big_sym == small_sym:
            raise Retry("same symbol")
        big_usd = rng.choice([1500, 2400, 3800])
        small_usd = rng.choice([200, 450, 700])
        w.set_hold(big_sym, big_ch, D(str(round(big_usd / w.price(big_sym), 4 if big_sym == "ETH" else 3))))
        w.set_hold(small_sym, small_ch, D(str(round(small_usd / w.price(small_sym), 4 if small_sym == "ETH" else 3))))
        if rng.random() < 0.5:
            w.set_hold("USDC", "base", D(str(rng.choice([90, 400, 4000]))))
        sym, ch = big_sym, big_ch
        tpl = P(rng, split, R7B)
        first = tpl.format(usd=usd_v)
    else:
        sym = rng.choice(["ETH", "ETH", "SOL"])
        ch = "solana" if sym == "SOL" else rng.choice(["base", "arbitrum", "ethereum"])
        w.set_hold(sym, ch, D(str(round(usd_v * rng.choice([2, 3, 5, 9]) / w.price(sym), 4 if sym == "ETH" else 3))))
        add_noise_holdings(rng, w, {sym})
        if variant == "a":
            first = P(rng, split, R7A).format(usd=usd_v, s=sym_word(rng, sym))
        else:
            first = None
    quote_cap = None
    if variant == "c":
        # decide pass or fail from the real quote
        px = w.price(sym)
        units = token_amount(usd_v / px, sym)
        q = w.quote(sym, ch, "USDC", ch, units)
        if "error" in q:
            raise Retry("quote")
        cost = q["total_cost_usd"]
        ok = rng.random() < 0.7
        caps = [0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 8, 10, 15]
        cl = [x for x in caps if x > cost] if ok else [x for x in caps if x < cost]
        if not cl:
            raise Retry("cap")
        quote_cap = rng.choice(cl[:3]) if ok else rng.choice(cl[-3:])
        first = P(rng, split, R7C).format(usd=usd_v, s=sym_word(rng, sym), cap=dstr(quote_cap))
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
    c.user(decorate(rng, first))
    if variant == "b":
        bal = c.call1("get_balances", {})
        non = [h for h in bal["holdings"] if h["symbol"] not in STABLES]
        top = max(non, key=lambda h: h["usd"])
        sym, ch = top["symbol"], top["chain"]
        pr = c.call1("get_price", {"symbol": sym})
    else:
        bal, pr = c.call(("get_balances", {}), ("get_price", {"symbol": sym}))
    price = pr["price"]
    held = find_bal(bal, sym, ch)
    r = c.call1("calculator", {"expression": f"{usd_v} / {dstr(price)}"})
    units = token_amount(r["result"], sym)
    if units > D(held):
        raise Retry("not enough")
    args = {"from_symbol": sym, "from_chain": ch, "to_symbol": "USDC", "to_chain": ch, "amount": num_json(units)}
    if variant == "c":
        quote = c.call1("get_quote", args)
        assert "error" not in quote
        if quote["total_cost_usd"] >= quote_cap:
            c.say(rng.choice([
                f"At ${pnum(price)} per {sym}, ${usd_v} is {dstr(units)} {sym}. The quote says the total cost is {usd(quote['total_cost_usd'])}, above your ${dstr(quote_cap)} limit, so I haven't prepared it. Tell me if you want it anyway.",
                f"${usd_v} of {sym} is {dstr(units)} {sym} at ${pnum(price)}. Swapping that costs {usd(quote['total_cost_usd'])} all in, over your ${dstr(quote_cap)} cap. Nothing is prepared; say so if you still want it."]))
            return finish_read(c, "long_chain", {"variant": variant, "rounds": 4, "cost_ok": False})
    res = c.call1("prepare_swap", args)
    assert ok_status(res), res
    c.step("s1", "prepare_swap", args)
    lead = f"${usd_v} of {sym} at ${pnum(price)} is {dstr(units)} {sym}."
    if variant == "b":
        lead = f"Your biggest non-stable holding is {sym} on {CHAIN_NAME[ch]}. " + lead
    if variant == "c":
        lead = f"{lead} The total cost is {usd(quote['total_cost_usd'])}, under your ${dstr(quote_cap)}."
    c.say(rng.choice([
        f"{lead} I've prepared the swap: {res['from']} to {res['to']} through {ROUTE}, waiting for your {bio}." + warn_tail(res),
        f"{lead} The swap is prepared ({res['from']} to {res['to']}, {ROUTE}). Approve it with {bio}." + warn_tail(res)]))
    return finish_read(c, "long_chain", {"variant": variant, "rounds": {"a": 4, "b": 5, "c": 5}[variant], "cost_ok": True})


# ------------------------------------------------------------------ R8: a result changes the next step
R8F = [
    "bridge {a} usdc from base to arbitrum, or whatever i have if it's less",
    "swap {a} usdc on base to eth. if i don't have that much do all of it",
    "move {a} usdc from base over to arbitrum, if i have less just take the full balance",
    "Bridge {a} USDC from Base to Arbitrum, or my whole balance if that's lower.",
    "swap {a} usdc on base into eth, or everything i have there if it's under that",
    "send {a} usdc from base to arbitrum via bridge, take the max i have if short",
    "i want {a} usdc swapped to eth on base. if my balance is smaller use all of it",
    "bridge up to {a} usdc from base to arbitrum, whatever i've got",
    "swap up to {a} usdc on base to eth",
    "bridge {a} usdc base -> arbitrum, or less if that's all i have",
]
R8N = [
    "if i have {a} usdc on base, bridge it to arbitrum",
    "check my balance and if i have at least {a} usdc on base swap it to eth",
    "bridge {a} usdc from base to arbitrum if i have enough",
    "Check I have {a} USDC on Base, then bridge it to Arbitrum.",
    "swap {a} usdc on base to eth, but only if i actually have it",
    "do i have {a} usdc on base? if so bridge it to arbitrum",
    "see if {a} usdc is on base and swap it to eth if it is",
    "if my base balance covers {a} usdc, bridge that to arbitrum",
    "first check my usdc on base, then bridge {a} of it to arbitrum if it covers it",
    "make sure {a} usdc is there on base before swapping it to eth",
]
R8Y = ["yes", "yes do that", "ok go with that", "yep", "ok, do it with that amount", "fine, use that", "sure go ahead", "yes please", "do it",
       "ok that works"]


def rc_balance_changes(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    asked = D(str(rng.choice([100, 150, 200, 250, 300, 400, 500, 750, 1000])))
    case = rng.choices(["enough", "fallback", "ask"], [24, 38, 38])[0]
    if case == "enough":
        bal = asked + D(str(rng.choice([0.5, 12.3, 40, 150, 310.25])))
    else:
        bal = D(str(round(float(asked) * rng.choice([0.2, 0.35, 0.5, 0.64, 0.8, 0.93]) + rng.choice([0, 0.37, 0.84]), 2)))
    w.set_hold("USDC", "base", bal)
    add_noise_holdings(rng, w, {"USDC"})
    bridge = rng.random() < 0.6
    pool = R8F if case == "fallback" else (R8N if case == "ask" else rng.choice([R8F, R8N]))
    tpl = P(rng, split, pool)
    to_bridge = "arbitrum" in tpl.lower()
    first = tpl.format(a=amt_user(rng, asked))
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
    c.user(decorate(rng, first))
    bres = c.call1("get_balances", {"chain": "base"})
    have = find_bal(bres, "USDC", "base")
    to_sym, to_ch = ("USDC", "arbitrum") if to_bridge else ("ETH", "base")
    if D(have) >= asked:
        amount, note = asked, ""
    elif case == "ask" or "if i have" in tpl.lower() and case != "fallback":
        amount, note = None, "ask"
    else:
        amount, note = D(have), "short"
    if note == "ask" or (case == "ask" and amount is None):
        thing = "bridge" if to_bridge else "swap"
        c.say(rng.choice([
            f"You have {have} USDC on Base, which is less than the {dstr(asked)} you asked for. I haven't prepared anything. Want me to {thing} {have} instead?",
            f"Your Base balance is {have} USDC, short of {dstr(asked)}. Nothing is prepared yet; shall I {thing} {have} USDC instead?",
            f"I only see {have} USDC on Base, not {dstr(asked)}. Do you want me to {thing} {have} instead?"]))
        if rng.random() < 0.35:
            c.user(decorate(rng, P(rng, split, R8Y), first=False))
            args = {"from_symbol": "USDC", "from_chain": "base", "to_symbol": to_sym, "to_chain": to_ch, "amount": num_json(D(have))}
            res = c.call1("prepare_swap", args)
            assert ok_status(res), res
            c.step("s1", "prepare_swap", args)
            c.say(announce_swap(rng, res, "bridge" if to_bridge else "swap", bio))
        return finish_read(c, "balance_changes", {"case": "ask"})
    args = {"from_symbol": "USDC", "from_chain": "base", "to_symbol": to_sym, "to_chain": to_ch, "amount": num_json(amount)}
    res = c.call1("prepare_swap", args)
    assert ok_status(res), res
    c.step("s1", "prepare_swap", args)
    if note == "short":
        lead = rng.choice([f"You have {have} USDC on Base, less than {dstr(asked)}, so I used all of it. ",
                           f"Your Base balance is {have} USDC, below the {dstr(asked)} you named, so I took the full balance. "])
    else:
        lead = rng.choice([f"You have {have} USDC on Base, which covers {dstr(asked)}. ", f"Base shows {have} USDC, enough for {dstr(asked)}. "])
    c.say(lead + announce_swap(rng, res, "bridge" if to_bridge else "swap", bio, cost=False))
    return finish_read(c, "balance_changes", {"case": case})


# ------------------------------------------------------------------ R9: the data is already in the context
R9A1 = [
    "what do i hold on base", "show me my base balances", "whats on base in my wallet", "what's in my wallet on base",
    "check my base holdings", "how much usdc and eth do i have on base", "balances on base pls", "list what i have on base",
]
R9A2 = [
    "swap {fr} of my usdc to eth", "ok swap {fr} my usdc into eth", "put {fr} of that usdc into eth", "convert {fr} my usdc to eth",
    "now swap {fr} of the usdc to eth", "bridge {fr} of my usdc to arbitrum", "ok turn {fr} of the usdc into eth please",
    "swap {fr} of my usdc for eth", "ok send {fr} the usdc over to arbitrum", "Swap {fr} of my USDC to ETH.",
    "now bridge {fr} of that usdc to arbitrum", "ok put {fr} of it into eth",
]
R9FR = [("half", "0.5"), ("50%", "0.5"), ("a quarter", "0.25"), ("25%", "0.25"), ("30%", "0.3"), ("40%", "0.4"), ("60%", "0.6"),
        ("75%", "0.75"), ("20%", "0.2"), ("10%", "0.1"), ("80%", "0.8"), ("a fifth", "0.2")]
R9B1 = ["whats eth at", "eth price?", "how much is eth right now", "price of eth", "what is eth trading at", "eth price please"]
R9B2 = [
    "how much eth is ${x}", "ok how many eth would ${x} buy", "and ${x} worth is how much eth", "what's ${x} in eth at that price",
    "how much eth for ${x} then", "so ${x} gets me how many eth", "convert ${x} to eth for me, just the number",
]
R9C = [
    "i have {h} usdc on base. bridge {a} of it to arbitrum", "my base balance is {h} usdc, bridge {a} to arbitrum",
    "i've got {h} usdc on base, swap {a} of it to eth", "I hold {h} USDC on Base. Swap {a} of it to ETH.",
    "with the {h} usdc i have on base, bridge {a} to arbitrum", "{h} usdc sitting on base: swap {a} into eth",
    "base: {h} usdc. move {a} of that to arbitrum", "i know i have {h} usdc on base, so swap {a} of it to eth, no need to check",
]


def rc_redundant(rng, split, bio):
    w, c = new_conv(rng, split, bio)
    case = rng.choices(["a", "b", "c"], [42, 24, 34])[0]
    w.s["activity"] = old_rows(rng, w, rng.randint(1, 3))
    if case == "a":
        usdc = D(str(rng.choice([120, 240, 360, 500, 640.5, 1000, 87.4])))
        eth = D(str(rng.choice([0.05, 0.1, 0.25, 0.5])))
        w.set_hold("USDC", "base", usdc)
        w.set_hold("ETH", "base", eth)
        add_noise_holdings(rng, w, {"USDC", "ETH"})
        c.user(decorate(rng, P(rng, "train" if split == "train" else "eval", R9A1)))
        bal = c.call1("get_balances", {"chain": "base"})
        parts = [f"{h['amount']} {h['symbol']} (${pnum(h['usd'])})" for h in bal["holdings"]]
        c.say("On Base you have " + " and ".join(parts) + ".")
        tpl = P(rng, split, R9A2)
        frac_words, frac_expr = rng.choice(R9FR)
        if "{fr} my" in tpl and frac_words in ("half", "a quarter", "a fifth"):
            tpl = tpl.replace("{fr} my", "{fr} of my")
        to_arb = "arbitrum" in tpl
        c.user(decorate(rng, tpl.format(fr=frac_words), first=False))
        have = find_bal(bal, "USDC", "base")
        r = c.call1("calculator", {"expression": f"{have} * {frac_expr}"})
        args = {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "USDC" if to_arb else "ETH",
                "to_chain": "arbitrum" if to_arb else "base", "amount": num_json(D(r["result"]))}
        res = c.call1("prepare_swap", args)
        assert ok_status(res), res
        c.step("s1", "prepare_swap", args)
        c.say(f"{cap(frac_words)} of your {have} USDC is {r['result']}. " + announce_swap(rng, res, "bridge" if to_arb else "swap", bio, cost=False))
        return finish_read(c, "redundant_read", {"case": "a"}, cut_last=True)
    if case == "b":
        c.user(decorate(rng, P(rng, "train" if split == "train" else "eval", R9B1)))
        pr = c.call1("get_price", {"symbol": "ETH"})
        c.say(f"ETH is at ${pnum(pr['price'])}, {chg_txt(pr['change_24h_pct'])} in the last 24 hours.")
        x = rng.choice([100, 250, 500, 750, 1000, 1500, 2000, 3000])
        c.user(decorate(rng, P(rng, split, R9B2).format(x=x), first=False))
        r = c.call1("calculator", {"expression": f"{x} / {dstr(pr['price'])}"})
        c.say(rng.choice([f"${x} is about {approx(float(r['result']))} ETH at ${pnum(pr['price'])}.",
                          f"At ${pnum(pr['price'])} per ETH, ${x} comes to about {approx(float(r['result']))} ETH."]))
        return finish_read(c, "redundant_read", {"case": "b"}, cut_last=True)
    h = D(str(rng.choice([120, 250, 400, 600, 900, 1500])))
    a = D(str(rng.choice([x for x in [20, 40, 50, 75, 100, 150, 200, 300] if x < float(h)])))
    w.set_hold("USDC", "base", h)
    add_noise_holdings(rng, w, {"USDC"})
    tpl = P(rng, split, R9C)
    to_arb = "arbitrum" in tpl
    c.user(decorate(rng, tpl.format(h=dstr(h), a=dstr(a))))
    args = {"from_symbol": "USDC", "from_chain": "base", "to_symbol": "USDC" if to_arb else "ETH", "to_chain": "arbitrum" if to_arb else "base",
            "amount": num_json(a)}
    res = c.call1("prepare_swap", args)
    assert ok_status(res), res
    c.step("s1", "prepare_swap", args)
    c.say(announce_swap(rng, res, "bridge" if to_arb else "swap", bio))
    return finish_read(c, "redundant_read", {"case": "c"})


# ------------------------------------------------------------------ registry of scenarios and allocation
MONEY_SHAPES = [("bridge_deposit", sc_bridge_deposit), ("swap_send", sc_swap_send), ("withdraw_then", sc_withdraw_then),
                ("close_withdraw", sc_close_withdraw), ("sell_send", sc_sell_send), ("independent", sc_independent),
                ("three_steps", sc_three_steps), ("planning", sc_planning)]
READ_KINDS = [("swap_quote", rc_swap_quote, 12, 6), ("risk_trade", rc_risk_trade, 18, 8), ("parallel_prices", rc_parallel_prices, 12, 5),
              ("yield_deposit", rc_yield_deposit, 16, 7), ("partial_close", rc_partial_close, 16, 7),
              ("funding_annualised", rc_funding, 14, 6), ("long_chain", rc_long_chain, 22, 10),
              ("balance_changes", rc_balance_changes, 16, 6), ("redundant_read", rc_redundant, 14, 5)]


def allocate(total, weights):
    s = sum(weights)
    raw = [total * w / s for w in weights]
    out = [int(x) for x in raw]
    rem = total - sum(out)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - out[i]))
    for i in order[:rem]:
        out[i] += 1
    return out


def key_text(rec):
    m = rec["meta"]
    users = [x["content"] for x in rec["messages"] if x["role"] == "user"]
    return users[m["cut"]] if m.get("cut") is not None else users[0]


def generate(seed, n_total, split, banned, money_share=0.8, eval_mode=False):
    rng = random.Random(seed)
    n_money = int(round(n_total * money_share))
    n_read = n_total - n_money
    m_counts = allocate(n_money, [1] * len(MONEY_SHAPES))
    r_counts = allocate(n_read, [(k[3] if eval_mode else k[2]) for k in READ_KINDS])
    jobs = [(name, fn, n) for (name, fn), n in zip(MONEY_SHAPES, m_counts)] + \
           [(k[0], k[1], n) for k, n in zip(READ_KINDS, r_counts)]
    out, seen_keys, seen_norms = [], set(), set()
    stats = Counter()
    errs = {}
    for name, fn, n in jobs:
        made = 0
        tries = 0
        while made < n:
            tries += 1
            if tries > 400 + 60 * n:
                raise RuntimeError(f"{name}: could not make {n} unique records ({made} done), errors: {errs.get(name)}")
            sub = random.Random(rng.getrandbits(64))
            bio = pick_bio(sub)
            try:
                rec = fn(sub, split, bio)
            except Retry:
                stats[name + ":retry"] += 1
                continue
            except (AssertionError, KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as e:
                import traceback
                stats[name + ":error"] += 1
                errs.setdefault(name, traceback.format_exc())
                continue
            users = [x["content"] for x in rec["messages"] if x["role"] == "user"]
            kt = norm(key_text(rec))
            norms = [norm(u) for u in users]
            if kt in seen_keys or kt in banned or any(x in banned for x in norms):
                stats[name + ":dup"] += 1
                continue
            seen_keys.add(kt)
            rec["meta"]["key_text"] = key_text(rec)
            out.append(rec)
            made += 1
    rng.shuffle(out)
    return out, stats, errs


# ------------------------------------------------------------------ eval items
GATES = ["no_unbacked_done", "no_early_prepare", "step_order", "exact_digits", "chain_complete", "step_acc", "terminates",
         "wrong_money", "redundant_reads"]
BANNED_RE_STRINGS = [
    r"\bRelay\b", r"went through", r"gone through", r"(?<!not )(?<!nothing )(?<!n't )\bhas been (sent|swapped|bridged|deposited|withdrawn|closed|sold|completed)",
    r"\bI(?:'ve| have)? (sent|bridged|swapped|deposited|withdrew|closed|sold)\b", r"successfully", r"funds (are|have) (in|arrived)",
    r"^\s*done\b", r"\bdone\b", r"not financial advice", "[·•]", r"\bAs an AI\b", r"\ball set\b", r"\bprobably\b", r"\bmaybe\b",
    r"\bI think\b", r"\bcompleted\b", r"when it arrives", r"I'll deposit it", r"I'll send it", r"will be deposited", r"\bshould have\b",
    r"what('s| is) (the )?(current )?price", r"your budget", r"can't see the future", r"\"name\"", r"tool_call", r"\bautomatic(ally)?\b",
]
BANNED_RE = [re.compile(p, re.I) for p in BANNED_RE_STRINGS]


def to_eval(rec, idx):
    m = rec["meta"]
    msgs = rec["messages"]
    uidx = [i for i, x in enumerate(msgs) if x["role"] == "user"]
    cut_turn = m["cut"]
    cut = uidx[cut_turn]
    ctx, ref = msgs[:cut + 1], msgs[cut + 1:]
    # per-turn gold rounds from the messages after each user turn
    turns = []
    cur = None
    for x in msgs[1:]:
        if x["role"] == "user":
            cur = {"user": x["content"], "rounds": [], "text": None}
            turns.append(cur)
        elif x["role"] == "assistant":
            if x.get("tool_calls"):
                cur["rounds"].append([{"name": c_["name"], "arguments": c_["arguments"]} for c_ in x["tool_calls"]])
            else:
                cur["text"] = x["content"]
                cur["rounds"].append("text")
    group = "read_chain" if m.get("suite_kind") == "read" else "money_seq"
    ac = {"suite": "chain_multi", "kind": m["kind"], "group": group, "steps": m["steps"], "money_by_turn": m["money_by_turn"],
          "turns": turns, "first_turn": cut_turn, "max_rounds": 5, "max_calls_per_turn": 6, "max_identical_calls": 2,
          "max_redundant_reads": 1, "banned_regex": BANNED_RE_STRINGS, "n_steps": len(m["steps"]), "outcome": m["tag"],
          "steps_reachable": len([st for st in m["steps"] if not st.get("blocked")]),
          "no_prepare_turns": [i for i, ms in enumerate(m["money_by_turn"]) if not ms]}
    item = {"id": f"chain_multi-{idx}", "kind": m["kind"], "suite": "chain_multi", "tools": True, "bio": m["bio"], "context": ctx,
            "reference": ref, "expect": m["expect"], "script": m["script"], "pack": m["pack"], "family": m["family_ok"],
            "world": {"stages": m["world"]["stages"], "first_turn": cut_turn}, "checks": GATES, "answer_checks": ac,
            "meta": {"generator": SRC, "tag": m["tag"], "shape": m["shape"], "device": m["device"], "group": group}}
    return item


# ------------------------------------------------------------------ self-check
ADDR_EVM = re.compile(r"0x[0-9a-fA-F]{40}")
B58_RE = re.compile(r"(?<![A-Za-z0-9])[1-9A-HJ-NP-Za-km-z]{32,44}(?![A-Za-z0-9])")
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
STRIP_PHRASES = ["Aave v3", "24 hours", "24h", "1Click", "v3"]
REQUIRED_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}
NEXT_RE = re.compile(r"\b(lands?|landed|next|once|after|approved|depends|waits?|isn't prepared|not prepared|unprepared|stopped|comes)\b", re.I)
COND_RE = re.compile(r"\b(until|once|when|if|whether|before|unless|after|not|n't|no row|can't|cannot|nothing)\b", re.I)
CONFIRM_RE = re.compile(r"\bconfirmed\b|\blanded\b|\barrived\b", re.I)
TYPO_WORDS = set(w2 for v in TYPOS.values() for w2 in v)


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
        okt = {"number": isinstance(v, (int, float)) and not isinstance(v, bool), "integer": isinstance(v, int) and not isinstance(v, bool),
               "string": isinstance(v, str), "boolean": isinstance(v, bool), "array": isinstance(v, list), "object": isinstance(v, dict)}.get(t, True)
        if not okt:
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


def turns_of(msgs):
    """[{user, rounds:[[(name,args,result_str)]], text, texts}] per user turn, from the message list."""
    turns = []
    i = 1
    n = len(msgs)
    while i < n:
        m = msgs[i]
        if m["role"] == "user":
            turns.append({"user": m["content"], "rounds": [], "texts": []})
            i += 1
        elif m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            res = msgs[i + 1:i + 1 + k]
            assert len(res) == k and all(r["role"] == "tool" for r in res), "tool messages do not match the calls"
            turns[-1]["rounds"].append([(c_["name"], c_["arguments"], r["content"]) for c_, r in zip(m["tool_calls"], res)])
            i += 1 + k
        elif m["role"] == "assistant":
            turns[-1]["texts"].append(m["content"])
            i += 1
        else:
            raise AssertionError(f"unexpected message order at {i}: {m['role']}")
    return turns


def nums_in(s):
    out = []
    for x in NUM_RE.findall(s):
        try:
            out.append(Decimal(x.replace(",", "")))
        except Exception:
            pass
    return out


def flat_numbers(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(abs(Decimal(repr(obj))))
    elif isinstance(obj, str):
        out.extend(nums_in(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            flat_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flat_numbers(v, out)


def places(x):
    return max(0, -x.as_tuple().exponent)


def num_backed(x, sources):
    p = places(x)
    for s in sources:
        if s == x:
            return True
        if p <= 6:
            q = Decimal(1).scaleb(-p)
            for rnd in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                if s.quantize(q, rounding=rnd) == x:
                    return True
    return False


def clean_for_nums(text):
    for ph in STRIP_PHRASES:
        text = text.replace(ph, " ")
    text = ADDR_EVM.sub(" ", text)
    text = B58_RE.sub(" ", text)
    return text


def check_record(ex, who):
    m = ex["meta"]
    msgs = ex["messages"]
    bio = m["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    assert ex["tools"] is True and m["source"] == SRC
    blob = json.dumps(msgs, ensure_ascii=False)
    assert "{BIO}" not in blob and other not in blob, f"{who}: bio word mismatch"
    assert not re.search(r"[{}]", "".join(x["content"] for x in msgs if x["role"] in ("user", "assistant"))), f"{who}: brace left in a text"
    assert "None" not in "".join(x["content"] for x in msgs if x["role"] in ("user", "assistant")), f"{who}: None in a text"
    s0 = msgs[0]
    assert s0["role"] == "system" and s0["content"].startswith(SYSTEM_V3.replace("{BIO}", bio)), f"{who}: system prompt"
    extra = s0["content"][len(SYSTEM_V3.replace("{BIO}", bio)):]
    assert extra == "" or re.fullmatch(r"\nThe user wants to be called \w+\.", extra), f"{who}: system suffix"
    pack = set(m["pack"])
    assert len(m["pack"]) <= 22 or "4b" not in m["family_ok"], f"{who}: pack too big"
    turns = turns_of(msgs)
    users = [x["content"] for x in msgs if x["role"] == "user"]
    assert len(turns) == len(users) == len(m["world"]["stages"]) == m["n_user_turns"], f"{who}: turn counts"
    assert m["script"] == users[1:] or m.get("cut", 0) > 0
    # ---- replay every call on the recorded stage; compare byte for byte
    resolved = set()
    all_results = []
    typed = []
    seen_triples = Counter()
    for k, t in enumerate(turns):
        seen_calls = Counter()
        typed.append(t["user"])
        w = World.from_state(m["world"]["stages"][k], typed=typed, resolved=resolved)
        assert 1 <= len(t["rounds"]) + len(t["texts"]) <= 6, f"{who}: too many rounds in turn {k}"
        assert len(t["rounds"]) <= 4 and len(t["texts"]) == 1, f"{who}: turn {k} rounds {len(t['rounds'])} texts {len(t['texts'])}"
        assert len(t["rounds"]) + 1 <= 5, f"{who}: more than 5 model rounds in turn {k}"
        money_here = 0
        for rd in t["rounds"]:
            assert len(rd) <= 6, f"{who}: more than 6 calls in a round"
            for name, args, content in rd:
                assert name in TOOL_SCHEMA, f"{who}: unknown tool {name}"
                assert name in pack, f"{who}: {name} not in the pack"
                sch = TOOL_SCHEMA[name]
                errs = validate_args(args, sch)
                for r_ in REQUIRED_OVERRIDE.get(name, sch.get("required", [])):
                    if r_ not in args:
                        errs.append(f"missing {r_}")
                assert not errs, f"{who}: {name} {errs}"
                res = w.call(name, args)
                again = dumps_result(res, bio)
                assert again == content, f"{who}: replay differs for {name}\n{again}\n{content}"
                try:
                    json.loads(content)
                except Exception:
                    raise AssertionError(f"{who}: tool content is not JSON")
                key = (name, json.dumps(args, sort_keys=True))
                seen_calls[key] += 1
                assert seen_calls[key] <= 2, f"{who}: identical call {name} more than twice in a turn"
                if name in READ_TOOLS and name not in ("calculator", "get_activity"):
                    tr = (name, json.dumps(args, sort_keys=True), content)
                    if m.get("suite_kind") != "read":
                        tr = tr + (k,)          # money chains re-check on purpose across turns; never twice in one turn
                    seen_triples[tr] += 1
                    assert seen_triples[tr] == 1, f"{who}: redundant read {name} {args}"
                if name in MONEY_TOOLS:
                    money_here += 1
                all_results.append((k, name, args, json.loads(content)))
            # keep the dynamic guard state in sync (get_address resolves contacts)
        resolved = w.resolved
        assert money_here <= 1, f"{who}: {money_here} reviews prepared in turn {k}"
        # no parallel money together with another money tool; one prepare per kind per turn is implied above
    # ---- texts
    sources = []
    for u in users:
        sources += nums_in(u)
    texts_by_turn = []
    for k, t in enumerate(turns):
        sources_k = list(sources)
        for (kk, name, args, res) in all_results:
            if kk <= k:
                flat_numbers(res, sources_k)
                flat_numbers(args, sources_k)
        for u in users[:k + 1]:
            sources_k += nums_in(u)
        text = t["texts"][0]
        texts_by_turn.append(text)
        assert text.strip() and len(text.split()) <= 125, f"{who}: text length {len(text.split())} words in turn {k}"
        for rx in BANNED_RE:
            assert not rx.search(text), f"{who}: banned phrase {rx.pattern!r} in {text!r}"
        assert other not in text and "·" not in text, f"{who}: device word / middle dot"
        for x in nums_in(clean_for_nums(text)):
            assert num_backed(x, sources_k), f"{who}: number {x} in turn {k} not backed: {text!r}"
        # addresses must be copied whole
        for a in ADDR_EVM.findall(text):
            assert a in "\n".join(users[:k + 1]) or any(a in json.dumps(r) for (kk, n_, ar, r) in all_results if kk <= k), f"{who}: address not from the chat"
        for frag in re.findall(r"0x[0-9a-fA-F]{6,}", text):
            assert len(frag) == 42, f"{who}: truncated address {frag!r}"
        for a in B58_RE.findall(text):
            assert a in "\n".join(users[:k + 1]) or any(a in json.dumps(r) for (kk, n_, ar, r) in all_results if kk <= k), f"{who}: base58 not from the chat"
        # a confirmed / landed claim needs a confirmed row read in this turn
        confirmed_here = False
        for rd in t["rounds"]:
            for name, args, content in rd:
                if name == "get_activity" and any(r_["status"] == "confirmed" for r_ in json.loads(content)["activity"]):
                    confirmed_here = True
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if CONFIRM_RE.search(sent) and not COND_RE.search(sent):
                assert confirmed_here, f"{who}: confirmed claim without a confirmed row: {sent!r}"
        assert not re.search(r"I've prepared[^.]*\b(deposit|send|bridge|swap)\b[^.]*\bit has landed", text)
    # ---- addresses in calls come from the chat
    typed_all = "\n".join(users)
    for (k, name, args, res) in all_results:
        txt = json.dumps(args)
        for a in ADDR_EVM.findall(txt) + B58_RE.findall(txt):
            seen = "\n".join(users[:k + 1]) + "".join(json.dumps(r) for (kk, n_, ar, r) in all_results if kk < k or (kk == k and n_ == "get_address"))
            assert a in seen, f"{who}: address in a call is not from the chat"
    # ---- step order
    for st in m["steps"]:
        t = turns[st["turn"]]
        found = None
        for ri, rd in enumerate(t["rounds"]):
            for name, args, content in rd:
                if name == st["name"] and args == st["arguments"]:
                    found = ri
        assert found is not None, f"{who}: step {st['id']} call not in turn {st['turn']}"
        if st["needs"]:
            ok_status_ = set(st["needs"]["status"].split("|"))
            good = False
            for rd in [rd_ for tt in turns[:st["turn"]] for rd_ in tt["rounds"]] + list(t["rounds"][:found]):
                for name, args, content in rd:
                    if name == "get_activity":
                        for r_ in json.loads(content)["activity"]:
                            if r_["what"] == st["needs"]["row"] and r_["status"] in ok_status_:
                                good = True
            assert good, f"{who}: step {st['id']} prepared without an activity row for step one"
            if st["name"] in ("prepare_deposit", "prepare_send", "prepare_swap"):
                amount = D(st["arguments"].get("amount", 0))
                srcs = set()
                prior = [rd for tt in turns[:st["turn"]] for rd in tt["rounds"]] + list(t["rounds"][:found])
                for rd in prior:
                    for name, args, content in rd:
                        r_ = json.loads(content)
                        if name == "get_balances":
                            srcs |= {D(h["amount"]) for h in r_["holdings"]}
                        elif name == "calculator" and "result" in r_:
                            srcs.add(D(r_["result"]))
                srcs |= {x for u in users for x in nums_in(u)}
                if st["name"] == "prepare_swap" and not any(True for _ in srcs):
                    pass
                assert amount in srcs, f"{who}: step amount {amount} not from a balance string, calculator result or the user"
                # never the quote's amount_out when it differs from the balance
                for (kk, name, args, res) in all_results:
                    if kk <= st["turn"] and name == "prepare_swap" and "amount_out" in res:
                        bal_strs = set()
                        for rd in prior:
                            for n2, a2, c2 in rd:
                                if n2 == "get_balances":
                                    bal_strs |= {D(h["amount"]) for h in json.loads(c2)["holdings"]}
                        if amount == D(res["amount_out"]) and amount not in bal_strs and amount not in {x for u in users for x in nums_in(u)}:
                            raise AssertionError(f"{who}: step amount is the quote's amount_out")
        else:
            # no dependency: a step with needs None must be the first money call of the whole chat or an explicit retry
            pass
    # first assistant text of a dependent chain names the next step
    if m["shape"] in ("bridge_deposit", "swap_send", "withdraw_then", "sell_send", "three_steps") and m["steps"] and m["steps"][0]["turn"] == 0 \
            and m["tag"] != "blocked0" and m.get("n_user_turns", 1) >= 1:
        if not (m["shape"] == "bridge_deposit" and m["n_user_turns"] > 1 and m["tag"] == "full" and False):
            assert NEXT_RE.search(texts_by_turn[0]), f"{who}: turn 0 text does not name the next step"
    # estimate prompt size with the pack block
    lines = "".join("\n" + json.dumps(t_, ensure_ascii=False) for t_ in TOOLS_V3 if t_["function"]["name"] in pack)
    est = (len(blob) + len(lines) + 400) / 3.3
    assert est < (6144 if "lite" in m["family_ok"] else 8192), f"{who}: prompt too long {est}"
    # render round trip
    saved = H.TOOLS_BLOCK
    H.TOOLS_BLOCK = saved.replace("Face ID", bio)
    try:
        H.render_text(msgs, True)
    finally:
        H.TOOLS_BLOCK = saved
    for x in msgs:
        if x["role"] == "assistant":
            calls, bad = H.parse_calls(H.assistant_text(x))
            want = [{"name": c_["name"], "arguments": c_["arguments"]} for c_ in (x.get("tool_calls") or [])]
            assert bad == 0 and calls == want, f"{who}: render round trip failed"
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--eval-seed", type=int, default=9404)
    ap.add_argument("--eval-n", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(TRAIN_DIR, "data4", "multistep.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(TRAIN_DIR, "eval4", "multistep.jsonl"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    assert a.seed != a.eval_seed
    ev, est_, eerrs = generate(a.eval_seed, a.eval_n, "eval", set(), money_share=0.5, eval_mode=True)

    def specific(rec):
        us = [x["content"] for x in rec["messages"] if x["role"] == "user"]
        return {norm(u) for u in us if len(norm(u)) > 25}
    banned = set()
    for r_ in ev:
        banned |= specific(r_)
        banned.add(norm(key_text(r_)))
    tr, tst, terrs = generate(a.seed, a.n, "train", banned)
    items = [to_eval(r_, i) for i, r_ in enumerate(ev)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r_ in tr:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    failures = []
    soft = []
    max_tok = 0
    try:
        rows_t = [json.loads(l) for l in open(a.out, encoding="utf-8")]
        rows_e = [json.loads(l) for l in open(a.eval_out, encoding="utf-8")]
        assert len(rows_t) == len(tr) and len(rows_e) == len(items)
        seen = {}
        for i, ex in enumerate(rows_t):
            try:
                max_tok = max(max_tok, check_record(ex, f"train[{i}]"))
            except AssertionError as e:
                soft.append(str(e))
            kt = norm(ex["meta"]["key_text"])
            assert kt not in seen, f"duplicate train user text: {ex['meta']['key_text']!r}"
            seen[kt] = i
            assert ex["meta"]["template_split"] == "train"
            assert 1 <= ex["meta"]["n_user_turns"] <= 6
        seen_e = set()
        for i, (it, ex) in enumerate(zip(rows_e, ev)):
            for k in ("id", "kind", "tools", "bio", "context", "reference", "expect", "world", "checks", "answer_checks", "script", "pack"):
                assert k in it, f"eval[{i}] missing {k}"
            try:
                max_tok = max(max_tok, check_record(ex, it["id"]))
            except AssertionError as e:
                soft.append(str(e))
            assert it["context"] + it["reference"] == ex["messages"], f"eval[{i}] context+reference != conversation"
            assert it["context"][-1]["role"] == "user"
            assert it["answer_checks"]["steps"] is not None
            kt = norm(ex["meta"]["key_text"])
            assert kt not in seen and kt not in seen_e, f"eval text not disjoint: {ex['meta']['key_text']!r}"
            seen_e.add(kt)
            for u in specific(ex):
                assert u not in {n for r_ in tr for n in specific(r_)}, f"eval user text in train: {u!r}"
            assert ex["meta"]["template_split"] == "eval"
            first_turn = it["world"]["first_turn"]
            assert len(it["world"]["stages"]) == len([x for x in ex["messages"] if x["role"] == "user"])
            assert it["script"] == [x["content"] for x in ex["messages"] if x["role"] == "user"][first_turn + 1:]
        # counts
        gm = Counter(r_["meta"]["group"] for r_ in rows_e) if False else Counter(i_["meta"]["group"] for i_ in rows_e)
        assert gm["money_seq"] == a.eval_n // 2 and gm["read_chain"] == a.eval_n - a.eval_n // 2, f"eval groups {gm}"
        bios = Counter(r_["meta"]["bio"] for r_ in rows_t)
        share = bios["Touch ID"] / len(rows_t)
        assert 0.22 <= share <= 0.38, f"Touch ID share {share:.2f}"
        assert abs(len(rows_t) - a.n) <= 0, f"train count {len(rows_t)}"
        if soft:
            raise AssertionError(f"{len(soft)} record checks failed; first: {soft[0]}")
    except AssertionError as e:
        failures.append(str(e))
        for m_ in soft[:int(os.environ.get("MS_SHOW", "25"))]:
            print("  FAIL", m_[:600], file=sys.stderr)
    shapes_t = Counter(r_["meta"]["shape"] for r_ in tr)
    tags_t = Counter(r_["meta"]["tag"] for r_ in tr if r_["meta"].get("suite_kind") != "read")
    shapes_e = Counter(r_["meta"]["shape"] for r_ in ev)
    turns_t = Counter(r_["meta"]["n_user_turns"] for r_ in tr)
    print("train shapes:", dict(shapes_t), file=sys.stderr)
    print("train tags:", dict(tags_t), file=sys.stderr)
    print("train user turns:", dict(sorted(turns_t.items())), file=sys.stderr)
    print("eval shapes:", dict(shapes_e), file=sys.stderr)
    print("bio:", dict(Counter(r_["meta"]["bio"] for r_ in tr)), " est max tokens:", int(max_tok), file=sys.stderr)
    print("build stats train:", {k: v for k, v in tst.items() if k.endswith((":error", ":dup"))}, file=sys.stderr)
    for k, v in {**terrs, **eerrs}.items():
        if a.verbose:
            print("FIRST ERROR", k, v, file=sys.stderr)
    if failures:
        print(f"multistep: {len(tr)} train, {len(items)} eval, self-check FAILED: {failures[0]}")
        sys.exit(1)
    print(f"multistep: {len(tr)} train, {len(items)} eval, self-check ok")


if __name__ == "__main__":
    main()
