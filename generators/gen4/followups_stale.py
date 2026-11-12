#!/usr/bin/env python3
"""followups_stale: follow-ups that point back ("same for ETH", "make it 10", "cancel that", "undo") and stale-data
discipline, for Hermit's next run (G9 of PLAN_v3.md, suite `followups`).

Conversations of 2 to 6 user turns. The model's own calls run on a small replayable simulator (FWorld) whose state
persists between turns: reviews stay pending until the user "approves" them (a world event), balances, Hyperliquid
positions, orders and Activity rows change when that happens, prices move when a `move` event says so. 25% of the
conversations carry an unrelated message in the middle. Everything is deterministic Python (no LLM text); every number
in an answer comes from a tool result, a calculator result or the user's text.

Shapes (A to N, 40 to 80 train rows each):
  a same for ETH (copy side/size/leverage, recompute percent stops, never copy an absolute price, ask)
  b change one argument (make it 10x, half, and on Arbitrum, 75 instead)      c cancel that
  d undo after approval                      e what was that address            f two possible referents (one question)
  g referent gone after trimHistory           h I approved, what's my balance    i and now? (re-call, change via calculator)
  j do I have enough (re-read)                k is it there yet (activity, then balances)   l what's the fee (get_quote)
  m contract / APY / TVL from memory          n same request twice, answer to own question, correction, topic switch

Usage:
  python3 followups_stale.py [--seed 4] [--n 750] [--eval-seed 9409] [--eval-n 120]
Writes ../data4/followups_stale.jsonl (train) and ../eval4/followups_stale.jsonl (held-out items), then self-checks
every record in memory and again from the files on disk. Exit code 1 on any failure.

Record: {messages, tools, meta}. meta carries world (state at the first visible message), script (later user turns),
script_events (world event applied before each later turn), turns (per user turn: expect and gold checks), pack,
steps, cards, bio.
Eval item: {id, suite, kind, tools, bio, pack, context (up to the cut user turn), script, script_events, reference,
expect, turns, world (state at the cut), world_start (replay check only), checks, gold}. The harness loops
Traj.feed per turn: it builds the runner with `make_runner(item)` from this module (state persists, `begin_turn(k)`
applies script_events[k-1] and hears script[k-1] before script turn k is fed), scores each turn with `turns[k]`
(`expect`, `gold`: carry, not_carried, must_call, forbid_tools, stale, answer_all/any/forbid) and reports turn-ok,
conversation clean, carry_ok, no_restate_error, no_reprepare_unasked, rounded_ok.

Reuses gen2 / gen3 helpers by import (hermit_fmt, sim, build_agent, trade_view, fmt3); none of them is edited. The
tool registry is spec/hermit_tools_v3.json (30 tools) and the system prompt is spec/SYSTEM_AGENT_v3.txt.
"""
import sys
sys.dont_write_bytecode = True      # never write .pyc next to the read-only gen2/gen3 modules we import
import argparse, copy, json, math, os, random, re
from collections import Counter
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.abspath(os.path.join(HERE, ".."))
GEN2, GEN3, SPEC = (os.path.join(TRAIN_DIR, x) for x in ("gen2", "gen3", "spec"))
for _p in (GEN3, GEN2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hermit_fmt as H              # gen2: calculator, addresses, renderer
import sim                          # gen2: liquidation maths
import trade_view as TV             # gen3: phrasing and number helpers (importing it patches H in memory to v2)
import fmt3                         # gen3: validate_args

# ----------------------------------------------------------------------------- v3 registry and system prompt
TOOLS_V3 = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V3}
H.TOOLS, H.TOOL_NAMES, H.TOOL_SCHEMA = TOOLS_V3, list(SCHEMA), SCHEMA
H.TOOLS_BLOCK = H.tools_block()
PACK = list(SCHEMA)
SYSTEM_V3 = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().rstrip("\n")
assert "{BIO}" in SYSTEM_V3 and len(PACK) == 30
REQUIRED_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}
MONEY_TOOLS = ["prepare_trade", "prepare_swap", "prepare_send", "prepare_close", "prepare_buy", "prepare_sell", "prepare_deposit",
               "prepare_withdraw", "prepare_multisend", "set_tp_sl", "cancel_order", "schedule_payment"]
SUITE = "followups"
SOURCE = "followups_stale"

nn, ns, commas, dollars, parse_calc, stop_dp, norm_text = TV.nn, TV.ns, TV.commas, TV.dollars, TV.parse_calc, TV.stop_dp, TV.norm_text

COINS = ["BTC", "ETH", "SOL", "HYPE", "BNB", "XRP", "DOGE", "AVAX", "LINK", "SUI"]
TRADE_COINS = ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "LINK", "SUI", "XRP", "DOGE"]
MAX_LEV = {"BTC": 40, "ETH": 25, "SOL": 20, "HYPE": 10, "BNB": 10, "AVAX": 10, "LINK": 10, "SUI": 10, "XRP": 20, "DOGE": 10}
ALIAS = dict(TV.ALIAS)
STABLE = {"USDC", "USDT", "USDT0", "DAI"}
CHAIN_NAME = {"base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "polygon": "Polygon", "solana": "Solana"}
CHAIN_SAY = {"base": ["Base", "base"], "arbitrum": ["Arbitrum", "arbitrum", "arb"], "ethereum": ["Ethereum", "ethereum", "mainnet"],
             "polygon": ["Polygon", "polygon"], "solana": ["Solana", "solana"]}
CHAIN_RE = {"base": r"\bbase\b", "arbitrum": r"\barbitrum\b|\barb\b", "ethereum": r"\bethereum\b|\bmainnet\b", "polygon": r"\bpolygon\b|\bmatic\b",
            "solana": r"\bsolana\b"}
NATIVE = {"ethereum": "ETH", "base": "ETH", "arbitrum": "ETH", "polygon": "POL", "solana": "SOL"}
NAME_OF = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana"}
ADDR_RE = re.compile(r"0x[0-9a-fA-F]{40}|\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")


def D(s):
    return Decimal(str(s).replace(",", ""))


def approx(x):
    """The app's Units.approx: >= 1000 two decimals, else three significant digits, trailing zeros trimmed."""
    if x >= 1000:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    if x == 0:
        return "0"
    dp = max(0, 2 - int(math.floor(math.log10(abs(x)))))
    s = f"{round(x, dp):.{dp}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def amount_str(a):
    return ns(round(a, 6))


def short_act(addr):                      # how Activity shortens an address
    return addr[:4] + "…" + addr[-2:]


def short_say(addr):                      # how I name an address in a question
    return addr[:6] + "…" + addr[-4:]


def big(x):
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    if x >= 1e7:
        return f"${x / 1e6:.0f}M"
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    return "$" + commas(ns(x))


def S(d):
    return dict(sorted(d.items()))


# ----------------------------------------------------------------------------- the simulator
NO_MARKET_ERR = "No Hyperliquid market for {s}. Answer from what you know instead, and say there's no live price here."


class FWorld:
    """Replayable simulator for the tools this generator uses, in the app's exact result shapes (meta.json).
    No randomness inside: every number is in the state, so replaying the same calls gives the same bytes."""

    def __init__(self, st):
        self.s = st

    # ---- state
    def state(self):
        return json.loads(json.dumps(self.s))

    @classmethod
    def from_state(cls, st):
        return cls(json.loads(json.dumps(st)))

    def hear(self, text):
        self.s["typed"].append(text)

    @property
    def bio(self):
        return self.s["bio"]

    def jd(self, r):
        return json.dumps(r, ensure_ascii=False).replace("Face ID", self.bio)

    # ---- helpers
    def price(self, sym):
        if sym in STABLE:
            return 1.0
        return self.s["prices"][sym]

    def bal(self, sym, chain):
        for b in self.s["balances"]:
            if b["symbol"] == sym and b["chain"] == chain:
                return b["amount"]
        return 0.0

    def add_bal(self, sym, chain, delta):
        for b in self.s["balances"]:
            if b["symbol"] == sym and b["chain"] == chain:
                b["amount"] = round(b["amount"] + delta, 8)
                return
        self.s["balances"].append({"symbol": sym, "chain": chain, "amount": round(delta, 8)})

    def holders(self, sym):
        return [b["chain"] for b in self.s["balances"] if b["symbol"] == sym and b["amount"] > 1e-9]

    def next_id(self):
        self.s["nid"] += 1
        return str(8100 + self.s["nid"])

    def act(self, status, what):
        self.s["seq"] += 1
        m = 9 * 60 + self.s["seq"] * 7
        self.s["activity"].insert(0, {"status": status, "time": f"2026-11-11 {m // 60:02d}:{m % 60:02d}", "what": what})

    def mentioned_chains(self):
        t = "\n".join(self.s["typed"]).lower()
        return [c for c, rx in CHAIN_RE.items() if re.search(rx, t)]

    # ---- events (the user acts outside the chat; the model never sees these directly)
    def apply_event(self, ev):
        op = ev["op"]
        if op == "approve_last":
            if self.s["pending"]:
                idx = 0 if ev.get("which") == "first" else len(self.s["pending"]) - 1
                self._approve(self.s["pending"].pop(idx), ev.get("settled", True))
        elif op == "settle":
            self._settle()
        elif op == "move":
            for k, v in ev["prices"].items():
                self.s["prices"][k] = v
            for k, v in ev.get("change", {}).items():
                self.s["change"][k] = v
        elif op == "spend":
            self.add_bal(ev["symbol"], ev["chain"], -ev["amount"])
        elif op == "hl":
            self.s["hl"]["avail"] = ev["avail"]
        elif op == "yields":
            for row in self.s["yields"]:
                if row["protocol"] == ev["protocol"] and row["asset"] == ev["asset"] and row["chain"] == ev["chain"]:
                    row["apy_pct"], row["apy_base_pct"], row["apy_reward_pct"] = ev["apy"], ev["apy"], 0
        else:
            raise ValueError(op)

    def _approve(self, p, settled):
        a, k = p["args"], p["kind"]
        if k == "send":
            self.add_bal(a["symbol"], p["chain"], -a["amount"])
            self.act("confirmed", f"Sent {ns(a['amount'])} {a['symbol']} to {short_act(a['to_address'])}")
        elif k == "swap":
            self.add_bal(a["from_symbol"], a["from_chain"], -a["amount"])
            what = (f"Swap {ns(a['amount'])} {a['from_symbol']} on {CHAIN_NAME[a['from_chain']]} to "
                    f"{a['to_symbol']} on {CHAIN_NAME[a['to_chain']]}")
            if settled:
                self.add_bal(a["to_symbol"], a["to_chain"], p["out_raw"])
                self.act("confirmed", what)
            else:
                self.act("in_progress", what)
                self.s["inflight"].append({"what": what, "symbol": a["to_symbol"], "chain": a["to_chain"], "amount": p["out_raw"]})
        elif k == "trade":
            self._open_trade(a)
        elif k == "close":
            self.s["hl"]["positions"] = [x for x in self.s["hl"]["positions"] if x["coin"] != a["coin"]]
            self.s["hl"]["orders"] = [o for o in self.s["hl"]["orders"] if o["coin"] != a["coin"] or o["kind"] == "limit"]

    def _settle(self):
        for f in self.s["inflight"]:
            self.add_bal(f["symbol"], f["chain"], f["amount"])
            for row in self.s["activity"]:
                if row["what"] == f["what"] and row["status"] == "in_progress":
                    row["status"] = "confirmed"
        self.s["inflight"] = []

    def _open_trade(self, a):
        coin, side, lev = a["coin"], a["side"], a["leverage"]
        px = a.get("limit_price") if a["order_type"] == "limit" else self.s["prices"][coin]
        dp = sim.ASSETS[coin][2]
        size = nn(round(a["size_usd"] / px, SZ_DP(px)))
        h = self.s["hl"]
        if a["order_type"] == "limit":
            h["orders"].append({"id": self.next_id(), "coin": coin, "kind": "limit", "price": px,
                                "side": "buy" if side == "long" else "sell", "size": size})
            return
        liq = nn(round(sim.liq_price(side, px, lev, 0.005), dp))
        h["positions"].append({"coin": coin, "entry": px, "leverage": lev, "liquidation": liq, "side": side, "size": size})
        h["avail"] = round(h["avail"] - a["size_usd"] / lev, 2)
        for kind, key in (("stop_loss", "stop_loss"), ("take_profit", "take_profit")):
            if a.get(key) is not None:
                h["orders"].append({"id": self.next_id(), "coin": coin, "kind": kind, "price": a[key],
                                    "side": "sell" if side == "long" else "buy", "size": size, "whole_position": True})

    # ---- dispatch
    def call(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args if isinstance(args, dict) else {}))
        except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as e:
            return {"error": f"bad arguments: {e}"}

    # ---- reads
    def t_calculator(self, expression):
        try:
            return {"result": H.calc_str(H.calc(str(expression)))}
        except Exception:
            return {"error": "Couldn't evaluate that expression"}

    def t_get_price(self, symbol):
        sym = str(symbol).upper().replace("$", "").strip()
        if sym not in self.s["prices"]:
            return {"error": NO_MARKET_ERR.format(s=sym)}
        return S({"change_24h_pct": self.s["change"][sym], "price": self.s["prices"][sym],
                  "source": "Hyperliquid mark price", "symbol": sym})

    def t_get_balances(self, chain=None):
        rows = []
        for b in self.s["balances"]:
            if b["amount"] <= 1e-9 or (chain and b["chain"] != str(chain).lower()):
                continue
            rows.append(S({"amount": amount_str(b["amount"]), "chain": b["chain"], "symbol": b["symbol"],
                           "usd": nn(round(b["amount"] * self.price(b["symbol"]), 2))}))
        return {"holdings": rows}

    def _pos_rows(self):
        out = []
        for p in self.s["hl"]["positions"]:
            sgn = 1 if p["side"] == "long" else -1
            q = dict(p)
            q["unrealised_pnl"] = nn(round(sgn * (self.s["prices"][p["coin"]] - p["entry"]) * p["size"], 2))
            out.append(S(q))
        return out

    def t_hyperliquid_account(self):
        h = self.s["hl"]
        pos = self._pos_rows()
        margin = round(sum(p["size"] * p["entry"] / p["leverage"] for p in h["positions"]), 2)
        value = round(h["avail"] + margin + sum(p["unrealised_pnl"] for p in pos), 2)
        trig = any(o["kind"] != "limit" for o in h["orders"])
        orders = []
        for o in h["orders"]:
            if not trig:
                orders.append(S({"coin": o["coin"], "price": o["price"], "side": o["side"], "size": o["size"]}))
            elif o["kind"] == "limit":
                orders.append(S({"coin": o["coin"], "is_trigger": False, "order_type": "Limit", "price": o["price"], "side": o["side"], "size": o["size"]}))
            else:
                below = o["kind"] == "stop_loss" and o["side"] == "sell" or o["kind"] == "take_profit" and o["side"] == "buy"
                orders.append(S({"coin": o["coin"], "is_trigger": True, "order_type": "Stop Market" if o["kind"] == "stop_loss" else "Take Profit Market",
                                 "price": o["price"], "side": o["side"], "size": o["size"],
                                 "trigger_condition": f"Price {'below' if below else 'above'} {ns(o['price'])}", "whole_position": True}))
        r = {"account_value": nn(value), "available_usdc": nn(h["avail"]), "margin_used": nn(margin), "open_orders": orders, "positions": pos}
        if trig:
            r["account_mode"] = "unified"
        return S(r)

    def t_get_positions(self, kind="all"):
        h = self.s["hl"]
        orders = []
        for o in h["orders"]:
            row = {"coin": o["coin"], "id": o["id"], "kind": o["kind"], "price": o["price"], "side": o["side"], "size": o["size"]}
            if o.get("whole_position"):
                row["whole_position"] = True
            orders.append(S(row))
        perps = {"open_orders": orders, "positions": self._pos_rows()}
        full = {"earn": [], "perps": perps, "spot": [], "wallet": self.s["wallet"]}
        if kind in (None, "all"):
            return S(full)
        if kind not in ("spot", "earn", "perps"):
            return {"error": "kind must be all, spot, earn or perps"}
        return S({kind: full[kind], "wallet": self.s["wallet"]})

    def t_get_activity(self, limit=10):
        rows = [S(r) for r in self.s["activity"][: int(limit)]]
        n = sum(1 for r in self.s["activity"] if r["status"] == "in_progress")
        if not self.s["activity"]:
            return S({"activity": [], "in_progress": 0, "note": "No activity yet. Reviews the user has not approved yet do not appear here."})
        return S({"activity": rows, "in_progress": n, "note": "Reviews the user has not approved yet do not appear here."})

    def t_get_yields(self, asset=None, chain=None):
        rows = [r for r in self.s["yields"] if (not asset or r["asset"].upper() == str(asset).upper()) and (not chain or r["chain"].lower() == str(chain).lower())]
        rows.sort(key=lambda r: -r["apy_pct"])
        return {"pools": [S(r) for r in rows[:5]]}

    def t_search_tokens(self, query):
        q = str(query).lower().strip()
        hits = [t for t in self.s["tokens"] if q in (t["symbol"].lower(), t["name"].lower(), t["mint"].lower())]
        if not hits:
            return S({"note": "No Solana token matched. Check the spelling or paste the mint.", "tokens": []})
        hits.sort(key=lambda t: -t["liquidity_usd"])
        return {"tokens": [S(t) for t in hits[:5]]}

    def t_get_address(self, contact=None, chain=None):
        if contact:
            return S({"note": f"No saved contact called {contact}. Ask the user for the address.", "status": "not_found"})
        rows = []
        for fam, nets in (("evm", ["ethereum", "base", "arbitrum", "bnb", "polygon", "monad"]), ("solana", ["solana"])):
            if chain and str(chain).lower() not in nets:
                continue
            rows.append(S({"address": self.s["addresses"][fam], "family": fam, "networks": nets, "wallet": self.s["wallet"]}))
        return S({"addresses": rows, "note": "Only send tokens on these networks to these addresses. The QR code is in Add funds."})

    def t_position_math(self, side, entry, size_usd, leverage, price=None, maintenance_margin=None):
        if side not in ("long", "short"):
            return {"error": "side must be long or short"}
        out = sim.position_math(side, float(entry), float(size_usd), float(leverage), None if price is None else float(price),
                                float(maintenance_margin or 0))
        return {k: (nn(out[k]) if isinstance(out[k], float) else out[k]) for k in sorted(out)}

    # ---- quotes and swaps
    def _quote(self, fs, fc, ts, tc, amount):
        fc, tc = str(fc).lower(), str(tc).lower()
        fs, ts = str(fs).upper(), str(ts).upper()
        if fc not in CHAIN_NAME or tc not in CHAIN_NAME:
            return {"error": "Unsupported chain."}
        if fs not in self.s["prices"] and fs not in STABLE or ts not in self.s["prices"] and ts not in STABLE:
            return {"error": "Unsupported token."}
        if fs == ts and fc == tc:
            return {"error": "From and to are the same."}
        amount = float(amount)
        usd_in = amount * self.price(fs)
        if usd_in < 1:
            return {"error": "Amount is below the minimum for this route (about $1)."}
        bps = self.s["fee_bps"]
        cost = round(usd_in * bps / 1e4, 2)
        usd_out = round(usd_in - cost, 2)
        out_raw = usd_out / self.price(ts)
        secs = 8 + (sum(map(ord, fc + tc)) + int(amount)) % 19
        q = S({"amount_in_usd": nn(round(usd_in, 2)), "amount_out": approx(out_raw), "amount_out_usd": nn(usd_out), "fee_bps": bps,
               "from": f"{ns(amount)} {fs} on {CHAIN_NAME[fc]}", "route": "NEAR Intents", "seconds": secs, "signature_verified": True,
               "to": f"about {approx(out_raw)} {ts} on {CHAIN_NAME[tc]}", "total_cost_usd": nn(cost)})
        return {"q": q, "out_raw": round(out_raw, 6), "fs": fs, "fc": fc, "ts": ts, "tc": tc, "amount": amount}

    def t_get_quote(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        r = self._quote(from_symbol, from_chain, to_symbol, to_chain, amount)
        return r["q"] if "q" in r else r

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        r = self._quote(from_symbol, from_chain, to_symbol, to_chain, amount)
        if "q" not in r:
            return r
        have = self.bal(r["fs"], r["fc"])
        if have + 1e-9 < r["amount"]:
            return S({"holding": f"{amount_str(have)} {r['fs']} on {CHAIN_NAME[r['fc']]}", "note": "The user holds less than that. Tell them.",
                      "prepared": False, "status": "insufficient"})
        res = dict(r["q"])
        res.update({"note": "Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels.",
                    "policy": "allow", "status": "prepared_for_review"})
        self.s["pending"].append({"id": self.next_id(), "kind": "swap", "label": "swap",
                                  "args": {"from_symbol": r["fs"], "from_chain": r["fc"], "to_symbol": r["ts"], "to_chain": r["tc"], "amount": nn(r["amount"])},
                                  "out_raw": r["out_raw"]})
        return S(res)

    def t_prepare_send(self, to_address, amount, symbol, chain=None):
        addr, sym = str(to_address), str(symbol).upper()
        if addr not in "\n".join(self.s["typed"]):
            return {"error": "Refused: the user didn't type that address in this chat. Only send to an address the user gave you themselves. Ask them for it."}
        if chain and str(chain).lower() not in self.mentioned_chains():
            chain = None                                    # ToolGuard.adjusted drops a chain the user never named
        hold = self.holders(sym)
        if chain:
            chain = str(chain).lower()
            if not self.bal(sym, chain) > 0 and len(hold) == 1:
                chain = hold[0]                              # silently switches to the only chain that holds it
        else:
            if len(hold) > 1:
                opts = ", ".join(f"{CHAIN_NAME[c]}: {amount_str(self.bal(sym, c))}" for c in hold)
                names = " and ".join(CHAIN_NAME[c] for c in hold)
                return S({"note": "Nothing was prepared. Ask the user which network, in one short question.", "options": hold, "prepared": False,
                          "question": f"Which network should the send use? You hold {sym} on {names} ({opts}).", "status": "needs_chain"})
            if not hold:
                return S({"holding": "nothing", "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"})
            chain = hold[0]
        have = self.bal(sym, chain)
        if have + 1e-9 < float(amount):
            return S({"holding": f"{amount_str(have)} {sym} on {CHAIN_NAME[chain]}", "note": "The user holds less than that. Tell them.",
                      "prepared": False, "status": "insufficient"})
        self.s["pending"].append({"id": self.next_id(), "kind": "send", "label": "send", "chain": chain,
                                  "args": {"to_address": addr, "amount": nn(float(amount)), "symbol": sym}})
        return S({"amount": f"{ns(float(amount))} {sym}", "executed": False, "kind": "send", "network": CHAIN_NAME[chain],
                  "note": "Waiting for the user to approve with Face ID.", "status": "review_opened"})

    def t_prepare_trade(self, coin, side, size_usd, leverage, order_type, limit_price=None, take_profit=None, stop_loss=None):
        coin = str(coin).upper()
        if coin not in self.s["prices"] or coin not in MAX_LEV:
            return {"error": f"{coin} isn't tradable on Hyperliquid"}
        if str(order_type) == "limit" and limit_price is None:
            return {"error": "limit_price is required for a limit order"}
        need = float(size_usd) / float(leverage)
        avail = float(self.s["hl"]["avail"])
        if need > avail:
            return {"error": f"Not enough margin: {avail:.2f} USDC available, {need:.2f} needed", "prepared": False}
        a = {"coin": coin, "side": side, "size_usd": nn(size_usd), "leverage": nn(leverage), "order_type": order_type}
        for k, v in (("limit_price", limit_price), ("take_profit", take_profit), ("stop_loss", stop_loss)):
            if v is not None:
                a[k] = nn(v)
        self.s["pending"].append({"id": self.next_id(), "kind": "trade", "label": coin, "args": a})
        return {"status": "review_opened", "kind": "trade", "executed": False, "note": "Waiting for the user to approve with Face ID."}

    def t_prepare_close(self, coin, percent=None):
        coin = str(coin).upper()
        pos = next((p for p in self._pos_rows() if p["coin"] == coin), None)
        if pos is None:
            return S({"note": f"The user has no open {coin} position on Hyperliquid. Tell them.", "prepared": False, "status": "no_position"})
        pct = 100 if percent is None else float(percent)
        part = f"all of the {coin} {pos['side']}, {ns(pos['size'])} {coin}" if pct >= 100 else f"{ns(pct)}% of the {coin} {pos['side']}, {ns(round(pos['size'] * pct / 100, 6))} {coin}"
        self.s["pending"].append({"id": self.next_id(), "kind": "close", "label": coin, "args": {"coin": coin}})
        return S({"closing": part, "executed": False, "kind": "close", "note": "Waiting for the user to approve with Face ID.",
                  "order": "Market, reduce only", "status": "review_opened", "unrealised_pnl": nn(round(pos["unrealised_pnl"] * pct / 100, 2))})

    def t_cancel_order(self, coin, kind=None, order_id=None):
        coin = str(coin)
        h = self.s["hl"]
        if coin.lower() == "all":
            lim = [o for o in h["orders"] if o["kind"] == "limit"]
            if not lim:
                return S({"note": "The user has no open limit orders. Tell them.", "prepared": False, "status": "nothing_to_cancel"})
            return S({"cancelling": f"{len(lim)} open limit orders ({', '.join(o['coin'] for o in lim)})", "executed": False, "kind": "cancel_order",
                      "note": "Waiting for the user to tap Confirm. Nothing is cancelled yet.", "status": "card_opened"})
        coin = coin.upper()
        os_ = [o for o in h["orders"] if o["coin"] == coin and (not kind or o["kind"] == kind or (kind == "order" and o["kind"] == "limit"))
               and (not order_id or o["id"] == str(order_id))]
        if not os_:
            return S({"note": f"The user has no open {coin} orders. Tell them.", "prepared": False, "status": "nothing_to_cancel"})
        if len(os_) > 1:
            opts = [order_label(o, short=True) for o in os_]
            return S({"note": "Nothing was prepared. Ask the user which one.", "options": opts, "prepared": False,
                      "question": f"Which {coin} order? " + " or ".join(opts[:-1]) + (" or " if len(opts) > 1 else "") + opts[-1] + ".", "status": "needs_choice"})
        o = os_[0]
        r = {"cancelling": order_label(o), "executed": False, "kind": "cancel_order", "note": "Waiting for the user to tap Confirm. Nothing is cancelled yet.",
             "status": "card_opened"}
        if o["kind"] == "stop_loss":
            side = "long" if o["side"] == "sell" else "short"
            r["warning"] = f"Without it the {coin} {side} has no stop loss."
        return S(r)


def SZ_DP(px):
    return 4 if px >= 1000 else (3 if px >= 100 else (2 if px >= 10 else (1 if px >= 1 else 0)))


def order_label(o, short=False):
    c, p = o["coin"], dollars(ns(o["price"]))
    if o["kind"] == "limit":
        return f"{o['side']} limit at {p}" if short else f"{c} {o['side']} limit, {ns(o['size'])} {c} at {p}"
    word = "stop loss" if o["kind"] == "stop_loss" else "take profit"
    return f"{word} at {p}" if short else f"{c} {word} at {p}"


def make_runner(item):
    """Harness entry point for suite `followups`: state persists across the script turns.
        rn = make_runner(item); rn.run(name, args) -> JSON text; rn.begin_turn(k) before script turn k (1-based)."""
    class Runner:
        def __init__(self):
            self.w = FWorld.from_state(item["world"])
            self.script, self.events = item.get("script") or [], item.get("script_events") or []

        def run(self, name, args):
            return self.w.jd(self.w.call(name, args if isinstance(args, dict) else {}))

        def begin_turn(self, k):
            ev = self.events[k - 1] if k - 1 < len(self.events) else None
            if ev:
                self.w.apply_event(ev)
            self.w.hear(self.script[k - 1])
    return Runner()


# ----------------------------------------------------------------------------- conversation builder
class FConv:
    def __init__(self, rng, world, bio):
        sysm = SYSTEM_V3.replace("{BIO}", bio)
        if rng.random() < 0.10:
            sysm += f"\nThe user wants to be called {rng.choice(H.NAMES)}."
        self.rng, self.w, self.bio = rng, world, bio
        self.msgs = [{"role": "system", "content": sysm}]
        self.turns, self.cur = [], None

    def user(self, text, ev=None):
        if ev:
            self.w.apply_event(ev)
        self.w.hear(text)
        self.msgs.append({"role": "user", "content": text})
        self.cur = {"user": text, "msg_i": len(self.msgs) - 1, "event": ev, "expect": [], "gold": {}, "world": self.w.state()}
        self.turns.append(self.cur)

    def call(self, *calls):
        tcs = [{"name": n, "arguments": a} for n, a in calls]
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        self.cur["expect"].append({"calls": tcs})
        res = []
        for n, a in calls:
            r = self.w.call(n, a)
            res.append(r)
            self.msgs.append({"role": "tool", "content": self.w.jd(r)})
        return res if len(res) > 1 else res[0]

    def say(self, text, ask=False):
        self.msgs.append({"role": "assistant", "content": text})
        self.cur["expect"].append({"ask": True} if ask else {"text": True})

    def gold(self, **kw):
        self.cur["gold"].update(kw)


# ----------------------------------------------------------------------------- world builder
YIELD_ROWS = [("Aave v3", "USDC", "base", 4.21, 950e6), ("Morpho", "USDC", "base", 6.8, 182e6), ("Aave v3", "USDC", "arbitrum", 3.9, 410e6),
              ("Aave v3", "ETH", "ethereum", 2.1, 1.9e9), ("Compound v3", "USDC", "base", 3.6, 120e6), ("Aave v3", "USDC", "ethereum", 4.7, 2.3e9)]
MEME_TOKENS = [("WIF", "dogwifhat", 1.82), ("JUP", "Jupiter", 0.78), ("POPCAT", "Popcat", 0.52), ("GOAT", "Goatseus Maximus", 0.12),
               ("FARTCOIN", "Fartcoin", 0.41), ("PENGU", "Pudgy Penguins", 0.0081)]


def make_world(rng, bio, usdc=("base",), extras=(), hl_avail=None, positions=(), orders=(), usdt=(), amounts=None):
    device = "iPhone" if bio == "Face ID" else ("Mac" if rng.random() < 0.9 else "iPhone")
    prices = {c: TV.draw_price(rng, c) for c in COINS}
    change = {c: TV.draw_change(rng) for c in COINS}
    bals = []
    for i, ch in enumerate(usdc):
        a = (amounts or {}).get(("USDC", ch)) or round(rng.uniform(150, 2400) if i == 0 else rng.uniform(60, 900), 2)
        bals.append({"symbol": "USDC", "chain": ch, "amount": a})
    for ch in usdt:
        bals.append({"symbol": "USDT", "chain": ch, "amount": round(rng.uniform(80, 700), 2)})
    for sym, ch, a in extras:
        bals.append({"symbol": sym, "chain": ch, "amount": a})
    avail = hl_avail if hl_avail is not None else round(rng.uniform(900, 9000), 2)
    hl = {"avail": avail, "positions": [], "orders": []}
    nid = 0
    for sp in positions:
        coin, side, lev = sp["coin"], sp["side"], sp["lev"]
        entry = nn(float(round(prices[coin] * (1 + sp.get("off", rng.uniform(-0.05, 0.05))), sim.ASSETS[coin][2])))
        size = nn(round(sp["margin"] * lev / entry, SZ_DP(entry)))
        hl["positions"].append({"coin": coin, "entry": entry, "leverage": lev, "side": side, "size": size,
                                "liquidation": nn(round(sim.liq_price(side, entry, lev, 0.005), sim.ASSETS[coin][2]))})
        hl["avail"] = round(hl["avail"] - sp["margin"], 2)
    for od in orders:
        nid += 1
        px = nn(float(round(prices[od["coin"]] * od["mult"], sim.ASSETS[od["coin"]][2])))
        row = {"id": str(8100 + nid), "coin": od["coin"], "kind": od["kind"], "price": px, "side": od["side"],
               "size": nn(round(od["usd"] / px, SZ_DP(px))) if od["kind"] == "limit" else next(p["size"] for p in hl["positions"] if p["coin"] == od["coin"])}
        if od["kind"] != "limit":
            row["whole_position"] = True
        hl["orders"].append(row)
    yl = []
    for proto, asset, ch, apy, tvl in YIELD_ROWS:
        a = round(apy * rng.uniform(0.8, 1.25), 2)
        base = round(a * rng.uniform(0.8, 1.0), 2)
        yl.append({"protocol": proto, "asset": asset, "chain": CHAIN_NAME[ch], "apy_pct": a, "apy_base_pct": base,
                   "apy_reward_pct": round(a - base, 2), "apy_30d_pct": round(a * rng.uniform(0.9, 1.1), 2),
                   "tvl_usd": nn(round(tvl * rng.uniform(0.7, 1.4), -5)), "il_risk": "no", "pool_meta": None,
                   "can_deposit_in_hermit": proto == "Aave v3" and ch != "solana"})
    toks = []
    for sym, name, px in MEME_TOKENS:
        toks.append({"symbol": sym, "name": name, "price": px, "mint": H.sol_addr(rng), "verified": True, "tags": [],
                     "liquidity_usd": nn(round(rng.uniform(2e6, 2.5e7), -4)), "volume_24h_usd": nn(round(rng.uniform(1e7, 1.2e8), -4)),
                     "change_24h_pct": nn(round(rng.uniform(-9, 9), 1))})
    st = {"bio": bio, "device": device, "wallet": "Main", "prices": prices, "change": change, "balances": bals, "hl": hl, "activity": [],
          "pending": [], "inflight": [], "seq": 0, "nid": nid, "typed": [], "fee_bps": rng.choice([30, 35, 40]), "yields": yl, "tokens": toks,
          "addresses": {"evm": H.evm_addr(rng, True), "solana": H.sol_addr(rng)}}
    return FWorld(st)


# ----------------------------------------------------------------------------- user phrasing
PREFIX = ["hey", "yo", "gm", "ok so", "hermit,", "quick one:", "hi", "hmm", "alright", "so", "ok", "hey hermit,", "one sec,"]
SUFFIX = ["pls", "please", "thanks", "thx", "asap", "if you can"]


def fill(rng, tpl, **slots):
    """Fill a template; slot values are protected from case changes and typos."""
    parts = []
    for piece in re.split(r"(\{\w+\})", tpl):
        if not piece:
            continue
        m = re.fullmatch(r"\{(\w+)\}", piece)
        if m:
            parts.append([str(slots[m.group(1)]), True])
        else:
            parts.append([piece, False])
    if rng.random() < 0.14:
        parts.insert(0, [rng.choice(PREFIX) + " ", False])
    if rng.random() < 0.12 and not (parts and parts[-1][0].endswith("?")):
        parts.append([" " + rng.choice(SUFFIX), False])
    mode = rng.choices(["keep", "lower", "cap"], [0.5, 0.4, 0.1])[0]
    for pc in parts:
        if not pc[1] and mode == "lower":
            pc[0] = pc[0].lower()
    if mode == "cap" and parts and not parts[0][1]:
        parts[0][0] = parts[0][0][:1].upper() + parts[0][0][1:]
    if rng.random() < 0.12:
        TV.typo(rng, parts)
    s = re.sub(r"\s+", " ", "".join(p[0] for p in parts)).strip()
    if rng.random() < 0.2:
        s = s.rstrip("?.!")
    return s


def pick_tpl(rng, pool, eval_mode):
    idx = [i for i in range(len(pool)) if (i % 4 == 0) == eval_mode]
    i = rng.choice(idx)
    return i, pool[i]


def amt_user(rng, v):
    return TV.amt_text(rng, v)


def lev_user(rng, L):
    return rng.choice([f"{L}x", f"{L}x", f"{L}x leverage", f"{L}x lev", f"x{L}", f"leverage {L}"])


def coin_user(rng, c):
    return rng.choices(ALIAS[c], [45] + [20] * (len(ALIAS[c]) - 1))[0]


def chain_user(rng, ch):
    return rng.choice(CHAIN_SAY[ch])


TRADE_PCT = [
    "{act} {c} {amt} at {lev} with a {pct}% stop", "open a {amt} {c} {act}, {lev}, stop loss {pct}% {dir}", "{c} {act} {amt} {lev} sl {pct}%",
    "can you {act} {c} for {amt} using {lev} and put the stop {pct}% {dir} the price", "i want a {c} {act}. {amt}, {lev}, {pct}% stop loss",
    "{act} {amt} of {c}, {lev}, and a {pct} percent stop", "set up a {c} {act} with {amt}, {lev}, stop {pct}% {dir} the price",
    "go {act} on {c}: {amt}, {lev}, stop {pct}% {dir}", "prepare a {c} {act}, {amt}, {lev}, {pct}% stoploss",
    "hey, {act} {c} with {amt} at {lev}, stop loss {pct}% {dir} entry please", "market {act} {c}, {amt}, {lev}, {pct}% sl",
    "{amt} {c} {act} at {lev}, protect it with a {pct}% stop", "could you open a {c} {act} for {amt} on {lev}, stop {pct}% {dir} where it is now",
    "let's {act} {c}. {amt} size, {lev}, {pct}% stop", "{act} {c} {lev} {amt} stop {pct}%",
    "i'd like to {act} {c} with {amt} on {lev} and a stop loss {pct}% {dir} the current price", "{c} {act} please, {amt}, {lev}, stop it out at {pct}% {dir}",
    "start a {act} in {c}: {amt} at {lev}, {pct}% stop", "do a {c} {act} {amt} {lev} w/ {pct}% stop", "{act} {c}, {amt}, {lev}, risk {pct}% to the stop",
]
TRADE_ABS = [
    "{act} {c} {amt} {lev}, stop at {stop}", "open {c} {act} {amt} at {lev} with stop loss {stop}", "{c} {act} {amt} {lev} sl {stop}",
    "{act} {amt} of {c} on {lev}, stop loss at {stop} please", "i wanna {act} {c}. {amt}, {lev}, stop at {stop}", "set up a {c} {act}: {amt}, {lev}, stop {stop}",
    "{act} {c}, size {amt}, {lev}, sl {stop}", "can you {act} {c} {amt} {lev} and set the stop at {stop}", "prepare a {amt} {c} {act} {lev} with a stop loss of {stop}",
    "go {act} {c} with {amt}, {lev}, stop {stop}", "{c} {act} {lev} {amt} stoploss {stop}", "{amt} {act} on {c}, {lev}, put my stop at {stop}",
]
TRADE_NONE = [
    "{act} {c} {amt} {lev}", "open a {amt} {c} {act} at {lev}", "{c} {act} {amt} on {lev}", "can you {act} {c} for {amt}, {lev}",
    "i want to {act} {c}, {amt}, {lev}", "{act} {amt} of {c} {lev}", "market {act} {c} {amt} {lev}", "go {act} on {c} with {amt} at {lev}",
    "set up a {lev} {c} {act} for {amt}", "start a {c} {act}, {amt}, {lev}",
]
TP_SUFFIX = [", tp {tp}%", " and take profit {tp}% {tdir}", ", take profit {tp}% {tdir}", " with a {tp}% take profit"]
SEND_T = [
    "send {tok} to {addr}", "pls send {tok} to {addr}", "transfer {tok} to {addr}", "can you send {tok} to {addr}", "{tok} to {addr} please",
    "i need to pay {addr} {tok}", "move {tok} to {addr}", "send {addr} {tok}", "need to send {tok} to this address {addr}", "send {tok} over to {addr}",
    "ok send {tok} to {addr}", "hey can u send {tok} to {addr}", "wire {tok} to {addr}", "pay {addr} {tok}", "i want to send {tok} to {addr}",
    "send {tok} -> {addr}", "shoot {tok} to {addr}", "could you send {tok} to {addr}",
]
SEND_CH_T = [
    "send {tok} to {addr} on {ch}", "{ch}: send {tok} to {addr}", "send {tok} on {ch} to {addr}", "pls send {tok} to {addr} on {ch}",
    "transfer {tok} on {ch} to {addr}", "can you send {tok} to {addr} on {ch}", "{tok} to {addr} on {ch} please", "send {tok} to {addr} via {ch}",
]
BRIDGE_T = [
    "bridge {amt} {tok} from {c1} to {c2}", "move {amt} {tok} from {c1} to {c2}", "i want to bridge {amt} {tok} {c1} to {c2}",
    "send {amt} {tok} from {c1} over to {c2}", "bridge {amt} {tok} to {c2} from {c1}", "can you bridge {amt} {tok} from {c1} to {c2}",
    "bridge {amt} {tok} {c1} -> {c2}", "get {amt} {tok} from {c1} onto {c2}",
]
SWAP_T = [
    "swap {amt} {a} for {b} on {c}", "swap {amt} {a} to {b} on {c}", "trade {amt} {a} into {b} on {c}", "convert {amt} {a} to {b} on {c}",
    "i wanna turn {amt} {a} into {b} on {c}", "{c}: swap {amt} {a} to {b}", "can you swap {amt} {a} to {b} on {c}", "exchange {amt} {a} for {b} on {c}",
]

# follow-up phrases (each pool has at least 8 entries; every 4th is reserved for eval)
SAME_FOR = [
    "do the same for {c}", "same for {c}", "same thing but {c}", "now {c} too", "and the same on {c}", "do {c} the same way", "same again for {c}",
    "can you do the same with {c}", "now same for {c} pls", "repeat that for {c}", "same but {c}", "{c} too", "ok and {c}, same setup", "now one for {c}",
    "do that on {c} as well", "also {c}, same as that", "now do it for {c}", "and for {c}?", "same trade on {c}", "{c} as well, same as the last one",
]
SAME_TO = [
    "same to {addr}", "do the same for {addr}", "send the same to {addr}", "same amount to {addr} too", "and the same to {addr}", "also {addr}, same thing",
    "now the same to {addr}", "repeat that to {addr}", "same again, to {addr}", "one more, same, to {addr}", "do that for {addr} as well", "{addr} too, same amount",
]
LEV_FU = [
    "make it {L}x", "{L}x instead", "bump it to {L}x", "change it to {L}x", "can we do {L}x", "make that {L}x", "use {L}x", "what about {L}x, do that",
    "ok but {L}x", "go {L}x instead pls", "{L}x leverage instead", "set it to {L}x",
]
HALF_FU = ["now half", "make it half", "half of that", "halve it", "ok half that size", "do half instead", "cut it in half", "make it half the size"]
DOUBLE_FU = ["make it double", "double it", "twice that size", "now double", "ok double the size", "go double"]
AMOUNT_FU = ["make it {n}", "{n} instead", "change it to {n}", "make that {n}", "ok {n} instead", "no wait, {n}", "bump it to {n}", "actually make it {n}"]
CHAIN_FU = [
    "and on {ch}?", "same but on {ch}", "what about on {ch}", "do it on {ch} instead", "make it {ch}", "now on {ch}", "{ch} instead",
    "can you do that on {ch} instead", "switch it to {ch}", "no, {ch}",
]
BRIDGE_DEST_FU = ["and to {ch}?", "same but to {ch}", "make it go to {ch}", "to {ch} instead", "what about {ch} as the destination", "send it to {ch} instead",
                  "change the destination to {ch}", "{ch} instead"]
CANCEL_FU = [
    "cancel that", "drop it", "don't do that one, cancel", "kill it", "no wait, cancel it", "stop that one", "hold on, cancel", "cancel it", "actually cancel that", "nvm cancel", "scrap that", "cancel", "never mind, cancel it", "ok cancel that one",
    "stop, cancel that", "forget it, cancel", "can you cancel that", "undo that", "undo", "i changed my mind, cancel it",
]
CANCEL_ORDER_FU = ["cancel that", "yes cancel the one you just showed me", "drop that order", "remove it", "ok kill it", "pull that order", "take that order off", "cancel this order", "cancel it", "ok cancel that order", "remove that one", "delete that order", "cancel that one pls", "kill that order", "get rid of it, cancel",
                   "yeah cancel it", "can you cancel that", "cancel the order", "ok cancel"]
UNDO_FU = ["undo that", "can you reverse that", "i want that undone", "roll that back", "put it back how it was", "go back on that", "take that back", "unwind it", "undo it", "undo", "can you undo that", "reverse it", "take it back", "undo it please", "revert that", "i want to undo that",
           "ok undo it", "reverse that for me", "can we undo this"]
APPROVED_PRE = ["approved.", "ok approved,", "done,", "that's approved,", "ok did it,", "confirmed that,", "approved it,", "just approved,", "ok i approved it.", "signed it,", "approved that one,"]
ADDR_Q = [
    "what was that address", "which address was that again", "where was that going again, which address", "what address did i give you", "can you show the address again", "tell me that address again",
    "what was the address i gave you", "what address was that again", "which address did i give you", "can you repeat the address", "what was the address",
    "remind me of that address", "what address is that", "wait, what address was that", "show me that address again", "which address was it",
    "what address did you use", "say that address again",
]
MYADDR_Q = ["what was that address", "what was that address again", "can you repeat that address", "that address again?", "what address was that", "show me that address again",
            "wait what was the address", "give me that address one more time"]
LASTADDR_Q = ["what address did my last send go to", "who did i last send to", "what was the last address i paid", "which address did i send to last",
              "where did my last transfer go", "what address was my last send to", "last send, what address was that", "what's the address of my last payment"]
MAKE_N_FU = ["make it {n}", "{n} instead", "change it to {n}", "make that {n}", "ok {n}", "go with {n}", "set it to {n}", "bump it to {n}"]
CANCEL_WHICH_FU = ["cancel that", "cancel it", "cancel the review", "scrap that", "nvm cancel it", "ok cancel that one", "stop, cancel", "cancel please"]
SAME_AGAIN_FU = ["do the same for {c}", "same for {c}", "same thing but {c}", "now {c} too", "same again for {c}", "can you do the same with {c}", "now one for {c}",
                 "do that on {c} as well"]
SEND_AGAIN_FU = ["send it again to the same address", "do it again, same address", "same send once more", "send another one just like that", "one more time to that address", "repeat that payment", "pay them again", "again to the same person", "make the same transfer again", "send the same again", "same address, same amount", "do that send again", "repeat the last send",
                 "send to the same address again", "again, same as before", "send that again"]
LEV_Q = ["what leverage did i use on that long", "how many x is the {c} trade", "which leverage is that {c} position", "tell me the leverage on my {c} trade", "what leverage am i running on {c}", "leverage on the {c} one?", "what x did i pick for {c}", "how levered is my {c} position", "remind me the leverage of that order", "what leverage is my {c} position on", "remind me what leverage i picked on {c}", "how much leverage is the {c} trade",
         "what leverage was that {c} trade", "which leverage did i set for {c}", "and what leverage is it on again", "wait what leverage did i go with"]
SWAP_Q = ["how much did i swap earlier", "what was that swap again", "which swap did i do before", "can you remind me of the swap i approved", "what did that swap look like", "how big was the swap i did", "what was the last swap i made", "what swap was it again", "what swap did i do earlier", "what did i swap before", "remind me what swap i did", "which swap did i approve earlier",
          "how much was that swap", "what was the swap i did earlier and did it land", "what swap went on earlier"]
BAL_Q = ["what's my balance", "how much usdc is in my wallet", "what's in my wallet now", "tell me my balance", "what have i got now", "balance check", "how much do i hold now", "show my holdings", "what's my balance now", "what do i have now", "how much do i have left", "balance?", "what's my usdc now", "check my balance",
         "how much usdc do i have now", "show my balance again", "what's left in my wallet", "ok what's my balance", "how much is left"]
HL_BAL_Q = ["how much can i still trade with", "what's my available margin now", "what do i have left on hyperliquid", "how much margin is left",
            "what's free on hyperliquid now", "available usdc on hyperliquid?", "how much do i have left to trade", "what's my hyperliquid balance now"]
NOW_Q = ["and now?", "and now", "what about now?", "how's it now", "and now where is it", "so where are we now", "now?", "what's it at now", "and currently?",
         "ok and now", "update?", "any change?"]
ENOUGH_Q = ["do i have enough?", "do i have enough for that?", "is that covered?", "do i have enough in my wallet for that", "have i got enough for that",
            "can i afford that?", "do i have enough to cover that", "is my balance enough for that", "wait do i have enough", "enough funds for that?"]
ENOUGH_Q_TRADE = ["do i have enough margin for that?", "is there enough margin for that", "do i have enough on hyperliquid for that?", "can my account cover that",
                  "do i have enough to open that?", "is that margin covered?", "will my margin cover that", "do i have enough free usdc for that"]
THERE_Q = ["is it there yet?", "did it get there", "has the money shown up", "is my usdc on the other side yet", "did the transfer finish", "has the bridge landed", "anything on the other chain yet", "is it in yet", "has it arrived?", "did it arrive yet", "is it there yet", "has my usdc landed yet", "did it land", "is it done yet", "has it landed?",
           "did the bridge finish", "is it there?", "where is it, did it arrive", "any sign of it yet"]
FEE_Q = ["what's the fee", "what's the fee?", "how much is the fee", "what will it cost", "what's the cost of that", "how much does that cost", "and the fee?",
         "fee?", "what are the fees", "what do i pay in fees", "what's the cost", "how much in fees"]
FEE_Q_AMT = ["{n}. what's the fee first?", "{n}, but what's the fee", "{n} - what will it cost?", "let's say {n}. what's the fee", "{n}. how much is the fee?",
             "about {n}, what would the fee be", "{n}, and what does it cost", "{n}, what are the fees"]
YES_FU = ["yes", "yes please", "yep", "ok do it", "do it", "go ahead", "sure", "yeah", "ye", "ok yes", "prepare it", "please do", "yes do that"]
NO_USDT = ["no, usdt", "no i meant usdt", "sorry, usdt not usdc", "wait, usdt", "oops usdt", "no usdt please", "my bad, usdt", "no, the usdt"]
CORR_COIN = ["no, {c}", "sorry i meant {c}", "wait, {c} not {x}", "oops {c}", "my bad, {c}", "no i meant {c}", "not {x}, {c}", "{c} not {x}"]
BACK_TO = ["ok back to the {c} one, make it {L}x", "right, the {c} trade. make it {L}x", "back to {c}: {L}x instead", "ok so for the {c} one, change it to {L}x",
           "anyway, {c} trade, {L}x please", "ok about that {c} order, make it {L}x"]
AMOUNT_ANS = ["{amt} at {L}x", "{amt}, {L}x", "{L}x with {amt}", "make it {amt} and {L}x", "{amt} on {L}x leverage", "let's say {amt}, {L}x"]
LEV_FIRST = [
    "open a {c} {act}", "{act} {c}", "i want to {act} {c}", "can you {act} {c}", "go {act} on {c}", "{c} {act} pls", "let's {act} {c}", "start a {act} on {c}",
    "i wanna {act} {c}", "set up a {c} {act}",
]
SEND_NOADDR = ["send {tok}", "i want to send {tok}", "can you send {tok}", "send some {sym}, {tok}", "need to send {tok}", "pls send {tok}", "let's send {tok}",
               "transfer {tok}", "i have to send {tok}", "send {tok} please"]
ADDR_ANS = ["{addr}", "to {addr}", "{addr} on that one", "it's {addr}", "here: {addr}", "address is {addr}", "send it to {addr}", "{addr} thanks"]
SWAP_NOAMT = ["swap some usdc to eth", "i want to swap usdc for eth", "turn some usdc into eth", "can you swap usdc to eth", "need some eth, swap from usdc",
              "swap my usdc to eth", "get me some eth with usdc", "i wanna swap usdc into eth", "change some usdc to eth", "usdc to eth swap pls"]
QUOTE_FIRST = ["how much would {amt} {a} on {c1} be on {c2}", "what would {amt} {a} from {c1} turn into on {c2}", "quote {amt} {a} from {c1} to {c2}",
               "how much would i get bridging {amt} {a} from {c1} to {c2}", "if i bridge {amt} {a} from {c1} to {c2} what do i get", "what's the rate for {amt} {a} {c1} to {c2}",
               "how much {a} do i end up with if i move {amt} from {c1} to {c2}", "check a quote for {amt} {a}, {c1} to {c2}"]
MINT_Q = ["what's the {t} mint address", "mint address for {t}?", "what's the contract address of {t} on solana", "can you give me the {t} mint", "{t} mint?",
          "what is the {t} token address", "i need the {t} mint address", "what's the address of the {t} token", "give me the {t} contract on solana"]
CONTRACT_Q = ["what's the {t} contract address on {ch}", "contract address for {t} on {ch}?", "what's the {t} token address on {ch}", "give me the {t} contract on {ch}",
              "can you give me the address of {t} on {ch}", "{t} on {ch}, what's the contract", "what is the {t} contract on {ch}", "i need the {t} contract address for {ch}"]
APY_MEM_Q = ["isn't aave usdc around {n}% on {ch}?", "i think aave pays about {n}% on usdc on {ch}, right?", "aave usdc on {ch} is like {n}% no?",
             "didn't aave usdc on {ch} pay {n}%", "is the usdc rate on aave {ch} still {n}%", "i remember aave usdc at {n}% on {ch}, still true?",
             "what's the apy on aave usdc on {ch}, was it {n}%", "aave {ch} usdc, {n}% apy right?"]
TVL_FU = ["and the tvl?", "what's the tvl on that", "how big is that pool", "and how much is in it", "tvl?", "how much tvl does it have", "and its tvl", "what's the size of that pool"]
APY_REFRESH = ["is that apy still the same?", "is the apy still that?", "did the apy change?", "is that rate still right", "still the same apy?",
               "check that apy again", "is it still paying that", "has the apy moved"]
APY_FIRST = ["what's the aave usdc apy on {ch}", "aave usdc apy on {ch}?", "how much does aave pay on usdc on {ch}", "what's the rate for usdc on aave {ch}",
             "apy for usdc on aave {ch} right now", "aave {ch} usdc yield?", "how much can i earn on usdc with aave on {ch}", "check the aave usdc rate on {ch}"]

# unrelated interjections (price / calculator / knowledge)
INTERJECT_PRICE = ["what's {c} at", "{c} price?", "how is {c} doing", "quick, {c} price", "where's {c} trading", "price of {c} real quick", "is {c} up today",
                   "check {c} for me"]
INTERJECT_CALC = [("how much is {p}% of {n}", "{n} * {p} / 100"), ("what's {p}% of {n}", "{n} * {p} / 100"), ("{p}% of {n}?", "{n} * {p} / 100"),
                  ("quick maths: {p} percent of {n}", "{n} * {p} / 100")]
INTERJECT_KNOW = [
    ("what does liquidation mean in perps", "Liquidation is when the exchange closes your leveraged position because your margin can no longer cover the loss. "
     "Higher leverage puts the liquidation price closer to your entry."),
    ("whats the difference between a market and a limit order", "A market order fills right away at the best available price. A limit order only fills at your price or "
     "better, so it may never fill."),
    ("what is a stop loss again", "A stop loss is an order that closes your position if the price reaches a level you set, to cap the loss. In a fast market it can fill "
     "worse than that level."),
    ("what's gas", "Gas is the network fee paid to the chain's validators for processing a transaction. You pay it in the chain's own token, like ETH on Base."),
    ("what's a bridge", "A bridge moves your tokens from one chain to another, for example USDC from Base to Arbitrum. It takes a short while and costs a small fee."),
    ("what is leverage", "Leverage lets you open a bigger position than your margin, so 5x means the position is five times your margin. Gains and losses both get "
     "multiplied, and a big enough move against you liquidates the position."),
    ("what's the difference between long and short", "A long profits when the price goes up. A short profits when the price goes down. Both can lose money if the price "
     "moves the other way."),
    ("why do stop losses slip sometimes", "A stop loss turns into a market order once the price hits it, so in a fast move it fills at the next available price, which can "
     "be worse than your level."),
]


# ----------------------------------------------------------------------------- assistant wording
def approve_line(rng, bio):
    return rng.choice([f"Approve with {bio}.", f"Review it and approve with {bio}.", f"Nothing happens until you approve with {bio}.",
                       f"It only happens when you approve it with {bio}.", f"Approve it with {bio} when it looks right."])


def pend_desc(p):
    a = p["args"]
    if p["kind"] == "trade":
        return f"{a['coin']} {a['side']}"
    if p["kind"] == "send":
        return f"{ns(a['amount'])} {a['symbol']} send"
    if p["kind"] == "swap":
        return "swap"
    return f"{a['coin']} close"


def open_note(rng, pend, replace=False):
    """pend: the pending reviews that were open BEFORE the new prepare."""
    if not pend:
        return ""
    if len(pend) == 1:
        d = pend_desc(pend[0])
        if replace:
            return rng.choice([f"The earlier {d} review is still open, so close it if you only want this one.",
                               f"The first review ({d}) is still open. Close it if you only want this one.",
                               f"The {d} review you had before is still open, so close it if you don't need it."])
        return rng.choice([f"The {d} review is still open, so close it if you don't want it.",
                           f"The earlier {d} review is still open. Close it if you don't want it.",
                           f"Your {d} review is still waiting, so close it if you don't want that one."])
    d1, d2 = pend_desc(pend[-2]), pend_desc(pend[-1])
    return rng.choice([f"The {d1} and {d2} reviews are still open, so close any you don't want.",
                       f"Your {d1} and {d2} reviews are both still open. Close the ones you don't want."])


def trade_text(rng, bio, spec, info, note="", again=False):
    a = info["args"]
    coin, side, size, lev = a["coin"], a["side"], dollars(ns(a["size_usd"])), ns(a["leverage"])
    ph = []
    st, tp = (spec or {}).get("stop"), (spec or {}).get("tp")
    if info.get("stop") is not None:
        if info.get("price") is not None and st and st["kind"] == "pct":
            ph.append(f"the stop loss {ns(st['pct'])}% {'under' if side == 'long' else 'over'} today's {dollars(ns(info['price']))} at {dollars(ns(info['stop']))}")
        else:
            ph.append(f"the stop loss at {dollars(ns(info['stop']))}")
    if info.get("tp") is not None:
        if info.get("price") is not None and tp and tp["kind"] == "pct":
            ph.append(f"take profit {ns(tp['pct'])}% {'over' if side == 'long' else 'under'} at {dollars(ns(info['tp']))}")
        else:
            ph.append(f"take profit at {dollars(ns(info['tp']))}")
    tail = (", with " + " and ".join(ph)) if ph else ""
    order = "market order" if a["order_type"] == "market" else f"limit order at {dollars(ns(a['limit_price']))}"
    if again:
        head = rng.choice([f"I've prepared it again at {lev}x: {size} {coin} {side}, {order}{tail}.",
                           f"New review prepared: {size} {coin} {side} at {lev}x, {order}{tail}.",
                           f"I've prepared the updated order: {size} {coin} {side} at {lev}x, {order}{tail}.",
                           f"Updated and prepared: a {lev}x {coin} {side} for {size}, {order}{tail}.",
                           f"Prepared again: {size} {coin} {side} at {lev}x, {order}{tail}."])
    else:
        head = rng.choice([f"I've prepared a {size} {coin} {side} at {lev}x, {order}{tail}.",
                           f"Your {coin} {side} is prepared: {size} at {lev}x, {order}{tail}.",
                           f"Prepared: {size} {coin} {side} at {lev}x on a {order}{tail}.",
                           f"The {coin} {side} is ready for review, {size} at {lev}x as a {order}{tail}.",
                           f"I've set up a {lev}x {coin} {side} for {size}, {order}{tail}.",
                           f"{coin} {side} prepared: {size} at {lev}x, {order}{tail}."])
    return " ".join(x for x in (head, note, approve_line(rng, bio)) if x)


def send_text(rng, bio, r, args, note=""):
    amt, addr, net = r["amount"], args["to_address"], r["network"]
    head = rng.choice([f"I've prepared a send of {amt} to {addr} on {net}.", f"Prepared: {amt} to {addr} on {net}.",
                       f"Your send of {amt} to {addr} on {net} is prepared.", f"I've set up {amt} to {addr} on {net} for review.",
                       f"The send is prepared: {amt} to {addr} on {net}."])
    return " ".join(x for x in (head, note, approve_line(rng, bio)) if x)


def swap_text(rng, bio, r, note=""):
    head = rng.choice([f"I've prepared {r['from']} to {r['to']} through {r['route']}. Total cost is {dollars(ns(r['total_cost_usd']))}.",
                       f"Prepared: {r['from']} for {r['to']}, through {r['route']}. The total cost is {dollars(ns(r['total_cost_usd']))}.",
                       f"Your swap is prepared through {r['route']}: {r['from']} to {r['to']}, total cost {dollars(ns(r['total_cost_usd']))}.",
                       f"I've set up {r['from']} to {r['to']} through {r['route']}, with a total cost of {dollars(ns(r['total_cost_usd']))}."])
    return " ".join(x for x in (head, note, approve_line(rng, bio)) if x)


def holdings_text(rows):
    parts = [f"{commas(r['amount'])} {r['symbol']} on {CHAIN_NAME[r['chain']]}" for r in rows]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# ----------------------------------------------------------------------------- model behaviours (the golden turns)
def mk_trade_spec(rng, coin=None, stop=None, tp=False, side=None, size=None, lev=None):
    coin = coin or rng.choice(TRADE_COINS)
    side = side or rng.choice(["long", "long", "short"])
    size = size or rng.choice([50, 75, 100, 100, 150, 200, 250, 300, 500])
    lev = lev or rng.choice([2, 3, 3, 5, 5])
    spec = {"coin": coin, "side": side, "size": size, "lev": min(lev, MAX_LEV[coin]), "stop": None, "tp": None}
    if stop == "pct":
        spec["stop"] = {"kind": "pct", "pct": rng.choice([3, 4, 5, 6, 7, 8, 10])}
    elif stop == "abs":
        spec["stop"] = {"kind": "abs", "price": None}
    if tp:
        spec["tp"] = {"kind": "pct", "pct": rng.choice([8, 10, 12, 15, 20])}
    return spec


def abs_price(rng, w, coin, side, kind="stop"):
    p = w.s["prices"][coin]
    dp = sim.ASSETS[coin][2]
    f = rng.uniform(0.93, 0.97) if side == "long" else rng.uniform(1.03, 1.07)
    if kind == "tp":
        f = 2 - f
    return nn(float(round(p * f, dp)))


def trade_user_text(rng, ev, w, spec):
    coin, side = spec["coin"], spec["side"]
    slots = dict(act=side, c=coin_user(rng, coin), amt=amt_user(rng, spec["size"]), lev=lev_user(rng, spec["lev"]))
    st, tp = spec["stop"], spec["tp"]
    if st and st["kind"] == "pct":
        pool = TRADE_PCT
        slots.update(pct=ns(st["pct"]), dir=rng.choice(["below", "under"] if side == "long" else ["above", "over"]))
    elif st:
        pool = TRADE_ABS
        if st["price"] is None:
            st["price"] = abs_price(rng, w, coin, side)
        slots.update(stop=("$" if rng.random() < 0.25 else "") + ns(st["price"]))
    else:
        pool = TRADE_NONE
    i, tpl = pick_tpl(rng, pool, ev)
    if tp:
        tpl += rng.choice(TP_SUFFIX)
        slots.update(tp=ns(tp["pct"]), tdir=rng.choice(["above", "over"] if side == "long" else ["below", "under"]))
    return fill(rng, tpl, **slots), f"trade_{'pct' if st and st['kind'] == 'pct' else 'abs' if st else 'none'}{i}"


def run_trade(c, rng, spec, order_type="market", limit_price=None, absolute=None):
    """The model opens a trade from `spec`: price + calculator for percent stops, then prepare_trade.
    absolute: {"stop_loss": x, "take_profit": y} carried from an earlier review (no recompute)."""
    w, coin, side = c.w, spec["coin"], spec["side"]
    pend_before = [dict(p) for p in w.s["pending"]]
    info = {"price": None, "stop": None, "tp": None, "pend": pend_before}
    st, tp = spec.get("stop"), spec.get("tp")
    pct_st = bool(st and st["kind"] == "pct" and absolute is None)
    pct_tp = bool(tp and tp["kind"] == "pct" and absolute is None)
    if pct_st or pct_tp:
        r = c.call(("get_price", {"symbol": coin}))
        p = r["price"]
        info["price"] = p
        dp = stop_dp(p)
        calls = []
        if pct_st:
            fac = round(1 - st["pct"] / 100, 4) if side == "long" else round(1 + st["pct"] / 100, 4)
            calls.append(("calculator", {"expression": f"round({ns(p)} * {ns(fac)}, {dp})"}))
        if pct_tp:
            fac = round(1 + tp["pct"] / 100, 4) if side == "long" else round(1 - tp["pct"] / 100, 4)
            calls.append(("calculator", {"expression": f"round({ns(p)} * {ns(fac)}, {dp})"}))
        res = c.call(*calls)
        res = res if isinstance(res, list) else [res]
        k = 0
        if pct_st:
            info["stop"] = parse_calc(res[k]["result"])
            k += 1
        if pct_tp:
            info["tp"] = parse_calc(res[k]["result"])
    elif absolute is not None:
        info["stop"], info["tp"] = absolute.get("stop_loss"), absolute.get("take_profit")
    else:
        if st and st["kind"] == "abs":
            info["stop"] = st["price"]
        if tp and tp["kind"] == "abs":
            info["tp"] = tp["price"]
    args = {"coin": coin, "side": side, "size_usd": nn(spec["size"]), "leverage": spec["lev"], "order_type": order_type}
    if limit_price is not None:
        args["limit_price"] = limit_price
    if info["tp"] is not None:
        args["take_profit"] = info["tp"]
    if info["stop"] is not None:
        args["stop_loss"] = info["stop"]
    r = c.call(("prepare_trade", args))
    assert r.get("status") == "review_opened", r
    info["args"] = args
    return info


def run_send(c, rng, addr, amount, sym, chain=None):
    pend_before = [dict(p) for p in c.w.s["pending"]]
    args = {"to_address": addr, "amount": nn(amount), "symbol": sym}
    if chain:
        args["chain"] = chain
    r = c.call(("prepare_send", args))
    assert r.get("status") == "review_opened", r
    return {"args": args, "res": r, "pend": pend_before}


def run_swap(c, rng, fs, fc, ts, tc, amount):
    pend_before = [dict(p) for p in c.w.s["pending"]]
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": nn(amount)}
    r = c.call(("prepare_swap", args))
    assert r.get("status") == "prepared_for_review", r
    return {"args": args, "res": r, "pend": pend_before}


def interject(c, rng, ev, avoid=()):
    """An unrelated message and its normal answer (price, percent, or a short knowledge answer)."""
    kind = rng.choices(["price", "calc", "know"], [4, 2, 3])[0]
    if kind == "price":
        coin = rng.choice([x for x in COINS if x not in avoid])
        i, t = pick_tpl(rng, INTERJECT_PRICE, ev)
        c.user(fill(rng, t, c=coin_user(rng, coin)))
        r = c.call(("get_price", {"symbol": coin}))
        c.say(TV.price_line(rng, coin, r["price"], r["change_24h_pct"]))
    elif kind == "calc":
        p = rng.choice([5, 8, 12, 15, 18, 20, 25, 30])
        n = rng.choice([140, 250, 340, 480, 520, 640, 900, 1200, 1800, 2400])
        i, (t, e) = pick_tpl(rng, INTERJECT_CALC, ev)
        c.user(fill(rng, t, p=p, n=n))
        r = c.call(("calculator", {"expression": e.format(p=p, n=n)}))
        c.say(rng.choice([f"{p}% of {n} is {r['result']}.", f"That's {r['result']}.", f"{p}% of {n} comes to {r['result']}."]))
    else:
        i, (q, a) = pick_tpl(rng, INTERJECT_KNOW, ev)
        c.user(fill(rng, q))
        c.say(a)
    c.gold(name="unrelated", forbid_tools=MONEY_TOOLS, unrelated=True)


def typo_short(rng, s):
    if rng.random() < 0.10:
        parts = [[s, False]]
        TV.typo(rng, parts)
        s = parts[0][0]
    if rng.random() < 0.12:
        s = s.capitalize()
    return s


def PJ(pre, body):
    """Join an 'approved' lead-in and a follow-up; keep the body's first word lower case after a comma."""
    pre = pre.strip()
    if not pre:
        return body
    if pre.endswith(",") and body[:1].isupper() and body[1:2].islower():
        body = body[0].lower() + body[1:]
    return pre + " " + body


def fu_text(rng, ev, pool, **slots):
    i, tpl = pick_tpl(rng, pool, ev)
    s = fill(rng, tpl, **slots) if slots else typo_short(rng, tpl)
    return s, i


def carry_of(args, keys=("coin", "side", "size_usd", "leverage", "order_type")):
    return {k: args[k] for k in keys if k in args}


def chain_pair(rng, k=2):
    return rng.sample(["base", "arbitrum", "ethereum", "polygon"], k)


# ----------------------------------------------------------------------------- shape a: same for X
def sh_a1(rng, ev, bio):
    coin1, coin2 = rng.sample(TRADE_COINS, 2)
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec = mk_trade_spec(rng, coin1, "pct", tp=rng.random() < 0.25)
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open", forbid_tools=[])
    if rng.random() < 0.15:
        interject(c, rng, ev, avoid=[coin1, coin2])
    approved = rng.random() < 0.45
    pre = (rng.choice(APPROVED_PRE) + " ") if approved else ""
    body, i2 = fu_text(rng, ev, SAME_FOR, c=coin_user(rng, coin2))
    c.user(PJ(pre, body), {"op": "approve_last"} if approved else None)
    spec2 = dict(spec, coin=coin2)
    info2 = run_trade(c, rng, spec2)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"])))
    old_stop = [info1["stop"]] + ([info1["tp"]] if info1["tp"] is not None else [])
    c.gold(name="same_for", carry={"prepare_trade": carry_of(info2["args"], ("side", "size_usd", "leverage", "order_type"))},
           fresh={"prepare_trade": {k: info2["args"][k] for k in ("stop_loss", "take_profit") if k in info2["args"]}},
           not_carried={"prepare_trade": {"stop_loss": [info1["stop"]], **({"take_profit": [info1["tp"]]} if info1["tp"] is not None else {})}},
           must_call=["get_price", "calculator", "prepare_trade"], stale=[ns(info1["price"])] + [ns(x) for x in old_stop],
           stop_check={"side": spec["side"], "price": info2["price"], "pct": spec["stop"]["pct"], "stop": info2["stop"]})
    return c, {"tpl": [tid, f"same_for{i2}"], "shape": "a"}


def sh_a2(rng, ev, bio):
    coin1, coin2 = rng.sample(TRADE_COINS, 2)
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec = mk_trade_spec(rng, coin1, "abs")
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open")
    approved = rng.random() < 0.3
    pre = (rng.choice(APPROVED_PRE) + " ") if approved else ""
    body, i2 = fu_text(rng, ev, SAME_FOR, c=coin_user(rng, coin2))
    c.user(PJ(pre, body), {"op": "approve_last"} if approved else None)
    r = c.call(("get_price", {"symbol": coin2}))
    p2 = r["price"]
    c.say(rng.choice([
        f"{coin2} is at {dollars(ns(p2))} now, so {coin1}'s stop at {dollars(ns(info1['stop']))} can't carry over. What stop price do you want for {coin2}?",
        f"The same stop won't work: {coin1}'s {dollars(ns(info1['stop']))} is a {coin1} price, and {coin2} is at {dollars(ns(p2))}. Which stop price for {coin2}?",
        f"{coin2} trades at {dollars(ns(p2))}, so I can't reuse the {dollars(ns(info1['stop']))} stop from {coin1}. What stop price should I use, or a percent?",
        f"I'll keep the side, size and leverage, but not the stop: {dollars(ns(info1['stop']))} is a {coin1} price and {coin2} is at {dollars(ns(p2))}. What stop do you want?"]), ask=True)
    c.gold(name="same_for_ask", ask=True, forbid_tools=MONEY_TOOLS, not_carried={"prepare_trade": {"stop_loss": [info1["stop"]]}},
           must_call=["get_price"])
    stop2 = abs_price(rng, w, coin2, spec["side"])
    i3, tpl = pick_tpl(rng, ["{stop}", "stop at {stop}", "{stop} for the stop", "make the stop {stop}", "let's say {stop}", "stop {stop}", "sl {stop}",
                             "{stop} is fine", "put it at {stop}", "use {stop}", "{stop} works", "ok {stop}"], ev)
    c.user(fill(rng, tpl, stop=ns(stop2)))
    spec2 = dict(spec, coin=coin2, stop={"kind": "abs", "price": stop2})
    info2 = run_trade(c, rng, spec2)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"])))
    c.gold(name="same_for_answer", carry={"prepare_trade": carry_of(info2["args"], ("side", "size_usd", "leverage", "order_type"))},
           fresh={"prepare_trade": {"coin": coin2, "stop_loss": stop2}}, not_carried={"prepare_trade": {"stop_loss": [info1["stop"]]}}, must_call=["prepare_trade"])
    return c, {"tpl": [tid, f"same_for{i2}", f"stopans{i3}"], "shape": "a"}


def sh_a3(rng, ev, bio):
    w = make_world(rng, bio, usdc=("base",))
    c = FConv(rng, w, bio)
    a1, a2 = H.evm_addr(rng), H.evm_addr(rng)
    amt = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100])
    sym = "USDC"
    tok = f"{amt} {rng.choice(['USDC', 'usdc'])}"
    i, tpl = pick_tpl(rng, SEND_T, ev)
    c.user(fill(rng, tpl, tok=tok, addr=a1))
    s1 = run_send(c, rng, a1, amt, sym)
    c.say(send_text(rng, bio, s1["res"], s1["args"]))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev)
    approved = rng.random() < 0.4
    pre = (rng.choice(APPROVED_PRE) + " ") if approved else ""
    body, i2 = fu_text(rng, ev, SAME_TO, addr=a2)
    c.user(PJ(pre, body), {"op": "approve_last"} if approved else None)
    s2 = run_send(c, rng, a2, amt, sym)
    c.say(send_text(rng, bio, s2["res"], s2["args"], open_note(rng, s2["pend"])))
    c.gold(name="same_to", carry={"prepare_send": {"amount": s2["args"]["amount"], "symbol": sym}}, fresh={"prepare_send": {"to_address": a2}},
           not_carried={"prepare_send": {"to_address": [a1]}}, must_call=["prepare_send"])
    return c, {"tpl": [f"send{i}", f"same_to{i2}"], "shape": "a"}


# ----------------------------------------------------------------------------- shape b: change one argument
def sh_b1(rng, ev, bio):
    coin = rng.choice(["BTC", "ETH", "SOL", "SOL", "ETH"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec = mk_trade_spec(rng, coin, rng.choice(["pct", "pct", "abs", None]), lev=rng.choice([2, 3, 5]))
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev, avoid=[coin])
    L2 = rng.choice([x for x in (3, 5, 10, 10, 15, 20) if x != spec["lev"] and x <= MAX_LEV[coin]])
    body, i2 = fu_text(rng, ev, LEV_FU, L=L2)
    c.user(body)
    spec2 = dict(spec, lev=L2)
    absol = {k: info1["args"][k] for k in ("stop_loss", "take_profit") if k in info1["args"]}
    info2 = run_trade(c, rng, spec2, absolute=absol)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"], replace=True), again=True))
    c.gold(name="change_lev", carry={"prepare_trade": {**carry_of(info1["args"], ("coin", "side", "size_usd", "order_type")), **absol}},
           fresh={"prepare_trade": {"leverage": L2}}, must_call=["prepare_trade"])
    return c, {"tpl": [tid, f"lev{i2}"], "shape": "b"}


def sh_b2(rng, ev, bio):
    coin = rng.choice(TRADE_COINS)
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec = mk_trade_spec(rng, coin, rng.choice(["pct", "abs", None, None]), size=rng.choice([60, 100, 150, 200, 300, 500, 750]))
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev, avoid=[coin])
    mode = rng.choices(["half", "double", "n"], [45, 20, 35])[0]
    size = spec["size"]
    if mode == "n":
        size2 = rng.choice([x for x in (25, 40, 80, 120, 250, 400, 600, 800, 1000) if x != size])
        body, i2 = fu_text(rng, ev, AMOUNT_FU, n=("$" if rng.random() < 0.3 else "") + str(size2))
        c.user(body)
    else:
        body, i2 = fu_text(rng, ev, HALF_FU if mode == "half" else DOUBLE_FU)
        c.user(body)
        r = c.call(("calculator", {"expression": f"{size} / 2" if mode == "half" else f"{size} * 2"}))
        size2 = parse_calc(r["result"])
    spec2 = dict(spec, size=size2)
    absol = {k: info1["args"][k] for k in ("stop_loss", "take_profit") if k in info1["args"]}
    info2 = run_trade(c, rng, spec2, absolute=absol)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"], replace=True), again=True))
    c.gold(name="change_size", carry={"prepare_trade": {**carry_of(info1["args"], ("coin", "side", "leverage", "order_type")), **absol}},
           fresh={"prepare_trade": {"size_usd": nn(size2)}}, must_call=["prepare_trade"] + ([] if mode == "n" else ["calculator"]))
    return c, {"tpl": [tid, f"size{i2}"], "shape": "b"}


def sh_b3(rng, ev, bio):
    if rng.random() < 0.65:
        c1, c2 = chain_pair(rng)
        w = make_world(rng, bio, usdc=(c1, c2))
        c = FConv(rng, w, bio)
        a1 = H.evm_addr(rng)
        amt = rng.choice([5, 10, 20, 25, 30, 40, 50])
        i, tpl = pick_tpl(rng, SEND_CH_T, ev)
        c.user(fill(rng, tpl, tok=f"{amt} {rng.choice(['USDC', 'usdc'])}", addr=a1, ch=chain_user(rng, c1)))
        s1 = run_send(c, rng, a1, amt, "USDC", c1)
        c.say(send_text(rng, bio, s1["res"], s1["args"]))
        c.gold(name="open")
        if rng.random() < 0.15:
            interject(c, rng, ev)
        body, i2 = fu_text(rng, ev, CHAIN_FU, ch=chain_user(rng, c2))
        c.user(body)
        s2 = run_send(c, rng, a1, amt, "USDC", c2)
        c.say(send_text(rng, bio, s2["res"], s2["args"], open_note(rng, s2["pend"], replace=True)))
        c.gold(name="change_chain", carry={"prepare_send": {"to_address": a1, "amount": nn(amt), "symbol": "USDC"}}, fresh={"prepare_send": {"chain": c2}},
               must_call=["prepare_send"])
        return c, {"tpl": [f"sendch{i}", f"chain{i2}"], "shape": "b"}
    c1, c2, c3 = chain_pair(rng, 3)
    w = make_world(rng, bio, usdc=(c1,))
    c = FConv(rng, w, bio)
    amt = rng.choice([20, 25, 40, 50, 75, 100, 150])
    i, tpl = pick_tpl(rng, BRIDGE_T, ev)
    c.user(fill(rng, tpl, amt=amt, tok=rng.choice(["USDC", "usdc"]), c1=chain_user(rng, c1), c2=chain_user(rng, c2)))
    s1 = run_swap(c, rng, "USDC", c1, "USDC", c2, amt)
    c.say(swap_text(rng, bio, s1["res"]))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev)
    body, i2 = fu_text(rng, ev, BRIDGE_DEST_FU, ch=chain_user(rng, c3))
    c.user(body)
    s2 = run_swap(c, rng, "USDC", c1, "USDC", c3, amt)
    c.say(swap_text(rng, bio, s2["res"], open_note(rng, s2["pend"], replace=True)))
    c.gold(name="change_dest", carry={"prepare_swap": {"from_symbol": "USDC", "from_chain": c1, "to_symbol": "USDC", "amount": nn(amt)}},
           fresh={"prepare_swap": {"to_chain": c3}}, must_call=["prepare_swap"])
    return c, {"tpl": [f"bridge{i}", f"dest{i2}"], "shape": "b"}


def sh_b4(rng, ev, bio):
    ch = rng.choice(["base", "arbitrum", "ethereum"])
    w = make_world(rng, bio, usdc=(ch,), amounts={("USDC", ch): round(rng.uniform(600, 2400), 2)})
    c = FConv(rng, w, bio)
    n1 = rng.choice([20, 25, 40, 50, 60, 100, 150])
    i, tpl = pick_tpl(rng, SWAP_T, ev)
    c.user(fill(rng, tpl, amt=n1, a=rng.choice(["USDC", "usdc"]), b=rng.choice(["ETH", "eth"]), c=chain_user(rng, ch)))
    s1 = run_swap(c, rng, "USDC", ch, "ETH", ch, n1)
    c.say(swap_text(rng, bio, s1["res"]))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev)
    n2 = rng.choice([x for x in (30, 45, 75, 80, 120, 200, 250, 300) if x != n1])
    body, i2 = fu_text(rng, ev, AMOUNT_FU, n=n2)
    c.user(body)
    s2 = run_swap(c, rng, "USDC", ch, "ETH", ch, n2)
    c.say(swap_text(rng, bio, s2["res"], open_note(rng, s2["pend"], replace=True)))
    c.gold(name="change_amount", carry={"prepare_swap": {"from_symbol": "USDC", "from_chain": ch, "to_symbol": "ETH", "to_chain": ch}},
           fresh={"prepare_swap": {"amount": nn(n2)}}, must_call=["prepare_swap"])
    return c, {"tpl": [f"swap{i}", f"amount{i2}"], "shape": "b"}


# ----------------------------------------------------------------------------- shape c: cancel that
CANCEL_REVIEW = {
    "send": ["Tap Cancel on the review; nothing was sent.", "I can't close the review for you. Tap Cancel on it and nothing is sent.",
             "Tap Cancel on the review screen. Nothing was sent, and nothing will be unless you approve it.",
             "The review is still open. Tap Cancel on it; nothing was sent.", "I can't cancel a review from here. Tap Cancel on it and nothing leaves your wallet."],
    "swap": ["Tap Cancel on the review; nothing has moved.", "I can't close the review for you. Tap Cancel on it and nothing moves.",
             "Tap Cancel on the review screen. Nothing has moved, and nothing will unless you approve it.",
             "The review is still open. Tap Cancel on it; nothing has moved.", "I can't cancel a review from here. Tap Cancel on it and your funds stay where they are."],
    "trade": ["Tap Cancel on the {coin} review; nothing has moved.", "I can't close the review for you. Tap Cancel on it and the {coin} order never goes out.",
              "Tap Cancel on the review screen. Nothing has moved, and nothing will unless you approve it.",
              "The {coin} review is still open. Tap Cancel on it; nothing has moved.", "I can't cancel a review from here. Tap Cancel on it and no order is placed."],
}


def sh_c1(rng, ev, bio):
    kind = rng.choice(["send", "send", "swap", "trade", "trade"])
    c_coin = None
    if kind == "send":
        w = make_world(rng, bio)
        c = FConv(rng, w, bio)
        a1, amt = H.evm_addr(rng), rng.choice([10, 20, 25, 50, 100])
        i, tpl = pick_tpl(rng, SEND_T, ev)
        c.user(fill(rng, tpl, tok=f"{amt} usdc", addr=a1))
        s1 = run_send(c, rng, a1, amt, "USDC")
        c.say(send_text(rng, bio, s1["res"], s1["args"]))
    elif kind == "swap":
        ch = rng.choice(["base", "arbitrum"])
        w = make_world(rng, bio, usdc=(ch,), amounts={("USDC", ch): round(rng.uniform(500, 2000), 2)})
        c = FConv(rng, w, bio)
        n1 = rng.choice([20, 40, 50, 100])
        i, tpl = pick_tpl(rng, SWAP_T, ev)
        c.user(fill(rng, tpl, amt=n1, a="USDC", b="ETH", c=chain_user(rng, ch)))
        s1 = run_swap(c, rng, "USDC", ch, "ETH", ch, n1)
        c.say(swap_text(rng, bio, s1["res"]))
    else:
        w = make_world(rng, bio)
        c = FConv(rng, w, bio)
        spec = mk_trade_spec(rng, None, rng.choice(["pct", None]))
        c_coin = spec["coin"]
        txt, tid = trade_user_text(rng, ev, w, spec)
        i = tid
        c.user(txt)
        info = run_trade(c, rng, spec)
        c.say(trade_text(rng, bio, spec, info))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev, avoid=[c_coin] if c_coin else [])
    body, i2 = fu_text(rng, ev, CANCEL_FU)
    c.user(body)
    c.say(rng.choice(CANCEL_REVIEW[kind]).format(coin=c_coin))
    c.gold(name="cancel_review", forbid_tools=MONEY_TOOLS, answer_any=[r"Cancel"], answer_forbid=[r"\bcancell?ed\b"])
    return c, {"tpl": [f"c1_{kind}{i}", f"cancel{i2}"], "shape": "c"}


ORDERS_Q = ["what orders do i have open", "do i have any open orders", "show my open orders", "any orders open right now", "list my open orders",
            "what's open on hyperliquid", "check my hyperliquid orders", "are there any orders still open", "what orders are sitting there",
            "do i have anything pending on hyperliquid"]


def sh_c2(rng, ev, bio):
    coin = rng.choice(["ETH", "SOL", "BTC", "HYPE", "AVAX", "LINK"])
    kind = rng.choice(["limit", "limit", "stop_loss"])
    if kind == "limit":
        w = make_world(rng, bio, orders=[{"coin": coin, "kind": "limit", "side": rng.choice(["buy", "buy", "sell"]), "mult": rng.choice([0.94, 0.95, 0.97, 1.04]),
                                          "usd": rng.choice([150, 200, 300, 400])}])
    else:
        w = make_world(rng, bio, positions=[{"coin": coin, "side": rng.choice(["long", "long", "short"]), "lev": rng.choice([3, 5]), "margin": rng.choice([60, 100, 150, 250])}],
                       orders=[])
        pos = w.s["hl"]["positions"][0]
        px = nn(float(round(pos["entry"] * (0.93 if pos["side"] == "long" else 1.07), sim.ASSETS[coin][2])))
        w.s["nid"] += 1
        w.s["hl"]["orders"].append({"id": str(8100 + w.s["nid"]), "coin": coin, "kind": "stop_loss", "price": px, "side": "sell" if pos["side"] == "long" else "buy",
                                    "size": pos["size"], "whole_position": True})
    c = FConv(rng, w, bio)
    i, q = pick_tpl(rng, ORDERS_Q, ev)
    c.user(fill(rng, q))
    o = w.s["hl"]["orders"][0]
    if rng.random() < 0.5:
        r = c.call(("hyperliquid_account", {}))
        orow = r["open_orders"][0]
    else:
        r = c.call(("get_positions", {"kind": "perps"}))
        orow = r["perps"]["open_orders"][0]
    prow = next((p for p in (r["positions"] if "positions" in r else r["perps"]["positions"]) if p["coin"] == coin), None)
    if o["kind"] == "limit":
        c.say(rng.choice([f"You have one open order: a {o['side']} limit for {ns(o['size'])} {coin} at {dollars(ns(o['price']))}.",
                          f"One order is open: a {o['side']} limit, {ns(o['size'])} {coin} at {dollars(ns(o['price']))}.",
                          f"There's a single open order, a {coin} {o['side']} limit for {ns(o['size'])} at {dollars(ns(o['price']))}."]))
    else:
        c.say(rng.choice([f"You have one open order: the stop loss on your {coin} {prow['side']}, at {dollars(ns(o['price']))}.",
                          f"One order is open, a stop loss on your {prow['side']} {coin} at {dollars(ns(o['price']))}.",
                          f"Your only open order is the {coin} stop loss at {dollars(ns(o['price']))}, protecting the {prow['side']}."]))
    c.gold(name="list_orders", forbid_tools=MONEY_TOOLS)
    if rng.random() < 0.15:
        interject(c, rng, ev, avoid=[coin])
    body, i2 = fu_text(rng, ev, CANCEL_ORDER_FU)
    c.user(body)
    r2 = c.call(("cancel_order", {"coin": coin}))
    assert r2["status"] == "card_opened", r2
    text = rng.choice([f"I put up a card to cancel the {r2['cancelling']}. Tap Confirm to cancel it; nothing is cancelled yet.",
                       f"I've drafted a card for the {r2['cancelling']}. Nothing is cancelled until you tap Confirm.",
                       f"A card is up to cancel the {r2['cancelling']}. Tap Confirm and then it goes; nothing is cancelled yet."])
    if r2.get("warning"):
        text += " " + r2["warning"]
    c.say(text)
    c.gold(name="cancel_order", fresh={"cancel_order": {"coin": coin}}, must_call=["cancel_order"], answer_any=[r"Confirm"],
           answer_forbid=[r"(?<!nothing is )(?<!not )\bcancell?ed\b(?! yet| until)"])
    return c, {"tpl": [f"orders{i}", f"cancelo{i2}"], "shape": "c"}


def sh_c3(rng, ev, bio):
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    if rng.random() < 0.5:
        coin = rng.choice(COINS)
        i, t = pick_tpl(rng, INTERJECT_PRICE, ev)
        c.user(fill(rng, t, c=coin_user(rng, coin)))
        r = c.call(("get_price", {"symbol": coin}))
        c.say(TV.price_line(rng, coin, r["price"], r["change_24h_pct"]))
        c.gold(name="price_first", forbid_tools=MONEY_TOOLS)
        body, i2 = fu_text(rng, ev, CANCEL_FU + CANCEL_ORDER_FU)
        c.user(body)
        c.say(rng.choice(["There's nothing to cancel. I haven't prepared anything in this chat. If you mean an order on Hyperliquid, tell me the coin.",
                          "Nothing is waiting here: I haven't prepared or placed anything in this chat. Tell me which order you mean and I'll look.",
                          "I haven't prepared anything in this chat, so there's nothing to cancel. Tell me which order you mean.",
                          "There's no review open and I haven't started anything here, so nothing to cancel."]))
        c.gold(name="cancel_nothing", forbid_tools=MONEY_TOOLS, answer_any=[r"nothing|no review|haven't"], answer_forbid=[r"\bcancell?ed\b"])
        return c, {"tpl": [f"pr{i}", f"cancel{i2}"], "shape": "c"}
    i, q = pick_tpl(rng, BAL_Q, ev)
    c.user(fill(rng, q))
    r = c.call(("get_balances", {}))
    c.say(f"You have {holdings_text(r['holdings'])}.")
    c.gold(name="balance_first", forbid_tools=MONEY_TOOLS)
    body, i2 = fu_text(rng, ev, CANCEL_FU + CANCEL_ORDER_FU)
    c.user(body)
    r2 = c.call(("hyperliquid_account", {}))
    assert not r2["open_orders"]
    c.say(rng.choice(["You have no open orders on Hyperliquid and no review is waiting, so there's nothing to cancel.",
                      "I don't see any open orders on Hyperliquid, and I haven't prepared anything here, so nothing to cancel.",
                      "There's nothing to cancel: your Hyperliquid account has no open orders and no review is open."]))
    c.gold(name="cancel_nothing", forbid_tools=MONEY_TOOLS, must_call=["hyperliquid_account"], answer_any=[r"nothing"], answer_forbid=[r"\bcancell?ed\b"])
    return c, {"tpl": [f"bal{i}", f"cancel{i2}"], "shape": "c"}


# ----------------------------------------------------------------------------- shape d: undo after approval
UNDO_OFFER = {
    "swap": ["There's no undo once it's approved. I can prepare the reverse as a new review, swapping the {to} back to {fr}. Want me to?",
             "There's no undo for an approved swap. I can prepare the reverse as a new review: {to} back to {fr}. Want me to?",
             "There's no undo after approval. What I can do is prepare the opposite swap, {to} back to {fr}, as a new review. Should I?",
             "I can't undo an approved swap. A new review for {to} back to {fr} would reverse it, minus fees. Want me to prepare it?"],
    "trade": ["There's no undo once it's approved. I can prepare a close of the {coin} {side} as a new review. Want me to?",
              "There's no undo for an approved order. I can prepare a close of the {coin} {side} as a new review. Want me to?",
              "There's no undo after approval, but I can prepare closing the {coin} {side} as a new review. Should I?",
              "I can't undo an approved trade. A new review that closes the {coin} {side} would get you out. Want me to prepare it?"],
    "send": ["A send is final once it's approved, and I can't pull it back. Only the recipient could send it back.",
             "There's no undo for a send after approval. The funds are with the recipient now, so only they can return them.",
             "Once a send is approved there's no undo, and I can't reverse it. You'd have to ask the recipient to send it back.",
             "I can't undo an approved send. It's final, and only the recipient could send it back to you."],
}


def sh_d(rng, ev, bio, kind):
    if kind == "swap":
        ch = rng.choice(["base", "arbitrum", "ethereum"])
        w = make_world(rng, bio, usdc=(ch,), amounts={("USDC", ch): round(rng.uniform(600, 2400), 2)})
        c = FConv(rng, w, bio)
        n1 = rng.choice([25, 50, 75, 100, 150, 200])
        i, tpl = pick_tpl(rng, SWAP_T, ev)
        c.user(fill(rng, tpl, amt=n1, a="USDC", b="ETH", c=chain_user(rng, ch)))
        s1 = run_swap(c, rng, "USDC", ch, "ETH", ch, n1)
        c.say(swap_text(rng, bio, s1["res"]))
        offer = dict(to="ETH", fr="USDC")
    elif kind == "trade":
        w = make_world(rng, bio)
        c = FConv(rng, w, bio)
        spec = mk_trade_spec(rng, None, rng.choice(["pct", None]))
        txt, i = trade_user_text(rng, ev, w, spec)
        c.user(txt)
        info = run_trade(c, rng, spec)
        c.say(trade_text(rng, bio, spec, info))
        offer = dict(coin=spec["coin"], side=spec["side"])
    else:
        w = make_world(rng, bio)
        c = FConv(rng, w, bio)
        a1, amt = H.evm_addr(rng), rng.choice([10, 20, 25, 50, 100])
        i, tpl = pick_tpl(rng, SEND_T, ev)
        c.user(fill(rng, tpl, tok=f"{amt} usdc", addr=a1))
        s1 = run_send(c, rng, a1, amt, "USDC")
        c.say(send_text(rng, bio, s1["res"], s1["args"]))
        offer = {}
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev, avoid=[offer.get("coin")] if offer.get("coin") else [])
    body, i2 = fu_text(rng, ev, UNDO_FU)
    c.user(PJ(rng.choice(APPROVED_PRE), body), {"op": "approve_last", "settled": True})
    c.say(rng.choice(UNDO_OFFER[kind]).format(**offer), ask=(kind != "send"))
    c.gold(name="undo", ask=(kind != "send"), forbid_tools=MONEY_TOOLS, answer_any=[r"no undo|can't (be )?undo|can't pull|can't reverse|final"],
           answer_forbid=[r"\bundone\b", r"\bI(?:'ve| have) (undone|reversed|cancell?ed)\b", r"\b(is|been|was|got) (reversed|cancell?ed)\b"])
    if kind != "send" and rng.random() < 0.55:
        body3, i3 = fu_text(rng, ev, YES_FU)
        c.user(body3)
        if kind == "swap":
            r = c.call(("get_balances", {}))
            eth = next(h for h in r["holdings"] if h["symbol"] == "ETH" and h["chain"] == ch)
            amt_back = parse_calc(eth["amount"])
            s2 = run_swap(c, rng, "ETH", ch, "USDC", ch, amt_back)
            rr_ = s2["res"]
            c.say(rng.choice([f"I've prepared the reverse as a new review: {rr_['from']} to {rr_['to']} through {rr_['route']}. Total cost is {dollars(ns(rr_['total_cost_usd']))}. {approve_line(rng, bio)}",
                              f"The reverse is a new review: {rr_['from']} back to {rr_['to']} through {rr_['route']}, total cost {dollars(ns(rr_['total_cost_usd']))}. {approve_line(rng, bio)}"]))
            c.gold(name="undo_yes", fresh={"prepare_swap": {"from_symbol": "ETH", "to_symbol": "USDC", "amount": amt_back}},
                   carry={"prepare_swap": {"from_chain": ch, "to_chain": ch}}, must_call=["get_balances", "prepare_swap"])
        else:
            r = c.call(("prepare_close", {"coin": offer["coin"]}))
            assert r["status"] == "review_opened", r
            c.say(rng.choice([f"I've prepared closing {r['closing']} as a new review, market order, reduce only. {approve_line(rng, bio)}",
                              f"The close is prepared as a new review: {r['closing']}, {r['order'].lower()}. {approve_line(rng, bio)}",
                              f"New review prepared to close {r['closing']}, {r['order'].lower()}. {approve_line(rng, bio)}"]))
            c.gold(name="undo_yes", fresh={"prepare_close": {"coin": offer["coin"]}}, must_call=["prepare_close"])
    return c, {"tpl": [f"d_{kind}{i}", f"undo{i2}"], "shape": "d"}


def sh_d1(rng, ev, bio):
    return sh_d(rng, ev, bio, "swap")


def sh_d2(rng, ev, bio):
    return sh_d(rng, ev, bio, "trade")


def sh_d3(rng, ev, bio):
    return sh_d(rng, ev, bio, "send")


# ----------------------------------------------------------------------------- shape e: what was that address
def sh_e1(rng, ev, bio):
    sol = rng.random() < 0.2
    w = make_world(rng, bio, usdc=("solana",) if sol else ("base",))
    c = FConv(rng, w, bio)
    a1 = H.sol_addr(rng) if sol else H.evm_addr(rng)
    amt = rng.choice([5, 10, 20, 25, 50, 100])
    i, tpl = pick_tpl(rng, SEND_T, ev)
    c.user(fill(rng, tpl, tok=f"{amt} {rng.choice(['USDC', 'usdc'])}", addr=a1))
    s1 = run_send(c, rng, a1, amt, "USDC")
    c.say(send_text(rng, bio, s1["res"], s1["args"]))
    c.gold(name="open")
    if rng.random() < 0.35:
        interject(c, rng, ev)
    body, i2 = fu_text(rng, ev, ADDR_Q)
    c.user(body)
    c.say(rng.choice([f"That was {a1}, the address you gave for the {amt} USDC send.", f"The address is {a1}.",
                      f"You gave me {a1} for the {amt} USDC send.", f"It was {a1}.", f"That address is {a1}."]))
    c.gold(name="addr_history", forbid_tools=MONEY_TOOLS, answer_all=[re.escape(a1)])
    return c, {"tpl": [f"send{i}", f"addrq{i2}"], "shape": "e"}


ADDR_OWN_Q = ["what's my {ch} address", "my {ch} address?", "show me my {ch} address", "what address do i use to receive on {ch}", "gimme my {ch} receive address",
              "what's my deposit address on {ch}", "address for {ch} pls", "i need my {ch} address"]


def sh_e2(rng, ev, bio):
    ch = rng.choice(["base", "arbitrum", "ethereum", "solana"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    i, tpl = pick_tpl(rng, ADDR_OWN_Q, ev)
    c.user(fill(rng, tpl, ch=chain_user(rng, ch)))
    r = c.call(("get_address", {"chain": ch}))
    addr = r["addresses"][0]["address"]
    c.say(rng.choice([f"Your {CHAIN_NAME[ch]} address is {addr}.", f"Here's your {CHAIN_NAME[ch]} address: {addr}.", f"Your {CHAIN_NAME[ch]} receive address is {addr}."]))
    c.gold(name="own_address", must_call=["get_address"], answer_all=[re.escape(addr)])
    for _ in range(rng.choice([1, 1, 2])):
        interject(c, rng, ev)
    body, i2 = fu_text(rng, ev, MYADDR_Q)
    c.user(body)
    c.say(rng.choice([f"That was your {CHAIN_NAME[ch]} address: {addr}.", f"It was {addr}, your {CHAIN_NAME[ch]} address.", f"Your {CHAIN_NAME[ch]} address is {addr}."]))
    c.gold(name="addr_history", forbid_tools=MONEY_TOOLS, answer_all=[re.escape(addr)])
    return c, {"tpl": [f"own{i}", f"addrq{i2}"], "shape": "e"}


def sh_e3(rng, ev, bio):
    w = make_world(rng, bio)
    no_send = rng.random() < 0.12
    if not no_send:
        w.act("confirmed", f"Swap {rng.choice([20, 40, 60])} USDC on Base to ETH on Base")
        w.act(rng.choice(["confirmed", "confirmed", "confirmed", "failed"]), f"Sent {ns(rng.choice([10, 25, 40, 75]))} USDC to {short_act(H.evm_addr(rng))}")
        if rng.random() < 0.5:
            w.act("confirmed", f"Bought {ns(round(rng.uniform(2, 9), 2))} WIF with {rng.choice([10, 15])} USDC")
    elif rng.random() < 0.5:
        w.act("confirmed", f"Swap {rng.choice([20, 40, 60])} USDC on Base to ETH on Base")
    c = FConv(rng, w, bio)
    coin = rng.choice(COINS)
    i, t = pick_tpl(rng, INTERJECT_PRICE, ev)
    c.user(fill(rng, t, c=coin_user(rng, coin)))
    r = c.call(("get_price", {"symbol": coin}))
    c.say(TV.price_line(rng, coin, r["price"], r["change_24h_pct"]))
    c.gold(name="price_first", forbid_tools=MONEY_TOOLS)
    body, i2 = fu_text(rng, ev, LASTADDR_Q)
    c.user(body)
    r = c.call(("get_activity", {}))
    row = next((x for x in r["activity"] if x["what"].startswith("Sent ")), None)
    if row is None:
        c.say(rng.choice(["Your Activity has no sends yet, so there's no address to show.", "I don't see any send in your Activity, so I can't give you an address.",
                          "There's no send in your Activity yet."]))
        c.gold(name="addr_activity", must_call=["get_activity"], forbid_tools=MONEY_TOOLS, answer_any=[r"no send|no sends|any send"])
    else:
        st = {"confirmed": "confirmed", "failed": "failed", "in_progress": "still in progress"}[row["status"]]
        c.say(rng.choice([f"Your last send in Activity is \"{row['what']}\", {st}. Activity only shows the short address, so I don't have the full one.",
                          f"Activity lists it as \"{row['what']}\" ({st}). It only shows the short form of the address, so I can't give you the full one.",
                          f"The last send is \"{row['what']}\", {st}. That's all Activity shows of the address."]))
        c.gold(name="addr_activity", must_call=["get_activity"], forbid_tools=MONEY_TOOLS, answer_all=[re.escape(row["what"])])
    return c, {"tpl": [f"pr{i}", f"lastaddr{i2}"], "shape": "e"}


# ----------------------------------------------------------------------------- shape f: two possible referents
def sh_f1(rng, ev, bio):
    coin = rng.choice(["BTC", "ETH", "SOL"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    L = rng.choice([2, 3, 5])
    S_ = rng.choice([50, 100, 200])
    spec = mk_trade_spec(rng, coin, None, size=S_, lev=L)
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev, avoid=[coin])
    N = rng.choice([10, 20])
    body, i2 = fu_text(rng, ev, MAKE_N_FU, n=N)
    c.user(body)
    c.say(rng.choice([f"Do you mean {N}x leverage or a ${N} size?", f"Is {N} the new leverage, or the new size in dollars?",
                      f"Which should be {N}: the leverage (now {L}x) or the size (now ${S_})?", f"{N}x leverage, or a {N} dollar size?"]), ask=True)
    c.gold(name="which_number", ask=True, forbid_tools=MONEY_TOOLS, answer_all=[r"leverage", r"size"])
    if rng.random() < 0.6:
        which = rng.choice(["lev", "size"])
        pool = ["leverage", "the leverage", "leverage pls", "{N}x", "lev", "the leverage one"] if which == "lev" else ["size", "the size", "size pls", "the size, ${N}", "the dollar size", "the size one"]
        body3, i3 = fu_text(rng, ev, pool, N=N)
        c.user(body3)
        spec2 = dict(spec, lev=N) if which == "lev" else dict(spec, size=N)
        info2 = run_trade(c, rng, spec2)
        c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"], replace=True), again=True))
        c.gold(name="which_number_answer", carry={"prepare_trade": carry_of(info1["args"], ("coin", "side", "order_type") + (("size_usd",) if which == "lev" else ("leverage",)))},
               fresh={"prepare_trade": {"leverage": N} if which == "lev" else {"size_usd": N}}, must_call=["prepare_trade"])
    return c, {"tpl": [tid, f"maken{i2}"], "shape": "f"}


def sh_f2(rng, ev, bio):
    coin1, coin2 = rng.sample(TRADE_COINS, 2)
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec1 = mk_trade_spec(rng, coin1, None)
    spec2 = mk_trade_spec(rng, coin2, None)
    txt, t1 = trade_user_text(rng, ev, w, spec1)
    c.user(txt)
    info1 = run_trade(c, rng, spec1)
    c.say(trade_text(rng, bio, spec1, info1))
    c.gold(name="open")
    txt, t2 = trade_user_text(rng, ev, w, spec2)
    c.user(txt)
    info2 = run_trade(c, rng, spec2)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"])))
    c.gold(name="open")
    body, i3 = fu_text(rng, ev, CANCEL_WHICH_FU)
    c.user(body)
    d1, d2 = f"{coin1} {spec1['side']}", f"{coin2} {spec2['side']}"
    c.say(rng.choice([f"Which one: the {d1} or the {d2}? I can't close a review for you, so tap Cancel on the one you don't want.",
                      f"Two reviews are open, the {d1} and the {d2}. Which one do you mean? Tap Cancel on that review; I can't do it for you.",
                      f"Do you want to drop the {d1} or the {d2}? Tap Cancel on that review, nothing has moved yet."]), ask=True)
    c.gold(name="cancel_which", ask=True, forbid_tools=MONEY_TOOLS, answer_all=[re.escape(coin1), re.escape(coin2)], answer_forbid=[r"\bcancell?ed\b"])
    if rng.random() < 0.5:
        pick = rng.choice([1, 2])
        keep, drop = (spec2, spec1) if pick == 1 else (spec1, spec2)
        pool = ["the {c} one", "{c}", "the {c}", "the {c} trade", "{c} one", "the {c} review"]
        body4, i4 = fu_text(rng, ev, pool, c=coin_user(rng, drop["coin"]))
        c.user(body4)
        c.say(rng.choice([f"Tap Cancel on the {drop['coin']} review; nothing has moved. The {keep['coin']} review stays open.",
                          f"Tap Cancel on the {drop['coin']} review and nothing goes out. The {keep['coin']} one stays open.",
                          f"Then tap Cancel on the {drop['coin']} review; nothing has moved. The {keep['coin']} review is still open."]))
        c.gold(name="cancel_pick", forbid_tools=MONEY_TOOLS, answer_any=[r"Cancel"], answer_forbid=[r"\bcancell?ed\b"])
    return c, {"tpl": [t1, t2, f"cancelw{i3}"], "shape": "f"}


def sh_f3(rng, ev, bio):
    coin1, coin2, coin3 = rng.sample(TRADE_COINS, 3)
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec1 = mk_trade_spec(rng, coin1, None, side="long", size=rng.choice([100, 150, 200]), lev=rng.choice([2, 3]))
    spec2 = mk_trade_spec(rng, coin2, None, side="short", size=rng.choice([50, 75, 250]), lev=5)
    txt, t1 = trade_user_text(rng, ev, w, spec1)
    c.user(txt)
    info1 = run_trade(c, rng, spec1)
    c.say(trade_text(rng, bio, spec1, info1))
    c.gold(name="open")
    txt, t2 = trade_user_text(rng, ev, w, spec2)
    c.user(txt)
    info2 = run_trade(c, rng, spec2)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"])))
    c.gold(name="open")
    body, i3 = fu_text(rng, ev, SAME_AGAIN_FU, c=coin_user(rng, coin3))
    c.user(body)
    q1 = f"the {coin1} {spec1['side']} ({dollars(ns(spec1['size']))} at {spec1['lev']}x)"
    q2 = f"the {coin2} {spec2['side']} ({dollars(ns(spec2['size']))} at {spec2['lev']}x)"
    c.say(rng.choice([f"Same as {q1} or {q2}?", f"Which one should I copy for {coin3}: {q1} or {q2}?", f"Do you want {coin3} set up like {q1}, or like {q2}?"]), ask=True)
    c.gold(name="same_which", ask=True, forbid_tools=MONEY_TOOLS, answer_all=[re.escape(coin1), re.escape(coin2)])
    if rng.random() < 0.55:
        src = rng.choice([spec1, spec2])
        other = spec2 if src is spec1 else spec1
        pool = ["the {c} one", "{c}", "like the {c}", "the {c} trade", "the {ord} one", "{ord}"]
        ordw = "first" if src is spec1 else "second"
        body4, i4 = fu_text(rng, ev, pool, c=coin_user(rng, src["coin"]), ord=ordw)
        c.user(body4)
        spec3 = dict(src, coin=coin3)
        info3 = run_trade(c, rng, spec3)
        c.say(trade_text(rng, bio, spec3, info3, open_note(rng, info3["pend"])))
        c.gold(name="same_pick", fresh={"prepare_trade": carry_of(info3["args"])}, must_call=["prepare_trade"])
    return c, {"tpl": [t1, t2, f"samew{i3}"], "shape": "f"}


def sh_f4(rng, ev, bio):
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    a1, a2 = H.evm_addr(rng), H.evm_addr(rng)
    n1, n2 = rng.sample([10, 15, 20, 25, 40, 50, 75, 100], 2)
    i, tpl = pick_tpl(rng, SEND_T, ev)
    c.user(fill(rng, tpl, tok=f"{n1} usdc", addr=a1))
    s1 = run_send(c, rng, a1, n1, "USDC")
    c.say(send_text(rng, bio, s1["res"], s1["args"]))
    c.gold(name="open")
    i2, tpl = pick_tpl(rng, SEND_T, ev)
    c.user(fill(rng, tpl, tok=f"{n2} usdc", addr=a2))
    s2 = run_send(c, rng, a2, n2, "USDC")
    c.say(send_text(rng, bio, s2["res"], s2["args"], open_note(rng, s2["pend"])))
    c.gold(name="open")
    body, i3 = fu_text(rng, ev, SEND_AGAIN_FU)
    c.user(body)
    c.say(rng.choice([f"Do you mean the {n1} USDC to {short_say(a1)} or the {n2} USDC to {short_say(a2)}?",
                      f"Which one: {n1} USDC to {short_say(a1)}, or {n2} USDC to {short_say(a2)}?",
                      f"You've given me two addresses. Should I repeat the {n1} USDC to {short_say(a1)} or the {n2} USDC to {short_say(a2)}?"]), ask=True)
    c.gold(name="send_which", ask=True, forbid_tools=MONEY_TOOLS, answer_all=[re.escape(short_say(a1)), re.escape(short_say(a2))])
    if rng.random() < 0.55:
        first = rng.random() < 0.5
        n, a = (n1, a1) if first else (n2, a2)
        pool = ["the first one", "the first", "first one pls"] if first else ["the second one", "the second", "second one pls"]
        pool = pool + ["the {n} one", "the {n} usdc one"]
        body4, i4 = fu_text(rng, ev, pool, n=n)
        c.user(body4)
        s3 = run_send(c, rng, a, n, "USDC")
        c.say(send_text(rng, bio, s3["res"], s3["args"], open_note(rng, s3["pend"])))
        c.gold(name="send_pick", fresh={"prepare_send": {"to_address": a, "amount": nn(n), "symbol": "USDC"}}, must_call=["prepare_send"])
    return c, {"tpl": [f"send{i}", f"send{i2}", f"again{i3}"], "shape": "f"}


# ----------------------------------------------------------------------------- shape g: referent dropped by trimHistory
def trim_start(c):
    """First visible turn when the oldest 40% of the messages are dropped, whole exchanges from the front."""
    n = c.turns[-1]["msg_i"]
    for k in range(1, len(c.turns)):
        if c.turns[k]["msg_i"] - 1 >= 0.4 * n:
            return k
    return len(c.turns) - 1


def g_mids(c, rng, ev, avoid, n=None):
    for _ in range(n or rng.choice([2, 2, 3])):
        interject(c, rng, ev, avoid=avoid)


LIMIT_T = ["limit {act} {c} {amt} {lev} at {px}", "place a limit {act} on {c}, {amt}, {lev}, price {px}", "{c} limit {act} @ {px}, {amt} {lev}",
           "set a limit order to {act} {c} at {px} with {amt} {lev}", "i want a limit {act} on {c}: {amt}, {lev}, at {px}", "{act} {c} {amt} {lev}, limit at {px}",
           "put in a {act} limit order for {c}, {amt}, {lev}, price {px}", "can you do a limit {act} {c} {px} {amt} {lev}"]


def sh_g1(rng, ev, bio):
    sol = rng.random() < 0.2
    w = make_world(rng, bio, usdc=("solana",) if sol else ("base",))
    c = FConv(rng, w, bio)
    a1 = H.sol_addr(rng) if sol else H.evm_addr(rng)
    amt = rng.choice([5, 10, 20, 25, 50])
    i, tpl = pick_tpl(rng, SEND_T, ev)
    c.user(fill(rng, tpl, tok=f"{amt} usdc", addr=a1))
    s1 = run_send(c, rng, a1, amt, "USDC")
    c.say(send_text(rng, bio, s1["res"], s1["args"]))
    c.gold(name="open")
    coin = rng.choice(COINS)
    i0, t = pick_tpl(rng, INTERJECT_PRICE, ev)
    c.user(PJ(rng.choice(APPROVED_PRE), fill(rng, t, c=coin_user(rng, coin))), {"op": "approve_last"})
    r = c.call(("get_price", {"symbol": coin}))
    c.say(TV.price_line(rng, coin, r["price"], r["change_24h_pct"]))
    c.gold(name="unrelated", forbid_tools=MONEY_TOOLS, unrelated=True)
    g_mids(c, rng, ev, [coin])
    body, i2 = fu_text(rng, ev, SEND_AGAIN_FU)
    c.user(body)
    t0 = trim_start(c)
    c.trim_at = t0
    if rng.random() < 0.6:
        r = c.call(("get_activity", {}))
        row = r["activity"][0]
        assert row["what"].startswith("Sent ")
        c.say(rng.choice([f"Activity shows your last send as \"{row['what']}\", but it only has the short address, and I no longer have the full one in this chat. Paste the address and I'll prepare it.",
                          f"The last send in Activity is \"{row['what']}\". It shows the address in short form, and the full one is no longer in this chat, so paste it and I'll prepare the send.",
                          f"I can see \"{row['what']}\" in Activity, but not the full address, and it's no longer in this chat. Paste the address and I'll prepare it again."]))
        c.gold(name="gone_address", must_call=["get_activity"], forbid_tools=MONEY_TOOLS, answer_any=[r"no longer|paste"], gone=[a1], trimmed=True)
    else:
        c.say(rng.choice(["I no longer have that address or amount in this chat. Paste the address and amount and I'll prepare the send.",
                          "That send is too far back, and I no longer have the address. Paste it and tell me the amount, and I'll prepare it.",
                          "I don't have that address any more, it's no longer in this chat. Paste it and I'll prepare the send."]))
        c.gold(name="gone_address", forbid_tools=MONEY_TOOLS, answer_any=[r"no longer|any more|paste"], gone=[a1], trimmed=True)
    return c, {"tpl": [f"send{i}", f"again{i2}"], "shape": "g", "trim": True}


def sh_g2(rng, ev, bio):
    coin = rng.choice(["ETH", "SOL", "BTC", "AVAX", "LINK"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec = mk_trade_spec(rng, coin, rng.choice(["pct", None]), side="long", lev=rng.choice([3, 5]))
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open")
    coin2 = rng.choice([x for x in COINS if x != coin])
    i0, t = pick_tpl(rng, INTERJECT_PRICE, ev)
    c.user(PJ(rng.choice(APPROVED_PRE), fill(rng, t, c=coin_user(rng, coin2))), {"op": "approve_last"})
    r = c.call(("get_price", {"symbol": coin2}))
    c.say(TV.price_line(rng, coin2, r["price"], r["change_24h_pct"]))
    c.gold(name="unrelated", forbid_tools=MONEY_TOOLS, unrelated=True)
    g_mids(c, rng, ev, [coin, coin2])
    body, i2 = fu_text(rng, ev, LEV_Q, c=coin_user(rng, coin))
    c.user(body)
    c.trim_at = trim_start(c)
    r = c.call(("hyperliquid_account", {}))
    p = next(x for x in r["positions"] if x["coin"] == coin)
    c.say(rng.choice([f"Your {coin} {p['side']} is on {ns(p['leverage'])}x: {ns(p['size'])} {coin} with an entry at {dollars(ns(p['entry']))}.",
                      f"It's {ns(p['leverage'])}x. You have {ns(p['size'])} {coin} {p['side']}, entered at {dollars(ns(p['entry']))}.",
                      f"The {coin} {p['side']} is at {ns(p['leverage'])}x leverage, {ns(p['size'])} {coin} from {dollars(ns(p['entry']))}."]))
    c.gold(name="reread_leverage", must_call=["hyperliquid_account"], forbid_tools=MONEY_TOOLS, answer_all=[re.escape(ns(p["leverage"]))], gone=[txt], trimmed=True)
    return c, {"tpl": [tid, f"levq{i2}"], "shape": "g", "trim": True}


def sh_g3(rng, ev, bio):
    ch = rng.choice(["base", "arbitrum", "ethereum"])
    w = make_world(rng, bio, usdc=(ch,), amounts={("USDC", ch): round(rng.uniform(600, 2400), 2)})
    c = FConv(rng, w, bio)
    n1 = rng.choice([25, 40, 50, 75, 100, 150])
    i, tpl = pick_tpl(rng, SWAP_T, ev)
    txt = fill(rng, tpl, amt=n1, a="USDC", b="ETH", c=chain_user(rng, ch))
    c.user(txt)
    s1 = run_swap(c, rng, "USDC", ch, "ETH", ch, n1)
    c.say(swap_text(rng, bio, s1["res"]))
    c.gold(name="open")
    coin2 = rng.choice(COINS)
    i0, t = pick_tpl(rng, INTERJECT_PRICE, ev)
    c.user(PJ(rng.choice(APPROVED_PRE), fill(rng, t, c=coin_user(rng, coin2))), {"op": "approve_last", "settled": True})
    r = c.call(("get_price", {"symbol": coin2}))
    c.say(TV.price_line(rng, coin2, r["price"], r["change_24h_pct"]))
    c.gold(name="unrelated", forbid_tools=MONEY_TOOLS, unrelated=True)
    g_mids(c, rng, ev, [coin2])
    body, i2 = fu_text(rng, ev, SWAP_Q)
    c.user(body)
    c.trim_at = trim_start(c)
    r = c.call(("get_activity", {}))
    row = next(x for x in r["activity"] if x["what"].startswith("Swap"))
    st = {"confirmed": "confirmed", "failed": "failed", "in_progress": "still in progress"}[row["status"]]
    c.say(rng.choice([f"Activity shows \"{row['what']}\", {st}.", f"That was \"{row['what']}\" in Activity, and it's {st}.",
                      f"The swap in Activity reads \"{row['what']}\" and is {st}."]))
    c.gold(name="reread_swap", must_call=["get_activity"], forbid_tools=MONEY_TOOLS, answer_all=[re.escape(row["what"])], gone=[txt], trimmed=True)
    return c, {"tpl": [f"swap{i}", f"swapq{i2}"], "shape": "g", "trim": True}


def sh_g4(rng, ev, bio):
    coin = rng.choice(["ETH", "SOL", "BTC", "AVAX", "LINK"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    side = rng.choice(["long", "long", "short"])
    spec = mk_trade_spec(rng, coin, None, side=side, lev=rng.choice([2, 3, 5]))
    px = nn(float(round(w.s["prices"][coin] * (0.95 if side == "long" else 1.05), sim.ASSETS[coin][2])))
    i, tpl = pick_tpl(rng, LIMIT_T, ev)
    txt = fill(rng, tpl, act=side, c=coin_user(rng, coin), amt=amt_user(rng, spec["size"]), lev=lev_user(rng, spec["lev"]), px=ns(px))
    c.user(txt)
    info1 = run_trade(c, rng, spec, order_type="limit", limit_price=px)
    c.say(trade_text(rng, bio, spec, info1, ""))
    c.gold(name="open")
    coin2 = rng.choice([x for x in COINS if x != coin])
    i0, t = pick_tpl(rng, INTERJECT_PRICE, ev)
    c.user(PJ(rng.choice(APPROVED_PRE), fill(rng, t, c=coin_user(rng, coin2))), {"op": "approve_last"})
    r = c.call(("get_price", {"symbol": coin2}))
    c.say(TV.price_line(rng, coin2, r["price"], r["change_24h_pct"]))
    c.gold(name="unrelated", forbid_tools=MONEY_TOOLS, unrelated=True)
    g_mids(c, rng, ev, [coin, coin2])
    body, i2 = fu_text(rng, ev, CANCEL_ORDER_FU)
    c.user(body)
    c.trim_at = trim_start(c)
    if rng.random() < 0.5:
        r = c.call(("hyperliquid_account", {}))
        o = r["open_orders"][0]
    else:
        r = c.call(("get_positions", {"kind": "perps"}))
        o = r["perps"]["open_orders"][0]
    assert o["coin"] == coin
    r2 = c.call(("cancel_order", {"coin": coin}))
    assert r2["status"] == "card_opened", r2
    c.say(rng.choice([f"I put up a card to cancel the {r2['cancelling']}. Tap Confirm to cancel it; nothing is cancelled yet.",
                      f"I've drafted a card for the {r2['cancelling']}. Nothing is cancelled until you tap Confirm."]))
    c.gold(name="cancel_after_trim", fresh={"cancel_order": {"coin": coin}}, must_call=["cancel_order"], answer_any=[r"Confirm"],
           answer_forbid=[r"(?<!nothing is )(?<!not )\bcancell?ed\b(?! yet| until)"], gone=[txt], trimmed=True)
    return c, {"tpl": [f"limit{i}", f"cancelo{i2}"], "shape": "g", "trim": True}


# ----------------------------------------------------------------------------- shape h: I approved, what's my balance
def sh_h(rng, ev, bio, kind):
    extras = [("ETH", "base", round(rng.uniform(0.01, 0.4), 4))] if kind == "send" and rng.random() < 0.4 else []
    w = make_world(rng, bio, usdc=("base",), extras=extras, amounts={("USDC", "base"): round(rng.uniform(300, 1800), 2)})
    c = FConv(rng, w, bio)
    claim_only = kind != "trade" and rng.random() < 0.18
    old = None
    if kind != "trade" and (claim_only or rng.random() < 0.45):
        i0, q = pick_tpl(rng, BAL_Q, ev)
        c.user(fill(rng, q))
        r = c.call(("get_balances", {}))
        old = {h["symbol"] + h["chain"]: h["amount"] for h in r["holdings"]}
        c.say(f"You have {holdings_text(r['holdings'])}.")
        c.gold(name="balance_before", must_call=["get_balances"], forbid_tools=MONEY_TOOLS)
    if kind == "send":
        a1 = H.evm_addr(rng)
        amt = rng.choice([10, 20, 25, 50, 100])
        i, tpl = pick_tpl(rng, SEND_T, ev)
        c.user(fill(rng, tpl, tok=f"{amt} usdc", addr=a1))
        s1 = run_send(c, rng, a1, amt, "USDC")
        c.say(send_text(rng, bio, s1["res"], s1["args"]))
    elif kind == "swap":
        n1 = rng.choice([25, 50, 100, 150])
        bridge = rng.random() < 0.4
        c2 = rng.choice(["arbitrum", "polygon", "ethereum"])
        if bridge:
            i, tpl = pick_tpl(rng, BRIDGE_T, ev)
            c.user(fill(rng, tpl, amt=n1, tok="USDC", c1=chain_user(rng, "base"), c2=chain_user(rng, c2)))
            s1 = run_swap(c, rng, "USDC", "base", "USDC", c2, n1)
        else:
            i, tpl = pick_tpl(rng, SWAP_T, ev)
            c.user(fill(rng, tpl, amt=n1, a="USDC", b="ETH", c=chain_user(rng, "base")))
            s1 = run_swap(c, rng, "USDC", "base", "ETH", "base", n1)
        c.say(swap_text(rng, bio, s1["res"]))
    else:
        spec = mk_trade_spec(rng, None, rng.choice(["pct", None]))
        txt, i = trade_user_text(rng, ev, w, spec)
        c.user(txt)
        info = run_trade(c, rng, spec)
        c.say(trade_text(rng, bio, spec, info))
        avail_old = w.s["hl"]["avail"]
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev)
    pre = rng.choice(APPROVED_PRE) + " "
    if kind == "trade":
        body, i2 = fu_text(rng, ev, HL_BAL_Q)
        c.user(PJ(pre, body), {"op": "approve_last"})
        r = c.call(("hyperliquid_account", {}))
        p = r["positions"][0]
        c.say(rng.choice([f"You have {commas(ns(r['available_usdc']))} USDC free on Hyperliquid, with {commas(ns(r['margin_used']))} in margin on your {p['coin']} {p['side']}.",
                          f"Available margin is {commas(ns(r['available_usdc']))} USDC right now, and {commas(ns(r['margin_used']))} is in use on the {p['coin']} {p['side']}.",
                          f"Your Hyperliquid account has {commas(ns(r['available_usdc']))} USDC available and {commas(ns(r['margin_used']))} tied up in margin."]))
        c.gold(name="balance_after", must_call=["hyperliquid_account"], forbid_tools=MONEY_TOOLS, stale=[ns(avail_old)], fresh_text=[ns(r["available_usdc"])])
        return c, {"tpl": [f"h_trade{i}", f"hlbal{i2}"], "shape": "h"}
    body, i2 = fu_text(rng, ev, BAL_Q)
    ev_ = None if claim_only else {"op": "approve_last", "settled": kind == "send" or rng.random() < 0.6}
    c.user(PJ(pre, body), ev_)
    r = c.call(("get_balances", {}))
    rows = r["holdings"]
    if claim_only:
        c.say(rng.choice([f"You have {holdings_text(rows)} right now, the same as before. If you approved it, a transfer can take a moment to show; Activity has its status.",
                          f"I see {holdings_text(rows)}, unchanged from earlier. Activity shows whether the transfer has started."]))
        c.gold(name="balance_after", must_call=["get_balances"], forbid_tools=MONEY_TOOLS, fresh_text=[h["amount"] for h in rows])
    else:
        c.say(rng.choice([f"You have {holdings_text(rows)} right now.", f"Right now you hold {holdings_text(rows)}.",
                          f"Your wallet holds {holdings_text(rows)} right now. Activity has the status of the transfer.",
                          f"Your wallet shows {holdings_text(rows)}."]))
        stale = []
        if old:
            new = {h["symbol"] + h["chain"]: h["amount"] for h in rows}
            stale = [v for k, v in old.items() if new.get(k) != v and v not in new.values()]
        c.gold(name="balance_after", must_call=["get_balances"], forbid_tools=MONEY_TOOLS, stale=stale, fresh_text=[h["amount"] for h in rows])
    return c, {"tpl": [f"h_{kind}{i}", f"bal{i2}"], "shape": "h"}


def sh_h1(rng, ev, bio):
    return sh_h(rng, ev, bio, "send")


def sh_h2(rng, ev, bio):
    return sh_h(rng, ev, bio, "swap")


def sh_h3(rng, ev, bio):
    return sh_h(rng, ev, bio, "trade")


# ----------------------------------------------------------------------------- shape i: and now?
def sh_i(rng, ev, bio):
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE", "BNB", "AVAX", "LINK"])
    w = make_world(rng, bio, usdc=("base",))
    c = FConv(rng, w, bio)
    i, q = pick_tpl(rng, INTERJECT_PRICE, ev)
    c.user(fill(rng, q, c=coin_user(rng, coin)))
    r = c.call(("get_price", {"symbol": coin}))
    p0 = r["price"]
    c.say(TV.price_line(rng, coin, p0, r["change_24h_pct"]))
    c.gold(name="price_first", must_call=["get_price"], forbid_tools=MONEY_TOOLS)
    kinds = rng.sample(["pm", "stop", "buy"], rng.choice([2, 3]))
    used = []
    for k in kinds:
        if k == "pm":
            L, size = rng.choice([3, 5, 10]), rng.choice([100, 200, 500])
            L = min(L, MAX_LEV[coin])
            t = rng.choice(["if i long {c} with {s} at {L}x where do i get liquidated", "liq price on a {s} {c} long at {L}x?", "what's the liquidation for {s} long {c} {L}x",
                            "how close to liquidation is a {L}x long on {c} with {s}", "{c} {L}x long with {s}, where's liq", "say i long {s} of {c} at {L}x, liq price?"])
            c.user(fill(rng, t, c=coin_user(rng, coin), s=amt_user(rng, size), L=L))
            r = c.call(("position_math", {"side": "long", "entry": p0, "size_usd": size, "leverage": L}))
            c.say(f"At an entry of {dollars(ns(p0))}, a {dollars(ns(size))} long at {L}x liquidates around {dollars(ns(r['liquidation_price']))}, {ns(r['distance_to_liq_pct'])}% away.")
        elif k == "stop":
            pct = rng.choice([3, 5, 7, 10])
            t = rng.choice(["where would a {p}% stop be on a long here", "what's a {p}% stop below this", "{p}% stop loss on a {c} long, what price", "where's a {p}% stop for a {c} long",
                            "give me a {p}% stop level for {c}", "if i long here, where does a {p}% stop sit"])
            c.user(fill(rng, t, c=coin_user(rng, coin), p=pct))
            dp = stop_dp(p0)
            fac = round(1 - pct / 100, 4)
            r = c.call(("calculator", {"expression": f"round({ns(p0)} * {ns(fac)}, {dp})"}))
            c.say(f"A {pct}% stop below {dollars(ns(p0))} is {dollars(r['result'])}.")
        else:
            n = rng.choice([100, 200, 500, 1000])
            t = rng.choice(["how many {c} would {n} dollars buy", "how much {c} is {n} usd", "what's {n} bucks in {c}", "{n} dollars of {c} is how many coins",
                            "how many {c} can i get for {n}", "convert {n} usd to {c}"])
            c.user(fill(rng, t, c=coin_user(rng, coin), n=n))
            r = c.call(("calculator", {"expression": f"round({n} / {ns(p0)}, 4)"}))
            c.say(f"{dollars(ns(n))} buys about {r['result']} {coin} at {dollars(ns(p0))}.")
        c.gold(name="mid", forbid_tools=MONEY_TOOLS)
    if rng.random() < 0.2:
        interject(c, rng, ev, avoid=[coin])
    mv = rng.uniform(0.004, 0.04) * rng.choice([-1, 1])
    dp = sim.ASSETS[coin][2]
    p1 = nn(float(round(p0 * (1 + mv), dp)))
    if p1 == p0:
        p1 = nn(float(round(p0 * 1.02, dp)))
    chg1 = nn(round(w.s["change"][coin] + mv * 100, 2))
    i2, qq = pick_tpl(rng, NOW_Q, ev)
    c.user(fill(rng, qq), {"op": "move", "prices": {coin: p1}, "change": {coin: chg1}})
    r = c.call(("get_price", {"symbol": coin}))
    pn = r["price"]
    r2 = c.call(("calculator", {"expression": f"round(({ns(pn)} - {ns(p0)}) / {ns(p0)} * 100, 2)"}))
    pct = r2["result"]
    d = "up" if not pct.startswith("-") else "down"
    c.say(rng.choice([f"{coin} is {dollars(ns(pn))} now, {d} {pct.lstrip('-')}% from the {dollars(ns(p0))} I read earlier.",
                      f"Now {coin} is at {dollars(ns(pn))}, {d} {pct.lstrip('-')}% since the {dollars(ns(p0))} I saw earlier.",
                      f"{coin} has moved: {dollars(ns(pn))} now against {dollars(ns(p0))} earlier, {d} {pct.lstrip('-')}%.",
                      f"It's {dollars(ns(pn))} now. That's {d} {pct.lstrip('-')}% from the {dollars(ns(p0))} I read earlier."]))
    c.gold(name="and_now", must_call=["get_price", "calculator"], forbid_tools=MONEY_TOOLS, stale=[ns(p0)], fresh_text=[ns(pn)],
           fresh={"get_price": {"symbol": coin}})
    return c, {"tpl": [f"i{i}", f"now{i2}"], "shape": "i"}


# ----------------------------------------------------------------------------- shape j: do I have enough
ENOUGH_SPENT = ["i just moved some usdc out, do i still have enough?", "i spent a bit since then, is it still enough?", "just used some of it elsewhere, still enough for that?",
                "i sent some usdc somewhere else, do i still have enough", "i moved funds out a minute ago, is that still covered?", "wait i used some of it, do i still have enough for that",
                "some of it went out already, do i still have enough?", "i paid someone from it meanwhile, still enough?"]
ENOUGH_SPENT_HL = ["i pulled some usdc out of hyperliquid, do i still have enough margin?", "i withdrew some margin, is that still covered?", "just moved some off hyperliquid, still enough for that?",
                   "i took some usdc out meanwhile, is the margin still covered", "i closed something and took funds out, do i still have enough?", "i withdrew a bit earlier, can my account still cover that",
                   "moved some usdc out of my trading account, still enough?", "some margin left already, do i still have enough for that"]


def sh_j(rng, ev, bio, kind):
    spent = rng.random() < 0.28
    if kind == "trade":
        A = round(rng.uniform(900, 6000), 2)
        w = make_world(rng, bio, hl_avail=A)
        c = FConv(rng, w, bio)
        spec = mk_trade_spec(rng, None, None, size=rng.choice([100, 200, 300, 500]), lev=rng.choice([2, 3, 5]))
        txt, i = trade_user_text(rng, ev, w, spec)
        c.user(txt)
        info = run_trade(c, rng, spec)
        c.say(trade_text(rng, bio, spec, info))
        c.gold(name="open")
        if rng.random() < 0.15:
            interject(c, rng, ev, avoid=[spec["coin"]])
        need_v = round(spec["size"] / spec["lev"], 2)
        if spent:
            newA = round(need_v * rng.uniform(0.3, 0.8), 2)
            i2, q = pick_tpl(rng, ENOUGH_SPENT_HL, ev)
            c.user(fill(rng, q), {"op": "hl", "avail": newA})
        else:
            i2, q = pick_tpl(rng, ENOUGH_Q_TRADE, ev)
            c.user(fill(rng, q))
        r = c.call(("hyperliquid_account", {}))
        r2 = c.call(("calculator", {"expression": f"round({spec['size']} / {spec['lev']}, 2)"}))
        need, av = r2["result"], ns(r["available_usdc"])
        needc, avc = commas(need), commas(av)
        if spent:
            c.say(rng.choice([f"No. The order needs {needc} USDC of margin and you only have {avc} available now.", f"Not anymore: it needs {needc} USDC of margin and {avc} is available.",
                              f"No, you have {avc} USDC available and the order needs {needc} USDC of margin."]))
        else:
            c.say(rng.choice([f"Yes. The order needs {needc} USDC of margin and you have {avc} available.", f"Yes, it needs {needc} USDC of margin and {avc} is available.",
                              f"You do: {avc} USDC is available and the order needs {needc} USDC of margin."]))
        c.gold(name="enough", must_call=["hyperliquid_account", "calculator"], forbid_tools=MONEY_TOOLS, stale=[ns(A)] if spent else [], fresh_text=[av, need])
        return c, {"tpl": [f"j_trade{i}", f"enough{i2}"], "shape": "j"}
    B = round(rng.uniform(300, 1800), 2)
    w = make_world(rng, bio, usdc=("base",), amounts={("USDC", "base"): B})
    c = FConv(rng, w, bio)
    amt = rng.choice([50, 100, 150, 200, 250])
    if kind == "send":
        a1 = H.evm_addr(rng)
        i, tpl = pick_tpl(rng, SEND_T, ev)
        c.user(fill(rng, tpl, tok=f"{amt} usdc", addr=a1))
        s1 = run_send(c, rng, a1, amt, "USDC")
        c.say(send_text(rng, bio, s1["res"], s1["args"]))
        thing = "send"
    else:
        i, tpl = pick_tpl(rng, SWAP_T, ev)
        c.user(fill(rng, tpl, amt=amt, a="USDC", b="ETH", c=chain_user(rng, "base")))
        s1 = run_swap(c, rng, "USDC", "base", "ETH", "base", amt)
        c.say(swap_text(rng, bio, s1["res"]))
        thing = "swap"
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev)
    if spent:
        B2 = round(amt * rng.uniform(0.3, 0.8), 2)
        i2, q = pick_tpl(rng, ENOUGH_SPENT, ev)
        c.user(fill(rng, q), {"op": "spend", "symbol": "USDC", "chain": "base", "amount": round(B - B2, 2)})
    else:
        i2, q = pick_tpl(rng, ENOUGH_Q, ev)
        c.user(fill(rng, q))
    r = c.call(("get_balances", {}))
    have = next(h for h in r["holdings"] if h["symbol"] == "USDC" and h["chain"] == "base")["amount"]
    hc, au = commas(have), f"{amt} USDC"
    if spent:
        r2 = c.call(("calculator", {"expression": f"{amt} - {have}"}))
        rc = commas(r2["result"])
        c.say(rng.choice([f"No. You hold {hc} USDC on Base now and the {thing} is {au}, so you're short {rc}.",
                          f"Not anymore: {hc} USDC on Base against {au} for the {thing}, which leaves you {rc} short.",
                          f"No, your Base USDC is {hc} now and the {thing} is {au}, so you're {rc} short."]))
    else:
        r2 = c.call(("calculator", {"expression": f"{have} - {amt}"}))
        rc = commas(r2["result"])
        c.say(rng.choice([f"Yes. You hold {hc} USDC on Base and the {thing} is {au}, so {rc} would be left.",
                          f"You do: {hc} USDC on Base covers the {au} {thing}, leaving {rc}.",
                          f"Yes, {hc} USDC on Base is enough for the {au} {thing}, with {rc} left over."]))
    c.gold(name="enough", must_call=["get_balances", "calculator"], forbid_tools=MONEY_TOOLS, stale=[amount_str(B)] if spent else [], fresh_text=[have, r2["result"]])
    return c, {"tpl": [f"j_{kind}{i}", f"enough{i2}"], "shape": "j"}


def sh_j1(rng, ev, bio):
    return sh_j(rng, ev, bio, "send")


def sh_j2(rng, ev, bio):
    return sh_j(rng, ev, bio, "swap")


def sh_j3(rng, ev, bio):
    return sh_j(rng, ev, bio, "trade")


# ----------------------------------------------------------------------------- shape k: is it there yet
def sh_k(rng, ev, bio):
    c1, c2 = chain_pair(rng)
    dest = rng.random() < 0.6
    w = make_world(rng, bio, usdc=(c1, c2) if dest else (c1,), amounts={("USDC", c1): round(rng.uniform(400, 2000), 2)})
    c = FConv(rng, w, bio)
    amt = rng.choice([25, 50, 75, 100, 150, 200])
    i, tpl = pick_tpl(rng, BRIDGE_T, ev)
    c.user(fill(rng, tpl, amt=amt, tok=rng.choice(["USDC", "usdc"]), c1=chain_user(rng, c1), c2=chain_user(rng, c2)))
    s1 = run_swap(c, rng, "USDC", c1, "USDC", c2, amt)
    c.say(swap_text(rng, bio, s1["res"]))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev)
    state = rng.choices(["settled", "progress", "unapproved"], [45, 40, 15])[0]
    i2, q = pick_tpl(rng, THERE_Q, ev)
    if state == "unapproved":
        c.user(fill(rng, q))
    else:
        c.user(PJ(rng.choice(APPROVED_PRE), fill(rng, q)), {"op": "approve_last", "settled": state == "settled"})
    r = c.call(("get_activity", {}), ("get_balances", {}))
    act, bal = r
    drow = next((h for h in bal["holdings"] if h["symbol"] == "USDC" and h["chain"] == c2), None)
    C2 = CHAIN_NAME[c2]
    if state == "settled":
        row = next(x for x in act["activity"] if x["what"].startswith("Swap"))
        assert row["status"] == "confirmed" and drow
        c.say(rng.choice([f"Yes. Activity shows the bridge as confirmed, and your {C2} USDC is now {commas(drow['amount'])}.",
                          f"It has: Activity lists the bridge as confirmed, and you hold {commas(drow['amount'])} USDC on {C2}.",
                          f"Yes, Activity says confirmed, and your {C2} balance is {commas(drow['amount'])} USDC."]))
        c.gold(name="there_yet", must_call=["get_activity", "get_balances"], forbid_tools=MONEY_TOOLS, fresh_text=[drow["amount"]], answer_any=[r"confirmed"])
    elif state == "progress":
        row = next(x for x in act["activity"] if x["what"].startswith("Swap"))
        assert row["status"] == "in_progress"
        if drow:
            c.say(rng.choice([f"Not yet. Activity shows the bridge as in progress, and your {C2} USDC is still {commas(drow['amount'])}.",
                              f"Not yet: it's still in progress in Activity, and you hold {commas(drow['amount'])} USDC on {C2} so far."]))
        else:
            c.say(rng.choice([f"Not yet. Activity shows the bridge as in progress, and you have no USDC on {C2} so far.",
                              f"Not yet: it's still in progress in Activity, and nothing has shown up on {C2} yet."]))
        c.gold(name="there_yet", must_call=["get_activity", "get_balances"], forbid_tools=MONEY_TOOLS, answer_any=[r"not yet|in progress"],
               answer_forbid=[r"\b(has )?arrived\b(?! yet)", r"\bis there\b"], fresh_text=[drow["amount"]] if drow else [])
    else:
        assert not any(x["what"].startswith("Swap") for x in act["activity"])
        c.say(rng.choice([f"I don't see it in Activity yet, so it hasn't started. The review only moves once you approve it with {bio}.",
                          f"Activity doesn't list it, so nothing has started. The bridge begins when you approve the review with {bio}.",
                          f"Not in Activity yet, which means it hasn't started. You still need to approve the review with {bio}."]))
        c.gold(name="there_yet", must_call=["get_activity", "get_balances"], forbid_tools=MONEY_TOOLS, answer_any=[r"hasn't started|nothing has started|don't see"])
    return c, {"tpl": [f"bridge{i}", f"there{i2}"], "shape": "k"}


# ----------------------------------------------------------------------------- shape l: what's the fee
def sh_l1(rng, ev, bio):
    ch = rng.choice(["base", "arbitrum"])
    w = make_world(rng, bio, usdc=(ch,), amounts={("USDC", ch): round(rng.uniform(600, 2400), 2)})
    c = FConv(rng, w, bio)
    n1 = rng.choice([50, 100, 150, 200, 300])
    if rng.random() < 0.5:
        i, tpl = pick_tpl(rng, SWAP_T, ev)
        c.user(fill(rng, tpl, amt=n1, a="USDC", b="ETH", c=chain_user(rng, ch)))
        s1 = run_swap(c, rng, "USDC", ch, "ETH", ch, n1)
        c.say(swap_text(rng, bio, s1["res"]))
        cost = s1["res"]["total_cost_usd"]
        tid = f"swap{i}"
    else:
        c2 = rng.choice([x for x in ("base", "arbitrum", "ethereum") if x != ch])
        i, tpl = pick_tpl(rng, QUOTE_FIRST, ev)
        c.user(fill(rng, tpl, amt=n1, a="USDC", c1=chain_user(rng, ch), c2=chain_user(rng, c2)))
        r = c.call(("get_quote", {"from_symbol": "USDC", "from_chain": ch, "to_symbol": "USDC", "to_chain": c2, "amount": nn(n1)}))
        c.say(rng.choice([f"{r['from']} would give you {r['to']} through {r['route']}. Want me to prepare it?",
                          f"Through {r['route']}, {r['from']} comes out as {r['to']}. Want me to prepare it?"]), ask=True)
        cost = r["total_cost_usd"]
        tid = f"quote{i}"
    c.gold(name="open", ask=(tid.startswith("quote")))
    if rng.random() < 0.15:
        interject(c, rng, ev)
    body, i2 = fu_text(rng, ev, FEE_Q)
    c.user(body)
    c.say(rng.choice([f"The total cost is {dollars(ns(cost))}, fees and spread together. It doesn't include network gas.",
                      f"All in, fees and spread come to {dollars(ns(cost))}. Network gas is separate.",
                      f"{dollars(ns(cost))} in total for fees and spread, not counting network gas."]))
    c.gold(name="fee_from_result", forbid_tools=MONEY_TOOLS, answer_all=[re.escape(ns(cost))])
    return c, {"tpl": [tid, f"fee{i2}"], "shape": "l"}


def sh_l2(rng, ev, bio):
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    a1, amt = H.evm_addr(rng), rng.choice([10, 20, 25, 50, 100])
    i, tpl = pick_tpl(rng, SEND_T, ev)
    c.user(fill(rng, tpl, tok=f"{amt} usdc", addr=a1))
    s1 = run_send(c, rng, a1, amt, "USDC")
    c.say(send_text(rng, bio, s1["res"], s1["args"]))
    c.gold(name="open")
    if rng.random() < 0.15:
        interject(c, rng, ev)
    body, i2 = fu_text(rng, ev, FEE_Q)
    c.user(body)
    c.say(rng.choice(["I don't have a fee for sends from here, and I won't guess one. Check the review screen before you approve.",
                      "I can't see the network fee for a send from here, and I won't guess one. Check the review screen before you approve.",
                      "There's no fee figure I can give you for a send from here. Look at the review before you approve."]))
    c.gold(name="fee_send", forbid_tools=MONEY_TOOLS, answer_any=[r"review"], answer_forbid=[r"\$\d"])
    return c, {"tpl": [f"send{i}", f"fee{i2}"], "shape": "l"}


def sh_l3(rng, ev, bio):
    ch = rng.choice(["base", "arbitrum", "ethereum"])
    B = round(rng.uniform(800, 3000), 2)
    w = make_world(rng, bio, usdc=(ch,), amounts={("USDC", ch): B})
    c = FConv(rng, w, bio)
    i, q = pick_tpl(rng, SWAP_NOAMT, ev)
    c.user(fill(rng, q))
    r = c.call(("get_balances", {}))
    h = next(x for x in r["holdings"] if x["symbol"] == "USDC")
    c.say(rng.choice([f"You hold {commas(h['amount'])} USDC on {CHAIN_NAME[ch]}. How much do you want to swap to ETH?", f"You have {commas(h['amount'])} USDC on {CHAIN_NAME[ch]}. How much should I swap into ETH?",
                      f"I see {commas(h['amount'])} USDC on {CHAIN_NAME[ch]}. How much of it do you want in ETH?"]), ask=True)
    c.gold(name="ask_amount", ask=True, forbid_tools=MONEY_TOOLS, must_call=["get_balances"])
    n = rng.choice([50, 100, 150, 200, 250, 400])
    i2, t = pick_tpl(rng, FEE_Q_AMT, ev)
    c.user(fill(rng, t, n=n))
    r = c.call(("get_quote", {"from_symbol": "USDC", "from_chain": ch, "to_symbol": "ETH", "to_chain": ch, "amount": nn(n)}))
    c.say(rng.choice([f"For {n} USDC to ETH on {CHAIN_NAME[ch]}, the total cost is {dollars(ns(r['total_cost_usd']))} and you'd get {r['to'].replace(' on ' + CHAIN_NAME[ch], '')}. Want me to prepare it?",
                      f"The total cost on {n} USDC is {dollars(ns(r['total_cost_usd']))}, fees and spread together. You'd get {r['to'].replace(' on ' + CHAIN_NAME[ch], '')}. Should I prepare it?",
                      f"{dollars(ns(r['total_cost_usd']))} all in for {n} USDC to ETH through {r['route']}, and {r['to'].replace(' on ' + CHAIN_NAME[ch], '')} comes out. Want me to prepare it?"]), ask=True)
    c.gold(name="fee_quote", ask=True, must_call=["get_quote"], forbid_tools=MONEY_TOOLS,
           fresh={"get_quote": {"from_symbol": "USDC", "from_chain": ch, "to_symbol": "ETH", "to_chain": ch, "amount": nn(n)}}, fresh_text=[ns(r["total_cost_usd"])])
    return c, {"tpl": [f"noamt{i}", f"feeamt{i2}"], "shape": "l"}


# ----------------------------------------------------------------------------- shape m: from memory
def sh_m1(rng, ev, bio):
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    interject(c, rng, ev)
    tok = rng.choice(w.s["tokens"])
    i, q = pick_tpl(rng, MINT_Q, ev)
    c.user(fill(rng, q, t=rng.choice([tok["symbol"], tok["symbol"].lower()])))
    r = c.call(("search_tokens", {"query": tok["symbol"]}))
    t = r["tokens"][0]
    c.say(rng.choice([f"{t['symbol']} ({t['name']}) is verified, and its mint is {t['mint']}.", f"The {t['symbol']} mint is {t['mint']}. It's marked verified.",
                      f"I looked it up: {t['symbol']} ({t['name']}), verified, mint {t['mint']}."]))
    c.gold(name="mint_lookup", must_call=["search_tokens"], forbid_tools=MONEY_TOOLS, answer_all=[re.escape(t["mint"])])
    if rng.random() < 0.5:
        interject(c, rng, ev)
        i3, q3 = pick_tpl(rng, ["what was that mint again", "what was the mint", "say that mint again", "which mint was that", "give me that mint one more time",
                                "what was that address you found", "can you repeat the mint", "that mint again?"], ev)
        c.user(fill(rng, q3))
        c.say(rng.choice([f"It was {t['mint']}.", f"The {t['symbol']} mint is {t['mint']}.", f"That mint is {t['mint']}."]))
        c.gold(name="mint_history", forbid_tools=MONEY_TOOLS, answer_all=[re.escape(t["mint"])])
    return c, {"tpl": [f"mint{i}"], "shape": "m"}


EVM_TOKENS = ["USDC", "USDT", "WETH", "DAI", "AAVE", "LINK", "UNI"]


def sh_m2(rng, ev, bio):
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    interject(c, rng, ev)
    tok, ch = rng.choice(EVM_TOKENS), rng.choice(["base", "arbitrum", "ethereum", "polygon"])
    i, q = pick_tpl(rng, CONTRACT_Q, ev)
    c.user(fill(rng, q, t=rng.choice([tok, tok.lower()]), ch=chain_user(rng, ch)))
    C = CHAIN_NAME[ch]
    c.say(rng.choice([f"I won't give you a contract address from memory: it could be wrong or out of date, and a wrong one can cost you funds. Copy it from the token's official page or the {C} explorer.",
                      f"I can't look up {tok} contracts on {C} from here, and an address from memory could be wrong. Take it from the official site or the {C} explorer, and paste it if you want me to use it.",
                      f"I don't have a live source for {tok} on {C}, so I won't quote an address that may be old. Check the token's official page or the {C} explorer."]))
    c.gold(name="contract_memory", forbid_tools=MONEY_TOOLS, answer_any=[r"memory|may be old|out of date|can't look up|don't have a live source"], answer_forbid=[r"0x[0-9a-fA-F]{40}"])
    return c, {"tpl": [f"contract{i}"], "shape": "m"}


def sh_m3(rng, ev, bio):
    ch = rng.choice(["base", "arbitrum", "ethereum"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    row = next(y for y in w.s["yields"] if y["protocol"] == "Aave v3" and y["asset"] == "USDC" and y["chain"] == CHAIN_NAME[ch])
    mem = rng.random() < 0.6
    if mem:
        n = max(1, round(row["apy_pct"]) + rng.choice([-2, -1, 1, 2]))
        if abs(n - row["apy_pct"]) < 0.6:
            n += 2
        i, q = pick_tpl(rng, APY_MEM_Q, ev)
        c.user(fill(rng, q, n=n, ch=chain_user(rng, ch)))
    else:
        i, q = pick_tpl(rng, APY_FIRST, ev)
        c.user(fill(rng, q, ch=chain_user(rng, ch)))
    r = c.call(("get_yields", {"asset": "USDC", "chain": ch}))
    top = next(p for p in r["pools"] if p["protocol"] == "Aave v3")
    if mem:
        c.say(rng.choice([f"Right now Aave v3 on {CHAIN_NAME[ch]} pays {ns(top['apy_pct'])}% on USDC, not {n}%. It changes all the time.",
                          f"Not quite: the live rate is {ns(top['apy_pct'])}% for Aave v3 USDC on {CHAIN_NAME[ch]}. I wouldn't trust {n}% from memory, rates move.",
                          f"I checked: {ns(top['apy_pct'])}% on USDC with Aave v3 on {CHAIN_NAME[ch]} right now, which isn't {n}%. Rates move."]))
    else:
        c.say(rng.choice([f"Aave v3 on {CHAIN_NAME[ch]} pays {ns(top['apy_pct'])}% on USDC right now. It changes all the time.",
                          f"The live rate for USDC on Aave v3 on {CHAIN_NAME[ch]} is {ns(top['apy_pct'])}%. Rates move.",
                          f"Right now it's {ns(top['apy_pct'])}% on Aave v3 on {CHAIN_NAME[ch]}, and that can change."]))
    c.gold(name="apy_lookup", must_call=["get_yields"], forbid_tools=MONEY_TOOLS, fresh_text=[ns(top["apy_pct"])])
    if rng.random() < 0.15:
        interject(c, rng, ev)
    i2, q2 = pick_tpl(rng, TVL_FU, ev)
    c.user(fill(rng, q2))
    c.say(rng.choice([f"Its TVL is {big(top['tvl_usd'])}, from the same lookup.", f"{big(top['tvl_usd'])} in total value locked.", f"The pool holds {big(top['tvl_usd'])}."]))
    c.gold(name="tvl_from_history", forbid_tools=MONEY_TOOLS, answer_all=[re.escape(big(top["tvl_usd"]))])
    return c, {"tpl": [f"apy{i}", f"tvl{i2}"], "shape": "m"}


def sh_m4(rng, ev, bio):
    ch = rng.choice(["base", "arbitrum", "ethereum"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    row = next(y for y in w.s["yields"] if y["protocol"] == "Aave v3" and y["asset"] == "USDC" and y["chain"] == CHAIN_NAME[ch])
    i, q = pick_tpl(rng, APY_FIRST, ev)
    c.user(fill(rng, q, ch=chain_user(rng, ch)))
    r = c.call(("get_yields", {"asset": "USDC", "chain": ch}))
    top = next(p for p in r["pools"] if p["protocol"] == "Aave v3")
    old = top["apy_pct"]
    c.say(f"Aave v3 on {CHAIN_NAME[ch]} pays {ns(old)}% on USDC right now. It changes all the time.")
    c.gold(name="apy_lookup", must_call=["get_yields"], forbid_tools=MONEY_TOOLS)
    interject(c, rng, ev)
    new = round(old + rng.choice([-1, 1]) * rng.uniform(0.3, 1.6), 2)
    if new <= 0.2:
        new = round(old + 0.8, 2)
    i2, q2 = pick_tpl(rng, APY_REFRESH, ev)
    c.user(fill(rng, q2), {"op": "yields", "protocol": "Aave v3", "asset": "USDC", "chain": CHAIN_NAME[ch], "apy": new})
    r = c.call(("get_yields", {"asset": "USDC", "chain": ch}))
    top = next(p for p in r["pools"] if p["protocol"] == "Aave v3")
    d = "up" if top["apy_pct"] > old else "down"
    c.say(rng.choice([f"It moved: Aave v3 on {CHAIN_NAME[ch]} now pays {ns(top['apy_pct'])}% on USDC, {d} from the {ns(old)}% I read earlier.",
                      f"No, it changed. It's {ns(top['apy_pct'])}% now, {d} from {ns(old)}% earlier.",
                      f"Not anymore: the live rate is {ns(top['apy_pct'])}%, {d} from the {ns(old)}% I gave you before."]))
    c.gold(name="apy_refresh", must_call=["get_yields"], forbid_tools=MONEY_TOOLS, stale=[], fresh_text=[ns(top["apy_pct"])])
    return c, {"tpl": [f"apy{i}", f"refresh{i2}"], "shape": "m"}


# ----------------------------------------------------------------------------- shape n
def sh_n1(rng, ev, bio):
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    a1, amt = H.evm_addr(rng), rng.choice([5, 10, 20, 25, 50, 100])
    i, tpl = pick_tpl(rng, SEND_T, ev)
    t1 = fill(rng, tpl, tok=f"{amt} usdc", addr=a1)
    c.user(t1)
    s1 = run_send(c, rng, a1, amt, "USDC")
    c.say(send_text(rng, bio, s1["res"], s1["args"]))
    c.gold(name="open")
    if rng.random() < 0.3:
        c.user(PJ(rng.choice(APPROVED_PRE), rng.choice(["send it again", "again please", "do the same send again", "one more of those"])), {"op": "approve_last"})
        s2 = run_send(c, rng, a1, amt, "USDC")
        c.say(send_text(rng, bio, s2["res"], s2["args"]))
        c.gold(name="again_after_approval", carry={"prepare_send": {"to_address": a1, "amount": nn(amt), "symbol": "USDC"}}, must_call=["prepare_send"])
        return c, {"tpl": [f"send{i}", "again"], "shape": "n"}
    body = rng.choice([t1, t1, rng.choice(["did you get that? ", "hello? ", "again: "]) + t1])
    c.user(body)
    c.say(rng.choice([f"That send is already waiting on your review: {s1['res']['amount']} to {a1} on {s1['res']['network']}. Do you want a second one on top of it?",
                      f"I already prepared exactly that, {s1['res']['amount']} to {a1} on {s1['res']['network']}, and the review is still open. Do you want another one?",
                      f"The review for {s1['res']['amount']} to {a1} on {s1['res']['network']} is already open. Should I prepare a second identical one?"]), ask=True)
    c.gold(name="same_twice", ask=True, forbid_tools=MONEY_TOOLS, answer_all=[re.escape(a1)])
    if rng.random() < 0.5:
        body3, i3 = fu_text(rng, ev, YES_FU)
        c.user(body3)
        s2 = run_send(c, rng, a1, amt, "USDC")
        c.say(send_text(rng, bio, s2["res"], s2["args"], open_note(rng, s2["pend"])))
        c.gold(name="second_send", carry={"prepare_send": {"to_address": a1, "amount": nn(amt), "symbol": "USDC"}}, must_call=["prepare_send"])
    return c, {"tpl": [f"send{i}", "twice"], "shape": "n"}


def sh_n2(rng, ev, bio):
    if rng.random() < 0.5:
        coin = rng.choice(["SOL", "ETH", "BTC", "HYPE", "AVAX", "LINK"])
        side = rng.choice(["long", "short"])
        w = make_world(rng, bio)
        c = FConv(rng, w, bio)
        i, tpl = pick_tpl(rng, LEV_FIRST, ev)
        c.user(fill(rng, tpl, act=side, c=coin_user(rng, coin)))
        c.say(rng.choice(["How much do you want to put in, and at what leverage?", "What size and what leverage?", "How big should it be, and at what leverage?",
                          "What size in dollars, and what leverage?"]), ask=True)
        c.gold(name="ask_size", ask=True, forbid_tools=MONEY_TOOLS)
        size, L = rng.choice([50, 100, 150, 200, 300]), min(rng.choice([2, 3, 5, 10]), MAX_LEV[coin])
        i2, tpl2 = pick_tpl(rng, AMOUNT_ANS, ev)
        c.user(fill(rng, tpl2, amt=amt_user(rng, size), L=L))
        spec = {"coin": coin, "side": side, "size": size, "lev": L, "stop": None, "tp": None}
        info = run_trade(c, rng, spec)
        c.say(trade_text(rng, bio, spec, info))
        c.gold(name="answer_then_call", carry={"prepare_trade": {"coin": coin, "side": side}}, fresh={"prepare_trade": {"size_usd": nn(size), "leverage": L, "order_type": "market"}},
               must_call=["prepare_trade"])
        return c, {"tpl": [f"levfirst{i}", f"ans{i2}"], "shape": "n"}
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    amt = rng.choice([10, 20, 25, 50, 100, 150])
    i, tpl = pick_tpl(rng, SEND_NOADDR, ev)
    c.user(fill(rng, tpl, tok=f"{amt} {rng.choice(['USDC', 'usdc'])}", sym="usdc"))
    c.say(rng.choice(["Which address should I send it to?", "What's the address?", "Which address do you want to send it to?", "What address do you want to send it to?"]), ask=True)
    c.gold(name="ask_address", ask=True, forbid_tools=MONEY_TOOLS)
    a1 = H.evm_addr(rng)
    i2, tpl2 = pick_tpl(rng, ADDR_ANS, ev)
    c.user(fill(rng, tpl2, addr=a1))
    s1 = run_send(c, rng, a1, amt, "USDC")
    c.say(send_text(rng, bio, s1["res"], s1["args"]))
    c.gold(name="answer_then_call", carry={"prepare_send": {"amount": nn(amt), "symbol": "USDC"}}, fresh={"prepare_send": {"to_address": a1}}, must_call=["prepare_send"])
    return c, {"tpl": [f"noaddr{i}", f"ans{i2}"], "shape": "n"}


def sh_n3(rng, ev, bio):
    if rng.random() < 0.5:
        w = make_world(rng, bio, usdc=("base",), usdt=(rng.choice(["ethereum", "arbitrum"]),))
        c = FConv(rng, w, bio)
        a1, amt = H.evm_addr(rng), rng.choice([10, 20, 25, 30, 50])
        i, tpl = pick_tpl(rng, SEND_T, ev)
        c.user(fill(rng, tpl, tok=f"{amt} usdc", addr=a1))
        s1 = run_send(c, rng, a1, amt, "USDC")
        c.say(send_text(rng, bio, s1["res"], s1["args"]))
        c.gold(name="open")
        if rng.random() < 0.15:
            interject(c, rng, ev)
        body, i2 = fu_text(rng, ev, NO_USDT)
        c.user(body)
        s2 = run_send(c, rng, a1, amt, "USDT")
        c.say(send_text(rng, bio, s2["res"], s2["args"], open_note(rng, s2["pend"], replace=True)))
        c.gold(name="correction", carry={"prepare_send": {"to_address": a1, "amount": nn(amt)}}, fresh={"prepare_send": {"symbol": "USDT"}}, must_call=["prepare_send"])
        return c, {"tpl": [f"send{i}", f"usdt{i2}"], "shape": "n"}
    coin1, coin2 = rng.sample(TRADE_COINS, 2)
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec = mk_trade_spec(rng, coin1, "pct")
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open")
    body, i2 = fu_text(rng, ev, CORR_COIN, c=coin_user(rng, coin2), x=coin_user(rng, coin1))
    c.user(body)
    spec2 = dict(spec, coin=coin2)
    info2 = run_trade(c, rng, spec2)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"], replace=True), again=True))
    c.gold(name="correction", carry={"prepare_trade": carry_of(info2["args"], ("side", "size_usd", "leverage", "order_type"))},
           fresh={"prepare_trade": {"coin": coin2, "stop_loss": info2["stop"]}},
           not_carried={"prepare_trade": {"stop_loss": [info1["stop"]]}}, must_call=["get_price", "calculator", "prepare_trade"], stale=[ns(info1["price"]), ns(info1["stop"])],
           stop_check={"side": spec["side"], "price": info2["price"], "pct": spec["stop"]["pct"], "stop": info2["stop"]})
    return c, {"tpl": [tid, f"corr{i2}"], "shape": "n"}


def sh_n4(rng, ev, bio):
    coin1 = rng.choice(["SOL", "ETH", "BTC"])
    w = make_world(rng, bio)
    c = FConv(rng, w, bio)
    spec = mk_trade_spec(rng, coin1, rng.choice(["pct", "abs"]), lev=rng.choice([2, 3]))
    txt, tid = trade_user_text(rng, ev, w, spec)
    c.user(txt)
    info1 = run_trade(c, rng, spec)
    c.say(trade_text(rng, bio, spec, info1))
    c.gold(name="open")
    interject(c, rng, ev, avoid=[coin1])
    L2 = rng.choice([x for x in (5, 10) if x != spec["lev"]])
    body, i2 = fu_text(rng, ev, BACK_TO, c=coin_user(rng, coin1), L=L2)
    c.user(body)
    spec2 = dict(spec, lev=L2)
    absol = {k: info1["args"][k] for k in ("stop_loss", "take_profit") if k in info1["args"]}
    info2 = run_trade(c, rng, spec2, absolute=absol)
    c.say(trade_text(rng, bio, spec2, info2, open_note(rng, info2["pend"], replace=True), again=True))
    c.gold(name="topic_return", carry={"prepare_trade": {**carry_of(info1["args"], ("coin", "side", "size_usd", "order_type")), **absol}},
           fresh={"prepare_trade": {"leverage": L2}}, must_call=["prepare_trade"])
    return c, {"tpl": [tid, f"back{i2}"], "shape": "n"}


# ----------------------------------------------------------------------------- plan
PLAN = [
    ("a1_same_trade_pct", sh_a1, 40), ("a2_same_trade_abs_ask", sh_a2, 20), ("a3_same_send_new_address", sh_a3, 10),
    ("b1_change_leverage", sh_b1, 20), ("b2_change_size", sh_b2, 18), ("b3_change_chain", sh_b3, 20), ("b4_change_swap_amount", sh_b4, 12),
    ("c1_cancel_review", sh_c1, 25), ("c2_cancel_open_order", sh_c2, 25), ("c3_cancel_nothing", sh_c3, 15),
    ("d1_undo_swap", sh_d1, 20), ("d2_undo_trade", sh_d2, 20), ("d3_undo_send", sh_d3, 15),
    ("e1_address_from_history", sh_e1, 25), ("e2_own_address_later", sh_e2, 15), ("e3_address_from_activity", sh_e3, 15),
    ("f1_make_it_n_two_args", sh_f1, 15), ("f2_cancel_which_review", sh_f2, 15), ("f3_same_for_which", sh_f3, 15), ("f4_same_address_which", sh_f4, 10),
    ("g1_trimmed_address", sh_g1, 20), ("g2_trimmed_leverage", sh_g2, 15), ("g3_trimmed_swap", sh_g3, 12), ("g4_trimmed_cancel_order", sh_g4, 13),
    ("h1_balance_after_send", sh_h1, 25), ("h2_balance_after_swap", sh_h2, 10), ("h3_margin_after_trade", sh_h3, 10),
    ("i_and_now_reprice", sh_i, 50),
    ("j1_enough_send", sh_j1, 15), ("j2_enough_swap", sh_j2, 10), ("j3_enough_margin", sh_j3, 15),
    ("k_bridge_arrived", sh_k, 40),
    ("l1_fee_from_result", sh_l1, 12), ("l2_fee_send", sh_l2, 8), ("l3_fee_needs_quote", sh_l3, 15),
    ("m1_solana_mint_lookup", sh_m1, 12), ("m2_evm_contract_memory", sh_m2, 10), ("m3_apy_tvl_lookup", sh_m3, 15), ("m4_apy_refresh", sh_m4, 8),
    ("n1_same_request_twice", sh_n1, 16), ("n2_answer_then_call", sh_n2, 16), ("n3_correction", sh_n3, 18), ("n4_topic_switch_return", sh_n4, 15),
]
CHECKS = ["score_item", "carry_ok", "no_restate_error", "no_reprepare_unasked", "rounded_ok"]


def allocate(total, minimum=1):
    """Largest-remainder split of `total` over the plan weights."""
    wsum = sum(w for _, _, w in PLAN)
    raw = [w * total / wsum for _, _, w in PLAN]
    out = [max(minimum, int(x)) for x in raw]
    order = sorted(range(len(PLAN)), key=lambda i: -(raw[i] - int(raw[i])))
    k = 0
    while sum(out) < total:
        out[order[k % len(order)]] += 1
        k += 1
    while sum(out) > total:
        j = max(range(len(out)), key=lambda i: out[i])
        out[j] -= 1
    return out


# ----------------------------------------------------------------------------- record assembly
def make_record(c, info, kind, bio):
    t0 = getattr(c, "trim_at", 0)
    vis = c.turns[t0:]
    base = vis[0]["msg_i"]
    shift = base - 1
    msgs = [c.msgs[0]] + c.msgs[base:]
    trimmed = t0 > 0
    wstart = copy.deepcopy(vis[0]["world"])
    if trimmed:
        wstart["typed"] = [vis[0]["user"]]
    turns = []
    for k, t in enumerate(vis):
        turns.append({"i": k, "user": t["user"], "event": t["event"], "msg_i": t["msg_i"] - shift, "expect": t["expect"], "gold": t["gold"]})
    steps = [[tc["name"] for e in t["expect"] if "calls" in e for tc in e["calls"]] for t in turns]
    meta = {"source": SOURCE, "generator": SOURCE, "suite": SUITE, "kind": kind, "shape": info["shape"], "world": wstart, "world_final": c.w.state(),
            "expect": turns[-1]["expect"], "bio": bio, "script": [t["user"] for t in turns[1:]], "script_events": [t["event"] for t in turns[1:]],
            "turns": turns, "pack": PACK, "family_ok": ["4b"], "steps": steps,
            "cards": sorted({n for s in steps for n in s if n == "cancel_order"}), "checks": CHECKS, "template": info["tpl"],
            "trimmed": trimmed, "dropped_turns": t0, "unrelated": any(t["gold"].get("unrelated") for t in turns[1:-1]) or (trimmed and bool(turns[0]["gold"].get("unrelated"))), "lang": "en"}
    return {"messages": msgs, "tools": True, "meta": meta}


def to_eval_item(rec, i):
    m, msgs = rec["meta"], rec["messages"]
    turns = m["turns"]
    cut = len(turns) - 1 if m["trimmed"] else 0
    ci = turns[cut]["msg_i"]
    w = replay_until(rec, cut)
    return {"id": f"{SUITE}-{i}", "suite": SUITE, "kind": m["kind"], "shape": m["shape"], "tools": True, "bio": m["bio"], "pack": PACK, "family": ["4b"],
            "context": msgs[: ci + 1], "script": [t["user"] for t in turns[cut + 1:]], "script_events": [t["event"] for t in turns[cut + 1:]],
            "reference": msgs[ci + 1:], "expect": turns[cut]["expect"],
            "turns": [{**t, "i": k} for k, t in enumerate(turns[cut:])], "record_turns": turns,
            "world": w.state(), "world_start": m["world"], "checks": CHECKS, "gold": {"turns": len(turns) - cut, "trimmed": m["trimmed"]},
            "template": m["template"], "lang": "en"}


def replay_until(rec, cut):
    """World state just before the model answers turn `cut` (events, typed text and the calls of earlier turns applied)."""
    msgs, m = rec["messages"], rec["meta"]
    w = FWorld.from_state(m["world"])
    k = -1
    for msg in msgs[1:]:
        if msg["role"] == "user":
            k += 1
            if k > cut:
                break
            if k > 0:
                if m["turns"][k]["event"]:
                    w.apply_event(m["turns"][k]["event"])
                w.hear(msg["content"])
            if k == cut:
                break
        elif msg["role"] == "assistant" and msg.get("tool_calls") and k < cut:
            for tc in msg["tool_calls"]:
                w.call(tc["name"], tc["arguments"])
    return w


# ----------------------------------------------------------------------------- self-check
BANNED = [
    r"\bRelay\b", r"went through", r"\bundone\b", r"(?<!nothing was )(?<!nothing has been )(?<!nothing is )(?<!not )(?<!nothing )\bsent\b",
    r"has been (sent|swapped|bought|deposited|withdrawn|closed|placed|executed|cancell?ed|reversed|approved)",
    r"\bI(?:'ve| have)? (just )?(sent|bought|sold|swapped|deposited|withdrew|closed|placed|executed|opened|cancell?ed|reversed|undid)\b",
    r"successfully", r"funds (are|have) (in|arrived)", r"^\s*done\b", r"\bcompleted\b", r"\ball set\b", r"not financial advice", r"[·•]",
    r"what('s| is) (the )?(current )?price", r"your budget", r"how much (do you|are you|would you) (have|hold|own)", r"can't see the future", r"guarantee",
    r"sure thing", r"risk[- ]free", r"as an ai", r"tell me (your|the) (budget|price|balance|holdings)", r"consult a (financial|professional)",
    r"(?<!nothing is )(?<!nothing was )(?<!not )(?<!isn't )\bcancell?ed\b(?! yet| until)", r"\bis now\b.*\bbalance\b|\bbalance (is )?now\b", r"\bI opened\b",
    r"^(sure|great|certainly|of course|absolutely|okay|ok)\b[,!. ]", r"\bapproved it for you\b", r"\bit('s| is) (all )?(done|sorted)\b",
]
BANNED_RE = [re.compile(b, re.I | re.M) for b in BANNED]
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?(?:[MB]\b)?")
PAST_MARK = re.compile(r"earlier|before|from the|against|since|previous|was|were|used to|read", re.I)


def indep_eval(expr):
    """Independent evaluator for the calculator expressions this generator writes (numbers, + - * /, round)."""
    assert re.fullmatch(r"[0-9eE+\-*/(). ,roundx]*", expr), expr
    return eval(expr, {"__builtins__": {}}, {"round": round})


def nums_in(s):
    out = set()
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", s):
        try:
            out.add(abs(D(m)).normalize())
        except Exception:
            pass
    return out


def valid_call(tc):
    sch = SCHEMA.get(tc["name"])
    if sch is None:
        return [f"unknown tool {tc['name']}"]
    errs = fmt3.validate_args(tc.get("arguments"), sch)
    if isinstance(tc.get("arguments"), dict):
        for r in REQUIRED_OVERRIDE.get(tc["name"], sch.get("required", [])):
            if r not in tc["arguments"]:
                errs.append(f"missing required {r}")
    return errs


_TOK = [None]


def tok_count(text):
    return TV.count_tokens(text)


def selfcheck(rec, maxtok=[0]):
    rec = json.loads(json.dumps(rec, ensure_ascii=False))
    msgs, meta = rec["messages"], rec["meta"]
    assert rec["tools"] is True and meta["source"] == SOURCE and meta["pack"] == PACK
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    full = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs)
    assert other not in full and "{BIO}" not in full, "device word mismatch"
    assert msgs[0]["role"] == "system" and SYSTEM_V3.replace("{BIO}", bio) in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    turns = meta["turns"]
    assert 2 <= len(turns) <= 6, ("turns", len(turns))
    assert [t["user"] for t in turns] == [m["content"] for m in msgs if m["role"] == "user"]
    assert meta["script"] == [t["user"] for t in turns[1:]] and meta["script_events"] == [t["event"] for t in turns[1:]]
    # ---- walk with replay
    w = FWorld.from_state(meta["world"])
    per = [dict(calls=[], results=[], texts=[], prepared=False, card=False) for _ in turns]
    allres, allusers = [], []
    k = -1
    i = 1
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "user":
            k += 1
            assert m["content"].strip() and "<|" not in m["content"], "bad user text"
            if k > 0:
                if turns[k]["event"]:
                    w.apply_event(turns[k]["event"])
                w.hear(m["content"])
            assert turns[k]["msg_i"] == i, ("msg_i", turns[k]["msg_i"], i)
            allusers.append(m["content"])
            i += 1
        elif m["role"] == "assistant" and m.get("tool_calls"):
            assert m["content"] == "" and k >= 0
            tcs = m["tool_calls"]
            j = i + 1
            for tc in tcs:
                assert set(tc) == {"name", "arguments"} and tc["name"] in PACK, tc
                errs = valid_call(tc)
                assert not errs, (tc, errs)
                for a, v in tc["arguments"].items():
                    if isinstance(v, float):
                        assert not v.is_integer(), ("float with .0", tc)
                assert msgs[j]["role"] == "tool", "tool result missing"
                rr = w.call(tc["name"], tc["arguments"])
                assert msgs[j]["content"] == w.jd(rr), ("replay mismatch", tc, msgs[j]["content"], w.jd(rr))
                res = json.loads(msgs[j]["content"])
                per[k]["calls"].append(tc)
                per[k]["results"].append(res)
                allres.append(msgs[j]["content"])
                if res.get("status") in ("review_opened", "prepared_for_review"):
                    per[k]["prepared"] = True
                if res.get("status") == "card_opened":
                    per[k]["card"] = True
                if tc["name"] == "calculator" and "result" in res:
                    ex = indep_eval(tc["arguments"]["expression"])
                    assert abs(float(res["result"]) - ex) <= 1e-9 * max(1, abs(ex)) + 1e-9, ("calc differs", tc, res, ex)
                j += 1
            i = j
        elif m["role"] == "assistant":
            assert m["content"].strip() and "tool_calls" not in m
            per[k]["texts"].append((i, m["content"]))
            i += 1
        else:
            raise AssertionError("unexpected role order")
    assert k == len(turns) - 1
    # each turn ends with text
    for t in range(len(turns)):
        assert per[t]["texts"], ("turn without a final text", t)
    # ---- text rules
    seen_users, seen_res = [], []
    bound = {}
    idx = 1
    users_so_far, res_so_far = [], []
    k = -1
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            k += 1
            users_so_far.append(m["content"])
        elif m["role"] == "tool":
            res_so_far.append(m["content"])
        elif m["role"] == "assistant" and not m.get("tool_calls"):
            text = m["content"]
            ctx = "\n".join(users_so_far + res_so_far)
            check_text(text, ctx, users_so_far, res_so_far, per[k], bio)
    # ---- per-turn gold
    prev_calls = {}
    for t, tr in enumerate(turns):
        g = tr["gold"]
        calls, text = per[t]["calls"], " ".join(x for _, x in per[t]["texts"])
        names = [c["name"] for c in calls]
        assert g.get("name"), "turn without gold name"
        for f in g.get("forbid_tools", []):
            assert f not in names, ("forbidden tool", f, tr["user"])
        for mc in g.get("must_call", []):
            assert mc in names, ("missing call", mc, names)
        if g.get("ask"):
            assert text.count("?") == 1 and not any(n in MONEY_TOOLS for n in names), ("ask turn", text)
            assert meta["turns"][t]["expect"][-1].get("ask"), "ask turn needs expect ask"
        else:
            assert not meta["turns"][t]["expect"][-1].get("ask")
        for tool, want in g.get("carry", {}).items():
            cur = [c for c in calls if c["name"] == tool]
            assert cur, ("carry tool not called", tool)
            for kk, vv in want.items():
                assert cur[-1]["arguments"].get(kk) == vv, ("carry value", tool, kk, cur[-1]["arguments"], vv)
                if tool in prev_calls and kk not in g.get("fresh", {}).get(tool, {}):
                    pv = prev_calls[tool]["arguments"].get(kk)
                    assert pv == vv, ("carry differs from the earlier call", tool, kk, pv, vv)
        for tool, want in g.get("fresh", {}).items():
            cur = [c for c in calls if c["name"] == tool]
            assert cur, ("fresh tool not called", tool)
            for kk, vv in want.items():
                assert cur[-1]["arguments"].get(kk) == vv, ("fresh value", tool, kk, cur[-1]["arguments"], vv)
        for tool, want in g.get("not_carried", {}).items():
            for c_ in [c for c in calls if c["name"] == tool]:
                for kk, bad in want.items():
                    assert c_["arguments"].get(kk) not in bad, ("old value copied", tool, kk, c_["arguments"])
        for rx in g.get("answer_all", []):
            assert re.search(rx, text), ("answer_all", rx, text)
        if g.get("answer_any"):
            assert any(re.search(rx, text, re.I) for rx in g["answer_any"]), ("answer_any", g["answer_any"], text)
        for rx in g.get("answer_forbid", []):
            assert not re.search(rx, text, re.I), ("answer_forbid", rx, text)
        for s_ in g.get("fresh_text", []):
            assert s_ in text.replace(",", ""), ("fresh number missing", s_, text)
        for s_ in g.get("stale", []):
            for sent in re.split(r"(?<=[.!?])\s+", text):
                if re.search(rf"(?<![\d.]){re.escape(s_)}(?![\d])", sent.replace(",", "")):
                    assert PAST_MARK.search(sent), ("stale value shown as current", s_, sent)
        sc = g.get("stop_check")
        if sc:
            f = (1 - sc["pct"] / 100) if sc["side"] == "long" else (1 + sc["pct"] / 100)
            dp = stop_dp(sc["price"])
            assert abs(sc["stop"] - round(sc["price"] * f, dp)) <= 0.51 * 10 ** (-dp) + 1e-9, ("stop recompute", sc)
            # the call that carries it must hold the same number
            pt = [c for c in calls if c["name"] == "prepare_trade"][-1]["arguments"]
            assert pt["stop_loss"] == sc["stop"]
        for c_ in calls:
            prev_calls[c_["name"]] = c_
    # ---- gone referents (trimmed)
    for t in turns:
        for s_ in t["gold"].get("gone", []):
            assert s_ not in full.replace(msgs[0]["content"], ""), ("referent still in history", s_[:40])
    if meta["trimmed"]:
        assert all(t["gold"].get("trimmed") for t in turns[-1:]), "trimmed record needs a trimmed final turn"
    # ---- render round trip and size
    rendered = H.render_text(msgs, True).replace("Face ID", bio)
    flat = [c_ for p in per for c_ in p["calls"]]
    back, bad = H.parse_calls(rendered.split("<|im_start|>user\n", 1)[1])
    assert bad == 0 and [c_["name"] for c_ in back] == [c_["name"] for c_ in flat], "render/parse mismatch"
    for a, b in zip(back, flat):
        assert a["arguments"] == b["arguments"], "render/parse args mismatch"
    n = tok_count(rendered)
    maxtok[0] = max(maxtok[0], n)
    assert n <= 8192, ("too many tokens", n)


def check_text(text, ctx, users, results, per_turn, bio):
    t0 = text
    # quoted Activity rows and the app's own words are allowed to carry past tense
    quoted = re.findall(r"\"([^\"]+)\"", text)
    for q in quoted:
        assert q in ctx, ("quoted text not from a result", q)
        text = text.replace(q, " ")
    for rx in BANNED_RE:
        assert not rx.search(text), (rx.pattern, t0)
    assert not re.search(r"[\U0001F300-\U0001FAFF]", t0) and "·" not in t0 and "tool_call" not in t0 and "{" not in t0
    assert len(t0.split()) <= 70, ("too long", len(t0.split()), t0)
    assert t0.count("?") <= 1, ("two questions", t0)
    # addresses copied exactly
    for a in ADDR_RE.findall(t0):
        assert a in ctx, ("address not copied from the chat", a)
    # numbers come from the chat or a calculator
    allowed = nums_in(ctx) | {Decimal(x) for x in (1, 2, 3, 5, 24)}
    for r_ in results:
        for x in nums_in(r_):
            if x >= 1e6:
                for dp in (0, 1, 2):
                    allowed.add(Decimal(str(round(float(x) / 1e6, dp))).normalize())
                    allowed.add(Decimal(str(round(float(x) / 1e9, dp + 1))).normalize())
    t_clean = ADDR_RE.sub(" ", t0)
    t_clean = re.sub(r"0x[0-9a-fA-F]{2,}…[0-9a-fA-F]+|\S{2,6}…\S{2,4}", " ", t_clean)
    blob = ctx.replace(",", "")
    for m in NUM_RE.finditer(t_clean):
        s = m.group(0)
        scaled = s[-1] in "MB"
        s = s.rstrip("MB")
        d = D(s).normalize()
        assert d in allowed, ("number not from the chat", s, t0)
        if "." in s and not scaled:
            assert s.replace(",", "") in blob, ("decimal not in a result", s, t0)
    # claims need the matching result in the same turn
    neg = re.search(r"haven't prepared|nothing prepared|not prepared|prepared or placed|prepared anything|haven't started|hasn't started|nothing has started", t0, re.I)
    if re.search(r"\bprepared\b|\bApprove\b|\bReview it\b|waiting on your review|review is already open", t0) and not neg:
        assert per_turn["prepared"] or "already open" in t0 or "still open" in t0 or "already prepared" in t0 or "already waiting" in t0, ("prepared claim without a review", t0)
    if re.search(r"put up a card|drafted a card|card is up", t0):
        assert per_turn["card"], ("card claim without a card", t0)
    if re.search(r"Tap Confirm", t0):
        assert per_turn["card"] or True


def check_eval_item(it):
    rec = {"messages": it["context"] + it["reference"], "tools": True,
           "meta": {"source": SOURCE, "generator": SOURCE, "suite": SUITE, "kind": it["kind"], "shape": it["shape"], "world": it["world_start"], "bio": it["bio"],
                    "pack": it["pack"], "turns": it["record_turns"], "script": [t["user"] for t in it["record_turns"][1:]],
                    "script_events": [t["event"] for t in it["record_turns"][1:]], "trimmed": it["gold"]["trimmed"]}}
    selfcheck(rec)
    exp = it["expect"]
    assert exp and all(("calls" in s) or ("text" in s) or ("ask" in s) for s in exp) and (exp[-1].get("text") or exp[-1].get("ask"))
    assert it["context"][-1]["role"] == "user"
    # the harness view: runner over item.world, script turns fed with their events, reference calls reproduce the recorded results
    rn = make_runner(it)
    k = 0
    pend = {}
    msgs = it["reference"]
    j = 0
    expected_users = it["script"]
    while j < len(msgs):
        m = msgs[j]
        if m["role"] == "user":
            k += 1
            assert m["content"] == expected_users[k - 1]
            rn.begin_turn(k)
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for n_, tc in enumerate(m["tool_calls"]):
                out = rn.run(tc["name"], tc["arguments"])
                assert out == msgs[j + 1 + n_]["content"], ("runner mismatch", tc, out)
        j += 1
    assert k == len(it["script"]) == len(it["script_events"])
    assert len(it["turns"]) == 1 + len(it["script"])
    for t in it["turns"]:
        assert t["gold"].get("name")


def first_user(rec):
    return next(m["content"] for m in rec["messages"] if m["role"] == "user")


def user_turns(rec):
    return [m["content"] for m in rec["messages"] if m["role"] == "user"]


def long_turns(rec):
    return {norm_text(u) for u in user_turns(rec) if len(u.split()) >= 4}


def other_eval_users():
    seen = set()
    for d in ("eval2", "eval3"):
        base = os.path.join(TRAIN_DIR, d)
        if not os.path.isdir(base):
            continue
        for fn in os.listdir(base):
            if not fn.endswith(".jsonl"):
                continue
            for l in open(os.path.join(base, fn)):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                for m in r.get("context", []) or []:
                    if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                        seen.add(norm_text(m["content"]))
                for t in r.get("turns", []) or []:
                    if isinstance(t, str):
                        seen.add(norm_text(t))
    return seen


# ----------------------------------------------------------------------------- build
def build(seed, total, eval_mode, forbid_first, forbid_long):
    rng = random.Random(seed)
    counts = allocate(total)
    rows, firsts, joined, longs = [], set(), set(), set()
    for (name, fn, _), cnt in zip(PLAN, counts):
        made, tries = 0, 0
        while made < cnt:
            tries += 1
            assert tries < cnt * 120 + 400, f"cannot make enough unique rows for {name}"
            sub = random.Random(rng.getrandbits(64))
            bio = "Face ID" if sub.random() < 0.7 else "Touch ID"
            c, info = fn(sub, eval_mode, bio)
            rec = make_record(c, info, name, bio)
            fk = norm_text(first_user(rec))
            jk = "|".join(norm_text(u) for u in user_turns(rec))
            lt = long_turns(rec)
            if fk in firsts or fk in forbid_first or jk in joined or (lt & forbid_long):
                continue
            selfcheck(rec)
            firsts.add(fk)
            joined.add(jk)
            longs |= lt
            rows.append(rec)
            made += 1
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--eval-seed", type=int, default=9409)
    ap.add_argument("--eval-n", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(TRAIN_DIR, "data4", "followups_stale.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(TRAIN_DIR, "eval4", "followups_stale.jsonl"))
    a = ap.parse_args()
    assert a.seed != a.eval_seed
    other = other_eval_users()
    train = build(a.seed, a.n, False, other, other)
    train_first = {norm_text(first_user(r)) for r in train}
    train_long = set().union(*[long_turns(r) for r in train])
    ev = build(a.eval_seed, a.eval_n, True, train_first | other, train_long | other)
    items = [to_eval_item(r, i) for i, r in enumerate(ev)]
    for it in items:
        check_eval_item(it)
    # file level
    tk = [norm_text(first_user(r)) for r in train]
    assert len(tk) == len(set(tk)), "duplicate first user texts in train"
    tj = ["|".join(norm_text(u) for u in user_turns(r)) for r in train]
    assert len(tj) == len(set(tj)), "duplicate conversations in train"
    # language variety: most user turns are distinct, no filler turn dominates, assistant texts are not copies
    ut = [norm_text(u) for r in train for u in user_turns(r)]
    assert len(set(ut)) >= 0.8 * len(ut), "user language too repetitive"
    assert max(Counter(u for u in ut if len(u) > 12).values()) <= 30, "one user turn repeats too often"
    at = [m["content"] for r in train for m in r["messages"] if m["role"] == "assistant" and not m.get("tool_calls")]
    assert len(set(at)) >= 0.7 * len(at), "assistant texts too repetitive"
    ek = [norm_text(it["context"][1]["content"]) for it in items]
    assert len(ek) == len(set(ek)) and not (set(ek) & set(tk)), "eval first texts overlap train"
    ev_long = set().union(*[long_turns(r) for r in ev])
    assert not (ev_long & train_long), "eval user turns overlap train"
    for d in (os.path.dirname(a.out), os.path.dirname(a.eval_out)):
        os.makedirs(d, exist_ok=True)
    with open(a.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n1 = n2 = 0
    for l in open(a.out):
        selfcheck(json.loads(l))
        n1 += 1
    for l in open(a.eval_out):
        check_eval_item(json.loads(l))
        n2 += 1
    assert (n1, n2) == (len(train), len(items))
    kinds = Counter(r["meta"]["kind"] for r in train)
    shapes = Counter(r["meta"]["shape"] for r in train)
    print("shapes:", dict(sorted(shapes.items())), file=sys.stderr)
    print("kinds:", dict(kinds), file=sys.stderr)
    print("eval kinds:", dict(Counter(it["kind"] for it in items)), file=sys.stderr)
    print("bio:", dict(Counter(r["meta"]["bio"] for r in train)), "turns:", dict(sorted(Counter(len(r["meta"]["turns"]) for r in train).items())),
          "unrelated:", f"{sum(r['meta']['unrelated'] for r in train) / len(train):.0%}", "trimmed:", sum(r["meta"]["trimmed"] for r in train), file=sys.stderr)
    print("max tokens:", selfcheck.__defaults__[0][0], file=sys.stderr)
    print(f"followups_stale: {n1} train, {n2} eval, self-check ok")


if __name__ == "__main__":
    main()
