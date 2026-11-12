#!/usr/bin/env python3
"""orders_protect generator (sft5 generator 2): cancel_order, set_tp_sl and the perps part of get_positions.

Replaces the old "I can't cancel orders" answer. Teaches:
  * cancel_order with `kind` only when the user's words name it (stop loss / take profit / limit order), without it
    when they do not (the app answers needs_choice and the model asks the app's question), `coin: "all"` for
    "cancel all my open orders" (limit orders only, the answer says TP/SL are not included), `order_id` from a
    get_positions row for "cancel the second one";
  * set_tp_sl on an OPEN position with the user's exact digits, or a price computed with the calculator from the
    entry (hyperliquid_account) for percent stops, or the entry itself for break-even; mention `replaces`;
  * reading protection and orders with get_positions {kind: "perps"} (stops and take profits are "protection",
    limits are "orders", ids are never read out);
  * errors (wrong side, no position, nothing to cancel, blocked, no wallet) and near misses (close = prepare_close,
    "cancel that" with a review open = no tool, "cancel" with nothing pending, how stops work, a stop on a NEW trade =
    prepare_trade, a stop with no price = ask, margin questions = hyperliquid_account);
  * a tool that is not in the pack (core + solana) = one sentence and open_page {page: trade, tab: positions}.
Cards (cancel_order) are never "done": "put up a card", "tap Confirm", "nothing is cancelled until you tap".
Review tools (set_tp_sl, prepare_*) are "prepared ... approve with Face ID", never "protected".

Reuses gen2 (hermit_fmt, sim, build_agent.Conv) and gen3 (close_position helpers) by import; nothing there is edited.
Tool registry, result shapes and system prompt come from ../spec (hermit_tools_v3.json, hermit_tools_v3.meta.json,
SYSTEM_AGENT_v3.txt). Every assistant sentence is a Python f-string filled from tool results, tool arguments or user
text; no number is typed by hand. Percent stops and break-even prices go through the calculator.

Usage:  python orders_protect.py --seed 4 --n 420 [--eval-seed 9402 --eval-n 80]
Writes ../data4/orders_protect.jsonl and ../eval4/orders_protect.jsonl, running the self-check on every record first
(exit 1 on any failure, nothing written). Replay an eval item with OPWorld.from_state(item["world"]).call(name, args).
"""
import argparse, json, os, random, re, sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(LAB, "gen3"), os.path.join(LAB, "gen2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hermit_fmt as H          # noqa: E402  (gen2, read only)
import sim                      # noqa: E402
import build_agent as BA        # noqa: E402  (Conv)
import close_position as CP     # noqa: E402  (gen3 helpers: money, coin pools, pool_for, scale_quota, ...)

SPEC = os.path.join(LAB, "spec")
TOOLS_V3 = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
META_V3 = json.load(open(os.path.join(SPEC, "hermit_tools_v3.meta.json")))
SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V3}
TOOL_ORDER = [t["function"]["name"] for t in TOOLS_V3]
SYSTEM_V3 = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert "{BIO}" in SYSTEM_V3
SYSTEM_FACE = SYSTEM_V3.replace("{BIO}", "Face ID")
GEN_NAME = "orders_protect"

CORE = ["calculator", "get_price", "get_balances", "get_portfolio", "get_address", "get_activity", "prepare_send",
        "prepare_swap", "get_quote", "open_page", "open_app", "open_add_funds"]
TRADE = ["hyperliquid_account", "position_math", "search_markets", "prepare_trade", "prepare_close", "cancel_order",
         "set_tp_sl", "get_positions"]
SOLANA = ["search_tokens", "prepare_buy", "prepare_sell", "get_positions"]
LITE_TOOLS = set(CORE) | {"hyperliquid_account", "prepare_trade", "prepare_close", "cancel_order"}
PACKS = {
    "core+trade": [t for t in TOOL_ORDER if t in set(CORE) | set(TRADE)],
    "core+solana": [t for t in TOOL_ORDER if t in set(CORE) | set(SOLANA)],
    "all30": list(TOOL_ORDER),
}
assert len(PACKS["core+trade"]) == 20 and len(PACKS["core+solana"]) == 16 and len(PACKS["all30"]) == 30
CARD_TOOLS = {"cancel_order"}
REVIEW_TOOLS = {"set_tp_sl", "prepare_trade", "prepare_close", "prepare_send", "prepare_swap"}

short, money, norm, sdec, join_and = CP.short, CP.money, CP.norm, CP.sdec, CP.join_and
COIN_WORDS, WORD2COIN, pick_coins, coin_text = CP.COIN_WORDS, CP.WORD2COIN, CP.pick_coins, CP.coin_text
scale_quota, art = CP.scale_quota, CP.art


def dec(x):
    return Decimal(str(x))


def num(d):
    d = Decimal(d)
    return int(d) if d == d.to_integral_value() else float(d)


def dfmt(d):
    """Plain decimal text without exponent or trailing zeros."""
    return format(Decimal(d).normalize(), "f")


def sk(d):
    """Alphabetical keys, like the app's toolbox results."""
    return {k: (sk(d[k]) if isinstance(d[k], dict) else d[k]) for k in sorted(d)}


def join_or(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " or " + items[-1]


def dp_for(coin):
    return max(1, sim.ASSETS[coin][2])


def sdp_of(mark):
    return 4 if mark >= 1000 else 2 if mark >= 10 else 1 if mark >= 1 else 0


def nice(rng, x):
    """A price a trader would type: round x to a human step."""
    x = float(x)
    if x >= 20000:
        steps = [500, 100, 100, 50, 10, 1, 0.5]
    elif x >= 2000:
        steps = [50, 25, 10, 5, 1]
    elif x >= 200:
        steps = [5, 2, 1, 1, 0.5]
    elif x >= 20:
        steps = [1, 1, 0.5, 0.5, 0.1, 0.25]
    elif x >= 2:
        steps = [0.1, 0.05, 0.01, 0.5]
    elif x >= 0.2:
        steps = [0.01, 0.005, 0.001, 0.0001]
    else:
        steps = [0.0001, 0.00005, 0.00001]
    st = dec(rng.choice(steps))
    d = (dec(x) / st).to_integral_value(rounding=ROUND_HALF_UP) * st
    if d <= 0:
        d = dec(x)
    return num(d)


def price_text(rng, v):
    """How a user types a price: 128, 94250.5, 94,250.5, $128, 94.5k."""
    d = dec(v)
    s = dfmt(d)
    out = s
    if d >= 1000 and rng.random() < 0.35:
        ip, _, fp = s.partition(".")
        out = f"{int(ip):,}" + ("." + fp if fp else "")
    if d >= 10000 and d % 100 == 0 and rng.random() < 0.15:
        out = dfmt(d / 1000) + "k"
    elif rng.random() < 0.15:
        out = "$" + out
    return out


def parse_price(tok):
    t = tok.lower().replace("$", "").replace(",", "")
    mult = 1
    if t.endswith("k"):
        t, mult = t[:-1], 1000
    return dec(t) * mult


PRICE_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?k?", re.I)


def user_numbers(text):
    return {parse_price(m.group(0)) for m in PRICE_RE.finditer(text)}


KIND_WORD = {"limit": "limit order", "stop_loss": "stop loss", "take_profit": "take profit"}
KIND_NOUN = {"order": "limit orders", "stop_loss": "stop loss", "take_profit": "take profit", None: "orders"}


def order_label(o):
    """The `cancelling` text of one order."""
    if o["kind"] == "limit":
        return f"{o['coin']} {o['side']} limit, {short(o['size'])} {o['coin']} at {money(o['price'])}"
    return f"{o['coin']} {KIND_WORD[o['kind']]} at {money(o['price'])}"


# ------------------------------------------------------------------ world
BLOCKED_REASONS = ["This order is above your daily limit for the assistant.",
                   "The assistant has reached its limit for today.",
                   "This order is above the per-trade limit for the assistant."]
MAXLEV = {"BTC": 40, "ETH": 25, "SOL": 20}


class OPWorld(sim.World):
    """gen2 World with the app's v3 result shapes (hermit_tools_v3.meta.json) for the tools this generator uses."""
    STATE_KEYS = sim.World.STATE_KEYS + ("orders", "device", "no_wallet", "blocked", "wallet_name", "next_id", "net")

    def __init__(self, rng, device="iPhone"):
        super().__init__(rng, empty_hl=True, empty_wallet=False)
        self.device, self.no_wallet, self.blocked = device, False, None
        self.wallet_name, self.net = "Main", "Base"
        self.hl_positions, self.hl_orders, self.orders = [], [], []
        self.next_id = 8000 + rng.randrange(0, 900)

    # ---- building
    def pos(self, coin):
        return next((p for p in self.hl_positions if p["coin"] == coin), None)

    def add_position(self, rng, coin, side, pnl_sign=None):
        mark = self.prices[coin]
        dp = sim.ASSETS[coin][2]
        sgn = 1 if side == "long" else -1
        for _ in range(80):
            lev = rng.choice([2, 3, 5, 5, 10, 10, 15, 20])
            lev = min(lev, MAXLEV.get(coin, 10))
            d = rng.uniform(0.012, min(0.09, 0.7 / lev))
            ps = pnl_sign if pnl_sign else rng.choice([1, -1])
            entry = round(mark * (1 - sgn * ps * d), dp)
            entry = short(float(entry)) if dp == 0 else entry
            if entry <= 0:
                continue
            size_usd = rng.uniform(0.05, 0.28) * self.hl_value * lev
            size = round(size_usd / entry, sdp_of(mark))
            if size <= 0:
                continue
            pnl = round(sgn * (mark - entry) * size, 2)
            if pnl == 0 or (pnl_sign and (pnl > 0) != (pnl_sign > 0)):
                continue
            liq = round(sim.liq_price(side, entry, lev, 0.005), dp or 0)
            self.hl_positions.append({"coin": coin, "side": side, "size": short(size), "entry": short(entry),
                                      "mark": mark, "liq": short(liq), "leverage": lev, "unrealized_pnl": pnl,
                                      "margin": round(size * entry / lev, 2)})
            return
        raise RuntimeError("could not build position")

    def _new_id(self, rng):
        self.next_id += rng.randint(1, 9)
        return str(self.next_id)

    def add_trigger(self, rng, coin, kind, price=None, whole=True):
        p = self.pos(coin)
        long_ = p["side"] == "long"
        mark = p["mark"]
        if price is None:
            if kind == "stop_loss":
                price = nice(rng, mark * (1 - rng.uniform(0.04, 0.18)) if long_ else mark * (1 + rng.uniform(0.04, 0.18)))
            else:
                price = nice(rng, mark * (1 + rng.uniform(0.05, 0.35)) if long_ else mark * (1 - rng.uniform(0.05, 0.30)))
        size = p["size"] if whole else round(p["size"] * rng.choice([0.3, 0.4, 0.5, 0.6]), sdp_of(mark)) or p["size"]
        self.orders.append({"id": self._new_id(rng), "coin": coin, "kind": kind, "price": price,
                            "side": "sell" if long_ else "buy", "size": short(size), "whole_position": bool(whole)})

    def add_limit(self, rng, coin, side=None):
        mark = self.prices[coin]
        side = side or rng.choice(["buy", "sell"])
        price = nice(rng, mark * (1 - rng.uniform(0.03, 0.10)) if side == "buy" else mark * (1 + rng.uniform(0.03, 0.10)))
        size = round(rng.uniform(300, 3000) / price, sdp_of(mark))
        size = size or 1
        self.orders.append({"id": self._new_id(rng), "coin": coin, "kind": "limit", "price": price, "side": side,
                            "size": short(size), "whole_position": None})

    # ---- tools
    def _row_account(self, o):
        if o["kind"] == "limit":
            return sk({"coin": o["coin"], "is_trigger": False, "order_type": "Limit", "price": short(o["price"]),
                       "side": o["side"], "size": short(o["size"])})
        p = self.pos(o["coin"])
        long_ = p["side"] == "long"
        below = (o["kind"] == "stop_loss") == long_
        return sk({"coin": o["coin"], "is_trigger": True,
                   "order_type": "Stop Market" if o["kind"] == "stop_loss" else "Take Profit Market",
                   "price": short(o["price"]), "side": o["side"], "size": short(o["size"]),
                   "trigger_condition": f"Price {'below' if below else 'above'} {short(o['price'])}",
                   "whole_position": o["whole_position"]})

    def _row_pos(self, p):
        return {"coin": p["coin"], "entry": short(p["entry"]), "leverage": p["leverage"], "liquidation": short(p["liq"]),
                "side": p["side"], "size": short(p["size"]), "unrealised_pnl": short(p["unrealized_pnl"])}

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
        return sk({"account_mode": "unified", "account_value": short(value),
                   "available_usdc": short(round(max(value - used, 0), 2)), "margin_used": short(used),
                   "open_orders": [self._row_account(o) for o in self.orders],
                   "positions": [self._row_pos(p) for p in self.hl_positions]})

    def t_get_positions(self, kind="all"):
        if self.no_wallet:
            return {"error": f"No wallet on this {self.device} yet"}
        if kind not in ("all", "spot", "earn", "perps"):
            return {"error": "kind must be all, spot, earn or perps"}
        out = {"wallet": self.wallet_name}
        if kind in ("all", "earn"):
            out["earn"] = []
        if kind in ("all", "perps"):
            rows = []
            for o in self.orders:
                r = {"coin": o["coin"], "id": o["id"], "kind": o["kind"], "price": short(o["price"]),
                     "side": o["side"], "size": short(o["size"])}
                if o["kind"] != "limit":
                    r["whole_position"] = o["whole_position"]
                rows.append(sk(r))
            out["perps"] = {"open_orders": rows, "positions": [self._row_pos(p) for p in self.hl_positions]}
        if kind in ("all", "spot"):
            out["spot"] = []
        return sk(out)

    def _card(self, label, warning=None):
        out = {"cancelling": label, "executed": False, "kind": "cancel_order",
               "note": "Waiting for the user to tap Confirm. Nothing is cancelled yet.", "status": "card_opened"}
        if warning:
            out["warning"] = warning
        return sk(out)

    def t_cancel_order(self, coin, kind=None, order_id=None):
        if self.no_wallet:
            return {"error": f"No wallet on this {self.device} to trade with"}
        if kind not in (None, "order", "take_profit", "stop_loss"):
            return {"error": "kind must be order, take_profit or stop_loss"}
        c = str(coin).strip().upper()
        if order_id is not None:
            o = next((x for x in self.orders if x["id"] == str(order_id)), None)
            if o is None:
                return sk({"note": "The user has no open order with that id. Tell them.", "prepared": False,
                           "status": "nothing_to_cancel"})
            matches = [o]
        elif c == "ALL":
            lim = [x for x in self.orders if x["kind"] == "limit"]
            if not lim:
                return sk({"note": "The user has no open limit orders. Tell them.", "prepared": False,
                           "status": "nothing_to_cancel"})
            coins = []
            for x in lim:
                if x["coin"] not in coins:
                    coins.append(x["coin"])
            n = len(lim)
            return self._card(f"{n} open limit order{'s' if n != 1 else ''} ({', '.join(coins)})")
        else:
            matches = [x for x in self.orders if x["coin"] == c]
            if kind:
                want = "limit" if kind == "order" else kind
                matches = [x for x in matches if x["kind"] == want]
            if not matches:
                return sk({"note": f"The user has no open {c} {KIND_NOUN[kind]}. Tell them.", "prepared": False,
                           "status": "nothing_to_cancel"})
            if len(matches) > 1 and kind is None:
                parts = []
                for i, x in enumerate(matches):
                    w = KIND_WORD[x["kind"]]
                    parts.append(f"{'A' if i == 0 else 'a'} {w} at {money(x['price'])}")
                return sk({"note": "Nothing was prepared. Ask the user which one.",
                           "options": [f"{x['kind']} at {money(x['price'])}" for x in matches], "prepared": False,
                           "question": f"Which {c} order? {join_or(parts)}.", "status": "needs_choice"})
        o = matches[0]
        warning = None
        if o["kind"] == "stop_loss":
            p = self.pos(o["coin"])
            if p:
                warning = f"Without it the {o['coin']} {p['side']} has no stop loss."
        return self._card(order_label(o), warning)

    def t_set_tp_sl(self, coin, take_profit=None, stop_loss=None):
        if self.no_wallet:
            return {"error": f"No wallet on this {self.device} to trade with"}
        if self.blocked:
            return sk({"note": "Nothing was prepared. Tell the user why.", "reasons": [self.blocked], "status": "blocked"})
        for v in (take_profit, stop_loss):
            if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0):
                return {"error": "Prices must be positive numbers", "prepared": False}
        c = str(coin).strip().upper()
        p = self.pos(c)
        if p is None:
            return sk({"note": f"The user has no open {c} position on Hyperliquid. Tell them.", "prepared": False,
                       "status": "no_position"})
        if take_profit is None and stop_loss is None:
            return {"error": "Give a take profit price, a stop loss price or both.", "prepared": False}
        mark, long_ = p["mark"], p["side"] == "long"
        if stop_loss is not None and ((long_ and stop_loss >= mark) or (not long_ and stop_loss <= mark)):
            return {"error": f"A stop loss for a {p['side']} must be {'below' if long_ else 'above'} the current price ({short(mark)}).",
                    "prepared": False}
        if take_profit is not None and ((long_ and take_profit <= mark) or (not long_ and take_profit >= mark)):
            return {"error": f"A take profit for a {p['side']} must be {'above' if long_ else 'below'} the current price ({short(mark)}).",
                    "prepared": False}
        out = {"executed": False, "kind": "tp_sl", "note": "Waiting for the user to approve with Face ID.",
               "order": "Trigger, market, reduce only", "position": f"{c} {p['side']}, {short(p['size'])} {c}",
               "status": "review_opened"}
        rep = []
        if stop_loss is not None:
            out["stop_loss"] = stop_loss
            old = next((o for o in self.orders if o["coin"] == c and o["kind"] == "stop_loss"), None)
            if old:
                rep.append(f"stop loss at {money(old['price'])}")
        if take_profit is not None:
            out["take_profit"] = take_profit
            old = next((o for o in self.orders if o["coin"] == c and o["kind"] == "take_profit"), None)
            if old:
                rep.append(f"take profit at {money(old['price'])}")
        if rep:
            out["replaces"] = " and ".join(rep)
        return sk(out)

    def t_prepare_close(self, coin, percent=None):
        if self.no_wallet:
            return {"error": f"No wallet on this {self.device} to trade with"}
        c = str(coin).strip().upper()
        p = self.pos(c)
        if p is None:
            return sk({"note": f"The user has no open {c} position on Hyperliquid. Tell them.", "prepared": False,
                       "status": "no_position"})
        if percent is not None and not (isinstance(percent, (int, float)) and 0 < percent <= 100):
            return {"error": "percent must be between 1 and 100"}
        sdp = sdp_of(p["mark"])
        if percent is None or percent == 100:
            closing = f"all of the {c} {p['side']}, {p['size']} {c}"
            upnl = p["unrealized_pnl"]
        else:
            qty = sdec(dec(p["size"]) * dec(percent) / 100, sdp + 2)
            closing = f"{percent}% of the {c} {p['side']}, {qty} {c}"
            upnl = sdec(dec(p["unrealized_pnl"]) * dec(percent) / 100, 2)
        return sk({"closing": closing, "executed": False, "kind": "close", "note": "Waiting for the user to approve with Face ID.",
                   "order": "Market, reduce only", "status": "review_opened", "unrealised_pnl": short(upnl)})

    def t_prepare_trade(self, coin, side, size_usd, leverage, order_type, limit_price=None, take_profit=None, stop_loss=None):
        if self.no_wallet:
            return {"error": f"No wallet on this {self.device} to trade with"}
        c = str(coin).strip().upper()
        if c not in sim.ASSETS:
            return {"error": f"{c} isn't tradable on Hyperliquid"}
        if leverage > MAXLEV.get(c, 10):
            return {"error": f"Max leverage for {c} is {MAXLEV.get(c, 10)}x", "prepared": False}
        used = sum(p["margin"] for p in self.hl_positions)
        avail = round(self.hl_value + sum(p["unrealized_pnl"] for p in self.hl_positions) - used, 2)
        need = round(size_usd / leverage, 2)
        if need > avail:
            return {"error": f"Not enough margin: {avail:.2f} USDC available, {need:.2f} needed", "prepared": False}
        return {"status": "review_opened", "kind": "trade", "executed": False,
                "note": "Waiting for the user to approve with Face ID."}

    def t_prepare_send(self, to_address, amount, symbol, chain=None):
        net = (chain or self.net).capitalize()
        return sk({"amount": f"{short(amount)} {str(symbol).upper()}", "executed": False, "kind": "send", "network": net,
                   "note": "Waiting for the user to approve with Face ID.", "status": "review_opened"})

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        fee = round(amount * 35 / 10000, 2)
        out_usd = round(amount - fee, 2)
        price = self.prices[str(to_symbol).upper()]
        q = out_usd / price
        digits = 3
        mag = 0 if q == 0 else int(Decimal(q).adjusted())
        out = dec(round(q, digits - 1 - mag))
        outs = dfmt(out)
        return sk({"amount_in_usd": short(float(amount)), "amount_out": outs, "amount_out_usd": out_usd, "fee_bps": 35,
                   "from": f"{short(amount)} {from_symbol} on {str(from_chain).capitalize()}",
                   "note": "Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels.",
                   "policy": "allow", "route": "NEAR Intents", "seconds": 14, "signature_verified": True,
                   "status": "prepared_for_review",
                   "to": f"about {outs} {to_symbol} on {str(to_chain).capitalize()}", "total_cost_usd": fee})

    def t_open_page(self, page, tab=None):
        out = {"note": f"The user taps it to open {str(page).capitalize()}.", "page": page, "screen": page,
               "status": "button_shown"}
        if tab:
            out["tab"] = tab
        return sk(out)


def make_world(rng, split, bio, must=None, extra=0, no_wallet=False, blocked=None, signs=None):
    """must = (coin, side, pnl_sign|None) forced in; `extra` more random positions."""
    dev = "Mac" if (bio == "Touch ID" and rng.random() < 0.9) else "iPhone"
    w = OPWorld(rng, dev)
    w.hl_value = round(rng.uniform(400, 25000), 2)
    specs = []
    if must:
        specs.append(must)
    for c in pick_coins(rng, split, extra, [must[0]] if must else []):
        specs.append((c, rng.choice(["long", "short"]), None))
    rng.shuffle(specs)
    for coin, side, ps in specs:
        w.add_position(rng, coin, side, ps)
    w.no_wallet, w.blocked = no_wallet, blocked
    return w


def add_noise(rng, w, exclude, p=0.3):
    """Orders on other coins, so the model has to pick the coin."""
    others = [x["coin"] for x in w.hl_positions if x["coin"] != exclude]
    if others and rng.random() < p:
        c = rng.choice(others)
        w.add_trigger(rng, c, rng.choice(["stop_loss", "take_profit"]))
    if rng.random() < p / 2:
        c = rng.choice([x for x in ["BTC", "ETH", "SOL", "HYPE"] if x != exclude])
        w.add_limit(rng, c)


# ------------------------------------------------------------------ phrase pools
# Tags "L|" / "S|" restrict a template to long / short worlds. About 1 in 5 templates is held out for eval.
CANCEL_KIND_T = [
    "cancel my {c} {k}", "cancel the {k} on my {c}", "remove my {c} {k}", "remove the {k} on {c}", "delete my {c} {k}",
    "get rid of my {c} {k}", "cancel {c} {k}", "can you cancel my {c} {k}?", "pls cancel my {c} {k}", "take off my {c} {k}",
    "I want to cancel my {c} {k}", "kill my {c} {k}", "drop the {k} on {c}", "scrap my {c} {k}", "I dont want the {c} {k} anymore",
    "cancel the {c} {k} please", "clear my {c} {k}", "pull my {c} {k}", "take the {k} off my {c} position",
    "remove the {k} from my {c} {s}", "cancel that {c} {k}", "hey cancel my {c} {k}", "{c} {k}: cancel it", "cancel my {k} on {c}",
    "get the {c} {k} off", "I need to cancel my {c} {k}", "remove {c} {k}", "delete the {c} {k}", "cancel {c} {k} asap",
    "can u cancel my {k} for {c}", "cancel the {k} I set on {c}", "please remove my {c} {k}", "nix the {c} {k}",
    "cancel my open {c} {k}", "i changed my mind, cancel the {c} {k}", "dont need my {c} {k} any more, cancel it",
    "cancel the {c} {k} for me", "yo cancel the {c} {k}", "can you remove the {k} on my {c} {s}?", "cancel {k} {c}",
]
CANCEL_NOKIND_T = [
    "cancel my {c} order", "remove my {c} order", "delete my {c} order", "cancel the {c} order", "get rid of my {c} order",
    "cancel {c} order", "cancel my {c}", "can you cancel my {c} order?", "I want to cancel my {c} order", "cancel the order on {c}",
    "remove the order on my {c} {s}", "pls cancel my {c} order", "cancel the {c} one", "kill my {c} order", "drop my {c} order",
    "hey cancel my {c} order", "cancel my open {c} order", "i need to cancel my {c} order", "scrap the {c} order",
    "cancel {c} order please", "take off my {c} order", "clear my {c} order", "cancel the order I have on {c}",
    "can u cancel my {c} order", "delete the {c} order", "cancel my order on {c}", "remove {c} order", "cancel my {c} order asap",
]
STOP_REPLY = ["the stop loss", "stop loss", "the stop", "stop", "the sl", "stop loss please", "the stop loss one", "stop loss pls",
              "the stop loss, cancel that", "my stop loss", "stoploss", "the stop one", "cancel the stop", "the stop loss order",
              "the stop-loss", "stop loss one", "it's the stop loss", "the stop pls", "SL", "stop one"]
TP_REPLY = ["the take profit", "take profit", "the tp", "tp", "take profit please", "the take profit one", "take profit pls",
            "the take profit, cancel that", "my take profit", "the take-profit", "take profit one", "it's the take profit",
            "cancel the take profit", "the tp one", "TP", "the tp pls", "take profit order", "tp please", "the take profit order",
            "take-profit"]
LIMIT_REPLY = ["the limit order", "limit order", "the limit", "limit", "the limit one", "limit order please", "my limit order",
               "the limit order, cancel that", "limit one", "it's the limit order", "limit pls"]
ORD_REPLY = ["the first one", "first one", "the first", "1st one", "the first please"]
ORD_REPLY2 = ["the second one", "second one", "the second", "2nd one", "the second please"]
ORD_REPLY3 = ["the third one", "third one", "the third"]
ALL_T = [
    "cancel all my open orders", "cancel all my orders", "cancel everything", "cancel all orders", "clear all my open orders",
    "remove all my open orders", "delete all my orders", "cancel every open order", "cancel all of my limit orders",
    "get rid of all my open orders", "cancel all open orders on hyperliquid", "nuke all my orders", "cancel all my pending orders",
    "can you cancel all my open orders?", "pls cancel all orders", "cancel them all", "cancel all my limit orders",
    "remove every order I have open", "wipe my open orders", "cancel ALL my orders", "i want all my open orders cancelled",
    "cancel all the orders", "cancel the lot", "cancel all of them", "kill all my open orders", "cancel all limit orders pls",
    "clear out my open orders", "cancel all my hyperliquid orders", "cancel all open orders asap", "cancel any open orders I have",
]
ALL_STOPS_T = [
    "cancel all my orders and stops", "cancel all my orders including stops", "remove all orders and stop losses",
    "cancel everything including my stops and take profits", "cancel all open orders, tp and sl too",
    "clear all my orders and stops", "cancel all orders including the stop losses", "cancel every order and stop I have",
]
LIST_T = [
    "what orders do I have", "show my open orders", "what's open on hyperliquid", "list my open orders", "any open orders?",
    "what orders are open", "show me my orders", "which orders do I have open", "what do I have open", "check my open orders",
    "what are my open orders", "what orders and stops do I have", "show me my stops and orders", "list my orders and stops",
    "what's pending on hyperliquid", "orders?", "do I have any orders open", "see my open orders", "what orders are live",
    "open orders pls",
]
ORD_WORDS = {1: ("first", "1st"), 2: ("second", "2nd"), 3: ("third", "3rd")}
CANCEL_ID_T = {
    1: ["cancel the first one", "cancel the first", "remove the first one", "get rid of the first one", "cancel number one",
        "delete the first one", "kill the first one", "cancel the 1st one", "cancel the first one pls", "remove the first"],
    2: ["cancel the second one", "cancel the second", "remove the second one", "get rid of the second one", "cancel number two",
        "delete the second one", "kill the second one", "cancel the 2nd one", "cancel the second one pls", "remove the second"],
    3: ["cancel the third one", "cancel the third", "remove the third one", "get rid of the third one", "cancel number three",
        "delete the third one", "kill the third one", "cancel the 3rd one", "cancel the third one pls", "remove the third"],
    -1: ["cancel the last one", "cancel the last", "remove the last one", "get rid of the last one", "delete the last one",
         "kill the last one", "cancel the last one pls", "remove the last"],
}

ABS_BOTH = [
    "stop at {sl} take profit {tp} on {c}", "set a stop loss at {sl} and a take profit at {tp} on my {c} {s}", "{c} sl {sl} tp {tp}",
    "put a stop at {sl} and tp at {tp} on {c}", "add a stop loss {sl} and take profit {tp} to my {c} position",
    "stop {sl}, take profit {tp} for my {c}", "can you set sl {sl} and tp {tp} on {c}?", "I want a stop at {sl} and take profit at {tp} on my {c}",
    "stop loss {sl} / take profit {tp} on {c}", "set my {c} stop to {sl} and tp to {tp}", "{c}: sl {sl}, tp {tp}",
    "add sl {sl} tp {tp} to my open {c} {s}", "please put stop loss at {sl} and take profit at {tp} on my {c}",
    "on my {c} {s} set the stop at {sl} and the take profit at {tp}", "stoploss {sl} takeprofit {tp} {c}", "{c} {s} stop {sl} tp {tp}",
    "can u add a stop at {sl} and tp {tp} to {c}", "set stop {sl} and take profit {tp} on {c} pls",
    "attach a stop loss at {sl} and a take profit at {tp} to my {c} trade", "need a stop at {sl} plus take profit {tp} on {c}",
    "give my {c} a stop at {sl} and a tp at {tp}", "stop {sl} tp {tp} {c}", "add stop loss {sl} and take profit {tp} on my {c} position please",
    "my open {c} {s} needs a stop at {sl} and a take profit at {tp}", "set {c} exits: stop {sl}, take profit {tp}",
    "stop out my {c} at {sl} and take profit at {tp}", "put sl at {sl} and tp at {tp} on my {c}", "{c} position: stop loss {sl}, take profit {tp}",
]
ABS_SL = [
    "put a stop loss at {sl} on my {c}", "stop at {sl} on {c}", "set a stop on my {c} {s} at {sl}", "{c} stop {sl}",
    "add a stop loss of {sl} to my {c} position", "stop loss {sl} for {c}", "can you put a stop at {sl} on {c}?",
    "i want a stop at {sl} on my {c}", "sl {sl} on {c}", "set my {c} stop to {sl}", "add a stop to my open {c} {s} at {sl}",
    "place a stop loss on {c} at {sl}", "pls set stop {sl} on {c}", "stop out my {c} {s} at {sl}", "protect my {c} {s} with a stop at {sl}",
    "give my {c} a stop at {sl}", "{c} {s}: stop at {sl}", "need a stop loss on {c} at {sl}", "attach a stop at {sl} to my {c}",
    "put my {c} stop at {sl}", "stop loss on my {c} at {sl} please", "add a {sl} stop to {c}", "set {c} sl {sl}",
    "I need a stop on my {c} {s}, {sl}",
]
ABS_TP = [
    "put a take profit at {tp} on my {c}", "take profit at {tp} on {c}", "set a tp on my {c} {s} at {tp}", "{c} tp {tp}",
    "add a take profit of {tp} to my {c} position", "take profit {tp} for {c}", "can you put a take profit at {tp} on {c}?",
    "i want a take profit at {tp} on my {c}", "tp {tp} on {c}", "set my {c} take profit to {tp}", "add a tp to my open {c} {s} at {tp}",
    "place a take profit on {c} at {tp}", "pls set tp {tp} on {c}", "close my {c} {s} with a take profit at {tp}", "attach a take profit at {tp} to my {c}",
    "put my {c} tp at {tp}", "take profit on my {c} at {tp} please", "add a {tp} take profit to {c}", "set {c} tp {tp}",
    "{c} {s}: take profit at {tp}", "need a take profit on {c} at {tp}", "lock a take profit on {c} at {tp}",
]
REP_SL = [
    "move my {c} stop to {sl}", "change my {c} stop loss to {sl}", "update my {c} stop to {sl}", "set my {c} stop at {sl} instead",
    "my stop on {c} should be {sl}", "put the {c} stop at {sl}", "adjust my {c} stop loss to {sl}", "make my {c} stop {sl}",
    "can you change the stop on {c} to {sl}?", "new stop for {c}: {sl}", "shift my {c} stop to {sl}", "reset my {c} stop to {sl}",
    "stop loss on {c} to {sl}", "change the stop on my {c} {s} to {sl}", "i want a different stop on {c}, {sl}", "edit my {c} stop to {sl}",
    "swap my {c} stop for one at {sl}", "set the {c} stop to {sl}",
]
REP_TP = [
    "move my {c} take profit to {tp}", "change my {c} tp to {tp}", "update my {c} take profit to {tp}", "set my {c} tp at {tp} instead",
    "my take profit on {c} should be {tp}", "put the {c} tp at {tp}", "adjust my {c} take profit to {tp}", "make my {c} tp {tp}",
    "can you change the tp on {c} to {tp}?", "new take profit for {c}: {tp}", "shift my {c} take profit to {tp}", "reset my {c} tp to {tp}",
    "take profit on {c} to {tp}", "change the take profit on my {c} {s} to {tp}", "edit my {c} tp to {tp}", "set the {c} take profit to {tp}",
]
REP_BOTH = [
    "change my {c} stop to {sl} and tp to {tp}", "update my {c} stop loss to {sl} and take profit to {tp}",
    "new levels for {c}: stop {sl}, tp {tp}", "move my {c} stop to {sl} and my take profit to {tp}", "reset {c} sl {sl} tp {tp}",
    "change the stop on my {c} {s} to {sl} and the tp to {tp}", "set my {c} stop at {sl} and take profit at {tp} instead",
    "update {c}: stop loss {sl}, take profit {tp}",
]
PCT_SL = [
    "stop {sp} {sd} my entry on {c}", "set a stop loss {sp} {sd} my entry on my {c}", "put my {c} stop {sp} {sd} entry",
    "{c} stop {sp} {sd} my entry price", "add a stop {sp} {sd} where I entered {c}", "stop loss {sp} {sd} entry on {c} {s}",
    "can you put a stop {sp} {sd} my entry on {c}?", "I want a {sp} stop from my entry on {c}", "set a {sp} stop loss from entry on my {c} {s}",
    "{sp} stop from entry on {c}", "put a stop on {c} {sp} {sd} my entry", "stop {sp} {sd} entry for my {c}", "pls put a stop {sp} {sd} my entry on {c}",
    "add a stop loss on {c}, {sp} {sd} entry", "my {c} stop should be {sp} {sd} my entry", "{c}: stop {sp} {sd} entry",
    "set stop {sp} {sd} my {c} entry", "stop loss at {sp} from my entry on {c}",
]
PCT_TP = [
    "take profit {tp} {td} my entry on {c}", "set a take profit {tp} {td} my entry on my {c}", "put my {c} tp {tp} {td} entry",
    "{c} take profit {tp} {td} my entry price", "add a tp {tp} {td} where I entered {c}", "tp {tp} {td} entry on {c} {s}",
    "can you put a take profit {tp} {td} my entry on {c}?", "I want a {tp} take profit from my entry on {c}", "set a {tp} tp from entry on my {c} {s}",
    "{tp} take profit from entry on {c}", "put a take profit on {c} {tp} {td} my entry", "take profit {tp} {td} entry for my {c}",
    "my {c} tp should be {tp} {td} my entry", "{c}: tp {tp} {td} entry", "set tp {tp} {td} my {c} entry",
]
PCT_BOTH = [
    "stop {sp} {sd} my entry and take profit {tp} {td} on {c}", "put a stop loss {sp} {sd} entry and a tp {tp} {td} entry on my {c}",
    "{c}: sl {sp} {sd} entry, tp {tp} {td} entry", "set my {c} stop {sp} {sd} my entry and take profit {tp} {td} it",
    "{c} {s}: stop {sp} {sd} entry, take profit {tp} {td} entry", "add a {sp} stop and a {tp} take profit from my entry on {c}",
    "I want a stop {sp} {sd} my entry and a tp {tp} {td} on {c}", "stop loss {sp} {sd} entry and take profit {tp} {td} entry for {c}, please",
    "protect my {c} {s}: stop {sp} {sd} my entry, tp {tp} {td}",
]
PRICE_SL = [
    "stop {sp} {sd} the current price on {c}", "put a stop loss {sp} {sd} the current price on my {c}", "{c} stop {sp} {sd} the current price",
    "set a stop {sp} {sd} the market price on {c}", "stop loss {sp} {sd} the current {c} price", "add a stop {sp} {sd} the current price on {c}",
]
PRICE_TP = [
    "take profit {tp} {td} the current price on {c}", "put a take profit {tp} {td} the current price on my {c}",
    "{c} tp {tp} {td} the current price", "set a tp {tp} {td} the market price on {c}", "take profit {tp} {td} the current {c} price",
]
BE_NAMED = [
    "move my {c} stop to break-even", "put my stop at breakeven on {c}", "set stop at entry on {c}", "move my {c} stop to entry",
    "break even stop on my {c} {s}", "lock in break-even on {c}", "stop to break even on {c} pls", "can you move my {c} stop to breakeven?",
    "move the stop on my {c} {s} to my entry", "put a stop at my entry price on {c}", "set my {c} stop at break even", "{c}: stop to breakeven",
    "i want my {c} stop at entry", "stop at entry for {c}", "bring my {c} stop to breakeven", "make my {c} stop break even",
    "stop loss to break even on {c}", "move {c} stop to be", "set a break-even stop on my {c}", "stop to my entry on {c}",
]
BE_UNNAMED = [
    "move my stop to break-even", "put my stop at breakeven", "set the stop at entry", "stop to break even pls", "move stop to my entry",
    "break even stop please", "lock in break-even", "move my stop to breakeven", "set a break-even stop",
]
READ_STOP = [
    "do I have a stop on {c}", "is there a stop loss on my {c}", "do I have a stop loss on {c}?", "do I have a stop set on my {c} {s}",
    "check if my {c} has a stop", "does my {c} {s} have a stop", "what's my stop on {c}", "where's my {c} stop loss", "any stop on my {c}?",
    "have I got a stop on {c}", "is there a stop on my {c} position", "stop loss on {c}?", "what stop do I have on {c}", "tell me my {c} stop",
    "did I put a stop on {c}", "is my {c} stop still there",
]
READ_TP = [
    "do I have a take profit on {c}", "is there a take profit on my {c}", "do I have a tp on {c}?", "do I have a take profit set on my {c} {s}",
    "check if my {c} has a take profit", "does my {c} {s} have a tp", "what's my take profit on {c}", "where's my {c} tp", "any tp on my {c}?",
    "have I got a take profit on {c}", "is there a tp on my {c} position", "take profit on {c}?",
]
READ_PROT = [
    "which of my positions have a stop", "what's my protection", "do all my positions have stops", "which positions have no stop loss",
    "show my stops and take profits", "what stops do I have", "list my stop losses", "do any of my positions have a take profit",
    "what tp and sl do I have", "check my stops", "do my positions have stops", "what stop losses are set",
]
READ_COIN_ORD = [
    "what orders do I have on {c}", "any open orders on {c}?", "show my {c} orders", "what's open on my {c}", "orders on {c}?",
    "anything set on my {c} {s}", "what stops or orders do I have on {c}", "list my {c} orders", "do I have orders on {c}",
    "what's pending on {c}",
]
TPSL_ASK_SL = [
    "put a stop on my {c}", "add a stop loss to my {c} position", "I want a stop on my {c} {s}", "set a stop loss on {c}",
    "add a stop to my {c}", "pls put a stop on {c}", "can you add a stop loss to my {c}?", "stop loss on my open {c}",
    "protect my {c} {s} with a stop", "I need a stop on {c}", "give my {c} a stop loss", "attach a stop to my {c} trade",
]
TPSL_ASK_TP = [
    "put a take profit on my {c}", "add a tp to my {c} position", "I want a take profit on my {c} {s}", "set a take profit on {c}",
    "add a take profit to my {c}", "pls put a tp on {c}", "can you add a take profit to my {c}?", "take profit on my open {c}",
]
ASK_Q_SL = ["What price should the stop go at? You can also give a percent from your entry.",
            "At what price do you want the stop loss? A percent under your entry works too.",
            "What stop price? You can say a number or a percent from your entry.",
            "Which price for the stop? A percent from your entry is fine too.",
            "What price should I use for the stop loss, or what percent from your entry?",
            "Tell me the stop price, or a percent from your entry, and I'll prepare it."]
ASK_Q_TP = ["What price should the take profit go at? You can also give a percent from your entry.",
            "At what price do you want the take profit? A percent above your entry works too.",
            "What take profit price? You can say a number or a percent from your entry.",
            "Which price for the take profit? A percent from your entry is fine too.",
            "What price should I use for the take profit, or what percent from your entry?",
            "Tell me the take profit price, or a percent from your entry, and I'll prepare it."]
MARGIN_T = [
    "how much margin am I using", "what's my available usdc on hyperliquid", "how much is in my hyperliquid account",
    "what's my hl balance", "how much can I still trade with", "what's my free margin", "how much usdc do I have free on hyperliquid",
    "what's my account value on hyperliquid", "how much margin do I have left", "how much is my hyperliquid account worth",
    "do I have margin left for another trade", "show my hyperliquid margin",
]
SWAP_CANCEL = ["cancel my swap", "cancel the swap", "cancel that", "cancel it", "nvm cancel the swap", "actually cancel that swap",
               "cancel swap", "abort the swap", "scrap that swap", "never mind, cancel it", "cancel the bridge", "wait cancel that",
               "stop, cancel the swap", "changed my mind, cancel it", "cancel this swap pls", "undo the swap"]
SEND_CANCEL = ["cancel the send", "cancel that", "cancel my transfer", "cancel the payment", "abort the send", "nvm cancel it",
               "cancel the transaction", "cancel it", "wait, cancel that send", "cancel this", "scrap the send",
               "changed my mind, cancel the send", "cancel the transfer pls", "stop the send"]
TRADE_CANCEL = ["cancel the trade", "cancel my order", "cancel that", "cancel it", "nevermind, cancel the trade", "scrap the trade",
                "cancel this order", "wait cancel the trade", "cancel the {s}", "abort the trade", "cancel the {c} trade",
                "changed my mind, cancel it", "cancel this trade pls", "undo the trade"]
CANCEL_NOTHING = ["cancel", "cancel that", "cancel it", "nvm cancel", "cancel please", "scrap it", "never mind, cancel",
                  "cancel the order", "cancel this", "stop, cancel", "cancel everything pending", "abort", "cancel my request",
                  "cancel what you just did", "i said cancel", "actually cancel", "cancel the thing", "cancel that one",
                  "undo", "nvm scrap it"]
SHORT_STOP_T = [
    "{w} {c} {usd} {lev}x with a stop at {sl}", "open a {c} {w} {usd} usd {lev}x, stop loss {sl}", "{w} {c} ${usd} at {lev}x stop {sl}",
    "{c} {w}, {usd} dollars, {lev}x leverage, stop loss at {sl}", "i want to {w} {c} with {usd} {lev}x and a stop at {sl}",
    "put on a {c} {w} for {usd} at {lev}x with sl {sl}", "{w} {c} {usd}$ {lev}x sl {sl}", "can you {w} {c} ${usd}, {lev}x, stop {sl}?",
    "{w} {usd} of {c} {lev}x and set the stop at {sl}", "new {c} {w}: ${usd}, {lev}x, stop at {sl}", "go {w} {c} ${usd} {lev}x, stop loss {sl}",
    "{w} {c} market ${usd} {lev}x stop loss {sl}", "open {c} {w} size {usd} lev {lev} stop {sl}", "{w} {c} {usd} usd {lev}x, protect it with a stop at {sl}",
]
SHORT_STOP_TP_T = [
    "{w} {c} ${usd} {lev}x, stop {sl}, take profit {tp}", "open a {c} {w} {usd} usd {lev}x with sl {sl} and tp {tp}",
    "{w} {c} {usd} {lev}x stop loss {sl} take profit {tp}", "{c} {w}, ${usd}, {lev}x, stop at {sl}, take profit at {tp}",
    "i want to {w} {c} with {usd} {lev}x, a stop at {sl} and a tp at {tp}", "put on a {c} {w} for ${usd} at {lev}x, sl {sl}, tp {tp}",
    "new {c} {w}: {usd} usd, {lev}x, stop {sl}, take profit {tp}", "can you {w} {c} ${usd} {lev}x with stop {sl} and tp {tp}?",
]
CLOSE_FULL = [
    "close my {c} position", "close my {c} {s}", "close {c}", "exit my {c} {s}", "get me out of {c}", "close the {c} trade",
    "close out my {c} {s}", "exit {c}", "close my {c} perp", "i want out of {c}", "flatten my {c}", "close my {c} on hyperliquid",
    "can you close my {c} {s}?", "pls close my {c} position", "shut my {c} {s}", "take me out of my {c} trade", "close my entire {c} {s}",
    "close the whole {c} position", "exit the {c} position", "end my {c} {s}", "need to get out of my {c} {s} asap", "{c}: close",
]
CLOSE_PART = ["close {p} of my {c} {s}", "close {p} of my {c} position", "take {p} off my {c} {s}", "trim my {c} {s} by {p}",
              "reduce my {c} position by {p}", "close {p} of {c}", "partial close on {c}, {p}", "scale out of {c} {s}, {p}"]
PCT_CHOICES = [("half", 50), ("a quarter", 25), ("50%", 50), ("25%", 25), ("75%", 75), ("30%", 30), ("10%", 10), ("20%", 20)]

HOW = {
    "stop_hits": {
        "q": ["what happens when my stop hits", "what happens when my stop loss triggers", "when my stop loss is hit what happens",
              "how does a stop loss work", "what does my stop do when price reaches it", "explain how a stop loss works",
              "what happens at my stop price", "so what actually happens when the stop gets hit", "how does my stop get filled",
              "what happens if price touches my stop loss", "stop loss: what happens when it triggers?", "how does a stop loss order work on hyperliquid"],
        "a": ["When the price reaches your stop, a market order closes the position. In a fast move it can fill worse than the stop price, so the loss can be bigger than planned.",
              "Your stop is a trigger. Once the price touches it, a market order closes the position at the next available price. In a fast market that price can be worse than the stop.",
              "At your stop price a market order fires and closes the position. The fill can land past the stop when the market moves quickly, so the loss may be larger than you set."]},
    "tp_hits": {
        "q": ["what happens when my take profit hits", "how does a take profit work", "what does my take profit do when price reaches it",
              "explain take profit", "when my tp triggers what happens", "what happens at my take profit price",
              "how does a tp order get filled", "what happens if price touches my take profit", "take profit: what happens when it triggers?",
              "how does a take profit order work on hyperliquid"],
        "a": ["When the price reaches your take profit, a market order closes the position at about that price and you keep the gain, minus fees. In a fast market the fill can differ a little.",
              "A take profit is a trigger too. Once the price touches it, a market order closes your position, so you bank the gain minus fees. The fill can be slightly off the trigger.",
              "At your take profit price a market order fires and closes the position. You keep the gain minus fees, and the fill can be a touch different from the trigger."]},
    "stop_whole": {
        "q": ["does a stop loss close my whole position", "will my stop close the full position", "does the stop cover all of my position",
              "is the stop for the whole position or part of it", "does my take profit close everything", "do stops close all of it or just some",
              "does the tp cover my entire position", "will the stop only close part of my position"],
        "a": ["The stops and take profits I prepare cover your whole position and are reduce only, so they can only close it, never open a new one.",
              "Yes. The ones I prepare cover the whole position and are reduce only. They can shrink or close it, never add to it or flip it."]},
    "guarantee": {
        "q": ["is a stop loss guaranteed", "can my stop loss fail", "will my stop always fill at my price", "can I lose more than my stop",
              "is a stop guaranteed to fill at the price I set", "does a stop loss guarantee my exit price", "can price gap past my stop",
              "is my stop loss a guarantee"],
        "a": ["No. A stop turns into a market order when it triggers, so the fill can land past your price. If the market gaps, you can lose more than the stop level suggests.",
              "Not at your exact price. Once triggered it is a market order, so in a fast or thin market it can fill worse. The stop limits the damage but it does not fix the exit price."]},
    "reduce_only": {
        "q": ["what does reduce only mean", "what's a reduce only order", "why are stops reduce only", "explain reduce only",
              "reduce only meaning?", "what is reduce-only on a stop", "why do stops say reduce only", "does reduce only mean anything for my stop"],
        "a": ["A reduce-only order can only shrink or close a position. It can never open one or flip you to the other side, which is why stops and take profits use it.",
              "Reduce only means the order can only make your position smaller. It can't add to it or open the opposite side, so a stop can't turn into a new trade."]},
    "stop_vs_liq": {
        "q": ["should my stop be above my liquidation price", "where should I put my stop relative to liquidation", "what if my stop is below my liquidation price",
              "does liquidation come before my stop", "stop loss vs liquidation, which hits first", "can I get liquidated before my stop triggers",
              "how far from liquidation should my stop be", "is my stop safe from liquidation"],
        "a": ["For a long, keep the stop above your liquidation price, or liquidation hits first and takes your margin. For a short, keep it below. I can check yours if you want.",
              "The stop has to trigger before liquidation to help. Longs need it above the liquidation price, shorts below. If the price jumps past both, liquidation can still come first."]},
    "need_stop": {
        "q": ["do I need a stop loss", "should I always use a stop", "is a stop loss worth it", "do I really need a stop on every trade",
              "should I put a stop on my trades", "are stop losses a good idea", "why use a stop loss at all", "stop or no stop?"],
        "a": ["It's your call. A stop limits how much one trade can lose, but it can trigger on a quick wick and it can slip. I can prepare one if you give me a price or a percent from your entry.",
              "A stop caps the loss on a trade so one bad move cannot drain the account, but a sharp wick can trigger it and the fill can slip. Tell me a price and I'll prepare one."]},
    "sl_vs_tp": {
        "q": ["what's the difference between a stop loss and a take profit", "stop loss vs take profit", "difference between sl and tp",
              "what is a stop loss and what is a take profit", "tp vs sl what's the difference", "explain stop loss and take profit",
              "what do sl and tp mean", "stop loss or take profit, what's the difference"],
        "a": ["A stop loss closes the position if the price moves against you. A take profit closes it when the price reaches your target in your favour. Both are triggers that send a market order.",
              "Stop loss: exits when the price goes against you, to cap the loss. Take profit: exits when the price reaches your target, to lock the gain. Both fire a market order."]},
}

# ------------------------------------------------------------------ text helpers
def pick(rng, pool, split, side=None):
    use = CP.pool_for(pool, split, rng)
    for _ in range(80):
        t = pool[rng.choice(use)]
        tag, text = (t.split("|", 1) if re.match(r"^[LS]\|", t) else ("", t))
        if tag == "L" and side != "long":
            continue
        if tag == "S" and side != "short":
            continue
        return text
    raise RuntimeError("no template")


def fill(rng, tpl, c=None, s=None, side_prob=0.75, **kw):
    sd = s if (s and rng.random() < side_prob) else ""
    return tpl.format(c=coin_text(rng, c) if c else "", s=sd, **kw)


PROTECT = set(WORD2COIN) | {"stop", "loss", "stoploss", "take", "profit", "takeprofit", "limit", "order", "orders", "long", "short",
                            "entry", "percent", "half", "first", "second", "third", "last", "both", "break", "even", "breakeven",
                            "below", "under", "above", "over", "price", "current", "market", "position", "cancel", "remove",
                            "delete", "swap", "send", "trade", "leverage", "one", "dollars", "usd", "open", "cover", "stops", "every", "everything", "wipe", "nuke", "clear", "them", "lot", "any", "quarter", "halve", "percent"}


def typo_layer(rng, text):
    words = text.split(" ")
    cand = [i for i, w in enumerate(words)
            if len(re.sub(r"\W", "", w)) >= 4 and not re.search(r"\d", w) and re.sub(r"\W", "", w).lower() not in PROTECT]
    for _ in range(4):
        op = rng.choice(["swap", "drop", "double", "abbr"])
        if op == "abbr":
            for a, b in (("please", rng.choice(["pls", "plz"])), ("pls", "plz"), ("right now", "rn"), ("you", "u")):
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


def pick_bio(rng):
    return "Face ID" if rng.random() < 0.7 else "Touch ID"


def approve(rng):
    return rng.choice(["Approve with Face ID.", "Approve it with Face ID when you're ready.", "Approve it with Face ID or cancel the review.",
                       "It's waiting for your Face ID.", "Check it and approve with Face ID.", "Review it and approve with Face ID.",
                       "Approve with Face ID if it looks right."])


def moved(rng):
    return rng.choice(["Nothing has moved yet.", "Nothing has moved until you approve.", "Nothing has moved yet, it's only prepared.",
                       "", "", ""])


def other_side_word(side):
    return "short" if side == "long" else "long"


def pct_text(rng, p):
    ps = dfmt(p)
    return f"{ps}%" if rng.random() < 0.85 else f"{ps} percent"


def device_of(err):
    m = re.search(r"this (iPhone|Mac)", err)
    return m.group(1) if m else "iPhone"


# ------------------------------------------------------------------ answers
def cancel_card_text(rng, r):
    """Answer for a card_opened cancel result."""
    what = r["cancelling"]
    lead = rng.choice([f"I've put up a card to cancel your {what}.", f"I've put up a card to cancel your {what}.",
                       f"A card is up to cancel your {what}.", f"I've put up a card for cancelling your {what}.",
                       f"I've put up a card for your {what}: it asks you to confirm the cancel."])
    if r["cancelling"].endswith(")") and "open limit order" in what:
        lead = rng.choice([f"I've put up a card to cancel your {what}.", f"I've put up a card to cancel {what}.",
                           f"A card is up to cancel {what}."])
    tap = rng.choice(["Tap Confirm if you want it gone.", "Tap Confirm to go ahead.", "Tap Confirm to remove it.", "Tap Confirm if that's right."])
    if "open limit order" in what:
        tap = rng.choice(["Tap Confirm to go ahead.", "Tap Confirm if that's right.", "Tap Confirm to clear them." if " orders" in what else "Tap Confirm to clear it."])
    warn = (" " + r["warning"]) if r.get("warning") else ""
    tail = rng.choice(["Nothing is cancelled until you tap.", "Nothing is cancelled yet.", "Nothing is cancelled until you tap Confirm."])
    return f"{lead} {tap}{warn} {tail}"


def tpsl_text(rng, r, sl_expl="", tp_expl="", extra=""):
    poslong, possize = r["position"].split(", ")
    items = []
    if "stop_loss" in r:
        items.append(f"a stop loss at {money(r['stop_loss'])}{sl_expl}")
    if "take_profit" in r:
        items.append(f"a take profit at {money(r['take_profit'])}{tp_expl}")
    what = " and ".join(items)
    t = rng.choice([f"I've prepared {what} on your {poslong}, reduce only.",
                    f"I've prepared {what} for your {poslong} ({possize}), reduce only.",
                    f"I've prepared {what} on your {poslong} of {possize}, reduce only.",
                    f"Prepared: {what} on your {poslong}, reduce only.",
                    f"I've prepared {what} on your {poslong}. It's reduce only."])
    if r.get("replaces"):
        t += " " + rng.choice([f"It replaces your {r['replaces']}.", f"This replaces your {r['replaces']}.",
                               f"That swaps out your {r['replaces']}."])
    if extra:
        t += " " + extra
    t += " " + approve(rng)
    m = moved(rng)
    return t + (" " + m if m else "")


def card_pointer_text(rng, what):
    if what == "cancel":
        return rng.choice([
            "Order tools aren't available in this chat. I put a button below that opens your Trade positions, where you can manage the order.",
            "Cancelling orders isn't available in this chat. I put a button below to your Trade positions instead.",
            "I don't have the order tools in this chat. I put a button below that opens your Trade positions.",
            "Order changes aren't available in this chat. I put a button below to your Trade positions, so you can do it there.",
            "That tool isn't available in this chat. I put a button below that opens Trade, positions tab.",
            "The order tools aren't loaded here. I put a button below to your Trade positions."])
    return rng.choice([
        "Stops and take profits aren't available in this chat. I put a button below that opens your Trade positions.",
        "I don't have the stop and take profit tools in this chat. I put a button below to your Trade positions.",
        "Setting a stop isn't available in this chat. I put a button below that opens Trade, positions tab, so you can add it there.",
        "That tool isn't available in this chat. I put a button below to your Trade positions.",
        "The stop and take profit tools aren't loaded here. I put a button below that opens your Trade positions.",
        "I can't prepare that from this chat. I put a button below to your Trade positions."])


def trig(r, coin, kind):
    return next((o for o in r["perps"]["open_orders"] if o["coin"] == coin and o["kind"] == kind), None)


def trig_phrase(o, p):
    s = f"a {KIND_WORD[o['kind']]} at {money(o['price'])}"
    if o.get("whole_position") is False and p:
        s += f" that covers {short(o['size'])} of your {short(p['size'])} {p['coin']}"
    return s


def limit_phrase(o):
    return f"a {o['side']} limit for {short(o['size'])} {o['coin']} at {money(o['price'])}"


def protection_sentence(r, coins=None):
    """'your SOL long has a stop loss at $128 and a take profit at $170; your ETH short has a stop loss at ...'"""
    parts = []
    for p in r["perps"]["positions"]:
        if coins and p["coin"] not in coins:
            continue
        ts = [o for o in r["perps"]["open_orders"] if o["coin"] == p["coin"] and o["kind"] != "limit"]
        ts.sort(key=lambda o: 0 if o["kind"] == "stop_loss" else 1)
        if ts:
            parts.append(f"your {p['coin']} {p['side']} has " + join_and([trig_phrase(o, p) for o in ts]))
    return "; ".join(parts)


def read_answer(rng, qtype, r, coin):
    pos = {p["coin"]: p for p in r["perps"]["positions"]}
    rows = r["perps"]["open_orders"]
    lim = [o for o in rows if o["kind"] == "limit"]
    if qtype in ("stop_on_coin", "tp_on_coin"):
        want, other = ("stop_loss", "take_profit") if qtype == "stop_on_coin" else ("take_profit", "stop_loss")
        w, o2 = trig(r, coin, want), trig(r, coin, other)
        p = pos.get(coin)
        if p is None:
            lm = next((o for o in lim if o["coin"] == coin), None)
            t = f"You have no open {coin} position on Hyperliquid."
            if lm:
                t += f" There is {limit_phrase(lm)}, but that's an order, not a {KIND_WORD[want]}."
            return t
        if w:
            t = f"Yes. Your {coin} {p['side']} has {trig_phrase(w, p)}"
            if o2:
                t += f" and {trig_phrase(o2, p)}"
            return t + "."
        t = f"No. Your {coin} {p['side']} has no {KIND_WORD[want]}"
        if o2:
            t += f", only {trig_phrase(o2, p)}"
        t += ". "
        if want == "stop_loss":
            t += rng.choice(["Tell me a price and I'll prepare one.", "Want me to prepare one?", "Give me a price or a percent from your entry and I'll prepare it."])
        else:
            t += rng.choice(["Tell me a price and I'll prepare one.", "Want me to prepare one?", "Give me a price or a percent from your entry and I'll prepare it."])
        return t
    if qtype in ("orders_all", "coin_orders"):
        if qtype == "coin_orders":
            lim = [o for o in lim if o["coin"] == coin]
            cs = {coin}
        else:
            cs = None
        prot = protection_sentence(r, cs)
        parts = []
        if lim:
            parts.append("Open orders: " + join_and([limit_phrase(o) for o in lim]) + ".")
        else:
            parts.append(f"You have no open {coin + ' ' if qtype == 'coin_orders' else ''}limit orders.")
        if prot:
            parts.append("Protection: " + prot + ".")
        elif pos and (qtype == "orders_all" or coin in pos):
            parts.append("No stop loss or take profit is set." if qtype == "orders_all" else f"Your {coin} {pos[coin]['side']} has no stop loss or take profit.")
        if not lim and not prot and not pos:
            return "You have no open orders or positions on Hyperliquid." if qtype == "orders_all" else f"You have nothing open on {coin} on Hyperliquid."
        return " ".join(parts)
    if qtype == "protection_all":
        parts = []
        for p in r["perps"]["positions"]:
            ts = [o for o in rows if o["coin"] == p["coin"] and o["kind"] != "limit"]
            ts.sort(key=lambda o: 0 if o["kind"] == "stop_loss" else 1)
            if ts:
                parts.append(f"your {p['coin']} {p['side']} has " + join_and([trig_phrase(o, p) for o in ts]))
            else:
                parts.append(f"your {p['coin']} {p['side']} has no stop loss or take profit")
        if not parts:
            return "You have no open positions on Hyperliquid, so there is nothing to put a stop on."
        t = rng.choice(["Here is where you stand: ", "Right now, ", "Checking your positions, "]) + "; ".join(parts) + "."
        return t[0].upper() + t[1:] if t[0].islower() else t
    raise ValueError(qtype)


# ------------------------------------------------------------------ conversation helper
class OConv(BA.Conv):
    def __init__(self, rng, world, pack_name):
        super().__init__(rng, world, system=SYSTEM_FACE)
        self.pack_name = pack_name
        self.turns = []          # (index of user message, expect list of that turn)
        self.cur_u = None
        self.turn_gold = []

    def user(self, text):
        if self.cur_u is not None:
            self.turns.append((self.cur_u, self.expect))
        super().user(text)
        self.cur_u = len(self.msgs) - 1

    def finish(self):
        if self.cur_u is not None:
            self.turns.append((self.cur_u, self.expect))
            self.cur_u = None


def apply_bio(ex):
    bio = ex["meta"]["bio"]
    if bio == "Face ID":
        return ex
    for m in ex["messages"]:
        if isinstance(m.get("content"), str):
            m["content"] = m["content"].replace("Face ID", bio)
    return ex


class Retry(Exception):
    pass


def rec(c, kind, bio, gold, checks, slots, seq, turn_golds=None, family_note=None):
    """Build the record. seq = the expected tool-name sequence of the whole conversation (written by the scenario)."""
    c.finish()
    calls = [tc for m in c.msgs for tc in (m.get("tool_calls") or [])]
    names = [tc["name"] for tc in calls]
    texts = [m["content"] for m in c.msgs if m["role"] == "assistant" and m.get("content")]
    pack_name = c.pack_name
    lite_ok = (pack_name != "core+solana" and set(names) <= LITE_TOOLS and len(names) <= 3
               and all(len(t.split()) <= 45 for t in texts))
    gold = dict(gold)
    gold["steps"] = c.turns[-1][1]
    ex = {"messages": c.msgs, "tools": True,
          "meta": {"source": GEN_NAME, "generator": GEN_NAME, "kind": kind, "bio": bio, "lang": "en",
                   "tools_registry": "spec/hermit_tools_v3.json", "system": "spec/SYSTEM_AGENT_v3.txt",
                   "pack": PACKS[pack_name], "pack_name": pack_name, "family_ok": ["4b", "lite"] if lite_ok else ["4b"],
                   "steps": [[tc["name"] for tc in e["calls"]] if "calls" in e else ("ask" if e.get("ask") else "text") for e in c.turns[-1][1]],
                   "cards": [n for n in names if n in CARD_TOOLS],
                   "script": [m["content"] for m in c.msgs if m["role"] == "user"][1:],
                   "world": c.world0, "world_final": c.w.state(), "expect": c.turns[-1][1], "gold": gold,
                   "checks": checks, "slots": slots, "seq": seq,
                   "turns": [{"user_index": u, "expect": e} for u, e in c.turns],
                   "turn_golds": turn_golds or [gold]}}
    return ex


def absent_pointer(rng, c, bio, kind, what, gold_extra, slots):
    c.call(("open_page", {"page": "trade", "tab": "positions"}))
    c.say(card_pointer_text(rng, what))
    gold = {"calls": [{"name": "open_page", "arguments": {"page": "trade", "tab": "positions"}}],
            "forbidden_tools": ["cancel_order", "set_tp_sl", "prepare_trade", "prepare_close"], "absent_tool": True}
    gold.update(gold_extra)
    return rec(c, kind, bio, gold, ["right_tool", "exact_digits", "no_card_done", "restraint"], dict(slots, absent=True),
               ["open_page"])


FORB_ORDERS = ["prepare_trade", "prepare_close"]


# ------------------------------------------------------------------ scenarios: cancel
def sc_cancel_kind(rng, split, pack):
    bio = pick_bio(rng)
    flavor = rng.choices(["stop_loss", "take_profit", "order"], [4, 3, 3])[0]
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 0, 1, 2]))
    if flavor == "order":
        w.add_limit(rng, coin)
        if rng.random() < 0.5:
            w.add_trigger(rng, coin, "stop_loss")
        if rng.random() < 0.3:
            w.add_trigger(rng, coin, "take_profit")
        o = next(x for x in w.orders if x["kind"] == "limit" and x["coin"] == coin)
        words = ["limit order", "limit order", "limit", f"{o['side']} limit", f"limit {o['side']}"]
        kw = rng.choice(words)
        kind_arg = "order"
    else:
        w.add_trigger(rng, coin, flavor)
        other = "take_profit" if flavor == "stop_loss" else "stop_loss"
        if rng.random() < 0.5:
            w.add_trigger(rng, coin, other)
        if rng.random() < 0.3:
            w.add_limit(rng, coin)
        kw = rng.choice(["stop loss", "stop loss", "stop", "stop-loss", "stoploss", "SL"] if flavor == "stop_loss"
                        else ["take profit", "take profit", "tp", "take-profit", "TP"])
        kind_arg = flavor
    add_noise(rng, w, coin)
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, pick(rng, CANCEL_KIND_T, split), c=coin, s=side, k=kw)))
    slots = {"coin": coin, "side": side, "kind_arg": kind_arg}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "cancel_kind", "cancel", {}, slots)
    r = c.call(("cancel_order", {"coin": coin, "kind": kind_arg}))
    if r.get("status") != "card_opened":
        raise Retry
    c.say(cancel_card_text(rng, r))
    gold = {"calls": [{"name": "cancel_order", "arguments": {"coin": coin, "kind": kind_arg}}], "forbidden_tools": FORB_ORDERS,
            "omit_args": {"cancel_order": ["order_id"]}, "answer_contains": ["Tap Confirm"] + ([r["warning"]] if r.get("warning") else [])}
    return rec(c, "cancel_kind", bio, gold, ["right_tool", "exact_digits", "arg_omission", "no_card_done", "warnings_repeated"], slots,
               ["cancel_order"])


def sc_cancel_nokind(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    single = rng.random() < 0.2
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 0, 1, 2]))
    if single:
        w.add_trigger(rng, coin, rng.choice(["stop_loss", "take_profit"]))
    else:
        w.add_trigger(rng, coin, "stop_loss")
        w.add_trigger(rng, coin, "take_profit")
        if rng.random() < 0.25:
            w.add_limit(rng, coin)
    add_noise(rng, w, coin)
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, pick(rng, CANCEL_NOKIND_T, split), c=coin, s=side)))
    slots = {"coin": coin, "side": side, "single": single}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "cancel_noKind", "cancel", {}, slots)
    r = c.call(("cancel_order", {"coin": coin}))
    first_gold = {"calls": [{"name": "cancel_order", "arguments": {"coin": coin}}], "forbidden_tools": FORB_ORDERS,
                  "omit_args": {"cancel_order": ["kind", "order_id"]}}
    if single:
        if r.get("status") != "card_opened":
            raise Retry
        c.say(cancel_card_text(rng, r))
        first_gold["answer_contains"] = ["Tap Confirm"] + ([r["warning"]] if r.get("warning") else [])
        return rec(c, "cancel_noKind", bio, first_gold, ["right_tool", "exact_digits", "arg_omission", "no_card_done", "warnings_repeated"],
                   slots, ["cancel_order"])
    if r.get("status") != "needs_choice":
        raise Retry
    q = r["question"]
    c.say(rng.choice([q, q, f"You have more than one open on {coin}. {q}", f"{q} Tell me which and I'll put up a card.",
                      f"I found a few. {q}"]), ask=True)
    first_gold["answer_contains"] = [q]
    first_gold["asks"] = True
    first_gold["calls"] = [{"name": "cancel_order", "arguments": {"coin": coin}}]
    # the user answers
    opts = [o["kind"] for o in w.orders if o["coin"] == coin]
    chosen = rng.choice(opts)
    if rng.random() < 0.12 and len(opts) >= 2 and chosen in opts[:3]:
        idx = opts.index(chosen) + 1
        reply = rng.choice({1: ORD_REPLY, 2: ORD_REPLY2, 3: ORD_REPLY3}[idx])
        by_ordinal = True
    else:
        pool = {"stop_loss": STOP_REPLY, "take_profit": TP_REPLY, "limit": LIMIT_REPLY}[chosen]
        reply = pick(rng, pool, split)
        by_ordinal = False
    c.user(finish_text(rng, reply))
    kind_arg = "order" if chosen == "limit" else chosen
    r2 = c.call(("cancel_order", {"coin": coin, "kind": kind_arg}))
    if r2.get("status") != "card_opened":
        raise Retry
    c.say(cancel_card_text(rng, r2))
    gold = {"calls": [{"name": "cancel_order", "arguments": {"coin": coin, "kind": kind_arg}}], "forbidden_tools": FORB_ORDERS,
            "omit_args": {"cancel_order": ["order_id"]}, "answer_contains": ["Tap Confirm"] + ([r2["warning"]] if r2.get("warning") else [])}
    slots.update({"chosen": kind_arg, "by_ordinal": by_ordinal, "options": opts})
    return rec(c, "cancel_noKind", bio, gold, ["right_tool", "exact_digits", "arg_omission", "no_card_done", "warnings_repeated"],
               slots, ["cancel_order", "cancel_order"], turn_golds=[first_gold, gold])


def sc_cancel_all(rng, split, pack):
    bio = pick_bio(rng)
    n_lim = rng.choice([1, 2, 2, 3])
    coins = pick_coins(rng, split, n_lim + 1, [])
    w = make_world(rng, split, bio, must=(coins[0], rng.choice(["long", "short"]), None), extra=rng.choice([0, 1]))
    for cn in coins[:n_lim]:
        w.add_limit(rng, cn)
    if rng.random() < 0.6:
        w.add_trigger(rng, coins[0], "stop_loss")
        if rng.random() < 0.5:
            w.add_trigger(rng, coins[0], "take_profit")
    with_stops = rng.random() < 0.2
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, pick(rng, ALL_STOPS_T if with_stops else ALL_T, split)))
    slots = {"n_limit": n_lim, "with_stops": with_stops}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "cancel_all", "cancel", {}, slots)
    r = c.call(("cancel_order", {"coin": "all"}))
    if r.get("status") != "card_opened":
        raise Retry
    lead = rng.choice([f"I've put up a card to cancel {r['cancelling']}.", f"A card is up to cancel {r['cancelling']}.",
                       f"I've put up a card for {r['cancelling']}."])
    tap = rng.choice(["Tap Confirm to go ahead.", "Tap Confirm if that's right.", "Tap Confirm to clear them." if n_lim > 1 else "Tap Confirm to clear it."])
    excl = rng.choice(["Stop losses and take profits are not included.", "Stop losses and take profits are not part of this.",
                       "This does not touch stop losses or take profits."])
    if with_stops:
        excl += " " + rng.choice(["Tell me which stop or take profit to remove and I'll put up a card for it.",
                                  "Say which stop or take profit to remove and I'll do it one at a time."])
    tail = rng.choice(["Nothing is cancelled until you tap.", "Nothing is cancelled yet."])
    c.say(f"{lead} {tap} {excl} {tail}")
    gold = {"calls": [{"name": "cancel_order", "arguments": {"coin": "all"}}], "forbidden_tools": FORB_ORDERS,
            "omit_args": {"cancel_order": ["kind", "order_id"]}, "answer_contains": ["Tap Confirm", "not included|not part of this|does not touch"]}
    return rec(c, "cancel_all", bio, gold, ["right_tool", "exact_digits", "arg_omission", "no_card_done", "restraint"], slots,
               ["cancel_order"])


def order_line(o):
    return order_label(o)


def sc_cancel_id(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 1]))
    w.add_trigger(rng, coin, "stop_loss")
    w.add_trigger(rng, coin, "take_profit")
    lim_coin = rng.choice([x for x in ["ETH", "BTC", "SOL", "HYPE"]])
    w.add_limit(rng, lim_coin)
    if rng.random() < 0.4:
        others = [p["coin"] for p in w.hl_positions if p["coin"] != coin]
        if others:
            w.add_trigger(rng, others[0], "stop_loss")
    rng.shuffle(w.orders)
    w.orders = w.orders[:4]
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, pick(rng, LIST_T, split)))
    slots = {"coin": coin}
    if pack == "core+solana":
        raise Retry          # get_positions exists in this pack; the absent case is covered by the cancel kinds
    r = c.call(("get_positions", {"kind": "perps"}))
    rows = r["perps"]["open_orders"]
    if len(rows) < 2:
        raise Retry
    lines = [f"{i + 1}. {order_label(o)}" for i, o in enumerate(w.orders)]
    c.say(rng.choice(["Here is what is open:", "These are your open items:", "You have these open:", "Open on Hyperliquid right now:"]) + "\n" + "\n".join(lines))
    which = rng.choice([1, 2, 2, 3, -1])
    if which == 3 and len(rows) < 3:
        which = 2
    idx = which - 1 if which > 0 else len(rows) - 1
    c.user(finish_text(rng, pick(rng, CANCEL_ID_T[which], split)))
    row = rows[idx]
    args = {"coin": row["coin"], "order_id": row["id"]}
    r2 = c.call(("cancel_order", args))
    if r2.get("status") != "card_opened":
        raise Retry
    c.say(cancel_card_text(rng, r2))
    gold = {"calls": [{"name": "cancel_order", "arguments": args}], "forbidden_tools": FORB_ORDERS,
            "omit_args": {"cancel_order": ["kind"]}, "answer_contains": ["Tap Confirm"] + ([r2["warning"]] if r2.get("warning") else [])}
    slots.update({"which": which, "idx": idx})
    return rec(c, "cancel_id", bio, gold, ["right_tool", "exact_digits", "arg_omission", "no_card_done", "warnings_repeated"], slots,
               ["get_positions", "cancel_order"])


# ------------------------------------------------------------------ scenarios: set_tp_sl
def valid_levels(rng, w, coin, want_sl, want_tp):
    """(sl, tp) typed-style prices on the right side of the mark."""
    p = w.pos(coin)
    mark, long_ = p["mark"], p["side"] == "long"
    sl = tp = None
    if want_sl:
        sl = nice(rng, mark * (1 - rng.uniform(0.04, 0.2)) if long_ else mark * (1 + rng.uniform(0.04, 0.2)))
    if want_tp:
        tp = nice(rng, mark * (1 + rng.uniform(0.05, 0.4)) if long_ else mark * (1 - rng.uniform(0.05, 0.3)))
    if sl is not None and ((long_ and sl >= mark) or (not long_ and sl <= mark)):
        raise Retry
    if tp is not None and ((long_ and tp <= mark) or (not long_ and tp >= mark)):
        raise Retry
    return sl, tp


def sc_tpsl_abs(rng, split, pack):
    bio = pick_bio(rng)
    flavor = rng.choices(["both", "sl", "tp"], [45, 30, 25])[0]
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 0, 1, 2]))
    add_noise(rng, w, coin, 0.2)
    sl, tp = valid_levels(rng, w, coin, flavor in ("both", "sl"), flavor in ("both", "tp"))
    pool = {"both": ABS_BOTH, "sl": ABS_SL, "tp": ABS_TP}[flavor]
    c = OConv(rng, w, pack)
    kw = {}
    if sl is not None:
        kw["sl"] = price_text(rng, sl)
    if tp is not None:
        kw["tp"] = price_text(rng, tp)
    c.user(finish_text(rng, fill(rng, pick(rng, pool, split), c=coin, s=side, **kw)))
    args = {"coin": coin}
    if sl is not None:
        args["stop_loss"] = sl
    if tp is not None:
        args["take_profit"] = tp
    slots = {"coin": coin, "side": side, "flavor": flavor}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "tpsl_abs", "tpsl", {}, slots)
    r = c.call(("set_tp_sl", args))
    if r.get("status") != "review_opened":
        raise Retry
    c.say(tpsl_text(rng, r))
    gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS + ["cancel_order"],
            "omit_args": {}, "answer_contains": ["prepared"]}
    return rec(c, "tpsl_abs", bio, gold, ["right_tool", "exact_digits", "no_card_done", "restraint"], slots, ["set_tp_sl"])


def calc_expr(entry, factor, dp):
    return f"round({entry} * {factor}, {dp})"


def sc_tpsl_pct(rng, split, pack):
    bio = pick_bio(rng)
    flavor = rng.choices(["sl", "tp", "both"], [50, 20, 30])[0]
    base_price = rng.random() < 0.15
    named = rng.random() < 0.85
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None), extra=0 if not named else rng.choice([0, 0, 1, 2]))
    add_noise(rng, w, coin, 0.2)
    p = w.pos(coin)
    long_ = side == "long"
    base = p["mark"] if base_price else p["entry"]
    if not named and len(w.hl_positions) != 1:
        raise Retry
    sp_vals = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20] if split == "train" else [2, 3, 5, 8, 10, 12, 15, 20, 2.5, 7.5, 9, 11]
    tp_vals = [5, 10, 12, 15, 20, 25, 30, 40, 50] if split == "train" else [10, 15, 20, 25, 30, 8.5, 35, 45]
    sp = rng.choice(sp_vals)
    tpv = rng.choice(tp_vals)
    sf = dec(1) - dec(sp) / 100 if long_ else dec(1) + dec(sp) / 100
    tf = dec(1) + dec(tpv) / 100 if long_ else dec(1) - dec(tpv) / 100
    sl_dir = rng.choice(["under", "below"]) if long_ else rng.choice(["above", "over"])
    tp_dir = rng.choice(["above", "over"]) if long_ else rng.choice(["below", "under"])
    kw = {}
    if flavor in ("sl", "both"):
        kw["sp"], kw["sd"] = pct_text(rng, sp), sl_dir
    if flavor in ("tp", "both"):
        kw["tp"], kw["td"] = pct_text(rng, tpv), tp_dir
    if base_price:
        pool = {"sl": PRICE_SL, "tp": PRICE_TP}.get(flavor)
        if pool is None:
            raise Retry
    else:
        pool = {"sl": PCT_SL, "tp": PCT_TP, "both": PCT_BOTH}[flavor]
    tpl = pick(rng, pool, split)
    if not named:
        tpl = tpl.replace(" on my {c}", "").replace(" on {c}", "").replace(" for my {c}", "").replace("{c}: ", "").replace(" {c} {s}", "").replace("{c} ", "")
        if "{c}" in tpl or "{s}" in tpl:
            raise Retry
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, tpl, c=coin if named else None, s=side, **kw)))
    slots = {"coin": coin, "side": side, "flavor": flavor, "base": "price" if base_price else "entry", "named": named,
             "sp": sp if flavor != "tp" else None, "tp_pct": tpv if flavor != "sl" else None}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "tpsl_pct", "tpsl", {}, slots)
    if base_price:
        ra, rp = c.call(("hyperliquid_account", {}), ("get_price", {"symbol": coin}))
        if "error" in ra or "error" in rp:
            raise Retry
        base_v = rp["price"]
        pre = ["hyperliquid_account", "get_price"]
    else:
        ra = c.call(("hyperliquid_account", {}))
        if "error" in ra:
            raise Retry
        base_v = next(x for x in ra["positions"] if x["coin"] == coin)["entry"]
        pre = ["hyperliquid_account"]
    dp = dp_for(coin)
    calls, labels = [], []
    if flavor in ("sl", "both"):
        calls.append(("calculator", {"expression": calc_expr(base_v, dfmt(sf), dp)}))
        labels.append("stop_loss")
    if flavor in ("tp", "both"):
        calls.append(("calculator", {"expression": calc_expr(base_v, dfmt(tf), dp)}))
        labels.append("take_profit")
    res = c.call(*calls)
    res = res if isinstance(res, list) else [res]
    args = {"coin": coin}
    for lab, rr in zip(labels, res):
        if "error" in rr:
            raise Retry
        args[lab] = num(dec(rr["result"]))
    r = c.call(("set_tp_sl", args))
    if r.get("status") != "review_opened":
        raise Retry
    basis = (lambda pc, d: f" ({pc} {d} the current price of {money(base_v)})") if base_price else (lambda pc, d: f" ({pc} {d} your {money(base_v)} entry)")
    sl_ex = basis(pct_text(rng, sp) if False else f"{dfmt(sp)}%", "under" if long_ else "above") if "stop_loss" in args else ""
    tp_ex = basis(f"{dfmt(tpv)}%", "above" if long_ else "below") if "take_profit" in args else ""
    c.say(tpsl_text(rng, r, sl_ex, tp_ex))
    gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS + ["cancel_order"], "omit_args": {},
            "answer_contains": ["prepared", money(base_v)]}
    seq = pre + ["calculator"] * len(calls) + ["set_tp_sl"]
    return rec(c, "tpsl_pct", bio, gold, ["right_tool", "exact_digits", "no_card_done", "restraint"], slots, seq)


def sc_tpsl_breakeven(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    ask_case = rng.random() < 0.1
    named = rng.random() < 0.7 and not ask_case
    extra = 1 if ask_case else rng.choice([0, 0, 1])
    w = make_world(rng, split, bio, must=(coin, side, 1), extra=extra)
    if ask_case and len(w.hl_positions) < 2:
        raise Retry
    if not named and not ask_case and len(w.hl_positions) != 1:
        raise Retry
    p = w.pos(coin)
    if not ask_case and rng.random() < 0.35:
        long_ = side == "long"
        old = nice(rng, p["entry"] * (0.9 if long_ else 1.1))
        if (long_ and old >= p["entry"]) or (not long_ and old <= p["entry"]):
            raise Retry
        w.add_trigger(rng, coin, "stop_loss", price=old)
    c = OConv(rng, w, pack)
    tpl = pick(rng, BE_NAMED if named else BE_UNNAMED, split)
    c.user(finish_text(rng, fill(rng, tpl, c=coin if named else None, s=side)))
    slots = {"coin": coin, "side": side, "named": named, "ask": ask_case}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "tpsl_breakeven", "tpsl", {}, slots)
    ra = c.call(("hyperliquid_account", {}))
    if "error" in ra:
        raise Retry
    if ask_case:
        names = [f"your {x['coin']} {x['side']}" for x in ra["positions"]]
        c.say(rng.choice([f"Which position: {join_or(names)}?", f"Which one should I move the stop on, {join_or(names)}?",
                          f"You have {join_and(names)} open. Which one?"]), ask=True)
        gold = {"calls": [], "asks": True, "forbidden_tools": FORB_ORDERS + ["set_tp_sl"], "omit_args": {}}
        return rec(c, "tpsl_breakeven", bio, gold, ["right_tool", "asked_when_needed", "restraint"], slots, ["hyperliquid_account"])
    entry = next(x for x in ra["positions"] if x["coin"] == coin)["entry"]
    args = {"coin": coin, "stop_loss": entry}
    r = c.call(("set_tp_sl", args))
    if r.get("status") != "review_opened":
        raise Retry
    c.say(tpsl_text(rng, r, sl_expl=rng.choice([" (your entry price)", " (right at your entry)", " (break-even, your entry)"])))
    gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS + ["cancel_order"], "omit_args": {},
            "answer_contains": ["prepared"]}
    return rec(c, "tpsl_breakeven", bio, gold, ["right_tool", "exact_digits", "no_card_done", "restraint"], slots,
               ["hyperliquid_account", "set_tp_sl"])


def sc_tpsl_replace(rng, split, pack):
    bio = pick_bio(rng)
    flavor = rng.choices(["sl", "tp", "both"], [50, 25, 25])[0]
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 1]))
    if flavor in ("sl", "both"):
        w.add_trigger(rng, coin, "stop_loss")
    if flavor in ("tp", "both"):
        w.add_trigger(rng, coin, "take_profit")
    if flavor == "sl" and rng.random() < 0.4:
        w.add_trigger(rng, coin, "take_profit")
    if flavor == "tp" and rng.random() < 0.4:
        w.add_trigger(rng, coin, "stop_loss")
    sl, tp = valid_levels(rng, w, coin, flavor in ("sl", "both"), flavor in ("tp", "both"))
    old = {o["kind"]: o["price"] for o in w.orders if o["coin"] == coin}
    if (sl is not None and sl == old.get("stop_loss")) or (tp is not None and tp == old.get("take_profit")):
        raise Retry
    kw = {}
    if sl is not None:
        kw["sl"] = price_text(rng, sl)
    if tp is not None:
        kw["tp"] = price_text(rng, tp)
    pool = {"sl": REP_SL, "tp": REP_TP, "both": REP_BOTH}[flavor]
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, pick(rng, pool, split), c=coin, s=side, **kw)))
    args = {"coin": coin}
    if sl is not None:
        args["stop_loss"] = sl
    if tp is not None:
        args["take_profit"] = tp
    slots = {"coin": coin, "side": side, "flavor": flavor}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "tpsl_replace", "tpsl", {}, slots)
    r = c.call(("set_tp_sl", args))
    if r.get("status") != "review_opened" or not r.get("replaces"):
        raise Retry
    c.say(tpsl_text(rng, r))
    gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS + ["cancel_order"], "omit_args": {},
            "answer_contains": ["prepared", r["replaces"]]}
    return rec(c, "tpsl_replace", bio, gold, ["right_tool", "exact_digits", "no_card_done", "warnings_repeated"], slots, ["set_tp_sl"])


# ------------------------------------------------------------------ scenario: positions_read
def sc_positions_read(rng, split, pack):
    bio = pick_bio(rng)
    qtype = rng.choices(["stop_on_coin", "tp_on_coin", "orders_all", "protection_all", "coin_orders"], [35, 15, 25, 15, 10])[0]
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    has_pos = rng.random() < 0.88
    w = make_world(rng, split, bio, must=(coin, side, None) if has_pos else None, extra=rng.choice([0, 1, 1]) if has_pos else rng.choice([1, 2]))
    if has_pos:
        if rng.random() < 0.55:
            w.add_trigger(rng, coin, "stop_loss", whole=rng.random() > 0.15)
        if rng.random() < 0.45:
            w.add_trigger(rng, coin, "take_profit")
    for p in w.hl_positions:
        if p["coin"] != coin and rng.random() < 0.4:
            w.add_trigger(rng, p["coin"], rng.choice(["stop_loss", "take_profit"]))
    nlim = rng.choice([0, 1, 1, 2])
    lcs = [x for x in ["ETH", "BTC", "SOL", "HYPE"]]
    rng.shuffle(lcs)
    for lc in lcs[:nlim]:
        w.add_limit(rng, lc)
    if qtype in ("orders_all", "protection_all") and len(w.orders) > 5:
        raise Retry
    if qtype == "coin_orders":
        w.add_limit(rng, coin) if not any(o["coin"] == coin and o["kind"] == "limit" for o in w.orders) and rng.random() < 0.5 else None
    c = OConv(rng, w, pack)
    pool = {"stop_on_coin": READ_STOP, "tp_on_coin": READ_TP, "orders_all": LIST_T, "protection_all": READ_PROT, "coin_orders": READ_COIN_ORD}[qtype]
    c.user(finish_text(rng, fill(rng, pick(rng, pool, split), c=coin, s=side)))
    r = c.call(("get_positions", {"kind": "perps"}))
    if "error" in r:
        raise Retry
    c.say(read_answer(rng, qtype, r, coin))
    slots = {"coin": coin, "side": side, "qtype": qtype, "has_pos": has_pos}
    gold = {"calls": [{"name": "get_positions", "arguments": {"kind": "perps"}}], "forbidden_tools": ["set_tp_sl", "cancel_order", "prepare_trade", "prepare_close"],
            "omit_args": {}, "answer_forbidden": ["\\b\\d{4}\\b ids"]}
    return rec(c, "positions_read", bio, gold, ["right_tool", "exact_digits", "restraint"], slots, ["get_positions"])


# ------------------------------------------------------------------ scenario: errors
def sc_err_wrong_side(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    long_ = side == "long"
    pct_flow = rng.random() < 0.3
    bad_kind = rng.choice(["stop_loss", "take_profit"])
    if pct_flow:
        w = make_world(rng, split, bio, must=(coin, side, -1), extra=rng.choice([0, 1]))
    else:
        w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 1]))
    p = w.pos(coin)
    mark = p["mark"]
    c = OConv(rng, w, pack)
    if pct_flow:
        bad_kind = "stop_loss"
        loss = abs(p["entry"] - mark) / p["entry"] * 100
        cands = [v for v in ([1, 2, 3, 4, 5, 6] if split == "train" else [1, 2, 3, 4, 5, 7]) if v < loss * 0.8]
        if not cands:
            raise Retry
        sp = rng.choice(cands)
        sd = rng.choice(["under", "below"]) if long_ else rng.choice(["above", "over"])
        c.user(finish_text(rng, fill(rng, pick(rng, PCT_SL, split), c=coin, s=side, sp=pct_text(rng, sp), sd=sd)))
        slots = {"coin": coin, "side": side, "flow": "pct", "sp": sp}
        if pack == "core+solana":
            return absent_pointer(rng, c, bio, "err_wrong_side", "tpsl", {}, slots)
        ra = c.call(("hyperliquid_account", {}))
        entry = next(x for x in ra["positions"] if x["coin"] == coin)["entry"]
        factor = dec(1) - dec(sp) / 100 if long_ else dec(1) + dec(sp) / 100
        rc = c.call(("calculator", {"expression": calc_expr(entry, dfmt(factor), dp_for(coin))}))
        sl = num(dec(rc["result"]))
        args = {"coin": coin, "stop_loss": sl}
        r = c.call(("set_tp_sl", args))
        if "error" not in r:
            raise Retry
        c.say(rng.choice([f"I couldn't prepare that. {r['error']} A {dfmt(sp)}% stop from your {money(entry)} entry works out to {money(sl)}, which is on the wrong side. Which price do you want?",
                          f"{r['error']} Your entry is {money(entry)}, so {dfmt(sp)}% from there is {money(sl)}. What price should I use instead?",
                          f"Nothing was prepared. {r['error']} {money(sl)} is {dfmt(sp)}% from your {money(entry)} entry, which is too close. Give me another price?"]), ask=True)
        gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS, "omit_args": {}, "answer_contains": [r["error"]], "asks": True}
        return rec(c, "err_wrong_side", bio, gold, ["right_tool", "exact_digits", "error_stated", "restraint"], slots,
                   ["hyperliquid_account", "calculator", "set_tp_sl"])
    # absolute price on the wrong side
    d = rng.uniform(0.02, 0.10)
    if bad_kind == "stop_loss":
        raw = mark * (1 + d) if long_ else mark * (1 - d)
    else:
        raw = mark * (1 - d) if long_ else mark * (1 + d)
    v = nice(rng, raw)
    if (bad_kind == "stop_loss" and ((long_ and v < mark) or (not long_ and v > mark))) or (bad_kind == "take_profit" and ((long_ and v > mark) or (not long_ and v < mark))):
        raise Retry
    if v == mark:
        raise Retry
    pool = ABS_SL if bad_kind == "stop_loss" else ABS_TP
    key = "sl" if bad_kind == "stop_loss" else "tp"
    c.user(finish_text(rng, fill(rng, pick(rng, pool, split), c=coin, s=side, **{key: price_text(rng, v)})))
    slots = {"coin": coin, "side": side, "flow": "abs", "bad": bad_kind}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "err_wrong_side", "tpsl", {}, slots)
    args = {"coin": coin, bad_kind: v}
    r = c.call(("set_tp_sl", args))
    if "error" not in r:
        raise Retry
    word = "stop loss" if bad_kind == "stop_loss" else "take profit"
    valid_dir = ("below" if long_ else "above") if bad_kind == "stop_loss" else ("above" if long_ else "below")
    c.say(rng.choice([f"I couldn't prepare that. {r['error']} What price should I use instead?",
                      f"{r['error']} Your {coin} {side} needs the {word} {valid_dir} that. Which price do you want?",
                      f"Nothing was prepared. {r['error']} Give me a {word} price on the {valid_dir} side and I'll try again.",
                      f"{r['error']} {money(v)} is on the wrong side. What price do you want?"]), ask=True)
    gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS, "omit_args": {}, "answer_contains": [r["error"]], "asks": True}
    return rec(c, "err_wrong_side", bio, gold, ["right_tool", "exact_digits", "error_stated", "restraint"], slots, ["set_tp_sl"])


def sc_err_no_position(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    others = pick_coins(rng, split, rng.choice([0, 1, 1, 2]), [coin])
    w = make_world(rng, split, bio, must=None, extra=0)
    for oc in others:
        w.add_position(rng, oc, rng.choice(["long", "short"]))
    flavor = rng.choice(["sl", "tp", "both"])
    mark = w.prices[coin]
    sl = nice(rng, mark * 0.9) if flavor in ("sl", "both") else None
    tp = nice(rng, mark * 1.2) if flavor in ("tp", "both") else None
    kw = {}
    if sl is not None:
        kw["sl"] = price_text(rng, sl)
    if tp is not None:
        kw["tp"] = price_text(rng, tp)
    pool = {"both": ABS_BOTH, "sl": ABS_SL, "tp": ABS_TP}[flavor]
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, pick(rng, pool, split), c=coin, s=None, **kw)))
    slots = {"coin": coin, "flavor": flavor}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "err_no_position", "tpsl", {}, slots)
    args = {"coin": coin}
    if sl is not None:
        args["stop_loss"] = sl
    if tp is not None:
        args["take_profit"] = tp
    r = c.call(("set_tp_sl", args))
    if r.get("status") != "no_position":
        raise Retry
    c.say(rng.choice([f"You have no open {coin} position on Hyperliquid, so there is nothing to put that on.",
                      f"There is no open {coin} position on Hyperliquid, so I can't add a stop or take profit. To open one with a stop, tell me side, size and leverage.",
                      f"I don't see an open {coin} position on Hyperliquid. Nothing was prepared. If you want to open one, give me side, size and leverage and I'll add the stop to it.",
                      f"No open {coin} position on Hyperliquid, so nothing was prepared."]))
    gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS, "omit_args": {}, "answer_contains": ["no open|don't see an open|There is no open"]}
    return rec(c, "err_no_position", bio, gold, ["right_tool", "exact_digits", "error_stated", "restraint"], slots, ["set_tp_sl"])


def sc_err_nothing_to_cancel(rng, split, pack):
    bio = pick_bio(rng)
    flavor = rng.choices(["stop_loss", "take_profit", "order", "none", "all"], [25, 20, 20, 15, 20])[0]
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None) if rng.random() < 0.7 else None, extra=rng.choice([0, 1]))
    if flavor == "stop_loss":
        if w.pos(coin) and rng.random() < 0.5:
            w.add_trigger(rng, coin, "take_profit")
        kw, kind_arg = rng.choice(["stop loss", "stop", "sl"]), "stop_loss"
    elif flavor == "take_profit":
        if w.pos(coin) and rng.random() < 0.5:
            w.add_trigger(rng, coin, "stop_loss")
        kw, kind_arg = rng.choice(["take profit", "tp"]), "take_profit"
    elif flavor == "order":
        if w.pos(coin) and rng.random() < 0.5:
            w.add_trigger(rng, coin, "stop_loss")
        kw, kind_arg = rng.choice(["limit order", "limit"]), "order"
    else:
        kw, kind_arg = None, None
    if flavor == "all":
        if w.hl_positions and rng.random() < 0.6:
            w.add_trigger(rng, w.hl_positions[0]["coin"], "stop_loss")
    c = OConv(rng, w, pack)
    if flavor == "all":
        c.user(finish_text(rng, pick(rng, ALL_T, split)))
        args = {"coin": "all"}
    elif flavor == "none":
        c.user(finish_text(rng, fill(rng, pick(rng, CANCEL_NOKIND_T, split), c=coin, s=side)))
        args = {"coin": coin}
    else:
        c.user(finish_text(rng, fill(rng, pick(rng, CANCEL_KIND_T, split), c=coin, s=side, k=kw)))
        args = {"coin": coin, "kind": kind_arg}
    slots = {"coin": coin, "flavor": flavor}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "err_nothing_to_cancel", "cancel", {}, slots)
    r = c.call(("cancel_order", args))
    if r.get("status") != "nothing_to_cancel":
        raise Retry
    if flavor == "all":
        t = rng.choice(["You have no open limit orders to cancel. Stop losses and take profits are not included in that.",
                        "There are no open limit orders to cancel. Stop losses and take profits are not part of it, so tell me if you want to remove one of those."])
    elif flavor == "none":
        t = rng.choice([f"You have no open {coin} orders to cancel.", f"I don't see any open {coin} orders, so there is nothing to cancel.",
                        f"There are no open {coin} orders. Nothing was put up to cancel."])
    else:
        noun = KIND_NOUN[kind_arg]
        t = rng.choice([f"You have no open {coin} {noun} to cancel.", f"I don't see an open {coin} {noun}, so there is nothing to cancel.",
                        f"There is no open {coin} {noun}. Nothing was put up to cancel."])
        if kind_arg == "order":
            t = t.replace("an open", "any open").replace("is no open", "are no open") if False else t
    if rng.random() < 0.3:
        t += " " + rng.choice(["If you think there is one, I can look at your open orders.", "I can list your open orders if you want to check."])
    c.say(t)
    gold = {"calls": [{"name": "cancel_order", "arguments": args}], "forbidden_tools": FORB_ORDERS, "omit_args": {"cancel_order": ["order_id"] + ([] if kind_arg else ["kind"])},
            "answer_contains": ["no open|don't see|There are no|There is no"]}
    return rec(c, "err_nothing_to_cancel", bio, gold, ["right_tool", "exact_digits", "arg_omission", "error_stated", "no_card_done"], slots, ["cancel_order"])


def sc_err_blocked(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    reason = rng.choice(BLOCKED_REASONS)
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 1]), blocked=reason)
    sl, tp = valid_levels(rng, w, coin, True, rng.random() < 0.5)
    kw = {"sl": price_text(rng, sl)}
    if tp is not None:
        kw["tp"] = price_text(rng, tp)
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, pick(rng, ABS_BOTH if tp is not None else ABS_SL, split), c=coin, s=side, **kw)))
    args = {"coin": coin, "stop_loss": sl}
    if tp is not None:
        args["take_profit"] = tp
    slots = {"coin": coin, "side": side}
    if pack == "core+solana":
        return absent_pointer(rng, c, bio, "err_blocked", "tpsl", {}, slots)
    r = c.call(("set_tp_sl", args))
    if r.get("status") != "blocked":
        raise Retry
    c.say(rng.choice([f"Nothing was prepared. {r['reasons'][0]}", f"Nothing was prepared. {r['reasons'][0]} I won't retry it.",
                      f"I couldn't prepare that. {r['reasons'][0]} Tell me if you want something different.",
                      f"Nothing was prepared. {r['reasons'][0]} I haven't tried again."]))
    gold = {"calls": [{"name": "set_tp_sl", "arguments": args}], "forbidden_tools": FORB_ORDERS, "omit_args": {}, "answer_contains": [r["reasons"][0]]}
    return rec(c, "err_blocked", bio, gold, ["right_tool", "exact_digits", "error_stated", "restraint"], slots, ["set_tp_sl"])


def sc_err_no_wallet(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=None, extra=0, no_wallet=True)
    flavor = rng.choice(["cancel", "tpsl", "read"])
    c = OConv(rng, w, pack)
    if flavor == "cancel":
        kw = rng.choice(["stop loss", "stop", "take profit", "tp", "limit order"])
        kind_arg = "stop_loss" if "stop" in kw else "take_profit" if kw in ("take profit", "tp") else "order"
        c.user(finish_text(rng, fill(rng, pick(rng, CANCEL_KIND_T, split), c=coin, s=side, k=kw)))
        call = ("cancel_order", {"coin": coin, "kind": kind_arg})
        what = "cancel"
    elif flavor == "tpsl":
        mark = w.prices[coin]
        sl = nice(rng, mark * 0.9)
        c.user(finish_text(rng, fill(rng, pick(rng, ABS_SL, split), c=coin, s=side, sl=price_text(rng, sl))))
        call = ("set_tp_sl", {"coin": coin, "stop_loss": sl})
        what = "protect"
    else:
        c.user(finish_text(rng, fill(rng, pick(rng, READ_STOP, split), c=coin, s=side)))
        call = ("get_positions", {"kind": "perps"})
        what = "read"
    slots = {"coin": coin, "flavor": flavor}
    if pack == "core+solana":
        if flavor == "read":
            raise Retry
        return absent_pointer(rng, c, bio, "err_no_wallet", "cancel" if flavor == "cancel" else "tpsl", {}, slots)
    r = c.call(call)
    if "error" not in r:
        raise Retry
    dev = device_of(r["error"])
    if what == "cancel":
        t = rng.choice([f"There is no wallet on this {dev} yet, so there is nothing to cancel. Set one up in the app first.",
                        f"I can't do that yet: this {dev} has no wallet, so there are no orders to cancel."])
    elif what == "protect":
        t = rng.choice([f"There is no wallet on this {dev} yet, so there is no position to put a stop on. Set one up in the app first.",
                        f"Nothing was prepared: this {dev} has no wallet yet, so there is no position to add a stop to."])
    else:
        t = rng.choice([f"There is no wallet on this {dev} yet, so I can't see any positions or stops. Set one up in the app first.",
                        f"This {dev} has no wallet yet, so there are no positions or stops to look at."])
    c.say(t)
    gold = {"calls": [{"name": call[0], "arguments": call[1]}], "forbidden_tools": FORB_ORDERS, "omit_args": {}, "answer_contains": ["no wallet"]}
    return rec(c, "err_no_wallet", bio, gold, ["right_tool", "exact_digits", "error_stated", "no_card_done"], slots, [call[0]])


# ------------------------------------------------------------------ near misses
def sc_neg_close(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 1, 2]))
    if rng.random() < 0.4:
        w.add_trigger(rng, coin, "stop_loss")
    partial = rng.random() < 0.3
    c = OConv(rng, w, pack)
    args = {"coin": coin}
    if partial:
        ptxt, pv = rng.choice(PCT_CHOICES)
        c.user(finish_text(rng, fill(rng, pick(rng, CLOSE_PART, split), c=coin, s=side, p=ptxt)))
        args["percent"] = pv
    else:
        c.user(finish_text(rng, fill(rng, pick(rng, CLOSE_FULL, split), c=coin, s=side)))
    slots = {"coin": coin, "side": side, "partial": partial}
    r = c.call(("prepare_close", args))
    if r.get("status") != "review_opened":
        raise Retry
    pnl = r["unrealised_pnl"]
    pn = ("+" if pnl > 0 else "-") + money(abs(pnl)) if pnl != 0 else "$0"
    c.say(rng.choice([f"I've prepared closing {r['closing']}. It's a {r['order'].lower()} order and the unrealised PnL is {pn}. Approve with Face ID.",
                      f"I've prepared closing {r['closing']}, {r['order'].lower()}. Unrealised PnL is {pn}. {approve(rng)}",
                      f"Prepared: closing {r['closing']}. Order: {r['order'].lower()}. PnL right now is {pn}. Approve with Face ID.",
                      f"I've prepared closing {r['closing']} as a {r['order'].lower()} order, with {pn} unrealised PnL. It waits for your Face ID."]))
    gold = {"calls": [{"name": "prepare_close", "arguments": args}], "forbidden_tools": ["cancel_order", "set_tp_sl", "prepare_trade"], "omit_args": {},
            "answer_contains": ["prepared"]}
    return rec(c, "neg_close", bio, gold, ["right_tool", "exact_digits", "restraint"], slots, ["prepare_close"])


def prior_context(rng, c, w, bio, kind):
    """A review is open from an earlier turn. Returns the user cancel-text pool."""
    if kind == "swap":
        amt = rng.choice([20, 25, 50, 75, 100, 150, 200, 250, 500])
        w.ensure_balance("USDC", "base", amt)
        to = rng.choice([("ETH", "arbitrum"), ("ETH", "base"), ("SOL", "solana"), ("BNB", "bsc")])
        if to[1] == "base":
            phr = [f"swap {amt} usdc to {to[0].lower()} on base", f"swap {amt} USDC for {to[0]} on base", f"can you swap {amt} usdc into {to[0]} on base?",
                   f"I want {to[0]} on base, swap {amt} usdc"]
        else:
            phr = [f"swap {amt} usdc on base to {to[0].lower()} on {to[1]}", f"bridge {amt} usdc from base to {to[0]} on {to[1]}",
                   f"can you swap {amt} USDC (base) for {to[0]} on {to[1]}?", f"I want {to[0]} on {to[1]}, swap {amt} usdc from base"]
        c.user(finish_text(rng, rng.choice(phr)))
        r = c.call(("prepare_swap", {"from_symbol": "USDC", "from_chain": "base", "to_symbol": to[0], "to_chain": to[1], "amount": amt}))
        c.say(f"I've prepared swapping {r['from']} for {r['to']} through {r['route']}. Total cost is {money(r['total_cost_usd'])}. Approve with Face ID.")
        return SWAP_CANCEL, {"prior": "prepare_swap"}
    if kind == "send":
        amt = rng.choice([5, 10, 25, 40, 50, 100, 250])
        addr = H.evm_addr(rng)
        w.ensure_balance("USDC", "base", amt)
        pre = rng.choice([f"send {amt} usdc to", f"transfer {amt} USDC to", f"pls send {amt} usdc to", f"can you send {amt} usdc to"])
        c.user(re.sub(r"\bA1\b", lambda _m: addr, finish_text(rng, pre + " A1"), flags=re.I))
        r = c.call(("prepare_send", {"to_address": addr, "amount": amt, "symbol": "USDC"}))
        c.say(f"I've prepared sending {r['amount']} to {addr} on {r['network']}. Approve with Face ID.")
        return SEND_CANCEL, {"prior": "prepare_send"}
    coin = rng.choice(["ETH", "BTC", "SOL", "HYPE"])
    side = rng.choice(["long", "short"])
    usd = rng.choice([50, 100, 150, 200, 250, 500, 1000])
    lev = rng.choice([2, 3, 5, 5, 10])
    w.hl_value = max(w.hl_value, usd * 2)
    c.user(finish_text(rng, rng.choice([f"{side} {coin.lower()} ${usd} {lev}x", f"open a {coin} {side} {usd} usd at {lev}x", f"{coin} {side}, {usd} dollars, {lev}x"])))
    args = {"coin": coin, "side": side, "size_usd": usd, "leverage": lev, "order_type": "market"}
    r = c.call(("prepare_trade", args))
    if r.get("status") != "review_opened":
        raise Retry
    c.say(f"I've prepared {art(coin)} {coin} {side}: ${usd} at {lev}x, market order. Approve with Face ID.")
    pool = [t.replace("{s}", side).replace("{c}", coin) for t in TRADE_CANCEL]
    return pool, {"prior": "prepare_trade", "pcoin": coin, "pside": side}


def sc_neg_cancel_review(rng, split, pack):
    bio = pick_bio(rng)
    kind = rng.choice(["swap", "swap", "send", "trade", "trade", "trade"])
    w = make_world(rng, split, bio, must=None, extra=rng.choice([0, 1, 2]))
    if rng.random() < 0.3 and w.hl_positions:
        w.add_trigger(rng, w.hl_positions[0]["coin"], "stop_loss")
    c = OConv(rng, w, pack)
    pool, slots = prior_context(rng, c, w, bio, kind)
    slots = dict(slots, flavor=kind)
    c.user(finish_text(rng, pick(rng, pool, split)))
    c.say(rng.choice(["Tap Cancel on the review; nothing was sent.", "Tap Cancel on the review; nothing was sent.", "Tap Cancel on the review; nothing was sent.",
                      "Tap Cancel on the review. Nothing was sent, so there is nothing to undo.", "I can't close the review for you. Tap Cancel on it; nothing was sent.",
                      "Just tap Cancel on the review. Nothing was sent."]))
    gold = {"calls": [], "forbidden_tools": ["cancel_order", "set_tp_sl", "prepare_close", "prepare_trade", "prepare_swap", "prepare_send"], "omit_args": {},
            "answer_contains": ["Tap Cancel on the review"], "only_tools": []}
    prior = slots["prior"]
    return rec(c, "neg_cancel_review", bio, gold, ["right_tool", "restraint", "no_card_done"], slots, [prior])


def sc_neg_cancel_nothing(rng, split, pack):
    bio = pick_bio(rng)
    w = make_world(rng, split, bio, must=None, extra=rng.choice([0, 1]))
    if w.hl_positions and rng.random() < 0.5:
        w.add_trigger(rng, w.hl_positions[0]["coin"], "stop_loss")
    c = OConv(rng, w, pack)
    seq = []
    if rng.random() < 0.25:
        coin = rng.choice(["BTC", "ETH", "SOL", "HYPE"])
        c.user(finish_text(rng, rng.choice([f"what's {coin} at", f"{coin} price?", f"price of {coin}", f"how much is {coin}"])))
        r = c.call(("get_price", {"symbol": coin}))
        c.say(f"{coin} is at {money(r['price'])}, {'up' if r['change_24h_pct'] >= 0 else 'down'} {abs(r['change_24h_pct'])}% over the day.")
        seq = ["get_price"]
    base = pick(rng, CANCEL_NOTHING, split)
    pre = rng.choice(["", "", "", "ok ", "hey ", "wait "])
    c.user(finish_text(rng, pre + base))
    c.say(rng.choice(["There's nothing waiting for your approval in this chat, so there's nothing to cancel. If you mean an open order on Hyperliquid, tell me which coin.",
                      "Nothing is pending right now, so there's nothing to cancel. If you mean an open order, tell me which coin and I'll put up a card.",
                      "I don't have anything pending to cancel. If you mean a stop, take profit or limit order, tell me which one.",
                      "Nothing is open for review here, so there's nothing to cancel. Tell me which order you mean if it's one on Hyperliquid.",
                      "There is nothing pending in this chat. If you want to cancel an order on Hyperliquid, tell me the coin.",
                      "I don't see anything pending. Name the order you want to cancel, for example a stop loss on a coin, and I'll sort it."]))
    gold = {"calls": [], "forbidden_tools": ["cancel_order", "set_tp_sl", "prepare_close", "prepare_trade"], "omit_args": {},
            "answer_contains": ["nothing|Nothing|don't have anything|don't see anything"], "only_tools": seq}
    return rec(c, "neg_cancel_nothing", bio, gold, ["right_tool", "restraint"], {"price_first": bool(seq)}, seq)


def sc_neg_what_happens(rng, split, pack):
    bio = pick_bio(rng)
    topic = rng.choice(list(HOW))
    w = make_world(rng, split, bio, must=None, extra=rng.choice([0, 1]))
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, pick(rng, HOW[topic]["q"], split)))
    c.say(rng.choice(HOW[topic]["a"]))
    gold = {"calls": [], "forbidden_tools": ["cancel_order", "set_tp_sl", "prepare_close", "prepare_trade", "get_positions", "hyperliquid_account"], "omit_args": {},
            "only_tools": []}
    return rec(c, "neg_what_happens", bio, gold, ["restraint"], {"topic": topic}, [])


def sc_neg_short_stop(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    long_ = side == "long"
    w = make_world(rng, split, bio, must=None, extra=rng.choice([0, 1, 2]))
    w.hl_positions = [p for p in w.hl_positions if p["coin"] != coin]
    w.hl_value = max(w.hl_value, 3000)
    mark = w.prices[coin]
    lev = rng.choice([2, 3, 5, 5, 10] if split == "train" else [2, 3, 4, 5, 10])
    lev = min(lev, MAXLEV.get(coin, 10))
    usd = rng.choice([50, 100, 150, 200, 250, 300, 500, 750, 1000])
    sl = nice(rng, mark * (1 - rng.uniform(0.03, 0.12)) if long_ else mark * (1 + rng.uniform(0.03, 0.12)))
    with_tp = rng.random() < 0.35
    tp = nice(rng, mark * (1 + rng.uniform(0.06, 0.3)) if long_ else mark * (1 - rng.uniform(0.06, 0.25))) if with_tp else None
    if (long_ and sl >= mark) or (not long_ and sl <= mark):
        raise Retry
    if tp is not None and ((long_ and tp <= mark) or (not long_ and tp >= mark)):
        raise Retry
    kw = {"w": side, "usd": usd, "lev": lev, "sl": price_text(rng, sl)}
    if tp is not None:
        kw["tp"] = price_text(rng, tp)
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, pick(rng, SHORT_STOP_TP_T if with_tp else SHORT_STOP_T, split), c=coin, s=None, **kw)))
    args = {"coin": coin, "side": side, "size_usd": usd, "leverage": lev, "order_type": "market"}
    if tp is not None:
        args["take_profit"] = tp
    args["stop_loss"] = sl
    r = c.call(("prepare_trade", args))
    if r.get("status") != "review_opened":
        raise Retry
    tail = f" and a take profit at {money(tp)}" if tp is not None else ""
    c.say(rng.choice([f"I've prepared {art(coin)} {coin} {side}: ${usd} at {lev}x, market order, with a stop loss at {money(sl)}{tail}. Approve with Face ID.",
                      f"I've prepared {art(coin)} {coin} {side} for ${usd} at {lev}x with a stop loss at {money(sl)}{tail}. {approve(rng)}",
                      f"Prepared: {coin} {side}, ${usd} at {lev}x, market order, stop loss at {money(sl)}{tail}. Approve with Face ID.",
                      f"I've prepared your {coin} {side} at {lev}x for ${usd}, stop loss {money(sl)}{tail}. It waits for your Face ID."]))
    gold = {"calls": [{"name": "prepare_trade", "arguments": args}], "forbidden_tools": ["set_tp_sl", "cancel_order", "prepare_close"], "omit_args": {},
            "answer_contains": ["prepared"]}
    return rec(c, "neg_short_stop", bio, gold, ["right_tool", "exact_digits", "restraint"], {"coin": coin, "side": side, "tp": with_tp}, ["prepare_trade"])


def sc_neg_tpsl_ask(rng, split, pack):
    bio = pick_bio(rng)
    coin = pick_coins(rng, split, 1)[0]
    side = rng.choice(["long", "short"])
    w = make_world(rng, split, bio, must=(coin, side, None), extra=rng.choice([0, 1]))
    is_sl = rng.random() < 0.7
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, fill(rng, pick(rng, TPSL_ASK_SL if is_sl else TPSL_ASK_TP, split), c=coin, s=side)))
    c.say(rng.choice(ASK_Q_SL if is_sl else ASK_Q_TP), ask=True)
    gold = {"calls": [], "asks": True, "forbidden_tools": ["set_tp_sl", "prepare_trade", "cancel_order", "prepare_close"], "omit_args": {}, "only_tools": []}
    return rec(c, "neg_tpsl_ask", bio, gold, ["restraint", "asked_when_needed"], {"coin": coin, "sl": is_sl}, [])


def sc_neg_margin_read(rng, split, pack):
    bio = pick_bio(rng)
    w = make_world(rng, split, bio, must=None, extra=rng.choice([0, 1, 1, 2]))
    if w.hl_positions and rng.random() < 0.5:
        w.add_trigger(rng, w.hl_positions[0]["coin"], "stop_loss")
    c = OConv(rng, w, pack)
    c.user(finish_text(rng, pick(rng, MARGIN_T, split)))
    r = c.call(("hyperliquid_account", {}))
    if "error" in r:
        raise Retry
    c.say(rng.choice([f"Your Hyperliquid account is worth {money(r['account_value'])}. You are using {money(r['margin_used'])} as margin and have {money(r['available_usdc'])} available.",
                      f"You have {money(r['available_usdc'])} available on Hyperliquid, with {money(r['margin_used'])} in use as margin. The account is worth {money(r['account_value'])}.",
                      f"{money(r['available_usdc'])} is free to trade. Margin in use is {money(r['margin_used'])} and the account value is {money(r['account_value'])}."]))
    gold = {"calls": [{"name": "hyperliquid_account", "arguments": {}}], "forbidden_tools": ["get_positions", "cancel_order", "set_tp_sl"], "omit_args": {}}
    return rec(c, "neg_margin_read", bio, gold, ["right_tool", "exact_digits", "restraint"], {}, ["hyperliquid_account"])


# ------------------------------------------------------------------ build
SCEN = {
    "cancel_kind": sc_cancel_kind, "cancel_noKind": sc_cancel_nokind, "cancel_all": sc_cancel_all, "cancel_id": sc_cancel_id,
    "tpsl_abs": sc_tpsl_abs, "tpsl_pct": sc_tpsl_pct, "tpsl_breakeven": sc_tpsl_breakeven, "tpsl_replace": sc_tpsl_replace,
    "positions_read": sc_positions_read, "err_wrong_side": sc_err_wrong_side, "err_no_position": sc_err_no_position,
    "err_nothing_to_cancel": sc_err_nothing_to_cancel, "err_blocked": sc_err_blocked, "err_no_wallet": sc_err_no_wallet,
    "neg_close": sc_neg_close, "neg_cancel_review": sc_neg_cancel_review, "neg_cancel_nothing": sc_neg_cancel_nothing,
    "neg_what_happens": sc_neg_what_happens, "neg_short_stop": sc_neg_short_stop, "neg_tpsl_ask": sc_neg_tpsl_ask,
    "neg_margin_read": sc_neg_margin_read,
}
# train counts (sum 420): 60+35+20+15 cancel, 45+40+15+15 tp/sl, 25 read, 40 errors, 110 near misses
QUOTA = [("cancel_kind", 60), ("cancel_noKind", 35), ("cancel_all", 20), ("cancel_id", 15), ("tpsl_abs", 45), ("tpsl_pct", 40),
         ("tpsl_breakeven", 15), ("tpsl_replace", 15), ("positions_read", 25),
         ("err_wrong_side", 12), ("err_no_position", 8), ("err_nothing_to_cancel", 8), ("err_blocked", 6), ("err_no_wallet", 6),
         ("neg_close", 18), ("neg_cancel_review", 28), ("neg_cancel_nothing", 12), ("neg_what_happens", 18),
         ("neg_short_stop", 16), ("neg_tpsl_ask", 8), ("neg_margin_read", 10)]
assert sum(n for _, n in QUOTA) == 420
# kinds whose tool is missing in core+solana (the absent-tool pointer); cancel_id / positions_read / negatives keep get_positions or no trade tool
ABSENT_OK = {"cancel_kind", "cancel_noKind", "cancel_all", "tpsl_abs", "tpsl_pct", "tpsl_breakeven", "tpsl_replace", "err_wrong_side",
             "err_no_position", "err_nothing_to_cancel", "err_blocked"}


def user_key(ex):
    return " | ".join(norm(m["content"]) for m in ex["messages"] if m["role"] == "user")


def build(seed, n, split, forbid_users=()):
    rng = random.Random(seed)
    quota = scale_quota(QUOTA, n)
    plan = [k for k, cnt in quota for _ in range(cnt)]
    idx = list(range(len(plan)))
    n_abs = round(0.05 * n)
    n_all = round(0.05 * n)
    elig = [i for i in idx if plan[i] in ABSENT_OK]
    absent = set(rng.sample(elig, n_abs))
    rest = [i for i in idx if i not in absent]
    all30 = set(rng.sample(rest, n_all))
    seen_all, seen_first, out = set(forbid_users), set(), []
    for i in idx:
        name = plan[i]
        pack = "core+solana" if i in absent else "all30" if i in all30 else "core+trade"
        for attempt in range(600):
            r = random.Random(rng.getrandbits(64))
            try:
                ex = SCEN[name](r, split, pack)
            except (Retry, RuntimeError, StopIteration):
                continue
            users = [norm(m["content"]) for m in ex["messages"] if m["role"] == "user"]
            key = user_key(ex)
            if key in seen_first or any(u in seen_all for u in users):
                continue
            seen_first.add(key)
            out.append(apply_bio(ex))
            break
        else:
            raise SystemExit(f"could not generate a unique example for {name} (pack {pack})")
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------ self-check
BANNED = [r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|cancell?ed|removed|set|placed|closed)",
          r"\bI(?:'ve| have)? (sent|bought|swapped|cancell?ed|removed|closed|sold|placed)\b", r"successfully",
          r"funds (are|have) (in|arrived)", r"^\s*done\b", r"not financial advice", r"[·•]", r"[^\x00-\x7f]", r"\bas an ai\b",
          r"you'?re (now )?(protected|covered|safe)", r"\bprotected\b", r"\bcan(?:'|no)t cancel", r"\bcannot cancel", r"unable to cancel",
          r"stop(?: loss)? (is|has been|was) (now )?(cancel|removed|deleted|set|placed|live|active|in place)", r"\ball set\b",
          r"what('s| is) (the )?(current )?price", r"your budget", r"how much (do you|are you)", r"\bsold\b", r"should have arrived",
          r"(^|[.!]\s)(sent|done|cancell?ed|removed|set|swapped|bought)[.!,]", r"\bI will cancel\b", r"\bI'?ll cancel it\b", r"is now (live|active|cancelled)", r"\bguaranteed to\b"]
NEG_OK = re.compile(r"(nothing|not|isn't|is not|never|until|yet|nor|no)\W+(?:\w+\W+){0,3}$", re.I)
CANCELLED_RE = re.compile(r"\b(cancell?ed|removed|deleted)\b", re.I)
NUM_LINE_IDX = re.compile(r"^\s*\d+\.\s", re.M)
STOP_RE = re.compile(r"\bstop|\bsl\b|stoploss|stop-loss|\bsl\d", re.I)
TP_RE = re.compile(r"take[- ]?profit|\btp\b|takeprofit", re.I)
LIMIT_RE = re.compile(r"\blimit", re.I)


def vcall(tc):
    name, args = tc["name"], tc["arguments"]
    if name not in SCHEMA:
        return [f"unknown tool {name}"]
    sch = SCHEMA[name]
    errs = []
    req_list = ["to_address", "amount", "symbol"] if name == "prepare_send" else sch.get("required", [])
    for req in req_list:
        if req not in args:
            errs.append(f"{name}: missing {req}")
    for k, v in args.items():
        pr = sch["properties"].get(k)
        if pr is None:
            errs.append(f"{name}: unknown arg {k}")
            continue
        t = pr["type"]
        ok = {"number": isinstance(v, (int, float)) and not isinstance(v, bool), "string": isinstance(v, str),
              "integer": isinstance(v, int) and not isinstance(v, bool), "array": isinstance(v, list), "object": isinstance(v, dict)}[t]
        if not ok:
            errs.append(f"{name}: {k} wrong type {v!r}")
        if pr.get("enum") and v not in pr["enum"]:
            errs.append(f"{name}: {k} bad enum {v!r}")
        if isinstance(v, float) and v.is_integer():
            errs.append(f"{name}: {k} should be an int {v!r}")
    return errs


def src_numbers(msgs):
    acc = set()
    for m in msgs:
        if m["role"] == "tool":
            CP.nums_in(json.loads(m["content"]), acc)
        elif m["role"] == "user":
            CP.nums_in(m["content"], acc)
            acc |= {d for d in user_numbers(m["content"])}
        for tc in m.get("tool_calls") or []:
            CP.nums_in(tc["arguments"], acc)
    return acc


def self_check_record(ex, is_eval):
    errs = []
    meta, msgs = ex["meta"], ex["messages"]
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    kind = meta["kind"]
    # 1 json validity and shape
    try:
        json.loads(json.dumps(ex, ensure_ascii=False))
        for m in msgs:
            if m["role"] == "tool":
                json.loads(m["content"])
    except Exception as e:
        return [f"json: {e}"]
    if ex["tools"] is not True or meta["source"] != GEN_NAME:
        errs.append("record shape")
    for key in ("pack", "pack_name", "family_ok", "steps", "cards", "script", "seq", "world", "gold"):
        if key not in meta:
            errs.append(f"meta lacks {key}")
    if errs:
        return errs
    sysm = msgs[0]
    if sysm["role"] != "system" or "{BIO}" in sysm["content"] or not sysm["content"].startswith(SYSTEM_V3.replace("{BIO}", bio)):
        errs.append("system prompt is not SYSTEM_AGENT_v3 with the device word")
    blob = json.dumps(msgs, ensure_ascii=False)
    if other in blob or "{BIO}" in blob:
        errs.append(f"bio word mix ({other} in a {bio} example)")
    if meta["pack"] != PACKS[meta["pack_name"]]:
        errs.append("pack list")
    # 2 calls: registry, pack, schema
    calls = [tc for m in msgs for tc in (m.get("tool_calls") or [])]
    names = [tc["name"] for tc in calls]
    for tc in calls:
        errs += vcall(tc)
        if tc["name"] not in meta["pack"]:
            errs.append(f"{tc['name']} not in pack {meta['pack_name']}")
    if names != meta["seq"]:
        errs.append(f"call sequence {names} != expected {meta['seq']}")
    if meta["cards"] != [n for n in names if n in CARD_TOOLS]:
        errs.append("meta.cards")
    # tool message pairing
    for j, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            if [x["role"] for x in msgs[j + 1:j + 1 + k]] != ["tool"] * k:
                errs.append("tool message count")
    # 3 replay against the simulator
    w0 = OPWorld.from_state(meta["world"], seed=0)
    tool_msgs = [m["content"] for m in msgs if m["role"] == "tool"]
    results = [w0.call(tc["name"], tc["arguments"]) for tc in calls]
    replay = [json.dumps(r, ensure_ascii=False).replace("Face ID", bio) for r in results]
    if replay != tool_msgs:
        errs.append("replay mismatch")
    res_by_i = list(zip(calls, results))
    users = [m["content"] for m in msgs if m["role"] == "user"]
    utext = " ".join(users)
    gold = meta["gold"]
    slots = meta["slots"]
    # 4 argument provenance
    last_u = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    final_calls = [tc for m in msgs[last_u:] if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])]
    all_nums = set()
    CP.nums_in(" ".join(users), all_nums)
    unums = user_numbers(utext)
    ucoins = CP.parse_coins(utext)
    pcoins = [p["coin"] for p in meta["world"]["hl_positions"]]
    for i, (tc, r) in enumerate(res_by_i):
        a = tc["arguments"]
        if tc["name"] == "cancel_order":
            ut_all = " ".join(users)
            ut_last = users[-1]
            has_kind = "kind" in a
            says_stop, says_tp, says_lim = bool(STOP_RE.search(ut_all)), bool(TP_RE.search(ut_all)), bool(LIMIT_RE.search(ut_all))
            if has_kind:
                want = a["kind"]
                if slots.get("by_ordinal"):
                    pass
                elif want == "stop_loss" and not STOP_RE.search(ut_last):
                    errs.append("kind stop_loss but the user text does not say stop")
                elif want == "take_profit" and not TP_RE.search(ut_last):
                    errs.append("kind take_profit but the user text does not say take profit")
                elif want == "order" and not LIMIT_RE.search(ut_last):
                    errs.append("kind order but the user text does not say limit")
            else:
                if kind in ("cancel_kind",) or (kind == "err_nothing_to_cancel" and slots.get("flavor") in ("stop_loss", "take_profit", "order")):
                    errs.append("cancel_order lacks kind although the user named it")
                if "order_id" not in a and a["coin"] != "all":
                    first_user = users[0] if kind == "cancel_noKind" else ut_last
                    if STOP_RE.search(first_user) or TP_RE.search(first_user) or LIMIT_RE.search(first_user):
                        if kind in ("cancel_noKind",):
                            errs.append("kind omitted but the user named a kind")
            if "order_id" in a:
                prev = [json.loads(m["content"]) for m in msgs[:last_u] if m["role"] == "tool"]
                rows = [o for pr in prev if "perps" in pr for o in pr["perps"]["open_orders"]]
                row = next((o for o in rows if o["id"] == a["order_id"]), None)
                if row is None or not isinstance(a["order_id"], str):
                    errs.append("order_id not from a get_positions row (string)")
                elif row["coin"] != a["coin"]:
                    errs.append("order_id coin mismatch")
            if a["coin"] == "all":
                if not re.search(r"\ball\b|\bevery|\bthem\b|\blot\b|\beverything\b|\bany\b|\bwipe|\bnuke|\bclear", utext, re.I):
                    errs.append("coin all without an all-word")
            elif a["coin"] not in ucoins and "order_id" not in a:
                errs.append(f"cancel coin {a['coin']} not in user text")
        if tc["name"] == "set_tp_sl":
            if a["coin"] not in pcoins and r.get("status") != "no_position" and "error" not in r:
                errs.append("set_tp_sl on a coin that is not a position")
            if kind in ("tpsl_abs", "tpsl_replace", "err_blocked", "err_no_position") or (kind == "err_wrong_side" and slots.get("flow") == "abs"):
                for key in ("stop_loss", "take_profit"):
                    if key in a and dec(a[key]) not in unums:
                        errs.append(f"{key} {a[key]} not typed by the user")
                has_sl = bool(STOP_RE.search(utext))
                has_tp = bool(TP_RE.search(utext))
                if ("stop_loss" in a) != has_sl or ("take_profit" in a) != has_tp:
                    errs.append("stop/take profit arguments do not match the user's words")
            if kind in ("tpsl_abs", "tpsl_pct", "tpsl_breakeven", "tpsl_replace") and a["coin"] not in ucoins and slots.get("named", True):
                errs.append("set_tp_sl coin not in user text")
            if kind == "tpsl_breakeven" and "stop_loss" in a:
                ra = json.loads(next(m["content"] for m in msgs if m["role"] == "tool"))
                ent = next(x for x in ra["positions"] if x["coin"] == a["coin"])["entry"]
                if a["stop_loss"] != ent or set(a) != {"coin", "stop_loss"}:
                    errs.append("break-even stop is not the entry from the account result")
            if kind == "tpsl_pct" or (kind == "err_wrong_side" and slots.get("flow") == "pct"):
                calc = [(c2, r2) for c2, r2 in res_by_i if c2["name"] == "calculator"]
                got = {float(dec(r2["result"])) for _, r2 in calc}
                for key in ("stop_loss", "take_profit"):
                    if key in a and float(a[key]) not in got:
                        errs.append(f"{key} {a[key]} is not a calculator result")
                for c2, r2 in calc:
                    mm = re.fullmatch(r"round\(([\d.]+) \* ([\d.]+), (\d)\)", c2["arguments"]["expression"])
                    if not mm:
                        errs.append(f"calculator expression {c2['arguments']['expression']}")
                        continue
                    exp = round(float(mm.group(1)) * float(mm.group(2)), int(mm.group(3)))
                    if abs(exp - float(dec(r2["result"]))) > 1e-9:
                        errs.append("calculator result differs from recomputation")
                    # the factor must come from the user's percent
                    fac = dec(mm.group(2))
                    pcs = {abs(dec(1) - fac) * 100}
                    if not any(p in {dec(str(x)) for x in re.findall(r"\d+(?:\.\d+)?", utext)} for p in pcs):
                        errs.append("percent factor not in the user text")
                ra = json.loads(next(m["content"] for m in msgs if m["role"] == "tool"))
                if slots.get("base") != "price":
                    ent = next(x for x in ra["positions"] if x["coin"] == a["coin"])["entry"]
                    if any(not c2["arguments"]["expression"].startswith(f"round({ent} * ") for c2, _ in calc):
                        errs.append("calculator does not use the entry")
        if tc["name"] == "prepare_send" and a["to_address"] not in utext:
            errs.append("to_address not typed by the user")
        if tc["name"] == "prepare_trade" and kind == "neg_short_stop":
            if dec(a["stop_loss"]) not in unums or dec(a["size_usd"]) not in unums or dec(a["leverage"]) not in unums:
                errs.append("prepare_trade numbers not typed by the user")
            if "take_profit" in a and dec(a["take_profit"]) not in unums:
                errs.append("take_profit not typed by the user")
        if tc["name"] == "prepare_close" and kind == "neg_close":
            pv = a.get("percent")
            if pv is not None:
                m = re.search(r"(\d+)\s*%", utext)
                exp = 50 if re.search(r"half", utext, re.I) else 25 if re.search(r"quarter", utext, re.I) else int(m.group(1)) if m else None
                if pv != exp:
                    errs.append("prepare_close percent differs from the user text")
            elif re.search(r"half|quarter|\d+\s*%", utext, re.I):
                errs.append("prepare_close lacks the percent the user gave")
    # kind-specific call patterns
    if kind in ("neg_cancel_review",) and final_calls:
        errs.append("neg_cancel_review: a tool call after the cancel text")
    if kind in ("neg_what_happens", "neg_tpsl_ask", "neg_cancel_nothing") and any(tc["name"] in set(TRADE) | {"prepare_close"} for tc in calls):
        errs.append(f"{kind}: trade tool called")
    if kind == "neg_cancel_review" and not any(tc["name"] in ("prepare_swap", "prepare_send", "prepare_trade") for tc in calls):
        errs.append("neg_cancel_review lacks the open review")
    if kind == "cancel_all" and not meta["slots"].get("absent") and calls[-1]["arguments"] != {"coin": "all"}:
        errs.append("cancel_all arguments")
    if kind in ("cancel_kind", "cancel_noKind", "cancel_all", "cancel_id") and not meta["slots"].get("absent"):
        if results[-1].get("status") != "card_opened":
            errs.append("cancel flow does not end in a card")
    # 5 assistant texts
    src = src_numbers(msgs)
    tool_blob = " ".join(tool_msgs) + " " + utext + json.dumps(calls)
    tickers_src = set(re.findall(r"\b[A-Z]{2,5}\b", tool_blob)) | ucoins
    texts = [m["content"] for m in msgs if m["role"] == "assistant" and m.get("content")]
    order_ids = {o["id"] for o in meta["world"]["orders"]}
    for t in texts:
        for pat in BANNED:
            if re.search(pat, t, re.I | re.M):
                errs.append(f"banned /{pat}/ in: {t[:90]}")
        for m_ in CANCELLED_RE.finditer(t):
            if not NEG_OK.search(t[max(0, m_.start() - 40):m_.start()]):
                errs.append(f"'{m_.group(0)}' stated as fact: {t[:90]}")
        words = len(t.split())
        if words > 70:
            errs.append(f"too long ({words}) {t[:50]}")
        t_clean = NUM_LINE_IDX.sub("", t)
        for tok in CP.NUM_RE.findall(t_clean):
            try:
                d = dec(tok.replace("$", "").replace(",", "").replace("+", "")).copy_abs()
            except Exception:
                continue
            if d not in src:
                errs.append(f"number {tok} not backed in: {t[:90]}")
        for tk in set(re.findall(r"\b[A-Z]{2,5}\b", t)) & set(sim.ASSETS):
            if tk not in tickers_src:
                errs.append(f"ticker {tk} invented")
        if t.count("?") > 1:
            errs.append("more than one question")
        for oid in order_ids:
            if re.search(r"(?<![\d.,])" + oid + r"(?![\d.,]\d)", t):
                errs.append("order id read out")
    final = msgs[-1]
    if final["role"] != "assistant" or final.get("tool_calls"):
        errs.append("last message must be assistant text")
    fc = final["content"]
    # wording by the last result
    last_res = results[-1] if (results and final_calls) else {}
    st = last_res.get("status") if isinstance(last_res, dict) else None
    if st == "card_opened" and not slots.get("absent"):
        if not re.search(r"\bcard\b", fc) or not re.search(r"\bTap Confirm\b", fc) or not re.search(r"nothing is cancelled", fc, re.I):
            errs.append("card answer lacks card / Tap Confirm / nothing is cancelled")
        if last_res.get("warning") and last_res["warning"] not in fc:
            errs.append("warning not repeated")
        if last_res["cancelling"] not in fc:
            errs.append("cancelling text not copied")
        if re.search(r"\b(Face ID|Touch ID)\b", fc):
            errs.append("card answer names a biometric")
        if kind == "cancel_all" and not re.search(r"not included|not part of this|does not touch", fc):
            errs.append("cancel_all answer lacks the TP/SL note")
    if st == "needs_choice":
        if last_res["question"] not in fc:
            errs.append("needs_choice question not asked")
    if st == "review_opened":
        if not re.search(r"\bprepared\b", fc, re.I):
            errs.append("review answer lacks 'prepared'")
        if not re.search(r"\b(Face ID|Touch ID)\b", fc):
            errs.append("review answer lacks the device word")
        if last_res.get("replaces") and last_res["replaces"] not in fc:
            errs.append("replaces not mentioned")
        for key in ("stop_loss", "take_profit"):
            if key in last_res and money(last_res[key]) not in fc:
                errs.append(f"{key} not stated with its exact digits")
    if re.search(r"\bprepared\b", fc, re.I) and st not in ("review_opened", "prepared_for_review") and \
            not re.search(r"nothing was prepared|couldn't prepare|can prepare|I'll prepare|prepare one|prepare it", fc, re.I):
        errs.append("claims prepared without a review")
    if kind == "neg_cancel_review" and not re.search(r"Tap Cancel on (the review|it)", fc, re.I):
        errs.append("cancel_review answer")
    if kind == "neg_what_happens" and CP.NUM_RE.search(fc):
        errs.append("knowledge answer holds a number")
    if kind == "err_wrong_side" and gold.get("answer_contains") and gold["answer_contains"][0] not in fc:
        errs.append("wrong_side error not stated")
    if kind == "err_blocked" and last_res.get("reasons") and last_res["reasons"][0] not in fc:
        errs.append("blocked reason not stated")
    if meta["pack_name"] == "core+solana" and not slots.get("absent"):
        errs.append("core+solana row is not an absent-tool row")
    if slots.get("absent"):
        if names != ["open_page"] or calls[0]["arguments"] != {"page": "trade", "tab": "positions"}:
            errs.append("absent-tool row must call open_page trade/positions")
        if "I put a button below" not in fc and "button below" not in fc:
            errs.append("absent-tool answer lacks 'button below'")
        if re.search(r"\bI opened\b", fc):
            errs.append("I opened")
    # 6 render round trip
    try:
        txt = "".join(t for t, trainable in H.render(msgs, True) if trainable)
        got = [{"name": tc["name"], "arguments": tc["arguments"]} for m in msgs for tc in (m.get("tool_calls") or [])]
        parsed, bad = H.parse_calls(txt)
        if bad or parsed != got:
            errs.append("render/parse_calls round trip")
    except Exception as e:
        errs.append(f"render error {e}")
    # 7 expect vs messages
    exp_calls = [x for x in meta["expect"] if "calls" in x]
    real_calls = [m["tool_calls"] for m in msgs[last_u:] if m["role"] == "assistant" and m.get("tool_calls")]
    if [x["calls"] for x in exp_calls] != real_calls:
        errs.append("expect != calls after last user")
    # 8 user language
    for u in users:
        if len(u) > 200 or re.search(r"[^\x00-\x7f]", u):
            errs.append("odd user text")
    if is_eval:
        for key in ("context", "reference", "gold", "checks", "world", "id", "script", "family", "pack"):
            if key not in ex:
                errs.append(f"eval item lacks {key}")
    return errs


def as_eval_item(ex, i):
    msgs = ex["messages"]
    m = ex["meta"]
    turns = m["turns"]
    cut = len(turns) - 1
    if m["kind"] == "cancel_noKind" and len(turns) == 2 and i % 2 == 0:
        cut = 0                      # evaluate the first turn (no kind -> needs_choice), second user turn goes to `script`
    u_idx = turns[cut]["user_index"]
    end = turns[cut + 1]["user_index"] if cut + 1 < len(turns) else len(msgs)
    gold = dict(m["turn_golds"][cut]) if len(m["turn_golds"]) > cut else dict(m["gold"])
    gold["steps"] = turns[cut]["expect"]
    script = [msgs[t["user_index"]]["content"] for t in turns[cut + 1:]]
    return {"id": f"orders_protect-{i:03d}", "kind": m["kind"], "tools": True, "bio": m["bio"], "pack": m["pack"], "pack_name": m["pack_name"],
            "family": m["family_ok"], "context": msgs[:u_idx + 1], "reference": msgs[u_idx + 1:end], "expect": turns[cut]["expect"],
            "script": script, "world": m["world"], "checks": m["checks"], "gold": gold, "messages": msgs, "meta": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=420)
    ap.add_argument("--eval-seed", type=int, default=9402)
    ap.add_argument("--eval-n", type=int, default=80)
    ap.add_argument("--train-out", default=os.path.join(LAB, "data4", "orders_protect.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval4", "orders_protect.jsonl"))
    a = ap.parse_args()
    if a.eval_seed == a.seed:
        raise SystemExit("eval seed must differ from train seed")
    train = build(a.seed, a.n, "train")
    train_users = {norm(m["content"]) for ex in train for m in ex["messages"] if m["role"] == "user"}
    ev = build(a.eval_seed, a.eval_n, "eval", forbid_users=train_users)
    items = [as_eval_item(ex, i) for i, ex in enumerate(ev)]
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
    if len(train) != a.n or len(items) != a.eval_n:
        print("count mismatch", len(train), len(items))
        bad += 1
    # distribution checks
    pk = Counter(r["meta"]["pack_name"] for r in train)
    if pk["core+solana"] != round(0.05 * a.n) or pk["all30"] != round(0.05 * a.n):
        print("pack shares", dict(pk))
        bad += 1
    bios = Counter(r["meta"]["bio"] for r in train)
    if not (0.6 <= bios["Face ID"] / len(train) <= 0.8):
        print("bio share", dict(bios))
        bad += 1
    first_users = [norm(next(m["content"] for m in r["messages"] if m["role"] == "user")) for r in train]
    uniq = len(set(first_users)) / len(first_users)
    if uniq < 0.95:
        print("first-user-text diversity", round(uniq, 3))
        bad += 1
    near = sum(1 for r in train if r["meta"]["kind"].startswith("neg_"))
    if not (0.24 <= near / len(train) <= 0.28):
        print("near-miss share", near, len(train))
        bad += 1
    if bad:
        print(f"orders_protect: self-check FAILED ({bad} problems); nothing written")
        raise SystemExit(1)
    for path, rows in ((a.train_out, train), (a.eval_out, items)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train kinds:", dict(sorted(Counter(r["meta"]["kind"] for r in train).items())))
    print("eval kinds:", dict(sorted(Counter(r["meta"]["kind"] for r in items).items())))
    print("train packs:", dict(pk), "bio:", dict(bios), "lite-ok rows:", sum(1 for r in train if "lite" in r["meta"]["family_ok"]),
          "first-turn diversity:", round(uniq, 3))
    print(f"orders_protect: {len(train)} train, {len(items)} eval, self-check ok")


if __name__ == "__main__":
    main()
