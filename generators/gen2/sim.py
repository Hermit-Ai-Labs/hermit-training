#!/usr/bin/env python3
"""Deterministic simulator for Hermit's app tools (hermit_tools.json).

Used by the data generators (to produce tool results the model learns to read) and by the eval
harness (to run real multi-step tool loops against the model). A World holds one user's state:
live prices, wallet balances and Hyperliquid account. Every number is produced here.

Liquidation formula (also used by position_math and the maths generator), isolated margin with
maintenance margin fraction mm, as the app computes it (HermitToolbox.liquidation), with s = +1 long, -1 short:
    liq = max(0, entry - s * entry * (1/lev - mm) / (1 - s * mm))
position_math defaults mm to Hyperliquid's, half the initial margin at the market's max leverage: 1 / (2 * max_lev),
or 1 / (2 * lev) when the max leverage isn't known.
"""
import json, random

# symbol -> (low, high) realistic band for sampling, decimals for display
ASSETS = {
    "BTC": (58000, 118000, 0), "ETH": (1800, 4800, 2), "SOL": (95, 260, 2), "BNB": (480, 900, 2),
    "HYPE": (18, 55, 3), "XRP": (0.45, 3.2, 4), "DOGE": (0.08, 0.42, 5), "AVAX": (14, 48, 2),
    "LINK": (9, 28, 3), "ARB": (0.3, 1.6, 4), "OP": (0.6, 3.4, 4), "SUI": (0.8, 4.8, 4), "TON": (2.2, 7.5, 3),
    "PEPE": (0.000006, 0.000026, 9), "WIF": (0.6, 3.8, 4), "BONK": (0.000012, 0.000045, 9), "JUP": (0.35, 1.4, 4),
    "AAVE": (80, 380, 2), "UNI": (5, 17, 3), "LTC": (60, 130, 2), "ADA": (0.3, 1.2, 4), "TRX": (0.1, 0.35, 5),
    "ENA": (0.25, 1.2, 4), "TAO": (250, 650, 2), "NEAR": (2, 8, 3), "APT": (4, 14, 3), "INJ": (12, 40, 3),
    "PENDLE": (2, 7, 3), "POL": (0.15, 0.7, 4), "MON": (0.02, 0.2, 5), "FARTCOIN": (0.4, 1.8, 4), "TRUMP": (6, 30, 3), "ZEC": (30, 400, 2), "MKR": (1100, 3200, 1),
}
STABLES = {"USDC": 1.0, "USDT": 1.0, "USDT0": 1.0, "DAI": 1.0, "USDe": 1.0}
CHAINS_EVM = ["ethereum", "base", "arbitrum", "bsc", "polygon", "monad"]
CHAINS_ALL = CHAINS_EVM + ["solana"]
CHAIN_NATIVE = {"ethereum": "ETH", "base": "ETH", "arbitrum": "ETH", "bsc": "BNB", "monad": "MON",
                "polygon": "POL", "solana": "SOL", "hyperliquid": "USDC"}
CHAIN_TOKENS = {"ethereum": ["ETH", "USDC", "USDT"], "base": ["ETH", "USDC"], "arbitrum": ["ETH", "USDC", "USDT0"],
                "bsc": ["BNB", "USDT", "USDC"], "polygon": ["POL", "USDC"], "monad": ["MON", "USDC"],
                "solana": ["SOL", "USDC", "USDT"], "hyperliquid": ["USDC"]}
PROTOCOLS = [("Aave v3", ["ethereum", "base", "arbitrum", "polygon", "bsc"], ["USDC", "USDT", "ETH", "DAI"]),
             ("Morpho", ["ethereum", "base"], ["USDC", "ETH"]), ("Compound v3", ["ethereum", "base", "arbitrum"], ["USDC", "ETH"]),
             ("Kamino", ["solana"], ["USDC", "SOL"]), ("Spark", ["ethereum"], ["DAI", "USDC"]),
             ("Fluid", ["ethereum", "arbitrum"], ["USDC", "ETH"]), ("Hyperliquid HLP", ["hyperliquid"], ["USDC"]),
             ("Ethena", ["ethereum"], ["USDe"]), ("Lido", ["ethereum"], ["ETH"]), ("Jito", ["solana"], ["SOL"])]


def rnd_price(sym, rng):
    if sym in STABLES:
        return 1.0
    lo, hi, dp = ASSETS[sym]
    p = rng.uniform(lo, hi)
    return round(p, dp) if dp else float(round(p))


class World:
    def __init__(self, rng, empty_hl=None, empty_wallet=None):
        self.rng = rng
        self.prices = {s: rnd_price(s, rng) for s in ASSETS}
        self.prices.update(STABLES)
        self.change = {s: round(rng.uniform(-9, 9), 2) for s in self.prices}
        for s in STABLES:
            self.change[s] = round(rng.uniform(-0.05, 0.05), 2)
        self.down = set()           # symbols whose price feed errors
        self.fail_next = None        # tool name that returns an error next time
        # wallet
        self.balances = []
        if not (empty_wallet if empty_wallet is not None else rng.random() < 0.08):
            chains = rng.sample(CHAINS_ALL, rng.choice([1, 2, 3, 4]))
            for c in chains:
                for sym in rng.sample(CHAIN_TOKENS[c], rng.choice([1, 2, 3]) if len(CHAIN_TOKENS[c]) > 2 else 1):
                    p = self.prices[sym]
                    usd_target = rng.choice([rng.uniform(5, 300), rng.uniform(300, 5000), rng.uniform(5000, 40000)])
                    amt = usd_target / p
                    amt = round(amt, 2 if p < 5 else (4 if p < 5000 else 6))
                    if amt <= 0:
                        continue
                    self.balances.append({"symbol": sym, "chain": c, "amount": amt, "usd": round(amt * p, 2)})
        # hyperliquid
        self.hl_positions, self.hl_orders = [], []
        hl_empty = empty_hl if empty_hl is not None else rng.random() < 0.2
        self.hl_value = 0.0 if hl_empty and rng.random() < 0.4 else round(rng.uniform(150, 25000), 2)
        if not hl_empty and self.hl_value > 0:
            for coin in rng.sample(["BTC", "ETH", "SOL", "HYPE", "DOGE", "XRP", "SUI", "AVAX"], rng.choice([1, 1, 2, 3])):
                side = rng.choice(["long", "short"])
                lev = rng.choice([2, 3, 5, 5, 10, 10, 15, 20])
                mark = self.prices[coin]
                entry = mark * (1 + rng.uniform(-0.08, 0.08))
                entry = round(entry, ASSETS[coin][2])
                size_usd = round(rng.uniform(0.05, 0.28) * self.hl_value * lev, 2)
                size = size_usd / entry
                sgn = 1 if side == "long" else -1
                pnl = round(sgn * (mark - entry) * size, 2)
                if lev * abs(mark - entry) / entry > 0.85:
                    entry = round(mark * (1 - sgn * 0.6 / lev), ASSETS[coin][2]); size = size_usd / entry; pnl = round(sgn * (mark - entry) * size, 2)
                liq = liq_price(side, entry, lev, 0.005)
                self.hl_positions.append({"coin": coin, "side": side, "size": round(size, 4), "entry": entry,
                                          "mark": mark, "liq": round(liq, ASSETS[coin][2] or 0), "leverage": lev,
                                          "unrealized_pnl": pnl, "margin": round(size_usd / lev, 2)})
            if rng.random() < 0.4:
                coin = rng.choice(["BTC", "ETH", "SOL"])
                self.hl_orders.append({"coin": coin, "side": rng.choice(["buy", "sell"]), "type": "limit",
                                       "price": round(self.prices[coin] * rng.choice([0.95, 0.97, 1.03, 1.05]), ASSETS[coin][2]),
                                       "size": round(rng.uniform(0.01, 2), 3)})
        self.prep_n = 0

    STATE_KEYS = ("prices", "change", "balances", "hl_positions", "hl_orders", "hl_value")

    def state(self):
        return json.loads(json.dumps({k: getattr(self, k) for k in self.STATE_KEYS} |
                                     {"down": sorted(self.down), "fail_next": self.fail_next}))

    @classmethod
    def from_state(cls, st, seed=0):
        w = cls.__new__(cls)
        w.rng = random.Random(seed)
        for k in cls.STATE_KEYS:
            setattr(w, k, json.loads(json.dumps(st[k])))
        w.down = set(st.get("down", []))
        w.fail_next = st.get("fail_next")
        w.prep_n = 0
        return w

    # ---------------- tools
    def call(self, name, args):
        if self.fail_next == name:
            self.fail_next = None
            return {"error": self.rng.choice(["network timeout, try again", "service unavailable", "request timed out"])}
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args if isinstance(args, dict) else {}))
        except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as e:
            return {"error": f"bad arguments: {e}"}

    def t_calculator(self, expression):
        from hermit_fmt import calc, calc_str
        try:
            return {"result": calc_str(calc(str(expression)))}
        except ZeroDivisionError:
            return {"error": "division by zero"}
        except Exception:
            return {"error": "could not evaluate expression"}

    def t_get_price(self, symbol):
        s = str(symbol).upper().replace("$", "").strip()
        if s in ("USDE",):
            s = "USDe"
        if s in self.down:
            return {"error": f"price feed unavailable for {s}"}
        if s not in self.prices:
            return {"error": f"unknown symbol {symbol}"}
        return {"symbol": s, "price": self.prices[s], "change_24h_pct": self.change[s],
                "source": "hyperliquid" if s in ASSETS and s not in ("PEPE", "BONK", "MKR") else "binance"}

    def t_get_balances(self, chain=None):
        rows = [b for b in self.balances if chain is None or b["chain"] == str(chain).lower()]
        return {"balances": rows, "total_usd": round(sum(b["usd"] for b in rows), 2)}

    def t_hyperliquid_account(self):
        used = round(sum(p["margin"] for p in self.hl_positions), 2)
        upnl = round(sum(p["unrealized_pnl"] for p in self.hl_positions), 2)
        value = round(self.hl_value + upnl, 2)
        return {"account_value": value, "margin_used": used, "available_usdc": round(max(value - used, 0), 2),
                "positions": self.hl_positions, "open_orders": self.hl_orders}

    def t_position_math(self, side, entry, size_usd, leverage, price=None, maintenance_margin=None, coin=None, max_leverage=None):
        side = str(side).lower()
        if side not in ("long", "short"):
            return {"error": "side must be long or short"}
        entry, size_usd, leverage = float(entry), float(size_usd), float(leverage)
        if entry <= 0 or size_usd <= 0 or leverage <= 0:
            return {"error": "entry, size_usd and leverage must be positive"}
        mm = None if maintenance_margin is None else float(maintenance_margin)
        max_lev = None if max_leverage is None else float(max_leverage)
        out = position_math(side, entry, size_usd, leverage, None if price is None else float(price), mm, max_lev)
        return out

    def t_search_markets(self, query):
        q = str(query).upper()
        hits = [s for s in ASSETS if s in q.replace("-", " ").split() or q.startswith(s) or s in q]
        if not hits:
            return {"results": []}
        out = []
        for s in hits[:2]:
            out.append({"name": f"{s}-PERP", "type": "perp", "venue": "hyperliquid", "price": self.prices[s],
                        "funding_8h_pct": round(self.rng.uniform(-0.02, 0.06), 4),
                        "open_interest_usd": round(self.rng.uniform(2e6, 3e9), -3)})
            if self.rng.random() < 0.6:
                out.append({"name": f"{s}/USDC", "type": "spot", "venue": "hyperliquid" if s == "HYPE" else "relay",
                            "price": self.prices[s]})
            if self.rng.random() < 0.4:
                out.append({"name": f"{s}-USDC pool", "type": "pool", "venue": self.rng.choice(["Uniswap v3", "Aerodrome", "Raydium"]),
                            "apy_pct": round(self.rng.uniform(3, 60), 2), "tvl_usd": round(self.rng.uniform(5e5, 9e7), -3)})
        return {"results": out}

    def t_get_yields(self, asset=None, chain=None):
        a = None if asset is None else str(asset).upper().replace("USDE", "USDe")
        c = None if chain is None else str(chain).lower()
        rows = []
        for proto, chains, assets in PROTOCOLS:
            for ch in chains:
                for s in assets:
                    if (a and s.upper() != a.upper()) or (c and ch != c):
                        continue
                    base = {"USDC": 5, "USDT": 5, "DAI": 5, "USDe": 9, "ETH": 2.8, "SOL": 6.5}.get(s, 4)
                    rows.append({"protocol": proto, "chain": ch, "asset": s,
                                 "apy_pct": round(max(0.3, base * self.rng.uniform(0.5, 1.8)), 2),
                                 "tvl_usd": round(self.rng.uniform(4e6, 2.5e9), -3)})
        rows.sort(key=lambda r: -r["apy_pct"])
        return {"pools": rows[:5]}

    def _prep(self, kind):
        self.prep_n += 1
        return {"status": "review_opened", "kind": kind, "review_id": f"{kind[:2]}-{self.rng.randint(1000, 9999)}",
                "note": "Waiting for the user to approve with Face ID."}

    def t_prepare_trade(self, coin, side, size_usd, leverage, order_type, limit_price=None, take_profit=None, stop_loss=None):
        if str(order_type) == "limit" and limit_price is None:
            return {"error": "limit_price is required for a limit order"}
        if float(leverage) > 40:
            return {"error": f"max leverage for {coin} is 40x"}
        if self.hl_value <= 0:
            return {"error": "no USDC in the Hyperliquid account"}
        need = float(size_usd) / float(leverage)
        avail = self.t_hyperliquid_account()["available_usdc"]
        if need > avail:
            return {"error": f"not enough margin: needs ${need:,.2f}, available ${avail:,.2f}"}
        return self._prep("trade")

    def t_prepare_swap(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        have = sum(b["amount"] for b in self.balances if b["symbol"].upper() == str(from_symbol).upper()
                   and b["chain"] == str(from_chain).lower())
        if have < float(amount):
            return {"error": f"insufficient balance: {from_symbol} on {from_chain} is {have:g}"}
        return self._prep("swap")

    def t_prepare_send(self, to_address, amount, symbol, chain):
        have = sum(b["amount"] for b in self.balances if b["symbol"].upper() == str(symbol).upper()
                   and b["chain"] == str(chain).lower())
        if have < float(amount):
            return {"error": f"insufficient balance: {symbol} on {chain} is {have:g}"}
        return self._prep("send")

    QUOTE_CHAINS = ("ethereum", "base", "arbitrum", "bsc", "polygon", "monad", "solana", "near", "bitcoin", "hyperliquid")

    def t_get_quote(self, from_symbol, from_chain, to_symbol, to_chain, amount):
        """NEAR Intents (1Click) quote: nothing is prepared."""
        fc, tc = str(from_chain).lower(), str(to_chain).lower()
        fs, ts = str(from_symbol).upper().replace("USDE", "USDe"), str(to_symbol).upper().replace("USDE", "USDe")
        for c in (fc, tc):
            if c not in self.QUOTE_CHAINS:
                return {"error": f"unsupported chain {c}"}
        if fc == "bitcoin":
            return {"error": "bitcoin is supported as a destination only"}
        if tc == "hyperliquid" and ts != "USDC":
            return {"error": "hyperliquid destination supports USDC only"}
        if fc == "hyperliquid":
            return {"error": "hyperliquid is supported as a destination only"}
        for x in (fs, ts):
            if x not in self.prices:
                return {"error": f"unknown token {x}"}
        amount = float(amount)
        if amount <= 0:
            return {"error": "amount must be positive"}
        usd_in = amount * self.prices[fs]
        if usd_in < 1:
            return {"error": f"amount too low: minimum is about $1 (this is ${usd_in:.2f})"}
        fee = round(max(0.05, usd_in * self.rng.uniform(0.0008, 0.003)), 2)
        usd_out = usd_in - fee
        out = usd_out / self.prices[ts]
        dp = 8 if self.prices[ts] > 1000 else (6 if self.prices[ts] > 1 else 2)
        if self.prices[ts] < 0.001:
            dp = 0
        return {"provider": "NEAR Intents", "from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc,
                "amount_in": amount, "amount_out": round(out, dp), "amount_in_usd": round(usd_in, 2),
                "amount_out_usd": round(usd_out, 2), "fee_usd": fee,
                "eta_seconds": self.rng.choice([20, 30, 45, 60, 90, 120, 300, 600 if tc == "bitcoin" else 120])}

    def t_open_add_funds(self, goal):
        if goal not in ("wallet", "hyperliquid"):
            return {"error": "goal must be wallet or hyperliquid"}
        return {"status": "opened", "screen": "add_funds", "goal": goal}

    def ensure_balance(self, symbol, chain, amount):
        """Make sure the wallet holds at least `amount` of symbol on chain (for success-path examples)."""
        for b in self.balances:
            if b["symbol"] == symbol and b["chain"] == chain:
                if b["amount"] < amount:
                    b["amount"] = round(amount * self.rng.uniform(1.05, 3), 6)
                    b["usd"] = round(b["amount"] * self.prices.get(symbol, 1.0), 2)
                return
        amt = round(amount * self.rng.uniform(1.05, 3), 6)
        self.balances.append({"symbol": symbol, "chain": chain, "amount": amt,
                              "usd": round(amt * self.prices.get(symbol, 1.0), 2)})


def liq_price(side, entry, lev, mm=0.0):
    s = 1 if side == "long" else -1
    return max(0.0, entry - s * entry * (1 / lev - mm) / (1 - s * mm))


def position_math(side, entry, size_usd, leverage, price=None, mm=None, max_lev=None):
    if mm is None:
        mm = 1 / (2 * max_lev) if max_lev else 1 / (2 * max(leverage, 1))
    margin = size_usd / leverage
    qty = size_usd / entry
    liq = liq_price(side, entry, leverage, mm)
    out = {"side": side, "entry": entry, "size_usd": size_usd, "leverage": leverage, "margin": round(margin, 2),
           "quantity": round(qty, 6), "liquidation_price": round(liq, 6 if liq < 1 else 2),
           "distance_to_liq_pct": round(abs(liq - entry) / entry * 100, 2), "maintenance_margin_used": round(mm, 6)}
    if price is not None:
        sgn = 1 if side == "long" else -1
        pnl = sgn * (price - entry) * qty
        out.update({"price": price, "pnl": round(pnl, 2), "roe_pct": round(pnl / margin * 100, 2)})
    return out


if __name__ == "__main__":
    w = World(random.Random(3))
    for n, a in [("get_price", {"symbol": "btc"}), ("get_balances", {}), ("hyperliquid_account", {}),
                 ("position_math", {"side": "long", "entry": 3000, "size_usd": 1000, "leverage": 10, "price": 3150}),
                 ("search_markets", {"query": "SOL perp"}), ("get_yields", {"asset": "USDC", "chain": "base"}),
                 ("calculator", {"expression": "0.5*3412.55"})]:
        print(n, json.dumps(w.call(n, a))[:300])
