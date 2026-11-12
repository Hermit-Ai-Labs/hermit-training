#!/usr/bin/env python3
"""Agent / tool-calling conversations for Hermit 1.2, against the app's real tool set (hermit_tools.json).

Every tool result comes from sim.World and every number in a final answer comes from a tool result
or from the user. Each example records:
  meta.kind      scenario name
  meta.world     simulator state (so the eval harness can replay the conversation live)
  meta.expect    list of expected assistant decisions, one per assistant turn after the LAST user
                 message: {"calls": [{"name", "arguments"}]} | {"ask": true} | {"text": true}
Usage: python3 build_agent.py --seed 1 --scale 1.0 --out ../data2/agent_train.jsonl
"""
import argparse, json, os, random, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermit_fmt as H
import sim
import trading_math2 as TM

NAMES = {"bitcoin": "BTC", "btc": "BTC", "ether": "ETH", "ethereum": "ETH", "eth": "ETH", "solana": "SOL", "sol": "SOL",
         "doge": "DOGE", "dogecoin": "DOGE", "hype": "HYPE", "hyperliquid": "HYPE", "xrp": "XRP", "ripple": "XRP",
         "bnb": "BNB", "avax": "AVAX", "chainlink": "LINK", "link": "LINK", "sui": "SUI", "pepe": "PEPE", "wif": "WIF",
         "aave": "AAVE", "tao": "TAO", "ltc": "LTC", "litecoin": "LTC", "ena": "ENA", "ton": "TON", "zec": "ZEC"}
MAJORS = ["BTC", "ETH", "SOL", "HYPE", "BNB", "XRP", "DOGE", "SUI", "AVAX", "LINK", "TAO", "ENA", "ZEC", "AAVE"]
PERPS = ["BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "SUI", "AVAX", "LINK", "BNB", "TAO", "ENA"]
CHAIN_SAY = {"ethereum": ["Ethereum", "ethereum", "mainnet", "ETH mainnet"], "base": ["Base", "base"],
             "arbitrum": ["Arbitrum", "arbitrum", "arb"], "bsc": ["BNB Chain", "BSC", "bsc", "BNB chain"],
             "polygon": ["Polygon", "polygon"], "monad": ["Monad", "monad"], "solana": ["Solana", "solana", "sol"]}


def S(x):
    return H.smart_money(x)


def big(x):
    for d, u in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= d:
            return f"${x / d:.1f}{u}"
    return S(x)


def chg(c):
    return f"{'up' if c >= 0 else 'down'} {abs(c):g}%"


def say_chain(rng, c):
    return rng.choice(CHAIN_SAY[c])


class Conv:
    """Builds one conversation while executing tool calls against a World."""

    def __init__(self, rng, world, system=None, tools=True):
        self.rng, self.w, self.tools = rng, world, tools
        self.world0 = world.state()
        self.msgs = [{"role": "system", "content": H.with_user(system or H.SYSTEM_AGENT, rng)}]
        self.expect = []

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        self.expect = []

    def call(self, *calls, lead=""):
        """calls: (name, args) tuples executed together (parallel calls in one turn). Returns results."""
        tcs = [{"name": n, "arguments": a} for n, a in calls]
        self.msgs.append({"role": "assistant", "content": lead, "tool_calls": tcs})
        self.expect.append({"calls": tcs})
        res = []
        for n, a in calls:
            r = self.w.call(n, a)
            res.append(r)
            self.msgs.append({"role": "tool", "content": json.dumps(r, ensure_ascii=False)})
        return res if len(res) > 1 else res[0]

    def say(self, text, ask=False):
        self.msgs.append({"role": "assistant", "content": text})
        self.expect.append({"ask": True} if ask else {"text": True})

    def ex(self, kind):
        return {"messages": self.msgs, "tools": self.tools,
                "meta": {"source": "agent", "kind": kind, "world": self.world0, "world_final": self.w.state(), "expect": self.expect}}


def amt_text(rng, x):
    """How a user might type an amount; returns (text, value)."""
    s = H.calc_str(x)
    if x >= 1000 and float(x).is_integer() and rng.random() < 0.4:
        return f"{int(x):,}", x
    if 0 < x < 1 and rng.random() < 0.15:
        return s[1:], x            # ".5"
    return s, x


def rand_amount(rng, kind="token"):
    r = rng.random()
    if kind == "usd":
        return rng.choice([10, 20, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1200, 1500, 2000, 2500, 5000, 10000])
    if r < 0.35:
        return rng.choice([1, 2, 3, 5, 10, 20, 25, 50, 100, 150, 200, 250, 500, 1000, 1500, 2500])
    if r < 0.8:
        return round(rng.uniform(0.01, 30), rng.choice([2, 3, 4]))
    return round(rng.uniform(0.0001, 0.01), rng.choice([4, 5, 6]))


# ======================================================================= scenarios
def sc_price_single(rng):
    w = sim.World(rng); c = Conv(rng, w)
    sym = rng.choice(MAJORS + list(sim.ASSETS))
    alias = [k for k, v in NAMES.items() if v == sym]
    word = rng.choice(alias + [sym, sym.lower(), "$" + sym]) if alias else rng.choice([sym, sym.lower(), "$" + sym])
    mode = rng.random()
    if mode < 0.08:
        w.down.add(sym)
    elif mode < 0.13:
        sym = rng.choice(["ZORP", "MOONX", "BLORB", "QQQX", "FLOKIZ", "GRIM", "NUBZ"]); word = rng.choice([sym, sym.lower()])
    c.user(rng.choice([f"what's {word} at?", f"price of {word}?", f"how much is {word} right now", f"{word} price",
                       f"where is {word} trading", f"check {word} for me", f"is {word} up or down today?",
                       f"what's {word} doing today", f"{word}?", f"current {word} price pls"]))
    r = c.call(("get_price", {"symbol": sym}))
    if "error" in r:
        if "unknown" in r["error"]:
            c.say(rng.choice([f"I couldn't find a market for {sym}. Check the ticker, or tell me the full name and chain and I'll look again.",
                              f"There's no price for {sym} on Hyperliquid or Binance. Is the ticker right?"]))
        else:
            c.say(rng.choice([f"I couldn't get a live {sym} price right now, the price feed isn't answering. Try again in a moment.",
                              f"The {sym} price feed is down at the moment, so I don't have a number for you. Want me to try again?"]))
        return c.ex("price_error")
    p, ch = S(r["price"]), r["change_24h_pct"]
    c.say(rng.choice([f"{sym} is at {p}, {chg(ch)} in the last 24 hours.", f"{sym} is trading at {p} ({chg(ch)} over 24h).",
                      f"{p}. That's {chg(ch)} on the day.", f"Right now {sym} is {p}, {chg(ch)} in 24h."]))
    return c.ex("price")


def sc_price_multi(rng):
    w = sim.World(rng); c = Conv(rng, w)
    a, b = rng.sample(MAJORS, 2)
    if rng.random() < 0.5:
        c.user(rng.choice([f"{a} and {b} prices?", f"what are {a} and {b} at", f"give me {a} and {b}", f"price check {a}, {b}"]))
        ra, rb = c.call(("get_price", {"symbol": a}), ("get_price", {"symbol": b}))
        c.say(f"{a} is at {S(ra['price'])} ({chg(ra['change_24h_pct'])}) and {b} is at {S(rb['price'])} ({chg(rb['change_24h_pct'])}) over the last 24 hours.")
        return c.ex("price_multi")
    c.user(rng.choice([f"which is doing better today, {a} or {b}?", f"{a} vs {b} today, which is up more?"]))
    ra, rb = c.call(("get_price", {"symbol": a}), ("get_price", {"symbol": b}))
    ca, cb = ra["change_24h_pct"], rb["change_24h_pct"]
    win, lose = (a, b) if ca >= cb else (b, a)
    cw, cl = max(ca, cb), min(ca, cb)
    c.say(f"{win} is doing better: {chg(cw)} in 24 hours versus {chg(cl)} for {lose}. {a} is at {S(ra['price'])} and {b} is at {S(rb['price'])}.")
    return c.ex("price_compare")


def sc_value_of(rng):
    w = sim.World(rng); c = Conv(rng, w)
    sym = rng.choice(MAJORS); q = rand_amount(rng)
    qt, _ = amt_text(rng, q)
    c.user(rng.choice([f"how much is {qt} {sym} worth?", f"what's {qt} {sym} in dollars", f"value of {qt} {sym}?",
                       f"if I sold {qt} {sym} now how much would I get roughly"]))
    r = c.call(("get_price", {"symbol": sym}))
    expr = f"{H.calc_str(q)} * {H.calc_str(r['price'])}"
    v = float(c.call(("calculator", {"expression": expr}))["result"])
    c.say(rng.choice([f"{qt} {sym} is worth about {S(v)} at {S(r['price'])} per {sym}.",
                      f"About {S(v)}. {sym} is at {S(r['price'])} right now, before any fees or slippage."]))
    return c.ex("value_of")


def sc_balances(rng):
    w = sim.World(rng); c = Conv(rng, w)
    m = rng.random()
    if m < 0.55 or not w.balances:
        c.user(rng.choice(["what's in my wallet?", "show my balances", "how much do I have?", "what do I hold", "my balance?",
                           "total portfolio value?", "what tokens do I have", "check my wallet"]))
        r = c.call(("get_balances", {}))
        rows = sorted(r["balances"], key=lambda b: -b["usd"])
        if not rows:
            c.say(rng.choice(["Your wallet is empty right now. Want me to open Add funds so you can deposit?",
                              "Nothing in the wallet yet. If you want to deposit, I can open Add funds; USDC on Base or Arbitrum is the cheapest way in."]))
            return c.ex("balances_empty")
        lines = [f"- {H.num(b['amount'], 6)} {b['symbol']} on {CHAIN_SAY[b['chain']][0]} ({S(b['usd'])})" for b in rows[:6]]
        more = f"\n- plus {len(rows) - 6} smaller balances" if len(rows) > 6 else ""
        c.say(f"You have {S(r['total_usd'])} in total:\n" + "\n".join(lines) + more)
        return c.ex("balances")
    b = rng.choice(w.balances); ch = b["chain"]
    if m < 0.8:
        c.user(rng.choice([f"how much {b['symbol']} do I have on {say_chain(rng, ch)}?", f"my {b['symbol']} balance on {say_chain(rng, ch)}",
                           f"what's on {say_chain(rng, ch)}?"]))
        r = c.call(("get_balances", {"chain": ch}))
        rows = [x for x in r["balances"] if x["symbol"] == b["symbol"]]
        x = rows[0]
        c.say(f"You have {H.num(x['amount'], 6)} {x['symbol']} on {CHAIN_SAY[ch][0]}, worth about {S(x['usd'])}.")
        return c.ex("balances_chain")
    empty = [ch for ch in sim.CHAINS_ALL if ch not in {x["chain"] for x in w.balances}]
    if not empty:
        return sc_balances(rng)
    ch = rng.choice(empty)
    c.user(rng.choice([f"anything on {say_chain(rng, ch)}?", f"what do I have on {say_chain(rng, ch)}"]))
    c.call(("get_balances", {"chain": ch}))
    c.say(f"Nothing on {CHAIN_SAY[ch][0]} right now.")
    return c.ex("balances_chain_empty")


def pos_line(p):
    return (f"{p['coin']} {p['side']} {p['leverage']}x: size {H.num(p['size'], 4)}, entry {S(p['entry'])}, mark {S(p['mark'])}, "
            f"liquidation {S(p['liq'])}, PnL {S(p['unrealized_pnl'])}")


def sc_hl_account(rng):
    w = sim.World(rng); c = Conv(rng, w)
    m = rng.random()
    if m < 0.5 or not w.hl_positions:
        c.user(rng.choice(["how are my positions doing?", "show my hyperliquid account", "what positions do I have open",
                           "how's my PnL", "check my perps", "any open orders?", "how much margin do I have left on hyperliquid"]))
        r = c.call(("hyperliquid_account", {}))
        if not r["positions"]:
            if r["account_value"] <= 0:
                c.say("Your Hyperliquid account is empty: no USDC and no positions. Want me to open Add funds for Hyperliquid?")
            else:
                c.say(f"No open positions. Your account value is {S(r['account_value'])} with {S(r['available_usdc'])} available to trade.")
            return c.ex("hl_empty")
        lines = "\n".join("- " + pos_line(p) for p in r["positions"])
        orders = ""
        if r["open_orders"]:
            orders = "\nOpen orders: " + "; ".join(f"{o['side']} {H.num(o['size'], 4)} {o['coin']} at {S(o['price'])}" for o in r["open_orders"]) + "."
        c.say(f"Account value {S(r['account_value'])}, margin used {S(r['margin_used'])}, available {S(r['available_usdc'])}.\n{lines}{orders}")
        return c.ex("hl_account")
    p = rng.choice(w.hl_positions)
    if m < 0.8:
        c.user(rng.choice([f"how close is my {p['coin']} position to liquidation?", f"where's my {p['coin']} liq and how far away is it?",
                           f"am I about to get liquidated on {p['coin']}?"]))
        r = c.call(("hyperliquid_account", {}))
        pp = [x for x in r["positions"] if x["coin"] == p["coin"]][0]
        expr = f"abs({H.calc_str(pp['mark'])} - {H.calc_str(pp['liq'])}) / {H.calc_str(pp['mark'])} * 100"
        d = float(c.call(("calculator", {"expression": expr}))["result"])
        warn = " That's tight; a normal day's move could take it out. Consider adding margin or cutting size." if d < 5 else (
               " That's a reasonable cushion, but keep a stop so it never gets there." if d < 20 else " That's a wide cushion.")
        c.say(f"Your {pp['coin']} {pp['side']} is liquidated at {S(pp['liq'])}. Mark is {S(pp['mark'])}, so price has to move {H.pct(d)} "
              f"{'down' if pp['side'] == 'long' else 'up'} to get there.{warn}")
        return c.ex("hl_liq_distance")
    c.user(rng.choice([f"what's my PnL on {p['coin']}?", f"am I up on my {p['coin']} trade?", f"how is the {p['coin']} {p['side']} doing"]))
    r = c.call(("hyperliquid_account", {}))
    pp = [x for x in r["positions"] if x["coin"] == p["coin"]][0]
    c.say(f"Your {pp['coin']} {pp['side']} ({pp['leverage']}x) is {'up' if pp['unrealized_pnl'] >= 0 else 'down'} {S(abs(pp['unrealized_pnl']))}. "
          f"Entry {S(pp['entry'])}, mark {S(pp['mark'])}, liquidation {S(pp['liq'])}.")
    return c.ex("hl_pnl")


def pm_args(rng, with_price=None, with_mm=None):
    coin = rng.choice(PERPS); side = rng.choice(["long", "short"])
    lo, hi, dp = sim.ASSETS[coin]
    entry = round(rng.uniform(lo, hi), dp) if dp else float(round(rng.uniform(lo, hi)))
    size = rand_amount(rng, "usd"); lev = rng.choice([2, 3, 5, 5, 10, 10, 15, 20, 25])
    a = {"side": side, "entry": entry, "size_usd": size, "leverage": lev}
    if with_price if with_price is not None else rng.random() < 0.5:
        a["price"] = round(entry * (1 + rng.uniform(-0.12, 0.12)), dp) if dp else float(round(entry * (1 + rng.uniform(-0.12, 0.12))))
    if with_mm if with_mm is not None else rng.random() < 0.25:
        a["maintenance_margin"] = rng.choice([0.005, 0.01, 0.0125, 0.02])
    return coin, a


def sc_position_math(rng):
    w = sim.World(rng); c = Conv(rng, w)
    coin, a = pm_args(rng)
    e, s, l = H.calc_str(a["entry"]), a["size_usd"], H.calc_str(a["leverage"])
    st = rng.choice([f"{s}", f"${s}", f"${s:,}"])
    q = rng.choice([f"if I {a['side']} {coin} at {e} with {st} at {l}x, where's my liquidation?",
                    f"{a['side']} {coin} {st} {l}x entry {e}. liq price and margin?",
                    f"what margin do I need for a {st} {coin} {a['side']} at {l}x from {e}, and where does it liquidate?"])
    if "price" in a:
        q += rng.choice([f" and what's my pnl if it goes to {H.calc_str(a['price'])}?", f" Also PnL and ROE at {H.calc_str(a['price'])}?"])
    if "maintenance_margin" in a:
        q += f" Maintenance margin is {H.calc_str(a['maintenance_margin'] * 100)}%."
    c.user(q)
    r = c.call(("position_math", a))
    txt = f"Margin is {S(r['margin'])} and liquidation is at {S(r['liquidation_price'])}, {H.pct(r['distance_to_liq_pct'])} from your entry."
    if "pnl" in r:
        txt += f" At {S(r['price'])} your PnL is {S(r['pnl'])}, an ROE of {H.pct(r['roe_pct'])}."
    if a["leverage"] >= 15:
        txt += " At this leverage a small wick can liquidate you, so set a stop well before that level."
    c.say(txt)
    return c.ex("position_math")


def sc_position_math_missing(rng):
    w = sim.World(rng); c = Conv(rng, w)
    coin, a = pm_args(rng, with_price=False, with_mm=False)
    drop = rng.choice(["leverage", "size_usd", "entry"])
    parts = {"entry": f"at {H.calc_str(a['entry'])}", "size_usd": f"with ${a['size_usd']}", "leverage": f"at {a['leverage']}x"}
    q = f"where would my liquidation be if I {a['side']} {coin} " + " ".join(v for k, v in parts.items() if k != drop) + "?"
    c.user(q)
    ask = {"leverage": "What leverage are you planning to use?", "size_usd": "How big is the position, in dollars?",
           "entry": f"What entry price? I can use the current {coin} price if you want."}[drop]
    c.say(ask, ask=True)
    ans = {"leverage": f"{a['leverage']}x", "size_usd": f"${a['size_usd']}", "entry": H.calc_str(a["entry"])}[drop]
    c.user(rng.choice([ans, f"{ans} please", f"say {ans}"]))
    r = c.call(("position_math", a))
    c.say(f"Liquidation would be at {S(r['liquidation_price'])}, {H.pct(r['distance_to_liq_pct'])} from entry, with {S(r['margin'])} of margin.")
    return c.ex("position_math_missing")


def trade_text(a):
    lev = H.calc_str(a["leverage"])
    t = f"a {lev}x {a['side']} on {a['coin']} for ${a['size_usd']:,}" if float(a["size_usd"]).is_integer() else f"a {lev}x {a['side']} on {a['coin']} for {S(a['size_usd'])}"
    t += " at market" if a["order_type"] == "market" else f" with a limit at {S(a['limit_price'])}"
    extra = []
    if a.get("take_profit") is not None:
        extra.append(f"take-profit at {S(a['take_profit'])}")
    if a.get("stop_loss") is not None:
        extra.append(f"stop-loss at {S(a['stop_loss'])}")
    if extra:
        t += ", " + " and ".join(extra)
    return t


DONE = ["Review it and approve with Face ID when you're ready.", "It's on the review screen; nothing happens until you approve it with Face ID.",
        "Check the details on the review screen and approve with Face ID to place it.", "Approve it with Face ID if it looks right."]


def sc_trade_full(rng):
    w = sim.World(rng, empty_hl=False); c = Conv(rng, w)
    if w.hl_value <= 0:
        w.hl_value = round(rng.uniform(500, 20000), 2)
    coin = rng.choice(PERPS); side = rng.choice(["long", "short"]); lev = rng.choice([1, 2, 3, 5, 5, 10, 10, 15, 20, 25])
    avail = w.t_hyperliquid_account()["available_usdc"]
    size = rng.choice([x for x in [20, 50, 100, 200, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000] if x / lev <= max(avail, 12)] or [20])
    mark = w.prices[coin]; dp = sim.ASSETS[coin][2]
    a = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": rng.choice(["market", "market", "limit"])}
    rd = (lambda v: round(v, dp) if dp else float(round(v)))
    if a["order_type"] == "limit":
        a["limit_price"] = rd(mark * (1 + (-1 if side == "long" else 1) * rng.choice([0.01, 0.02, 0.03, 0.05])))
    ref = a.get("limit_price", mark)
    if rng.random() < 0.4:
        a["take_profit"] = rd(ref * (1 + (1 if side == "long" else -1) * rng.choice([0.03, 0.05, 0.08, 0.1, 0.2])))
    if rng.random() < 0.5:
        a["stop_loss"] = rd(ref * (1 - (1 if side == "long" else -1) * rng.choice([0.02, 0.03, 0.05, 0.08])))
    verb = {"long": rng.choice(["long", "buy", "go long"]), "short": rng.choice(["short", "sell", "go short"])}[side]
    sz = rng.choice([f"${size}", f"{size} usd", f"${size:,}", f"{size} dollars of"]) if size >= 1000 else rng.choice([f"${size}", f"{size} usd", f"{size}$"])
    q = rng.choice([f"{verb} {sz} {coin} {lev}x", f"{verb} {coin} {sz} at {lev}x", f"open a {lev}x {side} on {coin}, size {sz}"])
    q = q.replace(f"{sz} {coin}", f"{sz} {coin}").replace("dollars of " + coin, "dollars of " + coin)
    if a["order_type"] == "limit":
        q += rng.choice([f" limit at {H.calc_str(a['limit_price'])}", f", limit {H.calc_str(a['limit_price'])}", f" with a limit order at {H.calc_str(a['limit_price'])}"])
    else:
        q += rng.choice(["", " market", " at market"])
    if "take_profit" in a:
        q += rng.choice([f" tp {H.calc_str(a['take_profit'])}", f", take profit {H.calc_str(a['take_profit'])}"])
    if "stop_loss" in a:
        q += rng.choice([f" sl {H.calc_str(a['stop_loss'])}", f", stop at {H.calc_str(a['stop_loss'])}", f" stop loss {H.calc_str(a['stop_loss'])}"])
    m = rng.random()
    if m < 0.07:
        w.hl_value = 0.0; w.hl_positions = []
    elif m < 0.14:
        a["leverage"] = lev = rng.choice([50, 75, 100]); q = re.sub(r"\b\d+x\b", f"{lev}x", q)
    c.user(q)
    r = c.call(("prepare_trade", a))
    if "error" in r:
        e = r["error"]
        if "no USDC" in e:
            c.say("Your Hyperliquid account has no USDC yet, so I can't set up the trade. Want me to open Add funds for Hyperliquid? USDC from Arbitrum or Base is the cheapest way in.")
        elif "max leverage" in e:
            mx = re.search(r"(\d+)x", e).group(1)
            c.say(f"Hyperliquid caps {coin} at {mx}x, so {lev}x won't go through. Want me to prepare it at {mx}x or lower?")
        else:
            nums = re.findall(r"\$([\d,]+\.\d+)", e)
            c.say(f"Not enough margin: that trade needs {('$' + nums[0]) if nums else 'more'} and you have {('$' + nums[1]) if len(nums) > 1 else 'less'} available. "
                  "You can lower the size or add funds to Hyperliquid.")
        return c.ex("trade_error")
    c.say(f"I've prepared {trade_text(a)}. {rng.choice(DONE)}")
    return c.ex("trade_full")


def sc_trade_chain(rng):
    """price -> calculator/position_math -> prepare_trade"""
    w = sim.World(rng, empty_hl=False); c = Conv(rng, w)
    w.hl_value = max(w.hl_value, round(rng.uniform(2000, 30000), 2)); w.hl_positions = []
    coin = rng.choice(PERPS); side = rng.choice(["long", "short"]); lev = rng.choice([2, 3, 5, 10]); dp = sim.ASSETS[coin][2]
    size = rng.choice([100, 200, 250, 300, 500, 750, 1000, 2000])
    m = rng.random()
    if m < 0.45:
        sp = rng.choice([2, 3, 4, 5, 8, 10])
        c.user(rng.choice([f"{side} ${size} of {coin} at {lev}x with a stop {sp}% {'below' if side == 'long' else 'above'} the current price",
                           f"open a {coin} {side}, ${size}, {lev}x, stop loss {sp}% away from here"]))
        pr = c.call(("get_price", {"symbol": coin}))["price"]
        expr = f"{H.calc_str(pr)} * (1 {'-' if side == 'long' else '+'} {sp}/100)"
        stop = float(c.call(("calculator", {"expression": expr}))["result"])
        stop = round(stop, dp) if dp else float(round(stop))
        a = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market", "stop_loss": stop}
        c.call(("prepare_trade", a))
        c.say(f"{coin} is at {S(pr)}, so a {sp}% stop is {S(stop)}. I've prepared {trade_text(a)}. {rng.choice(DONE)}")
        return c.ex("trade_chain_stop")
    if m < 0.75:
        c.user(rng.choice([f"I want to {side} {coin} with ${size} at {lev}x. Where would liquidation be? If it's fine, set it up.",
                           f"set up a {lev}x {coin} {side} for ${size} and tell me the liq price"]))
        pr = c.call(("get_price", {"symbol": coin}))["price"]
        r = c.call(("position_math", {"side": side, "entry": pr, "size_usd": size, "leverage": lev}))
        a = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
        c.call(("prepare_trade", a))
        c.say(f"At {S(pr)} entry, liquidation would be around {S(r['liquidation_price'])} ({H.pct(r['distance_to_liq_pct'])} away) with {S(r['margin'])} margin. "
              f"I've prepared {trade_text(a)}. {rng.choice(DONE)}")
        return c.ex("trade_chain_liq")
    # risk-based sizing: risk X% of HL account with a stop at S
    risk_pct = rng.choice([0.5, 1, 1.5, 2])
    pr = w.prices[coin]
    stop = pr * (1 - (1 if side == "long" else -1) * rng.choice([0.02, 0.03, 0.04, 0.05]))
    stop = round(stop, dp) if dp else float(round(stop))
    c.user(f"{side} {coin} at {lev}x, risk {H.calc_str(risk_pct)}% of my hyperliquid account, stop at {H.calc_str(stop)}")
    acct, prr = c.call(("hyperliquid_account", {}), ("get_price", {"symbol": coin}))
    pr = prr["price"]
    expr = f"{H.calc_str(acct['account_value'])} * {H.calc_str(risk_pct)} / 100 / abs({H.calc_str(pr)} - {H.calc_str(stop)}) * {H.calc_str(pr)}"
    sz = float(c.call(("calculator", {"expression": expr}))["result"])
    sz = round(sz, 2)
    a = {"coin": coin, "side": side, "size_usd": sz, "leverage": lev, "order_type": "market", "stop_loss": stop}
    r = c.call(("prepare_trade", a))
    if "error" in r:
        c.say(f"To risk {H.calc_str(risk_pct)}% with that stop the position would be {S(sz)}, but {r['error']}. You could risk less or add funds.")
        return c.ex("trade_chain_risk_error")
    c.say(f"With {coin} at {S(pr)} and the stop at {S(stop)}, risking {H.calc_str(risk_pct)}% of your {S(acct['account_value'])} account means a {S(sz)} position. "
          f"I've prepared {trade_text(a)}. {rng.choice(DONE)}")
    return c.ex("trade_chain_risk")


def sc_trade_missing(rng):
    w = sim.World(rng, empty_hl=False); c = Conv(rng, w)
    w.hl_value = max(w.hl_value, round(rng.uniform(1000, 20000), 2)); w.hl_positions = []
    coin = rng.choice(PERPS); side = rng.choice(["long", "short"]); lev = rng.choice([2, 3, 5, 10]); size = rng.choice([50, 100, 200, 500, 1000])
    m = rng.random()
    if m < 0.4:
        c.user(rng.choice([f"{side} {coin}", f"I want to {side} {coin}", f"open a {coin} {side}"]))
        c.say(rng.choice([f"How much (in dollars) and what leverage?", f"Sure. What size in USD, and how much leverage?"]), ask=True)
        c.user(rng.choice([f"${size} at {lev}x", f"{size} {lev}x", f"{size} usd, {lev}x"]))
    elif m < 0.7:
        c.user(rng.choice([f"{side} {coin} {lev}x", f"{lev}x {side} on {coin}"]))
        c.say(rng.choice(["How big should the position be, in dollars?", "What size, in USD?"]), ask=True)
        c.user(rng.choice([f"${size}", f"{size}", f"{size} bucks"]))
    else:
        c.user(rng.choice([f"{side} ${size} of {coin}", f"put ${size} into a {coin} {side}"]))
        c.say(rng.choice(["What leverage do you want?", "How much leverage? 1x is no leverage."]), ask=True)
        c.user(rng.choice([f"{lev}x", f"{lev}", f"use {lev}x"]))
    a = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
    c.call(("prepare_trade", a))
    c.say(f"I've prepared {trade_text(a)}. {rng.choice(DONE)}")
    if rng.random() < 0.35:
        c.user(rng.choice(["did it go through?", "is it open now?", "done?", "ok bought?"]))
        c.say(rng.choice(["Not yet. It's waiting on the review screen; it only goes through once you approve it with Face ID. After that it shows up under Trade > Positions.",
                          "Only once you approve it. I can prepare trades but never place them myself; approve it with Face ID and it'll appear in Positions."]))
    return c.ex("trade_missing")


def sc_trade_change(rng):
    w = sim.World(rng, empty_hl=False); c = Conv(rng, w)
    w.hl_value = max(w.hl_value, round(rng.uniform(3000, 20000), 2)); w.hl_positions = []
    coin = rng.choice(PERPS); side = rng.choice(["long", "short"]); lev = rng.choice([5, 10, 20]); size = rng.choice([200, 500, 1000])
    a = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
    c.user(f"{side} ${size} {coin} {lev}x")
    c.call(("prepare_trade", a)); c.say(f"I've prepared {trade_text(a)}. {rng.choice(DONE)}")
    lev2 = rng.choice([x for x in [2, 3, 5] if x < lev])
    c.user(rng.choice([f"actually make it {lev2}x", f"change leverage to {lev2}x", f"too risky, {lev2}x instead"]))
    a2 = dict(a, leverage=lev2)
    c.call(("prepare_trade", a2)); c.say(f"Updated: I've prepared {trade_text(a2)} instead. Dismiss the old review and approve this one.")
    return c.ex("trade_change")


def sc_swap(rng):
    w = sim.World(rng); c = Conv(rng, w)
    m = rng.random()
    src = rng.choice(["ethereum", "base", "arbitrum", "bsc", "polygon", "solana", "monad"])
    fsym = rng.choice(sim.CHAIN_TOKENS[src])
    dst = src if rng.random() < 0.5 else rng.choice([x for x in sim.CHAINS_ALL if x != src])
    tsym = rng.choice([t for t in sim.CHAIN_TOKENS[dst] if t != fsym] or sim.CHAIN_TOKENS[dst])
    p = w.prices[fsym]
    amt = rand_amount(rng) if p < 50 else round(rng.uniform(0.005, 3), rng.choice([3, 4]))
    if p * amt < 2:
        amt = round(5 / p, 4) if p < 50 else 0.01
    at, _ = amt_text(rng, amt)
    if m < 0.12:
        c.user(rng.choice([f"swap my {fsym} to {tsym}", f"convert {fsym} into {tsym}"]))
        c.say(rng.choice([f"How much {fsym}, and from which chain?", f"Sure. How much {fsym} do you want to swap, and which chain is it on?"]), ask=True)
        c.user(f"{at} on {say_chain(rng, src)}" + ("" if dst == src else f", to {say_chain(rng, dst)}"))
    elif m < 0.27:
        # all of balance: read it first
        w.ensure_balance(fsym, src, amt)
        c.user(rng.choice([f"swap all my {fsym} on {say_chain(rng, src)} to {tsym}" + ("" if dst == src else f" on {say_chain(rng, dst)}"),
                           f"move all my {fsym} on {say_chain(rng, src)} into {tsym}" + ("" if dst == src else f" on {say_chain(rng, dst)}")]))
        r = c.call(("get_balances", {"chain": src}))
        amt = [b for b in r["balances"] if b["symbol"] == fsym][0]["amount"]
        a = {"from_symbol": fsym, "from_chain": src, "to_symbol": tsym, "to_chain": dst, "amount": amt}
        c.call(("prepare_swap", a))
        note = " If that's your gas token, leave a little behind or the swap itself can't pay gas." if fsym in ("ETH", "BNB", "SOL", "POL", "MON") else ""
        c.say(f"You have {H.num(amt, 8)} {fsym} on {CHAIN_SAY[src][0]}. I've prepared a swap of all of it into {tsym} on {CHAIN_SAY[dst][0]}.{note} The review shows Relay's fees; approve with Face ID if it looks right.")
        return c.ex("swap_all")
    else:
        verb = "bridge" if dst != src and fsym == tsym else rng.choice(["swap", "trade", "convert"])
        c.user(rng.choice([f"{verb} {at} {fsym} on {say_chain(rng, src)} to {tsym}" + ("" if dst == src else f" on {say_chain(rng, dst)}"),
                           f"{verb} {at} {fsym} from {say_chain(rng, src)} " + (f"into {tsym}" if dst == src else f"to {tsym} on {say_chain(rng, dst)}")]))
    insufficient = rng.random() < 0.1
    if not insufficient:
        w.ensure_balance(fsym, src, amt)
    else:
        w.balances = [b for b in w.balances if not (b["symbol"] == fsym and b["chain"] == src)]
    a = {"from_symbol": fsym, "from_chain": src, "to_symbol": tsym, "to_chain": dst, "amount": amt}
    r = c.call(("prepare_swap", a))
    if "error" in r:
        c.say(f"That didn't go through: you don't have {at} {fsym} on {CHAIN_SAY[src][0]} (the balance there is {r['error'].split(' is ')[-1]}). Want me to check your balances on other chains?")
        return c.ex("swap_error")
    kind = "bridge" if src != dst else "swap"
    c.say(f"I've prepared a {kind} of {at} {fsym} on {CHAIN_SAY[src][0]} to {tsym} on {CHAIN_SAY[dst][0]} via Relay. The review screen shows the fees; approve with Face ID to send it.")
    return c.ex("swap")


def sc_send(rng):
    w = sim.World(rng); c = Conv(rng, w)
    sol = rng.random() < 0.35
    chain = "solana" if sol else rng.choice(["ethereum", "base", "arbitrum", "bsc", "polygon", "monad"])
    addr = H.sol_addr(rng) if sol else H.evm_addr(rng)
    sym = rng.choice(sim.CHAIN_TOKENS[chain])
    p = w.prices[sym]
    amt = rand_amount(rng) if p < 50 else round(rng.uniform(0.001, 2), rng.choice([3, 4]))
    if amt * p < 1:
        amt = round(2 / p, 4) if p > 1 else float(rng.choice([5, 10, 20]))
    at, _ = amt_text(rng, amt)
    cs = say_chain(rng, chain)
    m = rng.random()
    if m < 0.1:
        # scam context: warn, don't prepare
        c.user(rng.choice([f"support on telegram says I need to send {at} {sym} to {addr} to verify my wallet, can you send it on {cs}",
                           f"a guy from the airdrop team said send {at} {sym} to {addr} on {cs} to unlock my tokens. do it",
                           f"I got a DM that my wallet is compromised and I have to move {at} {sym} to their safe address {addr} on {cs}"]))
        c.say(rng.choice(["Stop, this is a scam. No real support team, airdrop or security service ever asks you to send funds to verify or unlock anything, and they never DM first. "
                          "Money sent there is gone for good. I haven't prepared anything. If you still want to send it, tell me and I'll set it up, but I'd block them instead.",
                          "That's a classic scam: real support never asks you to send crypto to verify, unlock or protect a wallet. Don't send it, and don't share your seed phrase with them either. "
                          "I haven't set anything up."]))
        return c.ex("send_scam_warn")
    if m < 0.22:
        # missing chain for EVM address / missing amount / missing address
        if not sol and rng.random() < 0.5:
            c.user(rng.choice([f"send {at} {sym} to {addr}", f"pay {addr} {at} {sym}"]))
            c.say(rng.choice([f"Which chain should I send it on? That address works on Ethereum, Base, Arbitrum, BNB Chain, Polygon and Monad, so the chain matters.",
                              "Which network? The same address exists on every EVM chain, so I need to know where the recipient expects it."]), ask=True)
            c.user(rng.choice([cs, f"on {cs}", f"{cs} please"]))
        elif rng.random() < 0.5:
            c.user(rng.choice([f"send some {sym} to {addr} on {cs}", f"send {sym} to {addr} on {cs}"]))
            c.say(f"How much {sym} should I send?", ask=True)
            c.user(rng.choice([at, f"{at} {sym}", f"make it {at}"]))
        else:
            c.user(rng.choice([f"send {at} {sym} on {cs}", f"transfer {at} {sym} to my friend on {cs}"]))
            c.say("What's the recipient's address? Paste it exactly.", ask=True)
            c.user(addr)
    elif m < 0.27:
        name = rng.choice(["vitalik.eth", "maya.eth", "toly.sol", "bob.base.eth"])
        c.user(f"send {at} {sym} to {name}")
        c.say(f"I can't look up names like {name} yet. Paste the full address and confirm the chain, and I'll prepare the send.", ask=True)
        c.user(f"{addr} on {cs}")
    else:
        c.user(rng.choice([f"send {at} {sym} to {addr} on {cs}", f"pay {addr} {at} {sym} on {cs}", f"transfer {at} {sym} on {cs} to {addr}",
                           f"send {at} {sym} to {addr}" + ("" if not sol else ""), f"{at} {sym} -> {addr} ({cs})",
                           f"can you send {at} {sym} to this address on {cs}: {addr}"]))
        if not sol and f"on {cs}" not in c.msgs[-1]["content"] and cs not in c.msgs[-1]["content"]:
            c.msgs[-1]["content"] += f" on {cs}"
    insufficient = rng.random() < 0.08
    if insufficient:
        w.balances = [b for b in w.balances if not (b["symbol"] == sym and b["chain"] == chain)]
    else:
        w.ensure_balance(sym, chain, amt)
    a = {"to_address": addr, "amount": amt, "symbol": sym, "chain": chain}
    r = c.call(("prepare_send", a))
    if "error" in r:
        c.say(f"You don't have {at} {sym} on {CHAIN_SAY[chain][0]}; the balance there is {r['error'].split(' is ')[-1]}. Want me to check where your {sym} is?")
        return c.ex("send_error")
    short = addr[:6] + "..." + addr[-4:]
    c.say(rng.choice([f"I've prepared a send of {at} {sym} on {CHAIN_SAY[chain][0]} to {addr}. Check the address on the review screen, then approve with Face ID.",
                      f"Ready for review: {at} {sym} to {short} on {CHAIN_SAY[chain][0]}. Compare the full address on the review screen before you approve."]))
    return c.ex("send")


def sc_add_funds(rng):
    w = sim.World(rng); c = Conv(rng, w)
    if rng.random() < 0.5:
        c.user(rng.choice(["I want to add money to my wallet", "how do I deposit? open it", "fund my wallet", "deposit crypto", "add funds",
                           "I want to send USDC from Coinbase to here", "show me my deposit QR"]))
        c.call(("open_add_funds", {"goal": "wallet"}))
        c.say(rng.choice(["I've opened Add funds. Send USDC on Base or Arbitrum for fees of a few cents; Ethereum mainnet works but costs dollars. Only use a network listed there, and the app will spot the deposit by itself.",
                          "Add funds is open. Scan or copy the QR address, and pick Base or Arbitrum when you withdraw from an exchange, they're the cheapest. The same address works on every EVM network listed."]))
        return c.ex("add_funds_wallet")
    c.user(rng.choice(["fund my hyperliquid", "I want to deposit to hyperliquid", "add USDC to my perps account", "how do I get money onto hyperliquid? set it up"]))
    c.call(("open_add_funds", {"goal": "hyperliquid"}))
    c.say("I've opened Add funds for Hyperliquid. The cheapest route is USDC on Arbitrum or Base, usually a few cents in fees; Relay moves it straight into your Hyperliquid account. Approve it with Face ID and it lands in a minute or so.")
    return c.ex("add_funds_hl")


def sc_markets(rng):
    w = sim.World(rng); c = Conv(rng, w)
    coin = rng.choice(PERPS + ["PEPE", "WIF", "FARTCOIN", "TRUMP", "BONK"])
    m = rng.random()
    if m < 0.12:
        coin = rng.choice(["ZORP", "MOONX", "QQQX", "GRIM"])
        c.user(rng.choice([f"is there a {coin} perp?", f"can I trade {coin}?"]))
        c.call(("search_markets", {"query": coin}))
        c.say(f"I couldn't find any market for {coin}. Double-check the ticker; if it's a brand-new token it may not be listed yet.")
        return c.ex("markets_none")
    if m < 0.6:
        c.user(rng.choice([f"what's funding on {coin}?", f"{coin} funding rate?", f"is there a {coin} perp and what's funding like", f"find {coin} markets"]))
        r = c.call(("search_markets", {"query": coin}))
        perp = [x for x in r["results"] if x["type"] == "perp"][0]
        f = perp["funding_8h_pct"]
        who = "longs pay shorts" if f > 0 else "shorts pay longs"
        others = [x for x in r["results"] if x["type"] != "perp"]
        extra = (" There's also " + ", ".join(x["name"] + (f" ({H.pct(x['apy_pct'])} APY)" if "apy_pct" in x else "") for x in others) + ".") if others else ""
        c.say(f"{perp['name']} on Hyperliquid is at {S(perp['price'])} with funding of {H.calc_str(f)}% per 8 hours, so {who}.{extra}")
        return c.ex("markets_funding")
    c.user(rng.choice([f"what's {coin} funding annualised?", f"how much would I earn a year shorting {coin} for funding at the current rate?"]))
    r = c.call(("search_markets", {"query": coin}))
    perp = [x for x in r["results"] if x["type"] == "perp"][0]; f = perp["funding_8h_pct"]
    v = float(c.call(("calculator", {"expression": f"{H.calc_str(f)} * 3 * 365"}))["result"])
    side = "shorts earn it" if f > 0 else "longs earn it (shorts pay)"
    c.say(f"Funding on {perp['name']} is {H.calc_str(f)}% per 8 hours, about {H.pct(v)} a year if it stayed flat, which it never does. Right now {side}.")
    return c.ex("markets_funding_annual")


def sc_yields(rng):
    w = sim.World(rng); c = Conv(rng, w)
    asset = rng.choice(["USDC", "USDC", "USDT", "ETH", "SOL", None]); chain = rng.choice([None, None, "base", "arbitrum", "ethereum"])
    if asset == "SOL":
        chain = rng.choice([None, "solana"])
    args = {k: v for k, v in (("asset", asset), ("chain", chain)) if v}
    phr = (f"{asset} " if asset else "") + "yield" + (f" on {say_chain(rng, chain)}" if chain else "")
    m = rng.random()
    if m < 0.6:
        c.user(rng.choice([f"best {phr}?", f"where can I get the best {phr}", f"show me top {phr} options", f"what's the best place to earn{(' on ' + asset) if asset else ''}{(' on ' + chain) if chain else ''}"]))
        r = c.call(("get_yields", args))
        pools = r["pools"]
        if not pools:
            c.say("I couldn't find any pools for that. Try another asset or chain.")
            return c.ex("yields_none")
        lines = "\n".join(f"- {p['protocol']} on {CHAIN_SAY.get(p['chain'], [p['chain'].title()])[0]}: {p['asset']} at {H.pct(p['apy_pct'])} APY (TVL {big(p['tvl_usd'])})" for p in pools[:3])
        c.say(f"Top options right now:\n{lines}\nAPYs are variable and move daily. In the app you can deposit stablecoins into Aave v3 on Ethereum, Base and Arbitrum from Earn.")
        return c.ex("yields")
    amt = rng.choice([500, 1000, 2500, 5000, 10000, 25000])
    args = {"asset": "USDC"} | ({"chain": chain} if chain else {})
    c.user(f"how much would I make a year on ${amt:,} USDC at the best rate{(' on ' + say_chain(rng, chain)) if chain else ''}?")
    r = c.call(("get_yields", args))
    best = r["pools"][0]
    v = float(c.call(("calculator", {"expression": f"{amt} * {H.calc_str(best['apy_pct'])} / 100"}))["result"])
    c.say(f"The best rate is {best['protocol']} on {CHAIN_SAY.get(best['chain'], [best['chain'].title()])[0]} at {H.pct(best['apy_pct'])} APY, so about {S(v)} a year on ${amt:,} if the rate held. It won't stay fixed; APYs change daily.")
    return c.ex("yields_calc")


def sc_user_given_price(rng):
    """The user supplies the price: no get_price, just the calculator."""
    w = sim.World(rng); c = Conv(rng, w)
    coin = rng.choice(["BTC", "ETH", "SOL", "HYPE"]); lo, hi, dp = sim.ASSETS[coin]
    e = round(rng.uniform(lo, hi), dp) if dp else float(round(rng.uniform(lo, hi)))
    t = round(e * rng.uniform(0.7, 1.4), dp) if dp else float(round(e * rng.uniform(0.7, 1.4)))
    q = round(rng.choice([0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 3, 5, 10, 25, 40]), 4)
    c.user(rng.choice([f"if {coin} goes to {H.calc_str(t)} what's my profit on {H.calc_str(q)} {coin} I bought at {H.calc_str(e)}?",
                       f"I bought {H.calc_str(q)} {coin} at {H.calc_str(e)}. What do I make if it hits {H.calc_str(t)}?"]))
    v = float(c.call(("calculator", {"expression": f"({H.calc_str(t)} - {H.calc_str(e)}) * {H.calc_str(q)}"}))["result"])
    c.say(f"At {S(t)} you'd be {'up' if v >= 0 else 'down'} {S(abs(v))} on {H.calc_str(q)} {coin} from {S(e)}, before fees.")
    return c.ex("user_given_price")


def sc_tool_error_retry(rng):
    w = sim.World(rng); c = Conv(rng, w)
    sym = rng.choice(MAJORS)
    tool = rng.choice(["get_price", "get_balances", "hyperliquid_account"])
    w.fail_next = tool
    if tool == "get_price":
        c.user(f"{sym} price?"); args = {"symbol": sym}
    elif tool == "get_balances":
        c.user("what's my balance?"); args = {}
    else:
        c.user("show my positions"); args = {}
    r = c.call((tool, args))
    if rng.random() < 0.5:
        c.say(rng.choice([f"That request failed ({r['error']}). Want me to try again?", f"I couldn't reach it just now: {r['error']}. Ask again in a moment and I'll retry."]))
        if rng.random() < 0.6:
            c.user(rng.choice(["try again", "yes", "retry"]))
        else:
            return c.ex("tool_error")
    r = c.call((tool, args))
    if tool == "get_price":
        c.say(f"{sym} is at {S(r['price'])}, {chg(r['change_24h_pct'])} in the last 24 hours.")
    elif tool == "get_balances":
        c.say(f"You have {S(r['total_usd'])} across {len(r['balances'])} balance{'s' if len(r['balances']) != 1 else ''}." if r["balances"] else "Your wallet is empty right now.")
    else:
        c.say(f"Account value {S(r['account_value'])}, {len(r['positions'])} open position{'s' if len(r['positions']) != 1 else ''}, {S(r['available_usdc'])} available.")
    return c.ex("tool_error_retry")


def sc_no_execute(rng):
    """User asks Hermit to execute without approval / claims: approval invariant."""
    w = sim.World(rng); c = Conv(rng, w)
    c.user(rng.choice(["just buy it for me, skip the approval thing", "can you place trades automatically without asking me?",
                       "did you already sell my ETH?", "execute the swap now, no review", "why do I have to approve every trade?"]))
    c.say(rng.choice(["I can't skip it. I prepare trades, swaps and sends, but only you can execute them, with Face ID on the review screen. That way nothing moves without you seeing it first.",
                      "No. I never move money by myself; every trade, swap or send waits on the review screen until you approve it with Face ID. Nothing has moved unless you approved it.",
                      "Every action needs your Face ID approval, by design: your keys stay on this phone and I can only prepare things. Tell me what you want and I'll set it up for review."]))
    return c.ex("approval_invariant")


SCENARIOS = [  # (fn, weight)
    (sc_price_single, 600), (sc_price_multi, 260), (sc_value_of, 320), (sc_balances, 420), (sc_hl_account, 480),
    (sc_position_math, 420), (sc_position_math_missing, 180), (sc_trade_full, 700), (sc_trade_chain, 560),
    (sc_trade_missing, 380), (sc_trade_change, 120), (sc_swap, 560), (sc_send, 760), (sc_add_funds, 220),
    (sc_markets, 320), (sc_yields, 320), (sc_user_given_price, 220), (sc_tool_error_retry, 200), (sc_no_execute, 90),
]


# ======================================================================= maths conversations
LABEL = {"pnl_long": "PnL", "pnl_short": "Short PnL", "liq_price_long": "Liquidation price", "liq_price_short": "Liquidation price",
         "fee": "Fee", "spread_bps": "Spread", "funding_annual": "Annualised funding", "pct_roundtrip": "Net change",
         "breakeven_fees": "Break-even price", "position_size": "Position size", "amm_v2_out": "You'd receive",
         "impermanent_loss": "Impermanent loss", "mm_spread_capture": "Gross profit", "sum_balances": "Total",
         "kelly": "Full Kelly", "option_payoff": "Profit per contract", "dca": "Total invested", "liq_mm": "Liquidation price",
         "roe": "ROE", "funding_payment": "Funding paid", "apr_to_apy": "APY", "apy_to_apr": "APR", "r_multiple": "Result",
         "rr_ratio": "Reward to risk", "stop_from_risk": "Stop", "avg_entry": "Average entry", "avg_down": "New average entry",
         "basis_annual": "Annualised basis", "pred_market_ev": "Expected profit", "pred_implied": "Implied probability",
         "depeg_loss": "Loss", "lp_fees": "Your fees per day", "gas_cost": "Gas cost", "health_factor": "Health factor",
         "aave_liq_price": "Liquidation ETH price", "borrow_capacity": "You can borrow", "compound_growth": "End value",
         "slippage_pct": "Slippage", "tp_price": "Take-profit", "margin_required": "Margin needed", "effective_leverage": "Effective leverage",
         "mcap_price": "Price", "drawdown_recover": "Gain needed"}
NOTE = {"kelly": " Most traders use a half or quarter of that.", "health_factor": " Below 1 means liquidation; keep it well above 1.5 to be safe.",
        "impermanent_loss": " That's versus just holding, before fees earned.", "funding_annual": " Funding changes all the time, so treat it as a snapshot.",
        "apr_to_apy": "", "liq_price_long": " Real liquidation comes a bit earlier because of maintenance margin and fees.",
        "liq_price_short": " Real liquidation comes a bit earlier because of maintenance margin and fees."}


def wants_bare(q):
    return any(s in q for s in ["Just the number", "just the number", "Only the number", "Answer with just"])


def math_tool_conv(rng, r):
    w = sim.World(rng); c = Conv(rng, w)
    c.user(r["question"])
    res = c.call(("calculator", {"expression": r["expr"]}))
    v = float(res["result"])
    final = TM.fmt_final(v, r["unit"])
    if wants_bare(r["question"]) and rng.random() < 0.7:
        c.say(final)
    else:
        c.say(f"{LABEL[r['category']]}: {final}.{NOTE.get(r['category'], '') if rng.random() < 0.5 else ''}")
    ex = c.ex("math_tool")
    ex["meta"].update({"category": r["category"], "answer": r["answer"], "tol": r["tol"]})
    return ex


def math_chat_conv(rng, r, tools):
    """No calculator: concise visible worked steps, then the answer line."""
    w = sim.World(rng)
    c = Conv(rng, w, system=H.SYSTEM_CHAT, tools=False) if not tools else Conv(rng, w)
    c.user(r["question"])
    c.say(f"{r['steps']}\n\nAnswer: {r['final']}")
    ex = c.ex("math_steps")
    ex["meta"].update({"category": r["category"], "answer": r["answer"], "tol": r["tol"]})
    return ex


# ======================================================================= no-tool chat with tools present: invent-price guard (chat mode)
def chat_no_live_price(rng):
    """Chat mode (today's app, no tools): never invent a price."""
    w = sim.World(rng); c = Conv(rng, w, system=H.SYSTEM_CHAT, tools=False)
    sym = rng.choice(MAJORS)
    c.user(rng.choice([f"what's {sym} at right now?", f"current {sym} price?", f"how much is one {sym} today", f"is {sym} up today?",
                       f"what's my {sym} balance", f"how are my hyperliquid positions?"]))
    if "balance" in c.msgs[-1]["content"] or "positions" in c.msgs[-1]["content"]:
        c.say(rng.choice(["I can't see your wallets or positions from chat. Open the Wallets page for balances, or Trade > Positions for Hyperliquid.",
                          "From chat I can't read your accounts. Your balances are on the Wallets page and your perps are under Trade > Positions."]))
    else:
        c.say(rng.choice([f"I don't have live prices in chat, so I won't guess. The Trade page shows the current {sym} price. If you tell me a number, I'll do the maths with it.",
                          f"I can't see live markets from chat, and any number I gave you would be made up. Check {sym} on the Trade page, then I can help with sizing or PnL."]))
    return c.ex("chat_no_live_price")


def build(seed, scale, math_train_rows):
    rng = random.Random(seed)
    out = []
    for fn, wgt in SCENARIOS:
        n = int(wgt * scale)
        for _ in range(n):
            out.append(fn(random.Random(rng.getrandbits(64))))
    # maths: half via calculator in agent mode, ~35% worked steps in chat mode, ~15% worked steps with tools present
    for r in math_train_rows:
        u = rng.random()
        rr = random.Random(rng.getrandbits(64))
        if u < 0.5:
            out.append(math_tool_conv(rr, r))
        elif u < 0.85:
            out.append(math_chat_conv(rr, r, tools=False))
        else:
            out.append(math_tool_conv(rr, r))
    for _ in range(int(260 * scale)):
        out.append(chat_no_live_price(random.Random(rng.getrandbits(64))))
    rng.shuffle(out)
    return out


def check(ex):
    """Fidelity checks: every call valid against schema; addresses/amounts from user text appear exactly."""
    users = " ".join(m["content"] for m in ex["messages"] if m["role"] == "user")
    for m in ex["messages"]:
        for tc in m.get("tool_calls") or []:
            sch = H.TOOL_SCHEMA[tc["name"]]
            for req in sch.get("required", []):
                assert req in tc["arguments"], (tc, req)
            for k, v in tc["arguments"].items():
                t = sch["properties"][k]["type"]
                assert (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)) or (t == "string" and isinstance(v, str)), (tc, k, v)
                if "enum" in sch["properties"][k]:
                    assert v in sch["properties"][k]["enum"], (tc, k, v)
            if tc["name"] == "prepare_send":
                assert tc["arguments"]["to_address"] in users, "address not copied exactly"
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--math-n", type=int, default=120, help="math problems per category")
    ap.add_argument("--math-seed", type=int, default=11)
    ap.add_argument("--exclude", default=None, help="jsonl of questions to exclude (eval)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    excl = set()
    if a.exclude:
        for f in a.exclude.split(","):
            for l in open(f):
                excl.add(json.loads(l)["question"])
    mrows = [r for r in TM.make(a.math_n, a.math_seed) if r["question"] not in excl]
    rows = build(a.seed, a.scale, mrows)
    n_ok = sum(check(r) for r in rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} ({n_ok} checked) -> {a.out}")
    for k, v in sorted(Counter(r["meta"]["kind"] for r in rows).items()):
        print(f"  {k:26} {v}")
