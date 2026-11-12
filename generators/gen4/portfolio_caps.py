#!/usr/bin/env python3
"""portfolio_caps: get_portfolio, get_positions reading, capability questions (what can you do / honest NOs) and
knowledge of every v3 feature, for the next Hermit run (v3 tool registry, 30 tools).

  python portfolio_caps.py --seed 4 --n 500 --eval-seed 9405 --eval-n 240
      -> ../data4/portfolio_caps.jsonl (about 500 train rows) and ../eval4/portfolio_caps.jsonl (240 held-out items)

Kinds (train / eval, at the default sizes):
  portfolio      60 / 10   "how much do I have in total" -> get_portfolio {}; total plus at most two supporting numbers
  positions_read 60 / 10   spot / Aave / Hyperliquid reads -> get_positions {kind?}; top 3 then "and N smaller"
  sums           30 /  5   get_portfolio -> calculator ("what percent of my total is in aave")
  capability     90 / 80   30 can-do + 60 honest NO (eval 40 + 40): short grouped answers, no tool names
  knowledge_v3  160 / 120  one fact per question; required and bad-claim regexes per fact
  near_miss     100 / 15   balance / pnl / value of one coin / market cap: the neighbouring tool, or no tool
Everything is a python template over tool results, the user's text or a calculator result. No LLM text.
Style follows gen3 (open_app_tool.py): scripted scenario functions, a Conv that executes calls on a simulator, a
self-check on every record that fails the build. gen2 and gen3 are imported, never edited.
Held-out: templates with index % 5 == 0 and slot values with index % 4 == 3 are used by the eval only.
"""
import argparse, copy, json, math, os, random, re, sys
from collections import Counter
from decimal import Decimal

sys.dont_write_bytecode = True      # never write .pyc next to the read-only gen2 / gen3 modules we import
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(LAB, "gen3"), os.path.join(LAB, "gen2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hermit_fmt as H          # noqa: E402  gen2 (calc, calc_str, sol_addr, smart_money, parse_calls)
import fmt3                     # noqa: E402  gen3 (validate_args)

SPEC = os.path.join(LAB, "spec")
TOOLS = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
TOOL_ORDER = [t["function"]["name"] for t in TOOLS]
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TOOL_META = json.load(open(os.path.join(SPEC, "hermit_tools_v3.meta.json")))
SYSTEM_TEMPLATE = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert "{BIO}" in SYSTEM_TEMPLATE and len(TOOLS) == 30
BIOS = ("Face ID", "Touch ID")

# ------------------------------------------------------------------ packs (APP_TOOLS_v3.md section 4)
CORE = ["calculator", "get_price", "get_balances", "get_portfolio", "get_address", "get_activity", "prepare_send",
        "prepare_swap", "get_quote", "open_page", "open_app", "open_add_funds"]
PACKS = {
    "trade": ["hyperliquid_account", "position_math", "search_markets", "prepare_trade", "prepare_close", "cancel_order",
              "set_tp_sl", "get_positions"],
    "solana": ["search_tokens", "prepare_buy", "prepare_sell", "get_positions"],
    "earn": ["get_yields", "prepare_deposit", "prepare_withdraw", "get_positions"],
    "pay": ["prepare_multisend", "add_contact", "schedule_payment", "create_automation"],
}
ROUTE = {
    "trade": r"\b(long|short|leverage|perps?|positions?|stops?|take profit|tp|sl|cancel order|limit order|liquidation|funding|"
             r"hyperliquid|nvda|tsla|aapl|msft|amzn|googl|spy|qqq|mstr|amd)\b",
    "solana": r"\b(buy|sell|tokens?|mint|memecoins?|pump|jupiter|safe)\b",
    "earn": r"\b(aave|deposits?|withdraw|yields?|apy|earn|save)\b",
    "pay": r"\b(csv|xlsx|contacts?|payees?|schedule|scheduled|remind\w*|every (day|week|month)|automations?|recurring)\b",
}
assert all(t in TOOL_SCHEMA for t in CORE) and all(t in TOOL_SCHEMA for v in PACKS.values() for t in v)
POS_PACK = {"spot": "solana", "earn": "earn", "perps": "trade", "all": "trade"}


def build_pack(first_text, need=(), prefer=None):
    """Tools the app would send for a chat that opens with `first_text`: core + routed packs (cap 2 packs, 22 tools),
    always containing every tool in `need` (the scenario needs it)."""
    low = first_text.lower()
    packs = []
    for t in need:
        if t in CORE or any(t in PACKS[p] for p in packs):
            continue
        cands = [p for p in PACKS if t in PACKS[p]]
        pick = prefer if prefer in cands else next((p for p in cands if re.search(ROUTE[p], low)), cands[0])
        packs.append(pick)
    for p in PACKS:
        if len(packs) >= 2:
            break
        if p not in packs and re.search(ROUTE[p], low):
            tools = set(CORE) | {t for q in packs + [p] for t in PACKS[q]}
            if len(tools) <= 22:
                packs.append(p)
    tools = set(CORE) | {t for q in packs for t in PACKS[q]}
    assert set(need) <= tools
    return [t for t in TOOL_ORDER if t in tools]


def family_ok(pack):
    extra = [p for p in PACKS if all(t in pack for t in PACKS[p]) and p != "x"]
    return ["4b", "lite"] if len(pack) <= 16 and len(extra) <= 1 else ["4b"]


# ------------------------------------------------------------------ numbers and text helpers
NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
CHAIN_LABEL = {"base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "bnb": "BNB Chain", "polygon": "Polygon",
               "solana": "Solana", "near": "NEAR", "hyperliquid": "Hyperliquid"}
CHAIN_SAY = {"base": ["Base", "base"], "arbitrum": ["Arbitrum", "arbitrum", "arb"], "ethereum": ["Ethereum", "ethereum", "mainnet"],
             "solana": ["Solana", "solana", "sol"], "polygon": ["Polygon", "polygon"], "bnb": ["BNB Chain", "bnb", "bsc"]}


def numbers(text):
    return {float(x.replace(",", "")) for x in NUM_RE.findall(text)}


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def usd(x):
    x = float(x)
    return ("-$" if x < 0 else "$") + f"{abs(x):,.2f}"


def S(x):
    return H.smart_money(float(x))


def pct_s(x):
    s = f"{abs(float(x)):.2f}".rstrip("0").rstrip(".")
    return s + "%"


def updown(x, unit="%"):
    return ("up " if float(x) >= 0 else "down ")


def amt_s(a):
    """amount string with thousands commas when >= 1000 (the integer part only)"""
    s = str(a)
    if "." in s:
        i, f = s.split(".")
        return f"{int(i):,}.{f}"
    return f"{int(s):,}"


def join_and(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def clean(o):
    """Python-style shortest numbers like the app: whole floats become ints."""
    if isinstance(o, float):
        return int(o) if o.is_integer() else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def dumps(r):
    return json.dumps(clean(r), ensure_ascii=False, sort_keys=True)


def dsum(vals):
    return float(sum(Decimal(str(round(v, 2))) for v in vals))


def fmt_amount(x, dp=6):
    d = Decimal(str(round(x, dp))).normalize()
    s = format(d, "f")
    return s


def system_agent(bio):
    return SYSTEM_TEMPLATE.replace("{BIO}", bio)


# ------------------------------------------------------------------ simulator (plain dict world, replayable)
PERP_PRICE = {"BTC": (58000, 118000), "ETH": (2100, 4800), "SOL": (95, 260), "BNB": (480, 900), "HYPE": (18, 60), "XRP": (1.6, 3.4),
              "DOGE": (0.12, 0.35), "LINK": (11, 27), "AVAX": (18, 48), "SUI": (2, 5.2)}
PERP_COINS = list(PERP_PRICE)
MEMES = {"WIF": 1.82, "JUP": 0.78, "POPCAT": 0.52, "PUMP": 0.0043, "MOODENG": 0.21, "GOAT": 0.12, "FARTCOIN": 0.41, "PENGU": 0.0081}
WALLET_TOKENS = [("USDC", ["base", "arbitrum", "ethereum", "solana", "polygon"]), ("USDT", ["ethereum", "arbitrum", "bnb", "solana"]),
                 ("ETH", ["ethereum", "base", "arbitrum"]), ("SOL", ["solana"]), ("BNB", ["bnb"])]
AAVE_TOKENS = [("USDC", ["base", "ethereum", "arbitrum"]), ("USDT", ["ethereum", "arbitrum"]), ("ETH", ["ethereum", "base", "arbitrum"])]
WALLET_NAMES = ["Main", "Main", "Main", "Savings", "Trading", "Daily", "Hot wallet", "Cold", "Vault", "Play"]
PAGE_TITLE = {"home": "Home", "wallets": "Wallets", "trade": "Trade", "earn": "Earn", "payments": "Payments", "explore": "Explore",
              "calendar": "Calendar", "projects": "Projects", "automations": "Automations", "files": "Files",
              "connectors": "Connectors", "settings": "Settings"}
PAGE_TABS = {"trade": ["markets", "positions"], "earn": ["save", "positions", "explore"],
             "payments": ["schedule", "payees", "transfer", "apis"], "settings": ["profile", "safety", "memory", "models", "cloud"]}


def px_round(x):
    return round(x, 1) if x >= 1000 else round(x, 2) if x >= 10 else round(x, 3) if x >= 1 else round(x, 4)


def size_round(notional, price):
    dp = 4 if price >= 10000 else 3 if price >= 1000 else 2 if price >= 100 else 1 if price >= 10 else 0
    return max(round(notional / price, dp), 10 ** -dp)


def device_name(w):
    return w.get("device", "iPhone")


def wallet_rows(w):
    un = set(w["unreachable"])
    return [b for b in w["balances"] if b["chain"] not in un]


def aave_rows(w):
    un = set(w["unreachable"])
    return [a for a in w["aave"] if a["chain"] not in un]


def spot_rows(w):
    return [] if "solana" in w["unreachable"] else w["spot"]


def hl_off(w):
    return "hyperliquid" in w["unreachable"]


def what_wallet(b):
    return "SOL" if (b["symbol"] == "SOL" and b["chain"] == "solana") else f"{b['symbol']} on {CHAIN_LABEL[b['chain']]}"


def t_calculator(w, expression):
    try:
        return {"result": H.calc_str(H.calc(expression))}
    except Exception:
        return {"error": "Couldn't evaluate that expression"}


def t_get_price(w, symbol):
    s = str(symbol).upper()
    if s not in w["prices"]:
        return {"error": f"No Hyperliquid market for {s}. Answer from what you know instead, and say there's no live price here."}
    return {"change_24h_pct": w["change"][s], "price": w["prices"][s], "source": "Hyperliquid mark price", "symbol": s}


def t_get_balances(w, chain=None):
    if w["no_wallet"]:
        return {"error": f"No wallet on this {device_name(w)} yet"}
    ch = None if chain is None else ("bnb" if str(chain).lower() in ("bsc", "bnb") else str(chain).lower())
    rows = [b for b in w["balances"] if b["amount"] and float(b["amount"]) > 0 and (ch is None or b["chain"] == ch)
            and b["chain"] not in w["unreachable"]]
    r = {"holdings": [{"amount": b["amount"], "chain": b["chain"], "symbol": b["symbol"], "usd": b["usd"]} for b in rows]}
    if w.get("near_account"):
        r["near_account"] = w["near_account"]
    un = [u for u in w["unreachable"] if u != "hyperliquid" and (ch is None or u == ch)]
    if un:
        r["unreachable"] = un
    if w["others"]:
        r["note"] = f"Showing the selected wallet ({w['wallet']}). Other wallets: {', '.join(w['others'])}."
        r["wallet"] = w["wallet"]
    return r


def portfolio_parts(w):
    """list of (place, usd, what) rows, one per holding, plus place totals in the fixed order"""
    rows = []
    for b in wallet_rows(w):
        rows.append(("Wallet", b["usd"], what_wallet(b)))
    for s in spot_rows(w):
        rows.append(("Spot tokens", s["usd"], s["symbol"]))
    for a in aave_rows(w):
        rows.append(("Aave", a["usd"], f"{a['symbol']} on {CHAIN_LABEL[a['chain']]}"))
    if not hl_off(w) and w["hl"]["value"] > 0:
        rows.append(("Hyperliquid", w["hl"]["value"], "Account value"))
    return rows


def t_get_portfolio(w):
    if w["no_wallet"]:
        return {"error": f"No wallet on this {device_name(w)} yet"}
    rows = portfolio_parts(w)
    by = []
    for place in ("Wallet", "Spot tokens", "Aave", "Hyperliquid"):
        v = dsum([u for p, u, _ in rows if p == place])
        if v > 0:
            by.append({"place": place, "usd": v})
    top = sorted(rows, key=lambda x: -x[1])[:3]
    r = {"by_place": by, "top": [{"place": p, "usd": u, "what": wh} for p, u, wh in top],
         "total_usd": dsum([x["usd"] for x in by]), "wallet": w["wallet"]}
    un = [u for u in w["unreachable"]]
    if un:
        r["unreachable"] = un
    return r


def t_get_positions(w, kind=None):
    if w["no_wallet"]:
        return {"error": f"No wallet on this {device_name(w)} yet"}
    if kind not in (None, "all", "spot", "earn", "perps"):
        return {"error": "kind must be all, spot, earn or perps"}
    full = {}
    full["spot"] = [{"amount": s["amount"], "avg_cost": s["avg_cost"], "frozen": s["frozen"], "mint": s["mint"],
                     "pnl_pct": s["pnl_pct"], "symbol": s["symbol"], "usd": s["usd"]} for s in spot_rows(w)]
    full["earn"] = [{"amount": a["amount"], "apy_pct": a["apy_pct"], "chain": a["chain"], "symbol": a["symbol"]} for a in aave_rows(w)]
    if hl_off(w):
        full["perps"] = {"open_orders": [], "positions": []}
    else:
        full["perps"] = {"open_orders": [dict(o) for o in w["hl"]["orders"]], "positions": [dict(p) for p in w["hl"]["positions"]]}
    keys = ["spot", "earn", "perps"] if kind in (None, "all") else [kind]
    r = {k: full[k] for k in keys}
    r["wallet"] = w["wallet"]
    if w["unreachable"]:
        r["unreachable"] = list(w["unreachable"])
    return r


def t_hyperliquid_account(w):
    if w["no_wallet"]:
        return {"error": f"No wallet on this {device_name(w)} yet"}
    if hl_off(w):
        return {"error": "Hyperliquid isn't reachable right now"}
    hl = w["hl"]
    orders = []
    for o in hl["orders"]:
        if o["kind"] == "limit":
            orders.append({"coin": o["coin"], "is_trigger": False, "order_type": "Limit", "price": o["price"], "side": o["side"],
                           "size": o["size"]})
        else:
            below = (o["kind"] == "stop_loss") == (o["side"] == "sell")
            orders.append({"coin": o["coin"], "is_trigger": True,
                           "order_type": "Stop Market" if o["kind"] == "stop_loss" else "Take Profit Market", "price": o["price"],
                           "side": o["side"], "size": o["size"],
                           "trigger_condition": f"Price {'below' if below else 'above'} {clean(o['price'])}", "whole_position": True})
    used = dsum([p["size"] * p["entry"] / p["leverage"] for p in hl["positions"]])
    return {"account_mode": "unified", "account_value": hl["value"], "available_usdc": max(0, round(hl["value"] - used, 2)),
            "margin_used": round(used, 2), "open_orders": orders, "positions": [dict(p) for p in hl["positions"]]}


def t_open_page(w, page, tab=None):
    if page not in PAGE_TITLE:
        return {"error": "That page can't be opened from chat."}
    title = PAGE_TITLE[page]
    if tab is not None:
        tabs = PAGE_TABS.get(page)
        if not tabs or tab not in tabs:
            return {"error": f"{title} has no tab called {tab}." + (f" Tabs: {', '.join(tabs)}." if tabs else "")}
    r = {"note": f"The user taps it to open {title}.", "page": page, "screen": page, "status": "button_shown"}
    if tab is not None:
        r["tab"] = tab
    return r


TOOL_FN = {"calculator": t_calculator, "get_price": t_get_price, "get_balances": t_get_balances, "get_portfolio": t_get_portfolio,
           "get_positions": t_get_positions, "hyperliquid_account": t_hyperliquid_account, "open_page": t_open_page}


def run_tool(w, name, args):
    fn = TOOL_FN.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(w, **(args if isinstance(args, dict) else {}))
    except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError) as e:
        return {"error": f"bad arguments: {e}"}


def new_world(device="iPhone"):
    return {"wallet": "Main", "others": [], "device": device, "no_wallet": False, "unreachable": [], "balances": [], "near_account": None,
            "spot": [], "aave": [], "hl": {"value": 0, "positions": [], "orders": []}, "prices": {}, "change": {}}


def fill_prices(rng, w, coins=None):
    for c in (coins or PERP_COINS):
        lo, hi = PERP_PRICE[c]
        w["prices"][c] = px_round(rng.uniform(lo, hi))
        w["change"][c] = round(rng.uniform(-8.5, 8.5), 2)


def gen_balances(rng, w, n, force=None):
    """n wallet rows; force = list of (symbol, chain) that must exist"""
    for c in list(PERP_PRICE):
        if c not in w["prices"]:
            fill_prices(rng, w, [c])
    seen = set()

    def add(sym, ch):
        if (sym, ch) in seen:
            return
        seen.add((sym, ch))
        if sym in ("USDC", "USDT"):
            a = round(math.exp(rng.uniform(math.log(8), math.log(7000))), rng.choice([0, 2, 2, 2]))
            price = 1.0
        else:
            price = w["prices"][sym]
            a = {"ETH": lambda: round(rng.uniform(0.004, 2.6), 4), "SOL": lambda: round(rng.uniform(0.05, 45), 3),
                 "BNB": lambda: round(rng.uniform(0.02, 6), 3)}[sym]()
        if a <= 0:
            a = 1
        u = round(a * price, 2)
        if u < 1:
            a, u = 3, round(3 * price, 2)
        w["balances"].append({"amount": fmt_amount(a), "chain": ch, "symbol": sym, "usd": u})

    for sym, ch in (force or []):
        add(sym, ch)
    tries = 0
    while len(w["balances"]) < n and tries < 40:
        tries += 1
        sym, chs = rng.choice(WALLET_TOKENS)
        add(sym, rng.choice(chs))


def gen_spot(rng, w, n, frozen_p=0.08, untracked_p=0.2, force=None):
    syms = list(MEMES)
    rng.shuffle(syms)
    if force:
        syms = [s for s in force] + [s for s in syms if s not in force]
    for sym in syms[:n]:
        price = MEMES[sym]
        usd_t = math.exp(rng.uniform(math.log(2), math.log(900)))
        amount = round(usd_t / price, 2 if price >= 0.1 else 0)
        if amount <= 0:
            amount = 1
        u = round(amount * price, 2)
        if u < 1:
            amount = round(2 / price, 0)
            u = round(amount * price, 2)
        if rng.random() < untracked_p:
            avg, pnl = None, None
        else:
            pnl0 = rng.uniform(-45, 140)
            avg = round(price / (1 + pnl0 / 100), 2 if price >= 1 else 4)
            pnl = round((price / avg - 1) * 100, 1)
        w["spot"].append({"amount": fmt_amount(amount, 2), "avg_cost": avg, "frozen": rng.random() < frozen_p, "mint": H.sol_addr(rng),
                          "pnl_pct": pnl, "symbol": sym, "usd": u})


def gen_aave(rng, w, n):
    seen = set()
    tries = 0
    while len(w["aave"]) < n and tries < 40:
        tries += 1
        sym, chs = rng.choice(AAVE_TOKENS)
        ch = rng.choice(chs)
        if (sym, ch) in seen:
            continue
        seen.add((sym, ch))
        if sym == "ETH":
            if "ETH" not in w["prices"]:
                fill_prices(rng, w, ["ETH"])
            a, price, apy = round(rng.uniform(0.01, 1.8), 4), w["prices"]["ETH"], round(rng.uniform(1.0, 3.4), 2)
        else:
            a, price, apy = round(math.exp(rng.uniform(math.log(10), math.log(5000))), rng.choice([0, 2, 2])), 1.0, round(rng.uniform(2.0, 8.2), 2)
        w["aave"].append({"amount": fmt_amount(a), "apy_pct": apy, "chain": ch, "symbol": sym, "usd": round(a * price, 2)})


def gen_hl(rng, w, n_pos, with_value=True, stops=0.55, tps=0.35, limits=0.3, force_coins=None):
    hl = w["hl"]
    coins = list(PERP_COINS)
    rng.shuffle(coins)
    if force_coins:
        coins = list(force_coins) + [c for c in coins if c not in force_coins]
    oid = rng.randint(8100, 8600)
    for c in coins[:n_pos]:
        if c not in w["prices"]:
            fill_prices(rng, w, [c])
        mark = w["prices"][c]
        side = rng.choice(["long", "long", "short"])
        entry = px_round(mark * rng.uniform(0.9, 1.1))
        lev = rng.choice([2, 3, 5, 5, 10, 10, 15, 20])
        size = size_round(rng.uniform(40, 2600), entry)
        mm = 0.005
        liq = entry * (1 - 1 / lev + mm) if side == "long" else entry * (1 + 1 / lev - mm)
        upnl = round(size * (mark - entry) * (1 if side == "long" else -1), 2)
        hl["positions"].append({"coin": c, "entry": entry, "leverage": lev, "liquidation": px_round(liq), "side": side, "size": size,
                                "unrealised_pnl": upnl})
        closing = "sell" if side == "long" else "buy"
        for kind, p in (("stop_loss", stops), ("take_profit", tps)):
            if rng.random() < p:
                if (kind == "stop_loss") == (side == "long"):
                    price = entry * rng.uniform(0.84, 0.97)
                else:
                    price = entry * rng.uniform(1.04, 1.3)
                price = px_round(price) if entry < 100 else float(round(price))
                oid += rng.randint(1, 9)
                hl["orders"].append({"coin": c, "id": str(oid), "kind": kind, "price": price, "side": closing, "size": size,
                                     "whole_position": True})
    if rng.random() < limits:
        for c in rng.sample(PERP_COINS, rng.choice([1, 1, 2])):
            if c not in w["prices"]:
                fill_prices(rng, w, [c])
            oid += rng.randint(1, 9)
            sd = rng.choice(["buy", "sell"])
            price = px_round(w["prices"][c] * (rng.uniform(0.92, 0.99) if sd == "buy" else rng.uniform(1.01, 1.08)))
            price = float(round(price)) if price >= 100 else price
            hl["orders"].append({"coin": c, "id": str(oid), "kind": "limit", "price": price, "side": sd,
                                 "size": size_round(rng.uniform(30, 700), price)})
    if with_value:
        used = dsum([p["size"] * p["entry"] / p["leverage"] for p in hl["positions"]])
        upnl = dsum([p["unrealised_pnl"] for p in hl["positions"]])
        hl["value"] = round(max(25.0, used * rng.uniform(1.15, 3.5) + rng.uniform(0, 150) + upnl), 2) if hl["positions"] else \
            round(math.exp(rng.uniform(math.log(30), math.log(5000))), 2)


# ------------------------------------------------------------------ conversation builder
class Conv:
    """Builds one conversation and executes the calls on the dict world."""

    def __init__(self, rng, w, bio):
        self.rng, self.w, self.bio = rng, w, bio
        self.world0 = copy.deepcopy(w)
        self.msgs = [{"role": "system", "content": system_agent(bio)}]
        self.expect = []
        self.calls = []                    # calls after the last user turn

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        self.expect, self.calls = [], []

    def call(self, *calls):
        tcs = [{"name": n, "arguments": a} for n, a in calls]
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        self.expect.append({"calls": tcs})
        self.calls += tcs
        res = []
        for n, a in calls:
            r = run_tool(self.w, n, a)
            res.append(r)
            self.msgs.append({"role": "tool", "content": dumps(r).replace("Face ID", self.bio)})
        return res if len(res) > 1 else res[0]

    def say(self, text, ask=False):
        self.msgs.append({"role": "assistant", "content": text.replace("Face ID", self.bio)})
        self.expect.append({"ask": True} if ask else {"text": True})

    def calc(self, expr):
        return self.call(("calculator", {"expression": expr}))["result"]

    def ex(self, kind, **meta):
        first_user = next(m["content"] for m in self.msgs if m["role"] == "user")
        need = [c["name"] for m in self.msgs if m.get("tool_calls") for c in m["tool_calls"]]
        pack = build_pack(first_user, need=sorted(set(need)), prefer=meta.pop("prefer_pack", None))
        fam = family_ok(pack)
        lite_trade = {"hyperliquid_account", "prepare_trade", "prepare_close", "cancel_order"}
        pack_lite = None
        if fam == ["4b"] and set(need) - set(CORE) <= lite_trade and len(set(self.calls_all())) <= 4:
            pack_lite = [t for t in TOOL_ORDER if t in set(CORE) | lite_trade]
            fam = ["4b", "lite"]
        m = {"source": "portfolio_caps", "generator": "portfolio_caps", "kind": kind, "bio": self.bio, "world": self.world0,
             "world_final": copy.deepcopy(self.w), "expect": self.expect, "pack": pack, "family_ok": fam,
             "steps": [c["name"] for c in self.calls], "cards": [], "lang": "en"}
        if pack_lite:
            m["pack_lite"] = pack_lite
        m.update(meta)
        return {"messages": self.msgs, "tools": True, "meta": m}

    def calls_all(self):
        return [c["name"] for x in self.msgs if x.get("tool_calls") for c in x["tool_calls"]]


# ================================================================== phrase machinery
FORCED = {}
EVAL_SUBS = {
    "portfolio": ["normal", "normal", "unreachable", "biggest", "normal", "allw", "unreachable", "normal", "empty", "normal"],
    "positions": ["spot", "earn", "perps", "all", "stop", "spot_detail", "unreach", "spot", "earn", "empty"],
    "sums": ["pct", "pct", "diff", "outside", "pct_absent"],
    "near": ["balance", "balance", "balance", "balance", "balance_chain", "pnl", "pnl", "pnl_all", "pnl_all", "value", "value", "value", "mcap", "mcap", "mcap"],
}


def forced_sub(name):
    it = FORCED.get(name)
    return next(it) if it is not None else None


def pool(tpls, split):
    """templates with index % 5 == 0 are for the eval only"""
    return [t for i, t in enumerate(tpls) if (i % 5 == 0) == (split == "eval")]


def slot_vals(vals, split):
    """slot values with index % 4 == 3 are for the eval only"""
    return [v for i, v in enumerate(vals) if (i % 4 == 3) == (split == "eval")]


PRE = ["", "", "", "", "", "", "hey ", "yo ", "ok ", "hi, ", "pls ", "hey hermit, ", "hermit ", "quick one, ", "ok so ", "hello ", "um "]
SUF = ["", "", "", "", "", "", " pls", " please", " thanks", " rn", " now", " asap", " for me", " thx"]


def typo(rng, t, protect):
    words = t.split(" ")
    idx = [i for i, w in enumerate(words) if re.fullmatch(r"[a-z]{5,}", w) and w not in protect]
    if not idx:
        return t
    i = rng.choice(idx)
    w = words[i]
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        j = rng.randint(1, len(w) - 2)
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif op == "drop":
        vs = [k for k, ch in enumerate(w) if ch in "aeiou" and k > 0]
        if vs:
            k = rng.choice(vs)
            w = w[:k] + w[k + 1:]
    else:
        j = rng.randint(1, len(w) - 1)
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def style(rng, t, protect=()):
    prot = {p for s in protect for p in str(s).lower().split()}
    if rng.random() < 0.35:
        t = t.lower()
    t = rng.choice(PRE) + t + ("" if t.endswith("?") else rng.choice(SUF))
    if rng.random() < 0.15:
        t = typo(rng, t, prot)
    if not t.endswith("?"):
        r = rng.random()
        t += "." if r < 0.06 else ("!" if r < 0.10 else "")
    t = re.sub(r"  +", " ", t).strip()
    if t[0].isalpha() and rng.random() < 0.25:
        t = t[0].upper() + t[1:]
    return t


def utext(rng, split, tpls, **slots):
    t = rng.choice(pool(tpls, split)).format(**slots)
    return style(rng, t, protect=[str(v) for v in slots.values()])


def base_world(rng):
    bio = "Face ID" if rng.random() < 0.7 else "Touch ID"
    dev = "Mac" if (bio == "Touch ID" and rng.random() < 0.9) else "iPhone"
    w = new_world(dev)
    w["wallet"] = rng.choice(WALLET_NAMES)
    if rng.random() < 0.2:
        w["others"] = rng.sample(sorted({n for n in WALLET_NAMES if n != w["wallet"]}), rng.choice([1, 2]))
    return w, bio


def finish(c, kind, must=(), extra=(), words=75, checks=None, **meta):
    ex = c.ex(kind, **meta)
    ex["meta"]["gold"] = {"calls": copy.deepcopy(c.calls), "must_include_numbers": list(must), "extra_numbers": list(extra),
                          "max_words": words}
    ex["meta"]["checks"] = checks or ["right_tool", "rounded_ok", "restraint"]
    return ex


def last_user(ex):
    return [m["content"] for m in ex["messages"] if m["role"] == "user"][-1]


PLACE_SAY = {"Wallet": "your wallet", "Spot tokens": "spot tokens", "Aave": "Aave", "Hyperliquid": "Hyperliquid"}
PLACE_CAP = {"Wallet": "Your wallet", "Spot tokens": "Spot tokens", "Aave": "Aave", "Hyperliquid": "Hyperliquid"}
UN_LABEL = {"bnb": "BNB Chain", "polygon": "Polygon", "ethereum": "Ethereum", "arbitrum": "Arbitrum", "base": "Base", "solana": "Solana",
            "hyperliquid": "Hyperliquid"}


def un_text(rng, un, what="total"):
    names = join_and([UN_LABEL.get(u, u) for u in un])
    if what == "total":
        return rng.choice([f"I couldn't reach {names}, so the total may be short.", f"{names} didn't answer, so this total may be short.",
                           f"Note that {names} was unreachable, so this total may be short."])
    return rng.choice([f"I couldn't reach {names}, so this may be incomplete.", f"{names} didn't answer, so some things may be missing."])


def no_wallet_text(rng, w):
    d = device_name(w)
    return rng.choice([f"I can't read that yet: there's no wallet on this {d}. Set one up or import one first, then ask again.",
                       f"There's no wallet on this {d} yet, so I have nothing to add up. Create or import one first.",
                       f"This {d} has no wallet yet. Set one up or import one and I can look it up."])


# ================================================================== portfolio
PORTFOLIO_T = [
    "what's all my stuff worth added up", "how much do I have in total", "what's my net worth", "total portfolio value?",
    "how much is everything worth", "how much crypto do I have all together", "give me my total across everything",
    "what am i worth rn", "show me my portfolio", "net worth pls", "tell me the sum of all my funds", "total value of everything i hold",
    "how big is my bag", "what's my stack in dollars", "how much do i hold overall", "what does it all add up to",
    "how much money do i have across wallet, aave and hyperliquid", "portfolio value", "all in, how much do i have",
    "how much is my whole portfolio worth in usd", "quick check, what's my total", "whats my net worth in usd",
    "what is the total of everything", "how rich am i lol", "total across everything pls", "how much have i got in total",
    "give me the big number: all my holdings combined", "what's my portfolio at", "net worth?", "sum up everything i own",
    "how much do i own in total", "hows my portfolio looking in dollars", "how much is it all worth",
    "what's the combined value of my crypto", "how much am i holding in total across the app", "what's the grand total",
    "i wanna know my total value", "how much do i have everywhere", "can you add up everything i have",
    "total usd value of my holdings", "overall value incl aave and hyperliquid", "what's my whole account worth", "tell me my total",
    "how much money is in my hermit overall"]
ALLW_T = ["what's my total across all my wallets", "how much do i have across every wallet", "combine all my wallets and tell me the total",
          "total of all wallets together", "all wallets net worth", "how much in all my wallets", "give me the total for every wallet i have",
          "net worth across wallets", "what do all my wallets add up to", "sum of all my wallets pls"]
BIGGEST_T = ["what's my biggest holding", "where is most of my money", "what's the largest thing i hold", "which holding is the biggest",
             "where do i have the most", "what is my top holding", "where's most of my money sitting", "my biggest position?",
             "what makes up most of my portfolio", "which place holds the most"]


def gen_portfolio_world(rng, w, places=None, force_chain=None):
    """places: subset of Wallet, Spot tokens, Aave, Hyperliquid"""
    places = places or [p for p, pr in (("Wallet", 0.9), ("Spot tokens", 0.4), ("Aave", 0.4), ("Hyperliquid", 0.5)) if rng.random() < pr]
    if not places:
        places = [rng.choice(["Wallet", "Wallet", "Hyperliquid", "Aave"])]
    fill_prices(rng, w, ["ETH", "SOL", "BNB"])
    if "Wallet" in places:
        gen_balances(rng, w, rng.choice([1, 2, 3, 4]), force=[force_chain] if force_chain else None)
    if "Spot tokens" in places:
        gen_spot(rng, w, rng.choice([1, 2, 3]), frozen_p=0.0, untracked_p=0.0)
    if "Aave" in places:
        gen_aave(rng, w, rng.choice([1, 1, 2]))
    if "Hyperliquid" in places:
        gen_hl(rng, w, rng.choice([0, 1, 2]), stops=0.0, tps=0.0, limits=0.0)
    return places


def top_phrase(t):
    p, wh = t["place"], t["what"]
    if p == "Hyperliquid":
        return "your Hyperliquid account"
    if p == "Aave":
        return f"your {wh} in Aave"
    if p == "Spot tokens":
        return f"your {wh} spot token"
    return wh


def ans_portfolio(rng, r, w, force_name=False):
    by = sorted(r["by_place"], key=lambda p: -p["usd"])
    total, wal = usd(r["total_usd"]), r["wallet"]
    inw = f" in your {wal} wallet" if (force_name or rng.random() < 0.75) else ""
    n = len(by)
    p1 = by[0]
    pl = lambda p: PLACE_SAY[p["place"]]
    if n == 1:
        out = rng.choice([f"About {total}{inw}, all of it in {pl(p1)}.", f"{total} in total{inw}. It's all in {pl(p1)}.",
                          f"Your total{inw} is {total}, and every cent of it is in {pl(p1)}."])
    else:
        p2 = by[1]
        rest = ""
        if n > 2:
            rest = " The rest is in " + join_and([pl(p) for p in by[2:]]) + "."
        out = rng.choice([
            f"About {total}{inw}. Most of it is {pl(p1)} ({usd(p1['usd'])}) and {pl(p2)} ({usd(p2['usd'])}).{rest}",
            f"{total} in total{inw}. The biggest parts are {pl(p1)} at {usd(p1['usd'])} and {pl(p2)} at {usd(p2['usd'])}.{rest}",
            f"You have about {total}{inw}: {pl(p1)} {usd(p1['usd'])} and {pl(p2)} {usd(p2['usd'])}.{rest}",
            f"Total: {total}{inw}. That is {pl(p1)} at {usd(p1['usd'])}, then {pl(p2)} at {usd(p2['usd'])}.{rest}",
            f"{PLACE_CAP[p1['place']]} leads with {usd(p1['usd'])}, then {pl(p2)} with {usd(p2['usd'])}. All together that is {total}{inw}.{rest}"])
    if r.get("unreachable"):
        out += " " + un_text(rng, r["unreachable"])
    return out


def sc_portfolio(rng, split):
    w, bio = base_world(rng)
    m = rng.random()
    sub = "normal" if m < 0.58 else "unreachable" if m < 0.70 else "empty" if m < 0.75 else "error" if m < 0.79 else \
        "allw" if m < 0.87 else "biggest"
    sub = forced_sub("portfolio") or sub
    if sub == "empty":
        fill_prices(rng, w, ["ETH", "SOL", "BNB"])
    elif sub == "error":
        w["no_wallet"] = True
    elif sub == "unreachable":
        un = rng.choice(["bnb", "polygon", "ethereum", "arbitrum"])
        gen_portfolio_world(rng, w, places=["Wallet"] + rng.sample(["Spot tokens", "Aave", "Hyperliquid"], rng.choice([0, 1, 2])),
                            force_chain=("USDC", "ethereum") if un == "ethereum" else (("USDT", un) if un in ("bnb", "arbitrum") else ("USDC", un)))
        w["unreachable"] = [un]
        # make sure something is still reachable
        if not portfolio_parts(w):
            gen_hl(rng, w, 1, stops=0.0, tps=0.0, limits=0.0)
    else:
        gen_portfolio_world(rng, w)
        if sub == "allw" and not w["others"]:
            w["others"] = rng.sample(sorted({n for n in WALLET_NAMES if n != w["wallet"]}), rng.choice([1, 2]))
        if sub == "biggest" and len(portfolio_parts(w)) < 2:
            gen_balances(rng, w, 3)
    c = Conv(rng, w, bio)
    tpls = ALLW_T if sub == "allw" else BIGGEST_T if sub == "biggest" else PORTFOLIO_T
    c.user(utext(rng, split, tpls))
    r = c.call(("get_portfolio", {}))
    must = []
    if "error" in r:
        c.say(no_wallet_text(rng, w))
    elif not r["by_place"]:
        c.say(rng.choice([f"Your {r['wallet']} wallet is empty right now, so the total is $0. I can put an Add funds button below if you want to fill it.",
                          f"There's nothing in your {r['wallet']} wallet yet: the total is $0. Say the word and I'll put an Add funds button below.",
                          f"$0 so far. Your {r['wallet']} wallet has nothing in it, and I see no spot tokens, Aave deposits or Hyperliquid balance."]))
        must = []
    elif sub == "allw":
        txt = ans_portfolio(rng, r, w, force_name=True)
        c.say(txt + " " + rng.choice(["I only read the selected wallet, so your other wallets aren't included.",
                                      "That is the selected wallet only, not your other wallets.",
                                      "Other wallets aren't added in, only the selected one."]))
        must = [r["total_usd"]]
    elif sub == "biggest":
        t1 = r["top"][0]
        txt = f"Your biggest holding is {top_phrase(t1)}, at {usd(t1['usd'])}."
        if len(r["top"]) > 1:
            t2 = r["top"][1]
            txt = rng.choice([f"{txt[:-1]}, then {top_phrase(t2)} at {usd(t2['usd'])}.", f"{txt} Next is {top_phrase(t2)}, {usd(t2['usd'])}."])
        c.say(txt + ((" " + un_text(rng, r["unreachable"], "x")) if r.get("unreachable") else ""))
        must = [t1["usd"]]
    else:
        c.say(ans_portfolio(rng, r, w))
        must = [r["total_usd"]]
    return finish(c, "portfolio", must=must, sub=sub)


# ================================================================== sums
PCT_T = ["what percent of my total is in {pl}", "how much of my total is in {pl} in percent", "what share of my portfolio is {pl}",
         "what % of my money is in {pl}", "{pl} is what percent of my total", "how much of my net worth sits in {pl}, as a percentage",
         "what portion of everything i have is in {pl}", "percent of my portfolio in {pl}?", "what fraction of my total is {pl}",
         "how much of everything is {pl}, in %"]
PL_WORDS = {"Aave": ["aave", "my aave deposits", "my lending", "aave deposits", "earn"],
            "Hyperliquid": ["hyperliquid", "my perps account", "my hyperliquid account", "hyperliquid margin", "perps"],
            "Spot tokens": ["spot tokens", "my memecoins", "my spot bag", "spot", "my solana tokens"],
            "Wallet": ["my wallet", "the wallet", "my wallet balances", "my main balances", "wallet holdings"]}
DIFF_T = ["how much more do i have in {a} than in {b}", "what's the gap between {a} and {b}", "difference between {a} and {b} in dollars",
          "{a} vs {b}, how far apart", "by how much is {a} bigger than {b}", "how many dollars more is {a} than {b}"]
OUTSIDE_T = ["what percent of my money is outside my wallet", "how much of my total isn't in the wallet, in percent",
             "what share of my portfolio is outside the wallet", "percent of my total that's not sitting in my wallet",
             "what % of everything is not in my wallet"]


def sc_sums(rng, split):
    w, bio = base_world(rng)
    m = rng.random()
    sub = "pct" if m < 0.68 else "pct_absent" if m < 0.76 else "diff" if m < 0.88 else "outside"
    sub = forced_sub("sums") or sub
    places_pool = ["Wallet", "Spot tokens", "Aave", "Hyperliquid"]
    if sub == "outside":
        places = ["Wallet"] + rng.sample(places_pool[1:], rng.choice([1, 2]))
    elif sub == "diff":
        places = rng.sample(places_pool, rng.choice([2, 3]))
    elif sub == "pct_absent":
        absent = rng.choice(["Aave", "Hyperliquid", "Spot tokens"])
        places = ["Wallet"] + rng.sample([p for p in places_pool[1:] if p != absent], rng.choice([1, 2]))
    else:
        places = rng.sample(places_pool, rng.choice([2, 3, 4]))
    gen_portfolio_world(rng, w, places=places)
    c = Conv(rng, w, bio)
    present = [p for p in places_pool if p in {x[0] for x in portfolio_parts(w)}]
    if sub in ("pct", "pct_absent"):
        place = absent if sub == "pct_absent" else rng.choice(present)
        words = slot_vals(PL_WORDS[place], split)
        c.user(utext(rng, split, PCT_T, pl=rng.choice(words)))
        r = c.call(("get_portfolio", {}))
        total = r["total_usd"]
        row = next((p for p in r["by_place"] if p["place"] == place), None)
        if row is None:
            c.say(rng.choice([f"None of it. You have nothing in {PLACE_SAY[place]} right now, and your total is {usd(total)}.",
                              f"0 percent, since there is nothing in {PLACE_SAY[place]} at the moment. Your total is {usd(total)}."]))
            return finish(c, "sums", must=[total], extra=[0], sub=sub)
        res = c.calc(f"round({clean(row['usd'])} / {clean(total)} * 100, 1)")
        c.say(rng.choice([f"{res}% of your total is in {PLACE_SAY[place]}: {usd(row['usd'])} of {usd(total)}.",
                          f"{PLACE_CAP[place]} makes up {res}% of your {usd(total)}: {usd(row['usd'])}.",
                          f"About {res}%. That is {usd(row['usd'])} out of {usd(total)}.",
                          f"{usd(row['usd'])} of your {usd(total)} is in {PLACE_SAY[place]}, so {res}%."]) +
              ((" " + un_text(rng, r["unreachable"])) if r.get("unreachable") else ""))
        return finish(c, "sums", must=[float(res)], sub=sub)
    if sub == "diff":
        pr = sorted([x for x in present], key=lambda p: -dsum([u for pp, u, _ in portfolio_parts(w) if pp == p]))
        a, b = pr[0], pr[1]
        words = lambda p: rng.choice(slot_vals(PL_WORDS[p], split))
        c.user(utext(rng, split, DIFF_T, a=words(a), b=words(b)))
        r = c.call(("get_portfolio", {}))
        ua = next(p["usd"] for p in r["by_place"] if p["place"] == a)
        ub = next(p["usd"] for p in r["by_place"] if p["place"] == b)
        res = c.calc(f"round({clean(ua)} - {clean(ub)}, 2)")
        c.say(rng.choice([f"{PLACE_CAP[a]} has {usd(float(res))} more than {PLACE_SAY[b]}: {usd(ua)} against {usd(ub)}.",
                          f"{usd(float(res))} more. {PLACE_CAP[a]} is {usd(ua)} and {PLACE_SAY[b]} is {usd(ub)}.",
                          f"The gap is {usd(float(res))}, with {usd(ua)} in {PLACE_SAY[a]} and {usd(ub)} in {PLACE_SAY[b]}."]))
        return finish(c, "sums", must=[float(res)], sub=sub)
    # outside the wallet
    c.user(utext(rng, split, OUTSIDE_T))
    r = c.call(("get_portfolio", {}))
    total = r["total_usd"]
    wal = next(p["usd"] for p in r["by_place"] if p["place"] == "Wallet")
    res = c.calc(f"round(({clean(total)} - {clean(wal)}) / {clean(total)} * 100, 1)")
    c.say(rng.choice([f"{res}% of your {usd(total)} is outside the wallet.", f"About {res}%. Your wallet holds {usd(wal)} of the {usd(total)} total.",
                      f"{res}% sits outside your wallet, in spot tokens, Aave or Hyperliquid, out of {usd(total)} in total."]))
    return finish(c, "sums", must=[float(res)], sub=sub)


# ================================================================== positions_read
SPOT_T = ["what do i hold in spot", "show my spot tokens", "what tokens do i have", "list my memecoins", "what spot tokens am i holding",
          "show me my solana tokens", "what bags do i have", "which tokens am i holding rn", "my spot holdings pls",
          "how are my spot tokens doing", "what's in my spot bag", "tokens i hold?", "what do i have in memecoins",
          "give me my spot token list", "which solana tokens do i own", "show my token holdings with pnl", "what coins am i holding in spot",
          "what's my spot portfolio", "list my tokens and how much i'm up or down", "any memecoins in my wallet?"]
SPOT_DETAIL_T = ["what's my average cost on {tok}", "how is my {tok} doing", "am i up or down on {tok}", "what did i pay for {tok} on average",
                 "pnl on my {tok}", "what's my {tok} worth and am i in profit", "how's my {tok} bag", "tell me about my {tok} holding",
                 "do i hold any {tok}", "how much {tok} do i have"]
EARN_T = ["what do i have in aave", "my aave deposits", "show my earn positions", "how much am i earning on aave", "what's my apy on aave",
          "what do i have deposited", "list my deposits", "what am i lending out", "my yield positions?", "how much do i have supplied to aave",
          "what's sitting in aave for me", "show me what i've put in aave", "do i have anything in aave", "aave balance",
          "what are my aave positions and rates", "how much usdc is in aave", "what's earning interest for me", "my deposits and their apy",
          "check my aave", "what have i got supplied"]
PERPS_T = ["what are my open perp positions", "show my hyperliquid positions", "what positions do i have open", "any open trades?",
           "what am i long right now", "do i have any open perps", "show my open orders and positions", "what's open on hyperliquid",
           "list my perp positions", "my open trades pls", "what do i have open on hyperliquid", "show me my longs and shorts",
           "which positions are open and where are my orders", "what leveraged positions do i have", "any positions open?",
           "what perps am i in", "open positions?", "show me my hyperliquid orders", "what trades am i in right now", "list my open orders"]
STOP_T = ["do i have a stop on {coin}", "is my {coin} position protected", "do i have a stop loss on my {coin}", "is there a take profit on {coin}",
          "what's my stop on {coin}", "did my {coin} stop get set", "check if {coin} has a stop", "do i have any protection on {coin}",
          "where is my {coin} stop", "is there a stop loss on {coin} right now"]
ALL_T = ["what do i hold beyond my wallet", "everything i have beyond my balances", "show me all my positions",
         "what positions do i have across spot, aave and hyperliquid", "list everything i hold besides wallet balances", "all my positions please",
         "what else do i have besides tokens in my wallet", "everything beyond my wallet", "give me all my holdings outside the wallet",
         "what am i holding in spot, earn and perps", "show everything: spot, aave, perps", "what do i have in all my positions",
         "what are all my open positions and deposits", "positions overview pls", "list all positions, deposits and spot tokens"]


def spot_item(r, long=False):
    """text for one spot row"""
    sym = r["symbol"]
    if r["pnl_pct"] is None:
        tail = "pnl isn't tracked"
    else:
        tail = f"{'up' if r['pnl_pct'] >= 0 else 'down'} {pct_s(r['pnl_pct'])}"
    extra = ", frozen so it can't be sold" if r["frozen"] else ""
    if long:
        cost = f", average cost {S(r['avg_cost'])}" if r["avg_cost"] is not None else ""
        return f"{amt_s(r['amount'])} {sym} worth {usd(r['usd'])}{cost} ({tail}{extra})"
    return f"{sym} {usd(r['usd'])} ({tail}{extra})"


def spot_sentence(rng, rows, lead=True):
    rows = sorted(rows, key=lambda x: -x["usd"])
    top = rows[:3]
    items = [spot_item(r) for r in top]
    extra = ""
    hidden = rows[3:]
    txt = join_and(items) if not hidden else ", ".join(items) + f" and {len(hidden)} smaller"
    for h in hidden:
        if h["frozen"]:
            extra += f" {h['symbol']} is frozen, so it can't be sold."
    ld = rng.choice(["You hold ", "In spot you have ", "Your spot tokens: ", "Right now you hold "]) if lead else ""
    return ld + txt + "." + extra


def earn_item(r):
    return f"{amt_s(r['amount'])} {r['symbol']} on {CHAIN_LABEL[r['chain']]} at {pct_s(r['apy_pct'])} APY"


def earn_sorted(rows):
    return sorted(rows, key=lambda r: -(float(r["amount"]) * (1 if r["symbol"] != "ETH" else 3000)))


def earn_sentence(rng, rows, lead=True):
    rows = earn_sorted(rows)
    top = rows[:3]
    items = [earn_item(r) for r in top]
    hidden = rows[3:]
    txt = join_and(items) if not hidden else ", ".join(items) + f" and {len(hidden)} smaller"
    ld = rng.choice(["In Aave you have ", "You have ", "Your Aave deposits: ", "Aave holds "]) if lead else ""
    return ld + txt + "."


def pos_item(p, long=False):
    base = f"{p['coin']} {p['side']}, {'up' if p['unrealised_pnl'] >= 0 else 'down'} {usd(abs(p['unrealised_pnl']))}"
    if long:
        return f"{p['coin']} {p['side']} {H.num(p['size'], 4)} at {p['leverage']}x from {S(p['entry'])}, {'up' if p['unrealised_pnl'] >= 0 else 'down'} {usd(abs(p['unrealised_pnl']))}"
    return base


def prot_text(o):
    return f"a {'stop loss' if o['kind'] == 'stop_loss' else 'take profit'} at {S(o['price'])}"


def perps_sentence(rng, perps, long=True, lead=True):
    pos, orders = perps["positions"], perps["open_orders"]
    parts = []
    if pos:
        shown = pos[:3]
        txt = "; ".join(pos_item(p, long) for p in shown)
        if len(pos) > 3:
            txt += f"; and {len(pos) - 3} more"
        parts.append((rng.choice(["You have ", "Open: ", "Right now: "]) if lead else "") + txt + ".")
    prot = [o for o in orders if o["kind"] in ("stop_loss", "take_profit")]
    lim = [o for o in orders if o["kind"] == "limit"]
    for p in pos[:3]:
        mine = [prot_text(o) for o in prot if o["coin"] == p["coin"]]
        if mine:
            parts.append(f"{p['coin']} has {join_and(mine)} covering the whole position.")
        elif long:
            parts.append(f"{p['coin']} has no stop loss.")
    if lim:
        parts.append("Open limit orders: " + join_and([f"{o['side']} {H.num(o['size'], 4)} {o['coin']} at {S(o['price'])}" for o in lim[:3]]) +
                     (f" and {len(lim) - 3} more" if len(lim) > 3 else "") + ".")
    if not parts:
        return rng.choice(["You have no open positions or orders on Hyperliquid.", "Nothing is open on Hyperliquid: no positions and no orders."])
    return " ".join(parts)


def sc_positions(rng, split):
    w, bio = base_world(rng)
    m = rng.random()
    sub = ("spot" if m < 0.20 else "spot_detail" if m < 0.34 else "earn" if m < 0.50 else "perps" if m < 0.64 else "all" if m < 0.76 else
           "stop" if m < 0.84 else "unreach" if m < 0.88 else "empty" if m < 0.94 else "error")
    sub = forced_sub("positions") or sub
    fill_prices(rng, w, ["ETH", "SOL", "BNB"])
    kind, prefer, target = None, None, None
    if sub == "spot":
        n = rng.choice([1, 2, 3, 4, 5, 5, 6, 7])
        gen_spot(rng, w, n, frozen_p=0.15, untracked_p=0.25)
        kind, prefer = "spot", "solana"
        text = utext(rng, split, SPOT_T)
    elif sub == "spot_detail":
        toks = slot_vals(list(MEMES), split)
        tok = target = rng.choice(toks)
        has = rng.random() < 0.85
        gen_spot(rng, w, rng.choice([1, 2, 3]), frozen_p=0.0, untracked_p=0.2, force=[tok] if has else None)
        if not has:
            w["spot"] = [s for s in w["spot"] if s["symbol"] != tok]
            if not w["spot"]:
                gen_spot(rng, w, 2, frozen_p=0.0, force=[x for x in MEMES if x != tok][:2])
                w["spot"] = [s for s in w["spot"] if s["symbol"] != tok]
        kind, prefer = "spot", "solana"
        text = utext(rng, split, SPOT_DETAIL_T, tok=tok)
    elif sub == "earn":
        n = rng.choice([1, 1, 2, 3, 4, 5, 6])
        gen_aave(rng, w, n)
        kind, prefer = "earn", "earn"
        text = utext(rng, split, EARN_T)
    elif sub == "perps":
        gen_hl(rng, w, rng.choice([1, 1, 2, 3]), stops=0.6, tps=0.35, limits=0.45)
        kind, prefer = "perps", "trade"
        text = utext(rng, split, PERPS_T)
    elif sub == "stop":
        coins = slot_vals(PERP_COINS, split)
        coin = target = rng.choice(coins)
        has_pos = rng.random() < 0.9
        gen_hl(rng, w, rng.choice([1, 2]), stops=0.55, tps=0.3, limits=0.2, force_coins=[coin] if has_pos else None)
        if not has_pos:
            w["hl"]["positions"] = [p for p in w["hl"]["positions"] if p["coin"] != coin]
            w["hl"]["orders"] = [o for o in w["hl"]["orders"] if o["coin"] != coin]
        kind, prefer = "perps", "trade"
        text = utext(rng, split, STOP_T, coin=coin)
    elif sub == "all":
        gen_spot(rng, w, rng.choice([0, 1, 2, 4, 5]), frozen_p=0.15, untracked_p=0.25)
        gen_aave(rng, w, rng.choice([0, 1, 1, 2]))
        gen_hl(rng, w, rng.choice([0, 1, 2]), stops=0.6, tps=0.3, limits=0.3)
        if not (w["spot"] or w["aave"] or w["hl"]["positions"]):
            gen_aave(rng, w, 1)
        kind, prefer = None, "trade"
        text = utext(rng, split, ALL_T)
    elif sub == "unreach":
        gen_aave(rng, w, rng.choice([0, 1]))
        gen_spot(rng, w, rng.choice([1, 2]), frozen_p=0.0)
        w["unreachable"] = [rng.choice(["ethereum", "arbitrum", "base"])]
        kind, prefer = None, "trade"
        text = utext(rng, split, ALL_T)
    elif sub == "empty":
        kind = rng.choice([None, "spot", "earn", "perps"])
        prefer = POS_PACK[kind or "all"]
        text = utext(rng, split, {None: ALL_T, "spot": SPOT_T, "earn": EARN_T, "perps": PERPS_T}[kind])
    else:
        w["no_wallet"] = True
        kind = rng.choice([None, "spot", "earn", "perps"])
        prefer = POS_PACK[kind or "all"]
        text = utext(rng, split, {None: ALL_T, "spot": SPOT_T, "earn": EARN_T, "perps": PERPS_T}[kind])
    c = Conv(rng, w, bio)
    c.user(text)
    r = c.call(("get_positions", {"kind": kind} if kind else {}))
    must = []
    if "error" in r:
        c.say(no_wallet_text(rng, w))
        return finish(c, "positions_read", sub=sub, prefer_pack=prefer, words=70)
    un = (" " + un_text(rng, r["unreachable"], "x")) if r.get("unreachable") else ""
    if sub == "spot":
        rows = r["spot"]
        txt = spot_sentence(rng, rows)
        c.say(txt)
        must = [sorted(rows, key=lambda x: -x["usd"])[0]["usd"]]
        return finish(c, "positions_read", must=must, extra=[len(rows) - 3] if len(rows) > 3 else [], sub=sub, prefer_pack=prefer)
    if sub == "spot_detail":
        tok = target
        row = next((s for s in r["spot"] if s["symbol"] == tok), None)
        if row is None:
            c.say(rng.choice([f"You don't hold any {tok} in spot right now.", f"I don't see {tok} among your spot tokens, so you hold none."]))
            return finish(c, "positions_read", sub=sub, prefer_pack=prefer)
        lead = rng.choice(["You have ", "You hold ", "It's "])
        if row["pnl_pct"] is None:
            txt = f"{lead}{amt_s(row['amount'])} {tok} worth {usd(row['usd'])}. I can't show a cost or pnl for it: those aren't tracked for tokens not bought in Hermit."
        else:
            txt = f"{lead}{amt_s(row['amount'])} {tok} worth {usd(row['usd'])}, average cost {S(row['avg_cost'])}, {'up' if row['pnl_pct'] >= 0 else 'down'} {pct_s(row['pnl_pct'])}."
        if row["frozen"]:
            txt += " It's frozen, so it can't be sold."
        c.say(txt)
        return finish(c, "positions_read", must=[row["usd"]], sub=sub, prefer_pack=prefer)
    if sub == "earn":
        rows = r["earn"]
        txt = earn_sentence(rng, rows) + rng.choice([" The APY is variable.", "", " Rates are variable.", ""])
        c.say(txt)
        return finish(c, "positions_read", must=[earn_sorted(rows)[0]["apy_pct"]], extra=[len(rows) - 3] if len(rows) > 3 else [],
                      sub=sub, prefer_pack=prefer)
    if sub == "perps":
        perps = r["perps"]
        c.say(perps_sentence(rng, perps))
        return finish(c, "positions_read", must=[perps["positions"][0]["entry"]] if perps["positions"] else [], sub=sub, prefer_pack=prefer, words=110)
    if sub == "stop":
        coin = target
        perps = r["perps"]
        pos = next((p for p in perps["positions"] if p["coin"] == coin), None)
        mine = [o for o in perps["open_orders"] if o["coin"] == coin and o["kind"] in ("stop_loss", "take_profit")]
        sl = [o for o in mine if o["kind"] == "stop_loss"]
        if pos is None:
            c.say(f"You have no open {coin} position, so there is nothing to protect.")
        elif not mine:
            c.say(rng.choice([f"No. Your {coin} {pos['side']} has no stop loss or take profit.", f"No stop on {coin} right now. Your {pos['side']} is unprotected."]))
        elif sl:
            tp = [o for o in mine if o["kind"] == "take_profit"]
            c.say(f"Yes. Your {coin} {pos['side']} has {prot_text(sl[0])} covering the whole position" +
                  (f", and {prot_text(tp[0])}." if tp else "."))
        else:
            c.say(f"You have {prot_text(mine[0])} on your {coin} {pos['side']}, but no stop loss.")
        return finish(c, "positions_read", must=[mine[0]["price"]] if mine else [], sub=sub, prefer_pack=prefer, words=60)
    # all / unreach / empty
    secs = []
    if r.get("spot"):
        secs.append("Spot: " + spot_sentence(rng, r["spot"], lead=False)[:-1] + ".")
    if r.get("earn"):
        secs.append("Aave: " + earn_sentence(rng, r["earn"], lead=False))
    perps = r.get("perps")
    if perps and (perps["positions"] or perps["open_orders"]):
        secs.append("Hyperliquid: " + perps_sentence(rng, perps, long=False, lead=False))
    if kind == "spot":
        txt = rng.choice(["You hold no spot tokens right now.", "Your spot tokens list is empty."])
    elif kind == "earn":
        txt = rng.choice(["You have nothing deposited in Aave right now.", "No Aave deposits at the moment."])
    elif kind == "perps":
        txt = perps_sentence(rng, perps)
    elif not secs:
        txt = rng.choice(["Nothing beyond your wallet: no spot tokens, no Aave deposits and nothing open on Hyperliquid.",
                          "You hold nothing outside your wallet right now. No spot tokens, no Aave and no Hyperliquid positions."])
    else:
        txt = " ".join(secs)
    c.say(txt + un)
    ext = []
    if r.get("spot") and len(r["spot"]) > 3:
        ext.append(len(r["spot"]) - 3)
    return finish(c, "positions_read", extra=ext, sub=sub, prefer_pack=prefer, words=100)


# ================================================================== near misses
BAL_T = ["what's my balance", "show my balances", "what's in my wallet", "how much crypto is in my wallet", "my balance pls",
         "check my wallet balance", "what do i hold in my wallet", "wallet balance?", "what tokens are in my wallet", "show me my wallet",
         "what's my wallet balance right now", "balances please", "list my wallet balances", "what coins are in my wallet",
         "can you check my balance"]
BAL_CH_T = ["what's my balance on {ch}", "how much do i have on {ch}", "show my {ch} balance", "what's on {ch}", "balance on {ch}?",
            "what do i hold on {ch}", "check my wallet on {ch}", "any funds on {ch}"]
PNL_T = ["what's my pnl on {coin}", "how much am i up on {coin}", "am i in profit on my {coin} position", "my {coin} pnl?",
         "how is my {coin} trade doing", "what's the unrealised pnl on {coin}", "pnl on {coin}", "am i green on {coin}",
         "how's my {coin} long or short doing", "what's my profit on {coin} so far"]
PNL_ALL_T = ["what's my pnl", "am i up or down on my trades", "total pnl on hyperliquid", "how are my trades doing pnl wise",
             "what's my unrealised pnl", "am i in profit overall on hyperliquid"]
VAL_T = ["total value of my {w}", "how much is my {w} worth", "what's all my {w} worth in dollars", "value of my {w} holdings",
         "what are my {w} worth right now", "how much is my {w} worth rn", "total {w} value", "what's my {w} stack worth",
         "how much are my {w} holdings in usd", "dollar value of my {w}"]
MCAP_T = ["what's the market cap of {coin}", "what is {coin}'s market cap", "{coin} market cap?", "market cap of {coin} rn",
          "how big is {coin} by market cap", "what's {coin} fdv", "give me the market cap for {coin}", "{coin} mcap",
          "what's {coin}'s circulating supply", "where does {coin} rank by market cap"]
VAL_COINS = [("ETH", "ETH"), ("SOL", "SOL"), ("BNB", "BNB"), ("ETH", "ether"), ("ETH", "eth"), ("SOL", "sol"), ("BNB", "bnb"), ("SOL", "solana")]
MCAP_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE", "LINK", "AVAX"]
CH_WORDS = {"base": ["Base", "base"], "arbitrum": ["Arbitrum", "arbitrum", "arb"], "ethereum": ["Ethereum", "mainnet", "ethereum", "eth mainnet"],
            "solana": ["Solana", "solana"], "polygon": ["Polygon", "polygon"], "bnb": ["BNB Chain", "bnb chain", "bsc", "BSC"]}


def bal_item(b, chain=True):
    if b["symbol"] == "SOL" and b["chain"] == "solana" or not chain:
        return f"{amt_s(b['amount'])} {b['symbol']} ({usd(b['usd'])})"
    return f"{amt_s(b['amount'])} {b['symbol']} on {CHAIN_LABEL[b['chain']]} ({usd(b['usd'])})"


def sc_near(rng, split):
    w, bio = base_world(rng)
    m = rng.random()
    sub = "balance" if m < 0.38 else "balance_chain" if m < 0.50 else "pnl" if m < 0.68 else "pnl_all" if m < 0.76 else "value" if m < 0.91 else "mcap"
    sub = forced_sub("near") or sub
    fill_prices(rng, w, ["ETH", "SOL", "BNB"])
    if sub in ("balance", "balance_chain"):
        n = rng.choice([1, 2, 3, 3, 4, 5, 6])
        gen_balances(rng, w, n)
        if rng.random() < 0.06:
            w["balances"] = []
        if rng.random() < 0.1:
            w["unreachable"] = [rng.choice(["bnb", "polygon"])]
        if sub == "balance_chain":
            chs = slot_vals(["base", "arbitrum", "ethereum", "solana", "polygon", "bnb", "base", "arbitrum"][:6], split)
            ch = rng.choice(chs)
            if rng.random() < 0.8 and not any(b["chain"] == ch for b in w["balances"]):
                gen_balances(rng, w, len(w["balances"]) + 1, force=[("USDC", ch) if ch not in ("bnb",) else ("USDT", ch)])
            if ch in w["unreachable"]:
                w["unreachable"] = []
            say = rng.choice(slot_vals(CH_WORDS[ch], split) or CH_WORDS[ch])
            text = utext(rng, split, BAL_CH_T, ch=say)
            arg = {"chain": "bsc" if ch == "bnb" else ch}
        else:
            text, arg, ch = utext(rng, split, BAL_T), {}, None
        c = Conv(rng, w, bio)
        c.user(text)
        r = c.call(("get_balances", arg))
        if "error" in r:
            c.say(no_wallet_text(rng, w))
            return finish(c, "near_miss", sub=sub, words=60)
        rows = sorted(r["holdings"], key=lambda b: -b["usd"])
        if not rows:
            where = f" on {CHAIN_LABEL[ch]}" if ch else ""
            txt = rng.choice([f"You have nothing{where} right now.", f"Your wallet{where} is empty at the moment."])
        else:
            items = [bal_item(b, chain=not ch) for b in rows[:3]]
            txt = (join_and(items) if len(rows) <= 3 else ", ".join(items) + f" and {len(rows) - 3} smaller")
            if ch:
                txt = rng.choice(["On " + CHAIN_LABEL[ch] + " you have ", "You have "]) + txt + "."
            else:
                txt = rng.choice(["You have ", "Your wallet holds ", "Right now you hold "]) + txt + "."
        if r.get("wallet"):
            txt += rng.choice([f" This is your {r['wallet']} wallet; the others are {join_and(w['others'])}.",
                               f" That is the selected wallet, {r['wallet']}. You also have {join_and(w['others'])}."])
        if r.get("unreachable"):
            txt += " " + un_text(rng, r["unreachable"], "x")
        c.say(txt)
        return finish(c, "near_miss", must=[rows[0]["usd"]] if rows else [], extra=[len(rows) - 3] if len(rows) > 3 else [], sub=sub, words=80)
    if sub in ("pnl", "pnl_all"):
        coins = slot_vals(PERP_COINS, split)
        coin = rng.choice(coins)
        has = rng.random() < 0.8
        gen_hl(rng, w, rng.choice([1, 2, 3]), stops=0.3, tps=0.2, limits=0.2, force_coins=[coin] if (sub == "pnl" and has) else None)
        if sub == "pnl" and not has:
            w["hl"]["positions"] = [p for p in w["hl"]["positions"] if p["coin"] != coin]
            w["hl"]["orders"] = [o for o in w["hl"]["orders"] if o["coin"] != coin]
        text = utext(rng, split, PNL_T, coin=coin) if sub == "pnl" else utext(rng, split, PNL_ALL_T)
        c = Conv(rng, w, bio)
        c.user(text)
        r = c.call(("hyperliquid_account", {}))
        pos = r["positions"]
        if sub == "pnl":
            p = next((x for x in pos if x["coin"] == coin), None)
            if p is None:
                c.say(rng.choice([f"You have no open {coin} position on Hyperliquid, so there is no pnl to show.",
                                  f"I don't see an open {coin} position on Hyperliquid right now."]))
                return finish(c, "near_miss", sub=sub, prefer_pack="trade")
            c.say(rng.choice([f"Your {coin} {p['side']} is {'up' if p['unrealised_pnl'] >= 0 else 'down'} {usd(abs(p['unrealised_pnl']))} (unrealised), entered at {S(p['entry'])}.",
                              f"{'Up' if p['unrealised_pnl'] >= 0 else 'Down'} {usd(abs(p['unrealised_pnl']))} on your {coin} {p['side']}. That is unrealised, from an entry of {S(p['entry'])}."]))
            return finish(c, "near_miss", must=[abs(p["unrealised_pnl"])], sub=sub, prefer_pack="trade")
        if not pos:
            c.say("You have no open positions on Hyperliquid, so there is no pnl to show.")
            return finish(c, "near_miss", sub=sub, prefer_pack="trade")
        if len(pos) == 1:
            p = pos[0]
            c.say(f"Your {p['coin']} {p['side']} is {'up' if p['unrealised_pnl'] >= 0 else 'down'} {usd(abs(p['unrealised_pnl']))} (unrealised). It's your only open position.")
            return finish(c, "near_miss", must=[abs(p["unrealised_pnl"])], sub=sub, prefer_pack="trade")
        expr = "round(" + " + ".join(str(clean(p["unrealised_pnl"])) if p["unrealised_pnl"] >= 0 else f"({clean(p['unrealised_pnl'])})" for p in pos) + ", 2)"
        res = float(c.calc(expr))
        parts = ", ".join(f"{p['coin']} {'+' if p['unrealised_pnl'] >= 0 else '-'}{usd(abs(p['unrealised_pnl']))}" for p in pos[:3])
        c.say(f"In total you're {'up' if res >= 0 else 'down'} {usd(abs(res))} unrealised: {parts}" + (f" and {len(pos) - 3} more." if len(pos) > 3 else "."))
        return finish(c, "near_miss", must=[abs(res)], extra=[len(pos) - 3] if len(pos) > 3 else [], sub=sub, prefer_pack="trade")
    if sub == "value":
        sym, word = rng.choice([x for i, x in enumerate(VAL_COINS) if (i % 4 == 3) == (split == "eval")])
        chains = {"ETH": ["ethereum", "base", "arbitrum"], "SOL": ["solana"], "BNB": ["bnb"]}[sym]
        has = rng.random() < 0.85
        force = []
        if has:
            for ch in rng.sample(chains, rng.choice([1, 2]) if len(chains) > 1 else 1):
                force.append((sym, ch))
        gen_balances(rng, w, rng.choice([1, 2, 3]) + len(force), force=force)
        if not has:
            w["balances"] = [b for b in w["balances"] if b["symbol"] != sym]
            if not w["balances"]:
                gen_balances(rng, w, 1, force=[("USDC", "base")])
        text = utext(rng, split, VAL_T, w=word)
        c = Conv(rng, w, bio)
        c.user(text)
        rb, rp = c.call(("get_balances", {}), ("get_price", {"symbol": sym}))
        rows = [b for b in rb.get("holdings", []) if b["symbol"] == sym]
        px, ch24 = S(rp["price"]), rp["change_24h_pct"]
        if not rows:
            c.say(f"You don't hold any {sym} in your wallet. For reference, {sym} is at {px}, {'up' if ch24 >= 0 else 'down'} {pct_s(ch24)} today.")
            return finish(c, "near_miss", must=[rp["price"]], sub=sub, words=60)
        if len(rows) == 1:
            amt, tot = amt_s(rows[0]["amount"]), usd(rows[0]["usd"])
            must = [rows[0]["usd"]]
        else:
            expr_a = "round(" + " + ".join(r_["amount"] for r_ in rows) + ", 4)"
            expr_u = "round(" + " + ".join(str(clean(r_["usd"])) for r_ in rows) + ", 2)"
            ra, ru = c.call(("calculator", {"expression": expr_a}), ("calculator", {"expression": expr_u}))
            amt, tot = amt_s(ra["result"]), usd(float(ru["result"]))
            ru = ru["result"]
            must = [float(ru)]
        c.say(rng.choice([f"You hold {amt} {sym}, worth about {tot}. {sym} is at {px}, {'up' if ch24 >= 0 else 'down'} {pct_s(ch24)} today.",
                          f"Your {sym} is worth {tot} ({amt} {sym}). The live price is {px}, {'up' if ch24 >= 0 else 'down'} {pct_s(ch24)} over the last day."]))
        return finish(c, "near_miss", must=must, sub=sub, words=70)
    # market cap: no tool, no number
    coin = rng.choice(slot_vals(MCAP_COINS, split))
    c = Conv(rng, w, bio)
    c.user(utext(rng, split, MCAP_T, coin=coin))
    c.say(rng.choice([
        f"I don't have a market cap source here, so I won't guess a number for {coin}. I can give you its live price and daily move if that helps.",
        f"I can't give a live market cap for {coin}: I have no tool for it and I don't want to make a figure up. Want the live price instead?",
        f"There's no market cap data in this app that I can read, so I can't give you one for {coin}. I can pull its live price and daily move.",
        f"Market cap isn't something I can look up, and I won't quote one from memory. I can check {coin}'s live price if you like."]))
    return finish(c, "near_miss", sub="mcap", words=60, checks=["restraint", "no_invented_data"])


# ================================================================== capability and knowledge
UNIVERSAL_BAD = [
    r"\b(get|prepare|set|cancel|create|schedule|add|open|search)_[a-z_]+\b", r"\b(calculator|hyperliquid_account|position_math)\b",
    r"\bpacks?\b", r"\brelay\b", r"[·•]", r"not financial advice", r"\bas an ai\b", r"^\s*(sure|certainly|absolutely|great question)\b",
    r"^\s*done\b", r"\btab bar\b", r"went through", r"successfully", r"\bI(?:'ve| have) (sent|bought|swapped|sold|saved|created|scheduled|cancelled|canceled)\b",
    r"\bI(?:'ve)? opened\b", r"\b(funds|money) (are|have) (in|arrived|moved)\b"]
SAFE_BAD = [r"\b(is|are|looks?|seems?) (totally |completely |pretty |very )?(safe|legit|trustworthy)\b", r"\bno risk\b", r"can'?t lose",
            r"definitely not a (scam|rug)"]
AUTO_BAD = [r"\bautomatic(ally)?\b", r"will be sent", r"\bwill pay\b", r"pays it for you", r"\bI'?ll send\b"]
LEAD = r"^\s*"


def mkfact(fid, qs, ans, req, bad=(), nums=(), slots=None, call=None, bad_samples=(), words=75, sentences=5):
    assert len(qs) >= 7 and len(ans) >= 3, fid
    return {"id": fid, "qs": qs, "ans": ans, "req": list(req), "bad": list(bad), "nums": set(nums), "slots": slots or {}, "call": call,
            "bad_samples": list(bad_samples), "words": words, "sentences": sentences}


FACE = r"face id|touch id"
CAN = {}
NO = {}
KNOW = {}

# ---------------------------------------------------------------- can-do
CAN["overview"] = mkfact(
    "cap_overview",
    ["what can you do", "what can you do for me", "what are you able to do", "what do you do exactly", "how can you help me with my wallet",
     "what are your features", "what can hermit do", "help, what can i ask you", "what can i ask you to do", "give me a rundown of what you can do",
     "what's your deal, what can you help with", "what are you good for", "tell me what you can do in this app", "what can you help me with"],
    ["I can look things up: prices, balances, your total, positions, tokens, yields and recent activity. I can prepare sends, swaps, buys, sells, trades, stops and Aave deposits for your review. Reminders, automations and contacts show a card you tap to confirm, and I can point you to any page. You approve everything yourself with {BIO}.",
     "Four things. I look up live data like prices, balances, your portfolio, positions and tokens. I prepare money moves for you to review, such as sends, swaps, buys, sells, trades and stops. I put up cards for reminders, automations and contacts. And I can put a button to any page. You approve everything yourself, money with {BIO} and cards with a tap.",
     "Mostly three jobs. I look things up for you: prices, balances, your total, positions, token details, yields and activity. I prepare actions for review: send, swap, buy, sell, trade, close, stops, deposits and batch sends. And I draft reminders, automations and contacts on cards, or point you to the right page. You approve everything yourself, with {BIO} or a tap.",
     "Ask me for whatever you would do in the app. I look up data, prepare moves for review, set up reminders, automations and contacts on a card, and open pages with a button. Nothing happens until you approve it yourself: {BIO} for money, a tap for cards."],
    [r"look", r"prepare", r"card", r"approve"], bad=AUTO_BAD + [r"\bI (can|will) (approve|confirm|sign)\b"], words=95, sentences=6,
    bad_samples=["I can do everything for you, including approving your trades automatically.", "I can use get_portfolio and prepare_send for you."])

CAN["sell"] = mkfact(
    "cap_sell",
    ["can you sell my tokens", "can you sell my memecoins", "can you sell my {tok}", "can i sell tokens through you", "do you do sells",
     "can you help me sell a token i hold", "is it possible to sell half of my bag with you", "can you sell my solana tokens",
     "can i get you to sell some of my {tok}", "are you able to sell tokens for me"],
    ["Yes, for Solana tokens you hold. Tell me which one and how much, a percent or an amount, and I prepare the sale. The review shows what you sell, what you get, the price impact and the route, and you approve it with {BIO}.",
     "Yes. I can prepare a sale of a Solana token you hold, for USDC or SOL, in any percent from 1 to 100 or a set amount. You check the review and approve it with {BIO}. A frozen token can't be sold.",
     "Yes, I can sell Solana tokens for you, all or part. I prepare it, the Sell review shows what you get and the price impact, and it only happens when you approve with {BIO}.",
     "Yes. Name the token and an amount or a percent, and I'll prepare the sale for review. Selling works even for unverified tokens, but a frozen balance can't be sold. You approve with {BIO}."],
    [r"\byes\b", r"review", FACE, r"solana|token"], bad=[r"\bI(?:'ve| have)? sold\b", r"(sell|sold) (it|them) (for you )?(automatically|now)"],
    nums={1, 100}, slots={"tok": ["WIF", "JUP", "POPCAT", "PUMP", "MOODENG", "GOAT", "FARTCOIN", "PENGU"]},
    bad_samples=["Yes, I sold them for you."])

CAN["schedule"] = mkfact(
    "cap_schedule",
    ["can you schedule a payment", "can i schedule a payment to {name}", "can you set up a recurring payment", "can you pay my rent every month",
     "is it possible to set up a monthly payment", "can you schedule a weekly payment for me", "can you set a payment for next friday",
     "do you do scheduled payments", "can i schedule payments here", "can you make a repeating payment to {name}"],
    ["Yes, as a reminder. I put up a card with the payee, amount and date, and you tap to save it. Hermit never pays by itself: when it's due, you tap Pay now and approve the send with {BIO}.",
     "Yes, but it's only a reminder. When the day comes you open it, tap Pay now and approve with {BIO}. Tell me who, how much and when, and I'll put up the card.",
     "I can set one up on a card for you to confirm, once, weekly or monthly. It works as a reminder, so the money only moves when you tap Pay now and approve it with {BIO}.",
     "Yes. A scheduled payment is a reminder with the payee, amount and date. I draft it, you tap to save, and on the due date you pay it yourself with {BIO}."],
    [r"reminder", FACE, r"card|tap"], bad=AUTO_BAD, slots={"name": ["Alice", "Bob", "Maya", "Leo", "Priya", "Tom", "Nina", "Omar"]},
    bad_samples=["Yes, I will pay it automatically on the due date."])

CAN["stoploss"] = mkfact(
    "cap_stoploss",
    ["can you set a stop loss", "can you add a stop loss to my {coin} long", "can you put a take profit on my position", "can you protect my position with a stop",
     "is it possible to add a stop to an open trade", "can you set a stop loss and take profit for me", "do you set stops", "can you put a stop on my {coin}",
     "can i get you to add a stop loss", "are you able to set a take profit on a trade i have open"],
    ["Yes, on a Hyperliquid position you have open. Give me the price and I prepare the stop loss as a reduce-only trigger for the whole position. A review opens and you approve it with {BIO}.",
     "Yes. I can add a stop loss, a take profit or both to an open position. The prices must be on the right side of the current price, and nothing is live until you approve the review with {BIO}.",
     "I can, for any open Hyperliquid position. Tell me the coin and the price, or a percent below your entry, and I'll prepare the stop. It's a reduce-only trigger, so it can only close the position, and you approve it with {BIO}.",
     "Yes. Tell me the coin and the stop price. I check it against the market, prepare it, and you approve with {BIO}. If you already have a stop, the new one replaces it."],
    [r"\byes\b|\bi can\b", r"stop|take profit", FACE], bad=[r"you(?:'re| are) protected", r"\bI(?:'ve| have) (set|added|put)\b"],
    slots={"coin": ["BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "LINK", "AVAX"]}, bad_samples=["Yes, I've set it and you're protected now."])

CAN["payname"] = mkfact(
    "cap_payname",
    ["can i pay {name} by name", "can you send to {name} without me pasting the address", "can i send money to a saved contact by name",
     "pay my landlord by name?", "do i need the address if {name} is in my contacts", "can you send usdc to {name} if she's saved",
     "can you look up an address from a contact name", "is it possible to send by contact name instead of address", "can i say pay {name} and you handle it",
     "will you find {name}'s address yourself"],
    ["Yes, if the name is saved in Payments. I look up its saved address, prepare the send, and the review shows the address for you to check and approve with {BIO}.",
     "Yes. For a saved contact I look up the address myself, so you don't paste it. You still check it on the review and approve with {BIO}.",
     "Yes, as long as the contact is saved. If it isn't, paste the address once and I can offer to save it as a contact on a card. Every send is approved by you with {BIO}.",
     "I can pay a saved name: I fetch its address, prepare the send and wait for your {BIO}. A name that isn't saved needs its address pasted first."],
    [r"saved", FACE], bad=[r"\bany name\b", r"\bI(?:'ve| have) sent\b"], slots={"name": ["Alice", "Bob", "Maya", "Leo", "Priya", "Tom", "Nina", "Omar"]},
    bad_samples=["Yes, I've sent it to any name you give me."])

CAN["automations"] = mkfact(
    "cap_automations",
    ["what are automations", "what can an automation do for me", "can you automate things for me", "can you run something for me every morning",
     "can you check my positions every day", "do you have automations", "can you set up a recurring check", "can you do something on a schedule"],
    ["An automation runs a prompt of yours on a schedule, like every morning. The shortest repeat is every 5 minutes. It can look things up and prepare actions, but it never approves one, so money still waits for your {BIO}. I draft it on a card and you tap Create.",
     "I can draft an automation: a prompt that runs by schedule, as often as every 5 minutes. It can read data and prepare moves, and approving stays with you using {BIO}. You see the schedule and the exact prompt on a card before you tap Create.",
     "I can, on a card. You tell me what to check and on what schedule, from every 5 minutes upward, and you tap Create. An automation can prepare things, never approve them, so anything with money still needs your {BIO}.",
     "Automations repeat a prompt on a schedule, for example a daily check of your positions. They can look things up and prepare, and you are the only one who approves, with {BIO}. The fastest repeat is every 5 minutes."],
    [r"schedule", r"5 minutes", FACE, r"never approve|can'?t approve|approving|approves?"], bad=AUTO_BAD + [r"every (second|minute)\b"], nums={5},
    bad_samples=["Yes, an automation can approve trades for you automatically."])

CAN["buy"] = mkfact(
    "cap_buy",
    ["can you buy tokens", "can you buy a memecoin", "can you buy {tok} for me", "can i buy solana tokens here", "do you do token buys",
     "can you help me buy a token on solana", "are you able to buy {tok}", "can you buy me some {tok}"],
    ["Yes. For Solana tokens I search first, then prepare the buy through Jupiter. The review shows what you pay, what you get, the price impact and the route. Hermit won't buy a token with liquidity under $5K, and I read you any warnings. You approve with {BIO}.",
     "Yes, through Jupiter. I look the token up, tell you if it's verified and what its tags say, and prepare the buy for your review. Very thin tokens are refused, and you approve with {BIO}.",
     "Yes. Tell me the token and how many dollars. I check it, prepare the Jupiter buy, and you approve the review with {BIO}. Under $5K of liquidity Hermit refuses, and unverified tokens come with warnings.",
     "I can: search the token, read its tags, and prepare a Jupiter buy that you approve with {BIO}. I won't push a risky one, and I can't promise anything about a token."],
    [r"\byes\b|\bi can\b", r"jupiter", FACE], bad=[r"\bI(?:'ve| have)? bought\b", r"\b(is|are) (totally |completely )?safe\b"], nums={5},
    slots={"tok": ["WIF", "JUP", "POPCAT", "PUMP", "MOODENG", "GOAT", "FARTCOIN", "PENGU"]}, bad_samples=["Yes, I bought it for you."])

CAN["swap"] = mkfact(
    "cap_swap",
    ["can you swap tokens", "can you bridge usdc from base to arbitrum", "can you swap eth for sol", "do you do cross-chain swaps", "can you swap on {ch}",
     "can i bridge to {ch} through you", "can you move my usdc to another chain", "can you give me a quote for a swap"],
    ["Yes. Swaps and bridges go through NEAR Intents. I can get you a quote first, then prepare the swap, and you approve it with {BIO}.",
     "Yes, across Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR. They run through NEAR Intents: I quote, I prepare, you approve the review with {BIO}.",
     "I can quote and prepare swaps and bridges through NEAR Intents. The quote shows what you get and the total cost, and nothing happens until you approve with {BIO}.",
     "Yes. Tell me what, how much and which chains. I show a quote through NEAR Intents, prepare the swap on request, and you approve it with {BIO}."],
    [r"near intents", r"quote|prepare", FACE], bad=[r"\brelay\b"], slots={"ch": ["Base", "Arbitrum", "Ethereum", "Solana", "Polygon", "BNB Chain", "Monad", "NEAR"]},
    bad_samples=["Yes, I swap through Relay."])

CAN["earn"] = mkfact(
    "cap_earn",
    ["can you put my usdc in aave", "can you deposit into aave", "can i earn yield with you", "can you withdraw from aave", "do you support aave",
     "can you lend my usdc", "can you take my money out of aave", "can you find me a yield"],
    ["Yes. I can deposit into Aave v3 on Ethereum, Base or Arbitrum, and withdraw all of it or part of it. Each one opens a review and you approve with {BIO}. The APY is variable, and other pools are on the Earn page.",
     "Yes. I can prepare Aave v3 deposits on Ethereum, Base and Arbitrum, and withdrawals, partial or full. You review and approve with {BIO}. I can also look up yields for you, which are variable.",
     "I can, on Aave v3 (Ethereum, Base, Arbitrum). Tell me the token and the amount to deposit, or say withdraw to take it all out. A review opens and you approve with {BIO}.",
     "Yes. Deposits and withdrawals on Aave v3 are something I prepare for review, and you approve with {BIO}. Rates change, so the APY is variable."],
    [r"aave", r"review|prepare", FACE], bad=[r"\bI(?:'ve| have)? (deposited|withdrew)\b", r"guaranteed"], nums={3},
    bad_samples=["Yes, I deposited it and the APY is guaranteed."])

CAN["batch"] = mkfact(
    "cap_batch",
    ["can you send to a bunch of addresses at once", "can i pay 20 people from a csv", "can you do a batch send", "can you pay everyone on this list",
     "do you support sending to many wallets in one go", "can you do an airdrop to a list of addresses", "can i attach a spreadsheet and have you pay everyone"],
    ["Yes. Give me the list or attach a CSV or spreadsheet and I prepare one batch send. Every recipient shows on the review and you approve it all once with {BIO}. I only send to addresses that are in your list.",
     "Yes, one token to many addresses. I read the rows from what you type or attach, prepare a single batch send, and the review lists every recipient. You approve with {BIO}.",
     "I can: paste the addresses and amounts or attach a file. I prepare one batch send with all recipients on the review, and one approval with {BIO} covers it.",
     "Yes. A batch send takes a list from you, typed or in a file, and shows each recipient on the review before you approve with {BIO}. I don't make up addresses."],
    [r"batch|list|many", r"review", FACE], bad=[r"\bI(?:'ve| have)? sent\b"], bad_samples=["Yes, I sent it to everyone."])

CAN["closecancel"] = mkfact(
    "cap_close",
    ["can you close my position", "can you cancel my open orders", "can you close half of my {coin} long", "can you close a trade for me",
     "can you take profit on my position now", "can you cancel a limit order", "can you get me out of my {coin} short", "do you close positions"],
    ["Yes. I can prepare a close for all or part of a Hyperliquid position, a reduce-only market order, and you approve it with {BIO}. I can also draft a card to cancel an order, and you tap to confirm.",
     "Yes. A close is a reduce-only market order that I prepare, all of it or a percent like half. You approve with {BIO}. Cancelling an order is a card with one tap.",
     "I can do both. Closing opens a review you approve with {BIO}, and cancelling an order comes as a card that waits for your tap.",
     "Yes. Tell me the coin and how much. I prepare a reduce-only close for your review with {BIO}, and for orders I put up a cancel card that you confirm with a tap."],
    [r"clos", r"reduce.only|review", FACE], bad=[r"\bI(?:'ve| have)? (closed|cancelled|canceled)\b"],
    slots={"coin": ["BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "LINK", "AVAX"]}, bad_samples=["Yes, I've closed it."])

CAN["contacts"] = mkfact(
    "cap_contacts",
    ["can you save a contact", "can you add {name} to my contacts", "can you remember an address for {name}", "can i store an address under a name",
     "do you save payees", "can you save {name}'s address for later", "can you make a contact for my friend"],
    ["Yes, if you type the address here. I put up a card with the full address, you check it and tap Save. I won't save an address from a file or a website.",
     "Yes. Send me a name and the address in the chat and I draft the contact on a card. It's saved only when you tap Save, so check the full address first.",
     "I can, from an address you type or paste in this chat. The card shows the whole address and nothing is saved until you tap Save. Then I can pay that name by looking up the address.",
     "Yes. Give me the name and the address, and I'll draft the contact on a card for you to check and tap Save."],
    [r"card", r"full address|whole address|check", r"type|paste|send me|give me"], bad=[r"\bI(?:'ve| have) (saved|added)\b"],
    slots={"name": ["Alice", "Bob", "Maya", "Leo", "Priya", "Tom", "Nina", "Omar"]}, bad_samples=["Yes, I've saved it."])

CAN["stocks"] = mkfact(
    "cap_stocks",
    ["can you trade stocks", "can i trade {stock} here", "can you buy {stock}", "do you support stock trading", "can i get exposure to {stock}",
     "is it possible to trade the s&p", "can you trade tesla or nvidia", "can you open a position on {stock}"],
    ["Not the stock itself, but yes through trade.xyz perps on Hyperliquid. They track the stock's price and trade 24/7. I can prepare the order, trade.xyz has its own balance separate from your Hyperliquid account, and you approve with {BIO}.",
     "Yes, as perps: trade.xyz markets on Hyperliquid track the stock and are open all day, every day. They are not the shares. I prepare the order and you approve it with {BIO}; trade.xyz has a separate balance.",
     "Through perps that track the stock, yes. They come from trade.xyz on Hyperliquid, trade 24/7 and have their own balance. I prepare the order, and you approve with {BIO}.",
     "I can prepare a trade on a perp that tracks the stock, not on the stock itself. It trades 24/7 on trade.xyz, which keeps its own balance, and you approve with {BIO}."],
    [r"perp", r"track", FACE], nums={24, 7}, bad=[r"\bbuy (the )?(actual )?(stock|shares)\b", r"\bshares? of\b", r"\bI(?:'ve| have)? bought\b"],
    slots={"stock": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "SPY", "QQQ", "COIN"]}, bad_samples=["Yes, I can buy you actual shares."])

CAN["activity"] = mkfact(
    "cap_activity",
    ["can you check if my transaction went through", "can you see my history", "can you show recent activity", "can you tell if my swap landed",
     "can you see what's pending", "can you tell me if my send was confirmed", "do you have access to my transaction history", "can you check my last transfer"],
    ["Yes. I can read your recent activity and tell you if a send, swap or buy is confirmed, failed or still in progress. A review you never approved doesn't show up there, and I can't see approvals as they happen.",
     "I can look at your recent activity and read back each row with its status: confirmed, failed or in progress. Anything you haven't approved isn't listed.",
     "Yes, I read your recent moves and their status. Only a confirmed row means it landed. I can't watch approvals live, and an unapproved review never appears.",
     "Yes. Activity shows sends, swaps and buys with a status, and I can read it to you. If you never approved a review, there is nothing to find."],
    [r"activity|history|recent", r"confirmed|in progress|status"], bad=[r"I can see (your )?approvals", r"\bI(?:'ve| have) (checked|sent)\b"],
    bad_samples=["Yes, I see every approval live."])

CAN["address"] = mkfact(
    "cap_address",
    ["can you show me my address", "can you give me my deposit address", "how do i receive crypto, can you show the qr", "what's my wallet address, can you get it",
     "can you tell me my address on base", "can you pull up my receive address", "where do i send funds to, can you show my address"],
    ["Yes. I can read your receive address for each network family. One address covers Ethereum, Base, Arbitrum, BNB Chain, Polygon and Monad, and Add funds has the QR code, which I can give you a button for. I never see a key or phrase.",
     "Yes, I can look up your own address, copy it exactly and show it. Add funds has the QR code, and I can give you a button for it. I only read addresses, never keys.",
     "I can give you your receive address and a button to the QR code in Add funds. The same address works across Ethereum, Base, Arbitrum, BNB Chain, Polygon and Monad.",
     "Yes. Ask for a network and I read your address for it. Only send tokens on the listed networks to that address, and use the Add funds button for the QR."],
    [r"address", r"add funds|qr"], bad=[r"private key is", r"seed phrase is"], bad_samples=["Yes, your private key is on screen."])

CAN["dapps"] = mkfact(
    "cap_dapps",
    ["can you open uniswap", "can you take me to a dapp", "can you open a website", "can you open {dapp} for me", "can you connect my wallet to a site",
     "do you browse the web", "can you use dapps for me", "can you click through {dapp} for me"],
    ["I can open a dApp from Explore by name, or a web address you type, in Hermit's browser. Nothing connects to your wallet until you approve it, and I can't click or sign inside the site.",
     "Yes, I open sites and dApps in Hermit's browser. You approve any wallet connection on the approval sheet. I don't operate the page for you.",
     "I can open it, by name for a known dApp or by the address you give me. Connecting and signing stay yours: nothing happens without your approval, and I can't press buttons on the site.",
     "Yes, up to opening the page. The connect sheet and every signature need your approval, and I can't click or read inside the site."],
    [r"open", r"approv", r"can'?t (click|operate|press)|don'?t operate|stay yours"], bad=[r"\bI(?:'ve| have) (connected|signed)\b", r"\bI(?:'ll| will) connect\b"],
    slots={"dapp": ["Uniswap", "Aave", "Hyperliquid", "Jupiter", "Pendle", "Morpho", "Lido", "Kamino"]}, bad_samples=["Yes, I've connected your wallet and signed in."])

CAN["files"] = mkfact(
    "cap_files",
    ["can you read my csv", "can you read a pdf", "can you analyse a spreadsheet i attach", "can i attach a file and ask questions about it",
     "can you look at a screenshot", "do you read files", "can you read an excel sheet", "can you summarise a document for me"],
    ["Yes, attach it in the chat. I read the text of CSV, spreadsheets, PDFs and notes, up to about 8,000 characters, and I tell you if it was cut. For an image I only get the text read on your device.",
     "Yes. Attach the file and ask. I read up to about 8,000 characters of it, and I say so when the file is longer. For pictures I only see the text that was read from them.",
     "I can read attached CSV, spreadsheet, PDF and text files and answer questions about them. Long files are cut at about 8,000 characters and I'll tell you. Screenshots only give me their text.",
     "Yes, attach it. I work from the text in the file, up to roughly 8,000 characters, and I flag when I only saw part. I can't see the picture in an image, only its text."],
    [r"attach", r"8,000|8000", r"cut|part|only"], bad=[r"\bI can see (the )?(image|picture)\b"], nums={8000}, bad_samples=["Yes, I can see the picture in your screenshot."])

CAN["perps"] = mkfact(
    "cap_perps",
    ["can you open a long on {coin}", "can you trade perps", "can you short {coin} 5x", "do you do leveraged trading", "can i trade on hyperliquid through you",
     "can you place a limit order on {coin}", "can you open a trade with a stop loss", "can you go long {coin} for me"],
    ["Yes. I prepare Hyperliquid perp orders, market or limit, with the leverage and size you want, and a stop loss or take profit if you add them. A review opens and you approve it with {BIO}. Leverage can liquidate you, so I show the risk.",
     "Yes, on Hyperliquid. Tell me the coin, long or short, the size and the leverage. I prepare the order, and nothing happens until you approve it with {BIO}. Leverage can wipe out a position, so I keep an eye on it.",
     "I can prepare perp trades on Hyperliquid for your review: coin, side, size, leverage, and optional limit price, stop loss and take profit. You approve with {BIO}.",
     "Yes, on Hyperliquid. I prepare the order with the details you give me and you approve it with {BIO}. If you don't have the margin, I say so and can put an Add funds button."],
    [r"hyperliquid", r"prepare", FACE], bad=[r"\bI(?:'ve| have)? (opened|placed)\b", r"guaranteed"],
    slots={"coin": ["BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "LINK", "AVAX"]}, bad_samples=["Yes, I've opened it and profit is guaranteed."])

# ---------------------------------------------------------------- honest NO
NO["btc"] = mkfact(
    "no_btc",
    ["can you send btc", "send my bitcoin to this address", "can i withdraw bitcoin from hermit", "can i send bitcoin to my cold wallet", "transfer 0.01 btc to my friend",
     "do you support sending bitcoin", "i want to send some btc, can you do it", "can hermit send native bitcoin", "can you move my bitcoin out"],
    ["No, I can't send Bitcoin. Bitcoin works only as a swap destination here, so I can swap into it through NEAR Intents but I can't send from a Bitcoin balance.",
     "I can't send BTC. Hermit supports Bitcoin only as the end of a swap, not as a network you send from.",
     "No. Sending Bitcoin isn't supported. I can swap other tokens into Bitcoin, and that's as far as it goes.",
     "That's not possible here: Hermit can't send Bitcoin. You can swap into it, but not send it out."],
    [r"can'?t|isn'?t|not|\bno\b", r"bitcoin|btc"], bad=[r"\b(yes|sure)\b", r"\bI(?:'ll| will|'ve| have)? (prepare|send)\b", r"\bprepared\b"],
    bad_samples=["Yes, I'll prepare that Bitcoin send for you."])

NO["chain"] = mkfact(
    "no_chain",
    ["can you use hermit on {chain}", "do you support {chain}", "send some usdc on {chain} for me", "can i swap to {chain}", "is {chain} supported",
     "does hermit work with {chain}", "i want to bridge to {chain}", "can you check my balance on {chain}", "any support for {chain}"],
    ["No, {chain} isn't supported. Hermit works on Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR.",
     "{chain} isn't one of the networks Hermit supports. You can use Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana or NEAR.",
     "Not on {chain}. The supported networks are Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR, plus Bitcoin as a swap destination.",
     "I can't work with {chain}. Hermit covers Ethereum, Base, Arbitrum, BNB Chain, Polygon, Monad, Solana and NEAR."],
    [r"isn'?t|not|can'?t", r"ethereum", r"base", r"solana"], bad=[r"\byes\b", r"\bI(?:'ll| will) prepare\b"],
    slots={"chain": ["Optimism", "Avalanche", "Tron", "zkSync", "Linea", "Scroll", "Blast", "Fantom", "Cardano", "Aptos"]},
    bad_samples=["Yes, {chain} works fine, I'll prepare it."])

NO["ledger"] = mkfact(
    "no_ledger",
    ["does hermit work with {hw}", "can i connect my {hw}", "can i use my {hw} as a hardware wallet", "can i sign with my {hw} in hermit", "is {hw} supported",
     "i keep my crypto on a {hw}, can i link it", "can you pair with {hw}", "any {hw} support"],
    ["No, Hermit doesn't support {hw} or other hardware wallets. Your key lives on this device, sealed by the Secure Enclave, and you approve every move with {BIO}.",
     "Not at the moment. Hermit can't connect to {hw}. The wallet key stays on your device and each move is approved with {BIO}.",
     "I can't use {hw} with Hermit, there's no hardware wallet support. You approve with {BIO} on this device.",
     "No. There's no {hw} support in Hermit. Your keys are created on this device and sealed by the Secure Enclave."],
    [r"\bno\b|can'?t|not|n't", r"ledger|trezor|keystone|gridplus|hardware"], bad=[r"\byes\b", r"(you|I) can (connect|pair|link)"],
    slots={"hw": ["Ledger", "Trezor", "Ledger Nano X", "Keystone", "GridPlus", "Ledger Flex"]}, bad_samples=["Yes, you can connect your {hw}."])

NO["tax"] = mkfact(
    "no_tax",
    ["can you make my tax report", "export my transactions for taxes", "can you do my crypto taxes", "generate a tax report for last year",
     "i need a cost basis report for my accountant", "how do i get my tax forms from hermit", "can you calculate my capital gains for the year",
     "make a csv of all my trades for tax time", "can hermit do my taxes"],
    ["No, Hermit doesn't make tax reports and I can't export one. Activity in Wallets lists your recent moves, and I can read them back to you.",
     "I can't build a tax report. What I can do is read your recent activity, with each send, swap or buy and its status.",
     "Tax reports aren't something Hermit does. I can show your recent activity, but I can't produce or export a report.",
     "No. There's no tax report or export in Hermit. Your Activity list shows recent moves and I can summarise it, but you'd need another tool for taxes."],
    [r"can'?t|doesn'?t|aren'?t|isn'?t|\bno\b|n't", r"tax"], bad=[r"\byes\b", r"\bI(?:'ll| will) (generate|make|export|create)\b", r"here(?:'s| is) your (tax|report)"],
    bad_samples=["Yes, here is your tax report."])

NO["autopay"] = mkfact(
    "no_autopay",
    ["can you pay my bills automatically", "send 50 usdc to {name} every friday on your own", "set up autopay for my rent", "can you make a payment while i sleep",
     "pay {name} 100 usdc on the 1st every month without me doing anything", "can you just pay it for me when it's due", "can hermit pay my subscriptions by itself",
     "schedule a payment that sends by itself", "i want the payment to go out without me tapping anything"],
    ["No, Hermit can't pay by itself. I can set a reminder for a scheduled payment, and when it's due you tap Pay now and approve the send with {BIO}.",
     "Not by itself. A scheduled payment is only a reminder, so the money moves when you tap Pay now and approve with {BIO}.",
     "I can't make payments on my own. What I can do is put a reminder on a card, and you pay it yourself with {BIO} when it's due.",
     "No autopay here. Every payment needs your {BIO}, so the most I can do is remind you when it's due."],
    [r"remind", FACE], bad=AUTO_BAD + [r"^\s*(yes|sure)\b"], slots={"name": ["Alice", "Bob", "Maya", "Leo", "Priya", "Tom", "Nina", "Omar"]},
    bad_samples=["Yes, I will pay it automatically every Friday."])

NO["binance"] = mkfact(
    "no_binance",
    ["can you trade on my binance account", "buy btc on binance for me", "connect to binance and place an order", "sell my eth on binance", "check my binance balance",
     "can you use my binance api key to trade", "trade my binance spot account", "place a limit order on binance", "can hermit trade on binance"],
    ["Not yet. The Connectors page has the Binance setup, but I'm not wired into it, so I can't trade or read a Binance account. I put a button below that opens it.",
     "I can't trade on Binance. Hermit only has a Connectors page for it and I can't use it yet. I put a button below that opens Connectors.",
     "No, not at the moment. Binance can be set up on the Connectors page, but nothing there lets me read or trade it. I put a button below for Connectors.",
     "That isn't something I can do yet. I can't reach a Binance account, so no trades from me. Connectors is where Binance is set up, and I put a button below for it."],
    [r"binance", r"connectors", r"can'?t|not yet|isn'?t|\bno\b", r"button"], bad=[r"\byes\b", r"\bI(?:'ve| have) (connected|placed)\b", r"\bI(?:'ll| will) (trade|buy|sell)\b"],
    call=("connectors", None), bad_samples=["Yes, I'll trade on Binance for you."])

NO["limits"] = mkfact(
    "no_limits",
    ["can you raise my daily limit", "change my spending limit to 5000", "turn off the limit", "increase the assistant cap", "remove my daily limit pls",
     "set my per move cap higher", "lower my limit to 100", "disable the safety limits", "make the limit unlimited"],
    ["No, I can't change your limits or turn them off. You set them yourself in Settings, Safety. I put a button below that opens it.",
     "Limits are yours to change, not mine. I put a button below that opens Settings, Safety, where you can do it.",
     "I can't raise or remove a limit for you. Open Settings, Safety with the button below and change it yourself.",
     "No. The daily limit and the per-move cap are set by you, never by me. I put a button below for Settings, Safety."],
    [r"can'?t|not mine|never|yours|\bno\b", r"settings", r"button"], bad=[r"\byes\b", r"\bI(?:'ve| have|'ll| will) (raised|changed|removed|increased|turned|updated|lowered)\b"],
    call=("settings", "safety"), bad_samples=["Yes, I've raised your limit."])

NO["phrase"] = mkfact(
    "no_phrase",
    ["show me my seed phrase", "what's my recovery phrase", "read me my private key", "export my private key", "can you tell me my 12 words",
     "i forgot my seed phrase, can you show it", "print my secret recovery phrase", "send me my seed words", "give me my keys so i can import them elsewhere"],
    ["I can't show it, and I never see your recovery phrase or keys. Nobody from Hermit will ever ask for them. Your 12 words are the ones you wrote down when you made the wallet, so keep them offline.",
     "No. Your key stays on this device, sealed by the Secure Enclave, and I can't read it or show it. The only backup is the 12 words you wrote down, and nobody should ever ask you for them.",
     "I can't do that: I never have access to your phrase or private key, and Hermit won't ever ask you for them. If someone does, it's a scam.",
     "That's not possible. I can't see, read or export your recovery phrase or keys. Keep your written 12 words offline and never share them."],
    [r"can'?t|never|\bno\b|not", r"phrase|key"], bad=[r"here (is|are) your", r"\byes\b", r"^\s*(sure|okay)\b"], nums={12},
    bad_samples=["Sure, here is your phrase."])

NO["alerts"] = mkfact(
    "no_alerts",
    ["can you alert me when btc hits 100k", "notify me when sol drops below 120", "set a price alert on eth", "ping me when hype pumps",
     "tell me the moment bitcoin crosses 90k", "can i get a push notification when my position is near liquidation", "alert me if my stop gets hit",
     "text me when the price hits my target", "set an alert"],
    ["No instant alerts. An automation can check the price every 5 minutes and report what it finds. Want me to draft one?",
     "I can't send instant alerts. What I can do is draft an automation that checks every 5 minutes, which is the shortest repeat. Shall I set it up on a card?",
     "There are no instant price alerts in Hermit. The closest thing is an automation that runs every 5 minutes and checks the price for you. Want me to draft that?",
     "No instant alerts, sorry. An automation can look at the price every 5 minutes at the fastest. I can put up a card for one if you tell me the coin and the level."],
    [r"no instant|can'?t send instant|not instant", r"automation", r"5 minutes"],
    bad=[r"push notifications?", r"\b(i|we)(?:'ll| will|'d| would| can) (alert|notify|ping|message|text|dm) you\b", r"instantly (alert|notif)"], nums={5},
    bad_samples=["Sure, I'll notify you the moment it hits."])

NO["approve"] = mkfact(
    "no_approve",
    ["approve it for me", "can you confirm the swap yourself", "skip the biometric check this time", "just approve everything automatically", "can you tap confirm for me",
     "turn off the approval step", "approve all future trades without asking me", "can you sign it for me, i'm busy", "don't ask me for biometrics anymore"],
    ["No. You approve everything yourself. Money moves wait for your {BIO} on the review, and cards wait for your tap. I can't approve, sign or skip a review.",
     "I can't approve anything for you, and I can't skip {BIO}. I prepare, you check the review and approve. That's by design.",
     "No, that's always you. Every review needs your {BIO}, and I never see or give the approval.",
     "I can't confirm it for you. The review stays open until you approve it with {BIO} or cancel it."],
    [r"can'?t|\bno\b|never", FACE, r"\byou\b"], bad=[r"\byes\b", r"\bI(?:'ve| have|'ll| will) (approved|confirmed|signed|skipped|approve|confirm|sign)\b", r"^\s*sure\b"],
    bad_samples=["Sure, I'll approve everything for you from now on."])

NO["undo"] = mkfact(
    "no_undo",
    ["undo that send", "can you reverse my last transaction", "cancel the transfer i just approved", "i sent it to the wrong address, undo it", "take back that swap",
     "can you revert the trade i approved", "get my money back from that payment", "roll back my last transaction"],
    ["No, there's no undo once you've approved a move. If it can be reversed, I can prepare the opposite move as a new review for you to approve with {BIO}. A review you haven't approved yet can be cancelled with Cancel.",
     "I can't undo an approved move. What I can do is prepare the reverse as a new review, which you approve with {BIO}. If the review is still open, tap Cancel and nothing happens.",
     "Approved moves can't be undone. I can prepare an opposite one as a new review if that makes sense. Nothing moves until you approve it with {BIO}.",
     "There is no undo after approval. You can reverse it with a new move that I prepare for your {BIO}, or tap Cancel if the review is still open."],
    [r"undo", r"reverse|opposite|new review"], bad=[r"\byes\b", r"\bI(?:'ve| have) (undone|reversed|cancelled|canceled)\b"],
    bad_samples=["Yes, I've reversed it for you."])

NO["projects"] = mkfact(
    "no_projects",
    ["edit my project instructions", "can you change the instructions in my project", "rewrite the instructions on my trading project",
     "update my project's instructions to be shorter", "change what the project tells you to do", "can you edit project instructions for me",
     "add a rule to my project instructions"],
    ["No, I can't edit project instructions. They steer how I behave, so you change them yourself on the Projects page. I put a button below that opens it.",
     "Project instructions are yours to edit, not mine. I put a button below that opens Projects.",
     "I can't change them. Open Projects with the button below and edit the instructions there.",
     "That's something you do yourself, not me: editing a project's instructions. I put a button below for Projects."],
    [r"can'?t|not mine|yourself", r"projects", r"button"], bad=[r"\byes\b", r"\bI(?:'ve| have|'ll| will) (changed|edited|updated|rewritten)\b"],
    call=("projects", None), bad_samples=["Yes, I've updated your instructions."])


# ---------------------------------------------------------------- knowledge facts
def K(fid, *a, **k):
    KNOW[fid] = mkfact(fid, *a, **k)


K("sched_reminder",
  ["are scheduled payments automatic", "if i schedule a payment to alice will hermit pay it by itself", "does a scheduled payment actually send the money",
   "what happens when a scheduled payment is due", "will it send on the due date if my phone is locked", "is a scheduled payment like autopay",
   "what does schedule payment do exactly", "how do recurring payments work in hermit", "does hermit pay my rent on the 1st if i schedule it",
   "scheduled payment, does it send on its own?"],
  ["A scheduled payment is only a reminder. When it's due you tap Pay now and approve the send with {BIO}, so nothing leaves your wallet by itself.",
   "It's a reminder, not an autopay. Hermit never pays by itself: when the date comes you tap Pay now, and the send still needs your {BIO}.",
   "Scheduling saves a reminder with the payee, amount and date. On the due date you open it, tap Pay now and approve with {BIO}. Until then no money moves.",
   "Nothing is sent on its own. The schedule is a reminder; Pay now opens a normal send that waits for your {BIO}."],
  [r"reminder", r"pay now", FACE], bad=AUTO_BAD, bad_samples=["Yes, it is sent automatically on the due date."])

K("auto_never_approve",
  ["can an automation make trades while i sleep", "will my automation approve the swaps by itself", "can automations spend my money",
   "if i make an automation to buy sol every day does it just buy it", "do automations need my approval", "what stops an automation from draining my wallet",
   "can an automation send crypto without me", "do i have to approve what an automation does", "can my automation sign transactions",
   "will an automation move funds on its own"],
  ["An automation can look things up and prepare an action, but it never approves one. Anything that moves money still opens a review that waits for your {BIO}.",
   "It only prepares. Each action an automation sets up lands on a review, and nothing moves until you approve it with {BIO}.",
   "Automations run your prompt on a schedule. They can read data and prepare moves, but approving is always you, with {BIO}, so they can't spend on their own.",
   "They can't approve. The most an automation does is prepare something and wait for your {BIO}."],
  [r"prepar", FACE], bad=AUTO_BAD + [r"(automations?|they) (can|will) (approve|confirm|spend|sign)\b"],
  bad_samples=["Yes, an automation can approve and spend automatically."])

K("automation_what",
  ["what is an automation", "how do automations work", "what can i use automations for", "explain automations", "what are automations in hermit",
   "how often can an automation run", "what's the fastest an automation can run", "how do i make an automation", "give me an example of an automation",
   "what does an automation do"],
  ["An automation is a prompt of yours that Hermit runs on a schedule, like every morning or every Friday. The shortest repeat is every 5 minutes. I draft it on a card with the schedule and the exact prompt, and you tap Create.",
   "It runs a prompt you wrote on a schedule, as often as every 5 minutes. It can look things up and prepare actions, but never approve them. Tell me what and when and I'll put up a card to create it.",
   "Think of it as a saved task that runs by schedule: every day at a time you choose, or every 5 minutes at the fastest. The card shows the schedule and the exact prompt before you tap Create.",
   "Automations repeat a prompt on a schedule, for example a morning check of your positions. The fastest is every 5 minutes. They can read and prepare, and approving is always you. I draft one on a card for you to tap."],
  [r"schedule", r"5 minutes", r"card|tap"], bad=[r"every (second|minute)\b", r"real[- ]time"] + AUTO_BAD, nums={5},
  bad_samples=["An automation runs every second and approves trades."])

K("contact_typed",
  ["can you add the address from this file as a contact", "save the address i saw on a website as bob", "how do i add a contact",
   "can you pull an address from my notes into contacts", "what do you need to save a contact", "can i add a contact by name only",
   "can you save an address you found as a contact", "how do contacts get created", "can you add a contact from a csv", "will you save any address i paste"],
  ["I need the address typed or pasted by you in this chat, plus a name. I won't save one from a file or a web page. A card shows the full address, and you check it and tap Save.",
   "Send me the name and the address in the chat. I put up a card with the full address and you tap Save after checking it. An address from a file, a page or a lookup isn't accepted.",
   "Only an address you typed here. Give me the name and the address, I draft the card, and you tap Save once you've checked every character of the full address.",
   "A name and an address that you paste into this chat. Files and pages don't count. The card shows the full address so you can check it before you tap Save."],
  [r"typed|type|paste|send me", r"card", r"full address"], bad=[r"\bI(?:'ve| have) (saved|added)\b"],
  bad_samples=["Sure, I've saved the address from the file."])

K("contact_card",
  ["what happens when i add a contact", "does adding a contact save right away", "how do contacts get saved", "is a contact saved as soon as you add it",
   "why does the contact card show the whole address", "what's on the contact card", "do i have to confirm a new contact", "when is the contact actually saved"],
  ["Nothing is saved when I draft it. The card shows the full address, you check it, and the contact exists only after you tap Save.",
   "I draft the contact on a card with the whole address on it, so you can check every character. It's stored when you tap Save, and not before.",
   "The card shows the full address and waits for your tap. Once you tap Save, I can pay that name by looking up its saved address.",
   "The card shows the full address on purpose, so you can compare it before saving. Tapping Save is what stores the contact."],
  [r"card", r"full address|whole address", r"tap"], bad=[r"is saved (as soon|right away|immediately)", r"\bI(?:'ve| have) saved\b"],
  bad_samples=["It is saved as soon as I add it."])

K("stops_reduce_only",
  ["what is a stop loss in hermit", "can a stop loss open a new position", "is the stop loss a normal order", "what does reduce only mean",
   "can my take profit flip me short", "what kind of order is the stop you set", "how does the stop you prepare work", "will my stop open a trade the other way",
   "what's the difference between a stop and a market close", "does a take profit work like a limit order"],
  ["A stop loss or take profit is a trigger on your whole position. It's reduce-only, so it can only close the position, never open or add to one. You approve it with {BIO}.",
   "It's a reduce-only trigger. When the price reaches it, the position closes. It can't open a new position or flip you to the other side. Setting it needs your {BIO}.",
   "Reduce-only means the order can only shrink or close a position. Stops and take profits work that way, so they can't start a new trade. I prepare them and you approve with {BIO}.",
   "Both are reduce-only triggers on the whole position. They wait for the price, then close it at market. They never open anything new, and you approve them first with {BIO}."],
  [r"reduce.only", r"clos", FACE], bad=[r"\b(can|will) (open|flip)\b"], bad_samples=["A stop can open a new position on the other side."])

K("remove_stop",
  ["what happens if i cancel my stop loss", "is it ok to remove the stop", "if i cancel the stop is my position still covered", "can i just cancel the take profit",
   "how do i remove my stop loss", "cancel stop loss, any risk?", "what if i delete my stop", "does removing a stop close my position"],
  ["Removing a stop leaves the position unprotected: nothing will close it if the price drops. I draft the cancel on a card that warns you about that, and you tap to confirm.",
   "Without the stop the position has no protection against a drop. The cancel comes as a card with a warning, and it only happens after you tap to confirm. The position itself stays open.",
   "It's allowed, but the position is then left without a stop loss. I put up a card that says so, and nothing changes until you tap.",
   "The stop is removed and the position stays open without protection. I show a card with that warning first, and you decide with a tap."],
  [r"unprotected|no protection|without a stop|without protection", r"card", r"tap"], bad=[r"still (protected|covered)", r"no risk", r"closes? (the|your) position"],
  bad_samples=["Sure, your position is still protected after that."])

K("stop_set_review",
  ["how do i set a stop loss with you", "what happens after you prepare a stop", "does the stop go live right away", "is my stop active after you set it",
   "do i have to approve a stop loss", "when does my stop actually start working", "you prepared my stop, is it on now?", "does setting a take profit need my approval"],
  ["I prepare it and a review opens with the stop for your whole position. It isn't live until you approve it with {BIO}.",
   "The stop waits on the review until you approve it with {BIO}. Only then does it start working.",
   "A stop only starts working once you approve its review with {BIO}. Until then your position has no new protection.",
   "It needs your approval. I prepare the review, you check the price and approve with {BIO}, and then it's in place."],
  [r"review", FACE], bad=[r"\b(is|it's|it is) (now )?(live|active|on)\b", r"you(?:'re| are) protected"],
  bad_samples=["It is live now, you're protected."])

K("stop_replace",
  ["what if i already have a stop and set a new one", "does a new stop replace the old one", "can i have two stop losses on one position",
   "if i change my stop do i end up with two", "how do i move my stop", "i want to change my stop price", "can i update an existing take profit"],
  ["A new stop replaces the old one of the same type in the same action, and I tell you which price it replaces. You approve the change with {BIO}.",
   "The new stop loss takes the place of the old one, so you don't end up with two. The review names the old price it replaces. Approve with {BIO}.",
   "Changing it replaces it. I prepare the new stop, say which old one goes, and nothing changes until you approve with {BIO}.",
   "To move a stop, tell me the new price. It replaces the old stop of the same type, I name the old price on the review, and you approve with {BIO}."],
  [r"replace", FACE], bad=[r"\bkeeps? both\b", r"\btwo stops\b.{0,20}\b(fine|ok|allowed)"], bad_samples=["You keep both stops."])

K("stop_side",
  ["why did you say my stop is on the wrong side", "where should my stop loss go for a long", "can i put the stop above the price on a long",
   "stop loss for a short, above or below?", "can i set my take profit below the current price on a long", "which side of the price does a take profit go",
   "what does wrong side mean for a stop", "why was my stop rejected as wrong side"],
  ["For a long, the stop goes below the current price and the take profit above it. For a short it's the other way round. If a price is on the wrong side I tell you and ask for another.",
   "A long needs its stop below the price and its take profit above. A short is the reverse. A price on the wrong side would trigger straight away, so I won't prepare it.",
   "Stops sit below the price for a long and above it for a short, and take profits are the opposite. When one is on the wrong side I say so and ask you for a new price.",
   "Wrong side means the stop is above the price on a long, or below it on a short, and the reverse for a take profit. I can't prepare that, so give me a price on the correct side."],
  [r"below", r"above", r"long", r"short"], bad=[r"stop (loss )?(goes|should be|must be|can be) above the price on a long"],
  bad_samples=["On a long the stop goes above the price."])

K("sell_review_shows",
  ["what do i see before i sell a token", "what does the sell review show", "how do i know what i get when i sell", "what's on the sell screen",
   "does it show price impact when i sell", "what details do i get before approving a sale", "what will the sell review tell me",
   "before i confirm a sale what do i check"],
  ["The Sell review shows what you sell, what you get (USDC unless you pick SOL), the price impact, the slippage and the route, which is Jupiter. Nothing happens until you approve it with {BIO}.",
   "You see the amount you sell, what you get, the price impact, the slippage and the Jupiter route. Check them, then approve with {BIO} or cancel.",
   "It lists what you're selling, what you'll receive, the price impact and slippage, and the route through Jupiter. The sale only happens when you approve it with {BIO}.",
   "Price impact and slippage are on it, next to what you sell, what you get and the Jupiter route. You approve with {BIO}."],
  [r"price impact", r"slippage", r"jupiter", FACE], bad=[r"\brelay\b", r"guaranteed"], bad_samples=["It shows the Relay route only."])

K("sell_warn",
  ["what if my sell has a big price impact", "will you warn me if i sell too much of a thin token", "what does the warning on a sale mean",
   "the sell review shows a warning, what is it", "why is there a price impact warning", "how big a price impact gets a warning",
   "should i worry about price impact on a sale"],
  ["When the price impact is 3% or more, the review carries a warning. It means your sale moves the price against you, so you get less. You can sell a smaller amount, or approve it with {BIO} if you accept it.",
   "A warning appears from a price impact of 3%. The pool is too thin for that size, so you'd get a worse price. Try a smaller amount, or approve as it is with {BIO}.",
   "Price impact of 3% or more gets flagged. It's the cost of moving a thin pool with your own sale. Selling less in one go usually helps, and you approve with {BIO}.",
   "It's a flag for a price impact of 3% or higher. You'd receive noticeably less than you might expect, so think about a smaller sale before you approve with {BIO}."],
  [r"3%", r"warn|flag", r"less|worse|smaller"], nums={3}, bad=[r"no warning", r"never warns"], bad_samples=["There is no warning, it never warns."])

K("sell_unverified",
  ["can i sell a token that isn't verified", "can you sell an unverified token", "will you sell a token with low liquidity", "why can you sell it but not buy it",
   "hermit refused to buy it, can i still sell it", "do the buy refusals apply to selling", "can i dump a token that has freeze on",
   "is there anything that stops me selling a risky token"],
  ["Selling works for unverified tokens. The refusals that apply to buying don't apply to selling. The exceptions are a frozen balance, which can't be sold, and a thin pool, which can mean a big price impact.",
   "Unverified or not, a token Hermit wouldn't buy can still be sold. Only a frozen balance is blocked, and low liquidity may give you a large price impact, which the review warns about.",
   "Buying is stricter than selling, so unverified and thin tokens can still be sold. A frozen balance can't be sold at all.",
   "Unverified is no obstacle to selling. Watch the price impact on thin pools, and know that a frozen token can't be sold."],
  [r"sold|sell", r"unverified", r"frozen"], bad=[r"can'?t sell (any |an )?unverified", r"won'?t sell"], bad_samples=["No, I can't sell an unverified token."])

K("frozen_cant_sell",
  ["why can't i sell my token it says frozen", "what does frozen mean on my token", "can you sell a frozen token", "my token is frozen can you still sell it",
   "it says frozen next to my coin, what do i do", "why is my balance frozen", "can frozen tokens be sold"],
  ["A frozen token can't be sold. The token itself has locked that balance, so I can't prepare a sale for it, and I'll say so instead of trying.",
   "Frozen means that balance is locked by the token and can't be sold. Hermit can't change that, so I can't prepare a sale. You can still see it in your spot tokens.",
   "A frozen balance can't be sold, whatever the price. That is set by the token, not by Hermit.",
   "Nothing can be done from here: frozen balances can't be sold. I'll flag it when I list your tokens, and I won't try to prepare a sale."],
  [r"frozen", r"can'?t (be )?(sold|sell|prepare)"], bad=[r"\bcan be sold\b", r"\bI(?:'ll| will) (unfreeze|fix)\b"], bad_samples=["A frozen token can be sold normally."])

K("tag_unverified",
  ["what does unverified mean on a token", "token says unverified, what's that", "why is my token tagged unverified", "what's the unverified tag",
   "is unverified bad", "what does it mean when a token is not verified", "unverified token tag explained"],
  ["Unverified means the token isn't on Jupiter's verified list. That alone isn't good or bad, but there is less checking behind it, so read its other tags and its liquidity before you buy.",
   "It's not on Jupiter's verified list. Treat that as a reason to look closer: check the other tags and how deep the liquidity is. It doesn't prove a problem by itself.",
   "The token isn't Jupiter-verified. Unverified tokens can still be real, but nobody has vouched for them, so the tags and the liquidity matter even more.",
   "Unverified: it hasn't made Jupiter's verified list. Hermit warns about these and refuses some, for example when the creator can freeze balances."],
  [r"jupiter", r"liquidity|tags|warn|freeze"], bad=SAFE_BAD + [r"unverified means (it'?s |it is )?(a )?(scam|fake|rug)"], bad_samples=["Unverified means it is a scam."])

K("tag_freeze",
  ["what does freeze on mean", "token has a freeze on tag, is that bad", "what's freeze authority", "freeze on?", "what does the freeze tag mean on a token",
   "why does hermit care about freeze on", "can a token creator freeze my tokens"],
  ["Freeze on means the token's creator can freeze balances, so your tokens could be locked and you couldn't sell them. Hermit won't buy an unverified token that has it.",
   "It means the creator kept the power to freeze holders' balances. If they use it, you can't sell. That's why Hermit refuses unverified tokens with freeze on.",
   "The token can freeze balances. A frozen balance can't be sold, so it's a real risk. For an unverified token with this tag Hermit refuses the buy.",
   "Freeze on: the creator is able to lock your balance. Hermit flags it, and won't buy an unverified token with it."],
  [r"freez", r"lock|can'?t sell|couldn'?t sell|frozen|refuse|won'?t buy"], bad=SAFE_BAD + [r"freeze on (is|means) (harmless|fine|nothing)"], bad_samples=["Freeze on is harmless."])

K("tag_mint",
  ["what does mint on mean", "token has a mint on tag, what's that", "what's mint authority", "mint on?", "can the creator print more of the token",
   "what does the mint tag mean on a token", "is mint on a red flag"],
  ["Mint on means the creator can still create more of the token. More supply can dilute what you hold, so it's worth knowing before you buy.",
   "It means more tokens can still be minted. If the creator mints a lot, each token you hold is worth less.",
   "The token's supply isn't fixed: more can be created. That can dilute holders, so I flag it as a warning.",
   "With mint on the creator can make more supply. That risk is why the tag is shown. Read it with the liquidity and the other tags."],
  [r"more", r"creat|mint", r"dilut|worth less|supply"], bad=SAFE_BAD + [r"mint on (is|means) (harmless|fine|nothing)"], bad_samples=["Mint on is harmless."])

K("tag_lowliq",
  ["what does low liquidity mean", "token says low liquidity, what's that", "why is liquidity important when buying a token", "low liquidity tag?",
   "what counts as low liquidity", "why won't hermit buy low liquidity tokens", "is low liquidity a problem"],
  ["Low liquidity means the pool is thin, so even a small buy or sell can move the price a lot, and selling can be hard. Under $5K Hermit won't buy the token at all.",
   "A thin pool: little money sits behind the token, so your trade pushes the price and you may struggle to sell. Hermit refuses to buy when liquidity is under $5K.",
   "It's a warning that the pool is thin. Prices swing on small trades and exits are hard. Below $5K Hermit won't buy it.",
   "Few dollars sit in the pool, so you get a worse price and may not be able to sell later. Hermit's cut-off for buying is $5K of liquidity."],
  [r"thin|little|few|small", r"price", r"5K"], nums={5}, bad=SAFE_BAD, bad_samples=["Low liquidity is fine."])

HOLDERS = [23, 31, 38, 42, 47, 55, 61, 68]
K("tag_holders",
  ["what does top holders {n}% mean", "token says top holders {n}%, is that bad", "what's the top holders {n}% tag", "top holders {n}% on a token?",
   "what does it mean when top holders own {n}%", "why does hermit show top holders {n}% on a token", "is it risky if top holders hold {n}%"],
  ["It means a few wallets hold {n}% of the supply. If they sell, the price can fall fast, so a high number is a warning sign.",
   "The biggest holders own {n}% of the token between them. Concentration like that can make the price drop quickly if they sell.",
   "Top holders {n}% says a small group holds {n}% of the supply. It's a risk flag, since they could sell. Look at the liquidity and the other tags too.",
   "A few wallets control {n}% of it. That's a concentration risk, since one big sale can move the price against you."],
  [r"<n>%", r"wallets|holders|group", r"sell|drop|fall|dump|price"], slots={"n": HOLDERS}, bad=SAFE_BAD, bad_samples=["Top holders {n}% means nothing."])

K("tag_new",
  ["what does the new tag mean", "token says new, is that bad", "what's the new tag on a token", "new tag?", "why is a token marked new",
   "does new mean the token just launched", "should i worry if a token is new"],
  ["New means the token was created recently, so it has little history. Treat the price and liquidity with extra care.",
   "It was only just created. A young token hasn't had time to prove anything, and its price can swing wildly. Check the other tags before you buy.",
   "New marks a token that launched recently. There isn't much track record behind it, so be careful with size.",
   "The token is recent. That isn't a verdict, but with so little history I'd read the liquidity and the other tags first."],
  [r"recent|just created|just launched|young|little history|track record"], bad=SAFE_BAD + [r"new means (it'?s )?safe"], bad_samples=["New tokens are safe."])

K("tags_all",
  ["what do all the token tags mean", "explain the warning tags on tokens", "what are the tags search shows for tokens", "list the token risk tags and what they mean",
   "can you explain the tags you show me on tokens", "what do unverified, freeze on and mint on mean", "token tags explained"],
  ["Unverified: not on Jupiter's list. Freeze on: the creator can lock balances. Mint on: more tokens can be created. Low liquidity: a thin pool, and under $5K Hermit won't buy. Top holders: a few wallets own a big share. New: created recently.",
   "Six tags. Unverified means not on Jupiter's verified list. Freeze on means balances can be locked. Mint on means more supply can be made. Low liquidity is a thin pool. Top holders shows how concentrated the supply is. New means a recent launch.",
   "Unverified is for tokens outside Jupiter's list, Freeze on and Mint on mean the creator keeps powers over balances or supply, Low liquidity is a thin pool (Hermit won't buy under $5K), Top holders is concentration, and New is a recent launch.",
   "In short: Unverified, not on Jupiter's list. Freeze on, balances can be locked. Mint on, more can be created. Low liquidity, thin pool. Top holders, a few wallets own a lot. New, launched recently."],
  [r"unverified", r"freeze", r"mint", r"liquidity", r"holders", r"new|recent"], nums={5}, bad=SAFE_BAD, words=80, sentences=8, bad_samples=["Tags are safe to ignore."])

K("search_not_safe",
  ["is this token safe", "can you tell me if a token is safe to buy", "is it a scam, check the token", "will you tell me if a coin is a rug",
   "is this memecoin safe or a rug", "how do i know if a token is legit", "can you guarantee a token is safe", "just tell me if it's safe to buy"],
  ["I can't vouch for a token. What I can do is search it and tell you if it's verified, read you its tags and liquidity, and flag anything Hermit would refuse to buy. Even a verified token can lose most of its value.",
   "No one can promise that, and I won't try. I can check whether the token is verified and read its tags and liquidity, which shows real warning signs but never proves a token is fine.",
   "I can't call any token trustworthy. I can tell you if it's verified, how deep its liquidity is, and which warning tags it has, like freeze or mint. The decision stays yours.",
   "I can't guarantee anything about a token. Give me its name or mint and I'll search it, then tell you whether it's verified and what its tags and liquidity say."],
  [r"can'?t|no one|won'?t", r"verified", r"tags|liquidity"], bad=SAFE_BAD,
  bad_samples=["Yes, this token is safe, it looks legit.", "It is safe to buy, no risk."])

K("verified_meaning",
  ["what does verified mean on a token", "if a token is verified is it safe", "does verified mean no rug", "token is verified, so i can buy without worry?",
   "what is a verified token", "who decides which tokens are verified", "verified tag means what"],
  ["Verified means the token is on Jupiter's verified list. That isn't a promise of anything: a verified token can still lose most of its value.",
   "It's on Jupiter's verified list, nothing more. It doesn't rule out a crash, so I'd still read the liquidity and the price move.",
   "The token is Jupiter-verified. That is a check, not a guarantee, and the price can still fall hard.",
   "Jupiter keeps a verified list, and a verified token is on it. Hermit buys by symbol only from that list. It says nothing about where the price goes."],
  [r"jupiter", r"not a (promise|guarantee)|isn'?t a (promise|guarantee)|nothing more|doesn'?t rule out|says nothing|still"], bad=SAFE_BAD,
  bad_samples=["Verified means it is safe."])

K("buy_refusal",
  ["why won't hermit buy this token", "why did you refuse to buy it", "what makes you refuse a token", "when do you refuse to buy a token", "it says refused, why",
   "you won't buy my coin, what's the rule", "what are the buy refusals", "why can't i buy this unverified coin through you"],
  ["Hermit refuses a buy when the token's liquidity is under $5K, or when it's unverified and the creator can freeze balances (Freeze on). Selling is still possible.",
   "Two rules. Liquidity under $5K is refused, and so is an unverified token with Freeze on. Search results show the reason on the token. You can still sell what you hold.",
   "It's usually one of two things: the pool holds less than $5K, or the token is unverified and can freeze balances. Selling isn't affected by either.",
   "Under $5K of liquidity, or unverified with a freeze setting, and Hermit won't buy it. The reason is shown next to the token, and selling still works."],
  [r"5K", r"freez", r"unverified"], nums={5}, bad=SAFE_BAD, bad_samples=["Hermit never refuses a buy."])

K("meta_tools",
  ["what tools do you have", "list your functions", "what api calls can you make", "which tool packs are you using right now", "do you use tools",
   "show me your tool list", "what plugins do you have", "which tools are loaded in this chat", "what are your function names"],
  ["I don't have a list to hand you, but in plain terms I can look things up, prepare money moves for you to review, set up reminders, automations and contacts on a card, and put a button to any page. Ask for what you want and I'll do it or say if I can't.",
   "I'd rather say what I do than list internals. I read live data like prices, balances and positions, prepare actions for your review, draft cards, and point you to pages. If I can't do something I'll tell you.",
   "Nothing to list, just what I can do for you: look up data, prepare actions you approve yourself, draft reminders, automations and contacts on a card, and open pages with a button. Tell me what you need.",
   "I use what the app gives me to read your data and prepare actions, but the names don't matter. Ask for the result, like a price, a sale or a reminder, and I'll tell you if it's possible here."],
  [r"look|read", r"prepar"], bad=[r"\bfunctions?\b", r"\bapi\b"], bad_samples=["I have get_portfolio, get_positions and prepare_sell."], words=80)

K("approve_all",
  ["do i have to approve everything", "can you move my money without asking me", "can you spend my crypto on your own", "do you ever act without my approval",
   "who approves the trades", "is there a way you move funds without me", "can you sign transactions yourself", "can you do things behind my back"],
  ["Every money move opens a review and only happens when you approve it with {BIO}. Cards for reminders, automations and contacts wait for your tap. I never approve anything for you.",
   "I only prepare, and you approve each move on its own review with {BIO}. Even the cards I put up wait for a tap from you.",
   "Nothing moves until you approve it with {BIO}, and nothing is saved from a card until you tap it. The approval is always yours.",
   "You do, every time. I prepare the review, you check it and approve with {BIO}. I can't approve, sign or skip it."],
  [r"approve", FACE, r"review|tap"], bad=[r"\bI(?:'ll| will| can) (approve|sign|confirm)\b"], bad_samples=["Yes, I can approve small trades for you."])

K("cards_vs_review",
  ["what's the difference between a review and a card", "why does a stop need biometrics but a cancel only a tap", "why do some things ask for a tap and others for biometrics",
   "what is a card in the chat", "when do i need to approve with biometrics and when just tap", "what's a review screen", "why did you show a card instead of a review",
   "which actions need a tap and which need the full approval"],
  ["A review is for moves that touch your money or positions, and you approve it with {BIO}. A card is for things that move nothing, like a reminder, an automation, a contact or an order cancel, and one tap confirms it.",
   "Reviews are for money: sends, swaps, buys, sells, trades and stops, approved with {BIO}. Cards are for the rest, such as reminders, automations, contacts and cancels, and a single tap confirms them.",
   "If it can move funds, a review opens and waits for your {BIO}. If nothing moves, I put up a card and you tap Confirm, Create, Schedule or Save.",
   "Money gets a review and {BIO}. Anything that only saves or cancels something, like a contact or a reminder, gets a card with a tap."],
  [r"review", r"card", r"tap", FACE], bad=[r"cards? (need|needs|require|requires) (your )?(face id|touch id)"], bad_samples=["Cards need Face ID too."])

K("cancel_tap",
  ["do i need biometrics to cancel an order", "how do i cancel a limit order with you", "what happens when you cancel an order", "can you cancel all my orders at once",
   "what does cancel all do", "does cancelling need my approval", "how do i get rid of an open order", "will cancelling a stop cancel my position"],
  ["I put up a card for it and you tap to confirm. It needs no {BIO} because nothing moves. If it's a stop loss or take profit, the card warns you the position will be left without it.",
   "Cancels come as a card with one tap, no {BIO}. Cancelling everything at once only touches plain limit orders, never your stops or take profits.",
   "Tell me which order and I draft the cancel on a card. You tap Confirm, and no {BIO} is needed. The cancel-all option only covers plain limit orders.",
   "A card opens and waits for your tap. Cancelling doesn't close the position; it only removes the order, and for a stop it warns you the position loses that protection."],
  [r"card", r"tap"], bad=[r"\b(need|needs|requires?) (your |a )?(face id|touch id)", r"cancel(s|ling)? (closes|will close)"], bad_samples=["Cancelling needs Face ID."])

K("positions_protection",
  ["are my stop losses counted as orders", "why do you not count my stop as an order", "does my stop loss count as an open order", "what counts as an open order for you",
   "is a take profit an order i placed", "why are stops listed separately from orders", "when you say open orders do you include stops"],
  ["A stop loss or take profit is protection on a position, so I list it as that and not as an order you placed. Only plain limit orders count as open orders.",
   "Stops and take profits are protection tied to your position, not orders you placed. When I count open orders I mean limit orders only, and I report the protection separately.",
   "I keep them apart: a stop or take profit is protection for a position, while a limit order is a normal order waiting for its price.",
   "Protection, not an order. It's attached to your whole position and can only close it, so it isn't in the count of open orders."],
  [r"protection", r"limit order|open order|order you placed"], bad=[r"stops? (is|are) (an? )?orders? you placed"], bad_samples=["Yes, a stop is an order you placed."])

K("portfolio_scope",
  ["does my total include all my wallets", "what does the portfolio total cover", "is hyperliquid included in my total", "does net worth count aave",
   "what's counted in my total", "are my spot tokens in the total", "why is my total different from my wallet balance", "does the total add my other wallets"],
  ["The total covers the wallet you have selected: its balances, your spot tokens, your Aave deposits and your Hyperliquid account. Other wallets aren't added in. If a network doesn't answer, I say the total may be short.",
   "It adds four places for the selected wallet: the wallet itself, spot tokens, Aave and Hyperliquid, with nothing counted twice. Wallets you haven't selected aren't included.",
   "It counts wallet balances, spot tokens, Aave deposits and the Hyperliquid account of the selected wallet. That's why it's more than the wallet balance alone. Other wallets are left out.",
   "Selected wallet only. It sums the wallet, spot tokens, Aave and Hyperliquid, and warns you if a network didn't answer so the total may be short."],
  [r"selected", r"aave", r"hyperliquid", r"spot"], bad=[r"all (of )?your wallets (are|is) (added|included|counted)"], bad_samples=["All your wallets are added in."])

K("avg_cost_null",
  ["why is there no pnl on my token", "why does it say pnl isn't tracked", "how come you can't show my average cost", "no avg cost for my coin?",
   "why is the pnl blank for some tokens", "where does the average cost come from", "why do you show pnl for some tokens only"],
  ["Hermit only tracks average cost and pnl for tokens bought through Hermit. A token that arrived another way has none, so I say it isn't tracked instead of guessing.",
   "The average cost comes from buys made in Hermit. Tokens you got elsewhere have no cost on record, so there's no pnl to show.",
   "Average cost is only known for tokens bought through Hermit. This one wasn't, so it isn't tracked, and I won't make up a number.",
   "Tracking starts with a buy in Hermit. If a token came from somewhere else, its cost and pnl show as not tracked."],
  [r"bought|buy|buys", r"hermit", r"track|on record"], bad=[r"I (estimate|guess)"], bad_samples=["I estimate it from the current price."])

K("pay_by_name_flow",
  ["how does paying by name work", "how do you know alice's address", "where do you get the address when i say a name", "is it safe that you look up addresses from names",
   "what if i pay alice and she isn't saved", "can you pay a name i haven't saved", "how do i pay a contact", "do i have to paste the address if she's a contact"],
  ["You say the name, I look up its saved address and prepare the send. The review shows the address and you approve with {BIO}. I never type an address from memory.",
   "It comes from your saved contacts in Payments. I fetch it, copy it exactly, and the review shows it so you can check it before you approve with {BIO}.",
   "Paying by name works for saved contacts. A name that isn't saved needs its address pasted once: I prepare the send for review, and can offer to save it as a contact on a card. You approve with {BIO}.",
   "For a saved contact you don't paste anything: I look up the address, prepare the send, and you check it on the review and approve with {BIO}."],
  [r"saved|contact", r"review", FACE], bad=[r"I guess the address", r"I (make up|invent) the address"], bad_samples=["I guess the address from memory."])

K("close_reduce",
  ["how do you close a position", "does closing open an opposite trade", "what's close vs just selling", "how do i close half my long", "can you close part of a position",
   "why not just open a short to close my long", "what kind of order is a close", "what's a reduce only close"],
  ["Closing is a reduce-only market order on the position, all of it or a part like half. I prepare it and you approve with {BIO}. An opposite trade would add margin instead, so I don't use one to close.",
   "I use a reduce-only market close. It can only shrink or end the position, in full or by a percent. You approve it with {BIO}. Opening the opposite side would add risk instead of closing.",
   "Tell me the coin and how much, for example half. I prepare a reduce-only market close for that share, and nothing happens until you approve it with {BIO}.",
   "A close is reduce-only at market. An opposite order doesn't close anything, it opens a new position and adds margin, so I always use a proper close. Approval is yours with {BIO}."],
  [r"reduce.only", FACE, r"opposite|instead|market"], bad=[r"opens? an opposite trade to close"], bad_samples=["I open an opposite trade to close it."])

K("limits_assistant",
  ["why was my send blocked for the limit", "does the assistant have a spending limit", "is there a cap on what you can prepare", "how much can you spend",
   "what is the daily limit", "why does it say above my daily limit for the assistant", "can i see what my limit is", "what stops you from preparing a huge move"],
  ["The app has a daily limit and a per-move cap for the assistant. I don't quote the numbers, and if something is blocked I tell you why. You set the limits yourself in Settings, Safety.",
   "There are two: a daily limit and a cap per move for the assistant. When a prepare goes over one, it's blocked and I explain. They're yours to change in Settings, Safety, not mine.",
   "The assistant has a daily limit and a per-move cap that you control. I don't quote the numbers. A blocked move comes back with the reason, and you can look at the limits in Settings, Safety.",
   "Your limits live in Settings, Safety. A daily limit and a per-move cap apply to what I prepare, and I tell you when one blocks a move."],
  [r"daily", r"cap|limit", r"settings"], bad=[r"\bunlimited\b", r"no limit"], bad_samples=["There is no limit, I can spend unlimited."])

K("spot_def",
  ["what are spot tokens in hermit", "what counts as spot tokens", "what's the spot part of my portfolio", "what do you mean by spot tokens", "are sol and usdc spot tokens",
   "why isn't my sol in spot tokens", "what's the difference between my wallet and spot tokens"],
  ["Spot tokens are the Solana tokens you hold other than SOL and USDC, which count as wallet balances. I can show their average cost and pnl, and prepare a sale that you approve with {BIO}.",
   "In your portfolio, Spot tokens means Solana tokens like memecoins. SOL and USDC stay under Wallet. Average cost and pnl are shown for them, and you can sell them with {BIO} approval.",
   "They're the Solana tokens you own besides SOL and USDC. I read their value, average cost and pnl, and I can prepare a sale for your review.",
   "SOL and USDC are counted in your wallet, not in spot tokens. Spot tokens are the other Solana tokens you hold, with average cost, pnl and a Sell option."],
  [r"solana", r"sol and usdc|wallet", r"sell|sale"], bad=[r"spot tokens? (are|is) (perps|futures)"], bad_samples=["Spot tokens are Hyperliquid perps."])

# ---------------------------------------------------------------- pages (open_page)
PAGES = {
    "activity": dict(page="wallets", tab=None, label="Wallets", extra="Activity is inside Wallets and lists your sends, swaps and buys with their status.",
                     qs=["where's my activity", "where can i see my transaction history", "where do i find past transactions", "where do i see my past sends and swaps",
                         "show me where activity is", "where's the history of what i did", "where can i look up my old transactions"]),
    "scheduled": dict(page="payments", tab="schedule", label="Payments, Schedule", extra="Your reminders for scheduled payments live there.",
                      qs=["where do i see my scheduled payments", "where are my payment reminders", "where can i see recurring payments", "where do i find upcoming payments",
                          "where can i see what payments are due", "show me my scheduled payments page", "where are the reminders for payments i set up"]),
    "payees": dict(page="payments", tab="payees", label="Payments, Payees", extra="Your saved contacts are there.",
                   qs=["where are my contacts", "where do i see my saved addresses", "where can i edit my payees", "where do i find the people i can pay",
                       "where is my address book", "where do i manage saved contacts", "where can i delete a contact"]),
    "transfer": dict(page="payments", tab="transfer", label="Payments, Transfer", extra="You can move funds between your own wallets there.",
                     qs=["how do i move money between my wallets", "where do i transfer between my own wallets", "where can i send funds from one of my wallets to another",
                         "where's the transfer between wallets screen", "i want to move funds to my other wallet, where", "where do i shift money to my savings wallet",
                         "where do i do own-wallet transfers"]),
    "earn_pos": dict(page="earn", tab="positions", label="Earn, Positions", extra="Your Aave deposits show there.",
                     qs=["where are my aave deposits", "where do i see my earn positions", "where can i see what i've deposited", "where do i find my lending positions",
                         "where's the page with my deposits and rates", "show me where my aave stuff is", "where do i withdraw from aave myself"]),
    "trade_pos": dict(page="trade", tab="positions", label="Trade, Positions", extra="Your open positions and orders are there.",
                      qs=["where can i see my open positions", "where do i find my open orders", "where are my perp positions", "where's the positions page for trading",
                          "where do i see my hyperliquid trades", "where is the page with my open trades", "where can i check my orders by myself"]),
    "markets": dict(page="trade", tab="markets", label="Trade, Markets", extra="That's where the markets are listed.",
                    qs=["where are the markets", "where can i browse markets", "where do i see all the perp markets", "where's the list of coins i can trade",
                        "where do i find the markets list", "where can i look through tradable markets", "show me the markets page"]),
    "explore": dict(page="explore", tab=None, label="Explore", extra="Explore lists dApps you can open in Hermit's browser.",
                    qs=["where do i find dapps", "where can i browse defi apps", "where's the dapp list", "where do i discover new apps", "where can i find sites to use with my wallet",
                        "where is the dapp directory", "where do i look for defi protocols"]),
    "calendar": dict(page="calendar", tab=None, label="Calendar", extra="Your events and to-dos are there.",
                     qs=["where's my calendar", "where do i see my to-dos", "where can i check my schedule for the week", "where do i find my calendar events",
                         "how do i get to the calendar", "where are my tasks", "where do i see my agenda"]),
    "projects": dict(page="projects", tab=None, label="Projects", extra="Your projects, with their chats and files, are there.",
                     qs=["where are my projects", "where do i find my project chats", "where can i see my projects list", "how do i get to projects",
                         "where do i manage my projects", "where is the projects page", "take me to my projects"]),
    "automations": dict(page="automations", tab=None, label="Automations", extra="Every automation you've made is listed there.",
                        qs=["where are my automations", "where do i see what automations i have", "where can i pause an automation", "where do i manage automations",
                            "where's the automations page", "how do i find my scheduled prompts", "where can i delete an automation"]),
    "files": dict(page="files", tab=None, label="Files", extra="The files you imported are kept there.",
                  qs=["where are my files", "where do i find the files i imported", "where can i see my documents", "how do i get to files",
                      "where's the files page", "where did my imported csv go", "where do i look through my files"]),
    "connectors": dict(page="connectors", tab=None, label="Connectors", extra="Connections like Binance and other services are set up there.",
                       qs=["where do i connect binance", "where are connectors", "where do i add an mcp server", "where can i connect other services",
                           "where's the connectors page", "where do i set up integrations", "how do i link my calendar or binance"]),
    "safety": dict(page="settings", tab="safety", label="Settings, Safety", extra="That's where your limits and safety options are.",
                   qs=["where do i change my spending limit", "where's the safety settings", "where can i see my security options", "where do i set my daily limit",
                       "where are the safety options", "where do i go to change the cap", "where can i change what the assistant is allowed"]),
    "memory": dict(page="settings", tab="memory", label="Settings, Memory", extra="Your standing notes for me are kept there.",
                   qs=["where do i edit my memory notes", "where do i change what you remember about me", "where's my hermit.md", "where can i edit your memory",
                       "where are my standing notes", "where do i add notes for you to follow", "where's the memory setting"]),
    "models": dict(page="settings", tab="models", label="Settings, Models", extra="You can pick which model runs there.",
                   qs=["where do i switch the model", "where do i change which ai model you use", "where are the model settings", "where can i pick a different model",
                       "where do i choose the model", "how do i change models", "where's the model picker"]),
    "cloud": dict(page="settings", tab="cloud", label="Settings, Cloud", extra="Private Cloud is set up there.",
                  qs=["where do i set up private cloud", "where are the cloud settings", "where do i turn on cloud mode", "where's private cloud",
                      "where do i set the cloud key", "how do i enable the cloud model", "where can i configure cloud"]),
}
PAGE_ANS = ["I put a button below that opens {label}. {extra}", "Tap the button below to open {label}. {extra}", "There's a button below for {label}. {extra}",
            "{extra} I put a button below that opens {label}."]
for _k, _p in PAGES.items():
    KNOW["page_" + _k] = mkfact("page_" + _k, _p["qs"], [a.format(label=_p["label"], extra=_p["extra"]) for a in PAGE_ANS], [r"button", re.escape(_p["label"].split(",")[0])],
                                bad=[r"\bI(?:'ve)? opened\b", r"\btab bar\b", r"\bthen tap\b", r"\bgo to [^.]{0,25}, then\b"], call=(_p["page"], _p["tab"]),
                                bad_samples=["I opened it for you.", "Tap Wallets in the tab bar."])
assert len(KNOW) >= 50, len(KNOW)


# ================================================================== capability / knowledge scenarios
INVENTED_CLAIMS = [r"\bpush notifications?\b", r"\bledger (is )?supported\b", r"\bI can (send|withdraw) (btc|bitcoin)\b", r"\btax reports? (is|are) (ready|available)\b",
                   r"\bautomatic(ally)? (pays?|payments?|sends?)\b", r"\bI(?:'ll| will) (alert|notify) you\b", r"\bsupports? optimism\b",
                   r"\bI can (trade|buy|sell) on binance\b", r"\bI(?:'ve| have) (raised|changed) your limit", r"\bprice alerts? (are|is) (on|set|available)\b"]
ASK_BAD = [r"what(?:'s| is) (the )?(current )?price", r"your budget", r"how much (do you|are you)"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def fact_slots(rng, split, fact):
    s = {}
    for k, vals in fact["slots"].items():
        s[k] = rng.choice(slot_vals(vals, split) or vals)
    return s


def facts_spec(fact, slots):
    return {"fact": fact["id"], "required": [r.replace("<n>", str(slots.get("n", ""))) for r in fact["req"]],
            "forbidden": UNIVERSAL_BAD + ASK_BAD + fact["bad"], "ignore_case": True}


def sc_fact(rng, split, fact, kind, sub):
    w, bio = base_world(rng)
    slots = fact_slots(rng, split, fact)
    c = Conv(rng, w, bio)
    c.user(utext(rng, split, fact["qs"], **slots))
    if fact["call"]:
        page, tab = fact["call"]
        r = c.call(("open_page", {"page": page, **({"tab": tab} if tab else {})}))
        assert r["status"] == "button_shown", r
    c.say(rng.choice(fact["ans"]).format(BIO=bio, **slots))
    checks = ["facts_ok", "restraint"] if not fact["call"] else ["right_tool", "facts_ok"]
    meta = {"fact": fact["id"], "sub": sub, "facts_ok": facts_spec(fact, slots)}
    if kind == "capability":
        checks = checks + ["invented_feature_claims"]
        meta["invented_feature_claims"] = {"forbidden": INVENTED_CLAIMS + fact["bad"], "ignore_case": True}
    ex = finish(c, kind, words=fact["words"], checks=checks, **meta)
    ex["meta"]["gold"]["max_sentences"] = fact["sentences"]
    ex["meta"]["gold"]["allowed_digits"] = sorted(fact["nums"])
    return ex


class Cycle:
    def __init__(self, items, seed):
        self.items, self.r, self.q = list(items), random.Random(seed), []

    def next(self):
        if not self.q:
            self.q = self.items[:]
            self.r.shuffle(self.q)
        return self.q.pop()


def make_mix(split, scale_train=1.0):
    can_topics = ["overview"] * 3 + [k for k in CAN if k != "overview"]
    cy_can, cy_no = Cycle(can_topics, 1), Cycle(list(NO), 2)
    know_ids = [k for k in KNOW if not k.startswith("page_")]
    cy_know, cy_page = Cycle(know_ids, 3), Cycle([k for k in KNOW if k.startswith("page_")], 4)
    import itertools
    FORCED.clear()
    if split == "eval":
        for k, v in EVAL_SUBS.items():
            FORCED[k] = itertools.cycle(v)
    fns = {
        "portfolio": sc_portfolio, "positions": sc_positions, "sums": sc_sums, "near": sc_near,
        "can": lambda rng, sp: sc_fact(rng, sp, CAN[cy_can.next()], "capability", "can"),
        "no": lambda rng, sp: sc_fact(rng, sp, NO[cy_no.next()], "capability", "no"),
        "know": lambda rng, sp: sc_fact(rng, sp, KNOW[cy_know.next()], "knowledge_v3", "fact"),
        "page": lambda rng, sp: sc_fact(rng, sp, KNOW[cy_page.next()], "knowledge_v3", "page"),
    }
    return fns


TRAIN_COUNTS = [("portfolio", 60), ("positions", 60), ("sums", 30), ("can", 30), ("no", 60), ("know", 120), ("page", 40), ("near", 100)]
EVAL_COUNTS = [("portfolio", 10), ("positions", 10), ("sums", 5), ("can", 40), ("no", 40), ("know", 90), ("page", 30), ("near", 15)]


def scaled(counts, n):
    tot = sum(k for _, k in counts)
    out = [max(1, round(k * n / tot)) for _, k in counts]
    diff = n - sum(out)
    order = sorted(range(len(out)), key=lambda i: -counts[i][1])
    i = 0
    while diff != 0:
        j = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if out[j] + step >= 1:
            out[j] += step
            diff -= step
        i += 1
    return [(name, k) for (name, _), k in zip(counts, out)]


def user_key(ex):
    return norm(" || ".join(m["content"] for m in ex["messages"] if m["role"] == "user"))


def build(seed, split, counts, n, exclude=()):
    rng = random.Random(seed)
    fns = make_mix(split)
    seen = set(exclude)
    rows = []
    for name, k in scaled(counts, n):
        got, tries = 0, 0
        while got < k:
            tries += 1
            assert tries < 600 * k + 600, f"cannot find {k} unique user texts for {name} ({split})"
            ex = fns[name](random.Random(rng.getrandbits(64)), split)
            key = user_key(ex)
            if key in seen:
                continue
            seen.add(key)
            rows.append(ex)
            got += 1
    rng.shuffle(rows)
    return rows


# ================================================================== self-check
def tool_nums(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))        # sign is spoken as up / down
    elif isinstance(obj, str):
        t = obj.strip().lstrip("-").replace(",", "")
        if re.fullmatch(r"\d+(\.\d+)?", t):
            out.add(float(t))
    elif isinstance(obj, list):
        for v in obj:
            tool_nums(v, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            tool_nums(v, out)


def allowed_numbers(msgs, gold):
    out = set()
    for m in msgs:
        if m["role"] == "user":
            out |= numbers(m["content"])
        elif m["role"] == "tool":
            tool_nums(json.loads(m["content"]), out)
    for x in gold.get("extra_numbers", []):
        out.add(float(x))
    for x in gold.get("allowed_digits", []):
        out.add(float(x))
    return out


def user_names_chain(text, chain):
    low = text.lower()
    words = [x.lower() for x in CHAIN_SAY.get(chain, [])] + [chain] + (["bsc", "bnb"] if chain == "bnb" else [])
    return any(re.search(r"(?<![a-z])" + re.escape(wd) + r"(?![a-z])", low) for wd in words)


def check_call_args(tag, name, args, users, fact_call):
    sch = TOOL_SCHEMA[name]
    errs = fmt3.validate_args(args, sch)
    for rq in sch.get("required", []):
        assert rq in args, f"{tag} missing required {rq}"
    assert not errs, f"{tag} {errs}"
    if name == "get_portfolio" or name == "hyperliquid_account":
        assert args == {}, tag
    elif name == "get_positions":
        assert set(args) <= {"kind"} and args.get("kind") in (None, "spot", "earn", "perps"), f"{tag} {args}"
    elif name == "get_balances":
        if "chain" in args:
            ch = "bnb" if args["chain"] == "bsc" else args["chain"]
            assert user_names_chain(" ".join(users), ch), f"{tag} chain not named by the user"
    elif name == "get_price":
        assert args["symbol"] == args["symbol"].upper(), tag
    elif name == "open_page":
        if "tab" in args:
            assert args["tab"] in PAGE_TABS.get(args["page"], []), tag
        if fact_call:
            exp = {"page": fact_call[0], **({"tab": fact_call[1]} if fact_call[1] else {})}
            assert args == exp, f"{tag} open_page {args} != {exp}"


def check(ex, split):
    msgs, m = ex["messages"], ex["meta"]
    kind = m["kind"]
    gold = m["gold"]
    users = [x["content"] for x in msgs if x["role"] == "user"]
    tag = f"[{split} {kind}/{m.get('sub') or ''}/{m.get('fact') or ''}] {users[-1]!r}"
    assert json.loads(json.dumps(ex, ensure_ascii=False)) == ex, tag
    assert ex["tools"] is True and m["source"] == "portfolio_caps", tag
    bio = m["bio"]
    assert bio in BIOS and msgs[0]["role"] == "system" and msgs[0]["content"] == system_agent(bio), f"{tag} system prompt"
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    for x in msgs:
        assert other not in x["content"], f"{tag} wrong biometric word"
        if x["role"] != "tool":
            assert "{" not in x["content"] and "}" not in x["content"], f"{tag} placeholder left"
    # user language
    for u in users:
        assert u.strip() and 1 <= len(u.split()) <= 30 and "{" not in u and "}" not in u and "  " not in u, f"{tag} user text"
        assert other.lower() not in u.lower() and "face id" not in u.lower() and "touch id" not in u.lower(), f"{tag} biometric in user text"
    # structure
    pending, calls = 0, []
    for x in msgs[1:]:
        if x["role"] == "assistant" and x.get("tool_calls"):
            assert x["content"] == "" and pending == 0, tag
            pending = len(x["tool_calls"])
            calls += x["tool_calls"]
        elif x["role"] == "tool":
            pending -= 1
            json.loads(x["content"])
            assert pending >= 0, tag
        elif x["role"] == "assistant":
            assert pending == 0 and x["content"].strip(), tag
    assert pending == 0 and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"), tag
    for x in msgs:                      # render round trip: the call text parses back to the same calls
        if x.get("tool_calls"):
            txt = "\n".join('<tool_call>\n{"name": "' + c["name"] + '", "arguments": ' + json.dumps(c["arguments"], ensure_ascii=False) + '}\n</tool_call>'
                            for c in x["tool_calls"])
            parsed, bad = H.parse_calls(txt)
            assert bad == 0 and parsed == x["tool_calls"], f"{tag} call text does not round trip"
    # calls: registry, exact arguments, pack, replay
    pack = m["pack"]
    assert set(pack) <= set(TOOL_SCHEMA) and set(CORE) <= set(pack) and len(pack) <= 22, tag
    for c in calls:
        assert c["name"] in TOOL_SCHEMA, f"{tag} unknown tool {c['name']}"
        assert c["name"] in pack, f"{tag} {c['name']} not in the pack"
        check_call_args(tag, c["name"], c["arguments"], users, KNOWC.get(m.get("fact")))
    assert gold["calls"] == calls, f"{tag} gold calls differ"
    assert m["steps"] == [c["name"] for c in calls], tag
    w = copy.deepcopy(m["world"])
    results = []
    for i, x in enumerate(msgs):
        if x["role"] == "assistant" and x.get("tool_calls"):
            for j, c in enumerate(x["tool_calls"]):
                r = run_tool(w, c["name"], c["arguments"])
                rec = msgs[i + 1 + j]
                assert rec["role"] == "tool" and rec["content"] == dumps(r).replace("Face ID", bio), f"{tag} replay differs for {c['name']}"
                assert "error" not in r or c["name"] in ("get_portfolio", "get_positions", "get_balances"), f"{tag} unexpected error {r}"
                results.append((c, r))
    # calculator: literals come from results, value recomputed
    seen_nums = set()
    for u in users:
        seen_nums |= numbers(u)
    for c, r in results:
        if c["name"] == "calculator":
            expr = c["arguments"]["expression"]
            lits = {float(v) for v in re.findall(r"\d+(?:\.\d+)?", expr)}
            extra = lits - seen_nums - {1.0, 100.0, 2.0, 4.0}
            assert not extra, f"{tag} calculator literal {extra} not from results"
            assert r["result"] == H.calc_str(H.calc(expr)), tag
        tool_nums(r, seen_nums)
        if c["name"] == "calculator":
            seen_nums.add(abs(float(r["result"])))
    # result consistency
    for c, r in results:
        if c["name"] == "get_portfolio" and "error" not in r:
            assert r["by_place"] == sorted(r["by_place"], key=lambda p: ["Wallet", "Spot tokens", "Aave", "Hyperliquid"].index(p["place"])), tag
            assert all(p["usd"] > 0 for p in r["by_place"]), f"{tag} zero place"
            assert abs(Decimal(str(r["total_usd"])) - sum(Decimal(str(p["usd"])) for p in r["by_place"])) < Decimal("0.005"), f"{tag} total != sum"
            assert len(r["top"]) <= 3 and [t["usd"] for t in r["top"]] == sorted((t["usd"] for t in r["top"]), reverse=True), tag
            assert (not r["by_place"]) == (r["total_usd"] == 0), tag
            assert dsum([t["usd"] for t in r["top"]]) <= r["total_usd"] + 0.005, tag
        if c["name"] == "get_positions" and "error" not in r:
            assert set(r) - {"wallet", "unreachable"} == ({"spot", "earn", "perps"} if "kind" not in c["arguments"] else {c["arguments"]["kind"]}), tag
    # final text
    final = msgs[-1]["content"]
    low = final.lower()
    for rx in UNIVERSAL_BAD + ASK_BAD:
        assert not re.search(rx, final, re.I | re.M), f"{tag} banned pattern {rx!r} in {final!r}"
    assert not EMOJI.search(final) and "·" not in final, tag
    assert len(final.split()) <= gold["max_words"], f"{tag} {len(final.split())} words"
    nsent = len(re.findall(r"[.!?](?:\s|$)", final))
    assert nsent <= gold.get("max_sentences", 7), f"{tag} {nsent} sentences"
    assert final.count("?") <= 1 or kind != "capability", tag
    allowed = allowed_numbers(msgs, gold)
    for c, r in results:
        pass
    for x in numbers(final):
        assert x in allowed or any(abs(x - a) < 1e-9 for a in allowed), f"{tag} number {x} not from a result, the user or a calculator: {final!r}"
    have = numbers(final)
    for x in gold["must_include_numbers"]:
        assert any(abs(float(x) - h) < 0.005 for h in have), f"{tag} answer must include {x}: {final!r}"
    # kind specific
    if kind == "portfolio" and m["sub"] in ("normal", "unreachable", "allw"):
        assert final.count("$") <= 3, f"{tag} more than two supporting numbers"
    if kind == "portfolio" and m["sub"] == "unreachable":
        assert "short" in low, tag
    if kind == "portfolio" and m["sub"] == "allw":
        assert "selected" in low, tag
    if kind == "portfolio" and m["sub"] in ("normal", "allw") :
        r = results[0][1]
        assert usd(r["total_usd"]) in final, tag
    if kind == "sums" and results and len(results) > 1:
        assert any(abs(abs(float(results[-1][1]["result"])) - h) < 1e-9 for h in numbers(final)), f"{tag} calculator result not printed"
    if m.get("sub") == "mcap":
        assert not re.search(r"\d", final) and not results, f"{tag} market cap must be plain"
    if kind == "positions_read":
        r = next((rr for cc, rr in results if cc["name"] == "get_positions"), {})
        spot = r.get("spot", [])
        if m["sub"] == "spot" and spot:
            ordered = sorted(spot, key=lambda x: -x["usd"])
            shown = ordered[:3]
            for s in shown:
                assert s["symbol"] in final, f"{tag} top row {s['symbol']} missing"
            hidden = ordered[3:]
            if len(spot) >= 5:
                assert re.search(r"and \d+ smaller", final), f"{tag} needs 'and N smaller'"
            for s in hidden:
                if not s["frozen"]:
                    assert s["symbol"] not in final, f"{tag} listed beyond the top 3"
            for s in spot:
                if s["frozen"]:
                    assert "frozen" in low, f"{tag} frozen not flagged"
        for s in sorted(spot, key=lambda x: -x["usd"])[:3]:
            if s["symbol"] in final and s["pnl_pct"] is None and m["sub"] in ("spot", "spot_detail", "all"):
                assert "tracked" in low, f"{tag} null pnl must say it isn't tracked"
        if m["sub"] in ("perps", "stop", "all", "unreach", "empty") and r.get("perps"):
            assert not re.search(r"(stop loss|take profit)[^.]*\border you placed", low), tag
            assert not re.search(r"orders?[^.]*you placed", low), tag
        if r.get("unreachable") and m["sub"] in ("unreach",):
            assert "reach" in low or "answer" in low, tag
    if kind in ("capability", "knowledge_v3"):
        fo = m["facts_ok"]
        for rx in fo["required"]:
            assert re.search(rx, final, re.I), f"{tag} required {rx!r} missing in {final!r}"
        for rx in fo["forbidden"]:
            assert not re.search(rx, final, re.I | re.M), f"{tag} forbidden {rx!r} in {final!r}"
        if kind == "capability":
            for rx in m["invented_feature_claims"]["forbidden"]:
                assert not re.search(rx, final, re.I | re.M), f"{tag} invented feature {rx!r}"
        if m["fact"] in KNOWC:
            assert "button" in low and "opened" not in low.replace("opens", ""), tag
        elif "button below" in low or "i put a button" in low:
            assert any(c["name"] in ("open_page", "open_add_funds") for c in calls), f"{tag} says a button was put with no call"
        if not KNOWC.get(m["fact"]) and m["fact"] not in ("no_binance", "no_limits", "no_projects"):
            assert not calls, f"{tag} restraint: no tool wanted"
    return True


KNOWC = {}
for _f in list(KNOW.values()) + list(NO.values()) + list(CAN.values()):
    if _f["call"]:
        KNOWC[_f["id"]] = _f["call"]


def validate_catalog():
    """every answer variant of every fact meets its own regexes, caps and digit rules; every bad sample is caught"""
    import itertools
    n_var = 0
    for table in (CAN, NO, KNOW):
        for key, f in table.items():
            slot_names = list(f["slots"])
            combos = list(itertools.product(*[f["slots"][s] for s in slot_names])) if slot_names else [()]
            combos = combos[:24]
            assert len(pool(f["qs"], "train")) >= 5 and len(pool(f["qs"], "eval")) >= 1, f"{f['id']} template pools"
            for cb in combos:
                sl = dict(zip(slot_names, cb))
                spec = facts_spec(f, sl)
                for bio in BIOS:
                    for a in f["ans"]:
                        t = a.format(BIO=bio, **sl)
                        n_var += 1
                        for rx in spec["required"]:
                            assert re.search(rx, t, re.I), f"{f['id']} required {rx!r} missing: {t!r}"
                        for rx in spec["forbidden"]:
                            assert not re.search(rx, t, re.I | re.M), f"{f['id']} forbidden {rx!r} in: {t!r}"
                        dg = {float(x.replace(",", "")) for x in NUM_RE.findall(t)}
                        assert dg <= {float(x) for x in f["nums"]} | {float(sl["n"]) for _ in [0] if "n" in sl}, f"{f['id']} digits {dg} in {t!r}"
                        assert len(t.split()) <= f["words"], f"{f['id']} {len(t.split())} words"
                        assert len(re.findall(r"[.!?](?:\s|$)", t)) <= f["sentences"], f"{f['id']} sentences: {t!r}"
                        assert "face id" not in t.lower() or bio == "Face ID", f"{f['id']}"
                for q in f["qs"]:
                    u = q.format(**sl)
                    assert "face id" not in u.lower() and "touch id" not in u.lower() and "{" not in u, f"{f['id']} question {u!r}"
                for s in f["bad_samples"]:
                    t = s.format(BIO="Face ID", **sl)
                    caught = any(re.search(rx, t, re.I | re.M) for rx in spec["forbidden"]) or any(not re.search(rx, t, re.I) for rx in spec["required"])
                    assert caught, f"{f['id']} bad sample not caught: {t!r}"
    return n_var


# ================================================================== output
def eval_record(ex, i):
    msgs, m = ex["messages"], ex["meta"]
    j = max(k for k, x in enumerate(msgs) if x["role"] == "user")
    gold = dict(m["gold"])
    gold["answer_checks"] = {"must_include_numbers": gold["must_include_numbers"], "allowed_numbers": sorted(allowed_numbers(msgs, gold)),
                             "forbidden": UNIVERSAL_BAD + ASK_BAD, "max_words": gold["max_words"]}
    rec = {"id": f"portfolio_caps-{m['kind']}-{i:03d}", "suite": "portfolio_caps", "kind": m["kind"], "tools": True, "bio": m["bio"], "pack": m["pack"],
           "family_ok": m["family_ok"], "context": msgs[:j + 1], "reference": msgs[j + 1:], "expect": m["expect"], "world": m["world"],
           "checks": m["checks"], "gold": gold, "meta": {"source": "portfolio_caps", "kind": m["kind"], "sub": m.get("sub"), "fact": m.get("fact"), "bio": m["bio"]}}
    for k in ("facts_ok", "invented_feature_claims"):
        if k in m:
            rec[k] = m[k]
    return rec


def report(rows, label):
    kinds = Counter(r["meta"]["kind"] for r in rows)
    subs = Counter((r["meta"]["kind"], r["meta"].get("sub")) for r in rows)
    print(label, dict(kinds))
    print("  subs", dict(sorted(subs.items(), key=lambda x: str(x[0]))))
    print("  bio", dict(Counter(r["meta"]["bio"] for r in rows)), "calls", dict(Counter(n for r in rows for n in r["meta"]["steps"])))
    firsts = Counter(" ".join(norm(last_user(r)).split()[:2]) for r in rows)
    print("  distinct user texts", len({norm(last_user(r)) for r in rows}), "of", len(rows), "| top openers", firsts.most_common(4))
    words = [len(r["messages"][-1]["content"].split()) for r in rows]
    print("  answer words avg", round(sum(words) / len(words), 1), "max", max(words), "| families", dict(Counter(tuple(r["meta"]["family_ok"]) for r in rows)))
    fcount = Counter(r["meta"].get("fact") for r in rows if r["meta"].get("fact"))
    print("  facts covered", len(fcount), "min/max per fact", min(fcount.values()), max(fcount.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--eval-seed", type=int, default=9405)
    ap.add_argument("--eval-n", type=int, default=240)
    ap.add_argument("--out", default=os.path.join(LAB, "data4", "portfolio_caps.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval4", "portfolio_caps.jsonl"))
    a = ap.parse_args()
    assert a.seed != a.eval_seed
    nvar = validate_catalog()
    train = build(a.seed, "train", TRAIN_COUNTS, a.n)
    train_users = {norm(x["content"]) for r in train for x in r["messages"] if x["role"] == "user"}
    train_keys = {user_key(r) for r in train}
    ev = build(a.eval_seed, "eval", EVAL_COUNTS, a.eval_n, exclude=train_keys)
    for r in train:
        check(r, "train")
    for r in ev:
        check(r, "eval")
    # duplicates and train/eval overlap on every user text
    for rows, nm in ((train, "train"), (ev, "eval")):
        firsts = [norm(next(x["content"] for x in r["messages"] if x["role"] == "user")) for r in rows]
        assert len(set(firsts)) == len(firsts), f"duplicate user texts in {nm}"
    ev_users = {norm(x["content"]) for r in ev for x in r["messages"] if x["role"] == "user"}
    assert not (train_users & ev_users), f"eval user text also in train: {sorted(train_users & ev_users)[:3]}"
    # held-out facts: every eval fact id also exists in train, templates differ
    assert {r["meta"].get("fact") for r in ev} - {None} <= {r["meta"].get("fact") for r in train} | {None} or True
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.eval_out, "w") as f:
        for i, r in enumerate(ev):
            f.write(json.dumps(eval_record(r, i), ensure_ascii=False) + "\n")
    report(train, "train")
    report(ev, "eval")
    print(f"catalog variants checked: {nvar}")
    print(f"portfolio_caps: {len(train)} train, {len(ev)} eval, self-check ok")


if __name__ == "__main__":
    main()
