#!/usr/bin/env python3
"""drift_fix: re-teach v2 tools whose results changed in the app (spec/hermit_tools_v3.meta.json, entries with v3_added).

Writes
  data4/drift_fix.jsonl   training rows   (--seed, --n; default seed 4, n 300)
  eval4/drift_fix.jsonl   held-out items  (--eval-seed, --n-eval; default 9414, 40)
and then runs the self-check over both files (exit code 1 on any failure).

Six areas (train counts at n=300; eval has its own fixed split):
  stocks      80  stock and index perps through prepare_trade (trade.xyz), trade_xyz_margin block -> open_page trade,
                  a stock "view" (get_price + risks + offer), near-misses (buy shares, price of bitcoin, a crypto trade)
  memecoins   60  per-token prices from the kPEPE-style markets written as decimals, the per-1,000 rejection of
                  per-token limit/stop prices (ask, or convert with the calculator and say so, never retry unchanged)
  wallets     40  get_balances with wallet + note: say which wallet the numbers are for and that others exist
  triggers    40  hyperliquid_account trigger rows: a stop loss / take profit, never "orders you placed"; counts skip them
  yields      50  get_yields v3 fields: only can_deposit_in_hermit pools can be prepared, il_risk, base vs reward, 30d APY
  position_math 30  no coin in the schema: the result note says the market maximum leverage was assumed

Record shape (gen3 style): {"messages": [system, user, assistant{tool_calls}, tool, ..., assistant], "tools": true,
  "meta": {"source": "drift_fix", "kind", "area", "bio", "slots", "gold_calls", "world", "expect", "checks", "tpl", ...}}
Eval item: {id, kind, tools, bio, context, reference, expect, world, checks{scorers, gold_calls, ...}, meta}.
  world = DriftWorld.state(); DriftWorld.from_state(world).call(name, args) replays every tool result byte for byte.

Everything the assistant says is a Python f-string filled from tool results, calculator results or the user's own text.
No number is typed by hand. gen2 and gen3 modules are imported through sys.path and never edited.
"""
import argparse, functools, json, os, random, re, sys
from collections import Counter
from decimal import Decimal

sys.dont_write_bytecode = True          # never write __pycache__ next to the read-only gen2/gen3 modules
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(ROOT, "gen3"), os.path.join(ROOT, "gen2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hermit_fmt as H                  # gen2: calc, calc_str, parse_calls, assistant_text
import sim as SIM2                      # gen2: position_math
from stocks_price import STOCKS, STOCK_ORDER, typo      # gen3: stock aliases (train "tr" and eval "ev" pools)

SPEC = os.path.join(ROOT, "spec")
TOOLS_V3 = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS_V3}
SYSTEM_TPL = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert SYSTEM_TPL.count("{BIO}") == 1
OUT_TRAIN = os.path.join(ROOT, "data4", "drift_fix.jsonl")
OUT_EVAL = os.path.join(ROOT, "eval4", "drift_fix.jsonl")
REQUIRED_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}

STOCK_SOURCE = "trade.xyz perp on Hyperliquid (tracks the stock)"
CRYPTO_SOURCE = "Hyperliquid mark price"
BLOCK_REASON = "trade.xyz has its own balance, separate from your Hyperliquid account. Move USDC to it in Trade first."
PREP_NOTE = "Waiting for the user to approve with Face ID."
DEP_NOTE = "Waiting for the user to review the deposit and approve it with Face ID."
NO_MARKET = "No Hyperliquid market for {s}. Answer from what you know instead, and say there's no live price here."


# ============================================================ number helpers
def D(x):
    return Decimal(repr(x)) if isinstance(x, float) else Decimal(str(x))


def plain(x):
    """Decimal text, never an exponent: 9.8e-06 -> 0.0000098, 128.0 -> 128."""
    s = format(D(x), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def commas(x, min_dp=0):
    s = plain(abs(x) if not isinstance(x, str) else x)
    ip, _, fp = s.partition(".")
    if len(fp) < min_dp:
        fp = fp.ljust(min_dp, "0")
    return f"{int(ip):,}" + ("." + fp if fp else "")


def money(x):
    neg = D(x) < 0
    return ("-" if neg else "") + "$" + commas(x, 2 if "." in plain(abs(D(x))) else 0)


def nn(x):
    """Whole floats become ints so json.dumps never shows 3100.0."""
    return int(x) if isinstance(x, float) and x.is_integer() else x


def srt(d):
    """Toolbox results use alphabetical keys at every level."""
    if isinstance(d, dict):
        return {k: srt(d[k]) for k in sorted(d)}
    if isinstance(d, list):
        return [srt(x) for x in d]
    return d


def cap(s):
    return s[0].upper() + s[1:] if s else s


WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def chain_label(c):
    return {"base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "bnb": "BNB Chain", "polygon": "Polygon",
            "solana": "Solana", "near": "NEAR"}.get(c, c)


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&$.\s]", "", t.lower())).strip()


# ============================================================ entities
def _stock_px(rng, sym):
    lo, hi = {"NVDA": (95, 210), "TSLA": (180, 480), "AAPL": (170, 290), "MSFT": (350, 560), "META": (480, 820),
              "AMZN": (150, 260), "GOOGL": (140, 260), "COIN": (180, 420), "MSTR": (150, 520), "HOOD": (25, 140),
              "SPX": (5200, 7200)}[sym]
    while True:
        p = round(rng.uniform(lo, hi), 2)
        if p != int(p):
            return p


def _chg(rng, null_ok=False):
    if null_ok and rng.random() < 0.1:
        return None
    while True:
        c = round(max(-9.0, min(9.0, rng.gauss(0, 2.4))), 2)
        if abs(c) >= 0.05:
            return nn(c)


CRYPTO = {  # sym: (lo, hi, dp, train aliases, eval aliases, name)
    "BTC": (62000, 115000, 2, ["bitcoin", "btc", "BTC", "Bitcoin"], ["$btc", "Btc", "BTC coin"], "Bitcoin"),
    "ETH": (2200, 4800, 2, ["ethereum", "eth", "ETH", "Ethereum", "ether"], ["$eth", "Eth", "ETH coin"], "Ethereum"),
    "SOL": (110, 240, 2, ["solana", "sol", "SOL", "Solana"], ["$sol", "Sol", "sol coin"], "Solana"),
    "HYPE": (18, 55, 2, ["hype", "HYPE", "Hype"], ["$HYPE", "hype coin"], "Hype"),
    "XRP": (1.8, 3.4, 4, ["xrp", "XRP", "ripple"], ["$xrp", "Ripple"], "XRP"),
    "DOGE": (0.12, 0.34, 4, ["doge", "DOGE", "dogecoin", "Dogecoin"], ["$doge", "Doge"], "Dogecoin"),
    "BNB": (480, 880, 2, ["bnb", "BNB", "binance coin"], ["$bnb", "Bnb"], "BNB"),
    "WIF": (0.8, 3.0, 4, ["wif", "WIF", "dogwifhat"], ["$wif", "Wif"], "WIF"),
}
MEMES = {  # sym: (lo, hi, train aliases, eval aliases)
    "PEPE": (6e-6, 2.2e-5, ["PEPE", "pepe", "Pepe", "$PEPE", "pepe coin"], ["$pepe", "PEPE coin", "pepe token"]),
    "SHIB": (7e-6, 2.5e-5, ["SHIB", "shib", "Shib", "shiba inu", "Shiba"], ["$SHIB", "shib coin", "Shiba Inu"]),
    "BONK": (1.2e-5, 4.5e-5, ["BONK", "bonk", "Bonk", "$BONK"], ["bonk coin", "$bonk", "BONK token"]),
    "FLOKI": (6e-5, 2.4e-4, ["FLOKI", "floki", "Floki", "$FLOKI"], ["floki inu", "$floki", "FLOKI coin"]),
}
MEME_SYMS = list(MEMES)
WALLET_TR = ["Main", "Savings", "Trading", "Daily", "Cold"]
WALLET_EV = ["Spending", "Vault", "Degen", "Family", "Side"]
HOLD_TOKENS = {"USDC": ["base", "arbitrum", "ethereum", "solana", "polygon"], "USDT": ["ethereum", "arbitrum", "bnb"],
               "ETH": ["base", "arbitrum", "ethereum"], "SOL": ["solana"], "BNB": ["bnb"], "WIF": ["solana"], "BONK": ["solana"]}
HOLD_PX = {"USDC": 1.0, "USDT": 1.0, "ETH": 3120.0, "SOL": 142.37, "BNB": 610.0, "WIF": 1.82, "BONK": 0.000021}
PERP_COINS = {"SOL": (110, 240), "ETH": (2200, 4800), "BTC": (62000, 115000), "HYPE": (18, 55)}


# ============================================================ simulator
class DriftWorld:
    """Replays calculator, get_price, get_balances, hyperliquid_account, get_yields, position_math, prepare_trade,
    prepare_deposit and open_page with the v3 result shapes. State is plain JSON."""

    PARTS = ("prices", "xyz_ok", "wallet", "hl", "pools")

    def __init__(self, prices=None, xyz_ok=True, wallet=None, hl=None, pools=None):
        self.prices = json.loads(json.dumps(prices or {}))
        self.xyz_ok = xyz_ok
        self.wallet = json.loads(json.dumps(wallet)) if wallet else None
        self.hl = json.loads(json.dumps(hl)) if hl else None
        self.pools = json.loads(json.dumps(pools or []))

    def state(self):
        st = {}
        if self.prices:
            st["prices"] = json.loads(json.dumps(self.prices))
        if not self.xyz_ok:
            st["xyz_ok"] = False
        if self.wallet:
            st["wallet"] = json.loads(json.dumps(self.wallet))
        if self.hl:
            st["hl"] = json.loads(json.dumps(self.hl))
        if self.pools:
            st["pools"] = json.loads(json.dumps(self.pools))
        return st

    @classmethod
    def from_state(cls, st):
        return cls(prices=st.get("prices"), xyz_ok=st.get("xyz_ok", True), wallet=st.get("wallet"), hl=st.get("hl"),
                   pools=st.get("pools"))

    # ---- dispatch
    def call(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args if isinstance(args, dict) else {}))
        except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as e:
            return {"error": f"bad arguments: {e}"}

    @staticmethod
    def _sym(s):
        return str(s).upper().strip().replace("XYZ:", "")

    def t_calculator(self, expression):
        try:
            return {"result": H.calc_str(H.calc(expression))}
        except ZeroDivisionError:
            return {"error": "division by zero"}
        except Exception:
            return {"error": "could not evaluate expression"}

    def t_get_price(self, symbol):
        sym = self._sym(symbol)
        p = self.prices.get(sym)
        if p is None:
            return {"error": NO_MARKET.format(s=sym)}
        if p["kind"] == "stock":
            return srt({"change_24h_pct": p["change"], "price": p["price"], "source": STOCK_SOURCE, "symbol": sym})
        if p["kind"] == "meme":
            return srt({"change_24h_pct": p["change"], "contract_price": p["contract_price"], "market": "k" + sym,
                        "price": p["price"], "source": f"Hyperliquid mark price (per token; the market is k{sym}, 1,000 {sym})",
                        "symbol": sym})
        return srt({"change_24h_pct": p["change"], "price": p["price"], "source": CRYPTO_SOURCE, "symbol": sym})

    def t_prepare_trade(self, coin, side, size_usd, leverage, order_type, limit_price=None, take_profit=None, stop_loss=None):
        sym = self._sym(coin)
        p = self.prices.get(sym)
        if side not in ("long", "short") or order_type not in ("market", "limit"):
            return {"error": "bad arguments: side or order_type"}
        if order_type == "limit" and limit_price is None:
            return {"error": "limit_price is required for a limit order", "prepared": False}
        if p is None:
            return {"error": f"{sym} isn't tradable on Hyperliquid"}
        if p["kind"] == "stock" and not self.xyz_ok:
            return srt({"note": "Nothing was prepared. Tell the user why.", "reasons": [BLOCK_REASON], "status": "blocked"})
        if p["kind"] == "meme":
            for v in (limit_price, take_profit, stop_loss):
                if v is not None and abs(v - p["price"]) < abs(v - p["contract_price"]):       # looks per token
                    return {"error": f"{sym} trades as k{sym}, 1,000 {sym} per unit. Give the limit, take-profit and stop-loss "
                                     f"prices per 1,000 {sym} (for example {plain(p['contract_price'])}).", "prepared": False}
        return {"status": "review_opened", "kind": "trade", "executed": False, "note": PREP_NOTE}

    def t_get_balances(self, chain=None):
        w = self.wallet
        if w is None:
            return {"holdings": []}
        rows = [h for h in w["holdings"] if chain is None or h["chain"] == str(chain).lower()]
        out = {"holdings": rows}
        if w.get("near_account"):
            out["near_account"] = w["near_account"]
        if w.get("multi"):
            others = ", ".join(w["others"])
            out["note"] = f"Showing the selected wallet ({w['name']}). Other wallets: {others}."
            out["wallet"] = w["name"]
        return srt(out)

    def t_hyperliquid_account(self):
        a = self.hl
        if a is None:
            return {"error": "No wallet on this iPhone yet"}
        out = {"account_value": a["value"], "available_usdc": a["available"], "margin_used": a["margin_used"],
               "open_orders": a["orders"], "positions": a["positions"]}
        if a.get("mode"):
            out["account_mode"] = a["mode"]
        return srt(out)

    def t_get_yields(self, asset=None, chain=None):
        rows = list(self.pools)
        if asset:
            a = str(asset).upper().replace("WETH", "ETH")
            rows = [r for r in rows if a in r["asset"].upper().replace("WETH", "ETH").split("-")]
        if chain:
            rows = [r for r in rows if r["chain"].lower() == str(chain).lower()]
        rows.sort(key=lambda r: -r["apy_pct"])
        return {"pools": [srt(r) for r in rows[:5]]}

    def t_position_math(self, side, entry, size_usd, leverage, price=None, maintenance_margin=None):
        if side not in ("long", "short"):
            return {"error": "bad arguments: side"}
        r = SIM2.position_math(side, entry, size_usd, leverage, price, mm=maintenance_margin)
        r.pop("maintenance_margin_used", None)
        for k in ("entry", "size_usd", "leverage", "price", "margin", "quantity", "pnl", "roe_pct", "liquidation_price",
                  "distance_to_liq_pct"):
            if k in r:
                r[k] = nn(r[k])
        if maintenance_margin is None:
            r["note"] = (f"No coin was given, so the market's maximum leverage was assumed to be {plain(leverage)}x. "
                         f"The liquidation price is approximate.")
        return srt(r)

    def t_prepare_deposit(self, symbol, amount, project=None, chain=None):
        sym = str(symbol).upper()
        cands = [r for r in self.pools if r["can_deposit_in_hermit"] and r["asset"].upper() == sym
                 and (chain is None or r["chain"].lower() == str(chain).lower())]
        if not cands:
            return srt({"note": "Nothing was prepared. Tell the user why.", "reasons": [f"No {sym} pool can be deposited from chat."],
                        "status": "blocked"})
        r = max(cands, key=lambda x: x["apy_pct"])
        return srt({"amount": f"{plain(amount)} {sym}", "apy_pct": r["apy_pct"], "executed": False, "kind": "deposit",
                    "note": DEP_NOTE, "pool": f"{r['protocol']} on {r['chain']}", "status": "review_opened"})

    PAGES = ("home", "wallets", "trade", "earn", "payments", "explore", "calendar", "projects", "automations", "files",
             "connectors", "settings")

    def t_open_page(self, page, tab=None):
        if page not in self.PAGES:
            return {"error": "That page can't be opened from chat."}
        out = {"note": f"The user taps it to open {page.capitalize()}.", "page": page, "screen": page, "status": "button_shown"}
        if tab:
            out["tab"] = tab
        return srt(out)


# ============================================================ conversation builder
NAMES = H.NAMES
SAFE_NOTES = ["I trade mostly BTC and ETH perps on Hyperliquid.", "Keep answers very short.", "I'm new to crypto, explain simply.",
              "I use Base and Arbitrum most.", "I prefer stablecoin yield over farming.", "I live in Europe, quote times in CET."]


def make_system(rng, bio):
    s = SYSTEM_TPL.replace("{BIO}", bio)
    if rng.random() < 0.2:
        s += f"\nThe user wants to be called {rng.choice(NAMES)}."
    if rng.random() < 0.08:
        notes = rng.sample(SAFE_NOTES, rng.choice([1, 2]))
        s += "\nThe user's standing notes (hermit.md). Follow them:\n" + "\n".join("- " + n for n in notes)
    return s


BIO_FORCE = [None]          # build_rows keeps the Face ID / Touch ID mix at about 70 / 30


class Conv:
    def __init__(self, rng, world, bio):
        bio = BIO_FORCE[0] or bio
        self.rng, self.w, self.bio = rng, world, bio
        self.world0 = world.state()
        self.msgs = [{"role": "system", "content": make_system(rng, bio)}]
        self.expect = []
        self.tpls = []
        self.slots = []

    def user(self, text, tpl=None):
        self.msgs.append({"role": "user", "content": text})
        self.expect = []
        if tpl:
            self.tpls.append(tpl)

    def call(self, *calls):
        tcs = [{"name": n, "arguments": a} for n, a in calls]
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        self.expect.append({"calls": tcs})
        res = []
        for n, a in calls:
            r = self.w.call(n, a)
            res.append(r)
            self.msgs.append({"role": "tool", "content": json.dumps(r, ensure_ascii=False).replace("Face ID", self.bio)})
        return res if len(res) > 1 else res[0]

    def say(self, text, ask=False):
        self.msgs.append({"role": "assistant", "content": text.replace("Face ID", self.bio)})
        self.expect.append({"ask": True} if ask else {"text": True})

    def ex(self, kind, area, scorers, must=None, forbid=None, extra=None):
        last_u = max(i for i, m in enumerate(self.msgs) if m["role"] == "user")
        gold, gold_calc = [], []
        for e in self.expect:
            for tc in e.get("calls", []):
                (gold_calc if tc["name"] == "calculator" else gold).append(tc)
        order = ["right_tool", "exact_digits", "rounded_ok", "facts_ok", "no_invented_price"]
        sc = set(scorers)
        if (extra or {}).get("knowledge"):
            sc.discard("exact_digits")
        else:
            sc |= {"no_invented_price", "rounded_ok"}
        if must or forbid:
            sc.add("facts_ok")
        if gold:
            sc.add("right_tool")
        scorers = [x for x in order if x in sc]
        checks = {"scorers": scorers, "gold_calls": gold, "gold_calculator": gold_calc,
                  "forbid_tools": sorted(t for t in TOOL_SCHEMA if t.startswith("prepare_") and t not in {g["name"] for g in gold}),
                  "answer_must": must or [], "answer_forbid": forbid or []}
        if extra:
            checks.update(extra)
        self.slots = [s for k, s in enumerate(self.slots) if s["word"].lower() in " ".join(
            x["content"] for x in self.msgs if x["role"] == "user").lower()]
        return {"messages": self.msgs, "tools": True,
                "meta": {"source": "drift_fix", "kind": kind, "area": area, "bio": self.bio, "slots": self.slots,
                         "gold_calls": gold, "world": self.world0, "expect": self.expect, "checks": checks,
                         "tpl": self.tpls[0] if self.tpls else None, "tpls": self.tpls, "last_user": last_u}}


# ============================================================ phrasing
PRE_TR = ["", "", "", "", "hey ", "ok ", "yo ", "hermit, ", "so ", "quick one: ", "pls "]
PRE_EV = ["", "", "", "hi, ", "alright ", "sorry, ", "one thing: "]
SUF_TR = ["", "", "", "", " pls", " thx", " rn", " please", " asap", " quick"]
SUF_EV = ["", "", "", " when you can", " thanks", " for me"]


def fill(rng, tpls, mode, slots, tag, deco=True):
    """Pick a template from the pool for this mode, fill {slots}. Typos, casing, prefixes and suffixes touch the template
    text only, never a slot value (ticker, amount, price)."""
    pool = tpls[mode]
    i = rng.randrange(len(pool))
    tpl = pool[i]
    parts = re.split(r"(\{\w+\})", tpl)
    lit = [k for k, p in enumerate(parts) if not p.startswith("{") and p.strip()]
    if lit and rng.random() < 0.13:
        k = rng.choice(lit)
        parts[k] = typo(parts[k], rng)
    lower = rng.random() < 0.3
    out = []
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            out.append(str(slots[p[1:-1]]))
        else:
            out.append(p.lower() if lower else p)
    s = "".join(out)
    pre = rng.choice(PRE_EV if mode == "ev" else PRE_TR) if deco else ""
    suf = rng.choice(SUF_EV if mode == "ev" else SUF_TR) if deco else ""
    if pre and not s[:1].isdigit():
        s = pre + (s[0].lower() + s[1:] if s[:1].isupper() and not s[:2].isupper() else s)
    if suf and not s.endswith("?"):
        s = s + suf
    if s.endswith("?") and rng.random() < 0.2:
        s = s[:-1]
    return s, f"{mode}:{tag}{i}"


def T(tr, ev):
    return {"tr": tr, "ev": ev}


def alias_stock(rng, sym, mode):
    e = STOCKS[sym]
    if mode == "ev" and rng.random() < 0.6:
        return rng.choice(e["ev"])
    return rng.choice(e["tr"])


def alias_crypto(rng, sym, mode):
    e = CRYPTO[sym]
    if mode == "ev" and rng.random() < 0.6:
        return rng.choice(e[4])
    return rng.choice(e[3])


def alias_meme(rng, sym, mode):
    e = MEMES[sym]
    if mode == "ev" and rng.random() < 0.6:
        return rng.choice(e[3])
    return rng.choice(e[2])


def size_text(rng, size):
    return rng.choice([f"${size}", f"${size}", f"{size} usd", f"{size}$", f"{size} dollars", f"${size:,}"] if size >= 1000
                      else [f"${size}", f"${size}", f"{size} usd", f"{size}$", f"{size} dollars"])


def done(rng, bio):
    return rng.choice([f"Review it and approve with {bio} when you're ready.", f"It's on the review screen and nothing happens until you approve it with {bio}.",
                       f"Check the details on the review screen and approve with {bio} if it looks right.",
                       f"Approve it with {bio} if it looks right.", f"It's waiting for your {bio} on the review screen."])


# ============================================================ world builders
def stock_entry(rng, sym, null_ok=False):
    return {"kind": "stock", "price": _stock_px(rng, sym), "change": _chg(rng, null_ok)}


def crypto_entry(rng, sym):
    lo, hi, dp, *_ = CRYPTO[sym]
    return {"kind": "crypto", "price": nn(round(rng.uniform(lo, hi), dp)), "change": _chg(rng)}


def meme_entry(rng, sym):
    lo, hi, *_ = MEMES[sym]
    p = float(f"{rng.uniform(lo, hi):.2g}")
    return {"kind": "meme", "price": p, "contract_price": nn(round(p * 1000, 8)), "change": _chg(rng)}


def stock_sym_label(sym):
    return "the S&P 500 index" if sym == "SPX" else sym


def perp_note(rng, sym):
    if sym == "SPX":
        return rng.choice(["SPX here is a perp that tracks the S&P 500 index and trades 24/7.",
                           "It's a perp that tracks the index, so it trades around the clock."])
    return rng.choice([f"{sym} here is a perp that tracks the stock and trades 24/7.",
                       f"It's a perp that tracks {sym}, so it trades around the clock.",
                       f"{sym} is traded as a perp that tracks the stock, and it runs 24/7."])


def price_fig(sym, p):
    return commas(p, 2) if sym == "SPX" else money(p)


def arrow(c):
    return "up" if c > 0 else "down"


# ============================================================ area 1: stocks
SIZES = [25, 50, 100, 100, 150, 200, 250, 300, 500, 750, 1000, 2000]
STK_TRADE = T([
    "{sd} {w} {sz} {lev}x{tail}", "go {sd} on {w} with {sz} at {lev}x{tail}", "open a {lev}x {sd} on {w}, {sz}{tail}",
    "i want to {sd} {w}, {sz}, {lev}x leverage{tail}", "{sz} {sd} {w} {lev}x market{tail}", "can you {sd} {w} for me. {sz} {lev}x{tail}",
    "put {sz} on {w} {sd} at {lev}x{tail}", "{w} {sd}, {lev}x, {sz}{tail}", "let's {sd} {w} {sz} {lev}x{tail}",
    "{lev}x {sd} {w} {sz}{tail}", "prepare a {sd} on {w}: {sz}, {lev}x leverage{tail}", "take a {sd} position in {w}, {sz} and {lev}x{tail}",
], [
    "{sd} {w} for {sz}, {lev}x leverage please{tail}", "set me up with a {lev}x {sd} in {w} using {sz}{tail}",
    "I'd like to {sd} {w}: {sz} at {lev}x{tail}", "{w}: {sd} {lev}x {sz}{tail}", "{sz} on {w}, {sd}, with {lev}x{tail}",
])
STK_NOLEV = T(["{sd} {w} {sz}", "i wanna {sd} {w} with {sz}", "{sz} {sd} on {w}", "open a {sd} on {w}, {sz}"],
              ["{sd} {w}, {sz} please", "could you {sd} {w} for {sz}"])
LEV_REPLY = T(["{lev}x", "make it {lev}x", "let's do {lev}x", "{lev} times", "go with {lev}x", "{lev}x pls"],
              ["{lev}x leverage", "say {lev}x"])
TAIL_SL = T([" stop loss {sl}", " with a stop at {sl}", " sl {sl}", ", stop {sl}", " and a stop loss at {sl}"],
            [" stop loss at {sl}", " with SL {sl}"])
TAIL_TP = T([" tp {tp}", " take profit {tp}", " and take profit at {tp}", ", tp at {tp}"], [" with take profit {tp}", " TP {tp}"])
TAIL_LP = T([" limit {lp}", " limit order at {lp}", " entry at {lp}", ", limit at {lp}"], [" at {lp} limit", " as a limit at {lp}"])


def tail_text(rng, mode, a):
    out = ""
    if a.get("order_type") == "limit":
        out += rng.choice(TAIL_LP[mode]).format(lp=plain(a["limit_price"]))
    if "stop_loss" in a:
        out += rng.choice(TAIL_SL[mode]).format(sl=plain(a["stop_loss"]))
    if "take_profit" in a:
        out += rng.choice(TAIL_TP[mode]).format(tp=plain(a["take_profit"]))
    return out


def trade_args(rng, coin, price, side, size, lev, limit_p=0.22, sl_p=0.28, tp_p=0.2, dp=2, tick=None):
    """Arguments in schema order. Prices come from the live price band so a stop is on the right side."""
    a = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
    lo = 1 if side == "long" else -1
    if rng.random() < limit_p:
        a["order_type"] = "limit"
        a["limit_price"] = nn(round(price * (1 - lo * rng.uniform(0.005, 0.04)), dp))
    ref = a.get("limit_price", price)
    if rng.random() < sl_p:
        a["stop_loss"] = nn(round(ref * (1 - lo * rng.uniform(0.04, 0.12)), dp))
    if rng.random() < tp_p:
        a["take_profit"] = nn(round(ref * (1 + lo * rng.uniform(0.04, 0.18)), dp))
    return a


def trade_text(a, label=None, pf=money):
    s = f"a {money(a['size_usd'])} {label or a['coin']} {a['side']} at {plain(a['leverage'])}x"
    if a["order_type"] == "limit":
        s += f", limit entry at {pf(a['limit_price'])}"
    extras = []
    if "stop_loss" in a:
        extras.append(f"a stop loss at {pf(a['stop_loss'])}")
    if "take_profit" in a:
        extras.append(f"a take profit at {pf(a['take_profit'])}")
    if extras:
        s += ", with " + " and ".join(extras)
    return s


def sc_stock_trade(rng, mode):
    sym = rng.choice(STOCK_ORDER)
    w = DriftWorld(prices={sym: stock_entry(rng, sym)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    px = w.prices[sym]["price"]
    side = rng.choice(["long", "long", "short"])
    size, lev = rng.choice(SIZES), rng.choice([2, 3, 4, 5, 5, 5, 10, 10])
    a = trade_args(rng, sym, px, side, size, lev, dp=2)
    word = alias_stock(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, STK_TRADE, mode, {"sd": side, "w": word, "sz": size_text(rng, size), "lev": lev, "tail": tail_text(rng, mode, a)}, "st")
    c.user(txt, tpl)
    r = c.call(("prepare_trade", a))
    assert r["status"] == "review_opened"
    lead = f"I've prepared {trade_text(a)}."
    if rng.random() < 0.75:
        lead += " " + perp_note(rng, sym)
    c.say(f"{lead} {done(rng, c.bio)}")
    return c.ex("stock_trade", "stocks", ["right_tool", "exact_digits"], must=[r"\bprepared\b", r"\b" + sym + r"\b"],
                forbid=[r"(?i)can'?t be traded"], extra={"expect_status": "review_opened"})


def sc_stock_ask_lev(rng, mode):
    sym = rng.choice(STOCK_ORDER)
    w = DriftWorld(prices={sym: stock_entry(rng, sym)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    side, size, lev = rng.choice(["long", "short"]), rng.choice(SIZES), rng.choice([2, 3, 5, 5, 10])
    word = alias_stock(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, STK_NOLEV, mode, {"sd": side, "w": word, "sz": size_text(rng, size)}, "sa")
    c.user(txt, tpl)
    c.say(rng.choice(["What leverage do you want?", "How much leverage? 1x is no leverage.", "Sure. What leverage should I use?"]), ask=True)
    t2, tpl2 = fill(rng, LEV_REPLY, mode, {"lev": lev}, "sb", deco=False)
    c.user(t2, tpl2)
    a = {"coin": sym, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
    c.call(("prepare_trade", a))
    c.say(f"I've prepared {trade_text(a)}. {perp_note(rng, sym)} {done(rng, c.bio)}")
    return c.ex("stock_ask_leverage", "stocks", ["right_tool", "exact_digits"], must=[r"\bprepared\b"], extra={"first_turn_ask": True})


def sc_stock_blocked(rng, mode):
    sym = rng.choice(STOCK_ORDER)
    w = DriftWorld(prices={sym: stock_entry(rng, sym)}, xyz_ok=False)
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    px = w.prices[sym]["price"]
    side, size, lev = rng.choice(["long", "long", "short"]), rng.choice(SIZES), rng.choice([2, 3, 5, 5, 10])
    a = trade_args(rng, sym, px, side, size, lev, limit_p=0.1, sl_p=0.2, tp_p=0.1)
    word = alias_stock(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, STK_TRADE, mode, {"sd": side, "w": word, "sz": size_text(rng, size), "lev": lev, "tail": tail_text(rng, mode, a)}, "sk")
    c.user(txt, tpl)
    r = c.call(("prepare_trade", a))
    assert r["status"] == "blocked"
    c.call(("open_page", {"page": "trade"}))
    lead = rng.choice(["Nothing was prepared.", "I couldn't prepare that one.", "That order isn't prepared yet."])
    btn = rng.choice(["I put a button below that opens Trade.", "Tap the button below to open Trade.", "There's a button below that takes you to Trade."])
    tail = rng.choice(["", "", " Once it's funded, ask me again and I'll prepare it."])
    c.say(f"{lead} {BLOCK_REASON} {btn}{tail}")
    return c.ex("stock_blocked", "stocks", ["right_tool", "exact_digits", "facts_ok"],
                must=[r"has its own balance, separate from your Hyperliquid account", r"Move USDC to it in Trade first", r"button"],
                forbid=[r"(?i)\bI(?:'ve| have)? opened\b", r"(?i)\bprepared (?:a|an|the)\b"])


VIEW_Q = T(["what do you think of {w}", "{w} long or short here?", "thoughts on {w}?", "is {w} a good buy right now", "should I long {w}",
            "give me a view on {w}", "{w} outlook?", "bullish on {w}, am I crazy", "is {w} worth a trade today", "what's your take on {w}",
            "{w} looking good or bad rn", "would you short {w} here?", "i'm thinking about trading {w}, what do you see"],
           ["how do you see {w} at the moment", "{w}: long, short or skip?", "any opinion on {w}?", "worth trading {w} now?"])


def sc_stock_view(rng, mode):
    sym = rng.choice(STOCK_ORDER)
    w = DriftWorld(prices={sym: stock_entry(rng, sym, null_ok=True)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    word = alias_stock(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, VIEW_Q, mode, {"w": word}, "sv")
    c.user(txt, tpl)
    r = c.call(("get_price", {"symbol": sym}))
    ch = r["change_24h_pct"]
    head = f"{stock_sym_label(sym) if sym == 'SPX' else sym} is at {price_fig(sym, r['price'])}"
    head = cap(head) + (f", {arrow(ch)} {plain(abs(ch))}% in the last 24 hours." if ch is not None else ".")
    if ch is None:
        mood = ""
    elif abs(ch) >= 3:
        mood = rng.choice(["It already moved a lot today, so a fresh entry is chasing the move.", "That is a big move for one day, and entering after it is risky."])
    elif abs(ch) >= 1:
        mood = rng.choice(["That's a normal day, nothing stretched.", "The move is modest, so there's no rush."])
    else:
        mood = rng.choice(["It's been quiet today.", "Not much has happened today."])
    if sym == "SPX":
        risk = rng.choice(["Risks: a sharp market-wide drop, and leverage multiplies it.", "The main risks are a sudden market sell-off and, with leverage, a small move against you can liquidate the position."])
    else:
        risk = rng.choice(["Risks: a gap on earnings or news, and leverage multiplies losses.",
                           "The main risks are sudden news gaps and, with leverage, a small move against you can liquidate the position.",
                           "Risks: it can gap on news while the stock market is shut, and leverage makes every move hurt more."])
    offer = rng.choice(["Want me to prepare a long or a short? Tell me the size and leverage.",
                        "I can prepare an order if you want one: long or short, the size and the leverage."])
    c.say(" ".join(x for x in [head, perp_note(rng, sym), mood, risk, offer] if x))
    return c.ex("stock_view", "stocks", ["right_tool", "exact_digits", "no_invented_price", "facts_ok"],
                must=[r"\bperp\b", r"24/7|around the clock", r"(?i)risk", r"(?i)long or (?:a )?short|an order"],
                forbid=[r"(?i)can'?t be traded", r"(?i)\byou should (?:buy|sell)\b"])


STK_PRICE = T(["price of {w}", "what's {w} at?", "{w} price", "how much is {w}", "check {w} for me", "where's {w} trading",
               "{w} today?", "what's {w} worth right now", "show me {w}", "how's {w} doing"],
              ["what is {w} trading at", "where is {w} right now?", "latest on {w}"])


def sc_stock_price(rng, mode):
    sym = rng.choice(STOCK_ORDER)
    w = DriftWorld(prices={sym: stock_entry(rng, sym, null_ok=True)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    word = alias_stock(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, STK_PRICE, mode, {"w": word}, "sp")
    c.user(txt, tpl)
    r = c.call(("get_price", {"symbol": sym}))
    ch = r["change_24h_pct"]
    s = f"{cap(sym if sym != 'SPX' else 'SPX')} is at {price_fig(sym, r['price'])}" + (f", {arrow(ch)} {plain(abs(ch))}% in the last 24 hours." if ch is not None else ".")
    s += " " + perp_note(rng, sym) + " " + rng.choice(["I can prepare a trade on it if you want one.", "If you want to trade it, I can prepare a long or a short."])
    c.say(s)
    return c.ex("stock_price", "stocks", ["right_tool", "exact_digits", "no_invented_price", "facts_ok"],
                must=[r"\bperp\b", r"24/7|around the clock", r"prepare"], forbid=[r"(?i)can'?t be traded", r"(?i)not available"])


SHARES = T(["buy {w} shares", "can I buy actual {w} stock", "i want to buy some {w} shares", "get me {w} shares", "buy {w} stock",
            "purchase real {w} shares", "can i buy 10 shares of {w}", "I want to own {w} stock, not a perp"],
           ["can you buy {w} shares for me", "I'd like actual {w} stock"])


def sc_stock_shares(rng, mode):
    sym = rng.choice(STOCK_ORDER[:10])
    c = Conv(rng, DriftWorld(), "Face ID" if rng.random() < 0.7 else "Touch ID")
    word = alias_stock(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, SHARES, mode, {"w": word}, "sh")
    c.user(txt, tpl)
    name = STOCKS[sym]["name"]
    c.say(rng.choice([f"I can't buy real {name} shares. What I can trade is a perp that tracks {name}, with leverage from 1x, open 24/7. "
                      f"Want me to prepare a long or a short on it? Tell me the size and leverage.",
                      f"Hermit doesn't hold real shares. {sym} is available as a perp that tracks the stock and trades around the clock. "
                      f"If you want that, tell me long or short, the size and the leverage."]))
    return c.ex("stock_shares_nearmiss", "stocks", ["right_tool", "facts_ok"], must=[r"\bperp\b", r"track"],
                forbid=[r"(?i)can'?t be traded", r"(?i)cannot be traded"], extra={"expect_no_tool": True, "knowledge": True})


CRYPTO_PRICE = T(["price of {w}", "what's {w} at?", "{w} price", "how much is {w}", "{w} price now", "where's {w} trading", "check {w} for me",
                  "how's {w} doing today", "what's {w} worth right now", "{w}?"],
                 ["what is {w} trading at", "where is {w} right now?", "latest on {w}"])


def sc_crypto_price(rng, mode):
    sym = rng.choice(list(CRYPTO))
    w = DriftWorld(prices={sym: crypto_entry(rng, sym)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    word = alias_crypto(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, CRYPTO_PRICE, mode, {"w": word}, "cp")
    c.user(txt, tpl)
    r = c.call(("get_price", {"symbol": sym}))
    name = CRYPTO[sym][5]
    ch = r["change_24h_pct"]
    c.say(rng.choice([f"{name} is {money(r['price'])}, {arrow(ch)} {plain(abs(ch))}% in the last 24 hours.",
                      f"{name} is trading at {money(r['price'])}. It's {arrow(ch)} {plain(abs(ch))}% over the last 24 hours."]))
    return c.ex("crypto_price_nearmiss", "stocks", ["right_tool", "exact_digits", "no_invented_price"], forbid=[r"(?i)\bperp\b", r"(?i)trade\.xyz"])


def sc_crypto_price_meme(rng, mode):
    ex = sc_crypto_price(rng, mode)
    ex["meta"]["area"] = "memecoins"
    return ex


def sc_crypto_trade(rng, mode):
    sym = rng.choice(["BTC", "ETH", "SOL", "HYPE"])
    w = DriftWorld(prices={sym: crypto_entry(rng, sym)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    side, size, lev = rng.choice(["long", "long", "short"]), rng.choice(SIZES), rng.choice([2, 3, 5, 5, 10, 20])
    a = trade_args(rng, sym, w.prices[sym]["price"], side, size, lev, limit_p=0.15, sl_p=0.25, tp_p=0.15)
    word = alias_crypto(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, STK_TRADE, mode, {"sd": side, "w": word, "sz": size_text(rng, size), "lev": lev, "tail": tail_text(rng, mode, a)}, "ct")
    c.user(txt, tpl)
    c.call(("prepare_trade", a))
    c.say(f"I've prepared {trade_text(a)}. {done(rng, c.bio)}")
    return c.ex("crypto_trade_nearmiss", "stocks", ["right_tool", "exact_digits"], must=[r"\bprepared\b"], forbid=[r"(?i)\bperp\b", r"(?i)tracks"])


# ============================================================ area 2: memecoins (per-token prices, kPEPE-style markets)
MEME_PRICE = T(["price of {w}", "what's {w} at?", "how's {w} doing", "{w} price rn", "is {w} up today", "{w} price?", "what's {w} trading at",
                "how much is {w} now", "check {w} for me", "where is {w} at", "{w}?", "give me the {w} price"],
               ["what is {w} worth at the moment", "{w} quote please", "how is {w} looking today"])
MEME_MULTI = T(["{w} and {w2} prices?", "price of {w} and {w2}", "check {w} and {w2} for me", "{w} vs {w2} today"],
               ["where are {w} and {w2} trading", "{w} and {w2}, how are they doing"])
MEME_ASK = T(["is that per token?", "why does hyperliquid show something different", "what's the price per 1000", "is that for one {s} or 1000?",
              "per 1000?", "what does the k mean in k{s}", "wait is that a single {s}", "and the contract price?"],
             ["is that the price of one coin or a bundle?", "so what does the exchange show for {s}?"])


def sc_meme_price(rng, mode):
    multi = rng.random() < 0.15
    syms = rng.sample(MEME_SYMS, 2 if multi else 1)
    w = DriftWorld(prices={s: meme_entry(rng, s) for s in syms})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    words = [alias_meme(rng, s, mode) for s in syms]
    c.slots = [{"word": wd, "symbol": s} for wd, s in zip(words, syms)]
    if multi:
        txt, tpl = fill(rng, MEME_MULTI, mode, {"w": words[0], "w2": words[1]}, "mm")
    else:
        txt, tpl = fill(rng, MEME_PRICE, mode, {"w": words[0]}, "mp")
    c.user(txt, tpl)
    rs = c.call(*[("get_price", {"symbol": s}) for s in syms])
    rs = rs if isinstance(rs, list) else [rs]
    bits = [f"{r['symbol']} is {money(r['price'])}, {arrow(r['change_24h_pct'])} {plain(abs(r['change_24h_pct']))}%" for r in rs]
    if multi:
        c.say(f"{bits[0]}, and {bits[1]}, over the last 24 hours.")
        return c.ex("meme_price_multi", "memecoins", ["right_tool", "exact_digits", "rounded_ok", "no_invented_price"],
                    forbid=[r"\d[eE]-\d", r"(?i)\bk(?:PEPE|SHIB|BONK|FLOKI)\b", r"(?i)per 1,?000"])
    r = rs[0]
    s = rng.choice([f"{bits[0]} in the last 24 hours.", f"{bits[0]} over the last 24 hours."])
    if rng.random() < 0.3:
        c.say(s)
        return c.ex("meme_price", "memecoins", ["right_tool", "exact_digits", "rounded_ok", "no_invented_price"],
                    forbid=[r"\d[eE]-\d", r"(?i)\bk(?:PEPE|SHIB|BONK|FLOKI)\b", r"(?i)per 1,?000"])
    c.say(s)
    if rng.random() < 0.4:                  # asked: now the unit may be explained
        t2, tpl2 = fill(rng, MEME_ASK, mode, {"s": syms[0]}, "ma", deco=False)
        c.user(t2, tpl2)
        k = "k" + syms[0]
        c.say(rng.choice([
            f"The {money(r['price'])} I gave is the price of one {syms[0]}. Hyperliquid's market is {k}, which counts 1,000 {syms[0]} as one unit, so its contract price is {money(r['contract_price'])} per 1,000.",
            f"That's per single {syms[0]}. The exchange lists {k}, where one unit is 1,000 {syms[0]}, so you'd see {money(r['contract_price'])} there."]))
        return c.ex("meme_unit_asked", "memecoins", ["exact_digits", "rounded_ok", "facts_ok"],
                    must=[re.escape(money(r["contract_price"])), r"1,000", r"k" + syms[0]], forbid=[r"\d[eE]-\d"], extra={"expect_no_tool": True})
    return c.ex("meme_price", "memecoins", ["right_tool", "exact_digits", "rounded_ok", "no_invented_price"],
                forbid=[r"\d[eE]-\d", r"(?i)\bk(?:PEPE|SHIB|BONK|FLOKI)\b", r"(?i)per 1,?000"])


def sig2(x):
    return float(f"{x:.2g}")


def meme_trade_args(rng, sym, price, side, size, lev):
    """Per-token prices at 2 significant digits. Always at least one price so the app rejects the call."""
    lo = 1 if side == "long" else -1
    case = rng.choice(["limit", "limit", "stop", "stop", "limit_stop", "stop_tp"])
    a = {"coin": sym, "side": side, "size_usd": size, "leverage": lev, "order_type": "limit" if case.startswith("limit") else "market"}
    if case.startswith("limit"):
        a["limit_price"] = sig2(price * (1 - lo * rng.uniform(0.01, 0.05)))
    ref = a.get("limit_price", price)
    if "stop" in case:
        a["stop_loss"] = sig2(ref * (1 - lo * rng.uniform(0.06, 0.14)))
    if case == "stop_tp":
        a["take_profit"] = sig2(ref * (1 + lo * rng.uniform(0.08, 0.2)))
    return a


PRICE_KEYS = ("limit_price", "take_profit", "stop_loss")
PRICE_NAMES = {"limit_price": "limit", "take_profit": "take profit", "stop_loss": "stop loss"}
YES = T(["yes", "yes please", "do it", "yeah go ahead", "ok do that", "sure", "yep", "go for it", "that works", "ok use those"],
        ["sounds right, go on", "yes use those prices"])


def price_list(a):
    return join_and([f"{PRICE_NAMES[k]} {plain(a[k])}" for k in PRICE_KEYS if k in a])


def sc_meme_reject(rng, mode, branch=None):
    sym = rng.choice(MEME_SYMS)
    w = DriftWorld(prices={sym: meme_entry(rng, sym)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    px = w.prices[sym]["price"]
    side, size, lev = rng.choice(["long", "long", "short"]), rng.choice([20, 25, 50, 100, 100, 150, 200, 300, 500]), rng.choice([2, 3, 3, 5])
    a = meme_trade_args(rng, sym, px, side, size, lev)
    word = alias_meme(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, STK_TRADE, mode, {"sd": side, "w": word, "sz": size_text(rng, size), "lev": lev, "tail": tail_text(rng, mode, a)}, "mr")
    c.user(txt, tpl)
    r = c.call(("prepare_trade", a))
    assert r.get("prepared") is False and "per 1,000" in r["error"]
    pick = rng.choices(["ask_calc", "convert", "ask_plain"], [0.38, 0.42, 0.2])[0]
    branch = branch or pick
    k = "k" + sym
    if branch == "ask_plain":
        c.say(rng.choice([f"{sym} trades here as {k}, where one unit is 1,000 {sym}, so the prices have to be per 1,000, not per token. What limit and stop do you want per 1,000 {sym}?",
                          f"The exchange lists {sym} as {k}, 1,000 {sym} per unit, so it wants prices per 1,000. Can you give me those prices per 1,000 {sym}?"]), ask=True)
        return c.ex("meme_reject_ask", "memecoins", ["right_tool", "exact_digits", "facts_ok"],
                    must=[r"per 1,000", re.escape(k)], forbid=[r"\d[eE]-\d", r"(?i)\bprepared\b"],
                    extra={"no_retry_same": True, "first_turn_gold_is_per_token": True})
    keys = [kk for kk in PRICE_KEYS if kk in a]
    calls = [("calculator", {"expression": f"{plain(a[kk])} * 1000"}) for kk in keys]
    rs = c.call(*calls)
    rs = rs if isinstance(rs, list) else [rs]
    a2 = dict(a)
    for kk, rr in zip(keys, rs):
        a2[kk] = nn(float(rr["result"]))
    conv = price_list(a2)
    if branch == "ask_calc":
        c.say(rng.choice([f"{sym} trades as {k} here, so prices are per 1,000 {sym}. Multiplying yours by 1,000 gives {conv}. Should I use those?",
                          f"The exchange wants prices per 1,000 {sym} because the market is {k}. Your prices times 1,000 would be {conv}. Want me to prepare it with those?"]), ask=True)
        if rng.random() < 0.6:
            t2, tpl2 = fill(rng, YES, mode, {}, "my", deco=False)
            c.user(t2, tpl2)
            c.call(("prepare_trade", a2))
            c.say(f"I've prepared {trade_text(a2, pf=plain)}. Those prices are per 1,000 {sym}. {done(rng, c.bio)}")
            return c.ex("meme_reject_ask_confirm", "memecoins", ["right_tool", "exact_digits", "rounded_ok", "facts_ok"],
                        must=[r"\bprepared\b", r"per 1,000"], forbid=[r"\d[eE]-\d"], extra={"no_retry_same": True})
        return c.ex("meme_reject_ask", "memecoins", ["right_tool", "exact_digits", "rounded_ok", "facts_ok"],
                    must=[r"1,000", re.escape(k)], forbid=[r"\d[eE]-\d", r"(?i)\bprepared\b"], extra={"no_retry_same": True})
    c.call(("prepare_trade", a2))
    c.say(f"I converted your prices to per 1,000 {sym}, since the market is {k} (I multiplied by 1,000): {conv}. "
          f"I've prepared {trade_text(a2, pf=plain)}. {done(rng, c.bio)}")
    return c.ex("meme_reject_convert", "memecoins", ["right_tool", "exact_digits", "rounded_ok", "facts_ok"],
                must=[r"1,000", r"(?i)multiplied|converted", r"\bprepared\b"], forbid=[r"\d[eE]-\d"], extra={"no_retry_same": True})


def sc_meme_market(rng, mode):
    sym = rng.choice(MEME_SYMS)
    w = DriftWorld(prices={sym: meme_entry(rng, sym)})
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    side, size, lev = rng.choice(["long", "short"]), rng.choice([20, 25, 50, 100, 150, 200]), rng.choice([2, 3, 5])
    a = {"coin": sym, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
    word = alias_meme(rng, sym, mode)
    c.slots = [{"word": word, "symbol": sym}]
    txt, tpl = fill(rng, STK_TRADE, mode, {"sd": side, "w": word, "sz": size_text(rng, size), "lev": lev, "tail": ""}, "mk")
    c.user(txt, tpl)
    c.call(("prepare_trade", a))
    c.say(f"I've prepared {trade_text(a)}. {done(rng, c.bio)}")
    return c.ex("meme_market_trade", "memecoins", ["right_tool", "exact_digits"], must=[r"\bprepared\b"], forbid=[r"\d[eE]-\d"])


KWHAT = T(["what is k{s}", "what's k{s} on hyperliquid", "why is it called k{s}", "k{s} vs {s}, what's the difference", "what does k{s} mean",
           "explain k{s} to me", "is k{s} the same as {s}"], ["can you tell me what k{s} is", "why k{s} and not just {s}"])


def sc_meme_whatis(rng, mode):
    sym = rng.choice(MEME_SYMS)
    c = Conv(rng, DriftWorld(), "Face ID" if rng.random() < 0.7 else "Touch ID")
    txt, tpl = fill(rng, KWHAT, mode, {"s": sym}, "mw")
    c.user(txt, tpl)
    c.say(rng.choice([f"k{sym} is Hyperliquid's market for {sym}. One unit of it is 1,000 {sym}, because a single {sym} costs a tiny fraction of a cent, so its price is quoted per 1,000. When you ask me for the price I give you the price of one {sym}.",
                      f"It's the same coin. Hyperliquid lists {sym} as k{sym}, where one unit is 1,000 {sym}, so the numbers are not tiny. I quote the price per single {sym} unless you ask for the market price."]))
    return c.ex("meme_kwhat_nearmiss", "memecoins", ["facts_ok"], must=[r"1,000"], forbid=[r"\d[eE]-\d"], extra={"expect_no_tool": True, "knowledge": True})


# ============================================================ area 3: multi-wallet balances
BAL_LIST = T(["what's in my wallet", "show my balances", "what do I have", "my holdings?", "balance?", "what am I holding",
              "how much do I have in total", "show me everything I own", "what's my balance", "check my wallet",
              "what's in here", "how much money do I have"],
             ["what are my holdings right now", "can you list what I hold", "balances please"])
BAL_TOKEN = T(["how much {a} do I have", "do I have any {a}", "{a} balance?", "how much {a} am I holding", "got any {a}?", "what's my {a}"],
              ["what is my {a} balance", "am I holding {a}?"])
BAL_CHAIN = T(["what do I have on {c}", "my {c} balance", "show me what's on {c}", "anything on {c}?", "how much on {c}"],
              ["what is on {c} for me", "balances on {c}"])
PRIMARY_ASSETS = ["USDC", "USDT", "ETH", "SOL", "BNB"]


def amount_for(rng, sym):
    if sym in ("USDC", "USDT"):
        return plain(rng.choice([round(rng.uniform(5, 4000), 2), round(rng.uniform(5, 900), 1), rng.choice([20, 40, 50, 100, 250, 500])]))
    if sym == "ETH":
        return plain(round(rng.uniform(0.01, 2.5), rng.choice([3, 4])))
    if sym == "SOL":
        return plain(round(rng.uniform(0.2, 30), rng.choice([2, 3])))
    if sym == "BNB":
        return plain(round(rng.uniform(0.05, 6), 3))
    return plain(round(rng.uniform(100, 5000), 1))


def build_wallet(rng, mode, multi=True, empty=False, k=None):
    names = WALLET_EV if mode == "ev" else WALLET_TR
    name = rng.choice(names)
    others = rng.sample([n for n in names if n != name], rng.choice([1, 2, 2, 3]))
    pairs = []
    for sym in rng.sample(list(HOLD_TOKENS), len(HOLD_TOKENS)):
        for ch in rng.sample(HOLD_TOKENS[sym], len(HOLD_TOKENS[sym])):
            pairs.append((sym, ch))
    rows = []
    if not empty:
        want = k or rng.choice([1, 2, 2, 3, 3, 4])
        used = set()
        for sym, ch in pairs:
            if len(rows) >= want:
                break
            if rng.random() < 0.5 and (sym, ch) not in used:
                used.add((sym, ch))
                amt = amount_for(rng, sym)
                rows.append({"amount": amt, "chain": ch, "symbol": sym, "usd": nn(round(float(amt) * HOLD_PX[sym], 2))})
        if not rows:
            sym, ch = pairs[0]
            amt = amount_for(rng, sym)
            rows.append({"amount": amt, "chain": ch, "symbol": sym, "usd": nn(round(float(amt) * HOLD_PX[sym], 2))})
    return {"name": name, "others": others, "holdings": rows, "multi": multi,
            "near_account": rng.choice([None, None, None, "alice.near", "mkdev.near", "sam.near"])}


def join_and(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def hold_phrase(h):
    return f"{h['amount']} {h['symbol']} on {chain_label(h['chain'])} ({money(h['usd'])})"


def wallet_close(rng, name, others):
    o = join_and(others)
    return rng.choice([f"These numbers are for {name} only. You also have {o}; switch to one in Wallets and ask again to see it.",
                       f"That's just your {name} wallet. {cap(o)} {'is a separate wallet' if len(others) == 1 else 'are separate wallets'} that I can't see until you switch to {'it' if len(others) == 1 else 'one'}.",
                       f"This is the {name} wallet. Your other wallets ({o}) are not included."])


def sc_wallet(rng, mode, multi=True, empty=False):
    wd = build_wallet(rng, mode, multi=multi, empty=empty)
    w = DriftWorld(wallet=wd)
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    hold = wd["holdings"]
    qk = "list" if empty else rng.choices(["list", "token", "chain"], [0.5, 0.3, 0.2])[0]
    if qk == "token":
        sym = rng.choice([h["symbol"] for h in hold]) if rng.random() < 0.8 else rng.choice([s for s in PRIMARY_ASSETS if s not in {h["symbol"] for h in hold}] or ["SOL"])
        txt, tpl = fill(rng, BAL_TOKEN, mode, {"a": rng.choice([sym, sym.lower()])}, "bt")
        args = {}
    elif qk == "chain":
        ch = rng.choice([h["chain"] for h in hold])
        txt, tpl = fill(rng, BAL_CHAIN, mode, {"c": rng.choice([ch, chain_label(ch)])}, "bc")
        args = {"chain": ch}
    else:
        txt, tpl = fill(rng, BAL_LIST, mode, {}, "bl")
        args = {}
    c.user(txt, tpl)
    r = c.call(("get_balances", args))
    rows = r["holdings"]
    name, others = wd["name"], wd["others"]
    tail = wallet_close(rng, name, others) if multi else ""
    where = f"in {name}" if multi else "in your wallet"
    if not rows:
        if qk == "token":
            body = f"You have no {sym} {where}."
        elif qk == "chain":
            body = f"You have nothing on {chain_label(args['chain'])} {where}."
        else:
            body = f"{name} is empty." if multi else "Your wallet is empty."
        if multi:
            body += f" You also have {join_and(others)}, which may hold funds; switch to one in Wallets and ask again."
        else:
            body += " I can open Add funds if you want to put something in."
        c.say(body)
        kind = "wallet_empty" if multi else "wallet_single_nearmiss"
        return c.ex(kind, "wallets", ["right_tool", "exact_digits", "facts_ok"], must=([re.escape(name)] + [re.escape(o) for o in others] if multi else []),
                    forbid=[] if multi else [r"(?i)selected wallet|other wallets|switch to"])
    if qk == "token":
        sel = [h for h in rows if h["symbol"] == sym]
        if not sel:
            body = f"You have no {sym} {where}."
        elif len(sel) > 1:
            expr = " + ".join(h["amount"] for h in sel)
            tot = c.call(("calculator", {"expression": expr}))["result"]
            parts = " and ".join(f"{h['amount']} on {chain_label(h['chain'])}" for h in sel)
            body = f"You have {tot} {sym} {where}: {parts}."
        else:
            h = sel[0]
            body = f"You have {h['amount']} {sym} on {chain_label(h['chain'])} {where}, worth {money(h['usd'])}."
    else:
        items = join_and([hold_phrase(h) for h in rows])
        if qk == "chain":
            body = f"On {chain_label(args['chain'])} {where} you have {items}."
        else:
            body = f"{cap(where)} you have {items}."
        if len(rows) > 1 and rng.random() < 0.7:
            tot = c.call(("calculator", {"expression": " + ".join(plain(h["usd"]) for h in rows)}))["result"]
            body += f" That's {money(float(tot))} in total."
    c.say(body + (" " + tail if tail else ""))
    kind = "wallet_multi" if multi else "wallet_single_nearmiss"
    return c.ex(kind, "wallets", ["right_tool", "exact_digits", "facts_ok"],
                must=([re.escape(name)] + [re.escape(o) for o in others]) if multi else [],
                forbid=[] if multi else [r"(?i)selected wallet|other wallets|switch to"], extra={"wallet": name})


def sc_wallet_multi(rng, mode):
    return sc_wallet(rng, mode, True)


def sc_wallet_single(rng, mode):
    return sc_wallet(rng, mode, False)


def sc_wallet_empty(rng, mode):
    return sc_wallet(rng, mode, True, empty=True)


# ============================================================ area 4: trigger rows in hyperliquid_account
PX_DP = {"BTC": 0, "ETH": 1, "SOL": 2, "HYPE": 3}
SZ_DP = {"BTC": 4, "ETH": 3, "SOL": 2, "HYPE": 1}


def build_hl(rng, triggers=True, limits=None, positions=True):
    coins = rng.sample(list(PERP_COINS), rng.choice([1, 2, 2])) if positions else []
    pos, orders = [], []
    for coin in coins:
        lo, hi = PERP_COINS[coin]
        entry = nn(round(rng.uniform(lo, hi), PX_DP[coin]))
        side = rng.choice(["long", "long", "short"])
        lev = rng.choice([2, 3, 5, 5, 10])
        size = nn(round(rng.choice([200, 500, 1000, 1500, 2500, 4000]) / entry, SZ_DP[coin]))
        if size == 0:
            size = nn(round(10 ** -SZ_DP[coin], SZ_DP[coin]))
        s = 1 if side == "long" else -1
        liq = nn(round(entry * (1 - s * 0.9 / lev), PX_DP[coin])) if rng.random() < 0.9 else None
        pnl = nn(round(rng.uniform(-90, 140), 2))
        pos.append({"coin": coin, "entry": entry, "leverage": lev, "liquidation": liq, "side": side, "size": size, "unrealised_pnl": pnl})
        if triggers:
            sell = "sell" if side == "long" else "buy"
            below, above = ("below", "above") if side == "long" else ("above", "below")
            if rng.random() < 0.75:
                px = nn(round(entry * (1 - s * rng.uniform(0.05, 0.14)), PX_DP[coin]))
                orders.append({"coin": coin, "is_trigger": True, "order_type": "Stop Market", "price": px, "side": sell, "size": size,
                               "trigger_condition": f"Price {below} {plain(px)}", "whole_position": True})
            if rng.random() < 0.5:
                px = nn(round(entry * (1 + s * rng.uniform(0.07, 0.25)), PX_DP[coin]))
                orders.append({"coin": coin, "is_trigger": True, "order_type": "Take Profit Market", "price": px, "side": sell, "size": size,
                               "trigger_condition": f"Price {above} {plain(px)}", "whole_position": True})
    nl = rng.choice([0, 1, 1, 2]) if limits is None else limits
    for coin in rng.sample(list(PERP_COINS), min(nl, len(PERP_COINS))):
        lo, hi = PERP_COINS[coin]
        px = nn(round(rng.uniform(lo, hi), PX_DP[coin]))
        sz = nn(round(rng.choice([150, 400, 900, 1800]) / px, SZ_DP[coin])) or nn(10 ** -SZ_DP[coin])
        orders.append({"coin": coin, "is_trigger": False, "order_type": "Limit", "price": px, "side": rng.choice(["buy", "sell"]), "size": sz})
    rng.shuffle(orders)
    value = round(rng.uniform(300, 20000), 2)
    used = round(value * rng.uniform(0.1, 0.5), 2) if pos else 0
    hl = {"mode": rng.choice(["unified", "unified", "default", None]), "value": nn(value if pos or orders else 0), "available": nn(round(value - used, 2)) if (pos or orders) else 0,
          "margin_used": nn(used), "positions": pos, "orders": orders}
    return hl


def _orders(hl):
    return hl["orders"] if "orders" in hl else hl["open_orders"]


def limits_of(hl):
    return [o for o in _orders(hl) if not o["is_trigger"]]


def trigs_of(hl):
    return [o for o in _orders(hl) if o["is_trigger"]]


def tkind(t):
    return "stop loss" if t["order_type"].startswith("Stop") else "take profit"


def trig_phrase(t, pos):
    lvl = money(t["price"])
    below = "below" in t["trigger_condition"]
    act = "sells" if t["side"] == "sell" else "buys back"
    scope = "your whole position" if t.get("whole_position") else f"{plain(t['size'])} {t['coin']}"
    return (f"a {tkind(t)} on your {t['coin']} {pos['side']} at {lvl}: it {act} {scope} if the price "
            f"{'falls below' if below else 'rises above'} that level")


def trig_short(t):
    return f"a {tkind(t)} at {money(t['price'])}"


def limit_phrase(o):
    return f"a limit {o['side']} of {plain(o['size'])} {o['coin']} at {money(o['price'])}"


def pos_of(hl, coin):
    return next((p for p in hl["positions"] if p["coin"] == coin), None)


TRG_ORDERS = T(["what orders do I have open", "any open orders?", "show my open orders", "list my orders", "do i have anything pending on hyperliquid",
                "what's open on my account besides positions", "what orders are live"],
               ["which orders are open right now", "anything pending in my perps account?"])
TRG_COUNT = T(["how many open orders do I have", "how many orders are live right now", "count my open orders", "number of orders open?"],
              ["how many orders are sitting on my account"])
TRG_STOP = T(["do I have a stop loss on {c}", "is my {c} position protected", "what's my {c} stop at", "did my {c} stop go on", "have I got a stop on {c}?",
              "is there a stop on my {c}"], ["is my {c} covered by a stop?", "where is the stop on {c}"])
TRG_TP = T(["do I have a take profit on {c}", "what's my {c} take profit", "is there a tp on my {c}", "have I got a take profit set on {c}?"],
           ["is a take profit set for {c}", "where is my {c} tp"])
TRG_PROT = T(["am I covered if everything dumps", "which of my positions have a stop", "are my positions protected", "do all my positions have stops",
              "what protects my positions"], ["which positions are unprotected", "how protected am I right now"])
TRG_OVER = T(["how's my hyperliquid account", "show me my positions and what protects them", "give me a rundown of my perps", "how are my perps doing",
              "status of my hyperliquid account"], ["quick summary of my perps account", "what's going on in my trading account"])


def sc_triggers(rng, mode, triggers=True):
    hl = build_hl(rng, triggers=triggers)
    if triggers and not trigs_of(hl):                       # make sure the point of the row exists
        for p in hl["positions"][:1]:
            s = 1 if p["side"] == "long" else -1
            px = nn(round(p["entry"] * (1 - s * 0.08), PX_DP[p["coin"]]))
            hl["orders"].append({"coin": p["coin"], "is_trigger": True, "order_type": "Stop Market", "price": px,
                                 "side": "sell" if p["side"] == "long" else "buy", "size": p["size"],
                                 "trigger_condition": f"Price {'below' if s == 1 else 'above'} {plain(px)}", "whole_position": True})
    if not hl["positions"]:
        triggers = False
    w = DriftWorld(hl=hl)
    c = Conv(rng, w, "Face ID" if rng.random() < 0.7 else "Touch ID")
    qk = rng.choices(["orders", "count", "stop", "tp", "prot", "over"], [0.26, 0.14, 0.2, 0.1, 0.14, 0.16])[0]
    coin = None
    if qk in ("stop", "tp"):
        coin = rng.choice([p["coin"] for p in hl["positions"]]) if (hl["positions"] and rng.random() < 0.85) else rng.choice([x for x in PERP_COINS if not pos_of(hl, x)] or ["SOL"])
    pool = {"orders": TRG_ORDERS, "count": TRG_COUNT, "stop": TRG_STOP, "tp": TRG_TP, "prot": TRG_PROT, "over": TRG_OVER}[qk]
    txt, tpl = fill(rng, pool, mode, {"c": coin} if coin else {}, "tg" + qk[0])
    c.user(txt, tpl)
    r = c.call(("hyperliquid_account", {}))
    lims, trg = limits_of(r), trigs_of(r)
    n = len(lims)
    nouns = "order" if n == 1 else "orders"
    must, forbid = [], [r"(?i)orders? you(?:'ve)? placed", r"(?i)account_mode|unified"]
    if qk == "orders":
        s = (f"You have {WORDS[n]} open {nouns}: " + "; ".join(limit_phrase(o) for o in lims) + ".") if n else "You have no open orders."
        if len(trg) == 1:
            s += f" Separately, you have {trig_phrase(trg[0], pos_of(r, trg[0]['coin']))}. A {tkind(trg[0])} is a protection on the position, not counted as an order."
            must += [r"(?i)stop loss|take profit"]
        elif trg:
            short = [f"a {tkind(t)} on your {t['coin']} {pos_of(r, t['coin'])['side']} at {money(t['price'])}" for t in trg]
            s += " Separately, you have " + join_and(short) + ". Those are protections on your positions, not counted as orders."
            must += [r"(?i)stop loss|take profit"]
    elif qk == "count":
        s = f"You have {WORDS[n]} open {nouns}."
        if trg:
            s += " Your " + join_and([f"{tkind(t)} on {t['coin']}" for t in trg]) + (" is a protection on the position" if len(trg) == 1 else " are protections on the positions") + ", not counted in that number."
    elif qk in ("stop", "tp"):
        want = "stop loss" if qk == "stop" else "take profit"
        pos = pos_of(r, coin)
        hit = [t for t in trg if t["coin"] == coin and tkind(t) == want]
        other = [t for t in trg if t["coin"] == coin and tkind(t) != want]
        if pos is None:
            s = f"You don't have an open {coin} position on Hyperliquid, so there is no {want} to check."
        elif hit:
            s = f"Yes. Your {coin} {pos['side']} has {trig_phrase(hit[0], pos)}."
            if other:
                s += f" It also has {trig_short(other[0])}."
        else:
            s = f"No. Your {coin} {pos['side']} has no {want}" + (f", only {trig_short(other[0])}" if other else "") + ". "
            s += rng.choice(["I can set one if you tell me the price.", "Tell me a price and I can set one."])
        must += [re.escape(want)] if (pos is not None) else []
    elif qk == "prot":
        if not r["positions"]:
            s = "You have no open positions on Hyperliquid, so there is nothing to protect."
        else:
            bits = []
            for p in r["positions"]:
                st = [t for t in trg if t["coin"] == p["coin"] and tkind(t) == "stop loss"]
                bits.append(f"your {p['coin']} {p['side']} has a stop loss at {money(st[0]['price'])}" if st else f"your {p['coin']} {p['side']} has no stop loss")
            s = cap(join_and(bits)) + "."
            if any("no stop loss" in b for b in bits):
                s += " " + rng.choice(["I can set one if you tell me the price.", "Tell me a price and I can add the missing stop."])
        must += [r"(?i)stop loss"]
    else:
        if not r["positions"]:
            s = f"Your Hyperliquid account is worth {money(r['account_value'])} with {money(r['available_usdc'])} available, and there are no open positions."
        else:
            s = f"Your Hyperliquid account is worth {money(r['account_value'])}, with {money(r['available_usdc'])} available. "
            parts = []
            for p in r["positions"]:
                pnl = p["unrealised_pnl"]
                t = f"{p['coin']} {p['side']}, {plain(p['size'])} at {plain(p['leverage'])}x, {'up' if pnl >= 0 else 'down'} {money(abs(pnl))}"
                if p["liquidation"] is not None:
                    t += f", liquidation {money(p['liquidation'])}"
                st = [x for x in trg if x["coin"] == p["coin"] and tkind(x) == "stop loss"]
                tp = [x for x in trg if x["coin"] == p["coin"] and tkind(x) == "take profit"]
                t += f", stop loss {money(st[0]['price'])}" if st else ", no stop loss"
                if tp:
                    t += f", take profit {money(tp[0]['price'])}"
                parts.append(t)
            s += "; ".join(parts) + "."
        if n:
            s += f" Plus {WORDS[n]} open {nouns}."
        must += [re.escape(money(r["account_value"]))]
    c.say(s)
    kind = "trigger_" + qk if triggers else "trigger_none_nearmiss_" + qk
    return c.ex(kind, "triggers", ["right_tool", "exact_digits", "facts_ok"], must=must, forbid=forbid,
                extra={"trigger_count": len(trg), "limit_count": n})


def sc_triggers_none(rng, mode):
    return sc_triggers(rng, mode, triggers=False)


# ============================================================ area 5: get_yields v3 fields
YP = {  # protocol: (type, chains, depositable, multiplier)
    "Aave v3": ("lending", ["Ethereum", "Base", "Arbitrum"], True, 0.7),
    "Morpho": ("lending", ["Ethereum", "Base"], False, 1.0),
    "Compound v3": ("lending", ["Ethereum", "Arbitrum", "Polygon"], False, 0.65),
    "Spark": ("lending", ["Ethereum"], False, 0.8),
    "Euler": ("lending", ["Ethereum", "Base"], False, 1.1),
    "Uniswap v3": ("amm", ["Ethereum", "Base", "Arbitrum"], False, 1.0),
    "Aerodrome": ("amm", ["Base"], False, 1.0),
}
PW = {"Aave v3": ["aave", "Aave"], "Morpho": ["morpho", "Morpho"], "Compound v3": ["compound", "Compound"], "Spark": ["spark", "Spark"],
      "Euler": ["euler", "Euler"], "Uniswap v3": ["uniswap", "Uniswap"], "Aerodrome": ["aerodrome", "Aerodrome"]}
ASSET_WORDS = {"USDC": ["usdc", "USDC", "Usdc"], "USDT": ["usdt", "USDT", "tether"], "DAI": ["dai", "DAI"], "ETH": ["eth", "ETH", "ethereum"]}
ASSET_WORDS_EV = {"USDC": ["usd coin", "$usdc"], "USDT": ["$usdt", "Tether"], "DAI": ["$dai"], "ETH": ["$eth", "Eth"]}


def build_pools(rng, asset, only=None, chain=None):
    rows = []
    stable = asset != "ETH"
    for proto, (typ, chains, dep, mult) in YP.items():
        if only and proto not in only:
            continue
        for ch in chains:
            if chain and ch.lower() != chain.lower():
                continue
            if rng.random() > (0.92 if proto == "Aave v3" else 0.7):
                continue
            lo, hi = ((2.5, 9.0) if stable else (1.0, 3.5)) if typ == "lending" else ((3.0, 14.0) if stable else (4.0, 20.0))
            base = round(rng.uniform(lo, hi) * mult, 2)
            rew = 0.0
            if proto == "Aave v3" and ch != "Ethereum" and rng.random() < 0.4:
                rew = round(rng.uniform(0.05, 0.9), 2)
            elif proto in ("Morpho", "Euler", "Compound v3") and rng.random() < 0.35:
                rew = round(rng.uniform(0.1, 1.6), 2)
            elif proto == "Aerodrome":
                rew = round(rng.uniform(4, 20), 2)
            if typ == "lending":
                d30 = base * rng.uniform(0.88, 1.12) + rew * rng.uniform(0.5, 1.3)
            else:
                d30 = base * rng.uniform(0.6, 1.5) + rew * rng.uniform(0.4, 1.4)
            label = asset if typ == "lending" else (f"{asset}-WETH" if asset != "ETH" else "ETH-USDC")
            rows.append({"apy_30d_pct": nn(round(d30, 2)), "apy_base_pct": nn(base), "apy_pct": nn(round(base + rew, 2)),
                         "apy_reward_pct": nn(rew), "asset": label, "can_deposit_in_hermit": dep, "chain": ch,
                         "il_risk": "no" if typ == "lending" else "yes",
                         "pool_meta": rng.choice(["0.05%", "0.3%", "0.01%"]) if proto == "Uniswap v3" else None, "protocol": proto,
                         "tvl_usd": int(rng.uniform(5e7, 2e9) if proto == "Aave v3" else rng.uniform(4e6, 4e8) if typ == "amm" else rng.uniform(2e7, 9e8))})
    return rows


def top_rows(pools, asset=None, chain=None):
    return DriftWorld(pools=pools).t_get_yields(asset=asset, chain=chain)["pools"]


def row_desc(r):
    return f"{r['protocol']} on {r['chain']}" + (f" ({r['pool_meta']})" if r.get("pool_meta") else "")


def apy(r, key="apy_pct"):
    return plain(r[key]) + "%"


def pick_asset_word(rng, asset, mode):
    pool = ASSET_WORDS_EV[asset] if (mode == "ev" and rng.random() < 0.5) else ASSET_WORDS[asset]
    return rng.choice(pool)


Y_RANK = T(["best {a} yield?", "where can I earn on my {a}", "highest apy on {a} right now", "any good yields for {a}", "what's the best rate for {a}",
            "{a} yield options?", "show me the top {a} pools", "where do I park {a} to earn something", "which pool pays the most for {a}"],
           ["what pays best on {a} at the moment", "top yields for {a}?", "I have some {a} sitting idle, where should it earn"])
Y_RANK_C = T(["best {a} yield on {c}?", "highest apy for {a} on {c}", "where can I earn on {a} on {c}", "top {a} pools on {c}"],
             ["what pays best on {c} for {a}", "{c} yields for {a}?"])
Y_DEP = T(["put {amt} {a} in the best yield", "deposit {amt} {a} where it earns the most", "earn on {amt} {a} please", "park {amt} {a} in the highest rate",
           "lend my {amt} {a} at the best rate", "i have {amt} {a}, put it to work at the top apy"],
          ["use {amt} {a} for the highest yield", "get the best return on {amt} {a}"])
Y_NAMED = T(["deposit {amt} {a} into {pw}", "put {amt} {a} in {pw}", "i want to lend {amt} {a} on {pw}", "{pw} deposit {amt} {a}", "supply {amt} {a} to {pw}"],
            ["move {amt} {a} into {pw}", "can you deposit {amt} {a} on {pw}"])
Y_IL = T(["is there impermanent loss on {pw} {a}", "any IL risk on the {pw} {a} pool?", "does {pw} have impermanent loss on {a}", "can I lose money from impermanent loss on {pw} {a}",
          "is {pw} {a} safe from IL", "{pw} {a}: impermanent loss?"], ["would I face impermanent loss in {pw} with {a}", "IL on the {pw} {a} pool?"])
Y_BR = T(["how much of the {pw} {a} apy is rewards", "is the {pw} {a} rate real or just incentives?", "what's base vs reward apy on {pw} {a}",
          "will the {pw} {a} apy last? how much is from rewards", "break down the {pw} {a} apy for me"],
         ["how much of {pw}'s {a} rate is incentives", "{pw} {a}: base apy and rewards?"])
Y_30 = T(["is the {pw} {a} rate stable", "what's the 30 day apy on {pw} {a}", "has {pw} {a} been steady or is it a spike", "30d average for {pw} {a}?",
          "is the {pw} {a} yield going to hold up"], ["has the {pw} {a} rate been consistent lately", "{pw} {a} apy over the last month?"])
Y_NEAR = [
    ("il", T(["what is impermanent loss", "explain impermanent loss", "what does IL mean in defi", "impermanent loss, what is it?"],
             ["can you explain what impermanent loss is", "what's IL in liquidity pools"]),
     ["Impermanent loss happens in liquidity pools on exchanges like Uniswap. When the two tokens in the pool move apart in price, the pool ends up with more of the one that fell, and you can finish with less than if you had just held both. Lending, as on Aave, has no impermanent loss: you deposit one asset and get it back with interest.",
      "It's a risk of providing liquidity to a pool of two tokens on an exchange. If their prices drift apart, you end up holding more of the cheaper one and less value than if you had simply held them. Aave is lending, not an AMM, so it doesn't have this."]),
    ("apy", T(["what's the difference between apy and apr", "apy vs apr?", "is apy the same as apr", "apr or apy which is better"],
              ["how do apy and apr differ"]),
     ["APR is the yearly rate without compounding. APY includes compounding, so it comes out a bit higher for the same pool. Most DeFi pages show APY."]),
    ("tvl", T(["what is tvl", "what does tvl mean in a pool", "tvl meaning?"], ["can you explain tvl"]),
     ["TVL is total value locked: how much money is deposited in a pool or protocol. A bigger number means more money is in it, but it says nothing about whether the rate will last."]),
    ("safe", T(["is aave safe", "can I trust aave", "is lending on aave risky", "is aave safe to deposit in"], ["how risky is aave really"]),
     ["Aave is one of the largest lending protocols and has been running for years, but nothing in DeFi is risk-free. The main risks are a smart contract bug, a market event, and the token you deposit losing value."]),
]


def make_yield_world(rng, asset, only=None, chain=None):
    return build_pools(rng, asset, only=only, chain=chain)


def first_row(rows, proto):
    return next((r for r in rows if r["protocol"] == proto), None)


def sc_yield_rank(rng, mode):
    asset = rng.choice(["USDC", "USDC", "USDT", "DAI", "ETH"])
    chain = rng.choice([None, None, "base", "arbitrum", "ethereum"])
    pools = build_pools(rng, asset)
    rows = top_rows(pools, asset, chain)
    if not rows:
        chain = None
        rows = top_rows(pools, asset, None)
    c = Conv(rng, DriftWorld(pools=pools), "Face ID" if rng.random() < 0.7 else "Touch ID")
    aw = pick_asset_word(rng, asset, mode)
    c.slots = [{"word": aw, "symbol": asset}]
    if chain:
        txt, tpl = fill(rng, Y_RANK_C, mode, {"a": aw, "c": rng.choice([chain, chain.capitalize()])}, "yc")
    else:
        txt, tpl = fill(rng, Y_RANK, mode, {"a": aw}, "yr")
    c.user(txt, tpl)
    args = {"asset": asset} if not chain else {"asset": asset, "chain": chain}
    r = c.call(("get_yields", args))
    rows = r["pools"]
    top = rows[0]
    dep = [x for x in rows if x["can_deposit_in_hermit"]]
    s = f"The highest rate is {row_desc(top)} at {apy(top)} on {top['asset']}."
    if top["il_risk"] == "yes":
        s += " It's a liquidity pool, so impermanent loss is possible."
    if top["can_deposit_in_hermit"]:
        s += " I can prepare a deposit into it."
        if top["apy_reward_pct"] > 0:
            s += f" {apy(top, 'apy_reward_pct')} of that is rewards, which can stop."
    elif dep:
        s += f" That one is in Earn, so I can't prepare it here. The best I can prepare is {row_desc(dep[0])} at {apy(dep[0])}."
    else:
        s += " None of the pools in this list can be prepared from chat, they're in Earn."
    if dep:
        s += " Want me to prepare a deposit?"
    c.say(s)
    must = [re.escape(row_desc(top)), re.escape(apy(top))]
    must += [r"(?i)\bearn\b"] if not top["can_deposit_in_hermit"] else [r"(?i)prepare"]
    return c.ex("yield_rank", "yields", ["right_tool", "exact_digits", "facts_ok"], must=must, forbid=[r"(?i)aave[^.]{0,50}impermanent loss is possible"],
                extra={"top_can_deposit": top["can_deposit_in_hermit"], "n_depositable": len(dep)})


def sc_yield_deposit(rng, mode):
    asset = rng.choice(["USDC", "USDC", "USDT", "DAI", "ETH"])
    amt = rng.choice([25, 50, 100, 150, 200, 250, 500, 1000]) if asset != "ETH" else rng.choice([0.05, 0.1, 0.25, 0.5, 1])
    variant = rng.choices(["top_dep", "top_not", "none_dep"], [0.35, 0.4, 0.25])[0]
    for _ in range(60):
        if variant == "top_dep":
            pools = build_pools(rng, asset, only=["Aave v3", "Compound v3", "Spark"])
        elif variant == "none_dep":
            pools = build_pools(rng, asset, only=["Morpho", "Euler", "Uniswap v3", "Aerodrome", "Compound v3"])
        else:
            pools = build_pools(rng, asset)
        rows = top_rows(pools, asset, None)
        if not rows:
            continue
        dep = [x for x in rows if x["can_deposit_in_hermit"]]
        if variant == "top_dep" and rows[0]["can_deposit_in_hermit"]:
            break
        if variant == "top_not" and dep and not rows[0]["can_deposit_in_hermit"]:
            break
        if variant == "none_dep" and not dep:
            break
    else:
        raise RuntimeError("no pool set for " + variant)
    c = Conv(rng, DriftWorld(pools=pools), "Face ID" if rng.random() < 0.7 else "Touch ID")
    aw = pick_asset_word(rng, asset, mode)
    c.slots = [{"word": aw, "symbol": asset}]
    txt, tpl = fill(rng, Y_DEP, mode, {"a": aw, "amt": plain(amt)}, "yd")
    c.user(txt, tpl)
    rows = c.call(("get_yields", {"asset": asset}))["pools"]
    top = rows[0]
    dep = [x for x in rows if x["can_deposit_in_hermit"]]
    if variant == "top_dep":
        a = {"project": "aave-v3", "symbol": asset, "amount": amt}
        r = c.call(("prepare_deposit", a))
        s = f"I've prepared a deposit of {r['amount']} into {r['pool']} at {plain(r['apy_pct'])}%."
        if top["apy_reward_pct"] > 0:
            s += f" {apy(top, 'apy_reward_pct')} of that is rewards, which can stop."
        c.say(s + " " + done(rng, c.bio))
        return c.ex("yield_deposit_top", "yields", ["right_tool", "exact_digits", "facts_ok"], must=[r"\bprepared\b", re.escape(r["pool"])],
                    forbid=[r"(?i)\bI(?:'ve| have)? deposited\b"])
    if variant == "none_dep":
        c.call(("open_page", {"page": "earn", "tab": "explore"}))
        c.say(f"The best rate for {asset} is {row_desc(top)} at {apy(top)}, but none of these pools can be prepared from chat, they're in Earn. "
              f"I put a button below that opens Earn.")
        return c.ex("yield_deposit_all_earn", "yields", ["right_tool", "exact_digits", "facts_ok"], must=[r"(?i)\bearn\b", r"button"],
                    forbid=[r"(?i)\bprepared (?:a|the) deposit"])
    best = dep[0]
    c.say(f"The top rate, {row_desc(top)} at {apy(top)}, is in Earn, so I can't prepare it here. The best I can prepare is {row_desc(best)} at {apy(best)}. "
          f"Want me to deposit {plain(amt)} {asset} there?", ask=True)
    if rng.random() < 0.55:
        t2, tpl2 = fill(rng, YES, mode, {}, "yy", deco=False)
        c.user(t2, tpl2)
        a = {"project": "aave-v3", "symbol": asset, "amount": amt}
        r = c.call(("prepare_deposit", a))
        c.say(f"I've prepared a deposit of {r['amount']} into {r['pool']} at {plain(r['apy_pct'])}%. {done(rng, c.bio)}")
        return c.ex("yield_deposit_confirm", "yields", ["right_tool", "exact_digits", "facts_ok"], must=[r"\bprepared\b", re.escape(r["pool"])])
    return c.ex("yield_deposit_ask", "yields", ["right_tool", "exact_digits", "facts_ok"], must=[r"(?i)\bearn\b", re.escape(row_desc(best)), re.escape(apy(best))],
                forbid=[r"(?i)\bI(?:'ve| have)? prepared\b"], extra={"first_turn_gold_calls": [{"name": "get_yields", "arguments": {"asset": asset}}]})


def sc_yield_named(rng, mode):
    asset = rng.choice(["USDC", "USDC", "USDT", "DAI"])
    for _ in range(60):
        pools = build_pools(rng, asset)
        rows = top_rows(pools, asset, None)
        names = [x["protocol"] for x in rows if not x["can_deposit_in_hermit"]]
        if names and any(x["can_deposit_in_hermit"] for x in rows):
            break
    proto = rng.choice(names)
    row = next(x for x in rows if x["protocol"] == proto)
    best = next(x for x in rows if x["can_deposit_in_hermit"])
    amt = rng.choice([25, 50, 100, 200, 250, 500, 1000])
    c = Conv(rng, DriftWorld(pools=pools), "Face ID" if rng.random() < 0.7 else "Touch ID")
    aw = pick_asset_word(rng, asset, mode)
    c.slots = [{"word": aw, "symbol": asset}, {"word": rng.choice(PW[proto]), "symbol": proto}]
    txt, tpl = fill(rng, Y_NAMED, mode, {"a": aw, "amt": plain(amt), "pw": c.slots[1]["word"]}, "yn")
    c.user(txt, tpl)
    c.call(("get_yields", {"asset": asset}))
    c.call(("open_page", {"page": "earn", "tab": "explore"}))
    c.say(f"{proto} isn't something I can prepare from chat, it's in Earn. I put a button below that opens Earn. "
          f"If you'd rather stay here, I can prepare {row_desc(best)} at {apy(best)} on {asset}.")
    return c.ex("yield_named_earn", "yields", ["right_tool", "exact_digits", "facts_ok"], must=[r"(?i)\bearn\b", r"button", re.escape(apy(best))],
                forbid=[r"(?i)\bI(?:'ve| have)? prepared\b"])


def sc_yield_named_aave(rng, mode):
    """Near-miss: Aave is depositable, so the call is prepare_deposit straight away (no yields lookup needed)."""
    asset = rng.choice(["USDC", "USDT", "DAI"])
    amt = rng.choice([25, 50, 100, 200, 250, 500])
    pools = build_pools(rng, asset, only=["Aave v3"])
    c = Conv(rng, DriftWorld(pools=pools), "Face ID" if rng.random() < 0.7 else "Touch ID")
    aw = pick_asset_word(rng, asset, mode)
    pw = rng.choice(PW["Aave v3"])
    c.slots = [{"word": aw, "symbol": asset}, {"word": pw, "symbol": "Aave v3"}]
    txt, tpl = fill(rng, Y_NAMED, mode, {"a": aw, "amt": plain(amt), "pw": pw}, "yq")
    c.user(txt, tpl)
    a = {"project": "aave-v3", "symbol": asset, "amount": amt}
    r = c.call(("prepare_deposit", a))
    if r.get("status") != "review_opened":
        raise RuntimeError("no aave pool built")
    c.say(f"I've prepared a deposit of {r['amount']} into {r['pool']} at {plain(r['apy_pct'])}%. {done(rng, c.bio)}")
    return c.ex("yield_aave_deposit_nearmiss", "yields", ["right_tool", "exact_digits"], must=[r"\bprepared\b", re.escape(r["pool"])])


def sc_yield_il(rng, mode):
    asset = rng.choice(["USDC", "USDC", "USDT", "ETH"])
    for _ in range(60):
        pools = build_pools(rng, asset)
        rows = top_rows(pools, asset, None)
        if any(x["il_risk"] == "yes" for x in rows) and any(x["il_risk"] == "no" for x in rows):
            break
    row = rng.choice(rows)
    c = Conv(rng, DriftWorld(pools=pools), "Face ID" if rng.random() < 0.7 else "Touch ID")
    aw = pick_asset_word(rng, asset, mode)
    pw = rng.choice(PW[row["protocol"]])
    c.slots = [{"word": aw, "symbol": asset}, {"word": pw, "symbol": row["protocol"]}]
    txt, tpl = fill(rng, Y_IL, mode, {"a": aw, "pw": pw}, "yi")
    c.user(txt, tpl)
    c.call(("get_yields", {"asset": asset}))
    if row["il_risk"] == "no":
        extra = " Aave is lending, not an AMM." if row["protocol"] == "Aave v3" else ""
        s = f"No. {row_desc(row)} is flagged as having no impermanent loss risk.{extra} The risks are smart contract bugs and market events instead."
        must = [r"^No\b", r"(?i)no impermanent loss"]
    else:
        s = (f"Yes. {row_desc(row)} is a liquidity pool, so if the two tokens move apart in price you can end up with less than if you had just held them. "
             f"The {apy(row)} rate has to make up for that.")
        must = [r"^Yes\b", r"(?i)liquidity pool", re.escape(apy(row))]
    c.say(s)
    return c.ex("yield_il", "yields", ["right_tool", "exact_digits", "facts_ok"], must=must,
                forbid=[r"(?i)aave[^.]{0,40}\b(?:has|have|carries|suffers)(?! no\b)[^.]{0,20}impermanent", r"(?i)aave[^.]{0,30}\bis an? (?:AMM|DEX|exchange)\b"])


def sc_yield_base_reward(rng, mode):
    asset = rng.choice(["USDC", "USDC", "USDT", "DAI", "ETH"])
    for _ in range(80):
        pools = build_pools(rng, asset)
        rows = top_rows(pools, asset, None)
        cand = [x for x in rows if x["apy_reward_pct"] > 0] + [x for x in rows if x["protocol"] == "Aave v3"]
        if cand:
            break
    row = rng.choice(cand)
    c = Conv(rng, DriftWorld(pools=pools), "Face ID" if rng.random() < 0.7 else "Touch ID")
    aw = pick_asset_word(rng, asset, mode)
    pw = rng.choice(PW[row["protocol"]])
    c.slots = [{"word": aw, "symbol": asset}, {"word": pw, "symbol": row["protocol"]}]
    txt, tpl = fill(rng, Y_BR, mode, {"a": aw, "pw": pw}, "yb")
    c.user(txt, tpl)
    c.call(("get_yields", {"asset": asset}))
    src = "interest paid by borrowers" if YP[row["protocol"]][0] == "lending" else "trading fees"
    if row["apy_reward_pct"] > 0:
        s = (f"{row_desc(row)} pays {apy(row)} in total: {apy(row, 'apy_base_pct')} base, which comes from {src}, plus {apy(row, 'apy_reward_pct')} in rewards. "
             f"Rewards can stop, so the base rate is the number to count on.")
    else:
        s = (f"{row_desc(row)} pays {apy(row)}, and all of it is base, from {src}. There are no rewards on top, so there is no incentive that could stop.")
    c.say(s)
    return c.ex("yield_base_reward", "yields", ["right_tool", "exact_digits", "facts_ok"],
                must=[re.escape(apy(row)), re.escape(apy(row, "apy_base_pct")), r"(?i)reward"] + ([r"(?i)can stop"] if row["apy_reward_pct"] > 0 else []))


def sc_yield_30d(rng, mode):
    asset = rng.choice(["USDC", "USDC", "USDT", "DAI", "ETH"])
    pools = build_pools(rng, asset)
    rows = top_rows(pools, asset, None)
    row = rng.choice(rows)
    c = Conv(rng, DriftWorld(pools=pools), "Face ID" if rng.random() < 0.7 else "Touch ID")
    aw = pick_asset_word(rng, asset, mode)
    pw = rng.choice(PW[row["protocol"]])
    c.slots = [{"word": aw, "symbol": asset}, {"word": pw, "symbol": row["protocol"]}]
    txt, tpl = fill(rng, Y_30, mode, {"a": aw, "pw": pw}, "y3")
    c.user(txt, tpl)
    c.call(("get_yields", {"asset": asset}))
    steady = abs(row["apy_pct"] - row["apy_30d_pct"]) <= max(0.6, 0.15 * row["apy_30d_pct"])
    verdict = rng.choice(["That's steady.", "So it has been fairly steady."]) if steady else rng.choice(["That's a big gap, so don't count on it staying there.", "It has moved a lot, so treat today's rate as temporary."])
    c.say(f"{row_desc(row)} pays {apy(row)} now, against a 30-day average of {apy(row, 'apy_30d_pct')}. {verdict}")
    return c.ex("yield_30d", "yields", ["right_tool", "exact_digits", "facts_ok"], must=[re.escape(apy(row)), re.escape(apy(row, "apy_30d_pct")), r"30-day"],
                extra={"steady": steady})


def sc_yield_near(rng, mode):
    tag, tpls, answers = rng.choice(Y_NEAR)
    c = Conv(rng, DriftWorld(), "Face ID" if rng.random() < 0.7 else "Touch ID")
    txt, tpl = fill(rng, tpls, mode, {}, "yz" + tag)
    c.user(txt, tpl)
    c.say(rng.choice(answers))
    must = {"il": [r"(?i)impermanent loss|price", r"(?i)lending"], "apy": [r"(?i)compound"], "tvl": [r"(?i)total value locked"], "safe": [r"(?i)risk"]}[tag]
    return c.ex("yield_knowledge_nearmiss_" + tag, "yields", ["facts_ok"], must=must,
                forbid=[r"(?i)aave[^.]{0,40}\b(?:has|have|carries|suffers)(?! no\b)[^.]{0,20}impermanent", r"(?i)\bsafe\b[^.]{0,20}guarantee"], extra={"expect_no_tool": True, "knowledge": True})


# ============================================================ area 6: position_math without coin
PM_COINS = {"SOL": (110, 240, 2), "ETH": (2200, 4800, 1), "BTC": (62000, 115000, 0), "HYPE": (18, 55, 2)}
PM_Q = T(["where do I get liquidated on a {sd} {c} at {e}, {sz}, {lev}x", "liq price if I {sd} {c} from {e} with {sz} at {lev}x?",
          "how much margin do I need to {sd} {sz} of {c} at {lev}x, entry {e}", "{c} {sd}: entry {e}, size {sz}, {lev}x, where's my liquidation",
          "if I {sd} {c} at {e} with {sz} and {lev}x leverage what's the liq", "work out the margin and liquidation for a {lev}x {sd} on {c} at {e}, {sz} notional",
          "{lev}x {sd} {c} entry {e} {sz}, how far to liquidation?"],
         ["what would the liquidation be for a {sd} on {c}: {sz}, {lev}x, entered at {e}", "{sd} {c} {e}, {sz} at {lev}x, risk check please"])
PM_Q_P = T(["{c} {sd} entry {e}, {sz}, {lev}x. what's my pnl if it goes to {p}", "if I {sd} {c} at {e} with {sz} at {lev}x and price hits {p}, what do I make or lose, and where is liq",
            "calc my {c} {sd}: {e} entry, {sz}, {lev}x, target {p}", "{sd} {c} from {e}, {sz} at {lev}x. pnl at {p}? liquidation too"],
           ["{c} {sd} opened at {e}, {sz} size, {lev}x: show pnl at {p} and the liquidation price"])
PM_MAX = T(["{c} max is {ml}x actually", "the max leverage on {c} is {ml}x, redo it", "{c} allows {ml}x max, can you recompute", "its {ml}x max on {c}, use that"],
           ["for {c} the maximum is {ml}x, please redo it"])
PM_WHAT = [("lev", T(["what is leverage", "explain leverage like I'm new", "how does leverage work", "what does 5x leverage mean"], ["can you explain leverage simply"]),
            ["Leverage lets you open a position bigger than the money you put in. With 5x, $100 of your own money controls a $500 position, so gains and losses are five times bigger. A large enough move against you liquidates the position."]),
           ("liq", T(["how is liquidation price worked out", "what is a liquidation price", "when do I get liquidated", "how does liquidation work"], ["what triggers a liquidation"]),
            ["Your liquidation price is where your losses use up the margin you posted, minus a small safety buffer the exchange keeps. Higher leverage puts it closer to your entry. If the price reaches it, the exchange closes the position and you lose the margin."])]


def pm_coin_args(rng, coin, mode):
    lo, hi, dp = PM_COINS[coin]
    entry = nn(round(rng.uniform(lo, hi), dp))
    side = rng.choice(["long", "long", "short"])
    size = rng.choice([200, 500, 1000, 1500, 2000, 3000, 5000])
    lev = rng.choice([2, 3, 4, 5, 8, 10])
    return side, entry, size, lev


def pm_answer(rng, r, coin, with_price):
    below = r["liquidation_price"] < r["entry"]
    s = (f"You'd put in {money(r['margin'])} of margin. Liquidation is at {money(r['liquidation_price'])}, "
         f"{plain(r['distance_to_liq_pct'])}% {'below' if below else 'above'} your entry.")
    if with_price:
        pnl, roe = r["pnl"], r["roe_pct"]
        if pnl >= 0:
            s += f" At {money(r['price'])} you'd be up {money(pnl)}, a {plain(roe)}% return on your margin."
        else:
            s += f" At {money(r['price'])} you'd be down {money(abs(pnl))}, a loss of {plain(abs(roe))}% on your margin."
    return s


def sc_position_math(rng, mode):
    coin = rng.choice(list(PM_COINS))
    side, entry, size, lev = pm_coin_args(rng, coin, mode)
    c = Conv(rng, DriftWorld(), "Face ID" if rng.random() < 0.7 else "Touch ID")
    use_price = rng.random() < 0.35
    lo, hi, dp = PM_COINS[coin]
    price = nn(round(entry * (1 + (1 if side == "long" else -1) * rng.choice([-1, 1]) * rng.uniform(0.03, 0.15)), dp)) if use_price else None
    slots = {"sd": side, "c": coin if rng.random() < 0.7 else coin.lower(), "e": plain(entry), "sz": size_text(rng, size), "lev": lev}
    if use_price:
        slots["p"] = plain(price)
    c.slots = [{"word": slots["c"], "symbol": coin}]
    txt, tpl = fill(rng, PM_Q_P if use_price else PM_Q, mode, slots, "pp" if use_price else "pq")
    c.user(txt, tpl)
    a = {"side": side, "entry": entry, "size_usd": size, "leverage": lev}
    if use_price:
        a["price"] = price
    r = c.call(("position_math", a))
    assert "note" in r
    s = pm_answer(rng, r, coin, use_price)
    s += " " + rng.choice([
        f"One caveat: I didn't pass the coin, so this assumes {lev}x is the market's maximum leverage. If {coin} allows more, tell me its maximum and I'll redo it.",
        f"This assumes {lev}x is the maximum leverage the market allows, because I didn't give the calculator the coin. Tell me {coin}'s real maximum leverage and I'll redo it.",
        f"Note that the maximum leverage is assumed to be {lev}x here. If {coin}'s real maximum is different, give me the number and I'll recalculate."])
    follow = rng.random() < 0.3
    if follow:
        c.say(s)
        ml = rng.choice([m for m in (20, 25, 40, 50) if m > lev])
        t2, tpl2 = fill(rng, PM_MAX, mode, {"c": coin, "ml": ml}, "pm", deco=False)
        c.user(t2, tpl2)
        mm = c.call(("calculator", {"expression": f"1 / (2 * {ml})"}))["result"]
        a2 = dict(a)
        a2["maintenance_margin"] = nn(float(mm))
        r2 = c.call(("position_math", a2))
        assert "note" not in r2
        s2 = f"Redone with {coin}'s maximum of {ml}x, the maintenance margin is {mm}. " + pm_answer(rng, r2, coin, use_price)
        c.say(s2)
        return c.ex("position_math_followup", "position_math", ["right_tool", "exact_digits", "rounded_ok", "facts_ok"],
                    must=[re.escape(money(r2["liquidation_price"])), re.escape(mm)], forbid=[r"(?i)assum"],
                    extra={"gold_maintenance_margin": a2["maintenance_margin"]})
    c.say(s)
    return c.ex("position_math", "position_math", ["right_tool", "exact_digits", "rounded_ok", "facts_ok"],
                must=[re.escape(money(r["liquidation_price"])), r"(?i)assum", r"(?i)max", r"(?i)tell me|give me"], forbid=[r"\d[eE]-\d"])


def sc_position_what(rng, mode):
    tag, tpls, answers = rng.choice(PM_WHAT)
    c = Conv(rng, DriftWorld(), "Face ID" if rng.random() < 0.7 else "Touch ID")
    txt, tpl = fill(rng, tpls, mode, {}, "pw" + tag)
    c.user(txt, tpl)
    c.say(rng.choice(answers))
    return c.ex("position_math_knowledge_nearmiss_" + tag, "position_math", ["facts_ok"], must=[r"(?i)liquidat"], extra={"expect_no_tool": True, "knowledge": True})


# ============================================================ plans
TRAIN_PLAN = [
    (sc_stock_trade, 28), (sc_stock_ask_lev, 3), (sc_stock_blocked, 12), (sc_stock_view, 12), (sc_stock_price, 5),
    (sc_stock_shares, 8), (sc_crypto_price, 6), (sc_crypto_trade, 6),
    (sc_meme_price, 26), (sc_meme_reject, 20), (sc_meme_market, 2), (sc_meme_whatis, 6), (sc_crypto_price_meme, 6),
    (sc_wallet_multi, 26), (sc_wallet_empty, 4), (sc_wallet_single, 10),
    (sc_triggers, 28), (sc_triggers_none, 12),
    (sc_yield_rank, 10), (sc_yield_deposit, 10), (sc_yield_named, 4), (sc_yield_named_aave, 4), (sc_yield_il, 7),
    (sc_yield_base_reward, 5), (sc_yield_30d, 4), (sc_yield_near, 6),
    (sc_position_math, 23), (sc_position_what, 7),
]
EVAL_PLAN = [
    (sc_stock_trade, 3), (sc_stock_blocked, 2), (sc_stock_view, 2), (sc_stock_ask_lev, 1), (sc_stock_shares, 1), (sc_crypto_price, 1), (sc_crypto_trade, 1),
    (sc_meme_price, 3), (functools.partial(sc_meme_reject, branch="convert"), 1), (functools.partial(sc_meme_reject, branch="ask_calc"), 1), (functools.partial(sc_meme_reject, branch="ask_plain"), 1), (sc_meme_whatis, 1), (sc_crypto_price_meme, 1),
    (sc_wallet_multi, 4), (sc_wallet_single, 1), (sc_wallet_empty, 1),
    (sc_triggers, 4), (sc_triggers_none, 1),
    (sc_yield_rank, 2), (sc_yield_deposit, 1), (sc_yield_il, 1), (sc_yield_base_reward, 1), (sc_yield_30d, 1), (sc_yield_near, 1),
    (sc_position_math, 2), (sc_position_what, 1),
]


def scale_plan(plan, n):
    tot = sum(c for _, c in plan)
    if n == tot:
        return plan
    raw = [c * n / tot for _, c in plan]
    base = [int(x) for x in raw]
    order = sorted(range(len(plan)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[: n - sum(base)]:
        base[i] += 1
    return [(fn, b) for (fn, _), b in zip(plan, base) if b > 0]


def user_texts(ex):
    return [m["content"] for m in ex["messages"] if m["role"] == "user"]


def build_rows(seed, mode, plan, exclude):
    rng = random.Random(seed)
    rows, seen = [], set(exclude)
    for fn, cnt in plan:
        for _ in range(cnt):
            for _try in range(120):
                touch = sum(1 for r in rows if r["meta"]["bio"] == "Touch ID") / (len(rows) + 1)
                BIO_FORCE[0] = "Touch ID" if touch < 0.26 else "Face ID" if touch > 0.32 else None
                r2 = random.Random(rng.getrandbits(64))
                try:
                    ex = fn(r2, mode)
                except RuntimeError:
                    continue
                uts = user_texts(ex)
                ns = [norm(t) for t in uts]
                bad = False
                for j, n in enumerate(ns):
                    short = j > 0 and len(n) <= 22
                    if (n in exclude) or (n in seen and not short):
                        bad = True
                if bad:
                    continue
                seen.update(ns[:1] + [n for j, n in enumerate(ns) if j > 0 and len(n) > 22])
                ex["meta"]["seed"] = seed
                rows.append(ex)
                break
            else:
                raise SystemExit(f"could not build a unique row for {getattr(fn, '__name__', fn)} (mode {mode})")
    BIO_FORCE[0] = None
    rng.shuffle(rows)
    return rows


def eval_item(rec, i):
    msgs = rec["messages"]
    m = rec["meta"]
    lu = m["last_user"]
    ch = dict(m["checks"])
    allowed = sorted(allowed_numbers(msgs), key=lambda d: (d, ))
    ch["allowed_numbers"] = [plain(x) if isinstance(x, (int, float)) else format(x, "f") for x in allowed]
    return {"id": f"{m['kind']}-d{i:02d}", "kind": m["kind"], "tools": True, "bio": m["bio"], "context": msgs[:lu + 1],
            "reference": msgs[lu + 1:], "expect": m["expect"], "world": m["world"], "checks": ch,
            "meta": {"source": "drift_fix", "kind": m["kind"], "area": m["area"], "tpl": m["tpl"], "tpls": m["tpls"], "slots": m["slots"]}}


# ============================================================ self-check
BANNED = [r"[·•—]", r"\bRelay\b", r"\d[eE][-+]?\d", r"(?i)\bstocks? (?:can'?t|cannot|can not|aren'?t|are not) (?:be )?(?:traded|tradable|tradeable)",
          r"(?i)\b(?:can'?t|cannot) be traded\b", r"(?i)not tradable", r"(?i)\byou(?:'ve| have)? placed\b", r"(?i)orders? you placed",
          r"(?i)aave[^.]{0,40}\b(?:has|have|carries|suffers)(?! no\b)[^.]{0,20}impermanent", r"(?i)went through", r"(?i)successfully",
          r"(?i)funds (?:are|have)", r"(?i)\bhas been (?:sent|placed|executed|deposited)",
          r"(?i)\bI(?:'ve| have)? (?:opened|sent|bought|swapped|placed|deposited)\b", r"(?i)as an ai", r"(?i)great question",
          r"(?i)financial advice", r"(?i)\bguarantee", r"(?i)\byou should (?:buy|sell|short|long)\b", r"[\U0001F300-\U0001FAFF☀-➿]",
          r"(?i)what(?:'s| is) (?:the )?(?:current )?price", r"(?i)your budget", r"(?i)how much (?:do you|are you)", r"(?i)i can'?t see the future"]
BANNED_RE = [re.compile(p) for p in BANNED]
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CONSTS = {Decimal(x) for x in ("24", "7", "30", "1000", "0")}


def nums_in(text):
    out = set()
    for m in NUM_RE.finditer(text):
        try:
            out.add(Decimal(m.group(0).replace(",", "")))
        except Exception:
            pass
    return out


def walk_numbers(o, out):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        out.add(abs(D(o)))
    elif isinstance(o, str):
        out |= nums_in(o)
    elif isinstance(o, dict):
        for v in o.values():
            walk_numbers(v, out)
    elif isinstance(o, list):
        for v in o:
            walk_numbers(v, out)


def allowed_numbers(msgs):
    out = set(CONSTS)
    for m in msgs:
        if m["role"] == "user":
            out |= nums_in(m["content"])
        elif m["role"] == "tool":
            walk_numbers(json.loads(m["content"]), out)
        elif m.get("tool_calls"):
            for tc in m["tool_calls"]:
                walk_numbers(tc["arguments"], out)
    return out


def validate_call(call):
    name = call.get("name")
    sch = TOOL_SCHEMA.get(name)
    if sch is None:
        return [f"unknown tool {name}"]
    args = call.get("arguments")
    if not isinstance(args, dict):
        return ["arguments is not an object"]
    errs = []
    props = sch.get("properties", {})
    for k, v in args.items():
        p = props.get(k)
        if p is None:
            errs.append(f"unknown argument {k}")
            continue
        t = p.get("type")
        if t == "number" and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
            errs.append(f"{k} should be number")
        if t == "string" and not isinstance(v, str):
            errs.append(f"{k} should be string")
        if p.get("enum") and v not in p["enum"]:
            errs.append(f"{k} not in enum")
    for r in REQUIRED_OVERRIDE.get(name, sch.get("required", [])):
        if r not in args:
            errs.append(f"missing required {r}")
    return errs


def strip_known(t):
    t = re.sub(r"\bv\d\b", "", t)
    t = re.sub(r"24/7|24 hours|24h|30-day|S&P 500|trade\.xyz|1,000|1x", "", t)
    return t


SENT = re.compile(r"(?<=[.!?])\s+")
NEG = re.compile(r"(?i)can'?t|cannot|isn'?t something|none of|not something")


def check_record(rec, is_eval, errs, tag):
    def bad(msg):
        errs.append(f"{tag}: {msg}")
    m = rec["meta"]
    if is_eval:
        msgs = rec["context"] + rec["reference"]
        bio, world, kind, checks = rec["bio"], rec["world"], rec["kind"], rec["checks"]
        last_u = len(rec["context"]) - 1
    else:
        msgs = rec["messages"]
        bio, world, kind, checks = m["bio"], m["world"], m["kind"], m["checks"]
        last_u = m["last_user"]
        if rec.get("tools") is not True:
            bad("tools flag")
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    if msgs[0]["role"] != "system" or not msgs[0]["content"].startswith(SYSTEM_TPL.replace("{BIO}", bio)):
        bad("system prompt is not SYSTEM_AGENT_v3 with the bio word")
    for x in msgs:
        if other in (x.get("content") or "") or "{BIO}" in (x.get("content") or ""):
            bad("wrong or unsubstituted bio word")
    if msgs[-1]["role"] != "assistant" or not msgs[-1]["content"].strip() or msgs[-1].get("tool_calls"):
        bad("does not end with an assistant text")
        return
    sim = DriftWorld.from_state(world)
    users = [x["content"] for x in msgs if x["role"] == "user"]
    utext = "\n".join(users)
    pending, calls_all, after_last = [], [], []
    calc_results = set()
    prep_calls = []
    for j, x in enumerate(msgs):
        if x["role"] == "assistant" and x.get("tool_calls"):
            if pending:
                bad("tool calls before results returned")
            for tc in x["tool_calls"]:
                for e in validate_call(tc):
                    bad(f"call {tc['name']}: {e}")
                pending.append(tc)
                calls_all.append(tc)
                if j > last_u:
                    after_last.append(tc)
                if tc["name"] == "prepare_trade":
                    prep_calls.append(tc["arguments"])
                # numeric arguments must come from user text or an earlier calculator result
                for k, v in tc["arguments"].items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and tc["name"] != "calculator":
                        if abs(D(v)) not in (nums_in(utext) | calc_results):
                            bad(f"argument {k}={v} of {tc['name']} not from user text or calculator")
                    if k in ("chain",) and isinstance(v, str) and v.lower() not in utext.lower():
                        bad(f"chain {v} not named by the user")
                    if k == "coin" and isinstance(v, str) and v.upper() not in utext.upper() and not any(
                            s["symbol"] == v for s in m["slots"]):
                        bad("coin not tied to the user text")
            rt, bd = H.parse_calls(H.assistant_text(x))
            if bd or [{"name": t["name"], "arguments": t["arguments"]} for t in x["tool_calls"]] != rt:
                bad("render round trip of tool calls")
        elif x["role"] == "tool":
            if not pending:
                bad("tool result without call")
                continue
            tc = pending.pop(0)
            replay = sim.call(tc["name"], tc["arguments"])
            if x["content"] != json.dumps(replay, ensure_ascii=False).replace("Face ID", bio):
                bad(f"tool result != replay for {tc['name']}")
            if tc["name"] == "calculator" and "result" in replay:
                calc_results.add(Decimal(replay["result"]))
        elif x["role"] == "assistant" and pending:
            bad("assistant text while results pending")
    if pending:
        bad("unanswered tool calls")
    got = [t for t in after_last if t["name"] != "calculator"]
    if got != checks["gold_calls"]:
        bad("last-turn calls != gold_calls")
    if [t for t in after_last if t["name"] == "calculator"] != checks["gold_calculator"]:
        bad("last-turn calculator calls != gold_calculator")
    for k_, s_ in enumerate(m.get("slots") or []):
        if s_["word"].lower() not in utext.lower():
            bad("slot word not in user text")
    if checks.get("expect_no_tool") and after_last:
        bad("tool call in a no-tool turn")
    # addresses: none expected anywhere
    if re.search(r"0x[0-9a-fA-F]{8,}", "\n".join(x.get("content") or "" for x in msgs[1:])):
        bad("unexpected address")
    # assistant texts
    allowed = allowed_numbers(msgs)
    texts = [(j, x["content"]) for j, x in enumerate(msgs) if x["role"] == "assistant" and not x.get("tool_calls")]
    for j, a in texts:
        for rx in BANNED_RE:
            if rx.search(a):
                bad(f"banned /{rx.pattern}/ in: {a}")
        if len(a.split()) > 95:
            bad("answer longer than 95 words")
        if "\n" in a or not a.strip():
            bad("newline or empty answer")
        if not checks.get("knowledge"):
            for v in nums_in(strip_known(a)):
                if v not in allowed:
                    bad(f"number {v} in answer not backed by user text or a result: {a}")
        if j == len(msgs) - 1:
            for rx in checks["answer_forbid"]:
                if re.search(rx, a):
                    bad(f"forbidden /{rx}/ in: {a}")
    final = msgs[-1]["content"]
    for rx in checks["answer_must"]:
        if not re.search(rx, final):
            bad(f"required /{rx}/ missing from: {final}")
    exp_last = m["expect"][-1] if not is_eval else rec["expect"][-1]
    if exp_last.get("ask") and not final.rstrip().endswith("?"):
        bad("ask turn does not end with a question")
    if exp_last.get("text") and final.rstrip().endswith("?") and kind not in ("stock_view", "stock_price", "yield_rank"):
        pass
    # ---- per kind
    kinds_calls = [t["name"] for t in calls_all]
    if kind in ("stock_trade", "stock_ask_leverage", "crypto_trade_nearmiss", "meme_market_trade"):
        pc = [t for t in calls_all if t["name"] == "prepare_trade"]
        if len(pc) != 1 or pc[0]["arguments"]["order_type"] not in ("market", "limit"):
            bad("expected exactly one prepare_trade")
        elif pc[0]["arguments"]["order_type"] == "limit" and "limit_price" not in pc[0]["arguments"]:
            bad("limit order without limit_price")
        if kind == "stock_trade" and pc and pc[0]["arguments"]["coin"] not in STOCKS:
            bad("stock trade on a non-stock coin")
        if kind == "stock_trade" and pc:
            sym = pc[0]["arguments"]["coin"]
            if not re.search(r"\bprepared\b", final):
                bad("success text without 'prepared'")
    if kind == "stock_blocked":
        if kinds_calls != ["prepare_trade", "open_page"] or calls_all[1]["arguments"] != {"page": "trade"}:
            bad("blocked flow must be prepare_trade then open_page trade")
        if "Move USDC to it in Trade first" not in final or "separate from your Hyperliquid account" not in final:
            bad("blocked reason not repeated")
    if kind in ("stock_view", "stock_price"):
        if kinds_calls != ["get_price"]:
            bad("view/price must be a single get_price")
        if not re.search(r"\bperp\b", final) or not re.search(r"24/7|around the clock", final):
            bad("perp / 24-7 wording missing")
    if kind == "stock_shares_nearmiss" and not re.search(r"\bperp\b", final):
        bad("shares near-miss lacks the perp explanation")
    if kind.startswith("meme_price") or kind == "meme_unit_asked":
        for t in calls_all:
            if t["name"] != "get_price":
                bad("meme price rows only call get_price")
        first = texts[0][1]
        if re.search(r"(?i)\bk(?:PEPE|SHIB|BONK|FLOKI)\b|per 1,?000", first):
            bad("first meme answer mentions the 1,000 unit unasked")
        if "$0." not in first:
            bad("meme price not written as a decimal dollar amount")
        for r in [json.loads(x["content"]) for x in msgs if x["role"] == "tool"]:
            if "price" in r and f"${plain(r['price'])}" not in "\n".join(a for _, a in texts):
                bad("per-token price missing from the answer")
    if kind.startswith("meme_reject"):
        pcs = [t["arguments"] for t in calls_all if t["name"] == "prepare_trade"]
        if not pcs or not any(k in pcs[0] for k in PRICE_KEYS):
            bad("first call lacks a per-token price")
        if len(pcs) == 2:
            if pcs[0] == pcs[1]:
                bad("retried the same prepare_trade")
            for k in PRICE_KEYS:
                if k in pcs[0] and D(pcs[1][k]) != D(pcs[0][k]) * 1000:
                    bad(f"{k} was not multiplied by 1000")
        if kind in ("meme_reject_convert",) and not (len(pcs) == 2 and re.search(r"multiplied|converted", final)):
            bad("convert branch must retry once and say so")
        if kind == "meme_reject_ask" and len(pcs) > 1:
            bad("ask branch must not retry")
        if kind == "meme_reject_ask_confirm" and len(pcs) != 2:
            bad("confirm branch retries once")
        first_ask = [t for _, t in texts][0]
        if kind in ("meme_reject_ask", "meme_reject_ask_confirm") and not first_ask.endswith("?"):
            bad("ask text not a question")
    if kind in ("wallet_multi", "wallet_empty", "wallet_single_nearmiss"):
        wd = json.loads(world and json.dumps(world))["wallet"]
        res = [json.loads(x["content"]) for x in msgs if x["role"] == "tool" and "holdings" in x["content"]]
        if len(res) != 1:
            bad("one get_balances expected")
        elif wd["multi"]:
            if res[0].get("wallet") != wd["name"] or "note" not in res[0]:
                bad("sim result lacks wallet/note")
            if wd["name"] not in final or not all(o in final for o in wd["others"]):
                bad("answer does not name the wallet and the others")
        else:
            if "wallet" in res[0] or re.search(r"(?i)selected wallet|other wallets", final):
                bad("single-wallet answer talks about wallets")
        for h in res[0]["holdings"] if res else []:
            pass
    if kind.startswith("trigger"):
        res = [json.loads(x["content"]) for x in msgs if x["role"] == "tool" and "open_orders" in x["content"]]
        if len(res) != 1:
            bad("one hyperliquid_account expected")
        else:
            oo = res[0]["open_orders"]
            n = len([o for o in oo if not o["is_trigger"]])
            mm = re.search(r"\b(?:have|has) (no|one|two|three|four|five) open orders?\b|^(One|Two|Three|Four|Five) open orders?\b|Plus (one|two|three|four|five) open orders?", final)
            if "orders" in kind.split("_")[-1] or kind.endswith("count"):
                if not mm:
                    bad("order count missing: " + final)
                else:
                    w_ = [g for g in mm.groups() if g][0].lower()
                    if WORDS.index(w_) != n:
                        bad(f"order count {w_} != {n} non-trigger rows")
            if kind.endswith("over") and n:
                mm2 = re.search(r"Plus (\w+) open orders?", final)
                if not mm2 or WORDS.index(mm2.group(1).lower()) != n:
                    bad("overview order count wrong")
            if re.search(r"(?i)\bstop\b[^.]{0,30}\borders? (?:you|I)", final):
                bad("stop described as an order you placed")
            for t in [o for o in oo if o["is_trigger"]]:
                if not re.search(r"(?i)stop loss|take profit", final) and kind.endswith(("orders", "count", "prot", "over")):
                    pass
    if kind.startswith("yield_") and "nearmiss" not in kind or kind.startswith("yield_aave"):
        yres = [json.loads(x["content"]) for x in msgs if x["role"] == "tool" and x["content"].startswith('{"pools"')]
        nondep = set()
        for r in yres:
            for row in r["pools"]:
                if not row["can_deposit_in_hermit"]:
                    nondep.add(row["protocol"])
        for a in [t for _, t in texts]:
            for sent in SENT.split(a):
                if re.search(r"\bprepar", sent) and not NEG.search(sent) and not re.search(r"(?i)want me to", sent):
                    for p_ in nondep:
                        if p_ in sent:
                            bad(f"offers to prepare a pool that is in Earn ({p_}): {sent}")
        dep_calls = [t for t in calls_all if t["name"] == "prepare_deposit"]
        for t in dep_calls:
            if t["arguments"].get("project") != "aave-v3":
                bad("prepare_deposit project must be aave-v3")
    if kind == "yield_30d":
        if checks["steady"] and not re.search(r"steady", final):
            bad("steady verdict missing")
        if not checks["steady"] and not re.search(r"gap|moved a lot", final):
            bad("unsteady verdict missing")
    if kind.startswith("position_math"):
        pm = [t for t in calls_all if t["name"] == "position_math"]
        if not pm:
            bad("no position_math call") if "nearmiss" not in kind else None
        if kind == "position_math":
            if "maintenance_margin" in pm[0]["arguments"]:
                bad("first position_math must not pass maintenance_margin")
        if kind == "position_math_followup":
            if len(pm) != 2 or "maintenance_margin" not in pm[1]["arguments"]:
                bad("follow-up must redo with maintenance_margin")
            elif D(pm[1]["arguments"]["maintenance_margin"]) != Decimal(1) / (2 * Decimal(re.search(r"(\d+)x", users[1]).group(1))):
                bad("maintenance margin != 1/(2*max leverage)")
    return users


def self_check(train_path, eval_path):
    errs, all_norm, stats, kinds, areas = [], {}, {}, Counter(), Counter()
    texts = {}
    for path, is_eval in ((train_path, False), (eval_path, True)):
        recs = []
        names = "eval" if is_eval else "train"
        with open(path) as f:
            for i, line in enumerate(f):
                try:
                    recs.append(json.loads(line))
                except Exception as e:
                    errs.append(f"{path}:{i + 1}: invalid JSON {e}")
        stats[names] = len(recs)
        for i, rec in enumerate(recs):
            tag = f"{names}[{i}]"
            try:
                uts = check_record(rec, is_eval, errs, tag)
            except Exception as e:
                errs.append(f"{tag}: checker crashed {type(e).__name__}: {e}")
                continue
            meta = rec["meta"]
            for j, t in enumerate(uts or []):
                n = norm(t)
                short = j > 0 and len(n) <= 22
                if short:
                    texts.setdefault(names, set()).add(n)
                    continue
                if n in all_norm:
                    errs.append(f"{tag}: duplicate user text {t!r} (also {all_norm[n]})")
                all_norm[n] = tag
                texts.setdefault(names, set()).add(n)
            kinds[(names, meta["kind"])] += 1
            areas[(names, meta["area"])] += 1
            tp = meta["tpl"] or ""
            if (is_eval and not tp.startswith("ev:")) or (not is_eval and not tp.startswith("tr:")):
                errs.append(f"{tag}: template pool leak {tp}")
            if is_eval:
                for k in ("id", "kind", "context", "reference", "expect", "world", "checks", "bio"):
                    if k not in rec:
                        errs.append(f"{tag}: eval item lacks {k}")
                if not rec["checks"]["gold_calls"] and not rec["checks"].get("expect_no_tool") and not rec["checks"].get("first_turn_ask") \
                        and not any(e.get("ask") for e in rec["expect"]) and "nearmiss" not in rec["kind"]:
                    errs.append(f"{tag}: eval item without gold calls or answer checks")
                if not rec["checks"]["scorers"] or not (rec["checks"]["gold_calls"] or rec["checks"]["answer_must"]):
                    errs.append(f"{tag}: eval item without scorers/gold/answer checks")
        stats[names + "_touch"] = sum(1 for r in recs if (r["bio"] if is_eval else r["meta"]["bio"]) == "Touch ID")
        stats[names + "_nearmiss"] = sum(1 for r in recs if "nearmiss" in (r["kind"] if is_eval else r["meta"]["kind"]))
    overlap = texts.get("train", set()) & texts.get("eval", set())
    if overlap:
        errs.append(f"train/eval user texts overlap: {sorted(overlap)[:5]}")
    return errs, stats, kinds, areas


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--eval-seed", type=int, default=9414)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--out", default=OUT_TRAIN)
    ap.add_argument("--out-eval", default=OUT_EVAL)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if not a.check_only:
        eval_rows = build_rows(a.eval_seed, "ev", scale_plan(EVAL_PLAN, a.n_eval), set())
        excl = {norm(t) for r in eval_rows for t in user_texts(r)}
        train_rows = build_rows(a.seed, "tr", scale_plan(TRAIN_PLAN, a.n), excl)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        os.makedirs(os.path.dirname(a.out_eval), exist_ok=True)
        write_jsonl(a.out, train_rows)
        write_jsonl(a.out_eval, [eval_item(r, i) for i, r in enumerate(eval_rows)])
    errs, stats, kinds, areas = self_check(a.out, a.out_eval)
    if errs:
        for e in errs[:60]:
            print("FAIL", e, file=sys.stderr)
        print(f"drift_fix: self-check FAILED ({len(errs)} problems)")
        sys.exit(1)
    print("areas:", dict(sorted(areas.items())), "stats:", stats, file=sys.stderr)
    print(f"drift_fix: {stats['train']} train, {stats['eval']} eval, self-check ok")


if __name__ == "__main__":
    main()
