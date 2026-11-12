#!/usr/bin/env python3
"""recovery_errors generator (sft4/v3 generator 8): what to do after a failure, a refusal or an odd result, plus the
error paths of every new v3 tool (eval suite tool_errors_v3).

Recovery kinds (train quota in brackets):
  quote_min [52]      quote/swap below the route minimum: give the minimum, size a valid amount with the calculator, offer it
  blocked [52]        status blocked + reasons: say the reason, never retry the same call, no cap numbers, alternative only if supported
  insufficient [52]   deposit/withdraw/buy/sell "insufficient": show the holding, offer the maximum as a question
  needs_ask [52]      needs_chain / needs_choice: ask the app's question verbatim, call again with the answer, name the network
  no_route [46]       no route for a pair: plan swap-to-USDC then bridge, prepare only the first step when asked
  parallel_fail [52]  one of two parallel reads fails: answer from the other, name what is missing, offer one retry
  correction [52]     user corrects ("no, Arbitrum", "make it 50"): prepare again, say the first review is still open
  network_differs[46] the result network differs from the named chain: say so in the FIRST sentence
  odd [40]            holdings [], unreachable, change_24h_pct null, liquidation null, the (hidden: ...) placeholder, a result
                      that contradicts the user (trust the tool)
  unsupported [22]    request no tool supports (BTC send, Optimism, Ledger, editing an order): no fake call, say what the app does
  timeout_ok [27]     first error says try again: ONE identical retry, then answer
  timeout_twice [20]  second failure: one sentence, stop
  tools [88]          8 error rows for each of the 11 new tools (prepare_sell, search_tokens, get_positions, get_portfolio,
                      cancel_order, set_tp_sl, create_automation, schedule_payment, add_contact, open_page, get_address)

Tool results are SCRIPTED from the shapes in spec/hermit_tools_v3.meta.json (result_example / other_results); every
assistant sentence is a Python f-string filled from tool results, user text or calculator results. Nothing is typed by hand
into a number slot. gen2 helpers (hermit_fmt, sim) are imported, never edited; gen3 fmt3 is imported for validate_args.

Usage: python recovery_errors.py --seed 4 --n 600 [--eval-seed 9404 --eval-n 120]
Writes ../data4/recovery_errors.jsonl and ../eval4/recovery_errors.jsonl after the self-check passes on every record
(exit 1 otherwise). Eval rows carry `script` (the scripted results the harness returns for each call, in order),
`checks` (the tool_errors_v3 scorer checks that apply) and `gold` (payload for those checks).

Eval harness contract for `script`: a list of rules {name, arguments, results}; the n-th call whose name AND arguments equal
a rule returns results[n] (the last one repeats); the calculator is executed for real; any other call returns
{"error": "No result scripted for this call."} (the scorer then also fails exact_digits for that call).
"""
import argparse, json, os, random, re, sys
from collections import Counter
from decimal import Decimal

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
SPEC = os.path.join(LAB, "spec")
for p in (os.path.join(LAB, "gen3"), os.path.join(LAB, "gen2")):
    if p not in sys.path:
        sys.path.insert(0, p)
import hermit_fmt as H          # noqa: E402  (gen2, read only)
import sim                      # noqa: E402  (gen2, read only)
import fmt3 as F3               # noqa: E402  (gen3, read only: validate_args)

GEN = "recovery_errors"
SUITE = "tool_errors_v3"
TOOLS = json.load(open(os.path.join(SPEC, "hermit_tools_v3.json")))
SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TOOL_NAMES = list(SCHEMA)
META = json.load(open(os.path.join(SPEC, "hermit_tools_v3.meta.json")))
SYSTEM_TPL = open(os.path.join(SPEC, "SYSTEM_AGENT_v3.txt")).read().strip("\n")
assert "{BIO}" in SYSTEM_TPL and len(TOOLS) == 30
REQ_OVERRIDE = {"prepare_send": ["to_address", "amount", "symbol"]}      # chain stays optional (app drops an unnamed chain)

# ------------------------------------------------------------------ small helpers
HIDDEN = "(hidden: this text read like instructions to the assistant)"


class Retry(Exception):
    pass


class Lit(dict):
    """A result the app writes with its literal key order (not sorted)."""


def ser(r):
    return json.dumps(r, ensure_ascii=False, sort_keys=not isinstance(r, Lit))


def short(x):
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def fnum(x):
    """Plain decimal text of a number (no e-notation, no trailing zeros)."""
    if isinstance(x, str):
        return x
    d = Decimal(str(x))
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def usd(x):
    """$ amount with the digits of x (>= 2 decimals unless whole), thousands comma."""
    d = Decimal(str(abs(x)))
    ip, _, fp = format(d, "f").partition(".")
    fp = fp.rstrip("0")
    if fp and len(fp) < 2:
        fp = fp.ljust(2, "0")
    return ("-" if x < 0 else "") + "$" + f"{int(ip):,}" + ("." + fp if fp else "")


def approx(v):
    """Units.approx of the app: >= 1000 -> 2 dp, else 3 significant digits, trailing zeros trimmed."""
    if v >= 1000:
        return fnum(round(v, 2))
    return fnum(float(f"{v:.3g}"))


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def join_and(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def first_sentence(t):
    m = re.search(r"(?<=[.!?])\s", t)
    return t[:m.start()] if m else t


def n_sentences(t):
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


CH = {"base": "Base", "arbitrum": "Arbitrum", "ethereum": "Ethereum", "polygon": "Polygon", "solana": "Solana",
      "bsc": "BNB Chain", "bnb": "BNB Chain", "monad": "Monad", "near": "NEAR"}
CH_WORDS = {"base": ["base", "Base", "base chain"], "arbitrum": ["arbitrum", "Arbitrum", "arb"], "ethereum": ["ethereum", "Ethereum", "eth mainnet"],
            "polygon": ["polygon", "Polygon", "matic"], "solana": ["solana", "Solana"], "bsc": ["bnb chain", "bsc", "BNB chain"]}
CHAIN_NAMES = ["Base", "Arbitrum", "Ethereum", "Polygon", "Solana", "BNB", "Monad", "NEAR", "Optimism", "Avalanche", "Bitcoin", "Hyperliquid"]
NAMES = ["Alice", "Bob", "Mike", "Sara", "Dan", "Priya", "Leo", "Maya", "Tom", "Nina", "Omar", "Zoe", "Ken", "Ivy", "Raj", "Carlos"]
SURNAMES = [("Alice", "Smith", "Chen"), ("Sam", "Lee", "Park"), ("Maria", "Gomez", "Rossi"), ("Chris", "Wong", "Evans"), ("Dana", "Kim", "Shah")]
MEMES = {"BONK": 0.00002, "WIF": 1.9, "JUP": 0.8, "TRUMP": 12.0, "FARTCOIN": 0.9, "POPCAT": 0.55, "MEW": 0.006, "PENGU": 0.03,
         "GOAT": 0.3, "MOODENG": 0.21, "BOME": 0.008, "PNUT": 0.5, "MYRO": 0.12, "SLERF": 0.2, "GIGA": 0.04, "BILLY": 0.07}
STOCKS = ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "META", "GOOGL", "SPX"]
COIN_SAY = {"BTC": ["BTC", "btc", "bitcoin"], "ETH": ["ETH", "eth", "ether"], "SOL": ["SOL", "sol", "solana"], "HYPE": ["HYPE", "hype"],
            "DOGE": ["DOGE", "doge"], "XRP": ["XRP", "xrp"], "SUI": ["SUI", "sui"], "AVAX": ["AVAX", "avax"], "LINK": ["LINK", "link"],
            "BNB": ["BNB", "bnb"], "ARB": ["ARB", "arb"], "ENA": ["ENA", "ena"], "TAO": ["TAO", "tao"], "NEAR": ["NEAR", "near"],
            "INJ": ["INJ", "inj"], "APT": ["APT", "apt"], "OP": ["OP"], "AAVE": ["AAVE", "aave"], "LTC": ["LTC", "ltc"]}
PERPS = ["BTC", "ETH", "SOL", "HYPE", "DOGE", "XRP", "SUI", "AVAX", "LINK", "BNB", "ENA", "TAO", "NEAR", "INJ", "APT", "AAVE", "LTC"]
TYPO_WORDS = {"please": "plase", "actually": "actualy", "withdraw": "withdrwa", "balance": "balnce", "position": "positon", "address": "adress",
              "portfolio": "portfolo", "account": "acount", "message": "mesage", "anything": "anythign", "deposit": "depsoit",
              "bridge": "brdige", "wanted": "wnated", "contact": "contcat", "automation": "automaton", "payment": "paymnet"}


def typo(rng, s, p=0.12):
    if rng.random() >= p:
        return s
    ws = [w for w in TYPO_WORDS if re.search(r"\b" + w + r"\b", s)]
    if not ws:
        return s
    w = rng.choice(ws)
    return re.sub(r"\b" + w + r"\b", TYPO_WORDS[w], s, count=1)


def sloppy(rng, s):
    """Casual register: sometimes drop the capital and the final period / question mark."""
    r = rng.random()
    if r < 0.30 and s[:1].isupper() and not re.match(r"[A-Z]{2,}", s) and not s.startswith("I "):
        s = s[0].lower() + s[1:]
    if rng.random() < 0.35:
        s = s.rstrip(".?!")
    return typo(rng, s)


def pick_bio(rng):
    return "Face ID" if rng.random() < 0.7 else "Touch ID"


def tp(rng, split, train_list, eval_list=(), p_eval=0.45):
    """Template pick: eval draws part of its phrasing from phrases the train split never uses."""
    if split == "eval" and eval_list and rng.random() < p_eval:
        return rng.choice(eval_list)
    return rng.choice(train_list)


YES = ["yes", "yes please", "yep", "ok do it", "go ahead", "sure", "yeah do that", "ok", "do it", "yes, that works", "sounds good, go",
       "y", "yup", "please do", "alright, go for it", "that works"]
YES_EVAL = ["yes go", "ok prepare it", "fine, do that", "yeah ok", "go on then", "perfect, do it"]
TRY_AGAIN = ["try again", "can you just retry it?", "run it again", "just try again please", "retry", "try it once more",
             "come on, try again", "do it again", "send it again", "can you try that again?"]
TRY_AGAIN_EVAL = ["one more go", "hit it again", "ok try again then", "again?", "just run the same thing again"]


def approve(rng):
    return rng.choice(["Review it and approve with Face ID.", "It's waiting for your Face ID approval.",
                       "Check the review, then approve with Face ID.", "Nothing happens until you approve it with Face ID.",
                       "The review is open. Approve it with Face ID when you're ready."])


class Mkt:
    """Prices and 24h moves for one conversation."""

    def __init__(self, rng):
        self.p = {s: sim.rnd_price(s, rng) for s in sim.ASSETS}
        self.p.update(sim.STABLES)
        for s, v in MEMES.items():
            if s not in self.p:
                self.p[s] = float(f"{v * rng.uniform(0.7, 1.4):.3g}")
        self.chg = {s: round(rng.uniform(-9, 9), 2) for s in self.p}

    def price(self, s):
        return self.p[s]

    def get_price(self, s):
        return {"change_24h_pct": short(self.chg[s]), "price": short(self.p[s]), "source": "Hyperliquid mark price", "symbol": s}


def ok_price(v):
    return short(v)


# ------------------------------------------------------------------ result builders (shapes from the v3 meta)
def r_prepared(kind):
    return Lit([("status", "review_opened"), ("kind", kind), ("executed", False),
                ("note", "Waiting for the user to approve with Face ID.")])


def r_blocked(*reasons):
    return {"note": "Nothing was prepared. Tell the user why.", "reasons": list(reasons), "status": "blocked"}


def r_quote(rng, mk, fs, fc, ts, tc, amt, prepared=False):
    pf, pt = mk.price(fs), mk.price(ts)
    usd_in = round(amt * pf, 2)
    bps = rng.choice([30, 35, 35, 40, 45])
    usd_out = round(usd_in * (1 - bps / 10000) - rng.uniform(0, 0.05 * max(1, usd_in) ** 0.5), 2)
    out = approx(usd_out / pt)
    r = {"amount_in_usd": short(usd_in), "amount_out": out, "amount_out_usd": short(usd_out), "fee_bps": bps,
         "from": f"{fnum(amt)} {fs} on {CH[fc]}", "route": "NEAR Intents", "seconds": rng.randint(9, 40),
         "signature_verified": True, "to": f"about {out} {ts} on {CH[tc]}", "total_cost_usd": short(round(usd_in - usd_out, 2))}
    if prepared:
        r.update({"note": "Nothing has moved. The Swap review is open with this NEAR Intents quote; the user approves it with Face ID or cancels.",
                  "policy": "allow", "status": "prepared_for_review"})
    return r


def r_send(amount_sym, network):
    return {"amount": amount_sym, "executed": False, "kind": "send", "network": network,
            "note": "Waiting for the user to approve with Face ID.", "status": "review_opened"}


def r_needs(kind, question, options, noun=None):
    return {"note": "Nothing was prepared. Ask the user which network, in one short question." if kind == "needs_chain"
            else "Nothing was prepared. Ask the user which one.",
            "options": options, "prepared": False, "question": question, "status": kind}


def r_page(page, tab=None):
    names = {"home": "Home", "wallets": "Wallets", "trade": "Trade", "earn": "Earn", "payments": "Payments", "explore": "Explore",
             "calendar": "Calendar", "projects": "Projects", "automations": "Automations", "files": "Files",
             "connectors": "Connectors", "settings": "Settings"}
    r = {"note": f"The user taps it to open {names[page]}.", "page": page, "screen": page, "status": "button_shown"}
    if tab:
        r["tab"] = tab
    return r


def addr_evm(rng):
    return H.evm_addr(rng, checksum=True)


def mint(rng):
    return H.sol_addr(rng)


def holdings_row(rng, mk, sym, chain, usd_target=None):
    p = mk.price(sym)
    t = usd_target if usd_target is not None else rng.choice([rng.uniform(8, 300), rng.uniform(300, 4000)])
    amt = round(t / p, 2 if p < 5 else (4 if p < 5000 else 6))
    if amt <= 0:
        amt = 1.0
    return {"amount": fnum(amt), "chain": chain, "symbol": sym, "usd": short(round(amt * p, 2))}


def spend_chains(rng, k):
    return rng.sample(["base", "arbitrum", "ethereum", "polygon"], k)

# ------------------------------------------------------------------ conversation plumbing
class Conv:
    """Builds one conversation from SCRIPTED tool results."""

    def __init__(self, rng, bio):
        self.rng, self.bio = rng, bio
        self.mac = bio == "Touch ID"
        self.dev = "Mac" if self.mac else "iPhone"
        self.click = "click" if self.mac else "tap"
        sysm = SYSTEM_TPL.replace("{BIO}", "Face ID")
        if rng.random() < 0.2:
            sysm += f"\nThe user wants to be called {rng.choice(H.NAMES)}."
        self.msgs = [{"role": "system", "content": sysm}]
        self.expect, self.turn, self.log = [], 0, []

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        self.expect = []
        self.turn += 1

    def call(self, *calls):
        """calls: (name, args, result). Several calls go in ONE assistant message (parallel)."""
        tcs = [{"name": n, "arguments": a} for n, a, _ in calls]
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        self.expect.append({"calls": tcs})
        for n, a, r in calls:
            self.msgs.append({"role": "tool", "content": ser(r)})
            self.log.append((self.turn, n, a, r))
        out = [r for _, _, r in calls]
        return out if len(out) > 1 else out[0]

    def calc(self, expr):
        r = json.loads(H.calc_result(expr))
        assert "result" in r, expr
        self.call(("calculator", {"expression": expr}, r))
        return r["result"]

    def say(self, text, ask=False):
        ask = ask or "?" in text
        self.msgs.append({"role": "assistant", "content": text})
        self.expect.append({"ask": True} if ask else {"text": True})

    def after_last_user(self):
        return [x for x in self.log if x[0] == self.turn]


def rec(c, kind, group, gold, checks):
    ex = {"messages": c.msgs, "tools": True,
          "meta": {"source": GEN, "generator": GEN, "suite": SUITE, "kind": kind, "group": group, "bio": c.bio, "device": c.dev,
                   "lang": "en", "tools_registry": "spec/hermit_tools_v3.json", "world": {"scripted": True, "device": c.dev},
                   "expect": c.expect, "gold": gold, "checks": checks,
                   "script": script_from_msgs(c.msgs, 1)}}
    return apply_bio(ex)


def apply_bio(ex):
    bio = ex["meta"]["bio"]
    if bio == "Face ID":
        return ex

    def fix(o):
        if isinstance(o, str):
            return o.replace("Face ID", bio)
        if isinstance(o, list):
            return [fix(x) for x in o]
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        return o
    for m in ex["messages"]:
        for k in ("content",):
            if isinstance(m.get(k), str):
                m[k] = fix(m[k])
    ex["meta"]["script"] = fix(ex["meta"]["script"])
    return ex


def gold_of(c, **kw):
    g = {"steps": c.expect}
    g.update(kw)
    return g


RETRY_RE = re.compile(r"try again|timed out|timeout|is busy", re.I)
TIMEOUT_ERRS = ["network timeout, try again", "The request timed out. Try again.", "The network is busy. Try again in a few seconds.",
                "network timeout, try again", "The request timed out. Try again."]
QUOTE_BUSY = "1Click is busy. Try again in a moment."
TRANSIENT_NO_RETRY = ["The price feed isn't responding right now.", "Hyperliquid didn't answer.", "Service unavailable.",
                      "Couldn't reach the network for that.", "The data source didn't respond."]

# ------------------------------------------------------------------ user-language helpers
def cw(rng, c):
    return rng.choice(CH_WORDS[c])


def atext(rng, x):
    if isinstance(x, (int, float)) and x >= 1000 and float(x).is_integer() and rng.random() < 0.35:
        return f"{int(x):,}"
    return fnum(x)


def swap_text(rng, split, fs, fc, ts, tc, a, quote=False):
    """User message asking for a swap / bridge (or a quote of it)."""
    f, t = cw(rng, fc), cw(rng, tc)
    bridge = fs == ts and fc != tc
    same = fc == tc
    if quote:
        if same:
            tr = [f"quote me {a} {fs} to {ts} on {f}", f"how much {ts} would I get for {a} {fs} on {f}",
                  f"what's the rate, {a} {fs} into {ts} on {f}?", f"price check {a} {fs} to {ts} on {f}"]
            ev = [f"if I swapped {a} {fs} for {ts} on {f}, what comes out?", f"can you quote {a} {fs}->{ts} on {f}"]
        elif bridge:
            tr = [f"quote me bridging {a} {fs} from {f} to {t}", f"how much would I receive bridging {a} {fs} {f} to {t}?",
                  f"what does it cost to bridge {a} {fs} from {f} to {t}", f"get me a quote, {a} {fs} {f} -> {t}"]
            ev = [f"what would I get if I moved {a} {fs} from {f} over to {t}?", f"bridge quote please: {a} {fs}, {f} to {t}"]
        else:
            tr = [f"quote me {a} {fs} on {f} to {ts} on {t}", f"how much {ts} on {t} do I get for {a} {fs} on {f}?",
                  f"what would {a} {fs} from {f} turn into as {ts} on {t}", f"quote {a} {fs} {f} to {ts} {t}"]
            ev = [f"if I swap {a} {fs} on {f} for {ts} on {t}, what do I end up with?", f"rate check: {a} {fs} ({f}) into {ts} ({t})"]
    else:
        if same:
            tr = [f"swap {a} {fs} to {ts} on {f}", f"can you swap {a} {fs} for {ts} on {f}", f"convert {a} {fs} to {ts} on {f} pls",
                  f"I want {ts} for {a} {fs} on {f}", f"change {a} {fs} into {ts}, on {f}"]
            ev = [f"turn {a} {fs} into {ts} on {f}", f"exchange {a} {fs} for {ts} on {f} please"]
        elif bridge:
            tr = [f"bridge {a} {fs} from {f} to {t}", f"move {a} {fs} {f} -> {t}", f"can you bridge {a} {fs} to {t}? it's on {f}",
                  f"send {a} {fs} from {f} over to {t} (bridge it)", f"bridge {a} {fs} {f} to {t} please"]
            ev = [f"I need {a} {fs} on {t}, take it from {f}", f"get {a} {fs} from {f} onto {t}"]
        else:
            tr = [f"swap {a} {fs} on {f} to {ts} on {t}", f"can you swap {a} {fs} from {f} into {ts} on {t}", f"convert {a} {fs} on {f} to {ts} ({t})",
                  f"I want to turn {a} {fs} on {f} into {ts} on {t}", f"{a} {fs} {f} -> {ts} {t}"]
            ev = [f"exchange {a} {fs} on {f} for {ts} over on {t}", f"put {a} {fs} from {f} into {ts} on {t}"]
    return sloppy(rng, tp(rng, split, tr, ev))


def cap(s):
    return s[:1].upper() + s[1:]


def txt_swap_prepared(rng, r):
    return rng.choice([
        f"I've prepared the swap: {r['from']} to {r['to']} through {r['route']}. Total cost is about ${r['total_cost_usd']}. {approve(rng)}",
        f"The swap is prepared: {r['from']} for {r['to']}, route {r['route']}, total cost about ${r['total_cost_usd']}. {approve(rng)}",
        f"I've prepared it. You pay {r['from']} and get {r['to']} through {r['route']}, total cost about ${r['total_cost_usd']}. {approve(rng)}"])


def txt_quote(rng, r):
    return rng.choice([
        f"For {r['from']} you'd get {r['to']} through {r['route']}. Total cost is about ${r['total_cost_usd']} and it takes around {r['seconds']} seconds. Want me to prepare it?",
        f"{cap(r['from'])} gets you {r['to']} through {r['route']}, total cost about ${r['total_cost_usd']}, around {r['seconds']} seconds. Shall I prepare it?"])


def r_buy(rng, mk, sym, amt_usd):
    return {"executed": False, "kind": "buy", "note": "Nothing has moved. The Buy review is open with this Jupiter quote; the user approves it with Face ID or cancels.",
            "price_impact_pct": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08]), "route": rng.choice(["Jupiter via Orca, Raydium", "Jupiter via Raydium"]),
            "slippage_pct": 0.5, "status": "review_opened", "symbol": sym, "you_get": f"about {approx(amt_usd / mk.price(sym))} {sym}",
            "you_pay": f"{fnum(amt_usd)} USDC on Solana"}


def txt_buy_prepared(rng, r):
    return rng.choice([
        f"I've prepared the buy: you pay {r['you_pay']} and get {r['you_get']} through {r['route']}. {approve(rng)}",
        f"The buy is prepared. {cap(r['you_pay'])} for {r['you_get']}, route {r['route']}. {approve(rng)}"])


def r_sell(rng, mk, sym, amount, all_of=True):
    p = mk.price(sym)
    return {"executed": False, "kind": "sell", "note": "Nothing has moved. The Sell review is open with this Jupiter quote; the user approves it with Face ID or cancels.",
            "price_impact_pct": rng.choice([0.01, 0.02, 0.04, 0.06]), "route": "Jupiter via Orca, Raydium", "slippage_pct": 0.5,
            "status": "review_opened", "symbol": sym, "you_get": f"about {approx(float(amount) * p * 0.995)} USDC on Solana",
            "you_sell": f"{fnum(amount)} {sym} ({'all of your ' + sym if all_of else 'part of your ' + sym})"}


def txt_sell_prepared(rng, r):
    return rng.choice([
        f"I've prepared the sale: you sell {r['you_sell']} and get {r['you_get']}. {approve(rng)}",
        f"The sell is prepared. You sell {r['you_sell']} for {r['you_get']} through {r['route']}. {approve(rng)}"])


def r_deposit(rng, amount_sym, chain, apy=None):
    return {"amount": amount_sym, "apy_pct": apy if apy is not None else rng.choice([2.9, 3.4, 3.87, 4.21, 4.6, 5.12]), "executed": False,
            "kind": "deposit", "note": "Waiting for the user to review the deposit and approve it with Face ID.",
            "pool": f"Aave v3 on {CH[chain]}", "status": "review_opened"}


def txt_deposit_prepared(rng, r):
    return rng.choice([f"I've prepared a deposit of {r['amount']} into {r['pool']} at {fnum(r['apy_pct'])}% APY. {approve(rng)}",
                       f"The deposit is prepared: {r['amount']} into {r['pool']}, currently {fnum(r['apy_pct'])}% APY. {approve(rng)}"])


# ================================================================== 1. quote below the minimum
STABLE_ROUTES = [("USDC", "base", "ETH", "arbitrum"), ("USDC", "base", "USDC", "arbitrum"), ("USDC", "arbitrum", "USDC", "base"),
                 ("USDC", "polygon", "USDC", "base"), ("USDC", "base", "SOL", "solana"), ("USDT", "ethereum", "USDC", "base"),
                 ("USDC", "solana", "USDC", "base"), ("USDC", "ethereum", "USDC", "arbitrum"), ("USDC", "base", "ETH", "base"),
                 ("USDC", "arbitrum", "ETH", "ethereum"), ("USDC", "polygon", "POL", "polygon")]
VOL_ROUTES = [("ETH", "base", "USDC", "arbitrum"), ("ETH", "arbitrum", "USDC", "base"), ("ETH", "ethereum", "USDC", "base"),
              ("SOL", "solana", "USDC", "base"), ("BNB", "bsc", "USDC", "base"), ("POL", "polygon", "USDC", "base"),
              ("ETH", "base", "USDC", "base"), ("SOL", "solana", "USDC", "solana")]


def sc_quote_min(rng, split):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    m = rng.choices([1, 2, 3, 5], [17, 1, 1, 1])[0]
    err = {"error": f"Amount is below the minimum for this route (about ${m})."}
    as_quote = rng.random() < 0.5
    volatile = rng.random() < 0.38
    verb = "quote" if as_quote else "prepare"
    if volatile:
        fs, fc, ts, tc = rng.choice(VOL_ROUTES)
        price = mk.price(fs)
        u = rng.uniform(0.3, 0.85) * m
        amt = float(f"{u / price:.2g}")
        if not (amt * price < m * 0.97):
            raise Retry
    else:
        fs, fc, ts, tc = rng.choice(STABLE_ROUTES)
        amt = rng.choice([0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.35, 0.45, 0.55])
        amt = round(amt * m, 2)
        if amt >= m:
            raise Retry
    a = atext(rng, amt)
    c.user(swap_text(rng, split, fs, fc, ts, tc, a, quote=as_quote))
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": short(amt)}
    c.call(("get_quote" if as_quote else "prepare_swap", args, err))
    if volatile:
        pr = c.call(("get_price", {"symbol": fs}, mk.get_price(fs)))
        raw = 1.5 * m / pr["price"]
        d = max(2, 2 - int(__import__("math").floor(__import__("math").log10(raw))))
        valid = c.calc(f"round({m} * 1.5 / {fnum(pr['price'])}, {d})")
        if not (float(valid) * pr["price"] >= m * 1.4) or float(valid) <= 0:
            raise Retry
        c.say(rng.choice([
            f"{a} {fs} is below the minimum for this route, about ${m}. {fs} is {usd(pr['price'])} now, so {valid} {fs} is a safe size above it. Want me to {verb} {valid} {fs}?",
            f"That's too small for this route, the minimum is about ${m}. At {usd(pr['price'])} per {fs}, {valid} {fs} would clear it. Shall I {verb} that instead?",
            f"I can't {verb} {a} {fs} here: the minimum is about ${m}. {valid} {fs} would be enough at today's {usd(pr['price'])}. Want that amount?"]), ask=True)
    else:
        expr = rng.choice([f"{m} * 1.5", f"{m} + 0.5", f"{m} * 2", f"round({m} * 1.2, 2)"])
        valid = c.calc(expr)
        if float(valid) < m:
            raise Retry
        c.say(rng.choice([
            f"{a} {fs} is below the minimum for this route, which is about ${m}. {valid} {fs} would clear it. Want me to {verb} that instead?",
            f"That's too small for this route: the minimum is about ${m}. If you make it {valid} {fs}, it clears the minimum. Shall I {verb} {valid}?",
            f"I can't {verb} {a} {fs} on this route. The minimum is about ${m}, so {valid} {fs} would be enough. Want that?"]), ask=True)
    follow = rng.random() < 0.45
    if follow:
        c.user(tp(rng, split, YES, YES_EVAL))
        args2 = dict(args, amount=short(float(valid)))
        if as_quote:
            r = c.call(("get_quote", args2, r_quote(rng, mk, fs, fc, ts, tc, float(valid))))
            c.say(txt_quote(rng, r))
        else:
            r = c.call(("prepare_swap", args2, r_quote(rng, mk, fs, fc, ts, tc, float(valid), prepared=True)))
            c.say(txt_swap_prepared(rng, r))
    return rec(c, "quote_min" + ("_yes" if follow else ""), "recovery",
               gold_of(c, retry="never", min_usd=m, valid_amount=valid, must_any=[f"${m}", "minimum"], first_calls=[{"name": "get_quote" if as_quote else "prepare_swap", "arguments": args}],
                       no_prepare_before_yes=True), STD + ["no_repeat_blocked"])


STD = ["retry_once_exact", "error_stated", "no_invented_data", "no_prepared_after_error", "terminates"]


# ================================================================== 2. blocked + reasons
LIMIT_ALTS = ["Tell me a smaller amount and I'll try it, or do it yourself in Wallets.",
              "Give me a smaller amount and I'll try again with that, or you can do it yourself in Wallets.",
              "A smaller amount might be allowed, or you can do it yourself in Wallets."]
TRADE_LIMIT_ALTS = ["Tell me a smaller size and I'll try it, or place the order yourself in Trade.",
                    "Give me a smaller size and I'll try that, or you can place it yourself in Trade."]
BLOCK_RETRY_REPLY = ["The reason hasn't changed, so the same request would be blocked again and I won't send it twice.",
                     "It would be blocked the same way, so I won't repeat it.",
                     "Running it again won't change the answer, so I won't retry it."]


def sc_blocked(rng, split, var=None):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    var = var or rng.choice(["swap_limit", "swap_today", "trade_limit", "trade_xyz", "send_scam", "buy_unverified", "buy_limit", "multisend_limit"])
    smaller_ok = False
    redo = None
    if var in ("swap_limit", "swap_today"):
        fs, fc, ts, tc = rng.choice([("USDC", "base", "ETH", "arbitrum"), ("USDC", "base", "USDC", "arbitrum"), ("USDC", "arbitrum", "ETH", "base"),
                                     ("USDC", "ethereum", "SOL", "solana"), ("USDT", "ethereum", "USDC", "base")])
        amt = rng.choice([500, 750, 900, 1000, 1200, 1500, 2000, 2500, 3000, 5000, 8000])
        a = atext(rng, amt)
        c.user(swap_text(rng, split, fs, fc, ts, tc, a))
        args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": amt}
        reason = "Amount is above your daily limit for the assistant." if var == "swap_limit" else "The assistant has reached its limit for today."
        c.call(("prepare_swap", args, r_blocked(reason)))
        if var == "swap_limit":
            why, alt, smaller_ok = f"{a} {fs} is above your daily limit for the assistant", rng.choice(LIMIT_ALTS), True
            must = ["daily limit"]
        else:
            why, alt = "the assistant has reached its limit for today", "You can still do the swap yourself in Wallets."
            must = ["limit for today", "reached its limit"]
        tool, targs = "prepare_swap", args
    elif var in ("trade_limit", "trade_xyz"):
        stock = var == "trade_xyz"
        coin = rng.choice(STOCKS[:6]) if stock else rng.choice(PERPS)
        side, lev = rng.choice(["long", "short"]), rng.choice([2, 3, 5, 10])
        size = rng.choice([500, 1000, 1500, 2000, 3000, 5000, 10000, 25000])
        say = rng.choice(COIN_SAY.get(coin, [coin]))
        c.user(sloppy(rng, tp(rng, split, [f"{side} {say} {atext(rng, size)} at {lev}x", f"open a {side} on {say}, ${atext(rng, size)}, {lev}x leverage",
                                        f"I want to {side} {say} with {atext(rng, size)} usd {lev}x", f"{say} {side} ${atext(rng, size)} {lev}x market"],
                              [f"put on a {lev}x {side} in {say} for {atext(rng, size)} dollars", f"go {side} {say}, size {atext(rng, size)}, leverage {lev}"])))
        args = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
        if stock:
            reason = "trade.xyz has its own balance, separate from your Hyperliquid account. Move USDC to it in Trade first."
            why, alt, must = "trade.xyz has its own balance, separate from your Hyperliquid account", "Move USDC to it in Trade first, then ask me again.", ["trade.xyz", "own balance"]
        else:
            reason = "This trade is above your daily limit for the assistant."
            why, alt, smaller_ok, must = "that trade is above your daily limit for the assistant", rng.choice(TRADE_LIMIT_ALTS), True, ["daily limit"]
        c.call(("prepare_trade", args, r_blocked(reason)))
        tool, targs = "prepare_trade", args
        redo = "trade"
    elif var == "send_scam":
        sym, ch = rng.choice([("USDC", "base"), ("USDC", "arbitrum"), ("ETH", "ethereum"), ("USDT", "ethereum")])
        amt = rng.choice([20, 50, 100, 250, 500, 1000, 0.5, 1.5])
        to = addr_evm(rng)
        c.user(tp(rng, split, [f"send {atext(rng, amt)} {sym} to {to}", f"transfer {atext(rng, amt)} {sym} to {to} on {cw(rng, ch)}",
                               f"pay {to} {atext(rng, amt)} {sym}", f"can you send {atext(rng, amt)} {sym} to {to}?"],
                  [f"{sym} {atext(rng, amt)} -> {to}", f"I need to send {atext(rng, amt)} {sym} to {to}, it's for a deal"]))
        named = bool(re.search(r"\bon\b", c.msgs[-1]["content"]))
        args = {"to_address": to, "amount": short(amt), "symbol": sym}
        if named:
            args["chain"] = ch
        reason = "Recipient flagged as a known scam address."
        c.call(("prepare_send", args, r_blocked(reason)))
        why, must = "that address is flagged as a known scam address", ["scam"]
        alt = rng.choice(["I won't send to it. If someone gave it to you, check it with them another way first.",
                          "Please don't send to it. If it came from someone, ask them for the address again through a channel you trust."])
        tool, targs = "prepare_send", args
    elif var in ("buy_unverified", "buy_limit"):
        sym = rng.choice(list(MEMES))
        if var == "buy_unverified":
            sym = rng.choice(["XYZ", "MOONCAT", "ZORP", "PEPEGOLD", "SHIBX", "LAMBO", "DOGEKING", "ROCKET", "GRIMACE", "BASED"])
            amt = rng.choice([10, 20, 25, 50, 100])
            reason = f"There's no verified {sym} token on Solana, so Hermit won't buy one."
            why, alt, must = f"there's no verified {sym} token on Solana, so Hermit won't buy one", "", ["verified"]
        else:
            amt = rng.choice([500, 1000, 2000, 5000])
            reason = "Amount is above your daily limit for the assistant."
            why, alt, smaller_ok, must = f"${atext(rng, amt)} is above your daily limit for the assistant", rng.choice(LIMIT_ALTS[:2]).replace("do it yourself in Wallets", "buy it yourself in Trade"), True, ["daily limit"]
        c.user(sloppy(rng, tp(rng, split, [f"buy ${atext(rng, amt)} of {sym}", f"get me {atext(rng, amt)} dollars of {sym}", f"I want to ape {atext(rng, amt)} usd into {sym}",
                                         f"buy {sym} with {atext(rng, amt)} usdc", f"grab ${atext(rng, amt)} {sym} for me"],
                              [f"pick up {atext(rng, amt)} bucks of {sym}", f"{sym} buy, {atext(rng, amt)} usd"])))
        args = {"symbol": sym, "amount_usd": amt}
        c.call(("prepare_buy", args, r_blocked(reason)))
        tool, targs = "prepare_buy", args
        if var == "buy_limit":
            mk.p.setdefault(sym, MEMES[sym])
            redo = "buy"
    else:  # multisend_limit
        sym, ch = "USDC", rng.choice(["base", "arbitrum"])
        k = rng.choice([3, 4])
        rows = [{"address": addr_evm(rng), "amount": rng.choice([100, 150, 200, 250, 300, 400, 500])} for _ in range(k)]
        lines = "\n".join(f"{r['address']} {r['amount']}" for r in rows)
        c.user(tp(rng, split, [f"send these USDC:\n{lines}", f"batch send in USDC please\n{lines}", f"pay all of these in usdc\n{lines}"],
                  [f"multisend usdc:\n{lines}", f"send each of them the amount shown, usdc\n{lines}"]))
        args = {"symbol": sym, "recipients": rows}
        reason = "Total is above your daily limit for the assistant."
        c.call(("prepare_multisend", args, r_blocked(reason)))
        why, alt, smaller_ok, must = "the total is above your daily limit for the assistant", "Tell me smaller amounts and I'll try again, or send the batch yourself in Wallets.", True, ["daily limit"]
        tool, targs = "prepare_multisend", args
        redo = None
    first = rng.choice([f"I couldn't prepare that: {why}.", f"I couldn't prepare that, because {why}.", f"That didn't go to review: {why}."])
    c.say((first + (" " + alt if alt else "")).strip())
    # follow-ups
    r = rng.random()
    kind = var
    if r < 0.20:
        c.user(tp(rng, split, TRY_AGAIN, TRY_AGAIN_EVAL))
        c.say(rng.choice(BLOCK_RETRY_REPLY) + (" " + alt if alt else ""))
        kind += "_retry_asked"
    elif r < 0.35 and smaller_ok and redo:
        if redo == "trade":
            ns = int(targs["size_usd"] // 5)
            c.user(sloppy(rng, tp(rng, split, [f"ok do {ns} then", f"try {ns} instead", f"what about {ns}"], [f"fine, size {ns}", f"go with {ns} then"])))
            a2 = dict(targs, size_usd=ns)
            rr = c.call(("prepare_trade", a2, r_prepared("trade")))
            c.say(f"I've prepared a {a2['side']} on {a2['coin']}, {usd(ns)} at {a2['leverage']}x. {approve(rng)}")
        else:
            ns = int(targs["amount_usd"] // 10)
            c.user(sloppy(rng, tp(rng, split, [f"ok do {ns} then", f"try {ns} instead", f"what about {ns} dollars"], [f"fine, {ns} then", f"go with {ns}"])))
            a2 = dict(targs, amount_usd=ns)
            rr = c.call(("prepare_buy", a2, r_buy(rng, mk, targs["symbol"], ns)))
            c.say(txt_buy_prepared(rng, rr))
        kind += "_smaller"
    elif r < 0.50 and "limit" in " ".join(must) + why:
        c.user(tp(rng, split, ["what is my limit?", "how much can you do per day?", "whats the limit then", "what's my daily limit set to"],
                  ["so what's the cap", "how high is the limit"]))
        c.call(("open_page", {"page": "settings", "tab": "safety"}, r_page("settings", "safety")))
        c.say(rng.choice(["I can't read the limit from here. I put a button below that opens Safety in Settings, where you can see it.",
                          "I don't have the number. The button below opens Safety in Settings, where the limit is."]))
        kind += "_ask_limit"
    return rec(c, "blocked_" + kind, "recovery",
               gold_of(c, retry="never", blocked_call={"name": tool, "arguments": targs}, must_any=must, reasons=[reason], no_cap_numbers=True),
               STD + ["no_repeat_blocked"])


# ================================================================== 3. insufficient: show the holding, offer the maximum
def sc_insufficient(rng, split, var=None):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    var = var or rng.choice(["deposit", "buy", "sell", "withdraw"])
    follow = rng.random() < 0.45
    if var == "deposit":
        sym = rng.choice(["USDC", "USDC", "USDC", "ETH"])
        chain = rng.choice(["base", "arbitrum", "ethereum"])
        want = rng.choice([50, 100, 200, 500, 1000, 2500]) if sym == "USDC" else rng.choice([0.5, 1, 2, 0.25])
        have = round(want * rng.choice([0.1, 0.2, 0.3, 0.4, 0.6]), 2 if sym == "USDC" else 3)
        named = rng.random() < 0.5
        w, ct = atext(rng, want), cw(rng, chain)
        c.user(sloppy(rng, tp(rng, split, [f"deposit {w} {sym} into aave" + (f" on {ct}" if named else ""), f"put {w} {sym} in aave" + (f" on {ct}" if named else ""),
                                         f"lend {w} {sym} on aave" + (f" ({ct})" if named else ""), f"I want to earn on {w} {sym}, deposit it in aave" + (f" on {ct}" if named else "")],
                              [f"supply {w} {sym} to aave" + (f" on {ct}" if named else ""), f"stake {w} {sym} in aave" + (f" on {ct}" if named else "")])))
        args = {"project": "aave-v3", "symbol": sym}
        if named:
            args["chain"] = chain
        args["amount"] = want
        holding = f"{fnum(have)} {sym} on {CH[chain]}"
        c.call(("prepare_deposit", args, {"holding": holding, "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"}))
        c.say(rng.choice([f"You hold {holding}, which is less than the {w} {sym} you asked for. Want me to deposit {fnum(have)} {sym} instead?",
                          f"You only have {holding}, not {w} {sym}. Shall I deposit the {fnum(have)} {sym} you have?",
                          f"I can't deposit {w} {sym} because you hold {holding}. Want me to deposit {fnum(have)} {sym}, the most you can?"]), ask=True)
        if follow:
            c.user(tp(rng, split, YES, YES_EVAL))
            a2 = dict(args, amount=short(have))
            rr = c.call(("prepare_deposit", a2, r_deposit(rng, f"{fnum(have)} {sym}", chain)))
            c.say(txt_deposit_prepared(rng, rr))
        must = [holding.split(" on ")[0], "hold"]
    elif var == "buy":
        sym = rng.choice(list(MEMES))
        want = rng.choice([10, 20, 25, 50, 100, 200])
        usdc = round(want * rng.choice([0.1, 0.16, 0.25, 0.32, 0.4]), 2)
        sol = rng.choice([0.01, 0.02, 0.005, 0.03])
        holding = f"{fnum(usdc)} USDC and {fnum(sol)} SOL on Solana"
        c.user(sloppy(rng, tp(rng, split, [f"buy ${atext(rng, want)} of {sym}", f"get me {atext(rng, want)} dollars of {sym}", f"ape {atext(rng, want)} usd into {sym}",
                                         f"buy {sym}, {atext(rng, want)} bucks"], [f"I'd like {atext(rng, want)} usd worth of {sym}", f"pick up {sym} for ${atext(rng, want)}"])))
        args = {"symbol": sym, "amount_usd": want}
        c.call(("prepare_buy", args, {"holding": holding, "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"}))
        c.say(rng.choice([f"You hold {holding}, less than the ${atext(rng, want)} you asked for. Want me to buy ${fnum(usdc)} of {sym} with the USDC instead?",
                          f"You only have {holding}. Shall I buy ${fnum(usdc)} of {sym}, the most your USDC covers?"]), ask=True)
        if follow:
            c.user(tp(rng, split, YES, YES_EVAL))
            a2 = {"symbol": sym, "amount_usd": short(usdc)}
            rr = c.call(("prepare_buy", a2, r_buy(rng, mk, sym, usdc)))
            c.say(txt_buy_prepared(rng, rr))
        must = [fnum(usdc), "hold"]
    elif var == "sell":
        sym = rng.choice(list(MEMES))
        have = round(rng.uniform(1, 9000) if mk.price(sym) < 1 else rng.uniform(1, 40), rng.choice([0, 1, 2]))
        have = short(float(have)) if have > 0 else 1
        want = short(round(have * rng.choice([1.5, 2, 3, 5]), 2))
        holding = f"{fnum(have)} {sym}"
        c.user(sloppy(rng, tp(rng, split, [f"sell {atext(rng, want)} {sym}", f"dump {atext(rng, want)} of my {sym}", f"sell {atext(rng, want)} {sym} for usdc",
                                         f"I want to sell {atext(rng, want)} {sym}"], [f"offload {atext(rng, want)} {sym}", f"get rid of {atext(rng, want)} {sym} please"])))
        args = {"token": sym, "amount": want}
        c.call(("prepare_sell", args, {"holding": holding, "note": "The user holds less than that. Tell them.", "prepared": False, "status": "insufficient"}))
        c.say(rng.choice([f"You hold {holding}, not {atext(rng, want)}. Want me to sell all {fnum(have)} {sym}?",
                          f"You only have {holding}, so I can't sell {atext(rng, want)}. Shall I sell all of it?"]), ask=True)
        if follow:
            c.user(tp(rng, split, YES, YES_EVAL))
            a2 = {"token": sym, "percent": 100}
            rr = c.call(("prepare_sell", a2, r_sell(rng, mk, sym, have)))
            c.say(txt_sell_prepared(rng, rr))
        must = [fnum(have), "hold"]
    else:
        sym = rng.choice(["USDC", "USDC", "USDT", "ETH"])
        chain = rng.choice(["base", "arbitrum", "ethereum"])
        have = round(rng.uniform(5, 400), 2) if sym != "ETH" else round(rng.uniform(0.05, 2), 3)
        want = short(round(have * rng.choice([1.4, 2, 3, 5]), 2))
        supplied = f"{fnum(have)} {sym}"
        w = atext(rng, want)
        c.user(sloppy(rng, tp(rng, split, [f"withdraw {w} {sym} from aave", f"take {w} {sym} out of aave", f"pull {w} {sym} from my aave deposit",
                                         f"I want {w} {sym} back from aave"], [f"redeem {w} {sym} from aave", f"get {w} {sym} out of aave"])))
        args = {"symbol": sym, "amount": want}
        c.call(("prepare_withdraw", args, {"note": f"Only {supplied} is in Aave on {CH[chain]}. Tell the user.", "prepared": False, "status": "insufficient", "supplied": supplied}))
        c.say(rng.choice([f"Only {supplied} is in Aave on {CH[chain]}, less than the {w} {sym} you asked for. Want me to withdraw all of it?",
                          f"You have {supplied} in Aave on {CH[chain]}, not {w}. Shall I withdraw everything that's there?"]), ask=True)
        if follow:
            c.user(tp(rng, split, YES, YES_EVAL))
            rr = c.call(("prepare_withdraw", {"symbol": sym}, r_prepared("withdraw")))
            c.say(rng.choice([f"I've prepared the withdrawal of all your {sym} from Aave. {approve(rng)}", f"The withdrawal of all your {sym} in Aave is prepared. {approve(rng)}"]))
        must = [fnum(have), "aave"]
    return rec(c, f"insufficient_{var}" + ("_yes" if follow else ""), "recovery",
               gold_of(c, retry="never", must_any=must, offer_max=True, no_prepare_before_yes=True), STD + ["no_repeat_blocked"])


# ================================================================== 4. needs_chain / needs_choice: ask verbatim, call again
def sc_needs_ask(rng, split, var=None):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    var = var or rng.choices(["send", "deposit", "sell", "buy", "withdraw"], [18, 9, 9, 8, 8])[0]
    if var in ("send", "deposit"):
        sym = "USDC" if var == "deposit" or rng.random() < 0.8 else rng.choice(["USDT", "ETH"])
        ch1, ch2 = rng.sample(["base", "arbitrum", "ethereum", "polygon"], 2)
        h1 = round(rng.uniform(40, 900), 1) if sym != "ETH" else round(rng.uniform(0.2, 3), 3)
        h2 = round(h1 * rng.choice([0.2, 0.35, 0.5, 0.7, 1.4]), 1 if sym != "ETH" else 3)
        q = f"Which network should the {var} use? You hold {sym} on {CH[ch1]} and {CH[ch2]} ({CH[ch1]}: {fnum(h1)}, {CH[ch2]}: {fnum(h2)})."
        nr = r_needs("needs_chain", q, [ch1, ch2])
        amt = rng.choice([5, 10, 12.5, 20, 25, 30, 40, 50, 75]) if var == "send" else rng.choice([5, 10, 20, 25])
        if amt > min(h1, h2):
            amt = min(5, int(min(h1, h2)))
        a = atext(rng, amt)
        if var == "send":
            to = addr_evm(rng)
            c.user(sloppy(rng, tp(rng, split, [f"send {a} {sym} to {to}", f"transfer {a} {sym} to {to}", f"pay {to} {a} {sym}", f"can you send {a} {sym} to {to}?"],
                                  [f"{a} {sym} to {to} please", f"i need to send {to} {a} {sym}"])))
            args = {"to_address": to, "amount": short(amt), "symbol": sym}
        else:
            c.user(sloppy(rng, tp(rng, split, [f"deposit {a} {sym} into aave", f"put {a} {sym} in aave", f"lend {a} {sym} on aave", f"aave deposit {a} {sym}"],
                                  [f"supply {a} {sym} to aave", f"earn on {a} {sym} via aave"])))
            args = {"project": "aave-v3", "symbol": sym, "amount": short(amt)}
        c.call(("prepare_send" if var == "send" else "prepare_deposit", args, nr))
        c.say(q, ask=True)
        pick = rng.choice([ch1, ch2])
        other = ch2 if pick == ch1 else ch1
        cwp = cw(rng, pick)
        c.user(sloppy(rng, tp(rng, split, [cwp, f"{cwp} please", f"on {cwp}", f"the {cwp} one", f"use {cwp}", f"{cwp}, thanks"], [f"go with {cwp}", f"{cwp} pls"])))
        a2 = dict(args)
        if var == "send":
            a2["chain"] = pick
            rr = c.call(("prepare_send", a2, r_send(f"{fnum(amt)} {sym}", CH[pick])))
            c.say(rng.choice([f"I've prepared the send of {rr['amount']} on {rr['network']}. {approve(rng)}",
                              f"The send is prepared: {rr['amount']} on {rr['network']}. {approve(rng)}"]))
        else:
            a2 = {"project": "aave-v3", "symbol": sym, "chain": pick, "amount": short(amt)}
            rr = c.call(("prepare_deposit", a2, r_deposit(rng, f"{fnum(amt)} {sym}", pick)))
            c.say(txt_deposit_prepared(rng, rr))
        must = [CH[ch1], CH[ch2]]
        kind = f"needs_chain_{var}"
        gold = gold_of(c, retry="never", ask_verbatim=q, must_any=must, answer_chain=pick, network_in_final=CH[pick])
    elif var in ("sell", "buy"):
        sym = rng.choice(["PUMP", "MOODENG", "WIF", "BONK", "GOAT", "PNUT", "MEW", "POPCAT", "GIGA", "BILLY"])
        m1, m2 = mint(rng), mint(rng)
        mk.p.setdefault(sym, MEMES.get(sym, 0.5))
        if var == "sell":
            a1, a2_ = rng.choice([120, 250, 5000, 1000, 40]), rng.choice([4, 12, 7, 90, 2])
            ex = lambda m: f"{m[:6]}…{m[-3:]}"
            q = f"You hold two tokens called {sym}: {a1} ({ex(m1)}) and {a2_} ({ex(m2)}). Which one?"
            opts = [{"amount": str(a1), "mint": m1, "symbol": sym}, {"amount": str(a2_), "mint": m2, "symbol": sym}]
            pct = rng.choice([100, 50, 25, 100, 100])
            say = {100: ["sell all my", "dump my", "sell everything in", "sell my"], 50: ["sell half my", "sell 50% of my"], 25: ["sell 25% of my", "sell a quarter of my"]}[pct]
            c.user(sloppy(rng, tp(rng, split, [f"{rng.choice(say)} {sym}"], [f"{rng.choice(say)} {sym} please"])))
            args = {"token": sym, "percent": pct}
            c.call(("prepare_sell", args, {"note": "Nothing was prepared. Ask the user which one.", "options": opts, "prepared": False, "question": q, "status": "needs_choice"}))
            c.say(q, ask=True)
            big = 0 if a1 > a2_ else 1
            which = rng.choice([("the bigger one", big), ("the one with %s" % (a1 if big == 0 else a2_), big), ("the smaller one", 1 - big), ("the %s one" % a1, 0), ("the %s one" % a2_, 1),
                                ("first one", 0), ("second", 1)])
            c.user(which[0])
            ch = opts[which[1]]
            rr = c.call(("prepare_sell", {"token": ch["mint"], "percent": pct}, r_sell(rng, mk, sym, ch["amount"], all_of=pct == 100)))
            c.say(txt_sell_prepared(rng, rr))
            must = [sym, "which"]
        else:
            l1, l2 = rng.choice([9100000, 4200000, 1500000]), rng.choice([6200000, 800000, 310000])
            q = f"Two tokens are called {sym}. Pick the one you mean on Spot or Memes, or paste its mint."
            opts = [{"liquidity_usd": l1, "mint": m1, "symbol": sym}, {"liquidity_usd": l2, "mint": m2, "symbol": sym}]
            amt = rng.choice([10, 20, 25, 50, 100])
            c.user(sloppy(rng, tp(rng, split, [f"buy ${amt} of {sym}", f"get me {amt} dollars of {sym}", f"buy {sym} with {amt} usdc", f"ape {amt} into {sym}"],
                                  [f"{sym} for {amt} bucks", f"pick up ${amt} {sym}"])))
            args = {"symbol": sym, "amount_usd": amt}
            c.call(("prepare_buy", args, {"note": "Nothing was prepared. Ask the user which one.", "options": opts, "prepared": False, "question": q, "status": "needs_choice"}))
            c.say(q, ask=True)
            hi = 0 if l1 > l2 else 1
            c.user(rng.choice([f"the one with more liquidity", f"the bigger one", f"the first one", f"the second one", f"the one with less liquidity"]))
            txt = c.msgs[-1]["content"]
            idx = hi if "more liquidity" in txt or "bigger" in txt else (1 - hi if "less" in txt else (0 if "first" in txt else 1))
            rr = c.call(("prepare_buy", {"symbol": opts[idx]["mint"], "amount_usd": amt}, r_buy(rng, mk, sym, amt)))
            c.say(txt_buy_prepared(rng, rr))
            must = [sym, "mint"]
        kind = f"needs_choice_{var}"
        gold = gold_of(c, retry="never", ask_verbatim=q, must_any=must, answer_mint=ch["mint"] if var == "sell" else opts[idx]["mint"])
    else:
        sym = "USDC"
        ch1, ch2 = rng.sample(["base", "arbitrum", "ethereum"], 2)
        s1, s2 = round(rng.uniform(10, 300), 2), round(rng.uniform(5, 120), 2)
        q = f"Which deposit? {fnum(s1)} {sym} on {CH[ch1]}, {fnum(s2)} {sym} on {CH[ch2]}."
        c.user(sloppy(rng, tp(rng, split, ["withdraw my usdc from aave", "take my USDC out of aave", "get my usdc back from aave", "pull everything out of aave, the usdc",
                               "withdraw usdc from aave", "I want my usdc out of aave", "take out my aave usdc", "can you withdraw my USDC from aave", "pull my usdc out of aave please",
                               "aave withdraw usdc", "get my usdc out of the aave pool", "cash out my aave usdc deposit", "remove my usdc from aave", "return my usdc from aave to my wallet"],
                              ["I want my aave usdc back", "close my usdc deposit in aave", "unwind my aave usdc"])))
        c.call(("prepare_withdraw", {"symbol": sym}, {"note": "Nothing was prepared. Ask the user which one.", "prepared": False, "question": q, "status": "needs_choice"}))
        c.say(q, ask=True)
        pick = rng.choice([ch1, ch2])
        cwp = cw(rng, pick)
        c.user(sloppy(rng, tp(rng, split, [f"the {cwp} one", f"{cwp}", f"{cwp} please", f"from {cwp}"], [f"the one on {cwp}", f"{cwp} deposit"])))
        rr = c.call(("prepare_withdraw", {"symbol": sym, "chain": pick}, r_prepared("withdraw")))
        c.say(rng.choice([f"I've prepared the withdrawal of all your {sym} in Aave on {CH[pick]}. {approve(rng)}", f"The withdrawal from Aave on {CH[pick]} is prepared. {approve(rng)}"]))
        must = [CH[ch1], CH[ch2]]
        kind = "needs_choice_withdraw"
        gold = gold_of(c, retry="never", ask_verbatim=q, must_any=must, answer_chain=pick, network_in_final=CH[pick])
    return rec(c, kind, "recovery", gold, STD + ["no_repeat_blocked"])


# ================================================================== 5. no route: plan two steps
NOROUTE_ERRS = ["No route for this pair right now. Try a larger or smaller amount, or another token.",
                "No solver quoted this route right now. Try another amount or token."]
NR_FROM = [("ETH", "base", 0.01, 0.8), ("ETH", "arbitrum", 0.01, 0.8), ("ETH", "ethereum", 0.02, 2), ("SOL", "solana", 0.2, 20), ("BNB", "bsc", 0.05, 3), ("POL", "polygon", 20, 600)]


def sc_no_route(rng, split):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    fs, fc, lo, hi = rng.choice(NR_FROM)
    amt = float(f"{rng.uniform(lo, hi):.2g}")
    amt = short(amt)
    tc = rng.choice([x for x in ["base", "arbitrum", "polygon", "ethereum", "solana", "bsc"] if x != fc])
    as_quote = rng.random() < 0.4
    a = atext(rng, amt)
    c.user(swap_text(rng, split, fs, fc, "USDC", tc, a, quote=as_quote))
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": "USDC", "to_chain": tc, "amount": amt}
    err = {"error": rng.choice(NOROUTE_ERRS)}
    c.call(("get_quote" if as_quote else "prepare_swap", args, err))
    plan = rng.choice([
        f"There's no direct route for {a} {fs} on {CH[fc]} to USDC on {CH[tc]} right now. We can do it in two steps: first swap {a} {fs} to USDC on {CH[fc]}, then bridge that USDC to {CH[tc]} once it lands. Want me to prepare step one?",
        f"No route for that pair right now. Here's a plan: step 1, swap {a} {fs} to USDC on {CH[fc]}. Step 2, bridge the USDC to {CH[tc]} after step 1 lands. Shall I prepare step 1?",
        f"I couldn't find a direct route from {fs} on {CH[fc]} to USDC on {CH[tc]}. Two steps would work: swap {a} {fs} to USDC on {CH[fc]} first, then bridge the USDC to {CH[tc]}. Want me to prepare the first one?"])
    c.say(plan, ask=True)
    follow = rng.random() < 0.6
    if follow:
        c.user(tp(rng, split, YES + ["ok start with step one", "yes do step 1", "yes, the swap first", "ok prepare the first one"], YES_EVAL))
        a1 = dict(args, to_chain=fc)
        rr = c.call(("prepare_swap", a1, r_quote(rng, mk, fs, fc, "USDC", fc, amt, prepared=True)))
        c.say(rng.choice([
            f"I've prepared step 1: {rr['from']} to {rr['to']} through {rr['route']}. {approve(rng)} Once it lands, tell me and I'll prepare the bridge to {CH[tc]}.",
            f"Step 1 is prepared: {rr['from']} for {rr['to']}, total cost about ${rr['total_cost_usd']}. {approve(rng)} After it lands, ask me for the bridge to {CH[tc]}."]))
    return rec(c, "no_route" + ("_step1" if follow else ""), "recovery",
               gold_of(c, retry="never", must_any=["two steps", "step 1", "step one", "no route", "no direct route"], plan_steps=2, step1_args=dict(args, to_chain=fc),
                       no_prepare_before_yes=True, no_bridge_prepared=True), STD + ["no_repeat_blocked"])


# ================================================================== 6. one of two parallel calls fails
def fmt_chg(x):
    return f"up {fnum(abs(x))}%" if x >= 0 else f"down {fnum(abs(x))}%"


def signed_usd(x):
    return ("up " if x >= 0 else "down ") + usd(abs(x))


def r_account(rng, mk, coin, side, lev, null_liq=False):
    p = mk.price(coin)
    entry = round(p * (1 + rng.uniform(-0.06, 0.06)), sim.ASSETS[coin][2])
    value = round(rng.uniform(400, 6000), 2)
    notional = value * lev * rng.uniform(0.1, 0.4)
    size = round(notional / entry, 4 if p < 1000 else 5)
    sgn = 1 if side == "long" else -1
    pnl = round(sgn * (p - entry) * size, 2)
    liq = round(sim.liq_price(side, entry, lev, 0.005), sim.ASSETS[coin][2] or 0)
    margin = round(size * entry / lev, 2)
    pos = {"coin": coin, "entry": short(entry), "leverage": lev, "liquidation": None if null_liq else short(liq), "side": side, "size": short(size),
           "unrealised_pnl": short(pnl)}
    return {"account_value": short(round(value + pnl, 2)), "available_usdc": short(round(max(value - margin, 1), 2)), "margin_used": short(margin),
            "open_orders": [], "positions": [pos]}


def pos_sentence(p):
    return (f"Your {p['coin']} {p['side']} is {fnum(p['size'])} {p['coin']} at an entry of {usd(p['entry'])}, {p['leverage']}x, "
            f"with unrealised PnL {signed_usd(p['unrealised_pnl'])}")


def hold_sentence(rows, k=3):
    parts = [f"{r['amount']} {r['symbol']} on {CH[r['chain']]} ({usd(r['usd'])})" for r in rows[:k]]
    return join_and(parts) + (f" and {len(rows) - k} more" if len(rows) > k else "")


def sc_parallel_fail(rng, split, var=None):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    var = var or rng.choices(["prices", "acct_price", "bal_price", "pos_port"], [14, 14, 12, 12])[0]
    bad_err = lambda: {"error": rng.choice(TRANSIENT_NO_RETRY)}
    fail_idx = rng.choice([0, 1])
    if var == "prices":
        a, b = rng.sample(PERPS, 2)
        sa, sb = rng.choice(COIN_SAY[a]), rng.choice(COIN_SAY[b])
        c.user(sloppy(rng, tp(rng, split, [f"price of {sa} and {sb}?", f"what are {sa} and {sb} at", f"{sa} {sb} prices", f"how are {sa} and {sb} doing right now",
                                         f"check {sa} and {sb} for me"], [f"where's {sa} and also {sb} trading", f"give me {sa} + {sb} price"])))
        calls = [("get_price", {"symbol": a}, mk.get_price(a)), ("get_price", {"symbol": b}, mk.get_price(b))]
        calls[fail_idx] = (calls[fail_idx][0], calls[fail_idx][1], bad_err())
        res = c.call(*calls)
        okc, badc = calls[1 - fail_idx], calls[fail_idx]
        ok = okc[2]
        miss = badc[1]["symbol"]
        retry_call = (badc[0], badc[1], mk.get_price(miss))
        body = f"{ok['symbol']} is {usd(ok['price'])}, {fmt_chg(ok['change_24h_pct'])} in 24 hours."
        what = f"the {miss} price"
        merge = lambda rr: f"{body} {miss} is {usd(rr['price'])}, {fmt_chg(rr['change_24h_pct'])} in 24 hours."
        must = [miss]
    elif var == "acct_price":
        coin = rng.choice(["BTC", "ETH", "SOL", "HYPE", "DOGE", "SUI"])
        side, lev = rng.choice(["long", "short"]), rng.choice([2, 3, 5, 10])
        acct = r_account(rng, mk, coin, side, lev)
        sc = rng.choice(COIN_SAY[coin])
        c.user(sloppy(rng, tp(rng, split, [f"how's my {sc} {side} and where is {sc} now", f"my {sc} position and the current {sc} price", f"show my {sc} {side} plus the live price",
                                         f"how is my {sc} trade doing, and what's {sc} at"], [f"{sc} price and my position on it", f"where's {sc} and how am I doing on it"])))
        calls = [("hyperliquid_account", {}, acct), ("get_price", {"symbol": coin}, mk.get_price(coin))]
        calls[fail_idx] = (calls[fail_idx][0], calls[fail_idx][1], bad_err())
        c.call(*calls)
        p = acct["positions"][0]
        if fail_idx == 0:
            pr = calls[1][2]
            body = f"{coin} is {usd(pr['price'])}, {fmt_chg(pr['change_24h_pct'])} in 24 hours."
            what, retry_call, must = "your Hyperliquid account", ("hyperliquid_account", {}, acct), ["account", "position"]
            merge = lambda rr: f"{body} {pos_sentence(rr['positions'][0])}."
        else:
            body = pos_sentence(p) + "."
            what, retry_call, must = f"the live {coin} price", ("get_price", {"symbol": coin}, mk.get_price(coin)), [coin, "price"]
            merge = lambda rr: f"{body} {coin} is {usd(rr['price'])} now, {fmt_chg(rr['change_24h_pct'])} in 24 hours."
        miss = "account" if fail_idx == 0 else "price"
    elif var == "bal_price":
        coin = rng.choice(["ETH", "SOL", "BTC", "BNB"])
        k = rng.choice([1, 2, 3])
        chains = rng.sample(["base", "arbitrum", "ethereum", "solana", "polygon"], k)
        rows = [holdings_row(rng, mk, rng.choice(["USDC", "ETH"] if ch != "solana" else ["SOL", "USDC"]), ch) for ch in chains]
        sc = rng.choice(COIN_SAY[coin])
        c.user(sloppy(rng, tp(rng, split, [f"what do I hold and what's {sc} at", f"my balances and the {sc} price", f"show balances + {sc} price", f"what's in my wallet, and how's {sc}"],
                              [f"wallet contents and where {sc} is", f"list my tokens and tell me the {sc} price"])))
        calls = [("get_balances", {}, {"holdings": rows}), ("get_price", {"symbol": coin}, mk.get_price(coin))]
        calls[fail_idx] = (calls[fail_idx][0], calls[fail_idx][1], bad_err())
        c.call(*calls)
        if fail_idx == 0:
            pr = calls[1][2]
            body = f"{coin} is {usd(pr['price'])}, {fmt_chg(pr['change_24h_pct'])} in 24 hours."
            what, retry_call, must = "your balances", ("get_balances", {}, {"holdings": rows}), ["balances"]
            merge = lambda rr: f"{body} You hold {hold_sentence(rr['holdings'])}."
        else:
            body = f"You hold {hold_sentence(rows)}."
            what, retry_call, must = f"the live {coin} price", ("get_price", {"symbol": coin}, mk.get_price(coin)), [coin, "price"]
            merge = lambda rr: f"{body} {coin} is {usd(rr['price'])} now, {fmt_chg(rr['change_24h_pct'])} in 24 hours."
        miss = "balances" if fail_idx == 0 else "price"
    else:
        sym = "USDC"
        amt = round(rng.uniform(20, 900), 2)
        ch = rng.choice(["base", "arbitrum", "ethereum"])
        apy = rng.choice([3.4, 3.87, 4.21, 4.6, 5.12])
        positions = {"earn": [{"amount": fnum(amt), "apy_pct": apy, "chain": ch, "symbol": sym}], "perps": {"open_orders": [], "positions": []}, "spot": [], "wallet": "Main"}
        total = round(amt + rng.uniform(10, 600), 2)
        portfolio = {"by_place": [{"place": "Wallet", "usd": short(round(total - amt, 2))}, {"place": "Aave", "usd": short(amt)}],
                     "top": [{"place": "Aave", "usd": short(amt), "what": "USDC on " + CH[ch]}], "total_usd": short(total), "wallet": "Main"}
        c.user(sloppy(rng, tp(rng, split, ["show my positions and my total value", "what do I have in positions, and what's it all worth", "positions + total portfolio value",
                                         "give me my positions and the total in dollars", "list my open positions and tell me what everything is worth", "what are my positions, and the grand total?",
                                         "positions and portfolio value together", "show me what I'm in and how much it all adds up to", "my positions plus total worth pls",
                                         "break down my positions and give me the total", "open positions and net worth", "what do I have open and what's the total value",
                                         "I want my positions and my total balance in dollars", "tell me my positions, then the total"],
                              ["what am I holding beyond the wallet, and the overall total", "positions and total please", "give me both, positions and the total"])))
        calls = [("get_positions", {}, positions), ("get_portfolio", {}, portfolio)]
        calls[fail_idx] = (calls[fail_idx][0], calls[fail_idx][1], bad_err())
        c.call(*calls)
        if fail_idx == 0:
            body = f"Your total in the Main wallet is {usd(total)}."
            what, retry_call, must = "your positions", ("get_positions", {}, positions), ["positions"]
            merge = lambda rr: f"{body} In Aave you have {rr['earn'][0]['amount']} {rr['earn'][0]['symbol']} on {CH[rr['earn'][0]['chain']]} at {fnum(rr['earn'][0]['apy_pct'])}% APY."
        else:
            body = f"In Aave you have {fnum(amt)} {sym} on {CH[ch]} at {fnum(apy)}% APY, and nothing in spot tokens or Hyperliquid."
            what, retry_call, must = "your total value", ("get_portfolio", {}, portfolio), ["total"]
            merge = lambda rr: f"{body} Your total in the Main wallet is {usd(rr['total_usd'])}."
        miss = "positions" if fail_idx == 0 else "total"
    c.say(rng.choice([f"{body} I couldn't get {what} just now. Want me to try that again?",
                      f"Here's what I have. {body} {cap(what)} didn't come back, so I can't give it yet. Shall I try again?",
                      f"{body} The part about {what} failed to load. Want me to retry it?"]), ask=True)
    follow = rng.random() < 0.3
    if follow:
        c.user(tp(rng, split, YES + ["yes try it again", "please retry"], YES_EVAL))
        rr = c.call(retry_call)
        c.say(merge(rr))
    return rec(c, f"parallel_fail_{var}" + ("_retry" if follow else ""), "recovery",
               gold_of(c, retry="offer" if not follow else "user", must_any=must, answer_from_other=True, missing=what), STD)


# ================================================================== 7. the user corrects the request
APPROVE_NOTES = ["The earlier review is still open, so close it with Cancel to avoid approving the wrong one.",
                 "The first review is still open. Close it with Cancel, then approve this one.",
                 "Heads up: the earlier review is still open. Close it with Cancel so only this one is left.",
                 "The previous review is still open, so close it with Cancel before you approve this one."]


def sc_correction(rng, split, var=None):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    var = var or rng.choice(["send_chain", "send_amount", "swap_amount", "swap_dest", "trade_lev", "trade_size", "deposit_amount", "buy_amount"])
    note = rng.choice(APPROVE_NOTES)
    if var.startswith("send"):
        sym = rng.choice(["USDC", "USDC", "USDT"])
        ch, ch2 = rng.sample(["base", "arbitrum", "ethereum", "polygon"], 2)
        amt, amt2 = rng.choice([10, 20, 25, 30, 40, 100, 150]), rng.choice([15, 50, 60, 75, 200, 5])
        to = addr_evm(rng)
        a = atext(rng, amt)
        c.user(sloppy(rng, tp(rng, split, [f"send {a} {sym} to {to} on {cw(rng, ch)}", f"transfer {a} {sym} on {cw(rng, ch)} to {to}", f"pay {to} {a} {sym} on {cw(rng, ch)}"],
                              [f"{a} {sym} to {to}, {cw(rng, ch)} network", f"please send {to} {a} {sym} over {cw(rng, ch)}"])))
        args = {"to_address": to, "amount": short(amt), "symbol": sym, "chain": ch}
        r1 = c.call(("prepare_send", args, r_send(f"{fnum(amt)} {sym}", CH[ch])))
        c.say(rng.choice([f"I've prepared the send of {r1['amount']} on {r1['network']}. {approve(rng)}", f"The send is prepared: {r1['amount']} on {r1['network']}. {approve(rng)}"]))
        if var == "send_chain":
            w2 = cw(rng, ch2)
            c.user(sloppy(rng, tp(rng, split, [f"no, {w2}", f"wait, {w2} not {cw(rng, ch)}", f"actually {w2} instead", f"oops i meant {w2}", f"no no {w2}"], [f"sorry, make that {w2}", f"change it to {w2}"])))
            a2 = dict(args, chain=ch2)
            r2 = c.call(("prepare_send", a2, r_send(f"{fnum(amt)} {sym}", CH[ch2])))
            what = f"{r2['amount']} on {r2['network']}"
        else:
            c.user(sloppy(rng, tp(rng, split, [f"make it {amt2}", f"no, {amt2}", f"actually {amt2} not {a}", f"wait make that {amt2}", f"change the amount to {amt2}"], [f"sorry I meant {amt2}", f"let's do {amt2} instead"])))
            a2 = dict(args, amount=short(amt2))
            r2 = c.call(("prepare_send", a2, r_send(f"{fnum(amt2)} {sym}", CH[ch])))
            what = f"{r2['amount']} on {r2['network']}"
        c.say(rng.choice([f"I've prepared it again: {what}. {note}", f"Prepared again as {what}. {note}"]))
    elif var.startswith("swap"):
        fs, fc, ts, tc = rng.choice([("USDC", "base", "ETH", "arbitrum"), ("USDC", "arbitrum", "ETH", "base"), ("USDC", "base", "USDC", "arbitrum"), ("USDC", "polygon", "USDC", "base")])
        amt, amt2 = rng.choice([20, 50, 100, 200, 250, 30]), rng.choice([10, 40, 60, 75, 150, 500])
        a = atext(rng, amt)
        c.user(swap_text(rng, split, fs, fc, ts, tc, a))
        args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": amt}
        r1 = c.call(("prepare_swap", args, r_quote(rng, mk, fs, fc, ts, tc, amt, prepared=True)))
        c.say(txt_swap_prepared(rng, r1))
        if var == "swap_amount":
            c.user(sloppy(rng, tp(rng, split, [f"make it {amt2}", f"no, {amt2}", f"actually {amt2} not {a}", f"change it to {amt2}"], [f"hmm {amt2} instead", f"let's say {amt2}"])))
            a2 = dict(args, amount=amt2)
            r2 = c.call(("prepare_swap", a2, r_quote(rng, mk, fs, fc, ts, tc, amt2, prepared=True)))
        else:
            tc2 = rng.choice([x for x in ["base", "arbitrum", "polygon", "ethereum"] if x not in (tc, fc)])
            ts2 = ts
            w2 = cw(rng, tc2)
            c.user(sloppy(rng, tp(rng, split, [f"no, to {w2}", f"actually send it to {w2} instead", f"wait, destination {w2}", f"nah {w2}"], [f"i meant to {w2}", f"change the target to {w2}"])))
            a2 = dict(args, to_chain=tc2)
            r2 = c.call(("prepare_swap", a2, r_quote(rng, mk, fs, fc, ts2, tc2, amt, prepared=True)))
        c.say(rng.choice([f"I've prepared it again: {r2['from']} to {r2['to']}. {note}", f"Prepared again as {r2['from']} for {r2['to']}. {note}"]))
    elif var.startswith("trade"):
        coin = rng.choice(PERPS[:10])
        side, lev, lev2 = rng.choice(["long", "short"]), rng.choice([2, 3, 5]), rng.choice([4, 6, 8, 10, 7])
        size, size2 = rng.choice([100, 200, 250, 500, 1000]), rng.choice([150, 300, 400, 750, 2000])
        sc_ = rng.choice(COIN_SAY[coin])
        c.user(sloppy(rng, tp(rng, split, [f"{side} {sc_} {size} at {lev}x", f"open a {lev}x {side} on {sc_} with {size} usd", f"{sc_} {side}, ${size}, {lev}x, market"],
                              [f"I want a {side} in {sc_}: {size} dollars, {lev}x", f"{size} usd {side} {sc_} {lev}x"])))
        args = {"coin": coin, "side": side, "size_usd": size, "leverage": lev, "order_type": "market"}
        c.call(("prepare_trade", args, r_prepared("trade")))
        c.say(f"I've prepared a {side} on {coin}, {usd(size)} at {lev}x. {approve(rng)}")
        if var == "trade_lev":
            c.user(sloppy(rng, tp(rng, split, [f"make it {lev2}x", f"no, {lev2}x", f"actually {lev2}x leverage", f"change leverage to {lev2}"], [f"hmm {lev2}x instead", f"bump it to {lev2}x"])))
            a2 = dict(args, leverage=lev2)
        else:
            c.user(sloppy(rng, tp(rng, split, [f"make it {size2}", f"no, size {size2}", f"actually {size2} usd", f"change the size to {size2}"], [f"hmm {size2} instead", f"let's go {size2}"])))
            a2 = dict(args, size_usd=size2)
        c.call(("prepare_trade", a2, r_prepared("trade")))
        c.say(rng.choice([f"I've prepared it again: a {side} on {coin}, {usd(a2['size_usd'])} at {a2['leverage']}x. {note}", f"Prepared again as a {a2['leverage']}x {side} on {coin} with {usd(a2['size_usd'])}. {note}"]))
    elif var == "deposit_amount":
        sym, ch = "USDC", rng.choice(["base", "arbitrum", "ethereum"])
        amt, amt2 = rng.choice([50, 100, 200, 25, 500]), rng.choice([30, 60, 75, 150, 300])
        a = atext(rng, amt)
        c.user(sloppy(rng, tp(rng, split, [f"deposit {a} {sym} into aave", f"put {a} {sym} in aave", f"lend {a} {sym} on aave"], [f"aave: supply {a} {sym}", f"earn on {a} {sym} with aave"])))
        args = {"project": "aave-v3", "symbol": sym, "amount": amt}
        r1 = c.call(("prepare_deposit", args, r_deposit(rng, f"{fnum(amt)} {sym}", ch)))
        c.say(txt_deposit_prepared(rng, r1))
        c.user(sloppy(rng, tp(rng, split, [f"make it {amt2}", f"no, {amt2}", f"actually {amt2}", f"change it to {amt2}"], [f"hmm {amt2} instead", f"let's do {amt2}"])))
        r2 = c.call(("prepare_deposit", dict(args, amount=amt2), r_deposit(rng, f"{fnum(amt2)} {sym}", ch, apy=r1["apy_pct"])))
        c.say(rng.choice([f"I've prepared it again: a deposit of {r2['amount']} into {r2['pool']}. {note}", f"Prepared again as {r2['amount']} into {r2['pool']}. {note}"]))
    else:
        sym = rng.choice(list(MEMES))
        amt, amt2 = rng.choice([10, 20, 25, 50, 100]), rng.choice([15, 30, 40, 75, 200])
        c.user(sloppy(rng, tp(rng, split, [f"buy ${amt} of {sym}", f"get me {amt} dollars of {sym}", f"buy {sym} with {amt} usdc"], [f"I want {amt} usd of {sym}", f"{sym} for ${amt}"])))
        args = {"symbol": sym, "amount_usd": amt}
        r1 = c.call(("prepare_buy", args, r_buy(rng, mk, sym, amt)))
        c.say(txt_buy_prepared(rng, r1))
        c.user(sloppy(rng, tp(rng, split, [f"make it {amt2}", f"no, {amt2} dollars", f"actually {amt2}", f"change it to {amt2}"], [f"hmm {amt2} instead", f"let's do {amt2}"])))
        r2 = c.call(("prepare_buy", dict(args, amount_usd=amt2), r_buy(rng, mk, sym, amt2)))
        c.say(rng.choice([f"I've prepared it again: you pay {r2['you_pay']} and get {r2['you_get']}. {note}", f"Prepared again as {r2['you_pay']} for {r2['you_get']}. {note}"]))
    return rec(c, "correction_" + var, "recovery", gold_of(c, retry="user", final_must=["still open", "Cancel"], earlier_review_open=True, changed=var), STD)


# ================================================================== 8. the result network differs from the named chain
def sc_network_differs(rng, split, var=None):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    var = var or rng.choice(["send", "deposit"])
    named, held = rng.sample(["base", "arbitrum", "ethereum", "polygon"], 2)
    sym = rng.choice(["USDC", "USDC", "USDT"]) if var == "send" else "USDC"
    amt = rng.choice([10, 15, 20, 25, 40, 50, 80, 100, 150, 200])
    a = atext(rng, amt)
    X, Y = CH[named], CH[held]
    wn = cw(rng, named)
    if var == "send":
        to = addr_evm(rng)
        c.user(sloppy(rng, tp(rng, split, [f"send {a} {sym} to {to} on {wn}", f"transfer {a} {sym} on {wn} to {to}", f"pay {to} {a} {sym} on {wn}", f"{a} {sym} to {to} via {wn}"],
                              [f"please send {to} {a} {sym}, {wn} network", f"can you send {a} {sym} on {wn} to {to}?"])))
        args = {"to_address": to, "amount": short(amt), "symbol": sym, "chain": named}
        r = c.call(("prepare_send", args, r_send(f"{fnum(amt)} {sym}", Y)))
        c.say(rng.choice([
            f"The review is on {Y}, not {X}, because your {sym} is on {Y}. Check the network on the review before you approve it with Face ID, or cancel it if {Y} is wrong.",
            f"This send is on {Y}, not the {X} you asked for, since that's where your {sym} is. Look at the network on the review, then approve with Face ID or cancel it.",
            f"It's prepared on {Y}, not {X}: your {sym} sits on {Y}. Check the network on the review before approving with Face ID, or cancel it."]))
    else:
        c.user(sloppy(rng, tp(rng, split, [f"deposit {a} USDC into aave on {wn}", f"put {a} usdc in aave on {wn}", f"lend {a} USDC on {wn} via aave"], [f"aave {wn}: deposit {a} USDC", f"supply {a} usdc to aave, {wn}"])))
        args = {"project": "aave-v3", "symbol": "USDC", "chain": named, "amount": amt}
        r = c.call(("prepare_deposit", args, r_deposit(rng, f"{fnum(amt)} USDC", held)))
        c.say(rng.choice([
            f"The deposit is on {Y}, not {X}, because your USDC is on {Y}. Check the pool on the review before you approve it with Face ID, or cancel it.",
            f"This deposit goes to {r['pool']}, not {X}: your USDC is on {Y}. Check the review, then approve with Face ID or cancel it."]))
    return rec(c, "network_differs_" + var, "recovery",
               gold_of(c, retry="never", network_first=Y, named_chain=X, result_network=Y, must_any=[Y], prepared_ok=True), STD + ["network_said"])


# ================================================================== 9. odd results
def sc_odd(rng, split, var=None):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    var = var or rng.choices(["empty", "unreachable", "null_change", "null_liq", "hidden", "contra_bal", "contra_price", "contra_liq"], [6, 6, 6, 6, 8, 3, 3, 2])[0]
    must = []
    if var == "empty":
        flt = rng.random() < 0.3
        ch = rng.choice(["base", "arbitrum", "ethereum", "solana"])
        t = tp(rng, split, ["what do I have in my wallet?", "show my balances", "how much do I hold", "whats in my wallet", "check my balance pls", "what tokens do I own",
                            "what's my balance", "list what I own", "do I have any crypto in here", "how much is in my wallet", "show me my coins", "any funds in my wallet?",
                            "tell me my balances", "what am I holding", "wallet check"],
               ["list my holdings", "what's my wallet look like", "give me the wallet contents", "is there anything in my wallet"])
        if flt:
            t = tp(rng, split, [f"how much do I have on {cw(rng, ch)}", f"what's my {cw(rng, ch)} balance", f"balance on {cw(rng, ch)}?"], [f"anything on {cw(rng, ch)}?"])
        c.user(sloppy(rng, t))
        c.call(("get_balances", {"chain": ch} if flt else {}, {"holdings": []}))
        where = f" on {CH[ch]}" if flt else ""
        c.say(rng.choice([f"Your wallet shows no holdings{where}, so there's nothing to list. Want an Add funds button below to deposit?",
                          f"You don't hold anything{where} right now. Want a button to add funds?",
                          f"Nothing{where} yet: the wallet is empty. Should I show you where to add funds?"]), ask=True)
        must = ["no holdings", "don't hold anything", "empty", "nothing"]
    elif var == "unreachable":
        bad = rng.choice(["bnb", "ethereum", "polygon", "solana"])
        k = rng.choice([1, 2, 3])
        rows = [holdings_row(rng, mk, rng.choice(["USDC", "ETH"]), ch) for ch in rng.sample([x for x in ["base", "arbitrum", "ethereum", "polygon"] if x != bad], k)]
        c.user(sloppy(rng, tp(rng, split, ["what's in my wallet", "balances pls", "show me what I hold", "what do i have", "check my holdings", "how much do I have", "list my tokens", "what am I holding",
                                           "show my balances", "token balances please", "what funds do I have", "wallet overview"], ["give me my token balances", "wallet balance?", "what's sitting in my wallet"])))
        c.call(("get_balances", {}, {"holdings": rows, "unreachable": [bad]}))
        c.say(rng.choice([f"You hold {hold_sentence(rows)}. {CH[bad]} didn't respond, so anything you hold there is missing from this list.",
                          f"Here's what I could read: {hold_sentence(rows)}. I couldn't reach {CH[bad]}, so that chain may have more."]))
        must = [CH[bad]]
    elif var == "null_change":
        s = rng.choice(STOCKS)
        mk.p[s] = round(rng.uniform(80, 900), 2)
        c.user(sloppy(rng, tp(rng, split, [f"price of {s}", f"what's {s} at", f"how is {s} doing today", f"{s} price and move?"], [f"where is {s} trading", f"check {s}"])))
        r = {"change_24h_pct": None, "price": mk.p[s], "source": "trade.xyz perp on Hyperliquid (tracks the stock)", "symbol": s}
        c.call(("get_price", {"symbol": s}, r))
        c.say(rng.choice([f"{s} is {usd(r['price'])}. That's a perp on trade.xyz that tracks the stock and trades 24/7, and it gave me no 24-hour change right now.",
                          f"{s} is at {usd(r['price'])}, a trade.xyz perp that tracks the stock. I don't have its 24-hour change at the moment."]))
        must = [s]
    elif var == "null_liq":
        coin = rng.choice(["BTC", "ETH", "SOL", "HYPE", "DOGE"])
        side, lev = rng.choice(["long", "short"]), rng.choice([2, 3, 5, 10])
        acct = r_account(rng, mk, coin, side, lev, null_liq=True)
        sc = rng.choice(COIN_SAY[coin])
        c.user(sloppy(rng, tp(rng, split, [f"where's my {sc} liquidation", f"how far is my {sc} {side} from liq", f"what's my liq price on {sc}", f"am I close to liquidation on {sc}"], [f"{sc} liq level?", f"how safe is my {sc} position"])))
        c.call(("hyperliquid_account", {}, acct))
        p = acct["positions"][0]
        c.say(rng.choice([f"Hyperliquid shows no liquidation price for your {coin} {side} right now, so I can't give you one. {pos_sentence(p)}.",
                          f"I don't have a liquidation price for your {coin} {side}: the account returned none. {pos_sentence(p)}."]))
        must = ["no liquidation", "liquidation price"]
    elif var == "hidden":
        src = rng.choice(["activity", "activity", "markets", "yields"])
        if src == "activity":
            rows = [{"status": "confirmed", "time": "2026-11-10 14:32", "what": f"Sent {fnum(rng.choice([0.5, 2, 15, 40]))} USDC to 0x12…ab"},
                    {"status": "confirmed", "time": "2026-11-10 09:05", "what": HIDDEN},
                    {"status": "failed", "time": "2026-11-09 18:20", "what": f"Bought {fnum(rng.choice([4.82, 12.3]))} WIF with 10 USDC"}]
            c.user(sloppy(rng, tp(rng, split, ["show my recent activity", "what happened in my wallet lately", "last transactions?", "list my activity", "what did I do recently",
                              "any recent transactions", "show me my history", "what have I done lately", "recent wallet activity", "latest sends and swaps?", "what's been going on in my wallet"],
                      ["recent moves in my wallet", "show activity", "what are my last few transactions"])))
            c.call(("get_activity", {}, {"activity": rows, "in_progress": 0, "note": "Reviews the user has not approved yet do not appear here."}))
            c.say(rng.choice([f"Your latest rows: {rows[0]['what']}, confirmed, and {rows[2]['what']}, failed. One row's text was withheld because it read like instructions, so I can't tell you what it said.",
                              f"I can read two rows: {rows[0]['what']} (confirmed) and {rows[2]['what']} (failed). The middle row was hidden by the app because its text read like instructions, and I won't guess what it was."]))
            must = ["withheld", "hidden"]
        elif src == "markets":
            q = rng.choice(["SOL perp", "ETH perp", "BTC", "AVAX perp"])
            rows = [{"funding_hourly_pct": 0.00125, "max_leverage": 40, "name": "BTC-PERP", "price": 97250, "type": "perp", "volume_24h_usd": 1840000000},
                    {"funding_hourly_pct": -0.0031, "max_leverage": 20, "name": HIDDEN, "price": 142.37, "type": "perp", "volume_24h_usd": 410000000}]
            c.user(sloppy(rng, tp(rng, split, [f"search markets for {q}", f"find {q} markets", f"what markets match {q}"], [f"look up {q}"])))
            c.call(("search_markets", {"query": q}, {"markets": rows}))
            c.say(f"I found BTC-PERP at {usd(97250)} with hourly funding of {fnum(0.00125)}%. The second result's name was withheld because it read like instructions, so I can't tell you which market it is.")
            must = ["withheld"]
        else:
            rows = [{"apy_pct": 4.21, "asset": "USDC", "chain": "Base", "protocol": "Aave v3", "tvl_usd": 950000000},
                    {"apy_pct": 9.8, "asset": "USDC", "chain": "Base", "protocol": HIDDEN, "tvl_usd": 12000000}]
            c.user(sloppy(rng, tp(rng, split, ["best usdc yields on base?", "where can I earn on usdc", "top yields for usdc", "show usdc pools"], ["any good usdc pools"])))
            c.call(("get_yields", {"asset": "USDC"}, {"pools": rows}))
            c.say("Aave v3 on Base pays 4.21% on USDC. The other pool's protocol name was withheld because it read like instructions, so I can't tell you what it is or recommend it.")
            must = ["withheld"]
    elif var == "contra_bal":
        claimed = rng.choice([200, 500, 800, 1000, 2000])
        sym, ch = "USDC", rng.choice(["base", "arbitrum", "ethereum"])
        have = round(claimed * rng.choice([0.05, 0.12, 0.24, 0.4]), 1)
        send = int(claimed * rng.choice([0.5, 0.6]))
        to = addr_evm(rng)
        c.user(sloppy(rng, tp(rng, split, [f"I have {claimed} USDC on {cw(rng, ch)}, send {send} of it to {to}", f"I've got {claimed} usdc on {cw(rng, ch)}. please send {send} to {to}"],
                              [f"my {cw(rng, ch)} wallet has {claimed} USDC, move {send} to {to}"])))
        c.call(("get_balances", {}, {"holdings": [{"amount": fnum(have), "chain": ch, "symbol": sym, "usd": short(have)}]}))
        c.say(rng.choice([f"Your wallet shows {fnum(have)} {sym} on {CH[ch]}, not {claimed}, so I can't prepare a send of {send}. Want me to send up to {fnum(have)} instead?",
                          f"The app shows {fnum(have)} {sym} on {CH[ch]}, not the {claimed} you mentioned, so {send} is too much. Shall I send {fnum(have)} or less?"]), ask=True)
        must = [fnum(have)]
    elif var == "contra_price":
        coin = rng.choice(["BTC", "ETH", "SOL"])
        mk_p = mk.price(coin)
        claim = float(f"{mk_p * rng.choice([0.6, 0.7, 0.75, 1.3, 1.4]):.3g}")
        sz, lev = rng.choice([200, 500, 1000]), rng.choice([3, 5])
        sc = rng.choice(COIN_SAY[coin])
        c.user(sloppy(rng, tp(rng, split, [f"{sc} is at {fnum(claim)} right? long it, {sz} at {lev}x", f"since {sc} is around {fnum(claim)} now, open a {lev}x long with {sz}"],
                              [f"{sc} sitting at {fnum(claim)}, go long {sz} {lev}x"])))
        pr = c.call(("get_price", {"symbol": coin}, mk.get_price(coin)))
        c.say(rng.choice([f"{coin} is {usd(pr['price'])} right now, not {usd(claim)}. Do you still want the {lev}x long with {usd(sz)} at the live price?",
                          f"That's off: {coin} trades at {usd(pr['price'])}, not {usd(claim)}. Shall I still prepare the {lev}x long for {usd(sz)}?"]), ask=True)
        must = [fnum(pr["price"]), usd(pr["price"])]
    else:
        coin = rng.choice(["SOL", "ETH", "BTC", "HYPE"])
        side, lev = rng.choice(["long", "short"]), rng.choice([3, 5, 10])
        acct = r_account(rng, mk, coin, side, lev)
        real = acct["positions"][0]["liquidation"]
        claim = short(round(real * rng.choice([0.8, 0.85, 1.15, 1.2]), 2))
        sc = rng.choice(COIN_SAY[coin])
        c.user(sloppy(rng, tp(rng, split, [f"my {sc} {side} liquidates at {fnum(claim)}, right?", f"my {sc} liq is {fnum(claim)} isn't it", f"{sc} {side}: liq price is {fnum(claim)}, yes?"], [f"I think my {sc} liquidation is {fnum(claim)}"])))
        c.call(("hyperliquid_account", {}, acct))
        c.say(rng.choice([f"Not quite: the account shows a liquidation price of {usd(real)} for your {coin} {side}, not {usd(claim)}.",
                          f"Your {coin} {side} has a liquidation price of {usd(real)}, not {usd(claim)}. I'm going by what the account returned."]))
        must = [usd(real)]
    return rec(c, "odd_" + var, "recovery", gold_of(c, retry="never", must_any=must, trust_tool=var.startswith("contra")), STD)


# ================================================================== 10. requests no tool supports
def btc_addr(rng):
    return "bc1q" + "".join(rng.choice("023456789acdefghjklmnpqrstuvwxyz") for _ in range(38))


def sc_unsupported(rng, split, var=None):
    bio = pick_bio(rng)
    c = Conv(rng, bio)
    var = var or rng.choice(["btc_send", "chain", "ledger", "edit_order"])
    if var == "btc_send":
        amt = rng.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
        to = btc_addr(rng)
        c.user(sloppy(rng, tp(rng, split, [f"send {fnum(amt)} BTC to {to}", f"transfer {fnum(amt)} btc to {to}", f"pay {to} {fnum(amt)} bitcoin", f"can you send {fnum(amt)} bitcoin to {to}?"],
                              [f"{fnum(amt)} btc -> {to} please", f"i need to send {fnum(amt)} btc to {to}"])))
        c.say(rng.choice([f"I can't send Bitcoin from this wallet. I can quote or prepare a swap into BTC, since bitcoin works as a destination, but sending BTC out isn't something I can do.",
                          f"Sending BTC isn't supported here, so I haven't prepared anything. Bitcoin only works as a swap destination: I can quote a swap into BTC if you want."]))
        must = ["can't send", "isn't supported"]
    elif var == "chain":
        ch = rng.choice(["Optimism", "Avalanche", "Optimism", "Avalanche"])
        sym = rng.choice(["USDC", "ETH"])
        amt = rng.choice([20, 50, 100, 250, 0.1, 0.5])
        w = ch if rng.random() < 0.7 else ch.lower()
        c.user(sloppy(rng, tp(rng, split, [f"bridge {fnum(amt)} {sym} to {w}", f"send {fnum(amt)} {sym} over to {w}", f"swap {fnum(amt)} {sym} to {sym} on {w}", f"move {fnum(amt)} {sym} onto {w}"],
                              [f"i want {fnum(amt)} {sym} on {w}", f"get {fnum(amt)} {sym} to {w} for me"])))
        c.say(rng.choice([f"{ch} isn't a network Hermit supports, so I haven't prepared anything. It supports Ethereum, Base, Arbitrum, Polygon, Monad, Solana, NEAR and BNB Chain. Want a quote to one of those?",
                          f"I can't do {ch}: it isn't supported. The networks I can swap and bridge between are Ethereum, Base, Arbitrum, Polygon, Monad, Solana, NEAR and BNB Chain. Should I use one of those?"]), ask=True)
        must = ["isn't supported", "can't do", "isn't a network"]
    elif var == "ledger":
        c.user(sloppy(rng, tp(rng, split, ["can I connect my ledger?", "use my ledger to sign trades", "connect ledger nano", "i want to use my hardware wallet with hermit", "how do I plug in my Ledger",
                               "does this support ledger", "I keep my funds on a Ledger, can I link it", "can hermit sign with my hardware wallet", "pair my ledger nano x", "use my cold wallet here",
                               "connect a trezor", "sign with ledger instead"],
                              ["does hermit work with a ledger", "link my trezor or ledger", "can I hook up my hardware key"])))
        if rng.random() < 0.5:
            c.call(("open_page", {"page": "wallets"}, r_page("wallets")))
            c.say(f"Hermit can't connect a Ledger or any other hardware wallet; it works with the wallet on this {c.dev}. I put a button below that opens Wallets.")
        else:
            c.say(f"Hermit can't connect a Ledger or any other hardware wallet. It works with the wallet on this {c.dev}, and you manage wallets in the Wallets page.")
        must = ["can't connect", "hardware wallet"]
    else:
        coin = rng.choice(["ETH", "BTC", "SOL", "HYPE"])
        newp = rng.choice([2800, 3000, 95000, 90000, 140, 150, 28, 32])
        sc = rng.choice(COIN_SAY[coin])
        c.user(sloppy(rng, tp(rng, split, [f"change my {sc} limit order to {newp}", f"edit my {sc} order price to {newp}", f"move my {sc} limit order to {newp}", f"update my open {sc} order, new price {newp}"],
                              [f"can you modify my {sc} order to {newp}", f"adjust the {sc} limit to {newp}"])))
        c.say(rng.choice([f"I can't edit an order in place. I can cancel the {coin} order and prepare a new one at {newp}. Want that?",
                          f"Editing an open order isn't something I can do. What I can do is cancel the {coin} order and prepare a new one at {newp}. Should I start with the cancel?"]), ask=True)
        must = ["can't edit", "isn't something"]
    return rec(c, "unsupported_" + var, "recovery", gold_of(c, retry="never", must_any=must, no_money_call=True, chain_list=(var == "chain")), STD)


# ================================================================== 11. timeouts: ONE identical retry, then stop
def tm_price(rng, split, c, mk):
    sym = rng.choice(PERPS)
    sc = rng.choice(COIN_SAY[sym])
    t = tp(rng, split, [f"price of {sc}?", f"what's {sc} at", f"{sc} price", f"how much is {sc} right now", f"check {sc} for me"], [f"where is {sc} trading", f"{sc} now?"])
    r = mk.get_price(sym)
    return t, "get_price", {"symbol": sym}, r, lambda r: f"{sym} is {usd(r['price'])}, {fmt_chg(r['change_24h_pct'])} in 24 hours.", "the price feed", sym


def tm_balances(rng, split, c, mk):
    k = rng.choice([1, 2, 3])
    rows = [holdings_row(rng, mk, rng.choice(["USDC", "ETH"]), ch) for ch in rng.sample(["base", "arbitrum", "ethereum", "polygon"], k)]
    t = tp(rng, split, ["what's in my wallet", "show my balances", "what do I hold", "balances please", "how much do I have"], ["list my tokens", "check my wallet"])
    return t, "get_balances", {}, {"holdings": rows}, lambda r: f"You hold {hold_sentence(r['holdings'])}.", "your balances", None


def tm_account(rng, split, c, mk):
    coin = rng.choice(["BTC", "ETH", "SOL", "HYPE"])
    acct = r_account(rng, mk, coin, rng.choice(["long", "short"]), rng.choice([2, 3, 5, 10]))
    t = tp(rng, split, ["how's my hyperliquid account", "show my open positions", "what positions do I have open", "check my perp account", "how are my trades doing",
                            "hyperliquid account please", "what's my account value on hyperliquid", "how much margin am I using", "show my perps", "am I in profit on my trades",
                            "check my hyperliquid", "what do I have open on hyperliquid", "my perp positions and balance"], ["my hyperliquid status", "perp positions?", "how's the perp account looking"])
    return t, "hyperliquid_account", {}, acct, lambda r: f"Your account value is {usd(r['account_value'])} with {usd(r['available_usdc'])} available. {pos_sentence(r['positions'][0])}.", "your Hyperliquid account", None


def tm_yields(rng, split, c, mk):
    asset = rng.choice(["USDC", "ETH", "USDT"])
    pools = [{"apy_pct": a, "asset": asset, "chain": ch, "protocol": pr, "tvl_usd": tv} for a, ch, pr, tv in
             [(6.8, "Base", "Morpho", 182000000), (4.21, "Base", "Aave v3", 950000000)]]
    pools[0]["apy_pct"] = rng.choice([5.2, 6.8, 7.4, 8.1])
    t = tp(rng, split, [f"best {asset} yields?", f"where can I earn on {asset}", f"top {asset} pools", f"show me {asset} yield options"], [f"{asset} apy right now", f"what pays most on {asset}"])
    return t, "get_yields", {"asset": asset}, {"pools": pools}, lambda r: f"{r['pools'][0]['protocol']} on {r['pools'][0]['chain']} pays {fnum(r['pools'][0]['apy_pct'])}% on {asset}, and {r['pools'][1]['protocol']} pays {fnum(r['pools'][1]['apy_pct'])}%.", "the yield data", asset


def tm_portfolio(rng, split, c, mk):
    total = round(rng.uniform(80, 9000), 2)
    w = round(total * rng.choice([0.3, 0.5, 0.7]), 2)
    r = {"by_place": [{"place": "Wallet", "usd": short(w)}, {"place": "Hyperliquid", "usd": short(round(total - w, 2))}], "top": [{"place": "Wallet", "usd": short(w), "what": "USDC on Base"}],
         "total_usd": short(total), "wallet": "Main"}
    t = tp(rng, split, ["how much do I have in total", "what's my total portfolio value", "total worth of everything?", "portfolio value", "what's everything worth",
                            "total value of my account", "how much am I worth here", "add up everything I own", "my total in dollars", "overall portfolio please", "what's my net worth in hermit"],
               ["what is all of it worth", "my net worth in here", "sum of all my holdings in usd"])
    return t, "get_portfolio", {}, r, lambda r: f"Your total in the {r['wallet']} wallet is {usd(r['total_usd'])}: {usd(r['by_place'][0]['usd'])} in the wallet and {usd(r['by_place'][1]['usd'])} on Hyperliquid.", "your portfolio value", None


def tm_quote(rng, split, c, mk):
    fs, fc, ts, tc = rng.choice([("USDC", "base", "ETH", "arbitrum"), ("USDC", "arbitrum", "USDC", "base"), ("USDC", "polygon", "ETH", "base")])
    amt = rng.choice([20, 50, 100, 250, 500])
    a = atext(rng, amt)
    t = swap_text(rng, split, fs, fc, ts, tc, a, quote=True)
    args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": amt}
    return t, "get_quote", args, r_quote(rng, mk, fs, fc, ts, tc, amt), lambda r: txt_quote(rng, r), "the quote service", None


TM = [tm_price, tm_price, tm_balances, tm_account, tm_yields, tm_portfolio, tm_quote]


def sc_timeout(rng, split, twice):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    swap_busy = (not twice) and rng.random() < 0.15
    if swap_busy:
        fs, fc, ts, tc = rng.choice([("USDC", "base", "ETH", "arbitrum"), ("USDC", "arbitrum", "USDC", "base")])
        amt = rng.choice([20, 50, 100, 250])
        a = atext(rng, amt)
        t = swap_text(rng, split, fs, fc, ts, tc, a)
        name = "prepare_swap"
        args = {"from_symbol": fs, "from_chain": fc, "to_symbol": ts, "to_chain": tc, "amount": amt}
        ok = r_quote(rng, mk, fs, fc, ts, tc, amt, prepared=True)
        ans, label, must = (lambda r: txt_swap_prepared(rng, r)), "the swap", None
        err1 = QUOTE_BUSY
    else:
        t, name, args, ok, ans, label, must = rng.choice(TM)(rng, split, c, mk)
        err1 = rng.choice(TIMEOUT_ERRS)
    c.user(sloppy(rng, t))
    c.call((name, args, {"error": err1}))
    if twice:
        c.call((name, args, {"error": rng.choice(TIMEOUT_ERRS)}))
        c.say(rng.choice([f"I couldn't reach {label} after two tries and I won't guess, so please try again in a moment.",
                          f"{cap(label)} didn't answer after two tries, so I have nothing reliable to tell you and I won't guess."]))
        kind, must_any = "timeout_twice", [label.replace("the ", ""), "two tries"]
    else:
        r = c.call((name, args, ok))
        c.say(ans(r))
        kind, must_any = "timeout_retry_ok", []
    return rec(c, kind + "_" + name, "recovery", gold_of(c, retry="auto", retry_call={"name": name, "arguments": args}, must_any=must_any, one_sentence=twice), STD)


# ================================================================== 12. errors of every new tool (8 per tool)
def tool_rec(c, tool, variant, must, retry="never", **extra):
    return rec(c, f"tool_{tool}_{variant}", "tools", gold_of(c, retry=retry, must_any=must, tool=tool, variant=variant, **extra), STD + ["no_repeat_blocked"])


def sol_token(rng):
    return rng.choice(list(MEMES))


def sell_user(rng, split, sym, pct):
    say = {100: ["sell all my", "dump my", "sell everything in", "sell my", "get rid of my"], 50: ["sell half my", "sell 50% of my", "cash out half of my"], 25: ["sell a quarter of my", "sell 25% of my"]}[pct]
    return sloppy(rng, tp(rng, split, [f"{rng.choice(say)} {sym}", f"{rng.choice(say)} {sym} for usdc", f"{rng.choice(say)} ${sym}"], [f"{rng.choice(say)} {sym} please", f"quick, {rng.choice(say)} {sym}"]))


def t_sell(rng, split, variant):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    sym = sol_token(rng)
    pct = rng.choice([100, 100, 50, 25])
    if variant == "not_held":
        c.user(sell_user(rng, split, sym, pct))
        c.call(("prepare_sell", {"token": sym, "percent": pct}, {"note": f"The user holds no {sym} on Solana. Tell them.", "prepared": False, "status": "not_held"}))
        c.say(rng.choice([f"You don't hold any {sym} on Solana, so there's nothing to sell. Want me to list what you do hold?",
                          f"I can't sell {sym} because you hold none on Solana. Shall I show you your holdings?"]), ask=True)
        return tool_rec(c, "sell", variant, [f"no {sym}", "hold none", "don't hold", "nothing to sell"])
    if variant == "frozen":
        c.user(sell_user(rng, split, sym, pct))
        reason = f"Your {sym} balance is frozen by the token's authority, so it can't be sold."
        c.call(("prepare_sell", {"token": sym, "percent": pct}, r_blocked(reason)))
        c.say(rng.choice([f"I couldn't prepare that: your {sym} balance is frozen by the token's authority, so it can't be sold. Nothing I can do changes that.",
                          f"Your {sym} balance is frozen by the token's authority, so it can't be sold and I haven't prepared anything."]))
        return tool_rec(c, "sell", variant, ["frozen"])
    if variant == "no_route":
        c.user(sell_user(rng, split, sym, pct))
        c.call(("prepare_sell", {"token": sym, "percent": pct}, {"error": "No route to sell that token right now."}))
        c.say(rng.choice([f"There's no route to sell {sym} right now, so I haven't prepared anything. Try again a bit later.",
                          f"I couldn't prepare the sale: no route to sell {sym} is available right now. Ask me again in a while."]))
        return tool_rec(c, "sell", variant, ["no route"])
    # needs_choice
    m1, m2 = mint(rng), mint(rng)
    a1, a2_ = rng.choice([120, 250, 5000, 1000, 40]), rng.choice([4, 12, 7, 90, 2])
    ex = lambda m: f"{m[:6]}…{m[-3:]}"
    q = f"You hold two tokens called {sym}: {a1} ({ex(m1)}) and {a2_} ({ex(m2)}). Which one?"
    opts = [{"amount": str(a1), "mint": m1, "symbol": sym}, {"amount": str(a2_), "mint": m2, "symbol": sym}]
    c.user(sell_user(rng, split, sym, pct))
    c.call(("prepare_sell", {"token": sym, "percent": pct}, {"note": "Nothing was prepared. Ask the user which one.", "options": opts, "prepared": False, "question": q, "status": "needs_choice"}))
    c.say(q, ask=True)
    idx = rng.choice([0, 1])
    pick = rng.choice([f"the {a1 if idx == 0 else a2_} one", "first one" if idx == 0 else "second one", f"the one ending {opts[idx]['mint'][-3:]}"])
    c.user(pick)
    ch = opts[idx]
    rr = c.call(("prepare_sell", {"token": ch["mint"], "percent": pct}, r_sell(rng, mk, sym, ch["amount"], all_of=pct == 100)))
    c.say(txt_sell_prepared(rng, rr))
    return tool_rec(c, "sell", variant, [sym, "which"], ask_verbatim=q, answer_mint=ch["mint"])


def t_search(rng, split, variant):
    bio = pick_bio(rng)
    c = Conv(rng, bio)
    q = rng.choice(["BONKERS", "wif gold", "moonrocket", "PEPEGOLD", "zorpcoin", "SOLDOGE", "grimace token", "babyfloki", "catwifhatt", "turbo frog", "MOOCOIN", "pumpkin"])
    buy = rng.random() < 0.4
    if buy:
        amt = rng.choice([10, 20, 50, 100])
        t = tp(rng, split, [f"buy ${amt} of {q}", f"I want to buy {q}, {amt} dollars", f"ape {amt} into {q}"], [f"get me {q} for ${amt}", f"{q} buy {amt} usd"])
    else:
        t = tp(rng, split, [f"is {q} safe", f"find the {q} token", f"look up {q} on solana", f"search {q}", f"what's {q}, is it legit"], [f"check out {q} for me", f"any token called {q}?"])
    c.user(sloppy(rng, t))
    if variant == "empty":
        c.call(("search_tokens", {"query": q}, {"note": "No Solana token matched. Check the spelling or paste the mint.", "tokens": []}))
        c.say(rng.choice([f"I found no Solana token matching {q}. Check the spelling, or paste its mint and I'll look it up." + (" I won't buy anything I can't find." if buy else ""),
                          f"Nothing on Solana matched {q}. Double-check the name, or paste the mint and I'll search again." + (" I haven't prepared a buy." if buy else "")]))
        return tool_rec(c, "search", variant, ["no solana token", "nothing on solana", "matched"])
    c.call(("search_tokens", {"query": q}, {"error": "Token search is unavailable right now."}))
    c.say(rng.choice([f"Token search is unavailable right now, so I can't check {q}. I won't guess whether it's safe" + (", and I haven't prepared a buy." if buy else ". Try again in a bit."),
                      f"I couldn't look up {q} because token search is unavailable right now. I won't guess whether it's safe" + (" or buy it unchecked." if buy else ", so try again later.")]))
    return tool_rec(c, "search", variant, ["unavailable"])


def err_wallet(c):
    return {"error": f"No wallet on this {c.dev} yet"}


def t_positions(rng, split, variant, portfolio=False):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    name = "get_portfolio" if portfolio else "get_positions"
    if portfolio:
        t = tp(rng, split, ["how much do I have in total", "what's my total portfolio value", "total worth of everything?", "portfolio value", "what's it all worth",
                            "how much am I worth in total", "give me my total balance across everything", "what is my net worth here", "total value of my stuff?",
                            "sum up my whole portfolio", "how big is my portfolio", "what's the grand total", "all-in value of my account", "how much money do I have overall",
                            "show my portfolio", "portfolio check", "what's my total in usd"],
               ["my overall total please", "everything I own in dollars", "how rich am I, in total", "total across wallet, aave and hyperliquid"])
        args = {}
    else:
        kind = rng.choice([None, None, "perps", "earn", "spot"])
        t = tp(rng, split, ["what positions do I have", "show my positions", "any open positions?", "what do I hold besides my wallet", "list my aave and perp positions",
                            "what am I in right now", "show me everything I have open", "do I have any positions", "positions please", "what's open on my account",
                            "any deposits or trades running?", "list my open stuff", "what do I have invested", "check my positions pls", "what are my current positions"],
               ["what else do I have besides balances", "my positions?", "anything running for me", "show open positions and deposits"])
        if kind:
            t = tp(rng, split, {"perps": ["show my perp positions", "any open perps?"], "earn": ["what do I have in aave", "show my earn deposits"], "spot": ["which solana tokens do I hold", "show my spot tokens"]}[kind], [])
        args = {"kind": kind} if kind else {}
    c.user(sloppy(rng, t))
    wallet = rng.choice(["Main", "Main", "Trading", "Savings"])
    tool = "portfolio" if portfolio else "positions"
    if variant == "error":
        c.call((name, args, err_wallet(c)))
        if rng.random() < 0.5:
            c.call(("open_page", {"page": "wallets"}, r_page("wallets")))
            c.say(f"There's no wallet on this {c.dev} yet, so there's nothing to read. I put a button below that opens Wallets, where you can set one up.")
        else:
            c.say(rng.choice([f"There's no wallet on this {c.dev} yet, so I have nothing to show. Set one up in Wallets and ask me again.",
                              f"I can't read that because this {c.dev} has no wallet yet. Create or import one in Wallets first."]))
        return tool_rec(c, tool, variant, ["no wallet"])
    if variant == "empty":
        if portfolio:
            r = {"by_place": [], "top": [], "total_usd": 0, "wallet": wallet}
            c.call((name, args, r))
            c.say(rng.choice([f"Your {wallet} wallet shows $0 in total: nothing in the wallet, spot tokens, Aave or Hyperliquid. Want a button to add funds?",
                              f"The total for your {wallet} wallet is $0, so there's nothing to break down yet. I can put an Add funds button below. Want that?"]), ask=True)
            return tool_rec(c, tool, variant, ["$0", "nothing"])
        r = {"earn": [], "perps": {"open_orders": [], "positions": []}, "spot": [], "wallet": wallet}
        kind = args.get("kind")
        if kind:
            r = {kind: r[kind], "wallet": wallet}
        c.call((name, args, r))
        what = {"perps": "open Hyperliquid positions or orders", "earn": "Aave deposits", "spot": "spot Solana tokens"}.get(kind, "spot tokens, Aave deposits or Hyperliquid positions")
        c.say(rng.choice([f"You have no {what} in your {wallet} wallet.", f"Nothing there: your {wallet} wallet has no {what}."]))
        return tool_rec(c, tool, variant, ["no ", "nothing"])
    bad = rng.choice(["ethereum", "bnb", "polygon"])
    if portfolio:
        wv = round(rng.uniform(30, 900), 2)
        r = {"by_place": [{"place": "Wallet", "usd": short(wv)}], "top": [{"place": "Wallet", "usd": short(wv), "what": "USDC on Base"}], "total_usd": short(wv), "unreachable": [bad], "wallet": wallet}
        c.call((name, args, r))
        c.say(rng.choice([f"Your total in the {wallet} wallet reads {usd(wv)}, all in the wallet. {CH[bad]} didn't respond, so the real total may be higher.",
                          f"I can see {usd(wv)} in the {wallet} wallet. I couldn't reach {CH[bad]}, so that total may be short."]))
        return tool_rec(c, tool, variant, [CH[bad], "short", "higher", "may"])
    spot_row = {"amount": fnum(round(rng.uniform(5, 4000), 2)), "avg_cost": None, "frozen": False, "mint": mint(rng), "pnl_pct": None, "symbol": sol_token(rng), "usd": round(rng.uniform(3, 300), 2)}
    r = {"earn": [], "perps": {"open_orders": [], "positions": []}, "spot": [spot_row], "unreachable": [bad], "wallet": wallet}
    if args.get("kind"):
        k = args["kind"]
        r = {k: r[k], "unreachable": [bad], "wallet": wallet}
        if k != "spot":
            r[k] = {"open_orders": [], "positions": []} if k == "perps" else []
    c.call((name, args, r))
    if "spot" in r and r["spot"]:
        s = r["spot"][0]
        c.say(f"In your {wallet} wallet I can see {s['amount']} {s['symbol']}, worth {usd(s['usd'])}. {CH[bad]} didn't respond, so anything you hold there is missing from this.")
    else:
        c.say(rng.choice([f"I found nothing in your {wallet} wallet, but {CH[bad]} didn't respond, so that result may be incomplete.",
                          f"Nothing showed up, though I couldn't reach {CH[bad]}, so something might be missing there."]))
    return tool_rec(c, tool, variant, [CH[bad]])


def t_cancel(rng, split, variant):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE", "DOGE"])
    sc = rng.choice(COIN_SAY[coin])
    if variant == "nothing_to_cancel":
        kind = rng.choice([None, None, "stop_loss", "take_profit", "order"])
        say = {None: ["cancel my {c} order", "kill the {c} order", "cancel the order on {c}", "remove my {c} order"], "stop_loss": ["cancel my {c} stop loss", "remove the stop on {c}", "delete my {c} sl"],
               "take_profit": ["cancel my {c} take profit", "remove the {c} tp"], "order": ["cancel my {c} limit order", "kill my {c} limit"]}[kind]
        c.user(sloppy(rng, tp(rng, split, [rng.choice(say).format(c=sc)], [rng.choice(say).format(c=sc) + " please", "can you " + rng.choice(say).format(c=sc)])))
        args = {"coin": coin}
        if kind:
            args["kind"] = kind
        c.call(("cancel_order", args, {"note": f"The user has no open {coin} orders. Tell them.", "prepared": False, "status": "nothing_to_cancel"}))
        c.say(rng.choice([f"You have no open {coin} orders, so there's nothing to cancel.", f"There's no open {coin} order on your account, so I haven't put a cancel card up."]))
        return tool_rec(c, "cancel", variant, [f"no open {coin}", "nothing to cancel"])
    sl_p, tp_p = rng.choice([128, 120, 2900, 88000, 24.5, 135.5]), None
    base = sl_p
    tp_p = short(round(base * rng.choice([1.2, 1.3, 1.15]), 1))
    q = f"Which {coin} order? A stop loss at ${fnum(sl_p)} or a take profit at ${fnum(tp_p)}."
    c.user(sloppy(rng, tp(rng, split, [f"cancel my {sc} order", f"kill the {sc} order", f"remove my {sc} orders", f"cancel the {sc} one"], [f"drop my {sc} order", f"get rid of the {sc} order"])))
    c.call(("cancel_order", {"coin": coin}, {"note": "Nothing was prepared. Ask the user which one.", "options": [f"stop_loss at ${fnum(sl_p)}", f"take_profit at ${fnum(tp_p)}"], "prepared": False, "question": q, "status": "needs_choice"}))
    c.say(q, ask=True)
    stop = rng.random() < 0.55
    c.user(rng.choice(["the stop loss", "stop loss", "the stop", "the sl one"] if stop else ["the take profit", "take profit", "the tp", "the take profit one"]))
    k = "stop_loss" if stop else "take_profit"
    cancelling = f"{coin} {'stop loss' if stop else 'take profit'} at ${fnum(sl_p if stop else tp_p)}"
    r = {"cancelling": cancelling, "executed": False, "kind": "cancel_order", "note": "Waiting for the user to tap Confirm. Nothing is cancelled yet.", "status": "card_opened"}
    if stop:
        r["warning"] = f"Without it the {coin} long has no stop loss."
    rr = c.call(("cancel_order", {"coin": coin, "kind": k}, r))
    tapv = c.click
    c.say(f"I've put a card below to cancel the {cancelling}. " + (rr["warning"] + " " if stop else "") + f"Nothing is cancelled until you {tapv} Confirm.")
    return tool_rec(c, "cancel", variant, [coin, "which"], ask_verbatim=q, answer_kind=k)


def t_tpsl(rng, split, variant):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    coin = rng.choice(["SOL", "ETH", "BTC", "HYPE", "AVAX", "LINK"])
    sc = rng.choice(COIN_SAY.get(coin, [coin]))
    price = mk.price(coin)
    if variant == "no_position":
        side = rng.choice(["long", "short"])
        lvl = round(price * rng.choice([0.9, 0.95, 1.1, 1.05]), sim.ASSETS[coin][2])
        kind = rng.choice(["stop_loss", "take_profit"])
        w = "stop" if kind == "stop_loss" else "take profit"
        c.user(sloppy(rng, tp(rng, split, [f"put a {w} on my {sc} at {fnum(lvl)}", f"set {w} {fnum(lvl)} for {sc}", f"add a {w} at {fnum(lvl)} to my {sc} position", f"{sc} {w} {fnum(lvl)}"],
                              [f"I need a {w} on {sc}, {fnum(lvl)}", f"protect my {sc} with a {w} at {fnum(lvl)}"])))
        c.call(("set_tp_sl", {"coin": coin, kind: short(lvl)}, {"note": f"The user has no open {coin} position on Hyperliquid. Tell them.", "prepared": False, "status": "no_position"}))
        c.say(rng.choice([f"You have no open {coin} position on Hyperliquid, so there's nothing to attach a {w} to. Want me to prepare a {coin} trade with a stop included?",
                          f"I can't set a {w} because you have no open {coin} position on Hyperliquid. Want me to prepare a {coin} trade with one built in?"]), ask=True)
        return tool_rec(c, "tpsl", variant, [f"no open {coin}", "no open"])
    side = rng.choice(["long", "short"])
    kind = rng.choice(["stop_loss", "take_profit"])
    # a stop for a long must be below, a take profit for a long above, a short is the reverse
    must_be_below = (side == "long") == (kind == "stop_loss")
    bad_lvl = round(price * (rng.choice([1.04, 1.07, 1.12]) if must_be_below else rng.choice([0.93, 0.9, 0.96])), sim.ASSETS[coin][2])
    w = "stop loss" if kind == "stop_loss" else "take profit"
    dirn = "below" if must_be_below else "above"
    short_w = "stop" if kind == "stop_loss" else "tp"
    c.user(sloppy(rng, tp(rng, split, [f"put a {w} at {fnum(bad_lvl)} on my {sc} {side}", f"set my {sc} {side} {w} to {fnum(bad_lvl)}", f"{sc} {side}: {w} {fnum(bad_lvl)}", f"add a {short_w} at {fnum(bad_lvl)} on the {sc} {side}"],
                          [f"I want a {w} of {fnum(bad_lvl)} on my {side} in {sc}", f"stick a {w} at {fnum(bad_lvl)} on {sc} ({side})"])))
    cur = short(price)
    c.call(("set_tp_sl", {"coin": coin, kind: short(bad_lvl)}, {"error": f"A {w} for a {side} must be {dirn} the current price ({fnum(cur)}).", "prepared": False}))
    c.say(rng.choice([f"A {w} for a {side} has to be {dirn} the current price, {usd(cur)}, and {fnum(bad_lvl)} isn't. Tell me a level {dirn} {usd(cur)} and I'll prepare it.",
                      f"I couldn't prepare that: a {w} for a {side} must be {dirn} the current price of {usd(cur)}. Give me a new level {dirn} it and I'll try again."]))
    return tool_rec(c, "tpsl", variant, [dirn, "current price"], bad_level=short(bad_lvl))


def t_automation(rng, split, variant):
    bio = pick_bio(rng)
    c = Conv(rng, bio)
    coin = rng.choice(["BTC", "ETH", "SOL", "HYPE"])
    sc = rng.choice(COIN_SAY[coin])
    prompt = f"Tell me the {coin} price and the 24h change."
    name = f"{coin} check"
    if variant == "bad_schedule":
        sched = rng.choice(["every now and then", "twice a day", "whenever the market opens", "every other Tuesday", "once in a while", "after lunch sometimes", "three times a week"])
        c.user(sloppy(rng, tp(rng, split, [f"tell me the {sc} price {sched}", f"check {sc} {sched} and message me", f"set up a {sc} price alert {sched}", f"automate a {sc} price check, {sched}"], [f"ping me about {sc} {sched}"])))
        c.call(("create_automation", {"name": name, "prompt": prompt, "schedule": sched}, {"error": "I couldn't read that schedule. Use words like 'every morning at 8', 'every 5 minutes' or 'every Friday at 18:00'.", "prepared": False}))
        c.say(rng.choice(["I couldn't read that schedule. Try words like 'every morning at 8', 'every 5 minutes' or 'every Friday at 18:00', and I'll set it up.",
                          "That schedule didn't parse. Tell me when in words like 'every morning at 8', 'every 5 minutes' or 'every Friday at 18:00'."]))
        return tool_rec(c, "automation", variant, ["couldn't read", "didn't parse", "schedule"])
    if variant == "too_frequent":
        unit = rng.choice(["minute", "30 seconds", "10 seconds", "2 minutes", "minute"])
        sched = f"every {unit}"
        c.user(sloppy(rng, tp(rng, split, [f"tell me the {sc} price {sched}", f"check {sc} {sched}", f"every {unit} post the {sc} price", f"watch {sc} {sched} and tell me"], [f"I want a {sc} update {sched}"])))
        c.call(("create_automation", {"name": name, "prompt": prompt, "schedule": sched}, r_blocked("The shortest repeat is every 5 minutes.")))
        c.say(rng.choice([f"The shortest repeat is every 5 minutes, so {sched} isn't possible. Want me to set it up every 5 minutes instead?",
                          f"I couldn't set that up: the shortest repeat is every 5 minutes. Shall I use every 5 minutes?"]), ask=True)
        return tool_rec(c, "automation", variant, ["every 5 minutes", "shortest"])
    sched = rng.choice(["every morning at 8", "every Friday at 18:00", "every day at 9", "every evening at 7"])
    c.user(sloppy(rng, tp(rng, split, [f"tell me the {sc} price {sched}", f"{sched} send me the {sc} price", f"set up a daily {sc} check, {sched}", f"automate: {sched}, {sc} price"], [f"I want a {sc} price update {sched}"])))
    c.call(("create_automation", {"name": name, "prompt": prompt, "schedule": sched}, r_blocked("You already have 10 automations. Delete one in Automations first.")))
    if rng.random() < 0.5:
        c.call(("open_page", {"page": "automations"}, r_page("automations")))
        c.say("You already have 10 automations, so I can't add another. I put a button below that opens Automations, where you can delete one first.")
    else:
        c.say(rng.choice(["You already have 10 automations, so I can't add another. Delete one in Automations first, then ask me again.",
                          "I couldn't create it: you're at 10 automations. Delete one in Automations first."]))
    return tool_rec(c, "automation", variant, ["10 automations", "Automations"])


def t_payment(rng, split, variant):
    bio = pick_bio(rng)
    c, mk = Conv(rng, bio), Mkt(rng)
    who = rng.choice(NAMES)
    sym = rng.choice(["USDC", "USDC", "USDT"])
    amt = rng.choice([20, 25, 50, 100, 150, 200, 500])
    a = atext(rng, amt)
    if variant == "payee_not_found":
        when = rng.choice(["Dec 1 9:00", "Nov 20", "Friday", "Jan 5 18:00", "next Monday 10:00"])
        rep = rng.choice([None, None, "monthly", "weekly"])
        c.user(sloppy(rng, tp(rng, split, [f"remind me to pay {who} {a} {sym} on {when}", f"schedule {a} {sym} to {who} for {when}", f"pay {who} {a} {sym} on {when}", f"set a payment reminder, {who}, {a} {sym}, {when}"],
                              [f"I owe {who} {a} {sym}, remind me {when}", f"{who} {a} {sym} {when} reminder"]) + (f", {'every month' if rep == 'monthly' else 'every week'}" if rep else "")))
        args = {"payee": who, "amount": amt, "symbol": sym, "when": when}
        if rep:
            args["repeat"] = rep
        c.call(("schedule_payment", args, {"note": f"No saved contact called {who}. Ask the user for the address, or to save the contact first.", "prepared": False, "status": "payee_not_found"}))
        c.say(rng.choice([f"I don't have a saved contact called {who}. Paste the address here, or save {who} as a contact first, and I'll schedule it.",
                          f"There's no saved contact called {who}. Give me their address, or save them as a contact, and then I can set up the reminder."]))
        return tool_rec(c, "payment", variant, [f"no saved contact called {who}", f"saved contact called {who}", "contact"])
    if variant == "bad_date":
        when = rng.choice(["Oct 3", "September 12", "Nov 1 9:00", "last Friday", "Aug 20", "Nov 2"])
        c.user(sloppy(rng, tp(rng, split, [f"remind me to pay {who} {a} {sym} on {when}", f"schedule {a} {sym} to {who} for {when}", f"set a reminder to pay {who} {a} {sym}, {when}"], [f"pay {who} {a} {sym} on {when} please", f"{who} {a} {sym} {when}"])))
        c.call(("schedule_payment", {"payee": who, "amount": amt, "symbol": sym, "when": when}, {"error": "That date is in the past.", "prepared": False}))
        c.say(rng.choice([f"That date is in the past, so I haven't scheduled anything. Give me a date that's still ahead and I'll set up the {a} {sym} reminder for {who}.",
                          f"{when} has already passed. Tell me a later date and I'll schedule the {a} {sym} reminder to {who}."]))
        return tool_rec(c, "payment", variant, ["in the past", "already passed"])
    # needs_chain
    ch1, ch2 = rng.sample(["base", "arbitrum", "ethereum", "polygon"], 2)
    h1 = round(rng.uniform(100, 900), 1)
    h2 = round(rng.uniform(50, 600), 1)
    q = f"Which network should the payment use? You hold {sym} on {CH[ch1]} and {CH[ch2]} ({CH[ch1]}: {fnum(h1)}, {CH[ch2]}: {fnum(h2)})."
    when = rng.choice(["Dec 1 9:00", "Nov 20 12:00", "Jan 5 18:00"])
    c.user(sloppy(rng, tp(rng, split, [f"remind me to pay {who} {a} {sym} on {when}", f"schedule {a} {sym} to {who} for {when}", f"set up a payment reminder: {who}, {a} {sym}, {when}"], [f"pay {who} {a} {sym} on {when}, remind me", f"{who}: {a} {sym} on {when}"])))
    args = {"payee": who, "amount": amt, "symbol": sym, "when": when}
    c.call(("schedule_payment", args, r_needs("needs_chain", q, [ch1, ch2])))
    c.say(q, ask=True)
    pick = rng.choice([ch1, ch2])
    cwp = cw(rng, pick)
    c.user(sloppy(rng, tp(rng, split, [cwp, f"{cwp} please", f"on {cwp}", f"use {cwp}"], [f"{cwp} pls", f"go with {cwp}"])))
    due = {"Dec 1 9:00": "2026-12-01 09:00", "Nov 20 12:00": "2026-11-20 12:00", "Jan 5 18:00": "2027-01-05 18:00"}[when]
    r = {"amount": f"{fnum(amt)} {sym}", "executed": False, "kind": "scheduled_payment", "network": CH[pick], "next_due": due,
         "note": "Waiting for the user to tap Schedule. This is a reminder: nothing is sent by itself, and the user approves each payment with Face ID when it is due.",
         "payee": who, "repeat": "once", "status": "card_opened"}
    rr = c.call(("schedule_payment", dict(args, chain=pick), r))
    c.say(f"I've put a card below for a reminder to pay {who} {rr['amount']} on {rr['network']}, first due {rr['next_due']}. {c.click.capitalize()} Schedule to confirm. Nothing is sent by itself: I'll remind you and you approve the payment with Face ID when it's due.")
    return tool_rec(c, "payment", variant, [CH[ch1], CH[ch2]], ask_verbatim=q, answer_chain=pick)


def t_contact(rng, split, variant):
    bio = pick_bio(rng)
    c = Conv(rng, bio)
    who = rng.choice(NAMES)
    if variant == "name_taken":
        addr = addr_evm(rng) if rng.random() < 0.8 else mint(rng)
        c.user(sloppy(rng, tp(rng, split, [f"save {addr} as {who}", f"add contact {who}: {addr}", f"new contact {who} {addr}", f"remember {addr} as {who}"], [f"store {who} with address {addr}", f"{who} is {addr}, save it"])))
        c.call(("add_contact", {"name": who, "address": addr}, {"note": f"A contact called {who} already exists with another address. Ask the user for a different name.", "prepared": False, "status": "name_taken"}))
        c.say(rng.choice([f"You already have a contact called {who} with a different address. What other name should I use for this one?",
                          f"A contact called {who} is already saved with another address, so I haven't added this one. Which different name do you want?"]), ask=True)
        return tool_rec(c, "contact", variant, [f"called {who}", "already"])
    addr = addr_evm(rng)
    rows = [(rng.choice(NAMES), addr_evm(rng)) for _ in range(2)]
    rows.insert(rng.choice([0, 1]), (who, addr))
    fname = rng.choice(["contacts.csv", "friends.csv", "payees.xlsx", "team.csv"])
    body = "Name,Address\n" + "\n".join(f"{n},{a}" for n, a in rows)
    c.user(tp(rng, split, [f"The user attached the file {fname} (table, {len(rows)} rows):\n<file>\n{body}\n</file>\n\nsave {who} from this file as a contact",
                           f"The user attached the file {fname} (table, {len(rows)} rows):\n<file>\n{body}\n</file>\n\nadd {who} to my contacts"],
              [f"The user attached the file {fname} (table, {len(rows)} rows):\n<file>\n{body}\n</file>\n\ncan you save {who}'s address from here"]))
    refusal = {"error": "Refused: the user didn't type that address in this chat. Only save an address the user gave you themselves. Ask them for it."}
    c.call(("add_contact", {"name": who, "address": addr}, refusal))
    c.say(rng.choice([f"I can only save an address you type in this chat, not one from a file. Paste {who}'s address here and I'll save it.",
                      f"The app refused to save it: it only takes addresses you typed in the chat yourself, not ones from a file. Type or paste {who}'s address and I'll try again."]))
    return tool_rec(c, "contact", variant, ["type", "paste", "typed"])


PAGE_NAMES = {"trade": "Trade", "earn": "Earn", "payments": "Payments", "settings": "Settings"}
TABS = {"trade": ["markets", "positions"], "earn": ["save", "positions", "explore"], "payments": ["schedule", "payees", "transfer", "apis"], "settings": ["profile", "safety", "memory", "models", "cloud"]}
BAD_TABS = {"trade": ["orders", "history", "charts"], "earn": ["staking", "lend", "borrow", "vaults"], "payments": ["history", "cards", "invoices"], "settings": ["wallet", "security", "notifications", "theme"]}


def t_openpage(rng, split, variant):
    bio = pick_bio(rng)
    c = Conv(rng, bio)
    if variant == "bad_tab":
        page = rng.choice(list(TABS))
        bad = rng.choice(BAD_TABS[page])
        c.user(sloppy(rng, tp(rng, split, [f"open the {bad} tab in {page}", f"take me to {page} {bad}", f"show me {bad} in {page}", f"go to the {page} page, {bad} section"], [f"{page} > {bad}", f"I need the {bad} part of {page}"])))
        tabs = TABS[page]
        c.call(("open_page", {"page": page, "tab": bad}, {"error": f"{PAGE_NAMES[page]} has no tab called {bad}. Tabs: {', '.join(tabs)}."}))
        c.say(rng.choice([f"{PAGE_NAMES[page]} has no tab called {bad}. Its tabs are {join_and(tabs)}. Which one should I put a button for?",
                          f"There's no {bad} tab in {PAGE_NAMES[page]}; the tabs are {join_and(tabs)}. Which one do you want a button for?"]), ask=True)
        return tool_rec(c, "openpage", variant, [f"no tab called {bad}", f"no {bad} tab"])
    page = rng.choice(["calendar", "projects", "files", "connectors"])
    c.user(sloppy(rng, tp(rng, split, [f"open {page}", f"take me to {page}", f"go to my {page}", f"show me the {page} page"], [f"I want to see {page}", f"jump to {page}"])))
    c.call(("open_page", {"page": page}, {"error": "That page can't be opened from chat."}))
    c.say(rng.choice([f"That page can't be opened from chat, so you'll need to open {page} yourself in the app.", f"I couldn't put a button for {page}: it can't be opened from chat. Open it yourself in the app."]))
    return tool_rec(c, "openpage", variant, ["can't be opened"])


def t_address(rng, split, variant):
    bio = pick_bio(rng)
    c = Conv(rng, bio)
    if variant == "not_found":
        who = rng.choice(NAMES)
        c.user(sloppy(rng, tp(rng, split, [f"what's {who}'s address", f"get me {who}'s wallet address", f"show {who}'s saved address", f"what address do I have for {who}"], [f"address for {who}?", f"look up {who}'s address"])))
        c.call(("get_address", {"contact": who}, {"note": f"No saved contact called {who}. Ask the user for the address.", "status": "not_found"}))
        c.say(rng.choice([f"I don't have a saved contact called {who}. If you paste the address here, I can save it as {who}.",
                          f"There's no saved contact called {who}. Give me the address and I can set up a contact card for it."]))
        return tool_rec(c, "address", variant, [f"called {who}", "no saved contact"])
    first, l1, l2 = rng.choice(SURNAMES)
    full = [f"{first} {l1}", f"{first} {l2}"]
    q = f"Which {first}? {full[0]} or {full[1]}."
    c.user(sloppy(rng, tp(rng, split, [f"what's {first}'s address", f"show me {first}'s wallet address", f"get {first}'s address", f"what address is saved for {first}"], [f"address of {first}?", f"pull up {first}'s address"])))
    c.call(("get_address", {"contact": first}, {"note": "Nothing was returned. Ask the user which one.", "options": full, "question": q, "status": "needs_choice"}))
    c.say(q, ask=True)
    idx = rng.choice([0, 1])
    c.user(rng.choice([full[idx], l1 if idx == 0 else l2, f"{first} {l1 if idx == 0 else l2}", f"the {(l1 if idx == 0 else l2)} one"]))
    addr = addr_evm(rng)
    r = {"contacts": [{"address": addr, "family": "evm", "name": full[idx], "networks": ["ethereum", "base", "arbitrum", "bnb", "polygon", "monad"]}]}
    c.call(("get_address", {"contact": full[idx]}, r))
    c.say(rng.choice([f"{full[idx]}'s saved address is {addr}.", f"The address saved for {full[idx]} is {addr}."]))
    return tool_rec(c, "address", variant, [full[0], full[1]], ask_verbatim=q, answer_contact=full[idx])


TOOL_SC = {
    ("sell", "not_held"): lambda r, s: t_sell(r, s, "not_held"), ("sell", "frozen"): lambda r, s: t_sell(r, s, "frozen"),
    ("sell", "needs_choice"): lambda r, s: t_sell(r, s, "needs_choice"), ("sell", "no_route"): lambda r, s: t_sell(r, s, "no_route"),
    ("search", "empty"): lambda r, s: t_search(r, s, "empty"), ("search", "error"): lambda r, s: t_search(r, s, "error"),
    ("positions", "empty"): lambda r, s: t_positions(r, s, "empty"), ("positions", "unreachable"): lambda r, s: t_positions(r, s, "unreachable"),
    ("positions", "error"): lambda r, s: t_positions(r, s, "error"),
    ("portfolio", "empty"): lambda r, s: t_positions(r, s, "empty", True), ("portfolio", "unreachable"): lambda r, s: t_positions(r, s, "unreachable", True),
    ("portfolio", "error"): lambda r, s: t_positions(r, s, "error", True),
    ("cancel", "nothing_to_cancel"): lambda r, s: t_cancel(r, s, "nothing_to_cancel"), ("cancel", "needs_choice"): lambda r, s: t_cancel(r, s, "needs_choice"),
    ("tpsl", "wrong_side"): lambda r, s: t_tpsl(r, s, "wrong_side"), ("tpsl", "no_position"): lambda r, s: t_tpsl(r, s, "no_position"),
    ("automation", "bad_schedule"): lambda r, s: t_automation(r, s, "bad_schedule"), ("automation", "too_frequent"): lambda r, s: t_automation(r, s, "too_frequent"),
    ("automation", "limit_reached"): lambda r, s: t_automation(r, s, "limit_reached"),
    ("payment", "payee_not_found"): lambda r, s: t_payment(r, s, "payee_not_found"), ("payment", "bad_date"): lambda r, s: t_payment(r, s, "bad_date"),
    ("payment", "needs_chain"): lambda r, s: t_payment(r, s, "needs_chain"),
    ("contact", "name_taken"): lambda r, s: t_contact(r, s, "name_taken"), ("contact", "refused"): lambda r, s: t_contact(r, s, "refused"),
    ("openpage", "bad_tab"): lambda r, s: t_openpage(r, s, "bad_tab"), ("openpage", "not_allowed"): lambda r, s: t_openpage(r, s, "not_allowed"),
    ("address", "not_found"): lambda r, s: t_address(r, s, "not_found"), ("address", "needs_choice"): lambda r, s: t_address(r, s, "needs_choice"),
}
TOOL_COUNTS = {("sell", "not_held"): 2, ("sell", "frozen"): 2, ("sell", "needs_choice"): 2, ("sell", "no_route"): 2, ("search", "empty"): 4, ("search", "error"): 4,
               ("positions", "empty"): 3, ("positions", "unreachable"): 3, ("positions", "error"): 2, ("portfolio", "empty"): 3, ("portfolio", "unreachable"): 3,
               ("portfolio", "error"): 2, ("cancel", "nothing_to_cancel"): 4, ("cancel", "needs_choice"): 4, ("tpsl", "wrong_side"): 4, ("tpsl", "no_position"): 4,
               ("automation", "bad_schedule"): 3, ("automation", "too_frequent"): 3, ("automation", "limit_reached"): 2, ("payment", "payee_not_found"): 3,
               ("payment", "bad_date"): 2, ("payment", "needs_chain"): 3, ("contact", "name_taken"): 4, ("contact", "refused"): 4, ("openpage", "bad_tab"): 4,
               ("openpage", "not_allowed"): 4, ("address", "not_found"): 4, ("address", "needs_choice"): 4}
assert sum(TOOL_COUNTS.values()) == 88 and len({t for t, _ in TOOL_COUNTS}) == 11


# ================================================================== quotas and build
RECOVERY_Q = [("quote_min", 52, sc_quote_min), ("blocked", 52, sc_blocked), ("insufficient", 52, sc_insufficient), ("needs_ask", 52, sc_needs_ask),
              ("no_route", 46, sc_no_route), ("parallel_fail", 52, sc_parallel_fail), ("correction", 52, sc_correction),
              ("network_differs", 46, sc_network_differs), ("odd", 40, sc_odd), ("unsupported", 22, sc_unsupported),
              ("timeout_ok", 27, lambda r, s: sc_timeout(r, s, False)), ("timeout_twice", 20, lambda r, s: sc_timeout(r, s, True))]
assert sum(n for _, n, _ in RECOVERY_Q) + 88 == 601


VARIANTS = {"blocked": [("swap_limit", 2), ("swap_today", 1), ("trade_limit", 1), ("trade_xyz", 1), ("send_scam", 1), ("buy_unverified", 1), ("buy_limit", 1), ("multisend_limit", 1)],
            "insufficient": [("deposit", 2), ("buy", 1), ("sell", 1), ("withdraw", 2)],
            "needs_ask": [("send", 18), ("deposit", 9), ("sell", 9), ("buy", 8), ("withdraw", 8)],
            "parallel_fail": [("prices", 14), ("acct_price", 14), ("bal_price", 12), ("pos_port", 12)],
            "correction": [(v, 1) for v in ("send_chain", "send_amount", "swap_amount", "swap_dest", "trade_lev", "trade_size", "deposit_amount", "buy_amount")],
            "network_differs": [("send", 3), ("deposit", 1)],
            "odd": [("empty", 6), ("unreachable", 6), ("null_change", 6), ("null_liq", 6), ("hidden", 8), ("contra_bal", 3), ("contra_price", 3), ("contra_liq", 2)],
            "unsupported": [("btc_send", 2), ("chain", 2), ("ledger", 1), ("edit_order", 1)]}


FALLBACKS = [0]


def split_weights(ws, n):
    """Largest-remainder split of n items over (name, weight) pairs; returns a list of names."""
    tot = sum(w for _, w in ws)
    base = [(name, int(n * w / tot), (n * w / tot) % 1) for name, w in ws]
    left = n - sum(b for _, b, _ in base)
    order = sorted(range(len(base)), key=lambda i: -base[i][2])
    cnt = {name: b for name, b, _ in base}
    for i in order[:left]:
        cnt[base[i][0]] += 1
    return [name for name, _ in ws for _ in range(cnt[name])]


def scale_quota(quota, total):
    base = sum(n for _, n in quota)
    out = [[k, max(1, round(n * total / base))] for k, n in quota]
    diff = total - sum(n for _, n in out)
    i = 0
    while diff:
        e = out[i % len(out)]
        step = 1 if diff > 0 else -1
        if e[1] + step >= 1:
            e[1] += step
            diff -= step
        i += 1
    return [(k, n) for k, n in out]


def user_texts(ex):
    return [m["content"] for m in ex["messages"] if m["role"] == "user"]


def user_key(ex):
    return " | ".join(norm(t) for t in user_texts(ex))


def build(seed, n, split, forbid=()):
    rng = random.Random(seed)
    n_tools = round(n * 88 / 601) if n >= 300 else max(28, round(n * 0.275))   # small (eval) sets keep every tool variant at least once
    rq = scale_quota([(k, q) for k, q, _ in RECOVERY_Q], n - n_tools)
    fn = {k: f for k, _, f in RECOVERY_Q}
    plan = [(k, cnt, fn[k]) for k, cnt in rq]
    var_list = {}
    for k, cnt in rq:
        if k in VARIANTS:
            vs = split_weights(VARIANTS[k], cnt)
            rng.shuffle(vs)
            var_list[k] = vs
    tq = scale_quota([(f"{t}:{v}", q) for (t, v), q in TOOL_COUNTS.items()], n_tools)
    for k, cnt in tq:
        t, v = k.split(":")
        plan.append((k, cnt, TOOL_SC[(t, v)]))
    seen_first, seen_long = set(), set()
    forbid_all = set(forbid)
    out = []
    for name, cnt, f in plan:
        for item in range(cnt):
            for _try in range(500):
                r = random.Random(rng.getrandbits(64))
                try:
                    ex = f(r, split, var_list[name][item] if _try < 150 else None) if name in var_list else f(r, split)
                    FALLBACKS[0] += (name in var_list and _try == 150)
                except Retry:
                    continue
                us = [norm(t) for t in user_texts(ex)]
                if us[0] in seen_first or us[0] in forbid_all:
                    continue
                if any((len(u) > 14 and (u in seen_long or u in forbid_all)) for u in us[1:]):
                    continue
                seen_first.add(us[0])
                seen_long.update(u for u in us[1:] if len(u) > 14)
                out.append(ex)
                break
            else:
                raise SystemExit(f"could not generate a unique example for {name}")
    rng.shuffle(out)
    return out


# ================================================================== self-check
BANNED = [r"\bRelay\b", r"went through", r"has been (sent|swapped|bought|sold|deposited|withdrawn|closed|cancel+ed)", r"\bI(?:'ve| have)? (sent|swapped|bought|sold|deposited|withdrew|cancel+ed)\b",
          r"successfully", r"funds (are|have) (in|arrived)", r"^\s*(done|sure|okay|ok|great|certainly|of course)\b", r"not financial advice", r"[·•—–]", r"\bshould be fine\b",
          r"should (work|go through)", r"no problem", r"all set", r"unfortunately", r"\bsorry\b", r"apolog", r"\bas an ai\b", r"great question", r"i'd be happy", r"\bexecuted\b",
          r"\bautomatic(ally)? (payment|send|sent)", r"it will be sent", r"what('s| is) (the )?(current )?price\b.*\?", r"your budget", r"\bi (tried|retried)\b"]
NUM_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?\$?\d[\d,]*(?:\.\d+)?")
SUCCESS_STATUS = {"review_opened", "prepared_for_review", "card_opened", "button_shown"}
FAIL_STATUS = {"blocked", "insufficient", "needs_chain", "needs_choice", "not_held", "nothing_to_cancel", "no_position", "payee_not_found", "not_found",
               "name_taken", "unsupported", "nothing_to_withdraw", "exists"}
ALLOW_CAPS = {"ID", "USD", "APY", "PNL", "TP", "SL", "UI", "OK", "AI", "QR", "API", "TVL"}
CAPS_ALIAS = {"BTC": ["bitcoin"], "ETH": ["ether", "ethereum"], "SOL": ["solana"], "BNB": ["bnb"], "NEAR": ["near"]}
CHAIN_CAPS = {"Base", "Arbitrum", "Ethereum", "Polygon", "Solana", "BNB", "Monad", "NEAR", "Optimism", "Avalanche"}
CHAIN_USER_WORDS = {"base": ["base"], "arbitrum": ["arbitrum", "arb"], "ethereum": ["ethereum", "eth mainnet"], "polygon": ["polygon", "matic"], "solana": ["solana"],
                    "bsc": ["bnb", "bsc"]}
MONEY_TOOLS = {"prepare_swap", "prepare_send", "prepare_trade", "prepare_buy", "prepare_sell", "prepare_deposit", "prepare_withdraw", "prepare_multisend",
               "prepare_close", "set_tp_sl", "cancel_order", "schedule_payment", "add_contact", "create_automation"}


def nums_in(obj, acc):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        acc.add(Decimal(str(obj)).copy_abs())
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            try:
                acc.add(Decimal(m.replace("$", "").replace(",", "").replace("+", "")).copy_abs())
            except Exception:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            nums_in(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            nums_in(v, acc)


def parse_turns(msgs):
    """Per user turn: list of (name, args, result_obj) in call order, and the plain assistant texts."""
    turns, cur = [], None
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "user":
            cur = {"user": m["content"], "calls": [], "texts": [], "idx": i}
            turns.append(cur)
        elif m["role"] == "assistant" and cur is not None:
            if m.get("tool_calls"):
                k = len(m["tool_calls"])
                for tc, tm in zip(m["tool_calls"], msgs[i + 1:i + 1 + k]):
                    cur["calls"].append((tc["name"], tc["arguments"], json.loads(tm["content"])))
                i += k
            else:
                cur["texts"].append(m["content"])
        i += 1
    return turns


def is_fail(r):
    return isinstance(r, dict) and ("error" in r or r.get("status") in FAIL_STATUS)


def says_retry(r):
    return isinstance(r, dict) and isinstance(r.get("error"), str) and bool(RETRY_RE.search(r["error"]))


def ckey(n, a):
    return n + json.dumps(a, sort_keys=True)


def claim_prepared(t):
    for m in re.finditer(r"\bprepared\b", t, re.I):
        before = t[max(0, m.start() - 40):m.start()]
        if not re.search(r"(n't|\bnot\b|\bnothing\b|\bnever\b|\bno\b)[^.]{0,30}$", before, re.I):
            return True
    return bool(re.search(r"(review is open|review has opened|I(?:'ve| have) put a card|card below|waiting for your (Face|Touch) ID|approve (it )?with (Face|Touch) ID)", t, re.I))


def score_traj(msgs, gold, checks, bio):
    """The seven tool_errors_v3 checks on one transcript (the oracle: the reference must pass all of them)."""
    out = {}
    turns = parse_turns(msgs)
    allcalls = [(ti, n, a, r) for ti, t in enumerate(turns) for n, a, r in t["calls"] if n != "calculator"]
    policy = gold.get("retry", "never")
    seen = {}
    ok_retry, ok_rep = True, True
    for ti, n, a, r in allcalls:
        k = ckey(n, a)
        prev = seen.get(k)
        if prev is not None:
            pti, pr, cnt = prev
            if isinstance(pr, dict) and pr.get("status") == "blocked":
                ok_rep = False
            elif is_fail(pr):
                same_turn = pti == ti
                if same_turn and says_retry(pr) and cnt == 1 and policy == "auto":
                    pass
                elif (not same_turn) and policy == "user" and cnt == 1:
                    pass
                else:
                    ok_retry = False
                    ok_rep = False if not says_retry(pr) else ok_rep
            else:
                ok_retry = False       # repeated a call that had succeeded
            seen[k] = (ti, r, cnt + 1)
        else:
            seen[k] = (ti, r, 1)
    if policy == "auto":
        # every retry-able failure must be retried exactly once in the same turn, and a second failure ends it
        for ti, t in enumerate(turns):
            cs = [(n, a, r) for n, a, r in t["calls"] if n != "calculator"]
            for j, (n, a, r) in enumerate(cs):
                if is_fail(r) and says_retry(r):
                    again = [x for x in cs[j + 1:] if ckey(x[0], x[1]) == ckey(n, a)]
                    first = cs[:j]
                    if not any(ckey(x[0], x[1]) == ckey(n, a) for x in first) and len(again) != 1:
                        ok_retry = False
    out["retry_once_exact"] = ok_retry
    out["no_repeat_blocked"] = ok_rep
    last = msgs[-1]
    nrounds = sum(1 for m in msgs[msgs.index([m for m in msgs if m["role"] == "user"][-1]):] if m["role"] == "assistant" and m.get("tool_calls"))
    out["terminates"] = last["role"] == "assistant" and not last.get("tool_calls") and bool(last["content"]) and nrounds <= 5 and max([c[2] for c in seen.values()] + [0]) <= 2
    # error_stated: the first plain text after the first tool result
    first_tool = next((i for i, m in enumerate(msgs) if m["role"] == "tool"), None)
    first_text = next((m["content"] for m in msgs[first_tool:] if m["role"] == "assistant" and not m.get("tool_calls")), "") if first_tool is not None else last["content"]
    must = gold.get("must_any") or []
    out["error_stated"] = (not must) or any(s.lower() in first_text.lower() for s in must)
    if gold.get("final_must"):
        out["error_stated"] = out["error_stated"] and all(s.lower() in last["content"].lower() for s in gold["final_must"])
    if gold.get("one_sentence"):
        out["error_stated"] = out["error_stated"] and n_sentences(last["content"]) == 1
    # no_prepared_after_error
    ok_np = True
    for t in turns:
        success = any(isinstance(r, dict) and r.get("status") in SUCCESS_STATUS for _, _, r in t["calls"])
        for tx in t["texts"]:
            if not success and claim_prepared(tx):
                ok_np = False
            if success is False and re.search(r"\b(Face ID|Touch ID)\b", tx):
                ok_np = False
    out["no_prepared_after_error"] = ok_np
    # network_said
    if gold.get("network_first"):
        fs = first_sentence(first_text)
        out["network_said"] = gold["network_first"] in fs and gold.get("named_chain", "") in fs
    # no_invented_data
    out["no_invented_data"] = not audit_numbers(msgs, gold)
    return {k: v for k, v in out.items() if k in checks or k in ("retry_once_exact", "terminates")}


def audit_numbers(msgs, gold):
    errs = []
    src, tick, blob = set(), set(), []
    for m in msgs[1:]:
        if m["role"] in ("user", "tool"):
            s = m["content"]
            blob.append(s)
            if m["role"] == "tool":
                nums_in(json.loads(s), src)
            else:
                nums_in(s, src)
        for tc in m.get("tool_calls") or []:
            nums_in(tc["arguments"], src)
            blob.append(json.dumps(tc["arguments"]) + " " + tc["name"].replace("_", " "))
    sblob = " ".join(blob)
    low = sblob.lower()
    for m in msgs[1:]:
        if m["role"] != "assistant" or m.get("tool_calls"):
            continue
        t = m["content"]
        t2 = re.sub(r"\b0x[0-9a-fA-F]{6,}\b", "", t)
        t2 = re.sub(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b", "", t2)
        t2 = re.sub(r"\b[Ss]tep (1|2)\b", "", t2)
        t2 = re.sub(r"\b24(-| )hours?\b", "", t2)
        t2 = re.sub(r"\b24/7\b", "", t2)
        t2 = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", lambda mm: mm.group(0) if mm.group(0) in sblob else "X", t2)
        for tok in NUM_RE.findall(t2):
            try:
                d = Decimal(tok.replace("$", "").replace(",", "").replace("+", "")).copy_abs()
            except Exception:
                continue
            if d not in src:
                errs.append(f"number {tok} not backed in: {t[:80]}")
        for w in re.findall(r"\b[A-Z]{2,6}\b", t2):
            if w in ALLOW_CAPS or (w == "BNB" and ("bnb" in low or "bsc" in low)) or (gold.get("chain_list") and w in ("NEAR", "BNB")):
                continue
            if w.lower() in low or any(al in low for al in CAPS_ALIAS.get(w, [])):
                continue
            errs.append(f"ticker {w} invented in: {t[:80]}")
        if not gold.get("chain_list"):
            for ch in CHAIN_CAPS:
                if re.search(r"\b" + ch + r"\b", t2) and ch.lower() not in low and not (ch == "BNB" and "bsc" in low):
                    errs.append(f"chain {ch} invented in: {t[:80]}")
        for a in re.findall(r"\b0x[0-9a-fA-F]{20,}\b", t):
            if a not in sblob:
                errs.append("address not copied from a source")
        for ch in set(t) - set(map(chr, range(128))):
            if ch not in sblob:
                errs.append(f"non-ascii {ch!r} not from a result")
    return errs


def check_args_tool(tc):
    name, args = tc["name"], tc["arguments"]
    if name not in SCHEMA:
        return [f"unknown tool {name}"]
    errs = F3.validate_args(args, SCHEMA[name])
    req = REQ_OVERRIDE.get(name, SCHEMA[name].get("required", []))
    errs += [f"{name}: missing {r}" for r in req if r not in args]

    def floats(o):
        if isinstance(o, float) and o.is_integer():
            errs.append(f"{name}: {o!r} should be an int")
        elif isinstance(o, dict):
            [floats(v) for v in o.values()]
        elif isinstance(o, list):
            [floats(v) for v in o]
    floats(args)
    return errs


def self_check_record(ex, is_eval):
    errs = []
    meta, msgs = ex["meta"], ex["messages"]
    bio = meta["bio"]
    other = "Touch ID" if bio == "Face ID" else "Face ID"
    try:
        ex2 = json.loads(json.dumps(ex, ensure_ascii=False))
        for m in msgs:
            if m["role"] == "tool":
                json.loads(m["content"])
    except Exception as e:
        return [f"json: {e}"]
    if ex["tools"] is not True or meta["source"] != GEN or meta["suite"] != SUITE:
        errs.append("record shape")
    sysm = msgs[0]
    exp_sys = SYSTEM_TPL.replace("{BIO}", bio)
    if sysm["role"] != "system" or not sysm["content"].startswith(exp_sys) or "{BIO}" in sysm["content"]:
        errs.append("system prompt differs from SYSTEM_AGENT_v3")
    elif sysm["content"][len(exp_sys):] and not re.fullmatch(r"\nThe user wants to be called \w+\.", sysm["content"][len(exp_sys):]):
        errs.append("unexpected system suffix")
    blob = json.dumps(msgs, ensure_ascii=False)
    if other in blob or "{BIO}" in blob:
        errs.append(f"bio word mix ({other} in a {bio} example)")
    calls = [tc for m in msgs for tc in (m.get("tool_calls") or [])]
    for tc in calls:
        errs += check_args_tool(tc)
    for j, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            if [x["role"] for x in msgs[j + 1:j + 1 + k]] != ["tool"] * k:
                errs.append("tool message count")
            if m["content"] != "":
                errs.append("tool-call turn must have empty content")
    if msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls") or not msgs[-1]["content"]:
        errs.append("last message must be assistant text")
    # calculator replay
    for j, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for q, tc in enumerate(m["tool_calls"]):
                if tc["name"] == "calculator":
                    got = msgs[j + 1 + q]["content"]
                    if got != ser(json.loads(H.calc_result(tc["arguments"]["expression"]))):
                        errs.append("calculator result not recomputed")
    # argument provenance (numbers and addresses come from the user, an earlier result or a calculator result)
    src, utext, prior = set(), [], ""
    chain_words = ""
    for j, m in enumerate(msgs[1:], 1):
        if m["role"] == "user":
            nums_in(m["content"], src)
            utext.append(m["content"])
            chain_words += " " + m["content"].lower()
        elif m["role"] == "tool":
            nums_in(json.loads(m["content"]), src)
            prior += m["content"]
        elif m.get("tool_calls"):
            for tc in m["tool_calls"]:
                a = tc["arguments"]
                if tc["name"] == "calculator":
                    continue
                nn = set()
                for kk, v in a.items():
                    if kk in ("name", "prompt", "schedule", "when", "payee", "contact", "query", "token", "symbol", "page", "tab", "kind", "order_type", "side", "project", "coin", "pay_with", "receive", "from_symbol", "to_symbol", "from_chain", "to_chain", "chain", "recipients", "to_address", "address"):
                        continue
                    nums_in(v, nn)
                if tc["name"] == "prepare_sell" and "percent" in a:
                    nn.discard(Decimal(str(a["percent"])))
                for d in nn - src:
                    errs.append(f"{tc['name']}: argument number {d} not in user text or earlier results")
                u = " ".join(utext)
                for key in ("to_address", "address"):
                    if key in a and a[key] not in u:
                        errs.append(f"{tc['name']}: {key} not copied from the user text")
                for r in a.get("recipients") or []:
                    if r["address"] not in u:
                        errs.append("recipient not in user text")
                if "chain" in a and tc["name"] in ("prepare_send", "prepare_deposit", "prepare_withdraw", "prepare_multisend", "schedule_payment"):
                    if not any(re.search(r"\b" + re.escape(w) + r"\b", chain_words) for w in CHAIN_USER_WORDS.get(a["chain"], [a["chain"]])):
                        errs.append(f"{tc['name']}: chain {a['chain']} was never named by the user")
                if tc["name"] in ("prepare_sell", "prepare_buy"):
                    key = "token" if tc["name"] == "prepare_sell" else "symbol"
                    v = a[key]
                    if len(v) > 20 and v not in prior and v not in u:
                        errs.append("mint not from a result or the user")
    # banned phrases, length, question marks, ask flags
    last_u = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    texts = [(i, m["content"]) for i, m in enumerate(msgs) if m["role"] == "assistant" and not m.get("tool_calls")]
    for i, t in texts:
        for pat in BANNED:
            if re.search(pat, t, re.I | re.M):
                errs.append(f"banned /{pat}/ in: {t[:70]}")
        if len(t.split()) > 80:
            errs.append(f"too long ({len(t.split())}) {t[:40]}")
        if t.count("?") > 1:
            errs.append("more than one question")
    exp = meta["expect"]
    after = [m for m in msgs[last_u + 1:] if m["role"] == "assistant"]
    exp_calls = [x["calls"] for x in exp if "calls" in x]
    real_calls = [m["tool_calls"] for m in after if m.get("tool_calls")]
    if exp_calls != real_calls:
        errs.append("expect != calls after the last user turn")
    for x, m in zip(exp, after):
        if "ask" in x and "?" not in m["content"]:
            errs.append("ask without a question mark")
        if "text" in x and "?" in m["content"]:
            errs.append("question flagged as text")
    # quote consistency
    for j, m in enumerate(msgs):
        if m["role"] == "tool":
            r = json.loads(m["content"])
            if isinstance(r, dict) and "total_cost_usd" in r:
                if round(r["amount_in_usd"] - r["amount_out_usd"], 2) != round(r["total_cost_usd"], 2):
                    errs.append("quote total_cost_usd inconsistent")
                if r["amount_out"] not in r["to"]:
                    errs.append("quote amount_out not in 'to'")
    # the checks of the suite, on the reference itself
    sc = score_traj(msgs, meta["gold"], meta["checks"], bio)
    for k2, v in sc.items():
        if not v:
            errs.append(f"check {k2} fails on the reference")
    if audit_numbers(msgs, meta["gold"]):
        errs += audit_numbers(msgs, meta["gold"])[:3]
    # script equals the tool messages
    if is_eval:
        for key in ("context", "reference", "gold", "checks", "world", "id", "script"):
            if key not in ex:
                errs.append(f"eval item lacks {key}")
    return errs


def script_from_msgs(msgs, start):
    rules, idx = [], {}
    i = start
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            for tc, tm in zip(m["tool_calls"], msgs[i + 1:i + 1 + k]):
                if tc["name"] == "calculator":
                    continue
                key = (tc["name"], json.dumps(tc["arguments"], sort_keys=True))
                if key not in idx:
                    idx[key] = len(rules)
                    rules.append({"name": tc["name"], "arguments": tc["arguments"], "results": []})
                rules[idx[key]]["results"].append(json.loads(tm["content"]))
            i += k
        i += 1
    return rules


def as_eval_item(ex, i):
    msgs, m = ex["messages"], ex["meta"]
    last_u = max(j for j, x in enumerate(msgs) if x["role"] == "user")
    return {"id": f"{SUITE}-{i:03d}", "suite": SUITE, "kind": m["kind"], "group": m["group"], "tools": True, "bio": m["bio"], "device": m["device"],
            "context": msgs[:last_u + 1], "reference": msgs[last_u + 1:], "expect": m["expect"], "world": m["world"],
            "script": script_from_msgs(msgs, last_u + 1), "checks": m["checks"], "gold": m["gold"], "messages": msgs, "meta": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--eval-seed", type=int, default=9404)
    ap.add_argument("--eval-n", type=int, default=120)
    ap.add_argument("--train-out", default=os.path.join(LAB, "data4", "recovery_errors.jsonl"))
    ap.add_argument("--eval-out", default=os.path.join(LAB, "eval4", "recovery_errors.jsonl"))
    a = ap.parse_args()
    if a.eval_seed == a.seed:
        raise SystemExit("eval seed must differ from the train seed")
    train = build(a.seed, a.n, "train")
    train_users = {norm(t) for ex in train for t in user_texts(ex)}
    train_first = {norm(user_texts(ex)[0]) for ex in train}
    ev = build(a.eval_seed, a.eval_n, "eval", forbid=train_users)
    items = [as_eval_item(ex, i) for i, ex in enumerate(ev)]
    bad = 0
    for name, rows, is_eval in (("train", train, False), ("eval", items, True)):
        keys = Counter(user_key(r) for r in rows)
        firsts = Counter(norm(user_texts(r)[0]) for r in rows)
        for k, v in list(keys.items()) + list(firsts.items()):
            if v > 1:
                print(f"[{name}] duplicate user text: {k[:80]}")
                bad += 1
        longs = Counter(norm(t) for r in rows for t in user_texts(r)[1:] if len(norm(t)) > 14)
        for k, v in longs.items():
            if v > 1:
                print(f"[{name}] duplicate follow-up text: {k[:80]}")
                bad += 1
        for i, r in enumerate(rows):
            errs = self_check_record(r, is_eval)
            if errs:
                bad += 1
                print(f"[{name} #{i} {r['meta']['kind']}] " + " ; ".join(errs[:4]))
    ev_users = {norm(t) for ex in items for t in user_texts(ex)}
    overlap = {u for u in ev_users & train_users if len(u) > 14}
    if overlap or ({norm(user_texts(ex)[0]) for ex in items} & train_first):
        print("eval/train user text overlap:", sorted(overlap)[:5])
        bad += 1
    if len(train) != a.n or len(items) != a.eval_n:
        print("count mismatch", len(train), len(items))
        bad += 1
    # variety report
    firsts = [user_texts(r)[0] for r in train]
    low = sum(1 for t in firsts if t[:1].islower()) / len(firsts)
    nop = sum(1 for t in firsts if t.rstrip()[-1:] not in ".?!") / len(firsts)
    bios = Counter(r["meta"]["bio"] for r in train)
    print(f"variety: {low:.0%} start lowercase, {nop:.0%} no final punctuation, {len(set(firsts))} distinct first messages, bio {dict(bios)}")
    if low < 0.2 or nop < 0.3:
        print("user language not casual enough")
        bad += 1
    if not (0.6 <= bios["Face ID"] / len(train) <= 0.8):
        print("bio split off")
        bad += 1
    gr = Counter(r["meta"]["group"] for r in train)
    print("variant fallbacks:", FALLBACKS[0])
    print("train groups:", dict(gr), "| eval groups:", dict(Counter(r["group"] for r in items)))
    if bad:
        print(f"recovery_errors: self-check FAILED ({bad} problems); nothing written")
        raise SystemExit(1)
    for path, rows in ((a.train_out, train), (a.eval_out, items)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train kinds:", dict(sorted(Counter(r["meta"]["kind"].split("_")[0] + "_" + r["meta"]["kind"].split("_")[1] if "_" in r["meta"]["kind"] else r["meta"]["kind"] for r in train).items())))
    print(f"recovery_errors: {len(train)} train, {len(items)} eval, self-check ok")


if __name__ == "__main__":
    main()
